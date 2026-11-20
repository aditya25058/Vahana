"""The `vahana` attention function: batch-, padding- and query-length-invariant attention for any
transformers model that calls `attention_interface(module, q, k, v, mask, scaling=..., sliding_window=...,
s_aux=..., softcap=...)`.

Each (row, head, query) is one program with a fixed reduction order that depends only on that query's own
allowed key set:
  - keys are visited in fixed blocks of BT, anchored at the query's first allowed key (so left padding, the
    batch it sits in, and whether it is computed in a prefill or a decode step never move block boundaries);
  - scores are 16-lane fp32 fma dot products over the head dimension, then a fixed tree (the twin arithmetic);
  - online softmax over blocks in order with Vahana's exp and fixed halving trees over each block's 64 slots;
  - the CPU (csrc attention_twin) computes exactly the same operations, so CPU and GPU attention give the same bits;
  - attention sinks (s_aux, gpt-oss) are a virtual key visited first;
  - optional logit soft-capping.
GQA / MQA by head grouping; the value head size may differ from the query/key size (MLA expanded form).
The mask is always an explicit boolean [B, 1, Q, T] (allowed = True); sliding windows arrive inside it.
"""
from __future__ import annotations

import torch

from ._triton import tl, triton
from .hybrid import C_LN2HI, C_LN2LO, C_LOG2E, P0, P1, P2, P3, P4, P5, _dexp
from .twinlin import _tree16

BT = 64          # fixed: the halving trees below are written for 64 slots


@triton.jit
def _tsum64(x):
    """Sum of 64 values by halving: slot c += slot c + 32, then + 16, ... (the CPU twin's tree64). Values >= 0."""
    a = tl.sum(tl.reshape(x, (2, 32)), axis=0)
    a = tl.sum(tl.reshape(a, (2, 16)), axis=0)
    a = tl.sum(tl.reshape(a, (2, 8)), axis=0)
    a = tl.sum(tl.reshape(a, (2, 4)), axis=0)
    a = tl.sum(tl.reshape(a, (2, 2)), axis=0)
    a = tl.sum(tl.reshape(a, (2, 1)), axis=0)
    return tl.sum(a, axis=0)


@triton.jit
def _tsum64_rows(x, C: tl.constexpr):
    """Column sums of x [64, C] by the same halving tree over rows."""
    a = tl.sum(tl.reshape(x, (2, 32, C)), axis=0)
    a = tl.sum(tl.reshape(a, (2, 16, C)), axis=0)
    a = tl.sum(tl.reshape(a, (2, 8, C)), axis=0)
    a = tl.sum(tl.reshape(a, (2, 4, C)), axis=0)
    a = tl.sum(tl.reshape(a, (2, 2, C)), axis=0)
    a = tl.sum(tl.reshape(a, (2, 1, C)), axis=0)
    return tl.reshape(a, (C,))


@triton.jit
def _attn(q, k, v, o, first, last, mask, sinks, scale, softcap, H, G, Q, T,
          sqb, sqh, sqq, skb, skh, skt, svb, svh, svt, sob, soq, soh, smb, smq,
          D: tl.constexpr, DV: tl.constexpr, NCH: tl.constexpr, DVP: tl.constexpr, BT: tl.constexpr,
          HAS_MASK: tl.constexpr, HAS_SINK: tl.constexpr, HAS_CAP: tl.constexpr,
          LOG2E: tl.constexpr, LN2HI: tl.constexpr, LN2LO: tl.constexpr, P0: tl.constexpr, P1: tl.constexpr,
          P2: tl.constexpr, P3: tl.constexpr, P4: tl.constexpr, P5: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    i = pid % Q
    h = (pid // Q) % H
    b = pid // (Q * H)
    hk = h // G
    ln = tl.arange(0, 16)
    dv = tl.arange(0, DVP)
    qb = q + b * sqb + h * sqh + i * sqq
    t_first = tl.load(first + b * Q + i).to(tl.int64)
    t_last = tl.load(last + b * Q + i).to(tl.int64)
    if HAS_SINK:
        m_i = tl.load(sinks + h).to(tl.float32)
        l_i = tl.full([], 1.0, tl.float32)
    else:
        m_i = tl.full([], -float("inf"), tl.float32)
        l_i = tl.full([], 0.0, tl.float32)
    acc = tl.zeros((DVP,), dtype=tl.float32)
    tt = tl.arange(0, BT).to(tl.int64)
    for t0 in range(t_first, t_last + 1, BT):
        t = t0 + tt
        ok = t <= t_last
        if HAS_MASK:
            ok = ok & (tl.load(mask + b * smb + i * smq + t, mask=ok & (t < T), other=0) != 0)
        sacc = tl.zeros((BT, 16), dtype=tl.float32)
        for c in tl.static_range(NCH):          # 16 fma lanes over d ascending, as the CPU kernel
            d = c * 16 + ln
            qc = tl.load(qb + d, mask=d < D, other=0.0).to(tl.float32)
            kc = tl.load(k + b * skb + hk * skh + t[:, None] * skt + d[None, :],
                         mask=ok[:, None] & (d[None, :] < D), other=0.0).to(tl.float32)
            sacc = tl.fma(kc, qc[None, :], sacc)
        s = _tree16(sacc, BT) * scale
        if HAS_CAP:                             # cap * tanh(s / cap), tanh(y) = 2 sigmoid(2y) - 1 with Vahana's exp
            y = tl.math.div_rn(s, softcap)
            e = _dexp(-2.0 * y, LOG2E, LN2HI, LN2LO, P0, P1, P2, P3, P4, P5)
            sg = tl.math.div_rn(1.0, 1.0 + e)
            s = softcap * (2.0 * sg - 1.0)
        s = tl.where(ok, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        ea = _dexp(m_i - m_new, LOG2E, LN2HI, LN2LO, P0, P1, P2, P3, P4, P5)
        alpha = tl.where(m_new == -float("inf"), 1.0, ea)     # no key seen yet: nothing to rescale
        p = tl.where(ok, _dexp(s - m_new, LOG2E, LN2HI, LN2LO, P0, P1, P2, P3, P4, P5), 0.0)
        vb = tl.load(v + b * svb + hk * svh + t[:, None] * svt + dv[None, :], mask=ok[:, None] & (dv[None, :] < DV),
                     other=0.0).to(tl.float32)
        l_i = tl.fma(l_i, alpha, _tsum64(p))
        acc = tl.fma(acc, alpha, _tsum64_rows(p[:, None] * vb, DVP))
        m_i = m_new
    den = tl.where(l_i > 0, l_i, 1.0)
    out = tl.where(l_i > 0, tl.math.div_rn(acc, tl.zeros((DVP,), tl.float32) + den), 0.0)
    tl.store(o + b * sob + i * soq + h * soh + dv, out.to(o.dtype.element_ty), mask=dv < DV)


def _bounds(mask: torch.Tensor | None, B: int, Q: int, T: int, device):
    """Per (row, query): first and last allowed key index (first > last when nothing is allowed)."""
    if mask is None:
        last = (T - Q + torch.arange(Q, device=device)).expand(B, Q)
        return torch.zeros(B, Q, dtype=torch.int64, device=device), last.to(torch.int64).contiguous()
    m = mask[:, 0].to(torch.bool)
    any_ = m.any(-1)
    idx = torch.arange(T, device=device)
    first = torch.where(m, idx, T).amin(-1)
    last = torch.where(m, idx, -1).amax(-1)
    first = torch.where(any_, first, torch.ones_like(first))
    last = torch.where(any_, last, torch.zeros_like(last))
    return first.to(torch.int64).contiguous(), last.to(torch.int64).contiguous()


def vahana_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, softcap=None,
                     s_aux=None, sliding_window=None, **kwargs):
    """query [B, H, Q, D], key [B, Hkv, T, D], value [B, Hkv, T, DV] -> ([B, Q, H, DV], None).
    GPU: the Triton kernel; CPU: the C++ twin (csrc attention_twin). Both give the same bits."""
    B, H, Q, D = query.shape
    Hkv, T, DV = key.shape[1], key.shape[2], value.shape[-1]
    assert H % Hkv == 0
    scale = scaling if scaling is not None else D ** -0.5
    mask = attention_mask
    if mask is not None and mask.dtype != torch.bool:          # additive float mask -> allowed where finite/0
        mask = mask == 0
    if mask is not None:
        mask = mask[..., :T].expand(B, 1, Q, T).contiguous()
    first, last = _bounds(mask, B, Q, T, query.device)
    mk = mask.to(torch.int8) if mask is not None else None
    if query.device.type == "cpu":
        from . import hybrid as _hy
        if D % 16:                                             # zero lanes, exactly as the GPU kernel's masked loads
            pad = lambda x: torch.nn.functional.pad(x, (0, 16 - D % 16))
            query, key = pad(query), pad(key)
        dt = query.dtype if query.dtype in (torch.bfloat16, torch.float32) else torch.float32
        c = lambda x: (x if x.stride(-1) == 1 else x.contiguous()).to(dt)
        out = _hy.ext().attention_twin(c(query), c(key), c(value), first, last,
                                       mk if mk is not None else torch.empty(0, dtype=torch.int8),
                                       s_aux.float() if s_aux is not None else torch.empty(0),
                                       float(scale), float(softcap or 1.0), bool(softcap))
        return out.to(query.dtype), None
    out = torch.empty(B, Q, H, DV, dtype=query.dtype, device=query.device)
    mkt = mk if mk is not None else first
    sinks = s_aux.float().contiguous() if s_aux is not None else first
    _attn[(B * H * Q,)](
        query, key, value, out, first, last, mkt, sinks, float(scale), float(softcap or 1.0), H, H // Hkv, Q, T,
        query.stride(0), query.stride(1), query.stride(2), key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(1), value.stride(2), out.stride(0), out.stride(1), out.stride(2),
        mkt.stride(0) if mk is not None else 0, mkt.stride(2) if mk is not None else 0,
        D=D, DV=DV, NCH=triton.cdiv(D, 16), DVP=triton.next_power_of_2(DV), BT=BT,
        HAS_MASK=mk is not None, HAS_SINK=s_aux is not None, HAS_CAP=bool(softcap),
        LOG2E=C_LOG2E, LN2HI=C_LN2HI, LN2LO=C_LN2LO, P0=P0, P1=P1, P2=P2, P3=P3, P4=P4, P5=P5,
        num_warps=4, enable_fp_fusion=False)
    return out, None


def _vahana_mask(*args, **kwargs):
    from transformers.masking_utils import sdpa_mask
    kwargs["allow_is_causal_skip"] = False      # always an explicit mask: the kernel never guesses positions
    kwargs["allow_torch_fix"] = False           # fully padded query rows stay empty (their output is 0)
    return sdpa_mask(*args, **kwargs)


def register() -> None:
    from transformers import AttentionInterface
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    AttentionInterface.register("vahana", vahana_attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register("vahana", _vahana_mask)


def use(model) -> None:
    """Route every attention layer of `model` through the vahana function."""
    register()
    model.set_attn_implementation("vahana")
