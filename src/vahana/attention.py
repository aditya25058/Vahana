"""The `vahana` attention function: batch-, padding- and query-length-invariant attention for any
transformers model that calls `attention_interface(module, q, k, v, mask, scaling=..., sliding_window=...,
s_aux=..., softcap=...)`.

Each (row, head, query) is one program with a fixed reduction order that depends only on that query's own
allowed key set:
  - keys are visited in fixed blocks of BT, anchored at the query's first allowed key (so left padding, the
    batch it sits in, and whether it is computed in a prefill or a decode step never move block boundaries);
  - scores are fp32 dot products over the head dimension (fixed tree), online softmax over blocks in order;
  - attention sinks (s_aux, gpt-oss) are a virtual key visited first;
  - optional logit soft-capping.
GQA / MQA by head grouping; the value head size may differ from the query/key size (MLA expanded form).
The mask is always an explicit boolean [B, 1, Q, T] (allowed = True); sliding windows arrive inside it.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

BT = 64


@triton.jit
def _attn(q, k, v, o, first, last, mask, sinks, scale, softcap, H, G, Q, T,
          sqb, sqh, sqq, skb, skh, skt, svb, svh, svt, sob, soq, soh, smb, smq,
          D: tl.constexpr, DV: tl.constexpr, DP: tl.constexpr, DVP: tl.constexpr, BT: tl.constexpr,
          HAS_MASK: tl.constexpr, HAS_SINK: tl.constexpr, HAS_CAP: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    i = pid % Q
    h = (pid // Q) % H
    b = pid // (Q * H)
    hk = h // G
    dd = tl.arange(0, DP)
    dv = tl.arange(0, DVP)
    qv = tl.load(q + b * sqb + h * sqh + i * sqq + dd, mask=dd < D, other=0.0).to(tl.float32)
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
            ok = ok & (tl.load(mask + b * smb + i * smq + t, mask=t < T, other=0) != 0)
        kb = tl.load(k + b * skb + hk * skh + t[:, None] * skt + dd[None, :], mask=ok[:, None] & (dd[None, :] < D),
                     other=0.0).to(tl.float32)
        s = tl.sum(kb * qv[None, :], axis=1) * scale
        if HAS_CAP:
            s = softcap * (2.0 * tl.sigmoid(2.0 * (s / softcap)) - 1.0)     # tanh, written exactly the same way always
        s = tl.where(ok, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.where(m_new == -float("inf"), 1.0, tl.exp(m_i - m_new))     # no key seen yet: nothing to rescale
        p = tl.where(ok, tl.exp(s - m_new), 0.0)
        vb = tl.load(v + b * svb + hk * svh + t[:, None] * svt + dv[None, :], mask=ok[:, None] & (dv[None, :] < DV),
                     other=0.0).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * vb, axis=0)
        m_i = m_new
    out = tl.where(l_i > 0, acc / l_i, 0.0)
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
    """query [B, H, Q, D], key [B, Hkv, T, D], value [B, Hkv, T, DV] -> ([B, Q, H, DV], None)."""
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
    out = torch.empty(B, Q, H, DV, dtype=query.dtype, device=query.device)
    mk = mask.to(torch.int8) if mask is not None else first
    sinks = s_aux.float().contiguous() if s_aux is not None else first
    _attn[(B * H * Q,)](
        query, key, value, out, first, last, mk, sinks, float(scale), float(softcap or 1.0), H, H // Hkv, Q, T,
        query.stride(0), query.stride(1), query.stride(2), key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(1), value.stride(2), out.stride(0), out.stride(1), out.stride(2),
        mk.stride(0) if mask is not None else 0, mk.stride(2) if mask is not None else 0,
        D=D, DV=DV, DP=triton.next_power_of_2(D), DVP=triton.next_power_of_2(DV), BT=BT,
        HAS_MASK=mask is not None, HAS_SINK=s_aux is not None, HAS_CAP=bool(softcap), num_warps=4)
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
