"""Batch-, padding- and query-length-invariant kernels (VAHANA_INVARIANT=1).

Goal: every token's computation gives the same bits no matter which other rows share the batch,
how much left padding its row has, or how many query positions are processed at once (prefill,
decode, speculative verify). Then any scheduling freedom (batch size, splitting, speculation)
keeps outputs bit-exact. In offload inference the step is link-bound, so the extra compute of
fixed-order kernels is hidden (Phase 21b: +25% on expert GEMMs, ~0.2 s of a ~8 s step).

Rules followed by every kernel here:
  - each output element is produced by one program with a reduction order that depends only on
    fixed constexpr tile sizes and the reduction length, never on M / batch / grid size;
  - no split-K, no atomics, fp32 accumulation;
  - attention reduces keys in fixed blocks anchored at the row's FIRST REAL token, so left
    padding and the number of queries do not move block boundaries.
This defines a new reference (not bit-equal to cuBLAS), deterministic and invariant.
"""
from __future__ import annotations

import os as _os

import torch
import triton
import triton.language as tl

_FAST_GATE = _os.environ.get("VAHANA_FAST_GATE") == "1"


# --------------------------------------------------------------------------- matmul
@triton.jit
def _bmm_kernel(a, b, c, M, N, K,
                sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, IEEE: tl.constexpr):
    pb = tl.program_id(2).to(tl.int64)          # 64-bit offsets: G*M*N can exceed 2**31
    pm = tl.program_id(0).to(tl.int64)
    pn = tl.program_id(1).to(tl.int64)
    rm = pm * BM + tl.arange(0, BM).to(tl.int64)
    rn = pn * BN + tl.arange(0, BN).to(tl.int64)
    rk = tl.arange(0, BK)
    a += pb * sab
    b += pb * sbb
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ak = k0 + rk
        x = tl.load(a + rm[:, None] * sam + ak[None, :] * sak,
                    mask=(rm[:, None] < M) & (ak[None, :] < K), other=0.0)
        w = tl.load(b + ak[:, None] * sbk + rn[None, :] * sbn,
                    mask=(ak[:, None] < K) & (rn[None, :] < N), other=0.0)
        if IEEE:
            acc += tl.dot(x, w, input_precision="ieee")
        else:
            acc += tl.dot(x, w)
    c += pb * scb
    tl.store(c + rm[:, None] * scm + rn[None, :] * scn, acc.to(c.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def inv_bmm(a: torch.Tensor, b: torch.Tensor, out_dtype=None) -> torch.Tensor:
    """[G, M, K] @ [G, K, N] -> [G, M, N]; row results independent of M and G."""
    G, M, K = a.shape
    N = b.shape[2]
    c = torch.empty((G, M, N), dtype=out_dtype or a.dtype, device=a.device)
    if M == 0:
        return c
    ieee = a.dtype == torch.float32
    BM, BN, BK = (32, 64, 32) if ieee else (64, 128, 64)
    ns = 3
    if ieee and _FAST_GATE and N <= 256:        # Phase 41: same BK / FMA order, more programs in flight
        BN, ns = 16, 6
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN), G)
    _bmm_kernel[grid](a, b, c, M, N, K,
                      a.stride(0), a.stride(1), a.stride(2), b.stride(0), b.stride(1), b.stride(2),
                      c.stride(0), c.stride(1), c.stride(2),
                      BM=BM, BN=BN, BK=BK, IEEE=ieee, num_warps=4, num_stages=ns)
    return c


def inv_linear(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """F.linear(x, w, bias) with fixed-order reduction (any leading dims)."""
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if w.dtype != x2.dtype:
        w = w.to(x2.dtype)
    y = inv_bmm(x2.unsqueeze(0), w.t().unsqueeze(0))[0]
    if bias is not None:
        y = y + bias
    return y.view(*lead, w.shape[0])


# --------------------------------------------------------------------------- RMSNorm
@triton.jit
def _rms_kernel(x, w, y, N, sx, sy, eps, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    v = tl.load(x + r * sx + cols, mask=m, other=0.0).to(tl.float32)
    var = tl.sum(v * v, axis=0) / N
    h = (v * tl.math.rsqrt(var + eps)).to(y.dtype.element_ty)       # stock: cast, then * weight
    ww = tl.load(w + cols, mask=m, other=0.0)
    tl.store(y + r * sy + cols, (ww * h).to(y.dtype.element_ty), mask=m)


def inv_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    out_dtype = torch.promote_types(weight.dtype, x.dtype)
    y = torch.empty(x2.shape, dtype=out_dtype, device=x.device)
    if x2.shape[0]:
        _rms_kernel[(x2.shape[0],)](x2, weight, y, x2.shape[1], x2.stride(0), y.stride(0), eps,
                                    BLOCK=triton.next_power_of_2(x2.shape[1]), num_warps=8)
    return y.view(shape)


# --------------------------------------------------------------------------- MLA attention
@triton.jit
def _mla_kernel(ql, qp, lat, kpe, out, first, basep, rowmap, Q, H, T, scale,
                sqb, sqq, sqh, spb, spq, sph, slb, slt, skb, skt, sob, soq, soh,
                R: tl.constexpr, RP: tl.constexpr, BH: tl.constexpr, BT: tl.constexpr):
    bq = tl.program_id(0).to(tl.int64)
    hb = tl.program_id(1).to(tl.int64)
    b = bq // Q
    q = bq % Q
    bc = tl.load(rowmap + b).to(tl.int64)   # cache row (slot) of query row b
    hs = hb * BH + tl.arange(0, BH)
    hm = hs < H
    rr = tl.arange(0, R)
    rp = tl.arange(0, RP)
    qlat = tl.load(ql + b * sqb + q * sqq + hs[:, None] * sqh + rr[None, :], mask=hm[:, None], other=0.0)
    qpe = tl.load(qp + b * spb + q * spq + hs[:, None] * sph + rp[None, :], mask=hm[:, None], other=0.0)
    start = tl.load(first + bc).to(tl.int64)
    end = tl.load(basep + bc).to(tl.int64) + q        # causal: last key this query may see (inclusive)
    m_i = tl.full((BH,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BH,), dtype=tl.float32)
    acc = tl.zeros((BH, R), dtype=tl.float32)
    tt = tl.arange(0, BT)
    for t0 in range(start, end + 1, BT):     # blocks anchored at the first real token
        t = t0 + tt
        tm = t <= end
        kl = tl.load(lat + bc * slb + t[:, None] * slt + rr[None, :], mask=tm[:, None], other=0.0)
        kp = tl.load(kpe + bc * skb + t[:, None] * skt + rp[None, :], mask=tm[:, None], other=0.0)
        s = tl.dot(qlat, tl.trans(kl)) + tl.dot(qpe, tl.trans(kp))
        s = s * scale
        s = tl.where(tm[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kl.dtype), kl)
        m_i = m_new
    l_i = tl.where(l_i > 0, l_i, 1.0)     # pad-only query (no real key yet): output 0, not NaN
    o = acc / l_i[:, None]
    tl.store(out + b * sob + q * soq + hs[:, None] * soh + rr[None, :], o.to(out.dtype.element_ty),
             mask=hm[:, None])


def inv_mla_attention(q_lat, q_pe, latent, kpe, first_valid, softmax_scale, base=None, rows=None):
    """rows: cache row (slot) for each query row (default: identity). first_valid / base are
    indexed by cache row."""
    """q_lat [B, Q, H, R], q_pe [B, Q, H, RP], latent [B, T, R], kpe [B, T, RP], first_valid [B]
    (int32 index of each row's first real key). Query q sits at key index T-Q+q (causal).
    Returns ctx [B, Q, H, R] = softmax(q.k) @ latent, per query independent of B, Q and padding."""
    B, Q, H, R = q_lat.shape
    RP = q_pe.shape[-1]
    T = latent.shape[1]
    q_lat, q_pe = q_lat.contiguous(), q_pe.contiguous()
    latent, kpe = latent.contiguous(), kpe.contiguous()
    out = torch.empty((B, Q, H, R), dtype=latent.dtype, device=latent.device)
    if rows is None:
        rows = torch.arange(B, device=latent.device)
    if base is None:                     # right-aligned cache: query 0 sits at key T-Q
        base = torch.full((B,), T - Q, dtype=torch.int32, device=latent.device)
    BH, BT = 16, 32
    grid = (B * Q, triton.cdiv(H, BH))
    _mla_kernel[grid](q_lat, q_pe, latent, kpe, out, first_valid.to(torch.int32).contiguous(),
                      base.to(torch.int32).contiguous(), rows.to(torch.int32).contiguous(),
                      Q, H, T, softmax_scale,
                      q_lat.stride(0), q_lat.stride(1), q_lat.stride(2),
                      q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
                      latent.stride(0), latent.stride(1), kpe.stride(0), kpe.stride(1),
                      out.stride(0), out.stride(1), out.stride(2),
                      R=R, RP=RP, BH=BH, BT=BT, num_warps=8, num_stages=2)
    return out


# --------------------------------------------------------------------------- install
_INSTALLED = False


def enabled() -> bool:
    import os
    return os.environ.get("VAHANA_INVARIANT", "0") == "1"


def install(model: torch.nn.Module) -> dict:
    """Route linears, RMSNorms and the MoE gate's F.linear through invariant kernels.
    Structural: nn.Linear (class-level, so expert slots / pools created later are covered),
    modules exposing `weight` + `variance_epsilon` (RMSNorm), and the modeling module's `F`."""
    global _INSTALLED
    import sys
    import types
    import torch.nn.functional as F
    n_norm, mods = 0, set()
    if not _INSTALLED:
        def _lin_forward(self, x):
            return inv_linear(x, self.weight, self.bias)
        torch.nn.Linear.forward = _lin_forward
        _INSTALLED = True
    for m in model.modules():
        if hasattr(m, "variance_epsilon") and isinstance(getattr(m, "weight", None), torch.Tensor):
            m.forward = types.MethodType(lambda self, x: inv_rmsnorm(x, self.weight, self.variance_epsilon), m)
            n_norm += 1
        mods.add(type(m).__module__)
    shimmed = 0
    for name in mods:
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "F", None) is F:
            shim = types.SimpleNamespace(**{k: getattr(F, k) for k in dir(F) if not k.startswith("__")})
            shim.linear = lambda x, w, b=None: inv_linear(x, w, b)
            mod.F = shim
            shimmed += 1
    return {"rmsnorm": n_norm, "f_shimmed_modules": shimmed, "linear": "class-level"}


def fast_gates(model: torch.nn.Module) -> dict:
    """Phase 41: MoEGate weights -> bf16 views of contiguous W^T (same values, same memory)."""
    n, nbytes = 0, 0
    for m in model.modules():
        if type(m).__name__ != "MoEGate":
            continue
        w = getattr(m, "weight", None)
        if not isinstance(w, torch.Tensor) or not w.is_cuda or w.dim() != 2:
            continue
        # bf16 view of a contiguous W^T: no extra memory (no room for fp32 copies in a 16 GB budget).
        # weight.type(float32) in the gate preserves strides, so inv_linear still gets a contiguous [K, N].
        w.data = w.detach().t().contiguous().t()
        n += 1
        nbytes += w.numel() * w.element_size()
    return {"gates": n, "mb": nbytes / 2**20}
