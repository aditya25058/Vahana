"""v0.3-c: fused twin kernels. Each kernel performs exactly the operations of a pure-torch twin definition (tmath), in
the same order and precision, so its bits equal the definition's on every input (checked by tests and bars). The
point is fewer kernel launches: a router softmax goes from ~40 small kernels to one.
"""
from __future__ import annotations

import torch

from ._triton import HAVE_TRITON, div_rn, tl, triton


@triton.jit
def _c64(v, like):
    """A float64 constant shaped like `like` (exact: no float32 rounding of the Python literal)."""
    return tl.full(like.shape, v, tl.float64)


@triton.jit
def _exp64(x):
    """tmath.exp64, operation for operation (Cephes exp in float64; each op rounds as torch's does)."""
    xc = tl.minimum(tl.maximum(x, _c64(-708.0, x)), _c64(709.0, x))
    px = tl.floor(xc * _c64(1.4426950408889634073599, x) + _c64(0.5, x))
    r = xc - px * _c64(6.93145751953125E-1, x)
    r = r - px * _c64(1.42860682030941723212E-6, x)
    xx = r * r
    p = _c64(1.26177193074810590878E-4, x)                  # polevl(xx, _EP): y = c0; y = y * xx + c
    p = p * xx + _c64(3.02994407707441961300E-2, x)
    p = p * xx + _c64(9.99999999999999999910E-1, x)
    pp = r * p
    q = _c64(3.00198505138664455042E-6, x)                  # polevl(xx, _EQ)
    q = q * xx + _c64(2.52448340349684104192E-3, x)
    q = q * xx + _c64(2.27265548208155028766E-1, x)
    q = q * xx + _c64(2.00000000000000000009E0, x)
    r = pp / (q - pp)                                       # float64: PTX div.rn.f64 (correctly rounded)
    r = _c64(1.0, x) + _c64(2.0, x) * r
    sc = ((px.to(tl.int64) + 1023) << 52).to(tl.float64, bitcast=True)
    y = r * sc
    y = tl.where(x > _c64(709.0, x), _c64(float("inf"), x), y)
    y = tl.where(x < _c64(-708.0, x), _c64(0.0, x), y)
    return tl.where(x != x, x, y)


@triton.jit
def _halve(a, N: tl.constexpr):
    return tl.sum(tl.reshape(a, (2, N // 2)), axis=0)


@triton.jit
def _tsum_pow2(a, BLOCK: tl.constexpr):
    """tmath.tsum_last over a row already padded with zeros to BLOCK (a power of two): halving tree."""
    if BLOCK >= 2:
        a = _halve(a, BLOCK)
    if BLOCK >= 4:
        a = _halve(a, BLOCK // 2)
    if BLOCK >= 8:
        a = _halve(a, BLOCK // 4)
    if BLOCK >= 16:
        a = _halve(a, BLOCK // 8)
    if BLOCK >= 32:
        a = _halve(a, BLOCK // 16)
    if BLOCK >= 64:
        a = _halve(a, BLOCK // 32)
    if BLOCK >= 128:
        a = _halve(a, BLOCK // 64)
    if BLOCK >= 256:
        a = _halve(a, BLOCK // 128)
    if BLOCK >= 512:
        a = _halve(a, BLOCK // 256)
    if BLOCK >= 1024:
        a = _halve(a, BLOCK // 512)
    if BLOCK >= 2048:
        a = _halve(a, BLOCK // 1024)
    if BLOCK >= 4096:
        a = _halve(a, BLOCK // 2048)
    if BLOCK >= 8192:
        a = _halve(a, BLOCK // 4096)
    if BLOCK >= 16384:
        a = _halve(a, BLOCK // 8192)
    if BLOCK >= 32768:
        a = _halve(a, BLOCK // 16384)
    if BLOCK >= 65536:
        a = _halve(a, BLOCK // 32768)
    return tl.sum(a, axis=0)                          # one slot left


@triton.jit
def _tsoftmax_rows(x, y, N, sx, sy, BLOCK: tl.constexpr):
    """tmath.tsoftmax over the last dim, one row per program: fp32 max; (x - max) in fp32 widened to float64; exp64;
    narrowed to fp32; zero padding to BLOCK; halving-tree sum; fp32 division; stored in y's dtype."""
    r = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    v = tl.load(x + r * sx + cols, mask=m, other=-float("inf")).to(tl.float32)
    mx = tl.max(v, axis=0)
    d = (v - mx).to(tl.float64)
    e = _exp64(d).to(tl.float32)
    e = tl.where(m, e, 0.0)
    s = _tsum_pow2(e, BLOCK)
    out = div_rn(e, tl.zeros_like(e) + s)
    o = out.to(y.dtype.element_ty)
    if y.dtype.element_ty == tl.float32:                    # tsoftmax's canonical NaN, after the cast
        qn = tl.full(o.shape, 0x7FC00000, tl.int32).to(tl.float32, bitcast=True)
    else:
        qn = tl.full(o.shape, 0x7FC0 if y.dtype.element_ty == tl.bfloat16 else 0x7E00, tl.int16).to(
            y.dtype.element_ty, bitcast=True)
    o = tl.where(out != out, qn, o)                         # NaN before the cast iff after (fp32 test: bf16 compares fail on Triton 3.0)
    tl.store(y + r * sy + cols, o, mask=m)


def softmax_lastdim(x: torch.Tensor, out_dtype) -> torch.Tensor | None:
    """tmath.tsoftmax(x, -1, out_dtype) in one kernel (CUDA); None when this case is not covered (the caller keeps the
    pure-torch path)."""
    if not (HAVE_TRITON and x.is_cuda and x.dtype in (torch.float32, torch.bfloat16, torch.float16)
            and out_dtype in (torch.float32, torch.bfloat16, torch.float16) and 0 < x.shape[-1] <= 65536):
        return None
    N = x.shape[-1]
    x2 = x.reshape(-1, N)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    y = torch.empty(x2.shape, dtype=out_dtype, device=x.device)
    if x2.shape[0]:
        _tsoftmax_rows[(x2.shape[0],)](x2, y, N, x2.stride(0), y.stride(0), BLOCK=triton.next_power_of_2(N),
                                       num_warps=4 if N <= 2048 else 8, enable_fp_fusion=False)
    return y.view(x.shape)
