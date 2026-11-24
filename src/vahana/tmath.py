"""P2: device-twin "glue" arithmetic, so a model gives the same bits on any CPU and any GPU.

The heavy operations already have twins (experts, linears: csrc + Triton; attention: attention.py + csrc). Everything
else a model does between them (norms, rotary cos/sin, router softmax / sigmoid / top-k, activations of dense MLPs,
small fp32 matmuls) runs under `TwinMode`, which replaces each such torch call with a version written only in
operations that IEEE-754 defines exactly (add, sub, mul, div, sqrt, floor, comparisons, casts, bit moves). Each of
those is one correctly rounded result on every device, and each torch call is its own kernel (nothing fuses them),
so the composed result is identical on CPU and GPU by construction.

Definitions (the reference; a fast kernel may replace one only if it is verified bit-identical to it):
  tsum      float sum over a dim: pad to a power of two with zeros, then halve (slot c += slot c + n/2) to one slot.
  mean      tsum / n.  sqrt: float64 sqrt, rounded (= correctly rounded).  rsqrt(x) = 1 / sqrt(x).  x ** 2 = x * x.
  exp, log, sin, cos, tanh: Cephes double-precision algorithms in float64, rounded to the input dtype (accurate to
            ~1 ulp of float32; exp / sin / cos are correctly rounded in practice).
  sigmoid   1 / (1 + exp(-x));  silu = x / (1 + exp(-x));  softmax = exp(x - max) / tsum(...)  (float32).
  matmul    products, then tsum over k (float32).  bf16 linears with K % 16 == 0 and R % 4 == 0 use the twin linear.
  topk      stable sort (ties: lower index first);  sort / argsort always stable.
Ops not listed and not known to be exact are recorded by `TwinMode(audit=...)`, so a new model family shows exactly
which call still needs a twin.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

# ------------------------------------------------------------------ exact building blocks


def _pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def tsum_last(x: torch.Tensor) -> torch.Tensor:
    """Halving-tree sum over the last dim (x float). Returns shape x.shape[:-1]."""
    n = x.shape[-1]
    if n == 0:
        return torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
    p = _pow2(n)
    if p != n:
        x = torch.cat([x, torch.zeros(*x.shape[:-1], p - n, dtype=x.dtype, device=x.device)], -1)
    while x.shape[-1] > 1:
        h = x.shape[-1] // 2
        x = x[..., :h] + x[..., h:]
    return x[..., 0]


def _dims(x, dim):
    if dim is None:
        return list(range(x.dim()))
    if isinstance(dim, int):
        return [dim % max(x.dim(), 1)]
    return sorted(d % max(x.dim(), 1) for d in dim)


def tsum(x: torch.Tensor, dim=None, keepdim=False, dtype=None) -> torch.Tensor:
    out_dt = dtype or x.dtype
    acc = torch.float64 if x.dtype == torch.float64 else torch.float32
    if x.dim() == 0:
        return x.to(out_dt)
    ds = _dims(x, dim)
    keep = [d for d in range(x.dim()) if d not in ds]
    y = x.to(acc).permute(*keep, *ds)
    y = y.reshape(*[x.shape[d] for d in keep], -1)
    r = tsum_last(y)
    if keepdim:
        for d in ds:
            r = r.unsqueeze(d)
    return r.to(out_dt)


def tmean(x, dim=None, keepdim=False, dtype=None):
    ds = _dims(x, dim)
    n = 1
    for d in ds:
        n *= x.shape[d]
    acc = torch.float64 if x.dtype == torch.float64 else torch.float32
    return tdiv(tsum(x, dim, keepdim, acc), n).to(dtype or x.dtype)


def _bits_pow2_f64(n: torch.Tensor) -> torch.Tensor:
    """2**n (n float64 holding an integer in [-1022, 1023]) exactly, via the exponent bits."""
    return ((n.to(torch.int64) + 1023) << 52).view(torch.float64)


def _polevl(x, c):
    y = torch.full_like(x, c[0])
    for a in c[1:]:
        y = y * x + a
    return y


def _p1evl(x, c):
    y = x + c[0]
    for a in c[1:]:
        y = y * x + a
    return y


_EP = (1.26177193074810590878E-4, 3.02994407707441961300E-2, 9.99999999999999999910E-1)
_EQ = (3.00198505138664455042E-6, 2.52448340349684104192E-3, 2.27265548208155028766E-1, 2.00000000000000000009E0)


def exp64(x: torch.Tensor) -> torch.Tensor:
    """Cephes exp in float64 (x float64)."""
    xc = x.clamp(-708.0, 709.0)
    px = torch.floor(xc * 1.4426950408889634073599 + 0.5)
    r = xc - px * 6.93145751953125E-1
    r = r - px * 1.42860682030941723212E-6
    xx = r * r
    pp = r * _polevl(xx, _EP)
    r = pp / (_polevl(xx, _EQ) - pp)
    r = 1.0 + 2.0 * r
    y = r * _bits_pow2_f64(px)
    y = torch.where(x > 709.0, torch.full_like(y, math.inf), y)
    y = torch.where(x < -708.0, torch.zeros_like(y), y)
    return torch.where(x != x, x, y)


_LR = (-7.89580278884799154124E-1, 1.63866645699558079767E1, -6.41409952958715622951E1)
_LS = (-3.56722798256324312549E1, 3.12093766372244180303E2, -7.69691943550460008604E2)


def log64(x: torch.Tensor) -> torch.Tensor:
    """Cephes log in float64 (the rational branch for every exponent)."""
    m, e = torch.frexp(x)
    e = e.to(torch.float64)
    small = m < 0.70710678118654752440
    e = torch.where(small, e - 1.0, e)
    z = torch.where(small, m - 0.5, (m - 0.5) - 0.5)
    y = torch.where(small, 0.5 * z + 0.5, 0.5 * m + 0.5)
    r = z / y
    zz = r * r
    zz = r * (zz * _polevl(zz, _LR) / _p1evl(zz, _LS))
    zz = zz - e * 2.121944400546905827679e-4
    zz = zz + r
    zz = zz + e * 0.693359375
    zz = torch.where(x == 0, torch.full_like(zz, -math.inf), zz)
    zz = torch.where(x < 0, torch.full_like(zz, math.nan), zz)
    zz = torch.where(torch.isinf(x) & (x > 0), x, zz)
    return torch.where(x != x, x, zz)


_SC = (1.58962301576546568060E-10, -2.50507477628578072866E-8, 2.75573136213857245213E-6,
       -1.98412698295895385996E-4, 8.33333333332211858878E-3, -1.66666666666666307295E-1)
_CC = (-1.13585365213876817300E-11, 2.08757008419747316778E-9, -2.75573141792967388112E-7,
       2.48015872888517045348E-5, -1.38888888888730564116E-3, 4.16666666666665929218E-2)


def _sincos64(x: torch.Tensor, cos: bool) -> torch.Tensor:
    """Cephes sin / cos in float64."""
    sign = torch.where(x < 0, -1.0, 1.0).to(torch.float64) if not cos else torch.ones_like(x)
    a = x.abs()
    y = torch.floor(a * 1.27323954473516268615)          # 4 / pi
    j = (y - 8.0 * torch.floor(y * 0.125)).to(torch.int64)
    odd = (j & 1) == 1
    j = torch.where(odd, j + 1, j)
    y = torch.where(odd, y + 1.0, y)
    j = j & 7
    big = j > 3
    j = torch.where(big, j - 4, j)
    sign = torch.where(big, -sign, sign)
    if cos:
        sign = torch.where(j > 1, -sign, sign)
    z = ((a - y * 7.85398125648498535156E-1) - y * 3.77489470793079817668E-8) - y * 2.69515142907905952645E-15
    zz = z * z
    ps = z + z * zz * _polevl(zz, _SC)
    pc = 1.0 - 0.5 * zz + zz * zz * _polevl(zz, _CC)
    use_c = (j == 1) | (j == 2)
    r = torch.where(use_c, ps, pc) if cos else torch.where(use_c, pc, ps)
    r = sign * r
    return torch.where(torch.isfinite(x), r, torch.full_like(r, math.nan))


_TP = (-9.64399179425052238628E-1, -9.92877231001918586564E1, -1.61468768441708447952E3)
_TQ = (1.12811678491632931402E2, 2.23548839060100448583E3, 4.84406305325125486048E3)


def tanh64(x: torch.Tensor) -> torch.Tensor:
    a = x.abs()
    s = exp64(2.0 * a.clamp(max=40.0))
    big = 1.0 - 2.0 / (s + 1.0)
    big = torch.where(x < 0, -big, big)
    z = x * x
    small = x + x * z * (_polevl(z, _TP) / _p1evl(z, _TQ))
    return torch.where(a > 0.625, big, small)


def _f64(x):
    return x.to(torch.float64)


def _out_dt(x):
    return x.dtype if x.is_floating_point() else torch.float32


def _cast(y, dt):
    """float64 -> dt through float32 (CPUs and GPUs round float64 -> bf16 differently; float64 -> float32 -> bf16
    is two exactly defined roundings)."""
    if dt == torch.float64 or y.dtype == dt:
        return y
    return y.to(torch.float32).to(dt)


def tdiv(x: torch.Tensor, d) -> torch.Tensor:
    """x / d with true IEEE division on every device. (CUDA computes tensor / CPU-scalar as tensor * (1 / scalar),
    which rounds twice; a device tensor divisor keeps it one correctly rounded division.)"""
    if isinstance(d, torch.Tensor):
        if d.dim() == 0 and d.device != x.device:
            d = d.to(x.device)
        return torch.div(x, d)
    return torch.div(x, torch.full((), d, dtype=x.dtype if x.is_floating_point() else torch.float32,
                                   device=x.device))


def texp(x):
    return _cast(exp64(_f64(x)), _out_dt(x))


def tlog(x):
    return _cast(log64(_f64(x)), _out_dt(x))


def tsin(x):
    return _cast(_sincos64(_f64(x), False), _out_dt(x))


def tcos(x):
    return _cast(_sincos64(_f64(x), True), _out_dt(x))


def ttanh(x):
    return _cast(tanh64(_f64(x)), _out_dt(x))


def tsigmoid(x):
    return _cast((1.0 / (1.0 + exp64(-_f64(x)))), _out_dt(x))


def tsilu(x):
    d = _f64(x)
    return _cast((d / (1.0 + exp64(-d))), _out_dt(x))


def tsoftplus(x, beta=1.0, threshold=20.0):
    d = _f64(x) * beta
    y = tdiv(log64(1.0 + exp64(d)), float(beta))
    return _cast(torch.where(d > threshold, _f64(x), y), _out_dt(x))


def tgelu(x, approximate="none"):
    d = _f64(x)
    if approximate == "tanh":
        y = 0.5 * d * (1.0 + tanh64(0.7978845608028654 * (d + 0.044715 * d * d * d)))
    else:
        y = 0.5 * d * (1.0 + _erf64(d * 0.7071067811865476))
    return _cast(y, _out_dt(x))


def _erf64(x):
    """erf via Abramowitz-Stegun 7.1.26-style continued exp form is not accurate enough; use erf = 1 - erfc with
    the Cephes rational forms (|x| < 1: T/U; else 1 - exp(-x^2) P/Q)."""
    T = (9.60497373987051638749E0, 9.00260197203842689217E1, 2.23200534594684319226E3, 7.00332514112805075473E3,
         5.55923013010394962768E4)
    U = (3.35617141647503099647E1, 5.21357949780152679795E2, 4.59432382970980127987E3, 2.26290000613890934246E4,
         4.92673942608635921086E4)
    P = (2.46196981473530512524E-10, 5.64189564831068821977E-1, 7.46321056442269912687E0, 4.86371970985681366614E1,
         1.96520832956077098242E2, 5.26445194995477358631E2, 9.34528527171957607540E2, 1.02755188689515710272E3,
         5.57535335369399327526E2)
    Q = (1.32281951154744992508E1, 8.67072140885989742329E1, 3.54937778887819891062E2, 9.75708501743205489753E2,
         1.82390916687909736289E3, 2.24633760818710981792E3, 1.65666309194161350182E3, 5.57535340817727675546E2)
    a = x.abs()
    z = x * x
    small = x * _polevl(z, T) / _p1evl(z, U)
    ac = a.clamp(max=26.0)
    erfc = exp64(-ac * ac) * _polevl(ac, P) / _p1evl(ac, Q)
    big = 1.0 - erfc
    big = torch.where(x < 0, -big, big)
    return torch.where(a < 1.0, small, big)


def tpow(a, b):
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        if b == 2:
            return a * a
        if b == 1:
            return a.clone()
        if b == 0.5:
            return tsqrt(a)
        if b == -0.5:
            return 1.0 / tsqrt(a)
        if b == -1:
            return 1.0 / a
        if b == 3:
            return a * a * a
        return _cast(exp64(b * log64(_f64(a))), _out_dt(a))
    if isinstance(a, (int, float)):
        dt = b.dtype if b.is_floating_point() else torch.float32
        la = log64(torch.tensor(float(a), dtype=torch.float64, device=b.device))
        return _cast(exp64(_f64(b) * la), dt)
    dt = torch.promote_types(a.dtype, b.dtype)
    dt = dt if dt.is_floating_point else torch.float32
    return _cast(exp64(_f64(b) * log64(_f64(a))), dt)


def tsqrt(x):
    """Correctly rounded sqrt on every device: float64 sqrt, then rounded (float64 has more than 2p + 2 bits, so the
    double rounding equals the correctly rounded float32 / bf16 result). PyTorch's CUDA float32 sqrt is not
    correctly rounded, its CPU one is."""
    dt = _out_dt(x)
    return _cast(torch.sqrt(_f64(x)), dt)


def trsqrt(x):
    if x.dtype not in (torch.float32, torch.float64):
        x = x.float()
    return 1.0 / tsqrt(x)


def tsoftmax(x, dim, dtype=None, fused=True):
    out = dtype or x.dtype
    if fused and dim in (-1, x.dim() - 1) and x.is_cuda:     # v0.3-c: one kernel, the same operations (fused.py)
        from .fused import softmax_lastdim
        y = softmax_lastdim(x, out)
        if y is not None:
            return y
    xf = x.float() if x.dtype != torch.float64 else x
    m = xf.amax(dim, keepdim=True)
    e = exp64(_f64(xf - m)).to(xf.dtype)
    s = tsum(e, dim, keepdim=True)
    r = (e / s).to(out)
    return r.masked_fill(r != r, float("nan"))              # one NaN bit pattern on every device (0x7fc00000 / 0x7fc0)


def tlog_softmax(x, dim, dtype=None):
    out = dtype or x.dtype
    xf = x.float() if x.dtype != torch.float64 else x
    m = xf.amax(dim, keepdim=True)
    d = xf - m
    s = tsum(exp64(_f64(d)).to(xf.dtype), dim, keepdim=True)
    return (d - log64(_f64(s)).to(xf.dtype)).to(out)


_CHUNK = 1 << 24          # elements of the product tensor per chunk (rows are independent: chunking is exact)


def tmatmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a [..., M, K] @ b [..., K, N] (or 1-D forms): products, then tsum over k. float32 (float64 stays)."""
    dt = torch.promote_types(a.dtype, b.dtype)
    acc = torch.float64 if dt == torch.float64 else torch.float32
    va, vb = a.dim() == 1, b.dim() == 1
    A = a.unsqueeze(0) if va else a
    Bm = b.unsqueeze(-1) if vb else b
    A, Bm = A.to(acc), Bm.to(acc)
    M, K, N = A.shape[-2], A.shape[-1], Bm.shape[-1]
    lead = torch.broadcast_shapes(A.shape[:-2], Bm.shape[:-2])
    A = A.expand(*lead, M, K)
    Bm = Bm.expand(*lead, K, N)
    bt = Bm.transpose(-1, -2)                               # [..., N, K]
    nb = 1
    for s in lead:
        nb *= s
    step = max(1, _CHUNK // max(1, nb * N * K))
    outs = []
    for m0 in range(0, M, step):
        p = A[..., m0:m0 + step, None, :] * bt[..., None, :, :]     # [..., m, N, K]
        outs.append(tsum_last(p))
    y = torch.cat(outs, -2) if len(outs) > 1 else outs[0]
    if va:
        y = y.squeeze(-2)
    if vb:
        y = y.squeeze(-1)
    return y.to(dt if dt.is_floating_point else acc)


def tlinear(x, w, bias=None):
    """F.linear in twin arithmetic: the twin linear kernels when they apply (bf16, K % 16 == 0, R % 4 == 0; same
    bits on CPU and GPU), otherwise products + tsum. A bias is added after rounding to the output dtype."""
    from . import runtime as _rt
    R, K = w.shape
    shp = x.shape
    if x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16 and K % 16 == 0 and R % 4 == 0:
        y = _rt.twin_linear(x.reshape(-1, K), w).view(*shp[:-1], R)
    else:
        y = tmatmul(x, w.t()).to(torch.promote_types(x.dtype, w.dtype))
    if bias is not None:
        y = (y.float() + bias.float()).to(y.dtype)
    return y


def tcumsum(x, dim, dtype=None):
    """Inclusive scan by doubling (x[i] += x[i - s] for s = 1, 2, 4, ...)."""
    out = dtype or x.dtype
    if not x.is_floating_point() and (dtype is None or not dtype.is_floating_point):
        return torch.cumsum(x, dim, dtype=dtype)
    y = x.to(torch.float64 if x.dtype == torch.float64 else torch.float32).movedim(dim, -1)
    n, s = y.shape[-1], 1
    while s < n:
        y = torch.cat([y[..., :s], y[..., s:] + y[..., :-s]], -1)
        s *= 2
    return y.movedim(-1, dim).to(out)


def trms_norm(x, normalized_shape, weight=None, eps=None):
    dims = list(range(x.dim() - len(normalized_shape), x.dim()))
    xf = x.float()
    e = eps if eps is not None else torch.finfo(x.dtype).eps
    y = xf * trsqrt(tmean(xf * xf, dims, keepdim=True) + e)
    y = y.to(x.dtype)
    return y * weight if weight is not None else y


def tlayer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    dims = list(range(x.dim() - len(normalized_shape), x.dim()))
    xf = x.float()
    mu = tmean(xf, dims, keepdim=True)
    d = xf - mu
    y = d * trsqrt(tmean(d * d, dims, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    if bias is not None:
        y = y + bias.float()
    return y.to(x.dtype)


def tnorm2(x, dim=None, keepdim=False):
    xf = x.float() if x.dtype != torch.float64 else x
    return _cast(tsqrt(tsum(xf * xf, dim, keepdim)), _out_dt(x))


def ttopk(x, k, dim=-1, largest=True, sorted=True):
    v, i = torch.sort(x, dim=dim, descending=largest, stable=True)
    return torch.return_types.topk((v.narrow(dim, 0, k), i.narrow(dim, 0, k)))


# ------------------------------------------------------------------ the mode

def _arg(args, kwargs, i, name, default=None):
    if name in kwargs:
        return kwargs[name]
    return args[i] if len(args) > i else default


def _h_sum(args, kw):
    x = args[0]
    if not x.is_floating_point() and not (kw.get("dtype") is not None and kw["dtype"].is_floating_point):
        return NotImplemented
    dim = _arg(args, kw, 1, "dim")
    return tsum(x, dim, _arg(args, kw, 2, "keepdim", False), kw.get("dtype"))


def _h_mean(args, kw):
    x = args[0]
    return tmean(x, _arg(args, kw, 1, "dim"), _arg(args, kw, 2, "keepdim", False), kw.get("dtype"))


def _h_softmax(args, kw, log=False):
    x = args[0]
    dim = _arg(args, kw, 1, "dim")
    dtype = kw.get("dtype")
    if dtype is None and len(args) > 3 and isinstance(args[3], torch.dtype):
        dtype = args[3]
    if dtype is None and len(args) > 2 and isinstance(args[2], torch.dtype):
        dtype = args[2]
    if dim is None:
        dim = 0 if x.dim() in (0, 1, 3) else 1
    return (tlog_softmax if log else tsoftmax)(x, dim, dtype)


def _h_pow(args, kw):
    a, b = args[0], _arg(args, kw, 1, "exponent")
    if isinstance(a, torch.Tensor) and not a.is_floating_point() and isinstance(b, int):
        return NotImplemented
    return tpow(a, b)


def _h_rpow(args, kw):
    return tpow(args[1], args[0])


def _h_topk(args, kw):
    return ttopk(args[0], _arg(args, kw, 1, "k"), _arg(args, kw, 2, "dim", -1), _arg(args, kw, 3, "largest", True),
                 _arg(args, kw, 4, "sorted", True))


def _h_sort(args, kw):
    x = args[0]
    dim = _arg(args, kw, 1, "dim", -1)
    desc = _arg(args, kw, 2, "descending", False)
    if isinstance(dim, bool):            # sort(input, stable, dim=, descending=) overload
        dim = kw.get("dim", -1)
    return torch.sort(x, dim=dim, descending=desc, stable=True)


def _h_argsort(args, kw):
    x = args[0]
    return torch.sort(x, dim=_arg(args, kw, 1, "dim", -1), descending=_arg(args, kw, 2, "descending", False),
                      stable=True)[1]


def _h_matmul(args, kw):
    a, b = args[0], args[1]
    if not (a.is_floating_point() or b.is_floating_point()):
        return NotImplemented
    return tmatmul(a, b)


def _h_rmatmul(args, kw):
    return tmatmul(args[1], args[0])


def _h_linear(args, kw):
    return tlinear(args[0], _arg(args, kw, 1, "weight"), _arg(args, kw, 2, "bias"))


def _h_addmm(args, kw):
    inp, a, b = args[0], args[1], args[2]
    beta, alpha = kw.get("beta", 1), kw.get("alpha", 1)
    y = tmatmul(a, b)
    if alpha != 1:
        y = y * alpha
    return y + (inp if beta == 1 else inp * beta)


def _h_ew(f):
    return lambda args, kw: f(args[0]) if args[0].is_floating_point() else NotImplemented


def _h_inplace(f):
    def h(args, kw):
        x = args[0]
        return x.copy_(f(x))
    return h


def _h_div(args, kw, inplace=False):
    x, d = args[0], _arg(args, kw, 1, "other")
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        return NotImplemented
    scalar = (isinstance(d, (int, float)) and not isinstance(d, bool)) or         (isinstance(d, torch.Tensor) and d.dim() == 0 and d.device != x.device)
    if not scalar:
        return NotImplemented
    dd = torch.full((), float(d), dtype=x.dtype, device=x.device) if not isinstance(d, torch.Tensor) else         d.to(x.device)
    rm = kw.get("rounding_mode")
    if inplace:
        return x.div_(dd, rounding_mode=rm) if rm else x.div_(dd)
    return torch.div(x, dd, rounding_mode=rm) if rm else torch.div(x, dd)


def _h_cmul(args, kw, inplace=False):
    """complex * complex as separate real operations (a single complex-multiply kernel may fuse a*b + c)."""
    a, b = args[0], _arg(args, kw, 1, "other")
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.is_complex() and b.is_complex()):
        return NotImplemented
    ra, rb = torch.view_as_real(a), torch.view_as_real(b)
    ar, ai, br, bi = ra[..., 0], ra[..., 1], rb[..., 0], rb[..., 1]
    y = torch.complex(ar * br - ai * bi, ar * bi + ai * br)
    return a.copy_(y) if inplace else y


def _h_polar(args, kw):
    ab, ang = args[0], _arg(args, kw, 1, "angle")
    return torch.complex(ab * tcos(ang), ab * tsin(ang))


def tconv1d(x, w, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """conv1d as gathered windows + twin matmul (products, tsum over Cin/groups * K), per output position."""
    st = stride[0] if isinstance(stride, (tuple, list)) else stride
    dl = dilation[0] if isinstance(dilation, (tuple, list)) else dilation
    Cout, Cg, K = w.shape
    span = (K - 1) * dl + 1
    if isinstance(padding, str):
        if padding == "valid":
            pl = pr = 0
        else:                                       # "same" (stride 1)
            tot = span - 1
            pl, pr = tot // 2, tot - tot // 2
    else:
        pl = pr = padding[0] if isinstance(padding, (tuple, list)) else padding
    unb = x.dim() == 2
    if unb:
        x = x.unsqueeze(0)
    B, Cin, _ = x.shape
    xp = torch.cat([torch.zeros(B, Cin, pl, dtype=x.dtype, device=x.device), x,
                    torch.zeros(B, Cin, pr, dtype=x.dtype, device=x.device)], -1)
    win = xp.unfold(-1, span, st)[..., ::dl]                       # [B, Cin, Tout, K]
    Tout = win.shape[2]
    win = win.reshape(B, groups, Cg, Tout, K).permute(0, 1, 3, 2, 4).reshape(B, groups, Tout, Cg * K)
    wg = w.reshape(groups, Cout // groups, Cg * K).transpose(-1, -2)          # [g, Cg*K, Cout/g]
    y = tmatmul(win, wg)                                                      # [B, g, Tout, Cout/g]
    y = y.permute(0, 1, 3, 2).reshape(B, Cout, Tout)
    if bias is not None:
        y = y + bias.float().view(1, Cout, 1)
    y = y.to(x.dtype)
    return y[0] if unb else y


def _h_conv1d(args, kw):
    names = ["input", "weight", "bias", "stride", "padding", "dilation", "groups"]
    defaults = [None, None, None, 1, 0, 1, 1]
    v = [_arg(args, kw, i, n, d) for i, (n, d) in enumerate(zip(names, defaults))]
    if not v[0].is_floating_point():
        return NotImplemented
    return tconv1d(*v)


def _h_cumsum(args, kw):
    return tcumsum(args[0], _arg(args, kw, 1, "dim"), kw.get("dtype"))


def _h_gelu(args, kw):
    return tgelu(args[0], kw.get("approximate", args[1] if len(args) > 1 else "none"))


def _h_softplus(args, kw):
    return tsoftplus(args[0], _arg(args, kw, 1, "beta", 1.0), _arg(args, kw, 2, "threshold", 20.0))


def _h_rms_norm(args, kw):
    return trms_norm(args[0], _arg(args, kw, 1, "normalized_shape"), _arg(args, kw, 2, "weight"),
                     _arg(args, kw, 3, "eps"))


def _h_layer_norm(args, kw):
    return tlayer_norm(args[0], _arg(args, kw, 1, "normalized_shape"), _arg(args, kw, 2, "weight"),
                       _arg(args, kw, 3, "bias"), _arg(args, kw, 4, "eps", 1e-5))


def _h_norm(args, kw):
    x = args[0]
    p = _arg(args, kw, 1, "p", 2) if "ord" not in kw else kw["ord"]
    if p not in (2, 2.0, "fro", None):
        return None
    return tnorm2(x, _arg(args, kw, 2, "dim"), _arg(args, kw, 3, "keepdim", False))


def _h_normalize(args, kw):
    x = args[0]
    p, dim, eps = _arg(args, kw, 1, "p", 2.0), _arg(args, kw, 2, "dim", 1), _arg(args, kw, 3, "eps", 1e-12)
    if p != 2:
        return None
    n = tnorm2(x, dim, True)
    return x / n.clamp_min(eps)


def _h_addsub(sign):
    def h(args, kw):
        alpha = kw.get("alpha", 1)
        if alpha == 1:
            return NotImplemented
        a, b = args[0], args[1]
        return a + sign * (b * alpha)
    return h


def _h_addcmul(args, kw):
    inp, t1, t2 = args[0], args[1], args[2]
    v = kw.get("value", 1)
    return inp + (t1 * t2) * v if v != 1 else inp + t1 * t2


def _h_addcdiv(args, kw):
    inp, t1, t2 = args[0], args[1], args[2]
    v = kw.get("value", 1)
    return inp + (t1 / t2) * v if v != 1 else inp + t1 / t2


def _h_lerp(args, kw):
    a, b, w = args[0], args[1], args[2]
    return a + w * (b - a)


def _h_logsumexp(args, kw):
    x, dim = args[0], _arg(args, kw, 1, "dim")
    keep = _arg(args, kw, 2, "keepdim", False)
    xf = x.float()
    m = xf.amax(dim, keepdim=True)
    s = tsum(texp(xf - m), dim, keepdim=True)
    r = m + tlog(s)
    return (r if keep else r.squeeze(dim)).to(x.dtype)


def _h_var(args, kw, std=False):
    x = args[0]
    dim = _arg(args, kw, 1, "dim")
    corr = kw.get("correction", 0 if kw.get("unbiased") is False else 1)
    keep = kw.get("keepdim", False)
    xf = x.float()
    ds = _dims(x, dim)
    n = 1
    for d in ds:
        n *= x.shape[d]
    d_ = xf - tmean(xf, ds, keepdim=True)
    v = tdiv(tsum(d_ * d_, ds, keep), float(max(n - corr, 1)))
    return (tsqrt(v) if std else v).to(x.dtype)


HANDLERS = {
    "sum": _h_sum, "mean": _h_mean, "nansum": _h_sum,
    "softmax": _h_softmax, "_softmax": _h_softmax, "log_softmax": lambda a, k: _h_softmax(a, k, log=True),
    "pow": _h_pow, "__pow__": _h_pow, "__rpow__": _h_rpow, "float_power": _h_pow,
    "exp": _h_ew(texp), "log": _h_ew(tlog), "sin": _h_ew(tsin), "cos": _h_ew(tcos), "tanh": _h_ew(ttanh),
    "sigmoid": _h_ew(tsigmoid), "silu": _h_ew(tsilu), "rsqrt": lambda a, k: trsqrt(a[0]).to(_out_dt(a[0])),
    "exp_": _h_inplace(texp), "sigmoid_": _h_inplace(tsigmoid), "tanh_": _h_inplace(ttanh),
    "rsqrt_": _h_inplace(lambda x: trsqrt(x).to(x.dtype)),
    "exp2": _h_ew(lambda x: tpow(2.0, x)), "expm1": _h_ew(lambda x: _cast(exp64(_f64(x)) - 1.0, _out_dt(x))),
    "log1p": _h_ew(lambda x: _cast(log64(1.0 + _f64(x)), _out_dt(x))), "log2": _h_ew(
        lambda x: _cast((log64(_f64(x)) * 1.4426950408889634), _out_dt(x))),
    "erf": _h_ew(lambda x: _cast(_erf64(_f64(x)), _out_dt(x))),
    "gelu": _h_gelu, "softplus": _h_softplus, "logsigmoid": _h_ew(lambda x: (-log64(1.0 + exp64(-_f64(x)))).to(
        _out_dt(x))),
    "matmul": _h_matmul, "__matmul__": _h_matmul, "__rmatmul__": _h_rmatmul, "mm": _h_matmul, "bmm": _h_matmul,
    "linear": _h_linear, "addmm": _h_addmm, "baddbmm": _h_addmm,
    "topk": _h_topk, "sort": _h_sort, "argsort": _h_argsort, "cumsum": _h_cumsum,
    "rms_norm": _h_rms_norm, "layer_norm": _h_layer_norm, "norm": _h_norm, "vector_norm": _h_norm,
    "normalize": _h_normalize, "add": _h_addsub(1), "sub": _h_addsub(-1), "add_": None, "sub_": None,
    "addcmul": _h_addcmul, "addcdiv": _h_addcdiv, "lerp": _h_lerp, "logsumexp": _h_logsumexp,
    "var": _h_var, "std": lambda a, k: _h_var(a, k, std=True),
    "sqrt": _h_ew(tsqrt), "sqrt_": _h_inplace(tsqrt), "div": _h_div,
    "mul": _h_cmul, "__mul__": _h_cmul, "__rmul__": _h_cmul, "multiply": _h_cmul,
    "mul_": lambda a, k: _h_cmul(a, k, inplace=True), "__imul__": lambda a, k: _h_cmul(a, k, inplace=True),
    "polar": _h_polar, "conv1d": _h_conv1d, "__truediv__": _h_div, "true_divide": _h_div, "divide": _h_div,
    "div_": lambda a, k: _h_div(a, k, inplace=True), "__itruediv__": lambda a, k: _h_div(a, k, inplace=True),
}
HANDLERS = {k: v for k, v in HANDLERS.items() if v is not None}

# Calls that are exact on every device (pure data movement, integer / boolean work, comparisons, one correctly
# rounded IEEE operation per element, or order-free selections such as max).
EXACT = set("""
__get__ __set__ __getitem__ __setitem__ __len__ __iter__ __contains__ __bool__ __index__ __int__ __float__ __format__
__repr__ __hash__ __eq__ __ne__ __lt__ __le__ __gt__ __ge__ __and__ __or__ __xor__ __invert__ __neg__ __abs__
__add__ __radd__ __iadd__ __sub__ __rsub__ __isub__ __mul__ __rmul__ __imul__ __truediv__ __rtruediv__ __itruediv__
__rdiv__ __rtruediv__ __floordiv__ __rfloordiv__ __mod__ __rmod__ __lshift__ __rshift__ __rand__ __ror__ __ixor__ __iand__ __ior__
__reduce_ex__ __deepcopy__ __setstate__ __array__ __dlpack__ __dlpack_device__ __rlshift__ __rrshift__
add add_ sub sub_ mul mul_ div div_ true_divide neg neg_ abs abs_ reciprocal reciprocal_ square
floor ceil round trunc frac sign signbit clamp clamp_ clamp_min clamp_max clamp_min_ clamp_max_ clip maximum minimum
fmax fmin max min amax amin argmax argmin where masked_fill masked_fill_ masked_scatter masked_select nan_to_num
eq ne lt le gt ge isnan isinf isfinite logical_and logical_or logical_not logical_xor bitwise_and bitwise_or
bitwise_not bitwise_xor bitwise_left_shift bitwise_right_shift any all count_nonzero nonzero argwhere
view view_as reshape reshape_as flatten unflatten squeeze unsqueeze squeeze_ unsqueeze_ permute transpose transpose_
t mT T movedim moveaxis expand expand_as repeat repeat_interleave tile contiguous clone detach detach_ narrow select
index_select gather take take_along_dim scatter scatter_ index_put index_put_ index_copy index_copy_ index_fill
index_fill_ split split_with_sizes chunk tensor_split unbind cat concat concatenate stack hstack vstack flip roll
diagonal triu tril triu_ tril_ to type float double half bfloat16 int long bool short byte char cpu cuda type_as
new_zeros new_ones new_full new_empty new_tensor zeros_like ones_like full_like empty_like rand_like randn_like
zero_ fill_ copy_ size dim numel element_size nelement stride storage_offset is_contiguous is_floating_point
is_complex data_ptr untyped_storage storage requires_grad_ retain_grad item tolist numpy pin_memory share_memory_
record_stream get_device is_pinned dtype device shape layout is_cuda is_meta grad_fn data names ndim
embedding one_hot pad arange linspace zeros ones full empty eye tensor as_tensor from_numpy empty_strided
bincount histc searchsorted bucketize unique unique_consecutive sort_values sum_to_size
dropout feature_alpha_dropout alpha_dropout identity broadcast_to broadcast_tensors meshgrid atleast_1d atleast_2d
cumprod_ relu relu_ relu6 hardtanh hardtanh_ threshold leaky_relu prelu view_as_complex view_as_real complex real imag conj
unfold
scaled_dot_product_attention_unused
""".split())


class TwinMode(TorchFunctionMode):
    """Route a model's glue calls through the twin definitions above. audit: a dict that collects calls that are
    neither handled nor known to be exact (name -> count)."""

    def __init__(self, audit: dict | None = None):
        super().__init__()
        self.audit = audit

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, "__name__", "")
        h = HANDLERS.get(name)
        if h is not None:
            try:
                r = h(args, kwargs)
            except Exception:  # noqa: BLE001 - an unexpected signature: record it, never guess
                r = None
                if self.audit is None:
                    raise
            if r is not NotImplemented and r is not None:
                return r
            if r is None and self.audit is not None:
                self.audit[name + "(unhandled form)"] = self.audit.get(name + "(unhandled form)", 0) + 1
        elif self.audit is not None and name not in EXACT:
            fl = any(isinstance(a, torch.Tensor) and a.is_floating_point() for a in args)
            key = name if fl else name + "(non-float)"
            if fl:
                self.audit[key] = self.audit.get(key, 0) + 1
        return func(*args, **kwargs)


def wrap_forward(model, audit: dict | None = None) -> None:
    """Every call of `model` (forward) runs under TwinMode."""
    import functools
    orig = model.forward

    @functools.wraps(orig)
    def fwd(*a, **k):
        with TwinMode(audit):
            return orig(*a, **k)
    model.forward = fwd
    model._vahana_twin_forward = orig


# ------------------------------------------------------------------ model-level helpers

def twin_rope(model) -> int:
    """Recompute every rotary inv_freq under TwinMode (the stock init uses the CPU's own pow, which differs between
    CPUs), so a laptop and a server start from the same bits."""
    n = 0
    for m in model.modules():
        inv = getattr(m, "inv_freq", None)
        if not isinstance(inv, torch.Tensor) or not hasattr(m, "config"):
            continue
        fn = getattr(m, "rope_init_fn", None) or getattr(m, "compute_default_rope_parameters", None)
        if fn is None:
            continue
        try:
            with TwinMode():
                new, _ = fn(m.config, torch.device("cpu"))
        except Exception:  # noqa: BLE001 - unknown signature: keep the stock buffer (still exact on one machine)
            continue
        if new.shape != inv.shape:
            continue
        with torch.no_grad():
            inv.copy_(new.to(inv.device, inv.dtype))
            if isinstance(getattr(m, "original_inv_freq", None), torch.Tensor):
                m.original_inv_freq = inv.clone()
        n += 1
    return n


def fast_norms(model) -> dict:
    """GPU only: RMSNorm modules whose own forward, run under TwinMode, equals the one-kernel twin RMSNorm bit for bit
    (checked per class on random input) use that kernel. Everything else keeps the TwinMode path."""
    import types
    from .twinlin import twin_rmsnorm
    verdict, n, kept = {}, 0, set()
    for m in model.modules():
        w = getattr(m, "weight", None)
        if not hasattr(m, "variance_epsilon") or not isinstance(w, torch.Tensor) or w.dim() != 1 or not w.is_cuda:
            continue
        cls = type(m)
        if cls not in verdict:
            ok = False
            try:
                g = torch.Generator(device=w.device).manual_seed(0)
                x = (torch.randn(7, w.shape[0], generator=g, device=w.device) * 3).to(w.dtype)
                with torch.no_grad():
                    with TwinMode():
                        ref = m(x)
                    ok = ref.shape == x.shape and ref.dtype == x.dtype and \
                        torch.equal(ref, twin_rmsnorm(x, w, m.variance_epsilon))
            except Exception:  # noqa: BLE001
                ok = False
            verdict[cls] = ok
        if verdict[cls]:
            m.forward = types.MethodType(lambda self, x: twin_rmsnorm(x, self.weight, self.variance_epsilon), m)
            n += 1
        else:
            kept.add(cls.__name__)
    return {"fast": n, "twinmode": sorted(kept)}
