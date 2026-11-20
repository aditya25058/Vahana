"""P2 unit: CPU twin == GPU twin, bit for bit, for attention and every glue op (random and edge inputs).
Also reports the glue functions' accuracy against float64 torch (max error in float32 ulps).
usage: p2_unit.py OUT.json
"""
import json
import sys

import torch

from vahana import attention as A
from vahana import hybrid as _hy
from vahana import tmath as T
from vahana.twinlin import twin_rmsnorm

out = {"isa": _hy.ext().isa()}
dev = torch.device("cuda")
g = torch.Generator().manual_seed(7)


def rn(*s, scale=1.0, dt=torch.bfloat16):
    return (torch.randn(*s, generator=g) * scale).to(dt)


def _bits(t):
    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32, torch.float64: torch.int64}[t.dtype])


def same(a, b):
    """Bit-identical, except that any NaN equals any NaN (NaN payload bits are not arithmetic)."""
    a, b = a.cpu(), b.cpu()
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if not a.is_floating_point():
        return torch.equal(a, b)
    na, nb = torch.isnan(a), torch.isnan(b)
    return torch.equal(na, nb) and torch.equal(_bits(a)[~na], _bits(b)[~nb])


# ---------------------------------------------------------------- attention
att = []
cases = []
for (B, H, Hkv, Q, Tk, D, DV) in [(2, 4, 2, 1, 37, 64, 64), (2, 4, 2, 7, 130, 64, 64), (1, 4, 4, 70, 70, 64, 32),
                                  (2, 8, 1, 1, 200, 128, 128), (2, 4, 2, 5, 90, 40, 40), (1, 2, 2, 129, 129, 96, 96)]:
    for kind in ("causal", "padded", "sliding", "none"):
        for sink in (False, True):
            for cap in (None, 30.0):
                for dt in (torch.bfloat16, torch.float32):
                    if dt == torch.float32 and (sink or cap):
                        continue
                    cases.append((B, H, Hkv, Q, Tk, D, DV, kind, sink, cap, dt))
for (B, H, Hkv, Q, Tk, D, DV, kind, sink, cap, dt) in cases:
    q, k, v = rn(B, H, Q, D, dt=dt), rn(B, Hkv, Tk, D, dt=dt), rn(B, Hkv, Tk, DV, dt=dt)
    if kind == "none":
        mask = None
    else:
        pos = torch.arange(Tk - Q, Tk).view(Q, 1)
        kk = torch.arange(Tk).view(1, Tk)
        m = kk <= pos
        if kind == "sliding":
            m = m & (kk > pos - 33)
        m = m.view(1, 1, Q, Tk).expand(B, 1, Q, Tk).clone()
        if kind == "padded":
            m[0, :, :, :min(11, Tk - 1)] = False
        mask = m
    s_aux = (torch.randn(H, generator=g) * 2).to(dt) if sink else None
    oc, _ = A.vahana_attention(None, q, k, v, mask, scaling=D ** -0.5, softcap=cap, s_aux=s_aux)
    og, _ = A.vahana_attention(None, q.to(dev), k.to(dev), v.to(dev), mask.to(dev) if mask is not None else None,
                               scaling=D ** -0.5, softcap=cap, s_aux=s_aux.to(dev) if s_aux is not None else None)
    ok = same(oc, og)
    att.append(ok)
    if not ok:
        d = (oc.float() - og.cpu().float()).abs()
        print("ATT MISMATCH", (B, H, Hkv, Q, Tk, D, DV, kind, sink, cap, dt), "nbad", int((d > 0).sum()),
              "max", float(d.max()), flush=True)
out["attention"] = {"cases": len(att), "equal": sum(att)}
print("attention", out["attention"], flush=True)

# noncontiguous q (transposed view, as models pass it) and a strided cache slice
q = rn(2, 9, 4, 64).transpose(1, 2)
k, v = rn(2, 2, 50, 64), rn(2, 2, 50, 64)
oc, _ = A.vahana_attention(None, q, k[:, :, :40], v[:, :, :40], None)
og, _ = A.vahana_attention(None, q.to(dev), k.to(dev)[:, :, :40], v.to(dev)[:, :, :40], None)
out["attention_strided"] = same(oc, og)

# ---------------------------------------------------------------- glue ops
edge = torch.tensor([0.0, -0.0, 1e-30, -1e-30, 1e-8, 0.5, 0.625, 1.0, -1.0, 2.0, 10.0, -10.0, 20.0, -20.0, 40.0,
                     -40.0, 87.0, -87.0, 88.5, -88.5, 100.0, -100.0, 700.0, -750.0, 3.14159265, 1e4, -1e4, 1.3e5,
                     float("inf"), float("-inf"), float("nan")])
xs = torch.cat([torch.randn(4000, generator=g) * s for s in (0.1, 1.0, 5.0, 30.0)] + [edge])
pos = torch.cat([torch.rand(4000, generator=g) * s for s in (1e-3, 1.0, 1e3, 1e6)] + [edge.abs()])
ang = torch.cat([(torch.arange(0, 4096).float()[:, None] * (1.0 / 10000 ** (torch.arange(0, 64, 2) / 64))).flatten(),
                 torch.randn(3000, generator=g) * 1e5, edge])
ops = {
    "exp": (T.texp, xs), "log": (T.tlog, pos), "sin": (T.tsin, ang), "cos": (T.tcos, ang), "tanh": (T.ttanh, xs),
    "sigmoid": (T.tsigmoid, xs), "silu": (T.tsilu, xs), "gelu": (T.tgelu, xs),
    "gelu_tanh": (lambda x: T.tgelu(x, "tanh"), xs), "softplus": (T.tsoftplus, xs),
    "rsqrt": (T.trsqrt, pos), "sqrt": (T.tsqrt, pos), "pow_1.7": (lambda x: T.tpow(x, 1.7), pos),
    "rpow": (lambda x: T.tpow(10000.0, x), torch.arange(0, 64, 2).float() / 64),
}
glue = {}
for name, (f, x) in ops.items():
    for dt in (torch.float32, torch.bfloat16):
        xx = x.to(dt)
        a, b = f(xx), f(xx.to(dev))
        glue[f"{name}_{str(dt)[6:]}"] = same(a, b)
# accuracy vs float64 torch (float32 inputs)
acc = {}
ref = {"exp": torch.exp, "log": torch.log, "sin": torch.sin, "cos": torch.cos, "tanh": torch.tanh,
       "sigmoid": torch.sigmoid, "silu": torch.nn.functional.silu, "sqrt": torch.sqrt}
for name, rf in ref.items():
    f, x = ops[name]
    x = x[torch.isfinite(x)]
    a = f(x).double()
    r = rf(x.double())
    good = torch.isfinite(r) & (r.abs() > 1e-37)
    ulp = (r[good].float().abs() * 2**-23).double().clamp_min(2**-149)
    acc[name] = float(((a[good] - r[good]).abs() / ulp).max())
out["glue_ulp_vs_fp64"] = acc

X = rn(6, 48, 333)
W = rn(5, 333)
for name, f in {
    "sum": lambda x: T.tsum(x.float(), -1), "sum_all": lambda x: T.tsum(x.float()),
    "mean_keep": lambda x: T.tmean(x.float(), (-1, -2), keepdim=True), "sum_bf16": lambda x: T.tsum(x, 1),
    "softmax": lambda x: T.tsoftmax(x, -1), "softmax_f32out": lambda x: T.tsoftmax(x, -1, torch.float32),
    "log_softmax": lambda x: T.tlog_softmax(x, -1),
    "cumsum": lambda x: T.tcumsum(x.float(), -1),
    "matmul": lambda x: T.tmatmul(x.float(), W.float().t().to(x.device)),
    "linear_fp32": lambda x: T.tlinear(x.float(), W.float().to(x.device)),
    "linear_bf16_odd": lambda x: T.tlinear(x, W.to(x.device)),
    "rms_norm": lambda x: T.trms_norm(x, (333,), W[0].to(x.device), 1e-6),
    "layer_norm": lambda x: T.tlayer_norm(x, (333,), W[0].to(x.device), W[1].to(x.device), 1e-5),
    "norm2": lambda x: T.tnorm2(x, -1),
    "topk": lambda x: torch.cat([T.ttopk(x.float().round(), 7, -1).indices.float(),
                                 T.ttopk(x.float().round(), 7, -1).values], -1),
}.items():
    glue[name] = same(f(X), f(X.to(dev)))
# bf16 twin linear (K % 16, R % 4) through tlinear: CPU kernel vs GPU Triton
Wl = rn(96, 512, scale=0.05)
xl = rn(37, 512)
glue["linear_twin_kernels"] = same(T.tlinear(xl, Wl), T.tlinear(xl.to(dev), Wl.to(dev)))
bl = rn(96)
glue["linear_twin_bias"] = same(T.tlinear(xl, Wl, bl), T.tlinear(xl.to(dev), Wl.to(dev), bl.to(dev)))
# model-style scalar division through TwinMode (CUDA would otherwise multiply by the reciprocal)
with T.TwinMode():
    glue["div_scalar_mode"] = same(X.float() / 7.3, X.float().to(dev) / 7.3)
    glue["div_scalar_bf16_mode"] = same(X / 3.1, X.to(dev) / 3.1)
xc = rn(3, 64, 50)
for nm, (w, kw) in {"conv1d_depthwise": (rn(64, 1, 4), dict(padding=3, groups=64)),
                    "conv1d_general": (rn(32, 16, 3), dict(padding=1, groups=4, stride=2)),
                    "conv1d_bias_same": (rn(64, 64, 5), dict(padding="same"))}.items():
    b = rn(w.shape[0]) if "bias" in nm else None
    glue[nm] = same(T.tconv1d(xc, w, b, **kw), T.tconv1d(xc.to(dev), w.to(dev), b.to(dev) if b is not None else None,
                                                         **kw))
ang2 = torch.randn(40, 32, generator=g) * 100
with T.TwinMode():
    fc = torch.polar(torch.ones_like(ang2), ang2)
    fg = torch.polar(torch.ones_like(ang2).to(dev), ang2.to(dev))
    zc = torch.view_as_complex(rn(40, 32, 2, dt=torch.float32))
    glue["polar"] = same(torch.view_as_real(fc), torch.view_as_real(fg))
    glue["complex_mul"] = same(torch.view_as_real(zc * fc), torch.view_as_real(zc.to(dev) * fg))
out["glue"] = glue
print("glue", {k: v for k, v in glue.items() if not v}, "equal", sum(glue.values()), "/", len(glue), flush=True)

# fast GPU RMSNorm == the TwinMode reference (a Llama-style RMSNorm module)


class RMS(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = torch.nn.Parameter(rn(n))
        self.variance_epsilon = 1e-6

    def forward(self, h):
        dt = h.dtype
        h = h.to(torch.float32)
        var = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(var + self.variance_epsilon)
        return self.weight * h.to(dt)


fr = {}
for n in (64, 256, 333, 2048, 7168):
    m = RMS(n)
    x = rn(9, n, scale=3.0)
    with torch.no_grad(), T.TwinMode():
        rc = m(x)
        m.to(dev)
        rg = m(x.to(dev))
    fr[n] = same(rc, rg) and same(rg, twin_rmsnorm(x.to(dev), m.weight, 1e-6))
out["fast_rmsnorm"] = fr
print("fast_rmsnorm", fr, flush=True)
audit = {}
with T.TwinMode(audit):
    y = torch.nn.functional.softmax(X.float(), dim=-1) @ torch.ones(333, 4)
out["audit_smoke"] = audit
out["pass"] = out["attention"]["equal"] == out["attention"]["cases"] and out["attention_strided"] and \
    all(glue.values()) and all(fr.values())
print("acc", acc, "PASS", out["pass"], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
