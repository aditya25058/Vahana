"""v0.3-c UNIT bar: the fused twin softmax equals the pure-torch twin softmax (GPU) and the CPU twin softmax bit for
bit; the grouped cached-expert kernels equal the per-slot kernels bit for bit.
usage: v3c_unit.py [OUT.json]"""
import json
import sys

import torch

from vahana import cache, tmath

dev = "cuda"
g = torch.Generator().manual_seed(0)
res = {"softmax": [], "experts": []}


def bits(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def rows(n, kind):
    x = torch.randn(9, n, generator=g) * (8.0 if kind == "wide" else 1.0)
    if kind == "edge":
        x[0] = 3.25                                   # all equal
        x[1, ::3] = -float("inf")                     # -inf entries
        x[2] = torch.linspace(-800, 800, n)           # beyond exp's clamp range
        x[3, 0] = 1e30
    return x


ok_all = True
for n in (1, 7, 64, 128, 1000, 50304):
    for kind in ("normal", "wide", "edge"):
        for din in (torch.float32, torch.bfloat16):
            for dout in (torch.float32, torch.bfloat16):
                xc = rows(n, kind).to(din)
                f = tmath.tsoftmax(xc.to(dev), -1, dout)                   # fused
                t = tmath.tsoftmax(xc.to(dev), -1, dout, fused=False)      # pure torch, GPU
                c = tmath.tsoftmax(xc, -1, dout)                            # CPU
                ok = torch.equal(bits(f.cpu()), bits(t.cpu())) and torch.equal(bits(f.cpu()), bits(c))
                ok_all &= ok
                res["softmax"].append({"n": n, "kind": kind, "in": str(din), "out": str(dout), "equal": ok})
                if not ok:
                    print("SOFTMAX MISMATCH", n, kind, din, dout, flush=True)

# grouped experts: random routing with repeats across tokens, some experts uncached
H, I, E, C = 256, 128, 16, 10
for N, K in ((1, 4), (2, 4), (5, 8), (13, 4), (64, 8), (300, 2)):
    x = (torch.randn(N, H, generator=g)).to(dev, torch.bfloat16)
    ti = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(N)]).to(dev)
    ids = sorted(torch.randperm(E, generator=g)[:C].tolist())
    lut = torch.full((E,), -1, dtype=torch.int64)
    for i, e in enumerate(ids):
        lut[e] = i
    lut = lut.to(dev)
    G = (torch.randn(C, I, H, generator=g) * 0.05).to(dev, torch.bfloat16)
    U = (torch.randn(C, I, H, generator=g) * 0.05).to(dev, torch.bfloat16)
    D = (torch.randn(C, H, I, generator=g) * 0.05).to(dev, torch.bfloat16)
    a = cache.gpu_slots(x, ti, lut, G, U, D, grouped=False)
    for mt in (2, 4, 8):
        b = cache.gpu_slots(x, ti, lut, G, U, D, grouped=True, MT=mt)
        ok = torch.equal(bits(a), bits(b))
        ok_all &= ok
        res["experts"].append({"N": N, "K": K, "MT": mt, "equal": ok})
        print("experts", N, K, mt, ok, flush=True)
res["pass"] = ok_all
print("UNIT", "PASS" if ok_all else "FAIL", sum(r["equal"] for r in res["softmax"]), "/", len(res["softmax"]),
      "softmax cases", flush=True)
if len(sys.argv) > 1:
    json.dump(res, open(sys.argv[1], "w"), indent=1)
