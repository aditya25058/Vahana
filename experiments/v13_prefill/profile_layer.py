"""v1.3: where a streamed prefill layer's time goes: gather (CPU, from RAM or the mmap'd cache), host->GPU copy,
GPU twin kernels. One layer, N prompt tokens with uniform-random routing (top-K of E)."""
import os
import sys
import time
from argparse import Namespace

from vahana import cli

model_dir, N = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 512
ns = Namespace(model=model_dir, gpu_budget=None, reserve=1.5, threads=None, keep_numa_balancing=False, device=None,
               gpu_cache="0")
setup = cli._setup(ns)
model, tok, torch = cli._load(ns, setup)
from vahana.cache import gpu_slots, gpu_slots_q4  # noqa: E402
from vahana.experts import VahanaExperts  # noqa: E402

layers = [m for m in model.modules() if isinstance(m, VahanaExperts)]
b = layers[len(layers) // 2]
s = b.spec
dev = model._vahana_device
E, K = s.E, 8
ids = list(range(E))
per = sum(t.numel() * t.element_size() for t in (b.g[0], b.u[0], b.d[0]))
print(f"layer: E={E} H={s.H} I={s.I} q4={b.q4} expert {per / 2**20:.2f} MB, layer {E * per / 2**30:.2f} GB")
sync = torch.cuda.synchronize


def t(f, rep=3):
    best = 1e9
    for _ in range(rep):
        sync()
        t0 = time.perf_counter()
        r = f()
        sync()
        best = min(best, time.perf_counter() - t0)
    return best, r


cap = 32
hb = [torch.empty((cap, *w[0].shape), dtype=w[0].dtype, pin_memory=True) for w in (b.g, b.u, b.d)]
gb = [torch.empty_like(h, device=dev) for h in hb]
L2 = layers[len(layers) // 2 + 1]                     # a layer not read yet (cold pages) for the first gather


def gather(lay, e0):
    for h, ws in zip(hb, (lay.g, lay.u, lay.d)):
        torch.stack([ws[e] for e in range(e0, e0 + cap)], out=h)


t0 = time.perf_counter()
gather(L2, 0)
cold = time.perf_counter() - t0
warm, _ = t(lambda: gather(b, 0))
h2d, _ = t(lambda: [g.copy_(h, non_blocking=True) for g, h in zip(gb, hb)])
big = torch.empty(cb if (cb := cap * per) else 1, dtype=torch.uint8, pin_memory=True)
bigg = torch.empty_like(big, device=dev)
h2d1, _ = t(lambda: bigg.copy_(big, non_blocking=True))
gb2 = [torch.empty(h.shape, dtype=h.dtype, device=dev) for h in hb]
h2d2, _ = t(lambda: [g.copy_(h, non_blocking=True) for g, h in zip(gb2, hb)])
h2d3, _ = t(lambda: [g.copy_(h, non_blocking=True) for g, h in zip(gb, hb)])
flat = [bigg[o:o + h.numel()].view(h.shape) for o, h in zip((0, hb[0].numel(), 2 * hb[0].numel()), hb)]
h2d4, _ = t(lambda: [g.copy_(h, non_blocking=True) for g, h in zip(flat, hb)])
print("fresh", round(cb / h2d2 / 1e9, 2), "old again", round(cb / h2d3 / 1e9, 2), "into flat", round(cb / h2d4 / 1e9, 2),
      [h.shape for h in hb], [g.is_contiguous() for g in gb], [h.is_contiguous() for h in hb])
print(f"GPU free {torch.cuda.mem_get_info()[0] / 2**30:.2f} GB; one flat {cb / 2**20:.0f} MB copy "
      f"{cb / h2d1 / 1e9:.2f} GB/s; pinned {[h.is_pinned() for h in hb]}")
g = torch.Generator().manual_seed(0)
ti = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(N)]).to(dev)
x = (torch.randn(N, s.H, generator=g) * 0.5).to(dev, torch.bfloat16)
lut = torch.full((E,), -1, dtype=torch.int64, device=dev)
lut[:cap] = torch.arange(cap, device=dev)
if b.q4:
    kern, _ = t(lambda: gpu_slots_q4(x, ti, lut, *gb, s.H, s.I))
else:
    kern, _ = t(lambda: gpu_slots(x, ti, lut, *gb))
sc = E / cap
cb = cap * per
print(f"per {cap} experts: gather cold {cold * 1e3:.0f} ms ({cb / cold / 1e9:.2f} GB/s), warm {warm * 1e3:.0f} ms "
      f"({cb / warm / 1e9:.2f} GB/s); H2D {h2d * 1e3:.0f} ms ({cb / h2d / 1e9:.2f} GB/s); kernels {kern * 1e3:.0f} ms")
print(f"per layer (x{sc:.0f}): gather cold {cold * sc:.2f} s / warm {warm * sc:.2f} s, H2D {h2d * sc:.2f} s, "
      f"kernels {kern * sc:.2f} s; x{len(layers)} layers: cold {cold * sc * len(layers):.1f} s, "
      f"H2D {h2d * sc * len(layers):.1f} s, kernels {kern * sc * len(layers):.1f} s")
