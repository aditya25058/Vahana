"""v1.4: one decode token's CPU expert job per layer, timed at the kernel boundary (submit + wait, CPU tensors ready).

  python kernel_layer.py MODEL_DIR
Prints ms per layer and per token (48 layers) for hot routing (the same 8 experts every time: in RAM) and the
bytes/s that implies, against the machine's read bandwidth.
"""
import os
import sys
import time
from argparse import Namespace

from vahana import cli

ns = Namespace(model=sys.argv[1], gpu_budget=None, reserve=1.5, threads=None, keep_numa_balancing=False,
               device="cpu", gpu_cache="0")
setup = cli._setup(ns)
model, tok, torch = cli._load(ns, setup)
from vahana import hybrid  # noqa: E402
from vahana.experts import VahanaExperts  # noqa: E402

k = hybrid.ext()
layers = [m for m in model.modules() if isinstance(m, VahanaExperts)]
s = layers[0].spec
per = sum(t.numel() * t.element_size() for t in (layers[0].g[0], layers[0].u[0], layers[0].d[0]))
x = (torch.randn(1, s.H) * 0.5).to(torch.bfloat16)
tw = torch.full((1, 8), 0.125)
out = torch.empty(1, s.H, dtype=torch.bfloat16)
ti = torch.arange(8).view(1, 8).to(torch.int64)
reps = int(os.environ.get("REPS", "20"))
if os.environ.get("CLONE") == "1":                  # experts in anonymous RAM instead of mmap'd file pages
    for L in layers:
        L.g, L.u, L.d = ([t.clone() if i < 8 else t for i, t in enumerate(W)] for W in (L.g, L.u, L.d))  # the 8 hot experts
        k.register_layer(L.spec.slot, L.g, L.u, L.d)
    print("experts cloned into anonymous memory", flush=True)
for L in layers:                                   # warm: page in
    k.submit_moe(L.spec.slot, x, ti, tw, out, L.numa)
    k.wait_done()
best = 1e9
for _ in range(reps):
    t = time.perf_counter()
    for L in layers:
        k.submit_moe(L.spec.slot, x, ti, tw, out, L.numa)
        k.wait_done()
    best = min(best, time.perf_counter() - t)
for kk in (1, 2, 4, 8):
    tik = torch.arange(kk).view(1, kk).to(torch.int64)
    twk = torch.full((1, kk), 1.0 / kk)
    b2 = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        for L in layers:
            k.submit_moe(L.spec.slot, x, tik, twk, out, L.numa)
            k.wait_done()
        b2 = min(b2, time.perf_counter() - t)
    print(f"  top-{kk}: {b2 * 1e3 / len(layers):.3f} ms/layer", flush=True)
nb = len(layers) * 8 * per
print(f"threads {os.environ.get('VAHANA_THREADS')}: {best * 1e3:.1f} ms/token ({best * 1e3 / len(layers):.3f} ms/layer), "
      f"{nb / best / 1e9:.1f} GB/s of expert weights", flush=True)
