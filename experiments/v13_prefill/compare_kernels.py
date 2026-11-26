"""v1.3: tiled prefill kernels vs the decode kernels (cache.gpu_slots[_q4]) on real expert weights: same bits? time?

  python compare_kernels.py MODEL_DIR [N=512,2048]
Uses one layer's first 32 experts (all slots routed among them, top-8, random routing)."""
import sys
import time
from argparse import Namespace

from vahana import cli

model_dir = sys.argv[1]
Ns = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "512,2048").split(",")]
ns = Namespace(model=model_dir, gpu_budget=None, reserve=1.5, threads=None, keep_numa_balancing=False, device=None,
               gpu_cache="0")
setup = cli._setup(ns)
model, tok, torch = cli._load(ns, setup)
from vahana import prefill_kernels as pk  # noqa: E402
from vahana.cache import gpu_slots, gpu_slots_q4  # noqa: E402
from vahana.experts import VahanaExperts  # noqa: E402

b = [m for m in model.modules() if isinstance(m, VahanaExperts)][10]
s = b.spec
dev = model._vahana_device
C, K = int(__import__('os').environ.get('CK_C', '128')), 8
G, U, D = (torch.stack([w[e] for e in range(C)]).to(dev) for w in (b.g, b.u, b.d))
lut = torch.full((s.E,), -1, dtype=torch.int64, device=dev)
lut[:C] = torch.arange(C, device=dev)


def tm(f, rep=3):
    best = 1e9
    for _ in range(rep):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = f()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t)
    return best, r


for N in Ns:
    g = torch.Generator().manual_seed(N)
    ti = torch.stack([torch.randperm(C, generator=g)[:K] for _ in range(N)]).to(dev)
    x = (torch.randn(N, s.H, generator=g) * 0.5).to(dev, torch.bfloat16)
    ref_t, ref = tm(lambda: gpu_slots_q4(x, ti, lut, G, U, D, s.H, s.I) if b.q4 else gpu_slots(x, ti, lut, G, U, D))
    fl = 2 * N * K * 3 * s.H * s.I
    print(f"N={N}: decode kernels {ref_t * 1e3:.1f} ms ({fl / ref_t / 1e12:.2f} TFLOP/s)", flush=True)
    for BM, BN, nw in ((16, 16, 4), (16, 32, 4), (32, 16, 4), (32, 16, 2), (16, 16, 2), (16, 32, 2), (32, 32, 4)):
        try:
            tt, y = tm(lambda: pk.slots(x, ti, lut, G, U, D, b.q4, s.H, s.I, BM, BN, nw))
            print(f"  tiled BM={BM:2d} BN={BN:2d} nw={nw}: {tt * 1e3:7.1f} ms ({fl / tt / 1e12:.2f} TFLOP/s) "
                  f"{ref_t / tt:5.2f}x  same bits: {torch.equal(y, ref)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  tiled BM={BM} BN={BN}: {type(e).__name__} {str(e)[:120]}")
