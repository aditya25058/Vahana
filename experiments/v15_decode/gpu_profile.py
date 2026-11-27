"""v1.4: GPU time per decode token by kernel (torch.profiler over graph replays), on this machine.

  python gpu_profile.py MODEL_DIR [N_TOKENS]
Prints each kernel's GPU ms per token (sum over its launches / tokens), largest first. `_mb_wait` is the GPU spinning
while the CPU experts run; the rest is GPU work on the critical path or overlapped with the CPU.
"""
import os
import sys
from argparse import Namespace

from vahana import cli

ns = Namespace(model=sys.argv[1], gpu_budget=None, reserve=1.5, threads=None, keep_numa_balancing=False,
               device=os.environ.get("BENCH_DEVICE"), gpu_cache=os.environ.get("BENCH_GPU_CACHE", "auto"))
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
setup = cli._setup(ns)
model, tok, torch = cli._load(ns, setup)
from vahana import bench, decode  # noqa: E402

dev = model._vahana_device
ids = tok(bench.TEXT * 20, return_tensors="pt").input_ids[:, :512].to(dev)
decode.generate(model, ids[:, :32], max_new_tokens=4)
decode.generate(model, ids[:, :8], max_new_tokens=N)
from torch.profiler import ProfilerActivity, profile  # noqa: E402

with profile(activities=[ProfilerActivity.CUDA]) as p:
    r = decode.generate(model, ids[:, :8], max_new_tokens=N)
torch.cuda.synchronize()
st = r["step_ms"]
print(f"steps: mean {sum(st) / len(st):.1f} ms over {len(st)}", flush=True)
rows = []
for e in p.key_averages():
    t = getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
    if t > 0:
        rows.append((t / 1e3 / N, e.count / N, e.key))
rows.sort(reverse=True)
tot = sum(r[0] for r in rows)
print(f"GPU kernel time per token: {tot:.1f} ms (includes an 8-token prefill)")
for ms, c, k in rows[:40]:
    print(f"{ms:8.2f} ms/token {c:7.1f} launches/token  {k[:110]}")
