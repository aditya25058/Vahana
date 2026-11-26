"""v1.3: prefill time and logits, CPU-expert path vs GPU-streamed path, in one process.

  python bench_prefill.py MODEL_DIR [lens=512,2048] [--cpu-first]
"""
import hashlib
import os
import sys
import time
from argparse import Namespace

from vahana import cli

model_dir = sys.argv[1]
lens = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "512,2048").split(",")]
ns = Namespace(model=model_dir, gpu_budget=None, reserve=1.5, threads=None, keep_numa_balancing=False, device=None,
               gpu_cache=os.environ.get("BENCH_GPU_CACHE", "auto"))
setup = cli._setup(ns)
model, tok, torch = cli._load(ns, setup)
from vahana import bench, prefill  # noqa: E402

dev = model._vahana_device
ids_all = tok(bench.TEXT * 200, return_tensors="pt").input_ids.to(dev)


def once(n, mode):
    os.environ["VAHANA_PREFILL_GPU"] = "32" if mode == "gpu" else "0"
    ids = ids_all[:, :n]
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        lg = model(ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    torch.cuda.synchronize()
    return time.perf_counter() - t, hashlib.sha256(lg.cpu().numpy().tobytes()).hexdigest()[:16]


once(64, "gpu")                                     # warm: Triton compiles, staging buffers
once(64, "cpu")
for n in lens:
    if prefill.PROF is not None:
        prefill.PROF.clear()
        prefill.PROF["_last"] = None
    tg, hg = once(n, "gpu")
    print("  profile", prefill.profile_report(), flush=True)
    tc, hc = once(n, "cpu") if os.environ.get("BENCH_SKIP_CPU") != "1" else (float("nan"), "skipped")
    print(f"prompt {n:5d}: gpu-streamed {tg:7.2f} s ({n / tg:7.1f} tok/s)   cpu experts {tc:7.2f} s "
          f"({n / tc:7.1f} tok/s)   speedup {tc / tg:5.2f}x   logits {hg} {'==' if hg == hc else '!='} {hc}",
          flush=True)
print("STATS", prefill.STATS)
