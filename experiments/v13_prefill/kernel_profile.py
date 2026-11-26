"""v1.3: CUDA time per kernel for one prefill (torch.profiler).  python kernel_profile.py MODEL_DIR [N]"""
import os
import sys
from argparse import Namespace

from vahana import cli

model_dir, N = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2048
ns = Namespace(model=model_dir, gpu_budget=None, reserve=1.5, threads=None, keep_numa_balancing=False, device=None,
               gpu_cache=os.environ.get("BENCH_GPU_CACHE", "auto"))
setup = cli._setup(ns)
model, tok, torch = cli._load(ns, setup)
from vahana import bench  # noqa: E402

dev = model._vahana_device
ids = tok(bench.TEXT * 200, return_tensors="pt").input_ids[:, :N].to(dev)
with torch.no_grad():
    model(ids[:, :64], use_cache=False, logits_to_keep=1)
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
        model(ids, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=18, max_name_column_width=60))
