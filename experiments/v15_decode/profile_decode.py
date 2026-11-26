"""v1.4: where a decode step goes on this machine.

  python profile_decode.py MODEL_DIR
1. Real decode: 64 tokens after a 512-token prompt (step ms, p50).
2. The CPU expert kernel alone, per layer, for one token: hot (same routing repeated: weights in cache) and cold-ish
   (random routing over all experts), summed over layers = expert ms per token.
3. Process page faults per decode token (Windows: PageFaultCount; Linux: majflt).
"""
import ctypes
import os
import statistics
import sys
import time
from argparse import Namespace

from vahana import cli

model_dir = sys.argv[1]
ns = Namespace(model=model_dir, gpu_budget=None, reserve=1.5, threads=None, keep_numa_balancing=False,
               device=os.environ.get("BENCH_DEVICE"),
               gpu_cache=os.environ.get("BENCH_GPU_CACHE", "auto"))
setup = cli._setup(ns)
model, tok, torch = cli._load(ns, setup)
from vahana import bench, decode, runtime  # noqa: E402
from vahana.cache import CachedExperts  # noqa: E402
from vahana.experts import VahanaExperts  # noqa: E402


def faults():
    if sys.platform == "win32":
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
                       [(n, ctypes.c_size_t) for n in ("a", "b", "c", "d", "e", "f", "g", "h")]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        k = ctypes.WinDLL("kernel32")
        k.GetCurrentProcess.restype = wintypes.HANDLE
        p = ctypes.WinDLL("psapi")
        p.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        p.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PageFaultCount
    return int(open("/proc/self/stat").read().split()[11])


dev = model._vahana_device
if os.environ.get("PROFILE_SKIP_DECODE") != "1":
    ids = tok(bench.TEXT * 20, return_tensors="pt").input_ids[:, :512].to(dev)
    decode.generate(model, ids[:, :32], max_new_tokens=4)
    f0 = faults()
    r = decode.generate(model, ids, max_new_tokens=64)
    f1 = faults()
    st = r["step_ms"]
    print(f"decode: {len(st) / (sum(st) / 1e3):.2f} tok/s, p50 {statistics.median(st):.1f} ms/step, "
          f"page faults/token {(f1 - f0) / 64:.0f} (incl. prefill)", flush=True)

if os.environ.get("PROFILE_DECODE_ONLY") == "1":
    sys.exit(0)
layers = [m.base if isinstance(m, CachedExperts) else m for m in model.modules()
          if isinstance(m, (VahanaExperts, CachedExperts))]
cached = sum(isinstance(m, CachedExperts) for m in model.modules())
s = layers[0].spec
print(f"{len(layers)} expert layers ({cached} with GPU-cached experts), E={s.E} K=8 H={s.H} I={s.I}", flush=True)
g = torch.Generator().manual_seed(0)
x = (torch.randn(1, s.H, generator=g) * 0.5).to(torch.bfloat16)
tw = torch.full((1, 8), 0.125)
saved = runtime.GRAPH["on"]
runtime.GRAPH["on"] = False                       # the eager CPU path: runtime.moe straight to the kernel
for mode in ("hot", "cold"):
    tot = []
    for rep in range(int(os.environ.get("PROFILE_REPS", "3"))):
        t_all = 0.0
        for L in layers:
            ti = torch.arange(8).view(1, 8) if mode == "hot" else torch.randperm(s.E, generator=g)[:8].view(1, 8)
            t = time.perf_counter()
            runtime.moe(L.spec.slot, x, ti, tw, L.numa)
            t_all += time.perf_counter() - t
        tot.append(t_all * 1e3)
    print(f"CPU experts alone, {mode}: {min(tot):.1f} ms/token (runs {[round(v, 1) for v in tot]})", flush=True)
runtime.GRAPH["on"] = saved
f0 = faults()
t = time.perf_counter()
for L in layers:
    runtime.GRAPH["on"] = False
    for _ in range(4):
        runtime.moe(L.spec.slot, x, torch.randperm(s.E, generator=g)[:8].view(1, 8), tw, L.numa)
runtime.GRAPH["on"] = saved
print(f"cold routing: page faults per layer-call {(faults() - f0) / (4 * len(layers)):.0f}", flush=True)
