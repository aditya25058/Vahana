"""v1.4 tail: how much of decode time is slow steps, how many page faults per token, and whether keeping the
process's working set from being trimmed helps.

  python tail.py MODEL_DIR [--ws-min-gb G]
Windows only. Decodes 96 tokens after a 512-token prompt and prints the step-time distribution, page faults per
token and the working set. --ws-min-gb sets a hard minimum working set first (SetProcessWorkingSetSizeEx), so Windows
does not trim mapped expert pages it already has; it never changes what is computed.
"""
import ctypes
import statistics
import sys
from argparse import Namespace
from ctypes import wintypes

from vahana import cli

model_dir = sys.argv[1]
ws = float(sys.argv[sys.argv.index("--ws-min-gb") + 1]) if "--ws-min-gb" in sys.argv else 0.0
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi")
k32.GetCurrentProcess.restype = wintypes.HANDLE
proc = k32.GetCurrentProcess()


class PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
               [(n, ctypes.c_size_t) for n in ("PeakWS", "WS", "a", "b", "c", "d", "e", "f")]


psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]


def pmc():
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    psapi.GetProcessMemoryInfo(proc, ctypes.byref(c), c.cb)
    return c.PageFaultCount, c.WS


if ws > 0:
    k32.SetProcessWorkingSetSizeEx.argtypes = [wintypes.HANDLE, ctypes.c_size_t, ctypes.c_size_t, wintypes.DWORD]
    ok = k32.SetProcessWorkingSetSizeEx(proc, int(ws * 2**30), int((ws + 4) * 2**30), 0x1 | 0x8)
    print(f"SetProcessWorkingSetSizeEx(min {ws} GB): {'ok' if ok else 'FAILED ' + str(ctypes.get_last_error())}")

ns = Namespace(model=model_dir, gpu_budget=None, reserve=1.5, threads=None, keep_numa_balancing=False, device=None,
               gpu_cache="auto")
setup = cli._setup(ns)
model, tok, torch = cli._load(ns, setup)
from vahana import bench, decode  # noqa: E402

dev = model._vahana_device
ids = tok(bench.TEXT * 20, return_tensors="pt").input_ids[:, :512].to(dev)
decode.generate(model, ids[:, :32], max_new_tokens=4)
f0, _ = pmc()
r = decode.generate(model, ids, max_new_tokens=96)
f1, wsb = pmc()
st = r["step_ms"]
q = sorted(st)
med = statistics.median(st)
slow = [x for x in st if x > 1.5 * med]
extra = sum(slow) - len(slow) * med
print(f"mean {sum(st) / len(st):.1f} ms, p50 {med:.1f}, p90 {q[int(0.9 * len(q))]:.1f}, max {q[-1]:.1f}; "
      f"{len(st) / (sum(st) / 1e3):.2f} tok/s; faults/token {(f1 - f0) / 96:.0f} (incl. prefill); "
      f"working set {wsb / 2**30:.1f} GB; slow steps {len(slow)}/{len(st)} cost {extra:.0f} ms "
      f"({extra / sum(st):.0%})", flush=True)
print("slow step indices and ms:", [(i, round(x)) for i, x in enumerate(st) if x > 1.5 * med], flush=True)
