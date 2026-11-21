"""vahana doctor: what this machine is, and which models it runs exactly, how, and at about what speed (P4).

  vahana doctor [MODEL_DIR ...] [--gpu-budget GB] [--no-probe]

Machine: OS, CPU and the kernel build it gets, RAM, GPU, disk. Probes (skipped with --no-probe): memory read
bandwidth, and the sequential read speed of the disk holding each model.
Per model (from config.json and the safetensors headers only, nothing is loaded): non-expert and expert bytes, bytes
an average token reads, and a plan:
  - non-expert weights on the GPU if they fit the budget, else in CPU RAM;
  - experts resident in RAM if they fit what is available, else streamed from disk through the OS page cache (the
    same bytes, so the same bits; only speed changes);
  - an estimated decode speed from the measured bandwidths (a rough guide, not a promise).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import struct
import sys
import time

EXPERT_RE = re.compile(r"\.experts\.|block_sparse_moe\.w[123]$")    # HF per-expert names; Mistral-native fused w1/w2/w3


def _gb(b: float) -> str:
    return f"{b / 2**30:.1f} GB"


def machine(probe: bool = True) -> dict:
    from . import osl
    m = {"os": f"{platform.system()} {platform.release()}", "arch": platform.machine(), "python": sys.version.split()[0],
         "cpu_threads": os.cpu_count() or 1, "cpu": platform.processor() or platform.machine()}
    if sys.platform == "win32" and sys.getwindowsversion().build >= 22000:      # Python reports Windows 11 as 10
        m["os"] = f"Windows 11 (build {sys.getwindowsversion().build})"
    try:
        if sys.platform.startswith("linux"):
            m["cpu"] = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
        elif sys.platform == "win32":
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            m["cpu"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        elif sys.platform == "darwin":
            import subprocess
            m["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                      text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    m["ram_available_gb"] = osl.mem_available_gb()
    m["ram_total_gb"] = _ram_total_gb()
    import torch
    from . import hybrid
    m["torch"] = torch.__version__
    m["isa"] = hybrid.cpu_isa()
    try:
        k = hybrid.ext()
        m["kernel"] = f"{k.isa()} ({os.path.basename(os.path.dirname(k.path))} build), {k.threads()} threads"
    except Exception as e:  # noqa: BLE001
        m["kernel"] = f"NOT AVAILABLE: {e}"
    m["gpu"] = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        from ._triton import HAVE_TRITON
        m["gpu"] = {"name": torch.cuda.get_device_name(0), "total_gb": total / 2**30, "free_gb": free / 2**30,
                    "triton": HAVE_TRITON}
    if probe:
        m["mem_bw_gbs"] = _mem_bw()
    return m


def _ram_total_gb() -> float:
    try:
        if sys.platform.startswith("linux"):
            return int([l for l in open("/proc/meminfo") if l.startswith("MemTotal")][0].split()[1]) / 2**20
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("x", ctypes.c_ulonglong * 5)]
            s = MS()
            s.len = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            return s.total / 2**30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except Exception:  # noqa: BLE001
        return 0.0


def _mem_bw() -> float:
    """Multi-threaded read bandwidth of RAM (GB/s): repeated sums over a 512 MB buffer."""
    import torch
    x = torch.ones(2**27, dtype=torch.float32)          # 512 MB
    x.sum()
    t = time.perf_counter()
    n = 4
    for _ in range(n):
        x.sum()
    return n * x.numel() * 4 / (time.perf_counter() - t) / 1e9


def disk_read_gbs(path: str, limit: int = 2 * 2**30) -> float:
    """Sequential read speed of a file (GB/s). If the file is already in the page cache this reads RAM instead,
    so the result is an upper bound; the first read of a cold file is the honest number."""
    t = time.perf_counter()
    n = 0
    with open(path, "rb", buffering=0) as f:
        while n < limit:
            b = f.read(16 * 2**20)
            if not b:
                break
            n += len(b)
    dt = time.perf_counter() - t
    return n / dt / 1e9 if dt > 0 else 0.0


def model(path: str) -> dict:
    """Sizes from config.json and the safetensors headers (no weights read)."""
    cfg = json.load(open(os.path.join(path, "config.json")))
    tc = cfg.get("text_config", cfg)
    shards = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    E = next((tc.get(k) for k in ("num_experts", "n_routed_experts", "num_local_experts") if tc.get(k)), None)
    exp = non = 0
    for s in shards:
        with open(os.path.join(path, s), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            b = meta["data_offsets"][1] - meta["data_offsets"][0]
            shp = meta.get("shape", [])
            if EXPERT_RE.search(name) or (E and len(shp) == 3 and shp[0] == E):    # per-expert or fused experts
                exp += b
            else:
                non += b
    K = tc.get("num_experts_per_tok") or tc.get("moe_k") or 0
    return {"name": os.path.basename(os.path.normpath(path)), "type": tc.get("model_type", cfg.get("model_type")),
            "layers": tc.get("num_hidden_layers"), "experts": E, "top_k": K, "expert_bytes": exp,
            "nonexpert_bytes": non, "active_expert_bytes": exp * K / E if E else 0.0,
            "shard": os.path.join(path, shards[0]) if shards else None, "disk_free_gb": _disk_free(path)}


def _disk_free(path: str) -> float:
    import shutil
    return shutil.disk_usage(path).free / 2**30


def plan(m: dict, mc: dict, gpu_budget_gb: float | None, disk_gbs: float | None) -> dict:
    """Where everything goes and a rough decode-speed estimate."""
    gpu = mc.get("gpu")
    budget = gpu_budget_gb if gpu_budget_gb is not None else (gpu["free_gb"] - 0.8 if gpu and gpu["triton"] else 0.0)
    non_gb, exp_gb, act_gb = m["nonexpert_bytes"] / 2**30, m["expert_bytes"] / 2**30, m["active_expert_bytes"] / 2**30
    gpu_gb = min(non_gb, budget) if budget > 0 else 0.0          # the loader fills the GPU, the rest goes to RAM
    cpu_gb = non_gb - gpu_gb
    ram = mc["ram_available_gb"] - cpu_gb - 2.0                   # 2 GB for runtime, KV cache, OS
    resident = exp_gb <= ram
    hit = 1.0 if resident else max(0.0, ram) / exp_gb             # page-cache hit rate, uniform routing
    bw = (mc.get("mem_bw_gbs") or 20.0) * 0.6                     # the expert kernel's share of peak
    dbw = disk_gbs or 1.5
    t = act_gb * (hit / bw + (1 - hit) / dbw) + cpu_gb / bw       # CPU reads its non-expert weights every token
    if gpu_gb and not cpu_gb:
        where = "GPU"
    elif gpu_gb:
        where = f"GPU {gpu_gb:.1f} GB + CPU RAM {cpu_gb:.1f} GB"
    else:
        where = "CPU RAM"
    return {"device": "cuda" if gpu_gb else "cpu", "nonexpert_on": where,
            "experts": "resident in RAM" if resident else f"streamed from disk (about {hit:.0%} cached in RAM)",
            "est_tok_s": 1.0 / t if t > 0 else None, "fits_disk": True}


def report(models: list[str], gpu_budget_gb=None, probe=True) -> int:
    mc = machine(probe)
    print("vahana doctor")
    print(f"  system   {mc['os']} {mc['arch']}, Python {mc['python']}, torch {mc['torch']}")
    print(f"  CPU      {mc['cpu']}, {mc['cpu_threads']} threads; kernel: {mc['kernel']}")
    print(f"  RAM      {mc['ram_total_gb']:.1f} GB total, {mc['ram_available_gb']:.1f} GB available"
          + (f", read {mc['mem_bw_gbs']:.0f} GB/s" if mc.get("mem_bw_gbs") else ""))
    g = mc["gpu"]
    if g:
        print(f"  GPU      {g['name']}, {g['total_gb']:.1f} GB ({g['free_gb']:.1f} free), "
              f"triton {'yes' if g['triton'] else 'NO (GPU unused: install triton / triton-windows)'}")
    else:
        print("  GPU      none usable: CPU-only (the same bits as on a GPU, slower)")
    for p in models:
        try:
            m = model(p)
        except Exception as e:  # noqa: BLE001
            print(f"\n  {p}: cannot read ({type(e).__name__}: {e})")
            continue
        dgbs = disk_read_gbs(m["shard"]) if probe and m["shard"] else None
        pl = plan(m, mc, gpu_budget_gb, dgbs)
        print(f"\n  {m['name']} ({m['type']}, {m['layers']} layers, {m['experts']} experts, top-{m['top_k']})")
        print(f"    weights   non-expert {_gb(m['nonexpert_bytes'])}, experts {_gb(m['expert_bytes'])}, "
              f"an average token reads {_gb(m['active_expert_bytes'])} of experts")
        if dgbs:
            print(f"    disk      {dgbs:.2f} GB/s sequential read (an upper bound if the file was cached)")
        print(f"    plan      device={pl['device']}: non-expert in {pl['nonexpert_on']}, experts {pl['experts']}")
        if pl["est_tok_s"]:
            print(f"    estimate  about {pl['est_tok_s']:.1f} tok/s decode (rough; run it to know)")
        print("    exact     yes: every placement gives the same bits (check this machine with `vahana selftest`)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("vahana doctor")
    ap.add_argument("models", nargs="*")
    ap.add_argument("--gpu-budget", type=float, default=None)
    ap.add_argument("--no-probe", action="store_true")
    a = ap.parse_args(argv)
    return report(a.models, a.gpu_budget, not a.no_probe)


if __name__ == "__main__":
    sys.exit(main())
