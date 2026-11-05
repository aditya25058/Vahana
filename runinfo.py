"""Run environment snapshot for measurement records (no GPU work).

Timing on a shared box is only interpretable next to the state it ran in:
code version, foreign GPU memory, host page cache, load. `snapshot()`
captures that cheaply; every field degrades to None instead of raising.
"""
from __future__ import annotations

import os
import platform
import subprocess
from typing import Any


def _git(*args: str) -> str | None:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(["git", "-C", here, *args], capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _meminfo_gb() -> dict[str, float] | None:
    try:
        kv = {}
        with open("/proc/meminfo") as f:
            for line in f:
                p = line.split()
                kv[p[0].rstrip(":")] = int(p[1]) / (1024 ** 2)
        return {k: round(kv[k], 1) for k in ("MemTotal", "MemAvailable", "Cached",
                                             "SwapTotal", "SwapFree") if k in kv}
    except Exception:
        return None


def _gpu() -> dict[str, Any] | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return {"name": torch.cuda.get_device_name(0),
                "total_gb": round(total / 1024 ** 3, 2),
                # Used before this process loads anything = someone else.
                "used_by_others_gb": round((total - free) / 1024 ** 3, 2)}
    except Exception:
        return None


def snapshot() -> dict[str, Any]:
    """Code version + host/GPU state at the moment of the call."""
    import torch
    try:
        import transformers
        tf = transformers.__version__
    except Exception:
        tf = None
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    try:
        load = os.getloadavg()
    except Exception:
        load = None
    return {
        "host": platform.node(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if dirty is None else bool(dirty),
        "torch": torch.__version__,
        "transformers": tf,
        "gpu": _gpu(),
        "mem_gb": _meminfo_gb(),
        "loadavg": load,
    }
