"""OS layer (P3): the few machine facts Vahana needs, on Linux, Windows and macOS. Nothing here affects arithmetic."""
from __future__ import annotations

import os
import subprocess
import sys


def mem_available_gb() -> float:
    """Memory the OS can give us now (GB)."""
    if sys.platform.startswith("linux"):
        try:
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576
        except OSError:
            pass
    if sys.platform == "win32":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("tpage", ctypes.c_ulonglong), ("apage", ctypes.c_ulonglong),
                        ("tvirt", ctypes.c_ulonglong), ("avirt", ctypes.c_ulonglong), ("aext", ctypes.c_ulonglong)]
        m = MS()
        m.len = ctypes.sizeof(MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return m.avail / 2**30
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            page = 16384 if "page size of 16384" in out else 4096
            pages = 0
            for line in out.splitlines():
                if line.startswith(("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")):
                    pages += int(line.split(":")[1].strip().rstrip("."))
            return pages * page / 2**30
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return 0.0


def cpus() -> list[int]:
    """CPUs this process may run on."""
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def set_cpus(cs) -> bool:
    """Pin the calling thread's process to `cs` where the OS supports it (Linux); elsewhere a no-op."""
    if hasattr(os, "sched_setaffinity") and cs:
        os.sched_setaffinity(0, set(cs))
        return True
    return False


def cpu_flags() -> set[str]:
    """x86 feature flags (Linux /proc/cpuinfo); empty elsewhere (callers fall back to torch's detection)."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()
