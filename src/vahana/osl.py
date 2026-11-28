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


def physical_cores(cs=None) -> int | None:
    """v1.4: physical cores among CPUs `cs` (default: this process's), or None if unknown."""
    cs = list(cs) if cs is not None else cpus()
    try:
        if sys.platform.startswith("linux"):
            seen = set()
            for c in cs:
                base = f"/sys/devices/system/cpu/cpu{c}/topology/"
                with open(base + "core_id") as f, open(base + "physical_package_id") as g:
                    seen.add((f.read().strip(), g.read().strip()))
            return len(seen) or None
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            n = wintypes.DWORD(0)
            k = ctypes.windll.kernel32
            k.GetLogicalProcessorInformation(None, ctypes.byref(n))
            buf = (ctypes.c_byte * n.value)()
            if not k.GetLogicalProcessorInformation(buf, ctypes.byref(n)):
                return None
            sz = 32 if ctypes.sizeof(ctypes.c_void_p) == 8 else 24     # SYSTEM_LOGICAL_PROCESSOR_INFORMATION
            off = ctypes.sizeof(ctypes.c_void_p)                       # Relationship follows ProcessorMask
            rel = [int.from_bytes(bytes(buf[i + off:i + off + 4]), "little") for i in range(0, n.value, sz)]
            cores = sum(1 for r in rel if r == 0)                       # RelationProcessorCore
            return min(cores, len(cs)) or None
        if sys.platform == "darwin":
            import subprocess
            return int(subprocess.run(["sysctl", "-n", "hw.physicalcpu"], capture_output=True, text=True).stdout)
    except (OSError, ValueError):
        return None
    return None


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


def release(ptr: int, nbytes: int) -> bool:
    """v1.4: tell the OS these pages (a mapped file range we no longer read) are the first to reclaim: on Windows
    VirtualUnlock moves them from our working set to the standby list, on Linux madvise(MADV_COLD). The data stays
    valid (a later read faults it back in); this only makes room in RAM for pages that are read. Never affects bits."""
    if not ptr or nbytes <= 0 or os.environ.get("VAHANA_RELEASE", "1") == "0":
        return False
    try:
        import ctypes
        if sys.platform == "win32":
            k = ctypes.windll.kernel32
            k.VirtualUnlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            k.VirtualUnlock(ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes))   # "not locked" error: pages still removed
            return True
        if sys.platform.startswith("linux"):
            pg = os.sysconf("SC_PAGE_SIZE")
            a = ptr & ~(pg - 1)
            libc = ctypes.CDLL(None, use_errno=True)
            libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            return libc.madvise(ctypes.c_void_p(a), ctypes.c_size_t(ptr + nbytes - a), 20) == 0    # MADV_COLD
    except Exception:  # noqa: BLE001
        pass
    return False


def release_tensor(t) -> bool:
    return release(t.data_ptr(), t.numel() * t.element_size())

