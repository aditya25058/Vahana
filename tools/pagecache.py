"""Page-cache state of a model's shard files: measure and evict (no root).

Timings on a box whose RAM barely holds the model depend on how much of
it is cached, so every measured run records residency, and "cold" runs
evict first. Only the given files are touched: eviction is
posix_fadvise(DONTNEED) on our own shards, not a global cache drop.

    python rudra/pagecache.py residency <model_dir>
    python rudra/pagecache.py evict <model_dir>
"""
from __future__ import annotations

import ctypes
import glob
import json
import mmap
import os
import sys

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                       ctypes.c_int, ctypes.c_int, ctypes.c_long]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                          ctypes.POINTER(ctypes.c_ubyte)]
_PAGE = os.sysconf("SC_PAGE_SIZE")
_MAP_FAILED = ctypes.c_void_p(-1).value


def shards(model_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))


def file_residency(path: str) -> tuple[int, int]:
    """(resident_bytes, size_bytes) via mincore on a read-only mapping."""
    size = os.path.getsize(path)
    if size == 0:
        return 0, 0
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = _libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr in (None, _MAP_FAILED):
            raise OSError(ctypes.get_errno(), f"mmap failed: {path}")
        try:
            n = (size + _PAGE - 1) // _PAGE
            vec = (ctypes.c_ubyte * n)()
            if _libc.mincore(addr, size, vec) != 0:
                raise OSError(ctypes.get_errno(), f"mincore failed: {path}")
            # Linux sets only bit 0 (resident); count non-zero bytes fast.
            resident = n - bytes(vec).count(0)
        finally:
            _libc.munmap(addr, size)
    finally:
        os.close(fd)
    return min(resident * _PAGE, size), size


def residency(model_dir: str) -> dict:
    res = tot = 0
    for p in shards(model_dir):
        r, s = file_residency(p)
        res, tot = res + r, tot + s
    return {"resident_bytes": res, "total_bytes": tot,
            "resident_gb": round(res / 1e9, 2), "total_gb": round(tot / 1e9, 2),
            "resident_frac": round(res / tot, 4) if tot else 0.0}


def evict(model_dir: str) -> dict:
    n = 0
    for p in shards(model_dir):
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            n += 1
        finally:
            os.close(fd)
    return {"evicted_shards": n, **residency(model_dir)}


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("residency", "evict"):
        sys.exit(__doc__)
    fn = residency if sys.argv[1] == "residency" else evict
    print(json.dumps(fn(sys.argv[2])))
