"""Register chosen layers' expert weights as pinned host memory (read-only).

Expert loads from the page cache are pageable host->GPU copies: the CPU
takes a page fault per first touch and the driver stages every byte, which
measured ~6.7 GB/s on H100. Registered (pinned) host memory is DMA'd by the
GPU directly (52.9 GB/s measured on one shard in September).

This maps each shard read-only and MAP_SHARED, and registers only the byte
span holding the selected layers' routed experts with
cudaHostRegisterReadOnly: the existing page-cache pages are pinned in place,
so nothing is copied and no extra RAM is used (a private writable mapping
would copy-on-write every pinned page). Tensors for registered keys are
zero-copy views into that mapping; every other key comes from the wrapped
handles unchanged. Same bytes either way (self-checked at startup).

Pinned pages cannot be evicted until the process exits, so keep the
selected span well under free RAM and the user's memlock limit.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import mmap
import os
import re
import time
from typing import Any

import torch

from .loading import _safetensors_dtype, is_routed_expert_key

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")
_REGISTER_READ_ONLY = 0x08  # cudaHostRegisterReadOnly (CUDA >= 11.1)
_PAGE = mmap.PAGESIZE


def _cudart():
    lib = ctypes.util.find_library("cudart")
    if lib is None:
        # torch wheels ship the runtime under a versioned name
        for name in ("libcudart.so.12", "libcudart.so.11.0", "libcudart.so"):
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue
        raise OSError("libcudart not found")
    return ctypes.CDLL(lib)


if os.name == "posix":
    _libc = ctypes.CDLL(None, use_errno=True)
    _libc.mmap.restype = ctypes.c_void_p
    _libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                           ctypes.c_int, ctypes.c_int, ctypes.c_long]
    _libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _MAP_FAILED = ctypes.c_void_p(-1).value
else:
    _libc = None
    _MAP_FAILED = None


class _SharedMap:
    """Whole-file MAP_SHARED, PROT_READ mapping (pages stay in the page cache)."""

    def __init__(self, path: str):
        self.size = os.path.getsize(path)
        self.fd = os.open(path, os.O_RDONLY)
        addr = _libc.mmap(None, self.size, mmap.PROT_READ, mmap.MAP_SHARED, self.fd, 0)
        if addr in (None, _MAP_FAILED):
            os.close(self.fd)
            raise OSError(ctypes.get_errno(), f"mmap failed: {path}")
        self.addr = int(addr)
        with open(path, "rb") as f:
            self.data_start = 8 + int.from_bytes(f.read(8), "little")

    def close(self) -> None:
        _libc.munmap(self.addr, self.size)
        os.close(self.fd)


def merge_page_spans(spans: list[tuple[int, int]], page: int) -> list[tuple[int, int]]:
    """Page-align [start, end) byte spans and merge any that touch or overlap.

    Two tensors sharing a page must land in one registration: registering the
    same page twice fails (cudaErrorHostMemoryAlreadyRegistered).
    """
    out: list[list[int]] = []
    for s, e in sorted(spans):
        a, b = s // page * page, (e + page - 1) // page * page
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def mem_available_bytes() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise OSError("MemAvailable not found in /proc/meminfo")


def within_margin(avail_at_start: int, pinned: int, span: int, margin: int) -> bool:
    """Pinned page-cache pages stop being reclaimable for everyone else, so count
    every pinned byte against the RAM that was available when pinning began."""
    return avail_at_start - pinned - span >= margin


class RegisteredHandles:
    """Weight handles whose selected layers' experts live in pinned host memory."""

    def __init__(self, base: Any, layers: set[int] | None = None, verify: int = 3,
                 experts: set[tuple[int, int]] | None = None,
                 margin_gb: float | None = None):
        self.base = base
        self.weight_map = base.weight_map
        self.model_dir = getattr(base, "model_dir", "")
        self.layers = set(int(x) for x in (layers or ()))
        self.experts = {(int(a), int(b)) for a, b in (experts or ())}
        self.served = 0  # get_tensor calls answered from pinned memory
        # Shared-box guard: stop pinning (keeping what is pinned) once less than
        # margin_gb of the RAM available at start would remain for everyone else.
        self.margin = int(margin_gb * 1e9) if margin_gb is not None else None
        self.avail_start = mem_available_bytes() if self.margin is not None else None
        self.truncated = False
        self._cu = _cudart()
        self._cu.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        self._cu.cudaHostRegister.restype = ctypes.c_int
        self._cu.cudaHostUnregister.argtypes = [ctypes.c_void_p]
        self._maps: dict[str, _SharedMap] = {}
        self._spans: list[int] = []
        self._span_sizes: list[tuple[int, int]] = []
        self._keys: dict[str, tuple[str, int, int, tuple, str]] = {}
        self.reg_bytes = 0
        self.reg_s = 0.0
        self._register()
        self.verified = self._verify(verify)

    @staticmethod
    def layer_of(key: str) -> int | None:
        m = _LAYER_RE.search(key)
        return int(m.group(1)) if m else None

    @staticmethod
    def expert_of(key: str) -> int | None:
        m = _EXPERT_RE.search(key)
        return int(m.group(1)) if m else None

    def selected(self, key: str) -> bool:
        if not is_routed_expert_key(key):
            return False
        lyr = self.layer_of(key)
        return lyr in self.layers or (lyr, self.expert_of(key)) in self.experts

    def _register(self) -> None:
        by_shard: dict[str, list[str]] = {}
        for k, shard in self.weight_map.items():
            if self.selected(k):
                by_shard.setdefault(shard, []).append(k)
        if not by_shard:
            raise ValueError(f"no routed expert keys for layers {sorted(self.layers)} "
                             f"/ {len(self.experts)} experts")
        t0 = time.perf_counter()
        for shard, keys in sorted(by_shard.items()):
            sm = _SharedMap(os.path.join(self.model_dir, shard))
            self._maps[shard] = sm
            spans = []
            for k in keys:
                info = self.base.header(k)
                b, e = info["data_offsets"]
                s = sm.data_start + b
                self._keys[k] = (shard, s, e - b, tuple(info["shape"]), info.get("dtype", "BF16"))
                spans.append((s, s + (e - b)))
            for start, end in merge_page_spans(spans, _PAGE):
                end = min(sm.size, end)
                if self.margin is not None and not within_margin(
                        self.avail_start, self.reg_bytes, end - start, self.margin):
                    self.truncated = True
                    break
                ptr = sm.addr + start
                err = self._cu.cudaHostRegister(ctypes.c_void_p(ptr), ctypes.c_size_t(end - start),
                                                ctypes.c_uint(_REGISTER_READ_ONLY))
                if err != 0:
                    self.close()
                    raise RuntimeError(f"cudaHostRegister(read-only) failed with code {err} on "
                                       f"{shard} span [{start}, {end})")
                self._spans.append(ptr)
                self._span_sizes.append((ptr, end - start))
                self.reg_bytes += end - start
            if self.truncated:
                break
        self.reg_s = time.perf_counter() - t0
        if self.truncated:
            self._drop_unpinned()

    def _drop_unpinned(self) -> None:
        """After a truncated pass, keep views only for keys fully inside a pinned span."""
        ranges: dict[str, list[tuple[int, int]]] = {}
        for ptr, sz in self._span_sizes:
            for sh, sm in self._maps.items():
                if sm.addr <= ptr < sm.addr + sm.size:
                    ranges.setdefault(sh, []).append((ptr - sm.addr, ptr - sm.addr + sz))
        self._keys = {k: v for k, v in self._keys.items()
                      if any(a <= v[1] and v[1] + v[2] <= b for a, b in ranges.get(v[0], []))}

    def _view(self, key: str) -> torch.Tensor:
        shard, s, n, shape, tag = self._keys[key]
        sm = self._maps[shard]
        dtype = _safetensors_dtype(tag)
        buf = (ctypes.c_char * n).from_address(sm.addr + s)
        return torch.frombuffer(buf, dtype=dtype).reshape(shape)

    def _verify(self, n: int) -> int:
        checked = 0
        for k in list(self._keys)[: max(0, n)]:
            v = self._view(k)
            if not v.is_pinned():
                raise RuntimeError(f"{k}: registered view not reported as pinned")
            if not torch.equal(v, self.base.get_tensor(k)):
                raise RuntimeError(f"{k}: registered bytes differ from the base handles")
            checked += 1
        return checked

    def get_tensor(self, key: str) -> torch.Tensor:
        if key in self._keys:
            self.served += 1
            return self._view(key)
        return self.base.get_tensor(key)

    def header(self, key: str) -> dict:
        return self.base.header(key)

    def shards(self) -> list[str]:
        return self.base.shards()

    def stats(self) -> dict:
        return {"layers": sorted(self.layers), "experts": len(self.experts),
                "keys": len(self._keys), "served_tensors": self.served,
                "truncated_by_margin": self.truncated,
                "margin_gb": (self.margin / 1e9) if self.margin is not None else None,
                "mem_available_start_gb": (self.avail_start / 1e9) if self.avail_start else None,
                "spans": len(self._spans), "registered_gb": self.reg_bytes / 1e9,
                "register_s": self.reg_s, "verified_keys": self.verified}

    def close(self) -> None:
        for ptr in self._spans:
            self._cu.cudaHostUnregister(ctypes.c_void_p(ptr))
        self._spans = []
        for sm in self._maps.values():
            sm.close()
        self._maps = {}

    def __getattr__(self, name):
        if name == "base":  # not yet set during a failed __init__
            raise AttributeError(name)
        return getattr(self.base, name)
