"""Lossless compressed expert plane: ANS-coded high bytes + raw low bytes, decoded on the GPU.

Store layout (built by rudra/encode_ans_store.py): for every routed expert tensor,
`index.json` gives an nvCOMP ANS blob of the high bytes (sign + 7 exponent bits of
each BF16 value) in `<shard>.ansh`, and `lo_index.json` gives the raw low bytes in
`<shard>.lob`. ~0.70 of the raw bytes travel over PCIe; the GPU decodes the high
bytes and interleaves both halves back into BF16. Lossless: the slot holds the
exact original bytes (checked at startup with torch.equal against the raw shard).

Both files are mapped read-only/shared and the needed spans are pinned in place
(cudaHostRegisterReadOnly), so nothing is copied in host RAM; the same 64 GB
shared-RAM margin guard as hostreg applies.

Pipelining: copies run on the caller's DMA stream into a ring of R staging
entries; decode + interleave run on a separate decode stream; the GEMM stream
waits on the decode stream. `stage()` can be called ahead of `finish()` so the
link stays busy even if a decode call blocks the host.
"""
from __future__ import annotations

import ctypes
import json
import mmap
import os
import time
from typing import Any

import torch

from .hostreg import _cudart, mem_available_bytes, merge_page_spans

_REGISTER_READ_ONLY = 0x08
_REGISTER_PORTABLE = 0x01   # visible to every device in the process (2-GPU mode)
_PLANE = None
_PEERS: dict = {}


def _nvcomp():
    # The venv's nvidia namespace can be shadowed by another 'nvidia' package; add
    # the site-packages copy next to torch (same fix as rudra/encode_ans_store.py).
    import nvidia
    import sys
    site = os.environ.get("COLOSSUS_NVCOMP_SITE") or os.path.join(
        sys.prefix, "lib", "python%d.%d" % sys.version_info[:2], "site-packages", "nvidia")
    if os.path.isdir(site) and site not in list(nvidia.__path__):
        nvidia.__path__.append(site)
    import nvidia.nvcomp as nvcomp
    return nvcomp


class AnsPlane:
    def __init__(self, store: str, device: torch.device, margin_gb: float = 64.0,
                 layers: set[int] | None = None, ring: int = 16):
        self.store, self.device, self.ring = store, device, int(ring)
        self.idx = json.load(open(os.path.join(store, "index.json")))
        self.lidx = json.load(open(os.path.join(store, "lo_index.json")))
        if layers is not None:
            keep = {k for k in self.idx if _layer(k) in layers}
            self.idx = {k: v for k, v in self.idx.items() if k in keep}
        self._cu = _cudart()
        self._libc = ctypes.CDLL(None, use_errno=True)
        self._libc.mmap.restype = ctypes.c_void_p
        self._libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_long]
        self._maps: dict[str, tuple[int, int, int]] = {}
        self._regs: list[int] = []
        self.reg_bytes, self.truncated = 0, False
        self.avail_start = mem_available_bytes()
        self.margin = int(margin_gb * (1 << 30))
        self._reg_flags = _REGISTER_READ_ONLY | (
            _REGISTER_PORTABLE if os.environ.get("COLOSSUS_SECOND_GPU") == "1" else 0)
        t0 = time.perf_counter()
        self._register()
        self.reg_s = time.perf_counter() - t0
        nv = _nvcomp()
        self._nv = nv
        self.dec_stream = torch.cuda.Stream(device=device)
        self.codec = nv.Codec(algorithm="ans", cuda_stream=self.dec_stream.cuda_stream)
        self._slots: list[dict | None] = [None] * self.ring
        self._next = 0
        self._staged: dict[tuple[int, int], int] = {}
        self._cfg: dict[int, Any] = {}
        self.stats = {"staged": 0, "decoded": 0, "wire_bytes": 0, "raw_bytes": 0}

    # -- mapping / pinning -------------------------------------------------
    def _map(self, fn: str) -> tuple[int, int, int]:
        if fn not in self._maps:
            path = os.path.join(self.store, fn)
            size = os.path.getsize(path)
            fd = os.open(path, os.O_RDONLY)
            addr = self._libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
            if addr in (None, ctypes.c_void_p(-1).value):
                raise OSError(f"mmap failed: {path}")
            self._maps[fn] = (int(addr), size, fd)
        return self._maps[fn]

    def _register(self) -> None:
        spans: dict[str, list[tuple[int, int]]] = {}
        for k, m in self.idx.items():
            spans.setdefault(m["blob"], []).append((m["offset"], m["offset"] + m["comp_len"]))
            lm = self.lidx[k]
            spans.setdefault(lm["lob"], []).append((lm["offset"], lm["offset"] + lm["len"]))
        self._pinned_ranges: dict[str, list[tuple[int, int]]] = {}
        for fn in sorted(spans):
            addr, size, _ = self._map(fn)
            for a, b in merge_page_spans(spans[fn], mmap.PAGESIZE):
                b = min(b, size)
                if self.avail_start - self.reg_bytes - (b - a) < self.margin:
                    self.truncated = True
                    break
                err = self._cu.cudaHostRegister(ctypes.c_void_p(addr + a), ctypes.c_size_t(b - a),
                                                ctypes.c_uint(self._reg_flags))
                if err != 0:
                    raise RuntimeError(f"cudaHostRegister(read-only) failed ({err}) on {fn}")
                self._regs.append(addr + a)
                self._pinned_ranges.setdefault(fn, []).append((a, b))
                self.reg_bytes += b - a
            if self.truncated:
                break
        # Keys whose blob and low bytes are both fully pinned are served from the plane.
        def inside(fn, s, e):
            return any(a <= s and e <= b for a, b in self._pinned_ranges.get(fn, []))
        self.served = {k for k, m in self.idx.items()
                       if inside(m["blob"], m["offset"], m["offset"] + m["comp_len"])
                       and inside(self.lidx[k]["lob"], self.lidx[k]["offset"],
                                  self.lidx[k]["offset"] + self.lidx[k]["len"])}

    def _view(self, fn: str, off: int, n: int) -> torch.Tensor:
        addr, _, _ = self._map(fn)
        return torch.frombuffer((ctypes.c_char * n).from_address(addr + off), dtype=torch.uint8)

    def has(self, keys) -> bool:
        return all(k in self.served for k in keys)

    # -- staging ring --------------------------------------------------------
    def _entry(self, sizes: list[int], comp_total: int) -> int:
        i = self._next % self.ring
        self._next += 1
        if i in self._staged.values():
            raise RuntimeError("ANS staging ring overrun: lookahead must be < ring size")
        e = self._slots[i]
        if e is None or e["comp"].numel() < comp_total or [t.numel() for t in e["lo"]] != sizes:
            old_ev = e["free"] if e else None
            e = {"lo": [torch.empty(n, dtype=torch.uint8, device=self.device) for n in sizes],
                 "hi": [torch.empty(n, dtype=torch.uint8, device=self.device) for n in sizes],
                 "comp": torch.empty(max(comp_total, 16 << 20), dtype=torch.uint8, device=self.device),
                 "free": old_ev}
            self._slots[i] = e
        return i

    def stage(self, tag: tuple[int, int], keys, dma_stream) -> None:
        """Issue the copies of one expert's low bytes + compressed high bytes (async)."""
        if tag in self._staged:
            return
        los = [self.lidx[k] for k in keys]
        cps = [self.idx[k] for k in keys]
        i = self._entry([m["len"] for m in los], sum(m["comp_len"] for m in cps) + 256 * len(cps))
        e = self._slots[i]
        with torch.cuda.stream(dma_stream):
            if e["free"] is not None:
                dma_stream.wait_event(e["free"])
            segs, off = [], 0
            for j, (lm, cm) in enumerate(zip(los, cps)):
                e["lo"][j].copy_(self._view(lm["lob"], lm["offset"], lm["len"]), non_blocking=True)
                n = cm["comp_len"]
                seg = e["comp"][off:off + n]
                seg.copy_(self._view(cm["blob"], cm["offset"], n), non_blocking=True)
                segs.append(seg)
                off = (off + n + 255) // 256 * 256      # nvCOMP ANS needs aligned input
            ev = torch.cuda.Event()
            ev.record(dma_stream)
        e["segs"], e["copied"], e["tag"] = segs, ev, tag
        self._staged[tag] = i
        self.stats["staged"] += 1
        self.stats["wire_bytes"] += sum(m["len"] for m in los) + sum(m["comp_len"] for m in cps)

    def drop_layer(self, layer: int, keep: set[int]) -> None:
        """Forget staged copies of `layer` that this forward will not decode (their ring
        entries are reused later; copies on one DMA stream stay ordered)."""
        for tag in [t for t in self._staged if t[0] == layer and t[1] not in keep]:
            self._staged.pop(tag)

    def finish(self, tag, keys, dsts, dma_stream, slot_free=None) -> torch.cuda.Event:
        """Decode the staged expert into `dsts` (BF16 weights); returns the ready event."""
        if tag not in self._staged:
            self.stage(tag, keys, dma_stream)
        i = self._staged.pop(tag)
        e = self._slots[i]
        d = self.dec_stream
        d.wait_event(e["copied"])
        if slot_free is not None:
            d.wait_event(slot_free)
        srcs = [self._nv.as_array(s) for s in e["segs"]]
        outs = [self._nv.as_array(h) for h in e["hi"]]
        n = len(srcs)
        if n not in self._cfg:
            # One DecompressConfig per batch size, built once from real blobs (one host sync).
            # Every high-byte blob decodes to the same size per projection shape, so later
            # decodes reuse it and never sync (nvCOMP otherwise parses headers per call).
            e["copied"].synchronize()
            self._cfg[n] = self.codec.decompression_config(srcs)
        with torch.cuda.stream(d):
            # Decode into our own buffers (out=): avoids nvCOMP-owned outputs and the dlpack
            # conversion that segfaulted in Phase 13a attempt 2.
            self.codec.decode(srcs, out=outs, decompression_config=self._cfg[n])
            for lo, hi, w in zip(e["lo"], e["hi"], dsts):
                torch.stack((lo, hi), dim=-1, out=w.view(torch.uint8).view(-1, 2))
            ev = torch.cuda.Event()
            ev.record(d)
        e["free"] = ev
        self.stats["decoded"] += 1
        self.stats["raw_bytes"] += sum(w.numel() * w.element_size() for w in dsts)
        return ev

    def finish_group(self, tags, keys_list, dsts_list, dma_stream, slot_frees) -> torch.cuda.Event:
        """Decode several staged experts with ONE nvCOMP call (Phase 17).

        Each decode call costs ~1.8 ms and blocks the host regardless of size, so one call
        per expert caps the plane at ~26 GB/s; 8 experts per call reach ~180 GB/s. Same
        bytes, same interleave: the slots receive exactly the bytes finish() would write.
        """
        for tag, keys in zip(tags, keys_list):
            if tag not in self._staged:
                self.stage(tag, keys, dma_stream)
        ents = [self._slots[self._staged.pop(t)] for t in tags]
        d = self.dec_stream
        for e in ents:
            d.wait_event(e["copied"])
        for f in slot_frees:
            if f is not None:
                d.wait_event(f)
        srcs = [self._nv.as_array(s) for e in ents for s in e["segs"]]
        outs = [self._nv.as_array(h) for e in ents for h in e["hi"]]
        n = len(srcs)
        if n not in self._cfg:
            for e in ents:
                e["copied"].synchronize()
            self._cfg[n] = self.codec.decompression_config(srcs)
        with torch.cuda.stream(d):
            self.codec.decode(srcs, out=outs, decompression_config=self._cfg[n])
            for e, dsts in zip(ents, dsts_list):
                for lo, hi, w in zip(e["lo"], e["hi"], dsts):
                    torch.stack((lo, hi), dim=-1, out=w.view(torch.uint8).view(-1, 2))
            ev = torch.cuda.Event()
            ev.record(d)
        for e, dsts in zip(ents, dsts_list):
            e["free"] = ev
            self.stats["decoded"] += 1
            self.stats["raw_bytes"] += sum(w.numel() * w.element_size() for w in dsts)
        self.stats["group_calls"] = self.stats.get("group_calls", 0) + 1
        return ev

    def self_check(self, keys, raw_tensors, dma_stream) -> bool:
        dsts = [torch.empty(t.shape, dtype=t.dtype, device=self.device) for t in raw_tensors]
        ev = self.finish((-1, -1), keys, dsts, dma_stream)
        ev.synchronize()
        return all(torch.equal(d, t.to(self.device)) for d, t in zip(dsts, raw_tensors))

    def info(self) -> dict:
        return {"store": self.store, "registered_gb": self.reg_bytes / 1e9, "register_s": self.reg_s,
                "truncated_by_margin": self.truncated, "served_tensors": len(self.served),
                "margin_gb": self.margin / (1 << 30), "mem_available_start_gb": self.avail_start / 1e9,
                **self.stats}

    def close(self) -> None:
        for p in self._regs:
            self._cu.cudaHostUnregister(ctypes.c_void_p(p))
        self._regs.clear()


def _layer(k: str) -> int | None:
    import re
    m = re.search(r"\.layers\.(\d+)\.", k)
    return int(m.group(1)) if m else None


def init_plane(store: str, device: torch.device, margin_gb: float = 64.0) -> AnsPlane:
    global _PLANE
    if _PLANE is None:
        _PLANE = AnsPlane(store, device, margin_gb=margin_gb,
                          ring=int(os.environ.get("COLOSSUS_ANS_RING", "16")))
    return _PLANE


def get_plane() -> AnsPlane | None:
    return _PLANE


def get_peer(device: torch.device) -> AnsPlane:
    """A plane on another GPU that reuses the primary plane's mapped + registered (portable)
    host spans, with its own copy stream, decode stream, codec and staging ring."""
    if device.index in _PEERS:
        return _PEERS[device.index]
    base = _PLANE
    if base is None:
        raise RuntimeError("primary ANS plane not initialised")
    p = object.__new__(AnsPlane)
    for k in ("store", "ring", "idx", "lidx", "_cu", "_libc", "_maps", "served", "margin",
              "avail_start", "_reg_flags", "_nv", "_pinned_ranges"):
        setattr(p, k, getattr(base, k))
    p.device = device
    p._regs, p.reg_bytes, p.truncated, p.reg_s = [], 0, False, 0.0
    with torch.cuda.device(device):
        p.dec_stream = torch.cuda.Stream(device=device)
        p.dma_stream = torch.cuda.Stream(device=device)
        p.codec = p._nv.Codec(algorithm="ans", device_id=device.index,
                              cuda_stream=p.dec_stream.cuda_stream)
    p._slots = [None] * p.ring
    p._next = 0
    p._staged = {}
    p._cfg = {}
    p.stats = {"staged": 0, "decoded": 0, "wire_bytes": 0, "raw_bytes": 0}
    _PEERS[device.index] = p
    return p
