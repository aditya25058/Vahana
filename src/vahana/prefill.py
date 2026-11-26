"""v1.3: long prompts run their CPU-resident experts on the GPU.

In decode, a token touches K experts per layer, and moving their weights to the GPU costs more than computing them
on the CPU. A long prompt is the opposite: N tokens touch nearly every expert of every layer, and the CPU computes
N*K expert rows per layer, so its time grows with N, while copying the layer's experts to the GPU once costs the
same for any N. From VAHANA_PREFILL_GPU tokens (default 32; 0 = off) every supported expert layer of a prefill
streams its experts through staging buffers on the GPU and runs them with the tiled GPU twin kernels
(prefill_kernels: the CPU kernel's arithmetic, so its bits), then combines with the CPU kernel's combine
(cache.combine). The prefill's logits are unchanged, bit for bit; it is only faster.

Pipelining:
  - chunk i of a layer computes on the GPU while chunk i+1 is copied (side stream) and later chunks are gathered;
  - read-ahead (VAHANA_PREFILL_READAHEAD=1, off by default: on the 4 GB laptop it moved ~40% more bytes than the
    experts actually used and its background reads competed with the foreground ones, 7.2 -> 9.7 s for 512 tokens;
    v13_prefill README): when the batch is big enough that a layer uses nearly all its experts (N*K >= 2E), every uncached
    expert of the layer is streamed whether used or not, so the plan does not depend on routing, and a background
    thread gathers the next layers' experts (pinned host ring) while the GPU works on this one. Unused experts are
    copied but never read (their slots do not exist), so the bits do not depend on the read-ahead either.
"""
from __future__ import annotations

import os
import queue
import threading
import weakref

import torch

STATS = {"layers": 0, "experts": 0, "readahead_layers": 0}
PROF = {} if os.environ.get("VAHANA_PREFILL_PROFILE") else None
REG: dict = {}                     # slot -> [weakref to VahanaExperts, weakref to CachedExperts or None]


def _reg(s):
    """(base, cached) registered for slot s, or (None, None) when gone."""
    r = REG.get(s)
    if r is None:
        return None, None
    return r[0](), (r[1]() if r[1] is not None else None)


def register(base, cached=None) -> None:
    """Called when an expert layer is built (slots are dense and ascending within a model): a newer model's layer
    replaces an older one's, and the older model's later slots are dropped."""
    s = base.spec.slot
    if cached is not None and _reg(s)[0] is base:
        REG[s][1] = weakref.ref(cached)
        return
    for k in [k for k in REG if k >= s]:
        del REG[k]
    REG[s] = [weakref.ref(base), weakref.ref(cached) if cached is not None else None]


def min_tokens() -> int:
    return int(os.environ.get("VAHANA_PREFILL_GPU", "32"))


def wanted(x: torch.Tensor, base) -> bool:
    m = min_tokens()
    if m <= 0 or x.device.type != "cuda" or x.shape[0] < m:
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    from .cache import CachedExperts
    return CachedExperts.supported(base)


def _stage_bytes(dev) -> int:
    gb = os.environ.get("VAHANA_PREFILL_STAGE_GB")
    if gb:
        return int(float(gb) * 2**30)
    free, _ = torch.cuda.mem_get_info(dev)
    return int(max(64 * 2**20, min(2 * 2**30, 0.4 * free)))


_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(int(os.environ.get("VAHANA_PREFILL_IO_THREADS", "4")), "vahana-prefill-io")
    return _POOL


def _now():
    import time
    return time.perf_counter()


def _acc(k, v):
    if PROF is not None:
        PROF[k] = PROF.get(k, 0.0) + v


def profile_report() -> dict:
    """VAHANA_PREFILL_PROFILE=1: totals so far (seconds)."""
    if PROF is None:
        return {}
    torch.cuda.synchronize()
    out = {k: round(v, 3) if isinstance(v, float) else v for k, v in PROF.items() if not k.startswith("_")}
    out["h2d_s"] = round(sum(a.elapsed_time(b) for a, b in PROF.get("_h2d", [])) / 1e3, 3)
    out["compute_s"] = round(sum(a.elapsed_time(b) for a, b in PROF.get("_cmp", [])) / 1e3, 3)
    return out


class _Stager:
    """NH pinned host slots and 2 device slots, each holding `cap` experts (g, u, d); a side stream for copies."""

    def __init__(self, dev, shapes, dtype, cap, nh):
        self.dev, self.cap = dev, cap
        self.host = [[torch.empty((cap, *s), dtype=dtype, pin_memory=True) for s in shapes] for _ in range(nh)]
        self.hev = [None] * nh                    # per host slot: the copy that last read it
        self.gpu = [[torch.empty((cap, *s), dtype=dtype, device=dev) for s in shapes] for _ in range(2)]
        self.done = [None, None]                  # per device slot: the compute that last read it
        self.stream = torch.cuda.Stream(dev)

    def gather(self, h, mats, ids):
        """Experts `ids` of (g, u, d) into host slot h (after the copy that last read it has finished)."""
        if self.hev[h] is not None:
            self.hev[h].synchronize()
        t0 = _now()
        jobs = [(hb[j], ws[e]) for hb, ws in zip(self.host[h], mats) for j, e in enumerate(ids)]
        list(_pool().map(lambda p: p[0].copy_(p[1]), jobs))     # parallel page-ins and copies (GIL released)
        _acc("gather_s", _now() - t0)

    def to_gpu(self, h, b, n):
        """Host slot h -> device slot b on the side stream, after the compute that last read device slot b."""
        if self.done[b] is not None:
            self.stream.wait_event(self.done[b])
        with torch.cuda.stream(self.stream):
            if PROF is not None:
                e0 = torch.cuda.Event(enable_timing=True)
                e0.record(self.stream)
            for hb, gb in zip(self.host[h], self.gpu[b]):
                gb[:n].copy_(hb[:n], non_blocking=True)
            ev = torch.cuda.Event(enable_timing=PROF is not None)
            ev.record(self.stream)
            if PROF is not None:
                PROF.setdefault("_h2d", []).append((e0, ev))
        self.hev[h] = ev
        return [g[:n] for g in self.gpu[b]], ev


_ST: dict = {}


def _stager(base, dev):
    g0, d0 = base.g[0], base.d[0]
    per = sum(t.numel() * t.element_size() for t in (base.g[0], base.u[0], base.d[0]))
    cap = max(1, min(-(-base.spec.E // int(os.environ.get("VAHANA_PREFILL_MIN_CHUNKS", "4"))),
                     _stage_bytes(dev) // (2 * per)))      # >= 4 chunks a layer: gather, copy and compute overlap
    nh = max(2, int(os.environ.get("VAHANA_PREFILL_HOST_SLOTS", "6")))
    key = (tuple(g0.shape), tuple(d0.shape), g0.dtype, cap, nh, str(dev))
    st = _ST.get(key)
    if st is None:
        _abandon()
        _ST.clear()                               # one shape at a time: free the old buffers first
        torch.cuda.empty_cache()
        st = _ST[key] = _Stager(dev, [tuple(g0.shape), tuple(g0.shape), tuple(d0.shape)], g0.dtype, cap, nh)
    return st


class _ReadAhead:
    """A background thread gathering (layer, chunk) after (layer, chunk) into the host ring, in the order the
    layers will run; the layers take them in the same order."""

    def __init__(self, st, plan):
        self.st, self.plan = st, plan             # plan: [(slot, [ids per chunk]), ...]
        self.free = queue.Queue()
        for h in range(len(st.host)):
            self.free.put(h)
        self.ready = queue.Queue()
        self.stop = False
        self.pos = 0                              # next plan entry a layer must consume
        self.th = threading.Thread(target=self._work, daemon=True, name="vahana-prefill-readahead")
        self.th.start()

    def _work(self):
        try:
            for slot, chunks in self.plan:
                base = _reg(slot)[0]
                mats = (base.g, base.u, base.d)
                for i, ids in enumerate(chunks):
                    h = self.free.get()
                    if self.stop:
                        return
                    self.st.gather(h, mats, ids)
                    self.ready.put((slot, i, h))
        except Exception as e:  # noqa: BLE001
            self.ready.put(e)

    def take(self, slot, i):
        item = self.ready.get()
        if isinstance(item, Exception):
            raise item
        assert item[:2] == (slot, i), (item, slot, i)
        return item[2]

    def release(self, h):
        self.free.put(h)

    def close(self):
        self.stop = True
        for _ in range(len(self.st.host)):
            self.free.put(0)
        self.th.join()


_RA = {"ra": None, "next": None, "n": None}


def _abandon():
    ra = _RA["ra"]
    if ra is not None:
        ra.close()
    _RA.update(ra=None, next=None, n=None)


def _plan_from(slot, st):
    plan = []
    for s in sorted(k for k in REG if k >= slot):
        base, cached = _reg(s)
        from .cache import CachedExperts
        if base is None or not CachedExperts.supported(base) or base.g[0].shape != st.host[0][0].shape[1:]:
            break
        E = base.spec.E
        ids = [e for e in range(E) if cached is None or int(cached.lut_cpu[e]) < 0]
        plan.append((s, [ids[i:i + st.cap] for i in range(0, len(ids), st.cap)]))
    return plan


def run(base, x, ti, tw, cached=None) -> torch.Tensor:
    if PROF is None:
        return _run(base, x, ti, tw, cached)
    torch.cuda.synchronize()
    t0 = _now()
    if PROF.get("_last") is not None:
        _acc("between_s", t0 - PROF["_last"])     # attention, norms, router: everything between expert layers
    out = _run(base, x, ti, tw, cached)
    torch.cuda.synchronize()
    PROF["_last"] = _now()
    _acc("moe_wall_s", PROF["_last"] - t0)
    return out


def _run(base, x, ti, tw, cached=None) -> torch.Tensor:
    """The MoE output [N, H] bf16 of a VahanaExperts layer `base` with every slot computed on the GPU. `cached`: a
    CachedExperts whose resident experts are used in place (no copy)."""
    from . import prefill_kernels as pk
    from . import runtime
    from .cache import combine
    dev = x.device
    spec = base.spec
    q4 = getattr(base, "q4", False)
    x = x.to(torch.bfloat16).contiguous()
    ti = ti.to(torch.int64).contiguous()
    N, K = ti.shape
    bn = 32 if spec.H % 32 == 0 and spec.I % 32 == 0 else 16    # measured: v13_prefill/compare_kernels.py
    slots = lambda xx, t, lut, G, U, D: pk.slots(xx, t, lut, G, U, D, q4, spec.H, spec.I, 16, bn)
    y = torch.empty((N, K, spec.H), dtype=torch.bfloat16, device=dev)
    if cached is not None:
        y = slots(x, ti, cached.lut, cached.G, cached.U, cached.D)
    st = _stager(base, dev)
    PROF is not None and PROF.__setitem__("cap", st.cap)
    cs = torch.cuda.current_stream(dev)
    STATS["layers"] += 1
    readahead = N * K >= 2 * spec.E and os.environ.get("VAHANA_PREFILL_READAHEAD", "0") == "1"
    if readahead:
        if _RA["ra"] is None or _RA["next"] != spec.slot or _RA["n"] != N or _reg(spec.slot)[0] is not base:
            _abandon()
            plan = _plan_from(spec.slot, st)
            if not plan or plan[0][0] != spec.slot:
                readahead = False
            else:
                _RA.update(ra=_ReadAhead(st, plan), next=spec.slot, n=N)
    if readahead:
        ra = _RA["ra"]
        slot, chunks = ra.plan[ra.pos]
        ra.pos += 1
        _RA["next"] = ra.plan[ra.pos][0] if ra.pos < len(ra.plan) else None
        STATS["readahead_layers"] += 1
    else:
        used = torch.zeros(spec.E, dtype=torch.bool, device=dev)
        used[ti.flatten()] = True
        if cached is not None:
            used &= cached.lut < 0
        need = used.nonzero().flatten().tolist()
        chunks = [need[i:i + st.cap] for i in range(0, len(need), st.cap)]
    STATS["experts"] += sum(len(c) for c in chunks)
    if chunks:
        _acc("chunks", len(chunks))
        luth = torch.full((len(chunks), spec.E), -1, dtype=torch.int64)
        for i, ids in enumerate(chunks):
            luth[i, ids] = torch.arange(len(ids))
        luts = luth.pin_memory().to(dev, non_blocking=True)      # no host sync inside the loop
        mats = (base.g, base.u, base.d)

        def fetch(i):                                             # host slot holding chunk i, gathered
            if readahead:
                return ra.take(spec.slot, i)
            h = i % len(st.host)
            st.gather(h, mats, chunks[i])
            return h
        h = fetch(0)
        pend = st.to_gpu(h, 0, len(chunks[0])) + (h,)
        for i in range(len(chunks)):
            (G, U, D), ev, h = pend
            cs.wait_event(ev)
            lut = luts[i]
            if PROF is not None:
                c0 = torch.cuda.Event(enable_timing=True)
                c0.record()
            yc = slots(x, ti, lut, G, U, D)                       # queued: the GPU computes chunk i ...
            if PROF is not None:
                c1 = torch.cuda.Event(enable_timing=True)
                c1.record()
                PROF.setdefault("_cmp", []).append((c0, c1))
            y = torch.where((lut[ti] >= 0).unsqueeze(-1), yc, y)
            st.done[i % 2] = torch.cuda.Event()
            st.done[i % 2].record(cs)
            if readahead:
                ra.release(h)                                     # reused only after its copy has finished
            if i + 1 < len(chunks):
                h2 = fetch(i + 1)                                 # ... while chunk i + 1 is gathered and copied
                pend = st.to_gpu(h2, (i + 1) % 2, len(chunks[i + 1])) + (h2,)
    if readahead and _RA["next"] is None:
        _abandon()
    return combine(y, ti, runtime.weights_arg(tw))
