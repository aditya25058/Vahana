"""v1.4: the GPU expert cache follows the text being generated (speed only: the bits never depend on it).

The cache used to hold a fixed set of experts, the most used ones on built-in calibration text. On real prompts that
set serves few routed uses (Qwen3-30B on a 4 GB laptop GPU, 386 cached experts: 8-19% of decode uses), because which
experts are hot depends strongly on the text: a set profiled on other prompts is no better than a random one
(experiments/v15_decode/routing_sim.py). This module keeps the cache's size and per-layer slots and changes what they
hold:
  - every cached layer adds its routed experts to a count tensor on the GPU (one small kernel per layer, captured in
    the decode graph);
  - after a prefill, each layer re-picks its experts from the prompt's own routing (`after_prefill`);
  - after each decode step, up to VAHANA_CACHE_SWAPS (default 4) swaps whose decayed use count most exceeds the
    coldest cached expert's (`after_step`).
A swap copies the expert's weights into the coldest slot, then updates the GPU lookup table and the CPU kernel's
resident set together, between steps (no CPU job is in flight and the GPU is idle), so in every step each routed
slot is computed exactly once, on the GPU if cached and on the CPU otherwise. Both give the same bits (tests), so the
output never depends on what the cache holds. VAHANA_ADAPTIVE_CACHE=0 keeps the cache fixed.
"""
from __future__ import annotations

import os

import torch

DECAY = 0.97          # per decode step
MARGIN = 1.0          # a swap must gain more than one (decayed) use


def enabled() -> bool:
    return os.environ.get("VAHANA_ADAPTIVE_CACHE", "1") != "0"


class Adaptive:
    def __init__(self, layers, device):
        self.layers = list(layers)
        E = self.layers[0].spec.E
        self.cnt = torch.zeros(len(self.layers), E, dtype=torch.float32, device=device)
        self.score = torch.zeros(len(self.layers), E, dtype=torch.float64)
        for i, m in enumerate(self.layers):
            m._acnt = self.cnt[i]
        self.swaps = int(os.environ.get("VAHANA_CACHE_SWAPS", "4"))
        self._ring, self._ri = {}, {}
        self.stats = {"swaps": 0, "steps": 0, "prefill_swaps": 0, "prefill_ms": 0.0}

    def _take(self) -> torch.Tensor:
        c = self.cnt.to("cpu", torch.float64)
        self.cnt.zero_()
        return c

    def _cached(self) -> torch.Tensor:
        return torch.stack([m.lut_cpu >= 0 for m in self.layers])

    def _best(self, cached: torch.Tensor):
        """Per layer: (gain, best uncached expert, coldest cached expert), all layers at once."""
        sc = self.score
        e_in = torch.where(cached, -1.0, sc).argmax(1)
        e_out = torch.where(cached, sc, float("inf")).argmin(1)
        gain = sc.gather(1, e_in[:, None])[:, 0] - sc.gather(1, e_out[:, None])[:, 0]
        ok = cached.any(1) & ~cached.all(1)
        return torch.where(ok, gain, float("-inf")), e_in, e_out

    RING = 6

    def _up(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        """dst (GPU) <- src (host, pageable: often the checkpoint's memory map) through a ring of pinned buffers: the
        CPU copy into one buffer overlaps the DMA out of the others (pageable copies ran at ~2.3 GB/s)."""
        src = src.to(dst.dtype).reshape(-1)
        k = (dst.dtype, src.numel())
        ring = self._ring.get(k)
        if ring is None:
            ring = self._ring[k] = [[torch.empty(src.numel(), dtype=dst.dtype, pin_memory=True), None]
                                    for _ in range(self.RING)]
        slot = ring[self._ri.get(k, 0)]
        self._ri[k] = (self._ri.get(k, 0) + 1) % self.RING
        if slot[1] is not None:
            slot[1].synchronize()                          # its previous DMA has finished: the buffer is free
        slot[0].copy_(src)
        dst.reshape(-1).copy_(slot[0], non_blocking=True)
        if slot[1] is None:
            slot[1] = torch.cuda.Event()
        slot[1].record()

    def _swap(self, i: int, e_in: int, e_out: int) -> None:
        m = self.layers[i]
        s = int(m.lut_cpu[e_out])
        b = m.base
        self._up(m.G[s], b.g[e_in])
        self._up(m.U[s], b.u[e_in])
        self._up(m.D[s], b.d[e_in])
        from . import osl
        for w in (b.g[e_in], b.u[e_in], b.d[e_in]):      # now read from the GPU: its host pages may go first
            osl.release_tensor(w)
        m.lut_cpu[e_out] = -1
        m.lut_cpu[e_in] = s
        self.stats["swaps"] += 1

    def _commit(self, touched) -> None:
        from . import hybrid as _hy
        for i in touched:
            m = self.layers[i]
            m.lut.copy_(m.lut_cpu)
            _hy.ext().register_resident(m.spec.slot, (m.lut_cpu >= 0).to(torch.uint8))

    def _round(self, cached, limit):
        gain, e_in, e_out = self._best(cached)
        order = torch.argsort(gain, descending=True)[:limit]
        done = []
        for i in order.tolist():
            if float(gain[i]) <= MARGIN:
                break
            a, o = int(e_in[i]), int(e_out[i])
            self._swap(i, a, o)
            cached[i, a], cached[i, o] = True, False
            done.append(i)
        return done

    @torch.no_grad()
    def after_prefill(self) -> None:
        """Re-pick every layer's cached experts from the prompt's routing (older use counts are halved)."""
        import time
        t0, n0 = time.perf_counter(), self.stats["swaps"]
        self.score.mul_(0.5).add_(self._take())
        cached, touched = self._cached(), set()
        for _ in range(max(m.G.shape[0] for m in self.layers)):     # at most every slot of a layer once
            done = self._round(cached, len(self.layers))
            if not done:
                break
            touched.update(done)
        self._commit(touched)
        torch.cuda.synchronize()
        self.stats["prefill_swaps"] = self.stats["swaps"] - n0
        self.stats["prefill_ms"] = (time.perf_counter() - t0) * 1e3

    @torch.no_grad()
    def after_step(self) -> None:
        self.score.mul_(DECAY).add_(self._take())
        self.stats["steps"] += 1
        if self.swaps > 0:
            self._commit(set(self._round(self._cached(), self.swaps)))


def of(model):
    return getattr(model, "_vahana_adaptive", None)
