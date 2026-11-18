"""Tiered exact MoE execution (chunk 2b).

`TieredMoEWrapper` ports the proven serving paths (dynamic slots, demand DMA,
sorted-dispatch GEMM, CPU executor, ZSSR prefetch, telemetry) from the
DeepSeek-coupled research scaffold to `MoELayerSpec` + explicit per-expert
weight keys. No model names, no key templates, no experimental flags.

Device policy: CUDA streams/events only when `device.type == "cuda"` and a
stream is provided; otherwise plain synchronous copies (CPU-testable).
"""
from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from . import timing as _timing

# Phase 17b: one decode pool per device shared by all MoE layers (only one layer runs at a
# time). Entries: [modules], [free events], next index.
_ANS_POOL: dict = {}
from .interface import MoELayerSpec


class FastSlot(nn.Module):
    """HBM slot holding one whole expert (gate/up [I,H], down [H,I]).

    Gated SwiGLU-family form; activation from spec (silu/gelu/relu).
    Non-gated (up+down only) experts are outside this contract.
    """

    def __init__(self, hidden: int, inter: int, device: torch.device,
                 dtype: torch.dtype = torch.bfloat16,
                 activation: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(hidden, inter, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(inter, hidden, bias=False, device=device, dtype=dtype)
        self.activation = activation
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(apply_gate(self.gate_proj(x), self.up_proj(x),
                                         self.activation))


def apply_gate(g: torch.Tensor, u: torch.Tensor, activation: str) -> torch.Tensor:
    """Gated activation shared by slot and CPU paths (one definition)."""
    if activation == "gelu":
        return F.gelu(g) * u
    if activation == "gelu_tanh":
        return F.gelu(g, approximate="tanh") * u
    if activation == "relu":
        return F.relu(g) * u
    return F.silu(g) * u


def _rusage():
    """Process resource usage (page-fault counters); None where unsupported."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF)
    except Exception:
        return None


def dma_after_compute(dma_stream: Any, device: torch.device) -> None:
    """Order the next copy on `dma_stream` after work queued on compute.

    Slots and cold scratch are reused buffers. Without this wait an async
    copy can overwrite weights a queued GEMM has not read yet
    (write-after-read). Event wait only, no host sync.
    """
    if dma_stream is not None and device.type == "cuda":
        dma_stream.wait_stream(torch.cuda.current_stream(device))


class TieredMoEWrapper(nn.Module):
    """Dynamic slot residency MoE layer (model-agnostic).

    Resident: gate + shared experts (as given). Slots: C whole experts,
    LRU-managed, demand-loaded from `handles` via explicit `expert_keys`.
    Args:
        expert_keys: per-expert-position {gate|up|down: weight key},
            aligned with spec.experts order.
        handles: weight source exposing .get_tensor(key) -> CPU tensor.
        dma_stream: CUDA stream for async copies (None = sync path).
    """

    def __init__(
        self,
        layer_idx: int,
        spec: MoELayerSpec,
        expert_keys: list[dict[str, str]],
        device: torch.device,
        capacity: int,
        handles: Any,
        dma_stream: Any = None,
        zssr_prefetch: bool = False,
        prefetch_topk: int = 8,
        prefetch_conf: float = 0.0,
        cpu_exec: bool = False,
        cpu_cache_cap: int = 128,
        tupled: bool = False,
    ):
        super().__init__()
        assert len(expert_keys) == spec.n_routed, "keys/experts misaligned"
        self.layer_idx = layer_idx
        self.spec = spec
        self.expert_keys = expert_keys
        self.device = device
        self.capacity = capacity
        self.handles = handles
        self.dma_stream = dma_stream
        self.zssr_enabled = bool(zssr_prefetch)
        self.prefetch_topk = int(prefetch_topk)
        self.prefetch_conf = float(prefetch_conf)
        self.cpu_exec = bool(cpu_exec)
        self.cpu_cache_cap = int(cpu_cache_cap)
        # Return convention must match the replaced block: Mixtral-style
        # `block_sparse_moe` forwards return (hidden, router_logits) while
        # `mlp` blocks return a bare tensor. Mapped explicitly (known cases).
        self.tupled = bool(tupled)
        self._cpu_cache: dict[int, dict[str, torch.Tensor]] = {}
        self._cpu_fifo: list[int] = []

        self.gate = spec.gate
        self.shared_experts = spec.shared
        # Streaming floor (capacity 0): one transient slot, bindings cleared
        # after every forward (every needed expert misses, honestly priced).
        self.streaming = (capacity == 0)
        if self.streaming:
            capacity = 1
        self.capacity = capacity
        self.slots: list[nn.Module] = nn.ModuleList([
            FastSlot(spec.hidden, spec.inter, device, spec.dtype, spec.activation)
            for _ in range(capacity)
        ])
        self.slot_to_expert: dict[int, int] = {}
        self.expert_to_slot: dict[int, int] = {}
        self.slot_lru: list[int] = list(range(capacity))

        self.hits = 0
        self.misses = 0
        self.dma_bytes = 0
        self.seed_loads = 0
        # Prefill self-profile: first forward's union + per-expert token
        # frequency (the request's own touch pattern — a zero-mispredict
        # predictor for decode, free with the prefill pass).
        self.prefill_union = None
        self.prefill_freq = None
        self.zssr_predictions = 0
        self.zssr_correct = 0
        self.zssr_suppressed = 0
        self._prefetched: dict[int, int] = {}
        self.routing_log = None
        self.cpu_ms = 0.0
        self.cpu_n = 0
        # Host wall time by phase (step-time attribution): fetching tensors
        # from the weight source, issuing slot copies (blocks the host for
        # pageable sources), and the whole forward call.
        self.fetch_s = 0.0
        self.copy_s = 0.0
        self.forward_s = 0.0
        # Process page faults taken while loading expert weights (minor =
        # page cached but not yet mapped; major = read from disk).
        self.load_minflt = 0
        self.load_majflt = 0
        # Opt-in (COLOSSUS_SLOT_EVENTS=1): order each slot copy after the last
        # GEMM that read THAT slot (per-slot event) instead of after all queued
        # compute, so copy i+1 overlaps compute i. Scheduling only: the same
        # weights reach the same slots before the same GEMMs, so values are
        # unchanged by construction.
        import os as _os
        self.slot_events = (_os.environ.get("COLOSSUS_SLOT_EVENTS", "0") == "1"
                            and device.type == "cuda" and dma_stream is not None)
        self._slot_evt: dict[int, Any] = {}
        # Optional lossless compressed weight source (COLOSSUS_ANS_STORE, see ansplane.py).
        from .ansplane import get_plane
        self.ans = get_plane() if device.type == "cuda" and dma_stream is not None else None
        self.ans_lookahead = int(_os.environ.get("COLOSSUS_ANS_LOOKAHEAD", "8"))
        # Phase 17: decode G missing experts per nvCOMP call (1 = per-expert, as before).
        self.ans_group = int(_os.environ.get("COLOSSUS_ANS_GROUP", "1"))
        self._ans_grouped: set[int] = set()
        self._pool_bind: dict[int, int] = {}
        # Phase 20b (COLOSSUS_SECOND_GPU=1): a second GPU is a second weight link. Alternate
        # groups of this layer's ANS misses are copied/decoded/computed there; outputs come
        # back at the end of the layer. Same bytes, same kernels: values unchanged.
        self.dev2 = None
        self._on2: set[int] = set()
        self._pend2: list = []
        if (self.ans is not None and self.ans_group > 1 and torch.cuda.device_count() > 1
                and _os.environ.get("COLOSSUS_SECOND_GPU") == "1"):
            from .ansplane import get_peer
            self.dev2 = torch.device("cuda", 1 if (device.index or 0) == 0 else 0)
            self.ans2 = get_peer(self.dev2)
        self._ans_todo: dict[int, int] = {}
        self._ans_list: list[int] = []
        # Opt-in (COLOSSUS_XLAYER_PREFETCH=k): at the end of this layer's MoE, load
        # up to k experts of the NEXT MoE layer into its free slots, guessed from that
        # layer's previous forward, so the link is busy during the next attention.
        # Data movement only: a slot always holds its expert's exact weights.
        self.xlayer_k = int(_os.environ.get("COLOSSUS_XLAYER_PREFETCH", "0"))
        self.next_wrapper = None
        self.last_needed: list[int] = []
        self.xlayer_loads = 0
        self.xlayer_used = 0
        self._xlayer_pending: set[int] = set()

    # -- loading ------------------------------------------------------
    def _expert_tensors(self, expert_id: int):
        ks = self.expert_keys[expert_id]
        t0 = time.perf_counter()
        out = (self.handles.get_tensor(ks["gate"]),
               self.handles.get_tensor(ks["up"]),
               self.handles.get_tensor(ks["down"]))
        self.fetch_s += time.perf_counter() - t0
        return out

    def _load_expert_to_slot(self, expert_id: int, slot_idx: int,
                             kind: str = "demand") -> int:
        ru0 = _rusage()
        try:
            return self._load_expert_to_slot_impl(expert_id, slot_idx)
        finally:
            ru1 = _rusage()
            if ru0 is not None and ru1 is not None:
                self.load_minflt += ru1.ru_minflt - ru0.ru_minflt
                self.load_majflt += ru1.ru_majflt - ru0.ru_majflt

    def _ans_keys(self, expert_id: int) -> list[str]:
        ks = self.expert_keys[expert_id]
        return [ks["gate"], ks["up"], ks["down"]]

    def _plane_for(self, e: int):
        if e in self._on2:
            return self.ans2, self.ans2.dma_stream, self.dev2
        return self.ans, self.dma_stream, self.device

    def _ans_stage_ahead(self, expert_id: int) -> None:
        """After loading expert_id, issue copies for the expert `lookahead` places later."""
        p = self._ans_todo.get(expert_id)
        if p is None:
            return
        q = p + self.ans_lookahead
        if q < len(self._ans_list):
            e = self._ans_list[q]
            ks = self._ans_keys(e)
            pl, ds, _ = self._plane_for(e)
            if pl.has(ks):
                pl.stage((self.layer_idx, e), ks, ds)

    def _ans_load_group(self, first: int) -> bool:
        """Bind up to `ans_group` upcoming ANS misses (from `first`, in loop order) to LRU
        slots and decode them with one call. Returns False if `first` is not servable."""
        p = self._ans_todo.get(first)
        if p is None or not self.ans.has(self._ans_keys(first)):
            return False
        group = []
        for e in self._ans_list[p:]:
            if len(group) == self.ans_group:
                break
            if e in self.expert_to_slot or e in self._pool_bind:
                continue
            if (e in self._on2) != (first in self._on2):
                break
            if not self.ans.has(self._ans_keys(e)):
                break
            group.append(e)
        t0 = time.perf_counter()
        plane, dstream, dev = self._plane_for(first)
        pool = _ANS_POOL.get(str(dev))
        if pool is None:
            import copy
            mods = [copy.deepcopy(self.slots[0]).to(dev) for _ in range(2 * self.ans_group)]
            pool = _ANS_POOL[str(dev)] = {"mods": mods, "free": [None] * len(mods), "next": 0}
        tags, keys, dsts, frees = [], [], [], []
        for e in group:
            ps = pool["next"] % len(pool["mods"])
            pool["next"] += 1
            free = pool["free"][ps]
            if free is None and not self.slot_events:
                free = torch.cuda.Event()
                free.record(torch.cuda.current_stream(dev))
            slot = pool["mods"][ps]
            self._pool_bind[e] = ps
            d = (slot.gate_proj.weight, slot.up_proj.weight, slot.down_proj.weight)
            tags.append((self.layer_idx, e))
            keys.append(self._ans_keys(e))
            dsts.append(d)
            frees.append(free)
            self.misses += 1
            self.dma_bytes += int(sum(w.numel() * w.element_size() for w in d))
        with torch.no_grad(), torch.cuda.device(dev):
            plane.finish_group(tags, keys, dsts, dstream, frees)
        if dev != self.device:
            torch.cuda.current_stream(dev).wait_stream(plane.dec_stream)
        for e in group:
            self._ans_stage_ahead(e)
        self._ans_grouped.update(group)
        self.copy_s += time.perf_counter() - t0
        return True

    def _load_expert_to_slot_impl(self, expert_id: int, slot_idx: int) -> int:
        slot = self.slots[slot_idx]
        if self.ans is not None and self.ans.has(self._ans_keys(expert_id)):
            t0 = time.perf_counter()
            free = self._slot_evt.get(slot_idx) if self.slot_events else None
            if free is None and not self.slot_events:
                free = torch.cuda.Event()            # stock ordering: after all queued compute
                free.record(torch.cuda.current_stream(self.device))
            dsts = (slot.gate_proj.weight, slot.up_proj.weight, slot.down_proj.weight)
            with torch.no_grad():
                self.ans.finish((self.layer_idx, expert_id), self._ans_keys(expert_id), dsts,
                                self.dma_stream, slot_free=free)
            self._ans_stage_ahead(expert_id)
            self.copy_s += time.perf_counter() - t0
            if slot_idx in self.slot_to_expert:
                self.expert_to_slot.pop(self.slot_to_expert.pop(slot_idx), None)
            self.slot_to_expert[slot_idx] = expert_id
            self.expert_to_slot[expert_id] = slot_idx
            nbytes = int(sum(w.numel() * w.element_size() for w in dsts))
            self.dma_bytes += nbytes
            return nbytes
        t_gate, t_up, t_down = self._expert_tensors(expert_id)
        nbytes = int(t_gate.nbytes + t_up.nbytes + t_down.nbytes)
        use_stream = (self.dma_stream is not None and self.device.type == "cuda")
        # Diagnostic: time each load through to completion. Pinned sources
        # copy asynchronously, so issue time alone would flatter them; queued
        # compute is drained first so neither source is charged for it.
        sync = use_stream and getattr(self, "sync_loads", False)
        if sync:
            torch.cuda.current_stream(self.device).synchronize()
            self.dma_stream.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            if use_stream:
                if self.slot_events:
                    ev = self._slot_evt.get(slot_idx)
                    if ev is not None:
                        self.dma_stream.wait_event(ev)
                else:
                    dma_after_compute(self.dma_stream, self.device)
                with torch.cuda.stream(self.dma_stream):
                    slot.gate_proj.weight.copy_(t_gate, non_blocking=True)
                    slot.up_proj.weight.copy_(t_up, non_blocking=True)
                    slot.down_proj.weight.copy_(t_down, non_blocking=True)
                if sync:
                    self.dma_stream.synchronize()
            else:
                slot.gate_proj.weight.copy_(t_gate)
                slot.up_proj.weight.copy_(t_up)
                slot.down_proj.weight.copy_(t_down)
        self.copy_s += time.perf_counter() - t0
        if slot_idx in self.slot_to_expert:
            self.expert_to_slot.pop(self.slot_to_expert.pop(slot_idx), None)
        self.slot_to_expert[slot_idx] = expert_id
        self.expert_to_slot[expert_id] = slot_idx
        self.dma_bytes += nbytes
        return nbytes

    def prefetch_guess(self, k: int) -> None:
        """Load the first k experts this layer used last time (index order, as the
        dispatch loop visits them) into its least-recently-used slots."""
        want = [e for e in self.last_needed if e not in self.expert_to_slot][:min(k, self.capacity)]
        self._ans_todo, self._ans_list = {}, []      # no ANS stage-ahead from a guess
        with torch.no_grad():
            for e in want:
                slot_idx = self.slot_lru.pop(0)
                self._load_expert_to_slot(e, slot_idx, kind="prefetch")
                self.slot_lru.append(slot_idx)
                self.xlayer_loads += 1
                self._xlayer_pending.add(e)

    def _sync_dma(self) -> None:
        if self.dma_stream is not None and self.device.type == "cuda":
            torch.cuda.current_stream(self.device).wait_stream(self.dma_stream)
            if self.ans is not None:
                torch.cuda.current_stream(self.device).wait_stream(self.ans.dec_stream)

    def seed_from_prefill(self, policy: str = "freq") -> int:
        """Pre-position decode slots from the prefill self-profile.

        Policies: "freq" (load top-C prefill-frequent experts, most
        frequent most-recent), anything else = no-op (control: LRU tail
        left by prefill). Data movement only — values never change, so
        exactness is unaffected by construction. Returns seed loads.
        """
        if policy != "freq" or not self.prefill_union:
            return 0
        import torch as _torch
        freq = self.prefill_freq or [1] * self.spec.n_routed
        # Ascending frequency; keep the top-C tail so the most frequent
        # expert is loaded last (most recent in LRU).
        order = sorted(self.prefill_union,
                       key=lambda e: freq[e] if e < len(freq) else 0)
        loaded = 0
        with torch.no_grad():
            for exp_id in order[-self.capacity:]:
                if exp_id in self.expert_to_slot:
                    slot_idx = self.expert_to_slot[exp_id]
                    self.slot_lru.remove(slot_idx)
                    self.slot_lru.append(slot_idx)
                    continue
                self.misses += 1
                slot_idx = self.slot_lru.pop(0)
                self._load_expert_to_slot(exp_id, slot_idx)
                self.slot_lru.append(slot_idx)
                loaded += 1
        self._sync_dma()
        self.seed_loads += loaded
        return loaded

    # -- ZSSR prefetch (whole experts; prediction moves data only) ----
    def zssr_prefetch(self, hidden_in: torch.Tensor):
        if not self.zssr_enabled:
            return []
        try:
            rw = self.spec.router_weight()
            if rw is None:
                return []
            h = hidden_in[:, -1, :].to(dtype=torch.float32, device=rw.device)
            scores = torch.softmax(h @ rw.detach().t(), dim=-1)
            k = min(self.prefetch_topk, self.spec.n_routed)
            topv, topi = torch.topk(scores, k=k, dim=-1)
            conf = topv.view(-1)[0].item()
            pred = topi.view(-1).tolist()
        except Exception:
            return []
        if conf < self.prefetch_conf:
            self.zssr_suppressed += 1
            return []
        new_ids = [e for e in dict.fromkeys(pred)
                   if e not in self.expert_to_slot and e not in self._prefetched]
        if not new_ids:
            return pred
        for exp_id in new_ids:
            slot_idx = self.slot_lru.pop(0)
            nbytes = self._load_expert_to_slot(exp_id, slot_idx, kind="prefetch")
            self.slot_lru.append(slot_idx)
            self._prefetched[exp_id] = self._prefetched.get(exp_id, 0) + nbytes
        self._sync_dma()
        self.zssr_predictions += len(new_ids)
        return pred

    # -- CPU executor (oneDNN BF16 GEMV; ulp1 grade) -------------------
    def _cpu_weights(self, expert_id: int):
        w = self._cpu_cache.get(expert_id)
        if w is None:
            tg, tu, td = self._expert_tensors(expert_id)
            w = {"gate": tg, "up": tu, "down": td}
            self._cpu_cache[expert_id] = w
            self._cpu_fifo.append(expert_id)
            if len(self._cpu_fifo) > self.cpu_cache_cap:
                self._cpu_cache.pop(self._cpu_fifo.pop(0), None)
        return w

    def _cpu_moe_forward(self, hidden_states, shared_out, topk_indices, topk_weights,
                         orig_shape):
        t0 = time.perf_counter()
        x_cpu = hidden_states.detach().to("cpu")
        flat = x_cpu.reshape(-1, x_cpu.shape[-1])
        K = topk_indices.shape[-1]
        TI = topk_indices.reshape(-1, K).cpu()
        TW = topk_weights.reshape(-1, K).cpu()
        groups: dict[int, list] = {}
        for b in range(TI.shape[0]):
            for k in range(TI.shape[1]):
                groups.setdefault(int(TI[b, k]), []).append((b, k))
        out = torch.zeros_like(flat)
        act = self.spec.activation
        for e, poses in groups.items():
            w = self._cpu_weights(e)
            rows = torch.tensor([b for b, _ in poses])
            xe = flat[rows]
            with torch.no_grad():
                ye = F.linear(apply_gate(F.linear(xe, w["gate"]),
                                         F.linear(xe, w["up"]), act),
                              w["down"])
            for (b, k), yrow in zip(poses, ye):
                out[b] += yrow * TW[b, k]
        ret = shared_out + out.reshape(orig_shape).to(
            shared_out.device, dtype=shared_out.dtype)
        self.cpu_ms += (time.perf_counter() - t0) * 1000.0
        self.cpu_n += 1
        return ret

    # -- forward -------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        t0 = time.perf_counter()
        try:
            return self._forward(hidden_states)
        finally:
            self.forward_s += time.perf_counter() - t0

    def _forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self._ans_todo, self._ans_list = {}, []
        self._ans_grouped = set()
        self._pool_bind = {}
        self._on2 = set()
        self._pend2 = []
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.spec.route(hidden_states)
        # Dispatch math is 2D [N, K] regardless of gate rank conventions.
        K = topk_indices.shape[-1]
        topk_indices = topk_indices.reshape(-1, K)
        topk_weights = topk_weights.reshape(-1, K)
        _T = _timing.REC
        _h = time.perf_counter()
        needed_experts = topk_indices.unique().tolist()
        if _T is not None:
            _T.add_host("route_sync", time.perf_counter() - _h)
            if self.dma_stream is not None:
                _dma0 = _T.ev(self.dma_stream)
        if self._xlayer_pending:
            self.xlayer_used += len(self._xlayer_pending & set(needed_experts))
            self._xlayer_pending = set()
        self.last_needed = needed_experts
        if self.routing_log is not None:
            self.routing_log.append((self.layer_idx, needed_experts))
        if self.prefill_union is None:
            # First forward is the prefill pass: record its union and
            # per-expert token counts (single bincount, off the hot path
            # thereafter). Values untouched — telemetry only.
            self.prefill_union = list(needed_experts)
            try:
                self.prefill_freq = torch.bincount(
                    topk_indices.reshape(-1).long(),
                    minlength=self.spec.n_routed).tolist()
            except Exception:
                self.prefill_freq = [1] * self.spec.n_routed

        if self._prefetched:
            actual = set(needed_experts)
            for exp_id in list(self._prefetched.keys()):
                if exp_id in actual:
                    self.zssr_correct += 1
                    self._prefetched.pop(exp_id)

        shared_out = self.shared_experts(identity) if self.shared_experts is not None \
            else torch.zeros_like(identity)
        if self.cpu_exec:
            out = self._cpu_moe_forward(hidden_states, shared_out,
                                        topk_indices, topk_weights, orig_shape)
            if self.tupled:
                return out, None
            return out

        if len(needed_experts) <= self.capacity:
            for exp_id in needed_experts:
                if exp_id in self.expert_to_slot:
                    self.hits += 1
                    slot_idx = self.expert_to_slot[exp_id]
                    self.slot_lru.remove(slot_idx)
                    self.slot_lru.append(slot_idx)
            missing = [e for e in needed_experts if e not in self.expert_to_slot]
            for exp_id in missing:
                self.misses += 1
                slot_idx = self.slot_lru.pop(0)
                self._load_expert_to_slot(exp_id, slot_idx)
                self.slot_lru.append(slot_idx)
            self._sync_dma()

        cnts = topk_indices.new_zeros((topk_indices.shape[0], self.spec.n_routed))
        cnts.scatter_(1, topk_indices, 1)
        _h = time.perf_counter()
        tokens_per_expert = cnts.sum(dim=0).cpu().numpy()
        if _T is not None:
            _T.add_host("count_sync", time.perf_counter() - _h)
        if self.ans is not None:
            # Copies for the first `lookahead` misses go out now, so the link stays busy.
            self._ans_list = [i for i, n in enumerate(tokens_per_expert)
                              if n > 0 and i not in self.expert_to_slot]
            self._ans_todo = {e: p for p, e in enumerate(self._ans_list)}
            self.ans.drop_layer(self.layer_idx, keep=set(self._ans_list))
            if self.dev2 is not None:
                G = self.ans_group
                self._on2 = {e for q, e in enumerate(self._ans_list) if (q // G) % 2 == 1}
                self.ans2.drop_layer(self.layer_idx, keep=self._on2)
            for e in self._ans_list[:self.ans_lookahead]:
                ks = self._ans_keys(e)
                pl, ds, _ = self._plane_for(e)
                if pl.has(ks):
                    pl.stage((self.layer_idx, e), ks, ds)
        idxs = topk_indices.view(-1).argsort()
        flat_x = hidden_states.reshape(-1, hidden_states.shape[-1])
        sorted_tokens = flat_x[idxs // topk_indices.shape[1]]
        outputs, start_idx = [], 0
        _prev = _T.ev() if _T is not None else None
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            if (i not in self.expert_to_slot and i not in self._pool_bind and self.ans is not None
                    and self.ans_group > 1 and len(needed_experts) > self.capacity
                    and self._ans_load_group(i)):
                self._sync_dma()
            if i in self._pool_bind and i in self._on2:
                dev = self.dev2
                pool = _ANS_POOL[str(dev)]
                ps = self._pool_bind.pop(i)
                with torch.cuda.device(dev):
                    x2 = sorted_tokens[start_idx:end_idx].to(dev, non_blocking=True)
                    y2 = pool["mods"][ps](x2)
                    fe = torch.cuda.Event()
                    fe.record(torch.cuda.current_stream(dev))
                    pool["free"][ps] = fe
                self._pend2.append((len(outputs), y2))
                outputs.append(None)
                start_idx = end_idx
                continue
            if i in self._pool_bind:
                pool = _ANS_POOL[str(self.device)]
                ps = self._pool_bind.pop(i)
                if _T is not None:
                    _a = _T.ev()
                    _T.span("moe_stall", _prev, _a)
                outputs.append(pool["mods"][ps](sorted_tokens[start_idx:end_idx]))
                fe = torch.cuda.Event()
                fe.record(torch.cuda.current_stream(self.device))
                pool["free"][ps] = fe
                if _T is not None:
                    _prev = _T.ev()
                    _T.span("expert_gemm", _a, _prev)
                start_idx = end_idx
                continue
            if i in self.expert_to_slot:
                slot_idx = self.expert_to_slot[i]
                if i in self._ans_grouped:
                    self._ans_grouped.discard(i)       # loaded this forward: not a hit
                elif len(needed_experts) > self.capacity:
                    self.hits += 1
                    self.slot_lru.remove(slot_idx)
                    self.slot_lru.append(slot_idx)
            else:
                self.misses += 1
                slot_idx = self.slot_lru.pop(0)
                self._load_expert_to_slot(i, slot_idx)
                self.slot_lru.append(slot_idx)
                self._sync_dma()
            expert = self.slots[slot_idx]
            if _T is not None:
                _a = _T.ev()
                _T.span("moe_stall", _prev, _a)
            outputs.append(expert(sorted_tokens[start_idx:end_idx]))
            if _T is not None:
                _prev = _T.ev()
                _T.span("expert_gemm", _a, _prev)
            if self.slot_events:
                ev = torch.cuda.Event()
                ev.record(torch.cuda.current_stream(self.device))
                self._slot_evt[slot_idx] = ev
            start_idx = end_idx
        if _T is not None and self.dma_stream is not None:
            _T.span("copy_layer", _dma0, _T.ev(self.dma_stream))
            if self.ans is not None:
                _T.span("decode_layer", _dma0, _T.ev(self.ans.dec_stream))
        for k, y2 in self._pend2:
            outputs[k] = y2.to(self.device, non_blocking=True)
        self._pend2 = []
        outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (new_x.view(*topk_indices.shape, -1)
                     .type(topk_weights.dtype)
                     .mul_(topk_weights.unsqueeze(dim=-1))
                     .sum(dim=1)
                     .type(new_x.dtype))
        out = shared_out + final_out.view(*orig_shape)
        nxt = self.next_wrapper
        if self.xlayer_k > 0 and nxt is not None and nxt.last_needed and not nxt.streaming:
            nxt.prefetch_guess(self.xlayer_k)
        if self.streaming:
            # Transient slot: bindings die here; next forward misses everything.
            self.expert_to_slot.clear()
            self.slot_to_expert.clear()
            self.slot_lru = list(range(len(self.slots)))
        if self.tupled:
            return out, None
        return out


def wrap_moe_layers(model: nn.Module, profile: Any, handles: Any,
                    device: torch.device, capacity: int,
                    dma_stream: Any = None, **opts) -> list[TieredMoEWrapper]:
    """Replace every MoE block with a TieredMoEWrapper (model-agnostic).

    Discovery: profile.decoder_layer_cls instances -> find_moe_block each.
    Weight keys: expert dotted paths (from named_modules) grouped by
    interface.expert_weight_keys against handles.weight_map.
    wrapper_cls: alternate executor with the same constructor contract
    (e.g. column-granular slots); extra opts pass through.
    Returns wrappers in layer order.
    """
    from .interface import MoELayerSpec, expert_weight_keys, find_moe_block

    import inspect as _inspect
    wrapper_cls = opts.pop("wrapper_cls", TieredMoEWrapper)
    hot_tiers = opts.pop("hot_tiers", None)
    tier_pools = opts.pop("tier_pools", None)
    try:
        _takes_fracs = "hot_fracs" in _inspect.signature(
            wrapper_cls.__init__).parameters
    except (TypeError, ValueError):
        _takes_fracs = False
    decoder_cls = getattr(profile, "decoder_layer_cls", None)
    if decoder_cls is not None:
        layers = [m for m in model.modules() if isinstance(m, decoder_cls)]
    else:
        layers = list(getattr(getattr(model, "model", model), "layers", []))
    dotted = {id(m): name for name, m in model.named_modules()}
    weight_keys = list(handles.weight_map.keys())
    wrappers = []
    for layer_idx, layer in enumerate(layers):
        found = find_moe_block(layer)
        if found is None:
            continue
        attr, block = found
        spec = MoELayerSpec.from_block(block)
        prefix = dotted.get(id(block), "")
        keys = []
        for e in range(spec.n_routed):
            if not prefix:
                raise KeyError(f"layer {layer_idx}: no dotted path for MoE block")
            dotted_e = f"{prefix}.{spec.container_attr}.{e}"
            keys.append(expert_weight_keys(dotted_e, weight_keys))
        wkw = dict(opts)
        if hot_tiers is not None and _takes_fracs:
            fr = hot_tiers.get(layer_idx)
            if fr is not None:
                wkw["hot_fracs"] = [float(f) for f in fr]
            if tier_pools is not None:
                wkw["tier_pools"] = {float(k): int(v)
                                     for k, v in tier_pools.items()}
        wrapper = wrapper_cls(
            layer_idx=layer_idx, spec=spec, expert_keys=keys, device=device,
            capacity=capacity, handles=handles, dma_stream=dma_stream,
            tupled=(attr == "block_sparse_moe"), **wkw)
        setattr(layer, attr, wrapper)
        if wrappers:
            wrappers[-1].next_wrapper = wrapper
        wrappers.append(wrapper)
    _want = int(getattr(profile, "num_experts", 0) or 0)
    _grouped = bool(getattr(profile, "grouped_moe", False))
    if _want > 0 and not wrappers and not _grouped:
        raise RuntimeError(
            "wrap_moe_layers: profile declares MoE experts but zero blocks "
            "were wrapped (block discovery failed silently)")
    return wrappers
