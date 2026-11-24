"""v0.4-b: exact continuous batching. Requests join and leave between decode steps; each gets exactly the bits it
gets when decoded alone.

Layout: S slots share one KV cache of C columns. Every active row ends at the same column c (the next write
position); a request's keys are contiguous in its row from its start column s to c, and its positions (0, 1, ...)
travel in position_ids. Vahana's attention visits each query's own keys in blocks anchored at its first allowed key
and every other kernel is row-wise, so a row computes exactly what a solo run computes (v0.3-e argument).

  admit   waiting requests are prefilled together as a left-padded batch (exact), then each one's KV is copied bit
          for bit into a free slot at columns [c - L, c). If c < L every active row is shifted right first.
  step    one CUDA graph per batch bucket (1, 2, 4, ... S); each bucket's cache layers are views of the first rows of
          the shared cache. If c reaches C, every row is shifted left by the smallest start (a copy).
  leave   a finished row is replaced by the last active row (a copy), so rows 0..n-1 stay packed.
Shifts and moves copy KV; positions never change, so nothing a row computes depends on where it sits.
Models with recurrent / linear-attention layers use HybridEngine below (v0.4-c); make() picks the right one.
"""
from __future__ import annotations

import collections
import itertools
import threading

import torch

from . import runtime as _rt
from .decode import _sync


def supported(model) -> bool:
    kinds = set(getattr(model.config.get_text_config(), "layer_types", None) or [])
    return not (kinds - {"full_attention", "sliding_attention", "attention"})


class Request:
    def __init__(self, rid, ids, max_new_tokens, eos, sampler, on_token, record_logits):
        self.rid, self.prompt = rid, [int(t) for t in ids]
        self.max_new, self.eos = int(max_new_tokens), set(eos or ())
        self.sampler, self.on_token, self.record = sampler, on_token, record_logits
        self.ids: list[int] = []
        self.logits: list[torch.Tensor] = []
        self.s = 0                      # start column in the shared cache
        self.done = threading.Event()
        self.finish = None
        self.error = None

    def _emit(self, t: int, logit: torch.Tensor | None) -> bool:
        """Record one token; True when the request is finished."""
        self.ids.append(t)
        if self.record and logit is not None:
            self.logits.append(logit.detach().to("cpu", torch.float32).clone())
        stop = bool(self.on_token(t)) if self.on_token is not None else False
        if t in self.eos:
            self.finish = "stop"
        elif stop:
            self.finish = "stop"
        elif len(self.ids) >= self.max_new:
            self.finish = "length"
        return self.finish is not None


def _static_cache(model, length):
    from transformers import StaticCache
    from transformers.cache_utils import StaticLayer, StaticSlidingWindowLayer
    c = StaticCache(config=model.config, max_cache_len=length)
    c.layers = [StaticLayer(max_cache_len=length) if isinstance(x, StaticSlidingWindowLayer) else x for x in c.layers]
    return c


class Engine:
    """Continuous-batching greedy / seeded-sampling decoder. submit() from any thread; step() (or run()) from one."""

    def __init__(self, model, slots: int = 16, max_len: int = 4096, graph: bool = True):
        if not supported(model):
            raise ValueError("continuous batching needs attention-only layers (this model has recurrent ones)")
        self.model = model
        self.dev = next(p.device for p in model.parameters() if p.device.type == "cuda") \
            if any(p.is_cuda for p in model.parameters()) else torch.device("cpu")
        self.graph = graph and self.dev.type == "cuda"
        self.S, self.C = int(slots), int(max_len)
        self.buckets = sorted({min(self.S, 1 << i) for i in range(0, self.S.bit_length() + 1)})
        self.V = model.config.get_text_config().vocab_size
        self.waiting: collections.deque[Request] = collections.deque()
        self.lock = threading.Lock()
        self.active: list[Request] = []          # row r holds active[r]
        self.c = 0                                # next write column
        self._ids = itertools.count()
        self.big_k = self.big_v = None            # per layer [S, Hkv, C, D]
        self.cl = None                            # per layer cumulative_length (shared by every bucket's layer i)
        self.am = torch.zeros(self.S, self.C, dtype=torch.long, device=self.dev)
        self.tok = torch.zeros(self.S, 1, dtype=torch.long, device=self.dev)
        self.pos = torch.zeros(self.S, 1, dtype=torch.long, device=self.dev)
        self.cpos = torch.zeros(1, dtype=torch.long, device=self.dev)
        self.ones = torch.ones(self.S, 1, dtype=torch.long, device=self.dev)
        self.lbuf = torch.empty(self.S, self.V, dtype=torch.float32, device=self.dev)
        self._b = {}                              # bucket -> {"cache", "graph", "warm", "fallback"}
        self.stats = {"steps": 0, "step_ms": [], "prefills": 0, "shift_right": 0, "shift_left": 0, "moves": 0,
                      "bucket_tokens": 0, "active_tokens": 0}

    # ---- public ----
    def submit(self, ids, max_new_tokens: int = 64, eos_token_id=None, sampler=None, on_token=None,
               record_logits: bool = False) -> Request:
        eos = eos_token_id if isinstance(eos_token_id, (list, tuple, set)) else (
            () if eos_token_id is None else (eos_token_id,))
        r = Request(next(self._ids), ids, max_new_tokens, eos, sampler, on_token, record_logits)
        if len(r.prompt) == 0 or len(r.prompt) + r.max_new > self.C:
            r.error = f"prompt ({len(r.prompt)}) + max_new_tokens ({r.max_new}) must be in [1, {self.C}]"
            r.finish = "error"
            r.done.set()
            return r
        with self.lock:
            self.waiting.append(r)
        return r

    def idle(self) -> bool:
        with self.lock:
            return not self.waiting and not self.active

    def run(self):
        """Step until nothing is waiting or active."""
        while not self.idle():
            self.step()

    @torch.no_grad()
    def step(self):
        """Admit what fits, then one decode step for every active request."""
        self._admit()
        if not self.active:
            return
        if self.c >= self.C:
            self._shift(-min(r.s for r in self.active))
        n = len(self.active)
        b = next(x for x in self.buckets if x >= n)
        import time
        _sync(self.dev)
        t0 = time.perf_counter()
        self._decode(b)
        _sync(self.dev)
        self.stats["step_ms"].append((time.perf_counter() - t0) * 1e3)
        self.stats["steps"] += 1
        self.stats["bucket_tokens"] += b
        self.stats["active_tokens"] += n
        self.c += 1
        toks = self.tok[:n, 0].tolist()
        fin = []
        for r, req in enumerate(self.active):
            t = toks[r]
            if req.sampler is not None:
                t = int(req.sampler(self.lbuf[r:r + 1]).view(-1)[0])
                self.tok[r, 0] = t
            if req._emit(t, self.lbuf[r] if req.record else None):
                fin.append(r)
        for r in sorted(fin, reverse=True):
            self._leave(r)

    # ---- internals ----
    def _init_cache(self, pc):
        self.big_k, self.big_v, self.cl = [], [], []
        for L in pc.layers:
            k, v = L.keys, L.values
            self.big_k.append(torch.zeros((self.S, k.shape[1], self.C, k.shape[3]), dtype=k.dtype, device=k.device))
            self.big_v.append(torch.zeros((self.S, v.shape[1], self.C, v.shape[3]), dtype=v.dtype, device=v.device))
            self.cl.append(torch.zeros(1, dtype=torch.long, device=k.device))

    def _bucket_cache(self, b):
        from transformers.cache_utils import StaticLayer
        c = _static_cache(self.model, self.C)
        layers = []
        for i in range(len(self.big_k)):
            L = StaticLayer(max_cache_len=self.C)
            k, v = self.big_k[i][:b], self.big_v[i][:b]
            L.dtype, L.device = k.dtype, k.device
            L.max_batch_size, L.num_heads = b, k.shape[1]
            L.k_head_dim, L.v_head_dim = k.shape[3], v.shape[3]
            L.keys, L.values, L.cumulative_length = k, v, self.cl[i]
            L.is_initialized = True
            layers.append(L)
        c.layers = layers
        return c

    def _set_c(self, c):
        self.c = c
        self.cpos.fill_(c)
        for t in self.cl:
            t.fill_(c)

    def _mask_row(self, r):
        req = self.active[r]
        self.am[r].zero_()
        self.am[r, req.s:self.c] = 1

    def _shift(self, d):
        """Move every active row's span [s, c) by d columns (a copy), and c with it."""
        if d == 0 or not self.active:
            return
        n, lo, hi = len(self.active), min(r.s for r in self.active), self.c
        for B in (self.big_k, self.big_v):
            for t in B:
                t[:n, :, lo + d:hi + d] = t[:n, :, lo:hi].clone()
        for r in self.active:
            r.s += d
        self._set_c(self.c + d)
        for r in range(n):
            self._mask_row(r)
        self.stats["shift_right" if d > 0 else "shift_left"] += 1

    def _leave(self, r):
        req = self.active[r]
        last = len(self.active) - 1
        if r != last:                                   # move the last row into r (a copy of its span)
            q = self.active[last]
            for B in (self.big_k, self.big_v):
                for t in B:
                    t[r, :, q.s:self.c] = t[last, :, q.s:self.c]
            self.tok[r] = self.tok[last]
            self.pos[r] = self.pos[last]
            self.active[r] = q
            self._mask_row(r)
            self.stats["moves"] += 1
        self.active.pop()
        self.am[last].zero_()
        req.done.set()

    def _admit(self):
        with self.lock:
            k = min(self.S - len(self.active), len(self.waiting))
            new = [self.waiting.popleft() for _ in range(k)]
        if not new:
            return
        T = max(len(r.prompt) for r in new)
        ids = torch.zeros(len(new), T, dtype=torch.long)
        amp = torch.zeros(len(new), T, dtype=torch.long)
        for j, r in enumerate(new):
            ids[j, T - len(r.prompt):] = torch.tensor(r.prompt)
            amp[j, T - len(r.prompt):] = 1
        ids, amp = ids.to(self.dev), amp.to(self.dev)
        pos0 = (amp.cumsum(-1) - 1).clamp(min=0)
        pc = _static_cache(self.model, T)
        out = self.model(ids, attention_mask=amp, position_ids=pos0, past_key_values=pc,
                         cache_position=torch.arange(T, device=self.dev), use_cache=True, logits_to_keep=1)
        logit = out.logits[:, -1].float()
        first = logit.argmax(-1).tolist()
        self.stats["prefills"] += 1
        if self.big_k is None:
            self._init_cache(pc)
        keep = []
        for j, r in enumerate(new):
            t = first[j] if r.sampler is None else int(r.sampler(logit[j:j + 1]).view(-1)[0])
            if r._emit(t, logit[j]):
                r.done.set()                            # finished on its first token: never takes a slot
            else:
                keep.append((j, r, t))
        if not keep:
            return
        Lmax = max(len(r.prompt) for _, r, _ in keep)
        if not self.active:
            self._set_c(Lmax)
        elif self.c < Lmax:
            self._shift(Lmax - self.c)
        for j, r, t in keep:
            L, row = len(r.prompt), len(self.active)
            r.s = self.c - L
            for B, P in ((self.big_k, [x.keys for x in pc.layers]), (self.big_v, [x.values for x in pc.layers])):
                for dst, src in zip(B, P):
                    dst[row, :, r.s:self.c] = src[j, :, T - L:T]
            self.active.append(r)
            self._mask_row(row)
            self.tok[row, 0] = t
            self.pos[row, 0] = L
        del pc, out

    def _step_fn(self, b, cache):
        am, tok, pos, lbuf, ones = self.am[:b], self.tok[:b], self.pos[:b], self.lbuf[:b], self.ones[:b]
        cpos = self.cpos

        def step():
            am.scatter_(1, cpos.view(1, 1).expand(b, 1), ones)
            o = self.model(tok, attention_mask=am, position_ids=pos, past_key_values=cache, cache_position=cpos,
                           use_cache=True, logits_to_keep=1)
            lbuf.copy_(o.logits[:, -1].float())
            tok.copy_(lbuf.argmax(-1, keepdim=True))
            pos.add_(1)
            cpos.add_(1)
        return step

    def _decode(self, b):
        st = self._b.get(b)
        if st is None:
            cache = self._bucket_cache(b)
            st = self._b[b] = {"cache": cache, "fn": self._step_fn(b, cache), "graph": None, "warm": False,
                               "fallback": None}
        if not self.graph or st["fallback"] is not None:
            st["fn"]()
            return
        if not st["warm"]:                              # first run: compile kernels / allocate pinned buffers
            _rt.GRAPH["on"] = True
            try:
                st["fn"]()
            finally:
                _rt.GRAPH["on"] = False
            st["warm"] = True
            return
        if st["graph"] is None:
            g = torch.cuda.CUDAGraph()
            pool = torch.cuda.graph_pool_handle()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            _rt.GRAPH["on"] = True
            inner = []
            import gc
            gc_was = gc.isenabled()
            gc.disable()        # a collection during capture could free another graph (a stream-capture error)
            try:
                with torch.cuda.stream(s), torch.cuda.graph(g, pool=pool, stream=s):
                    try:
                        st["fn"]()                      # recorded, not executed
                    except Exception:  # noqa: BLE001 - keep the real cause, not capture_end's
                        import traceback
                        inner.append(traceback.format_exc())
                        raise
            except Exception as e:  # noqa: BLE001 - not capturable: this bucket runs eagerly
                st["capture_trace"] = inner[0][-1500:] if inner else None
                old = getattr(torch._C, "_cuda_endAllocateCurrentStreamToPool", None)
                for f in ((old, torch._C._cuda_releasePool) if old is not None else ()):
                    try:
                        f(self.dev.index if self.dev.index is not None else torch.cuda.current_device(), pool)
                    except Exception:  # noqa: BLE001
                        pass
                _sync(self.dev)
                st["fallback"] = f"{type(e).__name__}: {str(e)[:200]}"
                g = None
            finally:
                _rt.GRAPH["on"] = False
                if gc_was:
                    gc.enable()
            torch.cuda.current_stream().wait_stream(s)
            _sync(self.dev)
            if g is None:
                st["fn"]()
                return
            st["graph"] = g
        st["graph"].replay()

    def fail_all(self, err: str):
        """After an exception in step(): end every active and waiting request with `err` and reset the slots."""
        with self.lock:
            reqs = list(self.active) + list(self.waiting)
            self.waiting.clear()
        self.active.clear()
        self.am.zero_()
        for r in reqs:
            r.error, r.finish = err, "error"
            r.done.set()

    def close(self):
        """Release the graphs and the cache now (the step closures refer back to the engine, so without this they
        wait for the garbage collector)."""
        _sync(self.dev)
        self._b.clear()
        self.big_k = self.big_v = self.cl = None
        self.active.clear()

    def graph_info(self) -> dict:
        return {b: ("graph" if v["graph"] is not None else ("eager: " + v["fallback"] if v["fallback"] else "warm"))
                for b, v in self._b.items()}


# ---- v0.4-c: models with recurrent / linear-attention layers -------------------------------------------------------
_KV = ("keys", "values")                              # [B, H, T, D]: length on dim 2, left-padded to a common length
_STATE = ("conv_states", "recurrent_states")          # [B, ...]: one state per row, no length


def _fields(cache):
    """Every per-row tensor of a dynamic (hybrid) cache: (holder, name or index, is_kv)."""
    out = []
    for L in cache.layers:
        extra = [k for k, v in vars(L).items() if isinstance(v, torch.Tensor) and k not in _KV + _STATE]
        if extra:
            raise ValueError(f"unknown per-row cache state {extra} in {type(L).__name__}")
        for k in _KV + _STATE:
            if isinstance(getattr(L, k, None), torch.Tensor):
                out.append((L, k, k in _KV))
    lc = getattr(cache, "linear_cache", None)          # MiniMaxCache keeps its linear-attention state here
    if isinstance(lc, list):
        for i, v in enumerate(lc):
            if isinstance(v, torch.Tensor):
                out.append((lc, i, False))
    return out


def _get(h, k):
    return h[k] if isinstance(h, list) else getattr(h, k)


def _set(h, k, v):
    if isinstance(h, list):
        h[k] = v
    else:
        setattr(h, k, v)


def hybrid_supported(model) -> bool:
    kinds = set(getattr(model.config.get_text_config(), "layer_types", None) or [])
    return bool(kinds) and kinds <= {"full_attention", "linear_attention"}


class HybridEngine:
    """Continuous batching for models with recurrent / linear-attention layers (eager; their cache is dynamic).

    Exact without padding the recurrence: every request is prefilled alone (B = 1, exactly its solo prefill), and only
    decode steps are batched, where each row processes its own one token. Recurrent state is per row (concatenated
    along the batch); attention KV is left-padded to a common length with the mask zero there (attention is anchored
    at each query's first allowed key, so padding never moves a rounding). Rows that finish are removed by index.
    """

    def __init__(self, model, slots: int = 16, max_len: int = 4096, graph: bool = False):
        if not hybrid_supported(model):
            raise ValueError("HybridEngine is for full + linear attention models")
        self.model = model
        self.dev = next(p.device for p in model.parameters() if p.device.type == "cuda") \
            if any(p.is_cuda for p in model.parameters()) else torch.device("cpu")
        self.S, self.C = int(slots), int(max_len)
        self.waiting: collections.deque[Request] = collections.deque()
        self.lock = threading.Lock()
        self.active: list[Request] = []
        self._ids = itertools.count()
        self.cache = None
        self.am = self.tok = self.pos = None
        self.stats = {"steps": 0, "step_ms": [], "prefills": 0, "active_tokens": 0, "bucket_tokens": 0, "trims": 0}

    submit = Engine.submit
    idle = Engine.idle
    run = Engine.run

    def graph_info(self) -> dict:
        return {"eager": "dynamic cache (recurrent layers)"}

    def fail_all(self, err: str):
        with self.lock:
            reqs = list(self.active) + list(self.waiting)
            self.waiting.clear()
        self.active.clear()
        self.cache = self.am = self.tok = self.pos = None
        for r in reqs:
            r.error, r.finish = err, "error"
            r.done.set()

    def close(self):
        self.cache = self.am = self.tok = self.pos = None
        self.active.clear()

    @staticmethod
    def _pad_kv(t, d):
        if d == 0:
            return t
        z = torch.zeros(t.shape[:2] + (d,) + t.shape[3:], dtype=t.dtype, device=t.device)
        return torch.cat([z, t], 2)

    def _merge(self, cache, am, tok, pos):
        """Add a prefilled batch (cache, am [b, T], tok, pos) to the running one, aligning both at the right edge."""
        if self.cache is None:
            self.cache, self.am, self.tok, self.pos = cache, am, tok, pos
            return
        T0, T1 = self.am.shape[1], am.shape[1]
        T = max(T0, T1)
        fa, fb = _fields(self.cache), _fields(cache)
        assert len(fa) == len(fb), "cache layouts differ"
        for (ha, ka, kv), (hb, kb, _) in zip(fa, fb):
            a, b = _get(ha, ka), _get(hb, kb)
            if kv:
                a, b = self._pad_kv(a, T - T0), self._pad_kv(b, T - T1)
            _set(ha, ka, torch.cat([a, b], 0))

        def z(m, d):
            return torch.cat([torch.zeros(m.shape[0], d, dtype=m.dtype, device=m.device), m], 1) if d else m
        self.am = torch.cat([z(self.am, T - T0), z(am, T - T1)], 0)
        self.tok = torch.cat([self.tok, tok], 0)
        self.pos = torch.cat([self.pos, pos], 0)

    def _select(self, keep):
        if not keep:
            self.cache = self.am = self.tok = self.pos = None
            return
        idx = torch.tensor(keep, device=self.dev)
        for h, k, _ in _fields(self.cache):
            _set(h, k, _get(h, k).index_select(0, idx))
        self.am, self.tok, self.pos = self.am[idx], self.tok[idx], self.pos[idx]
        lead = int((self.am.sum(0) == 0).long().cumprod(0).sum())      # columns no row uses any more
        if lead:
            for h, k, kv in _fields(self.cache):
                if kv:
                    _set(h, k, _get(h, k)[:, :, lead:].contiguous())
            self.am = self.am[:, lead:].contiguous()
            self.stats["trims"] += 1

    def _admit(self):
        with self.lock:
            k = min(self.S - len(self.active), len(self.waiting))
            new = [self.waiting.popleft() for _ in range(k)]
        for r in new:                                   # each alone: its recurrence never sees padding
            ids = torch.tensor([r.prompt], device=self.dev)
            am = torch.ones_like(ids)
            out = self.model(ids, attention_mask=am, position_ids=(am.cumsum(-1) - 1), use_cache=True,
                             logits_to_keep=1)
            self.stats["prefills"] += 1
            logit = out.logits[:, -1].float()
            t = int(logit.argmax(-1)) if r.sampler is None else int(r.sampler(logit).view(-1)[0])
            if r._emit(t, logit[0]):
                r.done.set()
                continue
            self._merge(out.past_key_values, am, torch.tensor([[t]], device=self.dev),
                        torch.tensor([[len(r.prompt)]], device=self.dev))
            self.active.append(r)

    @torch.no_grad()
    def step(self):
        self._admit()
        if not self.active:
            return
        import time
        n = len(self.active)
        _sync(self.dev)
        t0 = time.perf_counter()
        self.am = torch.cat([self.am, torch.ones(n, 1, dtype=self.am.dtype, device=self.dev)], 1)
        o = self.model(self.tok, attention_mask=self.am, position_ids=self.pos, past_key_values=self.cache,
                       use_cache=True, logits_to_keep=1)
        self.cache = o.past_key_values
        lg = o.logits[:, -1].float()
        self.tok = lg.argmax(-1, keepdim=True)
        self.pos = self.pos + 1
        _sync(self.dev)
        self.stats["step_ms"].append((time.perf_counter() - t0) * 1e3)
        self.stats["steps"] += 1
        self.stats["active_tokens"] += n
        self.stats["bucket_tokens"] += n
        toks = self.tok[:, 0].tolist()
        keep = []
        for r, req in enumerate(self.active):
            t = toks[r]
            if req.sampler is not None:
                t = int(req.sampler(lg[r:r + 1]).view(-1)[0])
                self.tok[r, 0] = t
            if req._emit(t, lg[r] if req.record else None):
                req.done.set()
            else:
                keep.append(r)
        if len(keep) < n:
            self.active = [self.active[r] for r in keep]
            self._select(keep)


def make(model, slots: int = 16, max_len: int = 4096, graph: bool = True):
    """The continuous-batching engine for this model: Engine (attention-only, CUDA graphs) or HybridEngine
    (recurrent / linear-attention layers); None if neither applies."""
    if supported(model):
        return Engine(model, slots=slots, max_len=max_len, graph=graph)
    if hybrid_supported(model):
        return HybridEngine(model, slots=slots, max_len=max_len)
    return None
