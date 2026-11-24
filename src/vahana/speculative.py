"""v0.3-b: exact speculative decoding (greedy, batch 1).

Draft: prompt lookup. The longest suffix of the text so far (up to `ngram` tokens) is found earlier in the text, and
the tokens that followed it are proposed (up to k). No draft model, so it works for every family.
Verify: the last token plus k draft slots go through the model in one pass of fixed width Q = k + 1 (short drafts are
padded, so one CUDA graph serves every step). Drafts are accepted while they equal the model's own greedy choice;
the model's next choice is always emitted too.

Why it is exact: Vahana's kernels give a position the same bits whether it is computed alone or with other positions
in the same pass (attention anchored at the row's first key, per-row linears / experts / glue; M3's split
invariance). So every emitted token, and the logits that chose it, equal plain greedy decoding's, bit for bit.
Rejected slots write KV entries beyond the accepted text; the next pass rewrites those positions (the static cache's
running length is reset to the accepted length every step) before any query
can see them (attention is causal), so they never influence an emitted token.
"""
from __future__ import annotations

import time

import torch

from . import runtime as _rt
from .decode import _sync


def _lookup(hist: list, ngram: int, k: int) -> list:
    """Tokens that followed the most recent earlier occurrence of the longest suffix of `hist` (n = ngram..1)."""
    n_h = len(hist)
    for n in range(min(ngram, n_h - 1), 0, -1):
        suf = hist[n_h - n:]
        for s in range(n_h - n - 1, -1, -1):              # most recent earlier occurrence first
            if hist[s:s + n] == suf:
                return hist[s + n:s + n + k]
    return []


@torch.no_grad()
def generate_spec(model, input_ids: torch.Tensor, max_new_tokens: int = 64, k: int = 4, ngram: int = 3,
                  graph: bool = True, eos_token_id=None, return_logits: bool = False, draft_fn=None,
                  min_draft: int = 1):
    """Greedy decoding with prompt-lookup speculation. input_ids [1, T]. Returns ids [1, n], per-token logits
    [1, n, V] when return_logits, step timings and acceptance statistics. draft_fn(hist, k) -> list replaces prompt
    lookup (tests use an oracle draft to force long accepted runs). min_draft: drafts shorter than this are dropped and
    the step is an ordinary one-token pass (its own CUDA graph); 1 keeps every draft."""
    from transformers import StaticCache
    from transformers.cache_utils import StaticLayer, StaticSlidingWindowLayer
    dev = input_ids.device
    graph = graph and dev.type == "cuda"
    assert input_ids.shape[0] == 1, "speculative decoding: batch 1"
    T = input_ids.shape[1]
    Q = k + 1
    L = T + max_new_tokens + Q
    cache = StaticCache(config=model.config, max_cache_len=L)
    cache.layers = [StaticLayer(max_cache_len=L) if isinstance(x, StaticSlidingWindowLayer) else x
                    for x in cache.layers]
    am = torch.zeros(1, L, dtype=torch.long, device=dev)
    am[:, :T] = 1
    t0 = time.perf_counter()
    out = model(input_ids, attention_mask=am, position_ids=torch.arange(T, device=dev).view(1, T),
                past_key_values=cache, cache_position=torch.arange(T, device=dev), use_cache=True, logits_to_keep=1)
    _sync(dev)
    prefill_s = time.perf_counter() - t0
    logit = out.logits[:, -1].float()
    first = int(logit.argmax(-1))
    V = logit.shape[-1]
    lanes = {}

    class Lane:
        """Fixed-width verify pass (width W) with its own buffers and CUDA graph."""

        def __init__(self, W):
            self.W, self.uses, self.g, self.fallback = W, 0, None, None
            self.tok = torch.zeros(1, W, dtype=torch.long, device=dev)
            self.pos = torch.zeros(1, W, dtype=torch.long, device=dev)
            self.cpos = torch.zeros(W, dtype=torch.long, device=dev)
            self.ones = torch.ones(1, W, dtype=torch.long, device=dev)
            self.lbuf = torch.empty(W, V, dtype=torch.float32, device=dev)
            self.gbuf = torch.empty(W, dtype=torch.long, device=dev)

        def step(self):
            am.scatter_(1, self.cpos.view(1, self.W), self.ones)
            o = model(self.tok, attention_mask=am, position_ids=self.pos, past_key_values=cache,
                      cache_position=self.cpos, use_cache=True, logits_to_keep=self.W)
            self.lbuf.copy_(o.logits[0].float())
            self.gbuf.copy_(self.lbuf.argmax(-1))

        def run(self):
            if not graph or self.fallback:
                self.step()
            elif self.uses == 0:
                _rt.GRAPH["on"] = True             # warm the graph-mode path outside capture (kernels, buffers)
                try:
                    self.step()
                finally:
                    _rt.GRAPH["on"] = False
            else:
                if self.g is None:
                    self.g = torch.cuda.CUDAGraph()
                    pool = torch.cuda.graph_pool_handle()  # ours, so a failed capture can still release it
                    st = torch.cuda.Stream()
                    st.wait_stream(torch.cuda.current_stream())
                    _rt.GRAPH["on"] = True
                    try:
                        with torch.cuda.stream(st), torch.cuda.graph(self.g, pool=pool, stream=st):
                            self.step()
                    except Exception as e:  # noqa: BLE001 - not capturable: continue eagerly (as decode.generate)
                        self.fallback, self.g = repr(e)[:200], None
                        old = getattr(torch._C, "_cuda_endAllocateCurrentStreamToPool", None)
                        for f in ((old, torch._C._cuda_releasePool) if old is not None else ()):
                            try:
                                f(dev.index if dev.index is not None else torch.cuda.current_device(), pool)
                            except Exception:  # noqa: BLE001
                                pass
                    finally:
                        _rt.GRAPH["on"] = False
                    torch.cuda.current_stream().wait_stream(st)
                    _sync(dev)
                    if self.g is None:
                        self.step()
                if self.g is not None:
                    self.g.replay()
            self.uses += 1

    hist = input_ids[0].tolist() + [first]
    emitted, logits = [first], [logit[0]] if return_logits else []
    p = T                                  # position of the last emitted token (not yet in the cache)
    lat, accepted, steps, wide = [], 0, 0, 0
    eos = set(eos_token_id if isinstance(eos_token_id, (list, tuple, set)) else
              ([eos_token_id] if eos_token_id is not None else []))
    while len(emitted) < max_new_tokens and emitted[-1] not in eos:
        draft = (draft_fn(hist, k) if draft_fn else _lookup(hist, ngram, k))[:k]
        if len(draft) < min_draft:                 # nothing worth verifying: an ordinary one-token step
            draft = []
        nd = len(draft)
        W = Q if nd else 1
        lane = lanes.get(W) or lanes.setdefault(W, Lane(W))
        wide += W > 1
        slots = [emitted[-1]] + draft + [emitted[-1]] * (W - 1 - nd)      # padding slots are never accepted
        lane.tok.copy_(torch.tensor([slots], device=dev))
        r = torch.arange(p, p + W, device=dev)
        lane.pos.copy_(r.view(1, W))
        lane.cpos.copy_(r)
        # transformers' static cache layers append at their own running length (they ignore cache_position); after a
        # pass with rejected slots that length is ahead of the accepted text, so it is reset to `p` every step
        # (outside the CUDA graph, which reads the same tensor).
        for lay in cache.layers:
            cl = getattr(lay, "cumulative_length", None)
            if isinstance(cl, torch.Tensor):
                cl.fill_(p)
        _sync(dev)
        t = time.perf_counter()
        lane.run()
        _sync(dev)
        lat.append(time.perf_counter() - t)
        steps += 1
        choice = lane.gbuf.tolist()
        new = []
        for i in range(W):
            new.append(choice[i])
            if return_logits:
                logits.append(lane.lbuf[i].clone())
            if i < nd and draft[i] == choice[i]:
                continue
            break
        accepted += len(new) - 1
        for tkn in new:
            if len(emitted) >= max_new_tokens or emitted[-1] in eos:
                break
            emitted.append(tkn)
            hist.append(tkn)
        if return_logits:
            logits = logits[:len(emitted)]
        p += len(new)
    g = any(ln.g is not None for ln in lanes.values())
    fallback = next((ln.fallback for ln in lanes.values() if ln.fallback), None)
    res = {"ids": torch.tensor([emitted[:max_new_tokens]], device=dev), "step_ms": [x * 1e3 for x in lat],
           "prefill_s": prefill_s, "steps": steps, "accepted_drafts": accepted,
           "tokens_per_step": (len(emitted) - 1) / max(1, steps), "graph_used": g,
           "graph_fallback": fallback, "k": k, "wide_steps": wide}
    if return_logits:
        res["logits"] = torch.stack(logits[:max_new_tokens]).unsqueeze(0)
    return res
