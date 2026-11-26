"""Generic greedy generation for any Vahana model: prefill, then one decode step per token, eager or as one
captured CUDA graph per step (CPU expert jobs, CPU linears and CPU embedding rows run as host nodes; see runtime).

The step reads only static device buffers (token, position, cache position, 2-D attention mask), so a captured
graph replays every step unchanged; it also advances those buffers itself. Graph and eager steps run the same
kernels on the same inputs, so their outputs are bit-identical.
"""
from __future__ import annotations

import time

import torch

from . import runtime as _rt


def _sync(dev) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


@torch.no_grad()
def generate(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, max_new_tokens: int = 64,
             graph: bool = True, eos_token_id=None, return_logits: bool = False, sampler=None, on_token=None,
             prefix_cache=None):
    """input_ids [B, T] (left-padded if attention_mask is given). Returns dict with ids [B, n], step_ms, and logits
    (per step, [B, V]) when return_logits.
    sampler(logits [B, V] fp32) -> [B, 1] token ids replaces the greedy choice (it runs on the host after each step,
    so a seeded sampler is exactly reproducible). on_token(ids [B, 1]) is called for every token; returning True
    stops generation.
    prefix_cache (v1.3, prefix.PrefixCache): reuse a stored prompt prefix's keys/values and prefill only the rest
    (batch 1, attention models; the logits are bit-identical to a full prefill)."""
    from transformers import StaticCache
    dev = input_ids.device
    graph = graph and dev.type == "cuda"          # CPU-only machines decode eagerly
    B, T = input_ids.shape
    L = T + max_new_tokens
    am = torch.zeros(B, L, dtype=torch.long, device=dev)
    am[:, :T] = 1 if attention_mask is None else attention_mask
    pos0 = (am[:, :T].cumsum(-1) - 1).clamp(min=0)
    from transformers.cache_utils import StaticLayer, StaticSlidingWindowLayer
    kinds = set(getattr(model.config.get_text_config(), "layer_types", None) or [])
    other = kinds - {"full_attention", "sliding_attention", "attention"}
    if other:                                     # recurrent / linear-attention layers: their own cache, eager only
        return _generate_dynamic(model, input_ids, attention_mask, max_new_tokens, return_logits, sampler, on_token,
                                 eos_token_id=eos_token_id, reason=f"non-attention layers {sorted(other)}")
    cache = StaticCache(config=model.config, max_cache_len=L)
    # Sliding-window static layers keep their fill position in a Python int, which a captured graph would freeze;
    # a full-length static layer is position-indexed, and the window is still enforced by the attention mask.
    cache.layers = [StaticLayer(max_cache_len=L) if isinstance(x, StaticSlidingWindowLayer) else x
                    for x in cache.layers]
    t0 = time.perf_counter()
    k0 = 0
    pc = prefix_cache if (prefix_cache is not None and B == 1 and (attention_mask is None or bool(attention_mask.all()))
                          and all(hasattr(x, "lazy_initialization") for x in cache.layers)) else None
    if pc is not None:
        k0, kv = pc.lookup(input_ids[0])
        if k0:
            from . import prefix as _prefix
            _prefix.fill(cache.layers, kv, k0, dev)
    try:
        out = model(input_ids[:, k0:], attention_mask=am, position_ids=pos0[:, k0:], past_key_values=cache,
                    cache_position=torch.arange(k0, T, device=dev), use_cache=True, logits_to_keep=1)
    except ValueError as e:                       # model insists on its own cache class (hybrids): eager only
        if "cache" not in str(e).lower():
            raise
        return _generate_dynamic(model, input_ids, attention_mask, max_new_tokens, return_logits, sampler, on_token,
                                 eos_token_id=eos_token_id, reason=f"own cache class: {str(e)[:160]}")
    _sync(dev)
    prefill_s = time.perf_counter() - t0
    if pc is not None:
        pc.store(input_ids[0], cache.layers)
    logit = out.logits[:, -1].float()
    tok = logit.argmax(-1, keepdim=True) if sampler is None else sampler(logit).to(dev).view(B, 1)
    pos = pos0[:, -1:] + 1
    cpos = torch.full((1,), T, dtype=torch.long, device=dev)
    ones = torch.ones(B, 1, dtype=torch.long, device=dev)
    lbuf = torch.empty(B, model.config.get_text_config().vocab_size, dtype=torch.float32, device=dev)

    def step():
        am.scatter_(1, cpos.view(1, 1).expand(B, 1), ones)
        o = model(tok, attention_mask=am, position_ids=pos, past_key_values=cache, cache_position=cpos,
                  use_cache=True, logits_to_keep=1)
        lbuf.copy_(o.logits[:, -1].float())
        tok.copy_(lbuf.argmax(-1, keepdim=True))
        pos.add_(1)
        cpos.add_(1)

    ids, logits, lat = [tok.clone()], [logit] if return_logits else [], []
    g = None
    fallback = None
    stop = on_token is not None and bool(on_token(tok))
    for i in range(max_new_tokens - 1):
        if stop or eos_token_id is not None and B == 1 and int(ids[-1][0, 0]) in (
                eos_token_id if isinstance(eos_token_id, (list, tuple, set)) else (eos_token_id,)):
            break
        _sync(dev)
        t = time.perf_counter()
        if not graph:
            step()
        elif i == 0 and B not in getattr(model, "_vahana_graph_warm", ()):
            # first step of the first call for this model and batch size: the graph-mode path run directly on the
            # stream (host jobs execute in stream order). It compiles every kernel and allocates every pinned buffer,
            # which is not allowed during capture. v1.4: later calls capture at their first step (an eager step costs
            # ~0.45 s on Windows: thousands of small launches and a host sync per CPU job).
            _rt.GRAPH["on"] = True
            try:
                step()
            finally:
                _rt.GRAPH["on"] = False
        else:
            if g is None:
                g = torch.cuda.CUDAGraph()
                pool = torch.cuda.graph_pool_handle()   # ours, so a failed capture can still release it
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                _rt.GRAPH["on"] = True
                inner = []
                try:
                    with torch.cuda.stream(s), torch.cuda.graph(g, pool=pool, stream=s):
                        try:
                            step()                    # recorded, not executed
                        except Exception as e:        # noqa: BLE001 - keep the real cause, not capture_end's
                            import traceback
                            inner.append(traceback.format_exc())
                            raise
                except Exception as e:  # noqa: BLE001
                    # Not capturable (e.g. a module that copies host tensors to the GPU inside forward). Release the
                    # half-made capture pool and continue eagerly; the reason is reported, never silent.
                    # Older torch (private _cuda_endAllocateCurrentStreamToPool present) leaves the pool allocating
                    # after a failed capture and needs this; newer torch cleans up itself, and a second release aborts.
                    old = getattr(torch._C, "_cuda_endAllocateCurrentStreamToPool", None)
                    for f in ((old, torch._C._cuda_releasePool) if old is not None else ()):
                        try:
                            f(dev.index if dev.index is not None else torch.cuda.current_device(), pool)
                        except Exception:  # noqa: BLE001
                            pass
                    _sync(dev)
                    lines = (inner[0] if inner else repr(e)).strip().splitlines()
                    errs = [x for x in lines if "Error" in x and not x.startswith(" ")]
                    where = [x.strip() for x in lines if x.strip().startswith("File ")][-1:]
                    fallback = " | ".join(errs[:1] + where)[:300]
                    graph, g = False, None
                finally:
                    _rt.GRAPH["on"] = False
                if g is not None:
                    torch.cuda.current_stream().wait_stream(s)
                    _sync(dev)
                    t = time.perf_counter()
                    try:
                        model._vahana_graph_warm = set(getattr(model, "_vahana_graph_warm", ())) | {B}
                    except Exception:  # noqa: BLE001
                        pass
            if g is not None:
                g.replay()
            else:
                step()
        _sync(dev)
        lat.append(time.perf_counter() - t)
        if sampler is not None:
            tok.copy_(sampler(lbuf).to(dev).view(B, 1))
        ids.append(tok.clone())
        if return_logits:
            logits.append(lbuf.clone())
        if on_token is not None and on_token(tok):
            stop = True
    res = {"ids": torch.cat(ids, 1), "step_ms": [x * 1e3 for x in lat], "prefill_s": prefill_s,
           "graph_used": g is not None, "graph_fallback": fallback}
    if return_logits:
        res["logits"] = torch.stack(logits, 1)
    return res


@torch.no_grad()
def _generate_dynamic(model, input_ids, attention_mask, max_new_tokens, return_logits, sampler, on_token, reason,
                      eos_token_id=None):
    """Eager decode with the model's own cache (models that reject a static cache)."""
    dev = input_ids.device
    B, T = input_ids.shape
    am = torch.ones(B, T, dtype=torch.long, device=dev) if attention_mask is None else attention_mask.clone()
    pos = (am.cumsum(-1) - 1).clamp(min=0)
    t0 = time.perf_counter()
    out = model(input_ids, attention_mask=am, position_ids=pos, use_cache=True, logits_to_keep=1)
    _sync(dev)
    prefill_s = time.perf_counter() - t0
    pkv, logit = out.past_key_values, out.logits[:, -1].float()
    pick = (lambda lg: lg.argmax(-1, keepdim=True)) if sampler is None else (lambda lg: sampler(lg).to(dev).view(B, 1))
    tok = pick(logit)
    ids, logits, lat = [tok], [logit] if return_logits else [], []
    stop = on_token is not None and bool(on_token(tok))
    eos = set(eos_token_id) if isinstance(eos_token_id, (list, tuple, set)) else (
        set() if eos_token_id is None else {eos_token_id})
    for _ in range(max_new_tokens - 1):
        if stop or B == 1 and int(ids[-1][0, 0]) in eos:     # v0.4-c: stop at EOS as the static path does
            break
        _sync(dev)
        t = time.perf_counter()
        am = torch.cat([am, torch.ones(B, 1, dtype=torch.long, device=dev)], 1)
        pos = pos[:, -1:] + 1
        o = model(tok, attention_mask=am, position_ids=pos, past_key_values=pkv, use_cache=True, logits_to_keep=1)
        pkv, logit = o.past_key_values, o.logits[:, -1].float()
        tok = pick(logit)
        _sync(dev)
        lat.append(time.perf_counter() - t)
        ids.append(tok)
        if return_logits:
            logits.append(logit)
        if on_token is not None and on_token(tok):
            stop = True
    res = {"ids": torch.cat(ids, 1), "step_ms": [x * 1e3 for x in lat], "prefill_s": prefill_s,
           "graph_used": False, "graph_fallback": reason}
    if return_logits:
        res["logits"] = torch.stack(logits, 1)
    return res
