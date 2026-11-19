"""Huge-model serving loop for COLOSSUS tiers (chunk 2c).

`serve_huge_moe`: prefill (optionally chunked) -> lockstep greedy decode
over a wrapped (TieredMoEWrapper) model with per-step ledger.
Model-agnostic: no names, no templates, no architecture branches.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import torch

from .loading import ShardHandles, materialize, split_routed
from .placement import TieredMoEWrapper, wrap_moe_layers

_compat_installed = False
_KEEP = [1]   # positions the last-logits hook keeps (k+1 while verifying drafts)


def _load_counters(wrappers) -> dict:
    """Cumulative expert-load counters summed over all MoE wrappers."""
    return {"misses": sum(getattr(w, "misses", 0) for w in wrappers),
            "dma_bytes": sum(getattr(w, "dma_bytes", 0) for w in wrappers),
            "copy_s": sum(getattr(w, "copy_s", 0.0) for w in wrappers),
            "minflt": sum(getattr(w, "load_minflt", 0) for w in wrappers)}


def _install_nvtx(model, wrappers) -> None:
    """NVTX push/pop around each attention module and each MoE wrapper (diagnostics)."""
    def pre(name):
        def _hook(_m, _a):
            torch.cuda.nvtx.range_push(name)   # must return None (a value would replace args)
        return _hook

    def post(_m, _a, _o):
        torch.cuda.nvtx.range_pop()
    for name, m in model.named_modules():
        if name.endswith(".self_attn"):
            m.register_forward_pre_hook(pre("attn"))
            m.register_forward_hook(post)
    for w in wrappers:
        w.register_forward_pre_hook(pre("moe"))
        w.register_forward_hook(post)


def _ans_info():
    try:
        from .ansplane import get_plane
        p = get_plane()
        return p.info() if p is not None else None
    except Exception:
        return None


def install_cache_compat(model_type: str | None = None) -> None:
    """Polyfills for modeling code written against older transformers cache APIs.

    get_usable_length is a pure method addition (harmless if unused).
    The causal-mask patch (None -> zeros) is gated to DeepSeek-V2-family
    modeling: other families (Param2, Mixtral, Qwen) rely on None meaning
    "no mask" and break when it becomes zeros. Verified by bisection:
    global patching corrupted Param2 prefill while DeepSeek needs it.
    """
    global _compat_installed
    import torch as _torch
    from transformers.cache_utils import DynamicCache as _DC
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter as _AMC

    def _get_usable_length(self, *args, **kwargs):
        layer_idx = 0
        if len(args) > 1 and isinstance(args[1], int):
            layer_idx = args[1]
        elif "layer_idx" in kwargs:
            layer_idx = kwargs["layer_idx"]
        return self.get_seq_length(layer_idx)

    _DC.get_usable_length = _get_usable_length
    want_mask_patch = bool(model_type and "deepseek" in model_type)
    has_mask_patch = bool(getattr(_AMC.to_causal_4d, "_colossus_patched", False))
    if want_mask_patch and not has_mask_patch:
        _orig = _AMC.to_causal_4d
        _AMC._colossus_orig = _orig

        def _patched_to_causal_4d(self, batch_size, query_length, key_value_length,
                                  dtype, device="cpu"):
            mask = _orig(self, batch_size, query_length, key_value_length, dtype, device)
            if mask is None:
                mask = _torch.zeros((batch_size, 1, query_length, key_value_length),
                                    dtype=dtype, device=device)
            return mask

        _patched_to_causal_4d._colossus_patched = True
        _AMC.to_causal_4d = _patched_to_causal_4d
    elif not want_mask_patch and has_mask_patch:
        _AMC.to_causal_4d = _AMC._colossus_orig
    _compat_installed = True


def prepare_model(model: torch.nn.Module, profile: Any, handles: ShardHandles,
                  device: torch.device, capacity: int,
                  dtype: torch.dtype = torch.bfloat16,
                  placement: str = "slots", dma_stream: Any = None,
                   zssr: bool = False, prefetch_topk: int = 8,
                   prefetch_conf: float = 0.0, config=None,
                   prefault: bool = False,
                   hot_col_frac: float | None = None,
                   **opts) -> list[TieredMoEWrapper]:
    """Materialize resident weights + wrap MoE layers. Returns wrappers."""
    if prefault and hasattr(handles, "weight_map"):
        from .loading import prefault_shards
        model_dir = getattr(handles, "model_dir", "") or ""
        if model_dir:
            prefault_shards(handles.weight_map, model_dir, verbose=True)
    resident, _ = split_routed(handles.weight_map)
    materialize(model, resident, handles, device, dtype)
    model.eval()
    # Tied LM head: indexes omit tied weights (no lm_head entry when tied to
    # embeddings). from_pretrained shares storage; replicate the tie when the
    # head is still meta after materialize. Silent meta heads produce
    # constant garbage logits (found via Param2: prefill argmax pinned at 0).
    try:
        _head = getattr(model, "lm_head", None)
        _hw = getattr(_head, "weight", None)
        if _head is not None and getattr(_hw, "is_meta", False):
            _emb = model.get_input_embeddings()
            _ew = getattr(_emb, "weight", None) if _emb is not None else None
            if _ew is not None and not getattr(_ew, "is_meta", False):
                _head.weight = _ew
    except Exception:
        pass
    # RoPE-style computed buffers are meta after empty init; best-effort init
    # via duck-typed hooks (any attention exposing _init_rope).
    for m in model.modules():
        init_rope = getattr(m, "_init_rope", None)
        if callable(init_rope):
            try:
                init_rope()
            except Exception:
                pass
        remb = getattr(m, "rotary_emb", None)
        if remb is not None:
            try:
                remb.to(device)
            except Exception:
                pass
    # Generic fallback: REMOVED (was corrupting real rope tables on Param2:
    # recompute fired on non-meta buffers via attribute confusion and rewrote
    # working rotary state). Empty-init models must carry valid rope state
    # through materialize/_init_rope paths only; silent recompute is banned.
    if config is not None:
        n_recomputed = 0
        for m in model.modules():
            fn = getattr(m, "rope_init_fn", None)
            inv = getattr(m, "inv_freq", None)
            if callable(fn) and inv is not None and getattr(inv, "is_meta", False):
                try:
                    new_inv, _ = fn(config, device)
                    m.register_buffer("inv_freq", new_inv.to(device),
                                      persistent=False)
                    if hasattr(m, "original_inv_freq"):
                        m.original_inv_freq = m.inv_freq
                    n_recomputed += 1
                except Exception:
                    pass
        if n_recomputed:
            import logging as _logging
            _logging.getLogger("bhaskera.serve").warning(
                "recomputed %d meta rope tables (verify output!), model=%s",
                n_recomputed, getattr(config, "model_type", "?"))
    wrappers = wrap_moe_layers(
        model, profile, handles, device, capacity, dma_stream,
        zssr_prefetch=zssr, prefetch_topk=prefetch_topk,
        prefetch_conf=prefetch_conf,
        cpu_exec=(placement == "cpu"), **opts)
    if getattr(profile, "grouped_moe", False):
        # Representation 2 (grouped weights): the grouped adapter replaces
        # experts containers in place; routing stays native upstream.
        from .grouped import wrap_grouped_layers
        _gopts: dict = {}
        if hot_col_frac is not None:
            from .groupcol import GroupedColumnWrapper
            _gopts = {"wrapper_cls": GroupedColumnWrapper,
                      "hot_frac": float(hot_col_frac)}
        wrappers = list(wrappers) + wrap_grouped_layers(
            model, profile, handles, device, capacity,
            dma_stream=dma_stream, **_gopts)
    return wrappers


def serve_huge_moe(model, tokenizer, profile, handles: ShardHandles,
                   device: torch.device, prompts: list[str],
                   max_new_tokens: int = 16, capacity: int = 12,
                   placement: str = "slots", prefill_chunk: int = 0,
                   use_cache: bool = True, zssr: bool = False,
                   prefetch_topk: int = 8, prefetch_conf: float = 0.0,
                   log_routing: str | None = None,
                   teacher_tokens: list[int] | None = None,
                   audit_logits: str | None = None,
                   config=None,
                   prefault: bool = False,
                   dump_ids: str | None = None,
                   hot_col_frac: float | None = None,
                   hot_tier_file: str | None = None,
                   cold_cache_cap: int = 0,
                   grouped_gemm: bool = False,
                   matrix_tiers: bool = False,
                   prefill_seed: str = "off",
                   kv_layout: str = "full",
                   diag_row_split: bool = False,
                   sync_loads: bool = False,
                   ) -> dict[str, Any]:
    """Greedy lockstep serve with ledger. Returns results dict."""
    from transformers.cache_utils import DynamicCache

    install_cache_compat(getattr(profile, "model_type", None))
    _wrap_opts: dict = {}
    if hot_col_frac is not None or hot_tier_file is not None:
        from .colres import ColumnTieredMoEWrapper
        _wrap_opts = {"wrapper_cls": ColumnTieredMoEWrapper}
        if cold_cache_cap > 0:
            _wrap_opts["cold_cache_cap"] = int(cold_cache_cap)
        if grouped_gemm:
            _wrap_opts["grouped_gemm"] = True
        if hot_tier_file is not None:
            from .colprofile import load_tier_file, load_tier_pools
            _wrap_opts["hot_tiers"] = load_tier_file(hot_tier_file)
            _wrap_opts["tier_pools"] = load_tier_pools(hot_tier_file)
        elif hot_col_frac is not None:
            _wrap_opts["hot_frac"] = float(hot_col_frac)
    elif matrix_tiers:
        from .matrix import MatrixTieredMoEWrapper
        _wrap_opts = {"wrapper_cls": MatrixTieredMoEWrapper}
    ans_store = os.environ.get("COLOSSUS_ANS_STORE")
    if ans_store and device.type == "cuda":
        # Lossless compressed expert plane (ansplane.py): pinned in place with the
        # same RAM margin as --host-register-margin-gb (default 64 GB).
        from .ansplane import init_plane
        init_plane(ans_store, device,
                   margin_gb=float(os.environ.get("COLOSSUS_ANS_MARGIN_GB", "64")))
    wrappers = prepare_model(model, profile, handles, device, capacity,
                             placement=placement,
                             dma_stream=(torch.cuda.Stream(device=device)
                                         if device.type == "cuda" else None),
                             zssr=zssr, prefetch_topk=prefetch_topk,
                             prefetch_conf=prefetch_conf, config=config,
                             prefault=prefault,
                             hot_col_frac=hot_col_frac, **_wrap_opts)
    if ans_store and device.type == "cuda" and wrappers:
        from .ansplane import get_plane
        _w = wrappers[0]
        _ks = _w._ans_keys(0)
        if get_plane().has(_ks):
            _raw = [handles.get_tensor(k) for k in _ks]
            if not get_plane().self_check(_ks, _raw, _w.dma_stream):
                raise RuntimeError("ANS plane self-check failed: decoded expert != raw bytes")
            print(f"ANS plane: self-check exact; {get_plane().info()['registered_gb']:.1f} GB pinned")
    mla_patched = 0
    if kv_layout == "latent":
        # Exact latent KV for MLA attention (see mla_cache.py); refuses
        # with a reason on non-MLA or non-eager models.
        from .mla_cache import install_latent_kv
        mla_patched = install_latent_kv(model)
    elif kv_layout != "full":
        raise ValueError(f"kv_layout {kv_layout!r} not in ('full', 'latent')")
    from . import invariant as _inv
    if _inv.enabled():
        if device.type != "cuda" or kv_layout != "latent":
            raise ValueError("COLOSSUS_INVARIANT=1 needs CUDA and --kv-layout latent")
        print("invariant mode:", _inv.install(model), flush=True)
    if sync_loads:
        for w in wrappers:
            w.sync_loads = True  # diagnostic: honest per-load copy timing
    detector = None
    if diag_row_split:
        # Probe only (identical-prompt batches): first module where rows split.
        from .rowsplit import RowSplitDetector
        detector = RowSplitDetector()
        detector.attach(model, profile)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    from . import refill as _refill
    if _refill.enabled():
        # Continuous refill + (optional) exact speculation; see refill.py.
        if os.environ.get("COLOSSUS_INVARIANT") != "1" or kv_layout != "latent" or device.type != "cuda":
            raise ValueError("COLOSSUS_REFILL needs COLOSSUS_INVARIANT=1, --kv-layout latent, CUDA")
        torch.cuda.reset_peak_memory_stats(device)
        _outs, _lat, _wall, _st = _refill.refill_generate(
            model, tokenizer, prompts, int(os.environ.get("COLOSSUS_REFILL_SLOTS", "512")),
            max_new_tokens, int(os.environ.get("COLOSSUS_SPEC", "0") or 0), device,
            tokenizer.eos_token_id, refill_min=int(os.environ.get("COLOSSUS_REFILL_MIN", "32")))
        _tot = sum(len(o) for o in _outs)
        if dump_ids:
            with open(dump_ids, "w") as f:
                json.dump({"continuation_ids": _outs, "new_counts": [len(o) for o in _outs],
                           "batch": len(prompts), "prompt_len": None}, f)
        return {"texts": tokenizer.batch_decode(_outs, skip_special_tokens=True),
                "batch_size": len(prompts), "max_new_tokens": max_new_tokens,
                "generated_tokens_total": _tot, "decode_tokens": _tot, "prefill_s": 0.0,
                "decode_s": _wall, "step_ms": [s * 1000.0 for s in _lat],
                "batch_decode_tps": _tot / _wall, "e2e_tps": _tot / _wall,
                "legacy_batch_decode_tps": _tot / _wall,
                "metric_definitions": {"e2e_tps": "all generated tokens / wall time of the whole "
                                                  "refill run (prefills included)"},
                "refill_stats": _st, "placement": placement, "capacity": capacity,
                "total_hits": sum(w.hits for w in wrappers),
                "total_misses": sum(w.misses for w in wrappers),
                "total_dma_mb": sum(w.dma_bytes for w in wrappers) / (1024 ** 2),
                "peak_vram_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                "kv_layout": kv_layout}
    # Lockstep decode reads logits[:, -1]: that is a real token for every
    # row only with left padding.
    if len(prompts) > 1 and getattr(tokenizer, "padding_side", "left") != "left":
        tokenizer.padding_side = "left"
    enc = tokenizer(prompts, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"]
    attn_mask = enc.get("attention_mask")
    B, prompt_len = input_ids.shape
    input_ids = input_ids.to(device)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)
    generated_ids = input_ids.clone()
    if attn_mask is None:
        attn_mask = torch.ones_like(input_ids)
    # Mixed-length batches are padded: pad positions must stay masked and
    # positions must count real tokens only (as generate() does), or rows
    # diverge from their B=1 runs. Unpadded batches keep the original call
    # path unchanged, so canonical counters stay reproducible.
    padded = bool((attn_mask == 0).any())
    pos_ids = (attn_mask.long().cumsum(-1) - 1).clamp(min=0) if padded else None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    # Every caller reads logits[:, -1] only. COLOSSUS_LAST_LOGITS=1 feeds the LM head the last
    # position alone, so prefill does not materialise B x S x vocab fp32 logits (40 GB at
    # B=4096). Off by default.
    if os.environ.get("COLOSSUS_LAST_LOGITS") == "1" and getattr(model, "lm_head", None) is not None:
        def _last_only(_mod, args):
            h = args[0]
            keep = _KEEP[0]                      # 1, or k+1 during speculative verify steps
            if h.dim() == 3 and h.shape[1] > keep:
                return (h[:, -keep:, :],) + tuple(args[1:])
            return None
        model.lm_head.register_forward_pre_hook(_last_only)
    _spec_k = int(os.environ.get("COLOSSUS_SPEC", "0") or 0)
    _spec_stats = None
    if _spec_k > 0 and not (os.environ.get("COLOSSUS_INVARIANT") == "1" and kv_layout == "latent"
                            and use_cache and teacher_tokens is None):
        raise ValueError("COLOSSUS_SPEC needs COLOSSUS_INVARIANT=1, --kv-layout latent, cache on")
    t_prefill = time.perf_counter()
    if use_cache and kv_layout == "latent":
        from .mla_cache import LatentKVCache
        if _spec_k > 0:
            from .spec import RaggedLatentCache
            past = RaggedLatentCache((attn_mask == 0).sum(dim=-1),
                                     cap=input_ids.shape[1] + max_new_tokens + _spec_k + 1)
        else:
            past = LatentKVCache()
    else:
        past = DynamicCache() if use_cache else None
    with torch.no_grad():
        if prefill_chunk > 0 and generated_ids.shape[1] > prefill_chunk:
            logits, seqlen = None, generated_ids.shape[1]
            for s in range(0, seqlen, prefill_chunk):
                chunk = generated_ids[:, s:s + prefill_chunk]
                cmask = attn_mask[:, :s + chunk.shape[1]]
                extra = ({"position_ids": pos_ids[:, s:s + chunk.shape[1]]}
                         if padded else {})
                out = model(input_ids=chunk, attention_mask=cmask,
                            past_key_values=past, use_cache=use_cache, **extra)
                logits = out.logits
                past = getattr(out, "past_key_values", None)
                if _spec_k > 0:
                    past.commit([chunk.shape[1]] * B)
        else:
            extra = {"position_ids": pos_ids} if padded else {}
            out = model(input_ids=generated_ids, attention_mask=attn_mask,
                        past_key_values=past, use_cache=use_cache, **extra)
            logits = out.logits
            if _spec_k > 0:
                getattr(out, "past_key_values", past).commit([generated_ids.shape[1]] * B)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past = getattr(out, "past_key_values", None)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_prefill = time.perf_counter() - t_prefill
    seed_loads = 0
    if prefill_seed and prefill_seed != "off":
        # Self-seeding: pre-position decode slots from each wrapper's
        # prefill self-profile (data movement only — values untouched).
        for w in wrappers:
            seed = getattr(w, "seed_from_prefill", None)
            if callable(seed):
                try:
                    seed_loads += seed(prefill_seed)
                except Exception:
                    pass
    if next_token.device != device:
        next_token = next_token.to(device)
    # Teacher-forced flip audit (B=1 only): feed reference tokens, record own
    # argmax + fp32 margin per position (isolates per-position noise).
    audit: list[dict] = []
    if teacher_tokens is not None:
        assert B == 1, "teacher protocol requires batch size 1"
        assert len(teacher_tokens) >= max_new_tokens, "teacher too short"
        tv, ti = logits[0, -1, :].float().topk(2)
        own = int(ti[0])
        audit.append({"pos": 1, "own": own, "ref": teacher_tokens[0],
                      "match": own == teacher_tokens[0],
                      "margin": float(tv[0] - tv[1]),
                      "own_bf16": int(logits[0, -1, :].argmax())})
        next_token = torch.tensor([[teacher_tokens[0]]], device=device)

    finished = [False] * B
    eos_id = tokenizer.eos_token_id
    new_counts = [1] * B
    # Prefill token is generated token #1: append BEFORE the loop
    # (previously dropped — continuation missed position 1 and every
    # teacher comparison shifted by one).
    generated_ids = torch.cat([generated_ids, next_token], dim=1)
    for bi in range(B):
        if int(next_token[bi, 0].item()) == eos_id:
            finished[bi] = True
    latencies: list[float] = []
    # Cumulative load counters after prefill and after every decode step;
    # diffs give per-step misses / bytes / copy time / page faults.
    step_loads = [_load_counters(wrappers)]
    if log_routing:
        for w in wrappers:
            w.routing_log = []
    if all(finished):
        latencies = []
    # Opt-in profiling (COLOSSUS_PROFILE_STEPS="a,b"): cudaProfilerStart before decode step a,
    # Stop after step b (for nsys --capture-range=cudaProfilerApi), with NVTX ranges on every
    # attention and MoE module. Diagnostics only; no effect on values.
    _prof = os.environ.get("COLOSSUS_PROFILE_STEPS")
    _prof_a, _prof_b = (int(x) for x in _prof.split(",")) if _prof else (-1, -1)
    if _prof and device.type == "cuda":
        _install_nvtx(model, wrappers)
    _trec = None
    if os.environ.get("COLOSSUS_TIMING") == "1" and device.type == "cuda":
        from . import timing as _timing
        _trec = _timing.install(model, wrappers)
    timing_steps: list[dict] = []
    if _spec_k > 0:
        from .spec import spec_decode
        real = [input_ids[b][attn_mask[b] == 1].tolist() for b in range(B)]
        _new, latencies, _spec_stats = spec_decode(
            model, past, real, next_token, max_new_tokens, _spec_k, eos_id, device,
            lambda n: _KEEP.__setitem__(0, n))
        new_counts = [len(n) for n in _new]
        width = max(new_counts)
        pad_tok = eos_id if eos_id is not None else 0
        gen = torch.tensor([n + [pad_tok] * (width - len(n)) for n in _new], device=device)
        generated_ids = torch.cat([input_ids, gen], dim=1)
        finished = [True] * B
    for step in range(0 if _spec_k > 0 else max_new_tokens - 1):
        if all(finished):
            break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if step == _prof_a and device.type == "cuda":
            torch.cuda.profiler.start()
        if _prof and device.type == "cuda":
            torch.cuda.nvtx.range_push(f"step{step}")
        if detector is not None:
            detector.step = step
        t0 = time.perf_counter()
        with torch.no_grad():
            extra = {}
            if padded:
                attn_mask = torch.cat([attn_mask, attn_mask.new_ones((B, 1))], dim=1)
                pos_ids = attn_mask.long().cumsum(-1) - 1
                extra = {"attention_mask": attn_mask}
            if use_cache and past is not None:
                if padded:
                    extra["position_ids"] = pos_ids[:, -1:]
                out = model(input_ids=next_token, past_key_values=past, use_cache=True,
                            **extra)
            else:
                if padded:
                    extra["position_ids"] = pos_ids.clamp(min=0)
                out = model(input_ids=torch.cat([generated_ids, next_token], dim=1),
                            use_cache=False, **extra)
            logits = out.logits
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if use_cache:
                past = getattr(out, "past_key_values", None)
            if teacher_tokens is not None:
                pos = step + 2
                if pos - 1 < len(teacher_tokens):
                    tv, ti = logits[0, -1, :].float().topk(2)
                    own = int(ti[0])
                    ref = teacher_tokens[pos - 1]
                    audit.append({"pos": pos, "own": own, "ref": ref,
                                  "match": own == ref,
                                  "margin": float(tv[0] - tv[1]),
                                  "own_bf16": int(logits[0, -1, :].argmax())})
                    next_token = torch.tensor([[ref]], device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append(time.perf_counter() - t0)
        if _trec is not None:
            timing_steps.append(_trec.flush(step, latencies[-1]))
            print("TIMING", json.dumps(timing_steps[-1]), flush=True)
        if _prof and device.type == "cuda":
            torch.cuda.nvtx.range_pop()
            if step == _prof_b:
                torch.cuda.profiler.stop()
        step_loads.append(_load_counters(wrappers))
        if next_token.device != device:
            next_token = next_token.to(device)
        for bi in range(B):
            if finished[bi]:
                continue
            new_counts[bi] += 1
            if int(next_token[bi, 0].item()) == eos_id:
                finished[bi] = True
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        if all(finished):
            break

    texts = tokenizer.batch_decode(generated_ids[:, prompt_len:],
                                   skip_special_tokens=True)
    from .mla_cache import cache_nbytes
    kv_bytes = cache_nbytes(past) if past is not None else None
    total_hits = sum(w.hits for w in wrappers)
    total_misses = sum(w.misses for w in wrappers)
    total_dma_mb = sum(w.dma_bytes for w in wrappers) / (1024 ** 2)
    hot_dma_mb = (sum(getattr(w, "hot_dma_bytes", 0) for w in wrappers)
                  / (1024 ** 2) or None)
    cold_dma_mb = (sum(getattr(w, "cold_dma_bytes", 0) for w in wrappers)
                   / (1024 ** 2) if hot_dma_mb is not None else None)
    total_decode = sum(latencies)
    agg_tokens = sum(new_counts)
    # Token #1 of every row comes from prefill, so decode throughput counts
    # only tokens produced by decode steps (the legacy figure counted the
    # prefill token against decode time, overstating it by ~B/steps).
    decode_tokens = sum(max(0, c - 1) for c in new_counts)
    decode_tps = decode_tokens / total_decode if total_decode > 0 else 0.0
    e2e_s = t_prefill + total_decode
    for w in wrappers:
        flush = getattr(w, "flush_timing", None)
        if callable(flush):
            flush()  # every step ended in a device sync: all events done
    peak = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            if device.type == "cuda" else 0.0)
    routing_steps = 0
    if log_routing:
        import json as _json
        entries = [{"layer": lyr, "experts": exps}
                   for w in wrappers for (lyr, exps) in (w.routing_log or [])]
        with open(log_routing, "w") as f:
            _json.dump(entries, f)
        routing_steps = (len(wrappers[0].routing_log or [])
                         if wrappers else 0)
    flip_audit = None
    if audit_logits:
        import json as _json2
        with open(audit_logits, "w") as f:
            _json2.dump({"audit": audit}, f)
        flip_audit = audit
    if dump_ids:
        import json as _json3
        # One row per sequence (left padding keeps [prompt_len:] aligned).
        # Rows that hit EOS keep lockstep tokens after it; new_counts says
        # how many are real (the batched exactness audit truncates there).
        with open(dump_ids, "w") as f:
            _json3.dump({"continuation_ids": generated_ids[:, prompt_len:].tolist(),
                         "new_counts": list(new_counts),
                         "batch": B, "prompt_len": prompt_len}, f)
    return {
        "texts": texts,
        "batch_size": B,
        "prompt_tokens": prompt_len,
        "max_new_tokens": max_new_tokens,
        "generated_tokens": max(new_counts) if new_counts else 0,
        "generated_tokens_per_row": list(new_counts),
        "generated_tokens_total": agg_tokens,
        "decode_tokens": decode_tokens,
        "prefill_s": t_prefill,
        "decode_s": total_decode,
        "step_ms": [s * 1000.0 for s in latencies],
        "decode_avg_ms": (total_decode / max(1, len(latencies))) * 1000.0,
        "batch_decode_tps": decode_tps,
        "e2e_tps": agg_tokens / e2e_s if e2e_s > 0 else 0.0,
        "legacy_batch_decode_tps": agg_tokens / total_decode if total_decode > 0 else 0.0,
        "metric_definitions": {
            "batch_decode_tps": "decode_tokens / decode_s (prefill token excluded)",
            "e2e_tps": "generated_tokens_total / (prefill_s + decode_s)",
            "legacy_batch_decode_tps": "generated_tokens_total / decode_s "
                                        "(definition used in COLOSSUS_REPORT up to section 36)",
            "decode_s": "sum of per-step wall times, each bracketed by device syncs",
            "stall_ms": "GPU-event time cold copies finished after their expert's "
                        "hot GEMMs (column slots only; 0 = fully hidden)",
        },
        "total_hits": total_hits,
        "total_misses": total_misses,
        "total_dma_mb": total_dma_mb,
        "peak_vram_gb": peak,
        "row_split_first": detector.first() if detector is not None else None,
        "row_split_events": detector.events if detector is not None else None,
        "kv_layout": kv_layout,
        "spec_stats": _spec_stats,
        "mla_layers_patched": mla_patched,
        "kv_cache_mb": None if kv_bytes is None else kv_bytes / (1024 ** 2),
        "placement": placement,
        "capacity": capacity,
        "hot_frac": hot_col_frac,
        "matrix_tiers": matrix_tiers,
        "role_misses": {r: sum(getattr(w, "role_misses", {}).get(r, 0)
                               for w in wrappers) or None for r in
                        ("gate", "up", "down")} if matrix_tiers else None,
        "hot_dma_mb": hot_dma_mb,
        "cold_dma_mb": cold_dma_mb,
        "cold_hits": sum(getattr(w, "cold_hits", 0) for w in wrappers),
        "stall_ms": sum(getattr(w, "stall_ms", 0.0) for w in wrappers),
        "stall_n": sum(getattr(w, "stall_n", 0) for w in wrappers),
        # Host time inside MoE wrappers over prefill + decode (expert slots).
        "moe_fetch_s": sum(getattr(w, "fetch_s", 0.0) for w in wrappers),
        "moe_copy_s": sum(getattr(w, "copy_s", 0.0) for w in wrappers),
        "moe_forward_s": sum(getattr(w, "forward_s", 0.0) for w in wrappers),
        "step_loads": [{k: b[k] - a[k] for k in a}
                       for a, b in zip(step_loads, step_loads[1:])],
        "load_minflt": sum(getattr(w, "load_minflt", 0) for w in wrappers),
        "load_majflt": sum(getattr(w, "load_majflt", 0) for w in wrappers),
        "per_layer_loads": [{"layer": w.layer_idx, "misses": w.misses,
                             "dma_bytes": w.dma_bytes,
                             "copy_s": getattr(w, "copy_s", 0.0),
                             "minflt": getattr(w, "load_minflt", 0)} for w in wrappers],
        "sync_loads": sync_loads,
        "host_register": (handles.stats() if hasattr(handles, "reg_bytes") else None),
        "ans_plane": _ans_info(),
        "xlayer_loads": sum(getattr(w, "xlayer_loads", 0) for w in wrappers),
        "xlayer_used": sum(getattr(w, "xlayer_used", 0) for w in wrappers),
        "seed_loads": seed_loads,
        "routing_log_steps": routing_steps,
        "flip_audit": flip_audit,
        "hybrid_stats": (dict(__import__("bhaskera.inference.colossus.hybrid", fromlist=["STATS"]).STATS) if os.environ.get("COLOSSUS_HYBRID", "") != "" else None),
    }
