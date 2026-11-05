"""Huge-model serving loop for COLOSSUS tiers (chunk 2c).

`serve_huge_moe`: prefill (optionally chunked) -> lockstep greedy decode
over a wrapped (TieredMoEWrapper) model with per-step ledger.
Model-agnostic: no names, no templates, no architecture branches.
"""
from __future__ import annotations

import time
from typing import Any

import torch

from .loading import ShardHandles, materialize, split_routed
from .placement import TieredMoEWrapper, wrap_moe_layers

_compat_installed = False


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
    wrappers = prepare_model(model, profile, handles, device, capacity,
                             placement=placement,
                             dma_stream=(torch.cuda.Stream(device=device)
                                         if device.type == "cuda" else None),
                             zssr=zssr, prefetch_topk=prefetch_topk,
                             prefetch_conf=prefetch_conf, config=config,
                             prefault=prefault,
                             hot_col_frac=hot_col_frac, **_wrap_opts)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    t_prefill = time.perf_counter()
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
        else:
            extra = {"position_ids": pos_ids} if padded else {}
            out = model(input_ids=generated_ids, attention_mask=attn_mask,
                        past_key_values=past, use_cache=use_cache, **extra)
            logits = out.logits
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
    if log_routing:
        for w in wrappers:
            w.routing_log = []
    if all(finished):
        latencies = []
    for step in range(max_new_tokens - 1):
        if all(finished):
            break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
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
        "seed_loads": seed_loads,
        "routing_log_steps": routing_steps,
        "flip_audit": flip_audit,
    }
