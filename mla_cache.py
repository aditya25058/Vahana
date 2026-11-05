"""Exact latent KV cache for MLA attention (multi-head latent attention).

MLA layers project each token to a small latent (kv_lora_rank) plus a
rotary key part (qk_rope_head_dim), then expand the latent to per-head
K_nope / V with kv_b_proj. Stock eager code caches the EXPANDED K and V:
num_heads * (q_head_dim + v_head_dim) values per token per layer. This
module caches the latent + rotated key part instead (e.g. DeepSeek-V2:
576 vs 40960 values, ~71x smaller) and re-expands with the model's own
kv_b_proj every step.

Exactness: the re-expansion uses the same weights, inputs and op, so it
is bitwise-identical to the stock cache provided the GEMM gives each row
the same bits regardless of the row count. That was measured on the
target GPU before this was built (rudra/bench_mla_recompute.py: torch.equal
for M = 1 .. 57,344 rows). Tensor layouts at the attention matmuls are
matched too: the stock DynamicCache (transformers 5.x) concatenates on
every update, so K and V reach attention contiguous; so do ours.

Selection is structural (attributes of the attention module), never by
model name, and only for the eager implementation; anything else is
refused with a reason.
"""
from __future__ import annotations

import sys
import types
from typing import Any

import torch
from torch import nn
from transformers.cache_utils import DynamicCache

_REQUIRED = ("kv_a_proj_with_mqa", "kv_b_proj", "kv_a_layernorm", "kv_lora_rank",
             "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim", "q_head_dim",
             "num_heads", "rotary_emb", "softmax_scale", "layer_idx")


class LatentKVCache(DynamicCache):
    """Per-layer latent [B, T, rank] + rotated key part [B, 1, T, rope]."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._latent: dict[int, torch.Tensor] = {}
        self._kpe: dict[int, torch.Tensor] = {}

    def append(self, layer_idx: int, latent: torch.Tensor, kpe: torch.Tensor):
        """Store this step's rows; return (all latents, all key parts, fresh)."""
        old = self._latent.get(layer_idx)
        if old is None:
            self._latent[layer_idx], self._kpe[layer_idx] = latent, kpe
            return latent, kpe, True
        self._latent[layer_idx] = torch.cat([old, latent], dim=1)
        self._kpe[layer_idx] = torch.cat([self._kpe[layer_idx], kpe], dim=2)
        return self._latent[layer_idx], self._kpe[layer_idx], False

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        t = self._latent.get(layer_idx or 0)
        return 0 if t is None else int(t.shape[1])

    def nbytes(self) -> int:
        return sum(t.nbytes for d in (self._latent, self._kpe) for t in d.values())


def is_mla_attention(module: Any) -> bool:
    return all(hasattr(module, a) for a in _REQUIRED)


def _latent_forward(self, hidden_states, attention_mask=None, position_ids=None,
                    past_key_value=None, output_attentions=False, use_cache=False,
                    **kwargs):
    if not isinstance(past_key_value, LatentKVCache):
        return self._colossus_stock_forward(
            hidden_states, attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=past_key_value, output_attentions=output_attentions,
            use_cache=use_cache, **kwargs)
    rope_fn = self._colossus_apply_rope
    nope, rope, vdim = self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim
    bsz, q_len, _ = hidden_states.size()

    # -- identical to the stock eager forward up to the cache ---------------
    if getattr(self, "q_lora_rank", None) is None:
        q = self.q_proj(hidden_states)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(q, [nope, rope], dim=-1)
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, rope], dim=-1)
    k_pe = k_pe.view(bsz, q_len, 1, rope).transpose(1, 2)
    latent = self.kv_a_layernorm(compressed_kv)
    kv_seq_len = q_len + past_key_value.get_seq_length(self.layer_idx)
    cos, sin = self.rotary_emb(latent, seq_len=kv_seq_len)  # x only sets dtype/device
    q_pe, k_pe = rope_fn(q_pe, k_pe, cos, sin, position_ids)

    # -- cache the latent, re-expand K/V for every cached token -------------
    latent_all, kpe_all, _ = past_key_value.append(self.layer_idx, latent, k_pe)
    t_all = latent_all.shape[1]
    kv = (self.kv_b_proj(latent_all)
          .view(bsz, t_all, self.num_heads, nope + vdim).transpose(1, 2))
    k_nope, value_states = torch.split(kv, [nope, vdim], dim=-1)

    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    query_states[:, :, :, :nope] = q_nope
    query_states[:, :, :, nope:] = q_pe
    key_states = kpe_all.new_empty(bsz, self.num_heads, t_all, self.q_head_dim)
    key_states[:, :, :, :nope] = k_nope
    key_states[:, :, :, nope:] = kpe_all
    # Stock DynamicCache (transformers 5.x) concatenates on every update,
    # first call included, so attention always sees contiguous V.
    value_states = value_states.contiguous()

    # -- identical to the stock eager forward from here ---------------------
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
    if attn_weights.size() != (bsz, self.num_heads, q_len, t_all):
        raise ValueError(f"attention weights {tuple(attn_weights.size())} != "
                         f"{(bsz, self.num_heads, q_len, t_all)}")
    if attention_mask is None:
        raise ValueError("latent MLA path needs the 4D attention mask (as stock eager)")
    if attention_mask.size() != (bsz, 1, q_len, t_all):
        raise ValueError(f"attention mask {tuple(attention_mask.size())} != "
                         f"{(bsz, 1, q_len, t_all)}")
    attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                         dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout,
                                         training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * vdim)
    attn_output = self.o_proj(attn_output)
    return attn_output, (attn_weights if output_attentions else None), past_key_value


def install_latent_kv(model: nn.Module) -> int:
    """Route every eager MLA attention module through the latent cache path.

    Returns the number of modules patched; raises with a reason when the
    model has no MLA attention or uses a non-eager implementation.
    """
    impl = getattr(getattr(model, "config", None), "_attn_implementation", None)
    if impl not in (None, "eager"):
        raise ValueError(f"latent KV supports eager attention only (model uses {impl!r})")
    n = 0
    for m in model.modules():
        if not is_mla_attention(m) or hasattr(m, "_colossus_stock_forward"):
            continue
        rope_fn = getattr(sys.modules.get(type(m).__module__), "apply_rotary_pos_emb", None)
        if rope_fn is None:
            raise ValueError(f"{type(m).__name__}: no apply_rotary_pos_emb in its module")
        m._colossus_apply_rope = rope_fn
        m._colossus_stock_forward = m.forward
        m.forward = types.MethodType(_latent_forward, m)
        n += 1
    if n == 0:
        raise ValueError("no MLA attention modules found (latent KV not applicable)")
    return n


def cache_nbytes(cache: Any) -> int | None:
    """Bytes held by a KV cache object (latent or stock); None if unknown."""
    if isinstance(cache, LatentKVCache):
        return cache.nbytes()
    total, seen = 0, False
    for layer in getattr(cache, "layers", None) or []:
        for name in ("keys", "values"):
            t = getattr(layer, name, None)
            if isinstance(t, torch.Tensor):
                total += t.nbytes
                seen = True
    for name in ("key_cache", "value_cache"):
        for t in getattr(cache, name, None) or []:
            if isinstance(t, torch.Tensor):
                total += t.nbytes
                seen = True
    return total if seen else None
