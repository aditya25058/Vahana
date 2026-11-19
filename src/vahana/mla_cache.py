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

    if getattr(past_key_value, "append_ragged", None) is not None:
        # Ragged cache (speculative decoding, invariant mode only): rows write at their own
        # offsets; attention reads [first_b, length_b + q]. No mask, no shifting.
        if not _inv_on():
            raise ValueError("ragged latent cache requires COLOSSUS_INVARIANT=1")
        from .invariant import inv_bmm, inv_mla_attention
        lat_buf, kpe_buf, first, base, slot_rows = past_key_value.append_ragged(self.layer_idx, latent, k_pe)
        wuk, wuvT = _absorb_weights(self)
        H, r = self.num_heads, self.kv_lora_rank
        qn = q_nope.permute(1, 0, 2, 3).reshape(H, bsz * q_len, nope).contiguous()
        q_lat = inv_bmm(qn, wuk).view(H, bsz, q_len, r).permute(1, 2, 0, 3)
        ctx = inv_mla_attention(q_lat, q_pe.permute(0, 2, 1, 3), lat_buf, kpe_buf, first,
                                self.softmax_scale, base=base, rows=slot_rows)
        c = ctx.permute(2, 0, 1, 3).reshape(H, bsz * q_len, r).contiguous()
        o = inv_bmm(c, wuvT).view(H, bsz, q_len, vdim).permute(1, 2, 0, 3).reshape(bsz, q_len, H * vdim)
        return self.o_proj(o), None, past_key_value

    # -- cache the latent, re-expand K/V for every cached token -------------
    latent_all, kpe_all, _ = past_key_value.append(self.layer_idx, latent, k_pe)
    t_all = latent_all.shape[1]
    if attention_mask is None:
        raise ValueError("latent MLA path needs the 4D attention mask (as stock eager)")
    if attention_mask.size() != (bsz, 1, q_len, t_all):
        raise ValueError(f"attention mask {tuple(attention_mask.size())} != "
                         f"{(bsz, 1, q_len, t_all)}")
    # Rows are independent in attention, so it can run on row chunks to bound
    # the transient expanded K/V (B*T*heads*(q_head+v_head) values per layer).
    # Off (0) by default; COLOSSUS_ATTN_ROWS=R runs R rows at a time.
    rows = _attn_rows()
    if _inv_on() and not output_attentions:
        # Invariant mode: absorbed latent attention for ANY query count through fixed-order
        # kernels (invariant.py): bits independent of batch, padding and q_len.
        from .invariant import inv_bmm, inv_mla_attention
        wuk, wuvT = _absorb_weights(self)
        H, r = self.num_heads, self.kv_lora_rank
        qn = q_nope.permute(1, 0, 2, 3).reshape(H, bsz * q_len, nope).contiguous()
        q_lat = inv_bmm(qn, wuk).view(H, bsz, q_len, r).permute(1, 2, 0, 3)
        first = (attention_mask[:, 0, -1, :] == 0).int().argmax(dim=-1)
        ctx = inv_mla_attention(q_lat, q_pe.permute(0, 2, 1, 3), latent_all, kpe_all.squeeze(1),
                                first, self.softmax_scale)
        c = ctx.permute(2, 0, 1, 3).reshape(H, bsz * q_len, r).contiguous()
        o = inv_bmm(c, wuvT).view(H, bsz, q_len, vdim).permute(1, 2, 0, 3).reshape(bsz, q_len, H * vdim)
        return self.o_proj(o), None, past_key_value
    if q_len == 1 and _absorb_on() and not output_attentions:
        # Decode step in latent space (weight absorption): no K/V expansion.
        # Same weights and the same math up to bf16 rounding order, so not
        # bitwise-identical to the stock path; opt-in (COLOSSUS_MLA_ABSORB=1).
        attn_output = self.o_proj(_attend_absorbed(self, q_nope, q_pe, latent_all,
                                                   kpe_all, attention_mask))
        return attn_output, None, past_key_value
    if rows <= 0 or rows >= bsz:
        out, w = _attend(self, q_nope, q_pe, latent_all, kpe_all, attention_mask)
    else:
        parts = [_attend(self, q_nope[s:s + rows], q_pe[s:s + rows], latent_all[s:s + rows],
                         kpe_all[s:s + rows], attention_mask[s:s + rows])
                 for s in range(0, bsz, rows)]
        out = torch.cat([p[0] for p in parts], dim=0)
        w = torch.cat([p[1] for p in parts], dim=0) if output_attentions else None
        del parts
    attn_output = self.o_proj(out)
    return attn_output, (w if output_attentions else None), past_key_value


def _attn_rows() -> int:
    import os
    try:
        return int(os.environ.get("COLOSSUS_ATTN_ROWS", "0"))
    except ValueError:
        return 0


def _inv_on() -> bool:
    import os
    return os.environ.get("COLOSSUS_INVARIANT", "0") == "1"


def _absorb_on() -> bool:
    import os
    return os.environ.get("COLOSSUS_MLA_ABSORB", "0") == "1"


def _absorb_weights(self):
    """Per-head W_UK [H, nope, rank] and W_UV^T [H, rank, vdim] from kv_b_proj."""
    w = getattr(self, "_colossus_wuk", None)
    if w is None:
        if getattr(self.kv_b_proj, "bias", None) is not None:
            raise ValueError("absorbed MLA assumes kv_b_proj has no bias")
        H, nope, vdim = self.num_heads, self.qk_nope_head_dim, self.v_head_dim
        W = self.kv_b_proj.weight.detach().view(H, nope + vdim, self.kv_lora_rank)
        self._colossus_wuk = W[:, :nope, :].contiguous()
        self._colossus_wuvT = W[:, nope:, :].transpose(1, 2).contiguous()
    return self._colossus_wuk, self._colossus_wuvT


def _attend_absorbed(self, q_nope, q_pe, latent_all, kpe_all, attention_mask):
    """Attention in latent space: score = [q_nope W_UK | q_pe] . [latent | k_pe],
    output = (softmax . latent) W_UV. Returns pre-o_proj output [B, q, H*vdim]."""
    if self.training:
        raise ValueError("absorbed MLA is inference-only")
    bsz, H, q_len, nope = q_nope.shape
    r, vdim, t_all = self.kv_lora_rank, self.v_head_dim, latent_all.shape[1]
    wuk, wuvT = _absorb_weights(self)
    qn = q_nope.permute(1, 0, 2, 3).reshape(H, bsz * q_len, nope)
    q_lat = torch.bmm(qn, wuk).view(H, bsz, q_len, r).permute(1, 0, 2, 3)
    query = torch.cat([q_lat, q_pe], dim=-1).reshape(bsz, H * q_len, -1)
    del qn, q_lat
    key = torch.cat([latent_all, kpe_all.squeeze(1)], dim=-1)          # [B, T, r+rope]
    scores = torch.matmul(query, key.transpose(1, 2)) * self.softmax_scale
    del query, key
    scores = scores.view(bsz, H, q_len, t_all) + attention_mask
    p = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(latent_all.dtype)
    del scores
    ctx = torch.matmul(p.view(bsz, H * q_len, t_all), latent_all)       # [B, H*q, r]
    del p
    ctx = ctx.view(bsz, H, q_len, r).permute(1, 0, 2, 3).reshape(H, bsz * q_len, r)
    o = torch.bmm(ctx, wuvT)                                           # [H, B*q, vdim]
    return o.view(H, bsz, q_len, vdim).permute(1, 2, 0, 3).reshape(bsz, q_len, H * vdim)


def _attend(self, q_nope, q_pe, latent_all, kpe_all, attention_mask):
    """Stock eager attention math for a block of rows; returns pre-o_proj output."""
    nope, vdim = self.qk_nope_head_dim, self.v_head_dim
    bsz, _, q_len, _ = q_nope.shape
    t_all = latent_all.shape[1]
    kv = (self.kv_b_proj(latent_all)
          .view(bsz, t_all, self.num_heads, nope + vdim).transpose(1, 2))
    k_nope, value_states = torch.split(kv, [nope, vdim], dim=-1)

    query_states = q_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    query_states[:, :, :, :nope] = q_nope
    query_states[:, :, :, nope:] = q_pe
    key_states = kpe_all.new_empty(bsz, self.num_heads, t_all, self.q_head_dim)
    key_states[:, :, :, :nope] = k_nope
    key_states[:, :, :, nope:] = kpe_all
    del kv, k_nope
    # Stock DynamicCache (transformers 5.x) concatenates on every update,
    # first call included, so attention always sees contiguous V.
    value_states = value_states.contiguous()

    # -- identical to the stock eager forward from here ---------------------
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
    if attn_weights.size() != (bsz, self.num_heads, q_len, t_all):
        raise ValueError(f"attention weights {tuple(attn_weights.size())} != "
                         f"{(bsz, self.num_heads, q_len, t_all)}")
    del key_states, query_states
    attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                         dtype=torch.float32).to(value_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout,
                                         training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output.reshape(bsz, q_len, self.num_heads * vdim), attn_weights


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
