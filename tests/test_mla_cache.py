"""Latent MLA KV cache vs the stock full K/V cache on a small MLA replica.

The replica mirrors the eager MLA attention structure (q LoRA, compressed
KV + rotary key part, kv_b_proj expansion). Bitwise equality is a GPU
property audited on hardware; here outputs must match to fp32 noise and
the cache must hold only latent + rotary part.
"""
from __future__ import annotations

import types

import pytest
import torch
from torch import nn
from transformers.cache_utils import DynamicCache

from vahana.mla_cache import (
    LatentKVCache,
    cache_nbytes,
    install_latent_kv,
)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Rotary(nn.Module):
    def __init__(self, dim, n=256):
        super().__init__()
        inv = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        f = torch.outer(torch.arange(n).float(), inv)
        emb = torch.cat((f, f), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        return (self.cos_cached[:seq_len].to(x.dtype), self.sin_cached[:seq_len].to(x.dtype))


class TinyMLA(nn.Module):
    """Eager MLA attention, same structure as stock MLA modeling code."""

    def __init__(self, layer_idx, hidden=32, heads=4, nope=8, rope=4, vdim=8, rank=12):
        super().__init__()
        self.layer_idx, self.num_heads = layer_idx, heads
        self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim = nope, rope, vdim
        self.q_head_dim, self.kv_lora_rank, self.q_lora_rank = nope + rope, rank, 16
        self.softmax_scale, self.attention_dropout = self.q_head_dim ** -0.5, 0.0
        self.q_a_proj = nn.Linear(hidden, 16, bias=False)
        self.q_a_layernorm = nn.LayerNorm(16)
        self.q_b_proj = nn.Linear(16, heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(hidden, rank + rope, bias=False)
        self.kv_a_layernorm = nn.LayerNorm(rank)
        self.kv_b_proj = nn.Linear(rank, heads * (nope + vdim), bias=False)
        self.o_proj = nn.Linear(heads * vdim, hidden, bias=False)
        self.rotary_emb = Rotary(rope)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False, **kw):
        bsz, q_len, _ = hidden_states.size()
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        ckv = self.kv_a_proj_with_mqa(hidden_states)
        ckv, k_pe = torch.split(ckv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (self.kv_b_proj(self.kv_a_layernorm(ckv))
              .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
              .transpose(1, 2))
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        kv_seq_len = q_len + past_key_value.get_seq_length(self.layer_idx)
        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
        qs = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        qs[..., :self.qk_nope_head_dim], qs[..., self.qk_nope_head_dim:] = q_nope, q_pe
        ks = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        ks[..., :self.qk_nope_head_dim], ks[..., self.qk_nope_head_dim:] = k_nope, k_pe
        ks, v = past_key_value.update(ks, v, self.layer_idx, {"sin": sin, "cos": cos})
        w = torch.matmul(qs, ks.transpose(2, 3)) * self.softmax_scale + attention_mask
        w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(qs.dtype)
        out = torch.matmul(w, v).transpose(1, 2).reshape(bsz, q_len, -1)
        return self.o_proj(out), None, past_key_value


class TinyModel(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.config = types.SimpleNamespace(_attn_implementation="eager")
        self.layers = nn.ModuleList([TinyMLA(i) for i in range(n_layers)])

    def forward(self, x, past):
        b, s, _ = x.shape
        t0 = past.get_seq_length(0)
        pos = torch.arange(t0, t0 + s).unsqueeze(0).expand(b, s)
        mask = torch.full((s, t0 + s), float("-inf")).triu(t0 + 1)
        mask = mask.view(1, 1, s, t0 + s).expand(b, 1, s, t0 + s)
        for layer in self.layers:
            x = x + layer(x, attention_mask=mask, position_ids=pos, past_key_value=past)[0]
        return x


def _run(model, cache, steps):
    outs = [model(x, cache) for x in steps]
    return torch.cat(outs, dim=1), cache


def test_latent_cache_matches_full_cache_and_is_smaller():
    torch.manual_seed(0)
    model = TinyModel()
    steps = [torch.randn(2, 5, 32)] + [torch.randn(2, 1, 32) for _ in range(4)]
    ref, full = _run(model, DynamicCache(), steps)
    assert install_latent_kv(model) == 2
    got, lat = _run(model, LatentKVCache(), steps)
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)
    assert lat.get_seq_length(0) == 9 and lat.get_seq_length(1) == 9
    # latent (12) + rotary key (4) per token per layer vs heads*(12+8) = 80
    assert cache_nbytes(lat) * 5 == cache_nbytes(full)
    # stock path still served when the cache is not latent
    again, _ = _run(model, DynamicCache(), steps)
    assert torch.allclose(again, ref, atol=1e-6)


def test_install_refuses_non_eager_and_non_mla():
    m = TinyModel()
    m.config._attn_implementation = "flash_attention_2"
    with pytest.raises(ValueError, match="eager"):
        install_latent_kv(m)
    plain = nn.Sequential(nn.Linear(4, 4))
    with pytest.raises(ValueError, match="no MLA"):
        install_latent_kv(plain)
