"""Phase 31: serve Mistral "consolidated" checkpoints (e.g. Mixtral-8x7B) through the HF model classes.

VAHANA_REMAP=mistral_consolidated wraps the shard handles. Every HF key maps to a lossless view or
reordering of a consolidated tensor (the same transformations as transformers'
convert_mixtral_weights_to_hf), done at load time instead of writing a converted copy:
  tok_embeddings -> model.embed_tokens; norm -> model.norm; output -> lm_head
  layers.L.attention_norm -> input_layernorm; ffn_norm -> post_attention_layernorm
  attention.wq / wk -> q_proj / k_proj with the rotary head permutation; wv -> v_proj; wo -> o_proj
  block_sparse_moe.gate -> gate
  block_sparse_moe.w1 / w3 [E*I, H] -> experts.e.w1 / w3 = rows [e*I, (e+1)*I)  (zero-copy views)
  block_sparse_moe.w2 [E*I, H]      -> experts.e.w2 = rows [e*I, (e+1)*I) transposed to [H, I] (one copy)
"""
from __future__ import annotations

import torch


def _permute(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    d1, d2 = w.shape
    return w.view(n_heads, d1 // n_heads // 2, 2, d2).transpose(1, 2).reshape(d1, d2).contiguous()


class MistralConsolidatedHandles:
    def __init__(self, base, cfg):
        self.base = base
        self.model_dir = getattr(base, "model_dir", "")
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.n_exp = cfg.num_local_experts
        self.inter = cfg.intermediate_size
        self._src: dict[str, torch.Tensor] = {}
        self.src_of: dict[str, tuple] = {}
        wm = {}
        bw = base.weight_map
        for k, shard in bw.items():
            if k == "tok_embeddings.weight":
                self._add(wm, "model.embed_tokens.weight", k, shard, "id")
            elif k == "norm.weight":
                self._add(wm, "model.norm.weight", k, shard, "id")
            elif k == "output.weight":
                self._add(wm, "lm_head.weight", k, shard, "id")
            elif k.startswith("layers."):
                L, rest = k.split(".", 2)[1:]
                p = f"model.layers.{L}."
                if rest == "attention_norm.weight":
                    self._add(wm, p + "input_layernorm.weight", k, shard, "id")
                elif rest == "ffn_norm.weight":
                    self._add(wm, p + "post_attention_layernorm.weight", k, shard, "id")
                elif rest == "attention.wq.weight":
                    self._add(wm, p + "self_attn.q_proj.weight", k, shard, ("perm", self.n_heads))
                elif rest == "attention.wk.weight":
                    self._add(wm, p + "self_attn.k_proj.weight", k, shard, ("perm", self.n_kv))
                elif rest == "attention.wv.weight":
                    self._add(wm, p + "self_attn.v_proj.weight", k, shard, "id")
                elif rest == "attention.wo.weight":
                    self._add(wm, p + "self_attn.o_proj.weight", k, shard, "id")
                elif rest == "block_sparse_moe.gate.weight":
                    # transformers 5: the router lives at mlp.gate (fused-expert MoE block)
                    self._add(wm, p + "mlp.gate.weight", k, shard, "id")
                elif rest in ("block_sparse_moe.w1", "block_sparse_moe.w2", "block_sparse_moe.w3"):
                    wn = rest.rsplit(".", 1)[1]
                    for e in range(self.n_exp):
                        self._add(wm, p + f"block_sparse_moe.experts.{e}.{wn}.weight", k, shard,
                                  ("rows_t" if wn == "w2" else "rows", e))
                else:
                    raise KeyError(f"unmapped consolidated key {k}")
            else:
                raise KeyError(f"unmapped consolidated key {k}")
        self.weight_map = wm

    def _add(self, wm, hf, src, shard, op):
        wm[hf] = shard
        self.src_of[hf] = (src, op)

    def _source(self, key: str) -> torch.Tensor:
        t = self._src.get(key)
        if t is None:
            t = self._src[key] = self.base.header_view(key)[0]       # zero-copy view of the shard
        return t

    def get_tensor(self, key: str) -> torch.Tensor:
        src, op = self.src_of[key]
        t = self._source(src)
        if op == "id":
            return t
        if op[0] == "perm":
            return _permute(t, op[1])
        e = op[1]
        rows = t[e * self.inter:(e + 1) * self.inter]
        if op[0] == "rows":
            return rows
        return rows.t().contiguous()                                  # w2: [I, H] -> HF down [H, I]

    def header(self, key: str) -> dict:
        src, op = self.src_of[key]
        h = dict(self.base.header(src))
        t_shape = list(h["shape"])
        if op != "id" and op[0] in ("rows", "rows_t"):
            t_shape = [self.inter, t_shape[1]] if op[0] == "rows" else [t_shape[1], self.inter]
        h["shape"] = t_shape
        return h

    def shards(self):
        return self.base.shards()

    def __getattr__(self, name):
        return getattr(self.base, name)
