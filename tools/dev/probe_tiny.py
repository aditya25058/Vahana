"""Build a tiny random model per MoE family (from the family's config class, shrunk) and report the MoE seam:
block class, experts class, experts.forward signature, parameter shapes, activation, router class."""
import inspect
import sys
import traceback

import torch
from transformers import AutoConfig, AutoModelForCausalLM

FAMS = sys.argv[1:] or ["mixtral", "qwen2_moe", "qwen3_moe", "olmoe", "deepseek_v2", "deepseek_v3", "glm4_moe",
                        "glm4_moe_lite", "minimax", "minimax_m2", "hunyuan_v1_moe", "ernie4_5_moe", "cohere2_moe",
                        "exaone_moe", "afmoe", "phimoe", "granitemoe", "granitemoeshared", "gpt_oss", "llama4_text",
                        "qwen3_5_moe_text", "lfm2_moe", "jamba"]
SMALL = dict(hidden_size=256, intermediate_size=512, moe_intermediate_size=128, num_hidden_layers=2,
             num_attention_heads=4, num_key_value_heads=2, head_dim=64, vocab_size=512, num_local_experts=8,
             num_experts=8, n_routed_experts=8, num_experts_per_tok=2, max_position_embeddings=256,
             kv_lora_rank=64, q_lora_rank=96, qk_nope_head_dim=32, qk_rope_head_dim=32, v_head_dim=32,
             n_group=2, topk_group=1, first_k_dense_replace=0, shared_expert_intermediate_size=256,
             n_shared_experts=1, moe_layer_freq=1, decoder_sparse_step=1, mlp_only_layers=[],
             expert_intermediate_size=128, num_shared_experts=1, sliding_window=None)


def tiny_config(fam):
    cfg = AutoConfig.for_model(fam)
    for k, v in SMALL.items():
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, v)
            except Exception:  # noqa: BLE001
                pass
    for k in ("layer_types", "mlp_layer_types", "layers_block_type"):
        if isinstance(getattr(cfg, k, None), list):
            setattr(cfg, k, getattr(cfg, k)[: cfg.num_hidden_layers])
    return cfg


for fam in FAMS:
    try:
        cfg = tiny_config(fam)
        with torch.device("meta"):
            m = AutoModelForCausalLM.from_config(cfg)
        blocks = [(n, mod) for n, mod in m.named_modules() if hasattr(mod, "experts") and isinstance(mod.experts, torch.nn.Module)]
        if not blocks:
            print(f"### {fam}: NO block with .experts")
            continue
        n, b = blocks[0]
        ex = b.experts
        sig = str(inspect.signature(type(ex).forward))
        ps = {k: tuple(v.shape) for k, v in ex.named_parameters(recurse=False)}
        sub = [k for k, _ in ex.named_children()][:4]
        router = [(k, type(v).__name__) for k, v in b.named_children() if k != "experts"]
        print(f"### {fam}: {len(blocks)} blocks, block={type(b).__name__} at {n}")
        print(f"    experts={type(ex).__name__}{sig} params={ps} children={sub} act={getattr(cfg, 'hidden_act', None)}")
        print(f"    siblings={router}")
    except Exception as e:  # noqa: BLE001
        print(f"### {fam}: ERROR {type(e).__name__}: {str(e)[:160]}")
