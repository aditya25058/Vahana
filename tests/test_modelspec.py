"""ModelSpec discovery on meta-device tiny models (CPU only, no weights)."""
import importlib

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from vahana.modelspec import UnsupportedModel, inspect

SMALL = dict(hidden_size=256, intermediate_size=512, moe_intermediate_size=128, num_hidden_layers=2,
             num_attention_heads=4, num_key_value_heads=2, head_dim=64, vocab_size=512, num_local_experts=8,
             num_experts=8, n_routed_experts=8, num_experts_per_tok=2, kv_lora_rank=64, q_lora_rank=96,
             qk_nope_head_dim=32, qk_rope_head_dim=32, v_head_dim=32, n_group=2, topk_group=1,
             first_k_dense_replace=0, shared_expert_intermediate_size=256, n_shared_experts=1,
             decoder_sparse_step=1, mlp_only_layers=[], num_shared_experts=1)
TIER1 = ["mixtral", "qwen2_moe", "qwen3_moe", "olmoe", "deepseek_v2", "deepseek_v3", "glm4_moe", "glm4_moe_lite",
         "minimax", "minimax_m2", "hunyuan_v1_moe", "ernie4_5_moe", "cohere2_moe", "exaone_moe", "afmoe", "phimoe",
         "qwen3_5_moe_text"]
NOT_YET = ["llama4_text"]


def meta_model(fam):
    cfg = AutoConfig.for_model(fam)
    for k, v in SMALL.items():
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, v)
            except Exception:  # noqa: BLE001
                pass
    for k in ("layer_types", "mlp_layer_types"):
        if isinstance(getattr(cfg, k, None), list):
            setattr(cfg, k, getattr(cfg, k)[: cfg.num_hidden_layers])
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(cfg)


@pytest.mark.parametrize("fam", TIER1)
def test_tier1_discovered(fam):
    specs = inspect(meta_model(fam))
    assert specs and all(s.layout == "fused_concat" and s.act == "silu_glu" for s in specs)
    assert [s.slot for s in specs] == list(range(len(specs)))
    s = specs[0]
    assert s.E == 8 and s.H == 256 and s.I in (128, 512)


@pytest.mark.parametrize("fam", NOT_YET)
def test_unsupported_layouts_fail_loudly(fam):
    with pytest.raises(UnsupportedModel) as e:
        inspect(meta_model(fam))
    assert "experts" in str(e.value) or "Experts" in str(e.value)


def test_gpt_oss_layout():
    specs = inspect(meta_model("gpt_oss"))
    s = specs[0]
    assert s.layout == "fused_interleaved_t" and s.act == "swiglu_oai" and s.bias
    assert (s.E, s.H) == (8, 256) and s.alpha == 1.702 and s.limit == 7.0


def _has(module: str, cls: str) -> bool:
    try:
        return hasattr(importlib.import_module(module), cls)
    except ImportError:
        return False


@pytest.mark.parametrize("fam,module,cls", [
    ("granitemoe", "transformers.models.granitemoe.modeling_granitemoe", "GraniteMoeExperts"),
    ("granitemoeshared", "transformers.models.granitemoeshared.modeling_granitemoeshared", "GraniteMoeSharedExperts"),
])
def test_granite_follows_transformers(fam, module, cls):
    """transformers 5.20 rebuilt Granite-MoE on the standard experts module: discovery must then accept it with no
    Vahana change; older transformers keep the ParallelExperts layout, which must be rejected loudly."""
    m = meta_model(fam)
    if _has(module, cls):
        specs = inspect(m)
        assert specs and specs[0].layout == "fused_concat" and specs[0].E == 8
    else:
        with pytest.raises(UnsupportedModel):
            inspect(m)
