"""M2: per-family exactness (M1 harness + gpt_oss, a non-64 width variant, 40-token prompts) of VahanaExperts on tiny random models (GPU + CPU kernel).

For each family: build a tiny model from the family's config (shrunk dims, random weights, seed fixed), run
  stock     transformers' own experts (bf16)
  ref       TwinGPUExperts  (Vahana arithmetic on the GPU)
  cpu       VahanaExperts   (Vahana arithmetic on the CPU kernel)
on the same prompt: prefill logits at every position + 8 greedy decode steps (logits and ids).
Saves cpu logits per family (for the thread-invariance comparison across processes).
usage: m1_families.py OUT.json OUT_LOGITS.pt [family ...]
"""
import json
import os
import sys
import traceback

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from vahana import experts as vx
from vahana.modelspec import inspect, parent_of

TIER1 = ["mixtral", "qwen2_moe", "qwen3_moe", "olmoe", "deepseek_v2", "deepseek_v3", "glm4_moe", "glm4_moe_lite",
         "minimax", "minimax_m2", "hunyuan_v1_moe", "ernie4_5_moe", "cohere2_moe", "exaone_moe", "afmoe", "phimoe",
         "qwen3_5_moe_text"]
EXTRA = ["gpt_oss", "mixtral@intermediate_size=96"]
PROMPT = 40
SMALL = dict(hidden_size=256, intermediate_size=512, moe_intermediate_size=128, num_hidden_layers=4,
             num_attention_heads=4, num_key_value_heads=2, vocab_size=512, num_local_experts=8,
             num_experts=8, n_routed_experts=8, num_experts_per_tok=2, max_position_embeddings=256,
             kv_lora_rank=64, q_lora_rank=96, qk_nope_head_dim=32, qk_rope_head_dim=32, v_head_dim=32,
             n_group=2, topk_group=1, first_k_dense_replace=0, shared_expert_intermediate_size=256,
             n_shared_experts=1, moe_layer_freq=1, decoder_sparse_step=1, mlp_only_layers=[],
             expert_intermediate_size=128, num_shared_experts=1)
dev = torch.device("cuda")
NEW = 8


def tiny_config(fam):
    """The family's own config class, constructed with small sizes so derived fields are consistent."""
    fam, _, over = fam.partition("@")
    base = AutoConfig.for_model(fam)
    kw = {k: v for k, v in SMALL.items() if hasattr(base, k)}
    for item in filter(None, over.split(",")):
        k, v = item.split("=")
        kw[k] = int(v)
    if hasattr(base, "kv_lora_rank"):                 # MLA: every query head has its own latent K/V
        kw["num_key_value_heads"] = kw["num_attention_heads"]
    cfg = AutoConfig.for_model(fam, **kw)
    if hasattr(cfg, "head_dim") and cfg.head_dim is None:
        cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads
    return cfg


def build(fam):
    cfg = tiny_config(fam)
    torch.manual_seed(1234)
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).to(dev).eval()
    gen = torch.Generator(device=dev).manual_seed(99)
    with torch.no_grad():
        for n, p in m.named_parameters():       # experts (and anything left uninitialised) get N(0, 0.02)
            zero_router = "router" in n and not p.any()
            if n.endswith("gate_up_proj") or n.endswith("down_proj") and p.dim() == 3 or n.endswith("_bias")                     and "experts" in n or zero_router or not torch.isfinite(p).all():
                p.copy_((torch.randn(p.shape, generator=gen, device=dev) * 0.02).to(p.dtype))
    return cfg, m


def run(m, ids):
    with torch.no_grad():
        pre = m(ids).logits.float()
        g = m.generate(ids, max_new_tokens=NEW, min_new_tokens=NEW, do_sample=False, output_logits=True,
                       return_dict_in_generate=True, pad_token_id=0)
    return pre, [l.float() for l in g.logits], g.sequences[0, ids.shape[1]:].tolist()


def main():
    OUT, OUTPT = sys.argv[1], sys.argv[2]
    FAMS = sys.argv[3:] or TIER1 + EXTRA
    res, saved = {"omp": os.environ.get("OMP_NUM_THREADS")}, {}
    for fam in FAMS:
        r = {}
        try:
            cfg, m = build(fam)
            specs = inspect(m)
            r["experts_modules"] = len(specs)
            r["E_I_H"] = [specs[0].E, specs[0].I, specs[0].H]
            orig = {s.path: getattr(*parent_of(m, s.path)) for s in specs}
            ids = torch.randint(0, cfg.vocab_size, (1, PROMPT), generator=torch.Generator().manual_seed(7)).to(dev)
            stock = run(m, ids)

            def swap(mode):
                for s in specs:
                    parent, leaf = parent_of(m, s.path)
                    setattr(parent, leaf, orig[s.path])
                vx.install(m, mode=mode)

            swap("gpu_twin")
            ref = run(m, ids)
            swap("cpu")
            cpu = run(m, ids)
            r["prefill_equal"] = torch.equal(ref[0], cpu[0])
            r["decode_logits_equal"] = all(torch.equal(a, b) for a, b in zip(ref[1], cpu[1]))
            r["ids_equal"] = ref[2] == cpu[2]
            r["exact"] = r["prefill_equal"] and r["decode_logits_equal"] and r["ids_equal"]
            scale = stock[0].abs().max().item()
            r["stock_vs_ref_max_rel"] = (stock[0] - ref[0]).abs().max().item() / max(scale, 1e-9)
            r["stock_ids_agree"] = sum(a == b for a, b in zip(stock[2], ref[2]))
            saved[fam] = {"prefill": cpu[0].cpu(), "decode": [x.cpu() for x in cpu[1]], "ids": cpu[2]}
        except Exception as e:  # noqa: BLE001
            r["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            r["trace"] = traceback.format_exc()[-800:]
        res[fam] = r
        print(fam, {k: v for k, v in r.items() if k != "trace"}, flush=True)
        torch.cuda.empty_cache()

    res["summary"] = {"families": len(FAMS), "exact": sum(1 for f in FAMS if res[f].get("exact")),
                      "errors": [f for f in FAMS if "error" in res[f]]}
    print("SUMMARY", res["summary"], flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    torch.save(saved, OUTPT)


if __name__ == "__main__":
    main()
