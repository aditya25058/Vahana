"""Survey MoE implementations in the installed transformers: experts params, activation, router, shared experts."""
import ast
import os
import sys

import transformers

T = os.path.join(os.path.dirname(transformers.__file__), "models")
FAM = sys.argv[1:] or ["mixtral", "qwen2_moe", "qwen3_moe", "qwen3_5_moe", "olmoe", "deepseek_v2", "deepseek_v3",
                       "deepseek_v32", "deepseek_v4", "gpt_oss", "llama4", "glm4_moe", "ernie4_5_moe", "granitemoe",
                       "granitemoeshared", "granitemoehybrid", "phimoe", "dbrx", "jamba", "jetmoe", "minimax",
                       "minimax_m2", "hunyuan_v1_moe", "lfm2_moe", "afmoe", "cohere2_moe", "exaone_moe",
                       "glm4_moe_lite", "glm_moe_dsa", "qwen3_vl_moe", "qwen3_omni_moe", "nllb_moe"]


def src(node, text):
    return ast.get_source_segment(text, node) or ""


for fam in FAM:
    f = os.path.join(T, fam, f"modeling_{fam}.py")
    if not os.path.exists(f):
        print(f"### {fam}: no modeling file")
        continue
    text = open(f).read()
    tree = ast.parse(text)
    out = []
    for c in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        n = c.name
        if not any(k in n for k in ("Experts", "Router", "Gate", "MoE", "Moe", "SparseMoe", "Sparse", "TopK")):
            continue
        if any(k in n for k in ("PreTrained", "Output", "Model", "Config", "Decoder", "Attention", "RMSNorm")):
            continue
        params, acts, calls = [], set(), set()
        for node in ast.walk(c):
            s = src(node, text) if isinstance(node, (ast.Assign,)) else ""
            if "nn.Parameter(" in s or "nn.Linear(" in s:
                params.append(s.split("=", 1)[0].strip().replace("self.", "") + ":" +
                              ("Param" if "Parameter" in s else "Linear") +
                              ("+bias" if "bias" in s and "bias=False" not in s else ""))
            if isinstance(node, ast.Call):
                fn = src(node.func, text)
                for k in ("softmax", "sigmoid", "topk", "act_fn", "clamp", "group", "e_score_correction_bias",
                          "routed_scaling_factor", "norm_topk_prob", "shared_expert_gate", "sparsemixer", "ACT2FN"):
                    if k in fn:
                        calls.add(k)
        for node in ast.walk(c):
            if isinstance(node, ast.Attribute) and node.attr in ("e_score_correction_bias", "routed_scaling_factor",
                                                                 "norm_topk_prob", "n_group", "topk_group",
                                                                 "shared_expert_gate", "alpha", "limit",
                                                                 "moe_softmax_order", "num_experts_per_tok"):
                calls.add(node.attr)
        out.append(f"  {n}: params={params[:6]} uses={sorted(calls)}")
    print(f"### {fam}")
    print("\n".join(out) if out else "  (no MoE classes found)")
