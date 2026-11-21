"""vahana doctor reads sizes from config.json and safetensors headers only, and plans placement."""
import json
import os
import tempfile

import torch
from safetensors.torch import save_file

from vahana import doctor


def test_model_sizes_and_plan():
    d = tempfile.mkdtemp()
    json.dump({"model_type": "qwen3_moe", "num_hidden_layers": 1, "num_experts": 4, "num_experts_per_tok": 2},
              open(os.path.join(d, "config.json"), "w"))
    t = {"model.embed_tokens.weight": torch.zeros(64, 32, dtype=torch.bfloat16)}
    for e in range(4):
        t[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = torch.zeros(16, 32, dtype=torch.bfloat16)
    save_file(t, os.path.join(d, "model.safetensors"))
    m = doctor.model(d)
    assert m["experts"] == 4 and m["top_k"] == 2
    assert m["expert_bytes"] == 4 * 16 * 32 * 2 and m["nonexpert_bytes"] == 64 * 32 * 2
    assert m["active_expert_bytes"] == m["expert_bytes"] / 2
    mc = {"ram_available_gb": 8.0, "gpu": None, "mem_bw_gbs": 20.0}
    p = doctor.plan(m, mc, None, 1.0)
    assert p["device"] == "cpu" and p["experts"] == "resident in RAM" and p["est_tok_s"] > 0
    p = doctor.plan(m, mc, 1e-6, 1.0)                       # a GPU budget smaller than the non-expert weights
    assert p["device"] == "cuda" and "CPU RAM" in p["nonexpert_on"]
