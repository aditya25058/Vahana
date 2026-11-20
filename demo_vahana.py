"""Vahana End-to-End Demo and Verification Script.

Demonstrates:
1. MoE Architecture Structural Inspection (ModelSpec)
2. Hardware Probe & Feasibility Planning
3. Exact Inference Simulation with TieredMoEWrapper
"""
import sys
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

import vahana
from vahana.feasibility import plan, render_table
from vahana.placement import TieredMoEWrapper
from vahana.serve import prepare_model, serve_huge_moe
from tests.test_interface import FakeDeepSeekBlock
from tests.test_placement import DictHandles
from tests.test_serve import FakeCausalMoE, FakeProfile, FakeTokenizer


def demo_modelspec():
    print("=" * 60)
    print("1. MoE Architecture Inspection (ModelSpec)")
    print("=" * 60)
    models_to_test = ["mixtral", "qwen2_moe", "deepseek_v2", "granitemoe"]
    for fam in models_to_test:
        cfg = AutoConfig.for_model(fam)
        cfg.num_hidden_layers = 2
        if hasattr(cfg, "num_local_experts"):
            cfg.num_local_experts = 8
        if hasattr(cfg, "num_experts"):
            cfg.num_experts = 8
        if hasattr(cfg, "hidden_size"):
            cfg.hidden_size = 256
        if hasattr(cfg, "intermediate_size"):
            cfg.intermediate_size = 512
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(cfg)
        specs = vahana.inspect(model)
        first = specs[0]
        print(f"  [+] Family: {fam:<15} Layers: {len(specs)} | Seam: {first.cls} | Layout: {first.layout} (E={first.E}, H={first.H}, I={first.I})")
    print()


def demo_feasibility():
    print("=" * 60)
    print("2. Hardware Probing & Feasibility Planning")
    print("=" * 60)
    hw = vahana.probe()
    print(f"  CPU Cores: {hw['cpu']['cores']}, Arch: {hw['cpu']['arch']}")
    print(f"  CUDA Available: {hw['cuda_available']} (PyTorch {hw['torch']})")
    
    # Simulate planning for a 236B parameter MoE model with 16GB GPU budget
    model_meta = {
        "model": {"name": "DeepSeek-Coder-V2-Lite"},
        "moe": {"routed_per_layer": 64, "shared": 2, "top_k": 6, "moe_layers": 28, "hidden": 2048, "intermediate": 1408},
        "weights": {"total_gb": 32.0, "resident_gb": 6.0, "routed_gb": 26.0, "dtype": "BF16"},
        "capabilities": {}
    }
    sim_hw = {
        "gpus": [{"total_gb": 4.0}],
        "ram_gb": {"available_gb": 32.0},
        "pcie_gbs": 16.0,
        "dram_gbs": 40.0,
        "cuda_available": True,
        "cpu": {"bf16": True}
    }
    workload = {"batch": 1, "gen_tokens": 32, "shared": True}
    res = plan(model_meta, sim_hw, workload)
    print("\nFeasibility Analysis Table:")
    print(render_table(res, model_meta, sim_hw))
    print()


def _stub_handles(model):
    table = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and ".experts." in name:
            table[name + ".weight"] = mod.weight.detach().clone()
    stubs = DictHandles(table)
    stubs.weight_map = dict(table)
    return stubs


def demo_inference():
    print("=" * 60)
    print("3. End-to-End Tiered MoE Inference Run")
    print("=" * 60)
    torch.manual_seed(0)
    model = FakeCausalMoE(n_layers=2, vocab=32, hidden=16)
    tok = FakeTokenizer()
    handles = _stub_handles(model)
    print("  Running serve_huge_moe generation loop (device: CPU)...")
    res = serve_huge_moe(
        model,
        tok,
        FakeProfile(),
        handles,
        torch.device("cpu"),
        ["Hello Vahana!", "Explain MoE routing"],
        max_new_tokens=4,
        capacity=8,
        placement="cpu"
    )
    print(f"  Generated {len(res['texts'])} sequences:")
    for i, t in enumerate(res["texts"]):
        print(f"    Output {i+1}: \"{t}\"")
    print(f"  Total misses: {res['total_misses']}")
    print(f"  Batch decode throughput: {res['batch_decode_tps']:.2f} tok/s")
    print()


if __name__ == "__main__":
    demo_modelspec()
    demo_feasibility()
    demo_inference()
    print("All Vahana demonstrations executed successfully!")
