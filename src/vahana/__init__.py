"""Vahana: exact inference for MoE models too big for your GPU.

    model, info = vahana.load("/path/to/model", gpu_budget_gb=16)
    out = vahana.generate(model, input_ids, max_new_tokens=128)

or from the shell: `vahana run MODEL --prompt ...` / `vahana serve MODEL`.

Importing the package loads nothing (no torch): the command line sets process-wide options (NUMA policy, OpenMP)
before torch is imported. Every name below resolves on first use.
"""
from __future__ import annotations

import importlib

__version__ = "0.1.0"

_LAZY = {
    '.offload': ('hook', '.offload'),
    'ColumnDirectory': ('directory', 'ColumnDirectory'),
    'ExpertOffloadManager': ('hook', 'ExpertOffloadManager'),
    'FastSlot': ('placement', 'FastSlot'),
    'GATE_DEEPSEEK_3TUPLE': ('interface', 'GATE_DEEPSEEK_3TUPLE'),
    'GATE_LOGITS': ('interface', 'GATE_LOGITS'),
    'MoELayerSpec': ('interface', 'MoELayerSpec'),
    'RangeFetcher': ('remote', 'RangeFetcher'),
    'RemoteShardHandles': ('remote', 'RemoteShardHandles'),
    'ShardHandles': ('loading', 'ShardHandles'),
    'ShardMap': ('loading', 'ShardMap'),
    'TieredHandles': ('remote', 'TieredHandles'),
    'TieredMoEWrapper': ('placement', 'TieredMoEWrapper'),
    'VahanaMoEHook': ('hook', 'VahanaMoEHook'),
    'ZSSRPredictor': ('predictor', 'ZSSRPredictor'),
    'apply_gate': ('placement', 'apply_gate'),
    'as': ('hwprobe', 'as'),
    'classify_role': ('interface', 'classify_role'),
    'columns_batched': ('columns', 'columns_batched'),
    'columns_loop': ('columns', 'columns_loop'),
    'dense_expert_forward': ('sa_ffn', 'dense_expert_forward'),
    'describe_model': ('inspector', 'describe_model'),
    'expert_weight_keys': ('interface', 'expert_weight_keys'),
    'experts_of': ('interface', 'experts_of'),
    'feasibility_plan': ('feasibility', 'feasibility_plan'),
    'find_moe_block': ('interface', 'find_moe_block'),
    'find_proj': ('interface', 'find_proj'),
    'from': ('hook', 'from'),
    'generate': ('decode', 'generate'),
    'hardware_probe': ('hwprobe', 'hardware_probe'),
    'import': ('hook', 'import'),
    'inspect': ('modelspec', 'inspect'),
    'inspect_weights': ('inspector', 'inspect_weights'),
    'install_cache_compat': ('serve', 'install_cache_compat'),
    'is_routed_expert_key': ('loading', 'is_routed_expert_key'),
    'load': ('loader', 'load'),
    'materialize': ('loading', 'materialize'),
    'normalize_activation': ('interface', 'normalize_activation'),
    'plan': ('feasibility', 'plan'),
    'plan_fixed_packets': ('directory', 'plan_fixed_packets'),
    'prepare': ('loader', 'prepare'),
    'prepare_model': ('serve', 'prepare_model'),
    'probe': ('hwprobe', 'probe'),
    'quantize_int8': ('predictor', 'quantize_int8'),
    'render_table': ('feasibility', 'render_table'),
    'sa_expert_forward': ('sa_ffn', 'sa_expert_forward'),
    'serve': ('server', 'serve'),
    'serve_huge_moe': ('serve', 'serve_huge_moe'),
    'split_routed': ('loading', 'split_routed'),
    'stack_expert_weights': ('columns', 'stack_expert_weights'),
    'verify_lossless': ('sa_ffn', 'verify_lossless'),
    'wrap_moe_layers': ('placement', 'wrap_moe_layers'),
    'write_inspect_json': ('inspector', 'write_inspect_json'),
    'write_probe_json': ('hwprobe', 'write_probe_json'),
}


def __getattr__(name):
    if name in _LAZY:
        mod, attr = _LAZY[name]
        value = getattr(importlib.import_module(f".{mod}", __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'vahana' has no attribute {name!r}")


__all__ = sorted(_LAZY)
