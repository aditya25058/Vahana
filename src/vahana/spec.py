"""ModelSpec: find every MoE experts module by structure, never by family name.

The seam (transformers >= 5): a sparse block owns an `experts` module called as
`experts(hidden_states, top_k_index, top_k_weights)` that returns the weighted sum of the selected experts.
v0.1 layout `fused_concat`: gate_up_proj [E, 2I, H] (gate = rows [0, I), up = rows [I, 2I)), down_proj [E, H, I],
SiLU-GLU, no biases. Anything else raises UnsupportedModel with the class name and the reason; Vahana never
falls back silently.
"""
from __future__ import annotations

import inspect as _inspect
import re
from dataclasses import dataclass

import torch.nn as nn

SEAM_ARGS = ("hidden_states", "top_k_index", "top_k_weights")


class UnsupportedModel(ValueError):
    pass


@dataclass(frozen=True)
class ExpertsSpec:
    path: str           # module path of the experts module, e.g. "model.layers.3.mlp.experts"
    layer: int          # decoder layer index parsed from the path
    slot: int           # dense id 0..n-1 used to register the layer with the CPU kernel
    cls: str            # experts class name (for messages)
    layout: str         # "fused_concat"
    act: str            # "silu_glu"
    E: int
    I: int
    H: int


def _act_name(mod: nn.Module) -> str | None:
    a = getattr(mod, "act_fn", None)
    if a is None:
        return None
    n = type(a).__name__.lower()
    if n in ("siluactivation", "silu"):
        return "silu_glu"
    return n


def _check(path: str, ex: nn.Module) -> tuple[str, str, int, int, int]:
    cls = type(ex).__name__
    try:
        params = list(_inspect.signature(type(ex).forward).parameters)[1:4]
    except (TypeError, ValueError):
        params = []
    if tuple(params) != SEAM_ARGS:
        raise UnsupportedModel(f"{cls} at {path}: forward{tuple(params)} is not the experts seam {SEAM_ARGS}")
    own = dict(ex.named_parameters(recurse=False))
    if set(own) != {"gate_up_proj", "down_proj"}:
        raise UnsupportedModel(f"{cls} at {path}: parameters {sorted(own)} (expected gate_up_proj, down_proj only)")
    gu, dn = own["gate_up_proj"], own["down_proj"]
    if gu.dim() != 3 or dn.dim() != 3:
        raise UnsupportedModel(f"{cls} at {path}: expected 3-D expert tensors, got {tuple(gu.shape)} / {tuple(dn.shape)}")
    E, I2, H = gu.shape
    if I2 % 2 or tuple(dn.shape) != (E, H, I2 // 2):
        raise UnsupportedModel(f"{cls} at {path}: gate_up {tuple(gu.shape)} / down {tuple(dn.shape)} is not "
                               f"[E, 2I, H] / [E, H, I] (transposed or interleaved layouts are not v0.1)")
    act = _act_name(ex)
    if act != "silu_glu":
        raise UnsupportedModel(f"{cls} at {path}: activation {act!r} (v0.1 kernel: SiLU-GLU)")
    return cls, "fused_concat", E, I2 // 2, H


def inspect(model: nn.Module) -> list[ExpertsSpec]:
    """Every experts module of the model, in module order. Raises UnsupportedModel if there is none, or if one
    does not match a supported layout."""
    out: list[ExpertsSpec] = []
    for name, mod in model.named_modules():
        ex = getattr(mod, "experts", None)
        if not isinstance(ex, nn.Module) or isinstance(ex, nn.ModuleList):
            continue
        path = f"{name}.experts" if name else "experts"
        cls, layout, E, I, H = _check(path, ex)
        m = re.search(r"layers\.(\d+)\.", path)
        out.append(ExpertsSpec(path=path, layer=int(m.group(1)) if m else -1, slot=len(out), cls=cls,
                               layout=layout, act="silu_glu", E=E, I=I, H=H))
    if not out:
        kinds = sorted({type(m).__name__ for m in model.modules() if "moe" in type(m).__name__.lower()
                        or "expert" in type(m).__name__.lower()})
        raise UnsupportedModel(f"{type(model).__name__}: no module with an `experts` child "
                               f"(MoE-like classes seen: {kinds[:6]})")
    return out


def parent_of(model: nn.Module, path: str) -> tuple[nn.Module, str]:
    parent_path, _, leaf = path.rpartition(".")
    return model.get_submodule(parent_path) if parent_path else model, leaf
