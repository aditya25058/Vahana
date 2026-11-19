"""ModelSpec: find every MoE experts module by structure, never by family name.

The seam (transformers >= 5): a sparse block owns an `experts` module called as
`experts(hidden_states, indices [N, K], weights [N, K])` that returns the weighted sum of the selected experts.
Supported layouts:
  fused_concat        gate_up_proj [E, 2I, H] (gate = rows [0, I), up = rows [I, 2I)), down_proj [E, H, I];
                      activation from `act_fn` (SiLU-GLU)
  fused_interleaved_t gate_up_proj [E, H, 2I] used as x @ W (gate = columns 0::2, up = 1::2),
                      down_proj [E, I, H], per-expert biases gate_up_proj_bias [E, 2I] / down_proj_bias [E, H];
                      activation = clamped SwiGLU with `alpha` / `limit` (the gpt-oss experts form)
Anything else raises UnsupportedModel with the class name and the reason; Vahana never falls back silently.
"""
from __future__ import annotations

import inspect as _inspect
import re
from dataclasses import dataclass

import torch.nn as nn

SEAM_ARGS = {("hidden_states", "top_k_index", "top_k_weights"), ("hidden_states", "router_indices", "routing_weights")}
ACT_IDS = {"silu_glu": 0, "swiglu_oai": 1}


class UnsupportedModel(ValueError):
    pass


@dataclass(frozen=True)
class ExpertsSpec:
    path: str           # module path of the experts module, e.g. "model.layers.3.mlp.experts"
    layer: int          # decoder layer index parsed from the path
    slot: int           # dense id 0..n-1 used to register the layer with the CPU kernel
    cls: str            # experts class name (for messages)
    layout: str         # "fused_concat" | "fused_interleaved_t"
    act: str            # "silu_glu" | "swiglu_oai"
    E: int
    I: int
    H: int
    bias: bool = False
    alpha: float = 1.702
    limit: float = 7.0


def _act_name(mod: nn.Module) -> str | None:
    a = getattr(mod, "act_fn", None)
    if a is None:
        return None
    n = type(a).__name__.lower()
    return "silu_glu" if n in ("siluactivation", "silu") else n


def _check(path: str, ex: nn.Module) -> dict:
    cls = type(ex).__name__
    try:
        params = tuple(list(_inspect.signature(type(ex).forward).parameters)[1:4])
    except (TypeError, ValueError):
        params = ()
    if params not in SEAM_ARGS:
        raise UnsupportedModel(f"{cls} at {path}: forward{params} is not the experts seam "
                               f"(hidden_states, indices, weights)")
    own = {k: tuple(v.shape) for k, v in ex.named_parameters(recurse=False)}
    if set(own) == {"gate_up_proj", "down_proj"}:
        gu, dn = own["gate_up_proj"], own["down_proj"]
        if len(gu) == 3 and len(dn) == 3 and gu[1] % 2 == 0 and dn == (gu[0], gu[2], gu[1] // 2):
            act = _act_name(ex)
            if act != "silu_glu":
                raise UnsupportedModel(f"{cls} at {path}: activation {act!r} (supported: SiLU-GLU)")
            return dict(cls=cls, layout="fused_concat", act=act, E=gu[0], I=gu[1] // 2, H=gu[2], bias=False)
        raise UnsupportedModel(f"{cls} at {path}: gate_up {gu} / down {dn} is not [E, 2I, H] / [E, H, I] "
                               f"(transposed layouts without the clamped-SwiGLU form are not supported)")
    if set(own) == {"gate_up_proj", "gate_up_proj_bias", "down_proj", "down_proj_bias"}:
        gu, gb, dn, db = own["gate_up_proj"], own["gate_up_proj_bias"], own["down_proj"], own["down_proj_bias"]
        E, H, I2 = gu if len(gu) == 3 else (0, 0, 0)
        shapes_ok = I2 % 2 == 0 and gb == (E, I2) and dn == (E, I2 // 2, H) and db == (E, H)
        form_ok = all(isinstance(getattr(ex, a, None), float) for a in ("alpha", "limit")) and \
            callable(getattr(ex, "_apply_gate", None))
        if shapes_ok and form_ok:
            return dict(cls=cls, layout="fused_interleaved_t", act="swiglu_oai", E=E, I=I2 // 2, H=H, bias=True,
                        alpha=float(ex.alpha), limit=float(ex.limit))
        raise UnsupportedModel(f"{cls} at {path}: biased experts {own} are not the [E, H, 2I] interleaved "
                               f"clamped-SwiGLU form")
    raise UnsupportedModel(f"{cls} at {path}: parameters {sorted(own)} (no supported experts layout)")


def inspect(model: nn.Module) -> list[ExpertsSpec]:
    """Every experts module of the model, in module order. Raises UnsupportedModel if there is none, or if one
    does not match a supported layout."""
    out: list[ExpertsSpec] = []
    for name, mod in model.named_modules():
        ex = getattr(mod, "experts", None)
        if not isinstance(ex, nn.Module) or isinstance(ex, nn.ModuleList):
            continue
        path = f"{name}.experts" if name else "experts"
        d = _check(path, ex)
        m = re.search(r"layers\.(\d+)\.", path)
        out.append(ExpertsSpec(path=path, layer=int(m.group(1)) if m else -1, slot=len(out), **d))
    if not out:
        kinds = sorted({type(m).__name__ for m in model.modules() if "moe" in type(m).__name__.lower()
                        or "expert" in type(m).__name__.lower()})
        raise UnsupportedModel(f"{type(model).__name__}: no module with an `experts` child "
                               f"(MoE-like classes seen: {kinds[:6]})")
    return out


def parent_of(model: nn.Module, path: str) -> tuple[nn.Module, str]:
    parent_path, _, leaf = path.rpartition(".")
    return model.get_submodule(parent_path) if parent_path else model, leaf
