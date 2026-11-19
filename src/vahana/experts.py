"""VahanaExperts: the one module Vahana puts into any supported MoE model (replaces `block.experts`).

Same call and return as the transformers experts module: forward(hidden_states [N, H], top_k_index [N, K],
top_k_weights [N, K]) -> [N, H], the weighted sum of the selected experts. Vahana's arithmetic (identical on
every device):
  per expert   gate, up = twin dot products (16 fp32 FMA lanes, k ascending, fixed tree) -> bf16;
               h = bf16(bf16(g / (1 + exp(-g))) * u) with Vahana's exp; y = twin dot(down, h) -> bf16
  combine      per token, selected experts in ascending id: acc = y0 * w0; acc = acc + yj * wj (fp32, no FMA);
               out = bf16(acc); routing weights are rounded to bf16 first.
VahanaExperts computes it with the CPU kernel (weights in CPU RAM, NX12 for decode); TwinGPUExperts computes it
on the GPU and is the reference: their outputs must be bit-identical.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import hybrid as _hy
from .spec import ExpertsSpec, inspect, parent_of


def _split(gate_up: torch.Tensor, down: torch.Tensor, I: int):
    """fused_concat [E, 2I, H] / [E, H, I] -> per-expert contiguous gate [I, H], up [I, H], down [H, I]."""
    g = [gate_up[e, :I].contiguous() for e in range(gate_up.shape[0])]
    u = [gate_up[e, I:].contiguous() for e in range(gate_up.shape[0])]
    d = [down[e].contiguous() for e in range(down.shape[0])]
    return g, u, d


class VahanaExperts(nn.Module):
    """Routed experts in CPU RAM, computed by the CPU twin kernel through the persistent worker."""

    def __init__(self, spec: ExpertsSpec, gate_up: torch.Tensor, down: torch.Tensor, numa: int = 2,
                 nx: bool = True):
        super().__init__()
        self.spec, self.numa = spec, numa
        cpu = lambda t: t.detach().to("cpu", torch.bfloat16).contiguous()
        self.g, self.u, self.d = _split(cpu(gate_up), cpu(down), spec.I)
        ext = _hy.ext()
        ext.register_layer(spec.slot, self.g, self.u, self.d)
        self.nx = nx and spec.H % 64 == 0 and spec.I % 64 == 0
        if self.nx:
            packs = []
            for e in range(spec.E):
                for W in (self.g[e], self.u[e], self.d[e]):
                    packs += list(ext.encode_nx(W))
            self._packs = packs
            ext.register_layer_nx(spec.slot, packs)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        dev, dt = hidden_states.device, hidden_states.dtype
        x = hidden_states.detach().to("cpu", torch.bfloat16).contiguous()
        ti = top_k_index.detach().to("cpu", torch.int64).contiguous()
        tw = top_k_weights.detach().to(torch.bfloat16).to("cpu").contiguous()
        out = torch.empty(x.shape, dtype=torch.bfloat16)
        ext = _hy.ext()
        ext.submit_moe(self.spec.slot, x, ti, tw, out, self.numa)
        ext.wait_done()
        return out.to(dev, dt)


class TwinGPUExperts(nn.Module):
    """The same arithmetic on the GPU (hybrid.twin_one per expert, combine in torch). Reference only."""

    def __init__(self, spec: ExpertsSpec, gate_up: torch.Tensor, down: torch.Tensor):
        super().__init__()
        self.spec = spec
        dev = gate_up.device
        g, u, d = _split(gate_up.detach().to(torch.bfloat16), down.detach().to(torch.bfloat16), spec.I)
        self.g, self.u, self.d = [t.to(dev) for t in g], [t.to(dev) for t in u], [t.to(dev) for t in d]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        x = hidden_states.to(torch.bfloat16).contiguous()
        N, K = top_k_index.shape
        w = top_k_weights.to(torch.bfloat16).float()
        y = {}
        for e in torch.unique(top_k_index).tolist():
            rows = (top_k_index == e).any(dim=1).nonzero().flatten()
            y[e] = (rows, _hy.twin_one(x[rows], self.g[e], self.u[e], self.d[e]))
        out = torch.empty_like(x)
        for t in range(N):
            ids = top_k_index[t].tolist()
            order = sorted(range(K), key=lambda k: ids[k])
            acc = None
            for q, k in enumerate(order):
                rows, ye = y[ids[k]]
                yt = ye[(rows == t).nonzero()[0, 0]].float()
                p = yt * w[t, k]
                acc = p if q == 0 else acc + p
            out[t] = acc.to(torch.bfloat16)
        return out.to(hidden_states.dtype)


def install(model: nn.Module, mode: str = "cpu", numa: int = 2, nx: bool = True) -> list[ExpertsSpec]:
    """Replace every experts module of `model` (mode "cpu": VahanaExperts, "gpu_twin": TwinGPUExperts).
    The original modules' gate_up_proj / down_proj are the weight source."""
    specs = inspect(model)
    for s in specs:
        parent, leaf = parent_of(model, s.path)
        old = getattr(parent, leaf)
        gu, dn = old.gate_up_proj, old.down_proj
        new = VahanaExperts(s, gu, dn, numa=numa, nx=nx) if mode == "cpu" else TwinGPUExperts(s, gu, dn)
        setattr(parent, leaf, new)
    return specs
