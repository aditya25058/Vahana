"""VahanaExperts: the one module Vahana puts into any supported MoE model (replaces `block.experts`).

Same call and return as the transformers experts module: forward(hidden_states [N, H], indices [N, K],
weights [N, K]) -> [N, H], the weighted sum of the selected experts. Vahana's arithmetic (identical on every
device):
  per expert   gate, up = twin dot products (16 fp32 FMA lanes, k ascending, fixed tree) -> bf16 (+ bias -> bf16);
               activation with fixed bf16 rounding points and Vahana's exp (SiLU-GLU, or gpt-oss clamped SwiGLU);
               y = twin dot(down, h) -> bf16 (+ bias -> bf16)
  combine      per token, selected experts in ascending id: acc = y0 * w0; acc = acc + yj * wj (fp32, no FMA);
               out = bf16(acc). Routing weights are used as given (fp32 or bf16), never rounded.
VahanaExperts computes it with the CPU kernel (weights in CPU RAM, NX12 for decode); TwinGPUExperts computes it
on the GPU and is the reference: their outputs must be bit-identical.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import hybrid as _hy
from .modelspec import ACT_IDS, ExpertsSpec, inspect, parent_of

CHUNK = 16          # tokens per CPU job: the kernel's batched path takes at most 16 rows per expert


def expert_weights(spec: ExpertsSpec, ex: nn.Module):
    """Per-expert row-major gate [I, H], up [I, H], down [H, I] (+ fp32 biases or None) from the HF module.
    Pure reordering; values unchanged."""
    I = spec.I
    gu, dn = ex.gate_up_proj.detach(), ex.down_proj.detach()
    if spec.layout == "fused_concat":
        g = [gu[e, :I].contiguous() for e in range(spec.E)]
        u = [gu[e, I:].contiguous() for e in range(spec.E)]
        d = [dn[e].contiguous() for e in range(spec.E)]
        return g, u, d, None, None, None
    if spec.layout == "fused_interleaved_t":        # x @ W, W [H, 2I], gate = cols 0::2, up = 1::2
        g = [gu[e, :, 0::2].t().contiguous() for e in range(spec.E)]
        u = [gu[e, :, 1::2].t().contiguous() for e in range(spec.E)]
        d = [dn[e].t().contiguous() for e in range(spec.E)]
        gb, db = ex.gate_up_proj_bias.detach(), ex.down_proj_bias.detach()
        bg = [gb[e, 0::2].float().contiguous() for e in range(spec.E)]
        bu = [gb[e, 1::2].float().contiguous() for e in range(spec.E)]
        bd = [db[e].float().contiguous() for e in range(spec.E)]
        return g, u, d, bg, bu, bd
    raise ValueError(spec.layout)


def _weights_arg(w: torch.Tensor) -> torch.Tensor:
    """Routing weights as given: fp32 stays fp32, bf16 stays bf16, anything else converts exactly to fp32."""
    return w.detach() if w.dtype in (torch.float32, torch.bfloat16) else w.detach().float()


class VahanaExperts(nn.Module):
    """Routed experts in CPU RAM, computed by the CPU twin kernel through the persistent worker."""

    def __init__(self, spec: ExpertsSpec, ex: nn.Module | None = None, numa: int = 2, nx: bool = True,
                 weights=None, place: bool = False, shared=None, margin_gb: float = 64.0):
        """Weights from the HF experts module `ex`, or `weights` = (g, u, d, bg, bu, bd) lists (e.g. checkpoint
        views). place: NX12 buffers split by row half across the two sockets. shared: (g, u, d) of a shared
        expert folded into this layer's CPU job. NX12 is skipped (raw BF16 kept) when it would cut MemAvailable
        below margin_gb + 8."""
        super().__init__()
        self.spec, self.numa = spec, numa
        cpu = lambda ts: [t if (t.device.type == "cpu" and t.dtype in (torch.bfloat16, torch.uint8) and t.is_contiguous())
                          else t.to("cpu").contiguous() if t.dtype == torch.uint8      # v1.2: twin-q4 records
                          else t.to("cpu", torch.bfloat16).contiguous() for t in ts]
        g, u, d, bg, bu, bd = weights if weights is not None else expert_weights(spec, ex)
        self.g, self.u, self.d = cpu(g), cpu(u), cpu(d)
        k = _hy.ext()
        k.register_layer(spec.slot, self.g, self.u, self.d)
        if spec.act != "silu_glu" or spec.bias:
            f = lambda ts: [t.to("cpu", torch.float32).contiguous() for t in ts] if ts is not None else []
            self._b = (f(bg), f(bu), f(bd))
            k.register_layer_act(spec.slot, ACT_IDS[spec.act], spec.alpha, spec.limit, *self._b)
        self.shared = None
        if shared is not None:
            self.shared = cpu(list(shared))
            k.register_shared(spec.slot, *self.shared)
        mats = [w for trip in zip(self.g, self.u, self.d) for w in trip] + (self.shared or [])
        need_gb = sum(w.numel() * 2 for w in mats) * 0.76 / 2**30
        nx_fast = getattr(k, "nx_fast", lambda: True)()      # portable builds: raw BF16 is faster than NX12
        self.q4 = self.g[0].dtype == torch.uint8           # v1.2: twin-q4 experts (no NX12 for them)
        self.nx = nx and nx_fast and not self.q4 and spec.H % 64 == 0 and spec.I % 64 == 0 and \
            _hy.mem_available_gb() - need_gb >= margin_gb + 8
        if self.nx:
            pk = lambda W: _hy.nx_pack(W, place) if place else list(k.encode_nx(W))
            self._packs = [t for e in range(spec.E) for W in (self.g[e], self.u[e], self.d[e]) for t in pk(W)]
            k.register_layer_nx(spec.slot, self._packs)
            if self.shared is not None:
                self._spacks = [t for W in self.shared for t in pk(W)]
                k.register_shared_nx(spec.slot, self._spacks)
        from . import prefill
        prefill.register(self)                           # v1.3: the read-ahead plans layers in slot order

    def forward(self, hidden_states, top_k_index, top_k_weights):
        from . import prefill, runtime
        if prefill.wanted(hidden_states, self):          # v1.3: long prompts run these experts on the GPU
            return prefill.run(self, hidden_states, top_k_index, top_k_weights).to(hidden_states.dtype)
        out = runtime.moe(self.spec.slot, hidden_states, top_k_index, top_k_weights, self.numa)
        return out.to(hidden_states.dtype)


class TwinGPUExperts(nn.Module):
    """The same arithmetic on the GPU (per expert: hybrid.twin_one or twin_act.expert; combine in torch)."""

    def __init__(self, spec: ExpertsSpec, ex: nn.Module | None = None, weights=None, device=None):
        super().__init__()
        self.spec = spec
        dev = device if device is not None else ex.gate_up_proj.device
        g, u, d, bg, bu, bd = weights if weights is not None else expert_weights(spec, ex)
        to = lambda ts, dt: [t.to(dev, dt).contiguous() for t in ts] if ts is not None else None
        self.g, self.u, self.d = to(g, torch.bfloat16), to(u, torch.bfloat16), to(d, torch.bfloat16)
        self.bg, self.bu, self.bd = to(bg, torch.float32), to(bu, torch.float32), to(bd, torch.float32)
        self.general = spec.act != "silu_glu" or spec.bias

    def _one(self, x, e):
        if not self.general:
            return _hy.twin_one(x, self.g[e], self.u[e], self.d[e])
        from . import twin_act
        b = (self.bg[e], self.bu[e], self.bd[e]) if self.bg is not None else (None, None, None)
        return twin_act.expert(x, self.g[e], self.u[e], self.d[e], ACT_IDS[self.spec.act], self.spec.alpha,
                               self.spec.limit, *b)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        x = hidden_states.to(torch.bfloat16).contiguous()
        N, K = top_k_index.shape
        w = _weights_arg(top_k_weights).float()
        y = {}
        for e in torch.unique(top_k_index).tolist():
            rows = (top_k_index == e).any(dim=1).nonzero().flatten()
            y[e] = (rows, self._one(x[rows], e))
        out = torch.empty_like(x)
        for t in range(N):
            ids = top_k_index[t].tolist()
            order = sorted(range(K), key=lambda k: ids[k])
            acc = None
            for q, k in enumerate(order):
                rows, ye = y[ids[k]]
                p = ye[(rows == t).nonzero()[0, 0]].float() * w[t, k]
                acc = p if q == 0 else acc + p
            out[t] = acc.to(torch.bfloat16)
        return out.to(hidden_states.dtype)


def install(model: nn.Module, mode: str = "cpu", numa: int = 2, nx: bool = True) -> list[ExpertsSpec]:
    """Replace every experts module of `model` (mode "cpu": VahanaExperts, "gpu_twin": TwinGPUExperts).
    The original modules are the weight source."""
    specs = inspect(model)
    for s in specs:
        parent, leaf = parent_of(model, s.path)
        old = getattr(parent, leaf)
        setattr(parent, leaf, VahanaExperts(s, old, numa=numa, nx=nx) if mode == "cpu" else TwinGPUExperts(s, old))
    return specs
