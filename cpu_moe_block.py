"""Phase 31: routed experts of fused-expert MoE blocks (transformers-5 Mixtral style) on the CPU, exactly.

COLOSSUS_FUSED_CPU_MOE=1 replaces each decoder layer's `mlp` (router + fused experts) with CPUExpertsMoE:
  routing: the model's own router on the GPU (unchanged);
  experts: COLOSSUS_EXPERTS=cpu (default) -> the C++ worker (moe_fused: group, twin-arithmetic expert MLPs,
           combine); weights are views of the host-resident checkpoint, never copied to the GPU.
           COLOSSUS_EXPERTS=gpu -> the same arithmetic on the GPU (Triton twin), weights copied per use:
           the reference for the placement-exactness test.
  combine (both): per token, experts in ascending id, acc = f32(y0)*f32(w0); acc = acc + f32(yj)*f32(wj)
           (no FMA), out = bf16(acc); routing weights are rounded to bf16 first (both paths).
"""
from __future__ import annotations

import os
import time

import torch
from torch import nn

from . import hybrid as _hy


class CPUExpertsMoE(nn.Module):
    def __init__(self, layer_idx, router, n_experts, keys, handles, device):
        super().__init__()
        self.layer_idx = layer_idx
        self.gate = router
        self.n_experts = n_experts
        self.keys = keys            # e -> (gate key, up key, down key)
        self.handles = handles
        self.device = device
        self.registered = False
        self.mode = os.environ.get("COLOSSUS_EXPERTS", "cpu")
        self.misses = 0
        self.hits = 0
        self.dma_bytes = 0
        self.copy_s = 0.0
        self.forward_s = 0.0
        self.cpu_ms = 0.0

    def _w(self, e):
        g, u, d = self.keys[e]
        return self.handles.get_tensor(g), self.handles.get_tensor(u), self.handles.get_tensor(d)

    def _register(self):
        G, U, D = [], [], []
        for e in range(self.n_experts):
            g, u, d = self._w(e)
            G.append(g)
            U.append(u)
            D.append(d)
        _hy.ext().register_layer(self.layer_idx, G, U, D)
        self.registered = True

    def forward(self, hidden_states):
        t_0 = time.perf_counter()
        shp = hidden_states.shape
        H = shp[-1]
        flat = hidden_states.reshape(-1, H)
        N = flat.shape[0]
        _, tw, ti = self.gate(flat)
        K = ti.shape[-1]
        tw = tw.to(torch.bfloat16)
        if self.mode == "cpu":
            if not self.registered:
                self._register()
            _hy.pool()
            xb = _hy.pinned("mx", (N, H), flat.dtype)
            tib = _hy.pinned("mti", (N, K), torch.int64)
            twb = _hy.pinned("mtw", (N, K), torch.bfloat16)
            ob = _hy.pinned("mout", (N, H), flat.dtype)
            xb.copy_(flat, non_blocking=True)
            tib.copy_(ti.reshape(N, K), non_blocking=True)
            twb.copy_(tw.reshape(N, K), non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            ev.synchronize()
            _hy.ext().submit_moe(self.layer_idx, xb, tib, twb, ob, int(os.environ.get("COLOSSUS_HYBRID_NUMA", "1")))
            _hy.ext().wait_done()
            self.cpu_ms += _hy.ext().kernel_ms()
            out = ob.to(self.device, non_blocking=True)
        else:
            TIl = ti.reshape(N, K).tolist()
            rows: dict[int, list[int]] = {}
            for b in range(N):
                for k in range(K):
                    rows.setdefault(TIl[b][k], []).append(b)
            ys = {}
            for e in sorted(rows):
                t0 = time.perf_counter()
                g, u, d = (t.to(self.device) for t in self._w(e))
                self.copy_s += time.perf_counter() - t0
                self.misses += 1
                self.dma_bytes += 3 * g.numel() * 2
                xe = flat.index_select(0, torch.tensor(rows[e], device=self.device))
                ys[e] = _hy.twin_one(xe, g, u, d)
            pos, cat = [], []
            for e in sorted(rows):
                for j, b in enumerate(rows[e]):
                    pos.append(b * K + TIl[b].index(e))
                cat.append(ys[e])
            new_x = torch.empty((N * K, H), dtype=flat.dtype, device=self.device)
            new_x[torch.tensor(pos, device=self.device)] = torch.cat(cat, 0)
            TI = ti.reshape(N, K)
            order = TI.argsort(dim=1)
            v = torch.take_along_dim(new_x.view(N, K, H), order.unsqueeze(-1), dim=1)
            wv = torch.take_along_dim(tw.reshape(N, K), order, dim=1).float()
            acc = v[:, 0].float() * wv[:, 0:1]
            for j in range(1, K):
                acc = acc + v[:, j].float() * wv[:, j:j + 1]
            out = acc.to(flat.dtype)
        self.forward_s += time.perf_counter() - t_0
        return out.view(*shp)


def install_cpu_moe(model, handles, device) -> list:
    layers = list(model.model.layers)
    wrappers = []
    for li, layer in enumerate(layers):
        mlp = layer.mlp
        n = mlp.experts.num_experts if hasattr(mlp, "experts") else mlp.gate.num_experts
        base = f"model.layers.{li}.block_sparse_moe.experts"
        keys = {e: (f"{base}.{e}.w1.weight", f"{base}.{e}.w3.weight", f"{base}.{e}.w2.weight") for e in range(n)}
        w = CPUExpertsMoE(li, mlp.gate, n, keys, handles, device)
        layer.mlp = w
        wrappers.append(w)
    return wrappers
