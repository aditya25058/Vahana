"""Phase 30: device-invariant linears and budgeted CPU placement (the small-box path).

COLOSSUS_TWIN=1 (on top of COLOSSUS_INVARIANT=1): every bf16 nn.Linear runs the twin arithmetic, i.e.
per output row 16 IEEE-fma lanes over k ascending, the fixed lane tree, and bf16 RNE. On the GPU this is a
Triton kernel (near cuBLAS speed, Phase 30a). On the CPU it is the C++ kernel, run on the expert worker
thread. Both give the same bits, so a weight can live on either device without changing any output.
Shared experts use the expert twin (GPU twin_one / CPU expert kernel).

COLOSSUS_CPU_PLACE=<regex>: resident (non-routed) weights whose key matches are kept in host memory
(in place, no copy) and computed by the CPU. Norms, the router gate and MLA attention stay on the GPU.
COLOSSUS_GPU_BUDGET_GB=<g>: caps the CUDA caching allocator at g GB (emulates a small card).
"""
from __future__ import annotations

import os
import re

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

from . import hybrid as _hy


def enabled() -> bool:
    return os.environ.get("COLOSSUS_TWIN", "0") == "1"


def on_cpu(key: str) -> bool:
    v = os.environ.get("COLOSSUS_CPU_PLACE", "")
    return bool(v) and re.search(v, key) is not None


@triton.jit
def _tree16(acc, BR: tl.constexpr):
    a = tl.sum(tl.reshape(acc, (BR, 2, 8)), axis=1)
    a = tl.sum(tl.reshape(a, (BR, 2, 4)), axis=1)
    a = tl.sum(tl.reshape(a, (BR, 2, 2)), axis=1)
    a = tl.sum(tl.reshape(a, (BR, 2, 1)), axis=1)
    return tl.reshape(a, (BR,))


@triton.jit
def _lin(x, w, y, K, R, BR: tl.constexpr, U: tl.constexpr):
    rb = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    rows = rb * BR + tl.arange(0, BR).to(tl.int64)
    ln = tl.arange(0, 16)
    acc = tl.zeros((BR, 16), dtype=tl.float32)
    xb = x + t * K
    for k0 in range(0, K, 16 * U):
        for u in tl.static_range(U):
            kk = k0 + 16 * u
            xv = tl.load(xb + kk + ln).to(tl.float32)
            wv = tl.load(w + rows[:, None] * K + kk + ln[None, :]).to(tl.float32)
            acc = tl.fma(wv, xv[None, :], acc)
    tl.store(y + t * R + rows, _tree16(acc, BR).to(tl.bfloat16))


def _cfg(R: int, K: int):
    """Launch shape only (every choice gives the same bits; Phase 30a measured the fast ones)."""
    if K >= 4096:
        BR, nw = 2, 1
    elif K >= 1024:
        BR, nw = 8, 1
    else:
        BR, nw = 16, 2
    while R % BR:
        BR //= 2
    U = 4 if K % 64 == 0 else (2 if K % 32 == 0 else 1)
    return max(BR, 2), U, nw


def gpu_linear(x2: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    n, K = x2.shape
    R = W.shape[0]
    BR, U, nw = _cfg(R, K)
    y = torch.empty((n, R), dtype=torch.bfloat16, device=x2.device)
    _lin[(R // BR, n)](x2, W, y, K, R, BR=BR, U=U, num_warps=nw)
    return y


def cpu_linear(x2: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    n, K = x2.shape
    R = W.shape[0]
    xc = x2.to("cpu")
    outs = []
    for s in range(0, n, 16):
        xs = xc[s:s + 16].contiguous()
        ob = _hy.pinned("lin_out", (xs.shape[0], R), torch.bfloat16)
        _hy.ext().submit_linear(xs, W, ob)
        _hy.ext().wait_done()
        outs.append(ob.clone())
    return torch.cat(outs).to(x2.device)


_ORIG = None
_NX_KEEP = []


def _twin_forward(self, x):
    W = self.weight
    if W.dtype != torch.bfloat16 or x.dtype != torch.bfloat16 or self.bias is not None \
            or W.shape[1] % 16 or W.shape[0] % 4:
        return _ORIG(self, x)
    shp = x.shape
    x2 = x.reshape(-1, shp[-1]).contiguous()
    y = gpu_linear(x2, W) if W.is_cuda else cpu_linear(x2, W)
    return y.view(*shp[:-1], W.shape[0])


def install(model: nn.Module, device: torch.device) -> dict:
    """Patch nn.Linear (class level, after invariant.install) and a CPU-resident embedding."""
    global _ORIG
    if _ORIG is None:
        _ORIG = nn.Linear.forward
        nn.Linear.forward = _twin_forward
    emb = model.model.embed_tokens
    info = {"cpu_linears": [], "embedding_on_cpu": emb.weight.device.type == "cpu"}
    if info["embedding_on_cpu"]:
        def _emb(ids, _w=emb.weight, _d=device):
            return F.embedding(ids.to("cpu"), _w).to(_d)
        emb.forward = _emb
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.weight.device.type == "cpu":
            info["cpu_linears"].append(n)
            if "shared_experts" not in n and m.weight.is_contiguous() and m.weight.shape[1] % 64 == 0 \
                    and os.environ.get("COLOSSUS_NX12") == "1":
                pk = _hy.nx_pack(m.weight, os.environ.get("COLOSSUS_NUMA_PLACE") == "1")   # Phase 40
                _hy.ext().register_linear_nx(m.weight, pk)
                _NX_KEEP.append(pk)
            elif os.environ.get("COLOSSUS_NUMA_PLACE") == "1" and m.weight.is_contiguous() and "shared_experts" not in n:
                _hy.ext().place_rows(m.weight)              # Phase 40: pairs with COLOSSUS_LIN_NUMA=2
    # Rotary tables: every layer holds an identical cos/sin table for max_position_embeddings positions
    # (5 GB on V2). Keep the first COLOSSUS_ROPE_ROWS rows of one table and share it: each row depends only
    # on its own position, so every value the model reads is unchanged.
    rows = int(os.environ.get("COLOSSUS_ROPE_ROWS", "8192"))
    shared = None
    n_rope = 0
    for mod in model.modules():
        if hasattr(mod, "cos_cached") and hasattr(mod, "sin_cached") and hasattr(mod, "max_seq_len_cached"):
            if mod.cos_cached.shape[0] <= rows:
                continue
            if shared is None:
                shared = (mod.cos_cached[:rows].clone(), mod.sin_cached[:rows].clone())
            elif not (torch.equal(mod.cos_cached[:rows], shared[0]) and torch.equal(mod.sin_cached[:rows], shared[1])):
                raise RuntimeError("rotary tables differ between layers; not sharing")
            mod.register_buffer("cos_cached", shared[0], persistent=False)
            mod.register_buffer("sin_cached", shared[1], persistent=False)
            mod.max_seq_len_cached = rows
            n_rope += 1
    info["rope_tables_shared"] = n_rope
    torch.cuda.empty_cache()
    _hy.pool()        # main thread: 1 torch thread (the worker owns the CPU team)
    return info


def shared_on_cpu(wrapper) -> bool:
    se = getattr(wrapper, "shared_experts", None)
    return se is not None and se.gate_proj.weight.device.type == "cpu"


def shared_gpu(wrapper, x2: torch.Tensor) -> torch.Tensor:
    """Shared experts on the GPU in expert-twin arithmetic. x2 [N, H] -> [N, H]."""
    se = wrapper.shared_experts
    return _hy.twin_one(x2, se.gate_proj.weight, se.up_proj.weight, se.down_proj.weight)
