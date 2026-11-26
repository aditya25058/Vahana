"""Phase 30: device-invariant linears and budgeted CPU placement (the small-box path).

VAHANA_TWIN=1 (on top of VAHANA_INVARIANT=1): every bf16 nn.Linear runs the twin arithmetic, i.e.
per output row 16 IEEE-fma lanes over k ascending, the fixed lane tree, and bf16 RNE. On the GPU this is a
Triton kernel (near cuBLAS speed, Phase 30a). On the CPU it is the C++ kernel, run on the expert worker
thread. Both give the same bits, so a weight can live on either device without changing any output.
Shared experts use the expert twin (GPU twin_one / CPU expert kernel).

VAHANA_CPU_PLACE=<regex>: resident (non-routed) weights whose key matches are kept in host memory
(in place, no copy) and computed by the CPU. Norms, the router gate and MLA attention stay on the GPU.
VAHANA_GPU_BUDGET_GB=<g>: caps the CUDA caching allocator at g GB (emulates a small card).
"""
from __future__ import annotations

import os
import re

import torch
import torch.nn.functional as F
from ._triton import div_rn, sqrt_rn, tl, triton
from torch import nn

from . import hybrid as _hy


def enabled() -> bool:
    return os.environ.get("VAHANA_TWIN", "0") == "1"


def on_cpu(key: str) -> bool:
    v = os.environ.get("VAHANA_CPU_PLACE", "")
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


# v0.3-d: several tokens per program. MT separate accumulators; each weight tile is loaded once and used for MT
# tokens. Each token's arithmetic is _lin's (16 fma lanes, k ascending, tree16, bf16 RNE), so the bits are the same.
@triton.jit
def _acc_t(acc, wv, x, t, n, K, kk, ln):
    xv = tl.load(x + t * K + kk + ln, mask=(ln < 16) & (t < n), other=0.0).to(tl.float32)
    return tl.fma(wv, xv[None, :], acc)


@triton.jit
def _st_t(y, t, n, R, rows, acc, BR: tl.constexpr):
    tl.store(y + t * R + rows, _tree16(acc, BR).to(tl.bfloat16), mask=(rows >= 0) & (t < n))


@triton.jit
def _linm(x, w, y, n, K, R, BR: tl.constexpr, MT: tl.constexpr, U: tl.constexpr):
    rb = tl.program_id(0).to(tl.int64)
    t0 = tl.program_id(1).to(tl.int64) * MT
    rows = rb * BR + tl.arange(0, BR).to(tl.int64)
    ln = tl.arange(0, 16)
    a0 = tl.zeros((BR, 16), dtype=tl.float32)
    a1 = tl.zeros((BR, 16), dtype=tl.float32)
    a2 = tl.zeros((BR, 16), dtype=tl.float32)
    a3 = tl.zeros((BR, 16), dtype=tl.float32)
    a4 = tl.zeros((BR, 16), dtype=tl.float32)
    a5 = tl.zeros((BR, 16), dtype=tl.float32)
    a6 = tl.zeros((BR, 16), dtype=tl.float32)
    a7 = tl.zeros((BR, 16), dtype=tl.float32)
    for k0 in range(0, K, 16 * U):
        for u in tl.static_range(U):
            kk = k0 + 16 * u
            wv = tl.load(w + rows[:, None] * K + kk + ln[None, :]).to(tl.float32)
            a0 = _acc_t(a0, wv, x, t0, n, K, kk, ln)
            if MT >= 2:
                a1 = _acc_t(a1, wv, x, t0 + 1, n, K, kk, ln)
            if MT >= 4:
                a2 = _acc_t(a2, wv, x, t0 + 2, n, K, kk, ln)
                a3 = _acc_t(a3, wv, x, t0 + 3, n, K, kk, ln)
            if MT >= 8:
                a4 = _acc_t(a4, wv, x, t0 + 4, n, K, kk, ln)
                a5 = _acc_t(a5, wv, x, t0 + 5, n, K, kk, ln)
                a6 = _acc_t(a6, wv, x, t0 + 6, n, K, kk, ln)
                a7 = _acc_t(a7, wv, x, t0 + 7, n, K, kk, ln)
    _st_t(y, t0, n, R, rows, a0, BR)
    if MT >= 2:
        _st_t(y, t0 + 1, n, R, rows, a1, BR)
    if MT >= 4:
        _st_t(y, t0 + 2, n, R, rows, a2, BR)
        _st_t(y, t0 + 3, n, R, rows, a3, BR)
    if MT >= 8:
        _st_t(y, t0 + 4, n, R, rows, a4, BR)
        _st_t(y, t0 + 5, n, R, rows, a5, BR)
        _st_t(y, t0 + 6, n, R, rows, a6, BR)
        _st_t(y, t0 + 7, n, R, rows, a7, BR)


# v1.3: prefill tiling. A program computes BM tokens x BN rows, each output with _lin's arithmetic (16 fma lanes, k
# ascending, tree16, bf16 RNE): activations and weights are reused across the tile, the bits are unchanged.
@triton.jit
def _lint(x, w, y, n, K, R, BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BN + tl.arange(0, BN).to(tl.int64)
    tok = tl.program_id(1).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    ok = tok < n
    ln = tl.arange(0, 16)
    acc = tl.zeros((BM, BN, 16), dtype=tl.float32)
    for k0 in range(0, K, 16):
        xv = tl.load(x + tok[:, None] * K + k0 + ln[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        wv = tl.load(w + rows[:, None] * K + k0 + ln[None, :]).to(tl.float32)
        acc = tl.fma(wv[None, :, :], xv[:, None, :], acc)
    a = tl.sum(tl.reshape(acc, (BM * BN, 2, 8)), axis=1)
    a = tl.sum(tl.reshape(a, (BM * BN, 2, 4)), axis=1)
    a = tl.sum(tl.reshape(a, (BM * BN, 2, 2)), axis=1)
    a = tl.sum(tl.reshape(a, (BM * BN, 2, 1)), axis=1)
    out = tl.reshape(a, (BM, BN)).to(tl.bfloat16)
    tl.store(y + tok[:, None] * R + rows[None, :], out, mask=ok[:, None])


TILE_MIN = int(os.environ.get("VAHANA_TILE_MIN", "64"))      # tokens from which linears use _lint (speed only)


def _cfg_m(n: int, R: int, K: int, MT: int | None = None):
    """Multi-token launch shape (v03d_multilin bench); every choice gives the same bits."""
    br, U, _ = _cfg(R, K)
    if MT is None:
        MT = 8 if (n >= 32 or (R >= 8192 and n >= 4)) else 2
    BR = br
    if MT == 8:
        while BR * 2 <= min(32, br * 4) and R % (BR * 2) == 0:
            BR *= 2
    return MT, BR, U, (1 if MT == 8 else 4)


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


def gpu_linear(x2: torch.Tensor, W: torch.Tensor, MT: int | None = None) -> torch.Tensor:
    n, K = x2.shape
    R = W.shape[0]
    y = torch.empty((n, R), dtype=torch.bfloat16, device=x2.device)
    if n == 1 and MT is None or MT == 1:
        BR, U, nw = _cfg(R, K)
        _lin[(R // BR, n)](x2, W, y, K, R, BR=BR, U=U, num_warps=nw)
    elif MT is None and n >= TILE_MIN and R % 16 == 0 and K % 16 == 0:
        _lint[(R // 16, triton.cdiv(n, 32))](x2, W, y, n, K, R, BM=32, BN=16, num_warps=2)   # laptop sweep
    elif n:
        MT, BR, U, nw = _cfg_m(n, R, K, MT)
        _linm[(R // BR, triton.cdiv(n, MT))](x2, W, y, n, K, R, BR=BR, MT=MT, U=U, num_warps=nw)
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
                    and os.environ.get("VAHANA_NX12") == "1":
                pk = _hy.nx_pack(m.weight, os.environ.get("VAHANA_NUMA_PLACE") == "1")   # Phase 40
                _hy.ext().register_linear_nx(m.weight, pk)
                _NX_KEEP.append(pk)
            elif os.environ.get("VAHANA_NUMA_PLACE") == "1" and m.weight.is_contiguous() and "shared_experts" not in n:
                _hy.ext().place_rows(m.weight)              # Phase 40: pairs with VAHANA_LIN_NUMA=2
    # Rotary tables: every layer holds an identical cos/sin table for max_position_embeddings positions
    # (5 GB on V2). Keep the first VAHANA_ROPE_ROWS rows of one table and share it: each row depends only
    # on its own position, so every value the model reads is unchanged.
    rows = int(os.environ.get("VAHANA_ROPE_ROWS", "8192"))
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


# ---- P2: one-kernel RMSNorm in the twin definition (tmath): x*x, halving-tree sum over the padded row, / N,
# 1 / sqrt(var + eps), bf16(x * r), bf16(w * h). Used only where verified equal to the module run under TwinMode.
@triton.jit
def _halve(a, N: tl.constexpr):
    return tl.sum(tl.reshape(a, (2, N // 2)), axis=0)


@triton.jit
def _trms(x, w, y, N, sx, sy, eps, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    v = tl.load(x + r * sx + cols, mask=m, other=0.0).to(tl.float32)
    a = v * v
    if BLOCK >= 2:
        a = _halve(a, BLOCK)
    if BLOCK >= 4:
        a = _halve(a, BLOCK // 2)
    if BLOCK >= 8:
        a = _halve(a, BLOCK // 4)
    if BLOCK >= 16:
        a = _halve(a, BLOCK // 8)
    if BLOCK >= 32:
        a = _halve(a, BLOCK // 16)
    if BLOCK >= 64:
        a = _halve(a, BLOCK // 32)
    if BLOCK >= 128:
        a = _halve(a, BLOCK // 64)
    if BLOCK >= 256:
        a = _halve(a, BLOCK // 128)
    if BLOCK >= 512:
        a = _halve(a, BLOCK // 256)
    if BLOCK >= 1024:
        a = _halve(a, BLOCK // 512)
    if BLOCK >= 2048:
        a = _halve(a, BLOCK // 1024)
    if BLOCK >= 4096:
        a = _halve(a, BLOCK // 2048)
    if BLOCK >= 8192:
        a = _halve(a, BLOCK // 4096)
    if BLOCK >= 16384:
        a = _halve(a, BLOCK // 8192)
    if BLOCK >= 32768:
        a = _halve(a, BLOCK // 16384)
    s = tl.sum(a, axis=0)                                  # one slot left (>= 0)
    var = div_rn(s, N.to(tl.float32))
    rs = div_rn(1.0, sqrt_rn(var + eps))
    h = (v * rs).to(tl.bfloat16).to(tl.float32)
    ww = tl.load(w + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(y + r * sy + cols, (ww * h).to(tl.bfloat16), mask=m)


def twin_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """bf16 x and weight (other dtypes: the caller keeps the TwinMode path)."""
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    y = torch.empty_like(x2)
    N = x2.shape[1]
    if x2.shape[0]:
        _trms[(x2.shape[0],)](x2, weight, y, N, x2.stride(0), y.stride(0), eps,
                              BLOCK=triton.next_power_of_2(N), num_warps=8, enable_fp_fusion=False)
    return y.view(shape)
