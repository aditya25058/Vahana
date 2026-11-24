"""GPU twin of the CPU kernel's general expert path (M2): any activation id, optional biases.

dot32: y[t, r] = tree16(16 fp32 FMA lanes over k ascending), left in fp32 (no rounding), the same lanes and tree
as the CPU kernel and twinlin._lin. act: the kernel's act_h() element for element (bf16 rounding points, Vahana's
exp). down: the kernel's down_o() (bf16(dot), then + bias, then bf16).
"""
from __future__ import annotations

import torch
from ._triton import div_rn, tl, triton

from .hybrid import C_LN2HI, C_LN2LO, C_LOG2E, P0, P1, P2, P3, P4, P5, _dexp
from .twinlin import _tree16


@triton.jit
def _dot32(x, w, y, K, R, BR: tl.constexpr):
    rb = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    rows = rb * BR + tl.arange(0, BR).to(tl.int64)
    ln = tl.arange(0, 16)
    acc = tl.zeros((BR, 16), dtype=tl.float32)
    xb = x + t * K
    for k0 in range(0, K, 16):
        xv = tl.load(xb + k0 + ln).to(tl.float32)
        wv = tl.load(w + rows[:, None] * K + k0 + ln[None, :]).to(tl.float32)
        acc = tl.fma(wv, xv[None, :], acc)
    tl.store(y + t * R + rows, _tree16(acc, BR))


@triton.jit
def _act(gd, ud, bg, bu, h, n, I, HAS_B: tl.constexpr, ACT: tl.constexpr, alpha, limit, BLK: tl.constexpr,
         LOG2E: tl.constexpr, LN2HI: tl.constexpr, LN2LO: tl.constexpr, P0: tl.constexpr, P1: tl.constexpr,
         P2: tl.constexpr, P3: tl.constexpr, P4: tl.constexpr, P5: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLK + tl.arange(0, BLK).to(tl.int64)
    m = i < n * I
    gb = tl.load(gd + i, mask=m, other=0.0).to(tl.bfloat16).to(tl.float32)
    ub = tl.load(ud + i, mask=m, other=0.0).to(tl.bfloat16).to(tl.float32)
    if HAS_B:
        r = i % I
        gb = (gb + tl.load(bg + r, mask=m, other=0.0)).to(tl.bfloat16).to(tl.float32)
        ub = (ub + tl.load(bu + r, mask=m, other=0.0)).to(tl.bfloat16).to(tl.float32)
    if ACT == 1:
        gb = tl.minimum(gb, limit)
        ub = tl.minimum(tl.maximum(ub, -limit), limit)
        z = (gb * alpha).to(tl.bfloat16).to(tl.float32)
        ex = _dexp(-z, LOG2E, LN2HI, LN2LO, P0, P1, P2, P3, P4, P5)
        sg = div_rn(1.0, 1.0 + ex).to(tl.bfloat16).to(tl.float32)
        glu = (gb * sg).to(tl.bfloat16).to(tl.float32)
        out = (ub + 1.0).to(tl.bfloat16).to(tl.float32) * glu
    else:
        ex = _dexp(-gb, LOG2E, LN2HI, LN2LO, P0, P1, P2, P3, P4, P5)
        s = div_rn(gb, 1.0 + ex).to(tl.bfloat16).to(tl.float32)
        out = s * ub
    tl.store(h + i, out.to(tl.bfloat16), mask=m)


def dot32(x: torch.Tensor, W: torch.Tensor, BR: int = 16) -> torch.Tensor:
    """x [n, K] bf16, W [R, K] bf16 -> fp32 [n, R] twin dot products (unrounded)."""
    x = x.contiguous()
    n, K = x.shape
    R = W.shape[0]
    assert K % 16 == 0 and R % BR == 0
    y = torch.empty((n, R), dtype=torch.float32, device=x.device)
    _dot32[(R // BR, n)](x, W, y, K, R, BR=BR, num_warps=4)
    return y


def expert(x, G, U, D, act_id: int, alpha: float, limit: float, bg=None, bu=None, bd=None) -> torch.Tensor:
    """One expert, the CPU kernel's general path on the GPU. x [n, H] bf16; G/U [I, H]; D [H, I]; biases fp32."""
    n = x.shape[0]
    I = G.shape[0]
    gd, ud = dot32(x, G), dot32(x, U)
    h = torch.empty((n, I), dtype=torch.bfloat16, device=x.device)
    has_b = bg is not None
    BLK = 1024
    _act[(triton.cdiv(n * I, BLK),)](gd, ud, bg if has_b else gd, bu if has_b else ud, h, n, I, HAS_B=has_b,
                                     ACT=act_id, alpha=alpha, limit=limit, BLK=BLK, LOG2E=C_LOG2E, LN2HI=C_LN2HI,
                                     LN2LO=C_LN2LO, P0=P0, P1=P1, P2=P2, P3=P3, P4=P4, P5=P5)
    y = dot32(h, D)
    if bd is None:
        return y.to(torch.bfloat16)
    return (y.to(torch.bfloat16).float() + bd).to(torch.bfloat16)
