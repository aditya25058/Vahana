"""v1.3 tiled prefill kernels for expert layers.

A program computes a BM x BN tile: BM slots routed to one expert (consecutive among the slots sorted by expert) times
BN weight rows. Every output keeps the CPU kernel's arithmetic exactly: 16 fp32 fma lanes, lane l taking k = l, l + 16,
... ascending (for twin-q4: k0 + l then k0 + 16 + l per 32-block), then twinlin._tree16's pairing, bf16 roundings
at the same points, Vahana's exp for SiLU. Only the tiling differs from the decode kernels in cache.py (one slot, or
MT slots of one expert, by BR rows), so the bits are the same and the weights and activations are reused across a
BM x BN tile instead of per slot.
"""
from __future__ import annotations

import torch

from ._triton import div_rn, tl, triton
from .cache import _q4w
from .hybrid import C_LN2HI, C_LN2LO, C_LOG2E, P0, P1, P2, P3, P4, P5, _dexp


@triton.jit
def _tree3(acc, BM: tl.constexpr, BN: tl.constexpr):
    """twinlin._tree16 on a (BM, BN, 16) accumulator: lanes l and l + 8 first, then l and l + 4, ..."""
    a = tl.sum(tl.reshape(acc, (BM * BN, 2, 8)), axis=1)
    a = tl.sum(tl.reshape(a, (BM * BN, 2, 4)), axis=1)
    a = tl.sum(tl.reshape(a, (BM * BN, 2, 2)), axis=1)
    a = tl.sum(tl.reshape(a, (BM * BN, 2, 1)), axis=1)
    return tl.reshape(a, (BM, BN))


@triton.jit
def _tile(tile_e, tile_s, tile_n, order, K, BM: tl.constexpr):
    t = tl.program_id(1).to(tl.int64)
    c = tl.load(tile_e + t)
    m = tl.arange(0, BM)
    ok = m < tl.load(tile_n + t)
    sl = tl.load(order + tl.load(tile_s + t) + m, mask=ok, other=0)
    return c, ok, sl, sl // K


@triton.jit
def _silu_mul(ag, au, BM: tl.constexpr, BN: tl.constexpr, LOG2E: tl.constexpr, LN2HI: tl.constexpr,
              LN2LO: tl.constexpr, P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
              P4: tl.constexpr, P5: tl.constexpr):
    gb = _tree3(ag, BM, BN).to(tl.bfloat16).to(tl.float32)
    ub = _tree3(au, BM, BN).to(tl.bfloat16).to(tl.float32)
    ex = _dexp(-gb, LOG2E, LN2HI, LN2LO, P0, P1, P2, P3, P4, P5)
    s = div_rn(gb, 1.0 + ex).to(tl.bfloat16).to(tl.float32)
    return (s * ub).to(tl.bfloat16)


@triton.jit
def _pgate_up(x, order, tile_e, tile_s, tile_n, g, u, h, H, I, K, BM: tl.constexpr, BN: tl.constexpr,
              LOG2E: tl.constexpr, LN2HI: tl.constexpr, LN2LO: tl.constexpr, P0: tl.constexpr, P1: tl.constexpr,
              P2: tl.constexpr, P3: tl.constexpr, P4: tl.constexpr, P5: tl.constexpr):
    c, ok, sl, tok = _tile(tile_e, tile_s, tile_n, order, K, BM)
    if c >= 0:
        rows = tl.program_id(0).to(tl.int64) * BN + tl.arange(0, BN).to(tl.int64)
        ln = tl.arange(0, 16)
        ag = tl.zeros((BM, BN, 16), dtype=tl.float32)
        au = tl.zeros((BM, BN, 16), dtype=tl.float32)
        gb_ = g + c * I * H
        ub_ = u + c * I * H
        for k0 in range(0, H, 16):
            xv = tl.load(x + tok[:, None] * H + k0 + ln[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
            wg = tl.load(gb_ + rows[:, None] * H + k0 + ln[None, :]).to(tl.float32)
            wu = tl.load(ub_ + rows[:, None] * H + k0 + ln[None, :]).to(tl.float32)
            ag = tl.fma(wg[None, :, :], xv[:, None, :], ag)
            au = tl.fma(wu[None, :, :], xv[:, None, :], au)
        hv = _silu_mul(ag, au, BM, BN, LOG2E, LN2HI, LN2LO, P0, P1, P2, P3, P4, P5)
        tl.store(h + sl[:, None] * I + rows[None, :], hv, mask=ok[:, None])


@triton.jit
def _pdown(hh, order, tile_e, tile_s, tile_n, d, y, H, I, K, BM: tl.constexpr, BN: tl.constexpr):
    c, ok, sl, tok = _tile(tile_e, tile_s, tile_n, order, K, BM)
    if c >= 0:
        rows = tl.program_id(0).to(tl.int64) * BN + tl.arange(0, BN).to(tl.int64)
        ln = tl.arange(0, 16)
        acc = tl.zeros((BM, BN, 16), dtype=tl.float32)
        db = d + c * H * I
        for k0 in range(0, I, 16):
            hv = tl.load(hh + sl[:, None] * I + k0 + ln[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
            w = tl.load(db + rows[:, None] * I + k0 + ln[None, :]).to(tl.float32)
            acc = tl.fma(w[None, :, :], hv[:, None, :], acc)
        tl.store(y + sl[:, None] * H + rows[None, :], _tree3(acc, BM, BN).to(tl.bfloat16), mask=ok[:, None])


@triton.jit
def _pgate_up_q4(x, order, tile_e, tile_s, tile_n, g, u, h, H, I, K, RBH, BM: tl.constexpr, BN: tl.constexpr,
                 GS: tl.constexpr, LOG2E: tl.constexpr, LN2HI: tl.constexpr, LN2LO: tl.constexpr, P0: tl.constexpr,
                 P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr, P4: tl.constexpr, P5: tl.constexpr):
    c, ok, sl, tok = _tile(tile_e, tile_s, tile_n, order, K, BM)
    if c >= 0:
        rows = tl.program_id(0).to(tl.int64) * BN + tl.arange(0, BN).to(tl.int64)
        ln = tl.arange(0, 16)
        ag = tl.zeros((BM, BN, 16), dtype=tl.float32)
        au = tl.zeros((BM, BN, 16), dtype=tl.float32)
        gb_ = g + c * I * RBH
        ub_ = u + c * I * RBH
        for k0 in range(0, H, 32):
            xlo = tl.load(x + tok[:, None] * H + k0 + ln[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
            xhi = tl.load(x + tok[:, None] * H + k0 + 16 + ln[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
            glo, ghi = _q4w(gb_, rows, RBH, k0, H, GS)
            ulo, uhi = _q4w(ub_, rows, RBH, k0, H, GS)
            ag = tl.fma(glo[None, :, :], xlo[:, None, :], ag)
            au = tl.fma(ulo[None, :, :], xlo[:, None, :], au)
            ag = tl.fma(ghi[None, :, :], xhi[:, None, :], ag)
            au = tl.fma(uhi[None, :, :], xhi[:, None, :], au)
        hv = _silu_mul(ag, au, BM, BN, LOG2E, LN2HI, LN2LO, P0, P1, P2, P3, P4, P5)
        tl.store(h + sl[:, None] * I + rows[None, :], hv, mask=ok[:, None])


@triton.jit
def _pdown_q4(hh, order, tile_e, tile_s, tile_n, d, y, H, I, K, RBI, BM: tl.constexpr, BN: tl.constexpr,
              GS: tl.constexpr):
    c, ok, sl, tok = _tile(tile_e, tile_s, tile_n, order, K, BM)
    if c >= 0:
        rows = tl.program_id(0).to(tl.int64) * BN + tl.arange(0, BN).to(tl.int64)
        ln = tl.arange(0, 16)
        acc = tl.zeros((BM, BN, 16), dtype=tl.float32)
        db = d + c * H * RBI
        for k0 in range(0, I, 32):
            hlo = tl.load(hh + sl[:, None] * I + k0 + ln[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
            hhi = tl.load(hh + sl[:, None] * I + k0 + 16 + ln[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
            wlo, whi = _q4w(db, rows, RBI, k0, I, GS)
            acc = tl.fma(wlo[None, :, :], hlo[:, None, :], acc)
            acc = tl.fma(whi[None, :, :], hhi[:, None, :], acc)
        tl.store(y + sl[:, None] * H + rows[None, :], _tree3(acc, BM, BN).to(tl.bfloat16), mask=ok[:, None])


def tiles(ti, lut, BM):
    """Slots sorted by (cache index, slot) and the tile table, all on the GPU with no host sync. Tiles past the real
    count get expert -1, and their programs exit."""
    N, K = ti.shape
    NK = N * K
    C = lut.shape[0]
    cs = lut[ti.flatten()]
    key = torch.where(cs >= 0, cs, torch.full_like(cs, C))
    order = torch.sort(key, stable=True)[1]
    cnt = torch.zeros(C + 1, dtype=torch.int64, device=ti.device).scatter_add_(0, key, torch.ones_like(key))[:C]
    # (not torch.bincount: it syncs the host to size its output, which would stall the copy/compute pipeline)
    first = torch.cumsum(cnt, 0) - cnt
    per = (cnt + BM - 1) // BM
    cum = torch.cumsum(per, 0)
    T = NK // BM + C + 1
    t = torch.arange(T, device=ti.device)
    e = torch.searchsorted(cum, t, right=True)
    valid = e < C
    ec = e.clamp(max=C - 1)
    j = t - (cum[ec] - per[ec])                      # tile index within its expert
    tile_s = (first[ec] + j * BM).contiguous()
    tile_n = torch.clamp(cnt[ec] - j * BM, max=BM).contiguous()
    tile_e = torch.where(valid, ec, torch.full_like(ec, -1)).contiguous()
    return order.contiguous(), tile_e, tile_s, tile_n, T


BM_, BN_ = 16, 16


def slots(x, ti, lut, G, U, D, q4: bool, H: int, I: int, BM: int | None = None, BN: int | None = None,
          nw: int = 4):
    """cache.gpu_slots / gpu_slots_q4 for many tokens: y [N, K, H] bf16 for the slots whose expert is in lut (the
    other slots are left unwritten)."""
    BM = BM or BM_
    BN = BN or BN_
    N, K = ti.shape
    ti = ti.contiguous()
    order, te, ts, tn, T = tiles(ti, lut, BM)
    h = torch.empty((N * K, I), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((N, K, H), dtype=torch.bfloat16, device=x.device)
    c = dict(LOG2E=C_LOG2E, LN2HI=C_LN2HI, LN2LO=C_LN2LO, P0=P0, P1=P1, P2=P2, P3=P3, P4=P4, P5=P5)
    if q4:
        RBH, RBI = G.shape[2], D.shape[2]
        GS = 2 * H // (RBH - H // 2)
        _pgate_up_q4[(I // BN, T)](x, order, te, ts, tn, G, U, h, H, I, K, RBH, BM=BM, BN=BN, GS=GS, num_warps=nw,
                                   **c)
        _pdown_q4[(H // BN, T)](h, order, te, ts, tn, D, y, H, I, K, RBI, BM=BM, BN=BN, GS=GS, num_warps=nw)
    else:
        _pgate_up[(I // BN, T)](x, order, te, ts, tn, G, U, h, H, I, K, BM=BM, BN=BN, num_warps=nw, **c)
        _pdown[(H // BN, T)](h, order, te, ts, tn, D, y, H, I, K, BM=BM, BN=BN, num_warps=nw)
    return y
