"""v0.3: GPU expert cache. The most-used experts of each layer live in spare GPU memory; the rest stay on the CPU.

Exact by construction:
  - a cached expert runs on the GPU with the CPU kernel's twin arithmetic (16-lane fma dots, tree16, bf16 rounding at
    the same points, Vahana's exp; the same ops as hybrid._gate_up / _down, here with the expert picked per slot);
  - the other slots run on the CPU (csrc moe_slots: per-slot expert outputs, the same per-expert arithmetic);
  - all K slot outputs are combined, by the CPU kernel itself (csrc moe_cached), in ascending expert id with the
    same operations as its fused path:
        acc = f32(y_0) * f32(w_0);  acc = acc + f32(y_j) * f32(w_j);  out = bf16(acc)
So which experts are cached, and how many, only changes speed, never a bit.
Supported: SiLU-GLU experts without biases and without a folded shared expert; other layers keep the CPU path.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ._triton import div_rn, tl, triton
from .hybrid import C_LN2HI, C_LN2LO, C_LOG2E, P0, P1, P2, P3, P4, P5, _dexp
from .twinlin import _tree16


@triton.jit
def _cgate_up(x, ti, lut, g, u, h, H, I, K, BR: tl.constexpr, LOG2E: tl.constexpr, LN2HI: tl.constexpr,
              LN2LO: tl.constexpr, P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
              P4: tl.constexpr, P5: tl.constexpr):
    rb = tl.program_id(0).to(tl.int64)
    sl = tl.program_id(1).to(tl.int64)                 # slot = token * K + k
    t = sl // K
    c = tl.load(lut + tl.load(ti + sl)).to(tl.int64)   # cache index of this slot's expert, -1 if not cached
    cs = tl.maximum(c, 0)
    rows = rb * BR + tl.arange(0, BR).to(tl.int64)
    ln = tl.arange(0, 16)
    ag = tl.zeros((BR, 16), dtype=tl.float32)
    au = tl.zeros((BR, 16), dtype=tl.float32)
    xb = x + t * H
    gb_ = g + cs * I * H
    ub_ = u + cs * I * H
    for k0 in range(0, H, 16):
        xv = tl.load(xb + k0 + ln).to(tl.float32)
        wg = tl.load(gb_ + rows[:, None] * H + k0 + ln[None, :]).to(tl.float32)
        wu = tl.load(ub_ + rows[:, None] * H + k0 + ln[None, :]).to(tl.float32)
        ag = tl.fma(wg, xv[None, :], ag)
        au = tl.fma(wu, xv[None, :], au)
    gb = _tree16(ag, BR).to(tl.bfloat16).to(tl.float32)
    ub = _tree16(au, BR).to(tl.bfloat16).to(tl.float32)
    ex = _dexp(-gb, LOG2E, LN2HI, LN2LO, P0, P1, P2, P3, P4, P5)
    s = div_rn(gb, 1.0 + ex).to(tl.bfloat16).to(tl.float32)
    tl.store(h + sl * I + rows, (s * ub).to(tl.bfloat16))


@triton.jit
def _cdown(hh, ti, lut, d, y, H, I, BR: tl.constexpr):
    rb = tl.program_id(0).to(tl.int64)
    sl = tl.program_id(1).to(tl.int64)
    c = tl.load(lut + tl.load(ti + sl)).to(tl.int64)
    cs = tl.maximum(c, 0)
    rows = rb * BR + tl.arange(0, BR).to(tl.int64)
    ln = tl.arange(0, 16)
    acc = tl.zeros((BR, 16), dtype=tl.float32)
    hb = hh + sl * I
    db = d + cs * H * I
    for k0 in range(0, I, 16):
        hv = tl.load(hb + k0 + ln).to(tl.float32)
        w = tl.load(db + rows[:, None] * I + k0 + ln[None, :]).to(tl.float32)
        acc = tl.fma(w, hv[None, :], acc)
    out = _tree16(acc, BR).to(tl.bfloat16)
    out = tl.where(c >= 0, out, tl.zeros_like(out))    # not cached: zeros (the CPU slot output is used instead)
    tl.store(y + sl * H + rows, out)


# v0.3-c grouped kernels: for several tokens, slots that use the same cached expert share one load of each weight
# tile. A slot's arithmetic is unchanged (the same fma chain per lane, k ascending, tree16, the same roundings): a
# group only stacks MT independent accumulators, so every output bit equals the per-slot kernels'.
@triton.jit
def _cgroup(ti, lut, order, rpos, NK, BLOCK: tl.constexpr):
    """Per slot p: its position in the slots sorted by (cache index, slot), and its rank within its expert's run."""
    p = tl.program_id(0).to(tl.int64)
    cp = tl.load(lut + tl.load(ti + p))
    less = tl.zeros((BLOCK,), dtype=tl.int32)
    same = tl.zeros((BLOCK,), dtype=tl.int32)
    for q0 in range(0, NK, BLOCK):
        q = q0 + tl.arange(0, BLOCK).to(tl.int64)
        mq = q < NK
        cq = tl.load(lut + tl.load(ti + q, mask=mq, other=0), mask=mq, other=0)
        before = (q < p) & mq
        less += ((mq & (cq < cp)) | ((cq == cp) & before)).to(tl.int32)
        same += ((cq == cp) & before).to(tl.int32)
    tl.store(order + tl.sum(less, axis=0), p)
    tl.store(rpos + p, tl.sum(same, axis=0))


@triton.jit
def _members(s, order, rpos, ti, lut, NK, BR: tl.constexpr, MT: tl.constexpr):
    """The group led from sorted position s: tile row j holds weight row j % BR for member j // BR. Returns the
    leader's cache index, whether s leads, and per tile row: the member's slot, whether it is a member, and the
    weight row."""
    p = tl.load(order + s)
    c = tl.load(lut + tl.load(ti + p)).to(tl.int64)
    lead = (tl.load(rpos + p) % MT) == 0
    j = tl.arange(0, MT * BR)
    sm = s + (j // BR).to(tl.int64)
    ok = sm < NK
    pj = tl.load(order + sm, mask=ok, other=0)
    cj = tl.load(lut + tl.load(ti + pj, mask=ok, other=0), mask=ok, other=-2)
    return c, lead, pj, ok & (cj == c), (j % BR).to(tl.int64)


@triton.jit
def _cgate_up_g(x, ti, lut, order, rpos, g, u, h, H, I, K, NK, BR: tl.constexpr, MT: tl.constexpr,
                LOG2E: tl.constexpr, LN2HI: tl.constexpr, LN2LO: tl.constexpr, P0: tl.constexpr, P1: tl.constexpr,
                P2: tl.constexpr, P3: tl.constexpr, P4: tl.constexpr, P5: tl.constexpr):
    rb = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1).to(tl.int64)
    c, lead, pj, vj, ij = _members(s, order, rpos, ti, lut, NK, BR, MT)
    if lead & (c >= 0):          # uncached slots: h is not read (_cdown_g writes zeros for them)
        tj = pj // K
        rows = rb * BR + ij
        ln = tl.arange(0, 16)
        ag = tl.zeros((MT * BR, 16), dtype=tl.float32)
        au = tl.zeros((MT * BR, 16), dtype=tl.float32)
        gb_ = g + c * I * H
        ub_ = u + c * I * H
        for k0 in range(0, H, 16):
            xv = tl.load(x + tj[:, None] * H + k0 + ln[None, :], mask=vj[:, None], other=0.0).to(tl.float32)
            wg = tl.load(gb_ + rows[:, None] * H + k0 + ln[None, :]).to(tl.float32)     # repeats: L1 hits
            wu = tl.load(ub_ + rows[:, None] * H + k0 + ln[None, :]).to(tl.float32)
            ag = tl.fma(wg, xv, ag)
            au = tl.fma(wu, xv, au)
        gb = _tree16(ag, MT * BR).to(tl.bfloat16).to(tl.float32)
        ub = _tree16(au, MT * BR).to(tl.bfloat16).to(tl.float32)
        ex = _dexp(-gb, LOG2E, LN2HI, LN2LO, P0, P1, P2, P3, P4, P5)
        sg = div_rn(gb, 1.0 + ex).to(tl.bfloat16).to(tl.float32)
        tl.store(h + pj * I + rows, (sg * ub).to(tl.bfloat16), mask=vj)


@triton.jit
def _cdown_g(hh, ti, lut, order, rpos, d, y, H, I, NK, BR: tl.constexpr, MT: tl.constexpr):
    rb = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1).to(tl.int64)
    c, lead, pj, vj, ij = _members(s, order, rpos, ti, lut, NK, BR, MT)
    rows = rb * BR + ij
    if lead:
        if c >= 0:
            ln = tl.arange(0, 16)
            acc = tl.zeros((MT * BR, 16), dtype=tl.float32)
            db = d + c * H * I
            for k0 in range(0, I, 16):
                hv = tl.load(hh + pj[:, None] * I + k0 + ln[None, :], mask=vj[:, None], other=0.0).to(tl.float32)
                w = tl.load(db + rows[:, None] * I + k0 + ln[None, :]).to(tl.float32)
                acc = tl.fma(w, hv, acc)
            out = _tree16(acc, MT * BR).to(tl.bfloat16)
        else:
            out = tl.zeros((MT * BR,), dtype=tl.bfloat16)
        tl.store(y + pj * H + rows, out, mask=vj)


# ---- v1.2 twin-q4 cached experts. Weights are row records (vahana.q4): K/2 nibble bytes then K/GS bf16 scales; in
# each 32-weight block byte b holds k = b (low) and k = b + 16 (high), as q + 8. Per lane the fma order is k0 + l then
# k0 + 16 + l, k0 ascending: exactly the CPU kernel's row_dots_q4, so the same bits.
@triton.jit
def _q4w(rec, rows, RB, k0, KH, GS: tl.constexpr):
    """Weight values (q*s, exact fp32) of k0..k0+15 (lo) and k0+16..k0+31 (hi) for tile rows; rec: uint8 base."""
    ln = tl.arange(0, 16)
    base = rec + rows * RB
    b = tl.load(base[:, None] + k0 // 2 + ln[None, :]).to(tl.int32)
    so = base + KH // 2 + 2 * (k0 // GS)
    sb = tl.load(so).to(tl.int32) | (tl.load(so + 1).to(tl.int32) << 8)
    sc = (sb << 16).to(tl.float32, bitcast=True)                  # bf16 bits -> fp32 (exact)
    wlo = ((b & 15) - 8).to(tl.float32) * sc[:, None]
    whi = ((b >> 4) - 8).to(tl.float32) * sc[:, None]
    return wlo, whi


@triton.jit
def _cgate_up_q4(x, ti, lut, g, u, h, H, I, K, RBH, BR: tl.constexpr, GS: tl.constexpr, LOG2E: tl.constexpr,
                 LN2HI: tl.constexpr, LN2LO: tl.constexpr, P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
                 P3: tl.constexpr, P4: tl.constexpr, P5: tl.constexpr):
    rb = tl.program_id(0).to(tl.int64)
    sl = tl.program_id(1).to(tl.int64)
    t = sl // K
    c = tl.load(lut + tl.load(ti + sl)).to(tl.int64)
    cs = tl.maximum(c, 0)
    rows = rb * BR + tl.arange(0, BR).to(tl.int64)
    ln = tl.arange(0, 16)
    ag = tl.zeros((BR, 16), dtype=tl.float32)
    au = tl.zeros((BR, 16), dtype=tl.float32)
    xb = x + t * H
    gb_ = g + cs * I * RBH
    ub_ = u + cs * I * RBH
    for k0 in range(0, H, 32):
        xlo = tl.load(xb + k0 + ln).to(tl.float32)
        xhi = tl.load(xb + k0 + 16 + ln).to(tl.float32)
        glo, ghi = _q4w(gb_, rows, RBH, k0, H, GS)
        ulo, uhi = _q4w(ub_, rows, RBH, k0, H, GS)
        ag = tl.fma(glo, xlo[None, :], ag)
        au = tl.fma(ulo, xlo[None, :], au)
        ag = tl.fma(ghi, xhi[None, :], ag)
        au = tl.fma(uhi, xhi[None, :], au)
    gb = _tree16(ag, BR).to(tl.bfloat16).to(tl.float32)
    ub = _tree16(au, BR).to(tl.bfloat16).to(tl.float32)
    ex = _dexp(-gb, LOG2E, LN2HI, LN2LO, P0, P1, P2, P3, P4, P5)
    s = div_rn(gb, 1.0 + ex).to(tl.bfloat16).to(tl.float32)
    tl.store(h + sl * I + rows, (s * ub).to(tl.bfloat16))


@triton.jit
def _cdown_q4(hh, ti, lut, d, y, H, I, RBI, BR: tl.constexpr, GS: tl.constexpr):
    rb = tl.program_id(0).to(tl.int64)
    sl = tl.program_id(1).to(tl.int64)
    c = tl.load(lut + tl.load(ti + sl)).to(tl.int64)
    cs = tl.maximum(c, 0)
    rows = rb * BR + tl.arange(0, BR).to(tl.int64)
    ln = tl.arange(0, 16)
    acc = tl.zeros((BR, 16), dtype=tl.float32)
    hb = hh + sl * I
    db = d + cs * H * RBI
    for k0 in range(0, I, 32):
        hlo = tl.load(hb + k0 + ln).to(tl.float32)
        hhi = tl.load(hb + k0 + 16 + ln).to(tl.float32)
        wlo, whi = _q4w(db, rows, RBI, k0, I, GS)
        acc = tl.fma(wlo, hlo[None, :], acc)
        acc = tl.fma(whi, hhi[None, :], acc)
    out = _tree16(acc, BR).to(tl.bfloat16)
    out = tl.where(c >= 0, out, tl.zeros_like(out))
    tl.store(y + sl * H + rows, out)


@triton.jit
def _cgate_up_g_q4(x, ti, lut, order, rpos, g, u, h, H, I, K, NK, RBH, BR: tl.constexpr, MT: tl.constexpr,
                   GS: tl.constexpr, LOG2E: tl.constexpr, LN2HI: tl.constexpr, LN2LO: tl.constexpr, P0: tl.constexpr,
                   P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr, P4: tl.constexpr, P5: tl.constexpr):
    rb = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1).to(tl.int64)
    c, lead, pj, vj, ij = _members(s, order, rpos, ti, lut, NK, BR, MT)
    if lead & (c >= 0):
        tj = pj // K
        rows = rb * BR + ij
        ln = tl.arange(0, 16)
        ag = tl.zeros((MT * BR, 16), dtype=tl.float32)
        au = tl.zeros((MT * BR, 16), dtype=tl.float32)
        gb_ = g + c * I * RBH
        ub_ = u + c * I * RBH
        for k0 in range(0, H, 32):
            xlo = tl.load(x + tj[:, None] * H + k0 + ln[None, :], mask=vj[:, None], other=0.0).to(tl.float32)
            xhi = tl.load(x + tj[:, None] * H + k0 + 16 + ln[None, :], mask=vj[:, None], other=0.0).to(tl.float32)
            glo, ghi = _q4w(gb_, rows, RBH, k0, H, GS)
            ulo, uhi = _q4w(ub_, rows, RBH, k0, H, GS)
            ag = tl.fma(glo, xlo, ag)
            au = tl.fma(ulo, xlo, au)
            ag = tl.fma(ghi, xhi, ag)
            au = tl.fma(uhi, xhi, au)
        gb = _tree16(ag, MT * BR).to(tl.bfloat16).to(tl.float32)
        ub = _tree16(au, MT * BR).to(tl.bfloat16).to(tl.float32)
        ex = _dexp(-gb, LOG2E, LN2HI, LN2LO, P0, P1, P2, P3, P4, P5)
        sg = div_rn(gb, 1.0 + ex).to(tl.bfloat16).to(tl.float32)
        tl.store(h + pj * I + rows, (sg * ub).to(tl.bfloat16), mask=vj)


@triton.jit
def _cdown_g_q4(hh, ti, lut, order, rpos, d, y, H, I, NK, RBI, BR: tl.constexpr, MT: tl.constexpr, GS: tl.constexpr):
    rb = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1).to(tl.int64)
    c, lead, pj, vj, ij = _members(s, order, rpos, ti, lut, NK, BR, MT)
    rows = rb * BR + ij
    if lead:
        if c >= 0:
            ln = tl.arange(0, 16)
            acc = tl.zeros((MT * BR, 16), dtype=tl.float32)
            db = d + c * H * RBI
            for k0 in range(0, I, 32):
                hlo = tl.load(hh + pj[:, None] * I + k0 + ln[None, :], mask=vj[:, None], other=0.0).to(tl.float32)
                hhi = tl.load(hh + pj[:, None] * I + k0 + 16 + ln[None, :], mask=vj[:, None], other=0.0).to(tl.float32)
                wlo, whi = _q4w(db, rows, RBI, k0, I, GS)
                acc = tl.fma(wlo, hlo, acc)
                acc = tl.fma(whi, hhi, acc)
            out = _tree16(acc, MT * BR).to(tl.bfloat16)
        else:
            out = tl.zeros((MT * BR,), dtype=tl.bfloat16)
        tl.store(y + pj * H + rows, out, mask=vj)


def gpu_slots_q4(x, ti, lut, G, U, D, H, I, BR: int = 16, MT: int | None = None, grouped: bool | None = None):
    """twin-q4 version of gpu_slots: G/U uint8 [C, I, RBH] and D uint8 [C, H, RBI] row records."""
    N = x.shape[0]
    K = ti.shape[1]
    RBH, RBI = G.shape[2], D.shape[2]
    GSH, GSI = 2 * H // (RBH - H // 2), 2 * I // (RBI - I // 2)
    assert GSH == GSI, "one group size per layer"
    ti = ti.contiguous()
    h = torch.empty((N * K, I), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((N, K, H), dtype=torch.bfloat16, device=x.device)
    if grouped is None:
        grouped = N > 1
    if MT is None:
        MT = 2 if N < 32 else 8
    c = dict(LOG2E=C_LOG2E, LN2HI=C_LN2HI, LN2LO=C_LN2LO, P0=P0, P1=P1, P2=P2, P3=P3, P4=P4, P5=P5)
    if grouped:
        NK = N * K
        order = torch.empty(NK, dtype=torch.int64, device=x.device)
        rpos = torch.empty(NK, dtype=torch.int32, device=x.device)
        _cgroup[(NK,)](ti, lut, order, rpos, NK, BLOCK=min(1024, triton.next_power_of_2(NK)), num_warps=4)
        _cgate_up_g_q4[(I // BR, NK)](x, ti, lut, order, rpos, G, U, h, H, I, K, NK, RBH, BR=BR, MT=MT, GS=GSH,
                                      num_warps=4, **c)
        _cdown_g_q4[(H // BR, NK)](h, ti, lut, order, rpos, D, y, H, I, NK, RBI, BR=BR, MT=MT, GS=GSI, num_warps=4)
        return y
    _cgate_up_q4[(I // BR, N * K)](x, ti, lut, G, U, h, H, I, K, RBH, BR=BR, GS=GSH, num_warps=4, **c)
    _cdown_q4[(H // BR, N * K)](h, ti, lut, D, y, H, I, RBI, BR=BR, GS=GSI, num_warps=4)
    return y


def gpu_slots(x, ti, lut, G, U, D, BR: int = 16, MT: int | None = None, grouped: bool | None = None):
    """x [N, H] bf16, ti [N, K] int64 (device), lut [E] int64, G/U [C, I, H], D [C, H, I] -> y [N, K, H] bf16."""
    N, H = x.shape
    K = ti.shape[1]
    I = G.shape[1]
    ti = ti.contiguous()
    h = torch.empty((N * K, I), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((N, K, H), dtype=torch.bfloat16, device=x.device)
    if grouped is None:
        grouped = N > 1          # one token: its K slots use K different experts, nothing to share
    if MT is None:               # group size: measured on the H100 (v03c_fused README), any size gives the same bits
        MT = 2 if N < 32 else 8
    if grouped:
        NK = N * K
        order = torch.empty(NK, dtype=torch.int64, device=x.device)
        rpos = torch.empty(NK, dtype=torch.int32, device=x.device)
        _cgroup[(NK,)](ti, lut, order, rpos, NK, BLOCK=min(1024, triton.next_power_of_2(NK)), num_warps=4)
        _cgate_up_g[(I // BR, NK)](x, ti, lut, order, rpos, G, U, h, H, I, K, NK, BR=BR, MT=MT, LOG2E=C_LOG2E,
                                   LN2HI=C_LN2HI, LN2LO=C_LN2LO, P0=P0, P1=P1, P2=P2, P3=P3, P4=P4, P5=P5,
                                   num_warps=4)
        _cdown_g[(H // BR, NK)](h, ti, lut, order, rpos, D, y, H, I, NK, BR=BR, MT=MT, num_warps=4)
        return y
    _cgate_up[(I // BR, N * K)](x, ti, lut, G, U, h, H, I, K, BR=BR, LOG2E=C_LOG2E, LN2HI=C_LN2HI, LN2LO=C_LN2LO,
                                P0=P0, P1=P1, P2=P2, P3=P3, P4=P4, P5=P5, num_warps=4)
    _cdown[(H // BR, N * K)](h, ti, lut, D, y, H, I, BR=BR, num_warps=4)
    return y


def combine(y: torch.Tensor, ti: torch.Tensor, tw: torch.Tensor) -> torch.Tensor:
    """The CPU kernel's combine: per token, slots in ascending expert id; acc = y0*w0, then acc + yj*wj (separate
    roundings); bf16. y [N, K, H] bf16, ti [N, K], tw [N, K] fp32 or bf16 -> [N, H] bf16."""
    N, K, H = y.shape
    order = torch.sort(ti, dim=1, stable=True)[1]
    ys = torch.gather(y, 1, order.unsqueeze(-1).expand(N, K, H)).float()
    ws = torch.gather(tw, 1, order).float()
    acc = ys[:, 0] * ws[:, 0:1]
    for q in range(1, K):
        acc = acc + ys[:, q] * ws[:, q:q + 1]
    return acc.to(torch.bfloat16)


class CachedExperts(nn.Module):
    """A VahanaExperts layer with `ids` of its experts also resident on the GPU."""

    def __init__(self, base, ids, device):
        super().__init__()
        self.base, self.spec, self.numa = base, base.spec, base.numa
        ids = sorted(int(i) for i in ids)
        self.ids = ids
        self.q4 = getattr(base, "q4", False)          # v1.2: twin-q4 row records stay uint8
        dt = torch.uint8 if self.q4 else torch.bfloat16
        st = lambda ws: torch.stack([ws[i] for i in ids]).to(device, dt).contiguous()
        self.G, self.U, self.D = st(base.g), st(base.u), st(base.d)
        lut = torch.full((self.spec.E,), -1, dtype=torch.int64)
        for c, e in enumerate(ids):
            lut[e] = c
        self.lut = lut.to(device)
        from . import hybrid as _hy
        _hy.ext().register_resident(self.spec.slot, (lut >= 0).to(torch.uint8))      # the CPU job derives its mask

    @staticmethod
    def supported(base) -> bool:
        s = base.spec
        if getattr(base, "q4", False) and (s.H % 32 or s.I % 32):
            return False
        return s.act == "silu_glu" and not s.bias and base.shared is None and s.H % 16 == 0 and s.I % 16 == 0

    def _slots(self, x, ti):
        if self.q4:
            return gpu_slots_q4(x, ti, self.lut, self.G, self.U, self.D, self.spec.H, self.spec.I)
        return gpu_slots(x, ti, self.lut, self.G, self.U, self.D)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        from . import runtime
        x = hidden_states.to(torch.bfloat16).contiguous()
        ti = top_k_index.to(torch.int64).contiguous()
        if runtime.overlap_on():
            # v1.1: post the CPU slots first, run the cached slots on the GPU while the CPU works, then combine on the
            # GPU with the CPU kernel's combine operations (cache.combine). Same bits as the sequential path below.
            hs = runtime.cpu_slots_post(self.spec.slot, x, ti, self.numa)
            y_gpu = self._slots(x, ti)
            y_cpu = runtime.cpu_slots_wait(hs, x.device)
            y = torch.where((self.lut[ti] >= 0).unsqueeze(-1), y_gpu, y_cpu)
            return combine(y, ti, runtime.weights_arg(top_k_weights)).to(hidden_states.dtype)
        y_gpu = self._slots(x, ti)       # cached slots (zeros elsewhere)
        # The CPU job runs the other slots and combines all K (nothing but one copy follows the CUDA-graph host node:
        # kernels queued behind a host node are expensive)
        out = runtime.moe_cached(self.spec.slot, x, ti, top_k_weights, y_gpu, self.numa)
        return out.to(hidden_states.dtype)


def popularity(model, run, layers) -> dict:
    """Count routed slots per (layer, expert) while `run()` drives the model (CPU-expert path)."""
    from .experts import VahanaExperts
    counts = {L: torch.zeros(m.spec.E, dtype=torch.int64) for L, m in layers.items()}
    orig = VahanaExperts.forward

    def fwd(self, h, ti, tw):
        if self.spec.slot in counts:
            counts[self.spec.slot] += torch.bincount(ti.flatten().cpu(), minlength=self.spec.E)
        return orig(self, h, ti, tw)
    VahanaExperts.forward = fwd
    try:
        run()
    finally:
        VahanaExperts.forward = orig
    return counts


def install(model, budget_bytes: float, counts: dict | None = None, frac: float | None = None, log=None) -> dict:
    """Cache the most-used (layer, expert) pairs that fit `budget_bytes` (or a fixed fraction `frac` of every layer),
    ranked globally by `counts` (uniform if None)."""
    from .experts import VahanaExperts
    from .modelspec import parent_of
    dev = next(p.device for p in model.parameters() if p.device.type == "cuda")
    layers = {m.spec.slot: (n, m) for n, m in model.named_modules()
              if isinstance(m, VahanaExperts) and CachedExperts.supported(m)}
    if not layers:
        return {"cached_experts": 0, "layers": 0, "gb": 0.0, "reason": "no supported layer"}
    pick = {L: [] for L in layers}
    if frac is not None:
        for L, (_, m) in layers.items():
            c = counts[L] if counts else torch.zeros(m.spec.E)
            order = sorted(range(m.spec.E), key=lambda e: (-int(c[e]), e))
            pick[L] = order[:max(1, round(frac * m.spec.E))]
    else:
        nb = lambda t: t.numel() * t.element_size()
        per = {L: nb(m.g[0]) + nb(m.u[0]) + nb(m.d[0]) for L, (_, m) in layers.items()}  # bytes per expert (g, u, d)
        cand = sorted(((-int(counts[L][e]) if counts else 0, L, e) for L, (_, m) in layers.items()
                       for e in range(m.spec.E)))
        used = 0
        for _, L, e in cand:
            if used + per[L] > budget_bytes:
                continue
            pick[L].append(e)
            used += per[L]
    n = gb = 0
    for L, (name, m) in layers.items():
        if not pick[L]:
            continue
        parent, leaf = parent_of(model, name)
        ce = CachedExperts(m, pick[L], dev)
        setattr(parent, leaf, ce)
        n += len(pick[L])
        gb += (ce.G.numel() + ce.U.numel() + ce.D.numel()) * ce.G.element_size() / 2**30
    hit = None
    if counts:
        tot = sum(int(c.sum()) for c in counts.values())
        hit = sum(int(counts[L][pick[L]].sum()) for L in pick if pick[L]) / max(1, tot)
    info = {"cached_experts": n, "layers": sum(1 for L in pick if pick[L]), "gb": gb, "profile_hit": hit}
    if log:
        log(f"vahana: GPU expert cache {n} experts in {info['layers']} layers, {gb:.1f} GB"
            + (f", {hit:.0%} of profiled expert uses" if hit is not None else ""))
    return info
