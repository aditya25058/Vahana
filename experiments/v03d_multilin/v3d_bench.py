"""v0.3-d design bench: multi-token twin linear variants vs the per-token _lin, bitwise check + time (H100)."""
import torch
import triton
import triton.language as tl

from vahana.twinlin import _cfg, _lin, _tree16, gpu_linear


@triton.jit
def _lin2d(x, w, y, n, K, R, BR: tl.constexpr, MT: tl.constexpr, U: tl.constexpr):
    """Tile row j: token tb*MT + j // BR, weight row j % BR (weight repeats are L1 hits)."""
    rb = tl.program_id(0).to(tl.int64)
    tb = tl.program_id(1).to(tl.int64)
    j = tl.arange(0, MT * BR)
    t = tb * MT + (j // BR).to(tl.int64)
    vt = t < n
    rows = rb * BR + (j % BR).to(tl.int64)
    ln = tl.arange(0, 16)
    acc = tl.zeros((MT * BR, 16), dtype=tl.float32)
    for k0 in range(0, K, 16 * U):
        for u in tl.static_range(U):
            kk = k0 + 16 * u
            xv = tl.load(x + t[:, None] * K + kk + ln[None, :], mask=vt[:, None], other=0.0).to(tl.float32)
            wv = tl.load(w + rows[:, None] * K + kk + ln[None, :]).to(tl.float32)
            acc = tl.fma(wv, xv, acc)
    tl.store(y + t * R + rows, _tree16(acc, MT * BR).to(tl.bfloat16), mask=vt)


@triton.jit
def _acc4(acc, wv, x, t, n, K, kk, ln):
    xv = tl.load(x + t * K + kk + ln, mask=(t < n) & (ln < 16), other=0.0).to(tl.float32)
    return tl.fma(wv, xv[None, :], acc)


@triton.jit
def _st(y, t, n, R, rows, acc, BR: tl.constexpr):
    tl.store(y + t * R + rows, _tree16(acc, BR).to(tl.bfloat16), mask=(rows >= 0) & (t < n))


@triton.jit
def _linu(x, w, y, n, K, R, BR: tl.constexpr, MT: tl.constexpr, U: tl.constexpr):
    """MT (1, 2, 4 or 8) separate accumulators: each weight tile is loaded once and used for every token."""
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
            a0 = _acc4(a0, wv, x, t0, n, K, kk, ln)
            if MT >= 2:
                a1 = _acc4(a1, wv, x, t0 + 1, n, K, kk, ln)
            if MT >= 4:
                a2 = _acc4(a2, wv, x, t0 + 2, n, K, kk, ln)
                a3 = _acc4(a3, wv, x, t0 + 3, n, K, kk, ln)
            if MT >= 8:
                a4 = _acc4(a4, wv, x, t0 + 4, n, K, kk, ln)
                a5 = _acc4(a5, wv, x, t0 + 5, n, K, kk, ln)
                a6 = _acc4(a6, wv, x, t0 + 6, n, K, kk, ln)
                a7 = _acc4(a7, wv, x, t0 + 7, n, K, kk, ln)
    _st(y, t0, n, R, rows, a0, BR)
    if MT >= 2:
        _st(y, t0 + 1, n, R, rows, a1, BR)
    if MT >= 4:
        _st(y, t0 + 2, n, R, rows, a2, BR)
        _st(y, t0 + 3, n, R, rows, a3, BR)
    if MT >= 8:
        _st(y, t0 + 4, n, R, rows, a4, BR)
        _st(y, t0 + 5, n, R, rows, a5, BR)
        _st(y, t0 + 6, n, R, rows, a6, BR)
        _st(y, t0 + 7, n, R, rows, a7, BR)


def run(kern, x, W, MT, BR=None, nw=None):
    n, K = x.shape
    R = W.shape[0]
    br, U, nw0 = _cfg(R, K)
    BR, nw = BR or br, nw or nw0
    y = torch.empty((n, R), dtype=torch.bfloat16, device=x.device)
    kern[(R // BR, triton.cdiv(n, MT))](x, W, y, n, K, R, BR=BR, MT=MT, U=U, num_warps=nw)
    return y


g = torch.Generator().manual_seed(0)
bad = 0
for (R, K, name) in ((2048, 2048, "olmoe qkvo"), (64, 2048, "olmoe router"), (50304, 2048, "olmoe lm_head"),
                     (4096, 2048, "qwen3 q"), (2048, 4096, "qwen3 o")):
    W = (torch.randn(R, K, generator=g) * 0.02).cuda().bfloat16()
    for n in (1, 2, 5, 8, 16, 64, 512):
        x = torch.randn(n, K, generator=g).cuda().bfloat16()
        ref = gpu_linear(x, W)
        row = {"base": triton.testing.do_bench(lambda: gpu_linear(x, W), warmup=25, rep=150)}
        for kern, kn in ((_lin2d, "2d"), (_linu, "u")):
            for MT in (2, 4, 8):
                for BRm in (1, 2, 4):
                    br = _cfg(R, K)[0] * BRm
                    if R % br or br > 64:
                        continue
                    for nw in (1, 2, 4):
                        y = run(kern, x, W, MT, br, nw)
                        if not torch.equal(y.view(torch.int16), ref.view(torch.int16)):
                            bad += 1
                            print("MISMATCH", name, n, kn, MT, br, nw, flush=True)
                        row[f"{kn} MT{MT} BR{br} w{nw}"] = triton.testing.do_bench(
                            lambda: run(kern, x, W, MT, br, nw), warmup=25, rep=150)
        best = min((v, k) for k, v in row.items() if k != "base")
        best2 = min(((v, k) for k, v in row.items() if k.startswith("2d")))
        bestu = min(((v, k) for k, v in row.items() if k.startswith("u")))
        print(f"{name:14s} n={n:4d} base {row['base']*1e3:8.1f}us  best {best[0]*1e3:8.1f}us {best[1]:18s} "
              f"x{row['base']/best[0]:.2f}  (2d {best2[1]} {best2[0]*1e3:.1f}, u {bestu[1]} {bestu[0]*1e3:.1f})",
              flush=True)
print("mismatches:", bad)
