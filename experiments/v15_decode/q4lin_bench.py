"""v1.4: the decode 4-bit linear (_lin_q4) against unrolled variants, Qwen3-30B attention shapes, on this GPU.

  python q4lin_bench.py
Every variant must give _lin_q4's bits (checked); prints microseconds per call and GB/s of q4 records.
"""
import torch

from vahana import q4lin
from vahana._triton import tl, triton
from vahana.cache import _q4w
from vahana.twinlin import _tree16


@triton.jit
def _lin_q4u(x, rec, y, K, R, RB, BR: tl.constexpr, GS: tl.constexpr, U: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    ln = tl.arange(0, 16)
    acc = tl.zeros((BR, 16), dtype=tl.float32)
    xb = x + t * K
    for k0 in range(0, K, 32 * U):
        for u in tl.static_range(U):
            kk = k0 + 32 * u
            wlo, whi = _q4w(rec, rows, RB, kk, K, GS)
            wlo = wlo.to(tl.bfloat16).to(tl.float32)
            whi = whi.to(tl.bfloat16).to(tl.float32)
            acc = tl.fma(wlo, tl.load(xb + kk + ln).to(tl.float32)[None, :], acc)
            acc = tl.fma(whi, tl.load(xb + kk + 16 + ln).to(tl.float32)[None, :], acc)
    tl.store(y + t * R + rows, _tree16(acc, BR).to(tl.bfloat16))


def rec_for(R, K, G=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    nib = torch.randint(0, 256, (R, K // 2), generator=g, dtype=torch.uint8)
    sc = (torch.rand(R, K // G, generator=g) * 0.02 + 0.001).to(torch.bfloat16).view(torch.uint8).view(R, 2 * K // G)
    return torch.cat([nib, sc], 1).cuda()


def timeit(fn, reps=100):
    """GPU time per call inside a CUDA graph of `reps` calls (as in graph decode: no launch overhead)."""
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=st):
        for _ in range(reps):
            fn()
    for _ in range(3):
        g.replay()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    best = 1e9
    for _ in range(5):
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / reps * 1e3)
    return best


for name, R, K in (("q", 4096, 2048), ("k/v", 512, 2048), ("o", 2048, 4096), ("qkv", 5120, 2048)):
    rec = rec_for(R, K)
    RB = rec.shape[1]
    G = 2 * K // (RB - K // 2)
    x = (torch.randn(1, K) * 0.5).to(torch.bfloat16).cuda()
    lin = q4lin.Q4Linear(rec, K, R)
    ref = lin(x)
    yb = torch.empty(1, R, dtype=torch.bfloat16, device="cuda")
    br0 = 16 if R >= 2048 else 8
    us = timeit(lambda: q4lin._lin_q4[(R // br0, 1)](x, rec, yb, K, R, RB, BR=br0, GS=G, num_warps=4 if K >= 4096 else 2))
    print(f"{name:4s} R={R} K={K}: _lin_q4 {us:7.1f} us  {rec.numel() / us / 1e3:6.1f} GB/s", flush=True)
    res = []
    for br in (4, 8, 16, 32):
        for U in (1, 2, 4, 8):
            for nw in (1, 2, 4):
                if R % br or (K // 32) % U:
                    continue
                y = torch.empty(1, R, dtype=torch.bfloat16, device="cuda")
                f = lambda: _lin_q4u[(R // br, 1)](x, rec, y, K, R, RB, BR=br, GS=G, U=U, num_warps=nw)
                try:
                    f()
                except Exception as ex:  # noqa: BLE001
                    continue
                same = torch.equal(y, ref)
                us2 = timeit(f, 100)
                res.append((us2, br, U, nw, same))
    assert all(r[4] for r in res), "a variant changed the bits"
    for us2, br, U, nw, _ in sorted(res)[:3]:
        print(f"     br={br:2d} U={U} nw={nw}: {us2:7.1f} us  {rec.numel() / us2 / 1e3:6.1f} GB/s  (same bits)", flush=True)
