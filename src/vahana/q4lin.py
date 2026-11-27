"""v1.4: 4-bit non-expert linears stay 4-bit on the GPU, with the bits of 1.3.

Up to 1.3, a pack-quantized attention projection (q/k/v/o of a w4a16 checkpoint) was expanded at load time to bf16:
W = bf16(q * s), and run by the twin linear (16 fp32 fma lanes, k ascending, tree16, bf16). Q4Linear keeps the
twin-q4 records on the GPU (4.125 bits per weight instead of 16) and rebuilds exactly that bf16 weight inside the
kernel, right before the same fma sequence. So the outputs are bit-identical to 1.3, while the layer takes about a
quarter of the GPU memory and a quarter of the memory traffic per decode token.

  - decode-sized batches (n < VAHANA_TILE_MIN tokens): _lin_q4, a fused 4-bit kernel;
  - prefill-sized batches: the weight is rebuilt in bf16 once per call and run by twinlin.gpu_linear (tiled).
`.weight` still returns the bf16 matrix (built on demand) for any code that reads it.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from ._triton import tl, triton
from .cache import _q4w
from .twinlin import _tree16


@triton.jit
def _lin_q4(x, rec, y, K, R, RB, BR: tl.constexpr, GS: tl.constexpr, U: tl.constexpr = 1):
    rows = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    ln = tl.arange(0, 16)
    acc = tl.zeros((BR, 16), dtype=tl.float32)
    xb = x + t * K
    # v1.4: U blocks of 32 weights per iteration (unrolled), so their loads are in flight together; the fma order per
    # lane is unchanged (k ascending), so the bits are too. Laptop RTX 3050 Ti, Qwen3-30B attention, per layer:
    # 132 -> 80 us (experiments/v15_decode/q4lin_bench.py).
    for k0 in range(0, K, 32 * U):
        for u in tl.static_range(U):
            kk = k0 + 32 * u
            wlo, whi = _q4w(rec, rows, RB, kk, K, GS)
            wlo = wlo.to(tl.bfloat16).to(tl.float32)          # the weight 1.3 stored: bf16(q * s), round to nearest even
            whi = whi.to(tl.bfloat16).to(tl.float32)
            acc = tl.fma(wlo, tl.load(xb + kk + ln).to(tl.float32)[None, :], acc)       # lane l: kk + l ...
            acc = tl.fma(whi, tl.load(xb + kk + 16 + ln).to(tl.float32)[None, :], acc)  # ... then kk + 16 + l
    tl.store(y + t * R + rows, _tree16(acc, BR).to(tl.bfloat16))


def _cfg(R: int, K: int) -> tuple[int, int, int]:
    """Launch shape (rows per program, unroll, warps), measured on the laptop GPU (v15_decode/q4lin_bench.py)."""
    br, nw = (16, 4 if K >= 4096 else 2) if R >= 2048 else (4, 1)
    U = 4 if K % 128 == 0 else 1
    br = int(os.environ.get("VAHANA_Q4LIN_BR", br))
    while R % br:
        br //= 2
    return br, int(os.environ.get("VAHANA_Q4LIN_U", U)), int(os.environ.get("VAHANA_Q4LIN_WARPS", nw))


def _bf16_weight(rec: torch.Tensor, K: int) -> torch.Tensor:
    from . import q4
    return q4.dequant(rec, K).to(torch.bfloat16)


class Q4Linear(nn.Module):
    """A bias-free linear whose weight is twin-q4 records [R, K/2 + 2K/G] on the GPU."""

    def __init__(self, rec: torch.Tensor, in_features: int, out_features: int):
        super().__init__()
        self.register_buffer("rec", rec, persistent=False)
        self.in_features, self.out_features = in_features, out_features
        RB = rec.shape[1]
        self.G = 2 * in_features // (RB - in_features // 2)
        self.bias = None

    @property
    def weight(self) -> torch.Tensor:
        return _bf16_weight(self.rec, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from . import twinlin
        shp = x.shape
        x2 = x.reshape(-1, shp[-1]).to(torch.bfloat16).contiguous()
        n, K, R = x2.shape[0], self.in_features, self.out_features
        if n >= twinlin.TILE_MIN or R % 8:
            y = twinlin.gpu_linear(x2, _bf16_weight(self.rec, K))
        else:
            y = torch.empty((n, R), dtype=torch.bfloat16, device=x2.device)
            br, U, nw = _cfg(R, K)
            if n:
                _lin_q4[(R // br, n)](x2, self.rec, y, K, R, self.rec.shape[1], BR=br, GS=self.G, U=U, num_warps=nw)
        return y.view(*shp[:-1], R).to(x.dtype)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, twin-q4 G={self.G}"


def supported(K: int, R: int) -> bool:
    return K % 32 == 0 and R % 2 == 0
