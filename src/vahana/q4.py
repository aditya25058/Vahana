"""v1.2 twin-q4: 4-bit expert weights with Vahana's exact (device-invariant) arithmetic.

A weight matrix [R, K] is stored as R row records (uint8 [R, K/2 + 2*K/G]): K/2 nibble bytes, then K/G bf16 scales.
In each 32-weight block, byte b holds k = b (low nibble) and k = b + 16 (high nibble), both as q + 8 with q in [-8, 7].
The weight value is q * s (exact in fp32), and every dot product follows the twin rule (16 fp32 lanes, k ascending,
fma, tree16, bf16 rounding), so the CPU kernel (every ISA) and the GPU give the same bits.

Sources: compressed-tensors "pack-quantized" symmetric int4 checkpoints (GPTQ / RTN, e.g. *-quantized.w4a16), or
round-to-nearest from bf16 (from_bf16, for tests)."""
from __future__ import annotations

import torch


def is_q4(t: torch.Tensor) -> bool:
    return t.dtype == torch.uint8


def _records(u: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """u: uint8 [R, K] values q + 8 in k order; scale [R, K/G] -> records uint8 [R, K/2 + 2K/G]."""
    R, K = u.shape
    v = u.reshape(R, K // 32, 2, 16)
    nib = (v[:, :, 0, :] | (v[:, :, 1, :] << 4)).reshape(R, K // 2)
    sc = scale.to(torch.bfloat16).contiguous().view(torch.uint8).reshape(R, -1)
    return torch.cat([nib, sc], 1).contiguous()


def from_compressed(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """compressed-tensors pack-quantized int4 (int32 [R, K/8]: nibble i of word j is k = 8j + i, stored q + 8;
    symmetric, no zero point) + scale [R, K/G] -> twin-q4 records. Lossless."""
    if scale.dtype != torch.bfloat16:
        raise ValueError(f"twin-q4 needs bf16 scales (got {scale.dtype})")
    R = packed.shape[0]
    p = packed.to(torch.int32)
    u = torch.stack([(p >> (4 * i)) & 15 for i in range(8)], -1).reshape(R, -1).to(torch.uint8)
    return _records(u, scale)


def dequant_compressed(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp32 values q * s of a pack-quantized matrix (for non-expert layers, which Vahana keeps in bf16)."""
    R = packed.shape[0]
    p = packed.to(torch.int32)
    q = torch.stack([(p >> (4 * i)) & 15 for i in range(8)], -1).reshape(R, -1) - 8
    return q.float() * scale.float().repeat_interleave(q.shape[1] // scale.shape[1], 1)


def from_bf16(W: torch.Tensor, G: int = 128) -> torch.Tensor:
    """Round-to-nearest symmetric int4 with bf16 scales (s = bf16(max|w| / 7)); for tests and own conversions."""
    R, K = W.shape
    Wf = W.float().reshape(R, K // G, G)
    s = (Wf.abs().amax(-1) / 7).to(torch.bfloat16)
    sf = s.float()
    q = torch.clamp(torch.round(Wf / torch.where(sf == 0, torch.ones_like(sf), sf)[..., None]), -8, 7)
    return _records((q.reshape(R, K) + 8).to(torch.uint8), s)


def dequant(rec: torch.Tensor, K: int) -> torch.Tensor:
    """Exact fp32 weights q * s of twin-q4 records [R, K/2 + 2K/G] (reference / GPU use)."""
    R = rec.shape[0]
    nib = rec[:, : K // 2].reshape(R, K // 32, 16).to(torch.int32)
    q = torch.stack([(nib & 15) - 8, (nib >> 4) - 8], 2).reshape(R, K)
    s = rec[:, K // 2:].contiguous().view(torch.bfloat16).float()
    return q.float() * s.repeat_interleave(K // s.shape[1], 1)


def ref_linear(x: torch.Tensor, rec: torch.Tensor) -> torch.Tensor:
    """Scalar twin reference (CPU torch): x [n, K] bf16 -> [n, R] bf16."""
    n, K = x.shape
    W = dequant(rec, K)
    R = W.shape[0]
    Wl = W.reshape(R, K // 16, 16)
    xf = x.float()
    out = torch.empty(n, R, dtype=torch.bfloat16)
    for t in range(n):
        acc = torch.zeros(R, 16)
        xl = xf[t].reshape(K // 16, 16)
        for j in range(K // 16):
            acc = acc + Wl[:, j, :] * xl[j]            # w * x is exact (<= 20 bits), so mul + add == fma
        for st in (8, 4, 2, 1):
            acc = acc[:, :st] + acc[:, st:2 * st]
        out[t] = acc[:, 0].to(torch.bfloat16)
    return out
