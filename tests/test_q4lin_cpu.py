"""v1.4: a 4-bit non-expert linear on the CPU gives the bits of its bf16 expansion (1.3's weight bf16(q * s))."""
import pytest
import torch

from vahana import q4, q4lin
from vahana.q4lin import Q4Linear, _bf16_weight
from vahana.runtime import cpu_linear


@pytest.mark.parametrize("K,R,G", [(2048, 512, 128), (4096, 2048, 128), (768, 64, 32), (2048, 4096, 64)])
@pytest.mark.parametrize("n", [1, 3, 16, 21])
def test_cpu_q4_linear_bits(K, R, G, n):
    g = torch.Generator().manual_seed(K + R + n)
    W = (torch.randn(R, K, generator=g) * 0.05).to(torch.bfloat16)
    rec = q4.from_bf16(W, G)
    x = (torch.randn(n, K, generator=g) * 2).to(torch.bfloat16)
    ref = cpu_linear(_bf16_weight(rec, K).contiguous(), x)            # 1.3: expand once, bf16 twin linear
    got = Q4Linear(rec, K, R)(x)
    assert got.dtype == torch.bfloat16 and got.shape == (n, R)
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))


def test_supported_cpu_rows_in_fours():
    assert q4lin.supported(2048, 4, "cpu") and not q4lin.supported(2048, 6, "cpu") and q4lin.supported(2048, 6)
