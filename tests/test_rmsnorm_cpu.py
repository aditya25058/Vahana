"""v1.4: the C++ twin RMSNorm (CPU) gives the bits of the TwinMode path of a real RMSNorm module."""
import pytest
import torch

from vahana import tmath
from vahana.twinlin import twin_rmsnorm

Norm = pytest.importorskip("transformers.models.qwen3_moe.modeling_qwen3_moe").Qwen3MoeRMSNorm


@pytest.mark.parametrize("N", [128, 768, 1000, 2048, 4096])
@pytest.mark.parametrize("rows", [1, 7, 64])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 300.0])
def test_cpu_rmsnorm_bits(N, rows, scale):
    g = torch.Generator().manual_seed(N * 7 + rows)
    m = Norm(N, eps=1e-6)
    with torch.no_grad():
        m.weight.copy_(torch.randn(N, generator=g) * 0.3 + 1)
    m = m.to(torch.bfloat16)
    x = (torch.randn(rows, N, generator=g) * scale).to(torch.bfloat16)
    with torch.no_grad(), tmath.TwinMode():
        ref = m(x)
    got = twin_rmsnorm(x, m.weight, m.variance_epsilon)
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))


def test_fast_norms_uses_cpu_kernel():
    m = torch.nn.Sequential(Norm(256, eps=1e-6).to(torch.bfloat16))
    info = tmath.fast_norms(m)
    assert info["fast"] == 1 and info["twinmode"] == []
    x = torch.randn(3, 256).to(torch.bfloat16)
    with torch.no_grad(), tmath.TwinMode():
        ref = Norm.forward(m[0], x)
    assert torch.equal(m[0](x), ref)
