"""v1.4: a 4-bit linear kept 4-bit on the GPU gives the bits of the bf16 expansion 1.3 used (needs CUDA)."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("R,K", [(4096, 2048), (512, 2048), (2048, 4096), (96, 256)])
@pytest.mark.parametrize("n", [1, 3, 16, 70])
def test_q4linear_equals_bf16_expansion(R, K, n):
    from vahana import q4, q4lin, tmath, twinlin
    g = torch.Generator().manual_seed(R + K + n)
    W = torch.randn(R, K, generator=g) * 0.05
    rec = q4.from_bf16(W.to(torch.bfloat16))
    Wb = q4.dequant(rec, K).to(torch.bfloat16).cuda()         # what 1.3's loader stored
    x = torch.randn(1, n, K, generator=g).to("cuda", torch.bfloat16)
    lin = q4lin.Q4Linear(rec.cuda(), K, R)
    ref = twinlin.gpu_linear(x.view(n, K), Wb).view(1, n, R)
    assert torch.equal(lin(x), ref)
    assert torch.equal(lin.weight, Wb)
    audit = {}
    with tmath.TwinMode(audit):
        y = lin(x)
    assert torch.equal(y, ref) and audit == {}


@pytest.mark.parametrize("U", ["1", "2", "4", "8"])
@pytest.mark.parametrize("br", ["4", "16"])
def test_q4linear_launch_shapes_same_bits(U, br, monkeypatch):
    """v1.4: the unrolled decode kernel (U blocks per iteration) and any rows-per-program give the same bits."""
    from vahana import q4, q4lin, twinlin
    R, K = 512, 2048
    g = torch.Generator().manual_seed(11)
    rec = q4.from_bf16((torch.randn(R, K, generator=g) * 0.05).to(torch.bfloat16))
    Wb = q4.dequant(rec, K).to(torch.bfloat16).cuda()
    x = torch.randn(1, 1, K, generator=g).to("cuda", torch.bfloat16)
    monkeypatch.setenv("VAHANA_Q4LIN_U", U)
    monkeypatch.setenv("VAHANA_Q4LIN_BR", br)
    assert torch.equal(q4lin.Q4Linear(rec.cuda(), K, R)(x), twinlin.gpu_linear(x.view(1, K), Wb).view(1, 1, R))
