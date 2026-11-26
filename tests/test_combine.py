"""v1.4: the fused combine kernel gives combine()'s bits (needs CUDA)."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("N,K,H", [(1, 8, 2048), (3, 8, 2048), (5, 4, 300), (1, 2, 64), (17, 6, 1024)])
@pytest.mark.parametrize("wdt", [torch.float32, torch.bfloat16])
def test_combine_fused_matches(N, K, H, wdt):
    from vahana import cache
    g = torch.Generator().manual_seed(N * 100 + K + H)
    E = 32
    ti = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(N)]).cuda()
    tw = torch.softmax(torch.randn(N, K, generator=g), -1).to(wdt).cuda()
    yg = (torch.randn(N, K, H, generator=g) * 3).to(torch.bfloat16).cuda()
    yc = (torch.randn(N, K, H, generator=g) * 3).to(torch.bfloat16).cuda()
    lut = torch.full((E,), -1, dtype=torch.int64)
    lut[torch.randperm(E, generator=g)[:E // 3]] = torch.arange(E // 3)
    lut = lut.cuda()
    ref = cache.combine(torch.where((lut[ti] >= 0).unsqueeze(-1), yg, yc), ti, tw)
    assert torch.equal(cache.combine_fused(yg, ti, tw, yc, lut), ref)
    assert torch.equal(cache.combine_fused(yg, ti, tw), cache.combine(yg, ti, tw))
