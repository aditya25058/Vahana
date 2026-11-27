"""v1.4: the adaptive GPU expert cache changes what the cache holds between steps, never a bit (needs CUDA)."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("fam", ["qwen3_moe", "olmoe", "deepseek_v3"])
def test_adaptive_cache_same_bits(fam, monkeypatch):
    from vahana import adaptive, decode, loader, selftest
    from vahana.cache import CachedExperts
    cfg, m = selftest.build(fam)
    model, _ = loader.prepare(m.to("cuda"), gpu_cache_frac=0.25)
    g = torch.Generator().manual_seed(5)
    calls = [(torch.randint(1, cfg.vocab_size, (1, n), generator=g).cuda(), k) for n, k in ((24, 12), (40, 10), (24, 12))]
    ref = [decode.generate(model, p, max_new_tokens=k, return_logits=True) for p, k in calls]
    layers = sorted((x for x in model.modules() if isinstance(x, CachedExperts)), key=lambda x: x.spec.slot)
    ad = adaptive.Adaptive(layers, torch.device("cuda"))
    ad.swaps = 8
    monkeypatch.setattr(adaptive, "MARGIN", -1.0)          # swap whenever any uncached expert was used: maximal churn
    model._vahana_adaptive = ad
    try:
        got = [decode.generate(model, p, max_new_tokens=k, return_logits=True) for p, k in calls]
    finally:
        model._vahana_adaptive = None
    assert ad.stats["swaps"] > 0
    for r, x in zip(ref, got):
        assert torch.equal(r["logits"], x["logits"]) and torch.equal(r["ids"], x["ids"])
    for x in layers:                                          # GPU table, CPU table and resident set agree
        assert torch.equal(x.lut.cpu(), x.lut_cpu)
        assert sorted(set(x.lut_cpu.tolist()) - {-1}) == list(range(x.G.shape[0]))
