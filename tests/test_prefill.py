"""v1.3: GPU-streamed prefill gives the CPU-expert path's bits (needs CUDA + Triton; skipped otherwise)."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("fam", ["olmoe", "qwen3_moe", "deepseek_v3"])
@pytest.mark.parametrize("cache_frac", [None, 0.5])
def test_prefill_paths_agree(fam, cache_frac, monkeypatch):
    from vahana import loader, prefill, selftest
    cfg, m = selftest.build(fam)
    model, _ = loader.prepare(m.to("cuda"), gpu_cache_frac=cache_frac)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1, cfg.vocab_size, (1, 96), generator=g).cuda()
    out = {}
    for mode in ("0", "8"):
        monkeypatch.setenv("VAHANA_PREFILL_GPU", mode)
        before = prefill.STATS["layers"]
        with torch.no_grad():
            out[mode] = model(ids, use_cache=False).logits.float()
        used = prefill.STATS["layers"] - before
        assert (used > 0) == (mode == "8")
    assert torch.equal(out["0"], out["8"])


def test_tiled_kernels_match_decode_kernels():
    from vahana import prefill_kernels as pk
    from vahana.cache import gpu_slots
    g = torch.Generator().manual_seed(1)
    C, H, I, N, K = 12, 256, 128, 200, 4
    G = (torch.randn(C, I, H, generator=g) * 0.05).to("cuda", torch.bfloat16)
    U = (torch.randn(C, I, H, generator=g) * 0.05).to("cuda", torch.bfloat16)
    D = (torch.randn(C, H, I, generator=g) * 0.05).to("cuda", torch.bfloat16)
    x = torch.randn(N, H, generator=g).to("cuda", torch.bfloat16)
    E = 20
    ti = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(N)]).cuda()
    lut = torch.full((E,), -1, dtype=torch.int64)
    lut[torch.randperm(E, generator=g)[:C]] = torch.arange(C)
    lut = lut.cuda()
    ref = gpu_slots(x, ti, lut, G, U, D)
    mask = (lut[ti] >= 0).unsqueeze(-1)
    for BM, BN in ((16, 16), (16, 32), (32, 16)):
        y = pk.slots(x, ti, lut, G, U, D, False, H, I, BM, BN)
        assert torch.equal(torch.where(mask, y, 0), torch.where(mask, ref, 0))


def test_tiled_linear_matches():
    from vahana import twinlin
    g = torch.Generator().manual_seed(2)
    x = torch.randn(300, 512, generator=g).to("cuda", torch.bfloat16)
    W = (torch.randn(384, 512, generator=g) * 0.05).to("cuda", torch.bfloat16)
    assert torch.equal(twinlin.gpu_linear(x, W), twinlin.gpu_linear(x, W, MT=1))


def test_second_generate_call_same_bits():
    """v1.4: a warmed model captures its CUDA graph at the first decode step; the outputs equal the first call's."""
    from vahana import decode, loader, selftest
    cfg, m = selftest.build("qwen3_moe")
    model, _ = loader.prepare(m.to("cuda"))
    ids = torch.randint(1, cfg.vocab_size, (1, 20), generator=torch.Generator().manual_seed(3)).cuda()
    a = decode.generate(model, ids, max_new_tokens=8, return_logits=True)
    b = decode.generate(model, ids, max_new_tokens=8, return_logits=True)
    assert a["graph_used"] and b["graph_used"]
    assert torch.equal(a["logits"], b["logits"]) and torch.equal(a["ids"], b["ids"])


@pytest.mark.parametrize("fam", ["qwen3_moe", "olmoe", "deepseek_v3", "gpt_oss"])
def test_graph_reuse_same_bits(fam, monkeypatch):
    """v1.4: decode buffers, static cache and CUDA graph reused across calls give each call's fresh bits."""
    from vahana import decode, loader, selftest
    cfg, m = selftest.build(fam)
    model, _ = loader.prepare(m.to("cuda"))
    g = torch.Generator().manual_seed(7)
    p1 = torch.randint(1, cfg.vocab_size, (1, 20), generator=g).cuda()
    p2 = torch.randint(1, cfg.vocab_size, (1, 33), generator=g).cuda()
    calls = [(p1, 8), (p2, 6), (p1, 8), (p2, 30)]
    monkeypatch.setenv("VAHANA_GRAPH_REUSE", "0")
    ref = [decode.generate(model, p, max_new_tokens=n, return_logits=True) for p, n in calls]
    monkeypatch.setenv("VAHANA_GRAPH_REUSE", "1")
    got = [decode.generate(model, p, max_new_tokens=n, return_logits=True) for p, n in calls]
    assert getattr(model, "_vahana_ctx", None) is not None and model._vahana_ctx["g"] is not None
    for r, x in zip(ref, got):
        assert x["graph_used"] and torch.equal(r["logits"], x["logits"]) and torch.equal(r["ids"], x["ids"])
