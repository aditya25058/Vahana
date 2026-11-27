"""v1.4: expert prefetch is a hint to the OS only. Every mode gives the same bits, and modes 3 and 4 fire."""
import pytest
import torch


@pytest.mark.parametrize("fam", ["qwen3_moe", "olmoe"])
def test_prefetch_modes_same_bits(fam, monkeypatch):
    from vahana import decode, hybrid, loader, selftest
    k = hybrid.ext()
    if not getattr(k, "has_prefetch", False):
        pytest.skip("kernel without prefetch")
    out = {}
    try:
        for mode in (0, 1, 2, 3, 4):
            cfg, m = selftest.build(fam)
            model, _ = loader.prepare(m, nx=False)                # streamed experts: what prefetch reads
            monkeypatch.setenv("VAHANA_PREFETCH", str(mode))
            info = loader._prefetch(model)
            assert info["mode"] == mode
            k.prefetch_stats(reset=True)
            ids = torch.randint(1, cfg.vocab_size, (1, 20), generator=torch.Generator().manual_seed(3))
            out[mode] = decode.generate(model, ids, max_new_tokens=8, return_logits=True)
            st = k.prefetch_stats()
            if mode >= 3:
                assert st["calls"] > 0                              # a fresh model's experts are all cold
            if mode == 4:
                assert st["pred_hit"] is not None                   # the next layer was predicted in decode
    finally:
        k.set_prefetch(0)
    for mode in (1, 2, 3, 4):
        assert torch.equal(out[0]["logits"], out[mode]["logits"]) and torch.equal(out[0]["ids"], out[mode]["ids"])
