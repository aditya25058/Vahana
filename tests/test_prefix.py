"""v1.3: the prefix cache gives the full prefill's logits and tokens, bit for bit."""
import pytest
import torch

from vahana import decode, loader, prefix, selftest


@pytest.mark.parametrize("fam", ["olmoe", "qwen3_moe", "deepseek_v3", "gpt_oss", "glm4_moe"])
def test_prefix_cache_is_exact(fam, tmp_path):
    cfg, m = selftest.build(fam)
    model, _ = loader.prepare(m)
    g = torch.Generator().manual_seed(5)
    doc = torch.randint(1, cfg.vocab_size, (1, 40), generator=g)
    q1 = torch.cat([doc, torch.randint(1, cfg.vocab_size, (1, 7), generator=g)], 1)
    q2 = torch.cat([doc, torch.randint(1, cfg.vocab_size, (1, 11), generator=g)], 1)
    ref1 = decode.generate(model, q1, max_new_tokens=5, return_logits=True, graph=False)
    ref2 = decode.generate(model, q2, max_new_tokens=5, return_logits=True, graph=False)
    pc = prefix.PrefixCache(str(tmp_path), model_key=fam, block=16)
    a = decode.generate(model, q1, max_new_tokens=5, return_logits=True, graph=False, prefix_cache=pc)
    assert pc.last == {"hit_tokens": 0, "stored_tokens": 32}
    b = decode.generate(model, q2, max_new_tokens=5, return_logits=True, graph=False, prefix_cache=pc)
    assert pc.last["hit_tokens"] == 32                   # the shared 40-token document, cut at a 16-token block
    pc2 = prefix.PrefixCache(str(tmp_path), model_key=fam, block=16)      # a new process: from disk
    c = decode.generate(model, q1, max_new_tokens=5, return_logits=True, graph=False, prefix_cache=pc2)
    assert pc2.last["hit_tokens"] == 32
    for r, x in ((ref1, a), (ref2, b), (ref1, c)):
        assert torch.equal(r["ids"], x["ids"]) and torch.equal(r["logits"], x["logits"])


def test_other_model_key_misses(tmp_path):
    cfg, m = selftest.build("olmoe")
    model, _ = loader.prepare(m)
    q = torch.randint(1, cfg.vocab_size, (1, 40), generator=torch.Generator().manual_seed(0))
    decode.generate(model, q, max_new_tokens=2, graph=False, prefix_cache=prefix.PrefixCache(str(tmp_path), "a", 16))
    pc = prefix.PrefixCache(str(tmp_path), "b", 16)
    decode.generate(model, q, max_new_tokens=2, graph=False, prefix_cache=pc)
    assert pc.last["hit_tokens"] == 0


def test_model_key_separates_lookalike_checkpoints(tmp_path):
    """v1.4: a fine-tune with the same shapes must not reuse its base model's cached prefixes."""
    import os
    import shutil
    import time

    import torch
    from safetensors.torch import save_file

    from vahana import prefix
    a = tmp_path / "a"
    a.mkdir()
    (a / "config.json").write_text('{"model_type": "x"}')
    save_file({"w": torch.zeros(4, 4)}, str(a / "model.safetensors"))
    k1 = prefix.model_key(str(a))
    assert prefix.model_key(str(a)) == k1                           # stable
    b = tmp_path / "b"
    shutil.copytree(a, b)
    assert prefix.model_key(str(b)) != k1                           # same headers, another checkpoint
    time.sleep(0.01)
    save_file({"w": torch.ones(4, 4)}, str(a / "model.safetensors"))   # same shapes, new weights in place
    os.utime(a / "model.safetensors", ns=(time.time_ns(), time.time_ns() + 10**9))
    assert prefix.model_key(str(a)) != k1
