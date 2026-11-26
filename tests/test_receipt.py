"""v1.2: Vahana Receipts (verifiable inference) on tiny selftest models."""
import copy

import numpy as np
import pytest
import torch

from vahana import decode, loader, receipt, selftest


def test_texp_accuracy():
    z = np.concatenate([np.linspace(-745, 0, 20001), -np.logspace(-12, 2, 2001)])
    rel = np.abs(receipt.texp(z) - np.exp(z)) / np.maximum(np.exp(z), 1e-300)
    assert rel[np.exp(z) > 1e-300].max() < 1e-14
    assert receipt.texp(np.array([-800.0]))[0] == 0.0


def test_sampler_is_a_pure_function_and_roughly_right():
    rs = np.random.RandomState(0)
    lg = rs.standard_normal(50).astype(np.float32) * 2
    a = [receipt.sample_row(lg, 0.8, 0.9, receipt._uniform(7, i, 0)) for i in range(4000)]
    b = [receipt.sample_row(lg, 0.8, 0.9, receipt._uniform(7, i, 0)) for i in range(4000)]
    assert a == b
    p = np.exp(lg.astype(np.float64) / 0.8)
    p /= p.sum()
    top = int(np.argmax(p))
    assert abs(a.count(top) / 4000 - p[top] / p[np.argsort(-p)][np.cumsum(p[np.argsort(-p)]) - p[np.argsort(-p)]
                                                     < 0.9].sum()) < 0.04


def _tiny(fam):
    cfg, m = selftest.build(fam)
    model, _ = loader.prepare(m)
    ids, _ = selftest.prompts(cfg.vocab_size)
    return model, ids[1:2, -20:]


@pytest.mark.parametrize("fam,temp", [("olmoe", 0.0), ("qwen3_moe", 0.7), ("qwen3_5_moe_text", 0.0)])
def test_receipt_roundtrip_and_tamper(fam, temp):
    model, p = _tiny(fam)
    smp = receipt.make_sampler(temp, 0.95, 3)
    r = decode.generate(model, p, max_new_tokens=6, return_logits=True, sampler=smp, graph=False)
    rec = receipt.make(model, None, p, r, 6, None, smp, fp={"sha256": "x", "mode": "headers"})
    v = receipt.verify(model, rec)
    assert v["ok"], v
    assert v["mode"] == ("replay" if fam == "qwen3_5_moe_text" else "prefill")

    t = copy.deepcopy(rec)                       # a changed answer, receipt id recomputed: logits/tokens disagree
    t["output_ids"][3] = (t["output_ids"][3] + 1) % model.config.vocab_size
    t["receipt_id"] = receipt._body_id(t)
    v = receipt.verify(model, t)
    assert not v["ok"] and v["first_bad_step"] is not None and v["first_bad_step"] <= 4

    t = copy.deepcopy(rec)                       # an edited receipt without a new id
    t["prompt_ids"][0] += 1
    assert not receipt.verify(model, t)["checks"]["intact"]
