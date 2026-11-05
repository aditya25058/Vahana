"""Measurement-protocol plumbing: metric definitions, env snapshot, page cache."""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch

from bhaskera.inference.colossus.runinfo import snapshot
from bhaskera.inference.colossus.serve import serve_huge_moe
from bhaskera.inference.colossus.tests.test_serve import (
    FakeCausalMoE,
    FakeProfile,
    FakeTokenizer,
    _stub_handles,
)

_RUDRA = os.path.join(os.path.dirname(__file__), *[".."] * 5, "rudra")


def test_decode_tps_excludes_prefill_token():
    model = FakeCausalMoE()
    tok = FakeTokenizer()
    tok.eos_token_id = -1
    res = serve_huge_moe(model, tok, FakeProfile(), _stub_handles(model),
                         torch.device("cpu"), ["ab", "cd"], max_new_tokens=4,
                         capacity=8)
    assert res["generated_tokens_per_row"] == [4, 4]
    assert res["generated_tokens_total"] == 8 and res["decode_tokens"] == 6
    assert len(res["step_ms"]) == 3
    d = res["decode_s"]
    assert res["batch_decode_tps"] == pytest.approx(6 / d)
    assert res["legacy_batch_decode_tps"] == pytest.approx(8 / d)
    assert res["e2e_tps"] == pytest.approx(8 / (res["prefill_s"] + d))
    assert "batch_decode_tps" in res["metric_definitions"]
    # Per-step load deltas: one per decode step, summing to the run totals
    # minus what prefill loaded.
    sl = res["step_loads"]
    assert len(sl) == 3 and set(sl[0]) == {"misses", "dma_bytes", "copy_s", "minflt"}
    assert all(s["misses"] >= 0 and s["dma_bytes"] >= 0 for s in sl)
    assert sum(s["misses"] for s in sl) <= res["total_misses"]


def test_env_snapshot_never_raises():
    s = snapshot()
    assert {"host", "git_commit", "torch", "gpu", "mem_gb", "loadavg"} <= set(s)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="mincore/fadvise")
def test_pagecache_residency_and_evict(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "pagecache", os.path.join(_RUDRA, "pagecache.py"))
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    p = tmp_path / "x.safetensors"
    p.write_bytes(os.urandom(1 << 20))
    with open(p, "rb") as f:
        f.read()  # just written + read: resident
    r = pc.residency(str(tmp_path))
    assert r["total_bytes"] == 1 << 20 and 0.0 <= r["resident_frac"] <= 1.0
    e = pc.evict(str(tmp_path))
    assert e["evicted_shards"] == 1 and e["resident_frac"] <= r["resident_frac"]
