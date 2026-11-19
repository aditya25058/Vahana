import importlib.util
import os
from collections import Counter

from vahana.hostreg import RegisteredHandles, merge_page_spans

_SEL = os.path.join(os.path.dirname(__file__), "../../../../../rudra/select_pin_experts.py")


def _sel_mod():
    spec = importlib.util.spec_from_file_location("select_pin_experts", _SEL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_merge_page_spans_shares_pages_and_keeps_gaps():
    P = 4096
    spans = [(10, 5000), (5000, 9000), (20000, 20010), (100, 200)]
    assert merge_page_spans(spans, P) == [(0, 12288), (16384, 20480)]


def test_merge_page_spans_adjacent_pages_merge():
    assert merge_page_spans([(0, 4096), (4096, 8192)], 4096) == [(0, 8192)]


def test_key_parsing_and_selection():
    k = "model.layers.15.mlp.experts.37.down_proj.weight"
    assert RegisteredHandles.layer_of(k) == 15 and RegisteredHandles.expert_of(k) == 37
    h = RegisteredHandles.__new__(RegisteredHandles)
    h.layers, h.experts = {5}, {(15, 37)}
    assert h.selected(k)
    assert h.selected("model.layers.5.mlp.experts.0.up_proj.weight")
    assert not h.selected("model.layers.15.mlp.experts.36.down_proj.weight")
    assert not h.selected("model.layers.5.mlp.shared_experts.up_proj.weight")


def test_select_counts_and_ties():
    m = _sel_mod()
    c = m.count_routing([{"layer": 1, "experts": [3, 4]}, {"layer": 1, "experts": [4]},
                         {"layer": 2, "experts": [0]}])
    assert c == Counter({(1, 4): 2, (1, 3): 1, (2, 0): 1})
    assert m.select(c, 2) == [(1, 4), (1, 3)]


def test_margin_guard_counts_pinned_bytes():
    from vahana.hostreg import within_margin
    G = 10**9
    assert within_margin(469 * G, 0, 400 * G, 64 * G)
    assert not within_margin(469 * G, 400 * G, 10 * G, 64 * G)
    assert within_margin(469 * G, 400 * G, 5 * G, 64 * G)
