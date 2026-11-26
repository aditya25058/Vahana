"""v1.1: per-transformers-version selftest references (no models needed)."""
import json
import os

from vahana.selftest import GOLDEN, _ver, expected_hashes

GOLD = json.load(open(GOLDEN))


def test_version_parse():
    assert _ver("5.14.1") == (5, 14, 1)
    assert _ver("5.14") == (5, 14, 0)
    assert _ver("5.20.0.dev0") == (5, 20, 0)
    assert _ver("5.14.0rc1") == (5, 14, 0)


def test_every_family_has_a_reference_in_the_validated_range():
    fams = GOLD["families"]
    for v in GOLD["_validated"]:
        exp, note = expected_hashes(GOLD, v)
        assert note is None, v
        assert set(exp) == set(fams), v


def test_reference_changes_at_the_recorded_releases():
    q = GOLD["families"]["qwen3_5_moe_text"]
    a, _ = expected_hashes(GOLD, "5.13.1")
    b, _ = expected_hashes(GOLD, "5.14.0")
    c, _ = expected_hashes(GOLD, "5.17.0")
    assert a["qwen3_5_moe_text"] == q[0]["hash"] != b["qwen3_5_moe_text"] == q[1]["hash"] != c["qwen3_5_moe_text"]
    assert a["mixtral"] == b["mixtral"] == c["mixtral"]


def test_note_outside_the_validated_range():
    _, note = expected_hashes(GOLD, "99.0.0")
    assert note and "outside the validated range" in note


def test_flat_1_0_format_still_reads():
    exp, note = expected_hashes({"_transformers": "5.12.1", "mixtral": "abc"}, "5.13.1")
    assert exp == {"mixtral": "abc"} and note
