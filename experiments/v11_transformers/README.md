# v1.1: transformers 5.12–5.24 with per-version selftest references

## Problem
1.0 pinned transformers to 5.12–5.13. CI's compat job showed that later releases changed two families' selftest hashes, so users with a newer transformers saw a failing selftest or a dependency conflict.

## Finding
No family errors on any release. The differences are transformers model-code changes, identical on every machine:
- **Hashes recorded on the H100 box's CPU** (Linux, AMD EPYC, AVX-512, Python 3.12, transformers 5.12.1 → 5.24.1).
- **5.20.2 cross-check:** the hashes equal the laptop's CPU *and* GPU (Windows, Intel AVX2, RTX 3050 Ti).
- **5.12.1 cross-check:** the hashes equal the shipped 1.0 reference, which was recorded on the H100 GPU.

| family | 5.12–5.13 | 5.14–5.16 | 5.17–5.18 | 5.19–5.24 |
|---|---|---|---|---|
| qwen3_5_moe_text | `049869a7…` | `bc9f04c7…` | `784e0096…` | `784e0096…` |
| gpt_oss | `4385e7be…` | `4385e7be…` | `4385e7be…` | `f3548c33…` |
| other 16 families | unchanged | unchanged | unchanged | unchanged |

## Change
- **`golden_tiny.json`:** per-family `{"since", "hash"}` entries plus `_validated` (the 13 releases), built by `tools/dev/golden_versions.py` from per-version `vahana selftest --write` recordings.
- **`vahana selftest`:** picks the reference for the installed version, and notes only when transformers is outside the validated range.
- **`pyproject.toml`:** `transformers>=5.12,<5.25`.
- **CI `compat`:** all 13 releases are now required; a `latest` canary installs the newest transformers without failing the build, and its notice prints each differing family's new hash.

## Results
- **Laptop, full selftest:** 18/18 on CPU and GPU, under transformers 5.12.1 and 5.20.2.
- **`tests/test_selftest_versions.py`:** 5/5 (version parsing, per-version selection, out-of-range note, 1.0-format compatibility).
