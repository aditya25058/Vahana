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

## Follow-up: a real cross-machine bug found by CI (fixed)
**Symptom.** The first CI run of this change failed on GitHub's runners for Qwen3.5 only, with three different hashes:
- `784e…`: the H100 and the laptop, and some x86 runners;
- `a05d…`: other x86 runners;
- `0469…`: ARM runners.

All other families were identical everywhere.

**Cause.** TwinMode's audit (now part of the selftest) showed that from transformers 5.17 on, Qwen3.5's gated delta rule calls `torch.linalg.solve_triangular` (6 calls in the tiny model). That LAPACK/BLAS solve blocks its sums differently by math library and CPU model (MKL on x86, OpenBLAS on ARM).
- Its float32 results differed in about 1–4% of elements from any fixed order.
- On the H100 and the laptop those differences disappeared at the bf16 roundings; on some runners they did not.

Ruled out on the H100 before the audit found it: ATen CPU capability, thread count, MKL/oneDNN ISA settings, hash seed, uninitialized memory, Vahana kernel ISA.

**Fix.**
- **`tmath.tsolve_triangular`:** substitution one row at a time, each row's sum a twin matmul (fixed halving tree), IEEE division unless unit-triangular. It is registered for `linalg_solve_triangular` in TwinMode.
- **Accuracy:** about 1e-7 relative to torch's solve.
- **Results:** the reference hashes did not change (all 13 releases re-recorded: identical). Every audit is now empty for all 18 families under 5.12.1 and 5.24.1.

**Guard.**
- `vahana selftest` now runs every family with TwinMode's audit. A family that uses any float op outside Vahana's exact math fails with **NOT EXACT** and the op's name, even when the local hash matches; `--write` refuses to record such references.
- Checked: with the new handler removed, the laptop reports `qwen3_5_moe_text … matches here, but NOT EXACT: ['linalg_solve_triangular']`.
- Tests: `tests/test_tsolve.py`.
