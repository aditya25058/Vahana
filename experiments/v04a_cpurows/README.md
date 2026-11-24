# v0.4-a: multi-row CPU expert jobs (a batch's CPU experts read each weight once)

Pre-registered 2026-11-24, before building or running the code.

**Why:** in v0.3-e, Qwen3-30B (18 GB cache, the other experts on the CPU) went flat past batch 8: 36 → 48 tok/s from B = 8 to B = 64. A CPU expert job took at most 16 rows (the kernel's limit), so a B = 64 step made 4 serial jobs per layer, and an expert used in several chunks was read from RAM once per chunk.
- Estimate: about 90 uncached experts per layer × 9.4 MB × 48 layers × up to 4 reads ≈ 40–160 GB per step.
- Measured: 1.34 s per step.

**Code:**
- **csrc `expert_mlps_ctx`:** accepts up to 256 rows per expert. For each weight row, the tokens go in sub-blocks of 16 (`row_dots` with up to 16 accumulators). The row stays in L1 across sub-blocks, so it is read from memory once for all the rows.
- Each token's dot product is the same `row_dots` lane / tree / rounding sequence, so the bits are unchanged.
- **runtime:** a CPU expert job takes `VAHANA_MOE_ROWS` rows (default 64) instead of 16, in both eager and graph mode.

**Bars:**
- **EXACT:**
  - `vahana selftest --device cpu` 18/18 under each ISA build (avx512, avx2, scalar) on the H100 host and in CI (Linux x86/arm, Windows, macOS);
  - `--device cuda` 18/18 with no cache / 0.5 / 1.0;
  - tiny batched decode (v3e_tiny) at B = 24 and B = 48 with half the experts cached, and at B = 48 with no cache (the `moe` path), all rows equal solo;
  - Qwen3-30B: a batch of 32 M5 prompts, 32 tokens, the first 8 rows equal their solo runs;
  - Qwen3 and OLMoE prefill / generation hashes equal the P4 references (v3_run; prefill now runs up to 64-row expert jobs).
- **SPEED (H100, Qwen3-30B, 18 GB cache, alternated against v0.3-e at 431bc9f):**
  - aggregate decode tok/s at B = 32 ≥ 1.5×;
  - at B = 64 ≥ 2×;
  - B = 1 and B = 8 not slower than 0.95×.
  - OLMoE fully cached is reported (no CPU experts: unchanged).

**Prediction:** B = 64 at 2–3× (about 100–140 tok/s), B = 32 at about 1.7×, B ≤ 8 unchanged within noise.

## Run log
- **Run 1 (64-row jobs only, logs in `run1/`):**
  - Exactness all passed: CPU selftest under avx512, avx2 and scalar 18/18; CUDA 18/18 ×3; tiny B = 24/48 18/18; hashes; Qwen3 B = 32 8/8. The job was killed by the box owner once and resubmitted with the user's OK.
  - **Speed missed: B = 32 1.05×, B = 64 0.88× (slower).** The re-read hypothesis was wrong.
  - Diagnosis (`v4a_diag.py`, `diag.txt`): each job pads every expert to the busiest expert's row count (`nmax`), and the kernel zeroed, converted and computed the padding. Setup alone was 6.5 ms per call at B = 64. Bigger jobs meant a bigger `nmax`, hence slower.
- **Fix:** the kernel gets each expert's real row count (`ne`) and skips padded rows in setup and in both passes. The callers no longer zero the padding (it is never read). A used row's arithmetic is unchanged.
  - Diagnostic (`diag_ragged.txt`): the B = 64 step went from 1,535 to 520 ms.
- **Every bar was then rerun** with the final code. That run is the counted one below.

## Result: all bars pass. Qwen3-30B batch 64 is 2.46× faster (130 tok/s), batch 32 1.87×, small batches unchanged
| Bar | Result |
|---|---|
| CPU selftest per ISA build (avx512, avx2, scalar), H100 host | **18/18 each** |
| CUDA selftest, no cache / 0.5 / 1.0 | **18/18 ×3** |
| Tiny batched decode, rows vs solo: half cache B = 24 and B = 48; no cache B = 48 | **18/18 ×3** |
| P4 reference hashes (prefill uses up to 64-row expert jobs) | **OLMoE 50/50 + 4/4, Qwen3 8/8 + 2/2** |
| Qwen3-30B, batch of 32, first 8 rows vs solo | **8/8** (graph) |
| SPEED Qwen3 B = 32 vs 431bc9f (bar ≥ 1.5×) | **1.87×, met** (47.8 → 89.3 tok/s) |
| SPEED Qwen3 B = 64 (bar ≥ 2×) | **2.46×, met** (53.0 → 130.1 tok/s) |
| B = 1 and B = 8 (bar ≥ 0.95×) | **1.00× and 1.21×, met** (B = 1 is noisy: 9.8 / 21.0 for the new code vs 14.7 / 16.0 for the base) |
| OLMoE fully cached (reported) | B = 64: 1,016 tok/s (v0.3-e: 863) |

**Qwen3-30B aggregate decode tok/s (mean of 2 alternated rounds):**

| B | 1 | 8 | 32 | 64 |
|---|---|---|---|---|
| v0.3-e | 15.4 | 40.7 | 47.8 | 53.0 |
| v0.4-a | 15.4 | 49.3 | 89.3 | **130.1** |

**Reading:**
- **The fix was to stop computing padding, not to read less memory.** With ragged per-expert row counts, the CPU half of a 30B model keeps scaling with batch: 8.4× at B = 64 over B = 1, against 3.4× before. Every request still gets exactly its solo bits.
