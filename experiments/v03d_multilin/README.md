# v0.3-d: multi-token twin linear (one weight load for several tokens, the same bits)

Pre-registered 2026-11-24, after the design bench (v3d_bench.py, bench_h100.txt) and before the counted runs.

**Why:** v0.3-c showed that the per-token twin linear (`_lin`, grid rows × tokens) is where verify passes and prefill lose time. Every token re-reads every dense weight: for example, the 5-token OLMoE `lm_head` takes 0.79 ms against 0.20 ms for one token.

**Code (twinlin.py):**
- `_linm` keeps MT separate accumulators (MT = 2, 4 or 8). Each weight tile (BR rows × 16 lanes) is loaded once and applied to MT tokens.
- Each token's arithmetic is the per-token kernel's: 16 fma lanes, k ascending, the same tree16, bf16 RNE. So the bits are unchanged by construction; the launch shape only changes speed.
- **Policy (from the bench):**
  - one token keeps `_lin`;
  - otherwise MT = 8 when n ≥ 32 or (R ≥ 8192 and n ≥ 4), else MT = 2;
  - BR = the per-token BR × 4 for MT = 8 (capped at 32 and dividing R);
  - num_warps 1 for MT = 8, 4 for MT = 2.
- **Design bench (H100, µs per call, outside CUDA graphs):** all 35 cells × every variant are bit-equal to `_lin`.
  - verify-sized (5 tokens): OLMoE qkvo 1.3×, `lm_head` 2.5×, Qwen3 q 1.3×;
  - prefill-sized (512 tokens): 2.6–3.7×.

**Bars:**
- **UNIT:** `gpu_linear` equals `_lin` bit for bit on 6 weight shapes (including R = 50,304 and K = 4,096) × n ∈ {2, 3, 5, 8, 17, 33, 64, 333}, and for forced MT ∈ {2, 4, 8} (H100 Triton 3.0 and laptop Triton 3.8).
- **EXACT:**
  - `vahana selftest --device cuda` 18/18 (no cache / 0.5 / 1.0), H100 and laptop;
  - tiny speculative 16/16;
  - OLMoE (50 prefills, 4 generations) and Qwen3-30B (8 prefills, 2 generations) equal the P4 reference hashes. Prefill runs the multi-token path.
- **SPEED (H100, same session, alternated against v0.3-c at 0d49551):**
  - OLMoE fully cached, prefill time over the 50 prompts: ≥ 1.5× faster;
  - OLMoE plain graph decode: ≥ 0.97× (the one-token path is unchanged);
  - OLMoE speculative k = 4 vs plain in the same run: copy-heavy ≥ 1.2×, generic ≥ 0.95×;
  - Qwen3-30B (18 GB cache): prefill and speculation reported.

**Prediction:**
- prefill 1.5–2.5× (the experts' CPU/cache path and the glue are untouched);
- copy-heavy speculation 1.1–1.4×;
- generic speculation about 0.9–1.0×.

## Result: UNIT and EXACT pass everywhere; SPEED not met except decode (prefill 0.77× by the pre-registered measure, 1.16× warm; speculation 0.93×)
| Bar | Result |
|---|---|
| UNIT: multi-token == per-token `_lin`, H100 (Triton 3.0) and laptop (Triton 3.8) | **192/192** on both |
| `vahana selftest --device cuda`, no cache / 0.5 / 1.0 | **18/18 ×3** on the H100 and on the laptop |
| Tiny speculative exactness | **16/16** (H100, H100 with half the experts cached, laptop) |
| Real checkpoints vs P4 hashes (prefill runs the multi-token path) | **OLMoE 50/50 + 4/4** (both rounds), **Qwen3-30B 8/8 + 2/2** |
| SPEED: OLMoE prefill over 50 prompts, mean of 2 alternated rounds (bar ≥ 1.5×) | **0.77×, not met** (base 2.77 / 2.30 s, new 4.59 / 1.99 s). Round 1 of the new code includes the first-ever Triton compile of `_linm` (base kernels were already in the disk cache). Round 2 alone: **1.16×** |
| OLMoE plain graph decode (bar ≥ 0.97×) | **1.016×, met** (121.8 → 123.8 tok/s; both codes ran about 2× faster in round 2, so machine state shifted during the session) |
| OLMoE speculative vs plain, same run (bars ≥ 1.2× copy-heavy, ≥ 0.95× generic) | **0.93× and 0.86×, not met.** The 0d49551 code in the same session: 0.92× / 0.87× |
| Qwen3-30B (reported) | prefill 1.18× (6.15 → 5.22 s, single run each); speculation 0.73× copy-heavy, 0.65× generic |

**Where a verify pass spends its time now (v3d_prof.py, OLMoE fully cached, eager forward, GPU kernel ms):**

| tokens in the pass | total | twin linears `_lin`/`_linm` | cached experts (`_cgate_up` + `_cdown`) | elementwise glue |
|---|---|---|---|---|
| 1 | 5.46 | 1.83 | 1.38 | 1.18 |
| 5 | 7.78 | **1.76** | **3.38** | 1.45 |
| 12 | 10.30 | **2.21** | **5.33** | 1.44 |

**Reading:**
- **The multi-token linear works and stays exact.** Dense-linear time barely grows with tokens per pass (1.83 → 1.76 → 2.21 ms for 1 → 5 → 12 tokens; before, 5 tokens cost about 2.3×). The bits are identical everywhere tested.
- **The wall is now the experts, and it is inherent to MoE at batch 1.** Five tokens route to about 30 distinct experts (8 of 64 each) instead of 8, and every distinct expert's weights must be read once, so expert time grows with distinct experts, not with tokens. Speculation therefore buys little on small-batch MoE: the verify pass costs about 1.4× a step in kernel time, and the measured 2.25 tokens per step on copy-heavy prompts doesn't cover that plus the per-step host work (draft lookup, cache-length resets, sync).
- **Prefill is the same story.** At 14–30 tokens per prompt, nearly every expert is touched, and the expert path dominates.
- **What would move MoE speculation:** drafts that reuse the same experts (expert-aware drafting), or batching several requests so expert reads are shared. Both are serving-level changes, not kernel changes. Prefill also gains from batching prompts.
