# v0.3-c: fewer, fused exact kernels (router softmax, grouped experts)

Pre-registered 2026-11-24, after profiling and before writing the kernels.

**Profile (v3c_prof.py, GPU-resident OLMoE, one forward, CUDA kernel time):**

| | 1 token | 5 tokens |
|---|---|---|
| Total | 7.65 ms (1,820 kernels) | 10.71 ms |
| Elementwise glue | 2.89 ms (1,384 kernels, about 86 per layer) | 3.37 ms |
| Twin linears | 2.00 ms | 2.32 ms |
| Cached-expert kernels (`_cgate_up` + `_cdown`) | 1.47 ms | 3.64 ms |

The glue is spread over many small ops (v3c_prof2.py), much of it the float64 Cephes `exp` inside the TwinMode router softmax.

**Plan (each exact by construction, and checked):**
1. **Fused twin softmax (tmath):** one Triton program per row performs the same operations as `tsoftmax`, in the same order and precision:
   - fp32 max; `x − max` in fp32, widened to float64;
   - the Cephes `exp` in float64 (the same constants, each operation rounded as torch rounds it);
   - narrowing to fp32;
   - the halving-tree sum over the row padded to a power of two;
   - fp32 division.

   It replaces the pure-torch path on CUDA for last-dimension softmax.
2. **Grouped cached-expert kernels (cache.py):**
   - the N·K slots are sorted by cached expert;
   - the program of the first slot of each run computes every slot of that run with one load of the weight tile;
   - the other programs exit.

   Per-slot arithmetic (16 fma lanes, k ascending, tree16) is unchanged.

**Bars:**
- **UNIT:** the fused softmax equals the pure-torch twin softmax (GPU) and the CPU twin softmax bit for bit, on random rows of widths 1, 7, 64, 128, 1000 and 50,304 (fp32 and bf16 in; fp32 and bf16 out; edge values including −inf entries and all-equal rows).
- **EXACT:**
  - `vahana selftest --device cuda` 18/18 with no cache and with `--gpu-cache 0.5` / `1.0`;
  - tiny speculative exactness 16/16 (v3b_tiny.py);
  - OLMoE and Qwen3-30B with the GPU cache equal the P4 reference hashes (v3_run.py, as in v0.3-a).
- **SPEED (H100, same session, alternated against the v0.3-b code at 85216b8):**
  - OLMoE fully cached, plain graph decode: tok/s ≥ 1.15×.
  - Qwen3-30B, 18 GB cache: decode tok/s reported.
  - OLMoE speculative (k = 4) on the 4 copy-heavy prompts: ≥ 1.3× over plain decode in the same run (v3b_real.py).

**Prediction:**
- The fused softmax removes about 30–40 kernels per layer: OLMoE decode 1.1–1.3×.
- Grouped experts make the 5-token verify about 1.2× a single step.
- Copy-heavy speculation lands at 1.3–1.6×.

## Run log (before the counted runs)
- **Float64 division:** `tl.math.div_rn` is fp32-only. The exp kernel's float64 quotient uses `/`, which PTX always compiles to `div.rn.f64` (correctly rounded).
- **Width-1 edge rows (laptop UNIT):** an all `-inf` row gives NaN, and the NaN bit pattern differed by device: GPU 0x7fffffff, CPU 0xffc00000. After the bf16 cast it was 0x7fff vs 0xffff. This was already the case in the pure-torch twin softmax before this milestone.
  - Fix: `tsoftmax` (both paths) now writes one canonical NaN after the cast to the output dtype (0x7fc00000 / 0x7fc0).
  - This changes only NaN outputs; every golden hash is unchanged.
- **Triton 3.0 (H100) compile error:** comparing bf16 values does not compile there. The NaN test is now done on the fp32 value before the cast; it is NaN before the cast exactly when it is NaN after.
- **Triton 3.0 flushes subnormals in libdevice:** it compiles libdevice in FTZ mode (`set_nvvm_reflect_ftz`), so `tl.math.div_rn` / `sqrt_rn` flushed subnormal values to 0. Newer Triton (laptop) and the CPU do not.
  - Seen as: one H100 UNIT case, where exp(−101.6) = 7e-45 came out as 0.
  - This was a latent cross-version gap in every twin kernel using those functions (twin linear RMS norm, SiLU in the experts, attention softcap / sigmoid / normalisation).
  - Fix: `_triton.div_rn` / `sqrt_rn` emit PTX `div.rn.f32` / `sqrt.rn.f32` themselves, and every twin kernel uses them. Golden hashes are unchanged (no subnormals occur there).
- **Harness slip (Qwen3):** the first Qwen3 run generated 24 tokens. The P4 reference has 16, so the generation-logit hashes could not match, even though the first 16 ids did. It was rerun at 16 (`run_exact.sh` fixed). The counted Qwen3 result is the rerun.

## Result: UNIT and EXACT pass everywhere; SPEED not met (OLMoE 1.09× against ≥ 1.15×; speculation 0.84× against ≥ 1.3×)
| Bar | Result |
|---|---|
| UNIT, H100 (Triton 3.0) and laptop RTX 3050 Ti (triton-windows 3.8) | **72/72 softmax cases** (widths 1 to 50,304; fp32/bf16 in and out; edge rows). Grouped experts equal per-slot on 6 routing shapes (N = 1 to 300) |
| `vahana selftest --device cuda`, no cache / 0.5 / 1.0 | **18/18, 18/18, 18/18** on the H100 and on the laptop |
| Tiny speculative exactness (v3b_tiny) | **16/16** on the H100, the H100 with half the experts cached, and the laptop |
| Real checkpoints vs the P4 reference hashes (16 GB cache) | **OLMoE 50/50 prefills, 4/4 generations; Qwen3-30B 8/8, 2/2** |
| SPEED: OLMoE fully cached (1,024 experts, 12 GB), plain graph decode, alternated with 85216b8 (bar ≥ 1.15×) | **1.089×, not met**: 76.1 → 82.9 tok/s (8 generations of 64 tokens per side; single runs vary 70–90) |
| Qwen3-30B, 18 GB cache (reported) | **1.12×**: 12.1 → 13.6 tok/s |
| OLMoE speculative k = 4 vs plain in the same run, copy-heavy (bar ≥ 1.3×) | **0.84×, not met** (67.6 vs 80.9 tok/s); generic 0.73×. The 85216b8 code in the same session: 0.96× / 0.90× |

All softmax calls in the 18 selftest families go through the fused kernel (201 calls, 0 fallbacks).

**Why the speculative pass got slower:** the counted grouped kernels kept a 3-D (group × rows × lanes) accumulator. A microbenchmark (OLMoE shapes, H100, µs per layer call) shows those tiles are slower than the per-slot kernels until about 64 tokens:

| tokens | per-slot | 3-D groups of 8 | 2-D groups of 8 | 2-D groups of 2 |
|---|---|---|---|---|
| 1 | 272 / 240 | 387 | 298 | 254 |
| 5 | 596 / 514 | 1,325 | 834 | **429** |
| 8 | 642 / 824 | 1,188 | 1,197 | **672** |
| 16 | 1,519 / 1,500 | 2,171 | 1,709 | **1,159** |
| 64 | 5,128 / 5,456 | 3,112 | **2,758** | 2,833 |

(Per-slot shows two runs: the first with the 3-D groups, the second with the 2-D groups.)

**Follow-up (after the counted runs, not counted):**
- **2-D tiles:** tile row j = member j // BR, weight row j % BR. A member's repeated weight loads hit L1. The group size is 2 below 32 tokens and 8 from 32 up.
- **Exactness:** UNIT passes for group sizes 2, 4 and 8 (H100 and laptop); selftest 18/18 (no cache and 1.0); tiny speculative 16/16 (H100 and laptop).
- **OLMoE plain decode:** 1.09× (75.1 → 81.9).
- **Speculation:** 0.89× copy-heavy, 0.87× generic (per prompt 0.62–1.09×).

**Reading:**
- **Kernel launches were the cheap part.** Removing about 40 kernels per router softmax gives about 1.09× (OLMoE) and 1.12× (Qwen3), short of the predicted 1.1–1.3× range for OLMoE.
- **The bigger finding is the FTZ gap.** Exactness across Triton versions had a hole that no golden test could see (subnormals). It is closed by issuing the PTX instructions directly. Vahana now defines its GPU divisions and square roots without depending on how a given Triton build compiles libdevice.
- **Speculation still doesn't pay.** Grouped experts help the verify pass from 5 tokens up, but each token still re-reads every dense weight in the twin linears (attention and router projections, `_lin`: about 2.0 ms of the 7.65 ms step) and runs its own glue.
- **Next milestone:** the multi-token twin linear. One weight tile is applied to every token, with each token's lanes and tree unchanged. That is where the verify pass and prefill lose their time.
