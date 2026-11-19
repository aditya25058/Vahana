# M2: kernel generalisation (activations, biases, fp32 routing weights, non-64 widths, long prefill)

Pre-registered 2026-11-20, after the code compiled and before any test run.

**Changes:**
- **csrc/cpu_moe.cpp** (patch_m2.py):
  - `ActCtx` carries a per-expert activation (0 SiLU-GLU, 1 gpt-oss clamped SwiGLU) and optional biases, through all four expert paths (raw single-token, raw batched, NX12, fused v16).
  - `ctx == nullptr` keeps the original arithmetic character for character.
  - Biases are added after the dot product's BF16 rounding, then rounded again, as stock does.
  - `register_layer_act`.
  - fp32 routing weights are used as given.
- **modelspec.py:** recognises the `fused_interleaved_t` layout (`[E, H, 2I]`, gate/up interleaved, biases, `alpha` / `limit`): the gpt-oss experts.
- **experts.py:**
  - layouts and biases;
  - routing weights passed as given (M1 rounded them to BF16);
  - CPU jobs chunked to 16 tokens (the batched kernel's per-expert limit).
- **twin_act.py:** the GPU twin of the general path (`dot32`, `act`, `down`).

**Families:**
- the 17 Tier 1 families;
- **gpt_oss** (tiny, BF16 experts; its all-zero default router weights are randomised so routing varies);
- **mixtral_i96** (expert width 96, not a multiple of 64, so the raw kernel path runs instead of NX12).

The prompt is 40 tokens (exercises chunking and more than 16 tokens per call), followed by 8 greedy steps.

**Bars:**
- **EXACT-1:** for all 19 cases, CPU == GPU twin bit for bit (prefill logits at 40 positions, 8 decode-step logits, 8 ids).
- **EXACT-3:** the CPU arm is identical at OMP 8 and OMP 24 threads.
- **NO-CHANGE:** on the Phase 41 microbench (real DeepSeek-V2 236B layers, BF16 routing weights, no context), the M2 kernel's outputs are bit-identical to the pre-M2 kernel's (`.pt` comparison).
- **SPEED:** on the same microbench at 24 threads, M2's MoE layer time is within ±3% of pre-M2. Old/new are run alternately ×2, and medians are compared.
- **SEMANTIC:** twin vs stock prefill logits ≤ 5% of max |stock| for gpt_oss and mixtral_i96. The M1 families are reported again with fp32 routing weights.

**Prediction:** all pass. Twin vs stock should shrink a little now that routing weights are no longer rounded.

## Result: EXACT-1, EXACT-3, NO-CHANGE, SPEED, SEMANTIC all PASS
**Kernel microbench** (bench/, Phase 41 bench.py; 4 real DeepSeek-V2 236B layers, 24 experts each + shared + o_proj; 24 threads; old/new alternated ×2):

| | pre-M2 | M2 |
|---|---|---|
| MoE layer call, kernel median (rep 2) | 1.636 / 1.641 ms | 1.649 / 1.648 ms (**+0.6%**, bar ±3%) |
| o_proj NX12 | 0.660 / 0.663 ms | 0.668 / 0.661 ms |
| Outputs (MoE, o_proj NX12, o_proj raw) | — | **bit-identical to pre-M2** |

**Families** (m2_omp24.json / m2_omp8.json): 40-token prompt + 8 greedy steps.
- **19/19 exact:** CPU == GPU twin bit for bit on prefill logits, decode logits and ids.
- **19/19 identical** at 8 and 24 threads.

| Case | E / I / H | Twin vs stock (logits) | Greedy ids equal to stock |
|---|---|---|---|
| **gpt_oss** (interleaved transposed weights, biases, clamped SwiGLU) | 8/512/256 | **0.97%** | 8/8 |
| **mixtral, I = 96** (raw path, no NX12) | 8/96/256 | **0.34%** | 8/8 |
| mixtral | 8/512/256 | 0.61% | 8/8 |
| qwen2_moe / qwen3_moe / olmoe | | 0.69% / 1.2% / 2.3% | 8/8 each |
| deepseek_v2 / v3 / glm4_moe / glm4_moe_lite | | 0.58% / 0.80% / 0.72% / 0.57% | 8/8 each |
| minimax / minimax_m2 / hunyuan / ernie | | 0.59% / 0.61% / 0.67% / 0.65% | 8/8 each |
| cohere2 / exaone / afmoe | | 2.7% / 0.56% / 0.51% | 8 / 6 / 0 of 8 |
| phimoe | | **8.8%** | 8/8 |
| qwen3_5_moe_text | | **15.7%** | 8/8 |

**Reading:**
- **The kernel now covers the gpt-oss expert form exactly:** biases, clamped SwiGLU, interleaved and transposed weights.
- **Widths that are not multiples of 64 run exactly on the raw path.**
- **Prefill longer than 16 tokens per expert runs exactly,** through chunking.
- **Existing models are untouched** (identical bits) and **no slower** (+0.6%).
- **The prediction (twin vs stock shrinks with fp32 routing weights) held only partly.** With 40-token prompts instead of 12, phimoe (8.8%) and qwen3_5 (15.7%) moved further from stock. M2's SEMANTIC bar covers only gpt_oss and mixtral_i96; these two are reported, not barred.
- **Greedy-id agreement with stock is a weak signal on tiny random models:** afmoe agrees on 0 of 8 steps at 0.51% logit distance. Near-ties dominate.
- **Real-weight quality against stock fp32 (M5) remains the deciding test.**
- **Out of M2:** gpt-oss's real checkpoints are MXFP4 (needs dequantise-on-load or an MXFP4 kernel path); Llama 4 and Granite need their own adapters.
