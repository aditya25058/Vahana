# M1: the generic experts adapter on tiny models

Pre-registered 2026-11-20, before any GPU run. (tests/test_spec.py already passed: 20/20, CPU only.)

**Code:**
- `vahana/modelspec.py`: finds every experts module by structure (the `experts(hidden_states, top_k_index, top_k_weights)` seam, `gate_up_proj [E, 2I, H]` / `down_proj [E, H, I]`, SiLU-GLU). Anything else raises UnsupportedModel with the class name.
- `vahana/experts.py`, `VahanaExperts`: the drop-in module; CPU kernel, weights in CPU RAM, NX12 for single-token calls.
- `vahana/experts.py`, `TwinGPUExperts`: the same arithmetic on the GPU, used as the reference.
- `install(model, mode)` replaces every experts module.

**Families:** the 17 Tier 1 families:
- mixtral, qwen2_moe, qwen3_moe, olmoe;
- deepseek_v2, deepseek_v3, glm4_moe, glm4_moe_lite;
- minimax, minimax_m2, hunyuan_v1_moe, ernie4_5_moe;
- cohere2_moe, exaone_moe, afmoe, phimoe, qwen3_5_moe_text.

**Test setup** (m1_families.py):
- Each model is the family's own config shrunk to H=256, 8 experts, top-2, 2 layers, vocab 512, with random BF16 weights (seeded).
- The prompt is 12 tokens, followed by 8 greedy decode steps.
- The rest of the model (attention, router, shared experts) runs as stock transformers code on the GPU in all arms.

**Bars:**
- **SPEC (already met):** all 17 families discovered; gpt_oss, llama4_text and granitemoe raise UnsupportedModel.
- **EXACT-1:** for every family, the CPU arm (`VahanaExperts`) equals the GPU twin reference (`TwinGPUExperts`) bit for bit:
  - prefill logits at all 12 positions;
  - all 8 decode-step logits;
  - the 8 generated ids.
- **EXACT-3** (thread invariance): the CPU-arm logits at OMP 24 threads are bit-identical to those at OMP 8 threads (two processes).
- **SEMANTIC** (catches wiring errors such as swapped gate/up, which EXACT-1 alone would miss): the twin reference vs stock transformers prefill logits differ by at most 5% of max |stock logit|.

**Deferred** (they need later milestones):
- batch / padding invariance needs the invariant attention function (M3);
- placement invariance needs GPU-budget wiring (M4/M5).

**Known difference from stock**, by design: Vahana rounds routing weights to BF16 and combines in fp32 in ascending expert id. Stock transformers accumulates in BF16 in hit order. SEMANTIC bounds the effect.

## Harness fixes before the counted run
- The first run built each tiny config by setting attributes after construction. That left derived fields stale: `qk_head_dim`, `layer_types`, `sliding_window=None`.
- 7 families then failed inside stock transformers attention / mask / cache code, never in Vahana code.
- The harness now passes the small sizes to each config class constructor:
  - 4 layers, so hybrid models get a full-attention layer;
  - MLA models get one K/V head per query head;
  - `head_dim` is set where a family leaves it None.
- The bars and the Vahana code are unchanged. The results below are from the fixed harness (m1_omp24.json / m1_omp8.json).

## Result: SPEC, EXACT-1, EXACT-3 PASS (17/17); SEMANTIC 16/17 (qwen3_5_moe_text NOT MET)
| Family | Experts modules | E / I / H | EXACT-1 | EXACT-3 (8 vs 24 thr) | Twin vs stock (logits) | Single call vs stock | Experts' effect on logits |
|---|---|---|---|---|---|---|---|
| mixtral | 4 | 8/512/256 | ✓ | ✓ | 0.65% | 0.35% | 29% |
| qwen2_moe | 4 | 8/128/256 | ✓ | ✓ | 0.75% | 0.68% | 6.8% |
| qwen3_moe | 4 | 8/128/256 | ✓ | ✓ | 0.62% | 0.68% | 5.0% |
| olmoe | 4 | 8/512/256 | ✓ | ✓ | 0.65% | 0.35% | 9.2% |
| deepseek_v2 | 4 | 8/128/256 | ✓ | ✓ | 0.73% | 0.68% | 13% |
| deepseek_v3 | 4 | 8/128/256 | ✓ | ✓ | 1.02% | 0.68% | 82% |
| glm4_moe | 4 | 8/128/256 | ✓ | ✓ | 0.72% | 0.68% | 18% |
| glm4_moe_lite | 3 | 8/128/256 | ✓ | ✓ | 0.57% | 0.68% | 42% |
| minimax | 4 | 8/512/256 | ✓ | ✓ | 0.59% | 0.35% | 3.4% |
| minimax_m2 | 4 | 8/512/256 | ✓ | ✓ | 0.63% | 0.35% | 20% |
| hunyuan_v1_moe | 4 | 8/512/256 | ✓ | ✓ | 0.75% | 0.35% | 27% |
| ernie4_5_moe | 3 | 8/128/256 | ✓ | ✓ | 0.65% | 0.68% | 12% |
| cohere2_moe | 4 | 8/512/256 | ✓ | ✓ | 0.64% | 0.35% | 9.1% |
| exaone_moe | 4 | 8/128/256 | ✓ | ✓ | 1.01% | 0.68% | 42% |
| afmoe | 3 | 8/128/256 | ✓ | ✓ | 0.00% | 0.68% | 55% |
| phimoe | 4 | 8/512/256 | ✓ | ✓ | 0.66% | 0.35% | 59% |
| qwen3_5_moe_text | 4 | 8/128/256 | ✓ | ✓ | **11.7%** | 0.68% | 92% |

How each column is measured:
- **EXACT-1:** CPU `VahanaExperts` equals the GPU twin bit for bit on prefill logits (12 positions), 8 decode-step logits and 8 ids.
- **EXACT-3:** the CPU-arm logits and ids are identical at OMP 8 and OMP 24 threads.
- **Twin vs stock (logits):** max |Δ prefill logits| / max |stock logits|.
- **Single call vs stock:** one experts call on the same input (16 tokens), stock vs twin, max |Δ| / max |stock|.
- **Experts' effect on logits:** how much the logits move when the experts' output is zeroed (m1_checks.json).

**Reading:**
- **Every Tier 1 family runs on Vahana's CPU experts with bit-identical results,** independent of thread count.
- **The experts reach the logits in every family** (3-92%), so no family's exactness check is trivial.
- **AFMoE matches stock exactly** although a single experts call differs by 0.68%. The difference does not survive to its final logits in this tiny model.
- **qwen3_5_moe_text misses SEMANTIC (11.7% > 5%).** Its single-call difference from stock (0.68%) is the same as every other family's, so there is no wiring error. The tiny random model amplifies it: the MoE drives 92% of the logits, and the output passes through its Gated-DeltaNet layers.
  - The bar is recorded as not met.
  - The real-weight check in M5 (quality vs stock fp32 on real prompts) is the deciding test for this family.
- **Deferred as pre-registered:**
  - batch / padding invariance, which needs the invariant attention function (M3);
  - placement invariance, which needs GPU-budget wiring (M4/M5).
