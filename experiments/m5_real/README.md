# M5: real checkpoints, exactness and quality

Pre-registered 2026-11-20, before any M5 run.

**Models:**
- OLMoE-1B-7B-0924 (64 experts, top-8);
- Qwen1.5-MoE-A2.7B (60 experts plus a gated shared expert);
- DeepSeek-V2-Lite (MLA, 64 experts plus 2 shared);
- Qwen3-30B-A3B (128 experts, top-8).

All four are native transformers classes, loaded from their checkpoints by `vahana.loader.load`.

**Arms** (m5_real.py, one process per model):
- **fp32:** stock transformers in fp32, GPU plus CPU offload where it does not fit. Prefill logits on 50 prompts (prompts.json), batch 1.
- **bf16:** stock transformers in bf16; the same quantities.
- **cpu:** Vahana with CPU experts and no GPU budget. 5 prompts × 48 greedy tokens with the decode graph, plus prefill logits on the 50 prompts.
- **twin:** Vahana with every expert on the GPU (same arithmetic, the reference). The same quantities, decoded eagerly.

The stock arms run first, because Vahana patches `nn.Linear` process-wide.

**Bars:**
- **EXACT:** for each model, cpu == twin bit for bit: 5 × 48 generated ids, every generated step's logits, and prefill logits on all 50 prompts.
- **QUALITY:** for each model, Vahana's next-token top-1 agreement with stock fp32, over every prompt position, is ≥ stock bf16's agreement minus 1.0 percentage point. In other words, Vahana is as close to full precision as ordinary bf16 inference is. Max |Δ logit| / max |fp32 logit| is reported for both.

**Not covered here:**
- Mixtral-8x7B and DeepSeek-Coder-V2 236B were validated in M4 (graph == eager, speed). Their fp32 references (188 GB / 944 GB) do not fit this box's GPU plus a usable offload.
- The M3 tiny-model SEMANTIC misses (deepseek_v3, afmoe, gpt_oss) have no real checkpoint here. This M5 QUALITY bar on real weights is the test of the same question for the families that are present.

**Prediction:** EXACT passes for all four. QUALITY agreement is ~97-99% for both Vahana and stock bf16, within a point of each other.

## Run log
Attempt 1 (logs in attempt1/) produced no results, for two harness / loader bugs. The bars and the models were unchanged:
- `loader.load(log=None)` called `log` unconditionally. Fixed in the loader.
- The harness released models through a helper, while the caller's frame still held them. Accelerate's offload hooks form reference cycles, so the fp32 model stayed on the GPU and the bf16 load ran out of memory on Qwen3-30B. Fixed: each arm deletes its model and collects garbage in its own frame.

## Result: EXACT PASS (4/4), QUALITY PASS (4/4)
| Model | Experts × top-k | Positions | CPU == twin (5 × 48 gen ids + logits, 50 prefills) | Vahana vs fp32: top-1 / max rel | Stock bf16 vs fp32: top-1 / max rel | Vahana GPU | Load |
|---|---|---|---|---|---|---|---|
| OLMoE-1B-7B-0924 | 64 × 8 | 603 | **✓** | 98.01% / 14.4% | 98.51% / 18.1% | 0.92 GB | 4.9 s |
| Qwen1.5-MoE-A2.7B (gated shared expert) | 60 × 4 | 590 | **✓** | **97.29%** / 13.8% | 96.27% / 11.8% | 3.49 GB | 7.9 s |
| DeepSeek-V2-Lite (MLA, 2 shared) | 64 × 6 | 594 | **✓** | 97.98% / 4.5% | 97.98% / 5.0% | 2.47 GB | 8.8 s |
| Qwen3-30B-A3B | 128 × 8 | 590 | **✓** | **96.78%** / 14.9% | 96.44% / 14.1% | 2.90 GB | 28.9 s |

Columns:
- **Positions:** every prompt position of the 50 prompts.
- **Vahana / stock bf16 vs fp32:** next-token agreement with stock fp32 over all positions, and max |Δ logit| / max |fp32 logit|.
- **Vahana GPU:** GPU memory of the Vahana model; the experts are in CPU RAM.

**Reading:**
- **Exact on real checkpoints.** For four architectures, CPU experts give the same bits as the same model with every expert on the GPU: 240 generated tokens and 50 prefills each.
- **Quality equals ordinary bf16 inference.** Vahana is as close to full precision as stock bf16 is: within 0.5 point on OLMoE, and ahead on the other three by 0 to 1.0 point. This also answers the M3 tiny-model SEMANTIC question for the families present here, on real weights.
- **All decoding ran through the generic graph** (`graph_used` true for every generation).
