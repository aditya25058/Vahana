# M4: generic decode graph (any MoE model), placement, speed

Pre-registered 2026-11-20, after a 2-family smoke run (mixtral and deepseek_v2 passed every tiny bar below) and before the full matrix and any real-model run.

**Code:**
- `vahana/runtime.py`: CPU expert jobs, CPU twin linears and CPU embedding rows, each run eagerly or as host nodes in a captured graph. It also installs the linear dispatch: CPU twin, GPU twin, or invariant GEMM.
- `vahana/loader.py`:
  - `load(model_dir, gpu_budget_gb)`: checkpoint → meta model; experts as zero-copy checkpoint views via transformers' own key conversions.
  - `prepare(model)`: the same for in-memory models.
  - Budget placement: shared experts folded into the CPU expert job when the block adds them directly, then the input embedding, then the largest bias-free linears.
  - Shared experts always use expert arithmetic.
- `vahana/decode.py`: generic greedy generation over a StaticCache; one CUDA graph per step after a first step through the graph-mode path.
- **Kernel:** host jobs for CPU linears and embedding rows.

**What changes from the old engine:** the per-family decoders (`graph_decode.py`, `mixtral_graph.py`) are not used. Attention is the M3 `vahana` function over transformers' own attention modules, so DeepSeek-V2 uses the expanded K/V form, not the old latent kernel. The bits therefore differ from Phases 41/42. The design's "ids == Phase 41/42" regression bars are dropped for that reason (stated before running). Speed bars stay.

**Bars:**
- **GRAPH (tiny, m4_tiny.py, 17 Tier 1 families + gpt_oss):**
  - batch of 2 left-padded prompts, 10 greedy steps;
  - graph == eager, bit for bit (ids and every step's logits), with all non-expert weights on the GPU;
  - and again with embedding, layer 1 o_proj and all shared experts in CPU RAM.
- **PLACE (tiny):** CPU-placed == all-GPU, bit for bit.
- **REAL-EXACT** (m4_real.py, same process): on Mixtral-8x7B and DeepSeek-Coder-V2 236B, the graph run's first 32 ids and step logits equal an eager run's, bit for bit.
- **SPEED** (prompt A, 128 tokens, 48 OMP threads on CPUs 0-24,32-56, membind 0,1, main thread on 55,56):
  - Mixtral-8x7B at a 4 GB GPU budget: decode ≥ **11 tok/s**.
  - DeepSeek-Coder-V2 236B at a 16 GB budget: decode ≥ **7.0 tok/s**.
  - Metric: generated steps / sum of step times, with the first step included and capture excluded.

**Prediction:**
- GRAPH and PLACE pass for all standard families. minimax and qwen3.5 hybrids should also pass, because these bars are about graph vs eager and placement, not batch invariance.
- Mixtral ~11-12 tok/s.
- 236B ~6.5-7.5 tok/s. The expanded-K/V attention and the HF forward add GPU time compared with Phase 41's latent kernel; host-node savings may or may not cover it.

## Run log
**Tiny matrix, run 1:** 9/18. Each problem was fixed in code, with the bars unchanged:
- The force-CPU test pattern also matched linears inside a shared expert that cannot be folded. **Fix:** shared-expert weights are placed only by folding.
- `StaticSlidingWindowLayer` keeps its fill position in a Python int, which a captured graph freezes; this made exaone's graph differ from eager. **Fix:** sliding layers become full-length static layers, with the window enforced by the mask.
- Phi-MoE recomputes RoPE frequencies from host tensors inside every forward, which cannot be captured. The failed capture then left the allocator mid-capture and broke later families. **Fix:** a failed capture releases its pool (created up front) and decoding continues eagerly, with the reason reported.
- **Robustness:**
  - An exception inside the C++ worker terminated the process. Worker errors now re-raise from `wait_done`.
  - The planner only places linears whose shapes the CPU twin linear accepts.
- Models whose `layer_types` include non-attention layers, or whose own cache class rejects `StaticCache` (minimax, qwen3.5), decode eagerly with their own cache. The reason is reported.

**Real 236B, attempt 1:** loaded (466 s), then failed in transformers' own DeepSeek-V2 block. Its group-limited routing reads `self.num_experts`, which the class never sets. **Fix:** the loader sets it from `n_routed_experts`.

## Result: GRAPH 15/18 (NOT MET for minimax, qwen3_5_moe_text, phimoe); PLACE 18/18; REAL-EXACT PASS; SPEED PASS (both)
**Tiny matrix** (m4_tiny.json + m4_tiny_b.json, final code): batch 2, left padding, 10 steps.

| Family | Graph == eager (GPU / CPU-placed) | CPU-placed == all-GPU | Graph used | Graph / eager step |
|---|---|---|---|---|
| mixtral, qwen2_moe, qwen3_moe, olmoe | ✓ / ✓ | ✓ | ✓ | 1.5-1.8 / 4.7-5.8 ms |
| deepseek_v2, deepseek_v3, glm4_moe, glm4_moe_lite (shared experts folded) | ✓ / ✓ | ✓ | ✓ | 1.4-1.8 / 5.5-6.4 ms |
| minimax_m2, hunyuan, ernie4_5, cohere2, exaone (folded), afmoe | ✓ / ✓ | ✓ | ✓ | 1.4-2.2 / 5.2-6.0 ms |
| gpt_oss (sinks, sliding window) | ✓ / ✓ | ✓ | ✓ | 1.6 / 5.3 ms |
| minimax (lightning attention) | eager both | ✓ | ✗ own cache class | — |
| qwen3_5_moe_text (Gated-DeltaNet) | eager both | ✓ | ✗ non-attention layers | — |
| phimoe (RoPE recomputed from host tensors) | eager fallback | ✓ | ✗ capture not possible | — |

**Real models** (prompt A, 128 tokens, 48 threads, membind, main thread on 55,56):

| | Mixtral-8x7B, 4 GB budget | DeepSeek-Coder-V2 236B, 16 GB budget |
|---|---|---|
| **Decode** | **11.40 tok/s** (bar ≥ 11) | **7.49 tok/s** (bar ≥ 7.0) |
| Steady step (median) | 83.0 ms (Phase 42: 80 ms) | 127.6 ms (Phase 41: 128 ms) |
| Eager steady step | 146 ms | 290 ms |
| Graph == eager, 32 steps (ids + logits) | ✓ | ✓ |
| Peak VRAM | 3.10 GB | 13.77 GB |
| Placement | all non-expert on GPU | 59 shared experts folded, embedding + 33 linears in CPU RAM |
| Load | 24 s (page cache warm) | 451 s (NX12 encode, disk-bound) |
| Prefill | 0.96 s | **37.8 s** (eager, CPU linears in 16-row jobs; a v0.2 item) |
| MemAvailable minimum | — | 147 GB |

**Reading:**
- **The generic path reaches the hand-built engines' speed** on both regression models, with no per-family decoder code. The 236B model, shared-expert folding included, comes entirely from structure discovery.
- **Graph capture is generic** for 15 of 18 tiny families. The three exceptions decode eagerly but correctly; they are hybrids or dynamic-RoPE models, which the pre-registration wrongly predicted would pass.
- **236B prefill (38 s) is slow** and is the next performance item.
