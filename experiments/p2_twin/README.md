# P2: CPU-only Vahana, and GPU arithmetic that matches it

Pre-registered 2026-11-21, after the unit test and a two-family smoke run (mixtral, qwen3_moe: both pass), before the
full runs below.

**Code:**
- **csrc `attention_twin`:** a CPU twin of the `vahana` attention function. Per (row, head, query):
  - key blocks of 64, anchored at the query's first allowed key;
  - scores as 16 fma lanes over the head dimension, then tree16;
  - online softmax with Vahana's `dexp`;
  - fixed halving trees over each block's 64 slots;
  - fma rescaling.
- **attention.py:** the Triton kernel computes exactly those operations.
  - It switches from `tl.exp` and `tl.sum` to `dexp` and the halving trees, with `div_rn`.
  - FP fusion is off.
  - CPU tensors go to `attention_twin`.
- **tmath.py:** the device-twin "glue" arithmetic, and `TwinMode`, a torch function mode that wraps the model's forward.
  - **Ops covered:** norms, rotary cos/sin, router softmax, sigmoid and top-k, activations, small fp32 matmuls, scalar division, sqrt.
  - **How:** each op is written only in IEEE-exact operations, so the result is the same on every device by construction.
  - **sqrt:** computed in float64 and rounded, because PyTorch's CUDA float32 sqrt is not correctly rounded.
  - **Scalar division:** a device tensor divisor, because CUDA computes `tensor / scalar` as a reciprocal multiply.
  - **Results:** transcendentals are rounded float64 → float32 → dtype.
  - **Audit:** records any float op that is neither twinned nor known to be exact.
- **twinlin `twin_rmsnorm`:** a one-kernel GPU RMSNorm in the same definition. It is used per class only when it equals the module's own forward run under TwinMode, bit for bit.
- **The rotary `inv_freq`** is recomputed under TwinMode. The stock init uses the CPU's own `pow`.
- **loader.prepare / load(twin=True)** is the default. The model's device can be the CPU, which gives CPU-only Vahana. Shared experts on the CPU use the CPU expert kernel.
- **decode** runs on the CPU, eager.
- **Triton is optional** (`_triton.py` shim), so CPU-only installs import without it.

**Unit (development check, p2_unit.py, H100):**
- attention 120/120 cases bit-identical CPU vs GPU (causal, padded, sliding, no mask; sinks; softcap; GQA; D ≠ DV; D = 40 padded; fp32);
- glue 47/47 identical;
- fast RMSNorm equal to the TwinMode reference at 5 widths;
- every twin transcendental within 0.5 ulp of float64.

**Bars:**
- **TWIN (H100, p2_families.py, 19 cases: 17 Tier 1 families + gpt_oss + mixtral width 96):** CPU-only run == GPU CUDA-graph run (ids and every step's logits, bit for bit), batch of 2 left-padded prompts, 10 greedy steps.
- **GRAPH:** GPU graph == GPU eager. Families that fall back to eager decode in M4 (MiniMax, Qwen3.5, Phi-MoE) are reported, and count as GRAPH pass when the fallback is reported.
- **BATCH:** a prompt alone == the same prompt as a padded row of the batch, on the GPU and on the CPU.
- **CROSS (laptop):** the H100's tiny-model weights and GPU logits are shipped to this laptop. The laptop is Ryzen 7 4800H, AVX2, WSL, torch 2.15 CPU, with the same transformers 5.12.1. Its CPU-only logits == the H100 GPU logits, bit for bit, for every family that passes TWIN.
- **REAL (H100):** OLMoE-1B-7B (real checkpoint), 4 prompts × 24 greedy tokens. CPU-only == GPU (non-expert weights on the GPU, experts on the CPU), bit for bit.
- **QUALITY (H100):** OLMoE-1B-7B, M5 harness. Twin agreement with the fp32 reference is within 1 point of v0.1's.
- **SPEED (H100):** Mixtral-8x7B at a 4 GB GPU budget, graph decode. tok/s within 5% of v0.1 (M4: 11.40).

**Prediction:**
- TWIN passes on the families with plain attention.
- MiniMax and Qwen3.5 (linear-attention layers) may show audit entries and fail TWIN until their ops are twinned.
- CROSS passes wherever TWIN passes.
- QUALITY and SPEED pass.

## Run log (before the counted runs)
- **Unit run 1:** glue 26/43. Three causes, all fixed in tmath before any counted run:
  - **Scalar division:** PyTorch's CUDA computes `tensor / python_number` as `tensor * (1 / number)`, which rounds twice; the CPU divides. TwinMode now passes the divisor as a device tensor.
  - **float64 → bf16:** the CPU and the GPU round this differently. Results now go float64 → float32 → bf16.
  - **The harness compared NaN payload bits.** Any NaN now equals any NaN.
- **Unit run 2:** rsqrt and layer_norm still differed. **PyTorch's CUDA float32 `sqrt` is not correctly rounded** (3124 of 16000 random inputs differ from the CPU, which is correct). The twin sqrt is float64 sqrt, rounded. That equals the correctly rounded float32 result, because float64 has more than 2p + 2 bits.
- **First full family run:** TWIN 19/19. The audit listed two families:
  - deepseek_v2: `polar` and `view_as_complex` (complex RoPE);
  - qwen3_5_moe_text: `conv1d`.

  Their results matched, but not by construction. `polar`, complex multiply (as separate real operations) and `conv1d` (windows + twin matmul) were twinned, and the run was repeated. That repeat is the counted run below.

## Result: TWIN, GRAPH, CROSS, REAL, QUALITY, SPEED pass; BATCH 18/19 (qwen3_5_moe_text, as in M3)
**Unit (p2_unit.json, final):**
- attention 120/120;
- glue 52/52;
- fast RMSNorm 5/5 widths;
- every twin transcendental within 0.5 ulp of float64.

**Families (p2_families.json, H100):**

| | Result |
|---|---|
| TWIN: CPU-only == GPU graph (ids + every step's logits) | **19/19** |
| GRAPH: GPU graph == eager | **19/19**. Graph used on 16. Reported fallbacks, the same three as M4: minimax and qwen3_5_moe_text (linear-attention layers, eager by design), phimoe (its rotary module touches the host during capture) |
| BATCH: alone == padded row, GPU and CPU | **18/19**. qwen3_5_moe_text fails on both devices, as in M3: its Gated-DeltaNet layers let pad tokens into the recurrent state (a model property; reported, not barred, since M3) |
| TwinMode audit (float ops neither twinned nor known exact) | **empty for all 19** |
| Fast GPU RMSNorm verified and used | 16 families. cohere2 (LayerNorm), phimoe (LayerNorm) and qwen3_5 (gated norm) keep the TwinMode path, still exact |

**CROSS (cross_laptop_*.json):**
- **Setup:** this laptop (AMD Ryzen 7 4800H, Zen 2, WSL Ubuntu, torch 2.15 CPU, **no Triton, no CUDA**, transformers 5.12.1) ran the H100's exact tiny models CPU-only.
- **Result: 19/19 families give logits and ids bit-identical to the H100's GPU run, under both the AVX2 and the plain-C kernel builds.**
- So: two machines, two CPU generations, two torch versions (2.4 / 2.15), GPU vs CPU, same bits.

**REAL (real_olmoe.json, OLMoE-1B-7B checkpoint):**
- CPU-only Vahana == GPU Vahana: 4 × 24 greedy tokens (ids and every step's logits), and the prefill logits of **all 50 prompts (603 positions)**, bit for bit.
- Decode step: 19 ms (GPU graph, experts on the CPU) vs 51 ms (CPU-only).

**QUALITY (OLMoE, top-1 agreement with stock fp32 over 603 positions):**

| | top-1 vs fp32 | max \|Δ\| / max \|fp32 logit\| |
|---|---|---|
| v0.1 (M5) | 98.01% | 0.144 |
| **P2 twin** | **98.18%** | 0.184 |
| stock bf16 | 98.51% | 0.181 |

**SPEED (Mixtral-8x7B, 4 GB GPU budget, graph decode, 128 tokens; v0.1 = fcf1645 run alternately in the same session):**

| | v0.1 r1 / r2 | P2 r1 / r2 |
|---|---|---|
| tok/s | 12.37 / 12.43 | 11.62 / 12.14 (−4.2%, within the 5% bar; above M4's 11.40) |
| steady step (median) | 80.26 / 79.59 ms | 81.28 / 81.42 ms (**+1.8%**) |
| first step | 109 / 112 ms | 607 / 158 ms (r1 compiles the new Triton kernels) |
| eager step | 153 / 155 ms | 224 / 222 ms (+45%: TwinMode's Python dispatch) |
| prefill, 7 tokens | 0.79 / 0.76 s | 1.70 / 0.92 s |
| graph == eager (32 ids + logits) | ✓ / ✓ | ✓ / ✓ |

**Reading:**
- **The promise holds on what was tested.** The same model and prompt give the same bits on a CPU-only laptop and on an H100, for every family tested.
- **Real checkpoints are exact too,** not just tiny models.
- **It costs 1.8% per graph decode step.** Eager work (prefill, and eager decode in the CPU-only and fallback families) is slower, because TwinMode dispatches in Python. Fused glue kernels (router, rotary) would remove that cost; that is P3/P4 work.
- **Quality is unchanged** (98.2% vs 98.0%).

**Limits:**
- Long-context RoPE uses the float64 Cephes reduction, which is accurate far beyond the tested positions. Positions above ~1e9 are untested.
- Sampling (the server's seeded sampler) still runs stock torch on the host CPU. It is reproducible on one machine, not yet across CPUs. Greedy is exact.
- p2_families.pt (355 MB: tiny-model weights for CROSS) is not committed. p2_families.py regenerates it.
