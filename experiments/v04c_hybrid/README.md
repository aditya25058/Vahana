# v0.4-c: exact batching for recurrent / linear-attention models (Qwen3.5-MoE, MiniMax)

Pre-registered 2026-11-24, after development runs on the laptop (below) and before the counted H100 runs.

**Why:** until now, the two hybrid families in the selftest (`qwen3_5_moe_text`: Gated DeltaNet + full attention; `minimax`: lightning attention + full attention) were served one request at a time. Left padding runs their recurrence over the pad tokens, so a padded row no longer matched its solo run (v0.3-e probe: only the unpadded row was exact).

**Design (engine.HybridEngine; `engine.make` picks it for these models):**
- **No padding ever enters a recurrence.** Each request is prefilled alone (batch 1, exactly its solo prefill).
- **Only decode steps are batched.** Each row processes its own one token. Recurrent state (`conv_states`, `recurrent_states`, MiniMax `linear_cache`) is per row and concatenated along the batch. Attention KV is left-padded to a common length with the mask zero there; Vahana's attention is anchored at each query's first allowed key, so this is exact.
- Both models ignore the 2-D mask in their linear-attention layers once there is cached state, so batched decode never sees padding there.
- Finished rows are removed by index, and leading columns no row uses any more are trimmed.
- The engine runs eagerly (the dynamic cache changes shape).
- `generate_batch` and the server use it; nothing in these families falls back to one at a time any more.
- **Also fixed:** `decode.generate`'s dynamic-cache path ignored `eos_token_id`, so `vahana run` on these models did not stop at EOS. It now stops as the static path does.

**Development runs (laptop, final engine code):**
- Engine, 20 requests in waves into 6 slots: minimax 20/20, qwen3_5 20/20 on GPU and on CPU. The first probe showed 15/20 on both; the 5 misses were exactly the early-EOS requests, because the solo reference (dynamic path) ignored EOS. That is the bug fixed above.
- `generate_batch` B = 4: 4/4 rows on both.

**Bars (H100, counted):**
- **EXACT-HYBRID:** v4c_tiny (20 requests, waves, 6 slots, early EOS), every request's ids and every logit equal solo `decode.generate`. 2/2 families on GPU, GPU with half the experts cached, and CPU.
- **EXACT-BATCH:** v3e_tiny `generate_batch` B = 4 and B = 8, all 18 families exact, all batched (no one-at-a-time fallback).
- **REGRESSION:** v4b_tiny (the attention-only engine) 16/16, and `vahana selftest --device cuda` 18/18.
- **REAL (conditional):** there is no hybrid checkpoint on the H100. The smallest, Qwen3.5-35B-A3B (about 70 GB), needs the user's approval to download. If approved, 16 M5 prompts are served through the engine and 8 are compared with solo runs; throughput vs one at a time is reported.

**Prediction:** all exact. Hybrid batching gives a large throughput gain over one at a time (decode is shared); it is eager, so per-step overhead is higher than the graph engine's.

## Result: every bar passes. Recurrent / linear-attention models now batch exactly; nothing is served one at a time any more
| Bar | Result |
|---|---|
| EXACT-HYBRID: 20 requests in waves into 6 slots (early EOS included), ids and every logit vs solo | **2/2 families** (minimax, qwen3_5_moe_text), 20/20 each, on the H100 GPU, the H100 GPU with half cache, and the H100 CPU (also laptop GPU and CPU during development) |
| EXACT-BATCH: `generate_batch` B = 4 and B = 8, rows vs solo | **18/18 families both**, all batched (0 fallbacks) |
| REGRESSION: attention-only engine (v4b_tiny) | **16/16** (the 2 hybrid families are covered by HybridEngine above) |
| `vahana selftest --device cuda` | **18/18** |
| REAL | **not run.** There is no hybrid checkpoint on the H100; Qwen3.5-35B-A3B (about 70 GB) waits for the user's approval to download |

**Reading:** every family Vahana supports can now be batched and served concurrently with bit-identical results per request. Attention-only models use the CUDA-graph engine; hybrid models use the eager engine, where each request's recurrence only ever sees its own tokens.

## Real checkpoint (pre-registered after the user approved the download, before the run)
- **Model:** Qwen/Qwen3.5-35B-A3B, 71.9 GB, a vision-language checkpoint whose text model is `qwen3_5_moe_text`: 40 layers, Gated DeltaNet plus full attention every 4th layer, 256 experts with 8 active, and a shared expert. Only the text model is loaded.
- **Loader changes needed:**
  - the text weights are under `model.language_model.` and are mapped to the causal-LM names;
  - experts are stored fused (`gate_up_proj` [E, 2I, H], `down_proj` [E, H, I]) and are sliced per expert as views of the mmap, gate = rows :I and up = rows I: (the module's own `chunk(2)`).
  - Smoke test: "The capital of France is" → " Paris." (1.9 tok/s solo, eager, no GPU cache).
- **Bar REAL-HYBRID (v4c_real.py; 50 GB GPU expert cache as planned, 25 GB in the counted run, see the run log):** 16 M5 prompts through HybridEngine in two waves (8 slots), 32 tokens. The first 8 equal solo `decode.generate` (ids and every logit). Engine decode tok/s vs one at a time is reported.
- **Run log:** the first run (50 GB requested) got a 38 GB cache (98% of profiled expert uses) and then ran out of GPU memory in prefill. Another user's vLLM engine, started outside the GPU scheduler, held 49 GB of the same H100. Rerun with a 25 GB cache to fit beside it.

## Result (real checkpoint): REAL-HYBRID passes. Qwen3.5-35B-A3B batched requests are bit-identical to solo, 4.35× the throughput
| Bar | Result |
|---|---|
| 16 M5 prompts in two waves through HybridEngine (8 slots), first 8 vs solo `decode.generate`, ids and every logit | **8/8** |
| Engine decode tok/s vs one at a time (25 GB GPU expert cache, 88% of profiled expert uses; the rest on the CPU) | **16.8 vs 3.86 tok/s: 4.35×** |

Sample: "The Fibonacci sequence is defined as: F(0) = 0, F(1) = 1, F(n) = F…"

**Reading:** a real 35B hybrid model (Gated DeltaNet linear attention, 256 experts) is served with continuous batching, and each request's output is bit-identical to running it alone. Its recurrence never sees padding, because each request is prefilled alone and only decode steps are shared.
