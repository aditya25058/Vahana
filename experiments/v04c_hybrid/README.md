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
