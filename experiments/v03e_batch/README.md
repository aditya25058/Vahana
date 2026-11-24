# v0.3-e: exact batched decoding (several requests per step, each with its solo bits)

Pre-registered 2026-11-24, after a probe (probe_laptop_b4.json) and before the counted runs.

**Why:** v0.3-d's profile showed that at batch 1, a pass's cost is set by the weights it must read:
- the dense linears, which are now flat in tokens per pass;
- the distinct experts, about 8 per token per layer.

Batching several requests shares the dense-weight reads completely and the expert reads partly. It is the standard way engines get throughput, but in other engines it changes each request's bits (the batch-invariance problem). Vahana's kernels are row-wise and its attention is padding- and batch-invariant by construction, so batching should change no bit.

**Probe (laptop GPU, before this milestone's code):** `decode.generate` with 4 left-padded prompts of lengths 5–20 vs each prompt alone, 16 tokens, ids and every logit.
- 17/18 tiny families are exact.
- The exception is `qwen3_5_moe_text`: its linear-attention (recurrent) layers run over the left padding, so only the unpadded row matches.

**Code:**
- `vahana.generate_batch(model, prompts, ...)` (batch.py):
  - takes a list of token-id prompts of any lengths;
  - left-pads them into one batch and runs graph decode;
  - stops when every row has produced EOS;
  - returns each request's tokens (trimmed at its EOS) and, optionally, logits.
- Models with recurrent / linear-attention layers decode each request alone, so the result stays exact.

**Bars:**
- **EXACT-TINY:** every family, batched rows equal solo runs (ids and every step's logits): B = 4 and B = 8 on the H100 GPU (graph), the H100 with half the experts cached, and the laptop GPU. 18/18 through `generate_batch`, with the recurrent family exact via its fallback.
- **EXACT-REAL (H100):** OLMoE (fully cached) and Qwen3-30B (18 GB cache), 8 M5 prompts decoded as one batch of 8, 32 tokens each. Every row equals that prompt's solo run (ids and logits).
- **SPEED (H100):** aggregate decode tok/s = B × (tokens − 1) / decode time, at B = 1, 2, 4, 8, 16, 32, 64 tokens each.
  - OLMoE fully cached: B = 8 ≥ 3× B = 1;
  - Qwen3-30B (18 GB cache): B = 8 ≥ 2× B = 1;
  - the curve is reported.

**Prediction:**
- OLMoE B = 8 about 4× and B = 32 about 8–10× (the v0.3-d profile: a 12-token pass costs about 1.9× a 1-token pass in kernel time);
- Qwen3 B = 8 about 2.5× (its uncached experts run on the CPU, whose work grows with distinct experts).

## Run log
- **First counted run:** the exactness bars all passed, and B = 1–16 were measured. B = 32 and 64 stopped at an assert: graph-mode host jobs (the CPU mailbox) took at most 16 rows, the CPU kernel's per-call limit. Logs are in `run1/`.
  - Fix: graph mode now uses one host job per 16-row chunk, exactly like the eager path. Rows are independent, so this can't change bits.
  - Laptop check: B = 24 exact on 5 families. Then everything was rerun: tiny B = 24 with half the cache on the H100 (all families), and the real harness (exactness plus the full curve).
- The counted numbers below come from the rerun. The first run's B ≤ 16 numbers agree: OLMoE B = 8 was 4.05×, Qwen3 B = 8 was 2.3×.

## Result: all bars pass. Every request gets its solo bits; OLMoE B = 8 is 5.1×, Qwen3 B = 8 is 2.9×
| Bar | Result |
|---|---|
| EXACT-TINY, `generate_batch` vs solo, ids and every logit | **18/18** on: H100 B = 4, B = 8; H100 with half cache B = 4, 8, 24; laptop B = 4, 8; laptop B = 24 (5 families). Recurrent families (minimax, qwen3_5_moe_text) are exact via the per-request fallback |
| EXACT-REAL: batch of 8 M5 prompts, 32 tokens, rows vs solo runs | **OLMoE 8/8, Qwen3-30B 8/8** (graph decode; both runs) |
| SPEED OLMoE (fully cached) B = 8 vs B = 1 (bar ≥ 3×) | **5.13×, met** |
| SPEED Qwen3-30B (18 GB cache) B = 8 vs B = 1 (bar ≥ 2×) | **2.92×, met** |

**Aggregate decode tok/s (H100, mean of 2 rounds, 64 tokens per request):**

| B | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| OLMoE-1B-7B, all experts on the GPU | 78.8 | 152.2 | 235.9 | 404.4 | 501.6 | 600.4 | **863.4** (11.0×) |
| Qwen3-30B-A3B, 18 GB of experts on the GPU, the rest on the CPU | 12.4 | 17.9 | 21.5 | 36.3 | 42.4 | 45.8 | **48.5** (3.9×) |

**Reading:**
- **Batching is where an exact MoE engine gets its throughput.** At B = 64, OLMoE serves 863 tok/s aggregate, and every request's ids and logits are identical to decoding it alone. Other engines' batched outputs depend on what else is in the batch.
- **Qwen3 flattens past B = 8.** Its uncached experts run on the CPU, and the CPU's work grows with distinct experts × rows, while the mailbox serialises 16-row chunks. More GPU cache or a faster CPU expert path (multi-row expert kernels) is the next lever there.
- **The step time grows slowly up to B = 8** (12.7 → 19.7 ms on OLMoE) because the dense linears are flat (v0.3-d) and expert reads are shared between rows.
