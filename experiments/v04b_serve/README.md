# v0.4-b: exact continuous batching (requests join and leave mid-flight, each with its solo bits)

Pre-registered 2026-11-24, before writing the engine.

**Why:** v0.3-e / v0.4-a made static batches exact and fast. A server receives requests at random times with different lengths. Continuous batching (admitting a request as soon as a slot frees) is how engines keep the GPU full. In other engines it is also a main source of nondeterminism: a request's output depends on what else is running.

**Design (engine.py), exact by the same argument as v0.3-e:**
- **Slots and columns:** S slots share one KV cache of C columns. All active rows end at the same column c (the next write position).
  - Each request's keys are contiguous in its row, from its start column s to c.
  - Its positions are its own (0, 1, …), carried in `position_ids`.
  - Attention visits each query's own keys in blocks anchored at its first allowed key, so a row sees exactly what a solo run sees. Every other kernel is row-wise.
- **Admission:** waiting requests are prefilled together as a left-padded batch (exact, v0.3-e). Each one's KV is then copied bit for bit into a free slot at columns [c − L, c), and its first token is the prefill's greedy choice.
  - If c < L, every active row is first shifted right; if c reaches C, every row is shifted left by the smallest start. Both are pure copies; positions never change.
- **Leaving:** a request ends at EOS, its token limit, or a stop string. Its slot is filled by moving the last active row into it (a copy), so active rows stay packed in rows 0..n−1.
- **Decode step:** one CUDA graph per batch bucket (1, 2, 4, …, S). Every bucket's cache layers are views of the first rows of the one shared cache.
- **Server:** the existing OpenAI-compatible server (v0.1, one request at a time) submits to the engine.
  - Greedy is exact.
  - Seeded sampling draws on the host from each row's fp32 logits (bit-identical to solo), so sampled requests are reproducible too.
  - Models with recurrent / linear-attention layers keep the one-at-a-time path.

**Bars:**
- **EXACT-TINY (GPU graph and CPU):** every family, 20 requests (prompt lengths 3–30, token limits 4–24, some with an EOS id that triggers early).
  - They are submitted in waves into an engine with 6 slots and C small enough to force right shifts, left shifts and compaction.
  - Every request's ids and every logit equal its solo `decode.generate`.
  - Recurrent families: exact via the fallback.
- **EXACT-REAL (H100):** OLMoE (fully cached) and Qwen3-30B (18 GB cache): 32 M5 prompts with staggered arrivals into 8 slots, 32 tokens. The first 12 requests are compared with solo runs (ids and logits).
- **HTTP (H100, OLMoE):** 16 concurrent clients hit `/v1/completions` (greedy). Every response's text equals the text of that prompt run alone through `vahana run`'s path.
- **SPEED (H100, OLMoE fully cached):** 128 requests (M5 prompts, 64 tokens), all arriving at once, through a 32-slot engine. Aggregate decode tok/s ≥ 0.8× the static `generate_batch` at B = 32 (v0.3-e path, same session) and ≥ 5× one-at-a-time. Qwen3 reported.

**Prediction:** all exact. Continuous batching at 0.85–0.95× static B = 32: the extra cost is prefill between steps and bucket rounding.

## Run log (before the counted runs)
- **Laptop, first probe:** the laptop's CI-built kernel predates v0.4-a (16 rows per CPU expert job), so laptop runs set `VAHANA_MOE_ROWS=16`. Chunking never changes bits.
- **The probe's summary counted errors as passes.** Fixed: a family counts only with an `exact` field.
- **No right shift happened in the first all-family run.** The first wave now has short prompts (3–6 tokens), so the longer prompts of the next waves force right shifts. Every family then shows 1 right and 1 left shift and 13–17 row moves.
- **`deepseek_v3` failed during CUDA-graph capture, only after other families had run in the same process.** The bucket step closures refer back to their engine, so a discarded engine's graphs are freed only by the garbage collector, which could run in the middle of a later capture.
  - Fix: the garbage collector is disabled during capture, and `Engine.close()` releases graphs and caches at once.

## Result: all bars pass. Every request gets its solo bits; continuous batching matches static batching's throughput
| Bar | Result |
|---|---|
| EXACT-TINY: 20 requests in waves into 6 slots, C = 64 (shifts both ways, moves, early EOS), ids and every logit vs solo | **16/16 families** on: H100 GPU (graphs), H100 with half cache, H100 CPU, laptop GPU, laptop CPU. 2 recurrent families keep the one-at-a-time path |
| EXACT-REAL: 32 M5 prompts in 3 waves into 8 slots, 32 tokens, first 12 vs solo | **OLMoE 12/12, Qwen3-30B 12/12** (graph) |
| HTTP: 16 concurrent clients (half streaming, staggered, mixed `max_tokens`) vs the one-at-a-time server path | **16/16 texts identical** (OLMoE, 8 slots, 4.1 s wall) |
| SPEED OLMoE: 128 requests × 64 tokens into 32 slots, vs static `generate_batch` B = 32 (bar ≥ 0.8×) | **0.97× decode tok/s, 1.005× end to end, met** (579.7 vs 598.3 decode; 484.6 vs 482.3 e2e tok/s) |
| SPEED OLMoE vs one at a time (bar ≥ 5×) | **7.38×, met** |
| Qwen3-30B (reported) | 1.01× static (77.8 vs 77.1 decode tok/s), 7.5× one at a time |

**Reading:**
- **An OpenAI-compatible server where concurrency cannot change an answer.** Requests arrive at any time with any lengths and share decode steps, and every response is bit-identical to running it alone, including streamed ones. Batching costs nothing in throughput: it matches the static batch and is 7.4× faster than serving one at a time.
- **Scope of the speed bar:** in the speed runs all requests had the same token limit, so the engine's schedule was close to static. The exactness runs (tiny) exercise the irregular cases.
