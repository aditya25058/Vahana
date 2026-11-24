# v0.3-b: exact speculative decoding (prompt lookup), the same bits as plain greedy decoding

Pre-registered 2026-11-24, after the code compiled and before any run.

**Code (speculative.generate_spec):**
- **Draft:** prompt lookup. The longest suffix (up to 3 tokens) of the text so far is found earlier in the text, and the next k tokens are proposed. There is no draft model.
- **Verify:** the last token plus k draft slots go through the model in one pass of fixed width k + 1 (short drafts are padded), so one CUDA graph serves every step.
- **Accept:** drafts are accepted while they equal the model's own greedy choice. The model's next choice is always emitted too.
- **Why it is exact:** Vahana's kernels give a position the same bits whether it is computed alone or with other positions in one pass (M3 split invariance; per-row linears, experts and glue). Rejected slots' KV entries lie beyond the accepted text and are rewritten before any query can see them.
- **Scope:** recurrent / linear-attention families (minimax, qwen3_5_moe_text) are skipped, because they need a position-indexed cache.

**Bars:**
- **SPEC-EXACT-TINY (v3b_tiny.py):** for every tested family (selftest's numpy-seeded tiny models), on the H100 GPU (graph) and on the CPU, three modes are compared with plain greedy `decode.generate`, 24 tokens:
  - prompt lookup (k = 4);
  - an oracle draft (the reference's own future tokens, so every draft is accepted), k = 4;
  - the same oracle draft, k = 7.

  The bar: identical ids and every emitted token's logit vector bit-identical. Also on the GPU with half the experts cached (`gpu_cache_frac=0.5`).
- **SPEC-EXACT-REAL (H100):** OLMoE-1B-7B (all experts cached on the GPU, mailbox) and Qwen3-30B-A3B (18 GB cache). On 8 prompts (4 copy-heavy, 4 from the M5 set), 64 tokens each, prompt lookup k = 4: ids and per-token logits equal plain greedy decoding's.
- **SPEC-SPEED (H100, OLMoE, GPU-resident via the full cache):**
  - on the 4 copy-heavy prompts (rewrite / extend given code or text), decode tok/s with speculation is ≥ 1.5× plain;
  - on the 4 M5 prompts, it is not slower than 0.9× plain (an adaptive k is future work);
  - Qwen3-30B with partial cache is reported (verifying extra tokens costs extra expert reads there).

**Prediction:**
- SPEC-EXACT passes everywhere.
- SPEC-SPEED on copy-heavy prompts is 1.5–2.5× (acceptance 2–3 tokens per step).
- Generic prompts land around 0.9–1.1×.
- Qwen3 with partial cache is near 1×.

## Run log (before the counted runs)
- **First tiny run: every family failed with an index error.** transformers' StaticLayer ignores `cache_position` and appends at its own running length (`cumulative_length`). A verify pass with rejected slots therefore left that length ahead of the accepted text.
  - Fix: before each pass, every layer's `cumulative_length` is reset to the accepted length (outside the CUDA graph).
- **A failed CUDA-graph capture** (Phi-MoE, which falls back by design) left the next family's capture broken.
  - Fix: the same failed-capture cleanup as `decode.generate`.

## Result: SPEC-EXACT passes everywhere; SPEC-SPEED not met (copy-heavy 1.07×), generic not slower than 0.9× (0.96×)
| Bar | Result |
|---|---|
| SPEC-EXACT-TINY, H100 GPU (graph) | **16/16 families**, 3 modes each: lookup, oracle k = 4, oracle k = 7. Oracle steps accept 4.6 / 7.7 tokens per pass; graph used everywhere except Phi-MoE (falls back by design). 2 recurrent families skipped |
| SPEC-EXACT-TINY, GPU with half the experts cached | **16/16** |
| SPEC-EXACT-TINY, CPU | **16/16** |
| SPEC-EXACT-REAL: OLMoE (all experts cached), Qwen3-30B (18 GB cache) | **8/8 prompts each**: ids and every token's logit vector equal plain greedy decoding |
| SPEC-SPEED, OLMoE, copy-heavy (bar ≥ 1.5×) | **1.07×, not met** (78.4 vs 73.1 tok/s). Per prompt: 0.65–1.40×; tokens per pass 1.2–3.2 |
| SPEC-SPEED, OLMoE, generic (bar ≥ 0.9×) | **0.96×, met** (72.1 vs 75.4). Per prompt: 0.62–1.45× |
| Qwen3-30B, partial cache (reported) | copy-heavy 0.68×, generic 0.59× |

**Follow-ups (after the counted runs, not counted):**
- **Adaptive lanes** (`min_draft`): a step without a draft is an ordinary one-token pass with its own graph.
  - Tiny exactness: 16/16.
  - OLMoE: 0.93–0.95× copy-heavy, 0.85–0.86× generic. Qwen3: 0.77× / 0.72×.
  - No gain: the draftless steps were not where the time went.
- **All-GPU layers** (all experts of a layer cached, so the combine runs on the GPU and there is no CPU round trip):
  - Exact: selftest 18/18 at cache 1.0, tiny speculation 16/16.
  - But plain OLMoE decode is slower: 63–68 vs 73–76 tok/s. The GPU combine is about 25 tiny kernels per layer, against one 12 µs mailbox trip with the combine on the CPU. **Reverted.**

**Reading:**
- **Speculative decoding in Vahana is exact by construction, and it is measured so.** Every token and logit equals plain greedy decoding on 16 families, CPU and GPU, real 7B / 30B checkpoints, and up to 8 tokens verified per pass.
- **As far as we found, no other engine guarantees this** (see the literature survey: vLLM's RFC #54506 is still open).
- **It does not pay yet, because Vahana's twin GPU kernels compute each token separately.** Every token re-reads the weights (twin linear grid (R/BR, n); the expert slot kernels per slot), so a 5-token verify pass costs about 1.7–2× a single step instead of about 1×. It wins only when 3+ tokens are accepted per pass (1.4–1.5× on such prompts).
- **The fix is multi-token twin kernels:** one program loads a weight tile once and applies it to every token. Each token's arithmetic (16 lanes, k ascending, tree) is unchanged, so the bits are unchanged. That fix would also speed up prefill, which shares the cost. It is the next milestone.
