# M3: the `vahana` attention function and whole-model batch invariance

Pre-registered 2026-11-20, before any run of the kernel.

**Code:** `vahana/attention.py`, registered with transformers' `AttentionInterface` and `ALL_MASK_ATTENTION_FUNCTIONS`, and enabled by `attention.use(model)`.
- One program per (row, head, query). Keys are visited in blocks of 64, anchored at that query's first allowed key, with fp32 scores and an online softmax in order.
- **Sinks** (`s_aux`) are a virtual key visited first.
- **Soft-cap** is optional.
- **GQA / MQA** by head grouping; the value size may differ from the key size (expanded MLA).
- **The mask is always explicit and boolean.** The mask builder disables the "pure causal, skip the mask" shortcut and the fix that fills fully padded rows. Sliding windows arrive inside the mask.

**Whole-model setup per family** (m3_invariance.py; tiny models as in M2):
- `invariant.install` (batch-invariant linears, RMSNorms, `F.linear` shim);
- `attention.use`;
- `experts.install(mode="cpu")`.

**Test inputs:** four prompts of lengths 7 / 19 / 33 / 40, then 6 greedy steps. Stock baselines are computed first, before anything is patched.

**Bars** (17 standard-attention cases: the 16 non-hybrid Tier 1 families plus gpt_oss with sinks + sliding window):
- **BATCH:** every row of the left-padded batch of 4 equals the same prompt run alone, bit for bit: prefill logits at its real positions, 6 decode-step logits, 6 ids.
- **SPLIT:** prefill of 39 tokens followed by one decode step gives logits bit-identical to the full 40-token prefill at that position.
- **SEMANTIC:** the full Vahana model vs stock transformers on prompt 4 is within 10% of max |stock logit|. This is wider than M1/M2's 5% because linears, norms and attention now all change arithmetic, not just the experts. It still catches wiring errors such as a wrong norm type or a wrong mask.

**Reported, not barred:** qwen3_5_moe_text. Its Gated-DeltaNet layers run stock kernels, whose padding behaviour Vahana does not control (design §3.4; a v0.2 item).

**Prediction:**
- BATCH and SPLIT pass for all 17.
- SEMANTIC lands around 1-10%.
- The risk is families whose norms have `variance_epsilon` but are not RMSNorm (cohere2's LayerNorm), which `invariant.install` would rewrite wrongly. SEMANTIC would show that.

## Run log
**Run 1** (m3.log, overwritten; its numbers are kept here). 0/18 fully invariant, for three reasons:
1. **Harness bug.** The batched prefill call passed no `position_ids`, so left-padded rows got the wrong positions. `generate` computes positions itself, so every row's decode steps and ids were already bit-identical to running alone. Fixed in the harness (`position_ids` from the mask).
2. **`invariant.install` replaced every module with `variance_epsilon`.** That rewrote cohere2's LayerNorm as an RMSNorm (8.8% from that alone, m3_diag.json) and crashed on qwen3.5's gated norm. Fixed (patch_inv_m3.py): each norm class is replaced only if it takes one input and `inv_rmsnorm` reproduces it bit for bit on a random input. Otherwise the stock forward stays and the class is listed.
3. **minimax failed SPLIT.** MiniMax-Text-01 interleaves lightning (linear) attention layers, so it is a hybrid. It was misclassified as standard attention in the pre-registration.

The bars and the attention kernel are unchanged. Run 2 below is after fixes 1 and 2.

## Result (run 2): BATCH PASS (17/17); SPLIT 16/17 (minimax NOT MET); SEMANTIC 14/17 (NOT MET for gpt_oss, deepseek_v3, afmoe)
**Attention accuracy, single call** (m3_attn_unit.json): vs an fp32 reference with transformers' mask / sink / soft-cap semantics.

| Case | Max abs error | Reference max | Row alone == in batch |
|---|---|---|---|
| GQA causal, left padding | 0.0075 | 3.09 | ✓ |
| GQA decode | 0.0019 | 0.93 | ✓ |
| MQA, sliding window 16 | 0.0077 | 2.92 | ✓ |
| Sinks + sliding window 16 | 0.0048 | 2.25 | ✓ |
| Sinks, full | 0.0041 | 2.18 | ✓ |
| MLA shape (D 192, DV 128) | 0.0072 | 3.13 | ✓ |
| Soft-cap 30 | 0.0078 | 3.28 | ✓ |
| Decode, T = 300 | 0.0009 | 0.32 | ✓ |

All errors are within half a BF16 ULP of the output magnitude: the kernel computes the same function as stock attention.

**Whole models** (m3.json):

| Family | Norms replaced / kept stock | BATCH | SPLIT | Vahana vs stock |
|---|---|---|---|---|
| mixtral | 9 / – | ✓ | ✓ | 6.4% |
| qwen2_moe | 9 / – | ✓ | ✓ | 1.6% |
| qwen3_moe | 17 / – | ✓ | ✓ | 2.3% |
| olmoe | 17 / – | ✓ | ✓ | 0.61% |
| deepseek_v2 | 17 / – | ✓ | ✓ | 0.97% |
| deepseek_v3 | 17 / – | ✓ | ✓ | **36.9%** |
| glm4_moe | 9 / – | ✓ | ✓ | 0.82% |
| glm4_moe_lite | 17 / – | ✓ | ✓ | 1.0% |
| minimax (hybrid) | 11 / – | ✓ | **✗** | 0.66% |
| minimax_m2 | 17 / – | ✓ | ✓ | 5.7% |
| hunyuan_v1_moe | 17 / – | ✓ | ✓ | 9.9% |
| ernie4_5_moe | 9 / – | ✓ | ✓ | 0.65% |
| cohere2_moe | 0 / LayerNorm | ✓ | ✓ | 3.0% (was 10.2%) |
| exaone_moe | 17 / – | ✓ | ✓ | 0.83% |
| afmoe | 25 / – | ✓ | ✓ | **21.7%** |
| phimoe | 0 / – | ✓ | ✓ | 8.1% |
| gpt_oss | 9 / – | ✓ | ✓ | **62.7%** |
| qwen3_5_moe_text (reported) | 0 / gated norm | ✗ (decode ids only) | ✗ | 26.6% |

**Reading:**
- **Batch invariance holds for every standard-attention family,** including gpt-oss with sinks and sliding windows and the MLA families. Every row of a left-padded batch is bit-identical to the same prompt run alone: prefill logits, decode logits, ids.
- **Prefill-vs-decode invariance holds for 16 of 17.** minimax fails because of its lightning-attention layers, which run stock kernels.
- **SEMANTIC is not met for gpt_oss, deepseek_v3 and afmoe.** The evidence points to amplification inside tiny random models, not a wiring error:
  - The attention kernel matches fp32 to half a BF16 ULP on every feature (table above).
  - The experts are bit-exact to the twin (M2) and differ from stock by 0.4-0.7% per call (M1).
  - In m3_diag.json a single component alone already moves these models' logits by 20-50%. For deepseek_v3, the invariant linears alone move it by 46%. These tiny models route through near-tie gates, so one flipped expert choice changes the logits wholesale.
  - Real-weight quality against stock fp32 (M5) decides.
- **Hybrid layers** (minimax lightning attention, qwen3.5 Gated-DeltaNet) run stock kernels and are not invariant: the planned v0.2 item.
