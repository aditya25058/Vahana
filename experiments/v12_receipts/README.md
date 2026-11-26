# v1.2 Receipts: verifiable inference, and `vahana bench`

## Question
Vahana gives the same bits on every machine. Can anyone use that to check an answer someone else produced, on different hardware, bit for bit and cheaply?

## Pre-registered bars
1. Feasibility: for attention models, one teacher-forced forward over prompt + output must reproduce every decode step's fp32 logits bit for bit. If it does not, the fallback is replay.
2. A receipt made on one machine must verify on a different CPU ISA, OS and GPU, for both greedy and seeded sampling.
3. A changed token must fail at its step, and an edited receipt must fail its integrity check.
4. For attention models, checking must cost less than generating.

## Results
**1. Prefill equals decode.** Measured with [prefill_eq.py](prefill_eq.py) on tiny selftest models, laptop CPU and GPU, 8 steps each:

| Family | Prefill logits == decode logits |
|---|---|
| mixtral | 8/8 steps, CPU and GPU |
| qwen3_moe | 8/8 steps, CPU and GPU |
| olmoe | 8/8 steps, CPU and GPU |
| deepseek_v3 | 8/8 steps, CPU and GPU |
| gpt_oss | 8/8 steps, CPU and GPU |
| glm4_moe | 8/8 steps, CPU and GPU |
| qwen3_5_moe_text (hybrid) | step 0 only |

Qwen3.5's Gated DeltaNet layers compute a chunked prefill and a recurrent decode, which round differently. Hybrids are therefore verified by replaying the decode (`mode: replay`).

**2. Cross-machine verification.**
- Producer: laptop, Windows, Ryzen 7 4800H (AVX2), RTX 3050 Ti 4 GB.
- Checker: H100 NVL server, Linux, EPYC 9354 (AVX-512).

| Receipt | Producer | Checked on | Result | Check time |
|---|---|---|---|---|
| OLMoE-1B-7B, greedy, 48 tokens | laptop GPU | laptop CPU / laptop GPU | PASS / PASS | 6.9 s / 7.6 s |
| | | H100 CPU / H100 GPU | PASS / PASS | 0.73 s / 0.23 s |
| OLMoE-1B-7B, T=0.8 top-p 0.95 seed 11, 40 tokens | laptop GPU | laptop CPU / laptop GPU | PASS / PASS | 7.5 s / 8.0 s |
| | | H100 CPU / H100 GPU | PASS / PASS | 0.67 s / 0.39 s |
| Qwen3-30B-A3B w4a16 (exact 4-bit), T=0.7 top-p 0.9 seed 5, 32 tokens | laptop GPU | laptop GPU | PASS | 15.6 s |
| | | H100 CPU / H100 GPU | PASS / PASS | 1.81 s / 1.24 s |
| Qwen3.5-35B-A3B (hybrid), greedy, 16 tokens | H100 GPU | H100 GPU / H100 CPU (replay) | PASS / PASS | 4.5 s / 16.6 s |

The laptop's three receipts are in this folder.

**3. Tampering.** Both on the OLMoE greedy receipt, checked on the H100 CPU:
- Output token 10 changed, with the receipt id recomputed: FAIL, `logits` and `tokens` mismatch, first differing step 10.
- The seed edited without recomputing the id: FAIL, `intact` mismatch.

Unit tests cover both cases on three tiny families, plus the exact exponential and the sampler: [tests/test_receipt.py](../../tests/test_receipt.py).

**4. Cost.** `vahana bench` runs a 512-token prompt and 64 decoded tokens, then verifies that run's receipt:

| | Decode | Time to first token | Verify 64 tokens |
|---|---|---|---|
| OLMoE, H100 | 147.7 tok/s (0.43 s for 64) | 0.22 s | 0.24 s |
| Qwen3-30B w4a16, H100 | 55.9 tok/s (1.1 s for 64) | 0.82 s | 0.96 s |
| OLMoE, laptop GPU | 6.75 tok/s (9.3 s for 64) | 15.4 s | 11.7 s |

All four bars are met. For decode-bound runs, checking is cheaper than generating, but not by as much as the prefill-vs-decode gap suggests. The check re-reads the 512-token prompt, so its cost is close to the time to first token plus a short prefill over the answer. On the laptop, prefill is CPU-bound (1.3 targets this), so checking costs about as much as the prefill did. Full bench outputs: `bench_*.json`.

## Exact sampling
1.0 sampled with `torch.multinomial` on a seeded `torch.Generator`. Its tokens depend on torch's implementation and version. 1.2 samples with only:
- IEEE basic operations: an exp built from `+ - * /` and `ldexp`, with relative error < 1e-14;
- sequential sums;
- a stable sort;
- inverse-CDF at a 53-bit uniform taken from SHA-256(seed, step, row).

So a seed gives the same token everywhere, and a verifier can re-derive each sampled token from the recomputed logits.

## What a receipt does not show
- It is not signed, so it does not show who produced it.
- The weight fingerprint covers file names, sizes and safetensors headers by default. Changed weight values are caught by verification itself, since the logits change.
- It does not show that the weights are any good.
