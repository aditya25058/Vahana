# Vahana

**Exact inference and serving for Mixture-of-Experts models too big for your GPU.**

Vahana (Sanskrit: *carrier*) runs Mixture-of-Experts language models on the hardware you have, from an H100 server down to a laptop with a 4 GB GPU:
- the GPU runs attention, routing and everything else, plus a cache of the most-used experts;
- the remaining experts run on an exact CPU kernel, from RAM or streamed from disk when they do not fit.

Its answers are **bit-identical** wherever and however the work runs. The same tokens and the same logits come out regardless of:
- **the machine:** a CPU-only laptop and an H100 agree bit for bit, on Linux, Windows and macOS, on AVX-512, AVX2, ARM NEON or plain C;
- **where the weights live:** GPU, GPU expert cache, RAM, or disk;
- **what else is running:** batching, continuous batching in the server, padding, speculative decoding, or prefill vs decode.

Other engines change a request's output when the batch around it changes. In Vahana, a request served alongside 63 others gives exactly the bits it gives alone, and that is measured, not assumed ([benchmarks](https://github.com/vahana-ai/Vahana/blob/main/docs/benchmarks.md)).

> **Status: 1.4 (beta).** The exactness guarantee is stable and validated on real checkpoints up to DeepSeek-V2 236B, with 18 model families in the selftest. New in 1.4: on a 4 GB laptop, Qwen3-30B 4-bit runs faster than llama.cpp on the same machine (real prompts: decode 1.4×, first token 2.6× sooner), with an adaptive GPU expert cache; 70.8 tok/s on an H100, with the same bits as everywhere else. Since 1.3: receipts (verifiable inference), exact 4-bit models, fast exact long-prompt prefill and a prefix cache. Expect rough edges outside the tested setups below; please open an issue.
> Every claim here comes from a pre-registered experiment in [`experiments/`](https://github.com/vahana-ai/Vahana/tree/main/experiments), including the bars it missed.
> See the [changelog](https://github.com/vahana-ai/Vahana/blob/main/CHANGELOG.md).

## First five minutes
Install Vahana, check this machine, run a small MoE model, and verify an answer:
```bash
pip install torch "vahana[gpu]"            # omit [gpu] on machines without an NVIDIA GPU
vahana selftest                            # 18 tiny models: does this machine give the reference bits?
hf download allenai/OLMoE-1B-7B-0924 --local-dir models/OLMoE-1B-7B-0924     # 14 GB (pip install huggingface_hub)
vahana run models/OLMoE-1B-7B-0924 --prompt "The capital of France is" --max-tokens 16 --receipt answer.json
vahana verify answer.json --model models/OLMoE-1B-7B-0924 --device cpu       # PASS: the same bits on the CPU
vahana bench  models/OLMoE-1B-7B-0924      # speed and memory on this machine, with an exactness check
```
For a 30B model on a small GPU, use the exact 4-bit checkpoint (16 GB of experts, runs with a 4 GB GPU and 16 GB of RAM):
`hf download RedHatAI/Qwen3-30B-A3B-quantized.w4a16 --local-dir models/Qwen3-30B-A3B-w4a16`.

## Quick start
```bash
pip install torch                      # the build for your machine (CUDA or CPU), from pytorch.org
pip install "vahana[gpu]"              # [gpu] adds Triton for NVIDIA GPUs (triton-windows on Windows); omit it for CPU-only
vahana selftest                        # does this machine give the reference bits? (18 tiny models, about a minute)
vahana doctor /models/Qwen3-30B-A3B    # what this machine runs, where the weights go; add --measure to time it

vahana run   /models/Qwen3-30B-A3B --prompt "Explain mixture-of-experts in one paragraph." --chat
vahana serve /models/Qwen3-30B-A3B --slots 32 --port 8000      # OpenAI-compatible, continuous batching
vahana run   /models/OLMoE-1B-7B-0924 --device cpu --prompt "Hello"   # no GPU: the same tokens, slower
vahana run   /models/Qwen3-30B-A3B --prompt "$(cat doc.txt) Q: ..." --prefix-cache   # later questions on doc.txt skip its prefill
```
From source: `pip install -e .` then `python src/vahana/csrc/build.py` (needs a C++20 compiler; no torch headers).

```python
import vahana
model, info = vahana.load("/models/Qwen3-30B-A3B", gpu_cache_gb="auto")   # or gpu_budget_gb=4, device="cpu"
out = vahana.generate(model, input_ids, max_new_tokens=128)               # greedy; ids, logits, per-step timings
batch = vahana.generate_batch(model, [ids_a, ids_b, ids_c], 64)           # each row == its solo run, bit for bit
spec = vahana.generate_spec(model, input_ids, 64, k=4)                    # exact speculative decoding
```
The server speaks the OpenAI API:
- `GET /health`, `GET /v1/models`;
- `POST /v1/completions` and `POST /v1/chat/completions`, with `stream`, `max_tokens`, `temperature`, `top_p`, `seed` and `stop`.

Concurrent requests share decode steps (continuous batching), and every response equals the same request run alone. Greedy decoding is bit-exact. Sampling is exact too: Vahana's sampler draws from the fp32 logits using only IEEE basic operations and a SHA-256 counter stream, so a given `seed` gives the same tokens on every machine.

## Receipts: verifiable inference
Because the bits do not depend on the machine, an answer can be checked by anyone who has the same weights.
```bash
vahana run    /models/Qwen3-30B-A3B --prompt "..." --receipt answer.json      # also --temperature/--top-p/--seed
vahana verify answer.json --model /models/Qwen3-30B-A3B                      # on any machine: PASS or FAIL
vahana bench  /models/Qwen3-30B-A3B --json bench.json                         # speed, memory, and a receipt check
```
A receipt records:
- the prompt's token ids and the decoding settings;
- a fingerprint of the weights;
- the output token ids;
- a SHA-256 digest of every step's logits.

`vahana verify` recomputes all of it and passes only if every step's logits and every token match bit for bit. There is no tolerance and no spot-checking.
- **Attention models:** the check is one teacher-forced pass over prompt + answer, so it is much faster than generating. 64 tokens of Qwen3-30B 4-bit take 0.96 s to check on an H100.
- **Hybrid models (Qwen3.5):** the decode is replayed.

Tested so far:
- Receipts made on a Windows laptop's RTX 3050 Ti (OLMoE, and Qwen3-30B 4-bit with sampling) verify on an H100 server, both on its GPU and on its AVX-512 CPU.
- A changed token fails at the step where it was changed, and an edited receipt fails its integrity check ([experiments/v12_receipts](https://github.com/vahana-ai/Vahana/blob/main/experiments/v12_receipts/README.md)).

A receipt shows that these weights compute this output for this prompt and these settings. It is not signed, so it does not show who ran it.

## Results
Full tables, the machines used, and the misses are in **[docs/benchmarks.md](https://github.com/vahana-ai/Vahana/blob/main/docs/benchmarks.md)**. Highlights (H100 NVL server, and a laptop with an RTX 3050 Ti 4 GB and 15 GB of RAM):

| | Result |
|---|---|
| DeepSeek-Coder-V2 236B (438 GB), 16 GB GPU budget | 7.49 tok/s |
| Qwen3-30B-A3B served to 64 requests at once (18 GB of experts on the GPU, the rest on the CPU) | 130 tok/s aggregate, every request bit-identical to running alone |
| OLMoE-1B-7B, 64 requests at once | 1,016 tok/s aggregate |
| Continuous batching (128 requests, 32 slots) | the throughput of a static batch, 7.4× one at a time |
| Qwen3.5-35B-A3B (Gated DeltaNet hybrid), batched | 4.35× one at a time, bit-identical |
| Qwen3-30B (57 GB) on the 15 GB laptop, streamed from NVMe | about 1 tok/s, the H100's logits bit for bit |
| Qwen3-30B 4-bit on the 4 GB laptop, 2,048-token prompt | 9.6 s to first token (was 82 s), same logits as the H100 |
| vs llama.cpp on the same laptop (1.4; [decode](https://github.com/vahana-ai/Vahana/blob/main/experiments/v15_decode/README.md), [1.3 comparison](https://github.com/vahana-ai/Vahana/blob/main/experiments/v14_llamacpp/README.md)) | on five real prompts: decode 1.57x (15.3 vs 9.7 tok/s), time to first token 2.8x faster (2.9 vs 8.2 s); on the benchmark text, decode 1.86x (21.4 vs 11.6 tok/s); prefill of 2,048 tokens 1.2–1.7x faster; on a free Colab T4, decode 1.96x on the benchmark text and 1.65x on real prompts, first token 3.4x sooner; **with no GPU (2-core Kaggle CPU), llama.cpp is about 3.5x faster at decode** (7.91 vs 2.21 tok/s after 1.4's CPU work; 4.7x before) ([reproduce](https://github.com/vahana-ai/Vahana/blob/main/experiments/reproduce/README.md)); closer to the bf16 original than Q4_K_M (KL divergence 0.065 vs 0.093, [quality](https://github.com/vahana-ai/Vahana/blob/main/experiments/v15_quality/README.md)); only Vahana gives the same answer on CPU and GPU |
| Cost of exactness | +1.8% per GPU decode step |

## Supported models
Vahana finds the experts by structure (transformers' `experts(hidden, indices, weights)` modules) instead of hard-coding families, and fails loudly on anything it does not recognise.

| Status | Families |
|---|---|
| **Real checkpoints validated** (exact + quality) | Mixtral, DeepSeek-V2 (Lite and 236B), Qwen1.5-MoE (Qwen2-MoE), Qwen3-MoE, OLMoE |
| **Validated on the real architecture, tiny random weights** (exact, batch-invariant, graph decode) | DeepSeek-V3, GLM-4-MoE (+Lite), MiniMax-M2, Hunyuan-MoE, Ernie-4.5-MoE, Cohere2-MoE, Exaone-MoE, AFMoE, gpt-oss (BF16 experts) |
| **Real checkpoint validated, eager decode** | Qwen3.5-MoE (Gated DeltaNet; Qwen3.5-35B-A3B batched and served exactly) |
| **Exact, eager decode only** (tiny random weights) | MiniMax (lightning attention), Phi-MoE (host-side RoPE) |
| **Not yet** | gpt-oss MXFP4 checkpoints, FP8 checkpoints (DeepSeek-V3/R1), Llama 4, Granite-MoE, JetMoE, DBRX, encoder-decoder MoE |

## How it is exact
- **Twin arithmetic.** CPU and GPU kernels evaluate every dot product in the same order (16 fp32 FMA lanes, a fixed reduction tree) with the same BF16 rounding points. An expert or linear layer gives the same bits on either device.
- **Batch-invariant kernels.** The `vahana` attention function and the linear and norm kernels reduce each row in a fixed order that depends only on that row. Batch neighbours, padding and prefill-vs-decode cannot change it.
- **NX12.** A lossless 12-bit encoding of BF16 expert weights (75% of the size) that the CPU kernel decodes at memory bandwidth.
- **One CUDA graph per decode step,** with the CPU jobs reached through a GPU↔CPU mailbox, for any model.
- **Batching without changing bits.** Every kernel computes each row on its own, and attention is anchored at each query's first key, so batch neighbours and padding never move a rounding. The continuous-batching engine only copies KV; recurrent layers never see padding.
- **Twin arithmetic for everything else (v0.2 P2).**
  - Attention has a CPU twin of the GPU kernel: the same blocks, Vahana's own `exp` and fixed reduction trees.
  - Every remaining operation is written only in IEEE-exact steps: norms, rotary embeddings, router softmax and top-k, activations, small matmuls.
  - Two notable cases:
    - PyTorch's CUDA `sqrt` is not correctly rounded, so Vahana's sqrt is.
    - CUDA divides by a scalar via its reciprocal, so Vahana divides by a device tensor.
  - **Result:** the whole model gives the same bits on any CPU or GPU.
  - **Cost:** +1.8% per GPU decode step.

## Requirements and limits
- **Python 3.10+, PyTorch 2.4+, transformers 5.12–5.24.** Vahana runs transformers' own model code, and some releases change a model's definition (Qwen3.5-MoE from 5.14, gpt-oss from 5.19), which changes its outputs on every machine alike. The selftest therefore keeps a reference per transformers version, and CI's `compat` job checks every supported release (plus a canary on the newest one); the range widens as new releases are validated. NVIDIA GPUs need Triton (`vahana[gpu]`). CPU-only works everywhere.
- **RAM:**
  - **Fast:** room for the experts, plus page cache.
  - **Less RAM still works, exactly:** experts stream from disk at disk speed. A saturated NVMe caps decode; for example Qwen3-30B on a 15 GB laptop runs at about 1 tok/s.
- **One NVIDIA GPU, or none.** Multi-GPU is not supported.
- **CPU:** x86-64 (AVX-512 fastest, AVX2, or plain) and ARM64 (NEON). Wheels: Linux x86-64 / arm64, Windows x86-64, macOS arm64. CI runs the selftest on each, under every instruction set.
- **Checkpoints:**
  - BF16;
  - 4-bit GPTQ in compressed-tensors pack-quantized format (for example RedHatAI's `w4a16` models), run exactly.
  - FP8 and MXFP4 checkpoints are not yet supported.
- **Hybrid linear-attention models** decode eagerly (no CUDA graph).
- **Speculative decoding is exact but rarely faster for batch-1 MoE:** each drafted token routes to new experts ([why](https://github.com/vahana-ai/Vahana/blob/main/docs/benchmarks.md#what-did-not-work)).
- **Prefill:**
  - Since 1.3, long prompts run their CPU-resident experts on the GPU (same bits).
  - Prefill is still slow on very large models whose dense linears sit in CPU RAM: 38 s for a 236B prompt.
- **Prefix cache:** batch 1, attention models, and not yet in `vahana serve`.

## Layout
- **Engine:** `src/vahana/`:
  - `loader`, `modelspec`, `experts`, `runtime` (GPU↔CPU mailbox), `cache` (GPU expert cache);
  - `attention`, `twinlin`, `fused`, `tmath` (twin arithmetic);
  - `decode`, `batch`, `speculative`, `engine` (continuous batching);
  - `prefill`, `prefill_kernels` (GPU-streamed exact prefill), `prefix` (prefix cache), `q4` (exact 4-bit);
  - `receipt` (receipts, exact sampler), `bench`, `server`, `cli`, `doctor`, `selftest`.
- **CPU kernel:** `src/vahana/csrc/`: `cpu_moe.cpp`, `vt.h`, `pool.h`, `build.py`. A plain shared library with a C ABI, loaded through `_kernel.py`. `osl.py` holds the OS-specific calls.
- **Earlier research modules** (column tiers, ANS plane, tiered serving) are kept for reference and are not on the default path.
- **Design:** `docs/design/`. **Evidence:** `experiments/` (M0–M5, P1–P4, v0.3-a … v0.5).

## License
Apache License 2.0; see [LICENSE](https://github.com/vahana-ai/Vahana/blob/main/LICENSE) and [NOTICE](https://github.com/vahana-ai/Vahana/blob/main/NOTICE). Copyright 2026 Aditya Sai Teja B.
