# Vahana

**Exact inference for MoE models too big for your GPU.**

Vahana (Sanskrit: *carrier*) runs Mixture-of-Experts language models whose experts do not fit in GPU memory:
- the routed experts stay in CPU RAM and run on a CPU kernel;
- the GPU runs attention, routing and everything else;
- anything still over your GPU budget moves to CPU RAM too.

Its answers are **bit-identical** wherever the work runs. The same tokens come out regardless of:
- whether an expert runs on the CPU or the GPU;
- how small the GPU budget is;
- what else is in the batch, or how it is padded;
- how many CPU threads are used;
- whether a token is computed in prefill or decode;
- **which machine it runs on.** A CPU-only laptop and an H100 give the same logits and tokens, bit for bit (v0.2 P2).

> **Status: v0.1, plus v0.2 portability in progress.**
> - **v0.1:** one GPU plus a large-RAM CPU host, BF16 checkpoints, and an OpenAI-compatible server (one request at a time).
> - **Done for v0.2:**
>   - one CPU kernel for AVX-512, AVX2, ARM NEON and plain C (P1);
>   - CPU-only machines, with the same bits as a GPU (P2);
>   - Linux, Windows and macOS: no libtorch or OpenMP dependency, one wheel per platform, and `vahana selftest` (P3).
> - **Next:** `vahana doctor` and an SSD streaming tier, for models bigger than RAM (P4).

## Quick start
```bash
pip install vahana-*.whl            # a wheel for your platform (CI builds them), then torch for your machine
pip install -e . && python src/vahana/csrc/build.py    # or from source: needs a C++20 compiler, no torch headers
vahana selftest                     # same bits as the reference H100? (18 tiny models, about a minute)

vahana run   /models/Qwen3-30B-A3B --prompt "Explain mixture-of-experts in one paragraph." --chat
vahana serve /models/DeepSeek-Coder-V2-Instruct --gpu-budget 16 --port 8000
vahana run   /models/OLMoE-1B-7B-0924 --device cpu --prompt "Hello"   # no GPU: same tokens, slower
```
```python
import vahana
model, info = vahana.load("/models/Mixtral-8x7B-v0.1", gpu_budget_gb=4)    # or device="cpu"
out = vahana.generate(model, input_ids, max_new_tokens=128)        # ids, per-step timings
```
The server speaks the OpenAI API:
- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`, with `stream`, `max_tokens`, `temperature`, `top_p`, `seed` and `stop`.

Greedy decoding is bit-exact. Sampling is reproducible: tokens are drawn on the host from fp32 logits with a seeded generator.

## Results
All measured on one machine: 1× H100 NVL, 2× EPYC 9354, 503 GB DDR5 with 4 of 12 channels per socket populated. Every number is from a pre-registered experiment in `experiments/`.

| Model | GPU budget | Decode | Exactness |
|---|---|---|---|
| DeepSeek-Coder-V2 236B (BF16, 438 GB) | 16 GB | **7.49 tok/s** (48 threads) | graph == eager, bit for bit |
| Mixtral-8x7B (BF16, 94 GB) | 4 GB | **11.40 tok/s** (48 threads) | graph == eager, bit for bit |
| Qwen3-30B-A3B | 2.9 GB used | ~20 tok/s through the server, prefill included (24 threads) | CPU experts == GPU experts, bit for bit |

**Quality on real checkpoints** (experiments/m5_real): next-token agreement with stock transformers in fp32, over every position of 50 prompts.

| Model | Vahana | Stock bf16 |
|---|---|---|
| OLMoE-1B-7B | 98.0% | 98.5% |
| Qwen1.5-MoE-A2.7B | 97.3% | 96.3% |
| DeepSeek-V2-Lite | 98.0% | 98.0% |
| Qwen3-30B-A3B | 96.8% | 96.4% |

Vahana is as close to full precision as ordinary bf16 inference. Unlike ordinary inference, it is also exactly reproducible.

## Supported models
Vahana finds the experts by structure (transformers' `experts(hidden, indices, weights)` modules) instead of hard-coding families, and fails loudly on anything it does not recognise.

| Status | Families |
|---|---|
| **Real checkpoints validated** (exact + quality) | Mixtral, DeepSeek-V2 (Lite and 236B), Qwen1.5-MoE (Qwen2-MoE), Qwen3-MoE, OLMoE |
| **Validated on the real architecture, tiny random weights** (exact, batch-invariant, graph decode) | DeepSeek-V3, GLM-4-MoE (+Lite), MiniMax-M2, Hunyuan-MoE, Ernie-4.5-MoE, Cohere2-MoE, Exaone-MoE, AFMoE, gpt-oss (BF16 experts) |
| **Exact, but eager decode only** | MiniMax (lightning attention), Qwen3.5-MoE (Gated-DeltaNet), Phi-MoE (host-side RoPE) |
| **Not yet** | gpt-oss MXFP4 checkpoints, FP8 checkpoints (DeepSeek-V3/R1), Llama 4, Granite-MoE, JetMoE, DBRX, encoder-decoder MoE |

## How it is exact
- **Twin arithmetic.** CPU and GPU kernels evaluate every dot product in the same order (16 fp32 FMA lanes, a fixed reduction tree) with the same BF16 rounding points. An expert or linear layer gives the same bits on either device.
- **Batch-invariant kernels.** The `vahana` attention function and the linear and norm kernels reduce each row in a fixed order that depends only on that row. Batch neighbours, padding and prefill-vs-decode cannot change it.
- **NX12.** A lossless 12-bit encoding of BF16 expert weights (75% of the size) that the CPU kernel decodes at memory bandwidth.
- **One CUDA graph per decode step,** with the CPU jobs as host nodes, for any model.
- **Twin arithmetic for everything else (v0.2 P2).**
  - Attention has a CPU twin of the GPU kernel: the same blocks, Vahana's own `exp` and fixed reduction trees.
  - Every remaining operation is written only in IEEE-exact steps: norms, rotary embeddings, router softmax and top-k, activations, small matmuls.
  - Two notable cases:
    - PyTorch's CUDA `sqrt` is not correctly rounded, so Vahana's sqrt is.
    - CUDA divides by a scalar via its reciprocal, so Vahana divides by a device tensor.
  - **Result:** the whole model gives the same bits on any CPU or GPU.
  - **Cost:** +1.8% per GPU decode step.

## Requirements and limits (v0.1)
- **RAM** for the experts: about 0.75 × their BF16 size as NX12, plus page cache for the checkpoint. 236B needs about 340 GB.
- **One NVIDIA GPU, or none.** CPU-only works, gives the same tokens, and is slower.
- **CPU:** any x86-64 (AVX-512 fastest, AVX2, or plain). The ARM NEON build is written but not yet run on ARM hardware.
- **OS:** Linux, Windows and macOS. CI builds a wheel for Linux x86-64 / arm64, Windows x86-64 and macOS arm64, and checks the selftest on each.
- **CUDA graphs** need Linux or Windows with an NVIDIA GPU. CPU-only works everywhere.
- **The server runs one request at a time.**
- **Prefill on very large models is slow** when many linears are in CPU RAM: 38 s for a 236B prompt.
- **Hybrid-attention models decode eagerly.**

## Layout
- **v0.1 path:** `src/vahana/{loader, modelspec, experts, runtime, attention, decode, server, cli}.py`.
- **Twin arithmetic (P2):** `tmath.py`, `attention.py`.
- **Kernel (P3):**
  - `src/vahana/csrc/` holds `cpu_moe.cpp`, `vt.h`, `pool.h` and `build.py`, built as a plain shared library with a C ABI.
  - Python reaches it through `_kernel.py`.
  - `osl.py` holds the OS-specific calls.
- **Earlier research engine:** the remaining modules (batched throughput, column tiers, ANS plane, speculative decoding). Not used by the v0.1 path.
- **Design:** `docs/design/v0.1-moe-adapter.md`, `docs/design/v0.2-portability.md`.
- **Evidence:** `experiments/` (M0-M5, server, P1 portable kernel, P2 twin).

## License
Not yet chosen; all rights reserved until then.
