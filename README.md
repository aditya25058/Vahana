# Vahana

**Exact inference for MoE models too big for your GPU.**

Vahana (Sanskrit: *carrier*) runs Mixture-of-Experts models that do not fit in GPU memory. It keeps the experts in CPU RAM and computes them on the CPU, while the GPU runs attention and routing. Its answers are **bit-identical** to a full-precision GPU run: the same tokens no matter where an expert runs, how small the GPU budget is, or what the batch size or thread count is.

> Status: **v0.1 in development.** The engine is a research prototype being turned into a server. Expect rough edges and environment-variable configuration.

## Results so far
All results are from one machine: 1× H100, 2× EPYC 9354, 503 GB DDR5 with 4 of 12 channels per socket populated. Prompts are greedy, batch 1, 128 tokens. Every run's tokens match the full-GPU reference.

| Model | GPU budget | Vahana |
|---|---|---|
| DeepSeek-Coder-V2 236B (BF16, 438 GB) | 16 GB | **7.39 tok/s** (48 threads), 6.39 (24 threads) |
| Mixtral-8x7B (BF16, 94 GB) | 4 GB | **11.7 tok/s** (48 threads) |

Against KTransformers on the same CPUs (Mixtral BF16, all experts on the CPU):
- **24 cores:** Vahana is 1.19× faster.
- **48 threads:** Vahana is 1.08× faster.
- **Exactness:** KTransformers' tokens change when experts move between CPU and GPU. Vahana's do not.

## How it is exact
- **Twin arithmetic.** The CPU kernel and the GPU kernels evaluate every dot product in the same order, with the same rounding points. A CPU expert therefore returns the same bits as a GPU expert.
- **Batch-invariant kernels.** Each token's result does not depend on which other rows share the batch or how they are padded.
- **NX12.** A lossless 12-bit encoding of BF16 weights (75% of the size) that decodes at memory bandwidth inside the kernel.

## Supported models
- **Validated:** Mixtral, DeepSeek-V2.
- **In progress (v0.1):** a generic MoE adapter for every MoE family in `transformers`, including Qwen-MoE, DeepSeek-V3, OLMoE, gpt-oss, Llama 4 and GLM-4-MoE.

## Build the CPU kernel
Requires AVX-512 with VBMI (AMD Zen 4+, Intel Sapphire Rapids+), GCC and libnuma.

```bash
bash src/vahana/csrc/build_kernel.sh python
```

## License
Not yet chosen; all rights reserved until then.
