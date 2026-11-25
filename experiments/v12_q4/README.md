# v1.2: exact 4-bit experts (twin-q4)

## What
Vahana runs 4-bit MoE checkpoints (compressed-tensors "pack-quantized" symmetric int4 with bf16 scales, e.g. GPTQ w4a16) with its exactness guarantee: identical bits on every CPU and GPU.

- **Arithmetic:**
  - weight = q · s, exact in fp32;
  - each dot product follows Vahana's twin rule: 16 fp32 lanes, k ascending, fma, tree16, bf16 rounding. w·x fits in ≤ 20 bits, so it is exact and fma = mul + add;
  - activation and combine are unchanged.
- **Code:**
  - `vahana/q4.py`: row records, conversion and a scalar reference;
  - C++ kernel: q4 branches in `expert_mlps_ctx` (AVX-512, AVX2, NEON, scalar) and a fused one-token decode (`experts_q4_fused`);
  - loader: reads pack-quantized tensors. Expert matrices are converted once into `~/.cache/vahana/q4/<model>/` and memory-mapped read-only. Other quantized linears are dequantized to bf16;
  - the GPU expert cache is not used for q4 layers yet (its kernels are bf16).

## Exactness
- **Kernel unit test:** the C++ expert job (group, MLPs, combine) on real Qwen3-30B experts vs an independent reference (scalar twin dots, Vahana exp, combine rule) is **bit-identical at N = 1, 5 and 40 on AVX2 (Intel laptop), scalar, and AVX-512 (AMD EPYC)**. The fused decode path is identical too.
- **End to end, RedHatAI/Qwen3-30B-A3B-quantized.w4a16, 3 prompts × 32 tokens:** the hashes of all ids and logits are **identical on the laptop (Windows, Intel, AVX2) and the H100 box (Linux, AMD EPYC, AVX-512)**: `bf4974f1c2239a4c`, `e7800dd32f66c81a`, `36e8f396dbc2f74c`.
- `vahana selftest`: 18/18.

## Speed
**Laptop (RTX 3050 Ti 4 GB, 15 GB RAM, NVMe):** no GPU memory is free for an expert cache, so all experts run on the CPU.

| | tok/s (mean of 3 prompts) | per prompt | disk GB / token |
|---|---|---|---|
| bf16 Qwen3-30B-A3B (v0.5 baseline) | 0.96 | 0.73 / 1.04 / 1.11 | 1.80 |
| **4-bit, run 1** (includes the one-time conversion before decoding) | **4.28** | 3.02 / 4.35 / 5.48 | 0.117 |
| **4-bit, run 2** | **4.66** | 3.47 / 4.74 / 5.78 | 0.087 |

- **4.5–4.9× faster than bf16** on the mean. That **misses the pre-registered 5× bar** on the mean; the warmed-up prompts reach 5.5–5.8 tok/s (5.7–6.0×).
- Disk traffic per token falls about 16–20×.

**H100 box (CPU experts, all in RAM, no disk):** 22.4 tok/s for 4-bit vs 20.8 for bf16 (with NX12). On a machine with plenty of RAM, 4-bit is about even with bf16.

## Quality
This is the published checkpoint's quality; Vahana does not change it.
- Publisher: about 98% MMLU recovery.
- Our strict short-prompt metric: 86% top-1 agreement with bf16 and +21% perplexity, the same as any plain or calibrated int4 we measured (`research`, private).
