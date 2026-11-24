# Vahana benchmarks

Every number here comes from a pre-registered experiment in [`experiments/`](../experiments). Each has a README with:
- its bars, written before the run;
- the run log;
- the raw JSON.

Missed bars are listed too (see [What did not work](#what-did-not-work)).

**Machines:**
- **H100 server:** 1× H100 NVL (94 GB), 2× AMD EPYC 9354 (64 cores), 503 GB DDR5 (4 of 12 channels per socket populated), Linux.
- **Laptop:** RTX 3050 Ti Laptop (4 GB), 15.4 GB RAM, NVMe SSD, Windows 11.

## 1. Exactness (the guarantee)
| What is compared | Result | Evidence |
|---|---|---|
| Same model on CPU-only vs GPU, any instruction set (AVX-512 / AVX2 / NEON / scalar), Linux / Windows / macOS | identical logits and tokens; `vahana selftest` 18/18 families on every CI platform | [p2_twin](../experiments/p2_twin), [p3_portable](../experiments/p3_portable) |
| Laptop (15 GB RAM, model streamed from disk) vs H100 | identical hashes: OLMoE 50/50 prefills + 4/4 generations; Qwen3-30B 8/8 + 2/2 | [p4_doctor_stream](../experiments/p4_doctor_stream) |
| GPU expert cache on / off, any size | identical | [v03_speed](../experiments/v03_speed) |
| Speculative decoding vs plain greedy | identical ids and every logit, 16/16 families, OLMoE and Qwen3-30B | [v03b_spec](../experiments/v03b_spec) |
| A request in a batch (B = 4 … 64, padded) vs alone | identical, 18/18 families, OLMoE and Qwen3-30B | [v03e_batch](../experiments/v03e_batch), [v04a_cpurows](../experiments/v04a_cpurows) |
| Continuous batching (requests join and leave mid-flight) and the HTTP server vs alone | identical: 16/16 families, OLMoE 12/12, Qwen3-30B 12/12, 16 concurrent HTTP clients 16/16 | [v04b_serve](../experiments/v04b_serve) |
| Linear-attention hybrids (Qwen3.5-MoE, MiniMax) batched vs alone | identical; real Qwen3.5-35B-A3B 8/8 | [v04c_hybrid](../experiments/v04c_hybrid) |
| Disk prefetch on / off | identical | [v05_stream](../experiments/v05_stream) |

**Quality** (next-token agreement with stock transformers in fp32, every position of 50 prompts; [m5_real](../experiments/m5_real)):

| Model | Vahana | Stock bf16 |
|---|---|---|
| OLMoE-1B-7B | 98.0% | 98.5% |
| Qwen1.5-MoE-A2.7B | 97.3% | 96.3% |
| DeepSeek-V2-Lite | 98.0% | 98.0% |
| Qwen3-30B-A3B | 96.8% | 96.4% |

**Cost of exactness:** +1.8% per GPU decode step for the twin arithmetic (Mixtral-8x7B, [p2_twin](../experiments/p2_twin)).

## 2. One request at a time (H100)
| Model | Setup | Decode tok/s | Evidence |
|---|---|---|---|
| DeepSeek-Coder-V2 236B (BF16, 438 GB) | 16 GB GPU budget, experts in RAM | 7.49 | [m4_decode](../experiments/m4_decode) |
| Mixtral-8x7B (94 GB) | 4 GB GPU budget | 11.40 | [m4_decode](../experiments/m4_decode) |
| Qwen3-30B-A3B | experts in RAM, GPU↔CPU mailbox | 14.7 (1.72× the v0.2 host-callback path) | [v03_speed](../experiments/v03_speed) |
| OLMoE-1B-7B | all experts in a GPU cache | 79–96 | [v03e_batch](../experiments/v03e_batch), [v04a_cpurows](../experiments/v04a_cpurows) |

## 3. Batched and served throughput (H100)
Aggregate decode tok/s, with every request bit-identical to running it alone:

| Model | B = 1 | B = 8 | B = 32 | B = 64 | Evidence |
|---|---|---|---|---|---|
| OLMoE-1B-7B (all experts on the GPU) | 79–96 | 404–442 | 600–708 | **863–1,016** | [v03e_batch](../experiments/v03e_batch), [v04a_cpurows](../experiments/v04a_cpurows) |
| Qwen3-30B-A3B (18 GB of experts on the GPU, the rest on the CPU) | 15.4 | 49.3 | 89.3 | **130.1** | [v04a_cpurows](../experiments/v04a_cpurows) |

**Continuous batching** ([v04b_serve](../experiments/v04b_serve)): 128 requests into 32 slots.
- OLMoE: 0.97× the static B = 32 batch in decode, 1.005× end to end, and **7.4×** serving one at a time.
- Qwen3-30B: 1.01× static, 7.5× one at a time.

**Hybrid linear attention** ([v04c_hybrid](../experiments/v04c_hybrid)): Qwen3.5-35B-A3B, 16 requests in 8 slots, 16.8 tok/s vs 3.86 one at a time (**4.35×**).

## 4. Small machines (laptop)
| Model | Setup | Decode tok/s | Evidence |
|---|---|---|---|
| OLMoE-1B-7B (13 GB) | 4 GB GPU, mailbox + 1.6 GB GPU cache | 11.9 (2.8× v0.2) | [v03_speed](../experiments/v03_speed) |
| Qwen3-30B-A3B (57 GB, about 4× the RAM) | experts streamed from NVMe | 0.94 | [v05_stream](../experiments/v05_stream) |
| Same, with next-layer prefetch (`VAHANA_PREFETCH=2`) | opt-in | 1.09 (1.15×, follow-up) | [v05_stream](../experiments/v05_stream) |

**Streaming is limited by the disk.** This NVMe reads 1.0–1.25 GB/s single-stream, and decode already pulls about 1.75 GB/s, so the drive is saturated.

## What did not work
Pre-registered bars that were missed, with the reason recorded in each README:
- **Speculative decoding is exact but not faster for batch-1 MoE** (0.84–0.96× on copy-heavy prompts). Each extra token in a verify pass routes to new experts, so expert reads grow with the draft ([v03b_spec](../experiments/v03b_spec), [v03c_fused](../experiments/v03c_fused), [v03d_multilin](../experiments/v03d_multilin)).
- **Fused kernels:** OLMoE decode 1.09× against a bar of 1.15× ([v03c_fused](../experiments/v03c_fused)).
- **Multi-token linears:** prefill 0.77× by the pre-registered measure, which included first-time compilation; 1.16× warm ([v03d_multilin](../experiments/v03d_multilin)).
- **Disk prefetch:** 0.76× as pre-registered (it predicted 2K experts per token, doubling reads), 1.15× with top-K; it is off by default ([v05_stream](../experiments/v05_stream)).
- **`vahana doctor`'s estimate** was off by 3.8× in P4. In v0.5, `--measure` is within 10% on the H100, but 0.71× / 1.23× on the laptop while the page cache warms up ([p4_doctor_stream](../experiments/p4_doctor_stream), [v05_stream](../experiments/v05_stream)).
