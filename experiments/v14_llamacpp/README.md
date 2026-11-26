# Vahana vs llama.cpp on the same laptop

## Question
How does Vahana compare with llama.cpp, the tool most people use to run MoE models on small machines? Measured on the same laptop, with the same model at a similar 4-bit size, each with its best settings.

## Setup
- **Machine:** Ryzen 7 4800H (8 cores, AVX2), 15 GB RAM, RTX 3050 Ti Laptop GPU 4 GB, NVMe, Windows 11.
- **Model:** Qwen3-30B-A3B.
  - llama.cpp: the official Qwen GGUF, Q4_K_M (17.3 GB).
  - Vahana: RedHatAI w4a16 (GPTQ int4, group 128), its exact 4-bit path.
  - The two formats are not the same quantization, so quality is not compared here. Q4_K_M mixes 4- and 6-bit blocks; w4a16 is int4 throughout.
- **llama.cpp:** release b12494 (2026-11-18), official Windows CUDA 12.4 build.
  - Tuned for MoE offload: attention and as many expert layers as fit on the GPU (`--n-cpu-moe`), the rest on the CPU.
  - Thread count swept.
- **Vahana:** the v1.3 commit, default settings (`vahana bench`).

## Pre-registered bars (written 2026-11-26, before any llama.cpp run)
1. **Reported either way.** Every number is published, including where llama.cpp is faster. These are not pass/fail bars for Vahana, so the launch claims cannot rest on cherry-picked settings.
2. **Decode:** Vahana at least 0.5x llama.cpp's tok/s (claim: "within 2x of llama.cpp while exact"). Below that, the README says so plainly.
3. **Time to first token,** 512 and 2,048 tokens: same rule, at least 0.5x.
4. **Determinism**, measured on both:
   - the same prompt run twice;
   - batch 1 vs the same prompt inside a batch of 4 (llama.cpp `--parallel`);
   - CPU-only vs GPU offload.
   The question is whether the outputs (tokens, and logits where available) are identical. Expected: Vahana identical in all three; llama.cpp not identical across CPU/GPU or batching.

## Results
These are noisy. Neither model fits the laptop's 15 GB of RAM (llama.cpp's GGUF is 18.6 GB; Vahana's 4-bit experts are 16 GB), so both read weights from the NVMe page cache, and run-to-run variation is large, especially for llama.cpp's 512-token prompt. Every run is listed, as ranges.

**llama.cpp settings sweep** (`llama-bench -ngl 99 --n-cpu-moe N -t T`, 2–3 repetitions; files `llamabench_*.jsonl`):
- Fastest: 42 of 48 expert layers on the CPU, 8 threads.
- Fewer CPU layers overflow the 4 GB of VRAM into shared system memory, and llama.cpp gets slower: decode 7–11 tok/s at N = 34–40.
- 14 threads was no better than 8.

**Speed and memory**, Qwen3-30B-A3B, best settings each:

| | llama.cpp b12494 (Q4_K_M) | Vahana v1.3 (w4a16, exact) | Vahana / llama.cpp |
|---|---|---|---|
| Decode, 64 tokens | 13.2–15.5 tok/s (4 runs) | 9.4–11.4 tok/s (2 runs; 6.4 in an earlier run) | about 0.6–0.85x |
| 512-token prompt | 29–117 tok/s (5 runs, very noisy) | 70–79 tok/s (4 runs) | comparable |
| 2,048-token prompt | 124–161 tok/s (4 runs) | 194–210 tok/s (2 runs) | **1.2–1.7x faster** |
| Peak GPU memory | 3.7 GB | 3.1 GB | |
| Peak RAM (working set) | 10.1 GB | 9.4–9.8 GB | |

Bars 2 and 3 are met: decode is above 0.5x, and time to first token is at least 0.5x at both lengths.
- llama.cpp decodes faster, by about 1.2–1.6x.
- Vahana's prompt processing is about as fast at 512 tokens and faster at 2,048 (v1.3's GPU-streamed prefill).
- Caveat: the quantizations differ. Q4_K_M has some 6-bit blocks and is 16% larger than w4a16, so llama.cpp moves more bytes per token. Quality is not compared here.

**Determinism.** The same prompt, greedy, 32 tokens; tokens and log-probabilities compared exactly. Files: `determinism_*.json`, scripts `determinism_llamacpp.py` and `determinism_vahana.py`.

| | llama.cpp | Vahana |
|---|---|---|
| Same prompt again (GPU offload) | identical | identical |
| Alongside 3 other requests (llama.cpp `--parallel 4`; Vahana batch of 4) | tokens identical; **log-probabilities differ from token 2 on** | identical (tokens and fp32 logits) |
| CPU only instead of GPU offload | **different answer**: the text changes at token 3 ("the reasons, the events, ..." vs "the reasons for the split, the key events, ..."); log-probabilities differ from the first token | identical (tokens and fp32 logits) |

Bar 4 is met as expected. In llama.cpp the answer depends on where it runs, and its probabilities depend on what else is being served. Vahana gives the same bits in all three cases. That is what makes Vahana's receipts checkable on someone else's hardware.

## Summary for the README
On a 4 GB laptop, llama.cpp decodes about 1.2–1.6x faster. Vahana processes long prompts faster (1.2–1.7x at 2,048 tokens). Vahana's answers are bit-identical across CPU and GPU and under batching; llama.cpp's are not.
