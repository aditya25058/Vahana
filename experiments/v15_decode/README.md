# v1.4 decode: closing the gap to llama.cpp on the 4 GB laptop

## Question
On the laptop (Ryzen 7 4800H, 15 GB RAM, RTX 3050 Ti 4 GB), llama.cpp b12494 decodes Qwen3-30B-A3B Q4_K_M at 13.2–15.5 tok/s, and Vahana 1.3 decodes the w4a16 checkpoint at 9.4–11.4 tok/s (experiments/v14_llamacpp). Where does Vahana's decode step go, and can it match or beat llama.cpp without changing a bit?

## Pre-registered bars (written 2026-11-27, before profiling)
1. **Profile first.** A per-step breakdown is written down before any optimisation: CPU expert time, GPU time, host overhead, and disk reads per token.
2. **Bits:**
   - Any change that keeps Vahana's existing arithmetic must give identical logits, checked by the selftest and by receipt verification against the 1.3 outputs.
   - A change that alters the arithmetic (for example integer dot products) must be a new, opt-in mode with its own references. It must also be exact across CPU and GPU, and its quality must be measured against bf16.
3. **Target:** Vahana decode at least 1.0x llama.cpp's on the same laptop, measured back to back in the same session, median of 3 runs each, 64 tokens after a 512-token prompt. **Stretch:** 1.2x.
4. No regression on the H100 (Qwen3-30B w4a16 decode ≥ 55 tok/s, the 1.3 bench).

## Results
Measured on the laptop (Ryzen 7 4800H, 15 GB RAM, RTX 3050 Ti 4 GB, Windows 11), Qwen3-30B-A3B w4a16 unless stated. This laptop's speed drifts a lot over a session: page cache contents, background processes (Defender, WSL), heat. Every comparison below therefore alternates its variants in the same session and reports medians, never numbers from different hours.

### 1. Profile: where a 1.3 decode step went
From [profile_decode.py](profile_decode.py) and [kernel_layer.py](kernel_layer.py):

| | |
|---|---|
| Decode step, p50 | about 82 ms (9.0 tok/s) |
| CPU expert kernel alone, weights in RAM | **57 ms per token**: 0.89 GB of 4-bit weights at 16.5 GB/s |
| The same kernel with cold pages (random experts) | 548 ms per token |
| GPU expert cache | **0 experts**: the 4 GB held 2.9 GB of non-expert weights (attention expanded to bf16: 1.35 GB; embedding and lm_head: 0.62 GB each) |

Other measurements:
- **Machine read bandwidth:** 28–30 GB/s.
- **The bare q4 dot product** ([bench_q4.cpp](bench_q4.cpp)): reaches 23–24 GB/s with 16 threads, so the kernel's task layout lost about 30%.
- **H100 CPU (EPYC Zen 4, 8 cores), the same kernel:** 34 ms (27 GB/s).
- **Thread scaling on the laptop:** 8 → 16 threads gave no gain. On Windows, logical CPUs 0–7 are only 4 physical cores.

### 2. What changed (all bit-identical to 1.3)
- **4-bit attention stays 4-bit on small GPUs** ([q4lin.py](https://github.com/vahana-ai/Vahana/blob/main/src/vahana/q4lin.py)).
  - Each q/k/v/o projection of a w4a16 checkpoint keeps its twin-q4 records on the GPU.
  - A Triton kernel rebuilds exactly the bf16 weight that 1.3 stored, bf16(q·s), and runs the same 16-lane fma sequence.
  - GPU memory: 1.35 → 0.36 GB. The kernel is 1.3–2× faster than the bf16 twin linear at decode shapes.
  - On GPUs of 12 GB and up it stays off: on the H100 the bf16 path is faster (decode 55.9 vs 34.0 tok/s).
- **Embedding table in CPU RAM on GPUs under 12 GB.** It is an exact lookup of one row per token.
- **The expert-cache reserve is capped at 25% of the GPU.** Together with the two items above, the laptop's GPU expert cache goes from 0 to 386 experts (0.9 GB), covering 24% of profiled expert uses.
- **q4 CPU kernel:**
  - the scale is loaded once per group (no division in the inner loop);
  - a nibble becomes a float with the magic-number trick (`0x4B000000 | n` is 2^23 + n exactly), with one `vpshufb` per 8 values instead of a zero-extend and int→float convert;
  - 64-row tasks (16 → 64 rows: 51.0 → 47.3 ms per token at the kernel boundary; 128 rows starves 16 threads).
- **Bits:**
  - A receipt made with 1.2 on the laptop GPU still verifies: replay on the GPU and prefill on the CPU.
  - `tests/test_q4_kernel.py`: the q4 kernel is byte-identical to the scalar C build on every ISA and task size, in CI on Linux x86-64 and ARM, Windows and macOS.
  - `tests/test_q4lin.py`: the 4-bit linear equals the bf16 expansion, including under the exactness audit.

### 3. Speed: 1.3 vs v1.4
Same session, alternating, three rounds; both use CI-built Windows wheels.

| | 1.3 | v1.4 (`ci/decode-v14`) |
|---|---|---|
| Decode step, p50 | 76.6 ms (76.6 / 70.1 / 80.1) | **63.1 ms** (63.1 / 62.7 / 68.2), 18% faster |
| Decode tok/s (mean step) | 10.7–12.9 | 12.3–12.7 |

The mean is well above the p50 because some steps wait for expert pages from disk: the 16 GB of experts do not fit in the 15 GB of RAM.

**H100** (Qwen3-30B w4a16, every expert cached), bar 4: decode 55.86 tok/s vs 55.84 in 1.3; time to first token 0.40 s in both; receipts verify.

### 4. Against llama.cpp
llama.cpp b12494, Q4_K_M, its best setting (`-ngl 99 --n-cpu-moe 42 -t 8`), `llama-bench -n 64`; alternating with Vahana, three rounds, in the same session:

| Round | llama.cpp | Vahana v1.4 |
|---|---|---|
| 1 | 4.30 tok/s | 8.40 tok/s |
| 2 | 4.76 | 9.52 |
| 3 | 4.80 | 10.91 |
| **Median** | **4.76** | **9.52 (2.0×)** |

Bar 3 (≥ 1.0×) and the stretch bar (1.2×) are met in this session, but this was late in a long benchmark session with 8.5 GB of RAM free. Both engines were slower than earlier in the day, and llama.cpp lost much more: its 18.6 GB model pages harder than Vahana's 16 GB of experts plus the GPU cache.

Earlier, in a fresher state (not interleaved), llama.cpp measured 13.2–15.5 tok/s (experiments/v14_llamacpp) and Vahana v1.4 about 12.3–12.7 tok/s at the mean (63 ms p50).

**So:** Vahana now beats llama.cpp decisively when RAM is under pressure, and is within about 10–20% of it on a fresh machine. On a fresh machine Vahana's typical step is already at llama.cpp's speed (63 ms p50 is 15.9 tok/s); slow steps from expert page-ins pull its average down.

### 5. The tail (quiet laptop, nothing else running)
- **The slow steps were not disk reads.**
  - With the machine quiet, every decode step but one ran at about 60 ms ([tail.py](tail.py)).
  - The exception was the **first decode step of every `generate` call: 430–840 ms**, whatever the prompt length or prefill path. It ran eagerly, outside the CUDA graph: thousands of small launches plus a host sync per CPU job, all slow under Windows WDDM.
  - **Fix:** once a model and batch size have been warmed, later calls capture the graph at their first step. That step drops to 56–92 ms. A 6-token decode, including the capture, takes 0.48–0.56 s instead of about 0.9 s or more.
  - **Bits:** a new test checks that a second call's logits equal the first's (tests/test_prefill.py). The 1.2 receipt replays bit for bit, and the GPU selftest is 18/18.
- **A hard minimum working set** (`SetProcessWorkingSetSizeEx`) made no difference, so RAM residency was not the problem on a quiet machine.

### 6. Against llama.cpp, quiet machine
The page cache decides these numbers: neither model fits in 15 GB of RAM, so whichever engine ran last owns the cache.

**Alternating** (llama.cpp `-r 2`, then Vahana, three rounds):

| Round | llama.cpp | Vahana v1.4 |
|---|---|---|
| 1 | 8.19 ± 3.65 tok/s | 16.53 tok/s |
| 2 | 8.34 ± 3.40 | 14.80 |
| 3 | 8.44 ± 3.24 | 16.40 |

llama.cpp's large spread is its first repetition reading its model back from disk.

**Each engine warm, at its best:**
- **llama.cpp,** three runs of `-r 4` back to back: the repetitions warm up 5.8 → 10.5 → 12.3–15.9 tok/s. Its best repetitions were **13.5, 12.4 and 15.9**.
- **Vahana** reached **14.8–16.5** in the quiet rounds above (p50 58–60 ms per step).
- Run straight after llama.cpp, Vahana measured 5.2, 12.3 and 8.2 tok/s: the cold cache again.

**Verdict:**
- **At their best, warm and quiet:** Vahana 14.8–16.5 tok/s against llama.cpp's 12.4–15.9. About level, with Vahana slightly ahead, but within this laptop's noise.
- **When they compete for RAM, or under memory pressure:** Vahana is about 2× faster.
- This is with Vahana's answers bit-identical on every machine, which llama.cpp's are not (experiments/v14_llamacpp).

### Tried and dropped
- **Windows thread pinning**, one thread per physical core first: 98–167 ms per step against 63 ms unpinned. Removed.
- **Expert prefetch** (`VAHANA_PREFETCH=1/2`): 130–150 ms per step against 84 ms off. It stays off.
- **The w64devkit (mingw g++) build** gives the same kernel-boundary speed as CI's clang/MSVC build, but about 30 ms slower decode steps. Measure only with CI-built wheels.

### Next
- **Reuse the captured graph across calls** (keep the static cache and its buffers), so even the capture is paid once per process.
- **Fidelity:** 1.3's bf16 expansion of 4-bit attention rounds q·s (up to 12 bits) to 8 bits of mantissa. An exact-q4 attention mode would follow the checkpoint more closely. It changes the bits, so it would be opt-in with its own references.
- **An exact integer mode** (int8 activations × int4 weights, integer accumulation, exact in any order) like llama.cpp's arithmetic. It is a different numeric model, so it is a research item.
