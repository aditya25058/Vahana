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

### 7. Round 2 of optimisations, and a controlled head-to-head
Same bits throughout (GPU selftest 18/18 on the laptop and the H100, with and without the expert cache; the 1.2 receipt replays bit for bit).
- **GPU expert kernels skip uncached slots** (they computed every routed slot and then discarded the uncached ones): GPU expert time 19.6 → 7.6 ms per token on the laptop.
- **Fused combine:** one kernel per layer instead of 25–30 small ones. It keeps the summation order (stable by expert id, `y0·w0` then `+ yj·wj`, no FMA contraction). About 2–3% off the laptop step. **H100: 55.9 → 70.8 tok/s (+27%).**
- **Graph and cache reuse across calls:** a 16-token call's decode takes about 0.75 s instead of 0.93–1.42 s.
- **q4 CPU kernel software prefetch:** about 4% at the kernel boundary. It is not in the CI wheel measured below.

**Head-to-head protocol** (`head_to_head.py`, results in `head_to_head_laptop.json`): two blocks, engines alternating A B / B A. Each engine does a discarded warm-up run, then 5 runs. llama.cpp gets `llama-bench -p 0 -n 64 -r 3 -ngl 99 --n-cpu-moe 42 -t 8`, scored as the median of its two warm in-process repetitions. Vahana gets 64 tokens after a 512-token prompt, using the CI-built wheel.

| | runs | median | min–max |
|---|---|---|---|
| llama.cpp b12494 Q4_K_M | 10 | 11.64 tok/s | 10.40–11.80 |
| Vahana 1.4 (w4a16) | 10 | **16.44 tok/s** | 15.12–17.10 |

**Ratio of medians: 1.41×.** Bar 3 (≥ 1.0×) and the stretch bar (1.2×) are met under a protocol fixed before the run. The ranges do not overlap.

Caveat, recorded as is: llama.cpp's best single repetitions in other sessions reached 12.4–15.9 tok/s (section 6). Its warm repetitions under this protocol were 11.5–11.8. The first version of this protocol scored llama.cpp on its cold first repetitions only (`-r 1`, 5.3–5.5 tok/s, a 3.05× ratio). That was unfair, so it was discarded and rerun.

### 8. Round 3: fewer CPU expert reads, faster 4-bit attention
Same bits throughout: 198 tests pass, the GPU selftest is 18/18 with and without the cache, the CPU selftest is 18/18, and the Qwen3-30B receipt replays bit for bit with every change on.

**Where a step went** (`gpu_profile.py`, per token): about 65% was the GPU waiting for the CPU experts. The 4-bit attention projections took 7.2 ms at about 67 GB/s on a 192 GB/s card, the bf16 LM head 3.9 ms, and about 3 ms went to small ops. The CPU expert kernel is compute-bound (`bench_q4.cpp`: about 7 cycles per 32 weights per core). Standalone it reaches the RAM's ceiling, about 25 GB/s. In situ it reaches 19.5–21.6 GB/s.
- **Tried, no gain:** an exact one-FMA weight rebuild, which gives the same bits and is 5% faster on one core but no faster with 8 threads; a subnormal-decode variant; experts in anonymous memory instead of the memory map; thread counts 8–16; prefetch distances and task sizes (all within ±10% noise).

**4-bit attention kernel** (`q4lin_bench.py`): 4 blocks of 32 weights per loop iteration, so their loads are in flight together, plus a launch shape per matrix size. Per layer, inside a CUDA graph: q 45.6 → 31.9 µs, k and v 11.6 → 6.0 µs each, o 62.8 → 36.1 µs, so **132 → 80 µs**. Decode 15.5–16.2 → 16.5–17.1 tok/s.

**Adaptive GPU expert cache** (`adaptive.py`, `routing_sim.py`). The finding: with a fixed cache of 386 experts profiled on built-in text, only **8–19% of real decode expert uses hit the cache**. Which experts are hot depends strongly on the text: a cache profiled on the other prompts is no better than a random one (1.5–13%). Simulated on recorded routing (5 prompts × 96 tokens, 8 slots per layer):

| policy | code | chat | math | story | Hindi |
|---|---|---|---|---|---|
| fixed (shipped in 1.3) | 8.8% | 16.9% | 18.7% | 8.3% | 15.6% |
| re-picked from the prompt's prefill | 30.7% | 36.1% | 27.0% | 31.7% | 50.1% |
| + up to 4 swaps per decode step | 35.1% | 39.4% | 31.7% | 44.4% | 53.4% |
| best fixed set in hindsight | 40.6% | 43.6% | 37.0% | 49.9% | 56.9% |

Decay and margin barely matter (41.6–41.9% mean). Capacity does: 6/8/12/16 slots per layer give 36/42/51/58%.

So the cache now keeps 8 slots per layer and changes what they hold. It re-picks from each prompt's routing after the prefill, and swaps up to 4 experts after each decode step. Each swap updates the GPU lookup table and the CPU kernel's resident set together, between steps, so every routed slot is computed exactly once. Cached and uncached experts give the same bits, so the output never depends on the cache's contents. `test_adaptive.py` checks this with a swap after every step.

The prompt re-pick costs 0.2–0.45 s of time to first token, bound by reading cold expert pages from the memory map. `VAHANA_ADAPTIVE_CACHE=0` turns it off.
- **Benchmark text** (512-token prompt): 14.3–16.5 → **19.6–22.0 tok/s** (p50 44–48 ms).
- **Real prompts** (`prompts_decode.py`): about level (median 13.5–14.6 against 14.4–14.6). Experts these prompts need are paged in from disk: the 16 GB of experts do not fit in this laptop's 15 GB of RAM, which bounds decode there whatever the cache does.

**Head-to-head, benchmark text** (`head_to_head.py`, same protocol as section 7; `head_to_head_laptop_v2.json`):

| | runs | median | min–max |
|---|---|---|---|
| llama.cpp b12494 Q4_K_M | 10 | 11.55 tok/s | 9.99–11.77 |
| Vahana 1.4 | 10 | **21.42 tok/s** | 19.77–22.15 |

**1.86×**, ranges not overlapping.

**Head-to-head, real prompts** (`real_prompts_h2h.py`, `real_prompts_laptop.json`). Five prompts (code, chat, math, story, Hindi), 96 greedy tokens each, engines alternating by block. llama.cpp gets `llama-completion` with its default warm-up at its best MoE setting, and its own timings. Vahana runs one process with one warm-up call. Two blocks:

| | decode median (min–max) | time to first token, median |
|---|---|---|
| llama.cpp | 9.98 tok/s (8.88–11.91) | 7.97 s |
| Vahana 1.4 | **14.01 tok/s** (11.01–18.58) | **3.07 s** |

**Decode 1.40×, time to first token 2.6× faster.** Vahana is faster on every prompt in both blocks, with answers bit-identical on any machine.

### 9. The RAM limit

On real prompts the laptop is bound by RAM, not compute:
- Qwen3-30B's 4-bit experts take 14.3 GB. The laptop has 15.8 GB, and Windows and apps hold about 5 GB, so about 72% of the experts stay resident.
- In one five-prompt run, free memory fell to about 250 MB and Vahana read 22.7 GB from disk.
- The missing experts arrive through page faults at 100–600 MB/s, against 1.4 GB/s from the SSD in large reads. That is an estimated 20–40 ms of a 70–80 ms decode step.
- Vahana's own private working set is about 0.3 GB. (Windows' private bytes overstate it, because torch's CPU allocator keeps freed large blocks committed.) The pressure comes from the experts themselves.

Two changes, both hints to the OS, so no bit depends on them:
- **Cold-expert prefetch** (`VAHANA_PREFETCH=3`). When a layer's job uses an expert it has not used in its last 64 jobs (`VAHANA_PREFETCH_COLD`), the kernel asks the OS to read that whole expert in one large request (`PrefetchVirtualMemory` / `MADV_WILLNEED`). Experts used recently are skipped, so a machine with enough RAM pays nothing. It is the default when CPU experts exceed 80% of RAM.
  - Real prompts, mingw build (relative A/B only): 9.0–10.6 tok/s on (8 runs) against 7.4–8.7 off (3 runs), **about +17%**.
  - The code prompt's first token: about 5 s → 3 s.
- **Page release.** After a GPU load, the checkpoint's non-expert tensors and the host copies of GPU-cached experts are not read again. `VirtualUnlock` / `MADV_COLD` puts their pages first in line to be dropped, so RAM goes to CPU experts. Effect within noise (+0–5%). Kept, because it costs nothing.

**Head-to-head, real prompts, with both** (CI-built wheel; same protocol as section 8; `real_prompts_laptop_v2.json`):

| | decode median (min–max) | time to first token, median |
|---|---|---|
| llama.cpp | 9.74 tok/s (7.36–12.46) | 8.21 s |
| Vahana 1.4 | **15.33 tok/s** (10.32–17.58) | **2.92 s** |

**Decode 1.57× (was 1.40×), time to first token 2.8× faster.** Vahana is faster on every prompt in both blocks.

### Tried and dropped
- **Windows thread pinning**, one thread per physical core first: 98–167 ms per step against 63 ms unpinned. Removed.
- **Whole-job expert prefetch** (`VAHANA_PREFETCH=1/2`): 130–150 ms per step against 84 ms off on the benchmark text, and real prompts 13 → 10 tok/s. Only the cold-expert mode 3 (section 9) is kept.
- **Predicting the next layer's cold experts** (`VAHANA_PREFETCH=4`: the next layer's router on this layer's input, a layer ahead). The guesses are good: 83% of the next layer's CPU experts. Decode is still slower than mode 3 (median 8.3 against about 10.3 tok/s, mingw build). It prefetched 51 GB against 25 GB, and with RAM full every wrong guess, about 2.4 MB, evicts an expert that will be needed. Kept as an opt-in, tested for the same bits.
- **The w64devkit (mingw g++) build** gives the same kernel-boundary speed as CI's clang/MSVC build, but about 30 ms slower decode steps. Measure only with CI-built wheels.

### Next
- **CPU expert kernel toward memory bandwidth:** about 20 GB/s today, against 28–30 GB/s raw.
- **Fuse the small GPU ops** (router, norms, rope) that remain between graph nodes.
- **Fidelity:** 1.3's bf16 expansion of 4-bit attention rounds q·s (up to 12 bits) to 8 bits of mantissa. An exact-q4 attention mode would follow the checkpoint more closely. It changes the bits, so it would be opt-in with its own references.
- **An exact integer mode** (int8 activations × int4 weights, integer accumulation, exact in any order) like llama.cpp's arithmetic. It is a different numeric model, so it is a research item.
