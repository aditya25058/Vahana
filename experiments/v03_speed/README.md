# v0.3-a: GPU expert cache (hot experts in spare GPU memory), still exact

Pre-registered 2026-11-24, after the popularity measurement and the tiny-model exactness check, before any real-model cache run.

**Why a cache, and why it can be exact:**
- With experts offloaded, decode speed is set by the expert bytes read per token.
- **Expert use is skewed** (skew_*.json: cache chosen on half the 50 prompts, hits measured on the other half):

  | Cached fraction of each layer | OLMoE-1B-7B (64 experts, top-8) | Qwen3-30B-A3B (128 experts, top-8) |
  |---|---|---|
  | 10% | 20% of expert uses | 27% |
  | 25% | 44% | 51% |
  | 50% | 71% | 78% |
- **The cache is exact by construction.** A cached expert runs on the GPU with the CPU kernel's twin arithmetic (cache.py `_cgate_up` / `_cdown`, the same ops as hybrid `_gate_up` / `_down`). The other slots run on the CPU through `csrc moe_slots`. All K slot outputs are combined in ascending expert id with the CPU kernel's own operations. So what is cached changes speed, never bits.
- **The cache choice:** experts are ranked globally by popularity. It is profiled once per model on 8 built-in texts (code, maths, English, Spanish, Chinese, email, science) and saved under `~/.cache/vahana/popularity`.

**Development check (before this README):** `vahana selftest --device cuda --gpu-cache f` on the H100, graph decode: **18/18 golden hashes at f = 0.1, 0.5 and 1.0.** gpt-oss layers are not cacheable (gated activation with biases) and keep the CPU path.

**Bars:**
- **CACHE-REAL (H100):** OLMoE-1B-7B and Qwen3-30B-A3B with `gpu_cache_gb=16`. Every hash equals the P4 references:
  - OLMoE: 50 prefills + 4 generations;
  - Qwen3: 8 prefills + 2 generations.
- **CACHE-SPEED (H100, Qwen3-30B-A3B, graph decode):** with a cache sized like a 24 GB consumer GPU (`gpu_cache_gb=18`), decode tok/s is ≥ 1.3× the no-cache run (same prompts, same session, alternated ×2).
- **CACHE-LAPTOP (RTX 3050 Ti, OLMoE, experts streamed from NVMe):** `gpu_cache_gb="auto"` (what's free on the 4 GB card). Bits equal the P4 references; decode tok/s is reported against P4's 4.1–4.4.

**Prediction:**
- CACHE-REAL passes.
- CACHE-SPEED is 1.5–2.5× (about 50–65% of expert uses served from VRAM; the CPU part shrinks accordingly).
- Laptop: 1.1–1.4× (about 1.5 GB cache ≈ 12% of experts ≈ 25% of uses; disk reads drop more than that, because the page cache then holds the next-hottest experts).

## Amendment (2026-11-24, before the counted runs below)
**The first CACHE-SPEED attempt (development, not counted) found the real bottleneck.** Qwen3-30B with an 18 GB cache was slower than without one: 4.1–4.8 vs 6.7–8.1 tok/s. Profiling (v3_prof*.py) traced the cause:
- **The CPU work is small:** the experts take 0.3–0.7 ms per layer.
- **The overhead is the round trip:** each CPU step inside a CUDA graph cost about 1.3–1.4 ms of fixed overhead, made of a host node (cudaLaunchHostFunc) plus small device↔pinned memcpy nodes. Graph step, Qwen3-30B decode:

  | CPU step per layer | Step |
  |---|---|
  | None | 21.8 ms |
  | An empty host job | 83–90 ms |
  | The real job | 107–128 ms |
- **Two fixes made along the way:**
  - **The cache-hit mask copy** (about 1 ms per layer) is gone. The CPU kernel now knows the resident set (`register_resident`).
  - **The combine moved into the CPU job** (csrc `moe_cached`).

**New: the GPU↔CPU mailbox (runtime.roundtrip, csrc mailbox_loop).** It is the default in graph decode; `VAHANA_MAILBOX=0` keeps the old path.
- **How:**
  - one Triton kernel per CPU job copies the job's inputs into pinned host memory (UVA stores);
  - it publishes a sequence number (system-scope release) and polls until the kernel's CPU thread stores it back;
  - then it copies the output to the device.
  - The CPU thread polls the request flags and runs the job directly.
- **No host node and no memcpy node:** an empty job per layer costs 22.4 ms per step vs 21.8 ms with no CPU step, about 12 µs per round trip.
- **Exactness:** the same job code and buffers, only a different signal. Development checks: selftest on the GPU 18/18, with and without `--gpu-cache 0.5`.

**Added bars (the original bars stand):**
- **MAILBOX-REAL (H100):** with the mailbox (and the 16 GB cache, as in CACHE-REAL), the OLMoE and Qwen3 hashes equal the P4 references.
- **MAILBOX-SPEED (H100, Qwen3-30B-A3B, no cache, graph decode):** decode tok/s with the mailbox is ≥ 1.5× the host-node path. Same process, alternated ×2, 4 generations × 64 tokens each.
- **MAILBOX-MIXTRAL (H100, Mixtral-8x7B, 4 GB budget, m4_real.py, the M4/P3 settings):** steady decode step vs P3's 78.5 / 78.8 ms. Reported, with the bar ≥ 1.2× faster.
- **CACHE-SPEED is now measured with the mailbox on:** cache18 vs no cache, bar ≥ 1.3× unchanged.

## Result (H100): MAILBOX-REAL, CACHE-REAL and MAILBOX-SPEED pass; CACHE-SPEED and MAILBOX-MIXTRAL not met
| Bar | Result |
|---|---|
| CACHE-REAL / MAILBOX-REAL: hashes == P4 references, mailbox + 16 GB cache | **OLMoE 50/50 prefills, 4/4 generations; Qwen3 8/8, 2/2** (ids and logits) |
| MAILBOX-SPEED: Qwen3-30B, no cache, mailbox vs host node | **1.72×**. Mean 14.7 vs 8.5 tok/s over 8 generations each (r1 15.3–16.6 / r2 10.5–15.0 vs 8.0–9.1); steady steps 59–64 ms vs 110–122 ms |
| CACHE-SPEED: Qwen3-30B, mailbox on, 18 GB cache vs none | **1.12×, not met (bar 1.3×)**. Mean 16.5 vs 14.7 tok/s; warm generations 18.0–18.5 vs 14.7–16.6; steady steps 52–53 ms vs 59–65 ms |
| MAILBOX-MIXTRAL: Mixtral-8x7B, 4 GB budget, M4/P3 settings | **Not met: no gain.** Steady step 123.6 / 130.2 ms vs 131.4 / 124.7 ms with host nodes in the same session |

- **Extra evidence:** all 24 Qwen3 speed generations (host node, mailbox, mailbox + cache) produced identical ids.
- **Mixtral graph == eager:** ✓ in all 4 runs.

**Reading:**
- **The mailbox is the big win where the CPU work per layer is small.** Qwen3-30B has 128 small experts and 48 layers. There a host node's fixed cost (about 1.3 ms) outweighed the expert work, and removing it makes decode about 1.7–1.9× faster at identical bits.
- **Mixtral is the opposite case:** two large experts per layer (about 16 GB of expert reads per token). Its step is CPU memory-bandwidth-bound, so the host-node cost did not show.
- **This session's Mixtral numbers (124–131 ms) are far above P3's 78.5 ms** with the same code path and settings. The host was shared and its CPUs busy (load average 6–8 from other users' jobs). They show that the mailbox did not help Mixtral; they do not show a regression.
- **The cache's gain is small once the mailbox is on.** With 70% of expert uses served from VRAM, the remaining step is mostly GPU work (about 22 ms) plus the CPU's share of the experts, which leaves 1.1–1.2×. The cache matters more where expert reads are the bottleneck: streaming from disk, and slower or busier CPUs.

## Result (laptop): CACHE-LAPTOP passes on bits; 2.8× faster than P4
**Setup:** Ryzen 7 4800H, RTX 3050 Ti 4 GB, native Windows 11, torch 2.11 + triton-windows. The kernel is the CI-built Windows wheel from ci/v03. OLMoE-1B-7B, experts streamed from NVMe; the cache was set to 1.6 GB because "auto" would leave less on a 4 GB card.

| | Result |
|---|---|
| Selftest, GPU, mailbox, `--gpu-cache 0.5` | **18/18** (laptop_v3_selftest_gpu_cache05.txt) |
| Mailbox + 1.6 GB cache (136 experts, 30% of profiled uses) vs the H100 P4 reference | **50/50 prefills, 4/4 generations** (ids and logits) |
| Decode, mailbox, no cache (2 runs × 4 generations, 32 tokens) | mean **8.7 tok/s** (6.8–11.5) |
| Decode, mailbox + cache (same) | mean **11.9 tok/s** (7.6–14.5), 1.36× over no cache |
| P4 on this laptop (host nodes, no cache) | 4.1–4.4 tok/s |

**Reading:**
- **Same bits, about 2.8× faster** than P4 on the same laptop and model.
- **The mailbox works on Windows with a consumer GPU** (system-scope atomics on pinned host memory under WDDM).
- **Here the cache adds 1.36×,** more than on the H100 (1.12×). On the laptop, expert reads (RAM / disk) are the bottleneck.
