# v0.5: streaming from disk with expert prefetch, and a measured `vahana doctor`

Pre-registered 2026-11-24, after the baseline and before the counted runs.

**Baseline (`base_laptop_qwen3.json`):** Qwen3-30B-A3B (57 GB) on the laptop (15.4 GB RAM, NVMe SSD, RTX 3050 Ti 4 GB, GPU cache empty because the card is full).
- 0.96 tok/s decode, reading 1.8 GB from disk per token.
- Decode is I/O-bound, and the I/O happens as synchronous page faults inside the CPU expert kernel: small reads, a shallow queue, no overlap with compute.

**Code (exact by construction: only hints to the OS, no data or arithmetic changes):**
- **csrc `prefetch_job`:** `PrefetchVirtualMemory` on Windows, `madvise(WILLNEED)` on Linux/macOS.
  - **Mode 1:** at the start of each CPU expert job, the selected experts' gate/up/down ranges are prefetched whole (large reads, deep queue).
  - **Mode 2:** also, the next MoE layer's experts are predicted with its router applied to this layer's input (top 2K per token, skipping GPU-cached ones) and prefetched by a background thread while this layer computes. The prediction hit rate is counted.
- **loader:** registers each layer's router (a host bf16 copy) and enables mode 2 when the CPU experts do not fit in available RAM. `VAHANA_PREFETCH=0/1/2` overrides.
- **`vahana doctor MODEL --measure`:** loads the model as `vahana run` would and decodes 16 tokens on a fixed prompt, reporting the measured decode tok/s next to the formula estimate.

**Bars:**
- **EXACT:**
  - `vahana selftest` with `VAHANA_PREFETCH=2`: CPU 18/18 under each ISA build and CUDA 18/18 (with and without half cache), on the laptop and the H100;
  - streaming runs: the hash of ids plus every logit is identical across prefetch modes 0 / 1 / 2 for every prompt.
- **SPEED (laptop, Qwen3-30B streamed):** decode tok/s with mode 2 ≥ 1.3× mode 0, over 2 alternated rounds of 3 prompts × 24 tokens. Mode 1 and the prediction hit rate are reported. H100 (RAM-resident): mode 2 not slower than 0.97× mode 0 (OLMoE, fully on the CPU).
- **DOCTOR:** `--measure` is within ±20% of a full 32-token run's decode tok/s on the same machine for laptop OLMoE, laptop Qwen3 and H100 Qwen3.

**Prediction:** mode 1 gives 1.1–1.4× (deeper I/O queue); mode 2 gives 1.3–2× with about 60–80% prediction hit; bits are identical in every mode.

## Results (laptop: counted). The H100 part is pending (see below)
| Bar | Result |
|---|---|
| EXACT: selftest with `VAHANA_PREFETCH=2`, laptop CPU (avx2, scalar) and GPU with half cache | **18/18** each |
| EXACT: streaming hashes (ids plus every logit) across modes 0 / 1 / 2, every prompt | **identical** (`55112183…`, `c36d64fe…`, `df8cbf08…` in all 6 runs) |
| SPEED laptop Qwen3-30B, mode 2 vs mode 0 (bar ≥ 1.3×) | **0.76×, not met** (0.713 vs 0.935 tok/s; disk 4.7 vs 1.85 GB per token). Mode 1: 0.98× (0.914) |
| Prediction hit rate (next-layer router on this layer's input) | **98%** of the experts actually used were among those predicted |
| DOCTOR laptop: `--measure` within ±20% of a full run | **Not met:** OLMoE 0.71× (6.80 vs 9.58 tok/s), Qwen3 1.23× (1.19 vs 0.96) |

**Why prefetch did not pay on this laptop:**
- Decode time per token (about 1.06 s) equals bytes read ÷ about 1.75 GB/s.
- Raw single-stream reads from this NVMe reach 1.0–1.25 GB/s (`rawread.py`: 9.4 MB random reads and 64 MB sequential).
- So demand paging already saturates the drive and compute is hidden. Prefetching overlaps I/O but cannot make a saturated drive faster.
- Mode 2 as pre-registered predicted 2K = 16 experts per token for 8 used, which doubled the bytes read (and evicted hot pages from 15 GB of RAM).

**Follow-ups (after the counted runs, not counted):**
- **Prediction width = K** (`VAHANA_PREFETCH_TOPK`, now the default): mode 2 runs at 1.107 and 1.081 tok/s vs 0.947 for mode 0, so **1.15×**. Disk reads are 2.6 GB per token.
- **Prefetch is off by default** (`VAHANA_PREFETCH=1/2` opts in), because the counted bar was not met and the gain depends on the drive.
- **`vahana run` / `serve` now use the GPU expert cache by default** (`--gpu-cache auto`), which they never did before.

**Reading:**
- Exactness is untouched by any of this; prefetching is only hints to the OS.
- On a laptop whose NVMe is already saturated, the next lever is to read fewer bytes. For example: stream Vahana's lossless NX12-compressed experts from disk (about 24% fewer bytes), or keep the hot experts pinned in RAM instead of relying on the OS's LRU.
- **The doctor's `--measure` is too noisy as a single 16-token decode while the page cache warms.** It needs a longer warm-up and several prompts.

## Results (H100: counted, rerun on GPU 1 only)
GPU 0 was held (49.8 GB) by another user's vLLM server running outside the scheduler, which always hands out GPU 0 first. So the counted run reserved both GPUs (`gpurun -g 2`, nothing preempted) and computed only on GPU 1 (`on_gpu1.sh` aborts unless GPU 1 is empty at start; it was), leaving GPU 0 untouched.

| Bar | Result |
|---|---|
| EXACT: selftest with `VAHANA_PREFETCH=2`: CPU per ISA (avx512, avx2, scalar), CUDA with no cache and with half | **18/18** each |
| RAM-resident OLMoE (all experts on the CPU), mode 2 vs mode 0, alternated (bar ≥ 0.97×) | **0.958×, not met** (60.7 vs 63.3 tok/s; per run: mode 0 62.1 / 64.5, mode 2 57.6 / 63.8). On this box the experts are in-RAM NX packs, so `prefetch_job` returns at once: the gap is run-to-run noise, not prefetch cost |
| DOCTOR H100 Qwen3-30B: `--measure` within ±20% of a full run | **0.90×, met** (56.4 vs 62.6 tok/s) |

Code note: the H100 runs used the kernel as first pre-registered (prediction width 2K, auto-enable when streaming); the top-K / off-by-default change came after them. It does not affect these H100 results: every run forced the mode explicitly, and on this box prefetch is a no-op (NX packs in RAM). The H100 checkout has since been rebuilt from the committed code.

The first (killed) H100 run had computed on GPU 0 beside that vLLM server: OLMoE there ran 34 tok/s, against 63 on GPU 1 here.

**Overall v0.5:**
- **Exact everywhere.**
- **Prefetch:** the counted speed bar was not met on the laptop. The top-K follow-up gives 1.15×. Prefetch ships off by default (opt-in).
- **Doctor `--measure`:** met on the H100 (0.90×), not on the laptop (0.71× / 1.23×, page-cache warm-up noise).
- **New default:** `vahana run` / `serve` use the GPU expert cache.

**Earlier H100 attempt:** the first run was stopped by `gpurun --kill` from the shared account. Before the stop: CPU selftest 18/18 per ISA and CUDA 18/18 ±cache with prefetch forced; RAM-resident OLMoE mode 2 at 33.9 / 35.4 vs mode 0 at 34.2 tok/s. Logs are in `run1/`. Rerun from scratch afterwards (above).
