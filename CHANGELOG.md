# Changelog

Each version's evidence is in `experiments/` (pre-registered bars, run logs, raw results). Missed bars are recorded there too.

## 1.4.0 (2026-11-27)
- **Faster decode on small GPUs, same bits.**
  - On GPUs under 12 GB:
    - 4-bit attention weights stay 4-bit on the GPU (a kernel rebuilds 1.3's exact bf16 weight);
    - the embedding table moves to CPU RAM;
    - the expert-cache reserve is capped at 25% of the GPU.
  - The 4-bit CPU kernel decodes nibbles more cheaply and uses 64-row tasks.
  - Laptop RTX 3050 Ti, Qwen3-30B 4-bit: typical decode step 76.6 → 58–63 ms (14.8–16.5 tok/s on a quiet machine).
  - Against llama.cpp on the same laptop, in an alternating warm protocol: **1.41× faster decode** (median 16.44 vs 11.64 tok/s, 10 runs each, ranges not overlapping). About 2× faster under memory pressure.
- **GPU expert kernels skip uncached slots** (GPU expert time 19.6 → 7.6 ms per token on the laptop). A **fused combine kernel** keeps the summation order. **H100: 55.9 → 70.8 tok/s.** Same bits.
- **Graph and cache reuse across `generate` calls:** about 0.2–0.7 s saved per call.
- **The first decode step of a call no longer runs eagerly** once the model is warm: it captures the CUDA graph straight away (430–840 → 56–92 ms on Windows). Same bits.
  - H100: unchanged.
  - Details: [experiments/v15_decode](https://github.com/vahana-ai/Vahana/blob/main/experiments/v15_decode/README.md).

## 1.3.2 (2026-11-27)
- The PyPI page's links work. The README, which is also the PyPI description, used relative links, and those break on PyPI; they are now absolute GitHub URLs. A test keeps them that way.

## 1.3.1 (2026-11-27)
- A GPU run without a working Triton now fails at once, with the import error and the fix. Before, every GPU path crashed with a cryptic `TypeError` from a stand-in kernel. `vahana selftest` and `vahana doctor` show the reason too.
- `vahana[gpu]` installs `setuptools` on Linux. Triton 3.2+ imports it at startup, and a fresh Python 3.12 venv lacks it.
- `vahana --help` lists `verify` and `bench`. The README has a first-five-minutes guide.
- Validated on a third GPU: NVIDIA A100 80GB. The GPU selftest is 18/18, and the prefill, prefix-cache and receipt tests pass, with 1.3.0 installed from PyPI.

## 1.3.0 (2026-11-26)
This release brings together the 1.1–1.3 work:
- transformers 5.12–5.24;
- exact 4-bit;
- receipts and bench;
- fast exact prefill and the prefix cache.

It also adds a same-laptop comparison with llama.cpp ([experiments/v14_llamacpp](experiments/v14_llamacpp/README.md)).

Behavior changes:
- **Seeded sampling.** Seeded outputs differ from 1.0, because sampling now uses the exact sampler. The same seed now gives the same tokens on every machine.
- **Long prompts on a GPU** take the new prefill path. It gives the same bits; `VAHANA_PREFILL_GPU=0` turns it off.

- **Persistent prefix cache** (`vahana run --prefix-cache [DIR]`, `generate(prefix_cache=...)`).
  - A prompt that starts like an earlier one (a document, a system prompt) reuses that prefix's attention state from disk and prefills only the rest.
  - The logits are bit-identical to a full prefill.
  - Laptop, Qwen3-30B 4-bit, 2,048-token document with a new question: 12.2 → 1.5 s to first token.
  - Batch 1, attention models.
- **Long prompts are much faster on small GPUs, with the same bits.**
  - From 32 tokens, a prefill streams each layer's CPU-resident experts to the GPU and runs them with new tiled kernels.
  - The kernels keep the CPU kernel's exact arithmetic. Only the tiling changes, so the logits are bit-identical (also across machines).
  - Qwen3-30B 4-bit on the 4 GB laptop: 512-token prompt 24.8 → 5.7 s, 2,048-token prompt 82 → 9.6 s.
  - H100 with a 4 GB expert cache: 2,048 tokens 16.0 → 3.8 s.
  - Twin linears also have a tiled prefill kernel (1.4–2× on projections).
  - `VAHANA_PREFILL_GPU=0` turns the path off.
  - Details: [experiments/v13_prefill](experiments/v13_prefill/README.md).
- **Receipts: verifiable inference** (`vahana run --receipt`, `vahana verify`).
  - A receipt records the prompt, settings, weight fingerprint, output and a digest of every step's logits. Anyone with the same weights can check it bit for bit on their own hardware.
  - Attention models are checked with one teacher-forced pass: prefill logits equal decode logits bit for bit in every tested family.
  - Hybrid models (Qwen3.5) are replayed.
  - Laptop RTX 3050 Ti (Windows) receipts verify on an H100 GPU and on an AVX-512 CPU (Linux): OLMoE greedy and sampled, Qwen3-30B 4-bit sampled. Qwen3.5-35B hybrid receipts made on the H100 GPU also verify on its CPU (replay).
  - Details: [experiments/v12_receipts](experiments/v12_receipts/README.md).
- **Exact sampling.** `temperature`/`top_p`/`seed` now use Vahana's exact sampler: IEEE basic operations only, and a SHA-256 counter stream instead of torch's generator. The same seed gives the same tokens on every machine and every numpy/torch version. This applies in the server and in `vahana run`; seeded outputs differ from 1.0's.
- **`vahana bench MODEL`**: one command reports:
  - load time, time to first token and prefill tok/s;
  - decode tok/s with p50/p95 per-token latency;
  - peak GPU memory and RAM;
  - a receipt verified on the spot, and two selftest families;
  - `--json` for publishing.
- **transformers 5.12–5.24 supported** (was 5.12–5.13).
  - Some releases change a model family's code (Qwen3.5-MoE at 5.14 and 5.17, gpt-oss at 5.19), which changes that family's outputs on every machine alike.
  - The selftest now keeps a reference per transformers version and only warns outside the validated range.
  - CI checks all 13 supported releases, plus a canary on the newest one.
  - Fix: transformers 5.17+ runs Qwen3.5 through a LAPACK triangular solve whose rounding varied by CPU (x86 vs ARM runners gave different bits). Vahana now computes it exactly (`tmath.tsolve_triangular`).
  - `vahana selftest` now audits every family and fails any that uses a float op outside Vahana's exact math, even when the local hash matches.
  - Details: [experiments/v11_transformers](experiments/v11_transformers/README.md).
- **Exact 4-bit experts (twin-q4).** Vahana runs 4-bit MoE checkpoints (compressed-tensors pack-quantized int4, e.g. GPTQ w4a16) with identical bits on every CPU instruction set and machine.
  - Qwen3-30B-A3B w4a16 on a 4 GB laptop GPU with 15 GB RAM: 0.96 → 4.66 tok/s, disk traffic per token 1.8 → 0.09 GB.
  - Identical output hashes on an Intel laptop (Windows, AVX2) and an AMD EPYC server (Linux, AVX-512).
  - Quality is the checkpoint's own.
  - Details: [experiments/v12_q4](experiments/v12_q4/README.md).
- **GPU-cached and CPU experts now run at the same time.** Previously the GPU's cached experts ran first and the CPU part ran after.
  - Qwen3-30B on an H100 with a 16 GB expert cache: median decode 13.2 → 18.0 tok/s, on a shared and noisy machine; about 1.16× at matched load.
  - Same bits. `VAHANA_OVERLAP=0` restores the old path.
  - No gain when the cache holds little, for example a 4 GB laptop GPU.
  - Details: [experiments/v11_overlap](experiments/v11_overlap/README.md).

## 1.0.1 (2026-11-25)
- The macOS wheel is tagged `macosx_11_0_arm64` instead of `universal2`. It only ever contained Apple-silicon kernels, so on an Intel Mac pip would install 1.0.0 and its kernel would fail to load. Intel Macs can build from the source distribution.

## 1.0.0 (2026-11-24)
First public release (beta). The exactness guarantee is stable. Licensed under Apache-2.0.
- Dependencies pinned to the validated range: transformers 5.12–5.13 (later releases change Qwen3.5-MoE and gpt-oss outputs; found by a clean-install test before publishing), torch ≥ 2.4. CI logs the versions it tested, and a `compat` job runs the selftest on every transformers release.
- Docs: README quick start, [docs/benchmarks.md](docs/benchmarks.md) (every number linked to its experiment), this changelog.
- `vahana --version`. `vahana --help` lists `selftest` and `doctor`.
- Release pipeline: a version tag builds and checks every platform's wheel plus an sdist, then publishes to PyPI by trusted publishing ([docs/RELEASING.md](docs/RELEASING.md)). `pip install "vahana[gpu]"` picks Triton on Linux and triton-windows on Windows.
- No engine changes since 0.5.

## 0.5 (streaming and doctor)
- **Expert prefetch for disk streaming** (`VAHANA_PREFETCH=1/2`, opt-in): `PrefetchVirtualMemory` / `madvise(WILLNEED)` hints.
  - Mode 2 predicts the next layer's experts with its router (98% recall).
  - Hints only, so the bits are identical. On a saturated laptop NVMe it gives 1.15×.
- `vahana doctor MODEL --measure`: loads as `vahana run` would and times a real decode.
- `vahana run` / `serve`: `--gpu-cache` (default `auto`) caches the hottest experts in spare GPU memory.

## 0.4 (serving)
- **0.4-c:** exact batching and serving for linear-attention hybrids (Qwen3.5-MoE, MiniMax). Each request is prefilled alone and decode steps are shared. Real Qwen3.5-35B-A3B: 8/8 bit-identical, 4.35× one at a time.
  - Loader: vision-language checkpoints (text model under `model.language_model.`) and fused 3-D expert tensors.
  - Fix: `decode.generate` now stops at EOS for dynamic-cache models.
- **0.4-b:** `engine.Engine`, continuous batching behind the OpenAI-compatible server (`vahana serve --slots`). Requests join and leave between steps, and every response is bit-identical to running it alone. 7.4× one at a time.
- **0.4-a:** CPU expert jobs take up to 64 rows and skip padding: Qwen3-30B at batch 64 is 2.46× faster.

## 0.3 (speed at the same bits)
- **0.3-e:** `vahana.generate_batch`: batched decoding where each request keeps its solo bits.
- **0.3-d:** multi-token twin linear (one weight load for up to 8 tokens).
- **0.3-c:** fused router softmax and grouped GPU expert kernels.
  - Fixes an FTZ issue: Triton 3.0 flushed subnormals in library division. Vahana now emits PTX `div.rn` / `sqrt.rn` directly.
  - The softmax uses one canonical NaN on every device.
- **0.3-b:** exact speculative decoding (prompt lookup), bit-identical to plain greedy.
- **0.3-a:** a GPU↔CPU mailbox replaces CUDA host callbacks (1.72× on Qwen3-30B), and a GPU expert cache for the most-used experts.

## 0.2 (portability)
- One CPU kernel for AVX-512, AVX2, NEON and scalar, behind a plain C ABI: no libtorch or OpenMP dependency; wheels for Linux x86-64 / arm64, Windows, macOS.
- Twin arithmetic for every operation, so CPU-only machines give the same bits as a GPU.
- `vahana selftest`, `vahana doctor`; models bigger than RAM are streamed from disk, still exact.

## 0.1
- MoE experts in CPU RAM on an exact CPU kernel, everything else on one GPU, CUDA-graph decode, and an OpenAI-compatible server.
- Batch-invariant attention. NX12 lossless expert encoding.
- Validated on DeepSeek-V2 236B, Mixtral-8x7B, Qwen-MoE and OLMoE.
