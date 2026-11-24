# Changelog

Each version's evidence is in `experiments/` (pre-registered bars, run logs, raw results). Missed bars are recorded there too.

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
