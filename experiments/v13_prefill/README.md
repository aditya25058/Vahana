# v1.3 GPU-streamed expert prefill

## Question
Long prompts are slow on small GPUs: Qwen3-30B-A3B w4a16 on a 4 GB laptop takes 92 s for a 2,048-token prompt, because the CPU computes every token's experts. Copying each layer's experts to the GPU costs the same for any prompt length. Above a threshold, does running the prompt's experts on the GPU (GPU twin kernels, so the CPU kernel's bits) cut time to first token, without changing a bit?

## Pre-registered bars (written 2026-11-26, before any timing)
1. Bits: the prefill logits with the GPU path equal the CPU path's, bit for bit, on real checkpoints. All 18 selftest families keep their reference hashes with the path forced on (threshold 8), with and without a GPU expert cache.
2. Laptop (RTX 3050 Ti 4 GB, 15 GB RAM), Qwen3-30B-A3B w4a16:
   - 2,048-token prompt in under 15 s (from 92 s);
   - 512-token prompt at least 3x faster than the CPU path.
3. H100, Qwen3-30B-A3B w4a16 with the default (auto) GPU cache: prefill no slower than the CPU path at 512 and 2,048 tokens.
4. Decode speed unchanged: the path only runs for batches of at least the threshold, never in a captured decode step.

## Results
All four bars are met.

**1. Bits.**
- With the path forced on (threshold 8), all 18 selftest families keep their reference hashes on the laptop GPU, both without a GPU expert cache and with half the experts cached. 65 expert layers took the new path.
- On real checkpoints, the last-position prefill logits are identical with the path on and off, for every run below.
- The prefill logits hash is also identical across machines. Qwen3-30B w4a16 gives `87526b48…` (512 tokens) and `37996244…` (2,048 tokens) on both the laptop and the H100.
- The tiled kernels equal the decode kernels bit for bit on real 4-bit weights ([compare_kernels.py](compare_kernels.py)). Unit tests: [tests/test_prefill.py](../../tests/test_prefill.py).

**2. Laptop** (RTX 3050 Ti 4 GB, 15 GB RAM; Qwen3-30B-A3B w4a16, 16 GB of experts, so they are read from the memory-mapped cache):

| Prompt | v1.2-equivalent CPU path | v1.3 CPU path (tiled linears) | v1.3 GPU-streamed | vs v1.2 |
|---|---|---|---|---|
| 512 tokens | 24.8 s | 20.0 s | **5.7 s** (3.5x vs the v1.3 CPU path) | 4.4x |
| 2,048 tokens | 82.2 s | 74.1 s | **9.6 s** (7.7x) | 8.6x |

The v1.2-equivalent column uses `VAHANA_PREFILL_GPU=0 VAHANA_TILE_MIN=10^8`; [bench_prefill.py](bench_prefill.py) has the commands.

**3. H100 NVL**, prefill vs the CPU-expert / cached-decode-kernel path:

| Setup | 512 tokens | 2,048 tokens |
|---|---|---|
| Qwen3-30B w4a16, every expert cached (auto) | 0.82 → 0.40 s | 5.27 → 1.66 s |
| Qwen3-30B bf16, every expert cached (auto) | 0.65 → 0.39 s | 4.57 → 1.61 s |
| Qwen3-30B bf16, 4 GB cache (27% of uses) | 3.48 → 2.65 s | 16.03 → 3.82 s |

**4. Decode.**
- The path runs only for batches of at least 32 tokens, and never while a CUDA graph is being captured.
- Laptop `vahana bench` decode: 6.38 tok/s (v1.2: 5.16 to 6.75 tok/s on this machine across runs). The bench's receipt verification passes.
- The selftest is 18/18 on CPU and GPU with default settings.

## What made it fast
Profiles (`VAHANA_PREFILL_PROFILE=1`, [kernel_profile.py](kernel_profile.py), [profile_layer.py](profile_layer.py)) found four problems in turn:
1. **Experts computed on the CPU.** At 2,048 tokens on the laptop, the CPU spent about 70 s on experts. The new path streams them to the GPU.
2. **Decode kernels used for prefill.** They tile one slot (or MT slots) by BR rows and reach about 0.9 TFLOP/s on the laptop GPU.
   - Tiled kernels (BM slots x BN rows x 16 lanes) reach 2.6 TFLOP/s with the same arithmetic.
   - The same tiling applied to the twin linears (`twinlin._lint`) gives 1.4 to 2x on attention projections.
3. **Host syncs that serialized the pipeline:**
   - a lookup table built from a Python list on the device;
   - `torch.bincount`, which syncs to size its output;
   - waiting on the whole compute stream instead of the one buffer's last reader.
   With these removed, gather, copy and compute overlap.
4. **Single-threaded gather** from the memory-mapped cache. Four threads are faster on the laptop; eight are slower (disk contention).

## Misses and what did not help
- **Read-ahead** streams each layer's experts before routing is known, assuming nearly all are used. It made the laptop slower: 7.2 → 9.7 s at 512 tokens.
  - It moved about 40% more bytes, because this prompt uses about 90 of 128 experts per layer.
  - Its background reads competed with the foreground ones.
  - On the H100 at 2,048 tokens it helped (3.8 → 3.7 s), but it hurt at 512 tokens.
  - Off by default (`VAHANA_PREFILL_READAHEAD=1` turns it on).
- **Attention** (`attention._attn`, one program per query) is now the largest GPU kernel in a 2,048-token prefill: about 3.2 s of 12.5 s of CUDA time on the laptop. A tiled exact attention is the next step.
- **The 512-token laptop prefill** is bound by reading the experts: about 14 GB per prefill on a 15 GB-RAM machine.
- **The bench prompt** is one paragraph repeated, which routes to fewer distinct experts than varied text would.

## Persistent prefix cache (part 2 of 1.3)
`vahana run --prefix-cache` and `vahana.generate(..., prefix_cache=prefix.PrefixCache(dir, model_key))`.
- A prompt that starts like an earlier one reuses the keys and values of that shared prefix from disk, at a 256-token block boundary. Only the rest is prefilled.
- It is exact because every token's row is computed the same way however the prompt is split.

Checks:
- Tests on five families compare cold, cached and reloaded-from-disk runs. All give the full prefill's logits and tokens, bit for bit, on transformers 5.12 and 5.20 ([tests/test_prefix.py](../../tests/test_prefix.py)).
- Transformers 5.20's static cache layers track their fill position in `cumulative_length`. Loading must set it, or the suffix overwrites the prefix.

Laptop, Qwen3-30B w4a16: a 2,048-token document followed by three different questions ([bench_prefix.py](bench_prefix.py)).

| | Prefill (time to first token) | Logits vs no cache |
|---|---|---|
| Question 1 (cold: stores 2,048 tokens, about 200 MB) | 12.5 s | identical |
| Question 2 (reuses 2,048 of 2,055) | **1.70 s** (10.3 s without the cache) | identical |
| Question 3 (reuses 2,048 of 2,055) | **1.49 s** (12.2 s without) | identical |

Limits:
- Batch 1 only.
- Attention-only models. Recurrent / linear-attention hybrids such as Qwen3.5 are skipped.
- Not yet in `vahana serve`.
