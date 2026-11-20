# P1: one CPU kernel for every CPU, identical bits

Pre-registered 2026-11-20, after the builds compiled and before any test.

**Code:**
- **csrc/cpu_moe.cpp:**
  - `V16` (16 fp32 lanes) is implemented four ways: AVX-512, AVX2+FMA (2 × 8), ARM NEON (4 × 4) and scalar (`std::fma`).
  - Every dot product runs the same per-lane fused multiply-adds and the same tree.
  - NX12 decodes with VBMI / VBMI2 when compiled in, with a portable decoder otherwise.
  - NUMA placement is optional: without `numaif.h` it is stubbed, and it never changes arithmetic.
  - `isa()`, `nx_fast()` and `have_numa()` report what was compiled.
- **build_kernel.sh:** one build per ISA (`avx512 | avx2 | scalar | neon`), with explicit flags rather than `-march=native`, so the AVX-512 build runs on any AVX-512+VBMI2 CPU. It links `-lnuma` only when the header exists.
- **hybrid.ext():** loads the best build this CPU supports (`VAHANA_ISA` forces one).
- **experts.py:** uses NX12 only where the fast decoder is compiled; portable builds use raw BF16.

**Scope change from the design (stated before testing):** the own thread pool (replacing OpenMP) moves to P3, where macOS and MSVC need it. P1 keeps OpenMP, which GCC and Clang provide on Linux and WSL.

**Bars:**
- **EXACT-ISA (H100, m2_families.py, 19 cases):** with `VAHANA_ISA` = avx512, avx2 and scalar in turn:
  - every run is 19/19 CPU == GPU twin, bit for bit (prefill logits, decode logits, ids);
  - the CPU logits are identical across the three ISAs.
- **HASH (p1_hash.py):** every kernel path's output hashes are identical across:
  - H100: avx512, avx2, scalar;
  - this laptop (AMD Ryzen 7 4800H, Zen 2, AVX2 only, WSL Ubuntu, torch 2.x CPU): avx2, scalar.

  Kernel paths covered: raw experts n=1 / n=3, twin linear n=1 / n=3, NX12 experts, MoE job N=5 with fp32 weights, fused N=1, gpt-oss activation with biases.
- **NO-CHANGE (H100, Phase 41 bench, real DeepSeek-V2 236B layers):** the avx512 build's outputs are bit-identical to the pre-P1 kernel.
- **SPEED:** the avx512 build is within ±3% of pre-P1 on the same bench (old / new alternated ×2).

**Prediction:** all pass. The avx2 and scalar builds are slower (raw BF16, no NX12): speed is reported, not barred.

## Run log (build fixes before any counted test)
- **The avx512 build failed to compile.** `_mm512_mask_expandloadu_epi8` is a VBMI2 instruction. Fixed by adding `-mavx512vbmi2` to the build and requiring the `avx512_vbmi2` flag in detection (Ice Lake and Zen 4 both have it).
- **WSL lacks `numaif.h`.** NUMA placement became optional in the kernel (stub) and in the build (`-lnuma` only when the header exists).
- **torch 2.15 headers require C++20.** The build uses C++20 when the compiler supports it and C++17 otherwise. After the switch, the H100 builds were rebuilt and re-hashed before any counted comparison.

## Result: EXACT-ISA, HASH, NO-CHANGE and SPEED all PASS
**HASH** (hash_*.json): identical hashes for every kernel path.

| Machine | CPU | OS | torch | Builds |
|---|---|---|---|---|
| H100 server | 2× EPYC 9354 (Zen 4, AVX-512) | Ubuntu | 2.4.0 | avx512, avx2, scalar |
| Laptop | Ryzen 7 4800H (Zen 2, AVX2 only) | Windows 11 → WSL Ubuntu | 2.15.0 (CPU) | avx2, scalar |

| Path | Hash (all 5 builds) |
|---|---|
| experts n=1 | 8279cdb07591a881 |
| experts n=3 | ce80cac9366f2030 |
| twin linear n=1 | 5180cff971937d2c |
| twin linear n=3 | 6693bb4439b6b424 |
| NX12 experts n=1 | 8279cdb07591a881 (= raw: lossless) |
| MoE job N=5, fp32 weights | d4da68130d10dcca |
| MoE job N=1 (fused path) | 40639f2ddb8df72e |
| gpt-oss activation + biases | 931d6ec1a532de1f |

**EXACT-ISA** (fam_*.json, 19 cases: 17 Tier 1 families + gpt-oss + a width-96 variant):
- avx512: 19/19 CPU == GPU twin.
- avx2: 19/19.
- scalar: 19/19.
- **The CPU logits and ids are identical across the three builds for 19/19.**

**NO-CHANGE / SPEED** (bench/, real DeepSeek-V2 236B layers, 24 threads):

| | pre-P1 | P1 avx512 build |
|---|---|---|
| MoE layer call (rep-2 kernel median) | 1.650 / 1.654 ms | 1.654 / 1.658 ms (**+0.2%**) |
| o_proj NX12 | 0.665 / 0.658 ms | 0.667 / 0.666 ms |
| Outputs | — | **bit-identical** |

**Reading:**
- **One kernel source now runs on any x86-64 CPU** (AVX-512, AVX2, or neither) and gives the same bits. The proof spans a server and a consumer laptop, two OSes and two torch versions.
- **The NEON path is written but not yet run on ARM hardware.** The P3 CI matrix (macOS arm64) is where it gets tested.
- **The AVX-512 build carries no cost** (+0.2%).
- **The portable builds use raw BF16 instead of NX12.** Their speed is not part of P1.
