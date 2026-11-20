# P3: one kernel library for every OS, wheels, and a selftest any machine can run

Pre-registered 2026-11-21, after the code below compiled on the H100. Two checks ran before this README existed and are reported as development checks, not bars:
- **The kernel-hash check:** the new library gives P1's hashes on all 8 kernel paths × 3 builds.
- **The CPU-only cross check:** 19/19 families × 3 builds reproduce P2's H100 GPU logits.

**Code:**
- **The kernel has no libtorch, OpenMP or pybind.**
  - **csrc/vt.h** is the small part of the `torch::Tensor` API the kernel uses (descriptors of torch-owned memory; owned buffers for results).
  - **csrc/pool.h** is Vahana's own thread team. It replaces OpenMP:
    - spin, then sleep;
    - `VAHANA_THREADS` or `OMP_NUM_THREADS`;
    - optional core binding, like `OMP_PLACES=cores`;
    - a sense-reversing barrier.
  - **A C ABI** (`vh_*`) replaces the pybind module.
  - **NUMA placement** calls the `move_pages` syscall directly, so no libnuma is needed.
  - **The CUDA host-function address** comes from Python (the runtime torch loaded), not dlopen.
- **csrc/build.py:** one library per ISA, built with g++ / clang++ / MSVC. Contraction is off everywhere.
- **vahana/_kernel.py:** a ctypes wrapper with the old extension's functions, so the rest of Vahana is unchanged. ctypes releases the GIL.
- **vahana/osl.py:**
  - available memory (Linux, Windows, macOS);
  - CPU affinity where it exists;
  - CPU flags.

  The CLI detects CUDA on Linux and Windows.
- **vahana/selftest.py (`vahana selftest`):**
  - 18 families, tiny models with numpy-seeded weights (the same on every machine);
  - 2 left-padded prompts, 6 greedy steps;
  - a hash of the logits and ids, compared with golden hashes recorded on the H100.
- **Packaging:** one wheel per platform, `py3-none-<platform>`, holding every ISA build. It works with any Python ≥ 3.10 and any torch ≥ 2.4. Triton becomes optional, the `gpu` extra.
- **CI (GitHub Actions):** wheels for Linux x86-64, Linux arm64, Windows x86-64 and macOS arm64. Each wheel is installed into a clean environment with CPU torch, and then:
  - the tests run;
  - `vahana selftest` runs under every ISA build the runner's CPU supports.

**Bars:**
- **GOLDEN (H100):** `vahana selftest` hashes on the GPU (graph decode) == CPU-only (avx512, avx2, scalar). The GPU hashes become `golden_tiny.json`.
- **TWIN-GPU (H100, p2_families.py):** with the new library, CPU-only == GPU graph for 19/19. GRAPH and BATCH are as in P2 (the same three graph fallbacks; Qwen3.5 batch reported).
- **SPEED:**
  - **GPU path:** Mixtral-8x7B at a 4 GB budget, graph decode. The steady step is within 3% of P2 (4f3d3ba), A/B alternated ×2.
  - **CPU-only:** OLMoE-1B-7B via `vahana run --device cpu`. tok/s within 5% of P2, A/B ×2.
- **CI:** on each of the 4 platforms:
  - the wheel builds and installs;
  - the CPU tests pass;
  - `vahana selftest` gives 18/18 hashes equal to the H100 golden, under each ISA build the runner can execute.
- **LAPTOP (native Windows, this laptop):** the CI-built Windows wheel, installed into native Windows Python (torch CPU, no WSL), gives 18/18 golden hashes.

**Prediction:**
- GOLDEN, TWIN-GPU and the GPU SPEED bar pass.
- CPU-only speed may move either way (a different thread team).
- CI needs iterations on Windows and macOS: compiler flags, MSVC/clang differences. Bits must match once it compiles. If they do not, the floating-point rules of that compiler are the suspect.

## Run log
- **CPU-only speed, first attempts:** 1.65 tok/s, against 17.9 for P2. Two causes, both fixed before the counted run:
  - **Pinning:** torch's libgomp (`OMP_PROC_BIND=true`) pins the main thread to core 0 when torch is imported. Pool threads created later inherited that one-core mask.
    - The pool now takes its CPU list from `VAHANA_CPUS`, which the CLI records before torch loads, or else from the online CPUs.
    - Every worker sets its own mask.
    - Only the kernel's job thread is ever pinned.
  - **Wake-up:** sleeping workers woke through one mutex. They now use C++20 atomic wait/notify, so the build is C++20.
- **CPU-only speed, counted run:** −5.3% (bar not met). Fix: the pool's atomics shared one cache line, so they were moved onto separate lines. Rerun: −3.7%.
- **CLI default for CPU-only:** torch's OpenMP team runs 1 thread (it only runs small glue ops), and the kernel's team spins for 1 ms.
- **CI run 1:**
  - Linux: `auditwheel` rejected libraries under `.data/purelib`. The wheel is now a binary distribution, so they sit in platlib.
  - macOS: the `macos-14` image is retired. Now `macos-15`.
  - Windows: passed.
- **CI run 2:** Linux x86-64 also tried a 32-bit build, so the wheels are now 64-bit only. Linux arm64, macOS and Windows passed.
- **CI run 3:** all pass.

## Result: GOLDEN, TWIN-GPU, GPU SPEED, CI pass; CPU-only SPEED not met in the counted run (−5.3%), met after a fix (−3.7%); LAPTOP pending
| Bar | Result |
|---|---|
| GOLDEN: selftest GPU == CPU-only avx512 / avx2 / scalar | **18/18 each** |
| TWIN-GPU: CPU-only == GPU graph (p2_families) | **19/19**. Graph and batch as in P2 (Qwen3.5 batch reported) |
| SPEED, GPU: Mixtral 4 GB steady step | **78.5 / 78.8 ms vs P2 82.5 / 81.0 (3.9% faster)**; tok/s 12.09 / 12.65 vs 11.96 / 11.98 |
| SPEED, CPU-only: OLMoE tok/s | Counted: 17.31 / 17.26 vs 18.18 / 18.29 (**−5.3%, not met**). After the false-sharing fix: 17.18 / 17.48 vs 18.20 / 17.80 (−3.7%). Prefill (6 tokens) 0.17 s vs 0.20 s |
| CI: wheel builds, installs with CPU torch; selftest == H100 golden under every ISA the runner can execute | **All 4 platforms, 9 ISA builds, 18/18 each** (ci_runs.txt) |

**CI detail (run 3):**

| Platform | ISA builds checked | Selftest |
|---|---|---|
| Linux x86-64 (manylinux_2_28 wheel) | avx512, avx2, scalar | 18/18 each |
| Linux arm64 | neon, scalar | 18/18 each |
| macOS arm64 | neon, scalar | 18/18 each |
| Windows x86-64 (native, clang) | avx2, scalar | 18/18 each |

**Development checks (before this README):**
- Kernel hashes == P1 on all 8 paths × 3 builds, through the C ABI and the new thread team. Rechecked after every pool change.
- CPU-only on the H100 reproduces P2's GPU logits, 19/19 × 3 builds.
- The H100 test suite: 122 passed, 3 skipped (GPU-only).

**Reading:**
- **Same bits everywhere tested.** The same model and prompt give the same logits and tokens, bit for bit, on:
  - an H100 GPU;
  - an EPYC (AVX-512, AVX2, plain C);
  - a Zen 2 laptop under WSL;
  - Linux on ARM;
  - an Apple-silicon Mac;
  - native Windows.
- **Compilers:** GCC, Clang and Apple Clang give the same bits, with contraction off. MSVC is supported by build.py but was not exercised (CI used clang on Windows).
- **NEON:** the ARM code ran for the first time here and matches.
- **One wheel per platform** serves every Python ≥ 3.10 and every torch ≥ 2.4: the kernel links neither.
- **Remaining speed gap:** CPU-only is 3.7% behind the OpenMP build.

**Not done:**
- **LAPTOP** (this laptop, native Windows) waits on downloading the CI wheel. GitHub's Windows runner already passed the same check.
- **No PyPI upload:** publishing is the owner's call. Wheels are CI artifacts.
