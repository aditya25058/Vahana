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
