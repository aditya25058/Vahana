# P4: `vahana doctor`, and models bigger than RAM, streamed from disk, still exact

Pre-registered 2026-11-21, after the H100 reference runs and before any laptop run.

**Code:**
- **`vahana doctor [MODEL_DIR ...]`** reports this machine:
  - CPU and kernel build;
  - RAM, plus a read-bandwidth probe;
  - GPU, plus whether Triton is present;
  - per model, the disk read speed.

  For each model it reads only `config.json` and the safetensors headers (no weights): non-expert bytes, expert bytes, and the expert bytes an average token reads. Then it gives a plan:
  - non-expert weights on the GPU if they fit, else in RAM;
  - experts resident in RAM or streamed from disk;
  - a rough decode-speed estimate.
- **The streaming tier is the loader's existing in-place path.** Expert weights are views of the memory-mapped safetensors shards; NX12 copies are made only when RAM allows. So when experts exceed RAM, the OS page cache keeps the hot experts in memory and reads the rest from disk on demand. The bytes are the same, so the bits are the same. Only speed depends on RAM.

**Reference (H100, GPU, graph decode; p4_run.py → ref_*_h100.json):**
- **OLMoE-1B-7B:** prefill-logit hashes for 50 prompts, 4 greedy generations of 24 tokens (ids + a hash of every step's logits).
- **Qwen3-30B-A3B:** prefill-logit hashes for 8 prompts, 2 generations of 16 tokens.

**Bars (this laptop: Ryzen 7 4800H, 15.4 GB RAM, RTX 3050 Ti 4 GB, NVMe; native Windows 11):**
- **LAPTOP-OLMOE:** OLMoE-1B-7B (12.9 GB of weights in 15.4 GB of RAM) loads and runs. Every hash equals the H100's: 50 prefills, 4 generations (ids + logits). Speed is reported.
- **LAPTOP-QWEN3:** Qwen3-30B-A3B (57 GB of weights, about 4× the RAM) runs with experts streamed from the NVMe. Every hash equals the H100's: 8 prefills, 2 generations. Decode speed is reported.
- **DOCTOR:**
  - for both models, doctor's plan (device; resident vs streamed) matches what the run did;
  - its speed estimate is within 3× of the measured decode speed.
- **Devices:** the laptop runs use the GPU (CUDA torch + triton-windows) if that works on Windows; otherwise CPU-only, which is reported. Either device must give the H100's bits.

**Prediction:**
- LAPTOP-OLMOE passes.
- LAPTOP-QWEN3 passes on bits at about 0.5–3 tok/s.
- The estimate is the weakest part (page-cache behaviour under Windows is not modelled).

## Run log (before the counted runs)
- **The laptop GPU path on native Windows** (torch 2.11 cu128, triton-windows 3.8) was checked with `vahana selftest --device cuda`. Result: 17/18. Phi-MoE failed because the cleanup after a failed graph capture called a private torch function that torch 2.11 no longer has.
  - Calling its renamed version aborts the process, because newer torch already cleans up.
  - Fix: the old cleanup runs only on torch that still has the old function.
  - Then 18/18 (laptop_gpu_selftest.txt, laptop_gpu_phimoe.txt).
- **First OLMoE load on the laptop:** WinError 1455, "the paging file is too small".
  - Cause: a copy-on-write mapping on Windows is charged in full against RAM + pagefile, so no model bigger than that could even be mapped.
  - Fix: on Windows, checkpoints are mapped read-only. Vahana never writes weight memory. Linux keeps ACCESS_COPY.

## Result: LAPTOP-OLMOE and LAPTOP-QWEN3 pass; DOCTOR not met
| Bar | Result |
|---|---|
| LAPTOP-OLMOE: 50 prefills + 4 generations == H100 | **50/50, 4/4 (ids and logits)**. GPU graph decode 4.1–4.4 tok/s (the first, cold generation 1.95); 4.2 GB RAM available at the start |
| LAPTOP-QWEN3: 8 prefills + 2 generations == H100 | **8/8, 2/2 (ids and logits)**. 57 GB of weights on a 15.4 GB machine: experts streamed from NVMe, non-expert weights 1.7 GB on the GPU and the rest in RAM (3.2 GB budget). Decode 0.54 / 0.59 tok/s (1.4–1.5 s per token), prefill 20–25 s per prompt |
| DOCTOR: plan matches the run; estimate within 3× | **Not met.** OLMoE: plan matched (GPU, experts streamed), but the estimate was 1.1 vs 4.2 tok/s measured (3.8×, too pessimistic). Qwen3: estimate 0.4 vs 0.56 (within 3×), but the plan said "device=cpu", while the run split non-expert weights between GPU and RAM (doctor could not describe a partial GPU placement) |

**After the counted runs:** doctor now describes split placements ("non-expert in GPU 1.7 GB + CPU RAM 1.2 GB") and charges only the CPU part per token. Its estimate for the Qwen3 run is now 0.5 tok/s. The OLMoE miss remains: doctor assumes uniform expert popularity, while real routing is skewed and the page cache holds the hot experts.

**Reading:**
- **A 57 GB model runs exactly on a 15 GB laptop.** It gives the same tokens and logits, bit for bit, as an H100 that holds everything in memory.
- **Different hardware gave the same bits:** GPU (RTX 3050 Ti vs H100), OS (Windows vs Linux), CPU (Zen 2 vs Zen 4), torch (2.11 vs 2.4), memory regime (streamed from disk vs resident).
- **Speed follows the disk:** about 0.55 tok/s for Qwen3-30B and about 4 tok/s for OLMoE on this laptop.
- **Next for speed** (v0.3): keep hot experts in spare VRAM, prefetch the next layer's experts, and exact speculative decoding.
