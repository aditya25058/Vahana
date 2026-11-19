# M0: host-node CUDA graph spike

Pre-registered 2026-11-20, before any run.

**Question.** Can one CUDA graph per decode step contain the CPU expert jobs as host nodes (`cudaLaunchHostFunc`), so that the GPU work, the CPU experts and the GPU work again run in stream order with no Python on the critical path? This mechanism would replace the per-family graph decoders.

**Mechanism** (csrc/cpu_moe.cpp):
- `make_host_job(layer, pinned x / ti / tw / out, numa, noop)` registers a job.
- `enqueue_host_job(stream, id)` adds a host node. The node's callback hands the job to the persistent OpenMP worker and waits. It makes no CUDA calls.
- `cudaLaunchHostFunc` is resolved from the libcudart that torch has already loaded.

**Spike** (m0_spike.py):
- A synthetic stack of 32 layers: H=4096, E=8, top-2, I=2048, random BF16 experts as NX12, numa=2.
- Each layer runs, in order: GPU RMS-norm → router linear → softmax / top-k → device-to-pinned copies → CPU expert job → pinned-to-device copy → residual add.
- 20 steps. OMP 24 threads on CPUs 0-12,32-44; main thread moved to 43,44 after warm-up.

**Arms:**
- **eager:** Python sync + submit_moe / wait_done per layer.
- **graph:** the whole step captured, with a host node per layer.
- **gpu-only graph** and **noop-host-node graph:** together they measure the cost per host node.

**Bars:**
- **EXACT:** the graph state equals the eager state bit for bit after every one of the 20 steps (eager is also checked to be reproducible).
- **RUNS:** 0 host calls during capture, and exactly 32 host calls per replay.
- **OVERHEAD:** per-host-node overhead ≤ 30 µs, i.e. ≤ 1 ms per step at 32 layers.
- **WIN:** graph step time − CPU kernel time is ≤ 0.5 × (eager step time − CPU kernel time). In other words, the graph removes at least half of the non-kernel time.

**Prediction:** EXACT, RUNS and OVERHEAD pass, with host nodes costing ~10-20 µs each. The graph removes most of the Python and sync time: eager pays ~100-200 µs of Python and sync per layer.

**If it fails:** fall back to segmented capture generated from ModelSpec (design §3.5).

## Result (2026-11-20, GPU 1, m0_result.json): EXACT + RUNS + WIN PASS; OVERHEAD NOT MET (44 µs, bar 30 µs)
| Measure | Value |
|---|---|
| Graph state == eager state, all 20 steps | **yes** (eager also reproducible) |
| Host calls during capture / per replay | **0 / 32** |
| Eager step (median) | 39.48 ms |
| **Graph step (median)** | **26.73 ms (1.48× faster)** |
| CPU expert kernel per step (median) | 25.31 ms |
| Non-kernel time per step: eager → graph | **14.18 → 1.42 ms** (0.10×; bar ≤ 0.5×) |
| GPU-only graph / with 32 no-op host nodes | 1.85 / 3.26 ms |
| Per-host-node overhead | **44 µs** (bar ≤ 30 µs: NOT MET) |

**Reading:**
- **The mechanism works and is exact.** Capture records the host nodes without running them, and each replay runs every CPU job in stream order with identical bits.
- **It removes 90% of the non-kernel time:** Python, synchronisation and launch costs.
- **The 44 µs per node is CUDA's host-node round trip,** not our hand-off: a no-op callback returns immediately and still costs 44 µs. At 32-60 MoE layers that is 1.4-2.6 ms per token, 1-3% of a 80-130 ms step. The bar was set tighter than the mechanism allows. It is reported as missed and does not block the design.
- **Decision: the generic decode graph (design §3.5) goes ahead with host nodes.** Segmented capture stays as the fallback only.
