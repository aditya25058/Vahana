# v1.1 overlap: GPU-cached and CPU experts run at the same time

## Change
Before, a GPU-cached MoE layer ran its GPU-resident experts first and then handed the whole layer to the CPU job, which computed the remaining experts and combined all of them. The GPU waited throughout.

Now (graph decode, mailbox on):
1. The CPU job for the non-resident experts is posted first (`_mb_post`, which does not wait).
2. The GPU computes its cached experts meanwhile.
3. `_mb_wait` collects the CPU outputs [N, K, H].
4. All K experts are combined on the GPU with the CPU kernel's combine operations (`cache.combine`).

New kernel job: `make_host_cpu_slots`, an optional symbol, so ABI 3 is unchanged. `VAHANA_OVERLAP=0` restores the sequential path.

## Pre-set rule
Keep the change only if median tok/s with overlap ≥ 1.15× the median without it, and the hashes of every generated id and logit are identical.

## Results
**Exactness:** every run, on and off, produced the same hashes on both machines. OLMoE also hashes identically on the laptop and the H100.

**Qwen3-30B-A3B, H100, one GPU, 16 GB expert cache (66% of routing hits the cache), 3 prompts × 32 tokens, alternating runs.** Load is other users' CPU load during the run:

| run | overlap | load | tok/s |
|---|---|---|---|
| 1 | on | 14 | 12.18 |
| 2 | off | 57 | 8.06 |
| 3 | off | 63 | 8.60 |
| 4 | on | 61 | 19.45 |
| 5 | on | 39 | 16.64 |
| 6 | off | 27 | 17.80 |
| 7 | off | 26 | 18.83 |
| 8 | on | 30 | 21.31 |

- Median on/off = 18.0 / 13.2 = **1.37×, which passes the rule**.
- The box was shared and its load swung from 13 to 63. At similar load (runs 6–8) the gain is about 1.16×. With overlap the speed degrades less under CPU contention.

**Other measurements (no gain):**
- **OLMoE, H100, 4 GB cache:** about 45 tok/s either way.
- **OLMoE, laptop (RTX 3050 Ti):** only 0.68 GB was free for the cache (16% hit), so there is little to overlap; about 8 tok/s either way.
