"""v0.4-a diagnosis: where a Qwen3-30B batched decode step goes (CPU expert kernel phases vs the rest).
usage: v4a_diag.py MODEL_DIR PROMPTS.json gpu_cache_gb"""
import json
import sys

from transformers import AutoTokenizer

from vahana import batch, hybrid, loader, runtime

MD, PJ, CACHE = sys.argv[1], sys.argv[2], float(sys.argv[3])
tok = AutoTokenizer.from_pretrained(MD)
P = [tok(t).input_ids for t in json.load(open(PJ))]
model, _ = loader.load(MD, device="cuda", gpu_cache_gb=CACHE, log=None)
k = hybrid.ext()
for rows in (64, 16):
    runtime.MOE_ROWS = rows
    for B in (8, 64):
        batch.generate_batch(model, [P[i % len(P)] for i in range(B)], 4)            # warm (graph capture)
        ph0 = k.phase_ms()
        r = batch.generate_batch(model, [P[i % len(P)] for i in range(B)], 17)
        ph1 = k.phase_ms()
        d = [a - b for a, b in zip(ph1, ph0)]
        steps = len(r["step_ms"]) + 1          # + prefill's expert calls are inside d too
        st = sum(r["step_ms"]) / len(r["step_ms"])
        print(f"rows={rows} B={B}: step {st:.1f} ms; expert kernel calls {d[4]:.0f} "
              f"setup {d[0]:.0f} pass0 {d[1]:.0f} pass1 {d[2]:.0f} tail {d[3]:.0f} ms total "
              f"-> per step ~{(d[0] + d[1] + d[2] + d[3]) / steps:.1f} ms in the expert kernel", flush=True)
