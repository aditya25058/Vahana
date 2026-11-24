"""v0.4-a real checkpoint (Qwen3-30B with a partial GPU cache: CPU experts in every batch step).
mode exact: a batch of 32 M5 prompts, 32 new tokens; the first 8 rows vs their solo runs (ids and every logit).
mode speed: aggregate decode tok/s at the given batch sizes, 64 new tokens each.
usage: v4a_real.py MODEL_DIR PROMPTS.json OUT.json gpu_cache_gb exact|speed [B list]"""
import json
import sys

import torch
from transformers import AutoTokenizer

from vahana import batch, decode, loader

MD, PJ, OUT, CACHE, MODE = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
BS = [int(x) for x in (sys.argv[6] if len(sys.argv) > 6 else "1,8,32,64").split(",")]
tok = AutoTokenizer.from_pretrained(MD)
P = [tok(t).input_ids for t in json.load(open(PJ))]
model, info = loader.load(MD, device="cuda", gpu_cache_gb=CACHE, log=print)
res = {"model": MD.rstrip("/").rsplit("/", 1)[-1], "gpu_cache": info.get("gpu_cache"), "mode": MODE}
if MODE == "exact":
    bat = batch.generate_batch(model, P[:32], 32, return_logits=True)
    rows = []
    for b in range(8):
        s = decode.generate(model, torch.tensor([P[b]]).cuda(), None, max_new_tokens=32, return_logits=True)
        o = bat["outputs"][b]
        rows.append(o["ids"] == s["ids"][0].tolist() and torch.equal(o["logits"].view(torch.int32), s["logits"][0].view(torch.int32)))
    res["exact_rows"], res["graph"] = sum(rows), bat["graph_used"]
    print("EXACT B=32 rows", sum(rows), "/ 8, graph", bat["graph_used"], flush=True)
else:
    res["speed"] = {}
    for B in BS:
        r = batch.generate_batch(model, [P[i % len(P)] for i in range(B)], 64)
        tps = B * len(r["step_ms"]) / (sum(r["step_ms"]) / 1e3)
        res["speed"][str(B)] = round(tps, 2)
        print(f"B={B}: {tps:.1f} tok/s, step {sum(r['step_ms'])/len(r['step_ms']):.1f} ms, graph {r['graph_used']}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
