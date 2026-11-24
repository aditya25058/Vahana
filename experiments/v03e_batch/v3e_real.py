"""v0.3-e real checkpoints: batched decoding is exact (every row == its solo run) and its aggregate throughput.
EXACT: the first 8 M5 prompts as one batch, 32 new tokens, vs each alone (ids and every step's logits).
SPEED: aggregate decode tok/s = B * (tokens - 1) / decode time, B = 1..64, 64 new tokens, 2 rounds.
usage: v3e_real.py MODEL_DIR PROMPTS.json OUT.json gpu_cache_gb [B list, e.g. 1,2,4,8,16,32,64]"""
import json
import sys

import torch
from transformers import AutoTokenizer

from vahana import batch, decode, loader

MD, PJ, OUT, CACHE = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
BS = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "1,2,4,8,16,32,64").split(",")]
tok = AutoTokenizer.from_pretrained(MD)
texts = json.load(open(PJ))
P = [tok(t).input_ids for t in texts]
model, info = loader.load(MD, device="cuda", gpu_cache_gb=CACHE, log=print)
res = {"model": MD.rstrip("/").rsplit("/", 1)[-1], "gpu_cache": info.get("gpu_cache"), "prompt_lens": [len(p) for p in P[:8]]}
bat = batch.generate_batch(model, P[:8], 32, return_logits=True)
rows = []
for b in range(8):
    s = decode.generate(model, torch.tensor([P[b]]).cuda(), None, max_new_tokens=32, return_logits=True)
    o = bat["outputs"][b]
    rows.append(o["ids"] == s["ids"][0].tolist() and torch.equal(o["logits"].view(torch.int32), s["logits"][0].view(torch.int32)))
res["exact_rows"] = sum(rows)
res["exact_graph"] = bat["graph_used"]
print("EXACT rows", sum(rows), "/ 8, graph", bat["graph_used"], flush=True)
res["speed"] = {}
for rnd in (1, 2):
    for B in BS:
        ps = [P[i % len(P)] for i in range(B)]
        r = batch.generate_batch(model, ps, 64)
        tps = B * len(r["step_ms"]) / (sum(r["step_ms"]) / 1e3) if B > 1 else len(r["step_ms"]) / (sum(r["step_ms"]) / 1e3)
        res["speed"].setdefault(str(B), []).append(round(tps, 2))
        print(f"round {rnd} B={B}: {tps:.1f} tok/s aggregate, step {sum(r['step_ms'])/len(r['step_ms']):.2f} ms, graph {r['graph_used']}", flush=True)
base = sum(res["speed"]["1"]) / 2
res["scale"] = {B: round(sum(v) / 2 / base, 2) for B, v in res["speed"].items()}
print("SCALE vs B=1", res["scale"], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
