"""v0.5 streaming bench: a model bigger than RAM, experts streamed from disk. Per prompt: greedy graph decode of NEW
tokens; decode tok/s, disk GB read per decoded token (system-wide counters), and a hash of the ids and every logit
(prefetching must change no bit).
usage: v5_stream.py MODEL_DIR PROMPTS.json OUT.json [gpu_cache_gb|auto|-] [new] [n_prompts]"""
import hashlib
import json
import sys
import time

import psutil
import torch
from transformers import AutoTokenizer

from vahana import decode, hybrid, loader

MD, PJ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
CA = sys.argv[4] if len(sys.argv) > 4 else "auto"
CACHE = None if CA == "-" else (CA if CA == "auto" else float(CA))
NEW = int(sys.argv[5]) if len(sys.argv) > 5 else 32
NP = int(sys.argv[6]) if len(sys.argv) > 6 else 4
tok = AutoTokenizer.from_pretrained(MD)
prompts = json.load(open(PJ))[:NP]
model, info = loader.load(MD, device="cuda", gpu_cache_gb=CACHE, log=print)
k = hybrid.ext()
res = {"model": MD.rstrip("/\\").rsplit("/", 1)[-1].rsplit("\\", 1)[-1], "gpu_cache": info.get("gpu_cache"),
       "ram_gb": round(psutil.virtual_memory().total / 2**30, 1), "prompts": []}
for p in prompts:
    ids = tok(p, return_tensors="pt").input_ids.cuda()
    d0 = psutil.disk_io_counters().read_bytes
    t0 = time.perf_counter()
    r = decode.generate(model, ids, None, max_new_tokens=NEW, return_logits=True)
    wall = time.perf_counter() - t0
    d1 = psutil.disk_io_counters().read_bytes
    st = r["step_ms"]
    h = hashlib.sha256(r["ids"].cpu().numpy().tobytes() + r["logits"].float().cpu().numpy().tobytes()).hexdigest()[:16]
    row = {"tok_s": round(len(st) / (sum(st) / 1e3), 3), "disk_gb_per_tok": round((d1 - d0) / 2**30 / (len(st) + 1), 3),
           "wall_s": round(wall, 1), "prefill_s": round(r["prefill_s"], 2), "hash": h, "graph": r["graph_used"]}
    pf = getattr(k, "prefetch_stats", None)
    if pf:
        row["prefetch"] = pf(reset=True)
    res["prompts"].append(row)
    print(row, flush=True)
res["mean_tok_s"] = round(sum(r["tok_s"] for r in res["prompts"]) / len(res["prompts"]), 3)
res["mean_disk_gb_per_tok"] = round(sum(r["disk_gb_per_tok"] for r in res["prompts"]) / len(res["prompts"]), 3)
print("SUMMARY", {k: v for k, v in res.items() if k != "prompts"}, flush=True)
json.dump(res, open(OUT, "w"), indent=1)
