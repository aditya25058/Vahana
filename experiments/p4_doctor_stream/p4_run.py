"""P4: run a real checkpoint through vahana.load and record hashes, so two machines can be compared bit for bit.

Per prompt (batch 1): the prefill logits at every position (hash), and for the first NGEN prompts a greedy generation
of NEW tokens (ids, and a hash of every step's logits). Also timings and page-cache facts.
usage: p4_run.py MODEL_DIR PROMPTS.json OUT.json [n_prefill] [n_gen] [new_tokens] [device] [gpu_budget_gb]
"""
import hashlib
import json
import os
import sys
import time

import torch
from transformers import AutoTokenizer

from vahana import decode, loader, osl

MD, PJ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NPRE = int(sys.argv[4]) if len(sys.argv) > 4 else 50
NGEN = int(sys.argv[5]) if len(sys.argv) > 5 else 4
NEW = int(sys.argv[6]) if len(sys.argv) > 6 else 24
DEV = sys.argv[7] if len(sys.argv) > 7 else ("cuda" if torch.cuda.is_available() else "cpu")
BUDGET = float(sys.argv[8]) if len(sys.argv) > 8 else None


def h(t):
    return hashlib.sha256(t.float().cpu().contiguous().numpy().tobytes()).hexdigest()[:20]


tok = AutoTokenizer.from_pretrained(MD)
prompts = json.load(open(PJ))
res = {"model": os.path.basename(os.path.normpath(MD)), "device": DEV, "torch": torch.__version__,
       "platform": sys.platform, "ram_available_gb_before": osl.mem_available_gb()}
t0 = time.time()
model, info = loader.load(MD, device=DEV, gpu_budget_gb=BUDGET, log=None)
res["load_s"] = time.time() - t0
res["load"] = {k: info.get(k) for k in ("nonexpert_gb", "gpu_gb", "nx_layers", "cpu_modules", "moe_layers")}
res["load"]["cpu_modules"] = len(info.get("cpu_modules") or [])
res["kernel"] = __import__("vahana.hybrid", fromlist=["ext"]).ext().isa()
d = torch.device(DEV)
gens = []
for i in range(NGEN):
    ids = tok(prompts[i], return_tensors="pt").input_ids.to(d)
    t = time.time()
    r = decode.generate(model, ids, max_new_tokens=NEW, graph=DEV == "cuda", return_logits=True)
    st = r["step_ms"]
    gens.append({"ids": r["ids"][0].tolist(), "logits": h(r["logits"]), "wall_s": time.time() - t,
                 "prefill_s": r["prefill_s"], "step_ms_median": sorted(st)[len(st) // 2] if st else None,
                 "tok_s": len(st) / (sum(st) / 1e3) if st else None, "graph": r["graph_used"],
                 "text": tok.decode(r["ids"][0])})
    print("gen", i, gens[-1]["logits"], f"{gens[-1]['tok_s'] or 0:.2f} tok/s", repr(gens[-1]["text"][:60]), flush=True)
res["gens"] = gens
pre = []
t = time.time()
with torch.no_grad():
    for i in range(NPRE):
        ids = tok(prompts[i], return_tensors="pt").input_ids.to(d)
        pre.append(h(model(ids).logits[0]))
res["prefill"] = pre
res["prefill_s"] = time.time() - t
res["ram_available_gb_after"] = osl.mem_available_gb()
json.dump(res, open(OUT, "w"), indent=1)
print("DONE", res["model"], DEV, f"load {res['load_s']:.0f}s", f"prefill {res['prefill_s']:.0f}s", flush=True)
