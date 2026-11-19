"""M4 real: load a checkpoint through vahana.loader, decode with the graph (speed) and eagerly (exactness).
usage: m4_real.py MODEL_DIR GPU_BUDGET_GB PROMPT_FILE OUT.json [new_tokens] [exact_tokens] [reserve_gb]
env VAHANA_MAIN_CPUS=a,b moves the main thread off the worker's core after loading.
"""
import json
import os
import sys
import time

import torch

from vahana import decode, loader
from vahana import hybrid as hy

MD, BUDGET, PF, OUT = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
NEW = int(sys.argv[5]) if len(sys.argv) > 5 else 128
EX = int(sys.argv[6]) if len(sys.argv) > 6 else 32
RES = float(sys.argv[7]) if len(sys.argv) > 7 else 1.5
dev = torch.device("cuda")
t0 = time.time()
model, info = loader.load(MD, gpu_budget_gb=BUDGET, reserve_gb=RES)
print("loaded", {k: v for k, v in info.items() if k not in ("cpu_modules", "invariant")}, flush=True)
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(MD)
ids = tok(open(PF).read().rstrip("\n"), return_tensors="pt").input_ids.to(dev)
mc = os.environ.get("VAHANA_MAIN_CPUS")
if mc:
    os.sched_setaffinity(0, {int(c) for c in mc.split(",")})
res = {"model": MD, "budget_gb": BUDGET, "prompt_tokens": ids.shape[1], "load": {k: v for k, v in info.items()
                                                                                    if k != "invariant"},
       "mem_available_gb_after_load": hy.mem_available_gb()}
g = decode.generate(model, ids, max_new_tokens=NEW, graph=True, return_logits=True)
st = g["step_ms"]
res["graph"] = {"tok_s": len(st) / (sum(st) / 1e3), "steady_ms_median": sorted(st[1:])[len(st[1:]) // 2],
                "first_step_ms": st[0], "prefill_s": g["prefill_s"], "ids": g["ids"][0].tolist()}
print("graph", {k: v for k, v in res["graph"].items() if k != "ids"}, flush=True)
e = decode.generate(model, ids, max_new_tokens=EX, graph=False, return_logits=True)
res["exact"] = {"ids_equal": torch.equal(e["ids"][0, :EX], g["ids"][0, :EX]),
                "logits_equal": torch.equal(e["logits"][:, :EX], g["logits"][:, :EX]), "n": EX,
                "eager_steady_ms_median": sorted(e["step_ms"][1:])[len(e["step_ms"][1:]) // 2]}
print("exact", res["exact"], flush=True)
res["peak_vram_gb"] = torch.cuda.max_memory_allocated(dev) / 2**30
res["text"] = tok.decode(g["ids"][0])
res["wall_s"] = time.time() - t0
json.dump(res, open(OUT, "w"), indent=1)
print("DONE", res["peak_vram_gb"], flush=True)
