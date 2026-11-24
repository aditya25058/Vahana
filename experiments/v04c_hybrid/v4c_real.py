"""v0.4-c real checkpoint (Qwen3.5-35B-A3B: Gated DeltaNet + full attention, 256 experts): 16 M5 prompts through
HybridEngine in two waves (8 slots), 32 tokens; the first 8 vs solo decode.generate (ids and every logit).
Throughput: engine decode tok/s (active rows / step time) vs one at a time (the solo runs' decode tok/s).
usage: v4c_real.py MODEL_DIR PROMPTS.json OUT.json gpu_cache_gb"""
import json
import sys

import torch
from transformers import AutoTokenizer

from vahana import decode, engine, loader

MD, PJ, OUT, CACHE = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
tok = AutoTokenizer.from_pretrained(MD)
P = [tok(t).input_ids for t in json.load(open(PJ))[:16]]
model, info = loader.load(MD, device="cuda", gpu_cache_gb=CACHE, log=print)
res = {"model": MD.rstrip("/").rsplit("/", 1)[-1], "gpu_cache": info.get("gpu_cache"), "load_s": info["load_s"]}
eng = engine.make(model, slots=8, max_len=4096)
res["engine"] = type(eng).__name__
hs = [eng.submit(P[i], 32, record_logits=True) for i in range(8)]
for _ in range(6):
    eng.step()
hs += [eng.submit(P[i], 32, record_logits=True) for i in range(8, 16)]
eng.run()
dec = eng.stats["active_tokens"] / (sum(eng.stats["step_ms"]) / 1e3)
ok, solo = [], []
for i in range(8):
    s = decode.generate(model, torch.tensor([P[i]]).cuda(), None, max_new_tokens=32, return_logits=True)
    solo.append(len(s["step_ms"]) / (sum(s["step_ms"]) / 1e3))
    ok.append(hs[i].ids == s["ids"][0].tolist() and torch.equal(
        torch.stack(hs[i].logits).view(torch.int32), s["logits"][0].float().cpu().contiguous().view(torch.int32)))
res.update(exact=sum(ok), of=8, engine_decode_tok_s=round(dec, 2), solo_decode_tok_s=round(sum(solo) / len(solo), 2),
           speedup=round(dec / (sum(solo) / len(solo)), 2), stats={k: v for k, v in eng.stats.items() if k != "step_ms"},
           sample=tok.decode(hs[0].ids)[:120])
print("REAL", {k: v for k, v in res.items() if k not in ("gpu_cache",)}, flush=True)
json.dump(res, open(OUT, "w"), indent=1)
