"""v0.4-b real checkpoints. mode exact: 32 M5 prompts with staggered arrivals into 8 slots, 32 tokens; the first 12
vs solo decode.generate (ids and every logit). mode speed: 128 requests (64 tokens) at once into a 32-slot engine vs
static generate_batch at B = 32 (4 batches) and one-at-a-time; decode tok/s (step time only) and end-to-end wall.
usage: v4b_real.py MODEL_DIR PROMPTS.json OUT.json gpu_cache_gb exact|speed"""
import json
import sys
import time

import torch
from transformers import AutoTokenizer

from vahana import batch, decode, engine, loader

MD, PJ, OUT, CACHE, MODE = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
tok = AutoTokenizer.from_pretrained(MD)
P = [tok(t).input_ids for t in json.load(open(PJ))]
model, info = loader.load(MD, device="cuda", gpu_cache_gb=CACHE, log=print)
res = {"model": MD.rstrip("/").rsplit("/", 1)[-1], "gpu_cache": info.get("gpu_cache"), "mode": MODE}
if MODE == "exact":
    eng = engine.Engine(model, slots=8, max_len=2048)
    hs = []
    for a, b, k in ((0, 8, 10), (8, 16, 10), (16, 32, 0)):
        hs += [eng.submit(P[i], 32, record_logits=True) for i in range(a, b)]
        for _ in range(k):
            eng.step()
    eng.run()
    ok = []
    for i in range(12):
        s = decode.generate(model, torch.tensor([P[i]]).cuda(), None, max_new_tokens=32, return_logits=True)
        ok.append(hs[i].ids == s["ids"][0].tolist() and torch.equal(
            torch.stack(hs[i].logits).view(torch.int32), s["logits"][0].float().cpu().contiguous().view(torch.int32)))
    res.update(exact=sum(ok), of=12, graph=eng.graph_info(), stats={k: v for k, v in eng.stats.items() if k != "step_ms"})
    print("EXACT", sum(ok), "/ 12", res["graph"], res["stats"], flush=True)
else:
    N, NEW, S = 128, 64, 32
    reqs = [P[i % len(P)] for i in range(N)]
    for rnd in (1, 2):
        eng = engine.Engine(model, slots=S, max_len=2048)
        t0 = time.perf_counter()
        for p in reqs:
            eng.submit(p, NEW)
        eng.run()
        wall = time.perf_counter() - t0
        dec = eng.stats["active_tokens"] / (sum(eng.stats["step_ms"]) / 1e3)
        res.setdefault("engine", []).append({"decode_tok_s": round(dec, 1), "wall_s": round(wall, 2),
                                             "e2e_tok_s": round(N * NEW / wall, 1),
                                             "steps": eng.stats["steps"], "prefills": eng.stats["prefills"],
                                             "bucket_eff": round(eng.stats["active_tokens"] / eng.stats["bucket_tokens"], 3)})
        eng.close()
        t0 = time.perf_counter()
        steps = []
        for k in range(0, N, S):
            r = batch.generate_batch(model, reqs[k:k + S], NEW)
            steps += r["step_ms"]
        wall = time.perf_counter() - t0
        res.setdefault("static", []).append({"decode_tok_s": round(S * len(steps) / (sum(steps) / 1e3), 1),
                                             "wall_s": round(wall, 2), "e2e_tok_s": round(N * NEW / wall, 1)})
        r = decode.generate(model, torch.tensor([reqs[0]]).cuda(), None, max_new_tokens=NEW)
        res.setdefault("solo_decode_tok_s", []).append(round(len(r["step_ms"]) / (sum(r["step_ms"]) / 1e3), 1))
        print("round", rnd, "engine", res["engine"][-1], "static", res["static"][-1], "solo", res["solo_decode_tok_s"][-1], flush=True)
    m = lambda k, f: sum(x[f] for x in res[k]) / len(res[k])
    res["vs_static"] = round(m("engine", "decode_tok_s") / m("static", "decode_tok_s"), 3)
    res["vs_static_e2e"] = round(m("engine", "e2e_tok_s") / m("static", "e2e_tok_s"), 3)
    res["vs_solo"] = round(m("engine", "decode_tok_s") / (sum(res["solo_decode_tok_s"]) / 2), 2)
    print("SPEED vs static", res["vs_static"], "e2e", res["vs_static_e2e"], "vs solo", res["vs_solo"], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
