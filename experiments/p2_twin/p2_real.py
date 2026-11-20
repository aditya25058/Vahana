"""P2 REAL + QUALITY: a real checkpoint, CPU-only Vahana vs GPU Vahana (both twin arithmetic), and quality vs fp32.

Arms (sequential; stock first, because vahana patches nn.Linear process-wide):
  fp32   stock transformers fp32 on the GPU: prefill logits on all prompts (batch 1)
  gpu    vahana.load (non-expert weights on the GPU, experts on the CPU kernel): greedy 24 tokens on the first 4
         prompts with the decode graph; prefill logits on all prompts
  cpu    vahana.load(device="cpu"): no CUDA in the forward; the same generations (eager) and prefill logits
Bars: REAL     cpu == gpu: generated ids, every step's logits, every prompt's prefill logits, bit for bit
      QUALITY  gpu top-1 agreement with fp32 over every prompt position within 1 point of v0.1 (M5)
usage: p2_real.py MODEL_DIR PROMPTS.json OUT.json
"""
import gc
import json
import sys
import time

import torch

from vahana import decode, loader

MD, PJ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NEW, NGEN = 24, 4
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MD)
prompts = json.load(open(PJ))
enc = [tok(p, return_tensors="pt").input_ids for p in prompts]
res = {"model": MD.rsplit("/", 1)[-1], "prompts": len(prompts), "positions": int(sum(e.shape[1] for e in enc))}


@torch.no_grad()
def prefill_logits(m, d):
    return [m(e.to(d)).logits.float()[0].cpu() for e in enc]


def flush():
    gc.collect()
    torch.cuda.empty_cache()


t = time.time()
m = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.float32, device_map={"": 0}).eval()
fp32 = prefill_logits(m, "cuda")
del m
flush()
res["fp32_s"] = time.time() - t


def arm(device, graph):
    t = time.time()
    m, info = loader.load(MD, device=device, log=None)
    d = torch.device(device)
    gens = [decode.generate(m, e.to(d), max_new_tokens=NEW, graph=graph, return_logits=True) for e in enc[:NGEN]]
    gens = [{k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in g.items()} for g in gens]
    t1 = time.time()
    pre = prefill_logits(m, d)
    out = {"load_s": info["load_s"], "invariant": info["invariant"], "norm_fast": info.get("norm_fast"),
           "rope_twin": info.get("rope_twin"), "graph_used": [g["graph_used"] for g in gens],
           "step_ms_median": [sorted(g["step_ms"][1:])[len(g["step_ms"][1:]) // 2] for g in gens],
           "prefill_all_s": time.time() - t1, "wall_s": time.time() - t}
    del m
    flush()
    return gens, pre, out


gg, gp, res["gpu"] = arm("cuda", True)
print("gpu", res["gpu"], flush=True)
cg, cp, res["cpu"] = arm("cpu", False)
print("cpu", res["cpu"], flush=True)
res["real"] = {
    "gen_ids_equal": all(torch.equal(a["ids"], b["ids"]) for a, b in zip(cg, gg)),
    "gen_logits_equal": all(torch.equal(a["logits"].view(torch.int32), b["logits"].view(torch.int32))
                            for a, b in zip(cg, gg)),
    "prefill_logits_equal": sum(torch.equal(a.view(torch.int32), b.view(torch.int32)) for a, b in zip(cp, gp)),
    "prefill_prompts": len(cp)}
res["real"]["pass"] = res["real"]["gen_ids_equal"] and res["real"]["gen_logits_equal"] and \
    res["real"]["prefill_logits_equal"] == len(cp)
print("real", res["real"], flush=True)


def quality(pre):
    agree = n = 0
    mx = 0.0
    for a, r in zip(pre, fp32):
        agree += int((a.argmax(-1) == r.argmax(-1)).sum())
        n += r.shape[0]
        mx = max(mx, ((a - r).abs().max() / r.abs().max()).item())
    return {"top1_agree": agree / n, "max_rel": mx}


res["quality_vs_fp32"] = {"vahana_twin": quality(gp)}
print("quality", res["quality_vs_fp32"], flush=True)
res["gen_text_0"] = tok.decode(gg[0]["ids"][0])
json.dump(res, open(OUT, "w"), indent=1)
print("DONE", flush=True)
