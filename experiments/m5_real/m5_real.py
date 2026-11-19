"""M5: a real checkpoint through Vahana: exactness against the GPU-twin reference, quality against stock fp32.

Arms (sequential, one model in memory at a time; stock first):
  cpu    vahana.load (CPU experts): greedy generation of 48 tokens on the first 5 prompts with the decode graph;
         prefill logits (batch 1) on all 50 prompts
  twin   vahana.load(experts="gpu_twin") (every expert on the GPU, same arithmetic): the same, decode eager
  fp32   stock transformers in fp32 (GPU, CPU offload if needed): prefill logits on the 50 prompts (batch 1)
  bf16   stock transformers in bf16: the same (baseline: how far plain bf16 already is from fp32)
Quality per arm vs fp32, over every prompt position: next-token top-1 agreement, max |delta| / max |fp32 logit|.
usage: m5_real.py MODEL_DIR PROMPTS.json OUT.json [skip_twin]
"""
import gc
import json
import sys
import time

import torch

from vahana import decode, loader

MD, PJ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SKIP_TWIN = len(sys.argv) > 4 and sys.argv[4] == "skip_twin"
dev = torch.device("cuda")
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MD)
prompts = json.load(open(PJ))
enc = [tok(p, return_tensors="pt").input_ids.to(dev) for p in prompts]
res = {"model": MD, "prompts": len(prompts), "positions": int(sum(e.shape[1] for e in enc))}


@torch.no_grad()
def prefill_logits(m):
    return [m(e).logits.float()[0].cpu() for e in enc]


def flush():
    gc.collect()
    torch.cuda.empty_cache()


def vahana_arm(mode, graph):
    t = time.time()
    m, info = loader.load(MD, experts=mode, log=None)
    gens = [decode.generate(m, e, max_new_tokens=48, graph=graph, return_logits=True) for e in enc[:5]]
    pre = prefill_logits(m)
    out = {"load_s": info["load_s"], "gpu_gb": info["gpu_gb"], "graph_used": [g["graph_used"] for g in gens],
           "wall_s": time.time() - t}
    del m                      # the model must be released in this frame (hooks form reference cycles)
    flush()
    return gens, pre, out


# stock arms first: vahana patches nn.Linear process-wide, so stock must run before any vahana arm
def stock(dtype):
    kw = dict(dtype=dtype, device_map="auto", max_memory={0: "72GiB", "cpu": "300GiB"})
    t = time.time()
    m = AutoModelForCausalLM.from_pretrained(MD, **kw).eval()
    pre = prefill_logits(m)
    del m
    flush()
    return pre, time.time() - t


fp32, res["fp32_s"] = stock(torch.float32)
bf16, res["bf16_s"] = stock(torch.bfloat16)


cpu_gen, cpu_pre, res["cpu"] = vahana_arm("cpu", True)
print("cpu", res["cpu"], flush=True)
if not SKIP_TWIN:
    tw_gen, tw_pre, res["twin"] = vahana_arm("gpu_twin", False)
    res["exact"] = {
        "gen_ids_equal": all(torch.equal(a["ids"], b["ids"]) for a, b in zip(cpu_gen, tw_gen)),
        "gen_logits_equal": all(torch.equal(a["logits"], b["logits"]) for a, b in zip(cpu_gen, tw_gen)),
        "prefill_logits_equal": all(torch.equal(a, b) for a, b in zip(cpu_pre, tw_pre))}
    print("exact", res["exact"], flush=True)


def quality(pre):
    agree = n = 0
    mx = 0.0
    for a, r in zip(pre, fp32):
        agree += int((a.argmax(-1) == r.argmax(-1)).sum())
        n += r.shape[0]
        mx = max(mx, ((a - r).abs().max() / r.abs().max()).item())
    return {"top1_agree": agree / n, "max_rel": mx}


res["quality_vs_fp32"] = {"vahana": quality(cpu_pre), "stock_bf16": quality(bf16)}
print("quality", res["quality_vs_fp32"], flush=True)
res["gen_text_0"] = tok.decode(cpu_gen[0]["ids"][0])
json.dump(res, open(OUT, "w"), indent=1)
print("DONE", flush=True)
