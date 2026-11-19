"""M4 tiny: generic decode graph == eager, and placement invariance, per family (tiny random models).

Per family, batch of 2 left-padded prompts (lengths 23 / 40), 10 greedy steps:
  gpu_eager   all non-expert weights on the GPU, eager steps
  gpu_graph   same, CUDA-graph steps (host nodes for the CPU experts)
  cpu_eager / cpu_graph   input embedding, layer 1 o_proj (if any) and every shared expert placed in CPU RAM
              (shared experts folded into the CPU expert job where the block adds them directly)
Bars: GRAPH  gpu_graph == gpu_eager and cpu_graph == cpu_eager (ids and every step's logits, bit for bit)
      PLACE  cpu_eager == gpu_eager (bit for bit)
usage: m4_tiny.py OUT.json [family ...]
"""
import json
import sys
import traceback

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0].replace("m4_decode", "m2_kernel"))
import m2_families as M  # noqa: E402
from vahana import decode, loader  # noqa: E402

OUT = sys.argv[1]
FAMS = sys.argv[2:] or M.TIER1 + ["gpt_oss"]
FORCE = r"embed_tokens$|layers\.1\.self_attn\.o_proj$|shared_expert"
dev = torch.device("cuda")


def run(fam, force, graph):
    cfg, m = M.build(fam)
    model, info = loader.prepare(m, force_cpu=force)
    rng = torch.Generator().manual_seed(3)
    p = [torch.randint(1, cfg.vocab_size, (L,), generator=rng) for L in (23, 40)]
    ids = torch.zeros(2, 40, dtype=torch.long)
    am = torch.zeros(2, 40, dtype=torch.long)
    for b, x in enumerate(p):
        ids[b, 40 - len(x):] = x
        am[b, 40 - len(x):] = 1
    r = decode.generate(model, ids.to(dev), am.to(dev), max_new_tokens=10, graph=graph, return_logits=True)
    del model, m
    torch.cuda.empty_cache()
    return r, info


res = {}
for fam in FAMS:
    out = {}
    try:
        runs = {}
        for name, force, graph in (("gpu_eager", None, False), ("gpu_graph", None, True),
                                   ("cpu_eager", FORCE, False), ("cpu_graph", FORCE, True)):
            runs[name], info = run(fam, force, graph)
            if name.startswith("cpu"):
                out["cpu_modules"], out["folded"] = info["cpu_modules"], info["folded_shared"]
        eq = lambda a, b: torch.equal(runs[a]["ids"], runs[b]["ids"]) and torch.equal(runs[a]["logits"], runs[b]["logits"])
        out["graph_eq_eager_gpu"] = eq("gpu_graph", "gpu_eager")
        out["graph_eq_eager_cpu"] = eq("cpu_graph", "cpu_eager")
        out["placement_invariant"] = eq("cpu_eager", "gpu_eager")
        out["graph_used"] = runs["gpu_graph"]["graph_used"] and runs["cpu_graph"]["graph_used"]
        out["graph_fallback"] = runs["gpu_graph"]["graph_fallback"] or runs["cpu_graph"]["graph_fallback"]
        out["graph_step_ms"] = sorted(runs["gpu_graph"]["step_ms"][1:])[len(runs["gpu_graph"]["step_ms"][1:]) // 2]
        out["eager_step_ms"] = sorted(runs["gpu_eager"]["step_ms"][1:])[len(runs["gpu_eager"]["step_ms"][1:]) // 2]
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        out["trace"] = traceback.format_exc()[-1500:]
    res[fam] = out
    print(fam, {k: v for k, v in out.items() if k not in ("trace", "cpu_modules")},
          len(out.get("cpu_modules", [])), flush=True)
ok = [f for f in FAMS if res[f].get("graph_eq_eager_gpu") and res[f].get("graph_eq_eager_cpu")
      and res[f].get("placement_invariant") and res[f].get("graph_used")]
res["summary"] = {"families": len(FAMS), "all_pass": len(ok), "not": [f for f in FAMS if f not in ok]}
print("SUMMARY", res["summary"], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
