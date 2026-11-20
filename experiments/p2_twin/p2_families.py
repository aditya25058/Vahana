"""P2: CPU-only Vahana == GPU Vahana, bit for bit, per family (tiny random models, M2/M4 builder).

Per family, a batch of 2 left-padded prompts (lengths 23 / 40), 10 greedy steps (vahana.decode.generate):
  gpu_graph   model on the GPU (twin arithmetic), CUDA-graph decode
  gpu_eager   same, eager decode
  cpu         the same weights with the model on the CPU, no CUDA anywhere in the forward (eager)
  gpu_alone / cpu_alone   the 23-token prompt alone (batch 1, no padding)
Bars: TWIN   cpu == gpu_graph (ids and every step's logits, bit for bit)
      GRAPH  gpu_graph == gpu_eager
      BATCH  *_alone == row 0 of the batch, on the GPU and on the CPU
Also: the TwinMode audit (float torch calls that are neither twinned nor known exact) per family.
usage: p2_families.py OUT.json OUT.pt [family ...]
"""
import json
import sys
import traceback

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/m2_kernel")
import m2_families as M  # noqa: E402
from vahana import decode, loader  # noqa: E402
from vahana import tmath as T  # noqa: E402

OUT, OUTPT = sys.argv[1], sys.argv[2]
FAMS = sys.argv[3:] or M.TIER1 + M.EXTRA
dev = torch.device("cuda")
NEW = 10


def prompts(cfg):
    rng = torch.Generator().manual_seed(3)
    p = [torch.randint(1, cfg.vocab_size, (L,), generator=rng) for L in (23, 40)]
    ids = torch.zeros(2, 40, dtype=torch.long)
    am = torch.zeros(2, 40, dtype=torch.long)
    for b, x in enumerate(p):
        ids[b, 40 - len(x):] = x
        am[b, 40 - len(x):] = 1
    return p, ids, am


EXPORT = {}


def run(fam, where, graph, alone, audit=None):
    cfg, m = M.build(fam)                     # built on the GPU with fixed seeds: identical weights every time
    if fam not in EXPORT:                     # CROSS: the exact weights and config, for the laptop
        EXPORT[fam] = {"config": cfg.to_dict(), "model_type": cfg.model_type,
                       "state": {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}}
    if where == "cpu":
        m = m.to("cpu")
        torch.cuda.empty_cache()
    model, info = loader.prepare(m)
    if audit is not None:
        model.forward = model._vahana_twin_forward
        T.wrap_forward(model, audit)
    p, ids, am = prompts(cfg)
    d = torch.device(where if where == "cpu" else "cuda")
    if alone:
        r = decode.generate(model, p[0].view(1, -1).to(d), None, max_new_tokens=NEW, graph=graph,
                            return_logits=True)
    else:
        r = decode.generate(model, ids.to(d), am.to(d), max_new_tokens=NEW, graph=graph, return_logits=True)
    r = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in r.items()}
    del model, m
    torch.cuda.empty_cache()
    return r, info


def eq(a, b, row=None):
    la, lb = a["logits"], b["logits"]
    ia, ib = a["ids"], b["ids"]
    if row is not None:
        lb, ib = lb[row:row + 1], ib[row:row + 1]
    return torch.equal(ia, ib) and la.shape == lb.shape and torch.equal(la.view(torch.int32), lb.view(torch.int32))


res, saved = {}, {}
for fam in FAMS:
    out = {}
    try:
        audit = {}
        runs = {}
        runs["gpu_graph"], info = run(fam, "cuda", True, False)
        out["norm_fast"], out["rope_twin"] = info.get("norm_fast"), info.get("rope_twin")
        runs["gpu_eager"], _ = run(fam, "cuda", False, False)
        runs["cpu"], _ = run(fam, "cpu", False, False, audit)
        runs["gpu_alone"], _ = run(fam, "cuda", True, True)
        runs["cpu_alone"], _ = run(fam, "cpu", False, True)
        out["twin"] = eq(runs["cpu"], runs["gpu_graph"])
        out["graph"] = eq(runs["gpu_graph"], runs["gpu_eager"])
        out["batch_gpu"] = eq(runs["gpu_alone"], runs["gpu_graph"], 0)
        out["batch_cpu"] = eq(runs["cpu_alone"], runs["cpu"], 0)
        out["graph_used"] = runs["gpu_graph"]["graph_used"]
        out["graph_fallback"] = runs["gpu_graph"]["graph_fallback"]
        out["audit"] = audit
        if not out["twin"]:
            a, b = runs["cpu"]["logits"], runs["gpu_graph"]["logits"]
            out["twin_first_bad_step"] = int((a != b).flatten(2).any(-1).any(0).nonzero()[0]) \
                if a.shape == b.shape else "shape"
        g = runs["gpu_graph"]
        out["step_ms_gpu"] = sorted(g["step_ms"][1:])[len(g["step_ms"][1:]) // 2] if g["step_ms"][1:] else None
        c = runs["cpu"]
        out["step_ms_cpu"] = sorted(c["step_ms"])[len(c["step_ms"]) // 2] if c["step_ms"] else None
        p, ids, am = prompts(M.tiny_config(fam))
        saved[fam] = {"ids_gpu": runs["gpu_graph"]["ids"], "logits_gpu": runs["gpu_graph"]["logits"],
                      "ids_cpu": runs["cpu"]["ids"], "logits_cpu": runs["cpu"]["logits"], "input_ids": ids,
                      "attention_mask": am, **EXPORT[fam]}
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        out["trace"] = traceback.format_exc()[-2000:]
    res[fam] = out
    print(fam, {k: v for k, v in out.items() if k != "trace"}, flush=True)
    if "trace" in out:
        print(out["trace"], flush=True)
ok = [f for f in FAMS if res[f].get("twin") and res[f].get("graph") and res[f].get("batch_gpu")
      and res[f].get("batch_cpu")]
res["summary"] = {"families": len(FAMS), "all_pass": len(ok), "twin": sum(bool(res[f].get("twin")) for f in FAMS),
                  "not": [f for f in FAMS if f not in ok]}
print("SUMMARY", res["summary"], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
torch.save(saved, OUTPT)
