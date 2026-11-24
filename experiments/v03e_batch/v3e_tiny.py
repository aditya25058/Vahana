"""v0.3-e exactness on tiny models: a batched (left-padded) greedy decode gives every sequence the same bits as
decoding it alone. Per family (selftest's numpy-seeded tiny models): B prompts of different lengths, 16 new tokens;
vahana.generate_batch vs B solo decode.generate runs: ids equal and every step's logit vector bit-identical.
usage: v3e_tiny.py OUT.json [cuda|cpu] [gpu_cache_frac|-] [B] [families]"""
import json
import sys

import torch

from vahana import batch, decode, loader, selftest

OUT = sys.argv[1]
dev = torch.device(sys.argv[2] if len(sys.argv) > 2 else "cuda")
CF = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] not in ("", "-") else None
B = int(sys.argv[4]) if len(sys.argv) > 4 else 4
FAMS = sys.argv[5].split(",") if len(sys.argv) > 5 else selftest.FAMILIES
NEW = 16
LENS = ([5, 9, 13, 20, 3, 17, 8, 11, 6, 15, 4, 19, 7, 12, 10, 14] * 4)[:B]
res = {}
for fam in FAMS:
    out = {}
    try:
        cfg, m = selftest.build(fam)
        model, _ = loader.prepare(m.to(dev), gpu_cache_frac=CF if dev.type == "cuda" else None)
        g = torch.Generator().manual_seed(11)
        prompts = [torch.randint(1, cfg.vocab_size, (n,), generator=g) for n in LENS]
        gr = dev.type == "cuda"
        bat = batch.generate_batch(model, prompts, NEW, graph=gr, return_logits=True)
        rows = []
        for b, p in enumerate(prompts):
            solo = decode.generate(model, p.view(1, -1).to(dev), None, max_new_tokens=NEW, graph=gr, return_logits=True)
            o = bat["outputs"][b]
            rows.append(o["ids"] == solo["ids"][0].tolist() and torch.equal(
                o["logits"].view(torch.int32), solo["logits"][0].view(torch.int32)))
        out = {"rows_exact": sum(rows), "B": B, "batched": bat["batched"], "graph": bat["graph_used"]}
        del model, m
    except Exception as e:  # noqa: BLE001
        import traceback
        out = {"error": f"{type(e).__name__}: {str(e)[:200]}", "trace": traceback.format_exc()[-1500:]}
    res[fam] = out
    print(fam, {k: v for k, v in out.items() if k != "trace"}, flush=True)
ok = [f for f in res if res[f].get("rows_exact") == B]
res["summary"] = {"tested": len(FAMS), "exact": len(ok), "not_exact": [f for f in FAMS if f not in ok]}
print("SUMMARY", res["summary"], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
