"""v0.3-b exactness on tiny models: speculative decoding == plain greedy decoding, bit for bit.

Per family (selftest's numpy-seeded tiny models), one prompt with a repeated pattern, 24 new tokens:
  plain    decode.generate (graph on the GPU), the reference
  lookup   speculative.generate_spec, prompt-lookup drafts, k = 4
  oracle   generate_spec with the reference's own future tokens as drafts (every draft accepted: each step verifies
           k + 1 real tokens), k = 4 and k = 7
Bar: ids equal and every emitted token's logit vector bit-identical to plain's, for every family and mode.
usage: v3b_tiny.py OUT.json [cuda|cpu] [gpu_cache_frac]
"""
import json
import sys

import torch

from vahana import decode, loader, selftest, speculative

OUT = sys.argv[1]
dev = torch.device(sys.argv[2] if len(sys.argv) > 2 else "cuda")
CF = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] not in ("", "-") else None
NEW = 24
res = {}
FAMS = sys.argv[4].split(",") if len(sys.argv) > 4 else selftest.FAMILIES
for fam in FAMS:
    out = {}
    try:
        cfg, m = selftest.build(fam)
        m = m.to(dev)
        model, _ = loader.prepare(m, gpu_cache_frac=CF if dev.type == "cuda" else None)
        g = torch.Generator().manual_seed(5)
        pat = torch.randint(1, cfg.vocab_size, (6,), generator=g)
        ids = torch.cat([pat, pat, pat, torch.randint(1, cfg.vocab_size, (3,), generator=g), pat[:4]]).view(1, -1).to(dev)
        if any(t not in ("full_attention", "sliding_attention", "attention")
               for t in (getattr(cfg, "layer_types", None) or [])):
            out["skipped"] = "recurrent / linear-attention layers (speculative decoding needs a position-indexed cache)"
            res[fam] = out
            print(fam, out, flush=True)
            continue
        ref = decode.generate(model, ids, None, max_new_tokens=NEW, graph=dev.type == "cuda", return_logits=True)
        rid, rlg = ref["ids"][0].tolist(), ref["logits"][0]
        def oracle(hist, k, n0=ids.shape[1]):
            done = len(hist) - n0
            return rid[done:done + k]
        for name, kw in (("lookup_k4", dict(k=4)), ("oracle_k4", dict(k=4, draft_fn=oracle)),
                         ("oracle_k7", dict(k=7, draft_fn=oracle))):
            r = speculative.generate_spec(model, ids, max_new_tokens=NEW, graph=dev.type == "cuda",
                                          return_logits=True, **kw)
            same_ids = r["ids"][0].tolist() == rid
            same_lg = r["logits"].shape == ref["logits"].shape and torch.equal(
                r["logits"][0].view(torch.int32), rlg.view(torch.int32))
            out[name] = {"ids": same_ids, "logits": same_lg, "tokens_per_step": round(r["tokens_per_step"], 2),
                         "graph": r["graph_used"], "fallback": r["graph_fallback"]}
        del model, m
    except Exception as e:  # noqa: BLE001
        import traceback
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["trace"] = traceback.format_exc()[-1200:]
    res[fam] = out
    print(fam, {k: v for k, v in out.items() if k != "trace"}, flush=True)
modes = ("lookup_k4", "oracle_k4", "oracle_k7")
ok = [f for f in res if "skipped" not in res[f] and all(res[f].get(m, {}).get("ids") and res[f].get(m, {}).get("logits")
                                                       for m in modes)]
res["summary"] = {"tested": sum(1 for f in res if "skipped" not in res[f]), "exact": len(ok),
                  "skipped": [f for f in res if "skipped" in res[f]]}
print("SUMMARY", res["summary"], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
