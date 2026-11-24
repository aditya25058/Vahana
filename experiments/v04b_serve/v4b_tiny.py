"""v0.4-b exactness on tiny models: continuous batching (engine.Engine) == solo decode.generate, bit for bit.
Per family: 20 requests (prompt lengths 3-6 in the first wave, then 3-30, token limits 4-24, some with an EOS id that triggers early), submitted
in 3 waves into 6 slots with a short cache (C = 64) so right shifts, left shifts and row moves all happen.
usage: v4b_tiny.py OUT.json [cuda|cpu] [gpu_cache_frac|-] [families]"""
import json
import sys

import torch

from vahana import decode, engine, loader, selftest

OUT = sys.argv[1]
dev = torch.device(sys.argv[2] if len(sys.argv) > 2 else "cuda")
CF = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] not in ("", "-") else None
FAMS = sys.argv[4].split(",") if len(sys.argv) > 4 else selftest.FAMILIES
res = {}
for fam in FAMS:
    out = {}
    try:
        cfg, m = selftest.build(fam)
        model, _ = loader.prepare(m.to(dev), gpu_cache_frac=CF if dev.type == "cuda" else None)
        if not engine.supported(model):
            res[fam] = {"skipped": "recurrent layers: the server serves these one at a time (exact)"}
            print(fam, res[fam], flush=True)
            continue
        g = torch.Generator().manual_seed(7)
        reqs = []
        for i in range(20):
            L = int(torch.randint(3, 7, (1,), generator=g)) if i < 8 else int(torch.randint(3, 31, (1,), generator=g))
            mx = int(torch.randint(4, 25, (1,), generator=g))
            reqs.append((torch.randint(1, cfg.vocab_size, (L,), generator=g).tolist(), mx))
        gr = dev.type == "cuda"
        refs, eos = [], []
        for i, (p, mx) in enumerate(reqs):
            r = decode.generate(model, torch.tensor([p], device=dev), None, max_new_tokens=mx, graph=gr,
                                return_logits=True)
            e = r["ids"][0, 2].item() if i % 4 == 1 and mx > 3 else None     # some stop early at their 3rd token
            if e is not None:
                r = decode.generate(model, torch.tensor([p], device=dev), None, max_new_tokens=mx, graph=gr,
                                    eos_token_id=e, return_logits=True)
            refs.append(r)
            eos.append(e)
        eng = engine.Engine(model, slots=6, max_len=64, graph=gr)
        handles = []
        waves = [(0, 8, 5), (8, 14, 3), (14, 20, 0)]
        for a, b, nsteps in waves:
            for i in range(a, b):
                handles.append(eng.submit(reqs[i][0], reqs[i][1], eos_token_id=eos[i], record_logits=True))
            for _ in range(nsteps):
                eng.step()
        eng.run()
        ok = []
        for h, r in zip(handles, refs):
            rid = r["ids"][0].tolist()
            same = h.ids == rid and len(h.logits) == r["logits"].shape[1] and torch.equal(
                torch.stack(h.logits).view(torch.int32), r["logits"][0].float().cpu().contiguous().view(torch.int32))
            ok.append(bool(same))
        out = {"exact": sum(ok), "of": len(ok), "graph": eng.graph_info(),
               "stats": {k: v for k, v in eng.stats.items() if k != "step_ms"}}
        eng.close()
        del model, m, eng
    except Exception as e:  # noqa: BLE001
        import traceback
        out = {"error": f"{type(e).__name__}: {str(e)[:200]}", "trace": traceback.format_exc()[-1500:]}
    res[fam] = out
    print(fam, {k: v for k, v in out.items() if k != "trace"}, flush=True)
tested = [f for f in res if "skipped" not in res[f]]
ok = [f for f in tested if "exact" in res[f] and res[f]["exact"] == res[f]["of"]]
res["summary"] = {"tested": len(tested), "exact": len(ok), "skipped": [f for f in res if "skipped" in res[f]],
                  "not_exact": [f for f in tested if f not in ok]}
print("SUMMARY", res["summary"], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
