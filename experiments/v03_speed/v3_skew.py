"""v0.3: expert popularity skew. Count routed (token, slot) per expert per layer over prompts (prefill + greedy
decode), then the hit rate if the top-f fraction of experts per layer were cached (f picked on half the prompts,
measured on the other half, so the cache choice is not fitted to the test)."""
import json, sys, collections
import torch
from transformers import AutoTokenizer
from vahana import decode, loader
from vahana.experts import VahanaExperts
MD, PJ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
tok = AutoTokenizer.from_pretrained(MD)
prompts = json.load(open(PJ))
model, info = loader.load(MD, device="cuda", log=None)
counts = {}          # phase -> layer -> Counter
phase = ["calib"]
orig = VahanaExperts.forward
def fwd(self, h, ti, tw):
    c = counts.setdefault(phase[0], {}).setdefault(self.spec.slot, collections.Counter())
    c.update(ti.flatten().tolist())
    return orig(self, h, ti, tw)
VahanaExperts.forward = fwd
for i, p in enumerate(prompts):
    phase[0] = "calib" if i % 2 == 0 else "test"
    ids = tok(p, return_tensors="pt").input_ids.cuda()
    decode.generate(model, ids, max_new_tokens=32, graph=False)
E = next(m for m in model.modules() if isinstance(m, VahanaExperts)).spec.E
res = {"model": MD.rsplit("/", 1)[-1], "experts": E}
for f in (0.05, 0.1, 0.125, 0.2, 0.25, 0.33, 0.5):
    hit = tot = 0
    for L, ct in counts["test"].items():
        top = {e for e, _ in counts["calib"][L].most_common(max(1, round(f * E)))}
        hit += sum(n for e, n in ct.items() if e in top); tot += sum(ct.values())
    res[f"hit@{f}"] = hit / tot
print(json.dumps(res), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
