import os, time, torch, sys, json
from transformers import AutoTokenizer
from vahana import loader, decode, cache
from vahana.modelspec import parent_of
MD = sys.argv[1]
model, info = loader.load(MD, device="cuda", log=None, gpu_cache_gb=18)
tok = AutoTokenizer.from_pretrained(MD)
ids = tok(json.load(open("experiments/m5_real/prompts.json"))[0], return_tensors="pt").input_ids.cuda()
mc = os.environ.get("VAHANA_MAIN_CPUS")
if mc:
    os.sched_setaffinity(0, {int(c) for c in mc.split(",")})
mods = [(n, m) for n, m in model.named_modules() if isinstance(m, cache.CachedExperts)]
real_lut = {n: m.lut.clone() for n, m in mods}
def set_mode(mode):
    for n, m in mods:
        parent, leaf = parent_of(model, n)
        if mode == "base":
            setattr(parent, leaf, m.base)
        else:
            setattr(parent, leaf, m)
            m.lut.copy_(real_lut[n] if mode == "cache" else torch.full_like(m.lut, -1))
for r in range(3):
    for mode in ("base", "cache"):
        set_mode(mode)
        g = decode.generate(model, ids, max_new_tokens=33, graph=True, return_logits=True)
        st = g["step_ms"][1:]
        print(f"r{r} {mode:6s} steady step {sorted(st)[len(st)//2]:7.1f} ms  graph {g['graph_used']}  ids {g['ids'][0,:6].tolist()}", flush=True)
