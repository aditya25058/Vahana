"""Cost of a bare host node per layer: experts replaced by (D2H copies, a no-op host job, H2D copy)."""
import os, torch, sys, json
from transformers import AutoTokenizer
from vahana import loader, decode, runtime, hybrid
from vahana.experts import VahanaExperts
MD = sys.argv[1]
model, info = loader.load(MD, device="cuda", log=None)
tok = AutoTokenizer.from_pretrained(MD)
ids = tok(json.load(open("experiments/m5_real/prompts.json"))[0], return_tensors="pt").input_ids.cuda()
k = hybrid.ext()
orig = VahanaExperts.forward
NOOP = {}
def noop(self, h, ti, tw):
    if not runtime.GRAPH["on"]:
        return orig(self, h, ti, tw)
    N, H = h.shape
    key = (self.spec.slot, N)
    if key not in NOOP:
        b = (runtime._pin((N, H), torch.bfloat16), runtime._pin(tuple(ti.shape), torch.int64),
             runtime._pin(tuple(tw.shape), tw.dtype), runtime._pin((N, H), torch.bfloat16))
        NOOP[key] = b + (k.make_host_job(self.spec.slot, *b, self.numa, True),)
    xb, tib, twb, ob, job = NOOP[key]
    xb.copy_(h, non_blocking=True); tib.copy_(ti, non_blocking=True); twb.copy_(tw, non_blocking=True)
    k.enqueue_host_job(runtime._stream(), job)
    out = torch.empty((N, H), dtype=torch.bfloat16, device=h.device)
    out.copy_(ob, non_blocking=True)
    return out
def nohost(self, h, ti, tw):
    if not runtime.GRAPH["on"]:
        return orig(self, h, ti, tw)
    return torch.zeros_like(h)
for mode, f in (("base", orig), ("noop_host", noop), ("no_host", nohost), ("base", orig), ("noop_host", noop)):
    VahanaExperts.forward = f
    g = decode.generate(model, ids, max_new_tokens=33, graph=True)
    st = g["step_ms"][1:]
    print(f"{mode:10s} steady {sorted(st)[len(st)//2]:7.1f} ms", flush=True)
