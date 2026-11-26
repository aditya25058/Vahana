"""v1.3 prefix cache: time to first token for a long shared document + different questions, cold vs cached, and
the logits compared with a run without the cache.   python bench_prefix.py MODEL_DIR [doc_tokens=2048]"""
import hashlib
import os
import sys
import tempfile
import time
from argparse import Namespace

from vahana import cli

model_dir = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
ns = Namespace(model=model_dir, gpu_budget=None, reserve=1.5, threads=None, keep_numa_balancing=False, device=None,
               gpu_cache="auto")
setup = cli._setup(ns)
model, tok, torch = cli._load(ns, setup)
from vahana import bench, decode, prefix  # noqa: E402

dev = model._vahana_device
doc = tok(bench.TEXT * 200, return_tensors="pt").input_ids[:, :n]
qs = [" Question: who built the roads?", " Question: name the three periods.", " Question: when did it fall?"]
d = tempfile.mkdtemp(prefix="vahana_prefix_")
pc = prefix.PrefixCache(d, model_key="bench")
decode.generate(model, doc[:, :40].to(dev), max_new_tokens=2)                  # warm
h = lambda r: hashlib.sha256(r["logits"].float().cpu().numpy().tobytes()).hexdigest()[:16]
for i, q in enumerate(qs):
    ids = torch.cat([doc, tok(q, return_tensors="pt", add_special_tokens=False).input_ids], 1).to(dev)
    t = time.perf_counter()
    a = decode.generate(model, ids, max_new_tokens=8, return_logits=True, prefix_cache=pc)
    ta = time.perf_counter() - t
    hit = pc.last["hit_tokens"]
    b = decode.generate(model, ids, max_new_tokens=8, return_logits=True)
    print(f"question {i}: prompt {ids.shape[1]} tokens, reused {hit}; prefill {a['prefill_s']:.2f} s with cache vs "
          f"{b['prefill_s']:.2f} s without; logits {h(a)} {'==' if h(a) == h(b) else '!='} {h(b)}, tokens "
          f"{'==' if torch.equal(a['ids'], b['ids']) else '!='}", flush=True)
print("cache dir bytes", sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)))
