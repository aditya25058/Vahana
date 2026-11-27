"""v1.4 research: how many routed expert uses could a GPU expert cache of fixed size serve, by caching policy?

  python routing_sim.py MODEL_DIR [N_NEW] [CAPACITY]
Records real routing (CPU device, eager) for prompts of different kinds, the prompt's prefill and N_NEW greedy decode
tokens each, then replays the decode routing against caches holding CAPACITY (layer, expert) pairs:
  static     the shipped policy: popularity profiled once on built-in text (~/.cache/vahana/popularity)
  prompt     re-chosen after each prompt's prefill, from that prefill's routing counts
  oracle     the best static set for that generation, chosen in hindsight (an upper bound for any static choice)
  lfu-s      start from `prompt`; after every decode step, swap in up to s experts whose decayed use count beats the
             coldest cached one (models background uploads during the GPU's idle wait)
Prints the hit rate (share of decode expert uses served by the cache) per prompt and policy.
"""
import heapq
import json
import os
import sys
from argparse import Namespace
from collections import Counter

from vahana import cli

ns = Namespace(model=sys.argv[1], gpu_budget=None, reserve=1.5, threads=None, keep_numa_balancing=False,
               device="cpu", gpu_cache="0")
N_NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 96
CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 386
setup = cli._setup(ns)
model, tok, torch = cli._load(ns, setup)
from vahana import decode  # noqa: E402
from vahana.experts import VahanaExperts  # noqa: E402

PROMPTS = {
    "code": "Write a Python function that parses a CSV file of transactions and returns the total amount per "
            "customer, with type hints and a docstring.\n\ndef",
    "chat": "Hi! I'm planning a three-day trip to Kyoto in autumn. What should I see, where should I eat, and how "
            "should I get around the city?",
    "math": "Solve step by step: a train leaves at 9:40 travelling at 72 km/h; a second train leaves the same "
            "station at 10:05 at 90 km/h on the same track. When does the second train catch up?",
    "story": "Once upon a time, in a village at the edge of a great salt desert, there lived a clockmaker who",
    "hindi": "भारत के इतिहास में मौर्य साम्राज्य का महत्व क्या था? विस्तार से समझाइए।",
}
rec = {"on": False, "steps": []}
orig = VahanaExperts.forward


def fwd(self, h, ti, tw):
    if rec["on"]:
        rec["cur"].append((self.spec.slot, ti.reshape(-1, ti.shape[-1]).tolist()))
    return orig(self, h, ti, tw)


VahanaExperts.forward = fwd
runs = {}
for name, text in PROMPTS.items():
    ids = tok(text, return_tensors="pt").input_ids
    rec["on"], rec["cur"] = True, []
    calls = []
    orig_fwd = model.forward

    def mf(*a, **k):
        rec["cur"] = []
        out = orig_fwd(*a, **k)
        calls.append(rec["cur"])
        return out
    model.forward = mf
    try:
        decode.generate(model, ids, max_new_tokens=N_NEW, graph=False)
    finally:
        model.forward = orig_fwd
        rec["on"] = False
    pre = Counter()
    for L, rows in calls[0]:
        for r in rows:
            pre.update((L, e) for e in r)
    steps = []
    for c in calls[1:]:
        steps.append([(L, e) for L, rows in c for e in rows[0]])
    runs[name] = (pre, steps)
    print(f"{name}: prompt {ids.shape[1]} tokens, {len(steps)} decode steps recorded", flush=True)
VahanaExperts.forward = orig
if os.environ.get("ROUTING_OUT"):
    json.dump({n: {"pre": [[L, e, c] for (L, e), c in pre.items()], "steps": steps} for n, (pre, steps) in runs.items()},
              open(os.environ["ROUTING_OUT"], "w"))

key = None
pop_dir = os.path.join(os.path.expanduser("~"), ".cache", "vahana", "popularity")
static = None
if os.path.isdir(pop_dir):
    E = model.config.num_experts
    for f in os.listdir(pop_dir):
        d = json.load(open(os.path.join(pop_dir, f)))
        if len(d) == model.config.num_hidden_layers and len(next(iter(d.values()))) == E:
            c = Counter({(int(L), e): v for L, vs in d.items() for e, v in enumerate(vs)})
            static = set(x for x, _ in c.most_common(CAP))


def hits(cache, steps):
    n = h = 0
    for s in steps:
        for x in s:
            n += 1
            h += x in cache
    return h / max(1, n)


def lfu(start, steps, swaps, decay=0.97):
    cache = set(start)
    score = Counter()
    n = h = 0
    for s in steps:
        for x in s:
            n += 1
            h += x in cache
        for x in list(score):
            score[x] *= decay
        for x in s:
            score[x] += 1.0
        if swaps:
            cand = heapq.nlargest(swaps, (x for x in score if x not in cache), key=lambda x: score[x])
            for x in cand:
                cold = min(cache, key=lambda y: score.get(y, 0.0))
                if score[x] > score.get(cold, 0.0) + 1.0:
                    cache.remove(cold)
                    cache.add(x)
    return h / max(1, n)


print(f"\ncache capacity {CAP} (layer, expert) pairs; decode hit rate by policy")
print(f"{'prompt':8s} {'static':>7s} {'loo':>7s} {'prompt':>7s} {'oracle':>7s} {'lfu-1':>7s} {'lfu-4':>7s} {'lfu-16':>7s}")
for name, (pre, steps) in runs.items():
    allc = Counter(x for s in steps for x in s)
    oracle = set(x for x, _ in allc.most_common(CAP))
    pset = set(x for x, _ in pre.most_common(CAP))
    others = Counter(x for n2, (_, st2) in runs.items() if n2 != name for s in st2 for x in s)
    loo = set(x for x, _ in others.most_common(CAP))            # static, profiled on the other prompts' decode routing
    row = [hits(static, steps) if static else float("nan"), hits(loo, steps), hits(pset, steps), hits(oracle, steps),
           lfu(pset, steps, 1), lfu(pset, steps, 4), lfu(pset, steps, 16)]
    print(f"{name:8s} " + " ".join(f"{v:7.1%}" for v in row), flush=True)
