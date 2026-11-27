"""v1.4: decode speed on varied real prompts (code, chat, math, story, Hindi), 96 tokens each after one warm-up call.

  python prompts_decode.py MODEL_DIR        (compare VAHANA_ADAPTIVE_CACHE=1 and =0)
"""
import os
import statistics
import sys
from argparse import Namespace

from vahana import cli

ns = Namespace(model=sys.argv[1], gpu_budget=None, reserve=1.5, threads=None, keep_numa_balancing=False, device=None,
               gpu_cache="auto")
setup = cli._setup(ns)
model, tok, torch = cli._load(ns, setup)
from vahana import bench, decode  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
dev = model._vahana_device
decode.generate(model, tok(bench.TEXT, return_tensors="pt").input_ids[:, :32].to(dev), max_new_tokens=8)
rates = []
for name, text in PROMPTS.items():
    r = decode.generate(model, tok(text, return_tensors="pt").input_ids.to(dev), max_new_tokens=96)
    st = r["step_ms"]
    rates.append(len(st) / (sum(st) / 1e3))
    print(f"{name:6s} decode {rates[-1]:.2f} tok/s (p50 {statistics.median(st):.1f} ms), ttft {r['prefill_s']:.2f} s",
          flush=True)
print(f"median over prompts: {statistics.median(rates):.2f} tok/s", flush=True)
