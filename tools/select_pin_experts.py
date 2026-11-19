"""Choose routed experts to pin under a byte budget from --log-routing traces.

Counts, per (layer, expert), the decode steps whose routed union contained it
(summed over trace files), then takes the most frequent experts until the
budget is spent. Ties break by (layer, expert) so the choice is deterministic.

Usage: python rudra/select_pin_experts.py --trace a.json [b.json ...]
           --expert-bytes 47185920 --budget-gb 60 --out sel60.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter


def count_routing(entries: list[dict]) -> Counter:
    c: Counter = Counter()
    for e in entries:
        for x in e["experts"]:
            c[(int(e["layer"]), int(x))] += 1
    return c


def select(counts: Counter, n: int) -> list[tuple[int, int]]:
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ranked[:n]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", nargs="+", required=True)
    ap.add_argument("--expert-bytes", type=int, required=True)
    ap.add_argument("--budget-gb", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    counts: Counter = Counter()
    for t in a.trace:
        with open(t) as f:
            counts += count_routing(json.load(f))
    n = int(a.budget_gb * 1e9 // a.expert_bytes)
    sel = select(counts, n)
    total = sum(counts.values())
    covered = sum(counts[k] for k in sel)
    with open(a.out, "w") as f:
        json.dump([list(k) for k in sel], f)
    print(json.dumps({"budget_gb": a.budget_gb, "experts": len(sel), "distinct_seen": len(counts),
                      "in_sample_coverage": covered / total if total else None}))


if __name__ == "__main__":
    main()
