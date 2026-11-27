"""v1.4: the head-to-head protocol, Vahana vs llama.cpp on one machine.

  python head_to_head.py --vahana-model DIR --gguf FILE --llama-dir DIR [--blocks 2] [--runs 5] [--out results.json]

For each block, each engine first warms up (one discarded run, so its weights are back in the page cache), then runs
--runs times. Engines alternate by block (A B A B ...), so neither always runs on a cache the other just evicted.
Decode: 64 tokens. llama.cpp: llama-bench -n 64 -p 0 -r 3 at its best MoE setting (-ngl 99 --n-cpu-moe N -t T),
the median of its two warm repetitions (its first repetition in a fresh process is slow). Vahana:
64 tokens after a 512-token prompt (experiments/v15_decode/profile_decode.py, decode only). Reported: per-run
tok/s, median and min-max per engine, and the ratio of medians.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--vahana-model", required=True)
ap.add_argument("--gguf", required=True)
ap.add_argument("--llama-dir", required=True)
ap.add_argument("--ncmoe", default="42")
ap.add_argument("--threads", default="8")
ap.add_argument("--ngl", default="99")
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--runs", type=int, default=5)
ap.add_argument("--out", default=None)
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))


def llama_once():
    exe = os.path.join(a.llama_dir, "llama-bench.exe" if os.name == "nt" else "llama-bench")
    r = subprocess.run([exe, "-m", a.gguf, "-ngl", a.ngl, *(["--n-cpu-moe", a.ncmoe] if a.ngl != "0" else []), "-t", a.threads, "-p", "0", "-n", "64",
                        "-r", "3", "-o", "jsonl"], capture_output=True, text=True, timeout=1800)
    # llama.cpp's first repetition in a fresh process is slow (its mmap'd weights fault in); later repetitions in
    # the same process are its warm speed. Vahana's runs also warm up in-process before measuring, so report the
    # median of llama.cpp's warm repetitions (2 and 3).
    ts = json.loads(r.stdout.strip().splitlines()[0])["samples_ts"]
    return statistics.median(ts[1:])


def vahana_once():
    env = dict(os.environ, PROFILE_DECODE_ONLY="1", OMP_WAIT_POLICY="PASSIVE")
    r = subprocess.run([sys.executable, os.path.join(here, "profile_decode.py"), a.vahana_model], capture_output=True,
                       text=True, timeout=1800, env=env)
    line = next((x for x in r.stdout.splitlines() if x.startswith("decode:")), None)
    if line is None:
        sys.exit(f"Vahana run failed (exit {r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
    return float(line.split()[1])


res = {"llama.cpp": [], "vahana": []}
order = [("llama.cpp", llama_once), ("vahana", vahana_once)]
for b in range(a.blocks):
    for name, fn in (order if b % 2 == 0 else order[::-1]):
        fn()                                            # warm-up, discarded
        for _ in range(a.runs):
            v = fn()
            res[name].append(round(v, 2))
            print(f"block {b} {name}: {v:.2f} tok/s", flush=True)
summary = {k: {"runs": v, "median": statistics.median(v), "min": min(v), "max": max(v)} for k, v in res.items()}
summary["ratio_of_medians"] = round(summary["vahana"]["median"] / summary["llama.cpp"]["median"], 3)
print(json.dumps(summary, indent=1))
if a.out:
    json.dump(summary, open(a.out, "w"), indent=1)
