"""v1.4: Vahana vs llama.cpp on real prompts, same machine, same prompts, 96 greedy tokens each.

  python real_prompts_h2h.py --vahana-model DIR --gguf FILE --llama-dir DIR [--blocks 2] [--out results.json]
Prompts: prompts_decode.PROMPTS (code, chat, math, story, Hindi). Engines alternate by block (A B, then B A).
llama.cpp: llama-completion -ngl 99 --n-cpu-moe N -t T --temp 0 -n 96 (defaults 42 and 8: the laptop's best), one process per prompt with its default
warm-up; its own timings (prompt eval = time to first token, eval = decode tok/s). Vahana: prompts_decode.py, one
process for all five prompts after one warm-up call (decode tok/s and prefill time per prompt).
Reported: per-prompt decode tok/s and time to first token, medians per engine, and the ratio of decode medians.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile

ap = argparse.ArgumentParser()
ap.add_argument("--vahana-model", required=True)
ap.add_argument("--gguf", required=True)
ap.add_argument("--llama-dir", required=True)
ap.add_argument("--ncmoe", default="42")
ap.add_argument("--threads", default="8")
ap.add_argument("--ngl", default="99")
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--out", default=None)
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "prompts_decode.py"), encoding="utf-8").read()
i = src.index("PROMPTS = {")
ns = {}
exec(src[i:src.index("}\n", i) + 1], ns)
PROMPTS = ns["PROMPTS"]
tmp = tempfile.mkdtemp()


def llama():
    exe = os.path.join(a.llama_dir, "llama-completion.exe" if os.name == "nt" else "llama-completion")
    out = {}
    for name, text in PROMPTS.items():
        f = os.path.join(tmp, name + ".txt")
        open(f, "w", encoding="utf-8", newline="").write(text)
        r = subprocess.run([exe, "-m", a.gguf, "-ngl", a.ngl, *(["--n-cpu-moe", a.ncmoe] if a.ngl != "0" else []),
                            "-t", a.threads, "-f", f, "-n", "96", "--temp", "0", "-no-cnv"], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=1800)
        log = r.stdout + r.stderr
        pe = re.search(r"prompt eval time =\s*([\d.]+) ms", log)
        ev = re.search(r" eval time =\s*[\d.]+ ms /\s*\d+ runs.*?([\d.]+) tokens per second", log)
        out[name] = {"decode": float(ev.group(1)), "ttft": float(pe.group(1)) / 1e3}
    return out


def vahana():
    env = dict(os.environ, OMP_WAIT_POLICY="PASSIVE")
    r = subprocess.run([sys.executable, os.path.join(here, "prompts_decode.py"), a.vahana_model], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=3600, env=env)
    out = {}
    for line in r.stdout.splitlines():
        m = re.match(r"(\w+)\s+decode ([\d.]+) tok/s .* ttft ([\d.]+) s", line)
        if m:
            out[m.group(1)] = {"decode": float(m.group(2)), "ttft": float(m.group(3))}
    if not out:
        sys.exit(f"Vahana run failed (exit {r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
    return out


res = {"llama.cpp": [], "vahana": []}
order = [("llama.cpp", llama), ("vahana", vahana)]
for b in range(a.blocks):
    for name, fn in (order if b % 2 == 0 else order[::-1]):
        r = fn()
        res[name].append(r)
        print(f"block {b} {name}: " + ", ".join(f"{k} {v['decode']:.2f} tok/s / {v['ttft']:.2f} s" for k, v in r.items()),
              flush=True)
summary = {}
for name, runs in res.items():
    dec = [v["decode"] for r in runs for v in r.values()]
    tt = [v["ttft"] for r in runs for v in r.values()]
    summary[name] = {"runs": runs, "decode_median": statistics.median(dec), "decode_min": min(dec),
                     "decode_max": max(dec), "ttft_median": statistics.median(tt)}
summary["decode_ratio_of_medians"] = round(summary["vahana"]["decode_median"] / summary["llama.cpp"]["decode_median"], 3)
summary["ttft_ratio_of_medians"] = round(summary["llama.cpp"]["ttft_median"] / summary["vahana"]["ttft_median"], 3)
print(json.dumps({k: v for k, v in summary.items() if k not in ("llama.cpp", "vahana")}
                 | {k: {x: y for x, y in summary[k].items() if x != "runs"} for k in ("llama.cpp", "vahana")}, indent=1))
if a.out:
    json.dump(summary, open(a.out, "w"), indent=1)
