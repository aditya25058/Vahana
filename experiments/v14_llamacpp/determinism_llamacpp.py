"""Does llama.cpp give the same output for the same prompt (a) run twice, (b) alongside other requests (--parallel),
(c) on the CPU instead of the GPU?  Greedy decoding; compares the generated tokens and each token's top-5 log
probabilities exactly, as returned by llama-server (/completion with n_probs).

  python determinism_llamacpp.py LLAMA_DIR MODEL.gguf OUT.json [n_cpu_moe=42] [threads=8]
"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

LLAMA, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NCMOE = sys.argv[4] if len(sys.argv) > 4 else "42"
TH = sys.argv[5] if len(sys.argv) > 5 else "8"
PORT = 8077
PROMPT = ("The history of the Roman Empire spans more than a thousand years. Historians divide it into the Kingdom, "
          "the Republic and the Empire. Explain in three sentences why the Empire split into East and West:")
OTHERS = ["Write a short poem about the sea.", "List five prime numbers and explain why they are prime.",
          "Describe how a bicycle works in simple terms."]


def server(args):
    exe = os.path.join(LLAMA, "llama-server.exe")
    p = subprocess.Popen([exe, "-m", MODEL, "--port", str(PORT), "-c", "4096", "-t", TH, "--seed", "1", *args],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(600):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if json.load(r).get("status") == "ok":
                    return p
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    p.kill()
    raise RuntimeError("server did not start")


def complete(prompt, n=32):
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "n_probs": 5,
                       "cache_prompt": False, "seed": 1}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        js = json.load(r)
    probs = js.get("completion_probabilities", [])
    toks = [p.get("id", p.get("token")) for p in probs]
    lp = [[(t.get("id", t.get("token")), t.get("logprob", t.get("prob"))) for t in p.get("top_logprobs", p.get("probs", []))]
          for p in probs]
    return {"text": js.get("content"), "tokens": toks, "top": lp}


def run_with_others(prompt):
    res = {}
    ths = [threading.Thread(target=lambda q=q, i=i: res.__setitem__(i, complete(q))) for i, q in enumerate(OTHERS)]
    for t in ths:
        t.start()
    time.sleep(0.05)
    mine = complete(prompt)
    for t in ths:
        t.join()
    return mine


out = {}
gpu = ["-ngl", "99", "--n-cpu-moe", NCMOE]
p = server(gpu + ["-np", "4"])
try:
    out["gpu_alone_1"] = complete(PROMPT)
    out["gpu_alone_2"] = complete(PROMPT)
    out["gpu_with_3_others"] = run_with_others(PROMPT)
finally:
    p.kill()
    p.wait()
p = server(["-ngl", "0", "-np", "1"])
try:
    out["cpu_alone"] = complete(PROMPT)
finally:
    p.kill()
    p.wait()
ref = out["gpu_alone_1"]
summary = {}
for k, v in out.items():
    first = next((i for i, (a, b) in enumerate(zip(v["top"], ref["top"])) if a != b), None)
    summary[k] = {"same_tokens": v["tokens"] == ref["tokens"], "same_top5_logprobs": v["top"] == ref["top"],
                  "first_step_with_different_logprobs": first, "text": v["text"][:80]}
json.dump({"runs": out, "summary": summary, "n_cpu_moe": NCMOE, "threads": TH}, open(OUT, "w"), indent=1)
for k, v in summary.items():
    print(k, v)
