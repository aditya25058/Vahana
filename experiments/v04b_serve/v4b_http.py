"""v0.4-b HTTP bar: 16 concurrent clients on /v1/completions (greedy; half streaming) against the continuous-batching
server; every response's text equals the same request run alone through the one-at-a-time path (v0.1 server path).
usage: v4b_http.py MODEL_DIR PROMPTS.json OUT.json gpu_cache_gb [slots]"""
import json
import sys
import threading
import time
import urllib.request

from transformers import AutoTokenizer

from vahana import loader, server

MD, PJ, OUT, CACHE = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
SLOTS = int(sys.argv[5]) if len(sys.argv) > 5 else 8
PORT = 18431
tok = AutoTokenizer.from_pretrained(MD)
prompts = json.load(open(PJ))[:16]
model, _ = loader.load(MD, device="cuda", gpu_cache_gb=CACHE, log=None)
threading.Thread(target=server.serve, args=(model, tok), kwargs=dict(port=PORT, name="m", slots=SLOTS, max_len=2048),
                 daemon=True).start()
for _ in range(600):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
        break
    except Exception:  # noqa: BLE001
        time.sleep(0.5)
got = [None] * len(prompts)


def client(i):
    body = {"prompt": prompts[i], "max_tokens": 32 + 4 * (i % 5), "stream": i % 2 == 1}
    rq = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(rq, timeout=1200) as r:
        if not body["stream"]:
            got[i] = json.loads(r.read())["choices"][0]["text"]
            return
        text = ""
        for line in r.read().decode().split("\n"):
            if line.startswith("data: ") and line[6:] != "[DONE]":
                text += json.loads(line[6:])["choices"][0].get("text", "")
        got[i] = text


t0 = time.perf_counter()
ths = [threading.Thread(target=client, args=(i,)) for i in range(len(prompts))]
for t in ths:
    t.start()
    time.sleep(0.05 * (t.name[-1] in "02468"))         # staggered arrivals
for t in ths:
    t.join()
wall = time.perf_counter() - t0
solo_state = server._State(model, tok, "m", slots=1)     # the one-at-a-time path
ok = []
for i, p in enumerate(prompts):
    ref, _, _, _ = server._run(solo_state, p, {"max_tokens": 32 + 4 * (i % 5)})
    ok.append(ref == got[i])
res = {"exact": sum(ok), "of": len(ok), "wall_s": round(wall, 2), "streamed": [i % 2 == 1 for i in range(len(prompts))],
       "texts": got}
print("HTTP exact", sum(ok), "/", len(ok), "wall", round(wall, 1), "s", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
