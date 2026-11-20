"""Server acceptance checks against a running `vahana serve` (standard library client).
usage: server_check.py PORT OUT.json"""
import json
import sys
import time
import urllib.request

PORT, OUT = int(sys.argv[1]), sys.argv[2]
BASE = f"http://127.0.0.1:{PORT}"


def get(path):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=30).read())


def post(path, body, stream=False):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=600)
    if not stream:
        return json.loads(r.read())
    chunks = []
    for line in r:
        line = line.decode().strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[6:]))
    return chunks


for _ in range(180):
    try:
        if get("/health")["status"] == "ok":
            break
    except Exception:  # noqa: BLE001
        time.sleep(5)
res = {"models": get("/v1/models")}
import urllib.error  # noqa: E402
c1 = post("/v1/completions", {"prompt": "The capital of France is", "max_tokens": 16})
c2 = post("/v1/completions", {"prompt": "The capital of France is", "max_tokens": 16})
res["greedy_text"] = c1["choices"][0]["text"]
res["greedy_repeatable"] = c1["choices"][0]["text"] == c2["choices"][0]["text"]
res["usage"] = c1["usage"]
msgs = [{"role": "user", "content": "Name three primary colors."}]
ch = post("/v1/chat/completions", {"messages": msgs, "max_tokens": 40})
res["chat_text"] = ch["choices"][0]["message"]["content"]
st = post("/v1/chat/completions", {"messages": msgs, "max_tokens": 40, "stream": True}, stream=True)
streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in st)
res["stream_equals_nonstream"] = streamed == res["chat_text"]
res["stream_chunks"] = len(st)
s1 = post("/v1/completions", {"prompt": "Once upon a time", "max_tokens": 24, "temperature": 0.8, "seed": 42})
s2 = post("/v1/completions", {"prompt": "Once upon a time", "max_tokens": 24, "temperature": 0.8, "seed": 42})
s3 = post("/v1/completions", {"prompt": "Once upon a time", "max_tokens": 24, "temperature": 0.8, "seed": 43})
res["sampled_same_seed_equal"] = s1["choices"][0]["text"] == s2["choices"][0]["text"]
res["sampled_other_seed_differs"] = s1["choices"][0]["text"] != s3["choices"][0]["text"]
res["sampled_text"] = s1["choices"][0]["text"]
nl = post("/v1/completions", {"prompt": "List:\n1.", "max_tokens": 40, "stop": ["\n"]})
res["stop_newline"] = {"text": nl["choices"][0]["text"], "finish": nl["choices"][0]["finish_reason"]}
free = post("/v1/completions", {"prompt": "The capital of France is", "max_tokens": 16})["choices"][0]["text"]
sp = post("/v1/completions", {"prompt": "The capital of France is", "max_tokens": 16, "stop": ["The capital"]})
res["stop_text"] = sp["choices"][0]["text"]
k = free.find("The capital")
res["stop_respected"] = k >= 0 and sp["choices"][0]["text"] == free[:k] and sp["choices"][0]["finish_reason"] == "stop"
noc = post("/v1/chat/completions", {"messages": msgs, "max_tokens": 4, "unknown_field": 1})
res["chat_ok_extra_fields"] = "choices" in noc
t = time.time()
post("/v1/completions", {"prompt": "Write a long story about a dragon.", "max_tokens": 128})
res["128_tokens_s"] = time.time() - t
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
