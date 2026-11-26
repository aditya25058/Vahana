"""The Vahana side of determinism_llamacpp.py: the same prompt, greedy, 32 tokens, (a) twice alone, (b) in a batch
with the same three other prompts, (c) CPU-only (run with --device cpu in a separate process). Prints a hash of the
generated ids and of every step's fp32 logits; equal hashes mean bit-identical output.

  python determinism_vahana.py MODEL_DIR OUT.json [cuda|cpu]
"""
import hashlib
import json
import sys
from argparse import Namespace

from vahana import cli

model_dir, out, device = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else None)
ns = Namespace(model=model_dir, gpu_budget=None, reserve=1.5, threads=None, keep_numa_balancing=False, device=device,
               gpu_cache="auto")
setup = cli._setup(ns)
model, tok, torch = cli._load(ns, setup)
from vahana import batch, decode  # noqa: E402

PROMPT = ("The history of the Roman Empire spans more than a thousand years. Historians divide it into the Kingdom, "
          "the Republic and the Empire. Explain in three sentences why the Empire split into East and West:")
OTHERS = ["Write a short poem about the sea.", "List five prime numbers and explain why they are prime.",
          "Describe how a bicycle works in simple terms."]
dev = model._vahana_device
ids = tok(PROMPT, return_tensors="pt").input_ids.to(dev)
h = lambda t: hashlib.sha256(t.float().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]
res = {}
for name in ("alone_1", "alone_2"):
    r = decode.generate(model, ids, max_new_tokens=32, return_logits=True)
    res[name] = {"ids": h(r["ids"]), "logits": h(r["logits"]), "text": tok.decode(r["ids"][0])[:80]}
if dev.type == "cuda":
    rows = [ids[0].cpu()] + [tok(q, return_tensors="pt").input_ids[0] for q in OTHERS]
    rb = batch.generate_batch(model, [x.to(dev) for x in rows], 32, return_logits=True)
    o = rb["outputs"][0]
    i0 = torch.as_tensor(o["ids"], dtype=torch.long).view(1, -1)
    l0 = torch.as_tensor(o["logits"]).float().view(1, -1, o["logits"].shape[-1])
    res["in_batch_of_4"] = {"ids": h(i0), "logits": h(l0), "text": tok.decode(i0[0])[:80],
                            "batched": rb.get("batched")}
json.dump({"device": str(dev), **res}, open(out, "w"), indent=1)
for k, v in res.items():
    print(k, v)
