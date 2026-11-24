"""v0.5 DOCTOR bar: `vahana doctor --measure`'s number vs a full run on the same loaded model: 3 M5 prompts x 32 tokens,
mean decode tok/s. The measure must be within +-20%.
usage: v5_doctor.py MODEL_DIR PROMPTS.json OUT.json"""
import json
import sys

from vahana import decode, doctor

MD, PJ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
m = doctor.measure(MD, 16, keep=True)
model, tok = m.pop("model"), m.pop("tokenizer")
print("measure", m, flush=True)
rates = []
for p in json.load(open(PJ))[:3]:
    ids = tok(p, return_tensors="pt").input_ids.to(model._vahana_device)
    r = decode.generate(model, ids, max_new_tokens=32)
    rates.append(len(r["step_ms"]) / (sum(r["step_ms"]) / 1e3))
full = sum(rates) / len(rates)
res = {"measure_tok_s": m["tok_s"], "full_tok_s": full, "ratio": m["tok_s"] / full, "rates": rates, "measure": m}
print("DOCTOR", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k != "measure"}, flush=True)
json.dump(res, open(OUT, "w"), indent=1)
