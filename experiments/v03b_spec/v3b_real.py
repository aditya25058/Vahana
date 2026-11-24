"""v0.3-b real checkpoints: speculative decoding (prompt lookup, k = 4) vs plain greedy, exactness and speed.

Per prompt (4 copy-heavy + 4 from the M5 set), 64 new tokens, graph decode: plain decode.generate, then
speculative.generate_spec, then both again (2 rounds, alternated). Exact: ids and every token's logit vector equal.
Speed: decode tok/s = (tokens - 1) / decode time (prefill excluded), per run.
usage: v3b_real.py MODEL_DIR M5_PROMPTS.json OUT.json [gpu_cache_gb|-] [min_draft]
"""
import json
import sys

import torch
from transformers import AutoTokenizer

from vahana import decode, loader, speculative

MD, PJ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
CACHE = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "-" else None
MIN_DRAFT = int(sys.argv[5]) if len(sys.argv) > 5 else 1
NEW = 64
CODE = '''def summarize(orders):
    total = 0
    count = 0
    for order in orders:
        if order["status"] == "paid":
            total += order["amount"]
            count += 1
    average = total / count if count else 0
    return {"total": total, "count": count, "average": average}
'''
COPY = [
    "Here is a Python function:\n\n" + CODE + "\nThe same function with the variable `total` renamed to `revenue`, "
    "everything else unchanged:\n\n",
    "Fix the spelling mistakes in this paragraph and write out the whole corrected paragraph.\n\nParagraph: The "
    "comittee recieved the reports on wensday and decided to postpone the meeting untill the folowing week, because "
    "several members were not availible.\n\nCorrected paragraph: The",
    "Convert this list to a numbered list, keeping every item exactly as written:\n- apples from the north farm\n"
    "- pears from the south farm\n- plums from the east farm\n- cherries from the west farm\n\nNumbered list:\n1.",
    "Continue the table with the same format for the next four years.\n\n| year | city | population |\n|---|---|---|\n"
    "| 2001 | Springfield | 104,000 |\n| 2002 | Springfield | 105,100 |\n| 2003 | Springfield | 106,250 |\n",
]
tok = AutoTokenizer.from_pretrained(MD)
prompts = [("copy", p) for p in COPY] + [("m5", p) for p in json.load(open(PJ))[:4]]
model, info = loader.load(MD, device="cuda", gpu_cache_gb=CACHE, log=print)
res = {"model": MD.rstrip("/").rsplit("/", 1)[-1], "gpu_cache": info.get("gpu_cache"), "min_draft": MIN_DRAFT, "prompts": []}


def tps(r):
    st = r["step_ms"]
    return (r["ids"].shape[1] - 1) / (sum(st) / 1e3) if st else None


for kind, p in prompts:
    ids = tok(p, return_tensors="pt").input_ids.cuda()
    row = {"kind": kind, "plain_tok_s": [], "spec_tok_s": []}
    for rnd in range(2):
        a = decode.generate(model, ids, None, max_new_tokens=NEW, graph=True, return_logits=True)
        b = speculative.generate_spec(model, ids, max_new_tokens=NEW, k=4, graph=True, return_logits=True,
                                      min_draft=MIN_DRAFT)
        if rnd == 0:
            row["ids_equal"] = a["ids"][0].tolist() == b["ids"][0].tolist()
            row["logits_equal"] = a["logits"].shape == b["logits"].shape and torch.equal(
                a["logits"].view(torch.int32), b["logits"].view(torch.int32))
            row["tokens_per_step"] = round(b["tokens_per_step"], 2)
            row["graph"] = [a["graph_used"], b["graph_used"]]
            row["text"] = tok.decode(a["ids"][0])[:160]
        row["plain_tok_s"].append(round(tps(a), 2))
        row["spec_tok_s"].append(round(tps(b), 2))
    res["prompts"].append(row)
    print(kind, {k: v for k, v in row.items() if k != "text"}, repr(row["text"][:70]), flush=True)
for kind in ("copy", "m5"):
    rs = [r for r in res["prompts"] if r["kind"] == kind]
    pl = sum(sum(r["plain_tok_s"]) for r in rs) / (2 * len(rs))
    sp = sum(sum(r["spec_tok_s"]) for r in rs) / (2 * len(rs))
    res[f"{kind}_speedup"] = round(sp / pl, 3)
    res[f"{kind}_plain_tok_s"], res[f"{kind}_spec_tok_s"] = round(pl, 2), round(sp, 2)
res["exact"] = all(r["ids_equal"] and r["logits_equal"] for r in res["prompts"])
print("SUMMARY", {k: v for k, v in res.items() if k not in ("prompts",)}, flush=True)
json.dump(res, open(OUT, "w"), indent=1)
