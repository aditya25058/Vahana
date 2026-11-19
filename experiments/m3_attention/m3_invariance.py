"""M3: batch / padding / prefill-vs-decode invariance of a whole Vahana model (tiny random models per family).

Per family, one model with: invariant linears and RMSNorms (vahana.invariant), the `vahana` attention function,
CPU experts (VahanaExperts). Prompts of lengths 7 / 19 / 33 / 40.
  ALONE    each prompt alone (batch 1, no padding): prefill logits at every position + 6 greedy steps
  BATCH    the 4 prompts left-padded into one batch: the same quantities per row
  SPLIT    prompt 4: prefill of its first 39 tokens, then one decode step on token 40, vs the full prefill's
           logits at position 40
  SEMANTIC prompt 4 alone vs stock transformers (sdpa, stock experts): max |delta logits| / max |stock|
usage: m3_invariance.py OUT.json [family ...]
"""
import json
import sys
import traceback

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0].replace("m3_attention", "m2_kernel"))
import m2_families as M  # noqa: E402
from vahana import attention as va  # noqa: E402
from vahana import experts as vx  # noqa: E402
from vahana import invariant as inv  # noqa: E402

OUT = sys.argv[1]
FAMS = sys.argv[2:] or M.TIER1 + ["gpt_oss"]
LENS = (7, 19, 33, 40)
NEW = 6
dev = torch.device("cuda")


def gen(m, ids, mask=None):
    with torch.no_grad():
        pos = None if mask is None else (mask.cumsum(-1) - 1).clamp(min=0)   # left padding: real positions
        pre = m(ids, attention_mask=mask, position_ids=pos).logits.float()
        g = m.generate(ids, attention_mask=mask, max_new_tokens=NEW, min_new_tokens=NEW, do_sample=False,
                       output_logits=True, return_dict_in_generate=True, pad_token_id=0)
    return pre, torch.stack([l.float() for l in g.logits], 1), g.sequences[:, ids.shape[1]:]


# pass 1: stock baselines before anything is patched (invariant.install patches nn.Linear process-wide)
STOCK = {}
for fam in FAMS:
    try:
        cfg, stock = M.build(fam)
        rng = torch.Generator().manual_seed(11)
        p = [torch.randint(1, cfg.vocab_size, (L,), generator=rng) for L in LENS][3]
        with torch.no_grad():
            STOCK[fam] = stock(p[None].to(dev)).logits.float()
        del stock
    except Exception as e:  # noqa: BLE001
        STOCK[fam] = e
    torch.cuda.empty_cache()

res = {}
for fam in FAMS:
    r = {}
    try:
        if isinstance(STOCK[fam], Exception):
            raise STOCK[fam]
        stock_logits = STOCK[fam]
        cfg, m = M.build(fam)                      # same seed -> same weights as the stock baseline
        rng = torch.Generator().manual_seed(11)
        prompts = [torch.randint(1, cfg.vocab_size, (L,), generator=rng) for L in LENS]
        r["inv"] = inv.install(m)
        va.use(m)
        vx.install(m, mode="cpu")
        alone = [gen(m, p[None].to(dev)) for p in prompts]
        T = max(LENS)
        ids = torch.zeros(len(LENS), T, dtype=torch.long)
        mask = torch.zeros(len(LENS), T, dtype=torch.long)
        for b, p in enumerate(prompts):
            ids[b, T - len(p):] = p
            mask[b, T - len(p):] = 1
        bpre, blog, bids = gen(m, ids.to(dev), mask.to(dev))
        rows = []
        for b, L in enumerate(LENS):
            a_pre, a_log, a_ids = alone[b]
            rows.append(dict(prefill=torch.equal(a_pre[0], bpre[b, T - L:]), steps=torch.equal(a_log[0], blog[b]),
                             ids=torch.equal(a_ids[0], bids[b])))
        r["batch_rows"] = rows
        r["batch_invariant"] = all(all(x.values()) for x in rows)
        p4 = prompts[3][None].to(dev)
        with torch.no_grad():
            out = m(p4[:, :-1], use_cache=True)
            step = m(p4[:, -1:], past_key_values=out.past_key_values, use_cache=True).logits.float()
        r["split_invariant"] = torch.equal(step[0, -1], alone[3][0][0, -1])
        r["stock_vs_vahana_max_rel"] = (stock_logits[0] - alone[3][0][0]).abs().max().item() / \
            max(stock_logits.abs().max().item(), 1e-9)
    except Exception as e:  # noqa: BLE001
        r["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        r["trace"] = traceback.format_exc()[-1200:]
    res[fam] = r
    print(fam, {k: v for k, v in r.items() if k not in ("trace", "batch_rows")},
          [tuple(int(v) for v in x.values()) for x in r.get("batch_rows", [])], flush=True)
    torch.cuda.empty_cache()
ok = [f for f in FAMS if res[f].get("batch_invariant") and res[f].get("split_invariant")]
res["summary"] = {"families": len(FAMS), "fully_invariant": len(ok), "not": [f for f in FAMS if f not in ok]}
print("SUMMARY", res["summary"], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
