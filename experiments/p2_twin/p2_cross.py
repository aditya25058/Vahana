"""P2 CROSS: this machine's CPU-only Vahana == the H100's GPU Vahana, bit for bit, per family.
Input: p2_families.pt from the H100 (exact tiny-model weights, configs, prompts, GPU ids and logits).
usage: p2_cross.py p2_families.pt OUT.json
"""
import json
import platform
import sys
import traceback

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from vahana import _triton, decode, hybrid, loader

data = torch.load(sys.argv[1], weights_only=False)
res = {"machine": platform.processor() or platform.machine(), "torch": torch.__version__,
       "isa": hybrid.ext().isa(), "triton": _triton.HAVE_TRITON, "cuda": torch.cuda.is_available()}
try:
    res["cpu"] = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
except Exception:  # noqa: BLE001
    pass
fams = [f for f in data if not f.startswith("_")]
for fam in fams:
    d = data[fam]
    out = {}
    try:
        cls = type(AutoConfig.for_model(d["model_type"]))
        cfg = cls.from_dict(d["config"])
        torch.manual_seed(0)
        m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
        missing, unexpected = m.load_state_dict(d["state"], strict=False)
        out["missing"], out["unexpected"] = list(missing), list(unexpected)
        model, info = loader.prepare(m)
        r = decode.generate(model, d["input_ids"], d["attention_mask"], max_new_tokens=10, graph=False,
                            return_logits=True)
        out["ids_equal"] = torch.equal(r["ids"], d["ids_gpu"])
        lg, lh = r["logits"], d["logits_gpu"]
        out["logits_equal"] = lg.shape == lh.shape and torch.equal(lg.view(torch.int32), lh.view(torch.int32))
        out["cross"] = out["ids_equal"] and out["logits_equal"] and not missing
        if not out["logits_equal"] and lg.shape == lh.shape:
            out["max_abs"] = float((lg - lh).abs().max())
        out["step_ms"] = sorted(r["step_ms"])[len(r["step_ms"]) // 2]
        del model, m
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        out["trace"] = traceback.format_exc()[-1500:]
    res[fam] = out
    print(fam, {k: v for k, v in out.items() if k != "trace"}, flush=True)
ok = [f for f in fams if res[f].get("cross")]
res["summary"] = {"families": len(fams), "cross_equal": len(ok), "not": [f for f in fams if f not in ok]}
print("SUMMARY", res["summary"], flush=True)
json.dump(res, open(sys.argv[2], "w"), indent=1)
