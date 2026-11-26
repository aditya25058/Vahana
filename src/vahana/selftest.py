"""vahana selftest: does this machine give the same bits as every other machine? (P3)

For each MoE family, a tiny model is built from the family's own transformers config with weights drawn from numpy's
RandomState (identical on every OS, CPU and numpy version), a batch of 2 left-padded prompts is decoded greedily
for 6 steps through Vahana, and the logits and ids are hashed. The hashes must equal the references in
golden_tiny.json for the installed transformers version (v1.1: transformers sometimes changes a family's model code,
which changes that family's outputs on every machine alike, so references are kept per version). Equal hashes mean
this machine produces the same tokens and the same logits, bit for bit, as the reference machines.

  python -m vahana.selftest [--device cpu|cuda] [--families a,b] [--write golden.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zlib

FAMILIES = ["mixtral", "qwen2_moe", "qwen3_moe", "olmoe", "deepseek_v2", "deepseek_v3", "glm4_moe", "glm4_moe_lite",
            "minimax", "minimax_m2", "hunyuan_v1_moe", "ernie4_5_moe", "cohere2_moe", "exaone_moe", "afmoe", "phimoe",
            "qwen3_5_moe_text", "gpt_oss"]
SMALL = dict(hidden_size=256, intermediate_size=512, moe_intermediate_size=128, num_hidden_layers=4,
             num_attention_heads=4, num_key_value_heads=2, vocab_size=512, num_local_experts=8,
             num_experts=8, n_routed_experts=8, num_experts_per_tok=2, max_position_embeddings=256,
             kv_lora_rank=64, q_lora_rank=96, qk_nope_head_dim=32, qk_rope_head_dim=32, v_head_dim=32,
             n_group=2, topk_group=1, first_k_dense_replace=0, shared_expert_intermediate_size=256,
             n_shared_experts=1, moe_layer_freq=1, decoder_sparse_step=1, mlp_only_layers=[],
             expert_intermediate_size=128, num_shared_experts=1)
STEPS, LENS, PAD = 6, (11, 24), 24
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_tiny.json")


def tiny_config(fam):
    from transformers import AutoConfig
    base = AutoConfig.for_model(fam)
    kw = {k: v for k, v in SMALL.items() if hasattr(base, k)}
    if hasattr(base, "kv_lora_rank"):
        kw["num_key_value_heads"] = kw["num_attention_heads"]
    if hasattr(base, "layer_types") and isinstance(getattr(base, "layer_types", None), list):
        kw["layer_types"] = None                       # rebuilt from num_hidden_layers by the config
    cfg = AutoConfig.for_model(fam, **kw)
    if getattr(cfg, "head_dim", 0) is None:
        cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads
    return cfg


def _bf16(a):
    import numpy as np
    import torch
    u = a.astype(np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return torch.from_numpy(r.view(np.int16).copy()).view(torch.bfloat16)


def build(fam):
    """The family's model with numpy-seeded weights: the same bits on every machine."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    cfg = tiny_config(fam)
    with torch.device("cpu"):
        m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        for name, p in sorted(m.named_parameters()):
            rs = np.random.RandomState(zlib.crc32(name.encode()) & 0x7FFFFFFF)
            a = rs.standard_normal(tuple(p.shape)).astype(np.float32) * 0.02
            if p.dim() == 1 and "norm" in name and "bias" not in name:
                a = a + 1.0
            p.copy_(_bf16(a).to(p.dtype))
    return cfg, m


def prompts(vocab):
    import numpy as np
    import torch
    rs = np.random.RandomState(3)
    ids = torch.zeros(2, PAD, dtype=torch.long)
    am = torch.zeros(2, PAD, dtype=torch.long)
    for b, L in enumerate(LENS):
        ids[b, PAD - L:] = torch.from_numpy(rs.randint(1, vocab, L).astype(np.int64))
        am[b, PAD - L:] = 1
    return ids, am


def run(fam, device, cache_frac=None):
    import torch
    from . import decode, loader
    cfg, m = build(fam)
    m = m.to(device)
    model, _ = loader.prepare(m, gpu_cache_frac=cache_frac)
    audit = _audit(model)
    ids, am = prompts(cfg.vocab_size)
    r = decode.generate(model, ids.to(device), am.to(device), max_new_tokens=STEPS, graph=device.type == "cuda",
                        return_logits=True)
    h = hashlib.sha256()
    h.update(r["logits"].float().cpu().contiguous().numpy().tobytes())
    h.update(r["ids"].cpu().to(torch.int64).contiguous().numpy().tobytes())
    return h.hexdigest()[:24], audit


def _audit(model) -> dict:
    """v1.1: run the model's twin forward with TwinMode's audit on. Any float op that is neither handled by Vahana's
    exact math nor known to be exact is recorded: such an op may round differently on another CPU or GPU even when
    this machine's hash matches (transformers 5.17 added a LAPACK triangular solve to Qwen3.5 that did exactly that)."""
    from . import tmath
    audit: dict = {}
    orig = getattr(model, "_vahana_twin_forward", None)
    if orig is not None:
        def fwd(*a, **k):
            with tmath.TwinMode(audit):
                return orig(*a, **k)
        model.forward = fwd
    return audit


def _ver(v: str) -> tuple:
    """'5.14.1' / '5.14.0.dev0' -> (5, 14, 1); missing parts are 0."""
    import re
    out = []
    for p in v.split(".")[:3]:
        m = re.match(r"\d+", p)                          # leading digits only: "0rc1" -> 0, "0dev" -> 0
        out.append(int(m.group()) if m else 0)
    return tuple(out + [0] * (3 - len(out)))


def expected_hashes(golden: dict, version: str) -> tuple[dict, str | None]:
    """Reference hash per family for this transformers version (v1.1: per-version references).

    golden_tiny.json holds, per family, a list of {"since": version, "hash": ...}: transformers sometimes changes a
    family's model code, which legitimately changes its outputs (on every machine alike: CPU and GPU still agree).
    The entry with the newest "since" not after the installed version applies. "_validated" lists the transformers
    releases checked in CI; outside that range a mismatch may be a new model-code change rather than a fault.
    The flat 1.0 format ({family: hash}) is still read."""
    fams = golden.get("families")
    if fams is None:                                    # 1.0 format
        note = None
        if golden.get("_transformers") and golden["_transformers"] != version:
            note = (f"golden hashes were recorded with transformers {golden['_transformers']}; model code from "
                    f"another version may legitimately differ")
        return {k: v for k, v in golden.items() if not k.startswith("_")}, note
    v = _ver(version)
    exp = {}
    for fam, entries in fams.items():
        ok = [e for e in entries if _ver(e["since"]) <= v]
        if ok:
            exp[fam] = max(ok, key=lambda e: _ver(e["since"]))["hash"]
    val = sorted((_ver(x) for x in golden.get("_validated", [])))
    note = None
    if val and (v[:2] > val[-1][:2] or v[:2] < val[0][:2]):
        lo, hi = golden["_validated"][0], golden["_validated"][-1]
        note = (f"transformers {version} is outside the validated range ({lo} to {hi}); a family that differs may "
                f"have new model code in this release rather than a fault in Vahana")
    return exp, note


def main(argv=None):
    ap = argparse.ArgumentParser("vahana selftest")
    ap.add_argument("--device", default=None)
    ap.add_argument("--families", default=None)
    ap.add_argument("--write", default=None, help="record hashes to this file instead of checking")
    ap.add_argument("--golden", default=GOLDEN)
    ap.add_argument("--gpu-cache", type=float, default=None,
                    help="cache this fraction of every layer's experts on the GPU (the bits must not change)")
    a = ap.parse_args(argv)
    import torch
    import transformers
    from . import hybrid
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fams = a.families.split(",") if a.families else FAMILIES
    info = {"device": str(dev), "isa": hybrid.ext().isa(), "torch": torch.__version__,
            "transformers": transformers.__version__, "platform": sys.platform}
    print("vahana selftest:", info, flush=True)
    gold = {}
    if not a.write:
        gold, note = expected_hashes(json.load(open(a.golden)), transformers.__version__)
        if note:
            print("  note:", note, flush=True)
    res, ok, inexact = {}, 0, {}
    for f in fams:
        t = time.time()
        try:
            res[f], aud = run(f, dev, a.gpu_cache)
            if aud:
                inexact[f] = aud
        except Exception as e:  # noqa: BLE001
            res[f] = f"error: {type(e).__name__}: {str(e)[:200]}"
        same = gold.get(f) == res[f] and f not in inexact
        ok += same
        mark = "" if a.write else ("  same bits" if same else f"  DIFFERENT (golden {gold.get(f)})"
                                   if f not in inexact or gold.get(f) != res[f] else "  matches here, but")
        if f in inexact:
            mark += f"  NOT EXACT: uses ops outside Vahana's exact math {sorted(inexact[f])}"
        print(f"  {f:18s} {res[f]}{mark}  {time.time() - t:.1f}s", flush=True)
    if inexact and a.write:
        print("refusing to record references: some families use ops outside Vahana's exact math", inexact)
        return 1
    if a.write:
        out = {"_transformers": transformers.__version__, "_recorded_on": info, **res}
        json.dump(out, open(a.write, "w"), indent=1)
        print("wrote", a.write)
        return 0
    print(f"vahana selftest: {ok}/{len(fams)} families give the same bits as the reference machine", flush=True)
    return 0 if ok == len(fams) else 1


if __name__ == "__main__":
    sys.exit(main())
