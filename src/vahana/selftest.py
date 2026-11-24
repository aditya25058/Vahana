"""vahana selftest: does this machine give the same bits as every other machine? (P3)

For each MoE family, a tiny model is built from the family's own transformers config with weights drawn from numpy's
RandomState (identical on every OS, CPU and numpy version), a batch of 2 left-padded prompts is decoded greedily
for 6 steps through Vahana, and the logits and ids are hashed. The hashes must equal golden_tiny.json, which was
recorded on an H100 (GPU and CPU-only agreeing). Equal hashes mean this machine produces the same tokens and the
same logits, bit for bit, as that H100.

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
    ids, am = prompts(cfg.vocab_size)
    r = decode.generate(model, ids.to(device), am.to(device), max_new_tokens=STEPS, graph=device.type == "cuda",
                        return_logits=True)
    h = hashlib.sha256()
    h.update(r["logits"].float().cpu().contiguous().numpy().tobytes())
    h.update(r["ids"].cpu().to(torch.int64).contiguous().numpy().tobytes())
    return h.hexdigest()[:24]


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
    gold = {} if a.write else json.load(open(a.golden))
    info = {"device": str(dev), "isa": hybrid.ext().isa(), "torch": torch.__version__,
            "transformers": transformers.__version__, "platform": sys.platform}
    print("vahana selftest:", info, flush=True)
    if not a.write and gold.get("_transformers") and gold["_transformers"] != transformers.__version__:
        print(f"  note: golden hashes were recorded with transformers {gold['_transformers']}; model code from "
              f"another version may legitimately differ", flush=True)
    res, ok = {}, 0
    for f in fams:
        t = time.time()
        try:
            res[f] = run(f, dev, a.gpu_cache)
        except Exception as e:  # noqa: BLE001
            res[f] = f"error: {type(e).__name__}: {str(e)[:200]}"
        same = gold.get(f) == res[f]
        ok += same
        mark = "" if a.write else ("  same bits" if same else f"  DIFFERENT (golden {gold.get(f)})")
        print(f"  {f:18s} {res[f]}{mark}  {time.time() - t:.1f}s", flush=True)
    if a.write:
        out = {"_transformers": transformers.__version__, "_recorded_on": info, **res}
        json.dump(out, open(a.write, "w"), indent=1)
        print("wrote", a.write)
        return 0
    print(f"vahana selftest: {ok}/{len(fams)} families give the same bits as the reference machine", flush=True)
    return 0 if ok == len(fams) else 1


if __name__ == "__main__":
    sys.exit(main())
