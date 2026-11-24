"""v0.3-e: exact batched decoding. Several requests share each decode step; every request gets exactly the bits it
gets when decoded alone.

Why it is exact: every Vahana kernel computes each row (token) separately with a fixed operation order (twin
linears, experts, glue), and attention visits each query's own keys in blocks anchored at its first allowed key,
so left padding and the other rows of the batch never move a rounding. Recurrent / linear-attention layers would
run over the padding, so models with them decode each request alone.
"""
from __future__ import annotations

import torch

from . import decode


def _recurrent(model) -> bool:
    kinds = set(getattr(model.config.get_text_config(), "layer_types", None) or [])
    return bool(kinds - {"full_attention", "sliding_attention", "attention"})


def _eos_set(eos_token_id):
    if eos_token_id is None:
        return set()
    return set(eos_token_id) if isinstance(eos_token_id, (list, tuple, set)) else {int(eos_token_id)}


def generate_batch(model, prompts, max_new_tokens: int = 64, eos_token_id=None, graph: bool = True,
                   return_logits: bool = False, pad_id: int = 0):
    """prompts: list of token-id sequences (lists or 1-D tensors), any lengths. Greedy.
    Returns {"outputs": [{"ids": [new tokens, through the first EOS], "logits": [n, V] fp32 (if asked)}, ...],
             "step_ms": [...], "batched": bool, "graph_used": bool}."""
    dev = next(p.device for p in model.parameters() if p.device.type == "cuda") \
        if any(p.is_cuda for p in model.parameters()) else torch.device("cpu")
    ps = [torch.as_tensor(p, dtype=torch.long).flatten() for p in prompts]
    eos = _eos_set(eos_token_id)
    if not ps:
        return {"outputs": [], "step_ms": [], "batched": False, "graph_used": False}

    def trim(ids, lg):
        n = len(ids)
        for i, t in enumerate(ids):
            if t in eos:
                n = i + 1
                break
        o = {"ids": ids[:n]}
        if lg is not None:
            o["logits"] = lg[:n]
        return o

    if _recurrent(model) or len(ps) == 1:
        outs, steps, gu = [], [], True
        for p in ps:
            r = decode.generate(model, p.view(1, -1).to(dev), None, max_new_tokens=max_new_tokens, graph=graph,
                                eos_token_id=eos_token_id, return_logits=return_logits)
            outs.append(trim(r["ids"][0].tolist(), r["logits"][0] if return_logits else None))
            steps += r["step_ms"]
            gu = gu and r["graph_used"]
        return {"outputs": outs, "step_ms": steps, "batched": False, "graph_used": gu}
    B, T = len(ps), max(len(p) for p in ps)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    am = torch.zeros(B, T, dtype=torch.long)
    for b, p in enumerate(ps):
        ids[b, T - len(p):] = p
        am[b, T - len(p):] = 1
    done = torch.zeros(B, dtype=torch.bool)

    def on_token(tok):
        if not eos:
            return False
        t = tok.view(-1).cpu()
        for e in eos:
            done.logical_or_(t == e)
        return bool(done.all())

    r = decode.generate(model, ids.to(dev), am.to(dev), max_new_tokens=max_new_tokens, graph=graph,
                        return_logits=return_logits, on_token=on_token if eos else None)
    outs = [trim(r["ids"][b].tolist(), r["logits"][b] if return_logits else None) for b in range(B)]
    return {"outputs": outs, "step_ms": r["step_ms"], "batched": True, "graph_used": r["graph_used"]}
