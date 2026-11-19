"""Exact continuous refill with speculative decoding (COLOSSUS_REFILL=1, needs COLOSSUS_INVARIANT=1).

A pool of S slots serves a queue of prompts. Each step is ONE pass over the layers that carries two
row groups:
  - decode rows: [last token + k n-gram drafts] (k = COLOSSUS_SPEC, 0 = plain decode),
  - refill rows: the next chunk of newly admitted prompts (chunked prefill).
Attention runs per group (each row reads only its own slot of the ragged latent cache); the MoE
layer runs ONCE on both groups' tokens concatenated, so every expert's weights cross the link once
per step for decode and prefill together. When a row finishes, its slot is freed; free slots are
refilled from the queue in groups (>= COLOSSUS_REFILL_MIN rows, or what the queue has left).

Exactness: every kernel is invariant to batch companions, padding and query count (Phase 22/23),
and the layer arithmetic below is the decoder layer's own (norm, attention, residual, norm, MLP,
residual), so each prompt's tokens equal its plain greedy invariant decode. Checked per prompt.
"""
from __future__ import annotations

import collections
import os
import time

import torch

from .spec import RaggedLatentCache


class _Drafter:
    """Incremental n-gram index per row: draft = k tokens after the most recent earlier
    occurrence of the longest matching suffix (n = 4..1)."""

    def __init__(self, ctx: list[int], nmax: int = 4):
        self.ctx, self.nmax = ctx, nmax
        self.idx = [dict() for _ in range(nmax + 1)]
        self.upto = 0                       # n-grams with end index < upto are indexed

    def draft(self, k: int) -> list[int]:
        c, L = self.ctx, len(self.ctx)
        while self.upto < L - 1:            # index n-grams ending before the last token
            e = self.upto
            for n in range(1, self.nmax + 1):
                if e - n + 1 >= 0:
                    self.idx[n][tuple(c[e - n + 1:e + 1])] = e + 1   # continuation starts here
            self.upto += 1
        for n in range(min(self.nmax, L), 0, -1):
            s = self.idx[n].get(tuple(c[L - n:]))
            if s is not None and s < L:
                cont = c[s:s + k]
                if cont:
                    return cont + [cont[-1]] * (k - len(cont))
        return [c[-1]] * k


def _driver(model, cache, groups):
    """One pass over all decoder layers for several row groups; returns final-normed hidden per group."""
    m = model.model
    H = model.config.hidden_size
    hs = [m.embed_tokens(g["ids"]) for g in groups]
    for layer in m.layers:
        for gi, g in enumerate(groups):
            x = layer.input_layernorm(hs[gi])
            cache.active_rows = g["rows"]
            a = layer.self_attn(hidden_states=x, attention_mask=None, position_ids=g["pos"],
                                past_key_value=cache, use_cache=True)[0]
            hs[gi] = hs[gi] + a
        cache.active_rows = None
        xs = [layer.post_attention_layernorm(h) for h in hs]
        sizes = [x.shape[0] * x.shape[1] for x in xs]
        y = layer.mlp(torch.cat([x.reshape(-1, H) for x in xs]).unsqueeze(0))
        y = (y[0] if isinstance(y, tuple) else y).reshape(-1, H)
        hs = [h + p.view_as(h) for h, p in zip(hs, torch.split(y, sizes))]
    return [m.norm(h) for h in hs]


def refill_generate(model, tokenizer, prompts: list[str], slots: int, max_new: int, k: int,
                    device, eos_id, prefill_tokens: int = 8192, refill_min: int = 32):
    enc = [tokenizer(p, add_special_tokens=True)["input_ids"] for p in prompts]
    P = len(prompts)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (eos_id or 0)
    cap = max(len(e) for e in enc) + max_new + k + 2
    cache = RaggedLatentCache(torch.zeros(slots, dtype=torch.int64, device=device), cap=cap)
    queue = collections.deque(range(P))
    state: list[dict | None] = [None] * slots
    out: list[list[int] | None] = [None] * P
    pre = None                    # active refill group: rows, padded ids, offset, pads
    lat, t_start = [], time.perf_counter()
    stats = collections.Counter()
    while queue or any(s is not None for s in state):
        free = [r for r in range(slots) if state[r] is None]
        if pre is None and queue and free and (len(free) >= min(refill_min, len(queue))
                                               or all(s is None for s in state)):
            rows = free[:len(queue)]
            ps = [queue.popleft() for _ in rows]
            L = max(len(enc[p]) for p in ps)
            pads = [L - len(enc[p]) for p in ps]
            ids = torch.tensor([[pad_id] * d + enc[p] for p, d in zip(ps, pads)], device=device)
            cache.reset_rows(rows, pads)
            for r, p in zip(rows, ps):
                state[r] = {"p": p, "phase": "prefill", "new": []}
            pre = {"rows": rows, "ids": ids, "off": 0, "pads": pads, "L": L,
                   "chunk": max(16, min(L, prefill_tokens // len(rows)))}
            stats["refills"] += 1
        groups, gmeta = [], []
        drows = [r for r in range(slots) if state[r] is not None and state[r]["phase"] == "decode"]
        if drows:
            drafts = [state[r]["drafter"].draft(k) if k > 0 else [] for r in drows]
            ids = torch.tensor([[state[r]["new"][-1]] + d for r, d in zip(drows, drafts)], device=device)
            pos = torch.tensor([[len(enc[state[r]["p"]]) + len(state[r]["new"]) - 1 + i for i in range(k + 1)]
                                for r in drows], device=device)
            groups.append({"ids": ids, "pos": pos, "rows": torch.tensor(drows, device=device)})
            gmeta.append(("decode", drows, drafts))
        if pre is not None:
            o, c = pre["off"], pre["chunk"]
            ids = pre["ids"][:, o:o + c]
            cols = torch.arange(o, o + ids.shape[1], device=device)
            pos = (cols.unsqueeze(0) - torch.tensor(pre["pads"], device=device).unsqueeze(1)).clamp(min=0)
            groups.append({"ids": ids, "pos": pos, "rows": torch.tensor(pre["rows"], device=device)})
            gmeta.append(("prefill", pre["rows"], None))
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            hs = _driver(model, cache, groups)
            res = []
            for (kind, rows, _), h in zip(gmeta, hs):
                if kind == "decode":
                    res.append(model.lm_head(h.reshape(-1, h.shape[-1])).argmax(-1).view(len(rows), -1).tolist())
                elif pre["off"] + h.shape[1] >= pre["L"]:
                    res.append(model.lm_head(h[:, -1, :]).argmax(-1).tolist())
                else:
                    res.append(None)
        torch.cuda.synchronize(device)
        lat.append(time.perf_counter() - t0)
        stats["steps"] += 1
        for (kind, rows, drafts), g in zip(gmeta, res):
            if kind == "decode":
                commit = []
                for j, r in enumerate(rows):
                    st, d = state[r], drafts[j]
                    a = 0
                    while a < k and d[a] == g[j][a]:
                        a += 1
                    toks = (d[:a] + [g[j][a]])[:max_new - len(st["new"])]
                    if eos_id is not None and eos_id in toks:
                        toks = toks[:toks.index(eos_id) + 1]
                    commit.append(len(toks))
                    st["new"].extend(toks)
                    st["drafter"].ctx.extend(toks)
                    stats["accepted"] += len(toks)
                    stats["row_steps"] += 1
                    if len(st["new"]) >= max_new or (eos_id is not None and toks[-1] == eos_id):
                        out[st["p"]] = st["new"]
                        state[r] = None
                cache.commit_rows(rows, commit)
            else:
                n = pre["ids"][:, pre["off"]:pre["off"] + pre["chunk"]].shape[1]
                cache.commit_rows(rows, [n] * len(rows))
                stats["prefill_tokens"] += n * len(rows)
                pre["off"] += n
                if g is not None:                       # prompt finished: first generated token
                    for j, r in enumerate(rows):
                        st = state[r]
                        st["new"] = [g[j]]
                        st["phase"] = "decode"
                        st["drafter"] = _Drafter(list(enc[st["p"]]) + [g[j]])
                        if max_new <= 1 or (eos_id is not None and g[j] == eos_id):
                            out[st["p"]] = st["new"]
                            state[r] = None
                    pre = None
    wall = time.perf_counter() - t_start
    stats = dict(stats)
    stats["tokens_per_row_step"] = stats.get("accepted", 0) / max(1, stats.get("row_steps", 1))
    return out, lat, wall, stats


def enabled() -> bool:
    return os.environ.get("COLOSSUS_REFILL", "0") == "1"
