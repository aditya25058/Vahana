"""Exact speculative decoding on the invariant engine (VAHANA_SPEC=k, needs VAHANA_INVARIANT=1).

Each step feeds [last token, k drafted tokens] per row and verifies them with ONE forward. In the
offload regime that forward costs about the same link time as a 1-token step. Greedy acceptance:
the longest draft prefix equal to the target's own argmax, plus the target's next token.

Exactness: the invariant kernels give every query the same bits whether it is decoded alone or
inside a k+1 verify window, with any batch companions (Phase 22a/22b), so the accepted tokens and
the KV they leave behind are bit-identical to plain greedy decoding in invariant mode. The test is
direct: spec output == non-spec invariant output, token for token.

RaggedLatentCache: preallocated latent / rotated-key buffers with per-row `first` (first real key)
and `length` (next write index). Verify windows are written at each row's own offset; rejected
entries are simply overwritten next step. No shifting, no masks.
Drafts: n-gram ("prompt lookup") from the row's own context, no draft model.
"""
from __future__ import annotations

import time

import torch

from .mla_cache import LatentKVCache


class RaggedLatentCache(LatentKVCache):
    def __init__(self, first: torch.Tensor, cap: int = 0):
        super().__init__()
        self.first = first.to(torch.int32)
        self.B = int(first.shape[0])
        self.length = torch.zeros(self.B, dtype=torch.int64, device=first.device)
        self.len_host = [0] * self.B
        self.cap = int(cap)
        self._buf_lat: dict[int, torch.Tensor] = {}
        self._buf_kpe: dict[int, torch.Tensor] = {}
        self.active_rows: torch.Tensor | None = None   # slot subset for the next attention call

    def _ensure(self, layer: int, latent: torch.Tensor, kpe: torch.Tensor, need: int) -> None:
        buf = self._buf_lat.get(layer)
        if buf is not None and buf.shape[1] >= need:
            return
        # Exact preallocation (cap = prompt + max_new + k + 1) avoids growth; if ever exceeded,
        # grow by a small step only (1.5x growth cost ~20 GB extra at B=768).
        cap = max(need, self.cap) if buf is None else need + 64
        lat = latent.new_zeros((self.B, cap, latent.shape[-1]))
        kp = kpe.new_zeros((self.B, cap, kpe.shape[-1]))
        if buf is not None:
            lat[:, :buf.shape[1]] = buf
            kp[:, :buf.shape[1]] = self._buf_kpe[layer]
        self._buf_lat[layer], self._buf_kpe[layer] = lat, kp

    def append_ragged(self, layer: int, latent: torch.Tensor, kpe: torch.Tensor):
        """latent [B, Q, r], kpe [B, 1, Q, rope] written at rows' own offsets (not committed)."""
        Q = latent.shape[1]
        rows = self.active_rows
        if rows is None:
            rows = torch.arange(self.B, device=latent.device)
        self._ensure(layer, latent, kpe.squeeze(1), max(self.len_host) + Q)
        idx = self.length[rows].unsqueeze(1) + torch.arange(Q, device=latent.device)   # [Bg, Q]
        r2 = rows.unsqueeze(1).expand(-1, Q)
        self._buf_lat[layer][r2, idx] = latent
        self._buf_kpe[layer][r2, idx] = kpe.squeeze(1)
        return self._buf_lat[layer], self._buf_kpe[layer], self.first, self.length, rows

    def reset_rows(self, rows: list[int], first: list[int]) -> None:
        """Give slots to new sequences: first real key index, empty length (old data is overwritten)."""
        for r, f in zip(rows, first):
            self.len_host[r] = 0
        self.length = torch.tensor(self.len_host, dtype=torch.int64, device=self.length.device)
        idx = torch.tensor(rows, device=self.first.device)
        self.first[idx] = torch.tensor(first, dtype=torch.int32, device=self.first.device)

    def commit_rows(self, rows: list[int], n: list[int]) -> None:
        for r, a in zip(rows, n):
            self.len_host[r] += a
        self.length = torch.tensor(self.len_host, dtype=torch.int64, device=self.length.device)

    def commit(self, n: list[int]) -> None:
        self.len_host = [a + b for a, b in zip(self.len_host, n)]
        self.length = torch.tensor(self.len_host, dtype=torch.int64, device=self.length.device)

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        return max(self.len_host) if self.len_host else 0

    def nbytes(self) -> int:
        return sum(t.nbytes for d in (self._buf_lat, self._buf_kpe) for t in d.values())


def draft_ngram(ctx: list[int], k: int, nmax: int = 4) -> list[int]:
    """k tokens following the most recent earlier occurrence of the longest matching suffix."""
    L = len(ctx)
    for n in range(min(nmax, L - 1), 0, -1):
        suf = ctx[L - n:]
        for s in range(L - n - 1, -1, -1):
            if ctx[s:s + n] == suf:
                cont = ctx[s + n:s + n + k]
                if cont:
                    return cont + [cont[-1]] * (k - len(cont))
    return [ctx[-1]] * k


def spec_decode(model, past: RaggedLatentCache, prompts_real: list[list[int]], first_tok: torch.Tensor,
                max_new_tokens: int, k: int, eos_id: int | None, device, keep_setter):
    """Returns (new_tokens per row, latencies per step, stats). first_tok: [B, 1] prefill token."""
    B = first_tok.shape[0]
    new = [[int(t)] for t in first_tok[:, 0].tolist()]
    done = [(eos_id is not None and n[0] == eos_id) or max_new_tokens <= 1 for n in new]
    latencies, accepted_hist = [], []
    keep_setter(k + 1)
    try:
        while not all(done):
            drafts = [draft_ngram(prompts_real[b] + new[b], k) for b in range(B)]
            inp = torch.tensor([[new[b][-1]] + drafts[b] for b in range(B)], device=device)
            pos = torch.tensor([[len(prompts_real[b]) + len(new[b]) - 1 + i for i in range(k + 1)]
                                for b in range(B)], device=device)
            mask = torch.ones((B, past.get_seq_length() + k + 1), dtype=torch.long, device=device)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(input_ids=inp, attention_mask=mask, position_ids=pos,
                            past_key_values=past, use_cache=True)
                g = out.logits[:, -(k + 1):, :].argmax(dim=-1).tolist()      # one host sync
            torch.cuda.synchronize(device)
            latencies.append(time.perf_counter() - t0)
            commit = []
            for b in range(B):
                if done[b]:
                    commit.append(0)
                    continue
                a = 0
                while a < k and drafts[b][a] == g[b][a]:
                    a += 1
                toks = drafts[b][:a] + [g[b][a]]
                room = max_new_tokens - len(new[b])
                toks = toks[:room]
                if eos_id is not None and eos_id in toks:
                    toks = toks[:toks.index(eos_id) + 1]
                # KV entries kept: the fed last token + accepted drafts that precede a kept token.
                commit.append(len(toks))
                new[b].extend(toks)
                accepted_hist.append(len(toks))
                if len(new[b]) >= max_new_tokens or (eos_id is not None and toks and toks[-1] == eos_id):
                    done[b] = True
            past.commit(commit)
    finally:
        keep_setter(1)
    stats = {"spec_k": k, "steps": len(latencies),
             "tokens_per_row_step": (sum(accepted_hist) / len(accepted_hist)) if accepted_hist else 0.0}
    return new, latencies, stats
