"""v1.4: how close is each 4-bit model to the original? Vahana's 4-bit checkpoint against llama.cpp's Q4_K_M, both
judged against the bf16 original on the same tokens, with llama.cpp's own statistics.

  python kld.py base  BF16_MODEL_DIR TEXT_FILE OUT.dat [N_CHUNK]   the bf16 original's log-probs, llama.cpp's format
  python kld.py score MODEL_DIR BASE.dat                           a Vahana model against them

The base file is the one `llama-perplexity --kl-divergence-base` reads, so llama.cpp scores its own model against
the same reference:
  llama-perplexity -m Q4_K_M.gguf --kl-divergence-base BASE.dat --kl-divergence -c 512 [-ngl 99 --n-cpu-moe 42]
Chunks of 512 tokens, scored on their second half (llama.cpp's protocol). Statistics as llama.cpp computes them:
PPL, mean and 99th-percentile KL divergence from the base, how often the top token is the same, RMS of the change in
the probability of the actual next token.
"""
import math
import os
import struct
import sys
from argparse import Namespace

import numpy as np

N_CTX = 512
FIRST = N_CTX // 2


def load(model_dir):
    from vahana import cli
    ns = Namespace(model=model_dir, gpu_budget=None, reserve=1.5, threads=None, keep_numa_balancing=False,
                   device=None, gpu_cache="auto")
    model, tok, torch = cli._load(ns, cli._setup(ns))
    return model, tok, torch


def chunk_logits(model, torch, ids):
    """Logits [N_CTX - 1 - FIRST, V] float32 of positions FIRST .. N_CTX - 2 (each predicts the next token)."""
    dev = model._vahana_device
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
    return out.logits[0, FIRST:N_CTX - 1].float().cpu().numpy()


def pack(lg):
    """One token's logits -> llama.cpp's record: float scale, float min_log_prob, then a uint16 per vocab entry."""
    V = lg.shape[0]
    mx = float(lg.max())
    mn = max(float(lg.min()), mx - 16)
    lse = math.log(float(np.exp((lg - mx).astype(np.float64)).sum()))
    min_lp = mn - mx - lse
    scale = (mx - mn) / 65535.0
    nv = 2 * ((V + 1) // 2) + 4
    rec = np.zeros(nv, dtype=np.uint16)
    rec[:4] = np.frombuffer(struct.pack("<ff", np.float32(scale), np.float32(min_lp)), dtype=np.uint16)
    if scale:
        q = np.where(lg > mn, np.rint((lg - mn) / np.float32(scale)), 0)
        rec[4:4 + V] = q.astype(np.uint16)
    return rec


def base(model_dir, text_file, out, n_chunk):
    model, tok, torch = load(model_dir)
    toks = tok(open(text_file, encoding="utf-8").read(), return_tensors="np").input_ids[0].astype(np.int32)
    n_chunk = min(n_chunk, len(toks) // N_CTX)
    toks = toks[:n_chunk * N_CTX]
    V = model.config.vocab_size
    with open(out, "wb") as f:
        f.write(b"_logits_" + struct.pack("<Iii", N_CTX, V, n_chunk))
        f.write(toks.tobytes())
        nll = 0.0
        for c in range(n_chunk):
            ids = toks[c * N_CTX:(c + 1) * N_CTX].tolist()
            lg = chunk_logits(model, torch, ids)
            for i in range(lg.shape[0]):
                f.write(pack(lg[i]).tobytes())
                row = lg[i].astype(np.float64)
                m = row.max()
                nll += m + math.log(np.exp(row - m).sum()) - row[ids[FIRST + i + 1]]
            n = (c + 1) * (N_CTX - 1 - FIRST)
            print(f"[{c + 1}/{n_chunk}] base PPL {math.exp(nll / n):.4f}", flush=True)


def score(model_dir, base_file):
    model, tok, torch = load(model_dir)
    with open(base_file, "rb") as f:
        assert f.read(8) == b"_logits_"
        n_ctx, V, n_chunk = struct.unpack("<Iii", f.read(12))
        assert n_ctx == N_CTX
        toks = np.frombuffer(f.read(4 * n_ctx * n_chunk), dtype=np.int32)
        nv = 2 * ((V + 1) // 2) + 4
        nll, nll_b, klds, pdiffs, same = [], [], [], [], 0
        for c in range(n_chunk):
            ids = toks[c * N_CTX:(c + 1) * N_CTX].tolist()
            lg = chunk_logits(model, torch, ids)[:, :V].astype(np.float64)
            recs = np.frombuffer(f.read(2 * nv * lg.shape[0]), dtype=np.uint16).reshape(lg.shape[0], nv)
            for i in range(lg.shape[0]):
                t = ids[FIRST + i + 1]
                scale, min_lp = struct.unpack("<ff", recs[i, :4].tobytes()[:8])
                lpb = scale * recs[i, 4:4 + V].astype(np.float64) + min_lp
                row = lg[i]
                m = row.max()
                lp = row - (m + math.log(np.exp(row - m).sum()))
                nll.append(-lp[t])
                nll_b.append(-lpb[t])
                k = lpb > -16
                klds.append(float((np.exp(lpb[k]) * (lpb[k] - lp[k])).sum()))
                same += int(row.argmax() == lpb.argmax())
                pdiffs.append(math.exp(lp[t]) - math.exp(lpb[t]))
            n = len(nll)
            print(f"[{c + 1}/{n_chunk}] PPL {math.exp(np.mean(nll)):.4f} base {math.exp(np.mean(nll_b)):.4f} "
                  f"KLD {np.mean(klds):.5f} same top {same / n:.2%}", flush=True)
    n = len(nll)
    pd = np.array(pdiffs)
    print(f"\nmodel {os.path.basename(model_dir.rstrip('/'))}, {n} tokens")
    print(f"Mean PPL(Q)    {math.exp(np.mean(nll)):.4f}")
    print(f"Mean PPL(base) {math.exp(np.mean(nll_b)):.4f}")
    print(f"Mean ln(PPL(Q)/PPL(base)) {np.mean(np.array(nll) - np.array(nll_b)):.6f}")
    print(f"Mean KLD       {np.mean(klds):.6f}")
    print(f"99.0% KLD      {np.percentile(klds, 99):.6f}")
    print(f"Same top p     {same / n:.3%}")
    print(f"RMS dp         {math.sqrt(np.mean(pd ** 2)):.3%}")


if __name__ == "__main__":
    if sys.argv[1] == "base":
        base(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 32)
    else:
        score(sys.argv[2], sys.argv[3])
