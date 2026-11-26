"""Vahana Receipts (v1.2): verifiable inference.

A receipt records what was asked (prompt token ids, decoding settings, model fingerprint) and what came back (the
token ids, and a SHA-256 digest of every step's logits). Because Vahana's arithmetic gives the same bits on every
supported CPU and GPU, anyone with the same weights can re-run the computation on their own machine and check the
receipt bit for bit: same logits at every step, same tokens. No trusted hardware, no tolerance, no sampling of
positions.

Checking is cheap for attention models: the logits of one teacher-forced forward over prompt + output are
bit-identical to the logits the decoder produced step by step (tested on every family), so the whole answer is
checked with one prefill-shaped pass. Models with recurrent / linear-attention layers (Qwen3.5) compute a chunked
prefill differently from their recurrent decode, so they are checked by replaying the decode.

Sampling is checkable too: ExactSampler draws from the exact logits using only IEEE-754 basic operations (+, -, *, /,
an exp built from them, sequential sums) and a SHA-256 counter stream, so a seeded sample is the same token on every
machine and every numpy/torch version.

A receipt proves that this output is what these weights compute for this prompt and these settings. It does not
prove who ran it (it is not signed) and it says nothing about the weights beyond their fingerprint.
"""
from __future__ import annotations

import hashlib
import json
import os

FORMAT = "vahana-receipt/1"
SAMPLER = "vahana-exact-sampler/1"


# ---------------------------------------------------------------------------------------------------- exact sampler
_LN2_HI = 6.93147180369123816490e-01          # fdlibm split of ln 2: k * _LN2_HI is exact for |k| < 2^11
_LN2_LO = 1.90821492927058770002e-10
_INV_LN2 = 1.44269504088896338700e+00


def texp(z):
    """exp(z) for fp64 z <= 0 using only IEEE basic operations (so the bits are the same everywhere).
    Relative error < 1e-15; exp(z) = 0 below -745."""
    import numpy as np
    z = np.asarray(z, dtype=np.float64)
    zc = np.maximum(z, -745.0)
    k = np.rint(zc * _INV_LN2)
    r = (zc - k * _LN2_HI) - k * _LN2_LO      # |r| <= 0.347
    p = np.full_like(r, 1.0 / 6227020800.0)   # Taylor to r^13 / 13!, Horner
    for n in range(12, -1, -1):
        p = p * r + 1.0 / float(_fact(n))
    out = np.ldexp(p, k.astype(np.int64))
    return np.where(z < -745.0, 0.0, out)


def _fact(n):
    f = 1
    for i in range(2, n + 1):
        f *= i
    return f


def _uniform(seed: int, step: int, row: int) -> float:
    """53-bit uniform in [0, 1) from SHA-256 of (seed, step, row): a counter stream with no state to drift."""
    h = hashlib.sha256(f"{SAMPLER}|{int(seed)}|{int(step)}|{int(row)}".encode()).digest()
    return (int.from_bytes(h[:8], "little") >> 11) / 9007199254740992.0


def sample_row(logits_row, temperature: float, top_p: float, u: float) -> int:
    """One token from one row of fp32 logits: softmax(logits / T) restricted to the top-p nucleus, inverse-CDF at u.
    Every operation is an IEEE basic operation in a fixed order."""
    import numpy as np
    x = np.asarray(logits_row, dtype=np.float32).astype(np.float64) / float(temperature)
    p = texp(x - x.max())
    order = np.argsort(-p, kind="stable")      # descending probability; ties by token id
    sp = p[order]
    cs = np.add.accumulate(sp)                 # strictly sequential sums
    if top_p < 1.0:
        n = int(np.count_nonzero((cs - sp) < float(top_p) * cs[-1]))
        n = max(n, 1)
    else:
        n = len(sp)
    t = u * cs[n - 1]
    i = int(np.searchsorted(cs[:n], t, side="right"))
    return int(order[min(i, n - 1)])


class ExactSampler:
    """sampler(logits [B, V]) -> [B, 1] for decode.generate / the server: the same token on every machine."""

    def __init__(self, temperature: float, top_p: float = 1.0, seed: int = 0):
        self.temperature, self.top_p, self.seed = float(temperature), float(top_p), int(seed)
        self.step = 0

    def __call__(self, logits):
        import torch
        lg = logits.detach().float().cpu().numpy()
        out = [sample_row(lg[b], self.temperature, self.top_p, _uniform(self.seed, self.step, b))
               for b in range(lg.shape[0])]
        self.step += 1
        return torch.tensor(out, dtype=torch.long).view(-1, 1)

    def params(self) -> dict:
        return {"sampler": SAMPLER, "temperature": self.temperature, "top_p": self.top_p, "seed": self.seed}


def make_sampler(temperature, top_p=1.0, seed=0):
    if not temperature or temperature <= 0:
        return None
    return ExactSampler(temperature, top_p, seed)


# ----------------------------------------------------------------------------------------------------- fingerprint
def fingerprint(model_dir: str, full: bool = False) -> dict:
    """Identifies the weights: SHA-256 over config.json and, for every safetensors file, its name, size and header
    (tensor names, dtypes, shapes, offsets). With full=True, over every byte of every file (slow for big models).
    Tampered weight values are caught by verification itself (the logits change); this names which weights."""
    h = hashlib.sha256()
    files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    with open(os.path.join(model_dir, "config.json"), "rb") as f:
        h.update(b"config.json\0" + f.read())
    total = 0
    for name in files:
        p = os.path.join(model_dir, name)
        size = os.path.getsize(p)
        total += size
        h.update(f"\0{name}\0{size}\0".encode())
        with open(p, "rb") as f:
            if full:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    h.update(chunk)
            else:
                n = int.from_bytes(f.read(8), "little")
                h.update(f.read(n))
    return {"sha256": h.hexdigest(), "mode": "full" if full else "headers", "files": len(files),
            "bytes": total}


# ------------------------------------------------------------------------------------------------------- digests
def step_digests(logits) -> list[str]:
    """logits [n, V] (one row per generated token) -> SHA-256 of each row's fp32 bytes (little-endian)."""
    import numpy as np
    a = logits.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return [hashlib.sha256(np.ascontiguousarray(a[i]).tobytes()).hexdigest() for i in range(a.shape[0])]


def chain(digests: list[str]) -> str:
    h = hashlib.sha256(b"vahana-logits-chain/1")
    for d in digests:
        h.update(bytes.fromhex(d))
    return h.hexdigest()


def _body_id(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k != "receipt_id"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _env(model) -> dict:
    import platform
    import sys

    import torch
    import transformers

    from . import __version__
    env = {"vahana": __version__, "transformers": transformers.__version__, "torch": torch.__version__,
           "device": str(getattr(model, "_vahana_device", "cpu")), "platform": sys.platform,
           "machine": platform.machine()}
    try:
        from . import hybrid
        env["isa"] = hybrid.ext().isa()
    except Exception:  # noqa: BLE001
        pass
    if env["device"].startswith("cuda"):
        env["gpu"] = torch.cuda.get_device_name()
    return env


def make(model, model_dir: str | None, prompt_ids, result: dict, max_new_tokens: int, eos_token_id=None,
         sampler: ExactSampler | None = None, fp: dict | None = None) -> dict:
    """A receipt for one generate() call (batch of 1, return_logits=True)."""
    out = [int(t) for t in result["ids"].view(-1).tolist()]
    lg = result["logits"]
    lg = lg.reshape(-1, lg.shape[-1])
    digs = step_digests(lg)
    eos = sorted(eos_token_id) if isinstance(eos_token_id, (list, tuple, set)) else eos_token_id
    rec = {
        "format": FORMAT,
        "model": {"type": model.config.get_text_config().model_type,
                  "fingerprint": fp if fp is not None else (fingerprint(model_dir) if model_dir else None)},
        "prompt_ids": [int(t) for t in prompt_ids.view(-1).tolist()],
        "decoding": {"max_new_tokens": int(max_new_tokens), "eos_token_id": eos,
                     **(sampler.params() if sampler is not None else {"sampler": "greedy"})},
        "output_ids": out,
        "logits": {"dtype": "float32", "rows": len(digs), "chain": chain(digs), "steps": [d[:16] for d in digs]},
        "produced_on": _env(model),
    }
    rec["receipt_id"] = _body_id(rec)
    return rec


# -------------------------------------------------------------------------------------------------------- verify
def _attention_only(model) -> bool:
    kinds = set(getattr(model.config.get_text_config(), "layer_types", None) or [])
    return not (kinds - {"full_attention", "sliding_attention", "attention"})


def recompute(model, rec: dict, mode: str = "auto"):
    """The logits [n, V] these weights give for the receipt's prompt and output. mode: 'prefill' (one teacher-forced
    pass), 'replay' (decode again with the receipt's settings), 'auto' (prefill for attention models)."""
    import torch
    from . import decode
    dev = getattr(model, "_vahana_device", torch.device("cpu"))
    p = torch.tensor([rec["prompt_ids"]], dtype=torch.long, device=dev)
    out = rec["output_ids"]
    if mode == "auto":
        mode = "prefill" if _attention_only(model) else "replay"
    if mode == "prefill":
        full = torch.cat([p, torch.tensor([out[:-1]], dtype=torch.long, device=dev)], 1)
        with torch.no_grad():
            lg = model(full, use_cache=False, logits_to_keep=len(out)).logits[0].float()
        return lg, None, mode
    d = rec["decoding"]
    smp = None if d["sampler"] == "greedy" else ExactSampler(d["temperature"], d["top_p"], d["seed"])
    r = decode.generate(model, p, max_new_tokens=d["max_new_tokens"], eos_token_id=d["eos_token_id"],
                        return_logits=True, sampler=smp, graph=dev.type == "cuda")
    return r["logits"].reshape(-1, r["logits"].shape[-1]).float(), [int(t) for t in r["ids"].view(-1).tolist()], mode


def verify(model, rec: dict, model_dir: str | None = None, mode: str = "auto", fp: dict | None = None) -> dict:
    """Re-run the receipt on this machine. Returns {"ok", "checks": {...}, "first_bad_step", "mode", ...}."""
    checks = {}
    checks["format"] = rec.get("format") == FORMAT
    checks["intact"] = rec.get("receipt_id") == _body_id(rec)
    if model_dir is not None or fp is not None:
        mine = fp if fp is not None else fingerprint(model_dir, full=rec["model"]["fingerprint"].get("mode") == "full")
        checks["weights"] = mine["sha256"] == rec["model"]["fingerprint"]["sha256"]
    checks["model_type"] = model.config.get_text_config().model_type == rec["model"]["type"]
    lg, ids, mode = recompute(model, rec, mode)
    digs = step_digests(lg)
    want = rec["logits"]["steps"]
    bad = next((i for i, (a, b) in enumerate(zip(digs, want)) if a[:16] != b), None)
    if bad is None and len(digs) != len(want):
        bad = min(len(digs), len(want))
    checks["logits"] = bad is None and chain(digs) == rec["logits"]["chain"]
    out = rec["output_ids"]
    if ids is None:                              # prefill mode: re-derive each token from the recomputed logits
        d = rec["decoding"]
        if d["sampler"] == "greedy":
            ids = [int(t) for t in lg.argmax(-1).tolist()]
        else:
            ids = [sample_row(lg[i].cpu().numpy(), d["temperature"], d["top_p"], _uniform(d["seed"], i, 0))
                   for i in range(lg.shape[0])]
        eos = d["eos_token_id"]
        eos = set(eos) if isinstance(eos, list) else ({eos} if eos is not None else set())
        stop = next((i for i, t in enumerate(out[:-1]) if t in eos), None)
        checks["stops"] = stop is None and (len(out) == d["max_new_tokens"] or out[-1] in eos)
    checks["tokens"] = ids == out
    tok_bad = next((i for i, (a, b) in enumerate(zip(ids, out)) if a != b), None)
    first = min([x for x in (bad, tok_bad) if x is not None], default=None)
    return {"ok": all(checks.values()), "checks": checks, "first_bad_step": first, "mode": mode,
            "steps": len(out), "checked_on": _env(model)}


def save(rec: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
