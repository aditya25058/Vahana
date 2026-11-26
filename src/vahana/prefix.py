"""v1.3: persistent prefix cache. A prompt that starts like an earlier one (a system prompt, a document, a long chat
history) reuses that prefix's attention keys and values from disk and prefills only the rest.

Exact: Vahana's attention, linears, norms and experts compute each token's row with the same operations whatever
the batch, the query length or how the prompt is split, so the keys and values of a prefix computed alone equal
those computed inside the full prompt, and prefilling the suffix against them gives the full prefill's logits, bit
for bit (tests/test_prefix.py). The same holds across devices, so a prefix cached on a GPU serves a CPU run too.

Layout (one directory): one safetensors file per stored prompt prefix, holding every layer's keys and values for its
first k tokens, k a multiple of `block` (256); index.json maps the hash of every block-aligned prefix to (file, k), so
a later prompt that shares only the first 512 tokens of a stored 2,048-token prefix still reuses those 512.
Keys include the model fingerprint and the vahana and transformers versions (a model-code change changes the bits).
Least recently used files are deleted beyond `max_gb`. Attention-only models, batch 1; recurrent / linear-attention
hybrids are skipped (their state cannot be cut at a block boundary of a longer prefill).
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import torch

BLOCK = 256


def default_dir() -> str:
    return os.environ.get("VAHANA_PREFIX_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "vahana", "prefix")


class PrefixCache:
    def __init__(self, path: str | None = None, model_key: str = "", block: int = BLOCK, max_gb: float = 20.0):
        import transformers

        from . import __version__
        self.path = path or default_dir()
        os.makedirs(self.path, exist_ok=True)
        self.block, self.max_bytes = int(block), max_gb * 2**30
        self.model_key = f"{model_key}|vahana {__version__}|transformers {transformers.__version__}"
        self.ixf = os.path.join(self.path, "index.json")
        try:
            with open(self.ixf) as f:
                self.ix = json.load(f)
        except (OSError, ValueError):
            self.ix = {"prefixes": {}, "files": {}}
        self.last = {"hit_tokens": 0, "stored_tokens": 0}

    def _h(self, ids, k):
        h = hashlib.sha256(self.model_key.encode())
        h.update(torch.as_tensor(ids[:k], dtype=torch.int64).cpu().numpy().astype("<i8").tobytes())
        return h.hexdigest()

    def _save_ix(self):
        tmp = self.ixf + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.ix, f)
        os.replace(tmp, self.ixf)

    def lookup(self, ids) -> tuple[int, dict | None]:
        """Longest cached block-aligned prefix strictly shorter than ids: (k, {name: tensor}) or (0, None)."""
        from safetensors.torch import load_file
        T = len(ids)
        for k in range((T - 1) // self.block * self.block, 0, -self.block):
            e = self.ix["prefixes"].get(self._h(ids, k))
            if e is None:
                continue
            f = os.path.join(self.path, e["file"])
            if not os.path.exists(f):
                continue
            kv = load_file(f)
            self.ix["files"].setdefault(e["file"], {})["used"] = time.time()
            self._save_ix()
            self.last["hit_tokens"] = k
            return k, {n: t[:, :, :k] for n, t in kv.items()}
        self.last["hit_tokens"] = 0
        return 0, None

    def store(self, ids, layers) -> int:
        """Save the keys/values of the longest block-aligned prefix of ids (shorter than ids) from `layers` (static
        cache layers holding at least that many positions), unless already stored. Returns tokens stored."""
        from safetensors.torch import save_file
        T = len(ids)
        k = (T - 1) // self.block * self.block
        self.last["stored_tokens"] = 0
        if k <= 0:
            return 0
        top = self._h(ids, k)
        if top in self.ix["prefixes"] and os.path.exists(os.path.join(self.path, self.ix["prefixes"][top]["file"])):
            return 0
        kv = {}
        for i, L in enumerate(layers):
            kv[f"{i}.k"] = L.keys[:, :, :k].to("cpu").contiguous()
            kv[f"{i}.v"] = L.values[:, :, :k].to("cpu").contiguous()
        name = top[:32] + ".safetensors"
        save_file(kv, os.path.join(self.path, name))
        size = os.path.getsize(os.path.join(self.path, name))
        self.ix["files"][name] = {"bytes": size, "used": time.time(), "tokens": k}
        for j in range(self.block, k + 1, self.block):
            self.ix["prefixes"][self._h(ids, j)] = {"file": name, "k": j}
        self._evict()
        self._save_ix()
        self.last["stored_tokens"] = k
        return k

    def _evict(self):
        files = self.ix["files"]
        total = sum(v.get("bytes", 0) for v in files.values())
        for name in sorted(files, key=lambda n: files[n].get("used", 0)):
            if total <= self.max_bytes:
                break
            try:
                os.remove(os.path.join(self.path, name))
            except OSError:
                pass
            total -= files.pop(name).get("bytes", 0)
            self.ix["prefixes"] = {h: e for h, e in self.ix["prefixes"].items() if e["file"] != name}


def fill(layers, kv: dict, k: int, device) -> None:
    """Load a cached prefix into static cache layers (initialising them from the stored shapes)."""
    import inspect
    for i, L in enumerate(layers):
        kk, vv = kv[f"{i}.k"], kv[f"{i}.v"]
        if not getattr(L, "is_initialized", False):
            ref_k, ref_v = kk.to(device), vv.to(device)          # stored in the cache's own dtype
            n = len(inspect.signature(L.lazy_initialization).parameters)
            L.lazy_initialization(ref_k, ref_v) if n >= 2 else L.lazy_initialization(ref_k)
        L.keys[:, :, :k].copy_(kk.to(L.keys.device, L.keys.dtype))
        L.values[:, :, :k].copy_(vv.to(L.values.device, L.values.dtype))
        cl = getattr(L, "cumulative_length", None)        # newer transformers track the fill position here
        if torch.is_tensor(cl):
            cl.fill_(k)
        elif isinstance(cl, int):
            L.cumulative_length = k
