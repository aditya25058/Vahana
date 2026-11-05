"""Diagnostic: where do identical batch rows stop being bitwise identical?

For a batch whose prompts are all the same, every row should carry the
same bits through every module. When rows diverge, this records the first
(step, layer, module) at which some row's output stops being torch.equal
to row 0, which rows, and by how much. Observation only: hooks read
outputs and never modify them. Costs one device sync per hooked module,
so it is opt-in and meant for probes, not measured runs.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


class RowSplitDetector:
    def __init__(self, max_events: int = 40):
        self.step = -1  # -1 = prefill; serve sets 0.. for decode steps
        self.events: list[dict] = []
        self.max_events = max_events
        self._seen: set[tuple] = set()
        self._handles: list[Any] = []

    def attach(self, model: nn.Module, profile: Any) -> int:
        """Hook attention + MLP/MoE children of every decoder layer."""
        cls = getattr(profile, "decoder_layer_cls", None)
        layers = ([m for m in model.modules() if isinstance(m, cls)] if cls is not None
                  else list(getattr(getattr(model, "model", model), "layers", [])))
        for li, layer in enumerate(layers):
            for name, child in layer.named_children():
                if name in ("self_attn", "attention", "mlp", "block_sparse_moe", "moe"):
                    self._handles.append(child.register_forward_hook(self._hook(li, name)))
        return len(self._handles)

    def _hook(self, layer_idx: int, name: str):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if not isinstance(t, torch.Tensor) or t.dim() < 2 or t.shape[0] < 2:
                return
            last = t[:, -1]  # newest position of each row
            same = (last == last[:1]).reshape(last.shape[0], -1).all(dim=1)
            if bool(same.all()):
                return
            key = (name, layer_idx)
            if key in self._seen or len(self.events) >= self.max_events:
                return
            self._seen.add(key)
            rows = (~same).nonzero().flatten().tolist()
            diff = (last.float() - last[:1].float()).abs().amax().item()
            self.events.append({"step": self.step, "layer": layer_idx, "module": name,
                                "rows": rows, "n_rows": len(rows), "max_abs_diff": diff})
        return hook

    def first(self) -> dict | None:
        return min(self.events, key=lambda e: (e["step"], e["layer"],
                                               e["module"] != "self_attn")) \
            if self.events else None

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
