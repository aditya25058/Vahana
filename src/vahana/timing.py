"""Opt-in step timeline (VAHANA_TIMING=1): CUDA events, no syncs inside the step.

Records, per decode step:
  - compute-stream events around every attention module and every MoE wrapper,
  - compute-stream events around every routed-expert GEMM (after its copy wait),
  - copy-stream (and ANS decode-stream) events at the start / end of each layer's loads,
  - host seconds spent in the MoE routing host syncs and in the whole MoE forward.
`flush()` (called after the step's own synchronize) turns them into a summary dict.
Diagnostics only; values are unchanged.
"""
from __future__ import annotations

import time

import torch

REC: "Recorder | None" = None


class Recorder:
    def __init__(self):
        self.reset()
        self.steps: list[dict] = []

    def reset(self):
        self.spans: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self.host: dict[str, float] = {}
        self.marks: list[tuple[str, torch.cuda.Event]] = []
        self.t0 = time.perf_counter()

    def ev(self, stream=None) -> torch.cuda.Event:
        e = torch.cuda.Event(enable_timing=True)
        e.record(stream) if stream is not None else e.record()
        return e

    def span(self, key: str, a: torch.cuda.Event, b: torch.cuda.Event) -> None:
        self.spans.setdefault(key, []).append((a, b))

    def add_host(self, key: str, s: float) -> None:
        self.host[key] = self.host.get(key, 0.0) + s

    def mark(self, key: str, stream=None) -> None:
        self.marks.append((key, self.ev(stream)))

    def flush(self, step: int, wall_s: float) -> dict:
        out = {"step": step, "wall_s": wall_s, "gpu_s": {}, "n": {}, "host_s": dict(self.host)}
        for k, v in self.spans.items():
            out["gpu_s"][k] = sum(a.elapsed_time(b) for a, b in v) / 1e3
            out["n"][k] = len(v)
        if len(self.marks) >= 2:
            first, last = self.marks[0][1], self.marks[-1][1]
            out["gpu_s"]["step_marks"] = first.elapsed_time(last) / 1e3
        # Union of copy-stream activity windows vs the step: how busy was the link?
        out["host_s"]["step_host"] = time.perf_counter() - self.t0
        self.steps.append(out)
        self.reset()
        return out


def enabled() -> bool:
    return REC is not None


def install(model, wrappers) -> Recorder:
    global REC
    REC = Recorder()

    def pre(name):
        def _hook(m, _a):
            m._vahana_t = (REC.ev(), time.perf_counter())   # returns None
        return _hook

    def post(name):
        def _hook(m, _a, _o):
            a, h = m._vahana_t
            REC.span(name, a, REC.ev())
            REC.add_host(name, time.perf_counter() - h)
        return _hook
    for n, m in model.named_modules():
        if n.endswith(".self_attn"):
            m.register_forward_pre_hook(pre("attn"))
            m.register_forward_hook(post("attn"))
    for w in wrappers:
        w.register_forward_pre_hook(pre("moe"))
        w.register_forward_hook(post("moe"))
    return REC
