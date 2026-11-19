"""Batch-1 decode with CUDA graphs around the GPU work (VAHANA_GRAPH=1; Phase 29, generalized in 30).

Needs VAHANA_INVARIANT=1 (length-agnostic Triton MLA on the ragged latent cache), --kv-layout latent,
B=1, and the all-CPU expert path (VAHANA_HYBRID=1, VAHANA_HYBRID_FAST=1).

A decode step is a fixed sequence of GPU work and host actions:
  [embedding] -> per layer: [attention] (split around o_proj when o_proj lives on the CPU)
              -> MoE: [norm, gate, D2H] host:{start CPU experts} [shared experts if on GPU]
                      host:{wait} [H2D, combine, residual]   |  dense: [MLP]
  -> [final norm, lm_head, argmax, position/length += 1]
Consecutive GPU items are captured as one CUDA graph; host items run between replays. Every kernel is the
one the eager path runs, on the same inputs; the attention kernel reads the row's start / end from
device memory, so a fixed-capacity cache gives the same bits as a growing one.
Exactness test: graph decode == the same sequence run eagerly == the stock invariant engine.
"""
from __future__ import annotations

import os
import time

import torch
from torch import nn

from . import hybrid as _hy


class _Grab(nn.Module):
    """Stands in for an o_proj that lives on the CPU: keeps its input in a static buffer."""

    def __init__(self, buf):
        super().__init__()
        self.buf = buf

    def forward(self, x):
        self.buf.copy_(x.reshape(self.buf.shape))
        return x


def graph_decode(model, wrappers, cache, next_token, max_new: int, eos_id, device):
    """Returns (new token ids incl. next_token, per-step latencies, stats)."""
    from . import twinlin as _tw
    twin = _tw.enabled()
    m = model.model
    layers = list(m.layers)
    wmap = {li: l.mlp for li, l in enumerate(layers) if l.mlp.__class__.__name__ == "TieredMoEWrapper"}
    H = model.config.hidden_size
    dt = next(p for p in model.parameters() if p.is_floating_point()).dtype
    use_graph = os.environ.get("VAHANA_GRAPH_EAGER", "0") != "1"
    numa = int(os.environ.get("VAHANA_HYBRID_NUMA", "1"))
    for w in wrappers:
        if not getattr(w, "_hy_registered", False):
            raise RuntimeError("graph decode: experts not registered (prefill must use the fast path)")

    T0 = cache.len_host[0]
    cache.get_seq_length = lambda layer_idx=0: cache.cap          # rotary table slice: fixed size
    tok = torch.zeros((1, 1), dtype=torch.long, device=device)
    tok.copy_(next_token.view(1, 1))
    pos = torch.full((1, 1), T0, dtype=torch.long, device=device)
    h = torch.zeros((1, 1, H), dtype=dt, device=device)            # residual stream
    r = torch.zeros((1, 1, H), dtype=dt, device=device)            # attention residual
    x2 = torch.zeros((1, 1, H), dtype=dt, device=device)           # MoE input
    sh = torch.zeros((1, 1, H), dtype=dt, device=device)           # shared-expert output (GPU)
    od = torch.zeros((1, H), dtype=dt, device=device)
    yd = torch.zeros((1, H), dtype=dt, device=device)
    xb = _hy.pinned("gx", (1, H), dt)
    ob = _hy.pinned("gout", (1, H), dt)
    yb = _hy.pinned("gy", (1, H), dt)
    ev = torch.cuda.Event()
    emb_cpu = m.embed_tokens.weight.device.type == "cpu"
    eb = _hy.pinned("gemb", (1, H), dt)
    cur = [int(next_token.view(-1)[0])]
    rt = {}

    # o_proj on the CPU: grab its input in a static buffer, compute on the worker
    grabs = {}
    for li, layer in enumerate(layers):
        op = layer.self_attn.o_proj
        if op.weight.device.type == "cpu":
            K = op.weight.shape[1]
            gb = torch.zeros((1, K), dtype=dt, device=device)
            grabs[li] = (op, gb, _hy.pinned(f"go{K}", (1, K), dt))
            layer.self_attn.o_proj = _Grab(gb)

    def attn(li):
        layer = layers[li]
        r.copy_(h)
        y = layer.self_attn(hidden_states=layer.input_layernorm(h), attention_mask=None, position_ids=pos,
                            past_key_value=cache, output_attentions=False, use_cache=True)[0]
        if li in grabs:
            grabs[li][2].copy_(grabs[li][1], non_blocking=True)      # D2H of the o_proj input
        else:
            h.copy_(r + y)

    def host_oproj(li):
        ev.record()
        ev.synchronize()
        _hy.ext().submit_linear(grabs[li][2], grabs[li][0].weight, yb)
        _hy.ext().wait_done()

    def after_oproj(li):
        yd.copy_(yb, non_blocking=True)
        h.copy_(r + yd.view(1, 1, H))

    def moe_pre(li):
        w = wmap[li]
        r.copy_(h)
        y = layers[li].post_attention_layernorm(h)
        x2.copy_(y)
        ti, tw = w.spec.route(y)
        Kk = ti.shape[-1]
        if "ti" not in rt:
            rt["ti"] = _hy.pinned("gti", (1, Kk), torch.int64)
            rt["tw"] = _hy.pinned("gtw", (1, Kk), tw.dtype)
        xb.copy_(y.reshape(1, H), non_blocking=True)
        rt["ti"].copy_(ti.reshape(1, Kk), non_blocking=True)
        rt["tw"].copy_(tw.reshape(1, Kk), non_blocking=True)

    def host_submit(li):
        ev.record()
        ev.synchronize()
        _hy.ext().submit_moe(li, xb, rt["ti"], rt["tw"], ob, numa)

    def shared(li):
        w = wmap[li]
        if twin:
            sh.copy_(_tw.shared_gpu(w, x2.view(1, H)).view(1, 1, H))
        else:
            sh.copy_(w.shared_experts(x2) if w.shared_experts is not None else torch.zeros_like(x2))

    def moe_post(li):
        od.copy_(ob, non_blocking=True)
        if twin and _tw.shared_on_cpu(wmap[li]):
            out = od.view(1, 1, H)                       # the CPU job already added the shared experts
        else:
            out = sh + od.view(1, 1, H)
        h.copy_(r + out)

    def dense(li):
        layer = layers[li]
        h.copy_(h + layer.mlp(layer.post_attention_layernorm(h)))

    def host_embed():
        eb.copy_(m.embed_tokens.weight[cur[0]].view(1, H))

    def embed():
        if emb_cpu:
            h.copy_(eb.view(1, 1, H), non_blocking=True)
        else:
            h.copy_(m.embed_tokens(tok))

    def final():
        logits = model.lm_head(m.norm(h))
        tok.copy_(logits[:, -1, :].argmax(dim=-1, keepdim=True))
        pos.add_(1)
        cache.length.add_(1)

    # ---- the step as a fixed sequence --------------------------------------------------------------
    seq = []
    if emb_cpu:
        seq.append(("host", host_embed, "embed"))
    seq.append(("gpu", embed))
    for li in range(len(layers)):
        seq.append(("gpu", lambda li=li: attn(li)))
        if li in grabs:
            seq.append(("host", lambda li=li: host_oproj(li), "oproj"))
            seq.append(("gpu", lambda li=li: after_oproj(li)))
        if li in wmap:
            seq.append(("gpu", lambda li=li: moe_pre(li)))
            seq.append(("host", lambda li=li: host_submit(li), "submit"))
            if not (twin and _tw.shared_on_cpu(wmap[li])):
                seq.append(("gpu", lambda li=li: shared(li)))
            seq.append(("host", lambda: _hy.ext().wait_done(), "wait"))
            seq.append(("gpu", lambda li=li: moe_post(li)))
        else:
            seq.append(("gpu", lambda li=li: dense(li)))
    seq.append(("gpu", final))
    items = []                                   # merge consecutive GPU work: ["gpu", [fns], graph] / ["host", fn]
    for ent in seq:
        kind, fn = ent[0], ent[1]
        if kind == "gpu" and items and items[-1][0] == "gpu":
            items[-1][1].append(fn)
        elif kind == "gpu":
            items.append(["gpu", [fn], None])
        else:
            items.append(["host", fn, ent[2] if len(ent) > 2 else "host"])
    HT = {}                                   # Phase 40 timing: label -> [wall s, kernel ms, n]
    KL = {}                                   # Phase 41: label -> every kernel ms (steady steps)
    timing = os.environ.get("VAHANA_GRAPH_TIMING") == "1"

    def step():
        for it in items:
            if it[0] == "host":
                if timing:
                    t_h = time.perf_counter()
                    it[1]()
                    h_ = HT.setdefault(it[2], [0.0, 0.0, 0])
                    h_[0] += time.perf_counter() - t_h
                    if it[2] in ("wait", "oproj"):
                        km_ = _hy.ext().kernel_ms()
                        h_[1] += km_
                        KL.setdefault(it[2], []).append(km_)
                    h_[2] += 1
                else:
                    it[1]()
            elif it[2] is not None:
                if timing:
                    t_h = time.perf_counter()
                    it[2].replay()
                    torch.cuda.synchronize()
                    h_ = HT.setdefault("gpu", [0.0, 0.0, 0])
                    h_[0] += time.perf_counter() - t_h
                    h_[2] += 1
                else:
                    it[2].replay()
            else:
                for fn in it[1]:
                    fn()
        cache.len_host[0] += 1

    def capture():
        pool = torch.cuda.graph_pool_handle()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for it in items:
                if it[0] == "gpu":
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=pool, stream=s):
                        for fn in it[1]:
                            fn()
                    it[2] = g
        torch.cuda.current_stream().wait_stream(s)

    out = [cur[0]]
    lat = []
    _mc = os.environ.get("VAHANA_MAIN_CPUS")
    if _mc:                                       # Phase 41: main thread off the worker's core
        os.sched_setaffinity(0, {int(c_) for c_ in _mc.split(",")})
    stats = {"graph": use_graph, "capture_s": None, "moe_layers": len(wmap),
             "graphs": sum(1 for it in items if it[0] == "gpu"), "host_items": sum(1 for it in items if it[0] == "host"),
             "cpu_oproj_layers": len(grabs), "embedding_on_cpu": emb_cpu}
    try:
        with torch.no_grad():
            for i in range(max_new - 1):
                if eos_id is not None and out[-1] == eos_id:
                    break
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                prof_out = os.environ.get("VAHANA_GRAPH_PROF")
                if prof_out and i == 4 and use_graph:      # Phase 41: step 4 runs under the profiler (diag only)
                    from torch.profiler import profile, ProfilerActivity
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof_:
                        step()
                        torch.cuda.synchronize()
                    with open(prof_out, "w") as f_:
                        f_.write(prof_.key_averages().table(sort_by="cuda_time_total", row_limit=60))
                else:
                    step()
                nt = int(tok.item())
                lat.append(time.perf_counter() - t0)
                out.append(nt)
                cur[0] = nt
                if i == 0 and use_graph:
                    # the first step ran eagerly (compiles / warms every kernel); capture records without
                    # executing, so the cache and position are untouched by it.
                    tc = time.perf_counter()
                    capture()
                    torch.cuda.synchronize()
                    stats["capture_s"] = time.perf_counter() - tc
                    HT.clear()                           # timing: steady-state steps only
                    KL.clear()
    finally:
        for li, (op, _, _) in grabs.items():
            layers[li].self_attn.o_proj = op
    if timing:
        stats["host_timing"] = {k: {"wall_s": v[0], "kernel_ms": v[1], "n": v[2]} for k, v in HT.items()}
        for k_, v_ in KL.items():
            q_ = sorted(v_)
            stats["host_timing"][k_]["kernel_p50_p90_max"] = [q_[len(q_) // 2], q_[int(len(q_) * 0.9)], q_[-1]]
            stats["host_timing"][k_]["kernel_all"] = v_
    stats["nx"] = dict(getattr(_hy, "NX_STATS", {}))
    return out, lat, stats
