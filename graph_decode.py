"""Phase 29b: batch-1 decode with CUDA graphs around the GPU work (COLOSSUS_GRAPH=1).

Needs COLOSSUS_INVARIANT=1 (length-agnostic Triton MLA on the ragged latent cache), --kv-layout latent,
B=1, and the all-CPU expert path (COLOSSUS_HYBRID=1, COLOSSUS_HYBRID_FAST=1).

Per decode step, per MoE layer i:
  A_i (graph): [finish layer i-1: H2D of the CPU expert output, shared + routed, residual] ->
               [layers up to i: norms, attention (writes the latent at the device-side length),
               dense MLPs] -> post-attention norm -> gate -> D2H of the hidden row + routing to pinned.
  host:        wait for A_i, start the CPU experts (C++ worker).
  B_i (graph): shared experts of layer i (run while the CPU computes the routed experts).
  host:        wait for the CPU experts.
  F   (graph): finish the last layer, final norm, lm_head, argmax, position += 1, length += 1.
Every kernel is the one the eager path runs, on the same inputs; the attention kernel reads the
row's start / end from device memory, so a fixed-capacity cache gives the same bits as a growing one.
Exactness test: graph decode == the same driver run eagerly == the stock invariant engine.
"""
from __future__ import annotations

import os
import time

import torch

from . import hybrid as _hy


class _Segment:
    def __init__(self, fn):
        self.fn = fn
        self.g = None


def graph_decode(model, wrappers, cache, next_token, max_new: int, eos_id, device):
    """Returns (new token ids incl. next_token, per-step latencies, stats)."""
    m = model.model
    layers = list(m.layers)
    wmap = {}
    for li, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and mlp.__class__.__name__ == "TieredMoEWrapper":
            wmap[li] = mlp
    moe_layers = sorted(wmap)
    H = model.config.hidden_size
    dt = next(model.parameters()).dtype
    use_graph = os.environ.get("COLOSSUS_GRAPH_EAGER", "0") != "1"
    numa = int(os.environ.get("COLOSSUS_HYBRID_NUMA", "1"))

    # ---- static state -------------------------------------------------------------------------
    T0 = cache.len_host[0]
    cache.get_seq_length = lambda layer_idx=0: cache.cap          # rotary table slice: fixed size
    tok = torch.zeros((1, 1), dtype=torch.long, device=device)
    tok.copy_(next_token.view(1, 1))
    pos = torch.full((1, 1), T0, dtype=torch.long, device=device)
    h = torch.zeros((1, 1, H), dtype=dt, device=device)            # residual stream
    resid = torch.zeros((1, 1, H), dtype=dt, device=device)        # residual at the MoE layer
    x2 = torch.zeros((1, 1, H), dtype=dt, device=device)           # MoE input (post-attn norm)
    sh = torch.zeros((1, 1, H), dtype=dt, device=device)           # shared-expert output
    K = None
    xb = _hy.pinned("gx", (1, H), dt)
    ob = _hy.pinned("gout", (1, H), dt)
    od = torch.zeros((1, H), dtype=dt, device=device)
    for w in wrappers:
        if not getattr(w, "_hy_registered", False):
            raise RuntimeError("graph decode: experts not registered (prefill must use the fast path)")

    def attn_block(li, x):
        layer = layers[li]
        r = x
        y = layer.input_layernorm(x)
        y = layer.self_attn(hidden_states=y, attention_mask=None, position_ids=pos,
                            past_key_value=cache, output_attentions=False, use_cache=True)[0]
        return r + y

    def finish_prev(li_prev):
        # layer li_prev's MoE output: shared + routed (bf16 add, as the wrapper), then residual
        od.copy_(ob, non_blocking=True)
        out = sh + od.view(1, 1, H)
        return resid + out

    tib = twb = None

    def seg_A(k):
        """MoE layer index k in moe_layers: finish the previous MoE layer, run up to this gate."""
        nonlocal tib, twb
        li = moe_layers[k]
        if k == 0:
            x = m.embed_tokens(tok)
            start = 0
        else:
            x = finish_prev(moe_layers[k - 1])
            start = moe_layers[k - 1] + 1
        for lj in range(start, li + 1):
            x = attn_block(lj, x)
            if lj < li:                                   # dense layer (no MoE wrapper)
                layer = layers[lj]
                x = x + layer.mlp(layer.post_attention_layernorm(x))
        resid.copy_(x)
        y = layers[li].post_attention_layernorm(x)
        x2.copy_(y)
        w = wmap[li]
        ti, tw = w.spec.route(y)
        Kk = ti.shape[-1]
        if tib is None:
            tib = _hy.pinned("gti", (1, Kk), torch.int64)
            twb = _hy.pinned("gtw", (1, Kk), tw.dtype)
        xb.copy_(y.reshape(1, H), non_blocking=True)
        tib.copy_(ti.reshape(1, Kk), non_blocking=True)
        twb.copy_(tw.reshape(1, Kk), non_blocking=True)

    def seg_B(k):
        w = wmap[moe_layers[k]]
        sh.copy_(w.shared_experts(x2) if w.shared_experts is not None else torch.zeros_like(x2))

    def seg_F():
        x = finish_prev(moe_layers[-1])
        for lj in range(moe_layers[-1] + 1, len(layers)):
            x = attn_block(lj, x)
            layer = layers[lj]
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        x = m.norm(x)
        logits = model.lm_head(x)
        tok.copy_(logits[:, -1, :].argmax(dim=-1, keepdim=True))
        pos.add_(1)
        cache.length.add_(1)

    segA = [_Segment(lambda k=k: seg_A(k)) for k in range(len(moe_layers))]
    segB = [_Segment(lambda k=k: seg_B(k)) for k in range(len(moe_layers))]
    segF = _Segment(seg_F)
    ev = torch.cuda.Event()

    def run(seg):
        if seg.g is not None:
            seg.g.replay()
        else:
            seg.fn()

    def step():
        for k, li in enumerate(moe_layers):
            run(segA[k])
            ev.record()
            ev.synchronize()
            _hy.ext().submit_moe(li, xb, tib, twb, ob, numa)
            run(segB[k])
            _hy.ext().wait_done()
        run(segF)
        cache.len_host[0] += 1

    def capture():
        pool = torch.cuda.graph_pool_handle()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for k, li in enumerate(moe_layers):
                for seg in (segA[k], segB[k]):
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=pool, stream=s):
                        seg.fn()
                    seg.g = g
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool, stream=s):
                segF.fn()
            segF.g = g
        torch.cuda.current_stream().wait_stream(s)

    out = [int(next_token.view(-1)[0])]
    lat = []
    stats = {"graph": use_graph, "capture_s": None, "moe_layers": len(moe_layers)}
    with torch.no_grad():
        for i in range(max_new - 1):
            if eos_id is not None and out[-1] == eos_id:
                break
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            step()
            nt = int(tok.item())
            lat.append(time.perf_counter() - t0)
            out.append(nt)
            if i == 0 and use_graph:
                # the first step ran eagerly (compiles / warms every kernel); capture now. Capture
                # records without executing, so the cache and position are untouched by it.
                tc = time.perf_counter()
                capture()
                torch.cuda.synchronize()
                stats["capture_s"] = time.perf_counter() - tc
    return out, lat, stats
