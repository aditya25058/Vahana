"""Batch-1 Mixtral decode with CUDA graphs (COLOSSUS_MIXTRAL_GRAPH=1; Phase 36).

For fused-expert MoE models served with COLOSSUS_FUSED_CPU_MOE=1 (cpu_moe_block.CPUExpertsMoE).
After the stock prefill, K/V move into a fixed-capacity static cache. A decode step is a fixed sequence:
  [embed] -> per layer: [norm, q/k/v, rotary, cache write, attention, o_proj, residual, norm, router, D2H]
             host:{sync, submit CPU experts}  host:{wait}  [H2D, residual]
  -> [final norm, lm_head, argmax, position += 1]
Consecutive GPU work is captured as one CUDA graph; host actions run between replays.
q/k/v projections, rotary, norms, router, o_proj, lm_head are the model's own modules. Attention is a Triton
GQA decode kernel that reads the number of valid keys from device memory and always walks key blocks
from position 0 in fixed order (fp32 online softmax), so one graph serves every length, and replay == eager.
COLOSSUS_GRAPH_EAGER=1 runs the same sequence without graphs (the exactness reference);
with COLOSSUS_EXPERTS=gpu (eager only) the experts run on the GPU twin instead of the CPU worker.
"""
from __future__ import annotations

import os
import time

import torch
import triton
import triton.language as tl

from . import hybrid as _hy


@triton.jit
def _gqa_decode(q, k, v, out, lenp, scale, skt, skh, G: tl.constexpr, D: tl.constexpr,
                BG: tl.constexpr, BT: tl.constexpr):
    kh = tl.program_id(0).to(tl.int64)
    L = tl.load(lenp).to(tl.int64)                       # valid keys: 0 .. L-1
    g = tl.arange(0, BG)
    gm = g < G
    d = tl.arange(0, D)
    qv = tl.load(q + (kh * G + g)[:, None] * D + d[None, :], mask=gm[:, None], other=0.0)
    m_i = tl.full((BG,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BG,), dtype=tl.float32)
    acc = tl.zeros((BG, D), dtype=tl.float32)
    tt = tl.arange(0, BT)
    for t0 in range(0, L, BT):
        t = (t0 + tt).to(tl.int64)
        tm = t < L
        kk = tl.load(k + t[:, None] * skt + kh * skh + d[None, :], mask=tm[:, None], other=0.0)
        s = tl.dot(qv, tl.trans(kk)) * scale
        s = tl.where(tm[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        vv = tl.load(v + t[:, None] * skt + kh * skh + d[None, :], mask=tm[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(vv.dtype), vv)
        m_i = m_new
    o = acc / l_i[:, None]
    tl.store(out + (kh * G + g)[:, None] * D + d[None, :], o.to(out.dtype.element_ty), mask=gm[:, None])


def gqa_decode(q, K, V, lenbuf, scale):
    """q [n_heads, D] bf16; K, V [cap, n_kv, D] bf16; lenbuf int32 [1] on device. Returns [n_heads, D]."""
    nh, D = q.shape
    nkv = K.shape[1]
    out = torch.empty_like(q)
    _gqa_decode[(nkv,)](q, K, V, out, lenbuf, scale, K.stride(0), K.stride(1),
                        G=nh // nkv, D=D, BG=16, BT=64, num_warps=4, num_stages=2)
    return out


def mixtral_graph_decode(model, cache, next_token, max_new: int, eos_id, device):
    """Returns (new token ids incl. next_token, per-step latencies, stats)."""
    from transformers.models.mixtral import modeling_mixtral as mm
    m = model.model
    layers = list(m.layers)
    cfg = model.config
    H = cfg.hidden_size
    nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    D = getattr(cfg, "head_dim", None) or H // nh
    dt = m.embed_tokens.weight.dtype
    use_graph = os.environ.get("COLOSSUS_GRAPH_EAGER", "0") != "1"
    gpu_experts = os.environ.get("COLOSSUS_EXPERTS", "cpu") == "gpu"
    if gpu_experts and use_graph:
        raise ValueError("COLOSSUS_EXPERTS=gpu is the eager reference; set COLOSSUS_GRAPH_EAGER=1")
    numa = int(os.environ.get("COLOSSUS_HYBRID_NUMA", "1"))
    wl = [l.mlp for l in layers]
    if not gpu_experts:
        for w in wl:
            if not w.registered:
                w._register()

    # ---- static KV cache from the prefill cache
    T0 = cache.get_seq_length()
    cap = T0 + max_new + 1
    KV = []
    for li in range(len(layers)):
        lk = cache.layers[li].keys[0].transpose(0, 1)      # [T0, n_kv, D]
        lv = cache.layers[li].values[0].transpose(0, 1)
        Kb = torch.zeros((cap, nkv, D), dtype=dt, device=device)
        Vb = torch.zeros((cap, nkv, D), dtype=dt, device=device)
        Kb[:T0].copy_(lk)
        Vb[:T0].copy_(lv)
        KV.append((Kb, Vb))
    tok = torch.zeros((1, 1), dtype=torch.long, device=device)
    tok.copy_(next_token.view(1, 1))
    pos = torch.full((1, 1), T0, dtype=torch.long, device=device)
    lenbuf = torch.zeros((1,), dtype=torch.int32, device=device)
    h = torch.zeros((1, 1, H), dtype=dt, device=device)
    r = torch.zeros((1, 1, H), dtype=dt, device=device)
    od = torch.zeros((1, H), dtype=dt, device=device)
    K2 = cfg.num_experts_per_tok
    xb = _hy.pinned("mgx", (1, H), dt)
    ob = _hy.pinned("mgout", (1, H), dt)
    tib = _hy.pinned("mgti", (1, K2), torch.int64)
    twb = _hy.pinned("mgtw", (1, K2), torch.bfloat16)
    ev = torch.cuda.Event()
    pe = {}

    def embed():
        h.copy_(m.embed_tokens(tok))
        pe["cs"] = m.rotary_emb(h, pos)
        lenbuf.copy_((pos.view(1) + 1).to(torch.int32))

    def attn(li):
        layer = layers[li]
        a = layer.self_attn
        r.copy_(h)
        x = layer.input_layernorm(h)
        q = a.q_proj(x).view(1, 1, nh, D).transpose(1, 2)
        k = a.k_proj(x).view(1, 1, nkv, D).transpose(1, 2)
        v = a.v_proj(x).view(1, 1, nkv, D).transpose(1, 2)
        cos, sin = pe["cs"]
        q, k = mm.apply_rotary_pos_emb(q, k, cos, sin)
        Kb, Vb = KV[li]
        Kb.index_copy_(0, pos.view(1), k.reshape(1, nkv, D))
        Vb.index_copy_(0, pos.view(1), v.reshape(1, nkv, D))
        o = gqa_decode(q.reshape(nh, D), Kb, Vb, lenbuf, a.scaling)
        y = a.o_proj(o.view(1, 1, nh * D))
        h.copy_(r + y)

    def moe_pre(li):
        w = wl[li]
        r.copy_(h)
        y = layers[li].post_attention_layernorm(h)
        flat = y.reshape(1, H)
        _, tw, ti = w.gate(flat)
        xb.copy_(flat, non_blocking=True)
        tib.copy_(ti.reshape(1, K2), non_blocking=True)
        twb.copy_(tw.to(torch.bfloat16).reshape(1, K2), non_blocking=True)

    def host_submit(li):
        ev.record()
        ev.synchronize()
        _hy.ext().submit_moe(li, xb, tib, twb, ob, numa)

    def moe_post(li):
        od.copy_(ob, non_blocking=True)
        h.copy_(r + od.view(1, 1, H))

    def moe_gpu(li):                                   # eager reference: experts on the GPU twin
        r.copy_(h)
        y = layers[li].post_attention_layernorm(h)
        h.copy_(r + wl[li](y))

    def final():
        logits = model.lm_head(m.norm(h))
        tok.copy_(logits[:, -1, :].argmax(dim=-1, keepdim=True))
        pos.add_(1)

    seq = [("gpu", embed)]
    for li in range(len(layers)):
        seq.append(("gpu", lambda li=li: attn(li)))
        if gpu_experts:
            seq.append(("gpu", lambda li=li: moe_gpu(li)))
        else:
            seq.append(("gpu", lambda li=li: moe_pre(li)))
            seq.append(("host", lambda li=li: host_submit(li)))
            seq.append(("host", lambda: _hy.ext().wait_done()))
            seq.append(("gpu", lambda li=li: moe_post(li)))
    seq.append(("gpu", final))
    items = []
    for kind, fn in seq:
        if kind == "gpu" and items and items[-1][0] == "gpu":
            items[-1][1].append(fn)
        elif kind == "gpu":
            items.append(["gpu", [fn], None])
        else:
            items.append(["host", fn])

    def step():
        for it in items:
            if it[0] == "host":
                it[1]()
            elif it[2] is not None:
                it[2].replay()
            else:
                for fn in it[1]:
                    fn()

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

    out = [int(next_token.view(-1)[0])]
    lat = []
    stats = {"graph": use_graph, "capture_s": None, "graphs": sum(1 for it in items if it[0] == "gpu"),
             "host_items": sum(1 for it in items if it[0] == "host"), "cap": cap, "gpu_experts": gpu_experts}
    _hy.pool()
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
                # step 0 ran eagerly (warms / compiles every kernel); capture records without executing,
                # so the cache and position are untouched by it.
                tc = time.perf_counter()
                capture()
                torch.cuda.synchronize()
                stats["capture_s"] = time.perf_counter() - tc
    return out, lat, stats
