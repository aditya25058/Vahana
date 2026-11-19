"""M3: single-call accuracy of vahana_attention against an fp32 reference (eager math, same mask / sinks /
soft-cap semantics as transformers), plus within-call invariance (each row alone == inside the batch).
usage: m3_attn_unit.py OUT.json
"""
import json
import sys

import torch

from vahana.attention import vahana_attention

dev = torch.device("cuda")
g = torch.Generator(device=dev).manual_seed(0)


def ref(q, k, v, allowed, scale, sinks=None, softcap=None):
    B, H, Q, D = q.shape
    Hkv = k.shape[1]
    kk = k.float().repeat_interleave(H // Hkv, 1)
    vv = v.float().repeat_interleave(H // Hkv, 1)
    s = q.float() @ kk.transpose(-1, -2) * scale
    if softcap:
        s = softcap * torch.tanh(s / softcap)
    s = s.masked_fill(~allowed, float("-inf"))
    if sinks is not None:
        s = torch.cat([s, sinks.float().view(1, H, 1, 1).expand(B, H, Q, 1)], -1)
    p = torch.softmax(s, -1)
    if sinks is not None:
        p = p[..., :-1]
    p = torch.nan_to_num(p)
    return (p @ vv).transpose(1, 2)                         # [B, Q, H, DV]


def masks(B, Q, T, pads, window=None):
    qpos = torch.arange(T - Q, T, device=dev)[:, None]
    kpos = torch.arange(T, device=dev)[None, :]
    m = (kpos <= qpos)
    if window:
        m = m & (kpos > qpos - window)
    out = m[None, None].expand(B, 1, Q, T).clone()
    for b, pad in enumerate(pads):
        out[b, :, :, :pad] = False
    return out


CASES = {
    "gqa_causal_leftpad": dict(B=3, H=4, Hkv=2, Q=40, T=40, D=64, DV=64, pads=[0, 9, 21]),
    "gqa_decode": dict(B=3, H=4, Hkv=2, Q=1, T=40, D=64, DV=64, pads=[0, 9, 21]),
    "mqa_sliding16": dict(B=2, H=4, Hkv=1, Q=40, T=40, D=64, DV=64, pads=[0, 5], window=16),
    "sinks_sliding16": dict(B=2, H=4, Hkv=2, Q=40, T=40, D=64, DV=64, pads=[0, 7], window=16, sinks=True),
    "sinks_full": dict(B=2, H=4, Hkv=2, Q=40, T=40, D=64, DV=64, pads=[0, 7], sinks=True),
    "mla_shape_192_128": dict(B=2, H=4, Hkv=4, Q=40, T=40, D=192, DV=128, pads=[0, 11]),
    "softcap30": dict(B=2, H=4, Hkv=2, Q=40, T=40, D=64, DV=64, pads=[0, 3], softcap=30.0),
    "long_T300_decode": dict(B=2, H=4, Hkv=2, Q=1, T=300, D=64, DV=64, pads=[0, 100]),
}
res = {}
for name, c in CASES.items():
    B, H, Hkv, Q, T, D, DV = (c[k] for k in ("B", "H", "Hkv", "Q", "T", "D", "DV"))
    q = torch.randn(B, H, Q, D, generator=g, device=dev).bfloat16()
    k = torch.randn(B, Hkv, T, D, generator=g, device=dev).bfloat16()
    v = torch.randn(B, Hkv, T, DV, generator=g, device=dev).bfloat16()
    sinks = torch.randn(H, generator=g, device=dev) if c.get("sinks") else None
    allowed = masks(B, Q, T, c["pads"], c.get("window"))
    scale = D ** -0.5
    out, _ = vahana_attention(None, q, k, v, allowed, scaling=scale, s_aux=sinks, softcap=c.get("softcap"))
    r = ref(q, k, v, allowed, scale, sinks, c.get("softcap"))
    valid = allowed.any(-1)[:, 0]                       # [B, Q] query rows that attend to something
    err = (out.float() - r).abs()[valid].max().item()
    # invariance inside the call: each row alone equals its slice of the batched call
    alone = all(torch.equal(vahana_attention(None, q[b:b + 1], k[b:b + 1], v[b:b + 1], allowed[b:b + 1],
                                             scaling=scale, s_aux=sinks, softcap=c.get("softcap"))[0][0], out[b])
                for b in range(B))
    res[name] = {"max_abs_err_vs_fp32": err, "ref_max": r[valid].abs().max().item(), "row_alone_equal": alone}
    print(name, res[name], flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
