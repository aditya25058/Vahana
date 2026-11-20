"""P3: kernel-level identity through the C ABI (vahana._kernel). Same inputs and hashes as P1.
kernel path, a hash per output. The same hashes on every CPU / ISA / OS mean the same bits.
usage: p1_hash.py BUILD_DIR OUT.json
"""
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch

build, out_path = sys.argv[1], sys.argv[2]
from vahana._kernel import Kernel, lib_name

k = Kernel(os.path.join(build, lib_name()))

rs = np.random.RandomState(1234)


def bf16(shape, scale):
    a = (rs.standard_normal(shape) * scale).astype(np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return torch.from_numpy(r.view(np.int16).copy()).view(torch.bfloat16)


def h(t):
    return hashlib.sha256(t.contiguous().view(torch.int16 if t.dtype == torch.bfloat16 else t.dtype).numpy().tobytes()
                          ).hexdigest()[:16]


H, I, E, K, N = 256, 128, 4, 2, 5
g = [bf16((I, H), 0.05) for _ in range(E)]
u = [bf16((I, H), 0.05) for _ in range(E)]
d = [bf16((H, I), 0.05) for _ in range(E)]
x1 = bf16((E, 1, H), 1.0)
x3 = bf16((E, 3, H), 1.0)
xl = bf16((3, 512), 1.0)
W = bf16((96, 512), 0.05)
xt = bf16((N, H), 1.0)
ti = torch.from_numpy(np.array([[0, 2], [1, 3], [3, 0], [2, 1], [0, 1]], dtype=np.int64))
tw = torch.from_numpy(rs.uniform(0.1, 0.9, (N, K)).astype(np.float32))
res = {"isa": k.isa(), "nx_fast": k.nx_fast(), "torch": torch.__version__}
res["experts_n1"] = h(k.expert_mlps(x1, g, u, d, 0)[0])
res["experts_n3"] = h(k.expert_mlps(x3, g, u, d, 0)[0])
res["linear_n1"] = h(k.linear_twin(xl[:1].contiguous(), W))
res["linear_n3"] = h(k.linear_twin(xl, W))
packs = [t for e in range(E) for M in (g[e], u[e], d[e]) for t in k.encode_nx(M)]
res["nx_experts_n1"] = h(k.expert_mlps_nx(x1, packs, I, 0))
k.register_layer(0, g, u, d)
k.register_layer_nx(0, packs)
out = torch.empty(N, H, dtype=torch.bfloat16)
k.submit_moe(0, xt, ti, tw, out, 0)
k.wait_done()
res["moe_N5_fp32w"] = h(out)
out1 = torch.empty(1, H, dtype=torch.bfloat16)
k.submit_moe(0, xt[:1].contiguous(), ti[:1].contiguous(), tw[:1].contiguous(), out1, 0)
k.wait_done()
res["moe_N1_fused"] = h(out1)
k.register_layer(1, g, u, d)
bg = [torch.from_numpy(rs.standard_normal(I).astype(np.float32) * 0.1) for _ in range(E)]
bu = [torch.from_numpy(rs.standard_normal(I).astype(np.float32) * 0.1) for _ in range(E)]
bd = [torch.from_numpy(rs.standard_normal(H).astype(np.float32) * 0.1) for _ in range(E)]
k.register_layer_act(1, 1, 1.702, 7.0, bg, bu, bd)
out2 = torch.empty(N, H, dtype=torch.bfloat16)
k.submit_moe(1, xt, ti, tw, out2, 0)
k.wait_done()
res["moe_oai_bias"] = h(out2)
json.dump(res, open(out_path, "w"), indent=1)
print(json.dumps(res))
