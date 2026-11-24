"""v0.3-d UNIT bar: the multi-token twin linear equals the per-token kernel (_lin) bit for bit.
usage: v3d_unit.py [OUT.json]"""
import json
import sys

import torch

from vahana.twinlin import gpu_linear

g = torch.Generator().manual_seed(0)
res, ok_all = [], True
for R, K in ((2048, 2048), (64, 2048), (50304, 2048), (4096, 2048), (2048, 4096), (100, 48)):
    W = (torch.randn(R, K, generator=g) * 0.02).cuda().bfloat16()
    for n in (2, 3, 5, 8, 17, 33, 64, 333):
        x = (torch.randn(n, K, generator=g) * 2).cuda().bfloat16()
        ref = gpu_linear(x, W, MT=1).view(torch.int16)
        for mt in (None, 2, 4, 8):
            ok = torch.equal(gpu_linear(x, W, MT=mt).view(torch.int16), ref)
            ok_all &= ok
            res.append({"R": R, "K": K, "n": n, "MT": mt, "equal": ok})
            if not ok:
                print("MISMATCH", R, K, n, mt, flush=True)
print("UNIT", "PASS" if ok_all else "FAIL", sum(r["equal"] for r in res), "/", len(res), flush=True)
if len(sys.argv) > 1:
    json.dump({"pass": ok_all, "cases": res}, open(sys.argv[1], "w"), indent=1)
