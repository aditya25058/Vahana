"""M1 follow-up diagnostics (not bars): per family
  experts_matter     max |logits(twin) - logits(twin with expert output zeroed)| / max|logits|  (does the MoE
                     output reach the logits at all? if ~0, that family's exactness check is weak)
  call_rel           one experts call, same input: max |stock - twin| / max|stock|   (per-call semantic error)
usage: m1_checks.py OUT.json [family ...]
"""
import json
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import m1_families as M  # noqa: E402  (reuses build / tiny_config; its module body only runs under __main__ guard)
from vahana import experts as vx  # noqa: E402
from vahana.spec import inspect, parent_of  # noqa: E402

OUT = sys.argv[1]
FAMS = sys.argv[2:] or M.TIER1
res = {}
for fam in FAMS:
    cfg, m = M.build(fam)
    specs = inspect(m)
    s0 = specs[0]
    stock_ex = getattr(*parent_of(m, s0.path))
    twin = vx.TwinGPUExperts(s0, stock_ex.gate_up_proj, stock_ex.down_proj)
    g = torch.Generator(device="cuda").manual_seed(5)
    N, K = 16, cfg.num_experts_per_tok
    x = torch.randn(N, s0.H, generator=g, device="cuda").bfloat16()
    ti = torch.stack([torch.randperm(s0.E, generator=g, device="cuda")[:K] for _ in range(N)])
    tw = torch.softmax(torch.randn(N, K, generator=g, device="cuda"), -1)
    with torch.no_grad():
        a, b = stock_ex(x, ti, tw).float(), twin(x, ti, tw).float()
    call_rel = (a - b).abs().max().item() / max(a.abs().max().item(), 1e-9)
    ids = torch.randint(0, cfg.vocab_size, (1, 12), generator=torch.Generator().manual_seed(7)).to("cuda")
    orig = {s.path: getattr(*parent_of(m, s.path)) for s in specs}
    vx.install(m, mode="gpu_twin")
    with torch.no_grad():
        l1 = m(ids).logits.float()
        for s in specs:
            mod = getattr(*parent_of(m, s.path))
            mod.forward = (lambda f: (lambda h, i, w: f(h, i, w) * 0))(mod.forward)
        l0 = m(ids).logits.float()
    matter = (l1 - l0).abs().max().item() / max(l1.abs().max().item(), 1e-9)
    res[fam] = {"experts_matter": matter, "call_rel": call_rel, "stock_out_maxabs": a.abs().max().item()}
    print(fam, res[fam], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
