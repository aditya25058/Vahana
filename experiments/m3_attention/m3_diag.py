"""M3 diagnostic (not a bar): which Vahana component moves a family's logits away from stock?
For each family, prompt 4 (40 tokens) prefill logits, max |delta| / max |stock| for:
  attn    only attention.use(model)
  experts only experts.install(model, "cpu")
  norms   only the RMSNorm part of invariant.install (instance forwards; no class-level Linear patch)
  linear  only the class-level nn.Linear -> inv_linear patch (applied last: it is process-wide)
Also lists the norm classes invariant.install would replace and their forward signatures.
usage: m3_diag.py OUT.json family ...
"""
import inspect
import json
import sys
import types

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0].replace("m3_attention", "m2_kernel"))
import m2_families as M  # noqa: E402
from vahana import attention as va  # noqa: E402
from vahana import experts as vx  # noqa: E402
from vahana import invariant as inv  # noqa: E402

OUT, FAMS = sys.argv[1], sys.argv[2:]
dev = torch.device("cuda")


def logits(m, p):
    with torch.no_grad():
        return m(p[None].to(dev)).logits.float()[0]


def rel(a, b):
    return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-9)


def norms_only(m):
    n = 0
    for mod in m.modules():
        if hasattr(mod, "variance_epsilon") and isinstance(getattr(mod, "weight", None), torch.Tensor):
            mod.forward = types.MethodType(lambda self, x: inv.inv_rmsnorm(x, self.weight, self.variance_epsilon), mod)
            n += 1
    return n


res = {}
stage = {}
for fam in FAMS:
    cfg, m = M.build(fam)
    p = [torch.randint(1, cfg.vocab_size, (L,), generator=torch.Generator().manual_seed(11)) for L in (7, 19, 33, 40)][3]
    base = logits(m, p)
    r = {"norm_classes": sorted({f"{type(x).__name__}{inspect.signature(type(x).forward)}" for x in m.modules()
                                 if hasattr(x, "variance_epsilon") and isinstance(getattr(x, "weight", None), torch.Tensor)})}
    for name, fn in (("attn", va.use), ("experts", lambda mm: vx.install(mm, "cpu")), ("norms", norms_only)):
        _, mm = M.build(fam)
        try:
            fn(mm)
            r[name] = rel(logits(mm, p), base)
        except Exception as e:  # noqa: BLE001
            r[name] = f"{type(e).__name__}: {str(e)[:150]}"
    stage[fam] = (base, p)
    res[fam] = r
torch.nn.Linear.forward = lambda self, x: inv.inv_linear(x, self.weight, self.bias)
for fam in FAMS:
    _, mm = M.build(fam)
    base, p = stage[fam]
    res[fam]["linear"] = rel(logits(mm, p), base)
    print(fam, res[fam], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
