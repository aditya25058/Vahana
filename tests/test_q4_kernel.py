"""v1.4: the twin-q4 CPU expert kernel gives the same bits under every instruction-set build (scalar C is the
reference), for one token (the fused decode path) and for several (the row path), and whatever the task size."""
import json
import os
import subprocess
import sys

import pytest

SCRIPT = r'''
import hashlib, json, sys, torch
from vahana import hybrid, q4
from vahana.experts import VahanaExperts
from vahana.modelspec import ExpertsSpec
g = torch.Generator().manual_seed(0)
E, H, I = 8, 512, 256
mk = lambda r, c: q4.from_bf16((torch.randn(r, c, generator=g) * 0.05).to(torch.bfloat16))
G = [mk(I, H) for _ in range(E)]; U = [mk(I, H) for _ in range(E)]; D = [mk(H, I) for _ in range(E)]
spec = ExpertsSpec(path="x", layer=0, slot=0, cls="test", layout="fused_concat", act="silu_glu", E=E, I=I, H=H)
m = VahanaExperts(spec, weights=(G, U, D, None, None, None), nx=False)
out = {}
for n in (1, 3, 17):
    x = (torch.randn(n, H, generator=g) * 0.5).to(torch.bfloat16)
    ti = torch.stack([torch.randperm(E, generator=g)[:4] for _ in range(n)])
    tw = torch.softmax(torch.randn(n, 4, generator=g), -1)
    y = m(x, ti, tw)
    out[n] = hashlib.sha256(y.view(torch.int16).numpy().tobytes()).hexdigest()
print(json.dumps({"isa": hybrid.ext().isa(), "h": out}))
'''


def _run(env_extra):
    env = dict(os.environ, **env_extra)
    r = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def _isas():
    """Kernel builds in this install that this CPU can run."""
    import vahana
    from vahana import hybrid
    d = os.path.join(os.path.dirname(vahana.__file__), "_build")
    have = set(os.listdir(d)) if os.path.isdir(d) else set()
    os.environ.pop("VAHANA_ISA", None)
    best = hybrid.cpu_isa()
    order = ["scalar", "avx2", "avx512"] if best != "neon" else ["scalar", "neon"]
    return [i for i in order[: order.index(best) + 1] if i in have]


def test_q4_kernel_same_bits_every_isa_and_task_size():
    isas = _isas()
    if "scalar" not in isas:
        pytest.skip("no scalar build to compare against")
    ref = _run({"VAHANA_ISA": "scalar"})
    for isa in isas:
        for tr in ("16", "32", "64"):
            got = _run({"VAHANA_ISA": isa, "VAHANA_Q4_TASK_ROWS": tr})
            assert got["h"] == ref["h"], (isa, tr, got, ref)
