"""v1.4: physical core count and the CPU-only thread default."""
import os
from argparse import Namespace

from vahana import cli, osl


def test_physical_cores_sane():
    pc = osl.physical_cores()
    assert pc is None or 1 <= pc <= len(osl.cpus())


def test_cpu_only_default(monkeypatch):
    monkeypatch.delenv("VAHANA_THREADS", raising=False)
    for pc, want in ((8, 8), (2, 16), (None, 16)):
        monkeypatch.setattr(osl, "physical_cores", lambda cs=None, _p=pc: _p)
        monkeypatch.setattr(osl, "cpus", lambda: list(range(16)))
        monkeypatch.delenv("VAHANA_THREADS", raising=False)
        info = cli._setup(Namespace(threads=None, device="cpu", keep_numa_balancing=True))
        assert info["threads"] == want, (pc, info)
