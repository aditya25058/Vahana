"""Vahana runtime: the CPU-side operations (expert jobs, twin linears, embedding rows) and the linear dispatch.

Every CPU operation runs one of two ways with identical arithmetic:
  eager   copy inputs to the CPU, run the job on the persistent worker, copy the result back (synchronous);
  graph   while a CUDA graph is being captured (GRAPH["on"]): one mailbox kernel per job (v0.3) moves the inputs into
          pinned host memory, signals the kernel's CPU polling thread and waits, then moves the output back (~12 us per
          round trip; VAHANA_MAILBOX=0 uses the older device->pinned copies + CUDA host node + copy back, ~1.4 ms).
          No Python on the replay path.
Linears: CPU-resident weights use the CPU twin linear; GPU weights without bias use the GPU twin linear (same bits
as the CPU one); anything else (biases, fp32) uses the batch-invariant GEMM.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from . import hybrid as _hy
from ._kernel import raw as _raw
from ._triton import tl, triton
from .invariant import inv_linear
from .twinlin import gpu_linear

GRAPH = {"on": False}
TWIN = {"on": False}         # P2: every linear in twin arithmetic (set by loader when twin=True)
ROWS = 16                    # rows per CPU job (the kernel's per-expert / per-linear batch limit)
MOE_ROWS = int(os.environ.get("VAHANA_MOE_ROWS", "64"))   # v0.4-a: rows per CPU expert job (kernel limit 256); chunks never change bits
_BUF: dict = {}


def _pin(shape, dtype):
    return torch.empty(shape, dtype=dtype).pin_memory()


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


# ---- v0.3 mailbox (default in graph decode; VAHANA_MAILBOX=0 uses CUDA host nodes instead)
_MB: dict = {}
_MAXJ = 4096


@triton.jit
def _mb_copy(src, dst, n, BLK: tl.constexpr):
    for s0 in range(0, n, BLK):
        i = s0 + tl.arange(0, BLK)
        m = i < n
        tl.store(dst + i, tl.load(src + i, mask=m, volatile=True), mask=m)


@triton.jit
def _mb_rt(counter, req, done, job, d0, h0, n0, d1, h1, n1, d2, h2, n2, d3, h3, n3, oh, od, no, BLK: tl.constexpr):
    """One CPU round trip in stream order: inputs device -> pinned host (UVA stores), publish seq for `job` (system-
    scope release), poll until the CPU thread stores done[job] = seq, acquire, then output pinned host -> device."""
    _mb_copy(d0, h0, n0, BLK)
    _mb_copy(d1, h1, n1, BLK)
    _mb_copy(d2, h2, n2, BLK)
    _mb_copy(d3, h3, n3, BLK)
    seq = tl.atomic_add(counter + job, 1) + 1
    tl.atomic_xchg(req + job, seq, sem="release", scope="sys")
    cur = tl.load(done + job, volatile=True)
    while cur != seq:
        cur = tl.load(done + job, volatile=True)
    tl.atomic_add(done + job, 0, sem="acquire", scope="sys")
    _mb_copy(oh, od, no, BLK)


@triton.jit
def _mb_post(counter, req, job, d0, h0, n0, d1, h1, n1, BLK: tl.constexpr):
    """v1.1: the first half of a round trip: inputs device -> pinned host, publish seq for `job`; does not wait, so the
    GPU goes on with other work while the CPU runs the job (_mb_wait collects it)."""
    _mb_copy(d0, h0, n0, BLK)
    _mb_copy(d1, h1, n1, BLK)
    seq = tl.atomic_add(counter + job, 1) + 1
    tl.atomic_xchg(req + job, seq, sem="release", scope="sys")


@triton.jit
def _mb_wait(counter, done, job, oh, od, no, BLK: tl.constexpr):
    """v1.1: the second half: poll until the CPU thread stores done[job] = seq (the seq _mb_post published), acquire,
    then output pinned host -> device."""
    seq = tl.load(counter + job, volatile=True)
    cur = tl.load(done + job, volatile=True)
    while cur != seq:
        cur = tl.load(done + job, volatile=True)
    tl.atomic_add(done + job, 0, sem="acquire", scope="sys")
    _mb_copy(oh, od, no, BLK)


def overlap_on() -> bool:
    """v1.1: GPU-cached and CPU experts of a layer run at the same time (graph decode, mailbox, kernel support).
    VAHANA_OVERLAP=0 restores the sequential v0.3 path; both give the same bits."""
    return (os.environ.get("VAHANA_OVERLAP", "1") != "0" and mailbox_on() and GRAPH["on"]
            and getattr(_hy.ext(), "has_cpu_slots", False))


def mailbox_on() -> bool:
    import os
    return os.environ.get("VAHANA_MAILBOX", "1") != "0"


def _words(t: torch.Tensor) -> torch.Tensor:
    """A flat view for word copies: int32 when the byte count allows, else uint8."""
    f = t.reshape(-1)
    return f.view(torch.int32) if (f.numel() * f.element_size()) % 4 == 0 else f.view(torch.uint8)


def roundtrip(job: int, ins, oh: torch.Tensor, od: torch.Tensor) -> None:
    """Run host job `job` on (device input, pinned host buffer) pairs `ins` and copy its pinned output `oh` into the
    device tensor `od`, in stream order (capturable). Mailbox by default; VAHANA_MAILBOX=0: copies + a host node."""
    k = _hy.ext()
    if not mailbox_on():
        for d, h in ins:
            h.copy_(d, non_blocking=True)
        k.enqueue_host_job(_stream(), job)
        od.copy_(oh, non_blocking=True)
        return
    if not _MB:
        dev = torch.device("cuda", torch.cuda.current_device())
        _MB["counter"] = torch.zeros(_MAXJ, dtype=torch.int32, device=dev)
        _MB["req"] = torch.zeros(_MAXJ, dtype=torch.int32).pin_memory()
        _MB["done"] = torch.zeros(_MAXJ, dtype=torch.int32).pin_memory()
        k.mailbox_start(_MB["req"], _MB["done"])
    assert job < _MAXJ, "too many host jobs for the mailbox"
    args = []
    for d, h in ins:
        dv = _words(d.to(h.dtype).contiguous())
        hv = _words(h)
        assert dv.dtype == hv.dtype and dv.numel() == hv.numel()
        args += [dv, hv, dv.numel()]
    while len(args) < 12:
        args += [_MB["counter"], _MB["counter"], 0]
    ov, hov = _words(od), _words(oh)
    _mb_rt[(1,)](_MB["counter"], _MB["req"], _MB["done"], job, *args, hov, ov, ov.numel(), BLK=1024)


def weights_arg(w: torch.Tensor) -> torch.Tensor:
    """Routing weights as given: fp32 stays fp32, bf16 stays bf16, anything else converts exactly to fp32."""
    return w.detach() if w.dtype in (torch.float32, torch.bfloat16) else w.detach().float()


def moe(slot: int, x: torch.Tensor, ti: torch.Tensor, tw: torch.Tensor, numa: int, _chunk: int = 0) -> torch.Tensor:
    """Weighted sum of the selected experts of layer `slot` (plus its folded shared expert, if registered)."""
    k = _hy.ext()
    tw = weights_arg(tw)
    N, H = x.shape
    if not GRAPH["on"]:
        xc = x.detach().to("cpu", torch.bfloat16).contiguous()
        tic = ti.detach().to("cpu", torch.int64).contiguous()
        twc = tw.to("cpu").contiguous()
        out = torch.empty((N, H), dtype=torch.bfloat16)
        for s in range(0, N, MOE_ROWS):               # rows are independent: chunking never changes bits
            k.submit_moe(slot, xc[s:s + MOE_ROWS], tic[s:s + MOE_ROWS], twc[s:s + MOE_ROWS], out[s:s + MOE_ROWS], numa)
            k.wait_done()
        return out.to(x.device)
    if N > MOE_ROWS:                                  # one host job per MOE_ROWS-row chunk (rows are independent)
        return torch.cat([moe(slot, x[s:s + MOE_ROWS], ti[s:s + MOE_ROWS], tw[s:s + MOE_ROWS], numa, _chunk=s)
                          for s in range(0, N, MOE_ROWS)])
    key = ("moe", slot, N, ti.shape[1], tw.dtype, _chunk)
    if key not in _BUF:
        b = (_pin((N, H), torch.bfloat16), _pin(tuple(ti.shape), torch.int64), _pin(tuple(tw.shape), tw.dtype),
             _pin((N, H), torch.bfloat16))
        _BUF[key] = b + (k.make_host_job(slot, *b, numa, False),)
    xb, tib, twb, ob, job = _BUF[key]
    out = torch.empty((N, H), dtype=torch.bfloat16, device=x.device)
    roundtrip(job, [(x, xb), (ti, tib), (tw, twb)], ob, out)
    return out


def moe_cached(slot: int, x: torch.Tensor, ti: torch.Tensor, tw: torch.Tensor, yg: torch.Tensor,
               numa: int, _chunk: int = 0) -> torch.Tensor:
    """v0.3 (GPU expert cache): slots whose expert is not GPU-resident (the set registered with register_resident) run
    on the CPU kernel; the resident ones arrive in yg [N, K, H] (GPU, twin arithmetic). The CPU kernel combines all
    slots as moe_fused does."""
    k = _hy.ext()
    tw = weights_arg(tw)
    N, H = x.shape
    K = ti.shape[1]
    if not GRAPH["on"]:
        c = lambda t, dt=None: t.detach().to("cpu", dt or t.dtype).contiguous()
        xc, tic, twc, ygc = c(x, torch.bfloat16), c(ti, torch.int64), c(tw), c(yg)
        out = torch.empty((N, H), dtype=torch.bfloat16)
        for s in range(0, N, MOE_ROWS):
            k.submit_moe_cached(slot, xc[s:s + MOE_ROWS], tic[s:s + MOE_ROWS], twc[s:s + MOE_ROWS], None, ygc[s:s + MOE_ROWS],
                                out[s:s + MOE_ROWS], numa)
            k.wait_done()
        return out.to(x.device)
    if N > MOE_ROWS:                                  # one host job per MOE_ROWS-row chunk (rows are independent)
        return torch.cat([moe_cached(slot, x[s:s + MOE_ROWS], ti[s:s + MOE_ROWS], tw[s:s + MOE_ROWS], yg[s:s + MOE_ROWS], numa,
                                     _chunk=s) for s in range(0, N, MOE_ROWS)])
    key = ("cached", slot, N, K, tw.dtype, _chunk)
    if key not in _BUF:
        b = (_pin((N, H), torch.bfloat16), _pin((N, K), torch.int64), _pin((N, K), tw.dtype),
             _pin((N, K, H), torch.bfloat16), _pin((N, H), torch.bfloat16))
        _BUF[key] = b + (k.make_host_cached(slot, b[0], b[1], b[2], None, b[3], b[4], numa),)
    xb, tib, twb, ygb, ob, job = _BUF[key]
    out = torch.empty((N, H), dtype=torch.bfloat16, device=x.device)
    roundtrip(job, [(x, xb), (ti, tib), (tw, twb), (yg, ygb)], ob, out)
    return out


def _mb_init():
    k = _hy.ext()
    if not _MB:
        dev = torch.device("cuda", torch.cuda.current_device())
        _MB["counter"] = torch.zeros(_MAXJ, dtype=torch.int32, device=dev)
        _MB["req"] = torch.zeros(_MAXJ, dtype=torch.int32).pin_memory()
        _MB["done"] = torch.zeros(_MAXJ, dtype=torch.int32).pin_memory()
        k.mailbox_start(_MB["req"], _MB["done"])


def cpu_slots_post(slot: int, x: torch.Tensor, ti: torch.Tensor, numa: int) -> list:
    """v1.1 (graph only): start the CPU part of a GPU-cached layer: the slots whose expert is not GPU-resident, computed
    by the CPU kernel into [N, K, H] (zeros elsewhere). Returns handles for cpu_slots_wait; the GPU is free meanwhile."""
    k = _hy.ext()
    _mb_init()
    N, H = x.shape
    K = ti.shape[1]
    hs = []
    for s0 in range(0, N, MOE_ROWS):                  # one job per MOE_ROWS-row chunk (rows are independent)
        n = min(MOE_ROWS, N - s0)
        key = ("cslots", slot, n, K, s0)
        if key not in _BUF:
            b = (_pin((n, H), torch.bfloat16), _pin((n, K), torch.int64), _pin((n, K, H), torch.bfloat16))
            _BUF[key] = b + (k.make_host_cpu_slots(slot, b[0], b[1], b[2], numa),)
        xb, tib, yb, job = _BUF[key]
        assert job < _MAXJ, "too many host jobs for the mailbox"
        dv, hv = _words(x[s0:s0 + n].to(torch.bfloat16).contiguous()), _words(xb)
        tv, thv = _words(ti[s0:s0 + n].to(torch.int64).contiguous()), _words(tib)
        _mb_post[(1,)](_MB["counter"], _MB["req"], job, dv, hv, dv.numel(), tv, thv, tv.numel(), BLK=1024)
        hs.append((job, yb, n))
    return hs


def cpu_slots_wait(hs: list, device) -> torch.Tensor:
    outs = []
    for job, yb, n in hs:
        out = torch.empty(tuple(yb.shape), dtype=torch.bfloat16, device=device)
        ov, hov = _words(out), _words(yb)
        _mb_wait[(1,)](_MB["counter"], _MB["done"], job, hov, ov, ov.numel(), BLK=1024)
        outs.append(out)
    return outs[0] if len(outs) == 1 else torch.cat(outs)


@_raw          # v1.4: allocation, copies and kernel calls only (no TwinMode-handled op): skip its Python dispatch
def cpu_linear(W: torch.Tensor, x2: torch.Tensor, _chunk: int = 0) -> torch.Tensor:
    """y [n, R] = twin linear(x2 [n, K] bf16, W [R, K] bf16 in CPU RAM)."""
    k = _hy.ext()
    n, R = x2.shape[0], W.shape[0]
    if not GRAPH["on"]:
        xc = x2.detach().to("cpu", torch.bfloat16).contiguous()
        outs = []
        for s in range(0, n, ROWS):
            ob = torch.empty((min(ROWS, n - s), R), dtype=torch.bfloat16)
            k.submit_linear(xc[s:s + ROWS], W, ob)
            k.wait_done()
            outs.append(ob)
        return torch.cat(outs).to(x2.device)
    if n > ROWS:                                  # one host job per MOE_ROWS-row chunk (rows are independent)
        return torch.cat([cpu_linear(W, x2[s:s + ROWS], _chunk=s) for s in range(0, n, ROWS)])
    key = ("lin", W.data_ptr(), n, _chunk)
    if key not in _BUF:
        xb, ob = _pin((n, W.shape[1]), torch.bfloat16), _pin((n, R), torch.bfloat16)
        _BUF[key] = (xb, ob, k.make_host_linear(W, xb, ob))
    xb, ob, job = _BUF[key]
    out = torch.empty((n, R), dtype=torch.bfloat16, device=x2.device)
    roundtrip(job, [(x2, xb)], ob, out)
    return out


def cpu_embed(W: torch.Tensor, ids: torch.Tensor, device) -> torch.Tensor:
    """Rows of an embedding table held in CPU RAM (exact copies)."""
    shp = tuple(ids.shape)
    if not GRAPH["on"]:
        return torch.nn.functional.embedding(ids.to("cpu"), W).to(device)
    n = ids.numel()
    key = ("emb", W.data_ptr(), n)
    if key not in _BUF:
        idb, ob = _pin((n,), torch.int64), _pin((n, W.shape[1]), W.dtype)
        _BUF[key] = (idb, ob, _hy.ext().make_host_embed(W, idb, ob))
    idb, ob, job = _BUF[key]
    out = torch.empty((n, W.shape[1]), dtype=W.dtype, device=device)
    roundtrip(job, [(ids.reshape(-1).to(torch.int64), idb)], ob, out)
    return out.view(*shp, W.shape[1])


def twin_linear(x2: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """bf16 twin linear wherever W lives (CPU kernel or GPU Triton twin: the same bits). x2 [n, K], K % 16 == 0,
    R % 4 == 0."""
    if W.device.type == "cpu":
        return cpu_linear(W, x2)
    return gpu_linear(x2.contiguous(), W)


def _linear_forward(self, x):
    W = self.weight
    if TWIN["on"]:
        from .tmath import tlinear
        return tlinear(x, W, self.bias)
    shp = x.shape
    if W.device.type == "cpu":
        if self.bias is not None:
            raise RuntimeError("CPU-placed linears must not have a bias")
        y = cpu_linear(W, x.reshape(-1, shp[-1]).to(torch.bfloat16))
        return y.view(*shp[:-1], W.shape[0]).to(x.dtype)
    if self.bias is None and W.dtype == torch.bfloat16 and x.dtype == torch.bfloat16 and W.shape[1] % 16 == 0 \
            and W.shape[0] % 2 == 0:
        return gpu_linear(x.reshape(-1, shp[-1]).contiguous(), W).view(*shp[:-1], W.shape[0])
    return inv_linear(x, W, self.bias)


def install_linear() -> None:
    """Class-level: every nn.Linear (now or created later) uses the dispatch above."""
    nn.Linear.forward = _linear_forward


class CPUEmbedding(nn.Module):
    def __init__(self, weight: torch.Tensor, device):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.device_ = device

    def forward(self, ids):
        return cpu_embed(self.weight, ids, self.device_)


class SharedExpert(nn.Module):
    """A shared expert (plain SiLU-GLU MLP) computed with expert arithmetic, so its bits never depend on where
    it lives: GPU twin (hybrid.twin_one) or folded into the layer's CPU expert job (returns zeros here; the CPU
    job adds it)."""

    def __init__(self, g, u, d, folded: bool):
        super().__init__()
        self.g, self.u, self.d, self.folded = g, u, d, folded

    def forward(self, x):
        if self.folded:
            return torch.zeros_like(x)
        shp = x.shape
        x2 = x.reshape(-1, shp[-1]).to(torch.bfloat16).contiguous()
        if self.g.device.type == "cpu":               # CPU expert kernel (the twin of twin_one), 16 rows per call
            k = _hy.ext()
            ys = [k.expert_mlps(x2[s:s + ROWS].cpu().unsqueeze(0), [self.g], [self.u], [self.d], 0)[0][0]
                  for s in range(0, x2.shape[0], ROWS)]
            y = torch.cat(ys).to(x.device)
        else:
            y = _hy.twin_one(x2, self.g, self.u, self.d)
        return y.view(shp).to(x.dtype)
