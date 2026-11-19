"""Vahana runtime: the CPU-side operations (expert jobs, twin linears, embedding rows) and the linear dispatch.

Every CPU operation runs one of two ways with identical arithmetic:
  eager   copy inputs to the CPU, run the job on the persistent worker, copy the result back (synchronous);
  graph   while a CUDA graph is being captured (GRAPH["on"]): device->pinned copies, a host node that runs the job
          (csrc make_host_* / enqueue_host_job), pinned->device copy; no Python on the replay path.
Linears: CPU-resident weights use the CPU twin linear; GPU weights without bias use the GPU twin linear (same bits
as the CPU one); anything else (biases, fp32) uses the batch-invariant GEMM.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import hybrid as _hy
from .invariant import inv_linear
from .twinlin import gpu_linear

GRAPH = {"on": False}
ROWS = 16                    # rows per CPU job (the kernel's per-expert / per-linear batch limit)
_BUF: dict = {}


def _pin(shape, dtype):
    return torch.empty(shape, dtype=dtype).pin_memory()


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def weights_arg(w: torch.Tensor) -> torch.Tensor:
    """Routing weights as given: fp32 stays fp32, bf16 stays bf16, anything else converts exactly to fp32."""
    return w.detach() if w.dtype in (torch.float32, torch.bfloat16) else w.detach().float()


def moe(slot: int, x: torch.Tensor, ti: torch.Tensor, tw: torch.Tensor, numa: int) -> torch.Tensor:
    """Weighted sum of the selected experts of layer `slot` (plus its folded shared expert, if registered)."""
    k = _hy.ext()
    tw = weights_arg(tw)
    N, H = x.shape
    if not GRAPH["on"]:
        xc = x.detach().to("cpu", torch.bfloat16).contiguous()
        tic = ti.detach().to("cpu", torch.int64).contiguous()
        twc = tw.to("cpu").contiguous()
        out = torch.empty((N, H), dtype=torch.bfloat16)
        for s in range(0, N, ROWS):               # rows are independent: chunking never changes bits
            k.submit_moe(slot, xc[s:s + ROWS], tic[s:s + ROWS], twc[s:s + ROWS], out[s:s + ROWS], numa)
            k.wait_done()
        return out.to(x.device)
    assert N <= ROWS, "graph decode: at most 16 rows per step"
    key = ("moe", slot, N, ti.shape[1], tw.dtype)
    if key not in _BUF:
        b = (_pin((N, H), torch.bfloat16), _pin(tuple(ti.shape), torch.int64), _pin(tuple(tw.shape), tw.dtype),
             _pin((N, H), torch.bfloat16))
        _BUF[key] = b + (k.make_host_job(slot, *b, numa, False),)
    xb, tib, twb, ob, job = _BUF[key]
    xb.copy_(x, non_blocking=True)
    tib.copy_(ti, non_blocking=True)
    twb.copy_(tw, non_blocking=True)
    k.enqueue_host_job(_stream(), job)
    out = torch.empty((N, H), dtype=torch.bfloat16, device=x.device)
    out.copy_(ob, non_blocking=True)
    return out


def cpu_linear(W: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
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
    assert n <= ROWS
    key = ("lin", W.data_ptr(), n)
    if key not in _BUF:
        xb, ob = _pin((n, W.shape[1]), torch.bfloat16), _pin((n, R), torch.bfloat16)
        _BUF[key] = (xb, ob, k.make_host_linear(W, xb, ob))
    xb, ob, job = _BUF[key]
    xb.copy_(x2, non_blocking=True)
    k.enqueue_host_job(_stream(), job)
    out = torch.empty((n, R), dtype=torch.bfloat16, device=x2.device)
    out.copy_(ob, non_blocking=True)
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
    idb.copy_(ids.reshape(-1), non_blocking=True)
    _hy.ext().enqueue_host_job(_stream(), job)
    out = torch.empty((n, W.shape[1]), dtype=W.dtype, device=device)
    out.copy_(ob, non_blocking=True)
    return out.view(*shp, W.shape[1])


def _linear_forward(self, x):
    W = self.weight
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
        y = _hy.twin_one(x.reshape(-1, shp[-1]).to(torch.bfloat16).contiguous(), self.g, self.u, self.d)
        return y.view(shp).to(x.dtype)
