"""Phase 26: exact CPU/GPU expert split (device-invariant twin arithmetic).

COLOSSUS_HYBRID=<f> (0..1) turns it on: every routed expert is computed with the twin arithmetic
(16 IEEE-fma lanes in increasing k, fixed lane tree, bf16 RNE at the stock points, shared hex-exact
exp, IEEE division), either by the CPU kernel (cpu_moe.cpp, weights read in place from host memory)
or by the Triton twin on the GPU (weights in an HBM slot). Both give the same bits (Phase 25c), so
WHERE an expert runs never changes the output. Per layer, a fraction f of the experts that are not
already in an HBM slot (and serve <= 16 tokens) runs on the CPU while the rest cross the link.
COLOSSUS_HYBRID_EXT: directory holding the built cpu_moe extension (GIL-releasing build).
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import triton
import triton.language as tl

C_LOG2E, C_LN2HI, C_LN2LO = (float.fromhex("0x1.715476p+0"), float.fromhex("0x1.63p-1"),
                             float.fromhex("-0x1.bd0106p-13"))
P0, P1, P2, P3, P4, P5 = (float.fromhex(h) for h in ("0x1.a0d2cep-13", "0x1.6e878ap-10",
                                                      "0x1.111210p-7", "0x1.555382p-5",
                                                      "0x1.555556p-3", "0x1.0p-1"))

STATS = {"cpu_experts": 0, "gpu_hit_experts": 0, "gpu_link_experts": 0, "cpu_wait_s": 0.0,
         "cpu_submit_s": 0.0, "layers": 0, "prep_s": 0.0, "gpu_issue_s": 0.0, "tail_s": 0.0,
         "cpu_kernel_s": 0.0, "main_threads": None}


def _timed(*a):
    t0 = time.perf_counter()
    r = ext().expert_mlps(*a)
    dt = time.perf_counter() - t0
    STATS["cpu_kernel_s"] += dt
    _EMA["cpu_ms"] = 0.9 * _EMA["cpu_ms"] + 0.1 * dt * 1e3 / a[0].shape[0]
    STATS["cpu_ms_per_expert_ema"] = _EMA["cpu_ms"]
    return r


def enabled() -> bool:
    return os.environ.get("COLOSSUS_HYBRID", "") != ""


def cpu_frac() -> float:
    return float(os.environ.get("COLOSSUS_HYBRID", "0") or 0)


# COLOSSUS_HYBRID=auto: balance the measured CPU time per expert (EMA over kernel calls) against
# the link time per expert (COLOSSUS_HYBRID_LINK_MS). The split depends on timing, which is allowed
# exactly because it cannot change the output bits.
_EMA = {"cpu_ms": float(os.environ.get("COLOSSUS_HYBRID_CPU_MS0", "0.6"))}


def split_count(m: int) -> int:
    v = os.environ.get("COLOSSUS_HYBRID", "0")
    if v != "auto":
        return int(round(float(v or 0) * m))
    tl_ = float(os.environ.get("COLOSSUS_HYBRID_LINK_MS", "0.94"))
    return int(round(m * tl_ / (tl_ + _EMA["cpu_ms"])))


_EXT = None
_POOL = None


def ext():
    global _EXT
    if _EXT is None:
        sys.path.insert(0, os.environ["COLOSSUS_HYBRID_EXT"])
        import cpu_moe  # noqa: E402
        _EXT = cpu_moe
    return _EXT


def pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="colossus-cpu-experts")
        # Main thread's torch ops are tiny here; its own OpenMP team would oversubscribe the
        # cores the expert kernel's team is bound to.
        mt = int(os.environ.get("COLOSSUS_HYBRID_MAIN_THREADS", "1"))
        if mt > 0:
            torch.set_num_threads(mt)
        STATS["main_threads"] = torch.get_num_threads()
    return _POOL


@triton.jit
def _dexp(x, LOG2E: tl.constexpr, LN2HI: tl.constexpr, LN2LO: tl.constexpr, P0: tl.constexpr,
          P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr, P4: tl.constexpr, P5: tl.constexpr):
    x = tl.minimum(tl.maximum(x, -87.0), 87.0)
    n = tl.floor(tl.fma(x, LOG2E, 0.5))
    r = tl.fma(-n, LN2HI, x)
    r = tl.fma(-n, LN2LO, r)
    p = tl.full(x.shape, P0, tl.float32)
    p = tl.fma(p, r, P1)
    p = tl.fma(p, r, P2)
    p = tl.fma(p, r, P3)
    p = tl.fma(p, r, P4)
    p = tl.fma(p, r, P5)
    rr = r * r
    p = tl.fma(p, rr, r)
    p = p + 1.0
    sc = ((n.to(tl.int32) + 127) << 23).to(tl.float32, bitcast=True)
    return p * sc


@triton.jit
def _tree16(acc, BR: tl.constexpr):
    a = tl.sum(tl.reshape(acc, (BR, 2, 8)), axis=1)
    a = tl.sum(tl.reshape(a, (BR, 2, 4)), axis=1)
    a = tl.sum(tl.reshape(a, (BR, 2, 2)), axis=1)
    a = tl.sum(tl.reshape(a, (BR, 2, 1)), axis=1)
    return tl.reshape(a, (BR,))


@triton.jit
def _gate_up(x, g, u, h, H, I, BR: tl.constexpr, LOG2E: tl.constexpr, LN2HI: tl.constexpr,
             LN2LO: tl.constexpr, P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
             P3: tl.constexpr, P4: tl.constexpr, P5: tl.constexpr):
    rb = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    rows = rb * BR + tl.arange(0, BR).to(tl.int64)
    ln = tl.arange(0, 16)
    ag = tl.zeros((BR, 16), dtype=tl.float32)
    au = tl.zeros((BR, 16), dtype=tl.float32)
    xb = x + t * H
    for k0 in range(0, H, 16):
        xv = tl.load(xb + k0 + ln).to(tl.float32)
        wg = tl.load(g + rows[:, None] * H + k0 + ln[None, :]).to(tl.float32)
        wu = tl.load(u + rows[:, None] * H + k0 + ln[None, :]).to(tl.float32)
        ag = tl.fma(wg, xv[None, :], ag)
        au = tl.fma(wu, xv[None, :], au)
    gb = _tree16(ag, BR).to(tl.bfloat16).to(tl.float32)
    ub = _tree16(au, BR).to(tl.bfloat16).to(tl.float32)
    ex = _dexp(-gb, LOG2E, LN2HI, LN2LO, P0, P1, P2, P3, P4, P5)
    s = tl.math.div_rn(gb, 1.0 + ex).to(tl.bfloat16).to(tl.float32)
    tl.store(h + t * I + rows, (s * ub).to(tl.bfloat16))


@triton.jit
def _down(hh, d, y, H, I, BR: tl.constexpr):
    rb = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    rows = rb * BR + tl.arange(0, BR).to(tl.int64)
    ln = tl.arange(0, 16)
    acc = tl.zeros((BR, 16), dtype=tl.float32)
    hb = hh + t * I
    for k0 in range(0, I, 16):
        hv = tl.load(hb + k0 + ln).to(tl.float32)
        w = tl.load(d + rows[:, None] * I + k0 + ln[None, :]).to(tl.float32)
        acc = tl.fma(w, hv[None, :], acc)
    tl.store(y + t * H + rows, _tree16(acc, BR).to(tl.bfloat16))


def twin_one(x: torch.Tensor, G: torch.Tensor, U: torch.Tensor, D: torch.Tensor,
             BR: int = 16) -> torch.Tensor:
    """One expert on the GPU with the CPU kernel's arithmetic. x [n, H] bf16; G/U [I, H]; D [H, I]."""
    x = x.contiguous()
    n, H = x.shape
    I = G.shape[0]
    h = torch.empty((n, I), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((n, H), dtype=torch.bfloat16, device=x.device)
    _gate_up[(I // BR, n)](x, G, U, h, H, I, BR=BR, LOG2E=C_LOG2E, LN2HI=C_LN2HI, LN2LO=C_LN2LO,
                           P0=P0, P1=P1, P2=P2, P3=P3, P4=P4, P5=P5, num_warps=4)
    _down[(H // BR, n)](h, D, y, H, I, BR=BR, num_warps=4)
    return y


class _Pending:
    def result(self):
        r = ext().wait_done()
        ms = ext().kernel_ms()
        STATS["cpu_kernel_s"] += ms / 1e3
        STATS["cpu_calls"] = STATS.get("cpu_calls", 0) + 1
        _EMA["cpu_ms"] = 0.9 * _EMA["cpu_ms"] + 0.1 * ms / max(1, self.E)
        STATS["cpu_ms_per_expert_ema"] = _EMA["cpu_ms"]
        STATS["kernel_phases_ms_setup_p0_p1_tail_calls"] = list(ext().phase_ms())
        return r

    def __init__(self, E):
        self.E = E


def cpu_submit(X: torch.Tensor, W: list):
    """Start the CPU kernel on X [E, n, H] (bf16, host) on the extension's own C++ worker thread
    (no GIL hand-off). COLOSSUS_HYBRID_PYPOOL=1 keeps the old Python-thread path for comparison."""
    t0 = time.perf_counter()
    if os.environ.get("COLOSSUS_HYBRID_PYPOOL") == "1":
        fut = pool().submit(_timed, X, [w["gate"] for w in W], [w["up"] for w in W],
                            [w["down"] for w in W], 1)
    else:
        pool()      # one-time main-thread setup (torch threads)
        ext().submit(X, [w["gate"] for w in W], [w["up"] for w in W], [w["down"] for w in W],
                     int(os.environ.get("COLOSSUS_HYBRID_NUMA", "1")))
        fut = _Pending(X.shape[0])
    STATS["cpu_submit_s"] += time.perf_counter() - t0
    return fut
