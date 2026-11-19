"""M0 spike: one CUDA graph per decode step, CPU expert jobs as host nodes.

Synthetic MoE stack (L layers, H=4096, E=8, top-2, I=2048, random BF16 experts as NX12): per layer the GPU does
norm -> router -> top-k, the CPU kernel computes the experts, the GPU adds the result to the residual.
Arms: eager (Python: sync, submit_moe, wait_done, copy back) vs graph (whole step captured; host node per layer).
Also: noop-host-node graph vs GPU-only graph -> per-host-node overhead.
usage: m0_spike.py OUT.json [layers] [steps]
"""
import json
import os
import sys
import time

import torch

from vahana import hybrid as hy

OUT = sys.argv[1]
L = int(sys.argv[2]) if len(sys.argv) > 2 else 32
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
H, I, E, K, NUMA = 4096, 2048, 8, 2, 2
dev = torch.device("cuda")
ext = hy.ext()
g = torch.Generator().manual_seed(0)
res = {"layers": L, "steps": STEPS, "H": H, "I": I, "E": E, "K": K, "omp": os.environ.get("OMP_NUM_THREADS")}

t = time.time()
keep = []
for l in range(L):
    gs = [(torch.randn(I, H, generator=g) * 0.02).bfloat16().contiguous() for _ in range(E)]
    us = [(torch.randn(I, H, generator=g) * 0.02).bfloat16().contiguous() for _ in range(E)]
    ds = [(torch.randn(H, I, generator=g) * 0.02).bfloat16().contiguous() for _ in range(E)]
    ext.register_layer(l, gs, us, ds)
    packs = []
    for e in range(E):
        for W in (gs[e], us[e], ds[e]):
            packs += list(ext.encode_nx(W))
    ext.register_layer_nx(l, packs)
    keep.append((gs, us, ds, packs))
res["setup_s"] = time.time() - t
print("setup", res["setup_s"], flush=True)

Wr = [(torch.randn(E, H, generator=g) * 0.05).bfloat16().to(dev) for _ in range(L)]
nw = [(1 + 0.1 * torch.randn(H, generator=g)).bfloat16().to(dev) for _ in range(L)]
h = torch.zeros(1, H, dtype=torch.bfloat16, device=dev)               # static residual state
h0 = (torch.randn(1, H, generator=g)).bfloat16().to(dev)
pin = lambda *s, dt=torch.bfloat16: torch.empty(*s, dtype=dt).pin_memory()
xb = [pin(1, H) for _ in range(L)]
tib = [pin(1, K, dt=torch.int64) for _ in range(L)]
twb = [pin(1, K) for _ in range(L)]
ob = [pin(1, H) for _ in range(L)]
od = torch.empty(1, H, dtype=torch.bfloat16, device=dev)


def gpu_pre(l):
    hf = h.float()
    x = (hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + 1e-6)).bfloat16() * nw[l]
    p = torch.softmax(torch.nn.functional.linear(x, Wr[l]).float(), dim=-1)
    tw, ti = torch.topk(p, K, dim=-1)
    tw = (tw / tw.sum(-1, keepdim=True)).bfloat16()
    xb[l].copy_(x, non_blocking=True)
    tib[l].copy_(ti, non_blocking=True)
    twb[l].copy_(tw, non_blocking=True)


def gpu_post(l):
    od.copy_(ob[l], non_blocking=True)
    h.copy_(h + od)


def step_eager():
    km = 0.0
    for l in range(L):
        gpu_pre(l)
        torch.cuda.current_stream().synchronize()
        ext.submit_moe(l, xb[l], tib[l], twb[l], ob[l], NUMA)
        ext.wait_done()
        km += ext.kernel_ms()
        gpu_post(l)
    return km


jobs = [ext.make_host_job(l, xb[l], tib[l], twb[l], ob[l], NUMA, False) for l in range(L)]
noops = [ext.make_host_job(l, xb[l], tib[l], twb[l], ob[l], NUMA, True) for l in range(L)]


def step_graphbody(kind):
    st = torch.cuda.current_stream().cuda_stream
    for l in range(L):
        gpu_pre(l)
        if kind == "real":
            ext.enqueue_host_job(st, jobs[l])
        elif kind == "noop":
            ext.enqueue_host_job(st, noops[l])
        gpu_post(l)


def run_eager(n):
    h.copy_(h0)
    states, km, lat = [], [], []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        km.append(step_eager())
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
        states.append(h.clone())
    return states, km, lat


def capture(kind):
    gr = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(gr, stream=s):
            step_graphbody(kind)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    return gr


def run_graph(gr, n):
    h.copy_(h0)
    states, lat = [], []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gr.replay()
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
        states.append(h.clone())
    return states, lat


med = lambda v: sorted(v)[len(v) // 2]
# warm-up (creates the worker, compiles kernels), then main thread off CPU 0 (Phase 41)
run_eager(2)
mc = os.environ.get("VAHANA_MAIN_CPUS")
if mc:
    os.sched_setaffinity(0, {int(c) for c in mc.split(",")})

eag, km, lat_e = run_eager(STEPS)
c0 = ext.host_calls()
g_real = capture("real")
calls_capture = ext.host_calls() - c0
grs, lat_g = run_graph(g_real, STEPS)
calls_replay = ext.host_calls() - c0
same = [torch.equal(a, b) for a, b in zip(eag, grs)]
# second eager pass: eager is itself reproducible
eag2, _, _ = run_eager(STEPS)
same_ee = all(torch.equal(a, b) for a, b in zip(eag, eag2))

g_gpu = capture("gpu")
g_noop = capture("noop")
_, lat_gpu = run_graph(g_gpu, STEPS)
_, lat_noop = run_graph(g_noop, STEPS)
res.update({
    "exact_graph_eq_eager_every_step": all(same), "per_step_equal": same, "eager_reproducible": same_ee,
    "host_calls_during_capture": calls_capture, "host_calls_per_replay": (calls_replay - calls_capture) / STEPS,
    "eager_step_ms_median": med(lat_e) * 1e3, "graph_step_ms_median": med(lat_g) * 1e3,
    "cpu_kernel_ms_per_step_median": med(km),
    "gpu_only_graph_ms_median": med(lat_gpu) * 1e3, "noop_hostnode_graph_ms_median": med(lat_noop) * 1e3,
    "per_host_node_overhead_us": (med(lat_noop) - med(lat_gpu)) * 1e6 / L,
    "eager_minus_kernel_ms": med(lat_e) * 1e3 - med(km), "graph_minus_kernel_ms": med(lat_g) * 1e3 - med(km),
})
for k, v in res.items():
    if k != "per_step_equal":
        print(k, v, flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print("DONE", flush=True)
