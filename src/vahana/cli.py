"""vahana command line.

  vahana run   MODEL_DIR --prompt "..." [--gpu-budget GB] [--max-tokens N]
  vahana serve MODEL_DIR [--gpu-budget GB] [--port 8000] [--host 127.0.0.1]

Process setup happens here, before torch is imported, because it must apply to every thread:
  - automatic NUMA balancing is turned off for the process (MPOL_BIND over all nodes; it stalls the CPU expert
    kernel by milliseconds), unless --keep-numa-balancing;
  - OpenMP: OMP_NUM_THREADS (default: physical cores in the affinity set minus 2), OMP_PROC_BIND=true,
    OMP_PLACES=cores, and a spin count: long with a GPU (the expert team waits for the next job), short CPU-only
    (the main thread's own OpenMP work shares the cores, and spinning teams would starve it: 1.5 vs 18 tok/s);
  - with a GPU, after loading, the main Python thread moves to the two CPUs the OpenMP team does not use.
"""
from __future__ import annotations

import argparse
import os
import sys


def _numa_off() -> str:
    import ctypes
    nodes = set()
    for part in open("/sys/devices/system/node/online").read().strip().split(","):
        lo, _, hi = part.partition("-")
        nodes.update(range(int(lo), int(hi or lo) + 1))
    mask = (ctypes.c_ulong * 16)()
    for n in nodes:
        mask[n // 64] |= 1 << (n % 64)
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.syscall(238, 2, mask, ctypes.c_ulong(16 * 64 + 1)) != 0:      # set_mempolicy(MPOL_BIND), x86_64
        raise OSError(ctypes.get_errno(), "set_mempolicy failed")
    return f"NUMA balancing off (MPOL_BIND nodes {sorted(nodes)})"


def _device(args) -> str:
    """cuda or cpu, decided before torch is imported."""
    if args.device:
        return args.device
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "" or not os.path.exists("/dev/nvidiactl"):
        return "cpu"
    return "cuda"


def _setup(args) -> dict:
    args.device = _device(args)
    cpus = sorted(os.sched_getaffinity(0))
    threads = args.threads or max(1, len(cpus) - 2)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("OMP_PROC_BIND", "true")
    os.environ.setdefault("OMP_PLACES", "cores")
    os.environ.setdefault("GOMP_SPINCOUNT", "10000000" if args.device == "cuda" else "1000")
    os.environ.setdefault("VAHANA_FAST_GATE", "1")
    info = {"threads": int(os.environ["OMP_NUM_THREADS"]), "cpus": len(cpus)}
    if not args.keep_numa_balancing and os.path.exists("/sys/devices/system/node/online"):
        info["numa"] = _numa_off()
    info["main_cpus"] = cpus[info["threads"]:][:2] if len(cpus) > info["threads"] else []
    return info


def _load(args, setup):
    import torch
    from . import loader
    device = args.device
    model, info = loader.load(args.model, gpu_budget_gb=args.gpu_budget, reserve_gb=args.reserve, device=device,
                              log=lambda m: print(m, file=sys.stderr, flush=True))
    if setup["main_cpus"] and device != "cpu":    # CPU-only: the main thread computes too, keep every core
        os.sched_setaffinity(0, set(setup["main_cpus"]))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"vahana: loaded {info['model_type']} on {model._vahana_device} in {info['load_s']:.0f}s, "
          f"GPU {info['gpu_gb']:.1f} GB, "
          f"{len(info['cpu_modules'])} modules in CPU RAM, {setup['threads']} CPU threads", file=sys.stderr)
    return model, tok, torch


def main(argv=None):
    p = argparse.ArgumentParser("vahana", description="Exact inference for MoE models too big for your GPU")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "serve"):
        s = sub.add_parser(name)
        s.add_argument("model", help="local model directory (safetensors + config)")
        s.add_argument("--gpu-budget", type=float, default=None, help="GPU memory budget in GB (default: no limit)")
        s.add_argument("--reserve", type=float, default=1.5, help="GB of the budget kept for KV cache / workspace")
        s.add_argument("--threads", type=int, default=None, help="CPU threads for the expert kernel")
        s.add_argument("--keep-numa-balancing", action="store_true")
        s.add_argument("--device", default=None, choices=["cuda", "cpu"],
                       help="default: cuda if an NVIDIA GPU is present; cpu gives the same tokens, slower")
        if name == "run":
            s.add_argument("--prompt", required=True)
            s.add_argument("--max-tokens", type=int, default=128)
            s.add_argument("--chat", action="store_true", help="apply the tokenizer's chat template")
        else:
            s.add_argument("--host", default="127.0.0.1")
            s.add_argument("--port", type=int, default=8000)
            s.add_argument("--name", default=None, help="model name reported by the API")
    args = p.parse_args(argv)
    setup = _setup(args)
    model, tok, torch = _load(args, setup)
    if args.cmd == "run":
        from . import decode
        text = args.prompt
        if args.chat:
            text = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                           add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").input_ids.to(model._vahana_device)
        r = decode.generate(model, ids, max_new_tokens=args.max_tokens, eos_token_id=tok.eos_token_id)
        print(tok.decode(r["ids"][0], skip_special_tokens=True))
        st = r["step_ms"]
        if st:
            print(f"\n[{len(st) + 1} tokens, {len(st) / (sum(st) / 1e3):.2f} tok/s decode, prefill "
                  f"{r['prefill_s']:.2f}s, graph {'on' if r['graph_used'] else 'off: ' + str(r['graph_fallback'])}]",
                  file=sys.stderr)
        return 0
    from .server import serve
    serve(model, tok, host=args.host, port=args.port, name=args.name or os.path.basename(args.model.rstrip("/")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
