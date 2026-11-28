"""vahana command line.

  vahana run   MODEL_DIR --prompt "..." [--gpu-budget GB] [--max-tokens N]
  vahana serve MODEL_DIR [--gpu-budget GB] [--port 8000] [--host 127.0.0.1]
  vahana selftest [--device cpu|cuda]        tiny models of 18 families: same bits as the reference H100?
  vahana doctor [MODEL_DIR ...]              this machine, and how each model would run here (and how fast)
  vahana run MODEL_DIR --prompt "..." --receipt r.json     also write a receipt (verifiable inference, v1.2)
  vahana verify r.json --model MODEL_DIR     re-run a receipt here: same logits and tokens, bit for bit?
  vahana bench MODEL_DIR [--json out.json]   time to first token, decode speed, memory, and exactness checks

Process setup happens here, before torch is imported, because it must apply to every thread:
  - automatic NUMA balancing is turned off for the process (MPOL_BIND over all nodes; it stalls the CPU expert
    kernel by milliseconds), unless --keep-numa-balancing;
  - the kernel's thread team: VAHANA_THREADS (default: CPUs in the affinity set minus 2), bound one per core, and a
    spin time before sleeping (10 ms with a GPU, 1 ms CPU-only); VAHANA_CPUS records the affinity set before torch's
    OpenMP pins the main thread; torch's own OpenMP team is 1 thread CPU-only (its glue ops are small);
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
    import platform
    nr = {"x86_64": 238, "aarch64": 237}.get(platform.machine())          # set_mempolicy's syscall number
    if nr is None:
        return f"NUMA balancing left on (no set_mempolicy number for {platform.machine()})"
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.syscall(nr, 2, mask, ctypes.c_ulong(16 * 64 + 1)) != 0:       # set_mempolicy(MPOL_BIND)
        # containers often forbid it (Docker's default seccomp profile, Kaggle): a speed hint, never fatal
        return f"NUMA balancing left on (set_mempolicy refused: {os.strerror(ctypes.get_errno())})"
    return f"NUMA balancing off (MPOL_BIND nodes {sorted(nodes)})"


def _device(args) -> str:
    """cuda or cpu, decided before torch is imported."""
    if args.device:
        return args.device
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return "cpu"
    if sys.platform.startswith("linux"):
        return "cuda" if os.path.exists("/dev/nvidiactl") else "cpu"
    if sys.platform == "win32":                       # the NVIDIA driver installs nvcuda.dll
        return "cuda" if os.path.exists(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32",
                                                     "nvcuda.dll")) else "cpu"
    return "cpu"                                      # macOS: no CUDA


def _setup(args) -> dict:
    args.device = _device(args)
    from .osl import cpus as _cpus
    cpus = _cpus()
    os.environ.setdefault("VAHANA_CPUS", ",".join(map(str, cpus)))     # before torch's OpenMP pins this thread
    gpu = args.device == "cuda"
    if args.threads:
        threads = args.threads
    elif gpu:
        threads = max(1, len(cpus) - 2)            # two CPUs stay free to feed the GPU
    else:
        # v1.4 CPU-only: one thread per physical core (laptop, 8 cores / 16 CPUs: 8 threads 2.75 tok/s, 14: 2.40,
        # 16: 2.10); with under 4 cores, every CPU (Kaggle, 2 cores / 4 CPUs: 4 threads 1.38 tok/s, 2: 1.20)
        from .osl import physical_cores
        pc = physical_cores(cpus)
        threads = pc if pc and pc >= 4 else len(cpus)
    os.environ.setdefault("VAHANA_THREADS", str(threads))          # the kernel's own thread team (pool.h)
    # torch's OpenMP team: with a GPU it barely runs; CPU-only it runs only small glue ops, and a second team of
    # `threads` on the same cores would fight the kernel's (measured: 1 thread is fastest)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads) if gpu else "1")
    os.environ.setdefault("OMP_PROC_BIND", "true")
    os.environ.setdefault("OMP_PLACES", "cores")
    os.environ.setdefault("GOMP_SPINCOUNT", "10000000" if gpu else "1000")
    os.environ.setdefault("VAHANA_SPIN_US", "10000" if gpu else "1000")
    os.environ.setdefault("VAHANA_FAST_GATE", "1")
    info = {"threads": int(os.environ["VAHANA_THREADS"]), "cpus": len(cpus)}
    if not args.keep_numa_balancing and os.path.exists("/sys/devices/system/node/online"):
        info["numa"] = _numa_off()
    info["main_cpus"] = cpus[info["threads"]:][:2] if len(cpus) > info["threads"] else []
    return info


def _load(args, setup):
    import torch
    from . import loader
    device = args.device
    gc = getattr(args, "gpu_cache", "auto")
    gc = None if gc in (None, "0", "off", "none") else (gc if gc == "auto" else float(gc))
    model, info = loader.load(args.model, gpu_budget_gb=args.gpu_budget, reserve_gb=args.reserve, device=device,
                              gpu_cache_gb=gc if device == "cuda" else None,
                              log=lambda m: print(m, file=sys.stderr, flush=True))
    if setup["main_cpus"] and device != "cpu":    # CPU-only: the main thread computes too, keep every core
        from .osl import set_cpus
        set_cpus(setup["main_cpus"])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"vahana: loaded {info['model_type']} on {model._vahana_device} in {info['load_s']:.0f}s, "
          f"GPU {info['gpu_gb']:.1f} GB, "
          f"{len(info['cpu_modules'])} modules in CPU RAM, {setup['threads']} CPU threads", file=sys.stderr)
    return model, tok, torch


def _verify(args) -> int:
    import json
    import time
    from . import receipt as _rc
    rec = _rc.load(args.receipt)
    args.model_dir = args.model
    setup = _setup(args)
    model, _, _ = _load(args, setup)
    t = time.time()
    v = _rc.verify(model, rec, args.model, args.mode)
    v["seconds"] = round(time.time() - t, 2)
    prod = rec.get("produced_on", {})
    print(f"vahana verify: {'PASS' if v['ok'] else 'FAIL'}  ({v['steps']} tokens, {v['mode']}, {v['seconds']}s)")
    print(f"  produced on {prod.get('device')} {prod.get('isa', '')} {prod.get('platform')}; checked on "
          f"{v['checked_on'].get('device')} {v['checked_on'].get('isa', '')} {v['checked_on'].get('platform')}")
    for k, ok in v["checks"].items():
        print(f"  {k:11s} {'ok' if ok else 'MISMATCH'}")
    if v["first_bad_step"] is not None:
        print(f"  first differing step: {v['first_bad_step']}")
    if os.environ.get("VAHANA_VERIFY_JSON"):
        print(json.dumps(v))
    return 0 if v["ok"] else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] in (["--version"], ["-V"]):
        from . import __version__
        print(f"vahana {__version__}")
        return 0
    if argv[:1] == ["selftest"]:                      # same bits as the reference machine? (tiny models, no download)
        from .selftest import main as selftest
        return selftest(argv[1:])
    if argv[:1] == ["bench"]:                         # speed, memory and exactness on this machine
        from .bench import main as bench
        return bench(argv[1:])
    if argv[:1] == ["doctor"]:                        # this machine, and which models it runs (and how fast)
        from .doctor import main as doctor
        return doctor(argv[1:])
    p = argparse.ArgumentParser(
        "vahana", description="Exact inference for MoE models too big for your GPU",
        epilog="also: `vahana selftest` (does this machine give the reference bits?), `vahana doctor [MODEL] "
               "[--measure]` (what this machine runs, where the weights go, how fast), `vahana verify RECEIPT --model "
               "MODEL` (check an answer bit for bit), `vahana bench MODEL` (speed, memory, exactness), `vahana --version`")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "serve", "verify"):
        s = sub.add_parser(name)
        if name == "verify":
            s.add_argument("receipt", help="receipt JSON written by `vahana run --receipt`")
            s.add_argument("--model", required=True, help="local model directory (the same weights)")
            s.add_argument("--mode", default="auto", choices=["auto", "prefill", "replay"],
                           help="prefill: one teacher-forced pass (attention models); replay: decode again")
        else:
            s.add_argument("model", help="local model directory (safetensors + config)")
        s.add_argument("--gpu-budget", type=float, default=None, help="GPU memory budget in GB (default: no limit)")
        s.add_argument("--reserve", type=float, default=1.5, help="GB of the budget kept for KV cache / workspace")
        s.add_argument("--threads", type=int, default=None, help="CPU threads for the expert kernel")
        s.add_argument("--keep-numa-balancing", action="store_true")
        s.add_argument("--gpu-cache", default="auto",
                       help="GB of spare GPU memory for the most-used experts ('auto' = what is free, 0 = off); "
                            "speed only, the same bits")
        s.add_argument("--device", default=None, choices=["cuda", "cpu"],
                       help="default: cuda if an NVIDIA GPU is present; cpu gives the same tokens, slower")
        if name == "run":
            s.add_argument("--prompt", required=True)
            s.add_argument("--max-tokens", type=int, default=128)
            s.add_argument("--chat", action="store_true", help="apply the tokenizer's chat template")
            s.add_argument("--receipt", default=None, help="write a receipt (prompt, settings, output, per-step "
                                                           "logits digests) that anyone can check with `vahana verify`")
            s.add_argument("--temperature", type=float, default=0.0, help="0 = greedy; >0 samples with Vahana's "
                                                                          "exact sampler (same token on every machine)")
            s.add_argument("--prefix-cache", nargs="?", const="", default=None, metavar="DIR",
                           help="reuse the attention state of earlier prompts that start the same way (stored on disk, "
                                "default ~/.cache/vahana/prefix): only the new part is prefilled, same bits")
            s.add_argument("--top-p", type=float, default=1.0)
            s.add_argument("--seed", type=int, default=0)
        elif name == "serve":
            s.add_argument("--host", default="127.0.0.1")
            s.add_argument("--port", type=int, default=8000)
            s.add_argument("--name", default=None, help="model name reported by the API")
            s.add_argument("--slots", type=int, default=16,
                           help="requests decoded together (continuous batching; 1 = one at a time). Every response "
                                "is bit-identical to running it alone")
            s.add_argument("--max-len", type=int, default=4096, help="KV columns shared by the slots (prompt + "
                                                                     "max_tokens must fit)")
    args = p.parse_args(argv)
    if args.cmd == "verify":
        return _verify(args)
    setup = _setup(args)
    model, tok, torch = _load(args, setup)
    if args.cmd == "run":
        from . import decode
        text = args.prompt
        if args.chat:
            text = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                           add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").input_ids.to(model._vahana_device)
        from . import receipt as _rc
        smp = _rc.make_sampler(args.temperature, args.top_p, args.seed)
        pc = None
        if args.prefix_cache is not None:
            from . import prefix as _prefix
            pc = _prefix.PrefixCache(args.prefix_cache or None, model_key=_prefix.model_key(args.model))
        r = decode.generate(model, ids, max_new_tokens=args.max_tokens, eos_token_id=tok.eos_token_id,
                            sampler=smp, return_logits=bool(args.receipt), prefix_cache=pc)
        if pc is not None:
            print(f"[prefix cache: reused {pc.last['hit_tokens']} of {ids.shape[1]} prompt tokens, stored "
                  f"{pc.last['stored_tokens']}]", file=sys.stderr)
        print(tok.decode(r["ids"][0], skip_special_tokens=True))
        if args.receipt:
            rec = _rc.make(model, args.model, ids, r, args.max_tokens, tok.eos_token_id, smp)
            _rc.save(rec, args.receipt)
            print(f"[receipt {rec['receipt_id'][:16]} written to {args.receipt}]", file=sys.stderr)
        st = r["step_ms"]
        if st:
            print(f"\n[{len(st) + 1} tokens, {len(st) / (sum(st) / 1e3):.2f} tok/s decode, prefill "
                  f"{r['prefill_s']:.2f}s, graph {'on' if r['graph_used'] else 'off: ' + str(r['graph_fallback'])}]",
                  file=sys.stderr)
        return 0
    from .server import serve
    serve(model, tok, host=args.host, port=args.port, name=args.name or os.path.basename(args.model.rstrip("/")),
          slots=args.slots, max_len=args.max_len)
    return 0


if __name__ == "__main__":
    sys.exit(main())
