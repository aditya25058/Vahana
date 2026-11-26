"""vahana bench (v1.2): what this machine does with this model, measured, in one command.

  vahana bench MODEL_DIR [--prompt-tokens 512] [--tokens 64] [--gpu-budget GB] [--device cpu|cuda] [--json out.json]

Loads the model exactly as `vahana run` does, then reports
  - load time, time to first token for a prompt of --prompt-tokens tokens, prefill tokens/s;
  - decode tokens/s and per-token latency (p50 / p95);
  - peak GPU memory and the process's resident RAM;
  - exactness, checked rather than claimed: the run writes a receipt and verifies it on the spot (one teacher-forced
    pass must reproduce every step's logits bit for bit), and two tiny-model selftest families must match the
    reference hashes recorded on other machines.
Everything goes to --json as well, with the machine description, so results can be compared and published.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

TEXT = ("The history of the Roman Empire spans more than a thousand years, from the founding of the city to the fall "
        "of Constantinople. Historians divide it into the Kingdom, the Republic and the Empire, and each period left "
        "its own institutions, roads, laws and languages across Europe, North Africa and the Near East. ")


def _rss_gb():
    try:
        if sys.platform.startswith("linux"):
            for line in open("/proc/self/status"):
                if line.startswith("VmHWM:"):                       # peak resident set
                    return int(line.split()[1]) / 2**20
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            k32 = ctypes.WinDLL("kernel32")
            psapi = ctypes.WinDLL("psapi")
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return c.PeakWorkingSetSize / 2**30
    except Exception:  # noqa: BLE001
        pass
    return None


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("vahana bench")
    ap.add_argument("model")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--gpu-budget", type=float, default=None)
    ap.add_argument("--gpu-cache", default="auto")
    ap.add_argument("--reserve", type=float, default=1.5)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--keep-numa-balancing", action="store_true")
    ap.add_argument("--no-exact", action="store_true", help="skip the receipt check and the selftest families")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    from . import cli
    setup = cli._setup(a)                      # before torch: process-wide thread / NUMA setup
    t0 = time.time()
    model, tok, torch = cli._load(a, setup)
    load_s = time.time() - t0
    from . import __version__, decode, doctor
    from . import receipt as rc
    dev = model._vahana_device
    ids = tok(TEXT * (a.prompt_tokens // 40 + 2), return_tensors="pt").input_ids[:, :a.prompt_tokens].to(dev)
    decode.generate(model, ids[:, :32], max_new_tokens=4)           # warm: kernels, graph capture, first page-ins
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"vahana bench: prompt {ids.shape[1]} tokens, decoding {a.tokens}", file=sys.stderr, flush=True)
    r = decode.generate(model, ids, max_new_tokens=a.tokens, return_logits=not a.no_exact)
    st = r["step_ms"]
    res = {
        "vahana": __version__, "model": os.path.basename(os.path.normpath(a.model)),
        "model_type": model.config.get_text_config().model_type,
        "device": str(dev), "load_s": round(load_s, 1),
        "prompt_tokens": int(ids.shape[1]), "ttft_s": round(r["prefill_s"], 3),
        "prefill_tok_s": round(ids.shape[1] / r["prefill_s"], 1),
        "decode_tokens": len(st) + 1,
        "decode_tok_s": round(len(st) / (sum(st) / 1e3), 2) if st else None,
        "step_ms_p50": round(_pct(st, 0.5), 1) if st else None,
        "step_ms_p95": round(_pct(st, 0.95), 1) if st else None,
        "cuda_graph": r["graph_used"],
        "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else None,
        "ram_peak_gb": None,
    }
    if not a.no_exact:
        rec = rc.make(model, a.model, ids, r, a.tokens, None)
        t = time.time()
        v = rc.verify(model, rec, a.model)
        res["receipt_verify"] = "PASS" if v["ok"] else f"FAIL {v['checks']}"
        res["receipt_verify_s"] = round(time.time() - t, 2)
        res["receipt_id"] = rec["receipt_id"]
        res["logits_chain"] = rec["logits"]["chain"]
        from . import selftest
        import transformers
        gold, _ = selftest.expected_hashes(json.load(open(selftest.GOLDEN)), transformers.__version__)
        st_ok = {}
        for fam in ("olmoe", "qwen3_moe"):
            h, aud = selftest.run(fam, dev)
            st_ok[fam] = h == gold.get(fam) and not aud
        res["selftest"] = "PASS" if all(st_ok.values()) else f"FAIL {st_ok}"
    res["ram_peak_gb"] = round(_rss_gb() or 0, 2) or None
    res["machine"] = doctor.machine(probe=False)

    w = 22
    print(f"\nvahana bench  {res['model']} ({res['model_type']}) on {res['device']}")
    rows = [("load", f"{res['load_s']} s"),
            ("time to first token", f"{res['ttft_s']} s  ({res['prompt_tokens']}-token prompt, "
                                    f"{res['prefill_tok_s']} tok/s prefill)"),
            ("decode", f"{res['decode_tok_s']} tok/s  (p50 {res['step_ms_p50']} ms, p95 {res['step_ms_p95']} ms per "
                       f"token, {'CUDA graph' if res['cuda_graph'] else 'eager'})"),
            ("GPU peak", f"{res['gpu_peak_gb']} GB" if res["gpu_peak_gb"] is not None else "-"),
            ("RAM peak", f"{res['ram_peak_gb']} GB" if res["ram_peak_gb"] else "-")]
    if not a.no_exact:
        rows += [("receipt verify", f"{res['receipt_verify']}  ({res['receipt_verify_s']} s to check "
                                    f"{res['decode_tokens']} tokens)"),
                 ("selftest (2 families)", res["selftest"])]
    m = res["machine"]
    rows.append(("machine", f"{m.get('cpu')}, {m.get('kernel')}"
                            + (f", {m['gpu']['name']}" if m.get("gpu") else "")))
    for k, v in rows:
        print(f"  {k:{w}s} {v}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1, default=str)
        print(f"  wrote {a.json}")
    return 0 if all(not str(res.get(k, "PASS")).startswith("FAIL") for k in ("receipt_verify", "selftest")) else 1


if __name__ == "__main__":
    sys.exit(main())
