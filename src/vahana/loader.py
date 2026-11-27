"""Turn any supported MoE model into a Vahana model.

load(model_dir, gpu_budget_gb)  checkpoint -> meta model -> Vahana model. Experts are read straight from the
                                checkpoint as zero-copy page-cache views (transformers' own key conversions) and never
                                enter GPU memory or the model's parameters.
prepare(model, ...)             the same for a model already in memory (tests, tiny models).

Placement: everything except the routed experts goes to the GPU unless the budget is short; then shared experts
(folded into the CPU expert job when the block simply adds them), then the input embedding, then the largest
bias-free linears move to CPU RAM. Placement never changes bits: CPU and GPU run the same twin arithmetic, and
shared experts always use expert arithmetic. Finally the invariant layer is installed: twin / invariant linears,
verified RMSNorms, the `vahana` attention function.
"""
from __future__ import annotations

import inspect as _insp
import os
import re
import time

import torch
import torch.nn as nn

from . import attention as _attn
from . import hybrid as _hy
from . import invariant as _inv
from . import runtime as _rt
from .experts import TwinGPUExperts, VahanaExperts, expert_weights
from .modelspec import inspect, parent_of

SHARED_NAMES = ("shared_experts", "shared_expert", "shared_mlp")


def _shared_glu(block: nn.Module):
    """(name, module) of a plain SiLU-GLU shared expert, or None."""
    for n in SHARED_NAMES:
        m = getattr(block, n, None)
        if m is None:
            continue
        lin = [getattr(m, a, None) for a in ("gate_proj", "up_proj", "down_proj")]
        act = type(getattr(m, "act_fn", None)).__name__.lower()
        if all(isinstance(x, nn.Linear) and x.bias is None for x in lin) and act in ("siluactivation", "silu"):
            return n, m
    return None


def _foldable(block: nn.Module, name: str) -> bool:
    """The block adds the shared expert's output to the experts' output directly (no gate, no scaling)."""
    if hasattr(block, "shared_expert_gate"):
        return False
    try:
        src = _insp.getsource(type(block).forward)
    except (OSError, TypeError):
        return False
    return re.search(rf"\+\s*self\.{name}\(", src) is not None


def _cpu_ok(m: nn.Linear) -> bool:
    """Shapes the CPU twin linear accepts (K % 16, R % 4)."""
    return m.weight.shape[1] % 16 == 0 and m.weight.shape[0] % 4 == 0


def _in_shared(name: str) -> bool:
    """Inside a shared-expert container: those weights are placed only by folding, never as loose linears."""
    return any(part in SHARED_NAMES for part in name.split("."))


def _plan(model, specs, budget_bytes, force_cpu: str | None):
    """Modules to put in CPU RAM and shared experts to fold."""
    pref = tuple(s.path + "." for s in specs)
    sizes = {n: p.numel() * 2 for n, p in model.named_parameters() if not n.startswith(pref)}
    total = sum(sizes.values())
    blocks = {s.path.rsplit(".", 1)[0] for s in specs}
    cpu, folded = set(), {}
    out_emb = model.get_output_embeddings()
    emb = model.get_input_embeddings()

    def size_of(mod_name):
        return sum(v for n, v in sizes.items() if n.startswith(mod_name + "."))

    def add_shared(bpath):
        block = model.get_submodule(bpath)
        sh = _shared_glu(block)
        if sh and _foldable(block, sh[0]):
            folded[bpath] = sh
            cpu.add(f"{bpath}.{sh[0]}")
            return size_of(f"{bpath}.{sh[0]}")
        return 0

    if force_cpu:
        rx = re.compile(force_cpu)
        for bpath in sorted(blocks):
            sh = _shared_glu(model.get_submodule(bpath))
            if sh and rx.search(f"{bpath}.{sh[0]}"):
                add_shared(bpath)
        for n, m in model.named_modules():
            if rx.search(n) and (isinstance(m, nn.Embedding) and m is emb and
                                 (out_emb is None or out_emb.weight is not m.weight) or
                                 isinstance(m, nn.Linear) and m.bias is None and m is not out_emb
                                 and _cpu_ok(m) and not _in_shared(n)) \
                    and not any(n.startswith(c + ".") for c in cpu):
                cpu.add(n)
    need = total - budget_bytes - sum(size_of(c) for c in cpu)
    if need > 0:
        for bpath in sorted(blocks, key=lambda b: int(re.search(r"(\d+)", b).group(1)) if re.search(r"\d", b) else 0):
            if need <= 0:
                break
            if bpath not in folded:
                need -= add_shared(bpath)
    if need > 0 and (out_emb is None or out_emb.weight is not emb.weight):
        en = [n for n, m in model.named_modules() if m is emb][0]
        if en not in cpu:
            cpu.add(en)
            need -= emb.weight.numel() * 2
    if need > 0:
        cands = sorted(((m.weight.numel() * 2, n) for n, m in model.named_modules()
                        if isinstance(m, nn.Linear) and m.bias is None and m is not out_emb and n not in cpu
                        and not any(n.startswith(c + ".") for c in cpu) and not n.endswith((".gate", ".router"))
                        and not _in_shared(n) and _cpu_ok(m)), key=lambda t: (-t[0], t[1]))
        for b, n in cands:
            if need <= 0:
                break
            cpu.add(n)
            need -= b
    if need > 0:
        raise MemoryError("GPU budget too small even after CPU placement")
    return cpu, folded, total


def _assemble(model, specs, get_param, get_experts, dev, cpu, folded, numa, nx, place, log, experts="cpu",
              twin=True, get_q4=None):
    pref = tuple(s.path + "." for s in specs)
    on_cpu = lambda mod_name: any(mod_name == c or mod_name.startswith(c + ".") for c in cpu)
    n_q4 = 0
    for n, p in list(model.named_parameters()):
        if n.startswith(pref):
            continue
        mod_name, pname = n.rsplit(".", 1)
        if get_q4 is not None and pname == "weight" and dev.type in ("cuda", "cpu") and not on_cpu(mod_name):
            mod = model.get_submodule(mod_name)
            rec = get_q4(n) if isinstance(mod, nn.Linear) and mod.bias is None else None
            from . import q4lin
            if rec is not None and q4lin.supported(mod.in_features, mod.out_features, dev.type):
                # v1.4: keep a 4-bit linear 4-bit on the GPU (same bits as the bf16 expansion; see q4lin)
                parent, leaf = mod_name.rsplit(".", 1)
                setattr(model.get_submodule(parent), leaf,
                        q4lin.Q4Linear(rec.to(dev), mod.in_features, mod.out_features))
                n_q4 += 1
                continue
        t = get_param(n)
        t = t.to(torch.bfloat16).to("cpu").clone().contiguous() if on_cpu(mod_name) \
            else t.to(dev, torch.bfloat16).contiguous()
        model.get_submodule(mod_name)._parameters[pname] = nn.Parameter(t, requires_grad=False)
    for n, b in list(model.named_buffers()):
        if n.startswith(pref):
            continue
        mod_name, bname = n.rsplit(".", 1) if "." in n else ("", n)
        (model.get_submodule(mod_name) if mod_name else model)._buffers[bname] = b.to(dev)
    _hy.ext()
    t1 = time.time()
    for s in specs:
        bpath = s.path.rsplit(".", 1)[0]
        shared = None
        if bpath in folded:
            sm = folded[bpath][1]
            shared = (sm.gate_proj.weight, sm.up_proj.weight, sm.down_proj.weight)
        parent, leaf = parent_of(model, s.path)
        if experts == "gpu_twin":                 # reference: the same arithmetic with every expert on the GPU
            setattr(parent, leaf, TwinGPUExperts(s, None, weights=get_experts(s), device=dev))
        else:
            setattr(parent, leaf, VahanaExperts(s, None, numa=numa, nx=nx, weights=get_experts(s), place=place,
                                                shared=shared))
        if bpath in folded:
            setattr(model.get_submodule(bpath), folded[bpath][0], _rt.SharedExpert(None, None, None, folded=True))
        if log and (s.slot % 8 == 0 or s.slot == len(specs) - 1):
            log(f"  experts {s.slot + 1}/{len(specs)}  {time.time() - t1:.0f}s  MemAvailable "
                f"{_hy.mem_available_gb():.0f} GB")
    for bpath in {s.path.rsplit(".", 1)[0] for s in specs} - set(folded):
        block = model.get_submodule(bpath)
        sh = _shared_glu(block)
        if sh:
            sm = sh[1]
            setattr(block, sh[0], _rt.SharedExpert(sm.gate_proj.weight, sm.up_proj.weight, sm.down_proj.weight,
                                                   folded=False))
    for n in sorted(cpu):
        m = model.get_submodule(n)
        if isinstance(m, nn.Embedding):
            parent, leaf = parent_of(model, n)
            setattr(parent, leaf, _rt.CPUEmbedding(m.weight, dev))
        elif isinstance(m, nn.Linear) and nx and m.weight.shape[1] % 64 == 0:
            m._vahana_nx = _hy.nx_pack(m.weight, place)
            _hy.ext().register_linear_nx(m.weight, m._vahana_nx)
    # transformers' DeepSeek-V2 block reads self.num_experts in its group-limited routing but never sets it
    nre = getattr(model.config.get_text_config(), "n_routed_experts", None)
    for s in specs:
        block = model.get_submodule(s.path.rsplit(".", 1)[0])
        if nre is not None and not hasattr(block, "num_experts"):
            block.num_experts = nre
    info = {"invariant": "twin" if twin else _inv.install(model), "experts_s": time.time() - t1,
            "nx_layers": sum(1 for m in model.modules() if isinstance(m, VahanaExperts) and m.nx),
            "q4_linears": n_q4}
    _rt.TWIN["on"] = twin
    model._vahana_device = dev            # where inputs go (server / cli)
    _rt.install_linear()
    _attn.use(model)
    if twin:                      # P2: every remaining op in twin arithmetic -> the same bits on any CPU or GPU
        from . import tmath as _tm
        info["rope_twin"] = _tm.twin_rope(model)
        info["norm_fast"] = _tm.fast_norms(model)
        _tm.wrap_forward(model)
    model.eval()
    return info


def load(model_dir: str, gpu_budget_gb: float | None = None, device: str = "cuda", numa: int = 2, nx: bool = True,
         place: bool = True, reserve_gb: float = 1.5, force_cpu: str | None = None, log=print, experts: str = "cpu",
         twin: bool = True, gpu_cache_gb: float | str | None = None):
    """Checkpoint directory -> (Vahana model, info).
    gpu_cache_gb: spare GPU memory for the most-used experts (v0.3; "auto" = what is free after loading, minus
    reserve_gb). Popularity is profiled once on built-in text and saved next to Vahana's cache. Speed only: the
    outputs are the same bits with or without it."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping

    from .loading import ShardHandles
    t0 = time.time()
    dev = torch.device(device)
    if dev.type == "cuda":
        from ._triton import require_gpu
        require_gpu()
    cfg = AutoConfig.from_pretrained(model_dir)
    qcfg = getattr(cfg, "quantization_config", None)
    if qcfg is not None:                       # v1.2: Vahana reads quantized tensors itself (twin-q4 experts)
        del cfg.quantization_config
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    specs = inspect(model)
    h = ShardHandles.open(model_dir, source="mmap")
    ren, merge = [], []
    for c in get_checkpoint_conversion_mapping(cfg.model_type) or []:
        src, tgt = list(c.source_patterns), list(c.target_patterns)
        if any("*" in x for x in src):
            merge.append((src, tgt[0]))
        elif len(src) == 1 and len(tgt) == 1:
            ren.append((src[0], tgt[0]))

    def rename(k):
        for a, b in ren:
            k = k.replace(a, b)
        return k
    ck = {rename(k): k for k in h.weight_map}
    # v0.4-c: vision-language checkpoints (e.g. Qwen3.5-MoE) keep the text model under model.language_model.; the
    # causal-LM class names it model. (the vision tower and MTP head are simply never requested)
    for k in list(h.weight_map):
        if k.startswith("model.language_model."):
            ck.setdefault("model." + k[len("model.language_model."):], k)

    from . import q4 as _q4

    def get_param(n):
        if n not in ck and n + "_packed" in ck:       # v1.2: pack-quantized non-expert weight -> bf16(q * s)
            return _q4.dequant_compressed(h.get_tensor(ck[n + "_packed"]), h.get_tensor(ck[n + "_scale"])).to(
                torch.bfloat16)
        if n not in ck:
            raise KeyError(f"no checkpoint tensor for {n}")
        return h.get_tensor(ck[n])

    q4root = []

    def q4_layer_file(L):
        """v1.2: twin-q4 records of every expert of layer L, converted once from the pack-quantized checkpoint into
        ~/.cache/vahana/q4/<model>/layer<L>.safetensors and then memory-mapped read-only (paged by the OS like the
        bf16 checkpoint views; converting into RAM would need the whole expert set resident)."""
        import hashlib
        import os
        if not q4root:
            idx = open(os.path.join(model_dir, "model.safetensors.index.json"), "rb").read()
            q4root.append(os.path.join(os.path.expanduser("~"), ".cache", "vahana", "q4",
                                       hashlib.sha256(idx).hexdigest()[:16]))
            os.makedirs(q4root[0], exist_ok=True)
        path = os.path.join(q4root[0], f"layer{L}.safetensors")
        if not os.path.exists(path):
            from safetensors.torch import save_file
            pre = f"model.layers.{L}.mlp.experts."
            out = {}
            for kk in ck:
                if kk.startswith(pre) and kk.endswith(".weight_packed"):
                    base = kk[: -len("_packed")]
                    out[base[len(pre):]] = _q4.from_compressed(h.get_tensor(ck[kk]), h.get_tensor(ck[base + "_scale"]))
            save_file(out, path + ".tmp")
            os.replace(path + ".tmp", path)
        return path

    def get_expert(k):
        """One expert matrix: a bf16 view, or (v1.2) twin-q4 records (memory-mapped cache) for a pack-quantized one."""
        if k not in ck and k + "_packed" in ck:
            from .loading import ShardMap
            L = int(k.split(".layers.", 1)[1].split(".", 1)[0])
            return ShardMap.get(q4_layer_file(L)).view_tensor(k.split(".mlp.experts.", 1)[1])
        return h.get_tensor(ck[k])

    def get_experts(s):
        got = {}
        for src, tgt in merge:
            leaf = tgt.rsplit(".", 1)[-1]
            tail = tgt if tgt.startswith(".") else "." + tgt
            pname = f"{s.path}.{leaf}"
            if not pname.endswith(tail):
                continue
            base = pname[: -len(tail)]
            got[leaf] = [[get_expert(base + (p if p.startswith(".") else "." + p).replace("*", str(e)))
                          for e in range(s.E)] for p in src]
        if not got and f"{s.path}.gate_up_proj" in ck and f"{s.path}.down_proj" in ck:
            # v0.4-c: fused 3-D expert tensors as the module stores them: gate_up_proj [E, 2I, H] (gate = rows :I,
            # up = rows I:, the module's chunk(2)), down_proj [E, H, I]. Per-expert slices are views of the mmap.
            gu, dp = get_param(f"{s.path}.gate_up_proj"), get_param(f"{s.path}.down_proj")
            I = gu.shape[1] // 2
            return ([gu[e, :I] for e in range(gu.shape[0])], [gu[e, I:] for e in range(gu.shape[0])],
                    [dp[e] for e in range(dp.shape[0])], None, None, None)
        if len(got.get("gate_up_proj", [])) != 2 or len(got.get("down_proj", [])) != 1:
            raise ValueError(f"{s.path}: no per-expert gate / up / down keys in the checkpoint")
        return got["gate_up_proj"][0], got["gate_up_proj"][1], got["down_proj"][0], None, None, None

    budget = (gpu_budget_gb - reserve_gb) * 2**30 if gpu_budget_gb else float("inf")
    emb_cpu = os.environ.get("VAHANA_EMBED_CPU", "auto")
    if force_cpu is None and dev.type == "cuda" and not getattr(cfg, "tie_word_embeddings", False) and (
            emb_cpu == "1" or emb_cpu == "auto" and torch.cuda.get_device_properties(dev).total_memory < 12 * 2**30):
        # v1.4: on a small GPU the input embedding table (one row read per token, an exact lookup) lives in CPU RAM,
        # so its GPU memory goes to the expert cache instead. Same bits.
        force_cpu = r"embed_tokens$"
    cpu, folded, total = _plan(model, specs, budget, force_cpu)
    if log:
        log(f"vahana: {cfg.model_type}, {len(specs)} MoE layers, non-expert {total / 2**30:.1f} GB, "
            f"{len(cpu)} modules in CPU RAM ({len(folded)} shared experts folded)")
    def get_q4(n):
        """v1.4: twin-q4 records of a pack-quantized non-expert weight (None for any other weight)."""
        mode = os.environ.get("VAHANA_Q4_LINEARS", "auto")
        if mode == "0" or n in ck or n + "_packed" not in ck:
            return None
        if mode == "auto" and dev.type == "cuda" and torch.cuda.get_device_properties(dev).total_memory >= 12 * 2**30:
            return None     # a large GPU reads bf16 faster than it decodes 4-bit (H100: 55.9 vs 34.0 tok/s)
        sc = h.get_tensor(ck[n + "_scale"])
        return _q4.from_compressed(h.get_tensor(ck[n + "_packed"]), sc) if sc.dtype == torch.bfloat16 else None

    info = _assemble(model, specs, get_param, get_experts, dev, cpu, folded, numa, nx, place, log, experts, twin,
                     get_q4=get_q4)
    if getattr(cfg, "tie_word_embeddings", False):
        model.tie_weights()
    if dev.type == "cuda":
        # v1.4: the non-expert weights now live on the GPU (or, for CPU modules, in their own copies), so their pages
        # in the checkpoint's memory map are never read again: let the OS reclaim them before expert pages.
        from . import osl
        rel = 0
        for k in h.weight_map:
            if ".experts." in k:
                continue
            try:
                t = h.get_tensor(k)
            except Exception:  # noqa: BLE001
                continue
            if t.device.type == "cpu" and osl.release_tensor(t):
                rel += t.numel() * t.element_size()
        info["released_gb"] = rel / 2**30
    info.update(model_type=cfg.model_type, moe_layers=len(specs), nonexpert_gb=total / 2**30,
                cpu_modules=sorted(cpu), folded_shared=len(folded), load_s=time.time() - t0,
                gpu_gb=torch.cuda.memory_allocated(dev) / 2**30 if dev.type == "cuda" else 0.0)
    if gpu_cache_gb and dev.type == "cuda":
        info["gpu_cache"] = _gpu_cache(model, model_dir, gpu_cache_gb, reserve_gb, log)
    info["prefetch"] = _prefetch(model, log)
    return model, info


_ROUTERS = []


def _prefetch(model, log=None) -> dict:
    """v0.5: let the kernel prefetch streamed (mmap) experts: each job's experts as whole ranges (1), plus the next
    layer's experts predicted by its router applied to this layer's input (2). Hints to the OS only: the outputs are
    the same bits with or without it. Off by default (no gain measured on a saturated disk); VAHANA_PREFETCH=1/2."""
    import os
    from . import hybrid as _hy
    from .experts import VahanaExperts
    from .modelspec import parent_of
    k = _hy.ext()
    if not getattr(k, "has_prefetch", False):
        return {"mode": 0, "reason": "kernel without prefetch"}
    exp = [(n, m) for n, m in model.named_modules() if isinstance(m, VahanaExperts)]
    if not exp:
        return {"mode": 0, "reason": "no CPU experts"}
    streamed = sum(w.numel() * w.element_size() for _, m in exp if not m.nx for trip in zip(m.g, m.u, m.d)
                   for w in trip)
    need = streamed / 2**30
    env = os.environ.get("VAHANA_PREFETCH")
    # v1.4: mode 3 (prefetch only the experts a layer has not used lately, which are likely evicted) is the default
    # when the CPU experts exceed 80% of physical RAM, so that some are always being paged in: on a 15 GB laptop with
    # Qwen3-30B's 14 GB of 4-bit experts, real-prompt decode +17% and a faster first token (experiments/v15_decode).
    # Modes 1 and 2 (every job's experts) cost more than they save. Off when the experts fit in RAM.
    ram = _total_ram_gb()
    mode = int(env) if env not in (None, "") else (3 if ram and need > 0.8 * ram else 0)
    n = 0
    if mode >= 2:
        for name, m in exp:
            parent, _ = parent_of(model, getattr(m.spec, "path", name))    # the MoE block (not a cache wrapper)
            E, H = m.spec.E, m.spec.H
            ts = list(parent.named_parameters()) + list(parent.named_buffers())
            W = next((p for pn, p in ts if p.dim() == 2 and tuple(p.shape) == (E, H)
                      and ".experts" not in "." + pn), None)
            if W is None:
                continue
            Wc = W.detach().to("cpu", torch.bfloat16).contiguous()
            _ROUTERS.append(Wc)
            k.register_router(m.spec.slot, Wc)
            n += 1
    k.set_prefetch(mode)
    info = {"mode": mode, "streamed_gb": round(need, 1), "routers": n}
    if log and mode:
        log(f"vahana: expert prefetch mode {mode} ({need:.0f} GB of CPU experts, {ram:.0f} GB RAM"
            + (f", {n} routers)" if mode in (2, 4) else ")"))
    return info


def _total_ram_gb() -> float:
    """Physical RAM in GB (0 if unknown)."""
    import os
    import sys
    try:
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 2**30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except Exception:  # noqa: BLE001
        return 0.0


_CALIB = ["The history of the Roman Empire spans more than a thousand years, from the founding of the city",
          "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]",
          "Explain photosynthesis to a ten-year-old in three short paragraphs.",
          "La economía de América Latina creció lentamente durante la última década, debido a",
          "Q: A train leaves at 3 pm travelling 60 km/h. How far has it gone by 5:30 pm?\nA: Let's think step by step.",
          "Write a short, friendly email asking a colleague to review a pull request before Friday.",
          "In quantum mechanics, the uncertainty principle states that",
          "这是一个关于人工智能如何改变教育的简短讨论。"]


def _gpu_cache(model, model_dir, gpu_cache_gb, reserve_gb, log):
    """Profile expert popularity (once per model, saved), then cache the most-used experts in GPU memory."""
    import hashlib
    import json
    import os
    from . import cache as _cache
    from .experts import VahanaExperts
    free, total = (v / 2**30 for v in torch.cuda.mem_get_info())
    # v1.4: a 4 GB card keeps 1 GB, not 1.5, for KV cache / workspace (VAHANA_CACHE_RESERVE_FRAC of the GPU)
    reserve_gb = min(reserve_gb, float(os.environ.get("VAHANA_CACHE_RESERVE_FRAC", "0.25")) * total)
    gb = free - reserve_gb if gpu_cache_gb == "auto" else float(gpu_cache_gb)
    gb = min(gb, free - 0.5)
    if gb <= 0:
        return {"cached_experts": 0, "reason": f"no free GPU memory ({free:.1f} GB free)"}
    layers = {m.spec.slot: m for m in model.modules() if isinstance(m, VahanaExperts)}
    key = hashlib.sha256(json.dumps(sorted(json.load(open(os.path.join(model_dir, "config.json"))).items()),
                                    default=str).encode()).hexdigest()[:16]
    path = os.path.join(os.path.expanduser("~"), ".cache", "vahana", "popularity", f"{key}.json")
    counts = None
    if os.path.exists(path):
        counts = {int(L): torch.tensor(v) for L, v in json.load(open(path)).items()}
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir)
        dev = next(p.device for p in model.parameters() if p.device.type == "cuda")

        def run():
            with torch.no_grad():
                for t in _CALIB:
                    model(tok(t, return_tensors="pt").input_ids.to(dev))
        counts = _cache.popularity(model, run, layers)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump({str(L): c.tolist() for L, c in counts.items()}, open(path, "w"))
    return _cache.install(model, gb * 2**30, counts, log=log)


def prepare(model: nn.Module, gpu_budget_gb: float | None = None, force_cpu: str | None = None, numa: int = 2,
            nx: bool = True, place: bool = False, log=None, twin: bool = True, experts: str = "cpu",
            gpu_cache_frac: float | None = None):
    """A model already in memory -> Vahana model, in place. The model's device is where non-expert weights run
    (a CUDA device, or the CPU for a CPU-only machine). twin: the device-twin arithmetic everywhere (P2), so the
    result is the same bits on any CPU or GPU."""
    specs = inspect(model)
    devs = [p.device for p in model.parameters()]
    dev = next((d for d in devs if d.type == "cuda"), devs[0])
    if dev.type == "cuda":
        from ._triton import require_gpu
        require_gpu()
    src = {n: p.detach() for n, p in model.named_parameters()}
    exp = {s.path: expert_weights(s, model.get_submodule(s.path)) for s in specs}
    budget = gpu_budget_gb * 2**30 if gpu_budget_gb else float("inf")
    cpu, folded, total = _plan(model, specs, budget, force_cpu)
    info = _assemble(model, specs, lambda n: src[n], lambda s: exp[s.path], dev, cpu, folded, numa, nx, place, log,
                     experts, twin)
    info.update(cpu_modules=sorted(cpu), folded_shared=len(folded))
    if gpu_cache_frac and dev.type == "cuda":         # tests: cache this fraction of every layer's experts
        from . import cache as _cache
        info["gpu_cache"] = _cache.install(model, 0, None, frac=gpu_cache_frac)
    return model, info
