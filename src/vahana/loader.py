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


def _assemble(model, specs, get_param, get_experts, dev, cpu, folded, numa, nx, place, log, experts="cpu"):
    pref = tuple(s.path + "." for s in specs)
    on_cpu = lambda mod_name: any(mod_name == c or mod_name.startswith(c + ".") for c in cpu)
    for n, p in list(model.named_parameters()):
        if n.startswith(pref):
            continue
        mod_name, pname = n.rsplit(".", 1)
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
    info = {"invariant": _inv.install(model), "experts_s": time.time() - t1,
            "nx_layers": sum(1 for m in model.modules() if isinstance(m, VahanaExperts) and m.nx)}
    _rt.install_linear()
    _attn.use(model)
    model.eval()
    return info


def load(model_dir: str, gpu_budget_gb: float | None = None, device: str = "cuda", numa: int = 2, nx: bool = True,
         place: bool = True, reserve_gb: float = 1.5, force_cpu: str | None = None, log=print, experts: str = "cpu"):
    """Checkpoint directory -> (Vahana model, info)."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping

    from .loading import ShardHandles
    t0 = time.time()
    dev = torch.device(device)
    cfg = AutoConfig.from_pretrained(model_dir)
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

    def get_param(n):
        if n not in ck:
            raise KeyError(f"no checkpoint tensor for {n}")
        return h.get_tensor(ck[n])

    def get_experts(s):
        got = {}
        for src, tgt in merge:
            leaf = tgt.rsplit(".", 1)[-1]
            tail = tgt if tgt.startswith(".") else "." + tgt
            pname = f"{s.path}.{leaf}"
            if not pname.endswith(tail):
                continue
            base = pname[: -len(tail)]
            got[leaf] = [[h.get_tensor(ck[base + (p if p.startswith(".") else "." + p).replace("*", str(e))])
                          for e in range(s.E)] for p in src]
        if len(got.get("gate_up_proj", [])) != 2 or len(got.get("down_proj", [])) != 1:
            raise ValueError(f"{s.path}: no per-expert gate / up / down keys in the checkpoint")
        return got["gate_up_proj"][0], got["gate_up_proj"][1], got["down_proj"][0], None, None, None

    budget = (gpu_budget_gb - reserve_gb) * 2**30 if gpu_budget_gb else float("inf")
    cpu, folded, total = _plan(model, specs, budget, force_cpu)
    if log:
        log(f"vahana: {cfg.model_type}, {len(specs)} MoE layers, non-expert {total / 2**30:.1f} GB, "
            f"{len(cpu)} modules in CPU RAM ({len(folded)} shared experts folded)")
    info = _assemble(model, specs, get_param, get_experts, dev, cpu, folded, numa, nx, place, log, experts)
    if getattr(cfg, "tie_word_embeddings", False):
        model.tie_weights()
    info.update(model_type=cfg.model_type, moe_layers=len(specs), nonexpert_gb=total / 2**30,
                cpu_modules=sorted(cpu), folded_shared=len(folded), load_s=time.time() - t0,
                gpu_gb=torch.cuda.memory_allocated(dev) / 2**30)
    return model, info


def prepare(model: nn.Module, gpu_budget_gb: float | None = None, force_cpu: str | None = None, numa: int = 2,
            nx: bool = True, place: bool = False, log=None):
    """A model already in memory (weights on any device) -> Vahana model, in place."""
    specs = inspect(model)
    dev = next(p for p in model.parameters() if p.device.type == "cuda").device
    src = {n: p.detach() for n, p in model.named_parameters()}
    exp = {s.path: expert_weights(s, model.get_submodule(s.path)) for s in specs}
    budget = gpu_budget_gb * 2**30 if gpu_budget_gb else float("inf")
    cpu, folded, total = _plan(model, specs, budget, force_cpu)
    info = _assemble(model, specs, lambda n: src[n], lambda s: exp[s.path], dev, cpu, folded, numa, nx, place, log)
    info.update(cpu_modules=sorted(cpu), folded_shared=len(folded))
    return model, info
