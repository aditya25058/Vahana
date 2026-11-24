"""Round-trip cost with the data moved by the mailbox kernel itself (no memcpy nodes): no-op CPU jobs."""
import torch, sys, json, triton, triton.language as tl
from transformers import AutoTokenizer
from vahana import loader, decode, runtime, hybrid
from vahana.experts import VahanaExperts

@triton.jit
def _rt(counter, req, done, job, x, xh, nx, ti, tih, nti, tw, twh, ntw, oh, o, no, BLK: tl.constexpr):
    # inputs: device -> pinned host (int32 words), through UVA stores
    for s in range(0, nx, BLK):
        i = s + tl.arange(0, BLK); m = i < nx
        tl.store(xh + i, tl.load(x + i, mask=m), mask=m)
    i = tl.arange(0, BLK)
    tl.store(tih + i, tl.load(ti + i, mask=i < nti), mask=i < nti)
    tl.store(twh + i, tl.load(tw + i, mask=i < ntw), mask=i < ntw)
    seq = tl.atomic_add(counter + job, 1) + 1
    tl.atomic_xchg(req + job, seq, sem="release", scope="sys")
    cur = tl.load(done + job, volatile=True)
    while cur != seq:
        cur = tl.load(done + job, volatile=True)
    tl.atomic_add(done + job, 0, sem="acquire", scope="sys")
    for s in range(0, no, BLK):
        i = s + tl.arange(0, BLK); m = i < no
        tl.store(o + i, tl.load(oh + i, mask=m, volatile=True), mask=m)

MD = sys.argv[1]
model, info = loader.load(MD, device="cuda", log=None)
tok = AutoTokenizer.from_pretrained(MD)
ids = tok(json.load(open("experiments/m5_real/prompts.json"))[0], return_tensors="pt").input_ids.cuda()
k = hybrid.ext()
orig = VahanaExperts.forward
NOOP = {}
def w32(t): return t.contiguous().view(torch.int32)
def noop_kc(self, h, ti, tw):
    if not runtime.GRAPH["on"]:
        return orig(self, h, ti, tw)
    N, H = h.shape
    key = (self.spec.slot, N)
    if key not in NOOP:
        b = (runtime._pin((N, H), torch.bfloat16), runtime._pin(tuple(ti.shape), torch.int64),
             runtime._pin(tuple(tw.shape), tw.dtype), runtime._pin((N, H), torch.bfloat16))
        NOOP[key] = b + (k.make_host_job(self.spec.slot, *b, self.numa, True),)
        if not runtime._MB:
            runtime.run_host_job.__globals__  # noqa
            dev = h.device
            runtime._MB["counter"] = torch.zeros(4096, dtype=torch.int32, device=dev)
            runtime._MB["req"] = torch.zeros(4096, dtype=torch.int32).pin_memory()
            runtime._MB["done"] = torch.zeros(4096, dtype=torch.int32).pin_memory()
            k.mailbox_start(runtime._MB["req"], runtime._MB["done"])
    xb, tib, twb, ob, job = NOOP[key]
    out = torch.empty((N, H), dtype=torch.bfloat16, device=h.device)
    X, TI, TW = w32(h.to(torch.bfloat16)), w32(ti), w32(tw)
    _rt[(1,)](runtime._MB["counter"], runtime._MB["req"], runtime._MB["done"], job, X, w32(xb), X.numel(),
              TI, w32(tib), TI.numel(), TW, w32(twb), TW.numel(), w32(ob), w32(out), out.numel() // 2, BLK=1024)
    return out
def nohost(self, h, ti, tw):
    if not runtime.GRAPH["on"]:
        return orig(self, h, ti, tw)
    return torch.zeros_like(h)
for mode, f in (("no_host", nohost), ("noop_kernelcopy", noop_kc), ("no_host", nohost), ("noop_kernelcopy", noop_kc)):
    VahanaExperts.forward = f
    g = decode.generate(model, ids, max_new_tokens=33, graph=True)
    st = g["step_ms"][1:]
    print(f"{mode:16s} steady {sorted(st)[len(st)//2]:7.1f} ms", flush=True)
