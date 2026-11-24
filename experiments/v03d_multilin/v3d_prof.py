"""Where a W-token pass spends GPU time vs a 1-token pass (GPU-resident OLMoE, eager forward, CUDA kernel totals)."""
import sys, torch, collections
from transformers import AutoTokenizer, StaticCache
from torch.profiler import profile, ProfilerActivity
from vahana import loader
MD = sys.argv[1]
model, info = loader.load(MD, device="cuda", gpu_cache_gb=16, log=None)
tok = AutoTokenizer.from_pretrained(MD)
ids = tok("The history of the Roman Empire spans more than a thousand years, from", return_tensors="pt").input_ids.cuda()
T = ids.shape[1]
def run(W, prof=False):
    cache = StaticCache(config=model.config, max_cache_len=T + 80)
    am = torch.ones(1, T + 80, dtype=torch.long, device="cuda")
    with torch.no_grad():
        model(ids, attention_mask=am, past_key_values=cache, cache_position=torch.arange(T, device="cuda"), use_cache=True, logits_to_keep=1)
        q = ids[:, -W:]
        cp = torch.arange(T, T + W, device="cuda")
        for lay in cache.layers:
            if isinstance(getattr(lay, "cumulative_length", None), torch.Tensor): lay.cumulative_length.fill_(T)
        if not prof:
            model(q, attention_mask=am, position_ids=cp.view(1, W), past_key_values=cache, cache_position=cp, use_cache=True, logits_to_keep=W)
            return None
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            model(q, attention_mask=am, position_ids=cp.view(1, W), past_key_values=cache, cache_position=cp, use_cache=True, logits_to_keep=W)
            torch.cuda.synchronize()
        return p
for W in (1, 5, 12):
    run(W); run(W)
    p = run(W, prof=True)
    agg = collections.Counter(); cnt = collections.Counter()
    for e in p.key_averages():
        if e.device_type.name == "CUDA" or e.self_device_time_total > 0:
            name = e.key
            for k in ("_lin", "_cgate_up", "_cdown", "_attn", "_trms", "_mb_", "_gate_up", "_down", "_dot32", "elementwise", "reduce", "copy", "gather", "sort", "index", "cat", "Memcpy"):
                if k in name: name = k; break
            agg[name] += e.self_device_time_total; cnt[name] += e.count
    tot = sum(agg.values())
    print(f"W={W}: GPU kernel time {tot/1e3:.2f} ms over {sum(cnt.values())} kernels")
    for n, t in agg.most_common(10):
        print(f"   {n[:40]:40s} {t/1e3:7.3f} ms  x{cnt[n]}")
