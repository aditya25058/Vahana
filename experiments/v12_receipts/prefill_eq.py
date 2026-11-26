"""Receipts feasibility: are the logits of one teacher-forced forward over prompt + generated tokens bit-identical to
the per-step logits of Vahana's decode? (If yes, verifying an answer is one batched prefill.)"""
import sys

import torch

from vahana import decode, loader, selftest

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cpu")
fams = sys.argv[2].split(",") if len(sys.argv) > 2 else ["mixtral", "qwen3_moe", "olmoe", "deepseek_v3", "gpt_oss",
                                                         "glm4_moe", "qwen3_5_moe_text"]
for fam in fams:
    cfg, m = selftest.build(fam)
    model, _ = loader.prepare(m.to(dev))
    ids, _ = selftest.prompts(cfg.vocab_size)
    p = ids[1:2, -24:].to(dev)                                   # one unpadded prompt (24 tokens)
    N = 8
    r = decode.generate(model, p, None, max_new_tokens=N, graph=dev.type == "cuda", return_logits=True)
    gen = r["ids"].view(1, -1)
    full = torch.cat([p, gen[:, :-1]], 1)
    with torch.no_grad():
        out = model(full, logits_to_keep=0).logits[0, p.shape[1] - 1:].float()
    dl = r["logits"]
    dl = torch.stack(list(dl), 0).view(-1, out.shape[-1]).float() if isinstance(dl, (list, tuple)) else \
        dl.view(-1, out.shape[-1]).float()
    same = [bool(torch.equal(out[i].cpu(), dl[i].cpu())) for i in range(min(len(out), len(dl)))]
    am = (out.argmax(-1).cpu() == gen[0, :len(out)].cpu()).tolist()
    print(f"{fam:18s} prefill==decode per step: {same}  argmax matches: {all(am)}", flush=True)
