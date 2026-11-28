"""v1.4: a pack-quantized (compressed-tensors w4a16) checkpoint loads through loader.load for more than one family,
and gives the same bits on the CPU and the GPU. Attention and experts are 4-bit, as in RedHat's w4a16 releases, so on
the CPU this runs the 4-bit CPU linear and the twin-q4 experts; on the GPU, Q4Linear and the expert cache."""
import json
import os

import numpy as np
import pytest
import torch

from vahana import decode, loader, selftest

G = 32


def _compress(W: torch.Tensor):
    """compressed-tensors pack-quantized, symmetric int4, group G: int32 [R, K/8] (nibble i of word j = k 8j + i,
    stored q + 8) and bf16 scales [R, K/G]."""
    R, K = W.shape
    Wf = W.float().reshape(R, K // G, G)
    s = (Wf.abs().amax(-1) / 7).to(torch.bfloat16)
    sf = torch.where(s.float() == 0, torch.ones_like(s.float()), s.float())
    q = torch.clamp(torch.round(Wf / sf[..., None]), -8, 7).reshape(R, K).to(torch.int64) + 8
    u = q.reshape(R, K // 8, 8).numpy().astype(np.uint64)
    words = np.zeros((R, K // 8), dtype=np.uint64)
    for i in range(8):
        words |= u[:, :, i] << np.uint64(4 * i)
    packed = torch.from_numpy(words.astype(np.uint32).view(np.int32).copy())
    return packed, s


def _q4_checkpoint(fam, d):
    from safetensors.torch import load_file, save_file
    cfg, m = selftest.build(fam)
    m.save_pretrained(d)
    sd = load_file(os.path.join(d, "model.safetensors"))
    out = {}
    for k, t in sd.items():
        quant = t.dim() == 2 and k.endswith(".weight") and (".self_attn." in k or ".experts." in k) \
            and t.shape[1] % G == 0
        if quant:
            p, s = _compress(t)
            out[k[: -len(".weight")] + ".weight_packed"] = p
            out[k[: -len(".weight")] + ".weight_scale"] = s
            out[k[: -len(".weight")] + ".weight_shape"] = torch.tensor(list(t.shape), dtype=torch.int32)
        else:
            out[k] = t
    os.remove(os.path.join(d, "model.safetensors"))
    save_file(out, os.path.join(d, "model.safetensors"), metadata={"format": "pt"})
    json.dump({"metadata": {}, "weight_map": {k: "model.safetensors" for k in out}},
              open(os.path.join(d, "model.safetensors.index.json"), "w"))
    c = json.load(open(os.path.join(d, "config.json")))
    c["quantization_config"] = {"quant_method": "compressed-tensors", "format": "pack-quantized",
                                "config_groups": {"group_0": {"targets": ["Linear"], "weights": {
                                    "num_bits": 4, "type": "int", "symmetric": True, "strategy": "group",
                                    "group_size": G}}}}
    json.dump(c, open(os.path.join(d, "config.json"), "w"))
    return cfg


def _run(d, cfg, device):
    model, info = loader.load(d, device=device, log=lambda *a: None)
    ids, am = selftest.prompts(cfg.vocab_size)
    dev = model._vahana_device
    r = decode.generate(model, ids.to(dev), am.to(dev), max_new_tokens=4, graph=False, return_logits=True)
    return r["logits"].float().cpu(), r["ids"].cpu(), info


@pytest.mark.parametrize("fam", ["mixtral", "qwen3_moe"])
def test_q4_checkpoint_loads_and_cpu_equals_gpu(fam, tmp_path, monkeypatch):
    monkeypatch.setenv("VAHANA_Q4_LINEARS", "1")          # keep attention 4-bit on any GPU size
    for v in ("HOME", "USERPROFILE"):                      # a private ~/.cache/vahana/q4 for this test
        monkeypatch.setenv(v, str(tmp_path / "home"))
    d = str(tmp_path / fam)
    os.makedirs(d)
    cfg = _q4_checkpoint(fam, d)
    lc, ic, info = _run(d, cfg, "cpu")
    assert info["q4_linears"] > 0, info                  # the CPU kept attention 4-bit
    assert torch.isfinite(lc).all()
    if not torch.cuda.is_available():
        pytest.skip("no GPU: CPU load checked only")
    lg, ig, _ = _run(d, cfg, "cuda")
    assert torch.equal(ic, ig)
    assert torch.equal(lc.view(torch.int32), lg.view(torch.int32))


def test_q4_cache_not_shared_between_checkpoints(tmp_path, monkeypatch):
    """Two checkpoints with the same tensor names (a model and its fine-tune) must not share converted experts."""
    from safetensors.torch import load_file, save_file
    for v in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(v, str(tmp_path / "home"))
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    os.makedirs(a)
    cfg = _q4_checkpoint("qwen3_moe", a)
    import shutil
    shutil.copytree(a, b)
    f = os.path.join(b, "model.safetensors")
    sd = load_file(f)
    for k in list(sd):
        if ".experts." in k and k.endswith(".weight_scale"):
            sd[k] = (sd[k].float() * 1.5).to(torch.bfloat16)      # a different model, identical index.json
    save_file(sd, f, metadata={"format": "pt"})
    la, _, _ = _run(a, cfg, "cpu")
    lb, _, _ = _run(b, cfg, "cpu")
    assert not torch.equal(la, lb)
