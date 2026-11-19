"""ShardMap views at odd file offsets (odd safetensors header length)."""
from __future__ import annotations

import json
import struct

import torch

from vahana.loading import ShardMap


def _write(path, header_pad):
    t = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    raw = t.view(torch.uint8).numpy().tobytes()
    hdr = json.dumps({"w": {"dtype": "BF16", "shape": [2, 3],
                            "data_offsets": [0, len(raw)]}}) + " " * header_pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)) + hdr.encode() + raw)
    return t


def test_view_tensor_at_odd_offset(tmp_path):
    for pad in (0, 1):  # one of the two header lengths is odd
        p = tmp_path / f"s{pad}.safetensors"
        ref = _write(p, pad)
        sm = ShardMap(str(p))
        assert torch.equal(sm.view_tensor("w"), ref)
    ShardMap.clear()


def test_mmap_and_safe_open_sources_return_identical_bytes(tmp_path):
    from safetensors.torch import save_file

    from vahana.loading import ShardHandles

    g = torch.Generator().manual_seed(3)
    tensors = {"a.weight": torch.randn(4, 6, generator=g).to(torch.bfloat16),
               "b.weight": torch.randn(3, dtype=torch.float32, generator=g)}
    save_file(tensors, str(tmp_path / "m-0.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "m-0.safetensors" for k in tensors}}))
    views = ShardHandles.open(str(tmp_path), source="mmap")
    safe = ShardHandles.open(str(tmp_path), source="safe_open")
    for k, ref in tensors.items():
        assert torch.equal(views.get_tensor(k), ref)
        assert torch.equal(safe.get_tensor(k), ref)
    ShardMap.clear()
