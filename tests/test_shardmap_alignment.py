"""ShardMap views at odd file offsets (odd safetensors header length)."""
from __future__ import annotations

import json
import struct

import torch

from bhaskera.inference.colossus.loading import ShardMap


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
