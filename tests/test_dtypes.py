"""safetensors dtype mapping: exact for known tags, hard error otherwise."""
from __future__ import annotations

import pytest
import torch

from bhaskera.inference.colossus.loading import _safetensors_dtype


def test_known_tags_map_exactly():
    assert _safetensors_dtype("BF16") is torch.bfloat16
    assert _safetensors_dtype("f32") is torch.float32
    assert _safetensors_dtype("I32") is torch.int32


def test_unknown_tag_is_a_hard_error():
    # Used to default to BF16, silently reinterpreting FP8 bytes.
    with pytest.raises(ValueError, match="F8_E4M3"):
        _safetensors_dtype("F8_E4M3")
