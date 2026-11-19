"""RowSplitDetector: finds the first module where identical batch rows split."""
from __future__ import annotations

import torch
from torch import nn

from vahana.rowsplit import RowSplitDetector


class Layer(nn.Module):
    def __init__(self, perturb_row=None):
        super().__init__()
        self.self_attn = nn.Identity()
        self.mlp = nn.Linear(4, 4)
        self.perturb_row = perturb_row

    def forward(self, x):
        x = self.self_attn(x)
        if self.perturb_row is not None:
            x = x.clone()
            x[self.perturb_row] += 1e-3
        return self.mlp(x)


class Profile:
    decoder_layer_cls = Layer


def test_detects_first_split_and_ignores_identical_rows():
    torch.manual_seed(0)
    model = nn.Sequential(Layer(), Layer(perturb_row=3), Layer())
    det = RowSplitDetector()
    assert det.attach(model, Profile()) == 6
    x = torch.randn(1, 2, 4).expand(5, 2, 4).contiguous()  # 5 identical rows
    det.step = 7
    with torch.no_grad():
        model(x)
    first = det.first()
    assert first["layer"] == 1 and first["module"] == "mlp" and first["rows"] == [3]
    assert first["step"] == 7 and first["max_abs_diff"] > 0
    det.detach()
    plain = nn.Sequential(Layer(), Layer())
    clean = RowSplitDetector()
    assert clean.attach(plain, Profile()) == 4
    with torch.no_grad():
        plain(x)
    assert clean.events == []
