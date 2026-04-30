"""Phase 1b model geometry test — V1b accepts 1-channel raw input and produces (N, 1) output."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nicon_v2.models.v1b_concat_aug import NiconV1b, build_nicon_v1b


@pytest.mark.parametrize("length", [401, 576, 700, 1154, 2151])
def test_v1b_forward_backward(length: int):
    model = build_nicon_v1b((1, length))
    x = torch.randn(4, 1, length)
    y = torch.randn(4)
    out = model(x).squeeze(-1)
    assert out.shape == (4,), f"unexpected V1b output shape {out.shape} for length={length}"
    loss = ((out - y) ** 2).mean()
    loss.backward()
    assert torch.isfinite(loss)
    grad_norm = sum(
        (p.grad.detach().norm().item() if p.grad is not None else 0.0)
        for p in model.parameters()
        if p.requires_grad
    )
    assert grad_norm > 0.0, f"no gradient propagated for length={length}"


def test_v1b_first_conv_has_three_input_channels():
    import torch.nn as nn

    model = NiconV1b((1, 700))
    first_conv = None
    for m in model.body:
        if isinstance(m, nn.Conv1d):
            first_conv = m
            break
    assert first_conv is not None
    assert first_conv.in_channels == 3, f"V1b first conv should have in_channels=3, got {first_conv.in_channels}"


def test_v1b_rejects_multichannel_input():
    with pytest.raises(ValueError, match="1-channel"):
        NiconV1b((2, 700))
