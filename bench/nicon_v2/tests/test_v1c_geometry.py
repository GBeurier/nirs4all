"""Phase 1c V1c GAP backbone — length robustness, parameter count, norm switch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from nicon_v2.models.v1c_gap_backbone import NiconV1c, build_nicon_v1c


@pytest.mark.parametrize("length", [50, 100, 401, 576, 700, 1154, 2151])
def test_v1c_forward_backward_any_length(length: int):
    """V1c is GAP-headed so the model accepts arbitrary spectrum lengths (≥ kernel size)."""
    if length < 32:
        pytest.skip("too short for 4-block max-pool stack")
    model = build_nicon_v1c((1, length))
    x = torch.randn(4, 1, length)
    y = torch.randn(4)
    out = model(x).squeeze(-1)
    assert out.shape == (4,), f"unexpected V1c output for length={length}: {out.shape}"
    loss = ((out - y) ** 2).mean()
    loss.backward()
    grad_norm = sum(p.grad.detach().norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0


@pytest.mark.parametrize("norm", ["layer", "batch", "group"])
def test_v1c_norm_switch(norm: str):
    model = NiconV1c((1, 700), norm=norm)
    x = torch.randn(4, 1, 700)
    out = model(x)
    assert out.shape == (4, 1)


@pytest.mark.parametrize("use_concat", [False, True])
def test_v1c_concat_derivatives_optional(use_concat: bool):
    model = NiconV1c((1, 700), use_concat_derivatives=use_concat)
    # Find the first conv: in_channels should be 3 with concat, 1 without.
    first_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            first_conv = m
            break
    assert first_conv is not None
    expected_in = 3 if use_concat else 1
    assert first_conv.in_channels == expected_in


def test_v1c_parameter_count_under_capacity_cap():
    """Default V1c on a 700-feature input should stay well under 1e6 parameters."""
    model = NiconV1c((1, 700))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params < 1_000_000, f"V1c has {n_params} params, > 1M cap"


def test_v1c_invalid_concat_with_multichannel_raises():
    with pytest.raises(ValueError, match="1-channel"):
        NiconV1c((2, 700), use_concat_derivatives=True)
