"""Geometry / parity tests for V1a (Codex round 2 finding #1).

We assert that V1a preserves the receptive field of upstream NICON (kernels,
strides, channels, effective output sequence length) and document the
intentional ordering change (Conv→Norm→Activation→Dropout in V1a vs the
upstream Conv→Activation→BatchNorm).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from nicon_v2.models.baseline import build_nicon_torch
from nicon_v2.models.v1a_minimal_repair import (
    NiconV1a,
    NiconV1aActivationOnly,
    NiconV1aHeadOnly,
)


@pytest.mark.parametrize("length", [401, 576, 700, 1154, 2151])
def test_v1a_matches_nicon_effective_seq_len(length: int):
    """The post-conv sequence length is computable from kernels/strides; V1a should
    match upstream NICON exactly."""
    v1a = NiconV1a((1, length))
    # Walk the upstream model and observe its post-stage shapes via a forward pass.
    nicon = build_nicon_torch((1, length))
    nicon.eval()
    with torch.no_grad():
        x = torch.zeros(1, 1, length)
        # Run upstream NICON conv stack only by walking modules until we hit a Linear / Flatten.
        out_shape: tuple[int, ...] = (1, 1, length)
        for m in nicon:
            if isinstance(m, (nn.Linear, nn.Flatten)):
                break
            x = m(x)
            out_shape = tuple(x.shape)
    # Upstream post-conv (batch, channels, length) for an input of size `length`.
    assert len(out_shape) == 3, f"unexpected shape {out_shape}"
    _, nicon_post_conv_ch, nicon_post_conv_len = out_shape
    assert v1a.effective_seq_len == nicon_post_conv_len
    assert v1a.flat_dim == nicon_post_conv_ch * nicon_post_conv_len


def test_v1a_head_only_keeps_upstream_backbone():
    """The H1-only ablation must reuse upstream NICON; only the final Linear is replaced."""
    model = NiconV1aHeadOnly((1, 700))
    # The model should be a single nn.Sequential with no nn.Sigmoid as last module.
    body = model.body
    assert isinstance(body, nn.Sequential)
    assert not isinstance(body[-1], nn.Sigmoid)
    # The new last Linear is `something → 1`.
    last_linear = None
    for m in body:
        if isinstance(m, nn.Linear):
            last_linear = m
    assert last_linear is not None
    assert last_linear.out_features == 1


def test_v1a_activation_only_keeps_sigmoid_output():
    """The H2-only ablation must keep a Sigmoid as the last activation."""
    model = NiconV1aActivationOnly((1, 700))
    head = model._inner.head
    assert isinstance(head, nn.Sequential)
    assert isinstance(head[-1], nn.Sigmoid)


def test_v1a_full_has_no_sigmoid():
    """The combined V1a (H1+H2) must not contain any Sigmoid module."""
    model = NiconV1a((1, 700))
    for m in model.modules():
        assert not isinstance(m, nn.Sigmoid), "V1a head should be linear (no sigmoid)"


@pytest.mark.parametrize("length", [401, 576, 700])
def test_all_v1a_variants_forward_backward(length: int):
    """Every V1a variant must run forward + backward without numerical issues."""
    for ctor in (NiconV1aHeadOnly, NiconV1aActivationOnly, NiconV1a):
        m = ctor((1, length))
        x = torch.randn(4, 1, length)
        y = torch.randn(4)
        out = m(x).squeeze(-1)
        loss = ((out - y) ** 2).mean()
        loss.backward()
        assert torch.isfinite(loss)
