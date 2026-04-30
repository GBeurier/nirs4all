"""Length robustness tests (Codex F5).

Every model that the benchmark runs against the curated cohort must accept the
spectrum lengths actually present in the cohort *and* produce a finite gradient
through the parameters. Concretely we test the cohort length set
{401 (DIESEL), 576 (Beer), 700 (CORN/BISCUIT), 1154 (AMYLOSE), 2151 (ECOSIS)}.

A model that crashes or produces a non-finite output on any of these lengths is
ineligible for cohort benchmarks.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nicon_v2.models.baseline import build_decon_torch, build_nicon_torch
from nicon_v2.models.v1a_minimal_repair import NiconV1a, build_nicon_v1a


COHORT_LENGTHS = (401, 576, 700, 1154, 2151)


@pytest.mark.parametrize("length", COHORT_LENGTHS)
def test_nicon_v1a_forward_backward(length: int):
    model = build_nicon_v1a((1, length))
    x = torch.randn(4, 1, length, requires_grad=False)
    y = torch.randn(4)
    out = model(x).squeeze(-1)
    assert out.shape == (4,), f"unexpected output shape {out.shape} for length={length}"
    loss = ((out - y) ** 2).mean()
    loss.backward()
    grad_norm = sum(
        (p.grad.detach().norm().item() if p.grad is not None else 0.0)
        for p in model.parameters()
        if p.requires_grad
    )
    assert torch.isfinite(loss), f"loss not finite for length={length}"
    assert grad_norm > 0.0, f"no parameter received a gradient for length={length}"


@pytest.mark.parametrize("length", COHORT_LENGTHS)
def test_nicon_baseline_forward_backward(length: int):
    """Upstream NICON should at least *run* on every cohort length (failure modes are about
    accuracy, not crashes)."""
    model = build_nicon_torch((1, length))
    x = torch.randn(4, 1, length)
    y = torch.randn(4, 1)
    out = model(x)
    if out.dim() == 1:
        out = out.unsqueeze(-1)
    assert out.shape == (4, 1), f"unexpected output shape {out.shape} for length={length}"
    loss = ((out - y) ** 2).mean()
    loss.backward()
    assert torch.isfinite(loss), f"loss not finite for length={length}"


@pytest.mark.parametrize("length", [576, 700, 1154, 2151])  # 401 too short for DECON
def test_decon_baseline_forward_backward(length: int):
    model = build_decon_torch((1, length))
    x = torch.randn(4, 1, length)
    y = torch.randn(4, 1)
    out = model(x)
    if out.dim() == 1:
        out = out.unsqueeze(-1)
    loss = ((out - y) ** 2).mean()
    loss.backward()
    assert torch.isfinite(loss), f"DECON loss not finite for length={length}"


def test_nicon_v1a_computes_effective_seq_len():
    model = NiconV1a((1, 700))
    # post-stage1: floor((700-15)/5+1) = 138
    # post-stage2: floor((138-21)/3+1) = 40
    # post-stage3: floor((40-5)/3+1) = 12
    assert model.effective_seq_len == 12
    assert model.flat_dim == 12 * 32


def test_nicon_v1a_rejects_too_short_input():
    with pytest.raises(ValueError, match="sequence_length"):
        NiconV1a((1, 50))
