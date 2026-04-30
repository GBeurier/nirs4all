"""Geometry / capacity / branch-fitting tests for NiconV2A."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nicon_v2.models.v2_aom_cnn import NiconV2A, build_nicon_v2a


COHORT_LENGTHS = (401, 576, 700, 1154, 2151)


@pytest.mark.parametrize("length", COHORT_LENGTHS)
def test_v2a_forward_backward_all_lengths(length: int):
    model = build_nicon_v2a((1, length))
    x = torch.randn(4, 1, length)
    model.fit_branches(x)
    out = model(x).squeeze(-1)
    assert out.shape == (4,)
    loss = ((out - torch.randn(4)) ** 2).mean()
    loss.backward()
    grad_norm = sum(p.grad.detach().norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0


@pytest.mark.parametrize("bank", ["compact", "extended", "compact_plus_cnn_extras", "full"])
def test_v2a_bank_choice(bank: str):
    model = NiconV2A((1, 700), bank=bank)
    expected = {"compact": 9, "extended": 11, "compact_plus_cnn_extras": 12, "full": 14}[bank]
    assert model.n_branches == expected


def test_v2a_capacity_under_500k():
    """Default V2A on a 700-feature input should stay under 500K parameters."""
    model = NiconV2A((1, 700))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params < 500_000, f"V2A has {n_params} params"


def test_v2a_trainable_ops_have_grad():
    model = NiconV2A((1, 700), bank="compact", trainable_ops=True, operator_reg_lambda=1e-3)
    x = torch.randn(4, 1, 700)
    model.fit_branches(x)
    out = model(x).squeeze(-1)
    loss = ((out - torch.randn(4)) ** 2).mean() + model.operator_regularisation_loss()
    loss.backward()
    # At least one trainable op should have a grad on its kernel/matrix.
    found = False
    for branch in model.branches:
        for name, param in branch.named_parameters(recurse=False):
            if param.grad is not None and torch.isfinite(param.grad).all():
                found = True
                break
    assert found, "no trainable operator had a finite gradient"


def test_v2a_frozen_ops_have_no_grad():
    model = NiconV2A((1, 700), bank="compact", trainable_ops=False)
    for branch in model.branches:
        for _, param in branch.named_parameters(recurse=False):
            assert not param.requires_grad


def test_v2a_msc_branch_fitted_on_train():
    """When `bank='full'`, MSC reference must equal train mean."""
    rng_x = torch.from_numpy(__import__("numpy").random.default_rng(0).normal(size=(64, 1, 200)).astype("float32"))
    model = NiconV2A((1, 200), bank="full")
    model.fit_branches(rng_x)
    msc = next(b for b in model.branches if b.__class__.__name__ == "MSCOperator")
    assert msc._fitted
    train_mean = rng_x[:, 0, :].mean(dim=0)
    assert torch.allclose(msc.reference, train_mean, atol=1e-5)


def test_v2a_rms_branch_norm_fitted_after_first_train_forward():
    model = NiconV2A((1, 200), bank="compact")
    x = torch.randn(8, 1, 200) * 5.0
    model.fit_branches(x)
    for norm in model.branch_norms:
        assert norm.fitted.item() == 1
        assert norm.rms.item() > 0
