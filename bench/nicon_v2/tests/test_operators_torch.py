"""Tests for the AOM-operator-as-PyTorch-layer utility."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import savgol_filter

torch = pytest.importorskip("torch")

from nicon_v2.operators_torch import (
    FrozenConvOperator,
    FrozenMatrixOperator,
    MSCOperator,
    SNVOperator,
    _detrend_matrix,
    _gaussian_kernel,
    _identity_kernel,
    _sg_kernel,
    aom_compact_branches_torch,
    extended_branches_torch,
)


@pytest.mark.parametrize("p", [401, 576, 700, 1154])
def test_compact_branches_preserve_length(p: int):
    branches = aom_compact_branches_torch(p=p)
    x = torch.randn(2, 1, p)
    for b in branches:
        out = b(x)
        assert out.shape == x.shape, f"{b.name} on p={p}: {out.shape}"


def test_extended_branches_count_and_names():
    """``extended_branches_torch`` is now an alias of ``full_branches_torch`` (Codex round 5):
    11 strict-linear AOM ops + 3 CNN-only extras = 14."""
    branches = extended_branches_torch(p=576)
    assert len(branches) == 14
    names = [b.name for b in branches]
    expected = {"identity", "sg_smooth_w11p2", "sg_smooth_w21p3", "sg_d1_w11p2",
                "sg_d1_w21p3", "sg_d2_w11p2", "detrend_d1", "detrend_d2",
                "fd_o1", "nw_g5_s5", "whittaker_l10",
                "gauss_s1.5", "snv", "msc"}
    assert set(names) == expected


def test_identity_kernel_is_identity():
    layer = FrozenConvOperator(_identity_kernel())
    x = torch.randn(2, 1, 200)
    assert torch.allclose(layer(x), x, atol=1e-6)


def test_sg_kernel_matches_scipy_on_interior():
    layer = FrozenConvOperator(_sg_kernel(11, 2, 1))
    x = np.random.default_rng(0).normal(size=400).astype(np.float32)
    out = layer(torch.from_numpy(x).view(1, 1, -1)).numpy().ravel()
    ref = savgol_filter(x.astype(np.float64), 11, 2, deriv=1, mode="interp")
    pad = 5
    assert np.allclose(out[pad:-pad], ref[pad:-pad], atol=1e-3)


def test_detrend_matrix_removes_linear_trend():
    p = 200
    A = _detrend_matrix(p, degree=1)
    t = np.linspace(-1, 1, p)
    # A linear-only signal should be projected to zero (up to numerical noise).
    y = (3.0 + 2.0 * t).astype(np.float32)
    x = torch.from_numpy(y).view(1, 1, -1)
    layer = FrozenMatrixOperator(A)
    out = layer(x).numpy().ravel()
    assert np.allclose(out, 0.0, atol=1e-4), f"max residual = {np.abs(out).max()}"


def test_detrend_matrix_preserves_quadratic_when_degree_1():
    p = 200
    A = _detrend_matrix(p, degree=1)
    t = np.linspace(-1, 1, p)
    y = (t * t).astype(np.float32)
    x = torch.from_numpy(y).view(1, 1, -1)
    layer = FrozenMatrixOperator(A)
    out = layer(x).numpy().ravel()
    # Quadratic part survives; mean and slope should be ~0.
    assert abs(out.mean()) < 0.05
    assert np.std(out) > 0.1


def test_snv_zero_mean_unit_var():
    snv = SNVOperator()
    x = torch.from_numpy(np.random.default_rng(0).normal(loc=5.0, scale=2.0, size=(8, 1, 200)).astype(np.float32))
    out = snv(x).numpy()
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(out.std(axis=-1), 1.0, atol=1e-3)


def test_msc_fit_uses_mean_of_supplied_rows_only():
    rng = np.random.default_rng(0)
    base = rng.normal(size=200).astype(np.float32)
    n = 32
    a = rng.uniform(-1, 1, size=n).astype(np.float32)
    b = rng.uniform(0.5, 1.5, size=n).astype(np.float32)
    X_train = (a[:, None] + b[:, None] * base).astype(np.float32)
    msc = MSCOperator(p=200, trainable=False).fit(X_train)
    # Reference must equal column-mean of X_train.
    assert np.allclose(msc.reference.detach().numpy(), X_train.mean(axis=0), atol=1e-5)
    out = msc(torch.from_numpy(X_train).unsqueeze(1)).squeeze(1).numpy()
    out_centered = out - out.mean(axis=-1, keepdims=True)
    base_centered = base - base.mean()
    cor = np.array([np.corrcoef(out_centered[i], base_centered)[0, 1] for i in range(n)])
    assert (cor > 0.99).all()


def test_trainable_kernel_has_grad():
    layer = FrozenConvOperator(_sg_kernel(11, 2, 1), trainable=True, reg_lambda=1e-3)
    x = torch.randn(2, 1, 200, requires_grad=False)
    y = torch.randn(2, 1, 200)
    out = layer(x)
    loss = ((out - y) ** 2).mean() + layer.regularisation_loss()
    loss.backward()
    assert layer.kernel.grad is not None
    assert torch.all(torch.isfinite(layer.kernel.grad))


def test_frozen_kernel_has_no_grad():
    layer = FrozenConvOperator(_sg_kernel(11, 2, 1), trainable=False)
    # Buffer (not a Parameter) → has no grad slot
    assert not isinstance(layer.kernel, torch.nn.Parameter)


def test_regularisation_loss_zero_when_kernel_unchanged():
    layer = FrozenConvOperator(_sg_kernel(11, 2, 1), trainable=True, reg_lambda=1e-2)
    val = layer.regularisation_loss().item()
    assert abs(val) < 1e-9
