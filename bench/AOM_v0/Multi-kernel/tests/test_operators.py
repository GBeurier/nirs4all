"""Tests for linear spectral operators.

Each operator must satisfy:

- `transform(X)` returns the right shape.
- linearity: `A(a x + b y) = a A x + b A y`.
- adjoint identity: `<A x, y> = <x, A^T y>`.
- covariance identity: `(X A^T)^T Y = A X^T Y`.
- explicit matrix consistency: `transform(X)[i] = (matrix @ X[i])` etc.
- composition consistency.
"""

from __future__ import annotations

import numpy as np
import pytest

from aompls.banks import compact_bank, default_bank, extended_bank
from aompls.operators import (
    ComposedOperator,
    DetrendProjectionOperator,
    ExplicitMatrixOperator,
    FiniteDifferenceOperator,
    IdentityOperator,
    LinearSpectralOperator,
    NorrisWilliamsOperator,
    SavitzkyGolayOperator,
    WhittakerOperator,
)


def _all_operators_for_p(p: int):
    """Return a representative set of operators bound to feature dim `p`."""
    rng = np.random.default_rng(0)
    return [
        IdentityOperator(p=p),
        SavitzkyGolayOperator(window_length=7, polyorder=2, deriv=0, p=p),
        SavitzkyGolayOperator(window_length=7, polyorder=2, deriv=1, p=p),
        SavitzkyGolayOperator(window_length=7, polyorder=3, deriv=2, p=p),
        FiniteDifferenceOperator(order=1, p=p),
        FiniteDifferenceOperator(order=2, p=p),
        DetrendProjectionOperator(degree=1, p=p),
        DetrendProjectionOperator(degree=2, p=p),
        NorrisWilliamsOperator(gap=2, smoothing=3, order=1, p=p),
        NorrisWilliamsOperator(gap=2, smoothing=3, order=2, p=p),
        WhittakerOperator(lam=10.0, p=p),
        ComposedOperator(
            [
                DetrendProjectionOperator(degree=1, p=p),
                FiniteDifferenceOperator(order=1, p=p),
            ],
            name="detrend1_fd1",
        ),
        ExplicitMatrixOperator(rng.standard_normal((p, p))),
    ]


def test_transform_shape():
    p = 32
    rng = np.random.default_rng(1)
    X = rng.standard_normal((10, p))
    for op in _all_operators_for_p(p):
        Y = op.transform(X)
        assert Y.shape == X.shape, f"{op.name}: shape mismatch"


@pytest.mark.parametrize("p", [16, 31])
def test_linearity(p):
    rng = np.random.default_rng(2)
    X1 = rng.standard_normal((4, p))
    X2 = rng.standard_normal((4, p))
    a, b = 0.7, -0.4
    for op in _all_operators_for_p(p):
        lhs = op.transform(a * X1 + b * X2)
        rhs = a * op.transform(X1) + b * op.transform(X2)
        assert np.allclose(lhs, rhs, atol=1e-9), f"{op.name}: linearity violated"


@pytest.mark.parametrize("p", [16, 25])
def test_adjoint_identity(p):
    rng = np.random.default_rng(3)
    for op in _all_operators_for_p(p):
        x = rng.standard_normal(p)
        y = rng.standard_normal(p)
        Ax = op.transform(x.reshape(1, -1)).ravel()  # (A x^T)^T  - acts on row
        # We need `< A x_col, y >` where A is the operator viewed as
        # left-multiplication on column vectors. With our row convention,
        # `A x_col` corresponds to `apply_cov(x)`.
        Ax_col = op.apply_cov(x)
        ATy = op.adjoint_vec(y)
        lhs = float(Ax_col @ y)
        rhs = float(x @ ATy)
        assert np.isclose(lhs, rhs, atol=1e-8), f"{op.name}: <Ax,y>={lhs} vs <x,A^T y>={rhs}"


@pytest.mark.parametrize("p", [16, 25])
def test_covariance_identity(p):
    """Verify (X A^T)^T Y = A X^T Y on explicit matrices."""
    rng = np.random.default_rng(4)
    n = 17
    q = 3
    X = rng.standard_normal((n, p))
    Y = rng.standard_normal((n, q))
    XtY = X.T @ Y
    for op in _all_operators_for_p(p):
        Xb = op.transform(X)
        lhs = Xb.T @ Y
        rhs = op.apply_cov(XtY)
        assert np.allclose(lhs, rhs, atol=1e-7), f"{op.name}: covariance identity violated"


@pytest.mark.parametrize("p", [12, 24])
def test_matrix_consistency(p):
    rng = np.random.default_rng(5)
    n = 7
    X = rng.standard_normal((n, p))
    for op in _all_operators_for_p(p):
        M = op.matrix(p)
        # transform(X) must equal X M^T
        out = op.transform(X)
        assert np.allclose(out, X @ M.T, atol=1e-8), f"{op.name}: transform vs matrix^T mismatch"
        # apply_cov(S) must equal M @ S
        S = rng.standard_normal((p, 2))
        assert np.allclose(op.apply_cov(S), M @ S, atol=1e-8), f"{op.name}: apply_cov mismatch"
        # adjoint_vec(v) must equal M^T v
        v = rng.standard_normal(p)
        assert np.allclose(op.adjoint_vec(v), M.T @ v, atol=1e-8), f"{op.name}: adjoint mismatch"


def test_composition_consistency():
    """Composed operator matrix matches the matrix product."""
    p = 20
    rng = np.random.default_rng(6)
    op1 = SavitzkyGolayOperator(window_length=5, polyorder=2, deriv=1, p=p)
    op2 = DetrendProjectionOperator(degree=1, p=p)
    op3 = FiniteDifferenceOperator(order=1, p=p)
    composed = ComposedOperator([op1, op2, op3])
    M_expected = op3.matrix(p) @ op2.matrix(p) @ op1.matrix(p)
    M_actual = composed.matrix(p)
    assert np.allclose(M_actual, M_expected, atol=1e-8)
    X = rng.standard_normal((5, p))
    Y_expected = X @ M_expected.T
    assert np.allclose(composed.transform(X), Y_expected, atol=1e-8)


def test_identity_explicit_application():
    p = 8
    op = IdentityOperator(p=p)
    rng = np.random.default_rng(7)
    X = rng.standard_normal((3, p))
    assert np.allclose(op.transform(X), X)
    assert np.allclose(op.matrix(p), np.eye(p))


def test_savgol_d0_preserves_constant():
    """A constant signal must pass through SG smoothing essentially unchanged
    in the interior."""
    op = SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=0, p=64)
    x = np.ones(64)
    out = op.transform(x.reshape(1, -1)).ravel()
    interior = out[10:-10]
    assert np.allclose(interior, 1.0, atol=1e-8)


def test_savgol_d1_recovers_linear_slope():
    """First-derivative SG on a linear signal must return the slope (interior)."""
    p = 64
    op = SavitzkyGolayOperator(window_length=7, polyorder=2, deriv=1, p=p)
    slope = 0.31
    x = slope * np.arange(p, dtype=float)
    out = op.transform(x.reshape(1, -1)).ravel()
    interior = out[5:-5]
    assert np.allclose(interior, slope, atol=1e-8)


def test_finite_difference_first_order_signal():
    p = 32
    op = FiniteDifferenceOperator(order=1, p=p)
    x = np.arange(p, dtype=float)
    out = op.transform(x.reshape(1, -1)).ravel()
    # Centered first difference of arange(p) is 1.0 in the interior.
    interior = out[1:-1]
    assert np.allclose(interior, 1.0, atol=1e-8)


def test_detrend_removes_linear_baseline():
    p = 64
    op = DetrendProjectionOperator(degree=1, p=p)
    rng = np.random.default_rng(8)
    base = rng.standard_normal(p) * 0.05
    t = np.linspace(-1, 1, p)
    x = 0.5 + 1.7 * t + base
    out = op.transform(x.reshape(1, -1)).ravel()
    # Linear trend should be removed; only the small noise remains.
    assert np.linalg.norm(out - (base - base.mean())) < 0.5  # rough sanity


def test_whittaker_symmetric_matrix():
    p = 30
    op = WhittakerOperator(lam=50.0, p=p)
    M = op.matrix(p)
    assert np.allclose(M, M.T, atol=1e-8)


def test_bank_presets_resolve():
    """All preset banks must produce non-empty operator lists with identity first."""
    for builder in (compact_bank, default_bank, extended_bank):
        bank = builder(p=64)
        assert len(bank) >= 1
        assert isinstance(bank[0], IdentityOperator)
        # All operators must be instances of the protocol class.
        for op in bank:
            assert isinstance(op, LinearSpectralOperator)
            assert op.is_linear_at_apply()


def test_apply_cov_accepts_1d_and_2d():
    """apply_cov must accept both vector and matrix inputs."""
    p = 16
    rng = np.random.default_rng(9)
    op = SavitzkyGolayOperator(window_length=5, polyorder=2, deriv=1, p=p)
    s = rng.standard_normal(p)
    out_vec = op.apply_cov(s)
    out_mat = op.apply_cov(s.reshape(-1, 1)).ravel()
    assert np.allclose(out_vec, out_mat, atol=1e-10)
