"""Tests for NIPALS engines."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression

from aompls.nipals import (
    nipals_adjoint,
    nipals_materialized_fixed,
    nipals_materialized_per_component,
    nipals_pls_standard,
    nipals_standard,
)
from aompls.operators import (
    DetrendProjectionOperator,
    ExplicitMatrixOperator,
    FiniteDifferenceOperator,
    IdentityOperator,
    SavitzkyGolayOperator,
)
from aompls.synthetic import make_regression, small_pls1_dataset


@pytest.fixture
def regression_data():
    ds = make_regression(n_train=60, n_test=30, p=80, random_state=0)
    Xc = ds.X_train - ds.X_train.mean(axis=0)
    yc = ds.y_train - ds.y_train.mean()
    return Xc, yc, ds


def test_nipals_standard_pls1_shapes(regression_data):
    Xc, yc, _ = regression_data
    K = 5
    W, T, P, Q, U = nipals_standard(Xc, yc, K)
    assert W.shape[0] == Xc.shape[1]
    assert W.shape[1] <= K
    assert T.shape == (Xc.shape[0], W.shape[1])


def test_nipals_pls_standard_matches_sklearn(regression_data):
    Xc, yc, _ = regression_data
    K = 5
    res = nipals_pls_standard(Xc, yc, K)
    coef = res.coef().ravel()
    ref = PLSRegression(n_components=K, scale=False)
    ref.fit(Xc, yc)
    sk_coef = ref.coef_.ravel()
    # Predictions on the training set should match within tolerance.
    y_aom = (Xc @ coef).ravel()
    y_sk = (Xc @ sk_coef).ravel()
    assert np.allclose(y_aom, y_sk, atol=1e-6)


def test_identity_only_aom_matches_pls_standard(regression_data):
    """Identity-only AOM/POP must reduce to standard PLS."""
    Xc, yc, _ = regression_data
    K = 6
    res_pls = nipals_pls_standard(Xc, yc, K)
    res_aom = nipals_materialized_per_component(
        Xc, yc, [IdentityOperator(p=Xc.shape[1])], [0] * K, K
    )
    res_pop = nipals_adjoint(Xc, yc, [IdentityOperator(p=Xc.shape[1])], [0] * K, K)
    pred_pls = Xc @ res_pls.coef().ravel()
    pred_aom = Xc @ res_aom.coef().ravel()
    pred_pop = Xc @ res_pop.coef().ravel()
    assert np.allclose(pred_pls, pred_aom, atol=1e-7)
    assert np.allclose(pred_pls, pred_pop, atol=1e-7)


def test_single_operator_materialized_matches_adjoint(regression_data):
    """Materialized fixed and adjoint engines must agree on a single operator."""
    Xc, yc, _ = regression_data
    K = 5
    op = SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=Xc.shape[1])
    res_mat = nipals_materialized_fixed(Xc, yc, op, K)
    res_adj = nipals_adjoint(Xc, yc, [op], [0] * K, K)
    pred_mat = Xc @ res_mat.coef().ravel()
    pred_adj = Xc @ res_adj.coef().ravel()
    assert np.allclose(pred_mat, pred_adj, atol=1e-6)


def test_pop_fixed_sequence_materialized_adjoint(regression_data):
    """POP with a fixed sequence must agree between materialized and adjoint."""
    Xc, yc, _ = regression_data
    K = 4
    operators = [
        IdentityOperator(p=Xc.shape[1]),
        SavitzkyGolayOperator(window_length=7, polyorder=2, deriv=1, p=Xc.shape[1]),
        DetrendProjectionOperator(degree=1, p=Xc.shape[1]),
        FiniteDifferenceOperator(order=1, p=Xc.shape[1]),
    ]
    seq = [0, 1, 2, 3]
    res_mat = nipals_materialized_per_component(Xc, yc, operators, seq, K)
    res_adj = nipals_adjoint(Xc, yc, operators, seq, K)
    pred_mat = Xc @ res_mat.coef().ravel()
    pred_adj = Xc @ res_adj.coef().ravel()
    assert np.allclose(pred_mat, pred_adj, atol=1e-6)


def test_covariance_convention_explicit():
    """Verify (X A^T)^T y = A X^T y for an explicit operator and covariance."""
    rng = np.random.default_rng(1)
    p, n = 12, 20
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    M = rng.standard_normal((p, p))
    op = ExplicitMatrixOperator(M)
    Xt = op.transform(X)
    lhs = Xt.T @ y
    rhs = op.apply_cov(X.T @ y)
    assert np.allclose(lhs, rhs, atol=1e-9)


def test_pls2_shapes(regression_data):
    Xc, yc, _ = regression_data
    Y2 = np.column_stack([yc, yc * 0.5 + 0.3])
    K = 4
    res = nipals_materialized_per_component(
        Xc, Y2, [IdentityOperator(p=Xc.shape[1])], [0] * K, K
    )
    assert res.Z.shape == (Xc.shape[1], K)
    assert res.Q.shape == (2, K)


def test_coefficients_predict_from_original(regression_data):
    """B = Z (P^T Z)^+ Q^T must reproduce predictions from the original space."""
    Xc, yc, _ = regression_data
    K = 5
    op = SavitzkyGolayOperator(window_length=9, polyorder=2, deriv=1, p=Xc.shape[1])
    res = nipals_adjoint(Xc, yc, [op], [0] * K, K)
    B = res.coef().ravel()
    pred = Xc @ B
    # Also reconstruct via the score-based formula y_hat = T @ Q^T (sanity).
    pred_t = res.T @ res.Q.T
    pred_t = pred_t.ravel()
    # Predictions must be close (not identical because pseudoinverse differs
    # from inverse for tail components, but small dataset gives good agreement).
    assert np.allclose(pred, pred_t, atol=5e-2)


def test_small_pls1_recovers_known_factors():
    X, y, Xt, yt = small_pls1_dataset(p=20, n=40, K=2, noise=0.01, random_state=0)
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    res = nipals_pls_standard(Xc, yc, 2)
    coef = res.coef().ravel()
    pred_t = (Xt - X.mean(axis=0)) @ coef + y.mean()
    rmse = np.sqrt(np.mean((pred_t - yt) ** 2))
    assert rmse < 0.1


def test_max_components_respected(regression_data):
    Xc, yc, _ = regression_data
    K_request = 30
    res = nipals_pls_standard(Xc, yc, min(K_request, Xc.shape[1], Xc.shape[0] - 1))
    assert res.Z.shape[1] <= min(K_request, Xc.shape[1], Xc.shape[0] - 1)
