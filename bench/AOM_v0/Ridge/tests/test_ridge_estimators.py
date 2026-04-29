"""Phase 3 tests for AOMRidgeRegressor (selection='superblock')."""

from __future__ import annotations

import numpy as np
import pytest
from aompls.banks import compact_bank
from aompls.operators import (
    ExplicitMatrixOperator,
    IdentityOperator,
    SavitzkyGolayOperator,
)
from aomridge.estimators import AOMRidgeRegressor
from aomridge.kernels import (
    clone_operator_bank,
    compute_block_scales_from_xt,
    explicit_superblock,
    fit_operator_bank,
    resolve_operator_bank,
)
from sklearn.linear_model import Ridge


def _make_data(n=60, p=48, q=1, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    coef = rng.normal(size=(p, q))
    Y = X @ coef + noise * rng.normal(size=(n, q))
    if q == 1:
        Y = Y.ravel()
    return X, Y


def test_identity_only_matches_sklearn_ridge_univariate():
    X, y = _make_data()
    alpha = 0.42
    est = AOMRidgeRegressor(
        selection="superblock",
        operator_bank=[IdentityOperator()],
        alpha=alpha,
        block_scaling="none",
        cv=3,
        random_state=0,
    )
    est.fit(X, y)
    sk = Ridge(alpha=alpha, fit_intercept=True).fit(X, y)
    np.testing.assert_allclose(est.coef_, sk.coef_, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(est.intercept_, sk.intercept_, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(est.predict(X), sk.predict(X), atol=1e-8, rtol=1e-8)


def test_identity_only_matches_sklearn_ridge_multioutput():
    X, Y = _make_data(q=3, seed=1)
    alpha = 1.5
    est = AOMRidgeRegressor(
        selection="superblock",
        operator_bank=[IdentityOperator()],
        alpha=alpha,
        block_scaling="none",
        cv=3,
        random_state=0,
    )
    est.fit(X, Y)
    sk = Ridge(alpha=alpha, fit_intercept=True).fit(X, Y)
    # sklearn Ridge stores coef_ with shape (q, p). AOMRidge uses (p, q).
    np.testing.assert_allclose(est.coef_.T, sk.coef_, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(est.intercept_, sk.intercept_, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(est.predict(X), sk.predict(X), atol=1e-8, rtol=1e-8)


def test_single_operator_dual_equals_materialized_ridge():
    """Single-operator dual Ridge math, exercised via kernel utilities directly.

    The estimator always includes identity in its bank (per spec). To verify
    the Phase 1/2 math for a single non-identity operator we go through the
    kernel + solver primitives without auto-prepending identity.
    """
    X, y = _make_data()
    alpha = 0.7
    p = X.shape[1]
    op = SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=p)

    # Center
    x_mean = X.mean(axis=0)
    y_mean = float(np.mean(y))
    Xc = X - x_mean
    yc = y - y_mean

    # Materialized reference: Ridge on Z = Xc A^T (no auto-identity)
    Z = op.transform(Xc)
    sk = Ridge(alpha=alpha, fit_intercept=False).fit(Z, yc)
    coef_mat = op.matrix(p).T @ sk.coef_  # (p,) original-space

    # Dual reference via kernel utilities, scales = 1 (no block scaling)
    from aomridge.kernels import linear_operator_kernel_train
    from aomridge.solvers import solve_dual_ridge

    fit_operator_bank([op], Xc)
    K, U = linear_operator_kernel_train(Xc, [op], np.array([1.0]))
    C = solve_dual_ridge(K, yc, alpha=alpha, method="cholesky")
    coef_dual = U @ C  # (p,)

    np.testing.assert_allclose(coef_dual, coef_mat, atol=1e-8, rtol=1e-8)

    # K_cross @ C must equal Xtest_c @ coef_dual
    rng = np.random.default_rng(2)
    X_te = rng.normal(size=(7, p))
    Xc_te = X_te - x_mean
    from aomridge.kernels import linear_operator_kernel_cross
    K_te = linear_operator_kernel_cross(Xc_te, U)
    np.testing.assert_allclose(K_te @ C, Xc_te @ coef_dual, atol=1e-9, rtol=1e-9)
    # Materialized prediction matches dual prediction
    sk_pred = Xc_te @ op.matrix(p).T @ sk.coef_ + y_mean
    dual_pred = Xc_te @ coef_dual + y_mean
    np.testing.assert_allclose(dual_pred, sk_pred, atol=1e-8, rtol=1e-8)


def test_superblock_dual_equals_explicit_concatenated_ridge():
    X, y = _make_data(n=40, p=32, q=1, seed=3)
    alpha = 1.1
    bank_name = "compact"
    est = AOMRidgeRegressor(
        selection="superblock",
        operator_bank=bank_name,
        alpha=alpha,
        block_scaling="rms",
        cv=3,
        random_state=0,
    )
    est.fit(X, y)
    # Explicit reference: build Phi(Xc) and run sklearn Ridge with no intercept
    x_mean = X.mean(axis=0)
    y_mean = float(np.mean(y))
    Xc = X - x_mean
    yc = y - y_mean
    ops = resolve_operator_bank(bank_name, p=X.shape[1])
    fit_operator_bank(ops, Xc)
    scales = compute_block_scales_from_xt(Xc.T, ops, block_scaling="rms")
    Phi = explicit_superblock(Xc, ops, scales)
    sk = Ridge(alpha=alpha, fit_intercept=False, solver="cholesky").fit(Phi, yc)
    # Dual prediction must match explicit
    rng = np.random.default_rng(4)
    X_te = rng.normal(size=(8, X.shape[1]))
    Xc_te = X_te - x_mean
    Phi_te = explicit_superblock(Xc_te, ops, scales)
    sk_pred = Phi_te @ sk.coef_ + y_mean
    np.testing.assert_allclose(est.predict(X_te), sk_pred, atol=1e-7, rtol=1e-7)
    # coef_ has original feature shape, never wide
    assert est.coef_.shape == (X.shape[1],)


def test_coef_original_space_identity():
    X, Y = _make_data(q=2, seed=5)
    alpha = 0.3
    est = AOMRidgeRegressor(
        selection="superblock",
        operator_bank="compact",
        alpha=alpha,
        block_scaling="rms",
        cv=3,
        random_state=0,
    )
    est.fit(X, Y)
    Xc = X - est.x_mean_
    # Xc @ coef_ == K @ dual_coef_
    # Reconstruct K from U and Xc
    p = X.shape[1]
    ops = resolve_operator_bank("compact", p=p)
    fit_operator_bank(ops, Xc)
    scales = compute_block_scales_from_xt(Xc.T, ops, block_scaling="rms")
    from aomridge.kernels import linear_operator_kernel_train

    K, U = linear_operator_kernel_train(Xc, ops, scales)
    np.testing.assert_allclose(Xc @ est.coef_, K @ est.dual_coef_, atol=1e-7, rtol=1e-7)


def test_predict_univariate_returns_1d():
    X, y = _make_data()
    est = AOMRidgeRegressor(
        operator_bank="compact", alpha=1.0, cv=3, random_state=0
    ).fit(X, y)
    pred = est.predict(X[:5])
    assert pred.ndim == 1


def test_predict_multioutput_returns_2d():
    X, Y = _make_data(q=3, seed=6)
    est = AOMRidgeRegressor(
        operator_bank="compact", alpha=1.0, cv=3, random_state=0
    ).fit(X, Y)
    pred = est.predict(X[:5])
    assert pred.shape == (5, 3)


def test_feature_std_matches_standard_scaler_ridge():
    """With identity-only bank and x_scale=feature_std, AOMRidge must match
    sklearn ``Pipeline(StandardScaler, Ridge)`` to floating-point precision.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _make_data(seed=10)
    alpha = 0.7
    est = AOMRidgeRegressor(
        operator_bank=[IdentityOperator()],
        alpha=alpha,
        block_scaling="none",
        x_scale="feature_std",
        cv=3,
        random_state=0,
    ).fit(X, y)
    sk = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=alpha, fit_intercept=True)),
        ]
    ).fit(X, y)
    np.testing.assert_allclose(est.predict(X), sk.predict(X), atol=1e-7, rtol=1e-7)


def test_block_scales_high_gain_duplicate():
    X, y = _make_data(p=32, seed=7)
    high_gain = ExplicitMatrixOperator(10.0 * np.eye(32), name="ten_I")
    est = AOMRidgeRegressor(
        operator_bank=[IdentityOperator(), high_gain],
        alpha=0.5,
        block_scaling="rms",
        cv=3,
        random_state=0,
    ).fit(X, y)
    scales = est.block_scales_
    # Both blocks land on equal Frobenius norm after scaling
    assert len(scales) == 2
    assert abs(scales[0] - 10.0 * scales[1]) / max(scales[0], 1e-12) < 1e-6
