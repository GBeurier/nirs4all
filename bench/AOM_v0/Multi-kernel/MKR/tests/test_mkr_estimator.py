"""Tests for AOMMultiKernelRidge (mkR) sklearn estimator."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.utils.estimator_checks import check_is_fitted

from aomridge.mkr_estimator import AOMMultiKernelRidge


def _smooth_X_y(n: int, p: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    grid = np.arange(p, dtype=float)
    X = np.zeros((n, p), dtype=float)
    for i in range(n):
        for k in range(3):
            c = rng.uniform(0.1 * p, 0.9 * p)
            w = rng.uniform(0.05 * p, 0.15 * p)
            a = rng.normal()
            X[i] += a * np.exp(-((grid - c) ** 2) / (2 * w ** 2))
    X += rng.normal(0, 0.05, size=(n, p))
    beta = rng.normal(0, 0.5, size=p)
    y = X @ beta + rng.normal(0, 0.1, size=n)
    return X, y


# ----------------------------------------------------------------------
# Sklearn API basics
# ----------------------------------------------------------------------


def test_clone_preserves_params():
    est = AOMMultiKernelRidge(weight_strategy="kta", weight_top_k=3)
    cloned = clone(est)
    assert cloned.weight_strategy == "kta"
    assert cloned.weight_top_k == 3
    # cloned must be unfitted
    assert not hasattr(cloned, "coef_") or cloned.__dict__.get("coef_") is None


def test_get_set_params():
    est = AOMMultiKernelRidge()
    p = est.get_params()
    assert "weight_strategy" in p
    est.set_params(weight_strategy="kta")
    assert est.weight_strategy == "kta"


def test_fit_predict_uniform():
    X, y = _smooth_X_y(50, 60, seed=0)
    est = AOMMultiKernelRidge(
        weight_strategy="uniform",
        alpha_grid_size=20,
        alpha_cv_n_splits=3,
        random_state=0,
    )
    est.fit(X, y)
    check_is_fitted(est, ["coef_", "dual_coef_", "alpha_", "eta_"])
    # eta is simplex
    assert est.eta_.sum() == pytest.approx(1.0, abs=1e-10)
    # predict returns finite (n,) array
    y_pred = est.predict(X)
    assert y_pred.shape == (50,)
    assert np.all(np.isfinite(y_pred))
    # score is finite
    assert np.isfinite(est.score(X, y))


def test_fit_predict_kta():
    X, y = _smooth_X_y(50, 60, seed=1)
    est = AOMMultiKernelRidge(
        weight_strategy="kta",
        alpha_grid_size=15,
        alpha_cv_n_splits=3,
        random_state=0,
    )
    est.fit(X, y)
    assert est.eta_.sum() == pytest.approx(1.0, abs=1e-10)


def test_fit_predict_manual():
    X, y = _smooth_X_y(50, 60, seed=2)
    # We need to know B; quick fit a uniform estimator first.
    quick = AOMMultiKernelRidge(weight_strategy="uniform", alpha_grid_size=5,
                                  alpha_cv_n_splits=2)
    quick.fit(X, y)
    B = quick.B_
    user_init = np.linspace(0.1, 1.0, B)
    est = AOMMultiKernelRidge(
        weight_strategy="manual",
        weight_init=user_init,
        alpha_grid_size=15,
        alpha_cv_n_splits=3,
        random_state=0,
    )
    est.fit(X, y)
    # Manual eta after projection should match the simplex of user_init.
    expected = user_init / user_init.sum()
    np.testing.assert_allclose(est.eta_, expected, rtol=1e-10)


def test_fit_predict_softmax_cv_runs():
    X, y = _smooth_X_y(60, 50, seed=3)
    est = AOMMultiKernelRidge(
        weight_strategy="softmax_cv",
        weight_n_restarts=1,
        weight_max_iter=10,
        alpha_grid_size=10,
        alpha_cv_n_splits=3,
        random_state=0,
    )
    est.fit(X, y)
    assert est.eta_.sum() == pytest.approx(1.0, abs=1e-10)
    assert est.alpha_ > 0.0


# ----------------------------------------------------------------------
# Primal / dual agreement
# ----------------------------------------------------------------------


def test_primal_dual_agreement():
    X_train, y_train = _smooth_X_y(40, 50, seed=10)
    X_test, _ = _smooth_X_y(15, 50, seed=11)
    est = AOMMultiKernelRidge(
        weight_strategy="uniform",
        alpha_grid_size=10,
        alpha_cv_n_splits=3,
        random_state=0,
    )
    est.fit(X_train, y_train)
    y_primal = est.predict(X_test)
    y_dual = est.predict_dual(X_test)
    np.testing.assert_allclose(y_primal, y_dual, rtol=1e-7, atol=1e-7)


def test_predict_train_matches_intercept_plus_coef():
    X, y = _smooth_X_y(30, 40, seed=12)
    est = AOMMultiKernelRidge(
        weight_strategy="uniform",
        alpha_grid_size=10,
        alpha_cv_n_splits=3,
        random_state=0,
    )
    est.fit(X, y)
    expected = X @ est.coef_ + est.intercept_
    np.testing.assert_allclose(est.predict(X), expected, rtol=1e-12)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def test_rejects_2d_y():
    X, y = _smooth_X_y(20, 30, seed=20)
    Y = np.column_stack([y, y])
    est = AOMMultiKernelRidge(weight_strategy="uniform", alpha_grid_size=5,
                                alpha_cv_n_splits=2)
    with pytest.raises(ValueError, match="multi-output"):
        # Reshape to (n, 1) is also rejected (only 1D y in v1).
        est.fit(X, Y)


def test_rejects_unknown_strategy():
    X, y = _smooth_X_y(20, 30, seed=21)
    est = AOMMultiKernelRidge(weight_strategy="bogus", alpha_grid_size=5,
                                alpha_cv_n_splits=2)
    with pytest.raises(ValueError, match="weight_strategy"):
        est.fit(X, y)
