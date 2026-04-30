"""Tests for AOMMultiKernelBLUP decomposition identity."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from blup.estimator import AOMMultiKernelBLUP


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
    y = X @ beta + rng.normal(0, 0.5, size=n)
    return X, y


@pytest.fixture(scope="module")
def fitted_blup():
    X, y = _smooth_X_y(40, 50, seed=0)
    est = AOMMultiKernelBLUP(
        operator_bank="compact",
        method="reml",
        n_random_restarts=2,
        max_iter=50,
        random_state=0,
    )
    est.fit(X, y)
    return est, X, y


# ----------------------------------------------------------------------
# Sklearn API
# ----------------------------------------------------------------------


def test_clone_preserves_params():
    est = AOMMultiKernelBLUP(method="ml", n_random_restarts=2)
    cloned = clone(est)
    assert cloned.method == "ml"
    assert cloned.n_random_restarts == 2


def test_get_set_params():
    est = AOMMultiKernelBLUP()
    p = est.get_params()
    assert "method" in p
    est.set_params(method="ml")
    assert est.method == "ml"


# ----------------------------------------------------------------------
# Decomposition structure
# ----------------------------------------------------------------------


def test_decomposition_keys(fitted_blup):
    est, X, y = fitted_blup
    comps = est.predict_components(X)
    assert set(comps.keys()) == {"fixed", "random", "total"}
    assert comps["fixed"].shape == (40,)
    assert comps["total"].shape == (40,)
    assert isinstance(comps["random"], dict)
    for name in est.block_names_:
        assert name in comps["random"]
        assert comps["random"][name].shape == (40,)


# ----------------------------------------------------------------------
# Decomposition identity (the central invariant)
# ----------------------------------------------------------------------


def test_decomposition_sum_equals_predict_train(fitted_blup):
    """Critical invariant: sum components == predict on training data."""
    est, X, y = fitted_blup
    comps = est.predict_components(X)
    y_pred = est.predict(X)
    np.testing.assert_allclose(comps["total"], y_pred, rtol=1e-10, atol=1e-10)


def test_decomposition_sum_equals_predict_test(fitted_blup):
    """Sum identity must hold on unseen data too."""
    est, X_train, y_train = fitted_blup
    rng = np.random.default_rng(12345)
    p = X_train.shape[1]
    X_test = rng.normal(size=(15, p))
    comps = est.predict_components(X_test)
    y_pred = est.predict(X_test)
    np.testing.assert_allclose(comps["total"], y_pred, rtol=1e-10, atol=1e-10)


def test_train_decompose(fitted_blup):
    est, X, y = fitted_blup
    comps = est.train_decompose()
    assert comps["total"].shape == (40,)
    np.testing.assert_allclose(comps["total"], est.predict(X), rtol=1e-10)


def test_contribution_table_shape(fitted_blup):
    pytest.importorskip("pandas")
    est, X, y = fitted_blup
    df = est.contribution_table(X)
    expected_cols = {
        "sample_index", "component_type", "block_name",
        "contribution", "contribution_norm", "contribution_relative",
    }
    assert set(df.columns) == expected_cols
    expected_rows = 40 * (1 + est.B_)
    assert len(df) == expected_rows


# ----------------------------------------------------------------------
# Predict identity check on a 2D scalar y
# ----------------------------------------------------------------------


def test_score_finite(fitted_blup):
    est, X, y = fitted_blup
    s = est.score(X, y)
    assert np.isfinite(s)


def test_predict_shape_test(fitted_blup):
    est, X_train, _ = fitted_blup
    rng = np.random.default_rng(99)
    X_test = rng.normal(size=(8, X_train.shape[1]))
    y_pred = est.predict(X_test)
    assert y_pred.shape == (8,)
    assert np.all(np.isfinite(y_pred))


def test_no_leakage_train_X_stored(fitted_blup):
    """predict_components must not mutate _train_X_ between calls."""
    est, X_train, _ = fitted_blup
    snapshot = est._train_X_.copy()
    _ = est.predict_components(X_train)
    np.testing.assert_array_equal(est._train_X_, snapshot)
