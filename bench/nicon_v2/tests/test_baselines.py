"""Quick sklearn-based baseline tests; deterministic on synthetic data."""

from __future__ import annotations

import numpy as np

from nicon_v2.models.baseline import PLSBaseline, RidgeBaseline


def _toy_problem(n: int = 80, p: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=0.3, size=p)
    X = rng.normal(size=(n, p))
    y = X @ w + rng.normal(scale=0.1, size=n)
    X_test = rng.normal(size=(n // 2, p))
    y_test = X_test @ w + rng.normal(scale=0.1, size=n // 2)
    return X, y, X_test, y_test


def test_ridge_baseline_fit_predict_beats_mean():
    X, y, X_test, y_test = _toy_problem()
    model = RidgeBaseline(seed=0).fit(X, y)
    pred = model.predict(X_test)
    rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
    rmse_mean = float(np.sqrt(np.mean((y_test - np.mean(y)) ** 2)))
    assert rmse < rmse_mean
    assert model.selected_alpha_ in model.alphas


def test_pls_baseline_selects_valid_n_components():
    X, y, X_test, y_test = _toy_problem(n=40, p=80)
    model = PLSBaseline(seed=0).fit(X, y)
    assert 1 <= model.selected_n_components_ <= min(X.shape[0] - 1, X.shape[1])
    pred = model.predict(X_test)
    rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
    rmse_mean = float(np.sqrt(np.mean((y_test - np.mean(y)) ** 2)))
    assert rmse < rmse_mean


def test_ridge_predict_shape_matches_input():
    X, y, X_test, _ = _toy_problem()
    model = RidgeBaseline(seed=0).fit(X, y)
    pred = model.predict(X_test)
    assert pred.shape == (X_test.shape[0],)
