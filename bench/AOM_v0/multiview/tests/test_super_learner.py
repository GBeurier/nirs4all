"""Tests for Phase 11 super-learner components."""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from multiview.super_learner import (
    AdaptiveSuperLearner,
    NNLSSimplexStacker,
    TrimmedMeanEnsemble,
    _ShrinkCalibrator,
    _solve_simplex,
)


def _toy_data(n=200, p=80, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = X[:, 20:50].sum(axis=1) + 0.1 * rng.standard_normal(n)
    return X, y


class TestTrimmedMean:
    def test_drops_extremes(self):
        X, y = _toy_data(n=120, p=50, seed=1)
        bases = [
            (f"pls{k}", PLSRegression(n_components=k)) for k in (3, 5, 7, 10, 15)
        ]
        est = TrimmedMeanEnsemble(bases=bases, n_drop=1)
        est.fit(X[:100], y[:100])
        pred = est.predict(X[100:])
        rmse = float(np.sqrt(((y[100:] - pred) ** 2).mean()))
        assert rmse < 0.5 * float(y[100:].std())


class TestSimplex:
    def test_simplex_weights_nonneg_sum_one(self):
        rng = np.random.default_rng(0)
        Z = rng.standard_normal((50, 3))
        y = Z[:, 0] * 0.5 + Z[:, 1] * 0.3 + Z[:, 2] * 0.2 + 0.05 * rng.standard_normal(50)
        w = _solve_simplex(Z, y)
        assert (w >= 0).all()
        assert abs(w.sum() - 1.0) < 1e-3


class TestShrinkCalibrator:
    def test_perfect_yhat_passthrough(self):
        rng = np.random.default_rng(2)
        y = rng.standard_normal(200)
        yhat = y.copy()
        cal = _ShrinkCalibrator().fit(yhat, y)
        # Should be ~ identity (a=0, b=1)
        assert abs(cal.a_) < 0.05
        assert abs(cal.b_ - 1.0) < 0.05

    def test_biased_yhat_calibrates(self):
        rng = np.random.default_rng(3)
        y = rng.standard_normal(200)
        yhat = 2.0 + 1.5 * y + 0.05 * rng.standard_normal(200)
        cal = _ShrinkCalibrator(shrinkage_lambda=0.1).fit(yhat, y)
        # Inverse mapping: y = a + b yhat, with yhat = 2 + 1.5 y → y = (yhat - 2) / 1.5
        # → a ≈ -2/1.5 ≈ -1.33, b ≈ 1/1.5 ≈ 0.667
        assert abs(cal.a_ - (-1.33)) < 0.2
        assert abs(cal.b_ - 0.667) < 0.1


class TestNNLSStacker:
    def test_stacker_runs(self):
        X, y = _toy_data(n=200, p=80, seed=4)
        bases = [
            (f"pls{k}", PLSRegression(n_components=k)) for k in (3, 5, 10)
        ]
        est = NNLSSimplexStacker(bases=bases, n_oof_folds=3, random_state=0)
        est.fit(X[:160], y[:160])
        pred = est.predict(X[160:])
        rmse = float(np.sqrt(((y[160:] - pred) ** 2).mean()))
        baseline = float(y[160:].std())
        assert rmse < 0.5 * baseline
        assert hasattr(est, "weights_")
        assert (est.weights_ >= 0).all()
        assert abs(est.weights_.sum() - 1.0) < 1e-3


class TestAdaptiveSuperLearner:
    def test_recipe_select_on_small_n(self):
        X, y = _toy_data(n=80, p=50, seed=5)
        bases_atoms = [
            (f"pls{k}", PLSRegression(n_components=k)) for k in (3, 5, 10)
        ]
        recipes = [
            ("pls3", PLSRegression(n_components=3)),
            ("pls10", PLSRegression(n_components=10)),
        ]
        est = AdaptiveSuperLearner(
            atoms=bases_atoms, recipes=recipes,
            small_threshold=100, big_threshold=200, random_state=0,
        )
        est.fit(X, y)
        # n=80 < 100 → recipe-select
        assert est.mode_ == "recipe-select"
        assert est.winner_ in {"pls3", "pls10"}

    def test_nnls_stack_on_big_n(self):
        X, y = _toy_data(n=300, p=50, seed=6)
        bases_atoms = [
            (f"pls{k}", PLSRegression(n_components=k)) for k in (3, 5, 10)
        ]
        est = AdaptiveSuperLearner(
            atoms=bases_atoms,
            small_threshold=100, big_threshold=200, random_state=0,
        )
        est.fit(X, y)
        assert est.mode_ == "nnls-stack"
