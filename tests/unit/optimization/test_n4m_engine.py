"""Unit tests for the native (n4m / libn4m) finetuning engine.

Skipped when the native optimizer is not installed
(``from n4m.model_selection.optimizer import Optimizer`` fails).
"""
from __future__ import annotations

import numpy as np
import pytest

from nirs4all.optimization.n4m_engine import N4MFinetuneManager

pytestmark = pytest.mark.skipif(
    not N4MFinetuneManager().is_available,
    reason="native n4m optimizer not available",
)


class _StubController:
    """Minimal controller implementing the finetune hook contract."""

    def __init__(self):
        self.seen_params = []

    def _get_model_instance(self, dataset, model_config, force_params=None):
        from sklearn.cross_decomposition import PLSRegression
        self.seen_params.append(dict(force_params or {}))
        return PLSRegression(**(force_params or {}))

    def _prepare_data(self, X, y, context):
        return np.asarray(X, float), np.asarray(y, float).ravel()

    def _train_model(self, model, X_tr, y_tr, X_val, y_val, **kw):
        model.fit(X_tr, y_tr)
        return model

    def _evaluate_model(self, model, X_val, y_val, metric=None, direction="minimize"):
        from sklearn.metrics import mean_squared_error
        return float(np.sqrt(mean_squared_error(y_val, model.predict(X_val))))


class _StubDataset:
    task_type = "regression"


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 12))
    # y depends on a few features so PLS n_components matters
    y = X[:, :4] @ np.array([2.0, -1.0, 0.5, 1.5]) + 0.1 * rng.standard_normal(80)
    from sklearn.model_selection import KFold
    folds = list(KFold(4, shuffle=True, random_state=1).split(X))
    return X, y, folds


def _run(engine, data, ft):
    X, y, folds = data
    return engine.finetune(
        _StubDataset(), {"model": None}, X, y, None, None, folds,
        dict(ft, engine="n4m"), None, _StubController())


def test_grouped_tpe_finds_best(data):
    res = _run(N4MFinetuneManager(), data, {
        "n_trials": 20, "sampler": "tpe", "approach": "grouped",
        "seed": 3, "metric": "rmse", "model_params": {"n_components": ("int", 1, 10)}})
    assert "n_components" in res.best_params
    assert 1 <= res.best_params["n_components"] <= 10
    assert res.best_value >= 0.0
    assert res.n_trials == 20
    assert len(res.trials) == 20


def test_determinism(data):
    ft = {"n_trials": 12, "sampler": "tpe", "approach": "grouped", "seed": 7,
          "metric": "rmse", "model_params": {"n_components": ("int", 1, 10)}}
    a = _run(N4MFinetuneManager(), data, ft).best_params
    b = _run(N4MFinetuneManager(), data, ft).best_params
    assert a == b


@pytest.mark.parametrize("sampler", ["random", "sobol", "lhs", "cmaes", "ga", "pso", "gp_ei", "ternary"])
def test_all_samplers_run(data, sampler):
    res = _run(N4MFinetuneManager(), data, {
        "n_trials": 10, "sampler": sampler, "approach": "grouped", "seed": 1,
        "metric": "rmse", "model_params": {"n_components": ("int", 1, 8)}})
    assert 1 <= res.best_params["n_components"] <= 8


@pytest.mark.parametrize("pruner", ["median", "successive_halving", "hyperband", "racing"])
def test_pruners_run(data, pruner):
    res = _run(N4MFinetuneManager(), data, {
        "n_trials": 15, "sampler": "tpe", "pruner": pruner, "approach": "grouped",
        "seed": 1, "metric": "rmse", "model_params": {"n_components": ("int", 1, 8)}})
    assert res.n_trials == 15
    assert res.n_pruned >= 0


def test_single_holdout(data):
    res = _run(N4MFinetuneManager(), data, {
        "n_trials": 8, "sampler": "tpe", "approach": "single", "seed": 1,
        "metric": "rmse", "model_params": {"n_components": ("int", 1, 8)}})
    assert 1 <= res.best_params["n_components"] <= 8


def test_matches_optuna_optimum(data):
    """On a clean problem the native TPE should reach the same integer optimum."""
    from nirs4all.optimization.optuna import OptunaManager
    X, y, folds = data
    ft = {"n_trials": 30, "sampler": "tpe", "approach": "grouped", "seed": 5,
          "metric": "rmse", "model_params": {"n_components": ("int", 1, 10)}}
    n4m = _run(N4MFinetuneManager(), data, ft)
    opt = OptunaManager().finetune(
        _StubDataset(), model_config={"model": None}, X_train=X, y_train=y,
        X_test=None, y_test=None, folds=folds, finetune_params=dict(ft), context=None,
        controller=_StubController())
    assert n4m.best_params["n_components"] == opt.best_params["n_components"]


def test_categorical_and_log_dsl(data):
    """Categorical + float_log axes compile and resolve to real Python values."""
    X, y, folds = data

    class RidgeCtl(_StubController):
        def _get_model_instance(self, dataset, model_config, force_params=None):
            from sklearn.linear_model import Ridge
            self.seen_params.append(dict(force_params or {}))
            return Ridge(**(force_params or {}))

    res = N4MFinetuneManager().finetune(
        _StubDataset(), {"model": None}, X, y, None, None, folds,
        {"engine": "n4m", "n_trials": 10, "sampler": "tpe", "approach": "grouped",
         "seed": 1, "metric": "rmse",
         "model_params": {"alpha": ("float_log", 1e-3, 1e2), "fit_intercept": [True, False]}},
        None, RidgeCtl())
    assert isinstance(res.best_params["alpha"], float)
    assert isinstance(res.best_params["fit_intercept"], bool)
    assert 1e-3 <= res.best_params["alpha"] <= 1e2
