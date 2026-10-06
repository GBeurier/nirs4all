"""Hyperparameter trials must fit supervised feature selection within each fold."""

import numpy as np
import pytest
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.controllers.models.sklearn_model import SklearnModelController
from nirs4all.optimization.preprocessing import finetune_fold, finetune_inputs


@pytest.mark.parametrize("engine,approach,phases", [
    ("optuna", "grouped", False), ("optuna", "individual", False), ("optuna", "single", False),
    ("optuna", "grouped", True), ("optuna", "single", True),
    ("n4m", "grouped", False), ("n4m", "individual", False), ("n4m", "single", False),
])
def test_finetuning_feature_selection_matches_independent_training_only_fit(tmp_path, monkeypatch, engine, approach, phases):
    rng = np.random.default_rng(19)
    X = rng.normal(size=(36, 12))
    y = rng.normal(size=36)
    observed = []
    original = SklearnModelController._train_model

    def observe(self, model, X_train, y_train, X_val=None, y_val=None, **kwargs):
        if X_val is not None and len(X_val):
            train_ids = np.array([np.argmin(np.abs(y - target)) for target in np.asarray(y_train).reshape(-1)])
            val_ids = np.array([np.argmin(np.abs(y - target)) for target in np.asarray(y_val).reshape(-1)])
            independent = SelectKBest(f_regression, k=2).fit(X[train_ids], y[train_ids])
            np.testing.assert_allclose(X_train, independent.transform(X[train_ids]))
            np.testing.assert_allclose(X_val, independent.transform(X[val_ids]))
            observed.append((tuple(train_ids), tuple(val_ids)))
        return original(self, model, X_train, y_train, X_val, y_val, **kwargs)

    monkeypatch.setattr(SklearnModelController, "_train_model", observe)
    tuning = {"engine": engine, "approach": approach, "n_trials": 1, "model_params": {"alpha": [1.0]}, "metric": "rmse"}
    if phases:
        tuning["phases"] = [{"n_trials": 1, "sampler": "random"}, {"n_trials": 1, "sampler": "random"}]
    result = nirs4all.run([KFold(3), SelectKBest(f_regression, k=2), {"model": Ridge(), "finetune_params": tuning}],
                         (X, y), engine="legacy", refit=False, verbose=0, save_charts=False,
                         workspace_path=tmp_path / "workspace")
    assert result.predictions.num_predictions > 0
    # Three final CV fits plus every hyperparameter evaluation must be observed.
    assert len(observed) >= 4


def test_meta_tuning_preserves_reconstructed_oof_features():
    """A meta learner's inputs are prediction columns, not raw spectral channels."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    raw_plan = Mock(active=True)
    context = SimpleNamespace(custom={"cv_preprocessing": raw_plan, "meta_operator": True,
                                      "finetune_sample_ids": np.arange(6), "finetune_preprocessing_cache": {}})
    predictions = np.column_stack([np.arange(6), np.arange(6) ** 2])
    controller = Mock()
    inputs, local = finetune_inputs(None, context, controller, predictions)
    assert inputs is predictions and local is context
    controller._get_partition_sample_indices.assert_not_called()
    raw_plan.raw_features.assert_not_called()
    train, validation = finetune_fold(None, context, inputs, [0, 1, 2, 3], [4, 5])
    np.testing.assert_array_equal(train, predictions[:4])
    np.testing.assert_array_equal(validation, predictions[4:])
    raw_plan.prepare_fold.assert_not_called()
