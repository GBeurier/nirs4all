"""Tests for task-type-aware filtering in Predictions.

Verifies that ``top()``, ``get_best()``, and ``filter_predictions()``
correctly separate regression and classification predictions using
the ``task_type`` parameter and its aliases.
"""

import warnings

import numpy as np
import pytest

from nirs4all.data.predictions import Predictions


@pytest.fixture
def base_regression_params():
    """Base parameters shared by all regression predictions."""
    return {
        "dataset_name": "ds_reg",
        "dataset_path": "/path/to/ds_reg",
        "config_path": "/path/to/config",
        "pipeline_uid": "pipe_reg",
        "step_idx": 0,
        "op_counter": 1,
        "n_samples": 3,
        "n_features": 100,
        "task_type": "regression",
        "metric": "rmse",
    }


@pytest.fixture
def base_classification_params():
    """Base parameters shared by all classification predictions."""
    return {
        "dataset_name": "ds_clf",
        "dataset_path": "/path/to/ds_clf",
        "config_path": "/path/to/config",
        "pipeline_uid": "pipe_clf",
        "step_idx": 0,
        "op_counter": 1,
        "n_samples": 4,
        "n_features": 100,
        "task_type": "binary_classification",
        "metric": "balanced_accuracy",
    }


@pytest.fixture
def mixed_predictions(base_regression_params, base_classification_params):
    """Create a Predictions object with both regression and classification entries."""
    preds = Predictions()

    # --- Regression predictions ---
    preds.add_prediction(
        config_name="cfg1",
        model_name="PLSRegression_1",
        model_classname="PLSRegression",
        fold_id=0,
        partition="val",
        val_score=0.5,
        y_true=np.array([1.0, 2.0, 3.0]),
        y_pred=np.array([1.1, 2.1, 2.9]),
        **base_regression_params,
    )
    preds.add_prediction(
        config_name="cfg2",
        model_name="Ridge_1",
        model_classname="Ridge",
        fold_id=0,
        partition="val",
        val_score=0.3,  # Better RMSE
        y_true=np.array([1.0, 2.0, 3.0]),
        y_pred=np.array([1.05, 1.95, 3.05]),
        **base_regression_params,
    )

    # --- Classification predictions ---
    preds.add_prediction(
        config_name="cfg3",
        model_name="RF_1",
        model_classname="RandomForestClassifier",
        fold_id=0,
        partition="val",
        val_score=0.85,
        y_true=np.array([0, 1, 0, 1]),
        y_pred=np.array([0, 1, 0, 0]),
        **base_classification_params,
    )
    preds.add_prediction(
        config_name="cfg4",
        model_name="SVC_1",
        model_classname="SVC",
        fold_id=0,
        partition="val",
        val_score=0.90,
        y_true=np.array([0, 1, 0, 1]),
        y_pred=np.array([0, 1, 0, 1]),
        **base_classification_params,
    )

    return preds


# =========================================================================
# top() with task_type
# =========================================================================

class TestTopWithTaskType:
    """Verify top() respects task_type filtering and aliases."""

    def test_top_regression_only(self, mixed_predictions):
        results = mixed_predictions.top(5, rank_metric="rmse", rank_partition="val", task_type="regression", score_scope="folds")
        assert len(results) == 2
        assert all(r["task_type"] == "regression" for r in results)

    def test_top_classification_only(self, mixed_predictions):
        results = mixed_predictions.top(5, rank_metric="balanced_accuracy", rank_partition="val", task_type="classification", score_scope="folds")
        assert len(results) == 2
        assert all("classification" in r["task_type"] for r in results)

    def test_top_clf_alias(self, mixed_predictions):
        results = mixed_predictions.top(5, rank_metric="balanced_accuracy", rank_partition="val", task_type="clf", score_scope="folds")
        assert len(results) == 2
        assert all("classification" in r["task_type"] for r in results)

    def test_top_reg_alias(self, mixed_predictions):
        results = mixed_predictions.top(5, rank_metric="rmse", rank_partition="val", task_type="reg", score_scope="folds")
        assert len(results) == 2
        assert all(r["task_type"] == "regression" for r in results)

    def test_top_binary_alias(self, mixed_predictions):
        results = mixed_predictions.top(5, rank_metric="balanced_accuracy", rank_partition="val", task_type="binary", score_scope="folds")
        assert len(results) == 2
        assert all(r["task_type"] == "binary_classification" for r in results)

    def test_top_no_filter_returns_all(self, mixed_predictions):
        """Without task_type filter, all predictions are returned."""
        results = mixed_predictions.top(10, rank_metric="rmse", rank_partition="val", score_scope="folds")
        assert len(results) == 4

    def test_top_mixed_warns(self, mixed_predictions):
        """Calling top() without task_type or rank_metric on mixed data warns."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mixed_predictions.top(5, rank_partition="val", score_scope="folds")
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert any("Mixed task types" in str(uw.message) for uw in user_warnings)

    def test_top_with_task_type_no_warning(self, mixed_predictions):
        """Specifying task_type suppresses the mixed-type warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mixed_predictions.top(5, rank_partition="val", task_type="regression")
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert not any("Mixed task types" in str(uw.message) for uw in user_warnings)


# =========================================================================
# get_best() with task_type
# =========================================================================

class TestGetBestWithTaskType:
    """Verify get_best() respects task_type filtering."""

    def test_get_best_regression(self, mixed_predictions):
        best = mixed_predictions.get_best(metric="rmse", task_type="regression", score_scope="folds")
        assert best is not None
        assert best["task_type"] == "regression"

    def test_get_best_classification(self, mixed_predictions):
        best = mixed_predictions.get_best(metric="balanced_accuracy", task_type="classification", score_scope="folds")
        assert best is not None
        assert "classification" in best["task_type"]

    def test_get_best_nonexistent_type(self, mixed_predictions):
        """No multiclass predictions exist, so get_best returns None."""
        best = mixed_predictions.get_best(metric="balanced_accuracy", task_type="multiclass")
        assert best is None


# =========================================================================
# filter_predictions() with task_type
# =========================================================================

class TestFilterPredictionsTaskType:
    """Verify filter_predictions() respects task_type filtering."""

    def test_filter_regression(self, mixed_predictions):
        results = mixed_predictions.filter_predictions(task_type="regression")
        assert len(results) == 2
        assert all(r["task_type"] == "regression" for r in results)

    def test_filter_classification(self, mixed_predictions):
        results = mixed_predictions.filter_predictions(task_type="classification")
        assert len(results) == 2
        assert all("classification" in r["task_type"] for r in results)

    def test_filter_binary(self, mixed_predictions):
        results = mixed_predictions.filter_predictions(task_type="binary")
        assert len(results) == 2
        assert all(r["task_type"] == "binary_classification" for r in results)

    def test_filter_no_task_type_returns_all(self, mixed_predictions):
        results = mixed_predictions.filter_predictions()
        assert len(results) == 4
