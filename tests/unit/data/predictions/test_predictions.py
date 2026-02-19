"""Tests for the Predictions facade (in-memory mode)."""

import numpy as np
import pytest

from nirs4all.data.predictions import Predictions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_sample_prediction(
    preds: Predictions,
    *,
    dataset_name: str = "wheat",
    model_name: str = "PLS",
    fold_id: str | int = "0",
    partition: str = "val",
    val_score: float | None = 0.5,
    test_score: float | None = None,
    train_score: float | None = None,
    metric: str = "rmse",
    task_type: str = "regression",
    n_samples: int = 10,
    y_true: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    branch_id: int | None = None,
    branch_name: str | None = None,
    preprocessings: str = "",
    refit_context: str | None = None,
) -> str:
    """Add a single prediction and return its ID."""
    if y_true is None:
        y_true = np.arange(n_samples, dtype=np.float64)
    if y_pred is None:
        y_pred = y_true + np.random.default_rng(42).normal(0, 0.1, size=len(y_true))
    return preds.add_prediction(
        dataset_name=dataset_name,
        model_name=model_name,
        fold_id=fold_id,
        partition=partition,
        val_score=val_score,
        test_score=test_score,
        train_score=train_score,
        metric=metric,
        task_type=task_type,
        n_samples=n_samples,
        y_true=y_true,
        y_pred=y_pred,
        branch_id=branch_id,
        branch_name=branch_name,
        preprocessings=preprocessings,
        refit_context=refit_context,
    )

# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestEmptyPredictions:
    """Behaviour of a fresh, empty Predictions instance."""

    def test_num_predictions_zero(self):
        preds = Predictions()
        assert preds.num_predictions == 0

    def test_top_returns_empty_list(self):
        preds = Predictions()
        results = preds.top(5)
        assert len(results) == 0

    def test_get_best_returns_none(self):
        preds = Predictions()
        assert preds.get_best() is None

    def test_filter_predictions_returns_empty(self):
        preds = Predictions()
        assert preds.filter_predictions() == []

    def test_get_datasets_returns_empty(self):
        preds = Predictions()
        assert preds.get_datasets() == []

    def test_to_dicts_returns_empty(self):
        preds = Predictions()
        assert preds.to_dicts() == []

    def test_clear_on_empty_is_noop(self):
        preds = Predictions()
        preds.clear()
        assert preds.num_predictions == 0

class TestAddPrediction:
    """Tests for add_prediction and basic properties."""

    def test_add_single_prediction(self):
        preds = Predictions()
        pred_id = _add_sample_prediction(preds)
        assert isinstance(pred_id, str)
        assert len(pred_id) > 0
        assert preds.num_predictions == 1

    def test_add_multiple_predictions(self):
        preds = Predictions()
        _add_sample_prediction(preds, model_name="PLS", val_score=0.5)
        _add_sample_prediction(preds, model_name="RF", val_score=0.3)
        _add_sample_prediction(preds, model_name="SVM", val_score=0.7)
        assert preds.num_predictions == 3

    def test_prediction_id_unique(self):
        preds = Predictions()
        id1 = _add_sample_prediction(preds, model_name="PLS")
        id2 = _add_sample_prediction(preds, model_name="RF")
        assert id1 != id2

    def test_get_prediction_by_id(self):
        preds = Predictions()
        pred_id = _add_sample_prediction(preds, model_name="PLS")
        row = preds.get_prediction_by_id(pred_id)
        assert row is not None
        assert row["model_name"] == "PLS"

    def test_get_prediction_by_id_missing(self):
        preds = Predictions()
        _add_sample_prediction(preds)
        assert preds.get_prediction_by_id("nonexistent") is None

class TestFilterPredictions:
    """Tests for filter_predictions with various criteria."""

    def test_filter_by_dataset_name(self):
        preds = Predictions()
        _add_sample_prediction(preds, dataset_name="wheat")
        _add_sample_prediction(preds, dataset_name="corn")
        results = preds.filter_predictions(dataset_name="wheat")
        assert len(results) == 1
        assert results[0]["dataset_name"] == "wheat"

    def test_filter_by_model_name(self):
        preds = Predictions()
        _add_sample_prediction(preds, model_name="PLS")
        _add_sample_prediction(preds, model_name="RF")
        _add_sample_prediction(preds, model_name="PLS")
        results = preds.filter_predictions(model_name="PLS")
        assert len(results) == 2

    def test_filter_by_partition(self):
        preds = Predictions()
        _add_sample_prediction(preds, partition="val")
        _add_sample_prediction(preds, partition="test")
        _add_sample_prediction(preds, partition="val")
        results = preds.filter_predictions(partition="val")
        assert len(results) == 2

    def test_filter_by_fold_id(self):
        preds = Predictions()
        _add_sample_prediction(preds, fold_id="0")
        _add_sample_prediction(preds, fold_id="1")
        _add_sample_prediction(preds, fold_id="0")
        results = preds.filter_predictions(fold_id="0")
        assert len(results) == 2

    def test_filter_by_branch_id(self):
        preds = Predictions()
        _add_sample_prediction(preds, branch_id=0)
        _add_sample_prediction(preds, branch_id=1)
        results = preds.filter_predictions(branch_id=0)
        assert len(results) == 1

    def test_filter_no_match(self):
        preds = Predictions()
        _add_sample_prediction(preds, model_name="PLS")
        results = preds.filter_predictions(model_name="XGBoost")
        assert len(results) == 0

    def test_filter_without_arrays(self):
        preds = Predictions()
        _add_sample_prediction(preds)
        results = preds.filter_predictions(load_arrays=False)
        assert len(results) == 1
        assert "y_true" not in results[0]
        assert "y_pred" not in results[0]

    def test_filter_combined_criteria(self):
        preds = Predictions()
        _add_sample_prediction(preds, dataset_name="wheat", model_name="PLS")
        _add_sample_prediction(preds, dataset_name="wheat", model_name="RF")
        _add_sample_prediction(preds, dataset_name="corn", model_name="PLS")
        results = preds.filter_predictions(dataset_name="wheat", model_name="PLS")
        assert len(results) == 1

class TestTopAndGetBest:
    """Tests for top() ranking and get_best() convenience."""

    def test_top_returns_sorted_ascending(self):
        """For RMSE (lower is better), top(3) should sort ascending."""
        preds = Predictions()
        _add_sample_prediction(preds, model_name="A", val_score=0.5, metric="rmse")
        _add_sample_prediction(preds, model_name="B", val_score=0.2, metric="rmse")
        _add_sample_prediction(preds, model_name="C", val_score=0.8, metric="rmse")
        results = preds.top(3, rank_metric="rmse")
        scores = [r["val_score"] for r in results]
        assert scores == sorted(scores), "RMSE should sort ascending (lower is better)"

    def test_top_returns_sorted_descending_r2(self):
        """For R2 (higher is better), top(3) should sort descending."""
        preds = Predictions()
        _add_sample_prediction(preds, model_name="A", val_score=0.5, metric="r2")
        _add_sample_prediction(preds, model_name="B", val_score=0.9, metric="r2")
        _add_sample_prediction(preds, model_name="C", val_score=0.3, metric="r2")
        results = preds.top(3, rank_metric="r2")
        scores = [r["val_score"] for r in results]
        assert scores == sorted(scores, reverse=True), "R2 should sort descending (higher is better)"

    def test_top_n_limits_results(self):
        preds = Predictions()
        for i in range(10):
            _add_sample_prediction(preds, model_name=f"M{i}", val_score=float(i))
        results = preds.top(3, rank_metric="rmse")
        assert len(results) == 3

    def test_get_best_returns_best_model(self):
        preds = Predictions()
        _add_sample_prediction(preds, model_name="Bad", val_score=1.0, metric="rmse")
        _add_sample_prediction(preds, model_name="Best", val_score=0.1, metric="rmse")
        _add_sample_prediction(preds, model_name="Mid", val_score=0.5, metric="rmse")
        best = preds.get_best(metric="rmse")
        assert best is not None
        assert best["model_name"] == "Best"

    def test_get_best_with_filters(self):
        preds = Predictions()
        _add_sample_prediction(preds, dataset_name="wheat", model_name="PLS", val_score=0.1, metric="rmse")
        _add_sample_prediction(preds, dataset_name="corn", model_name="RF", val_score=0.05, metric="rmse")
        best = preds.get_best(metric="rmse", dataset_name="wheat")
        assert best is not None
        assert best["dataset_name"] == "wheat"

class TestMergePredictions:
    """Tests for merge_predictions combining two instances."""

    def test_merge_combines_buffers(self):
        pred1 = Predictions()
        pred2 = Predictions()
        _add_sample_prediction(pred1, model_name="PLS")
        _add_sample_prediction(pred2, model_name="RF")
        pred1.merge_predictions(pred2)
        assert pred1.num_predictions == 2

    def test_merge_inherits_repetition_column(self):
        pred1 = Predictions()
        pred2 = Predictions()
        pred2.set_repetition_column("Sample_ID")
        pred1.merge_predictions(pred2)
        assert pred1.repetition_column == "Sample_ID"

    def test_merge_does_not_overwrite_existing_repetition(self):
        pred1 = Predictions()
        pred2 = Predictions()
        pred1.set_repetition_column("Batch")
        pred2.set_repetition_column("Sample_ID")
        pred1.merge_predictions(pred2)
        assert pred1.repetition_column == "Batch"

    def test_merge_empty_into_populated(self):
        pred1 = Predictions()
        pred2 = Predictions()
        _add_sample_prediction(pred1, model_name="PLS")
        pred1.merge_predictions(pred2)
        assert pred1.num_predictions == 1

    def test_merge_populated_into_empty(self):
        pred1 = Predictions()
        pred2 = Predictions()
        _add_sample_prediction(pred2, model_name="PLS")
        pred1.merge_predictions(pred2)
        assert pred1.num_predictions == 1

class TestMetadataUtilities:
    """Tests for get_unique_values, get_datasets, etc."""

    def test_get_datasets(self):
        preds = Predictions()
        _add_sample_prediction(preds, dataset_name="wheat")
        _add_sample_prediction(preds, dataset_name="corn")
        _add_sample_prediction(preds, dataset_name="wheat")
        datasets = preds.get_datasets()
        assert sorted(datasets) == ["corn", "wheat"]

    def test_get_models(self):
        preds = Predictions()
        _add_sample_prediction(preds, model_name="PLS")
        _add_sample_prediction(preds, model_name="RF")
        models = preds.get_models()
        assert sorted(models) == ["PLS", "RF"]

    def test_get_partitions(self):
        preds = Predictions()
        _add_sample_prediction(preds, partition="val")
        _add_sample_prediction(preds, partition="test")
        parts = preds.get_partitions()
        assert sorted(parts) == ["test", "val"]

    def test_get_folds(self):
        preds = Predictions()
        _add_sample_prediction(preds, fold_id="0")
        _add_sample_prediction(preds, fold_id="1")
        _add_sample_prediction(preds, fold_id="0")
        folds = preds.get_folds()
        assert sorted(folds) == ["0", "1"]

class TestClearAndSlice:
    """Tests for clear() and slice_after()."""

    def test_clear(self):
        preds = Predictions()
        _add_sample_prediction(preds)
        _add_sample_prediction(preds)
        preds.clear()
        assert preds.num_predictions == 0

    def test_slice_after(self):
        preds = Predictions()
        _add_sample_prediction(preds, model_name="A")
        _add_sample_prediction(preds, model_name="B")
        n_before = preds.num_predictions
        _add_sample_prediction(preds, model_name="C")
        sliced = preds.slice_after(n_before)
        assert sliced.num_predictions == 1
        assert sliced.filter_predictions()[0]["model_name"] == "C"

    def test_slice_after_inherits_repetition(self):
        preds = Predictions()
        preds.set_repetition_column("ID")
        _add_sample_prediction(preds)
        sliced = preds.slice_after(0)
        assert sliced.repetition_column == "ID"

class TestConversion:
    """Tests for to_dataframe() and to_dicts()."""

    def test_to_dicts_with_arrays(self):
        preds = Predictions()
        _add_sample_prediction(preds)
        dicts = preds.to_dicts(load_arrays=True)
        assert len(dicts) == 1
        assert "y_true" in dicts[0]
        assert "y_pred" in dicts[0]

    def test_to_dicts_without_arrays(self):
        preds = Predictions()
        _add_sample_prediction(preds)
        dicts = preds.to_dicts(load_arrays=False)
        assert len(dicts) == 1
        assert "y_true" not in dicts[0]
        assert "y_pred" not in dicts[0]

    def test_to_dataframe(self):
        preds = Predictions()
        _add_sample_prediction(preds, model_name="PLS")
        _add_sample_prediction(preds, model_name="RF")
        df = preds.to_dataframe()
        assert len(df) == 2
        assert "model_name" in df.columns

class TestContextManager:
    """Tests for context manager protocol."""

    def test_enter_exit(self):
        """Predictions can be used as a context manager in in-memory mode."""
        with Predictions() as preds:
            _add_sample_prediction(preds)
            assert preds.num_predictions == 1
        # After exit, the instance should still be usable (no store to close)
        assert preds.num_predictions == 1

class TestRepetitionColumn:
    """Tests for repetition column management."""

    def test_default_repetition_column_is_none(self):
        preds = Predictions()
        assert preds.repetition_column is None

    def test_set_repetition_column(self):
        preds = Predictions()
        preds.set_repetition_column("Sample_ID")
        assert preds.repetition_column == "Sample_ID"

    def test_set_repetition_column_to_none(self):
        preds = Predictions()
        preds.set_repetition_column("ID")
        preds.set_repetition_column(None)
        assert preds.repetition_column is None
