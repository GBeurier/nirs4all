"""Tests for store-backed Predictions facade.

Verifies that the Predictions class correctly integrates with WorkspaceStore
for buffer/flush, ranking, filtering, and array round-trip operations.
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path: Path) -> WorkspaceStore:
    """Create a WorkspaceStore rooted at *tmp_path*."""
    return WorkspaceStore(tmp_path / "workspace")


def _setup_store_hierarchy(store: WorkspaceStore, dataset_name: str = "wheat") -> tuple[str, str]:
    """Create run -> pipeline -> chain hierarchy and return (pipeline_id, chain_id).

    The DuckDB schema enforces foreign key constraints, so predictions
    require valid pipeline_id and chain_id references.
    """
    run_id = store.begin_run(
        "test_run",
        config={"metric": "rmse"},
        datasets=[{"name": dataset_name}],
    )
    pipeline_id = store.begin_pipeline(
        run_id=run_id,
        name="0001_pls_test",
        expanded_config=[{"model": "PLSRegression"}],
        generator_choices=[],
        dataset_name=dataset_name,
        dataset_hash="abc123",
    )
    chain_id = store.save_chain(
        pipeline_id=pipeline_id,
        steps=[{"step_idx": 0, "operator_class": "PLSRegression", "params": {}, "artifact_id": None, "stateless": False}],
        model_step_idx=0,
        model_class="sklearn.cross_decomposition.PLSRegression",
        preprocessings="",
        fold_strategy="per_fold",
        fold_artifacts={},
        shared_artifacts={},
    )
    return pipeline_id, chain_id


def _make_predictions_with_store(tmp_path: Path) -> tuple[Predictions, WorkspaceStore, str, str]:
    """Create a Predictions instance backed by a WorkspaceStore.

    Returns:
        Tuple of (Predictions, WorkspaceStore, pipeline_id, chain_id).
    """
    store = _make_store(tmp_path)
    pipeline_id, chain_id = _setup_store_hierarchy(store)
    preds = Predictions(store=store)
    return preds, store, pipeline_id, chain_id


def _add_sample_predictions(preds: Predictions, n: int = 10) -> list[str]:
    """Add *n* sample predictions to the buffer and return their IDs."""
    ids = []
    for i in range(n):
        pred_id = preds.add_prediction(
            dataset_name="wheat",
            dataset_path="/data/wheat",
            config_name="pls_config",
            config_path="/configs/pls.yaml",
            pipeline_uid=f"pipe_{i:03d}",
            step_idx=0,
            op_counter=i,
            model_name=f"PLS_{i + 1}",
            model_classname="PLSRegression",
            fold_id=0,
            partition="val",
            y_true=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            y_pred=np.array([1.0, 2.0, 3.0, 4.0, 5.0]) + np.random.randn(5) * 0.1 * (i + 1),
            val_score=0.1 * (i + 1),
            test_score=0.12 * (i + 1),
            train_score=0.08 * (i + 1),
            metric="rmse",
            task_type="regression",
            n_samples=5,
            n_features=100,
        )
        ids.append(pred_id)
    return ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPredictionsBufferFlush:
    """Test buffer accumulation and flush to store."""

    def test_buffer_flush_100_predictions(self, tmp_path):
        """Add 100 predictions, flush, and verify all are persisted in store."""
        preds, store, pipeline_id, chain_id = _make_predictions_with_store(tmp_path)

        # Add 100 predictions
        for i in range(100):
            preds.add_prediction(
                dataset_name="wheat",
                dataset_path="/data/wheat",
                config_name="config",
                config_path="/configs/c.yaml",
                model_name=f"model_{i}",
                model_classname="PLSRegression",
                fold_id=0,
                partition="val",
                y_true=np.array([1.0, 2.0]),
                y_pred=np.array([1.1, 2.1]),
                val_score=0.1 + i * 0.001,
                metric="rmse",
                task_type="regression",
                n_samples=2,
                n_features=50,
            )

        assert len(preds._buffer) == 100

        # Flush with valid pipeline_id
        preds.flush(pipeline_id=pipeline_id, chain_id=chain_id)

        # Buffer should be cleared
        assert len(preds._buffer) == 0

        # Store should have all 100
        df = store.query_predictions()
        assert df.height == 100

    def test_flush_without_store_is_noop(self):
        """Flush on a store-less Predictions does nothing."""
        preds = Predictions()
        preds.add_prediction(
            dataset_name="wheat",
            model_name="PLS",
            model_classname="PLSRegression",
            fold_id=0,
            partition="val",
            y_true=np.array([1.0]),
            y_pred=np.array([1.1]),
            val_score=0.1,
            metric="rmse",
            task_type="regression",
            n_samples=1,
            n_features=50,
        )

        # Should not raise
        preds.flush(pipeline_id="test")

        # Buffer still intact (no store to flush to)
        assert len(preds._buffer) == 1


class TestPredictionsTop:
    """Test top() ranking from in-memory buffer."""

    def test_top_5_returns_correct_ranking(self):
        """top(5) returns the 5 best predictions sorted by score."""
        preds = Predictions()
        _add_sample_predictions(preds, n=10)

        # Lower RMSE is better (ascending)
        top_5 = preds.top(5, rank_metric="", rank_partition="val", ascending=True)

        assert len(top_5) == 5

        # Verify ascending order
        scores = [r["rank_score"] for r in top_5]
        assert scores == sorted(scores)

        # The best (lowest) score should be first
        assert top_5[0]["val_score"] == pytest.approx(0.1, abs=0.01)

    def test_top_returns_rank_score_field(self):
        """Each result from top() should have a rank_score field."""
        preds = Predictions()
        _add_sample_predictions(preds, n=3)

        results = preds.top(3, rank_metric="", rank_partition="val", ascending=True)

        for r in results:
            assert "rank_score" in r
            assert r["rank_score"] is not None

    def test_top_empty_returns_empty(self):
        """top() on empty predictions returns empty list."""
        preds = Predictions()
        results = preds.top(5, rank_metric="rmse", rank_partition="val")
        assert len(results) == 0


class TestPredictionsFilter:
    """Test filter_predictions on in-memory buffer."""

    def test_filter_by_dataset(self):
        """filter(dataset_name='wheat') returns only wheat predictions."""
        preds = Predictions()

        # Add wheat predictions
        for i in range(3):
            preds.add_prediction(
                dataset_name="wheat",
                model_name=f"model_{i}",
                model_classname="PLSRegression",
                fold_id=0,
                partition="val",
                y_true=np.array([1.0]),
                y_pred=np.array([1.1]),
                val_score=0.1,
                metric="rmse",
                task_type="regression",
                n_samples=1,
                n_features=50,
            )

        # Add corn predictions
        for i in range(2):
            preds.add_prediction(
                dataset_name="corn",
                model_name=f"model_{i}",
                model_classname="PLSRegression",
                fold_id=0,
                partition="val",
                y_true=np.array([1.0]),
                y_pred=np.array([1.1]),
                val_score=0.1,
                metric="rmse",
                task_type="regression",
                n_samples=1,
                n_features=50,
            )

        wheat_preds = preds.filter_predictions(dataset_name="wheat")
        assert len(wheat_preds) == 3
        assert all(p["dataset_name"] == "wheat" for p in wheat_preds)

        corn_preds = preds.filter_predictions(dataset_name="corn")
        assert len(corn_preds) == 2


class TestPredictionsArraysRoundtrip:
    """Test array storage and retrieval through flush."""

    def test_arrays_roundtrip(self, tmp_path):
        """Save y_true/y_pred arrays, flush, verify via store."""
        preds, store, pipeline_id, chain_id = _make_predictions_with_store(tmp_path)

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.3])

        preds.add_prediction(
            dataset_name="wheat",
            model_name="PLS_10",
            model_classname="PLSRegression",
            fold_id=0,
            partition="val",
            y_true=y_true,
            y_pred=y_pred,
            val_score=0.15,
            metric="rmse",
            task_type="regression",
            n_samples=5,
            n_features=100,
        )

        preds.flush(pipeline_id=pipeline_id, chain_id=chain_id)

        # Query from store to get the prediction ID
        df = store.query_predictions()
        assert df.height == 1
        pred_id = df["prediction_id"][0]

        # Load with arrays
        loaded = store.get_prediction(pred_id, load_arrays=True)
        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded["y_true"], y_true)
        np.testing.assert_array_almost_equal(loaded["y_pred"], y_pred)

    def test_weights_roundtrip(self, tmp_path):
        """Save and load sample weights through flush."""
        preds, store, pipeline_id, chain_id = _make_predictions_with_store(tmp_path)

        weights = np.array([0.5, 1.0, 1.5, 2.0, 0.8])

        preds.add_prediction(
            dataset_name="wheat",
            model_name="PLS_10",
            model_classname="PLSRegression",
            fold_id=0,
            partition="val",
            y_true=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            y_pred=np.array([1.1, 2.2, 2.8, 4.1, 5.3]),
            weights=weights,
            val_score=0.15,
            metric="rmse",
            task_type="regression",
            n_samples=5,
            n_features=100,
        )

        preds.flush(pipeline_id=pipeline_id, chain_id=chain_id)

        df = store.query_predictions()
        pred_id = df["prediction_id"][0]
        loaded = store.get_prediction(pred_id, load_arrays=True)
        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded["weights"], weights)


class TestResultBestScore:
    """Test that result scores are correctly computed from store-backed predictions."""

    def test_best_rmse_from_buffer(self):
        """get_best() should return the correct best prediction from buffer."""
        preds = Predictions()

        # Add predictions with known RMSE scores
        preds.add_prediction(
            dataset_name="wheat",
            model_name="PLS_3",
            model_classname="PLSRegression",
            fold_id=0,
            partition="val",
            y_true=np.array([1.0, 2.0, 3.0]),
            y_pred=np.array([1.0, 2.0, 3.0]),  # Perfect prediction
            val_score=0.05,
            test_score=0.06,
            metric="rmse",
            task_type="regression",
            n_samples=3,
            n_features=50,
        )

        preds.add_prediction(
            dataset_name="wheat",
            model_name="PLS_5",
            model_classname="PLSRegression",
            fold_id=0,
            partition="val",
            y_true=np.array([1.0, 2.0, 3.0]),
            y_pred=np.array([1.5, 2.5, 3.5]),  # Worse prediction
            val_score=0.5,
            test_score=0.6,
            metric="rmse",
            task_type="regression",
            n_samples=3,
            n_features=50,
        )

        best = preds.get_best(metric="", ascending=True)
        assert best is not None
        assert best["model_name"] == "PLS_3"
        assert best["val_score"] == 0.05

    def test_top_with_precomputed_scores(self):
        """top() should use pre-computed scores from scores dict."""
        preds = Predictions()

        scores = {
            "val": {"rmse": 0.99, "r2": 0.01},
            "test": {"rmse": 1.01, "r2": -0.01},
        }

        preds.add_prediction(
            dataset_name="wheat",
            model_name="PLS_5",
            model_classname="PLSRegression",
            fold_id=0,
            partition="val",
            y_true=np.array([1.0, 2.0, 3.0]),
            y_pred=np.array([1.1, 2.1, 2.9]),
            val_score=0.99,
            scores=scores,
            metric="rmse",
            task_type="regression",
            n_samples=3,
            n_features=50,
        )

        results = preds.top(1, rank_partition="val", rank_metric="rmse")
        assert len(results) == 1
        assert results[0]["rank_score"] == 0.99

        results_r2 = preds.top(1, rank_partition="val", rank_metric="r2")
        assert len(results_r2) == 1
        assert results_r2[0]["rank_score"] == 0.01


class TestFlushAndQueryStore:
    """End-to-end: buffer -> flush -> query store."""

    def test_flush_and_query_top_predictions(self, tmp_path):
        """Flush predictions and query top from store."""
        preds, store, pipeline_id, chain_id = _make_predictions_with_store(tmp_path)

        for i in range(5):
            preds.add_prediction(
                dataset_name="wheat",
                model_name=f"PLS_{i + 1}",
                model_classname="PLSRegression",
                fold_id=0,
                partition="val",
                y_true=np.array([1.0, 2.0, 3.0]),
                y_pred=np.array([1.0, 2.0, 3.0]) + 0.1 * (i + 1),
                val_score=0.1 * (i + 1),
                test_score=0.12 * (i + 1),
                metric="rmse",
                task_type="regression",
                n_samples=3,
                n_features=100,
            )

        preds.flush(pipeline_id=pipeline_id, chain_id=chain_id)

        # Query top 3 from store
        top_df = store.top_predictions(n=3, metric="val_score", ascending=True, partition="val")
        assert top_df.height == 3

        # Best should have lowest val_score
        best_score = top_df["val_score"][0]
        assert best_score == pytest.approx(0.1, abs=0.01)

    def test_flush_preserves_scores_dict(self, tmp_path):
        """Flush preserves the scores JSON in the store."""
        preds, store, pipeline_id, chain_id = _make_predictions_with_store(tmp_path)

        scores = {
            "val": {"rmse": 0.15, "r2": 0.85, "mae": 0.12},
            "test": {"rmse": 0.18, "r2": 0.82, "mae": 0.14},
        }

        preds.add_prediction(
            dataset_name="wheat",
            model_name="PLS_10",
            model_classname="PLSRegression",
            fold_id=0,
            partition="val",
            y_true=np.array([1.0, 2.0]),
            y_pred=np.array([1.1, 2.1]),
            val_score=0.15,
            scores=scores,
            metric="rmse",
            task_type="regression",
            n_samples=2,
            n_features=100,
        )

        preds.flush(pipeline_id=pipeline_id, chain_id=chain_id)

        df = store.query_predictions()
        pred_id = df["prediction_id"][0]
        loaded = store.get_prediction(pred_id)
        assert loaded is not None

        loaded_scores = loaded["scores"]
        assert loaded_scores["val"]["rmse"] == 0.15
        assert loaded_scores["val"]["r2"] == 0.85
        assert loaded_scores["test"]["mae"] == 0.14
