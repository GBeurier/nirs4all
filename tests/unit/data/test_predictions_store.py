"""Tests for store-backed Predictions facade.

Verifies that the Predictions class correctly integrates with WorkspaceStore
for buffer/flush, ranking, filtering, and array round-trip operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import pytest

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

        # Buffer is preserved (downstream code reads from it after flush)
        assert len(preds._buffer) == 100

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

    def test_flush_with_chain_id_resolver(self, tmp_path):
        """flush() supports per-row chain resolution for runtime persistence."""
        preds, store, pipeline_id, chain_id_1 = _make_predictions_with_store(tmp_path)
        chain_id_2 = store.save_chain(
            pipeline_id=pipeline_id,
            steps=[{"step_idx": 1, "operator_class": "Ridge", "params": {}, "artifact_id": None, "stateless": False}],
            model_step_idx=1,
            model_class="sklearn.linear_model.Ridge",
            preprocessings="",
            fold_strategy="per_fold",
            fold_artifacts={},
            shared_artifacts={},
        )

        preds.add_prediction(
            dataset_name="wheat",
            model_name="PLS",
            model_classname="PLSRegression",
            fold_id=0,
            partition="val",
            y_true=np.array([1.0]),
            y_pred=np.array([1.1]),
            val_score=0.2,
            metric="rmse",
            task_type="regression",
            n_samples=1,
            n_features=10,
        )
        preds.add_prediction(
            dataset_name="wheat",
            model_name="Ridge",
            model_classname="Ridge",
            fold_id=0,
            partition="val",
            y_true=np.array([1.0]),
            y_pred=np.array([0.9]),
            val_score=0.1,
            metric="rmse",
            task_type="regression",
            n_samples=1,
            n_features=10,
        )

        preds.flush(
            pipeline_id=pipeline_id,
            chain_id_resolver=lambda row: chain_id_1 if row["model_name"] == "PLS" else chain_id_2,
        )

        df = store.query_predictions(pipeline_id=pipeline_id)
        assert len(df) == 2
        assert set(df["chain_id"].to_list()) == {chain_id_1, chain_id_2}

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

# ---------------------------------------------------------------------------
# Phase 3 — User-Facing API tests
# ---------------------------------------------------------------------------

def _flush_predictions_to_store(
    store: WorkspaceStore,
    pipeline_id: str,
    chain_id: str,
    n: int = 10,
    dataset_name: str = "wheat",
) -> Predictions:
    """Create, add, and flush *n* predictions to the store."""
    preds = Predictions(store=store)
    for i in range(n):
        preds.add_prediction(
            dataset_name=dataset_name,
            model_name=f"PLS_{i + 1}",
            model_classname="PLSRegression",
            fold_id=0,
            partition="val",
            y_true=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            y_pred=np.array([1.0, 2.0, 3.0, 4.0, 5.0]) + 0.1 * (i + 1),
            val_score=0.1 * (i + 1),
            test_score=0.12 * (i + 1),
            metric="rmse",
            task_type="regression",
            n_samples=5,
            n_features=100,
        )
    preds.flush(pipeline_id=pipeline_id, chain_id=chain_id)
    return preds

class TestPredictionsFromWorkspacePath:
    """Test Predictions(db_path=workspace_dir)."""

    def test_predictions_from_workspace_path(self, tmp_path):
        """Predictions('/path/to/workspace') loads predictions."""
        store = _make_store(tmp_path)
        pipeline_id, chain_id = _setup_store_hierarchy(store)
        _flush_predictions_to_store(store, pipeline_id, chain_id, n=5)
        store.close()

        workspace_dir = tmp_path / "workspace"
        preds = Predictions(workspace_dir)
        assert preds.num_predictions == 5
        assert len(preds.get_models()) == 5
        preds.close()

    def test_predictions_from_duckdb_file(self, tmp_path):
        """Predictions.from_file('store.duckdb') works."""
        store = _make_store(tmp_path)
        pipeline_id, chain_id = _setup_store_hierarchy(store)
        _flush_predictions_to_store(store, pipeline_id, chain_id, n=3)
        store.close()

        db_file = tmp_path / "workspace" / "store.duckdb"
        preds = Predictions.from_file(db_file)
        assert preds.num_predictions == 3
        preds.close()

class TestPredictionsFromParquet:
    """Test Predictions.from_parquet() portable mode."""

    def _write_portable_parquet(self, tmp_path, n: int = 5, dataset_name: str = "wheat") -> Path:
        """Write a portable Parquet file with *n* rows."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        records = []
        for i in range(n):
            records.append({
                "prediction_id": f"pred_{i:04d}",
                "dataset_name": dataset_name,
                "model_name": f"PLS_{i + 1}",
                "fold_id": "0",
                "partition": "val",
                "metric": "rmse",
                "val_score": 0.1 * (i + 1),
                "task_type": "regression",
                "y_true": [1.0, 2.0, 3.0],
                "y_pred": [1.1 + 0.1 * i, 2.1, 3.1],
                "y_proba": None,
                "y_proba_shape": None,
                "sample_indices": None,
                "weights": None,
            })

        schema = pa.schema([
            ("prediction_id", pa.utf8()),
            ("dataset_name", pa.utf8()),
            ("model_name", pa.utf8()),
            ("fold_id", pa.utf8()),
            ("partition", pa.utf8()),
            ("metric", pa.utf8()),
            ("val_score", pa.float64()),
            ("task_type", pa.utf8()),
            ("y_true", pa.list_(pa.float64())),
            ("y_pred", pa.list_(pa.float64())),
            ("y_proba", pa.list_(pa.float64())),
            ("y_proba_shape", pa.list_(pa.int32())),
            ("sample_indices", pa.list_(pa.int32())),
            ("weights", pa.list_(pa.float64())),
        ])

        columns = {field.name: [rec[field.name] for rec in records] for field in schema}
        table = pa.table(columns, schema=schema)
        path: Path = tmp_path / f"{dataset_name}.parquet"
        pq.write_table(table, path)
        return cast(Path, path)

    def test_from_parquet(self, tmp_path):
        """Predictions.from_parquet() loads arrays and metadata."""
        path = self._write_portable_parquet(tmp_path, n=5)
        preds = Predictions.from_parquet(path)
        assert preds.num_predictions == 5
        assert set(preds.get_models()) == {f"PLS_{i}" for i in range(1, 6)}

        # Check arrays are loaded
        entry = preds.filter_predictions(model_name="PLS_1")[0]
        assert entry["y_true"] is not None
        np.testing.assert_array_almost_equal(entry["y_true"], [1.0, 2.0, 3.0])

    def test_auto_detect_parquet(self, tmp_path):
        """Predictions('wheat.parquet') auto-detects portable mode."""
        path = self._write_portable_parquet(tmp_path, n=3)
        preds = Predictions(path)
        assert preds.num_predictions == 3
        assert preds._store is None

    def test_portable_mode_rejects_store_methods(self, tmp_path):
        """Store-requiring methods raise RuntimeError in portable mode."""
        path = self._write_portable_parquet(tmp_path, n=2)
        preds = Predictions.from_parquet(path)
        with pytest.raises(RuntimeError, match="requires a workspace store"):
            preds.query("SELECT 1")
        with pytest.raises(RuntimeError, match="requires a workspace store"):
            preds.store_stats()

class TestMergeStores:
    """Test Predictions.merge_stores()."""

    def test_merge_stores(self, tmp_path):
        """Merge 2 workspace stores into a target."""
        from nirs4all.data.predictions import MergeReport

        # Create source A with 3 predictions
        store_a = WorkspaceStore(tmp_path / "src_a")
        pid_a, cid_a = _setup_store_hierarchy(store_a)
        _flush_predictions_to_store(store_a, pid_a, cid_a, n=3, dataset_name="wheat")
        store_a.close()

        # Create source B with 2 predictions (different dataset)
        store_b = WorkspaceStore(tmp_path / "src_b")
        pid_b, cid_b = _setup_store_hierarchy(store_b, dataset_name="corn")
        _flush_predictions_to_store(store_b, pid_b, cid_b, n=2, dataset_name="corn")
        store_b.close()

        target_dir = tmp_path / "target"
        report = Predictions.merge_stores(
            sources=[tmp_path / "src_a", tmp_path / "src_b"],
            target=target_dir,
        )

        assert isinstance(report, MergeReport)
        assert report.total_sources == 2
        assert report.predictions_merged == 5
        assert set(report.datasets_merged) == {"wheat", "corn"}

        # Verify target has all predictions
        with Predictions(target_dir) as p:
            assert p.num_predictions == 5

    def test_merge_with_dataset_filter(self, tmp_path):
        """merge_stores with datasets filter only merges requested datasets."""
        store_a = WorkspaceStore(tmp_path / "src_a")
        pid_a, cid_a = _setup_store_hierarchy(store_a)
        _flush_predictions_to_store(store_a, pid_a, cid_a, n=3, dataset_name="wheat")
        store_a.close()

        store_b = WorkspaceStore(tmp_path / "src_b")
        pid_b, cid_b = _setup_store_hierarchy(store_b, dataset_name="corn")
        _flush_predictions_to_store(store_b, pid_b, cid_b, n=2, dataset_name="corn")
        store_b.close()

        target_dir = tmp_path / "target"
        report = Predictions.merge_stores(
            sources=[tmp_path / "src_a", tmp_path / "src_b"],
            target=target_dir,
            datasets=["wheat"],
        )

        assert report.predictions_merged == 3
        assert report.datasets_merged == ["wheat"]

class TestCleanDeadLinks:
    """Test clean_dead_links maintenance helper."""

    def test_clean_dead_links_dry_run(self, tmp_path):
        """clean_dead_links dry_run reports without deleting."""
        store = _make_store(tmp_path)
        pipeline_id, chain_id = _setup_store_hierarchy(store)
        _flush_predictions_to_store(store, pipeline_id, chain_id, n=5)

        preds = Predictions(store=store)
        result = preds.clean_dead_links(dry_run=True)
        # No orphans expected in a healthy store
        assert result["metadata_orphans_removed"] == 0
        assert result["array_orphans_found"] == 0
        store.close()

class TestRemoveBottom:
    """Test remove_bottom maintenance helper."""

    def test_remove_bottom_20_percent(self, tmp_path):
        """remove_bottom(0.2) removes the worst 20% of predictions."""
        store = _make_store(tmp_path)
        pipeline_id, chain_id = _setup_store_hierarchy(store)
        _flush_predictions_to_store(store, pipeline_id, chain_id, n=10)

        preds = Predictions(store=store)
        result = preds.remove_bottom(0.2, metric="val_score")

        assert result["removed"] == 2
        assert result["remaining"] == 8
        assert result["threshold_score"] is not None

        # Verify store has 8 remaining
        df = store.query_predictions()
        assert len(df) == 8
        store.close()

class TestRemoveDataset:
    """Test remove_dataset maintenance helper."""

    def test_remove_dataset(self, tmp_path):
        """remove_dataset removes all predictions for a dataset."""
        store = _make_store(tmp_path)
        pipeline_id, chain_id = _setup_store_hierarchy(store)
        _flush_predictions_to_store(store, pipeline_id, chain_id, n=5, dataset_name="wheat")

        # Add corn predictions
        pid2, cid2 = _setup_store_hierarchy(store, dataset_name="corn")
        _flush_predictions_to_store(store, pid2, cid2, n=3, dataset_name="corn")

        preds = Predictions(store=store)

        # Verify both datasets exist
        df_before = store.query_predictions()
        assert len(df_before) == 8

        result = preds.remove_dataset("wheat")
        assert result["predictions_removed"] == 5
        assert result["parquet_deleted"] is True

        # Verify only corn remains
        df_after = store.query_predictions()
        assert len(df_after) == 3
        assert all(row == "corn" for row in df_after["dataset_name"].to_list())
        store.close()

    def test_remove_dataset_dry_run(self, tmp_path):
        """remove_dataset dry_run reports without deleting."""
        store = _make_store(tmp_path)
        pipeline_id, chain_id = _setup_store_hierarchy(store)
        _flush_predictions_to_store(store, pipeline_id, chain_id, n=5)

        preds = Predictions(store=store)
        result = preds.remove_dataset("wheat", dry_run=True)
        assert result["predictions_removed"] == 5
        assert result["parquet_deleted"] is True

        # Verify nothing was deleted
        df = store.query_predictions()
        assert len(df) == 5
        store.close()

class TestRemoveRun:
    """Test remove_run maintenance helper."""

    def test_remove_run(self, tmp_path):
        """remove_run removes a run and all its descendants."""
        store = _make_store(tmp_path)
        run_id = store.begin_run(
            "test_run",
            config={"metric": "rmse"},
            datasets=[{"name": "wheat"}],
        )
        pipeline_id = store.begin_pipeline(
            run_id=run_id,
            name="0001_pls_test",
            expanded_config=[{"model": "PLSRegression"}],
            generator_choices=[],
            dataset_name="wheat",
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

        _flush_predictions_to_store(store, pipeline_id, chain_id, n=3)

        preds = Predictions(store=store)
        result = preds.remove_run(run_id)
        assert result["rows_removed"] > 0

        # Verify predictions are gone
        df = store.query_predictions()
        assert len(df) == 0
        store.close()

class TestCompact:
    """Test compact() maintenance helper."""

    def test_compact(self, tmp_path):
        """compact() applies tombstones and deduplicates."""
        store = _make_store(tmp_path)
        pipeline_id, chain_id = _setup_store_hierarchy(store)
        _flush_predictions_to_store(store, pipeline_id, chain_id, n=5)

        # Delete 2 predictions to create tombstones
        df = store.query_predictions()
        pids = df["prediction_id"].to_list()
        store.delete_prediction(pids[0])
        store.delete_prediction(pids[1])

        preds = Predictions(store=store)
        stats = preds.compact()

        assert "wheat" in stats
        assert stats["wheat"]["rows_removed"] >= 2
        store.close()

class TestStoreStats:
    """Test store_stats() helper."""

    def test_store_stats(self, tmp_path):
        """store_stats() returns combined DuckDB + Parquet stats."""
        store = _make_store(tmp_path)
        pipeline_id, chain_id = _setup_store_hierarchy(store)
        _flush_predictions_to_store(store, pipeline_id, chain_id, n=5)

        preds = Predictions(store=store)
        stats = preds.store_stats()

        assert "db_file_bytes" in stats
        assert stats["db_file_bytes"] > 0
        assert "tables" in stats
        assert stats["tables"]["predictions"] == 5
        assert stats["tables"]["runs"] == 1
        assert "arrays" in stats
        assert stats["arrays"]["total_rows"] == 5
        store.close()

class TestQuerySQL:
    """Test query() method."""

    def test_query_sql(self, tmp_path):
        """query() runs arbitrary SQL and returns Polars DataFrame."""
        store = _make_store(tmp_path)
        pipeline_id, chain_id = _setup_store_hierarchy(store)
        _flush_predictions_to_store(store, pipeline_id, chain_id, n=5)

        preds = Predictions(store=store)
        result = preds.query("SELECT COUNT(*) AS cnt FROM predictions")
        assert isinstance(result, pl.DataFrame)
        assert result["cnt"][0] == 5

        # More complex query
        result2 = preds.query("SELECT dataset_name, COUNT(*) AS n FROM predictions GROUP BY dataset_name")
        assert len(result2) == 1
        assert result2["n"][0] == 5
        store.close()

class TestContextManager:
    """Test context manager protocol."""

    def test_context_manager(self, tmp_path):
        """with Predictions(path) as p: works and closes store."""
        store = _make_store(tmp_path)
        pipeline_id, chain_id = _setup_store_hierarchy(store)
        _flush_predictions_to_store(store, pipeline_id, chain_id, n=3)
        store.close()

        workspace_dir = tmp_path / "workspace"
        with Predictions(workspace_dir) as p:
            assert p.num_predictions == 3
            assert p._store is not None
            assert p._owns_store is True

        # After exiting, store should be closed
        assert p._store is None
        assert p._owns_store is False

    def test_context_manager_no_store(self):
        """Context manager works in memory-only mode too."""
        with Predictions() as p:
            p.add_prediction(
                dataset_name="wheat",
                model_name="PLS",
                fold_id=0,
                partition="val",
                metric="rmse",
                task_type="regression",
            )
            assert p.num_predictions == 1
        # No crash — close is a no-op when no store
