"""Tests for the aggregated predictions VIEW and WorkspaceStore methods.

Covers:
- VIEW creation and idempotency
- Correct grouping and aggregation
- Metric-aware ranking (ascending for error metrics, descending for score metrics)
- Drill-down from aggregated to partition/fold predictions
- get_prediction_arrays retrieval
- Deletion cascade: VIEW reflects removal immediately
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nirs4all.pipeline.storage.workspace_store import WorkspaceStore, _infer_metric_ascending


# =========================================================================
# Helpers
# =========================================================================


def _make_store(tmp_path: Path) -> WorkspaceStore:
    """Create a WorkspaceStore rooted at *tmp_path*."""
    return WorkspaceStore(tmp_path / "workspace")


def _populate_store(store: WorkspaceStore, *, n_folds: int = 3) -> dict:
    """Populate a store with a run containing multiple folds and partitions.

    Creates:
    - 1 run, 1 pipeline, 1 chain
    - n_folds predictions for partition "val"
    - n_folds predictions for partition "test"
    - Arrays for each prediction

    Returns dict with all IDs.
    """
    run_id = store.begin_run("test_run", config={"metric": "rmse"}, datasets=[{"name": "wheat"}])
    pipeline_id = store.begin_pipeline(
        run_id=run_id,
        name="0001_pls",
        expanded_config=[{"step": "SNV"}, {"model": "PLSRegression"}],
        generator_choices=[],
        dataset_name="wheat",
        dataset_hash="abc123",
    )
    chain_id = store.save_chain(
        pipeline_id=pipeline_id,
        steps=[
            {"step_idx": 0, "operator_class": "SNV", "params": {}, "artifact_id": None, "stateless": True},
            {"step_idx": 1, "operator_class": "PLSRegression", "params": {"n_components": 10}, "artifact_id": None, "stateless": False},
        ],
        model_step_idx=1,
        model_class="sklearn.cross_decomposition.PLSRegression",
        preprocessings="SNV",
        fold_strategy="per_fold",
        fold_artifacts={},
        shared_artifacts={},
        branch_path=None,
        source_index=None,
    )

    prediction_ids = []
    for partition in ("val", "test"):
        for fold_idx in range(n_folds):
            # Deterministic scores for testing
            val_score = 0.10 + fold_idx * 0.02
            test_score = 0.12 + fold_idx * 0.03
            train_score = 0.05 + fold_idx * 0.01

            pred_id = store.save_prediction(
                pipeline_id=pipeline_id,
                chain_id=chain_id,
                dataset_name="wheat",
                model_name="PLSRegression",
                model_class="sklearn.cross_decomposition.PLSRegression",
                fold_id=f"fold_{fold_idx}",
                partition=partition,
                val_score=val_score,
                test_score=test_score,
                train_score=train_score,
                metric="rmse",
                task_type="regression",
                n_samples=100,
                n_features=200,
                scores={"val": {"rmse": val_score}, "test": {"rmse": test_score}},
                best_params={"n_components": 10},
                branch_id=None,
                branch_name=None,
                exclusion_count=0,
                exclusion_rate=0.0,
                preprocessings="SNV",
            )
            prediction_ids.append(pred_id)

            # Save arrays
            rng = np.random.default_rng(fold_idx)
            y_true = rng.standard_normal(100)
            y_pred = y_true + rng.standard_normal(100) * 0.1
            store.save_prediction_arrays(pred_id, y_true, y_pred)

    store.complete_pipeline(pipeline_id, best_val=0.10, best_test=0.12, metric="rmse", duration_ms=1000)
    store.complete_run(run_id, summary={"total_pipelines": 1})

    return {
        "run_id": run_id,
        "pipeline_id": pipeline_id,
        "chain_id": chain_id,
        "prediction_ids": prediction_ids,
    }


def _populate_multi_model_store(store: WorkspaceStore) -> dict:
    """Populate a store with multiple chains (models) for ranking tests.

    Creates 1 run, 1 pipeline, 3 chains with different model classes
    and varying scores.
    """
    run_id = store.begin_run("multi_run", config={"metric": "rmse"}, datasets=[{"name": "corn"}])
    pipeline_id = store.begin_pipeline(
        run_id=run_id,
        name="0001_multi",
        expanded_config=[],
        generator_choices=[],
        dataset_name="corn",
        dataset_hash="def456",
    )

    chains = []
    models = [
        ("PLSRegression", "sklearn.cross_decomposition.PLSRegression", 0.08),
        ("RandomForest", "sklearn.ensemble.RandomForestRegressor", 0.15),
        ("SVR", "sklearn.svm.SVR", 0.12),
    ]

    for model_name, model_class, base_score in models:
        chain_id = store.save_chain(
            pipeline_id=pipeline_id,
            steps=[{"step_idx": 0, "operator_class": model_name, "params": {}, "artifact_id": None, "stateless": False}],
            model_step_idx=0,
            model_class=model_class,
            preprocessings="",
            fold_strategy="per_fold",
            fold_artifacts={},
            shared_artifacts={},
        )

        # Create predictions for val partition
        for fold_idx in range(2):
            store.save_prediction(
                pipeline_id=pipeline_id,
                chain_id=chain_id,
                dataset_name="corn",
                model_name=model_name,
                model_class=model_class,
                fold_id=f"fold_{fold_idx}",
                partition="val",
                val_score=base_score + fold_idx * 0.01,
                test_score=base_score + 0.02 + fold_idx * 0.01,
                train_score=base_score - 0.02,
                metric="rmse",
                task_type="regression",
                n_samples=50,
                n_features=100,
                scores={},
                best_params={},
                branch_id=None,
                branch_name=None,
                exclusion_count=0,
                exclusion_rate=0.0,
            )

        chains.append({"chain_id": chain_id, "model_name": model_name, "base_score": base_score})

    store.complete_pipeline(pipeline_id, best_val=0.08, best_test=0.10, metric="rmse", duration_ms=2000)
    store.complete_run(run_id, summary={"total_pipelines": 1})

    return {
        "run_id": run_id,
        "pipeline_id": pipeline_id,
        "chains": chains,
    }


# =========================================================================
# VIEW creation and schema
# =========================================================================


class TestViewCreation:
    """Verify the v_aggregated_predictions VIEW is created correctly."""

    def test_view_exists_after_schema_creation(self, tmp_path):
        """VIEW is created as part of schema initialization."""
        store = _make_store(tmp_path)
        conn = store._ensure_open()
        result = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_type = 'VIEW' AND table_name = 'v_aggregated_predictions'"
        ).fetchone()
        assert result is not None
        store.close()

    def test_view_creation_idempotent(self, tmp_path):
        """Creating the store twice does not error (IF NOT EXISTS)."""
        store1 = _make_store(tmp_path)
        store1.close()
        store2 = WorkspaceStore(tmp_path / "workspace")
        conn = store2._ensure_open()
        result = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_type = 'VIEW' AND table_name = 'v_aggregated_predictions'"
        ).fetchone()
        assert result is not None
        store2.close()

    def test_view_empty_when_no_data(self, tmp_path):
        """VIEW returns empty result when no predictions exist."""
        store = _make_store(tmp_path)
        df = store.query_aggregated_predictions()
        assert len(df) == 0
        store.close()


# =========================================================================
# Aggregation correctness
# =========================================================================


class TestAggregation:
    """Verify VIEW returns correct grouping and aggregates."""

    def test_one_row_per_chain_metric_dataset(self, tmp_path):
        """VIEW groups to one row per (chain_id, metric, dataset_name)."""
        store = _make_store(tmp_path)
        ids = _populate_store(store, n_folds=3)
        df = store.query_aggregated_predictions()
        assert len(df) == 1  # One chain, one metric, one dataset
        store.close()

    def test_fold_count(self, tmp_path):
        """fold_count reflects number of distinct folds."""
        store = _make_store(tmp_path)
        ids = _populate_store(store, n_folds=4)
        df = store.query_aggregated_predictions()
        row = df.row(0, named=True)
        assert row["fold_count"] == 4
        store.close()

    def test_partition_count(self, tmp_path):
        """partition_count reflects number of distinct partitions."""
        store = _make_store(tmp_path)
        ids = _populate_store(store, n_folds=3)
        df = store.query_aggregated_predictions()
        row = df.row(0, named=True)
        assert row["partition_count"] == 2  # val and test
        store.close()

    def test_partitions_list(self, tmp_path):
        """partitions list contains all distinct partition names."""
        store = _make_store(tmp_path)
        ids = _populate_store(store, n_folds=3)
        df = store.query_aggregated_predictions()
        row = df.row(0, named=True)
        partitions = row["partitions"]
        assert sorted(partitions) == ["test", "val"]
        store.close()

    def test_score_aggregates(self, tmp_path):
        """min/max/avg scores are mathematically correct."""
        store = _make_store(tmp_path)
        ids = _populate_store(store, n_folds=3)
        df = store.query_aggregated_predictions()
        row = df.row(0, named=True)

        # val_score values: 0.10, 0.12, 0.14 (for val) and 0.10, 0.12, 0.14 (for test)
        # All 6 predictions produce val_score
        val_scores = [0.10, 0.12, 0.14, 0.10, 0.12, 0.14]
        assert row["min_val_score"] == pytest.approx(min(val_scores))
        assert row["max_val_score"] == pytest.approx(max(val_scores))
        assert row["avg_val_score"] == pytest.approx(sum(val_scores) / len(val_scores))
        store.close()

    def test_prediction_ids_list(self, tmp_path):
        """prediction_ids list contains all prediction IDs for the chain."""
        store = _make_store(tmp_path)
        ids = _populate_store(store, n_folds=3)
        df = store.query_aggregated_predictions()
        row = df.row(0, named=True)
        pred_ids = row["prediction_ids"]
        # 3 folds * 2 partitions = 6 predictions
        assert len(pred_ids) == 6
        # All IDs should match
        assert set(pred_ids) == set(ids["prediction_ids"])
        store.close()

    def test_chain_metadata(self, tmp_path):
        """VIEW includes chain metadata columns."""
        store = _make_store(tmp_path)
        ids = _populate_store(store, n_folds=2)
        df = store.query_aggregated_predictions()
        row = df.row(0, named=True)
        assert row["model_class"] == "sklearn.cross_decomposition.PLSRegression"
        assert row["preprocessings"] == "SNV"
        assert row["model_name"] == "PLSRegression"
        assert row["metric"] == "rmse"
        assert row["dataset_name"] == "wheat"
        assert row["run_id"] == ids["run_id"]
        assert row["pipeline_id"] == ids["pipeline_id"]
        assert row["chain_id"] == ids["chain_id"]
        store.close()

    def test_multiple_chains_produce_multiple_rows(self, tmp_path):
        """Multiple chains produce separate aggregated rows."""
        store = _make_store(tmp_path)
        ids = _populate_multi_model_store(store)
        df = store.query_aggregated_predictions()
        # 3 models = 3 chains = 3 rows (all same metric & dataset)
        assert len(df) == 3
        store.close()


# =========================================================================
# Filtering
# =========================================================================


class TestFiltering:
    """Verify query filters work correctly."""

    def test_filter_by_run_id(self, tmp_path):
        """Filtering by run_id returns only that run's predictions."""
        store = _make_store(tmp_path)
        ids = _populate_store(store, n_folds=2)
        df = store.query_aggregated_predictions(run_id=ids["run_id"])
        assert len(df) == 1
        store.close()

    def test_filter_by_nonexistent_run_id(self, tmp_path):
        """Filtering by unknown run_id returns empty."""
        store = _make_store(tmp_path)
        _populate_store(store, n_folds=2)
        df = store.query_aggregated_predictions(run_id="nonexistent")
        assert len(df) == 0
        store.close()

    def test_filter_by_chain_id(self, tmp_path):
        """Filtering by chain_id returns only that chain."""
        store = _make_store(tmp_path)
        ids = _populate_multi_model_store(store)
        target_chain = ids["chains"][0]["chain_id"]
        df = store.query_aggregated_predictions(chain_id=target_chain)
        assert len(df) == 1
        assert df.row(0, named=True)["chain_id"] == target_chain
        store.close()

    def test_filter_by_model_class(self, tmp_path):
        """Filtering by model_class works."""
        store = _make_store(tmp_path)
        ids = _populate_multi_model_store(store)
        df = store.query_aggregated_predictions(model_class="sklearn.svm.SVR")
        assert len(df) == 1
        assert df.row(0, named=True)["model_name"] == "SVR"
        store.close()

    def test_filter_by_dataset_name(self, tmp_path):
        """Filtering by dataset_name works."""
        store = _make_store(tmp_path)
        _populate_store(store, n_folds=2)
        df = store.query_aggregated_predictions(dataset_name="wheat")
        assert len(df) == 1
        df2 = store.query_aggregated_predictions(dataset_name="corn")
        assert len(df2) == 0
        store.close()

    def test_filter_by_metric(self, tmp_path):
        """Filtering by metric works."""
        store = _make_store(tmp_path)
        _populate_store(store, n_folds=2)
        df = store.query_aggregated_predictions(metric="rmse")
        assert len(df) == 1
        df2 = store.query_aggregated_predictions(metric="r2")
        assert len(df2) == 0
        store.close()


# =========================================================================
# Metric-aware ranking
# =========================================================================


class TestMetricAwareRanking:
    """Verify metric-direction-aware ranking."""

    def test_infer_ascending_rmse(self):
        """RMSE is lower-is-better (ascending=True)."""
        assert _infer_metric_ascending("rmse") is True
        assert _infer_metric_ascending("RMSE") is True

    def test_infer_ascending_r2(self):
        """R² is higher-is-better (ascending=False)."""
        assert _infer_metric_ascending("r2") is False
        assert _infer_metric_ascending("R2") is False

    def test_infer_ascending_accuracy(self):
        """Accuracy is higher-is-better."""
        assert _infer_metric_ascending("accuracy") is False

    def test_infer_ascending_mae(self):
        """MAE is lower-is-better."""
        assert _infer_metric_ascending("mae") is True

    def test_top_aggregated_rmse_ascending(self, tmp_path):
        """Top aggregated for RMSE sorts ascending (lower is better)."""
        store = _make_store(tmp_path)
        ids = _populate_multi_model_store(store)
        df = store.query_top_aggregated_predictions(metric="rmse", n=3)
        assert len(df) == 3
        # PLS (0.08 avg) should be first, then SVR (0.12), then RF (0.15)
        names = df["model_name"].to_list()
        assert names[0] == "PLSRegression"
        assert names[2] == "RandomForest"
        store.close()

    def test_top_aggregated_n_limit(self, tmp_path):
        """n parameter limits the number of results."""
        store = _make_store(tmp_path)
        _populate_multi_model_store(store)
        df = store.query_top_aggregated_predictions(metric="rmse", n=1)
        assert len(df) == 1
        store.close()

    def test_top_aggregated_with_filter(self, tmp_path):
        """Ranking with additional filter."""
        store = _make_store(tmp_path)
        ids = _populate_multi_model_store(store)
        df = store.query_top_aggregated_predictions(
            metric="rmse", n=10,
            dataset_name="corn",
        )
        assert len(df) == 3
        store.close()


# =========================================================================
# Drill-down: chain → partition → fold → arrays
# =========================================================================


class TestDrillDown:
    """Verify drill-down from aggregated to individual predictions."""

    def test_get_chain_predictions_all(self, tmp_path):
        """get_chain_predictions returns all predictions for a chain."""
        store = _make_store(tmp_path)
        ids = _populate_store(store, n_folds=3)
        df = store.get_chain_predictions(ids["chain_id"])
        assert len(df) == 6  # 3 folds × 2 partitions
        store.close()

    def test_get_chain_predictions_by_partition(self, tmp_path):
        """Filtering by partition returns only that partition."""
        store = _make_store(tmp_path)
        ids = _populate_store(store, n_folds=3)
        df = store.get_chain_predictions(ids["chain_id"], partition="val")
        assert len(df) == 3
        assert all(row["partition"] == "val" for row in df.iter_rows(named=True))
        store.close()

    def test_get_chain_predictions_by_fold(self, tmp_path):
        """Filtering by fold_id returns only that fold."""
        store = _make_store(tmp_path)
        ids = _populate_store(store, n_folds=3)
        df = store.get_chain_predictions(ids["chain_id"], fold_id="fold_0")
        assert len(df) == 2  # fold_0 for val and test
        assert all(row["fold_id"] == "fold_0" for row in df.iter_rows(named=True))
        store.close()

    def test_get_chain_predictions_by_partition_and_fold(self, tmp_path):
        """Filtering by both partition and fold returns one prediction."""
        store = _make_store(tmp_path)
        ids = _populate_store(store, n_folds=3)
        df = store.get_chain_predictions(ids["chain_id"], partition="val", fold_id="fold_1")
        assert len(df) == 1
        row = df.row(0, named=True)
        assert row["partition"] == "val"
        assert row["fold_id"] == "fold_1"
        store.close()

    def test_get_prediction_arrays(self, tmp_path):
        """get_prediction_arrays returns numpy arrays."""
        store = _make_store(tmp_path)
        ids = _populate_store(store, n_folds=2)
        pred_id = ids["prediction_ids"][0]
        arrays = store.get_prediction_arrays(pred_id)
        assert arrays is not None
        assert isinstance(arrays["y_true"], np.ndarray)
        assert isinstance(arrays["y_pred"], np.ndarray)
        assert len(arrays["y_true"]) == 100
        assert len(arrays["y_pred"]) == 100
        store.close()

    def test_get_prediction_arrays_nonexistent(self, tmp_path):
        """get_prediction_arrays returns None for nonexistent prediction."""
        store = _make_store(tmp_path)
        result = store.get_prediction_arrays("nonexistent_id")
        assert result is None
        store.close()


# =========================================================================
# Deletion cascade
# =========================================================================


class TestDeletionCascade:
    """Verify VIEW reflects deletions immediately."""

    def test_delete_run_removes_from_view(self, tmp_path):
        """Deleting a run empties the VIEW for that run."""
        store = _make_store(tmp_path)
        ids = _populate_store(store, n_folds=2)
        # Verify data exists
        df = store.query_aggregated_predictions()
        assert len(df) == 1
        # Delete the run
        store.delete_run(ids["run_id"])
        # VIEW should be empty
        df = store.query_aggregated_predictions()
        assert len(df) == 0
        store.close()

    def test_delete_one_run_preserves_others(self, tmp_path):
        """Deleting one run does not affect another run in the VIEW."""
        store = _make_store(tmp_path)
        ids1 = _populate_store(store, n_folds=2)

        # Create a second run
        run_id2 = store.begin_run("run2", config={}, datasets=[])
        pipeline_id2 = store.begin_pipeline(
            run_id=run_id2, name="pipe2",
            expanded_config=[], generator_choices=[],
            dataset_name="corn", dataset_hash="xyz",
        )
        chain_id2 = store.save_chain(
            pipeline_id=pipeline_id2,
            steps=[{"step_idx": 0, "operator_class": "Model", "params": {}, "artifact_id": None, "stateless": False}],
            model_step_idx=0, model_class="SomeModel",
            preprocessings="", fold_strategy="per_fold",
            fold_artifacts={}, shared_artifacts={},
        )
        store.save_prediction(
            pipeline_id=pipeline_id2, chain_id=chain_id2,
            dataset_name="corn", model_name="SomeModel",
            model_class="SomeModel", fold_id="fold_0",
            partition="val", val_score=0.5, test_score=0.6,
            train_score=0.3, metric="rmse", task_type="regression",
            n_samples=50, n_features=100, scores={}, best_params={},
            branch_id=None, branch_name=None,
            exclusion_count=0, exclusion_rate=0.0,
        )

        # Verify 2 aggregated rows
        df = store.query_aggregated_predictions()
        assert len(df) == 2

        # Delete first run
        store.delete_run(ids1["run_id"])

        # Only second run's data remains
        df = store.query_aggregated_predictions()
        assert len(df) == 1
        assert df.row(0, named=True)["run_id"] == run_id2
        store.close()

    def test_delete_prediction_updates_view(self, tmp_path):
        """Deleting an individual prediction updates aggregates."""
        store = _make_store(tmp_path)
        ids = _populate_store(store, n_folds=3)

        # Get initial fold_count
        df = store.query_aggregated_predictions()
        initial_pred_count = len(df.row(0, named=True)["prediction_ids"])

        # Delete one prediction
        store.delete_prediction(ids["prediction_ids"][0])

        # Aggregation should update
        df = store.query_aggregated_predictions()
        assert len(df) == 1
        new_pred_count = len(df.row(0, named=True)["prediction_ids"])
        assert new_pred_count == initial_pred_count - 1
        store.close()
