"""Comprehensive unit tests for WorkspaceStore (DuckDB implementation).

Tests cover the full CRUD lifecycle, queries, export operations,
artifact deduplication, garbage collection, cascade deletion, and
concurrent access.  Every test uses ``tmp_path`` for isolation.
"""

from __future__ import annotations

import json
import threading
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

# =========================================================================
# Helpers
# =========================================================================


class _WavelengthAwareAdder:
    """Simple transformer that requires wavelengths in transform()."""

    def transform(self, X, wavelengths=None):  # noqa: ANN001
        if wavelengths is None:
            raise ValueError("wavelengths are required")
        wl = np.asarray(wavelengths, dtype=float).reshape(1, -1)
        return np.asarray(X, dtype=float) + wl


class _SummingModel:
    """Simple model used for chain replay tests."""

    def predict(self, X):  # noqa: ANN001
        return np.asarray(X, dtype=float).sum(axis=1)

def _make_store(tmp_path: Path) -> WorkspaceStore:
    """Create a WorkspaceStore rooted at *tmp_path*."""
    return WorkspaceStore(tmp_path / "workspace")


def _create_full_run(store: WorkspaceStore, *, dataset_name: str = "wheat") -> dict:
    """Create a run -> pipeline -> chain -> prediction hierarchy and return all IDs."""
    run_id = store.begin_run("test_run", config={"metric": "rmse"}, datasets=[{"name": dataset_name}])
    pipeline_id = store.begin_pipeline(
        run_id=run_id,
        name="0001_pls_abc",
        expanded_config=[{"step": "MinMaxScaler"}, {"model": "PLSRegression"}],
        generator_choices=[{"_or_": "MinMaxScaler"}],
        dataset_name=dataset_name,
        dataset_hash="abc123",
    )

    # Save an artifact so the chain has something to reference
    from sklearn.preprocessing import MinMaxScaler

    art_id = store.save_artifact(MinMaxScaler(), "sklearn.preprocessing.MinMaxScaler", "transformer", "joblib")

    chain_id = store.save_chain(
        pipeline_id=pipeline_id,
        steps=[
            {"step_idx": 0, "operator_class": "MinMaxScaler", "params": {}, "artifact_id": art_id, "stateless": False},
            {"step_idx": 1, "operator_class": "PLSRegression", "params": {"n_components": 10}, "artifact_id": None, "stateless": False},
        ],
        model_step_idx=1,
        model_class="sklearn.cross_decomposition.PLSRegression",
        preprocessings="MinMax",
        fold_strategy="per_fold",
        fold_artifacts={"fold_0": art_id},
        shared_artifacts={"0": art_id},
    )

    pred_id = store.save_prediction(
        pipeline_id=pipeline_id,
        chain_id=chain_id,
        dataset_name=dataset_name,
        model_name="PLSRegression",
        model_class="sklearn.cross_decomposition.PLSRegression",
        fold_id="fold_0",
        partition="val",
        val_score=0.12,
        test_score=0.15,
        train_score=0.08,
        metric="rmse",
        task_type="regression",
        n_samples=100,
        n_features=200,
        scores={"val": {"rmse": 0.12}, "test": {"rmse": 0.15}},
        best_params={"n_components": 10},
        branch_id=None,
        branch_name=None,
        exclusion_count=0,
        exclusion_rate=0.0,
        preprocessings="MinMax",
    )

    return {
        "run_id": run_id,
        "pipeline_id": pipeline_id,
        "chain_id": chain_id,
        "pred_id": pred_id,
        "artifact_id": art_id,
    }


# =========================================================================
# test_run_lifecycle
# =========================================================================

class TestRunLifecycle:
    """begin -> complete -> query."""

    def test_run_lifecycle(self, tmp_path):
        """Full run lifecycle: begin, query, complete, verify."""
        store = _make_store(tmp_path)

        run_id = store.begin_run("experiment_1", config={"k": 5}, datasets=[{"name": "wheat"}])
        assert isinstance(run_id, str)
        assert len(run_id) > 0

        # Query the run
        run = store.get_run(run_id)
        assert run is not None
        assert run["name"] == "experiment_1"
        assert run["status"] == "running"
        assert run["config"] == {"k": 5}
        assert run["datasets"] == [{"name": "wheat"}]

        # Complete
        store.complete_run(run_id, summary={"total": 5})
        run = store.get_run(run_id)
        assert run["status"] == "completed"
        assert run["summary"] == {"total": 5}
        assert run["completed_at"] is not None

        store.close()


# =========================================================================
# test_run_failure
# =========================================================================

class TestRunFailure:
    """begin -> fail -> verify no leaked data."""

    def test_run_failure(self, tmp_path):
        """Failed run records error and status."""
        store = _make_store(tmp_path)

        run_id = store.begin_run("fail_run", config={}, datasets=[])
        store.fail_run(run_id, error="OOM error")

        run = store.get_run(run_id)
        assert run["status"] == "failed"
        assert run["error"] == "OOM error"
        assert run["completed_at"] is not None

        store.close()


# =========================================================================
# test_pipeline_crud
# =========================================================================

class TestPipelineCrud:
    """Create, query, complete, fail pipelines."""

    def test_pipeline_crud(self, tmp_path):
        """Pipeline lifecycle: begin, query, complete."""
        store = _make_store(tmp_path)

        run_id = store.begin_run("run1", config={}, datasets=[])
        pid = store.begin_pipeline(
            run_id=run_id,
            name="pipe_001",
            expanded_config={"steps": ["scaler", "pls"]},
            generator_choices=[],
            dataset_name="corn",
            dataset_hash="def456",
        )

        pipeline = store.get_pipeline(pid)
        assert pipeline is not None
        assert pipeline["name"] == "pipe_001"
        assert pipeline["status"] == "running"
        assert pipeline["dataset_name"] == "corn"
        assert pipeline["expanded_config"] == {"steps": ["scaler", "pls"]}

        # Complete
        store.complete_pipeline(pid, best_val=0.05, best_test=0.08, metric="rmse", duration_ms=1234)
        pipeline = store.get_pipeline(pid)
        assert pipeline["status"] == "completed"
        assert pipeline["best_val"] == 0.05
        assert pipeline["best_test"] == 0.08
        assert pipeline["metric"] == "rmse"
        assert pipeline["duration_ms"] == 1234

        store.close()

    def test_pipeline_failure(self, tmp_path):
        """Pipeline failure records error."""
        store = _make_store(tmp_path)

        run_id = store.begin_run("run1", config={}, datasets=[])
        pid = store.begin_pipeline(run_id, "pipe", {}, [], "ds", "hash")

        store.fail_pipeline(pid, error="convergence failure")

        pipeline = store.get_pipeline(pid)
        assert pipeline["status"] == "failed"
        assert pipeline["error"] == "convergence failure"

        store.close()

    def test_list_pipelines(self, tmp_path):
        """list_pipelines with filters."""
        store = _make_store(tmp_path)

        run_id = store.begin_run("run1", config={}, datasets=[])
        store.begin_pipeline(run_id, "p1", {}, [], "wheat", "h1")
        store.begin_pipeline(run_id, "p2", {}, [], "corn", "h2")
        store.begin_pipeline(run_id, "p3", {}, [], "wheat", "h3")

        # All pipelines
        df = store.list_pipelines()
        assert len(df) == 3

        # Filter by dataset
        df = store.list_pipelines(dataset_name="wheat")
        assert len(df) == 2

        # Filter by run_id
        df = store.list_pipelines(run_id=run_id)
        assert len(df) == 3

        store.close()


# =========================================================================
# test_chain_save_load
# =========================================================================

class TestChainSaveLoad:
    """Save chain with fold_artifacts, shared_artifacts; load by ID."""

    def test_chain_save_load(self, tmp_path):
        """Chain round-trip: save and load."""
        store = _make_store(tmp_path)
        ids = _create_full_run(store)

        chain = store.get_chain(ids["chain_id"])
        assert chain is not None
        assert chain["model_class"] == "sklearn.cross_decomposition.PLSRegression"
        assert chain["model_step_idx"] == 1
        assert chain["preprocessings"] == "MinMax"
        assert chain["fold_strategy"] == "per_fold"
        assert isinstance(chain["steps"], list)
        assert len(chain["steps"]) == 2
        assert chain["fold_artifacts"] == {"fold_0": ids["artifact_id"]}
        assert chain["shared_artifacts"] == {"0": ids["artifact_id"]}

        store.close()

    def test_get_chain_not_found(self, tmp_path):
        """get_chain returns None for unknown chain_id."""
        store = _make_store(tmp_path)
        assert store.get_chain("nonexistent") is None
        store.close()

    def test_get_chains_for_pipeline(self, tmp_path):
        """get_chains_for_pipeline returns DataFrame."""
        store = _make_store(tmp_path)
        ids = _create_full_run(store)

        df = store.get_chains_for_pipeline(ids["pipeline_id"])
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 1
        assert df["chain_id"][0] == ids["chain_id"]

        store.close()

    def test_chain_with_branch_path(self, tmp_path):
        """Chain with branch_path and source_index."""
        store = _make_store(tmp_path)
        run_id = store.begin_run("run", config={}, datasets=[])
        pid = store.begin_pipeline(run_id, "p", {}, [], "ds", "h")

        chain_id = store.save_chain(
            pipeline_id=pid,
            steps=[{"step_idx": 0, "operator_class": "M", "params": {}, "artifact_id": None, "stateless": True}],
            model_step_idx=0,
            model_class="Model",
            preprocessings="",
            fold_strategy="per_fold",
            fold_artifacts={},
            shared_artifacts={},
            branch_path=[0, 1],
            source_index=2,
        )

        chain = store.get_chain(chain_id)
        assert chain["branch_path"] == [0, 1]
        assert chain["source_index"] == 2

        store.close()


class TestChainReplay:
    """Chain replay behavior including wavelength passthrough."""

    def test_replay_chain_passes_wavelengths_to_transformers(self, tmp_path):
        """replay_chain forwards wavelengths when transformer supports it."""
        store = _make_store(tmp_path)
        run_id = store.begin_run("run", config={}, datasets=[])
        pipeline_id = store.begin_pipeline(run_id, "pipe", {}, [], "ds", "hash")

        transformer_id = store.save_artifact(
            _WavelengthAwareAdder(),
            "tests._WavelengthAwareAdder",
            "transformer",
            "joblib",
        )
        model_id = store.save_artifact(
            _SummingModel(),
            "tests._SummingModel",
            "model",
            "joblib",
        )

        chain_id = store.save_chain(
            pipeline_id=pipeline_id,
            steps=[
                {"step_idx": 0, "operator_class": "Adder", "params": {}, "artifact_id": transformer_id, "stateless": False},
                {"step_idx": 1, "operator_class": "Model", "params": {}, "artifact_id": model_id, "stateless": False},
            ],
            model_step_idx=1,
            model_class="tests._SummingModel",
            preprocessings="",
            fold_strategy="per_fold",
            fold_artifacts={"fold_0": model_id},
            shared_artifacts={"0": [transformer_id]},
        )

        X = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        wavelengths = np.array([10.0, 20.0, 30.0])

        y_pred = store.replay_chain(chain_id=chain_id, X=X, wavelengths=wavelengths)

        expected = (X + wavelengths.reshape(1, -1)).sum(axis=1)
        np.testing.assert_allclose(y_pred, expected)
        store.close()


# =========================================================================
# test_prediction_save_query
# =========================================================================

class TestPredictionSaveQuery:
    """Save 100 predictions; query by dataset, model, partition."""

    def test_prediction_save_query(self, tmp_path):
        """Bulk prediction save and filtered queries."""
        store = _make_store(tmp_path)
        run_id = store.begin_run("run", config={}, datasets=[])
        pid = store.begin_pipeline(run_id, "p", {}, [], "wheat", "h")
        chain_id = store.save_chain(pid, [{"step_idx": 0, "operator_class": "M", "params": {}, "artifact_id": None, "stateless": True}], 0, "Model", "", "per_fold", {}, {})

        # Save 100 predictions: 50 val + 50 test
        # Each prediction has a unique natural key (pipeline_id, chain_id, fold_id, partition, model_name)
        pred_ids = []
        for i in range(100):
            part = "val" if i < 50 else "test"
            model_cls = "PLSRegression" if i % 2 == 0 else "Ridge"
            pred_id = store.save_prediction(
                pipeline_id=pid,
                chain_id=chain_id,
                dataset_name="wheat",
                model_name=f"{model_cls}_{i}",
                model_class=f"sklearn.{model_cls}",
                fold_id=f"fold_{i % 5}",
                partition=part,
                val_score=0.1 + i * 0.001,
                test_score=0.15 + i * 0.001,
                train_score=0.05 + i * 0.001,
                metric="rmse",
                task_type="regression",
                n_samples=100,
                n_features=200,
                scores={"val": {"rmse": 0.1 + i * 0.001}},
                best_params={},
                branch_id=None,
                branch_name=None,
                exclusion_count=0,
                exclusion_rate=0.0,
            )
            pred_ids.append(pred_id)

        # Query all
        df = store.query_predictions()
        assert len(df) == 100

        # By dataset
        df = store.query_predictions(dataset_name="wheat")
        assert len(df) == 100

        # By partition
        df = store.query_predictions(partition="val")
        assert len(df) == 50

        # By model_class (LIKE pattern)
        df = store.query_predictions(model_class="sklearn.PLS%")
        assert len(df) == 50

        # By fold
        df = store.query_predictions(fold_id="fold_0")
        assert len(df) == 20

        # With limit
        df = store.query_predictions(limit=10)
        assert len(df) == 10

        # By pipeline_id
        df = store.query_predictions(pipeline_id=pid)
        assert len(df) == 100

        # By run_id
        df = store.query_predictions(run_id=run_id)
        assert len(df) == 100

        store.close()

    def test_prediction_upsert_guard_for_explicit_prediction_id(self, tmp_path):
        """Explicit-ID writes with same natural key should update in-place."""
        store = _make_store(tmp_path)
        run_id = store.begin_run("run", config={}, datasets=[])
        pid = store.begin_pipeline(run_id, "p", {}, [], "wheat", "h")
        chain_id = store.save_chain(
            pid,
            [{"step_idx": 0, "operator_class": "M", "params": {}, "artifact_id": None, "stateless": True}],
            0,
            "Model",
            "",
            "per_fold",
            {},
            {},
        )

        first_id = store.save_prediction(
            pipeline_id=pid,
            chain_id=chain_id,
            dataset_name="wheat",
            model_name="PLS",
            model_class="PLS",
            fold_id="fold_0",
            partition="val",
            val_score=0.50,
            test_score=0.60,
            train_score=0.40,
            metric="rmse",
            task_type="regression",
            n_samples=10,
            n_features=5,
            scores={},
            best_params={},
            branch_id=None,
            branch_name=None,
            exclusion_count=0,
            exclusion_rate=0.0,
            prediction_id="pred_explicit",
        )

        second_id = store.save_prediction(
            pipeline_id=pid,
            chain_id=chain_id,
            dataset_name="wheat",
            model_name="PLS",
            model_class="PLS",
            fold_id="fold_0",
            partition="val",
            val_score=0.10,
            test_score=0.20,
            train_score=0.05,
            metric="rmse",
            task_type="regression",
            n_samples=10,
            n_features=5,
            scores={},
            best_params={},
            branch_id=None,
            branch_name=None,
            exclusion_count=0,
            exclusion_rate=0.0,
            prediction_id="pred_explicit_new",
        )

        df = store.query_predictions(pipeline_id=pid)
        assert len(df) == 1
        assert first_id == second_id
        assert df["prediction_id"][0] == "pred_explicit"
        assert df["val_score"][0] == pytest.approx(0.10)

        store.close()


# =========================================================================
# test_prediction_arrays
# =========================================================================

class TestPredictionArrays:
    """Save y_true/y_pred as DOUBLE[]; load; verify numpy roundtrip."""

    def test_prediction_arrays(self, tmp_path):
        """Array round-trip through DuckDB native DOUBLE[]."""
        store = _make_store(tmp_path)
        ids = _create_full_run(store)

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.2])
        sample_indices = np.array([0, 1, 2, 3, 4])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        store.save_prediction_arrays(
            prediction_id=ids["pred_id"],
            y_true=y_true,
            y_pred=y_pred,
            sample_indices=sample_indices,
            weights=weights,
        )

        pred = store.get_prediction(ids["pred_id"], load_arrays=True)
        assert pred is not None
        np.testing.assert_array_almost_equal(pred["y_true"], y_true)
        np.testing.assert_array_almost_equal(pred["y_pred"], y_pred)
        np.testing.assert_array_equal(pred["sample_indices"], sample_indices)
        np.testing.assert_array_almost_equal(pred["weights"], weights)

        store.close()

    def test_prediction_arrays_none_fields(self, tmp_path):
        """Arrays with None fields are handled correctly."""
        store = _make_store(tmp_path)
        ids = _create_full_run(store)

        store.save_prediction_arrays(
            prediction_id=ids["pred_id"],
            y_true=np.array([1.0, 2.0]),
            y_pred=np.array([1.1, 2.1]),
            y_proba=None,
            sample_indices=None,
            weights=None,
        )

        pred = store.get_prediction(ids["pred_id"], load_arrays=True)
        assert pred["y_proba"] is None
        assert pred["sample_indices"] is None
        assert pred["weights"] is None

        store.close()

    def test_prediction_without_arrays(self, tmp_path):
        """get_prediction with load_arrays=False omits array fields."""
        store = _make_store(tmp_path)
        ids = _create_full_run(store)

        pred = store.get_prediction(ids["pred_id"], load_arrays=False)
        assert pred is not None
        assert "y_true" not in pred
        assert "y_pred" not in pred

        store.close()


# =========================================================================
# test_top_predictions
# =========================================================================

class TestTopPredictions:
    """Top-N with group_by, ascending/descending."""

    def test_top_predictions(self, tmp_path):
        """Top-N ranking returns correct order."""
        store = _make_store(tmp_path)
        run_id = store.begin_run("run", config={}, datasets=[])
        pid = store.begin_pipeline(run_id, "p", {}, [], "wheat", "h")
        cid = store.save_chain(pid, [{"step_idx": 0, "operator_class": "M", "params": {}, "artifact_id": None, "stateless": True}], 0, "M", "", "per_fold", {}, {})

        # Create predictions with different val_scores (unique fold_id per row)
        for i in range(20):
            store.save_prediction(
                pipeline_id=pid, chain_id=cid, dataset_name="wheat",
                model_name="PLS", model_class="PLS" if i < 10 else "Ridge",
                fold_id=f"f{i}", partition="val",
                val_score=float(i), test_score=float(i), train_score=float(i),
                metric="rmse", task_type="regression",
                n_samples=50, n_features=100,
                scores={}, best_params={},
                branch_id=None, branch_name=None,
                exclusion_count=0, exclusion_rate=0.0,
            )

        # Top 5 ascending (lower is better)
        df = store.top_predictions(5, metric="val_score", ascending=True)
        assert len(df) == 5
        scores = df["val_score"].to_list()
        assert scores == sorted(scores)

        # Top 5 descending (higher is better)
        df = store.top_predictions(5, metric="val_score", ascending=False)
        assert len(df) == 5
        scores = df["val_score"].to_list()
        assert scores == sorted(scores, reverse=True)

        store.close()

    def test_top_predictions_with_group_by(self, tmp_path):
        """Top-N per group."""
        store = _make_store(tmp_path)
        run_id = store.begin_run("run", config={}, datasets=[])
        pid = store.begin_pipeline(run_id, "p", {}, [], "wheat", "h")
        cid = store.save_chain(pid, [{"step_idx": 0, "operator_class": "M", "params": {}, "artifact_id": None, "stateless": True}], 0, "M", "", "per_fold", {}, {})

        # 5 PLS + 5 Ridge predictions (unique fold_id per row)
        for i in range(10):
            model = "PLS" if i < 5 else "Ridge"
            store.save_prediction(
                pipeline_id=pid, chain_id=cid, dataset_name="wheat",
                model_name=model, model_class=model,
                fold_id=f"f{i}", partition="val",
                val_score=float(i), test_score=0.0, train_score=0.0,
                metric="rmse", task_type="regression",
                n_samples=50, n_features=100,
                scores={}, best_params={},
                branch_id=None, branch_name=None,
                exclusion_count=0, exclusion_rate=0.0,
            )

        # Top 2 per model_class
        df = store.top_predictions(2, group_by="model_class", ascending=True)
        # Should have 2 PLS + 2 Ridge = 4 rows
        assert len(df) == 4

        store.close()

    def test_top_predictions_with_dataset_filter(self, tmp_path):
        """Top-N with dataset filter."""
        store = _make_store(tmp_path)
        run_id = store.begin_run("run", config={}, datasets=[])
        pid = store.begin_pipeline(run_id, "p", {}, [], "wheat", "h")
        cid = store.save_chain(pid, [{"step_idx": 0, "operator_class": "M", "params": {}, "artifact_id": None, "stateless": True}], 0, "M", "", "per_fold", {}, {})

        for i in range(10):
            ds = "wheat" if i < 5 else "corn"
            store.save_prediction(
                pipeline_id=pid, chain_id=cid, dataset_name=ds,
                model_name="PLS", model_class="PLS",
                fold_id=f"f{i}", partition="val",
                val_score=float(i), test_score=0.0, train_score=0.0,
                metric="rmse", task_type="regression",
                n_samples=50, n_features=100,
                scores={}, best_params={},
                branch_id=None, branch_name=None,
                exclusion_count=0, exclusion_rate=0.0,
            )

        df = store.top_predictions(3, dataset_name="wheat")
        assert len(df) == 3
        assert all(ds == "wheat" for ds in df["dataset_name"].to_list())

        store.close()


# =========================================================================
# test_artifact_dedup
# =========================================================================

class TestArtifactDedup:
    """Save same binary twice -> same artifact_id, ref_count=2."""

    def test_artifact_dedup(self, tmp_path):
        """Content-addressed deduplication increments ref_count."""
        store = _make_store(tmp_path)
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.mean_ = np.array([1.0, 2.0, 3.0])
        scaler.scale_ = np.array([0.5, 0.5, 0.5])

        aid1 = store.save_artifact(scaler, "sklearn.StandardScaler", "transformer", "joblib")
        aid2 = store.save_artifact(scaler, "sklearn.StandardScaler", "transformer", "joblib")

        assert aid1 == aid2

        # Check ref_count
        conn = store._ensure_open()
        result = conn.execute("SELECT ref_count FROM artifacts WHERE artifact_id = $1", [aid1]).fetchone()
        assert result[0] == 2

        store.close()


# =========================================================================
# test_artifact_gc
# =========================================================================

class TestArtifactGC:
    """Delete chain -> ref_count decrements -> gc removes file."""

    def test_artifact_gc(self, tmp_path):
        """Artifact garbage collection removes orphaned files."""
        store = _make_store(tmp_path)
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        scaler.data_min_ = np.array([0.0])
        scaler.data_max_ = np.array([1.0])

        aid = store.save_artifact(scaler, "sklearn.MinMaxScaler", "transformer", "joblib")
        artifact_path = store.get_artifact_path(aid)
        assert artifact_path.exists()

        # Decrement ref_count to 0
        conn = store._ensure_open()
        conn.execute("UPDATE artifacts SET ref_count = 0 WHERE artifact_id = $1", [aid])

        # GC should remove the file
        removed = store.gc_artifacts()
        assert removed == 1
        assert not artifact_path.exists()

        store.close()


# =========================================================================
# test_log_step
# =========================================================================

class TestLogStep:
    """Log events; query per pipeline; aggregate per run."""

    def test_log_step(self, tmp_path):
        """Log step events and query them."""
        store = _make_store(tmp_path)
        run_id = store.begin_run("run", config={}, datasets=[])
        pid = store.begin_pipeline(run_id, "p", {}, [], "ds", "h")

        store.log_step(pid, 0, "MinMaxScaler", "start", message="Starting")
        store.log_step(pid, 0, "MinMaxScaler", "end", duration_ms=42, message="Done")
        store.log_step(pid, 1, "PLSRegression", "start")
        store.log_step(pid, 1, "PLSRegression", "end", duration_ms=150)
        store.log_step(pid, 2, "Validator", "warning", message="Low R2", level="warning")

        # Pipeline log
        df = store.get_pipeline_log(pid)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 5

        # Run log summary
        summary = store.get_run_log_summary(run_id)
        assert isinstance(summary, pl.DataFrame)
        assert len(summary) == 1
        row = summary.row(0, named=True)
        assert row["log_count"] == 5
        assert row["warning_count"] == 1
        assert row["total_duration_ms"] == 192  # 42 + 150

        store.close()

    def test_log_with_details(self, tmp_path):
        """Log step with structured details."""
        store = _make_store(tmp_path)
        run_id = store.begin_run("run", config={}, datasets=[])
        pid = store.begin_pipeline(run_id, "p", {}, [], "ds", "h")

        store.log_step(
            pid, 0, "Scaler", "end",
            duration_ms=10,
            details={"transform_shape": [100, 200]},
            level="info",
        )

        df = store.get_pipeline_log(pid)
        assert len(df) == 1

        store.close()


# =========================================================================
# test_delete_cascade
# =========================================================================

class TestDeleteCascade:
    """Delete run -> pipelines, chains, predictions, arrays deleted."""

    def test_delete_cascade(self, tmp_path):
        """Deleting a run cascades to all dependents."""
        store = _make_store(tmp_path)
        ids = _create_full_run(store)

        # Save arrays too
        store.save_prediction_arrays(
            ids["pred_id"],
            y_true=np.array([1.0, 2.0]),
            y_pred=np.array([1.1, 2.1]),
        )

        # Also add a log entry
        store.log_step(ids["pipeline_id"], 0, "Scaler", "end", duration_ms=10)

        # Delete
        total_deleted = store.delete_run(ids["run_id"])
        assert total_deleted > 0

        # Verify everything is gone
        assert store.get_run(ids["run_id"]) is None
        assert store.get_pipeline(ids["pipeline_id"]) is None
        assert store.get_chain(ids["chain_id"]) is None
        assert store.get_prediction(ids["pred_id"]) is None

        # Check tables are empty
        conn = store._ensure_open()
        for table in ["runs", "pipelines", "chains", "predictions", "prediction_arrays", "logs"]:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert count == 0, f"{table} should be empty after delete_run"

        store.close()

    def test_delete_prediction(self, tmp_path):
        """Delete a single prediction and its arrays."""
        store = _make_store(tmp_path)
        ids = _create_full_run(store)

        store.save_prediction_arrays(ids["pred_id"], np.array([1.0]), np.array([1.1]))

        result = store.delete_prediction(ids["pred_id"])
        assert result is True

        assert store.get_prediction(ids["pred_id"]) is None

        # Trying again returns False
        result = store.delete_prediction(ids["pred_id"])
        assert result is False

        store.close()


# =========================================================================
# test_export_chain_n4a
# =========================================================================

class TestExportChainN4a:
    """Export chain -> valid .n4a ZIP with manifest + artifacts."""

    def test_export_chain_n4a(self, tmp_path):
        """Exported .n4a is a valid ZIP with manifest.json and chain.json."""
        store = _make_store(tmp_path)
        ids = _create_full_run(store)

        output = tmp_path / "model.n4a"
        result_path = store.export_chain(ids["chain_id"], output)

        assert result_path.exists()
        assert result_path.suffix == ".n4a"

        # Verify ZIP contents
        with zipfile.ZipFile(result_path, "r") as zf:
            names = zf.namelist()
            assert "manifest.json" in names
            assert "chain.json" in names

            # Check manifest
            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["chain_id"] == ids["chain_id"]
            assert "model_class" in manifest
            assert "exported_at" in manifest

            # Check chain
            chain_data = json.loads(zf.read("chain.json"))
            assert "steps" in chain_data
            assert "fold_artifacts" in chain_data

            # Check artifacts are included
            artifact_files = [n for n in names if n.startswith("artifacts/")]
            assert len(artifact_files) > 0

        store.close()

    def test_export_chain_not_found(self, tmp_path):
        """Exporting a nonexistent chain raises KeyError."""
        store = _make_store(tmp_path)
        with pytest.raises(KeyError):
            store.export_chain("nonexistent", tmp_path / "out.n4a")
        store.close()


# =========================================================================
# test_export_pipeline_config
# =========================================================================

class TestExportPipelineConfig:
    """Export pipeline config -> valid JSON."""

    def test_export_pipeline_config(self, tmp_path):
        """Exported pipeline config is valid JSON matching expanded_config."""
        store = _make_store(tmp_path)
        ids = _create_full_run(store)

        output = tmp_path / "config.json"
        result_path = store.export_pipeline_config(ids["pipeline_id"], output)

        assert result_path.exists()
        assert result_path.suffix == ".json"

        content = json.loads(result_path.read_text())
        assert isinstance(content, list)
        assert len(content) == 2

        store.close()

    def test_export_pipeline_config_not_found(self, tmp_path):
        """Exporting a nonexistent pipeline raises KeyError."""
        store = _make_store(tmp_path)
        with pytest.raises(KeyError):
            store.export_pipeline_config("nonexistent", tmp_path / "out.json")
        store.close()


# =========================================================================
# test_export_run
# =========================================================================

class TestExportRun:
    """Export run -> valid YAML with all pipelines."""

    def test_export_run(self, tmp_path):
        """Exported run YAML contains run metadata and pipeline list."""
        store = _make_store(tmp_path)
        ids = _create_full_run(store)
        store.complete_pipeline(ids["pipeline_id"], 0.12, 0.15, "rmse", 500)
        store.complete_run(ids["run_id"], summary={"best_rmse": 0.12})

        output = tmp_path / "run.yaml"
        result_path = store.export_run(ids["run_id"], output)

        assert result_path.exists()
        assert result_path.suffix == ".yaml"

        with open(result_path) as f:
            data = yaml.safe_load(f)

        assert data["run_id"] == ids["run_id"]
        assert data["name"] == "test_run"
        assert data["status"] == "completed"
        assert "pipelines" in data
        assert len(data["pipelines"]) == 1
        assert data["pipelines"][0]["pipeline_id"] == ids["pipeline_id"]

        store.close()

    def test_export_run_not_found(self, tmp_path):
        """Exporting a nonexistent run raises KeyError."""
        store = _make_store(tmp_path)
        with pytest.raises(KeyError):
            store.export_run("nonexistent", tmp_path / "out.yaml")
        store.close()


# =========================================================================
# test_concurrent_read_write
# =========================================================================

class TestConcurrentReadWrite:
    """Two threads: one writing, one reading -> no corruption."""

    def test_concurrent_read_write(self, tmp_path):
        """Concurrent read and write threads do not corrupt data."""
        workspace = tmp_path / "workspace"

        # DuckDB uses a single-writer model, so concurrent connections
        # writing simultaneously will conflict.  Instead, test that a
        # single store can be used safely from multiple threads (the
        # connection serialises access).
        store = WorkspaceStore(workspace)
        errors = []

        def writer():
            try:
                for i in range(20):
                    run_id = store.begin_run(f"run_{i}", config={}, datasets=[])
                    store.complete_run(run_id, summary={"i": i})
            except Exception as e:
                errors.append(("writer", e))

        def reader():
            try:
                for _ in range(20):
                    store.list_runs()
            except Exception as e:
                errors.append(("reader", e))

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)

        t1.start()
        t2.start()

        t1.join(timeout=30)
        t2.join(timeout=30)
        store.close()

        assert errors == [], f"Concurrent access errors: {errors}"


# =========================================================================
# test_empty_workspace
# =========================================================================

class TestEmptyWorkspace:
    """All queries return empty DataFrames on fresh store."""

    def test_empty_workspace(self, tmp_path):
        """Fresh store returns empty results for all queries."""
        store = _make_store(tmp_path)

        # Runs
        assert store.get_run("nonexistent") is None
        df = store.list_runs()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

        # Pipelines
        assert store.get_pipeline("nonexistent") is None
        df = store.list_pipelines()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

        # Predictions
        assert store.get_prediction("nonexistent") is None
        df = store.query_predictions()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

        df = store.top_predictions(10)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

        # Chains
        assert store.get_chain("nonexistent") is None
        df = store.get_chains_for_pipeline("nonexistent")
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

        # Logs
        df = store.get_pipeline_log("nonexistent")
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

        df = store.get_run_log_summary("nonexistent")
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

        store.close()


# =========================================================================
# test_schema_creation (via WorkspaceStore)
# =========================================================================

class TestSchemaCreationViaStore:
    """Create store from scratch; verify all tables exist."""

    def test_schema_creation(self, tmp_path):
        """WorkspaceStore constructor creates all 7 tables and the aggregation VIEW."""
        store = _make_store(tmp_path)
        conn = store._ensure_open()

        result = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_type = 'BASE TABLE' "
            "ORDER BY table_name"
        ).fetchall()
        tables = sorted([row[0] for row in result])

        from nirs4all.pipeline.storage.store_schema import TABLE_NAMES
        assert tables == sorted(TABLE_NAMES)

        # Verify aggregation VIEW also exists
        views = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_type = 'VIEW'"
        ).fetchall()
        view_names = [row[0] for row in views]
        assert "v_aggregated_predictions" in view_names

        store.close()

    def test_artifacts_directory_created(self, tmp_path):
        """WorkspaceStore constructor creates artifacts/ directory."""
        store = _make_store(tmp_path)
        artifacts_dir = tmp_path / "workspace" / "artifacts"
        assert artifacts_dir.exists()
        assert artifacts_dir.is_dir()
        store.close()

    def test_store_duckdb_file_created(self, tmp_path):
        """WorkspaceStore constructor creates store.duckdb file."""
        store = _make_store(tmp_path)
        db_path = tmp_path / "workspace" / "store.duckdb"
        assert db_path.exists()
        store.close()


# =========================================================================
# test_export_predictions_parquet
# =========================================================================

class TestExportPredictionsParquet:
    """Export filtered predictions via Polars write_parquet."""

    def test_export_predictions_parquet(self, tmp_path):
        """Exported parquet is readable by Polars."""
        store = _make_store(tmp_path)
        ids = _create_full_run(store)

        output = tmp_path / "preds.parquet"
        result_path = store.export_predictions_parquet(output, dataset_name="wheat")

        assert result_path.exists()
        df = pl.read_parquet(result_path)
        assert len(df) == 1
        assert df["dataset_name"][0] == "wheat"

        store.close()

    def test_export_empty_predictions(self, tmp_path):
        """Exporting with no matching predictions produces an empty parquet."""
        store = _make_store(tmp_path)
        output = tmp_path / "empty.parquet"
        result_path = store.export_predictions_parquet(output)
        assert result_path.exists()
        df = pl.read_parquet(result_path)
        assert len(df) == 0
        store.close()


# =========================================================================
# test_artifact_load
# =========================================================================

class TestArtifactLoad:
    """Save and load artifact round-trip."""

    def test_artifact_save_load(self, tmp_path):
        """Artifact round-trip: save object, load back, verify equality."""
        store = _make_store(tmp_path)
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.mean_ = np.array([1.0, 2.0, 3.0])
        scaler.scale_ = np.array([0.5, 0.5, 0.5])

        aid = store.save_artifact(scaler, "sklearn.StandardScaler", "transformer", "joblib")
        loaded = store.load_artifact(aid)

        assert type(loaded).__name__ == "StandardScaler"
        np.testing.assert_array_equal(loaded.mean_, scaler.mean_)
        np.testing.assert_array_equal(loaded.scale_, scaler.scale_)

        store.close()

    def test_load_unknown_artifact(self, tmp_path):
        """Loading an unknown artifact raises KeyError."""
        store = _make_store(tmp_path)
        with pytest.raises(KeyError):
            store.load_artifact("nonexistent")
        store.close()

    def test_get_artifact_path(self, tmp_path):
        """get_artifact_path returns a valid Path."""
        store = _make_store(tmp_path)
        from sklearn.preprocessing import MinMaxScaler

        aid = store.save_artifact(MinMaxScaler(), "sklearn.MinMaxScaler", "scaler", "joblib")
        path = store.get_artifact_path(aid)
        assert isinstance(path, Path)
        assert path.exists()

        store.close()

    def test_get_artifact_path_unknown(self, tmp_path):
        """get_artifact_path for unknown ID raises KeyError."""
        store = _make_store(tmp_path)
        with pytest.raises(KeyError):
            store.get_artifact_path("nonexistent")
        store.close()


# =========================================================================
# test_close
# =========================================================================

class TestClose:
    """Verify close and post-close behavior."""

    def test_close_idempotent(self, tmp_path):
        """close() can be called multiple times."""
        store = _make_store(tmp_path)
        store.close()
        store.close()  # Should not raise

    def test_operations_after_close_raise(self, tmp_path):
        """Operations after close raise RuntimeError."""
        store = _make_store(tmp_path)
        store.close()
        with pytest.raises(RuntimeError):
            store.begin_run("test", config={}, datasets=[])


# =========================================================================
# test_vacuum
# =========================================================================

class TestVacuum:
    """Verify vacuum runs without error."""

    def test_vacuum(self, tmp_path):
        """vacuum() executes without raising."""
        store = _make_store(tmp_path)
        ids = _create_full_run(store)
        store.delete_run(ids["run_id"])
        store.vacuum()  # Should not raise
        store.close()


# =========================================================================
# test_list_runs
# =========================================================================

class TestListRuns:
    """Verify list_runs filtering."""

    def test_list_runs_by_status(self, tmp_path):
        """Filter runs by status."""
        store = _make_store(tmp_path)

        r1 = store.begin_run("run1", config={}, datasets=[])
        r2 = store.begin_run("run2", config={}, datasets=[])
        store.complete_run(r1, summary={})

        df = store.list_runs(status="completed")
        assert len(df) == 1
        assert df["run_id"][0] == r1

        df = store.list_runs(status="running")
        assert len(df) == 1
        assert df["run_id"][0] == r2

        store.close()

    def test_list_runs_pagination(self, tmp_path):
        """Pagination with limit and offset."""
        store = _make_store(tmp_path)

        for i in range(10):
            store.begin_run(f"run_{i}", config={}, datasets=[])

        df = store.list_runs(limit=3, offset=0)
        assert len(df) == 3

        df = store.list_runs(limit=3, offset=7)
        assert len(df) == 3

        df = store.list_runs(limit=100, offset=0)
        assert len(df) == 10

        store.close()

    def test_list_runs_by_dataset(self, tmp_path):
        """Filter runs by dataset name in datasets JSON."""
        store = _make_store(tmp_path)

        store.begin_run("run1", config={}, datasets=[{"name": "wheat"}])
        store.begin_run("run2", config={}, datasets=[{"name": "corn"}])
        store.begin_run("run3", config={}, datasets=[{"name": "wheat"}, {"name": "corn"}])

        df = store.list_runs(dataset="wheat")
        assert len(df) == 2  # run1 and run3

        store.close()


# =========================================================================
# test_sql_injection_guard
# =========================================================================

class TestSQLInjectionGuard:
    """Verify that metric and group_by parameters are validated."""

    def test_invalid_metric_rejected(self, tmp_path):
        """top_predictions rejects invalid metric column names."""
        store = _make_store(tmp_path)
        with pytest.raises(ValueError, match="Invalid metric column"):
            store.top_predictions(5, metric="val_score; DROP TABLE runs--")
        store.close()

    def test_invalid_group_by_rejected(self, tmp_path):
        """top_predictions rejects invalid group_by column names."""
        store = _make_store(tmp_path)
        with pytest.raises(ValueError, match="Invalid group_by column"):
            store.top_predictions(5, group_by="1; DROP TABLE runs--")
        store.close()
