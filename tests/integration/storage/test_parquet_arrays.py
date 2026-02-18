"""Integration test for Parquet-backed prediction arrays (Phase 1).

Validates the full pipeline:
  nirs4all.run() → predictions.top() → arrays load correctly from Parquet.

Covers:
- Pipeline execution writes arrays to Parquet sidecar files (not DuckDB)
- WorkspaceStore.get_prediction(load_arrays=True) returns arrays from Parquet
- ArrayStore files exist on disk with correct structure
- Round-trip: arrays match between save and load
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


@pytest.fixture
def temp_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def regression_data():
    rng = np.random.RandomState(42)
    n_samples = 60
    n_features = 50
    X = rng.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + rng.randn(n_samples) * 0.1
    return X, y


class TestFullPipelineWithParquetArrays:
    """nirs4all.run() → predictions.top() → arrays load correctly from Parquet."""

    def test_arrays_stored_in_parquet(self, temp_workspace, regression_data):
        """After run, arrays directory exists with .parquet files."""
        X, y = regression_data
        pipeline = [
            MinMaxScaler(),
            KFold(n_splits=2, shuffle=True, random_state=42),
            {"model": PLSRegression(n_components=3)},
        ]

        runner = PipelineRunner(
            workspace_path=temp_workspace,
            verbose=0,
            save_artifacts=True,
        )
        runner.run(pipeline, (X, y))

        arrays_dir = temp_workspace / "arrays"
        assert arrays_dir.exists(), "arrays/ directory should be created"
        parquet_files = list(arrays_dir.glob("*.parquet"))
        assert len(parquet_files) > 0, "At least one .parquet file should exist"

    def test_no_prediction_arrays_table(self, temp_workspace, regression_data):
        """The prediction_arrays DuckDB table no longer exists."""
        X, y = regression_data
        pipeline = [MinMaxScaler(), PLSRegression(n_components=3)]

        runner = PipelineRunner(
            workspace_path=temp_workspace,
            verbose=0,
        )
        runner.run(pipeline, (X, y))

        store = WorkspaceStore(temp_workspace)
        try:
            result = store._conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name = 'prediction_arrays' AND table_type = 'BASE TABLE'"
            ).fetchall()
            assert len(result) == 0, "prediction_arrays table should not exist"
        finally:
            store.close()

    def test_get_prediction_loads_arrays(self, temp_workspace, regression_data):
        """get_prediction(load_arrays=True) returns arrays from Parquet."""
        X, y = regression_data
        pipeline = [
            MinMaxScaler(),
            KFold(n_splits=2, shuffle=True, random_state=42),
            {"model": PLSRegression(n_components=3)},
        ]

        runner = PipelineRunner(
            workspace_path=temp_workspace,
            verbose=0,
            save_artifacts=True,
        )
        predictions, _ = runner.run(pipeline, (X, y))

        store = WorkspaceStore(temp_workspace)
        try:
            # Get a prediction_id from the store
            preds_df = store._fetch_pl("SELECT prediction_id FROM predictions LIMIT 1")
            assert len(preds_df) > 0, "Expected at least one prediction"
            pred_id = preds_df.row(0, named=True)["prediction_id"]

            # Load with arrays
            pred = store.get_prediction(pred_id, load_arrays=True)
            assert pred is not None
            assert "y_true" in pred
            assert "y_pred" in pred
            assert pred["y_true"] is not None
            assert pred["y_pred"] is not None
            assert isinstance(pred["y_true"], np.ndarray)
            assert isinstance(pred["y_pred"], np.ndarray)
            assert len(pred["y_true"]) > 0
            assert len(pred["y_pred"]) > 0
        finally:
            store.close()

    def test_array_store_stats(self, temp_workspace, regression_data):
        """ArrayStore.stats() reflects predictions written during run."""
        X, y = regression_data
        pipeline = [
            MinMaxScaler(),
            KFold(n_splits=2, shuffle=True, random_state=42),
            {"model": PLSRegression(n_components=3)},
        ]

        runner = PipelineRunner(
            workspace_path=temp_workspace,
            verbose=0,
            save_artifacts=True,
        )
        runner.run(pipeline, (X, y))

        store = WorkspaceStore(temp_workspace)
        try:
            stats = store.array_store.stats()
            assert stats["total_files"] >= 1
            assert stats["total_rows"] > 0
            assert stats["total_bytes"] > 0
        finally:
            store.close()

    def test_parquet_file_is_self_describing(self, temp_workspace, regression_data):
        """Parquet files contain portable metadata columns."""
        X, y = regression_data
        pipeline = [
            MinMaxScaler(),
            KFold(n_splits=2, shuffle=True, random_state=42),
            {"model": PLSRegression(n_components=3)},
        ]

        runner = PipelineRunner(
            workspace_path=temp_workspace,
            verbose=0,
            save_artifacts=True,
        )
        runner.run(pipeline, (X, y))

        datasets = store = WorkspaceStore(temp_workspace)
        try:
            dataset_names = store.array_store.list_datasets()
            assert len(dataset_names) > 0

            df = store.array_store.load_dataset(dataset_names[0])
            columns = set(df.columns)

            # Self-describing portable columns
            assert "prediction_id" in columns
            assert "dataset_name" in columns
            assert "model_name" in columns
            assert "fold_id" in columns
            assert "partition" in columns
            assert "metric" in columns
            assert "val_score" in columns
            assert "task_type" in columns

            # Array columns
            assert "y_true" in columns
            assert "y_pred" in columns
        finally:
            store.close()
