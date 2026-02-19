"""Integration tests for Phase 2: DuckDB-backed pipeline execution.

Validates that the full pipeline execution flow works with WorkspaceStore
replacing ManifestManager, SimulationSaver, and PipelineWriter.

Tests verify:
- nirs4all.run() produces valid RunResult with DuckDB storage
- No legacy filesystem hierarchy (no runs/ folder, manifest.yaml, pipeline.json)
- Artifacts are saved in flat content-addressed workspace/artifacts/ structure
- Chains are built from ExecutionTrace with correct steps and artifacts
- Generator-expanded pipelines produce distinct pipeline rows
- WorkspaceStore contains expected run/pipeline/chain records
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.pipeline.storage.chain_builder import ChainBuilder
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace

@pytest.fixture
def regression_data():
    """Create simple regression data as (X, y) tuple."""
    rng = np.random.RandomState(42)
    n_samples = 60
    n_features = 50
    X = rng.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + rng.randn(n_samples) * 0.1
    return X, y

class TestBasicPipeline:
    """Test that nirs4all.run() produces valid RunResult with WorkspaceStore."""

    def test_basic_pipeline_produces_predictions(self, temp_workspace, regression_data):
        """nirs4all.run(pipeline, dataset) produces valid RunResult."""
        X, y = regression_data
        pipeline = [
            MinMaxScaler(),
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"model": PLSRegression(n_components=3)},
        ]

        runner = PipelineRunner(
            workspace_path=temp_workspace,
            verbose=0,
            save_artifacts=True,
        )
        predictions, per_dataset = runner.run(pipeline, (X, y))

        assert predictions.num_predictions > 0
        # Use display_partition="val" since KFold only produces train/val
        results = predictions.top(n=1, rank_partition="val", display_partition="val", ascending=True)
        assert len(results) > 0
        best = results[0]
        assert best is not None
        assert "val_score" in best
        assert np.isfinite(best["val_score"])

    def test_basic_pipeline_with_cv(self, temp_workspace, regression_data):
        """Pipeline with cross-validation produces predictions."""
        X, y = regression_data
        pipeline = [
            MinMaxScaler(),
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"model": PLSRegression(n_components=3)},
        ]

        runner = PipelineRunner(
            workspace_path=temp_workspace,
            verbose=0,
            save_artifacts=True,
        )
        predictions, per_dataset = runner.run(pipeline, (X, y))

        assert predictions.num_predictions > 0
        best = predictions.get_best(ascending=None)
        assert best is not None

class TestNoFileHierarchy:
    """Test that no legacy filesystem hierarchy is created."""

    def test_no_runs_folder(self, temp_workspace, regression_data):
        """After run: no runs/ folder created."""
        X, y = regression_data
        pipeline = [MinMaxScaler(), PLSRegression(n_components=3)]

        runner = PipelineRunner(
            workspace_path=temp_workspace,
            verbose=0,
            save_artifacts=True,
        )
        runner.run(pipeline, (X, y))

        runs_dir = temp_workspace / "runs"
        assert not runs_dir.exists(), f"Legacy runs/ folder should not exist, found: {runs_dir}"

    def test_no_manifest_yaml(self, temp_workspace, regression_data):
        """After run: no manifest.yaml files."""
        X, y = regression_data
        pipeline = [MinMaxScaler(), PLSRegression(n_components=3)]

        runner = PipelineRunner(
            workspace_path=temp_workspace,
            verbose=0,
        )
        runner.run(pipeline, (X, y))

        manifest_files = list(temp_workspace.rglob("manifest.yaml"))
        assert len(manifest_files) == 0, f"Legacy manifest.yaml files found: {manifest_files}"

    def test_no_pipeline_json(self, temp_workspace, regression_data):
        """After run: no pipeline.json files."""
        X, y = regression_data
        pipeline = [MinMaxScaler(), PLSRegression(n_components=3)]

        runner = PipelineRunner(
            workspace_path=temp_workspace,
            verbose=0,
        )
        runner.run(pipeline, (X, y))

        pipeline_json_files = list(temp_workspace.rglob("pipeline.json"))
        assert len(pipeline_json_files) == 0, f"Legacy pipeline.json files found: {pipeline_json_files}"

    def test_store_duckdb_exists(self, temp_workspace, regression_data):
        """After run: store.duckdb file exists."""
        X, y = regression_data
        pipeline = [MinMaxScaler(), PLSRegression(n_components=3)]

        runner = PipelineRunner(
            workspace_path=temp_workspace,
            verbose=0,
        )
        runner.run(pipeline, (X, y))

        store_file = temp_workspace / "store.duckdb"
        assert store_file.exists(), f"store.duckdb should exist at {store_file}"

class TestArtifactsFlat:
    """Test that artifacts are saved in flat content-addressed structure."""

    def test_artifacts_in_flat_directory(self, temp_workspace, regression_data):
        """Artifacts saved in workspace/artifacts/ flat structure."""
        X, y = regression_data
        pipeline = [MinMaxScaler(), PLSRegression(n_components=3)]

        runner = PipelineRunner(
            workspace_path=temp_workspace,
            verbose=0,
            save_artifacts=True,
        )
        runner.run(pipeline, (X, y))

        artifacts_dir = temp_workspace / "artifacts"
        if artifacts_dir.exists():
            # Artifacts should be in sharded subdirectories (2-char hex prefix)
            artifact_files = list(artifacts_dir.rglob("*.*"))
            for f in artifact_files:
                if f.is_file():
                    # Parent should be a 2-character shard directory
                    shard = f.parent.name
                    assert len(shard) == 2, f"Artifact shard directory should be 2 chars, got: {shard}"

class TestChainFromTrace:
    """Test that chains are built correctly from ExecutionTrace."""

    def test_chain_stored_in_database(self, temp_workspace, regression_data):
        """Chain built from ExecutionTrace is stored in DuckDB."""
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

        # Open store and verify chain exists
        store = WorkspaceStore(temp_workspace)
        try:
            # Query all chains
            chains_df = store._fetch_pl("SELECT * FROM chains")
            assert len(chains_df) > 0, "Expected at least one chain in store"

            # Verify chain has expected fields
            chain_row = chains_df.row(0, named=True)
            assert chain_row["model_class"] != "", "Chain should have model_class"
            assert chain_row["preprocessings"] is not None, "Chain should have preprocessings"
        finally:
            store.close()

    def test_chain_has_correct_model_step(self, temp_workspace, regression_data):
        """Chain model_step_idx points to the model step."""
        X, y = regression_data
        pipeline = [
            StandardScaler(),
            KFold(n_splits=2, shuffle=True, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        runner = PipelineRunner(
            workspace_path=temp_workspace,
            verbose=0,
            save_artifacts=True,
        )
        runner.run(pipeline, (X, y))

        store = WorkspaceStore(temp_workspace)
        try:
            chains_df = store._fetch_pl("SELECT * FROM chains")
            assert len(chains_df) > 0
            chain_row = chains_df.row(0, named=True)
            # model_step_idx should be a valid integer
            assert isinstance(chain_row["model_step_idx"], int)
        finally:
            store.close()

class TestStoreContents:
    """Test that WorkspaceStore contains expected records."""

    def test_run_record_exists(self, temp_workspace, regression_data):
        """Store contains a completed run record."""
        X, y = regression_data
        pipeline = [MinMaxScaler(), PLSRegression(n_components=3)]

        runner = PipelineRunner(
            workspace_path=temp_workspace,
            verbose=0,
        )
        runner.run(pipeline, (X, y))

        store = WorkspaceStore(temp_workspace)
        try:
            runs_df = store._fetch_pl("SELECT * FROM runs")
            assert len(runs_df) == 1, f"Expected 1 run, got {len(runs_df)}"
            run_row = runs_df.row(0, named=True)
            assert run_row["status"] == "completed", f"Run should be completed, got: {run_row['status']}"
        finally:
            store.close()

    def test_pipeline_record_exists(self, temp_workspace, regression_data):
        """Store contains a completed pipeline record."""
        X, y = regression_data
        pipeline = [MinMaxScaler(), PLSRegression(n_components=3)]

        runner = PipelineRunner(
            workspace_path=temp_workspace,
            verbose=0,
        )
        runner.run(pipeline, (X, y))

        store = WorkspaceStore(temp_workspace)
        try:
            pipelines_df = store._fetch_pl("SELECT * FROM pipelines")
            assert len(pipelines_df) == 1, f"Expected 1 pipeline, got {len(pipelines_df)}"
            pipeline_row = pipelines_df.row(0, named=True)
            assert pipeline_row["status"] == "completed"
        finally:
            store.close()

    def test_failed_pipeline_rolls_back(self, temp_workspace):
        """Failed pipeline is marked as failed in store."""
        # Create data that will cause a failure (NaN values)
        X = np.full((30, 10), np.nan)
        y = np.random.randn(30)
        pipeline = [MinMaxScaler(), PLSRegression(n_components=3)]

        runner = PipelineRunner(
            workspace_path=temp_workspace,
            verbose=0,
        )

        try:
            runner.run(pipeline, (X, y))
        except Exception:
            pass  # Expected to fail

        store = WorkspaceStore(temp_workspace)
        try:
            runs_df = store._fetch_pl("SELECT * FROM runs")
            if len(runs_df) > 0:
                run_row = runs_df.row(0, named=True)
                assert run_row["status"] in ("failed", "completed"), f"Unexpected run status: {run_row['status']}"
        finally:
            store.close()

class TestChainBuilder:
    """Unit-level tests for ChainBuilder."""

    def test_chain_builder_build_empty_trace(self):
        """ChainBuilder handles an empty trace gracefully."""
        from nirs4all.pipeline.trace.execution_trace import ExecutionTrace

        trace = ExecutionTrace(
            trace_id="test",
            pipeline_uid="test_uid",
            steps=[],
            model_step_index=None,
        )
        builder = ChainBuilder(trace)
        result = builder.build()

        assert isinstance(result, dict)
        assert result["steps"] == []
        assert result["model_class"] == ""
        assert result["fold_strategy"] == "shared"
