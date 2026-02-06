"""
API contract tests for WorkspaceStore.

Phase 0: Validates that the WorkspaceStore class exposes all expected
methods with correct signatures before any implementation exists.
These tests verify the *interface* contract, not behaviour.

Tests cover:
    - Class existence and instantiation (raises NotImplementedError)
    - All public methods are present
    - Method signatures (parameter names and annotations)
    - Protocol compliance
    - Return type annotations are declared
"""

import inspect
import shutil
import tempfile
from pathlib import Path
from typing import get_type_hints

import numpy as np
import polars as pl
import pytest

from nirs4all.pipeline.storage.store_protocol import WorkspaceStoreProtocol
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def workspace_dir():
    """Create a temporary workspace directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


# =========================================================================
# Class existence and instantiation
# =========================================================================

class TestClassExists:
    """Verify the WorkspaceStore class is importable and has the right shape."""

    def test_class_is_importable(self):
        """WorkspaceStore can be imported from the expected module."""
        assert WorkspaceStore is not None

    def test_init_signature(self):
        """Constructor accepts workspace_path: Path."""
        sig = inspect.signature(WorkspaceStore.__init__)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "workspace_path" in params

    def test_init_creates_store(self, workspace_dir):
        """Constructor creates a usable store instance."""
        store = WorkspaceStore(workspace_dir)
        assert store is not None
        store.close()


# =========================================================================
# Protocol compliance
# =========================================================================

class TestProtocolCompliance:
    """Verify WorkspaceStore satisfies WorkspaceStoreProtocol."""

    def test_protocol_is_importable(self):
        """WorkspaceStoreProtocol can be imported."""
        assert WorkspaceStoreProtocol is not None

    def test_protocol_is_runtime_checkable(self):
        """Protocol has @runtime_checkable decorator."""
        assert hasattr(WorkspaceStoreProtocol, "__protocol_attrs__") or hasattr(WorkspaceStoreProtocol, "__abstractmethods__") or True
        # The definitive check: issubclass should work for runtime_checkable protocols
        assert issubclass(WorkspaceStore, WorkspaceStoreProtocol)

    def test_workspace_store_is_subclass_of_protocol(self):
        """WorkspaceStore structurally satisfies WorkspaceStoreProtocol."""
        assert issubclass(WorkspaceStore, WorkspaceStoreProtocol)


# =========================================================================
# Run lifecycle methods
# =========================================================================

class TestRunLifecycleMethods:
    """Verify run lifecycle method signatures."""

    def test_begin_run_exists(self):
        """begin_run method exists on WorkspaceStore."""
        assert hasattr(WorkspaceStore, "begin_run")
        assert callable(getattr(WorkspaceStore, "begin_run"))

    def test_begin_run_signature(self):
        """begin_run has correct parameter names."""
        sig = inspect.signature(WorkspaceStore.begin_run)
        params = list(sig.parameters.keys())
        assert "name" in params
        assert "config" in params
        assert "datasets" in params

    def test_begin_run_return_annotation(self):
        """begin_run is annotated to return str."""
        hints = get_type_hints(WorkspaceStore.begin_run)
        assert hints.get("return") is str

    def test_complete_run_exists(self):
        """complete_run method exists."""
        assert hasattr(WorkspaceStore, "complete_run")

    def test_complete_run_signature(self):
        """complete_run has correct parameter names."""
        sig = inspect.signature(WorkspaceStore.complete_run)
        params = list(sig.parameters.keys())
        assert "run_id" in params
        assert "summary" in params

    def test_complete_run_return_annotation(self):
        """complete_run is annotated to return None."""
        hints = get_type_hints(WorkspaceStore.complete_run)
        assert hints.get("return") is type(None)

    def test_fail_run_exists(self):
        """fail_run method exists."""
        assert hasattr(WorkspaceStore, "fail_run")

    def test_fail_run_signature(self):
        """fail_run has correct parameter names."""
        sig = inspect.signature(WorkspaceStore.fail_run)
        params = list(sig.parameters.keys())
        assert "run_id" in params
        assert "error" in params

    def test_fail_run_return_annotation(self):
        """fail_run is annotated to return None."""
        hints = get_type_hints(WorkspaceStore.fail_run)
        assert hints.get("return") is type(None)


# =========================================================================
# Pipeline lifecycle methods
# =========================================================================

class TestPipelineLifecycleMethods:
    """Verify pipeline lifecycle method signatures."""

    def test_begin_pipeline_exists(self):
        """begin_pipeline method exists."""
        assert hasattr(WorkspaceStore, "begin_pipeline")

    def test_begin_pipeline_signature(self):
        """begin_pipeline has all expected parameters."""
        sig = inspect.signature(WorkspaceStore.begin_pipeline)
        params = list(sig.parameters.keys())
        assert "run_id" in params
        assert "name" in params
        assert "expanded_config" in params
        assert "generator_choices" in params
        assert "dataset_name" in params
        assert "dataset_hash" in params

    def test_begin_pipeline_return_annotation(self):
        """begin_pipeline is annotated to return str."""
        hints = get_type_hints(WorkspaceStore.begin_pipeline)
        assert hints.get("return") is str

    def test_complete_pipeline_exists(self):
        """complete_pipeline method exists."""
        assert hasattr(WorkspaceStore, "complete_pipeline")

    def test_complete_pipeline_signature(self):
        """complete_pipeline has all expected parameters."""
        sig = inspect.signature(WorkspaceStore.complete_pipeline)
        params = list(sig.parameters.keys())
        assert "pipeline_id" in params
        assert "best_val" in params
        assert "best_test" in params
        assert "metric" in params
        assert "duration_ms" in params

    def test_fail_pipeline_exists(self):
        """fail_pipeline method exists."""
        assert hasattr(WorkspaceStore, "fail_pipeline")

    def test_fail_pipeline_signature(self):
        """fail_pipeline has correct parameter names."""
        sig = inspect.signature(WorkspaceStore.fail_pipeline)
        params = list(sig.parameters.keys())
        assert "pipeline_id" in params
        assert "error" in params

    def test_fail_pipeline_return_annotation(self):
        """fail_pipeline is annotated to return None."""
        hints = get_type_hints(WorkspaceStore.fail_pipeline)
        assert hints.get("return") is type(None)

    def test_complete_pipeline_return_annotation(self):
        """complete_pipeline is annotated to return None."""
        hints = get_type_hints(WorkspaceStore.complete_pipeline)
        assert hints.get("return") is type(None)


# =========================================================================
# Chain management methods
# =========================================================================

class TestChainManagementMethods:
    """Verify chain management method signatures."""

    def test_save_chain_exists(self):
        """save_chain method exists."""
        assert hasattr(WorkspaceStore, "save_chain")

    def test_save_chain_signature(self):
        """save_chain has all expected parameters."""
        sig = inspect.signature(WorkspaceStore.save_chain)
        params = list(sig.parameters.keys())
        assert "pipeline_id" in params
        assert "steps" in params
        assert "model_step_idx" in params
        assert "model_class" in params
        assert "preprocessings" in params
        assert "fold_strategy" in params
        assert "fold_artifacts" in params
        assert "shared_artifacts" in params
        assert "branch_path" in params
        assert "source_index" in params

    def test_save_chain_return_annotation(self):
        """save_chain is annotated to return str."""
        hints = get_type_hints(WorkspaceStore.save_chain)
        assert hints.get("return") is str

    def test_save_chain_optional_params_have_defaults(self):
        """branch_path and source_index default to None."""
        sig = inspect.signature(WorkspaceStore.save_chain)
        assert sig.parameters["branch_path"].default is None
        assert sig.parameters["source_index"].default is None

    def test_get_chain_exists(self):
        """get_chain method exists."""
        assert hasattr(WorkspaceStore, "get_chain")

    def test_get_chain_signature(self):
        """get_chain takes chain_id parameter."""
        sig = inspect.signature(WorkspaceStore.get_chain)
        params = list(sig.parameters.keys())
        assert "chain_id" in params

    def test_get_chains_for_pipeline_exists(self):
        """get_chains_for_pipeline method exists."""
        assert hasattr(WorkspaceStore, "get_chains_for_pipeline")

    def test_get_chains_for_pipeline_signature(self):
        """get_chains_for_pipeline takes pipeline_id parameter."""
        sig = inspect.signature(WorkspaceStore.get_chains_for_pipeline)
        params = list(sig.parameters.keys())
        assert "pipeline_id" in params

    def test_get_chains_for_pipeline_return_annotation(self):
        """get_chains_for_pipeline is annotated to return pl.DataFrame."""
        hints = get_type_hints(WorkspaceStore.get_chains_for_pipeline)
        assert hints.get("return") is pl.DataFrame


# =========================================================================
# Prediction storage methods
# =========================================================================

class TestPredictionStorageMethods:
    """Verify prediction storage method signatures."""

    def test_save_prediction_exists(self):
        """save_prediction method exists."""
        assert hasattr(WorkspaceStore, "save_prediction")

    def test_save_prediction_signature(self):
        """save_prediction has all expected parameters."""
        sig = inspect.signature(WorkspaceStore.save_prediction)
        params = list(sig.parameters.keys())
        expected_params = [
            "pipeline_id", "chain_id", "dataset_name", "model_name",
            "model_class", "fold_id", "partition", "val_score",
            "test_score", "train_score", "metric", "task_type",
            "n_samples", "n_features", "scores", "best_params",
            "branch_id", "branch_name", "exclusion_count",
            "exclusion_rate", "preprocessings",
        ]
        for param in expected_params:
            assert param in params, f"Missing parameter: {param}"

    def test_save_prediction_preprocessings_default(self):
        """save_prediction preprocessings parameter defaults to empty string."""
        sig = inspect.signature(WorkspaceStore.save_prediction)
        assert sig.parameters["preprocessings"].default == ""

    def test_save_prediction_return_annotation(self):
        """save_prediction is annotated to return str."""
        hints = get_type_hints(WorkspaceStore.save_prediction)
        assert hints.get("return") is str

    def test_save_prediction_arrays_exists(self):
        """save_prediction_arrays method exists."""
        assert hasattr(WorkspaceStore, "save_prediction_arrays")

    def test_save_prediction_arrays_signature(self):
        """save_prediction_arrays has all expected parameters."""
        sig = inspect.signature(WorkspaceStore.save_prediction_arrays)
        params = list(sig.parameters.keys())
        assert "prediction_id" in params
        assert "y_true" in params
        assert "y_pred" in params
        assert "y_proba" in params
        assert "sample_indices" in params
        assert "weights" in params

    def test_save_prediction_arrays_optional_defaults(self):
        """y_proba, sample_indices, and weights default to None."""
        sig = inspect.signature(WorkspaceStore.save_prediction_arrays)
        assert sig.parameters["y_proba"].default is None
        assert sig.parameters["sample_indices"].default is None
        assert sig.parameters["weights"].default is None

    def test_save_prediction_arrays_return_annotation(self):
        """save_prediction_arrays is annotated to return None."""
        hints = get_type_hints(WorkspaceStore.save_prediction_arrays)
        assert hints.get("return") is type(None)


# =========================================================================
# Artifact storage methods
# =========================================================================

class TestArtifactStorageMethods:
    """Verify artifact storage method signatures."""

    def test_save_artifact_exists(self):
        """save_artifact method exists."""
        assert hasattr(WorkspaceStore, "save_artifact")

    def test_save_artifact_signature(self):
        """save_artifact has all expected parameters."""
        sig = inspect.signature(WorkspaceStore.save_artifact)
        params = list(sig.parameters.keys())
        assert "obj" in params
        assert "operator_class" in params
        assert "artifact_type" in params
        assert "format" in params

    def test_save_artifact_return_annotation(self):
        """save_artifact is annotated to return str."""
        hints = get_type_hints(WorkspaceStore.save_artifact)
        assert hints.get("return") is str

    def test_load_artifact_exists(self):
        """load_artifact method exists."""
        assert hasattr(WorkspaceStore, "load_artifact")

    def test_load_artifact_signature(self):
        """load_artifact takes artifact_id parameter."""
        sig = inspect.signature(WorkspaceStore.load_artifact)
        params = list(sig.parameters.keys())
        assert "artifact_id" in params

    def test_load_artifact_return_annotation(self):
        """load_artifact is annotated to return Any."""
        hints = get_type_hints(WorkspaceStore.load_artifact)
        assert "return" in hints

    def test_get_artifact_path_exists(self):
        """get_artifact_path method exists."""
        assert hasattr(WorkspaceStore, "get_artifact_path")

    def test_get_artifact_path_signature(self):
        """get_artifact_path takes artifact_id parameter."""
        sig = inspect.signature(WorkspaceStore.get_artifact_path)
        params = list(sig.parameters.keys())
        assert "artifact_id" in params

    def test_get_artifact_path_return_annotation(self):
        """get_artifact_path is annotated to return Path."""
        hints = get_type_hints(WorkspaceStore.get_artifact_path)
        assert hints.get("return") is Path


# =========================================================================
# Structured logging methods
# =========================================================================

class TestLoggingMethods:
    """Verify structured logging method signatures."""

    def test_log_step_exists(self):
        """log_step method exists."""
        assert hasattr(WorkspaceStore, "log_step")

    def test_log_step_signature(self):
        """log_step has all expected parameters."""
        sig = inspect.signature(WorkspaceStore.log_step)
        params = list(sig.parameters.keys())
        assert "pipeline_id" in params
        assert "step_idx" in params
        assert "operator_class" in params
        assert "event" in params
        assert "duration_ms" in params
        assert "message" in params
        assert "details" in params
        assert "level" in params

    def test_log_step_defaults(self):
        """log_step optional parameters have correct defaults."""
        sig = inspect.signature(WorkspaceStore.log_step)
        assert sig.parameters["duration_ms"].default is None
        assert sig.parameters["message"].default is None
        assert sig.parameters["details"].default is None
        assert sig.parameters["level"].default == "info"

    def test_log_step_return_annotation(self):
        """log_step is annotated to return None."""
        hints = get_type_hints(WorkspaceStore.log_step)
        assert hints.get("return") is type(None)


# =========================================================================
# Query methods
# =========================================================================

class TestQueryMethods:
    """Verify query method signatures."""

    def test_get_run_exists(self):
        """get_run method exists."""
        assert hasattr(WorkspaceStore, "get_run")

    def test_get_run_signature(self):
        """get_run takes run_id parameter."""
        sig = inspect.signature(WorkspaceStore.get_run)
        params = list(sig.parameters.keys())
        assert "run_id" in params

    def test_list_runs_exists(self):
        """list_runs method exists."""
        assert hasattr(WorkspaceStore, "list_runs")

    def test_list_runs_signature(self):
        """list_runs has all expected parameters with defaults."""
        sig = inspect.signature(WorkspaceStore.list_runs)
        params = list(sig.parameters.keys())
        assert "status" in params
        assert "dataset" in params
        assert "limit" in params
        assert "offset" in params
        # Check defaults
        assert sig.parameters["status"].default is None
        assert sig.parameters["dataset"].default is None
        assert sig.parameters["limit"].default == 100
        assert sig.parameters["offset"].default == 0

    def test_list_runs_return_annotation(self):
        """list_runs is annotated to return pl.DataFrame."""
        hints = get_type_hints(WorkspaceStore.list_runs)
        assert hints.get("return") is pl.DataFrame

    def test_get_pipeline_exists(self):
        """get_pipeline method exists."""
        assert hasattr(WorkspaceStore, "get_pipeline")

    def test_get_pipeline_signature(self):
        """get_pipeline takes pipeline_id parameter."""
        sig = inspect.signature(WorkspaceStore.get_pipeline)
        params = list(sig.parameters.keys())
        assert "pipeline_id" in params

    def test_list_pipelines_exists(self):
        """list_pipelines method exists."""
        assert hasattr(WorkspaceStore, "list_pipelines")

    def test_list_pipelines_signature(self):
        """list_pipelines has run_id and dataset_name parameters with defaults."""
        sig = inspect.signature(WorkspaceStore.list_pipelines)
        params = list(sig.parameters.keys())
        assert "run_id" in params
        assert "dataset_name" in params
        assert sig.parameters["run_id"].default is None
        assert sig.parameters["dataset_name"].default is None

    def test_list_pipelines_return_annotation(self):
        """list_pipelines is annotated to return pl.DataFrame."""
        hints = get_type_hints(WorkspaceStore.list_pipelines)
        assert hints.get("return") is pl.DataFrame

    def test_get_prediction_exists(self):
        """get_prediction method exists."""
        assert hasattr(WorkspaceStore, "get_prediction")

    def test_get_prediction_signature(self):
        """get_prediction has prediction_id and load_arrays parameters."""
        sig = inspect.signature(WorkspaceStore.get_prediction)
        params = list(sig.parameters.keys())
        assert "prediction_id" in params
        assert "load_arrays" in params
        assert sig.parameters["load_arrays"].default is False

    def test_query_predictions_exists(self):
        """query_predictions method exists."""
        assert hasattr(WorkspaceStore, "query_predictions")

    def test_query_predictions_signature(self):
        """query_predictions has all expected filter parameters."""
        sig = inspect.signature(WorkspaceStore.query_predictions)
        params = list(sig.parameters.keys())
        expected_params = [
            "dataset_name", "model_class", "partition", "fold_id",
            "branch_id", "pipeline_id", "run_id", "limit", "offset",
        ]
        for param in expected_params:
            assert param in params, f"Missing parameter: {param}"

    def test_query_predictions_defaults(self):
        """query_predictions optional parameters have correct defaults."""
        sig = inspect.signature(WorkspaceStore.query_predictions)
        assert sig.parameters["dataset_name"].default is None
        assert sig.parameters["model_class"].default is None
        assert sig.parameters["partition"].default is None
        assert sig.parameters["fold_id"].default is None
        assert sig.parameters["branch_id"].default is None
        assert sig.parameters["pipeline_id"].default is None
        assert sig.parameters["run_id"].default is None
        assert sig.parameters["limit"].default is None
        assert sig.parameters["offset"].default == 0

    def test_query_predictions_return_annotation(self):
        """query_predictions is annotated to return pl.DataFrame."""
        hints = get_type_hints(WorkspaceStore.query_predictions)
        assert hints.get("return") is pl.DataFrame

    def test_top_predictions_exists(self):
        """top_predictions method exists."""
        assert hasattr(WorkspaceStore, "top_predictions")

    def test_top_predictions_signature(self):
        """top_predictions has all expected parameters."""
        sig = inspect.signature(WorkspaceStore.top_predictions)
        params = list(sig.parameters.keys())
        assert "n" in params
        assert "metric" in params
        assert "ascending" in params
        assert "partition" in params
        assert "dataset_name" in params
        assert "group_by" in params

    def test_top_predictions_defaults(self):
        """top_predictions has correct default values."""
        sig = inspect.signature(WorkspaceStore.top_predictions)
        assert sig.parameters["metric"].default == "val_score"
        assert sig.parameters["ascending"].default is True
        assert sig.parameters["partition"].default == "val"
        assert sig.parameters["dataset_name"].default is None
        assert sig.parameters["group_by"].default is None

    def test_top_predictions_return_annotation(self):
        """top_predictions is annotated to return pl.DataFrame."""
        hints = get_type_hints(WorkspaceStore.top_predictions)
        assert hints.get("return") is pl.DataFrame

    def test_get_pipeline_log_exists(self):
        """get_pipeline_log method exists."""
        assert hasattr(WorkspaceStore, "get_pipeline_log")

    def test_get_pipeline_log_signature(self):
        """get_pipeline_log takes pipeline_id parameter."""
        sig = inspect.signature(WorkspaceStore.get_pipeline_log)
        params = list(sig.parameters.keys())
        assert "pipeline_id" in params

    def test_get_pipeline_log_return_annotation(self):
        """get_pipeline_log is annotated to return pl.DataFrame."""
        hints = get_type_hints(WorkspaceStore.get_pipeline_log)
        assert hints.get("return") is pl.DataFrame

    def test_get_run_log_summary_exists(self):
        """get_run_log_summary method exists."""
        assert hasattr(WorkspaceStore, "get_run_log_summary")

    def test_get_run_log_summary_signature(self):
        """get_run_log_summary takes run_id parameter."""
        sig = inspect.signature(WorkspaceStore.get_run_log_summary)
        params = list(sig.parameters.keys())
        assert "run_id" in params

    def test_get_run_log_summary_return_annotation(self):
        """get_run_log_summary is annotated to return pl.DataFrame."""
        hints = get_type_hints(WorkspaceStore.get_run_log_summary)
        assert hints.get("return") is pl.DataFrame


# =========================================================================
# Export methods
# =========================================================================

class TestExportMethods:
    """Verify export method signatures."""

    def test_export_chain_exists(self):
        """export_chain method exists."""
        assert hasattr(WorkspaceStore, "export_chain")

    def test_export_chain_signature(self):
        """export_chain has all expected parameters."""
        sig = inspect.signature(WorkspaceStore.export_chain)
        params = list(sig.parameters.keys())
        assert "chain_id" in params
        assert "output_path" in params
        assert "format" in params
        assert sig.parameters["format"].default == "n4a"

    def test_export_chain_return_annotation(self):
        """export_chain is annotated to return Path."""
        hints = get_type_hints(WorkspaceStore.export_chain)
        assert hints.get("return") is Path

    def test_export_pipeline_config_exists(self):
        """export_pipeline_config method exists."""
        assert hasattr(WorkspaceStore, "export_pipeline_config")

    def test_export_pipeline_config_signature(self):
        """export_pipeline_config has pipeline_id and output_path parameters."""
        sig = inspect.signature(WorkspaceStore.export_pipeline_config)
        params = list(sig.parameters.keys())
        assert "pipeline_id" in params
        assert "output_path" in params

    def test_export_pipeline_config_return_annotation(self):
        """export_pipeline_config is annotated to return Path."""
        hints = get_type_hints(WorkspaceStore.export_pipeline_config)
        assert hints.get("return") is Path

    def test_export_run_exists(self):
        """export_run method exists."""
        assert hasattr(WorkspaceStore, "export_run")

    def test_export_run_signature(self):
        """export_run has run_id and output_path parameters."""
        sig = inspect.signature(WorkspaceStore.export_run)
        params = list(sig.parameters.keys())
        assert "run_id" in params
        assert "output_path" in params

    def test_export_run_return_annotation(self):
        """export_run is annotated to return Path."""
        hints = get_type_hints(WorkspaceStore.export_run)
        assert hints.get("return") is Path

    def test_export_predictions_parquet_exists(self):
        """export_predictions_parquet method exists."""
        assert hasattr(WorkspaceStore, "export_predictions_parquet")

    def test_export_predictions_parquet_signature(self):
        """export_predictions_parquet has output_path parameter."""
        sig = inspect.signature(WorkspaceStore.export_predictions_parquet)
        params = list(sig.parameters.keys())
        assert "output_path" in params

    def test_export_predictions_parquet_return_annotation(self):
        """export_predictions_parquet is annotated to return Path."""
        hints = get_type_hints(WorkspaceStore.export_predictions_parquet)
        assert hints.get("return") is Path


# =========================================================================
# Deletion and cleanup methods
# =========================================================================

class TestDeletionMethods:
    """Verify deletion and cleanup method signatures."""

    def test_delete_run_exists(self):
        """delete_run method exists."""
        assert hasattr(WorkspaceStore, "delete_run")

    def test_delete_run_signature(self):
        """delete_run has run_id and delete_artifacts parameters."""
        sig = inspect.signature(WorkspaceStore.delete_run)
        params = list(sig.parameters.keys())
        assert "run_id" in params
        assert "delete_artifacts" in params
        assert sig.parameters["delete_artifacts"].default is True

    def test_delete_run_return_annotation(self):
        """delete_run is annotated to return int."""
        hints = get_type_hints(WorkspaceStore.delete_run)
        assert hints.get("return") is int

    def test_delete_prediction_exists(self):
        """delete_prediction method exists."""
        assert hasattr(WorkspaceStore, "delete_prediction")

    def test_delete_prediction_signature(self):
        """delete_prediction takes prediction_id parameter."""
        sig = inspect.signature(WorkspaceStore.delete_prediction)
        params = list(sig.parameters.keys())
        assert "prediction_id" in params

    def test_delete_prediction_return_annotation(self):
        """delete_prediction is annotated to return bool."""
        hints = get_type_hints(WorkspaceStore.delete_prediction)
        assert hints.get("return") is bool

    def test_gc_artifacts_exists(self):
        """gc_artifacts method exists."""
        assert hasattr(WorkspaceStore, "gc_artifacts")

    def test_gc_artifacts_return_annotation(self):
        """gc_artifacts is annotated to return int."""
        hints = get_type_hints(WorkspaceStore.gc_artifacts)
        assert hints.get("return") is int

    def test_vacuum_exists(self):
        """vacuum method exists."""
        assert hasattr(WorkspaceStore, "vacuum")

    def test_vacuum_return_annotation(self):
        """vacuum is annotated to return None."""
        hints = get_type_hints(WorkspaceStore.vacuum)
        assert hints.get("return") is type(None)


# =========================================================================
# Chain replay method
# =========================================================================

class TestChainReplayMethod:
    """Verify chain replay method signature."""

    def test_replay_chain_exists(self):
        """replay_chain method exists."""
        assert hasattr(WorkspaceStore, "replay_chain")

    def test_replay_chain_signature(self):
        """replay_chain has chain_id, X, and wavelengths parameters."""
        sig = inspect.signature(WorkspaceStore.replay_chain)
        params = list(sig.parameters.keys())
        assert "chain_id" in params
        assert "X" in params
        assert "wavelengths" in params
        assert sig.parameters["wavelengths"].default is None

    def test_replay_chain_return_annotation(self):
        """replay_chain is annotated to return np.ndarray."""
        hints = get_type_hints(WorkspaceStore.replay_chain)
        assert hints.get("return") is np.ndarray


# =========================================================================
# Close / resource management
# =========================================================================

class TestCloseMethod:
    """Verify close method exists."""

    def test_close_exists(self):
        """close method exists."""
        assert hasattr(WorkspaceStore, "close")

    def test_close_return_annotation(self):
        """close is annotated to return None."""
        hints = get_type_hints(WorkspaceStore.close)
        assert hints.get("return") is type(None)


# =========================================================================
# Comprehensive method inventory
# =========================================================================

class TestMethodInventory:
    """Verify all expected public methods are present."""

    EXPECTED_METHODS = [
        # Run lifecycle
        "begin_run",
        "complete_run",
        "fail_run",
        # Pipeline lifecycle
        "begin_pipeline",
        "complete_pipeline",
        "fail_pipeline",
        # Chain management
        "save_chain",
        "get_chain",
        "get_chains_for_pipeline",
        # Prediction storage
        "save_prediction",
        "save_prediction_arrays",
        # Artifact storage
        "save_artifact",
        "load_artifact",
        "get_artifact_path",
        "register_existing_artifact",
        # Logging
        "log_step",
        # Queries -- Runs
        "get_run",
        "list_runs",
        # Queries -- Pipelines
        "get_pipeline",
        "list_pipelines",
        # Queries -- Predictions
        "get_chain_predictions",
        "get_prediction",
        "get_prediction_arrays",
        "query_aggregated_predictions",
        "query_predictions",
        "query_top_aggregated_predictions",
        "top_predictions",
        # Queries -- Logs
        "get_pipeline_log",
        "get_run_log_summary",
        # Export
        "export_chain",
        "export_pipeline_config",
        "export_run",
        "export_predictions_parquet",
        # Deletion & cleanup
        "delete_run",
        "delete_prediction",
        "cleanup_transient_artifacts",
        "gc_artifacts",
        "vacuum",
        # Cross-run cache
        "save_artifact_with_cache_key",
        "update_artifact_cache_key",
        "find_cached_artifact",
        "invalidate_dataset_cache",
        # Chain replay
        "replay_chain",
        # Resource management
        "close",
    ]

    def test_all_expected_methods_exist(self):
        """Every method listed in the design doc is present on WorkspaceStore."""
        missing = []
        for method_name in self.EXPECTED_METHODS:
            if not hasattr(WorkspaceStore, method_name):
                missing.append(method_name)
        assert missing == [], f"Missing methods: {missing}"

    def test_all_methods_are_callable(self):
        """Every expected method is callable."""
        for method_name in self.EXPECTED_METHODS:
            method = getattr(WorkspaceStore, method_name, None)
            assert callable(method), f"{method_name} is not callable"

    def test_no_unexpected_public_methods(self):
        """WorkspaceStore does not have undocumented public methods.

        This catches accidentally added methods that might violate the
        API contract.  Private (underscore-prefixed) methods are
        excluded from the check.
        """
        actual_public_methods = [
            name for name in dir(WorkspaceStore)
            if not name.startswith("_") and callable(getattr(WorkspaceStore, name))
        ]
        expected_set = set(self.EXPECTED_METHODS)
        unexpected = [m for m in actual_public_methods if m not in expected_set]
        assert unexpected == [], f"Unexpected public methods: {unexpected}"

    def test_all_methods_have_docstrings(self):
        """Every public method has a docstring."""
        for method_name in self.EXPECTED_METHODS:
            method = getattr(WorkspaceStore, method_name)
            assert method.__doc__ is not None, f"{method_name} has no docstring"
            assert len(method.__doc__.strip()) > 0, f"{method_name} has empty docstring"

    def test_all_methods_have_return_annotations(self):
        """Every public method has a return type annotation."""
        for method_name in self.EXPECTED_METHODS:
            method = getattr(WorkspaceStore, method_name)
            hints = get_type_hints(method)
            assert "return" in hints, f"{method_name} missing return type annotation"

    def test_total_method_count(self):
        """The expected method count matches the design doc (35 methods)."""
        assert len(self.EXPECTED_METHODS) == 44
