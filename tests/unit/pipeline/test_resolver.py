"""
Tests for the PredictionResolver module (Phase 3).

Tests the PredictionResolver class for resolving various prediction sources
to executable components.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from nirs4all.pipeline.resolver import (
    PredictionResolver,
    ResolvedPrediction,
    SourceType,
    FoldStrategy,
)


class TestSourceType:
    """Tests for SourceType enum."""

    def test_values(self):
        """Test enum values."""
        assert SourceType.PREDICTION.value == "prediction"
        assert SourceType.FOLDER.value == "folder"
        assert SourceType.RUN.value == "run"
        assert SourceType.ARTIFACT_ID.value == "artifact_id"
        assert SourceType.BUNDLE.value == "bundle"
        assert SourceType.TRACE_ID.value == "trace_id"
        assert SourceType.UNKNOWN.value == "unknown"

    def test_str(self):
        """Test string conversion."""
        assert str(SourceType.PREDICTION) == "prediction"
        assert str(SourceType.BUNDLE) == "bundle"


class TestFoldStrategy:
    """Tests for FoldStrategy enum."""

    def test_values(self):
        """Test enum values."""
        assert FoldStrategy.AVERAGE.value == "average"
        assert FoldStrategy.WEIGHTED_AVERAGE.value == "weighted_average"
        assert FoldStrategy.SINGLE.value == "single"


class TestResolvedPrediction:
    """Tests for ResolvedPrediction dataclass."""

    def test_create_empty(self):
        """Test creating empty resolved prediction."""
        resolved = ResolvedPrediction()

        assert resolved.source_type == SourceType.UNKNOWN
        assert resolved.minimal_pipeline == []
        assert resolved.artifact_provider is None
        assert resolved.trace is None
        assert resolved.fold_strategy == FoldStrategy.WEIGHTED_AVERAGE
        assert resolved.fold_weights == {}

    def test_has_trace(self):
        """Test checking for trace."""
        resolved = ResolvedPrediction()
        assert resolved.has_trace() is False

        from nirs4all.pipeline.trace import ExecutionTrace
        resolved.trace = ExecutionTrace()
        assert resolved.has_trace() is True

    def test_has_fold_artifacts(self):
        """Test checking for fold artifacts."""
        resolved = ResolvedPrediction()
        assert resolved.has_fold_artifacts() is False

        resolved.fold_weights = {0: 0.5}
        assert resolved.has_fold_artifacts() is False  # Only 1 fold

        resolved.fold_weights = {0: 0.5, 1: 0.5}
        assert resolved.has_fold_artifacts() is True

    def test_get_preprocessing_chain(self):
        """Test getting preprocessing chain."""
        resolved = ResolvedPrediction()
        assert resolved.get_preprocessing_chain() == ""

        from nirs4all.pipeline.trace import ExecutionTrace
        trace = ExecutionTrace(preprocessing_chain="SNV>SG")
        resolved.trace = trace
        assert resolved.get_preprocessing_chain() == "SNV>SG"


class TestPredictionResolver:
    """Tests for PredictionResolver class."""

    @pytest.fixture
    def mock_workspace(self, tmp_path):
        """Create a mock workspace directory."""
        workspace = tmp_path / "workspace"
        runs_dir = workspace / "runs"
        runs_dir.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def resolver(self, mock_workspace):
        """Create a resolver with mock workspace."""
        return PredictionResolver(mock_workspace)

    def test_init(self, resolver, mock_workspace):
        """Test resolver initialization."""
        assert resolver.workspace_path == mock_workspace
        assert resolver.runs_dir == mock_workspace / "runs"

    def test_init_custom_runs_dir(self, mock_workspace, tmp_path):
        """Test resolver with custom runs directory."""
        custom_runs = tmp_path / "custom_runs"
        custom_runs.mkdir()

        resolver = PredictionResolver(mock_workspace, runs_dir=custom_runs)
        assert resolver.runs_dir == custom_runs

    def test_detect_source_type_prediction_dict(self, resolver):
        """Test detecting prediction dict source."""
        source = {
            "id": "abc123",
            "pipeline_uid": "test_pipeline",
            "model_name": "PLS"
        }

        source_type = resolver._detect_source_type(source)
        assert source_type == SourceType.PREDICTION

    def test_detect_source_type_artifact_id(self, resolver):
        """Test detecting artifact ID source."""
        # Standard format
        assert resolver._detect_source_type("0001:4:all") == SourceType.ARTIFACT_ID
        assert resolver._detect_source_type("abc123:2:0") == SourceType.ARTIFACT_ID

        # With branch
        assert resolver._detect_source_type("0001:0:4:all") == SourceType.ARTIFACT_ID

    def test_detect_source_type_bundle(self, resolver):
        """Test detecting bundle source."""
        assert resolver._detect_source_type("model.n4a") == SourceType.BUNDLE
        assert resolver._detect_source_type("/path/to/model.n4a") == SourceType.BUNDLE
        assert resolver._detect_source_type("model.n4a.py") == SourceType.BUNDLE

    def test_detect_source_type_trace_id(self, resolver):
        """Test detecting trace ID source."""
        assert resolver._detect_source_type("trace:xyz789") == SourceType.TRACE_ID
        assert resolver._detect_source_type("trace:abc123def") == SourceType.TRACE_ID

    def test_detect_source_type_folder(self, resolver, tmp_path):
        """Test detecting folder source."""
        folder = tmp_path / "test_folder"
        folder.mkdir()

        source_type = resolver._detect_source_type(str(folder))
        assert source_type == SourceType.FOLDER

    def test_detect_source_type_run(self, resolver):
        """Test detecting Run object source."""
        class MockRun:
            predictions = MagicMock()
            def best(self):
                return {"id": "123"}

        source_type = resolver._detect_source_type(MockRun())
        assert source_type == SourceType.RUN

    def test_detect_source_type_unknown(self, resolver):
        """Test detecting unknown source."""
        assert resolver._detect_source_type(12345) == SourceType.UNKNOWN
        assert resolver._detect_source_type(None) == SourceType.UNKNOWN

    def test_looks_like_artifact_id_valid(self, resolver):
        """Test artifact ID pattern matching - valid."""
        assert resolver._looks_like_artifact_id("0001:4:all") is True
        assert resolver._looks_like_artifact_id("abc:2:0") is True
        assert resolver._looks_like_artifact_id("pipeline:0:4:1") is True

    def test_looks_like_artifact_id_invalid(self, resolver):
        """Test artifact ID pattern matching - invalid."""
        assert resolver._looks_like_artifact_id("0001:4") is False
        assert resolver._looks_like_artifact_id("not_an_artifact") is False
        assert resolver._looks_like_artifact_id("0001:4:invalid") is False

    def test_resolve_unknown_raises(self, resolver):
        """Test that resolving unknown source raises error."""
        with pytest.raises(ValueError, match="Cannot resolve prediction source"):
            resolver.resolve(12345)

    def test_resolve_from_prediction_missing_uid(self, resolver):
        """Test error when prediction has no pipeline_uid."""
        source = {"id": "abc123"}  # Missing pipeline_uid

        with pytest.raises(ValueError, match="no pipeline_uid"):
            resolver._resolve_from_prediction(source)

    def test_resolve_from_bundle_not_found(self, resolver, tmp_path):
        """Test error when bundle file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Bundle not found"):
            resolver._resolve_from_bundle("nonexistent.n4a")

    def test_resolve_from_trace_id_invalid(self, resolver):
        """Test error with invalid trace reference."""
        with pytest.raises(ValueError, match="Invalid trace reference"):
            resolver._resolve_from_trace_id("invalid_trace")


class TestPredictionResolverIntegration:
    """Integration tests for PredictionResolver with real files."""

    @pytest.fixture
    def setup_mock_run(self, tmp_path):
        """Set up a mock run directory with manifest."""
        import json
        import yaml

        workspace = tmp_path / "workspace"
        runs_dir = workspace / "runs"
        run_dir = runs_dir / "2024-12-14_test"
        pipeline_dir = run_dir / "test_pipeline_abc123"

        pipeline_dir.mkdir(parents=True)

        # Create manifest.yaml
        manifest = {
            "pipeline_uid": "test_pipeline_abc123",
            "dataset": "test",
            "artifacts": {
                "items": [
                    {
                        "artifact_id": "test:1:all",
                        "step_index": 1,
                        "artifact_type": "transformer"
                    }
                ]
            }
        }
        with open(pipeline_dir / "manifest.yaml", "w") as f:
            yaml.dump(manifest, f)

        # Create pipeline.json
        pipeline_config = {
            "steps": [{"transform": "MinMaxScaler"}]
        }
        with open(pipeline_dir / "pipeline.json", "w") as f:
            json.dump(pipeline_config, f)

        return {
            "workspace": workspace,
            "runs_dir": runs_dir,
            "run_dir": run_dir,
            "pipeline_dir": pipeline_dir,
            "pipeline_uid": "test_pipeline_abc123"
        }

    def test_resolve_from_folder(self, setup_mock_run):
        """Test resolving from folder path."""
        resolver = PredictionResolver(
            setup_mock_run["workspace"],
            runs_dir=setup_mock_run["runs_dir"]
        )

        resolved = resolver.resolve(str(setup_mock_run["pipeline_dir"]))

        assert resolved.source_type == SourceType.FOLDER
        assert resolved.pipeline_uid == "test_pipeline_abc123"
        assert len(resolved.minimal_pipeline) > 0

    def test_resolve_from_prediction_dict(self, setup_mock_run):
        """Test resolving from prediction dictionary."""
        resolver = PredictionResolver(
            setup_mock_run["workspace"],
            runs_dir=setup_mock_run["runs_dir"]
        )

        prediction = {
            "id": "test123",
            "pipeline_uid": "test_pipeline_abc123",
            "model_name": "TestModel",
            "run_dir": str(setup_mock_run["run_dir"])
        }

        resolved = resolver.resolve(prediction)

        assert resolved.source_type == SourceType.PREDICTION
        assert resolved.pipeline_uid == "test_pipeline_abc123"


class TestModelFileResolution:
    """Tests for MODEL_FILE source type resolution."""

    @pytest.fixture
    def mock_workspace(self, tmp_path):
        """Create a mock workspace directory."""
        workspace = tmp_path / "workspace"
        runs_dir = workspace / "runs"
        runs_dir.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def resolver(self, mock_workspace):
        """Create a resolver with mock workspace."""
        return PredictionResolver(mock_workspace)

    @pytest.fixture
    def sklearn_model_file(self, tmp_path):
        """Create a sklearn model saved to file."""
        import joblib
        from sklearn.cross_decomposition import PLSRegression
        import numpy as np

        model = PLSRegression(n_components=3)
        X = np.random.randn(20, 10)
        y = np.random.randn(20)
        model.fit(X, y)

        model_path = tmp_path / "pls_model.joblib"
        joblib.dump(model, model_path)
        return model_path

    def test_detect_source_type_model_file_joblib(self, resolver, sklearn_model_file):
        """Test detecting .joblib file as MODEL_FILE."""
        source_type = resolver._detect_source_type(str(sklearn_model_file))
        assert source_type == SourceType.MODEL_FILE

    def test_detect_source_type_model_file_pkl(self, resolver, tmp_path):
        """Test detecting .pkl file as MODEL_FILE."""
        import cloudpickle
        from sklearn.linear_model import Ridge

        model_path = tmp_path / "model.pkl"
        with open(model_path, 'wb') as f:
            cloudpickle.dump(Ridge(), f)

        source_type = resolver._detect_source_type(str(model_path))
        assert source_type == SourceType.MODEL_FILE

    def test_detect_source_type_model_file_h5(self, resolver, tmp_path):
        """Test detecting .h5 file as MODEL_FILE."""
        pytest.importorskip("tensorflow")
        from tensorflow import keras

        model = keras.Sequential([keras.layers.Dense(1, input_shape=(5,))])
        model_path = tmp_path / "model.h5"
        model.save(str(model_path))

        source_type = resolver._detect_source_type(str(model_path))
        assert source_type == SourceType.MODEL_FILE

    def test_detect_source_type_model_file_pt(self, resolver, tmp_path):
        """Test detecting .pt file as MODEL_FILE."""
        torch = pytest.importorskip("torch")

        model_path = tmp_path / "model.pt"
        torch.save({'dummy': 'model'}, model_path)

        source_type = resolver._detect_source_type(str(model_path))
        assert source_type == SourceType.MODEL_FILE

    def test_is_model_folder_autogluon(self, resolver, tmp_path):
        """Test detecting AutoGluon model folder."""
        folder = tmp_path / "ag_model"
        folder.mkdir()
        (folder / "predictor.pkl").write_bytes(b"dummy")

        assert resolver._is_model_folder(folder) is True

    def test_is_model_folder_tensorflow_savedmodel(self, resolver, tmp_path):
        """Test detecting TensorFlow SavedModel folder."""
        folder = tmp_path / "tf_model"
        folder.mkdir()
        (folder / "saved_model.pb").write_bytes(b"dummy")

        assert resolver._is_model_folder(folder) is True

    def test_is_model_folder_pipeline_folder(self, resolver, tmp_path):
        """Test that pipeline folders are not detected as model folders."""
        folder = tmp_path / "pipeline_folder"
        folder.mkdir()
        (folder / "manifest.yaml").write_text("pipeline: true")

        assert resolver._is_model_folder(folder) is False

    def test_resolve_from_model_file(self, resolver, sklearn_model_file):
        """Test resolving from model file."""
        resolved = resolver.resolve(str(sklearn_model_file))

        assert resolved.source_type == SourceType.MODEL_FILE
        assert resolved.model_step_index == 0
        assert resolved.fold_strategy == FoldStrategy.SINGLE
        assert 0 in resolved.fold_weights
        assert resolved.artifact_provider is not None

        # Check that model is in artifact provider
        artifacts = resolved.artifact_provider.get_artifacts_for_step(0)
        assert len(artifacts) == 1
        artifact_id, model = artifacts[0]
        # artifact_id is a string like "model_file:pls_model:0:0"
        assert isinstance(artifact_id, str)
        assert "pls_model" in artifact_id
        assert hasattr(model, 'predict')

    def test_resolve_model_file_pipeline_uid(self, resolver, sklearn_model_file):
        """Test that resolved model file has appropriate pipeline_uid."""
        resolved = resolver.resolve(str(sklearn_model_file))

        assert "pls_model" in resolved.pipeline_uid
        assert resolved.pipeline_uid.startswith("model_file_")

    def test_detect_model_framework_sklearn(self, resolver):
        """Test framework detection for sklearn models."""
        from sklearn.linear_model import Ridge
        model = Ridge()

        framework = resolver._detect_model_framework(model)
        assert framework == 'sklearn'

    def test_detect_model_framework_tensorflow(self, resolver):
        """Test framework detection for TensorFlow models."""
        pytest.importorskip("tensorflow")
        from tensorflow import keras

        model = keras.Sequential([keras.layers.Dense(1)])

        framework = resolver._detect_model_framework(model)
        assert framework == 'tensorflow'

    def test_detect_model_framework_pytorch(self, resolver):
        """Test framework detection for PyTorch models."""
        torch = pytest.importorskip("torch")

        model = torch.nn.Linear(5, 1)

        framework = resolver._detect_model_framework(model)
        assert framework == 'pytorch'
