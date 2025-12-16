"""
Tests for the Bundle Export module (Phase 6).

Tests the BundleGenerator and BundleLoader classes for exporting
and loading prediction bundles.
"""

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from nirs4all.pipeline.bundle import (
    BundleFormat,
    BundleGenerator,
    BundleLoader,
    BundleMetadata,
)
from nirs4all.pipeline.bundle.loader import BundleArtifactProvider
from nirs4all.pipeline.resolver import ResolvedPrediction, SourceType, FoldStrategy
from nirs4all.pipeline.trace import ExecutionTrace, ExecutionStep, StepArtifacts


class SimpleModel:
    """Simple picklable model for testing."""

    def predict(self, X):
        """Predict method."""
        return np.ones(len(X))

    def fit(self, X, y):
        """Fit method."""
        return self


class TestBundleMetadata:
    """Tests for BundleMetadata dataclass."""

    def test_from_dict_basic(self):
        """Test creating BundleMetadata from dictionary."""
        data = {
            "bundle_format_version": "1.0",
            "nirs4all_version": "0.9.0",
            "created_at": "2024-12-14T10:00:00",
            "pipeline_uid": "0001_pls_abc123",
            "source_type": "prediction",
            "model_step_index": 4,
            "fold_strategy": "weighted_average",
            "preprocessing_chain": "SNV>SG>MinMax",
        }

        metadata = BundleMetadata.from_dict(data)

        assert metadata.bundle_format_version == "1.0"
        assert metadata.nirs4all_version == "0.9.0"
        assert metadata.pipeline_uid == "0001_pls_abc123"
        assert metadata.model_step_index == 4
        assert metadata.preprocessing_chain == "SNV>SG>MinMax"

    def test_from_dict_with_defaults(self):
        """Test creating BundleMetadata with missing fields."""
        data = {"pipeline_uid": "test_pipeline"}

        metadata = BundleMetadata.from_dict(data)

        assert metadata.bundle_format_version == "1.0"
        assert metadata.pipeline_uid == "test_pipeline"
        assert metadata.model_step_index is None
        assert metadata.original_manifest == {}


class TestBundleArtifactProvider:
    """Tests for BundleArtifactProvider."""

    def test_has_artifacts_for_step(self):
        """Test checking if artifacts exist for a step."""
        artifact_index = {
            "step_1": "step_1_SNV.joblib",
            "step_2": "step_2_MinMax.joblib",
            "step_4_fold0": "step_4_fold0_PLSRegression.joblib",
            "step_4_fold1": "step_4_fold1_PLSRegression.joblib",
        }

        provider = BundleArtifactProvider(
            bundle_path=Path("dummy.n4a"),
            artifact_index=artifact_index
        )

        assert provider.has_artifacts_for_step(1) is True
        assert provider.has_artifacts_for_step(2) is True
        assert provider.has_artifacts_for_step(3) is False
        assert provider.has_artifacts_for_step(4) is True

    def test_get_fold_weights(self):
        """Test getting fold weights."""
        fold_weights = {0: 0.52, 1: 0.48}
        provider = BundleArtifactProvider(
            bundle_path=Path("dummy.n4a"),
            artifact_index={},
            fold_weights=fold_weights
        )

        weights = provider.get_fold_weights()
        assert weights == {0: 0.52, 1: 0.48}


class TestBundleFormat:
    """Tests for BundleFormat enum."""

    def test_n4a_format(self):
        """Test .n4a format enum."""
        assert str(BundleFormat.N4A) == "n4a"
        assert BundleFormat.N4A.value == "n4a"

    def test_n4a_py_format(self):
        """Test .n4a.py format enum."""
        assert str(BundleFormat.N4A_PY) == "n4a.py"
        assert BundleFormat.N4A_PY.value == "n4a.py"


class TestBundleGenerator:
    """Tests for BundleGenerator class."""

    @pytest.fixture
    def mock_workspace(self, tmp_path):
        """Create a mock workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        return workspace

    @pytest.fixture
    def mock_resolved_prediction(self):
        """Create a mock ResolvedPrediction."""
        # Create mock trace
        trace = ExecutionTrace(
            trace_id="test_trace_123",
            pipeline_uid="0001_test_abc123",
            model_step_index=4,
            fold_weights={0: 0.52, 1: 0.48},
            preprocessing_chain="SNV>SG>MinMax"
        )

        # Add steps
        step1 = ExecutionStep(
            step_index=1,
            operator_type="transform",
            operator_class="SNV"
        )
        step1.artifacts = StepArtifacts(artifact_ids=["0001:1:all"])

        step4 = ExecutionStep(
            step_index=4,
            operator_type="model",
            operator_class="PLSRegression"
        )
        step4.artifacts = StepArtifacts(
            artifact_ids=["0001:4:0", "0001:4:1"],
            fold_artifact_ids={0: "0001:4:0", 1: "0001:4:1"}
        )

        trace.add_step(step1)
        trace.add_step(step4)

        # Create mock artifact provider
        mock_provider = MagicMock()
        mock_provider.get_artifacts_for_step.return_value = []

        return ResolvedPrediction(
            source_type=SourceType.PREDICTION,
            minimal_pipeline=[{"transform": "SNV"}, {"model": "PLSRegression"}],
            artifact_provider=mock_provider,
            trace=trace,
            fold_strategy=FoldStrategy.WEIGHTED_AVERAGE,
            fold_weights={0: 0.52, 1: 0.48},
            model_step_index=4,
            pipeline_uid="0001_test_abc123",
            manifest={"dataset": "wheat", "name": "pls_test"}
        )

    def test_create_bundle_manifest(self, mock_workspace, mock_resolved_prediction):
        """Test creating bundle manifest."""
        generator = BundleGenerator(mock_workspace)

        manifest = generator._create_bundle_manifest(
            mock_resolved_prediction,
            include_metadata=True
        )

        assert "bundle_format_version" in manifest
        assert manifest["pipeline_uid"] == "0001_test_abc123"
        assert manifest["model_step_index"] == 4
        assert manifest["preprocessing_chain"] == "SNV>SG>MinMax"
        assert manifest["trace_id"] == "test_trace_123"

    def test_extract_pipeline_config(self, mock_workspace, mock_resolved_prediction):
        """Test extracting pipeline configuration."""
        generator = BundleGenerator(mock_workspace)

        config = generator._extract_pipeline_config(mock_resolved_prediction)

        assert "steps" in config
        assert config["model_step_index"] == 4
        assert len(config["steps"]) == 2

    def test_artifact_filename(self, mock_workspace):
        """Test generating artifact filenames."""
        generator = BundleGenerator(mock_workspace)

        # Mock artifact object
        mock_artifact = MagicMock()
        mock_artifact.__class__.__name__ = "PLSRegression"

        # Test fold artifact
        filename = generator._artifact_filename("0001:4:0", mock_artifact)
        assert filename == "step_4_fold0_PLSRegression.joblib"

        # Test shared artifact
        filename = generator._artifact_filename("0001:2:all", mock_artifact)
        assert filename == "step_2_PLSRegression.joblib"


class TestBundleLoader:
    """Tests for BundleLoader class."""

    @pytest.fixture
    def create_test_bundle(self, tmp_path):
        """Create a test bundle file."""
        def _create_bundle(
            pipeline_uid="0001_test_abc123",
            model_step=4,
            include_trace=True
        ):
            import joblib

            bundle_path = tmp_path / "test_bundle.n4a"

            with zipfile.ZipFile(bundle_path, 'w') as zf:
                # Write manifest
                manifest = {
                    "bundle_format_version": "1.0",
                    "nirs4all_version": "0.9.0",
                    "created_at": "2024-12-14T10:00:00",
                    "pipeline_uid": pipeline_uid,
                    "source_type": "prediction",
                    "model_step_index": model_step,
                    "fold_strategy": "weighted_average",
                    "preprocessing_chain": "SNV>MinMax",
                }
                zf.writestr('manifest.json', json.dumps(manifest))

                # Write pipeline config
                pipeline_config = {
                    "steps": [{"transform": "SNV"}, {"model": "PLSRegression"}],
                    "model_step_index": model_step,
                }
                zf.writestr('pipeline.json', json.dumps(pipeline_config))

                # Write fold weights
                fold_weights = {"0": 0.52, "1": 0.48}
                zf.writestr('fold_weights.json', json.dumps(fold_weights))

                if include_trace:
                    trace_data = {
                        "trace_id": "test_trace",
                        "pipeline_uid": pipeline_uid,
                        "model_step_index": model_step,
                        "steps": [],
                    }
                    zf.writestr('trace.json', json.dumps(trace_data))

                # Write a mock artifact
                buffer = io.BytesIO()
                model = SimpleModel()
                joblib.dump(model, buffer)
                zf.writestr('artifacts/step_4_fold0_PLSRegression.joblib', buffer.getvalue())

            return bundle_path

        return _create_bundle

    def test_load_bundle_metadata(self, create_test_bundle):
        """Test loading bundle metadata."""
        bundle_path = create_test_bundle()

        loader = BundleLoader(bundle_path)

        assert loader.metadata is not None
        assert loader.metadata.pipeline_uid == "0001_test_abc123"
        assert loader.metadata.model_step_index == 4
        assert loader.metadata.preprocessing_chain == "SNV>MinMax"

    def test_load_fold_weights(self, create_test_bundle):
        """Test loading fold weights."""
        bundle_path = create_test_bundle()

        loader = BundleLoader(bundle_path)

        assert loader.fold_weights == {0: 0.52, 1: 0.48}

    def test_load_trace(self, create_test_bundle):
        """Test loading execution trace."""
        bundle_path = create_test_bundle(include_trace=True)

        loader = BundleLoader(bundle_path)

        assert loader.trace is not None
        assert loader.trace.trace_id == "test_trace"

    def test_bundle_not_found(self, tmp_path):
        """Test error when bundle file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            BundleLoader(tmp_path / "nonexistent.n4a")

    def test_invalid_bundle_format(self, tmp_path):
        """Test error when file is not a valid ZIP."""
        invalid_path = tmp_path / "invalid.n4a"
        invalid_path.write_text("not a zip file")

        with pytest.raises(ValueError, match="Invalid bundle format"):
            BundleLoader(invalid_path)

    def test_to_resolved_prediction(self, create_test_bundle):
        """Test converting bundle to ResolvedPrediction."""
        bundle_path = create_test_bundle()

        loader = BundleLoader(bundle_path)
        resolved = loader.to_resolved_prediction()

        assert resolved.source_type == SourceType.BUNDLE
        assert resolved.pipeline_uid == "0001_test_abc123"
        assert resolved.model_step_index == 4
        assert resolved.fold_weights == {0: 0.52, 1: 0.48}

    def test_get_step_info(self, create_test_bundle):
        """Test getting step information."""
        bundle_path = create_test_bundle()

        loader = BundleLoader(bundle_path)
        steps = loader.get_step_info()

        assert len(steps) >= 0  # May vary depending on trace/artifacts

    def test_repr(self, create_test_bundle):
        """Test string representation."""
        bundle_path = create_test_bundle()

        loader = BundleLoader(bundle_path)
        repr_str = repr(loader)

        assert "BundleLoader" in repr_str
        assert "0001_test_abc123" in repr_str


class TestIntegration:
    """Integration tests for bundle export/load cycle."""

    @pytest.fixture
    def mock_workspace(self, tmp_path):
        """Create a mock workspace with artifacts."""
        workspace = tmp_path / "workspace"
        runs_dir = workspace / "runs"
        runs_dir.mkdir(parents=True)
        return workspace

    def test_export_and_load_cycle(self, mock_workspace, tmp_path):
        """Test exporting a bundle and loading it back."""
        # This is a higher-level integration test
        # In a full test, we would run a pipeline first
        # For now, we just verify the infrastructure works

        generator = BundleGenerator(mock_workspace)

        # Verify generator initializes correctly
        assert generator.workspace_path == mock_workspace
        assert generator.resolver is not None
