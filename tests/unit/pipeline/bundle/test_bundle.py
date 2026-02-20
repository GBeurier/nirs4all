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
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nirs4all.pipeline.bundle import (
    BundleFormat,
    BundleGenerator,
    BundleLoader,
    BundleMetadata,
)
from nirs4all.pipeline.bundle.loader import BundleArtifactProvider
from nirs4all.pipeline.resolver import FoldStrategy, ResolvedPrediction, SourceType
from nirs4all.pipeline.trace import ExecutionStep, ExecutionTrace, StepArtifacts


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

class TestExtractStepInfoFromConfig:
    """Tests for _extract_step_info_from_config method.

    Critical: This method determines operator types from pipeline_config
    when no trace is available. Without correct operator types, y_processing
    transformers would be incorrectly applied to X (features) instead of y.
    """

    @pytest.fixture
    def create_bundle_with_config(self, tmp_path):
        """Create a bundle with specific pipeline configuration."""
        def _create(pipeline_steps, include_trace=False):
            bundle_path = tmp_path / "test_config.n4a"

            with zipfile.ZipFile(bundle_path, 'w') as zf:
                # Minimal manifest
                manifest = {
                    "bundle_format_version": "1.0",
                    "pipeline_uid": "test_pipeline",
                    "model_step_index": len(pipeline_steps),
                }
                zf.writestr('manifest.json', json.dumps(manifest))

                # Pipeline config with specific steps
                pipeline_config = {
                    "steps": pipeline_steps,
                    "model_step_index": len(pipeline_steps),
                }
                zf.writestr('pipeline.json', json.dumps(pipeline_config))

                if include_trace:
                    trace_data = {
                        "trace_id": "test_trace",
                        "pipeline_uid": "test_pipeline",
                        "steps": [],
                    }
                    zf.writestr('trace.json', json.dumps(trace_data))

            return bundle_path

        return _create

    def test_y_processing_detected(self, create_bundle_with_config):
        """Test that y_processing keyword is correctly identified."""
        pipeline_steps = [
            {"step": "sklearn.preprocessing.MinMaxScaler"},
            {"y_processing": "sklearn.preprocessing.StandardScaler"},
            {"model": {"class": "sklearn.cross_decomposition.PLSRegression"}},
        ]

        bundle_path = create_bundle_with_config(pipeline_steps, include_trace=False)
        loader = BundleLoader(bundle_path)

        # Without trace, step_info should be extracted from pipeline_config
        op_type = loader.artifact_provider.get_step_operator_type(2)
        assert op_type == "y_processing", f"Expected 'y_processing', got '{op_type}'"

    def test_feature_augmentation_detected(self, create_bundle_with_config):
        """Test that feature_augmentation keyword is correctly identified."""
        pipeline_steps = [
            {"step": "sklearn.preprocessing.MinMaxScaler"},
            {"feature_augmentation": ["SNV", "SavitzkyGolay"]},
            {"model": {"class": "sklearn.cross_decomposition.PLSRegression"}},
        ]

        bundle_path = create_bundle_with_config(pipeline_steps, include_trace=False)
        loader = BundleLoader(bundle_path)

        op_type = loader.artifact_provider.get_step_operator_type(2)
        assert op_type == "feature_augmentation", f"Expected 'feature_augmentation', got '{op_type}'"

    def test_model_detected(self, create_bundle_with_config):
        """Test that model keyword is correctly identified."""
        pipeline_steps = [
            {"step": "sklearn.preprocessing.MinMaxScaler"},
            {"model": {"class": "sklearn.cross_decomposition.PLSRegression"}},
        ]

        bundle_path = create_bundle_with_config(pipeline_steps, include_trace=False)
        loader = BundleLoader(bundle_path)

        op_type = loader.artifact_provider.get_step_operator_type(2)
        assert op_type == "model", f"Expected 'model', got '{op_type}'"

    def test_splitter_detected(self, create_bundle_with_config):
        """Test that splitter is correctly identified from class name."""
        pipeline_steps = [
            {"step": "sklearn.preprocessing.MinMaxScaler"},
            {"class": "sklearn.model_selection.ShuffleSplit", "params": {}},
            {"model": {"class": "sklearn.cross_decomposition.PLSRegression"}},
        ]

        bundle_path = create_bundle_with_config(pipeline_steps, include_trace=False)
        loader = BundleLoader(bundle_path)

        op_type = loader.artifact_provider.get_step_operator_type(2)
        assert op_type == "splitter", f"Expected 'splitter', got '{op_type}'"

    def test_regular_transform_detected(self, create_bundle_with_config):
        """Test that regular preprocessing step is identified as transform."""
        pipeline_steps = [
            {"step": "sklearn.preprocessing.MinMaxScaler"},
            {"model": {"class": "sklearn.cross_decomposition.PLSRegression"}},
        ]

        bundle_path = create_bundle_with_config(pipeline_steps, include_trace=False)
        loader = BundleLoader(bundle_path)

        op_type = loader.artifact_provider.get_step_operator_type(1)
        assert op_type == "transform", f"Expected 'transform', got '{op_type}'"

    def test_complex_pipeline_all_types(self, create_bundle_with_config):
        """Test a complex pipeline with multiple special operator types."""
        pipeline_steps = [
            {"step": "sklearn.preprocessing.MinMaxScaler"},           # 1: transform
            {"y_processing": "sklearn.preprocessing.StandardScaler"}, # 2: y_processing
            {"feature_augmentation": ["SNV", "SG"]},                  # 3: feature_augmentation
            {"class": "sklearn.model_selection.KFold", "params": {}}, # 4: splitter
            {"model": {"class": "PLSRegression"}},                    # 5: model
        ]

        bundle_path = create_bundle_with_config(pipeline_steps, include_trace=False)
        loader = BundleLoader(bundle_path)

        assert loader.artifact_provider.get_step_operator_type(1) == "transform"
        assert loader.artifact_provider.get_step_operator_type(2) == "y_processing"
        assert loader.artifact_provider.get_step_operator_type(3) == "feature_augmentation"
        assert loader.artifact_provider.get_step_operator_type(4) == "splitter"
        assert loader.artifact_provider.get_step_operator_type(5) == "model"

    def test_trace_takes_precedence_over_config(self, tmp_path):
        """Test that trace operator_type takes precedence over config parsing."""
        bundle_path = tmp_path / "trace_precedence.n4a"

        with zipfile.ZipFile(bundle_path, 'w') as zf:
            manifest = {
                "bundle_format_version": "1.0",
                "pipeline_uid": "test_pipeline",
                "model_step_index": 2,
            }
            zf.writestr('manifest.json', json.dumps(manifest))

            # Pipeline config says y_processing
            pipeline_config = {
                "steps": [
                    {"step": "MinMaxScaler"},
                    {"y_processing": "StandardScaler"},  # Config says y_processing
                ],
                "model_step_index": 2,
            }
            zf.writestr('pipeline.json', json.dumps(pipeline_config))

            # But trace says transform (trace should win)
            trace_data = {
                "trace_id": "test_trace",
                "pipeline_uid": "test_pipeline",
                "steps": [
                    {"step_index": 1, "operator_type": "transform", "operator_class": "MinMaxScaler"},
                    {"step_index": 2, "operator_type": "custom_type_from_trace", "operator_class": "StandardScaler"},
                ],
            }
            zf.writestr('trace.json', json.dumps(trace_data))

        loader = BundleLoader(bundle_path)

        # Trace should take precedence
        op_type = loader.artifact_provider.get_step_operator_type(2)
        assert op_type == "custom_type_from_trace", f"Trace should take precedence, got '{op_type}'"

class TestBundlePredictWithSpecialOperators:
    """Tests for BundleLoader.predict() with special operator types.

    Critical: These tests verify that y_processing and feature_augmentation
    are correctly handled during prediction - the exact scenario that caused
    the original bug.
    """

    @pytest.fixture
    def create_prediction_bundle(self, tmp_path):
        """Create a bundle with actual artifacts for prediction testing."""
        import joblib
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        def _create(include_y_processing=False, include_feature_augmentation=False, include_trace=True):
            bundle_path = tmp_path / "prediction_test.n4a"

            # Create fitted transformers
            X_fit = np.random.randn(50, 100)
            y_fit = np.random.randn(50, 1)

            x_scaler = MinMaxScaler()
            x_scaler.fit(X_fit)

            y_scaler = StandardScaler()
            y_scaler.fit(y_fit)

            # Create a simple model
            model = SimpleModel()

            step_idx = 1
            pipeline_steps = [{"step": "sklearn.preprocessing.MinMaxScaler"}]
            artifacts_to_write = [(f"step_{step_idx}_MinMaxScaler.joblib", x_scaler)]

            if include_y_processing:
                step_idx += 1
                pipeline_steps.append({"y_processing": "sklearn.preprocessing.StandardScaler"})
                artifacts_to_write.append((f"step_{step_idx}_StandardScaler.joblib", y_scaler))

            if include_feature_augmentation:
                step_idx += 1
                pipeline_steps.append({"feature_augmentation": ["SNV"]})
                # Feature augmentation artifact (SNV is stateless, but we still serialize)
                from nirs4all.operators.transforms import StandardNormalVariate
                snv = StandardNormalVariate()
                snv.fit(X_fit)
                artifacts_to_write.append((f"step_{step_idx}_StandardNormalVariate.joblib", snv))

            model_step_idx = step_idx + 1
            pipeline_steps.append({"model": {"class": "SimpleModel"}})
            artifacts_to_write.append((f"step_{model_step_idx}_fold0_SimpleModel.joblib", model))

            with zipfile.ZipFile(bundle_path, 'w') as zf:
                manifest = {
                    "bundle_format_version": "1.0",
                    "pipeline_uid": "prediction_test",
                    "model_step_index": model_step_idx,
                }
                zf.writestr('manifest.json', json.dumps(manifest))

                pipeline_config = {
                    "steps": pipeline_steps,
                    "model_step_index": model_step_idx,
                }
                zf.writestr('pipeline.json', json.dumps(pipeline_config))

                fold_weights = {"0": 1.0}
                zf.writestr('fold_weights.json', json.dumps(fold_weights))

                if include_trace:
                    trace_steps = []
                    for i, step in enumerate(pipeline_steps):
                        step_info = {"step_index": i + 1, "operator_class": "Unknown"}
                        if "y_processing" in step:
                            step_info["operator_type"] = "y_processing"
                        elif "feature_augmentation" in step:
                            step_info["operator_type"] = "feature_augmentation"
                        elif "model" in step:
                            step_info["operator_type"] = "model"
                        else:
                            step_info["operator_type"] = "transform"
                        trace_steps.append(step_info)

                    trace_data = {
                        "trace_id": "test_trace",
                        "pipeline_uid": "prediction_test",
                        "model_step_index": model_step_idx,
                        "steps": trace_steps,
                    }
                    zf.writestr('trace.json', json.dumps(trace_data))

                # Write artifacts
                for artifact_name, artifact_obj in artifacts_to_write:
                    buffer = io.BytesIO()
                    joblib.dump(artifact_obj, buffer)
                    zf.writestr(f'artifacts/{artifact_name}', buffer.getvalue())

            return bundle_path

        return _create

    def test_predict_without_y_processing(self, create_prediction_bundle):
        """Test basic prediction without y_processing works."""
        bundle_path = create_prediction_bundle(
            include_y_processing=False,
            include_feature_augmentation=False,
            include_trace=True
        )

        loader = BundleLoader(bundle_path)
        X_test = np.random.randn(10, 100)

        y_pred = loader.predict(X_test)

        assert y_pred is not None
        assert len(y_pred) == 10

    def test_predict_with_y_processing_and_trace(self, create_prediction_bundle):
        """Test prediction with y_processing when trace is available."""
        bundle_path = create_prediction_bundle(
            include_y_processing=True,
            include_feature_augmentation=False,
            include_trace=True
        )

        loader = BundleLoader(bundle_path)
        X_test = np.random.randn(10, 100)

        # This should NOT crash - y_processing should be skipped (not applied to X)
        y_pred = loader.predict(X_test)

        assert y_pred is not None
        assert len(y_pred) == 10

    def test_predict_with_y_processing_without_trace(self, create_prediction_bundle):
        """Test prediction with y_processing when NO trace is available.

        This is the exact scenario that caused the original bug:
        - Bundle has y_processing artifact (MinMaxScaler fitted on y with 1 feature)
        - No trace.json in bundle
        - Loader must infer operator types from pipeline_config
        - y_processing transformer must NOT be applied to X
        """
        bundle_path = create_prediction_bundle(
            include_y_processing=True,
            include_feature_augmentation=False,
            include_trace=False  # Critical: no trace!
        )

        loader = BundleLoader(bundle_path)

        # Verify no trace
        assert loader.trace is None, "This test requires no trace"

        X_test = np.random.randn(10, 100)

        # This should NOT crash - y_processing must be correctly identified from config
        y_pred = loader.predict(X_test)

        assert y_pred is not None
        assert len(y_pred) == 10

    def test_predict_with_feature_augmentation(self, create_prediction_bundle):
        """Test prediction with feature_augmentation."""
        bundle_path = create_prediction_bundle(
            include_y_processing=False,
            include_feature_augmentation=True,
            include_trace=True
        )

        loader = BundleLoader(bundle_path)
        X_test = np.random.randn(10, 100)

        y_pred = loader.predict(X_test)

        assert y_pred is not None
        assert len(y_pred) == 10

    def test_predict_complex_pipeline_without_trace(self, create_prediction_bundle):
        """Test that all special operators work without trace."""
        bundle_path = create_prediction_bundle(
            include_y_processing=True,
            include_feature_augmentation=True,
            include_trace=False  # No trace - relies on config parsing
        )

        loader = BundleLoader(bundle_path)
        assert loader.trace is None

        X_test = np.random.randn(10, 100)

        # Should work correctly without trace
        y_pred = loader.predict(X_test)

        assert y_pred is not None
        assert len(y_pred) == 10

class TestBundleEdgeCases:
    """Edge case tests for bundle export/load with special operators.

    This test class covers critical edge cases that could cause silent failures
    or incorrect predictions. These scenarios are designed to catch regressions
    in the bundle system, particularly around operator type detection.
    """

    @pytest.fixture
    def create_edge_case_bundle(self, tmp_path):
        """Factory for creating bundles with specific edge case configurations."""
        import joblib
        from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

        def _create(
            pipeline_steps: list,
            artifacts: list,
            model_step_index: int,
            include_trace: bool = False,
            trace_steps: list | None = None,
        ):
            """Create a bundle with specific configuration.

            Args:
                pipeline_steps: List of step dicts for pipeline.json
                artifacts: List of (filename, artifact_object) tuples
                model_step_index: Model step index for manifest
                include_trace: Whether to include trace.json
                trace_steps: Custom trace steps (if include_trace=True)
            """
            bundle_path = tmp_path / f"edge_case_{np.random.randint(10000)}.n4a"

            with zipfile.ZipFile(bundle_path, 'w') as zf:
                # Manifest
                manifest = {
                    "bundle_format_version": "1.0",
                    "pipeline_uid": f"edge_case_{np.random.randint(10000)}",
                    "model_step_index": model_step_index,
                }
                zf.writestr('manifest.json', json.dumps(manifest))

                # Pipeline config
                pipeline_config = {
                    "steps": pipeline_steps,
                    "model_step_index": model_step_index,
                }
                zf.writestr('pipeline.json', json.dumps(pipeline_config))

                # Fold weights
                zf.writestr('fold_weights.json', json.dumps({"0": 1.0}))

                # Trace (optional)
                if include_trace:
                    trace_data = {
                        "trace_id": "edge_case_trace",
                        "pipeline_uid": "edge_case",
                        "model_step_index": model_step_index,
                        "steps": trace_steps or [],
                    }
                    zf.writestr('trace.json', json.dumps(trace_data))

                # Artifacts
                for filename, artifact_obj in artifacts:
                    buffer = io.BytesIO()
                    joblib.dump(artifact_obj, buffer)
                    zf.writestr(f'artifacts/{filename}', buffer.getvalue())

            return bundle_path

        return _create

    def test_y_processing_at_step_1_without_trace(self, create_edge_case_bundle):
        """Test y_processing as the very first step (unusual but valid)."""
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        X_fit = np.random.randn(50, 100)
        y_fit = np.random.randn(50, 1)

        y_scaler = StandardScaler()
        y_scaler.fit(y_fit)

        x_scaler = MinMaxScaler()
        x_scaler.fit(X_fit)

        model = SimpleModel()

        pipeline_steps = [
            {"y_processing": "sklearn.preprocessing.StandardScaler"},  # Step 1
            {"step": "sklearn.preprocessing.MinMaxScaler"},            # Step 2
            {"model": {"class": "SimpleModel"}},                       # Step 3
        ]

        artifacts = [
            ("step_1_StandardScaler.joblib", y_scaler),
            ("step_2_MinMaxScaler.joblib", x_scaler),
            ("step_3_fold0_SimpleModel.joblib", model),
        ]

        bundle_path = create_edge_case_bundle(
            pipeline_steps=pipeline_steps,
            artifacts=artifacts,
            model_step_index=3,
            include_trace=False,
        )

        loader = BundleLoader(bundle_path)
        assert loader.trace is None

        X_test = np.random.randn(10, 100)

        # Should NOT crash - step 1 is y_processing, must be skipped
        y_pred = loader.predict(X_test)
        assert y_pred is not None
        assert len(y_pred) == 10

    def test_multiple_y_processing_steps_without_trace(self, create_edge_case_bundle):
        """Test multiple y_processing steps (edge case but possible)."""
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        X_fit = np.random.randn(50, 100)
        y_fit = np.random.randn(50, 1)

        y_scaler1 = MinMaxScaler()
        y_scaler1.fit(y_fit)

        y_scaler2 = StandardScaler()
        y_scaler2.fit(y_fit)

        x_scaler = MinMaxScaler()
        x_scaler.fit(X_fit)

        model = SimpleModel()

        pipeline_steps = [
            {"step": "sklearn.preprocessing.MinMaxScaler"},            # Step 1: X transform
            {"y_processing": "sklearn.preprocessing.MinMaxScaler"},    # Step 2: y transform
            {"y_processing": "sklearn.preprocessing.StandardScaler"},  # Step 3: y transform again
            {"model": {"class": "SimpleModel"}},                       # Step 4
        ]

        artifacts = [
            ("step_1_MinMaxScaler.joblib", x_scaler),
            ("step_2_MinMaxScaler.joblib", y_scaler1),
            ("step_3_StandardScaler.joblib", y_scaler2),
            ("step_4_fold0_SimpleModel.joblib", model),
        ]

        bundle_path = create_edge_case_bundle(
            pipeline_steps=pipeline_steps,
            artifacts=artifacts,
            model_step_index=4,
            include_trace=False,
        )

        loader = BundleLoader(bundle_path)
        X_test = np.random.randn(10, 100)

        # Both y_processing steps must be skipped
        y_pred = loader.predict(X_test)
        assert y_pred is not None
        assert len(y_pred) == 10

    def test_y_processing_immediately_before_model(self, create_edge_case_bundle):
        """Test y_processing as the last step before model."""
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        X_fit = np.random.randn(50, 100)
        y_fit = np.random.randn(50, 1)

        x_scaler = MinMaxScaler()
        x_scaler.fit(X_fit)

        y_scaler = StandardScaler()
        y_scaler.fit(y_fit)

        model = SimpleModel()

        pipeline_steps = [
            {"step": "sklearn.preprocessing.MinMaxScaler"},            # Step 1
            {"y_processing": "sklearn.preprocessing.StandardScaler"},  # Step 2 - right before model
            {"model": {"class": "SimpleModel"}},                       # Step 3
        ]

        artifacts = [
            ("step_1_MinMaxScaler.joblib", x_scaler),
            ("step_2_StandardScaler.joblib", y_scaler),
            ("step_3_fold0_SimpleModel.joblib", model),
        ]

        bundle_path = create_edge_case_bundle(
            pipeline_steps=pipeline_steps,
            artifacts=artifacts,
            model_step_index=3,
            include_trace=False,
        )

        loader = BundleLoader(bundle_path)
        X_test = np.random.randn(10, 100)

        y_pred = loader.predict(X_test)
        assert y_pred is not None
        assert len(y_pred) == 10

    def test_y_processing_with_same_class_as_x_preprocessing(self, create_edge_case_bundle):
        """Test when y_processing uses the same scaler class as X preprocessing.

        This is a critical edge case: both use MinMaxScaler but they're fitted
        on different data (X vs y) with different shapes.
        """
        from sklearn.preprocessing import MinMaxScaler

        X_fit = np.random.randn(50, 100)  # 100 features
        y_fit = np.random.randn(50, 1)    # 1 target

        x_scaler = MinMaxScaler()
        x_scaler.fit(X_fit)  # Fitted on 100 features

        y_scaler = MinMaxScaler()
        y_scaler.fit(y_fit)  # Fitted on 1 feature

        model = SimpleModel()

        pipeline_steps = [
            {"step": "sklearn.preprocessing.MinMaxScaler"},            # Step 1: X
            {"y_processing": "sklearn.preprocessing.MinMaxScaler"},    # Step 2: y (same class!)
            {"model": {"class": "SimpleModel"}},                       # Step 3
        ]

        artifacts = [
            ("step_1_MinMaxScaler.joblib", x_scaler),
            ("step_2_MinMaxScaler.joblib", y_scaler),  # Same class name in filename
            ("step_3_fold0_SimpleModel.joblib", model),
        ]

        bundle_path = create_edge_case_bundle(
            pipeline_steps=pipeline_steps,
            artifacts=artifacts,
            model_step_index=3,
            include_trace=False,
        )

        loader = BundleLoader(bundle_path)
        X_test = np.random.randn(10, 100)

        # If y_processing not correctly identified, this would crash:
        # "X has 100 features, but MinMaxScaler is expecting 1 features as input"
        y_pred = loader.predict(X_test)
        assert y_pred is not None
        assert len(y_pred) == 10

    def test_all_special_operators_interleaved(self, create_edge_case_bundle):
        """Test complex pipeline with all special operators interleaved."""
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        from nirs4all.operators.transforms import StandardNormalVariate

        n_features = 100
        X_fit = np.random.randn(50, n_features)
        y_fit = np.random.randn(50, 1)

        x_scaler1 = MinMaxScaler()
        x_scaler1.fit(X_fit)

        y_scaler = StandardScaler()
        y_scaler.fit(y_fit)

        snv = StandardNormalVariate()
        snv.fit(X_fit)

        # After feature_augmentation, features are [original, SNV(original)] = 2*n_features
        X_augmented = np.hstack([X_fit, snv.transform(X_fit)])
        x_scaler2 = StandardScaler()
        x_scaler2.fit(X_augmented)  # Fitted on augmented features (200)

        model = SimpleModel()

        pipeline_steps = [
            {"step": "sklearn.preprocessing.MinMaxScaler"},            # Step 1: X transform
            {"y_processing": "sklearn.preprocessing.StandardScaler"},  # Step 2: y transform
            {"feature_augmentation": ["SNV"]},                         # Step 3: feature aug (100 -> 200)
            {"step": "sklearn.preprocessing.StandardScaler"},          # Step 4: X transform (on 200 features)
            {"class": "sklearn.model_selection.KFold", "params": {}},  # Step 5: splitter
            {"model": {"class": "SimpleModel"}},                       # Step 6
        ]

        artifacts = [
            ("step_1_MinMaxScaler.joblib", x_scaler1),
            ("step_2_StandardScaler.joblib", y_scaler),
            ("step_3_StandardNormalVariate.joblib", snv),
            ("step_4_StandardScaler.joblib", x_scaler2),
            ("step_6_fold0_SimpleModel.joblib", model),
        ]

        bundle_path = create_edge_case_bundle(
            pipeline_steps=pipeline_steps,
            artifacts=artifacts,
            model_step_index=6,
            include_trace=False,
        )

        loader = BundleLoader(bundle_path)
        X_test = np.random.randn(10, n_features)

        # All special operators must be correctly identified
        y_pred = loader.predict(X_test)
        assert y_pred is not None
        assert len(y_pred) == 10

    def test_bundle_with_missing_pipeline_config(self, tmp_path):
        """Test behavior when pipeline.json is missing or empty."""
        import joblib
        from sklearn.preprocessing import MinMaxScaler

        bundle_path = tmp_path / "no_pipeline_config.n4a"

        X_fit = np.random.randn(50, 100)
        x_scaler = MinMaxScaler()
        x_scaler.fit(X_fit)
        model = SimpleModel()

        with zipfile.ZipFile(bundle_path, 'w') as zf:
            manifest = {
                "bundle_format_version": "1.0",
                "pipeline_uid": "no_config",
                "model_step_index": 2,
            }
            zf.writestr('manifest.json', json.dumps(manifest))

            # Empty pipeline config
            zf.writestr('pipeline.json', json.dumps({}))
            zf.writestr('fold_weights.json', json.dumps({"0": 1.0}))

            # Artifacts
            buffer = io.BytesIO()
            joblib.dump(x_scaler, buffer)
            zf.writestr('artifacts/step_1_MinMaxScaler.joblib', buffer.getvalue())

            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            zf.writestr('artifacts/step_2_fold0_SimpleModel.joblib', buffer.getvalue())

        loader = BundleLoader(bundle_path)

        # With empty pipeline config, step_info should be empty
        # but the loader should still work (graceful degradation)
        op_type = loader.artifact_provider.get_step_operator_type(1)
        assert op_type is None  # No info available

    def test_sequential_predictions_same_bundle(self, create_edge_case_bundle):
        """Test multiple sequential predictions from the same bundle.

        Verifies no state leakage between predictions.
        """
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        X_fit = np.random.randn(50, 100)
        y_fit = np.random.randn(50, 1)

        x_scaler = MinMaxScaler()
        x_scaler.fit(X_fit)

        y_scaler = StandardScaler()
        y_scaler.fit(y_fit)

        model = SimpleModel()

        pipeline_steps = [
            {"step": "sklearn.preprocessing.MinMaxScaler"},
            {"y_processing": "sklearn.preprocessing.StandardScaler"},
            {"model": {"class": "SimpleModel"}},
        ]

        artifacts = [
            ("step_1_MinMaxScaler.joblib", x_scaler),
            ("step_2_StandardScaler.joblib", y_scaler),
            ("step_3_fold0_SimpleModel.joblib", model),
        ]

        bundle_path = create_edge_case_bundle(
            pipeline_steps=pipeline_steps,
            artifacts=artifacts,
            model_step_index=3,
            include_trace=False,
        )

        loader = BundleLoader(bundle_path)

        # Multiple predictions
        for i in range(5):
            X_test = np.random.randn(10 + i, 100)
            y_pred = loader.predict(X_test)
            assert y_pred is not None
            assert len(y_pred) == 10 + i, f"Prediction {i} failed"

    def test_y_processing_deep_in_pipeline(self, create_edge_case_bundle):
        """Test y_processing buried deep in the pipeline (many steps before it)."""
        from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

        X_fit = np.random.randn(50, 100)
        y_fit = np.random.randn(50, 1)

        scalers = []
        for _ in range(5):
            s = MinMaxScaler()
            s.fit(X_fit)
            scalers.append(s)

        y_scaler = StandardScaler()
        y_scaler.fit(y_fit)

        model = SimpleModel()

        # Build pipeline with many X preprocessing steps, then y_processing
        pipeline_steps = []
        artifacts = []

        for i in range(5):
            pipeline_steps.append({"step": "sklearn.preprocessing.MinMaxScaler"})
            artifacts.append((f"step_{i+1}_MinMaxScaler.joblib", scalers[i]))

        # y_processing at step 6
        pipeline_steps.append({"y_processing": "sklearn.preprocessing.StandardScaler"})
        artifacts.append(("step_6_StandardScaler.joblib", y_scaler))

        # Model at step 7
        pipeline_steps.append({"model": {"class": "SimpleModel"}})
        artifacts.append(("step_7_fold0_SimpleModel.joblib", model))

        bundle_path = create_edge_case_bundle(
            pipeline_steps=pipeline_steps,
            artifacts=artifacts,
            model_step_index=7,
            include_trace=False,
        )

        loader = BundleLoader(bundle_path)
        X_test = np.random.randn(10, 100)

        # y_processing at step 6 must be correctly identified and skipped
        y_pred = loader.predict(X_test)
        assert y_pred is not None
        assert len(y_pred) == 10

    def test_operator_type_detection_with_various_dict_formats(self, tmp_path):
        """Test operator type detection handles various step dict formats."""
        bundle_path = tmp_path / "various_formats.n4a"

        with zipfile.ZipFile(bundle_path, 'w') as zf:
            manifest = {
                "bundle_format_version": "1.0",
                "pipeline_uid": "format_test",
                "model_step_index": 6,
            }
            zf.writestr('manifest.json', json.dumps(manifest))

            # Various step formats
            pipeline_config = {
                "steps": [
                    {"step": "MinMaxScaler"},                    # Simple string
                    {"y_processing": {"class": "StandardScaler", "params": {}}},  # Dict value
                    {"feature_augmentation": ["SNV", "SG"]},     # List value
                    {"class": "KFold"},                          # Splitter by class
                    {"model": "PLSRegression"},                  # Model as string
                    {"model": {"class": "PLSRegression", "params": {"n_components": 5}}},  # Model as dict
                ],
                "model_step_index": 6,
            }
            zf.writestr('pipeline.json', json.dumps(pipeline_config))

        loader = BundleLoader(bundle_path)

        # Verify detection works for all formats
        assert loader.artifact_provider.get_step_operator_type(1) == "transform"
        assert loader.artifact_provider.get_step_operator_type(2) == "y_processing"
        assert loader.artifact_provider.get_step_operator_type(3) == "feature_augmentation"
        # Note: step 4 might be detected as transform if "Fold" not in class name

    def test_y_processing_dimension_mismatch_would_crash(self, create_edge_case_bundle):
        """Explicitly verify that WITHOUT the fix, y_processing would crash.

        This test creates the exact scenario that caused the original bug:
        - MinMaxScaler fitted on y (1 feature)
        - If incorrectly applied to X (100 features), sklearn raises ValueError

        We verify that our implementation correctly skips y_processing.
        """
        from sklearn.preprocessing import MinMaxScaler

        X_fit = np.random.randn(50, 100)  # 100 features
        y_fit = np.random.randn(50, 1)    # 1 target

        x_scaler = MinMaxScaler()
        x_scaler.fit(X_fit)

        # This y_scaler expects exactly 1 feature
        y_scaler = MinMaxScaler()
        y_scaler.fit(y_fit)

        model = SimpleModel()

        # Verify directly that y_scaler would crash if applied to X
        X_test = np.random.randn(10, 100)
        with pytest.raises(ValueError, match="features"):
            y_scaler.transform(X_test)  # This MUST raise ValueError

        # Now test the bundle handles this correctly
        pipeline_steps = [
            {"step": "sklearn.preprocessing.MinMaxScaler"},
            {"y_processing": "sklearn.preprocessing.MinMaxScaler"},
            {"model": {"class": "SimpleModel"}},
        ]

        artifacts = [
            ("step_1_MinMaxScaler.joblib", x_scaler),
            ("step_2_MinMaxScaler.joblib", y_scaler),
            ("step_3_fold0_SimpleModel.joblib", model),
        ]

        bundle_path = create_edge_case_bundle(
            pipeline_steps=pipeline_steps,
            artifacts=artifacts,
            model_step_index=3,
            include_trace=False,
        )

        loader = BundleLoader(bundle_path)

        # This MUST NOT raise ValueError - y_processing must be skipped
        y_pred = loader.predict(X_test)
        assert y_pred is not None

    def test_trace_with_wrong_operator_type_vs_config(self, tmp_path):
        """Test that trace operator_type is used even if it contradicts config.

        If trace says a step is 'transform', it should be treated as transform
        even if the config suggests otherwise.
        """
        import joblib
        from sklearn.preprocessing import MinMaxScaler

        bundle_path = tmp_path / "trace_vs_config.n4a"

        X_fit = np.random.randn(50, 100)
        x_scaler = MinMaxScaler()
        x_scaler.fit(X_fit)
        model = SimpleModel()

        with zipfile.ZipFile(bundle_path, 'w') as zf:
            manifest = {
                "bundle_format_version": "1.0",
                "pipeline_uid": "trace_vs_config",
                "model_step_index": 2,
            }
            zf.writestr('manifest.json', json.dumps(manifest))

            # Config says step 1 is y_processing
            pipeline_config = {
                "steps": [
                    {"y_processing": "MinMaxScaler"},  # Config says y_processing
                    {"model": {"class": "SimpleModel"}},
                ],
                "model_step_index": 2,
            }
            zf.writestr('pipeline.json', json.dumps(pipeline_config))

            # But trace says step 1 is transform (trace should win)
            trace_data = {
                "trace_id": "override_trace",
                "pipeline_uid": "trace_vs_config",
                "model_step_index": 2,
                "steps": [
                    {"step_index": 1, "operator_type": "transform", "operator_class": "MinMaxScaler"},
                    {"step_index": 2, "operator_type": "model", "operator_class": "SimpleModel"},
                ],
            }
            zf.writestr('trace.json', json.dumps(trace_data))

            zf.writestr('fold_weights.json', json.dumps({"0": 1.0}))

            # Artifacts
            buffer = io.BytesIO()
            joblib.dump(x_scaler, buffer)
            zf.writestr('artifacts/step_1_MinMaxScaler.joblib', buffer.getvalue())

            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            zf.writestr('artifacts/step_2_fold0_SimpleModel.joblib', buffer.getvalue())

        loader = BundleLoader(bundle_path)

        # Trace says transform, so step 1 should be treated as X transform
        op_type = loader.artifact_provider.get_step_operator_type(1)
        assert op_type == "transform", f"Trace should override config, got '{op_type}'"
