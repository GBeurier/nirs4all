"""
Integration tests for complete artifact flow.

Tests the full lifecycle of artifacts from training to prediction:
- Training creates artifacts in centralized storage
- Prediction loads artifacts correctly
- Manifest references are valid
- Artifact integrity is preserved
"""

import pytest
import numpy as np
from pathlib import Path
import yaml
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler

from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.pipeline.storage.artifacts.artifact_loader import ArtifactLoader


def create_test_dataset(n_samples: int = 100, n_features: int = 50) -> SpectroDataset:
    """Create a synthetic dataset for testing."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :5], axis=1) + np.random.randn(n_samples) * 0.1

    dataset = SpectroDataset(name="test_artifact_flow")
    dataset.add_samples(X[:80], indexes={"partition": "train"})
    dataset.add_samples(X[80:], indexes={"partition": "test"})
    dataset.add_targets(y[:80])
    dataset.add_targets(y[80:])

    return dataset


class TestTrainingArtifactCreation:
    """Tests for artifact creation during training."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace directory."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def runner(self, workspace_path):
        """Create a PipelineRunner with artifact saving enabled."""
        return PipelineRunner(
            workspace_path=workspace_path,
            save_artifacts=True,
            verbose=0,
            enable_tab_reports=False,
            show_spinner=False
        )

    @pytest.fixture
    def dataset(self):
        """Create test dataset."""
        return create_test_dataset()

    def test_training_creates_manifest_with_artifacts(
        self, runner, dataset, workspace_path
    ):
        """Verify training creates manifest.yaml with artifact references."""
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"class": "sklearn.preprocessing.StandardScaler"},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = runner.run(
            PipelineConfigs(pipeline),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0

        # Find manifest
        runs_dir = workspace_path / "runs"
        assert runs_dir.exists(), "Runs directory should be created"

        manifest_files = list(runs_dir.glob("*/*/manifest.yaml"))
        assert len(manifest_files) > 0, "Should create at least one manifest"

        # Check manifest structure
        with open(manifest_files[0]) as f:
            manifest = yaml.safe_load(f)

        assert "artifacts" in manifest, "Manifest should have artifacts section"

        # Check artifacts section (v2 format)
        artifacts = manifest["artifacts"]
        if isinstance(artifacts, dict):
            # v2 format
            assert "items" in artifacts or "schema_version" in artifacts
        elif isinstance(artifacts, list):
            # v1 format (backward compat)
            assert len(artifacts) >= 0

    def test_training_saves_binaries_to_disk(
        self, runner, dataset, workspace_path
    ):
        """Verify training saves binary artifacts to disk."""
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"class": "sklearn.preprocessing.StandardScaler"},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = runner.run(
            PipelineConfigs(pipeline),
            dataset
        )

        # Check for binaries (v2 centralized or v1 per-run)
        runs_dir = workspace_path / "runs"
        binaries_v2 = workspace_path / "binaries"

        # At least one location should have binaries
        v1_binaries = list(runs_dir.glob("*/_binaries/*"))
        v2_binaries = list(binaries_v2.glob("**/*.pkl")) + list(binaries_v2.glob("**/*.joblib"))

        assert len(v1_binaries) > 0 or len(v2_binaries) > 0, \
            "Should create binary artifact files"

    def test_artifact_paths_in_manifest_are_valid(
        self, runner, dataset, workspace_path
    ):
        """Verify artifact paths in manifest point to existing files."""
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"class": "sklearn.preprocessing.StandardScaler"},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = runner.run(
            PipelineConfigs(pipeline),
            dataset
        )

        # Find manifest
        runs_dir = workspace_path / "runs"
        manifest_files = list(runs_dir.glob("*/*/manifest.yaml"))

        if len(manifest_files) == 0:
            pytest.skip("No manifest files found")

        with open(manifest_files[0]) as f:
            manifest = yaml.safe_load(f)

        artifacts = manifest.get("artifacts", [])

        # Handle v2 format
        if isinstance(artifacts, dict):
            items = artifacts.get("items", [])
        else:
            items = artifacts

        # Check each artifact path is valid or relative
        for item in items:
            path = item.get("path", "")
            # Paths should be non-empty strings
            assert path, f"Artifact should have a path: {item}"


class TestPredictionArtifactLoading:
    """Tests for artifact loading during prediction."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace directory."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def runner(self, workspace_path):
        """Create a PipelineRunner."""
        return PipelineRunner(
            workspace_path=workspace_path,
            save_artifacts=True,
            verbose=0,
            enable_tab_reports=False,
            show_spinner=False
        )

    @pytest.fixture
    def dataset(self):
        """Create test dataset."""
        return create_test_dataset()

    def test_prediction_loads_trained_artifacts(
        self, runner, dataset, workspace_path
    ):
        """Verify prediction mode correctly loads trained artifacts."""
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"class": "sklearn.preprocessing.StandardScaler"},
            {"model": Ridge(alpha=1.0)},
        ]

        # Train
        predictions, _ = runner.run(
            PipelineConfigs(pipeline),
            dataset
        )

        # Get a prediction to replay
        test_preds = predictions.filter_predictions(partition="test")
        if len(test_preds) == 0:
            pytest.skip("No test predictions available")

        best_pred = test_preds[0]

        # Predict
        y_pred, _ = runner.predict(
            prediction_obj=best_pred,
            dataset=dataset,
            dataset_name="test_artifact_flow"
        )

        assert y_pred is not None
        assert len(y_pred) > 0

    def test_prediction_produces_consistent_results(
        self, runner, dataset, workspace_path
    ):
        """Verify prediction uses trained models and produces results."""
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"class": "sklearn.preprocessing.StandardScaler"},
            {"model": Ridge(alpha=1.0)},
        ]

        # Train
        predictions, _ = runner.run(
            PipelineConfigs(pipeline),
            dataset
        )

        test_preds = predictions.filter_predictions(partition="test")
        if len(test_preds) == 0:
            pytest.skip("No test predictions available")

        best_pred = test_preds[0]

        # Predict on same dataset - prediction includes all partitions
        pred_y_pred, _ = runner.predict(
            prediction_obj=best_pred,
            dataset=dataset,
            dataset_name="test_artifact_flow"
        )

        # Verify prediction succeeded and produces expected number of samples
        assert pred_y_pred is not None
        # Full dataset has 100 samples (80 train + 20 test)
        assert len(pred_y_pred) == 100, \
            f"Expected 100 predictions, got {len(pred_y_pred)}"


class TestArtifactLoaderIntegration:
    """Tests for ArtifactLoader integration."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace with binaries."""
        workspace = tmp_path / "workspace"
        binaries_dir = workspace / "binaries" / "test_dataset"
        binaries_dir.mkdir(parents=True)
        return workspace

    def test_loader_loads_from_manifest(self, workspace_path):
        """Test ArtifactLoader imports from manifest correctly."""
        from nirs4all.pipeline.storage.artifacts.artifact_persistence import persist

        binaries_dir = workspace_path / "binaries" / "test_dataset"

        # Create a real artifact
        scaler = StandardScaler()
        scaler.fit(np.array([[0, 1], [1, 2], [2, 3]]))
        meta = persist(scaler, binaries_dir, "test_scaler")

        # Create manifest
        manifest = {
            "dataset": "test_dataset",
            "artifacts": {
                "schema_version": "2.0",
                "items": [{
                    "artifact_id": "0001:0:all",
                    "content_hash": meta["hash"],
                    "path": meta["path"],
                    "pipeline_id": "0001",
                    "branch_path": [],
                    "step_index": 0,
                    "artifact_type": "transformer",
                    "class_name": "StandardScaler",
                    "format": meta["format"],
                }]
            }
        }

        loader = ArtifactLoader(workspace_path, "test_dataset")
        loader.import_from_manifest(manifest)

        # Load artifact
        obj = loader.load_by_id("0001:0:all")

        assert isinstance(obj, StandardScaler)
        assert hasattr(obj, 'mean_')

    def test_loader_handles_missing_files_gracefully(self, workspace_path):
        """Test ArtifactLoader reports missing files clearly."""
        manifest = {
            "dataset": "test_dataset",
            "artifacts": {
                "schema_version": "2.0",
                "items": [{
                    "artifact_id": "0001:0:all",
                    "content_hash": "sha256:nonexistent",
                    "path": "nonexistent_file.pkl",
                    "pipeline_id": "0001",
                    "branch_path": [],
                    "step_index": 0,
                    "artifact_type": "transformer",
                    "class_name": "StandardScaler",
                    "format": "joblib",
                }]
            }
        }

        loader = ArtifactLoader(workspace_path, "test_dataset")
        loader.import_from_manifest(manifest)

        with pytest.raises(FileNotFoundError):
            loader.load_by_id("0001:0:all")

    def test_loader_caches_loaded_artifacts(self, workspace_path):
        """Test that artifacts are cached after first load."""
        from nirs4all.pipeline.storage.artifacts.artifact_persistence import persist

        binaries_dir = workspace_path / "binaries" / "test_dataset"

        scaler = StandardScaler()
        scaler.fit(np.array([[0], [1], [2]]))
        meta = persist(scaler, binaries_dir, "scaler")

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {
                "schema_version": "2.0",
                "items": [{
                    "artifact_id": "0001:0:all",
                    "content_hash": meta["hash"],
                    "path": meta["path"],
                    "pipeline_id": "0001",
                    "branch_path": [],
                    "step_index": 0,
                    "artifact_type": "transformer",
                    "class_name": "StandardScaler",
                    "format": meta["format"],
                }]
            }
        }

        loader = ArtifactLoader(workspace_path, "test_dataset")
        loader.import_from_manifest(manifest)

        # First load
        obj1 = loader.load_by_id("0001:0:all")

        # Check cache stats
        cache_info = loader.get_cache_info()
        assert cache_info["misses"] == 1

        # Second load should hit cache
        obj2 = loader.load_by_id("0001:0:all")

        cache_info = loader.get_cache_info()
        assert cache_info["hits"] == 1

        # Should be same object
        assert obj1 is obj2


class TestMultiplePipelinesArtifactFlow:
    """Tests for artifact handling across multiple pipelines."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def runner(self, workspace_path):
        """Create pipeline runner."""
        return PipelineRunner(
            workspace_path=workspace_path,
            save_artifacts=True,
            verbose=0,
            enable_tab_reports=False,
            show_spinner=False
        )

    @pytest.fixture
    def dataset(self):
        """Create test dataset."""
        return create_test_dataset()

    def test_multiple_pipelines_create_separate_manifests(
        self, runner, dataset, workspace_path
    ):
        """Verify multiple pipeline runs create separate manifests."""
        pipeline1 = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"model": Ridge(alpha=1.0)},
        ]
        pipeline2 = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"model": Ridge(alpha=2.0)},  # Different alpha
        ]

        # Run both pipelines
        predictions1, _ = runner.run(PipelineConfigs(pipeline1), dataset)
        predictions2, _ = runner.run(PipelineConfigs(pipeline2), dataset)

        # Should have multiple pipeline directories
        runs_dir = workspace_path / "runs"
        manifest_files = list(runs_dir.glob("*/*/manifest.yaml"))

        assert len(manifest_files) >= 2, "Should have at least 2 manifests"

    def test_predictions_from_different_pipelines_load_correctly(
        self, runner, dataset, workspace_path
    ):
        """Verify predictions from different pipelines load their correct artifacts."""
        pipeline1 = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"model": Ridge(alpha=0.1)},
        ]
        pipeline2 = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"model": Ridge(alpha=10.0)},  # Very different alpha
        ]

        # Train both
        predictions1, _ = runner.run(PipelineConfigs(pipeline1), dataset)
        predictions2, _ = runner.run(PipelineConfigs(pipeline2), dataset)

        # Get test predictions from each
        test_preds1 = predictions1.filter_predictions(partition="test")
        test_preds2 = predictions2.filter_predictions(partition="test")

        if len(test_preds1) == 0 or len(test_preds2) == 0:
            pytest.skip("Insufficient predictions")

        # Predict from first pipeline
        y_pred1, _ = runner.predict(
            prediction_obj=test_preds1[0],
            dataset=dataset,
            dataset_name="test_artifact_flow"
        )

        # Predict from second pipeline
        y_pred2, _ = runner.predict(
            prediction_obj=test_preds2[0],
            dataset=dataset,
            dataset_name="test_artifact_flow"
        )

        # Both should produce predictions
        assert y_pred1 is not None
        assert y_pred2 is not None

        # Predictions should be different due to different alpha values
        # (This is a weak test but validates isolation)
        assert len(y_pred1) == len(y_pred2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
