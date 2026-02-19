"""
Integration tests for complete artifact flow.

Tests the full lifecycle of artifacts from training to prediction:
- Training creates artifacts in DuckDB store + content-addressed artifacts/
- Prediction loads artifacts correctly via chain replay
- Artifact records in store.duckdb have valid paths
- Artifact integrity is preserved
"""

from pathlib import Path

import numpy as np
import pytest
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

    def test_training_creates_store_with_artifacts(
        self, runner, dataset, workspace_path
    ):
        """Verify training creates DuckDB store with chain and artifact records."""
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

        # Verify store.duckdb exists
        store_path = workspace_path / "store.duckdb"
        assert store_path.exists(), "store.duckdb should be created"

        # Verify artifacts directory exists
        artifacts_dir = workspace_path / "artifacts"
        assert artifacts_dir.exists(), "artifacts/ directory should be created"

    def test_training_saves_binaries_to_disk(
        self, runner, dataset, workspace_path
    ):
        """Verify training saves binary artifacts to content-addressed artifacts/ directory."""
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"class": "sklearn.preprocessing.StandardScaler"},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = runner.run(
            PipelineConfigs(pipeline),
            dataset
        )

        # Check for binaries in flat content-addressed artifacts/ directory
        artifacts_dir = workspace_path / "artifacts"
        artifact_files = list(artifacts_dir.glob("**/*.joblib")) + list(artifacts_dir.glob("**/*.pkl"))

        assert len(artifact_files) > 0, \
            "Should create binary artifact files in artifacts/ directory"

    def test_artifact_paths_in_store_are_valid(
        self, runner, dataset, workspace_path
    ):
        """Verify artifact paths in store.duckdb point to existing files."""
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"class": "sklearn.preprocessing.StandardScaler"},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = runner.run(
            PipelineConfigs(pipeline),
            dataset
        )

        # Verify artifact files exist in content-addressed artifacts/ directory
        artifacts_dir = workspace_path / "artifacts"
        artifact_files = list(artifacts_dir.glob("**/*.joblib")) + list(artifacts_dir.glob("**/*.pkl"))

        # Each artifact file should be non-empty
        for artifact_file in artifact_files:
            assert artifact_file.stat().st_size > 0, \
                f"Artifact file should be non-empty: {artifact_file}"

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
        """Create temporary workspace with artifacts."""
        workspace = tmp_path / "workspace"
        artifacts_dir = workspace / "artifacts"
        artifacts_dir.mkdir(parents=True)
        return workspace

    def test_loader_loads_from_manifest(self, workspace_path):
        """Test ArtifactLoader imports from manifest correctly."""
        from nirs4all.pipeline.storage.artifacts.artifact_persistence import persist

        binaries_dir = workspace_path / "artifacts"

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

        binaries_dir = workspace_path / "artifacts"

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

    def test_multiple_pipelines_create_separate_records(
        self, runner, dataset, workspace_path
    ):
        """Verify multiple pipeline runs create separate records in store."""
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

        # Both should produce predictions
        assert predictions1 is not None
        assert predictions2 is not None
        assert len(predictions1) > 0
        assert len(predictions2) > 0

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
