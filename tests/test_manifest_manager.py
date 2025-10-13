"""
Unit tests for the ManifestManager.

Tests UID-based pipeline management, dataset indexing,
and manifest YAML persistence.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import yaml

from nirs4all.pipeline.manifest_manager import ManifestManager


@pytest.fixture
def results_dir():
    """Create temporary results directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def manager(results_dir):
    """Create ManifestManager instance."""
    return ManifestManager(results_dir)


class TestManifestManagerInit:
    """Test ManifestManager initialization."""

    def test_creates_directories(self, results_dir):
        """Test that initialization creates required directories."""
        manager = ManifestManager(results_dir)

        assert manager.artifacts_dir.exists()
        assert manager.pipelines_dir.exists()
        assert manager.datasets_dir.exists()

        assert (results_dir / "artifacts" / "objects").exists()
        assert (results_dir / "pipelines").exists()
        assert (results_dir / "datasets").exists()


class TestCreatePipeline:
    """Test pipeline creation."""

    def test_create_pipeline_returns_uid(self, manager):
        """Test that create_pipeline returns a valid UID."""
        config = {"steps": [{"class": "sklearn.preprocessing.StandardScaler"}]}
        uid = manager.create_pipeline("test_pipeline", "test_dataset", config)

        assert isinstance(uid, str)
        assert len(uid) == 36  # UUID format
        assert "-" in uid

    def test_create_pipeline_creates_manifest(self, manager):
        """Test that create_pipeline creates manifest file."""
        config = {"steps": [{"class": "sklearn.preprocessing.StandardScaler"}]}
        uid = manager.create_pipeline("test_pipeline", "test_dataset", config)

        manifest_path = manager.pipelines_dir / uid / "manifest.yaml"
        assert manifest_path.exists()

    def test_create_pipeline_manifest_structure(self, manager):
        """Test manifest has correct structure."""
        config = {"steps": [{"class": "sklearn.preprocessing.StandardScaler"}]}
        metadata = {"n_samples": 100, "n_features": 50}

        uid = manager.create_pipeline("test_pipeline", "test_dataset", config, metadata)
        manifest = manager.load_manifest(uid)

        assert manifest["uid"] == uid
        assert manifest["name"] == "test_pipeline"
        assert manifest["dataset"] == "test_dataset"
        assert "created_at" in manifest
        assert manifest["version"] == "1.0"
        assert manifest["pipeline"] == config
        assert manifest["metadata"] == metadata
        assert manifest["artifacts"] == []
        assert manifest["predictions"] == []

    def test_create_pipeline_registers_in_dataset(self, manager):
        """Test that pipeline is registered in dataset index."""
        config = {"steps": []}
        uid = manager.create_pipeline("test_pipeline", "test_dataset", config)

        index_path = manager.datasets_dir / "test_dataset" / "index.yaml"
        assert index_path.exists()

        with open(index_path, "r", encoding="utf-8") as f:
            index = yaml.safe_load(f)

        assert "pipelines" in index
        assert "test_pipeline" in index["pipelines"]
        assert index["pipelines"]["test_pipeline"] == uid


class TestSaveLoadManifest:
    """Test manifest save/load operations."""

    def test_save_and_load_manifest(self, manager):
        """Test saving and loading manifest."""
        uid = "test-uid-123"
        manifest = {
            "uid": uid,
            "name": "test",
            "dataset": "dataset1",
            "created_at": "2025-01-01T00:00:00Z",
            "version": "1.0",
            "pipeline": {},
            "metadata": {},
            "artifacts": [],
            "predictions": []
        }

        manager.save_manifest(uid, manifest)
        loaded = manager.load_manifest(uid)

        assert loaded == manifest

    def test_load_nonexistent_manifest(self, manager):
        """Test loading nonexistent manifest raises error."""
        with pytest.raises(FileNotFoundError):
            manager.load_manifest("nonexistent-uid")

    def test_update_manifest(self, manager):
        """Test updating manifest fields."""
        config = {"steps": []}
        uid = manager.create_pipeline("test", "dataset1", config)

        # Update metadata
        manager.update_manifest(uid, {"metadata": {"accuracy": 0.95}})

        manifest = manager.load_manifest(uid)
        assert manifest["metadata"] == {"accuracy": 0.95}


class TestArtifactManagement:
    """Test artifact management in manifests."""

    def test_append_artifacts(self, manager):
        """Test appending artifacts to manifest."""
        config = {"steps": []}
        uid = manager.create_pipeline("test", "dataset1", config)

        artifacts = [
            {
                "hash": "sha256:abc123",
                "name": "scaler_0",
                "path": "objects/ab/abc123.pkl",
                "format": "sklearn_pickle",
                "size": 1024,
                "step": 0
            },
            {
                "hash": "sha256:def456",
                "name": "model_1",
                "path": "objects/de/def456.pkl",
                "format": "sklearn_pickle",
                "size": 2048,
                "step": 1
            }
        ]

        manager.append_artifacts(uid, artifacts)

        manifest = manager.load_manifest(uid)
        assert len(manifest["artifacts"]) == 2
        assert manifest["artifacts"][0]["name"] == "scaler_0"
        assert manifest["artifacts"][1]["name"] == "model_1"

    def test_append_artifacts_multiple_times(self, manager):
        """Test appending artifacts in multiple calls."""
        config = {"steps": []}
        uid = manager.create_pipeline("test", "dataset1", config)

        artifact1 = [{"hash": "sha256:abc", "name": "a1", "path": "p1", "format": "pkl", "size": 10, "step": 0}]
        artifact2 = [{"hash": "sha256:def", "name": "a2", "path": "p2", "format": "pkl", "size": 20, "step": 1}]

        manager.append_artifacts(uid, artifact1)
        manager.append_artifacts(uid, artifact2)

        manifest = manager.load_manifest(uid)
        assert len(manifest["artifacts"]) == 2


class TestPredictionManagement:
    """Test prediction history management."""

    def test_append_prediction(self, manager):
        """Test appending prediction to manifest."""
        config = {"steps": []}
        uid = manager.create_pipeline("test", "dataset1", config)

        prediction = {
            "id": "pred_001",
            "timestamp": "2025-01-01T12:00:00Z",
            "input_hash": "sha256:input123",
            "output_hash": "sha256:output456"
        }

        manager.append_prediction(uid, prediction)

        manifest = manager.load_manifest(uid)
        assert len(manifest["predictions"]) == 1
        assert manifest["predictions"][0]["id"] == "pred_001"


class TestDatasetIndex:
    """Test dataset index operations."""

    def test_get_pipeline_uid(self, manager):
        """Test getting pipeline UID from dataset index."""
        config = {"steps": []}
        uid = manager.create_pipeline("pipeline1", "dataset1", config)

        found_uid = manager.get_pipeline_uid("dataset1", "pipeline1")
        assert found_uid == uid

    def test_get_pipeline_uid_nonexistent(self, manager):
        """Test getting nonexistent pipeline returns None."""
        uid = manager.get_pipeline_uid("nonexistent", "pipeline1")
        assert uid is None

    def test_list_pipelines(self, manager):
        """Test listing all pipelines for a dataset."""
        config = {"steps": []}
        uid1 = manager.create_pipeline("pipeline1", "dataset1", config)
        uid2 = manager.create_pipeline("pipeline2", "dataset1", config)
        uid3 = manager.create_pipeline("pipeline3", "dataset2", config)

        pipelines = manager.list_pipelines("dataset1")
        assert len(pipelines) == 2
        assert "pipeline1" in pipelines
        assert "pipeline2" in pipelines
        assert pipelines["pipeline1"] == uid1
        assert pipelines["pipeline2"] == uid2

        # Dataset2 should only have one pipeline
        pipelines2 = manager.list_pipelines("dataset2")
        assert len(pipelines2) == 1

    def test_list_pipelines_empty_dataset(self, manager):
        """Test listing pipelines for empty dataset."""
        pipelines = manager.list_pipelines("empty_dataset")
        assert pipelines == {}

    def test_register_in_dataset(self, manager):
        """Test manually registering pipeline in dataset."""
        manager.register_in_dataset("dataset1", "manual_pipeline", "uid-123")

        uid = manager.get_pipeline_uid("dataset1", "manual_pipeline")
        assert uid == "uid-123"

    def test_unregister_from_dataset(self, manager):
        """Test removing pipeline from dataset index."""
        config = {"steps": []}
        uid = manager.create_pipeline("pipeline1", "dataset1", config)

        manager.unregister_from_dataset("dataset1", "pipeline1")

        found_uid = manager.get_pipeline_uid("dataset1", "pipeline1")
        assert found_uid is None


class TestDeletePipeline:
    """Test pipeline deletion."""

    def test_delete_pipeline(self, manager):
        """Test deleting a pipeline."""
        config = {"steps": []}
        uid = manager.create_pipeline("pipeline1", "dataset1", config)

        # Verify it exists
        assert manager.pipeline_exists(uid)

        # Delete it
        manager.delete_pipeline(uid)

        # Verify it's gone
        assert not manager.pipeline_exists(uid)
        assert not (manager.pipelines_dir / uid).exists()

    def test_delete_pipeline_removes_from_index(self, manager):
        """Test that deletion removes pipeline from dataset index."""
        config = {"steps": []}
        uid = manager.create_pipeline("pipeline1", "dataset1", config)

        manager.delete_pipeline(uid)

        found_uid = manager.get_pipeline_uid("dataset1", "pipeline1")
        assert found_uid is None

    def test_delete_nonexistent_pipeline(self, manager):
        """Test deleting nonexistent pipeline doesn't crash."""
        # Should not raise error
        manager.delete_pipeline("nonexistent-uid")


class TestPipelineExists:
    """Test pipeline existence checks."""

    def test_pipeline_exists(self, manager):
        """Test checking if pipeline exists."""
        config = {"steps": []}
        uid = manager.create_pipeline("pipeline1", "dataset1", config)

        assert manager.pipeline_exists(uid)

    def test_pipeline_not_exists(self, manager):
        """Test checking nonexistent pipeline."""
        assert not manager.pipeline_exists("nonexistent-uid")


class TestGetPipelinePath:
    """Test getting pipeline paths."""

    def test_get_pipeline_path(self, manager):
        """Test getting pipeline directory path."""
        config = {"steps": []}
        uid = manager.create_pipeline("pipeline1", "dataset1", config)

        path = manager.get_pipeline_path(uid)
        assert path == manager.pipelines_dir / uid
        assert path.exists()


class TestListAllPipelines:
    """Test listing all pipelines."""

    def test_list_all_pipelines(self, manager):
        """Test listing all pipelines across datasets."""
        config = {"steps": []}
        uid1 = manager.create_pipeline("p1", "dataset1", config)
        uid2 = manager.create_pipeline("p2", "dataset1", config)
        uid3 = manager.create_pipeline("p3", "dataset2", config)

        all_pipelines = manager.list_all_pipelines()

        assert len(all_pipelines) == 3
        uids = [p["uid"] for p in all_pipelines]
        assert uid1 in uids
        assert uid2 in uids
        assert uid3 in uids

    def test_list_all_pipelines_contains_info(self, manager):
        """Test that list_all_pipelines returns complete info."""
        config = {"steps": []}
        uid = manager.create_pipeline("pipeline1", "dataset1", config)

        # Add some artifacts
        artifacts = [{"hash": "sha256:abc", "name": "a1", "path": "p1", "format": "pkl", "size": 10, "step": 0}]
        manager.append_artifacts(uid, artifacts)

        all_pipelines = manager.list_all_pipelines()
        pipeline_info = next(p for p in all_pipelines if p["uid"] == uid)

        assert pipeline_info["name"] == "pipeline1"
        assert pipeline_info["dataset"] == "dataset1"
        assert "created_at" in pipeline_info
        assert pipeline_info["num_artifacts"] == 1
        assert pipeline_info["num_predictions"] == 0

    def test_list_all_pipelines_empty(self, manager):
        """Test listing when no pipelines exist."""
        all_pipelines = manager.list_all_pipelines()
        assert all_pipelines == []


class TestMultipleDatasets:
    """Test operations with multiple datasets."""

    def test_multiple_datasets_separate_indexes(self, manager):
        """Test that different datasets have separate indexes."""
        config = {"steps": []}

        uid1 = manager.create_pipeline("pipeline1", "dataset1", config)
        uid2 = manager.create_pipeline("pipeline1", "dataset2", config)  # Same name, different dataset

        # Should have different UIDs
        assert uid1 != uid2

        # Each dataset should have its own index
        found_uid1 = manager.get_pipeline_uid("dataset1", "pipeline1")
        found_uid2 = manager.get_pipeline_uid("dataset2", "pipeline1")

        assert found_uid1 == uid1
        assert found_uid2 == uid2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
