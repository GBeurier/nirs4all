"""
Unit tests for the ManifestManager.

Tests sequential pipeline numbering (0001_hash, 0002_hash, etc.)
and manifest YAML persistence in flat structure.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

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

    def test_creates_artifacts_directory(self, results_dir):
        """Test that initialization creates _binaries directory."""
        manager = ManifestManager(results_dir)

        assert manager.artifacts_dir.exists()
        assert (results_dir / "_binaries").exists()
class TestGetNextPipelineNumber:
    """Test sequential pipeline numbering."""

    def test_first_pipeline_is_0001(self, manager):
        """Test that first pipeline gets number 0001."""
        num = manager.get_next_pipeline_number()
        assert num == 1

    def test_increments_with_existing_pipelines(self, manager):
        """Test that numbering increments correctly."""
        # Create some pipeline directories
        (manager.results_dir / "0001_abc123").mkdir()
        (manager.results_dir / "0002_def456").mkdir()

        num = manager.get_next_pipeline_number()
        assert num == 3

    def test_ignores_artifacts_directory(self, manager):
        """Test that _binaries directory doesn't affect numbering."""
        # _binaries directory is created in __init__
        num = manager.get_next_pipeline_number()
        assert num == 1

    def test_ignores_non_numbered_directories(self, manager):
        """Test that non-numbered directories are ignored."""
        (manager.results_dir / "0001_abc123").mkdir()
        (manager.results_dir / "some_other_dir").mkdir()

        num = manager.get_next_pipeline_number()
        assert num == 2


class TestCreatePipeline:
    """Test pipeline creation with sequential numbering."""

    def test_create_pipeline_returns_pipeline_id_and_dir(self, manager):
        """Test that create_pipeline returns (pipeline_id, pipeline_dir) tuple."""
        config = {"steps": [{"class": "sklearn.preprocessing.StandardScaler"}]}
        pipeline_hash = "abc123"

        result = manager.create_pipeline("test_pipeline", "test_dataset", config, pipeline_hash)

        assert isinstance(result, tuple)
        assert len(result) == 2

        pipeline_id, pipeline_dir = result
        assert isinstance(pipeline_id, str)
        assert isinstance(pipeline_dir, Path)

    def test_first_pipeline_gets_0001(self, manager):
        """Test that first pipeline gets 0001 prefix."""
        config = {"steps": []}
        pipeline_hash = "abc123"

        pipeline_id, pipeline_dir = manager.create_pipeline("test_pipeline", "test_dataset", config, pipeline_hash)

        assert pipeline_id.startswith("0001_")
        assert "abc123" in pipeline_id

    def test_sequential_numbering(self, manager):
        """Test that pipelines get sequential numbers."""
        config = {"steps": []}

        id1, _ = manager.create_pipeline("pipeline1", "dataset", config, "hash1")
        id2, _ = manager.create_pipeline("pipeline2", "dataset", config, "hash2")
        id3, _ = manager.create_pipeline("pipeline3", "dataset", config, "hash3")

        assert id1.startswith("0001_")
        assert id2.startswith("0002_")
        assert id3.startswith("0003_")

    def test_custom_name_in_pipeline_id(self, manager):
        """Test that custom name is included in pipeline_id."""
        config = {"steps": []}
        pipeline_hash = "abc123"

        pipeline_id, _ = manager.create_pipeline("my_custom_pipeline", "dataset", config, pipeline_hash)

        assert "my_custom_pipeline" in pipeline_id
        assert pipeline_id == "0001_my_custom_pipeline_abc123"

    def test_generic_name_excluded_from_id(self, manager):
        """Test that generic 'pipeline' name is not included."""
        config = {"steps": []}
        pipeline_hash = "abc123"

        pipeline_id, _ = manager.create_pipeline("pipeline", "dataset", config, pipeline_hash)

        # Should be "0001_abc123" not "0001_pipeline_abc123"
        parts = pipeline_id.split("_")
        assert len(parts) == 2  # number and hash only
        assert pipeline_id == "0001_abc123"

    def test_creates_pipeline_directory(self, manager):
        """Test that pipeline directory is created."""
        config = {"steps": []}
        pipeline_hash = "abc123"

        pipeline_id, pipeline_dir = manager.create_pipeline("test", "dataset", config, pipeline_hash)

        assert pipeline_dir.exists()
        assert pipeline_dir.is_dir()
        assert pipeline_dir == manager.results_dir / pipeline_id

    def test_creates_manifest_file(self, manager):
        """Test that manifest.yaml is created."""
        config = {"steps": []}
        pipeline_hash = "abc123"

        pipeline_id, pipeline_dir = manager.create_pipeline("test", "dataset", config, pipeline_hash)

        manifest_path = pipeline_dir / "manifest.yaml"
        assert manifest_path.exists()

    def test_manifest_structure(self, manager):
        """Test that manifest has correct structure."""
        config = {"steps": [{"class": "sklearn.preprocessing.StandardScaler"}]}
        metadata = {"n_samples": 100, "n_features": 50}
        pipeline_hash = "abc123"

        pipeline_id, _ = manager.create_pipeline("test_pipeline", "test_dataset", config, pipeline_hash, metadata)
        manifest = manager.load_manifest(pipeline_id)

        assert "uid" in manifest  # Internal UID for backwards compatibility
        assert manifest["pipeline_id"] == pipeline_id
        assert manifest["name"] == "test_pipeline"
        assert manifest["dataset"] == "test_dataset"
        assert "created_at" in manifest
        assert manifest["version"] == "1.0"
        assert manifest["pipeline"] == config
        assert manifest["metadata"] == metadata
        assert manifest["artifacts"] == []
        assert manifest["predictions"] == []


class TestSaveLoadManifest:
    """Test manifest save/load operations."""

    def test_save_and_load_manifest(self, manager):
        """Test saving and loading manifest."""
        config = {"steps": []}
        pipeline_id, _ = manager.create_pipeline("test", "dataset", config, "hash1")

        # Load and verify
        manifest = manager.load_manifest(pipeline_id)
        assert manifest["pipeline_id"] == pipeline_id
        assert manifest["name"] == "test"

    def test_update_manifest(self, manager):
        """Test updating manifest."""
        config = {"steps": []}
        pipeline_id, _ = manager.create_pipeline("test", "dataset", config, "hash1")

        # Load, modify, save
        manifest = manager.load_manifest(pipeline_id)
        manifest["metadata"]["updated"] = True
        manager.save_manifest(pipeline_id, manifest)

        # Load again and verify
        updated_manifest = manager.load_manifest(pipeline_id)
        assert updated_manifest["metadata"]["updated"] is True


class TestArtifactManagement:
    """Test artifact management in manifests."""

    def test_append_artifacts(self, manager):
        """Test appending artifacts to manifest."""
        config = {"steps": []}
        pipeline_id, _ = manager.create_pipeline("test", "dataset", config, "hash1")

        artifacts = ["abc123", "def456"]
        manager.append_artifacts(pipeline_id, artifacts)

        manifest = manager.load_manifest(pipeline_id)
        assert manifest["artifacts"] == artifacts

    def test_append_artifacts_multiple_times(self, manager):
        """Test appending artifacts multiple times accumulates."""
        config = {"steps": []}
        pipeline_id, _ = manager.create_pipeline("test", "dataset", config, "hash1")

        manager.append_artifacts(pipeline_id, ["abc123"])
        manager.append_artifacts(pipeline_id, ["def456"])

        manifest = manager.load_manifest(pipeline_id)
        assert "abc123" in manifest["artifacts"]
        assert "def456" in manifest["artifacts"]


class TestPredictionManagement:
    """Test prediction management in manifests."""

    def test_append_prediction(self, manager):
        """Test appending prediction to manifest."""
        config = {"steps": []}
        pipeline_id, _ = manager.create_pipeline("test", "dataset", config, "hash1")

        prediction = {"model": "PLS", "score": 0.95}
        manager.append_prediction(pipeline_id, prediction)

        manifest = manager.load_manifest(pipeline_id)
        assert len(manifest["predictions"]) == 1
        assert manifest["predictions"][0] == prediction


class TestPipelineExists:
    """Test pipeline existence checks."""

    def test_pipeline_exists(self, manager):
        """Test checking if pipeline exists."""
        config = {"steps": []}
        pipeline_id, _ = manager.create_pipeline("test", "dataset", config, "hash1")

        assert manager.pipeline_exists(pipeline_id)

    def test_pipeline_not_exists(self, manager):
        """Test checking non-existent pipeline."""
        assert not manager.pipeline_exists("0001_nonexistent")


class TestGetPipelinePath:
    """Test getting pipeline paths."""

    def test_get_pipeline_path(self, manager):
        """Test getting pipeline path."""
        config = {"steps": []}
        pipeline_id, pipeline_dir = manager.create_pipeline("test", "dataset", config, "hash1")

        path = manager.get_pipeline_path(pipeline_id)
        assert path == pipeline_dir
        assert path == manager.results_dir / pipeline_id


class TestDeletePipeline:
    """Test pipeline deletion."""

    def test_delete_pipeline(self, manager):
        """Test deleting a pipeline."""
        config = {"steps": []}
        pipeline_id, pipeline_dir = manager.create_pipeline("test", "dataset", config, "hash1")

        assert pipeline_dir.exists()

        manager.delete_pipeline(pipeline_id)

        assert not pipeline_dir.exists()
        assert not manager.pipeline_exists(pipeline_id)


class TestListAllPipelines:
    """Test listing all pipelines."""

    def test_list_all_pipelines_empty(self, manager):
        """Test listing pipelines when none exist."""
        pipelines = manager.list_all_pipelines()
        assert pipelines == []

    def test_list_all_pipelines(self, manager):
        """Test listing all pipelines."""
        config = {"steps": []}

        id1, _ = manager.create_pipeline("pipeline1", "dataset", config, "hash1")
        id2, _ = manager.create_pipeline("pipeline2", "dataset", config, "hash2")

        pipelines = manager.list_all_pipelines()

        assert len(pipelines) == 2
        pipeline_ids = [p["pipeline_id"] for p in pipelines]
        assert id1 in pipeline_ids
        assert id2 in pipeline_ids

    def test_list_all_pipelines_contains_info(self, manager):
        """Test that listed pipelines contain useful information."""
        config = {"steps": [{"class": "StandardScaler"}]}
        metadata = {"n_samples": 100}

        pipeline_id, _ = manager.create_pipeline("test", "dataset", config, "hash1", metadata)

        pipelines = manager.list_all_pipelines()

        assert len(pipelines) == 1
        p = pipelines[0]

        assert p["pipeline_id"] == pipeline_id
        assert p["name"] == "test"
        assert p["dataset"] == "dataset"
        assert "created_at" in p


class TestMultipleDatasets:
    """Test handling multiple datasets in same results_dir."""

    def test_multiple_datasets_in_flat_structure(self, manager):
        """Test that multiple datasets can coexist in flat structure."""
        config = {"steps": []}

        # Create pipelines for different datasets
        id1, _ = manager.create_pipeline("pipeline1", "dataset_A", config, "hash1")
        id2, _ = manager.create_pipeline("pipeline2", "dataset_B", config, "hash2")
        id3, _ = manager.create_pipeline("pipeline3", "dataset_A", config, "hash3")

        # All should have sequential numbers
        assert id1.startswith("0001_")
        assert id2.startswith("0002_")
        assert id3.startswith("0003_")

        # All should exist
        assert manager.pipeline_exists(id1)
        assert manager.pipeline_exists(id2)
        assert manager.pipeline_exists(id3)

        # Can differentiate by dataset in manifest
        manifest1 = manager.load_manifest(id1)
        manifest2 = manager.load_manifest(id2)
        assert manifest1["dataset"] == "dataset_A"
        assert manifest2["dataset"] == "dataset_B"
