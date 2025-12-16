"""
Integration tests for Phase 2: Pipeline Core with Manifest Manager.

Tests the integration between SimulationSaver, ManifestManager, and PipelineRunner
using the new flat sequential structure (0001_hash, 0002_hash, etc.).
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

from nirs4all.pipeline.storage.io import SimulationSaver
from nirs4all.pipeline.storage.manifest_manager import ManifestManager
from nirs4all.pipeline.storage.artifacts.artifact_persistence import persist, load


@pytest.fixture
def results_dir():
    """Create temporary results directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def saver(results_dir):
    """Create SimulationSaver instance."""
    return SimulationSaver(results_dir)


@pytest.fixture
def manifest_manager(results_dir):
    """Create ManifestManager instance."""
    return ManifestManager(results_dir)


class TestSimulationSaverPersist:
    """Test SimulationSaver with new persist_artifact method."""

    def test_persist_artifact_creates_content_addressed_storage(self, saver):
        """Test that persist_artifact uses content-addressed storage."""
        # Register with pipeline_id
        saver.register("0001_test_abc123")

        # Create and persist object
        scaler = StandardScaler()
        scaler.fit(np.array([[0], [1], [2]]))

        artifact = saver.persist_artifact(
            step_number=0,
            name="scaler_0",
            obj=scaler
        )

        # Check artifact metadata
        assert "hash" in artifact
        assert artifact["hash"].startswith("sha256:")
        assert artifact["name"] == "scaler_0"
        assert artifact["step"] == 0
        assert "path" in artifact
        assert artifact["format"] in ['sklearn_pickle', 'joblib']

        # Check file exists in _binaries location
        artifact_path = saver.base_path / "_binaries" / artifact["path"]
        assert artifact_path.exists()

    def test_persist_artifact_deduplication(self, saver):
        """Test that identical objects share same storage."""
        saver.register("0001_test_abc123")

        # Create identical scalers
        scaler1 = StandardScaler()
        scaler1.fit(np.array([[0], [1], [2]]))

        scaler2 = StandardScaler()
        scaler2.fit(np.array([[0], [1], [2]]))

        artifact1 = saver.persist_artifact(0, "scaler_0", scaler1)
        artifact2 = saver.persist_artifact(1, "scaler_1", scaler2)

        # Same content → same hash → same file
        assert artifact1["hash"] == artifact2["hash"]
        assert artifact1["path"] == artifact2["path"]

    def test_persist_artifact_saves_file(self, saver):
        """Test that persist_artifact saves files correctly."""
        saver.register("0001_test_abc123")

        scaler = StandardScaler()
        scaler.fit(np.array([[0], [1]]))

        artifact = saver.persist_artifact(0, "scaler_0", scaler)

        # Verify artifact was saved with correct structure
        assert artifact["name"] == "scaler_0"
        assert "hash" in artifact
        assert "path" in artifact

        # Verify the file was actually saved
        artifact_path = saver.base_path / "_binaries" / artifact["path"]
        assert artifact_path.exists()


class TestManifestManagerIntegration:
    """Test ManifestManager integration with pipeline workflow."""

    def test_create_pipeline_with_artifacts(self, manifest_manager, results_dir):
        """Test complete workflow: create pipeline, add artifacts."""
        # Create pipeline with hash
        config = {"steps": [{"class": "sklearn.preprocessing.StandardScaler"}]}
        pipeline_id, pipeline_dir = manifest_manager.create_pipeline(
            "test_pipeline", "test_dataset", config, "abc123"
        )

        # Verify pipeline was created
        assert pipeline_id.startswith("0001_")
        assert pipeline_dir.exists()

        # Persist artifacts
        artifacts_dir = results_dir / "_binaries"
        scaler = StandardScaler()
        scaler.fit(np.array([[0], [1], [2]]))

        artifact = persist(scaler, artifacts_dir, "scaler_0")
        artifact["step"] = 0

        # Add to manifest
        manifest_manager.append_artifacts(pipeline_id, [artifact])

        # Load manifest and verify
        manifest = manifest_manager.load_manifest(pipeline_id)
        # v2 schema has artifacts as {"items": [...], "schema_version": "2.0"}
        artifacts_items = manifest["artifacts"]["items"]
        assert len(artifacts_items) == 1
        assert artifacts_items[0]["name"] == "scaler_0"
        assert artifacts_items[0]["step"] == 0

    def test_load_artifacts_from_manifest(self, manifest_manager, results_dir):
        """Test loading artifacts via manifest."""
        # Create pipeline
        config = {"steps": []}
        pipeline_id, _ = manifest_manager.create_pipeline(
            "test_pipeline", "test_dataset", config, "abc123"
        )

        # Persist artifacts
        artifacts_dir = results_dir / "_binaries"
        scaler = StandardScaler()
        X_train = np.array([[0], [1], [2], [3]])
        scaler.fit(X_train)

        artifact = persist(scaler, artifacts_dir, "scaler_0")
        artifact["step"] = 0

        manifest_manager.append_artifacts(pipeline_id, [artifact])

        # Load manifest and artifact
        manifest = manifest_manager.load_manifest(pipeline_id)
        # v2 schema has artifacts as {"items": [...], "schema_version": "2.0"}
        artifacts_items = manifest["artifacts"]["items"]
        loaded_scaler = load(artifacts_items[0], results_dir)

        # Verify loaded scaler works
        assert hasattr(loaded_scaler, 'mean_')
        np.testing.assert_array_almost_equal(loaded_scaler.mean_, scaler.mean_)


class TestPipelineWorkflow:
    """Test complete pipeline workflow with new flat architecture."""

    def test_complete_workflow(self, saver, manifest_manager):
        """Test: create manifest → register saver → persist artifacts → load."""
        # 1. Create manifest with sequential numbering
        config = {
            "steps": [
                {"class": "sklearn.preprocessing.StandardScaler"},
                {"class": "sklearn.svm.SVC", "params": {"kernel": "rbf"}}
            ]
        }
        pipeline_id, pipeline_dir = manifest_manager.create_pipeline(
            "svm_baseline", "corn_m5", config, "def456"
        )

        # Verify sequential numbering
        assert pipeline_id.startswith("0001_")
        assert "svm_baseline" in pipeline_id
        assert "def456" in pipeline_id

        # 2. Register saver with pipeline_id
        saver.register(pipeline_id)

        # 3. Persist artifacts during "training"
        scaler = StandardScaler()
        scaler.fit(np.array([[0, 0], [1, 1], [2, 2]]))

        artifact1 = saver.persist_artifact(0, "StandardScaler_0", scaler)

        from sklearn.svm import SVC
        model = SVC(kernel='rbf')
        model.fit(np.array([[0, 0], [1, 1]]), np.array([0, 1]))

        artifact2 = saver.persist_artifact(1, "SVC_1_model", model)

        # 4. Add artifacts to manifest
        manifest_manager.append_artifacts(pipeline_id, [artifact1, artifact2])

        # 5. Load manifest
        manifest = manifest_manager.load_manifest(pipeline_id)

        # Verify manifest structure
        assert manifest["pipeline_id"] == pipeline_id
        assert manifest["name"] == "svm_baseline"
        assert manifest["dataset"] == "corn_m5"
        # v2 schema has artifacts as {"items": [...], "schema_version": "2.0"}
        artifacts_items = manifest["artifacts"]["items"]
        assert len(artifacts_items) == 2
        assert artifacts_items[0]["name"] == "StandardScaler_0"
        assert artifacts_items[1]["name"] == "SVC_1_model"

        # 6. Load artifacts for prediction
        loaded_scaler = load(artifacts_items[0], saver.base_path)
        loaded_model = load(artifacts_items[1], saver.base_path)

        # Verify they work
        assert hasattr(loaded_scaler, 'mean_')
        assert hasattr(loaded_model, 'support_vectors_')

    def test_multiple_pipelines_sequential_numbering(self, manifest_manager, saver):
        """Test that multiple pipelines get sequential numbers."""
        config = {"steps": []}

        # Create three pipelines
        id1, _ = manifest_manager.create_pipeline("pipeline1", "dataset1", config, "hash1")
        id2, _ = manifest_manager.create_pipeline("pipeline2", "dataset2", config, "hash2")
        id3, _ = manifest_manager.create_pipeline("pipeline3", "dataset1", config, "hash3")

        # Verify sequential numbering
        assert id1.startswith("0001_")
        assert id2.startswith("0002_")
        assert id3.startswith("0003_")

        # Register savers and persist artifacts
        for pipeline_id in [id1, id2, id3]:
            saver_instance = SimulationSaver(manifest_manager.results_dir)
            saver_instance.register(pipeline_id)

            scaler = StandardScaler()
            scaler.fit(np.array([[0], [1]]))
            artifact = saver_instance.persist_artifact(0, "scaler", scaler)

            manifest_manager.append_artifacts(pipeline_id, [artifact])

        # Verify all manifests exist and are correct
        manifest1 = manifest_manager.load_manifest(id1)
        manifest2 = manifest_manager.load_manifest(id2)
        manifest3 = manifest_manager.load_manifest(id3)

        assert manifest1["name"] == "pipeline1"
        assert manifest2["name"] == "pipeline2"
        assert manifest3["name"] == "pipeline3"

        # v2 schema has artifacts as {"items": [...], "schema_version": "2.0"}
        assert len(manifest1["artifacts"]["items"]) == 1
        assert len(manifest2["artifacts"]["items"]) == 1
        assert len(manifest3["artifacts"]["items"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
