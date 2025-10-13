"""
Integration tests for Phase 2: Pipeline Core with Manifest Manager.

Tests the integration between SimulationSaver, ManifestManager, and PipelineRunner.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

from nirs4all.pipeline.io import SimulationSaver
from nirs4all.pipeline.manifest_manager import ManifestManager
from nirs4all.utils.serializer import persist, load


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
        saver.register("test_dataset", "test_pipeline", "train")

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

        # Check file exists in content-addressed location
        artifact_path = saver.base_path / "artifacts" / artifact["path"]
        assert artifact_path.exists()

    def test_persist_artifact_deduplication(self, saver):
        """Test that identical objects share same storage."""
        saver.register("test_dataset", "test_pipeline", "train")

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

    def test_persist_artifact_updates_metadata(self, saver):
        """Test that persist_artifact updates metadata correctly."""
        saver.register("test_dataset", "test_pipeline", "train")

        scaler = StandardScaler()
        scaler.fit(np.array([[0], [1]]))

        artifact = saver.persist_artifact(0, "scaler_0", scaler)

        metadata = saver.get_metadata()
        assert "0" in metadata["binaries"]
        assert len(metadata["binaries"]["0"]) == 1
        assert metadata["binaries"]["0"][0]["name"] == "scaler_0"
        assert metadata["binaries"]["0"][0]["hash"] == artifact["hash"]


class TestManifestManagerIntegration:
    """Test ManifestManager integration with pipeline workflow."""

    def test_create_pipeline_with_artifacts(self, manifest_manager, results_dir):
        """Test complete workflow: create pipeline, add artifacts."""
        # Create pipeline
        config = {"steps": [{"class": "sklearn.preprocessing.StandardScaler"}]}
        uid = manifest_manager.create_pipeline("test_pipeline", "test_dataset", config)

        # Persist artifacts
        artifacts_dir = results_dir / "artifacts" / "objects"
        scaler = StandardScaler()
        scaler.fit(np.array([[0], [1], [2]]))

        artifact = persist(scaler, artifacts_dir, "scaler_0")
        artifact["step"] = 0

        # Add to manifest
        manifest_manager.append_artifacts(uid, [artifact])

        # Load manifest and verify
        manifest = manifest_manager.load_manifest(uid)
        assert len(manifest["artifacts"]) == 1
        assert manifest["artifacts"][0]["name"] == "scaler_0"
        assert manifest["artifacts"][0]["step"] == 0

    def test_load_artifacts_from_manifest(self, manifest_manager, results_dir):
        """Test loading artifacts via manifest."""
        # Create pipeline
        config = {"steps": []}
        uid = manifest_manager.create_pipeline("test_pipeline", "test_dataset", config)

        # Persist artifacts
        artifacts_dir = results_dir / "artifacts" / "objects"
        scaler = StandardScaler()
        X_train = np.array([[0], [1], [2], [3]])
        scaler.fit(X_train)

        artifact = persist(scaler, artifacts_dir, "scaler_0")
        artifact["step"] = 0

        manifest_manager.append_artifacts(uid, [artifact])

        # Load manifest and artifact
        manifest = manifest_manager.load_manifest(uid)
        loaded_scaler = load(manifest["artifacts"][0], results_dir)

        # Verify loaded scaler works
        assert hasattr(loaded_scaler, 'mean_')
        np.testing.assert_array_almost_equal(loaded_scaler.mean_, scaler.mean_)


class TestPipelineWorkflow:
    """Test complete pipeline workflow with new architecture."""

    def test_complete_workflow(self, saver, manifest_manager):
        """Test: register → persist artifacts → create manifest → load."""
        # 1. Register pipeline
        saver.register("corn_m5", "svm_baseline", "train")

        # 2. Create manifest
        config = {
            "steps": [
                {"class": "sklearn.preprocessing.StandardScaler"},
                {"class": "sklearn.svm.SVC", "params": {"kernel": "rbf"}}
            ]
        }
        uid = manifest_manager.create_pipeline("svm_baseline", "corn_m5", config)

        # 3. Persist artifacts during "training"
        scaler = StandardScaler()
        scaler.fit(np.array([[0, 0], [1, 1], [2, 2]]))

        artifact1 = saver.persist_artifact(0, "StandardScaler_0", scaler)

        from sklearn.svm import SVC
        model = SVC(kernel='rbf')
        model.fit(np.array([[0, 0], [1, 1]]), np.array([0, 1]))

        artifact2 = saver.persist_artifact(1, "SVC_1_model", model)

        # 4. Add artifacts to manifest
        manifest_manager.append_artifacts(uid, [artifact1, artifact2])

        # 5. Load manifest
        manifest = manifest_manager.load_manifest(uid)

        # Verify manifest structure
        assert manifest["uid"] == uid
        assert manifest["name"] == "svm_baseline"
        assert manifest["dataset"] == "corn_m5"
        assert len(manifest["artifacts"]) == 2
        assert manifest["artifacts"][0]["name"] == "StandardScaler_0"
        assert manifest["artifacts"][1]["name"] == "SVC_1_model"

        # 6. Load artifacts for prediction
        loaded_scaler = load(manifest["artifacts"][0], saver.base_path)
        loaded_model = load(manifest["artifacts"][1], saver.base_path)

        # Verify they work
        assert hasattr(loaded_scaler, 'mean_')
        assert hasattr(loaded_model, 'support_vectors_')

    def test_dataset_index_lookup(self, manifest_manager):
        """Test looking up pipeline by dataset and name."""
        # Create pipeline
        config = {"steps": []}
        uid = manifest_manager.create_pipeline("pipeline1", "dataset1", config)

        # Lookup via dataset index
        found_uid = manifest_manager.get_pipeline_uid("dataset1", "pipeline1")
        assert found_uid == uid

        # Load manifest via found UID
        manifest = manifest_manager.load_manifest(found_uid)
        assert manifest["name"] == "pipeline1"
        assert manifest["dataset"] == "dataset1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
