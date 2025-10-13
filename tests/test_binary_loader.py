"""
Unit tests for updated BinaryLoader with manifest support.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from nirs4all.pipeline.binary_loader import BinaryLoader
from nirs4all.utils.serializer import persist


@pytest.fixture
def results_dir():
    """Create temporary results directory."""
    temp_dir = tempfile.mkdtemp()
    results_path = Path(temp_dir)
    (results_path / "artifacts" / "objects").mkdir(parents=True, exist_ok=True)
    yield results_path
    shutil.rmtree(temp_dir)


class TestBinaryLoaderWithManifests:
    """Test BinaryLoader with new manifest-based architecture."""

    def test_init_groups_by_step(self, results_dir):
        """Test that BinaryLoader groups artifacts by step number."""
        artifacts_dir = results_dir / "artifacts" / "objects"

        # Create artifacts
        scaler = StandardScaler()
        scaler.fit(np.array([[0], [1]]))
        artifact1 = persist(scaler, artifacts_dir, "scaler_0")
        artifact1["step"] = 0

        model = SVC()
        model.fit(np.array([[0, 0], [1, 1]]), np.array([0, 1]))
        artifact2 = persist(model, artifacts_dir, "model_1")
        artifact2["step"] = 1

        # Create loader
        loader = BinaryLoader([artifact1, artifact2], results_dir)

        # Verify grouping
        assert 0 in loader.artifacts_by_step
        assert 1 in loader.artifacts_by_step
        assert len(loader.artifacts_by_step[0]) == 1
        assert len(loader.artifacts_by_step[1]) == 1

    def test_get_step_binaries_loads_artifacts(self, results_dir):
        """Test loading artifacts for a specific step."""
        artifacts_dir = results_dir / "artifacts" / "objects"

        # Create and persist artifacts
        scaler = StandardScaler()
        X_train = np.array([[0], [1], [2], [3]])
        scaler.fit(X_train)

        artifact = persist(scaler, artifacts_dir, "StandardScaler_0")
        artifact["step"] = 0

        # Create loader and load binaries
        loader = BinaryLoader([artifact], results_dir)
        binaries = loader.get_step_binaries("0")

        # Verify
        assert len(binaries) == 1
        name, loaded_scaler = binaries[0]
        assert name == "StandardScaler_0"
        assert hasattr(loaded_scaler, 'mean_')
        np.testing.assert_array_almost_equal(loaded_scaler.mean_, scaler.mean_)

    def test_get_step_binaries_handles_step_formats(self, results_dir):
        """Test that get_step_binaries handles different step_id formats."""
        artifacts_dir = results_dir / "artifacts" / "objects"

        scaler = StandardScaler()
        scaler.fit(np.array([[0], [1]]))
        artifact = persist(scaler, artifacts_dir, "scaler")
        artifact["step"] = 0

        loader = BinaryLoader([artifact], results_dir)

        # All these formats should work
        binaries1 = loader.get_step_binaries("0")
        binaries2 = loader.get_step_binaries("0_0")
        binaries3 = loader.get_step_binaries(0)

        assert len(binaries1) == 1
        assert len(binaries2) == 1
        assert len(binaries3) == 1

    def test_get_step_binaries_caching(self, results_dir):
        """Test that loaded binaries are cached."""
        artifacts_dir = results_dir / "artifacts" / "objects"

        scaler = StandardScaler()
        scaler.fit(np.array([[0], [1]]))
        artifact = persist(scaler, artifacts_dir, "scaler")
        artifact["step"] = 0

        loader = BinaryLoader([artifact], results_dir)

        # First load
        binaries1 = loader.get_step_binaries("0")

        # Second load (should be cached)
        binaries2 = loader.get_step_binaries("0")

        # Should return same objects (from cache)
        assert binaries1 is binaries2

    def test_get_step_binaries_empty_step(self, results_dir):
        """Test loading binaries for non-existent step."""
        loader = BinaryLoader([], results_dir)
        binaries = loader.get_step_binaries("99")

        assert binaries == []

    def test_get_step_binaries_multiple_artifacts(self, results_dir):
        """Test loading multiple artifacts for same step."""
        artifacts_dir = results_dir / "artifacts" / "objects"

        # Create multiple artifacts for step 0
        scaler1 = StandardScaler()
        scaler1.fit(np.array([[0], [1]]))
        artifact1 = persist(scaler1, artifacts_dir, "scaler1")
        artifact1["step"] = 0

        scaler2 = StandardScaler()
        scaler2.fit(np.array([[2], [3]]))
        artifact2 = persist(scaler2, artifacts_dir, "scaler2")
        artifact2["step"] = 0

        loader = BinaryLoader([artifact1, artifact2], results_dir)
        binaries = loader.get_step_binaries("0")

        assert len(binaries) == 2
        names = [name for name, _ in binaries]
        assert "scaler1" in names
        assert "scaler2" in names

    def test_has_binaries_for_step(self, results_dir):
        """Test checking if binaries exist for a step."""
        artifacts_dir = results_dir / "artifacts" / "objects"

        scaler = StandardScaler()
        scaler.fit(np.array([[0], [1]]))
        artifact = persist(scaler, artifacts_dir, "scaler")
        artifact["step"] = 0

        loader = BinaryLoader([artifact], results_dir)

        assert loader.has_binaries_for_step(0)
        assert not loader.has_binaries_for_step(1)

    def test_clear_cache(self, results_dir):
        """Test clearing the cache."""
        artifacts_dir = results_dir / "artifacts" / "objects"

        scaler = StandardScaler()
        scaler.fit(np.array([[0], [1]]))
        artifact = persist(scaler, artifacts_dir, "scaler")
        artifact["step"] = 0

        loader = BinaryLoader([artifact], results_dir)

        # Load to populate cache
        loader.get_step_binaries("0")
        assert len(loader._cache) > 0

        # Clear cache
        loader.clear_cache()
        assert len(loader._cache) == 0

    def test_get_cache_info(self, results_dir):
        """Test getting cache information."""
        artifacts_dir = results_dir / "artifacts" / "objects"

        scaler = StandardScaler()
        scaler.fit(np.array([[0], [1]]))
        artifact = persist(scaler, artifacts_dir, "scaler")
        artifact["step"] = 0

        loader = BinaryLoader([artifact], results_dir)

        # Before loading
        info = loader.get_cache_info()
        assert info["cache_size"] == 0
        assert info["total_available_artifacts"] == 1
        assert 0 in info["available_steps"]

        # After loading
        loader.get_step_binaries("0")
        info = loader.get_cache_info()
        assert info["cache_size"] == 1

    def test_from_manifest_classmethod(self, results_dir):
        """Test creating BinaryLoader from manifest."""
        artifacts_dir = results_dir / "artifacts" / "objects"

        scaler = StandardScaler()
        scaler.fit(np.array([[0], [1]]))
        artifact = persist(scaler, artifacts_dir, "scaler")
        artifact["step"] = 0

        manifest = {
            "uid": "test-uid",
            "name": "test_pipeline",
            "dataset": "test_dataset",
            "artifacts": [artifact],
            "predictions": []
        }

        loader = BinaryLoader.from_manifest(manifest, results_dir)

        # Verify it works
        binaries = loader.get_step_binaries("0")
        assert len(binaries) == 1

    def test_handles_missing_artifact_gracefully(self, results_dir):
        """Test that missing artifacts are handled gracefully."""
        # Create artifact metadata but don't actually create the file
        artifact = {
            "hash": "sha256:nonexistent",
            "name": "missing_scaler",
            "path": "objects/no/nonexistent.pkl",
            "format": "sklearn_pickle",
            "size": 100,
            "step": 0,
            "saved_at": "2025-01-01T00:00:00Z"
        }

        loader = BinaryLoader([artifact], results_dir)

        # Should return empty list and warn, not crash
        with pytest.warns(UserWarning, match="Artifact file not found"):
            binaries = loader.get_step_binaries("0")

        assert binaries == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
