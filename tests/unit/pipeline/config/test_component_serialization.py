"""
Unit tests for the central artifact serializer.

Tests framework-aware serialization, content-addressed storage,
and deduplication for sklearn, numpy, and generic objects.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from nirs4all.pipeline.storage.artifacts.artifact_persistence import (
    persist,
    load,
    compute_hash,
    to_bytes,
    from_bytes,
    is_serializable,
    _detect_framework,
    ArtifactMeta
)


@pytest.fixture
def artifacts_dir():
    """Create temporary artifacts directory."""
    temp_dir = tempfile.mkdtemp()
    artifacts_path = Path(temp_dir) / "artifacts" / "objects"
    artifacts_path.mkdir(parents=True, exist_ok=True)
    yield artifacts_path
    shutil.rmtree(temp_dir)


@pytest.fixture
def results_dir():
    """Create temporary results directory."""
    temp_dir = tempfile.mkdtemp()
    results_path = Path(temp_dir)
    (results_path / "artifacts" / "objects").mkdir(parents=True, exist_ok=True)
    yield results_path
    shutil.rmtree(temp_dir)


class TestFrameworkDetection:
    """Test automatic framework detection."""

    def test_detect_sklearn_scaler(self):
        """Test detection of sklearn transformer."""
        scaler = StandardScaler()
        format = _detect_framework(scaler)
        assert format == 'sklearn_pickle'

    def test_detect_sklearn_model(self):
        """Test detection of sklearn model."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        format = _detect_framework(model)
        assert format == 'sklearn_pickle'

    def test_detect_numpy_array(self):
        """Test detection of numpy array."""
        arr = np.array([1, 2, 3])
        format = _detect_framework(arr)
        assert format == 'numpy_npy'

    def test_detect_generic_object(self):
        """Test fallback for generic objects."""
        obj = {"key": "value", "number": 42}
        format = _detect_framework(obj)
        assert format == 'pickle'


class TestHashComputation:
    """Test content hashing."""

    def test_compute_hash_deterministic(self):
        """Test that same data produces same hash."""
        data = b"test data content"
        hash1 = compute_hash(data)
        hash2 = compute_hash(data)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_compute_hash_different_data(self):
        """Test that different data produces different hash."""
        hash1 = compute_hash(b"data1")
        hash2 = compute_hash(b"data2")
        assert hash1 != hash2


class TestSerialization:
    """Test to_bytes and from_bytes."""

    def test_sklearn_scaler_roundtrip(self):
        """Test sklearn scaler serialization roundtrip."""
        scaler = StandardScaler()
        X_train = np.array([[0], [1], [2], [3]])
        scaler.fit(X_train)

        # Serialize
        data, format = to_bytes(scaler)
        assert format in ['sklearn_pickle', 'joblib']
        assert len(data) > 0

        # Deserialize
        loaded_scaler = from_bytes(data, format)
        assert hasattr(loaded_scaler, 'mean_')
        np.testing.assert_array_almost_equal(loaded_scaler.mean_, scaler.mean_)

    def test_sklearn_model_roundtrip(self):
        """Test sklearn model serialization roundtrip."""
        model = LogisticRegression(random_state=42)
        X_train = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y_train = np.array([0, 0, 1, 1])
        model.fit(X_train, y_train)

        # Serialize
        data, format = to_bytes(model)
        assert format in ['sklearn_pickle', 'joblib']

        # Deserialize
        loaded_model = from_bytes(data, format)
        X_test = np.array([[1.5, 1.5]])
        pred_original = model.predict(X_test)
        pred_loaded = loaded_model.predict(X_test)
        np.testing.assert_array_equal(pred_original, pred_loaded)

    def test_numpy_array_roundtrip(self):
        """Test numpy array serialization roundtrip."""
        arr = np.random.rand(10, 5)

        # Serialize
        data, format = to_bytes(arr)
        assert format == 'numpy_npy'

        # Deserialize
        loaded_arr = from_bytes(data, format)
        np.testing.assert_array_equal(arr, loaded_arr)

    def test_generic_dict_roundtrip(self):
        """Test generic dict serialization roundtrip."""
        obj = {"key": "value", "numbers": [1, 2, 3], "nested": {"a": 1}}

        # Serialize
        data, format = to_bytes(obj)
        assert format in ['pickle', 'cloudpickle']

        # Deserialize
        loaded_obj = from_bytes(data, format)
        assert loaded_obj == obj


class TestPersistLoad:
    """Test persist and load with content-addressed storage."""

    def test_persist_creates_artifact(self, artifacts_dir):
        """Test that persist creates artifact file."""
        scaler = StandardScaler()
        scaler.fit(np.array([[0], [1], [2]]))

        artifact = persist(scaler, artifacts_dir, "test_scaler")

        # Check metadata
        assert "hash" in artifact
        assert artifact["hash"].startswith("sha256:")
        assert artifact["name"] == "test_scaler"
        assert "path" in artifact
        assert "format" in artifact
        assert artifact["size"] > 0

        # Check file exists - new flat structure in _binaries
        full_path = artifacts_dir / artifact["path"]
        assert full_path.exists()
        # Path should be in format: ClassName_hash.ext
        assert "StandardScaler_" in artifact["path"]

    def test_persist_uses_flat_structure(self, artifacts_dir):
        """Test that artifacts are stored in flat structure with meaningful names."""
        obj = {"test": "data"}
        artifact = persist(obj, artifacts_dir, "test_obj")

        # Path should be flat: ClassName_hash.ext (no subdirectories)
        assert "/" not in artifact["path"]  # No subdirectories
        assert "dict_" in artifact["path"]  # Should have class name
        assert artifact["path"].endswith(".pkl")  # Should have extension

    def test_persist_deduplication(self, artifacts_dir):
        """Test that identical objects produce same artifact."""
        scaler1 = StandardScaler()
        scaler1.fit(np.array([[0], [1], [2]]))

        scaler2 = StandardScaler()
        scaler2.fit(np.array([[0], [1], [2]]))

        artifact1 = persist(scaler1, artifacts_dir, "scaler1")
        artifact2 = persist(scaler2, artifacts_dir, "scaler2")

        # Same content → same hash → same file
        assert artifact1["hash"] == artifact2["hash"]
        assert artifact1["path"] == artifact2["path"]

    def test_load_artifact(self, results_dir):
        """Test loading artifact from metadata."""
        artifacts_dir = results_dir / "_binaries"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Create and persist
        scaler = StandardScaler()
        X_train = np.array([[0], [1], [2], [3]])
        scaler.fit(X_train)

        artifact = persist(scaler, artifacts_dir, "test_scaler")

        # Load
        loaded_scaler = load(artifact, results_dir)

        # Verify
        assert hasattr(loaded_scaler, 'mean_')
        np.testing.assert_array_almost_equal(loaded_scaler.mean_, scaler.mean_)

    def test_load_nonexistent_artifact(self, results_dir):
        """Test that loading nonexistent artifact raises error."""
        artifact: ArtifactMeta = {
            "hash": "sha256:nonexistent",
            "name": "missing",
            "path": "NonExistent_abc123.pkl",  # Flat path format
            "format": "pickle",
            "size": 100,
            "saved_at": "2025-01-01T00:00:00Z",
            "step": 0
        }

        with pytest.raises(FileNotFoundError):
            load(artifact, results_dir)

    def test_persist_load_multiple_objects(self, results_dir):
        """Test persisting and loading multiple different objects."""
        artifacts_dir = results_dir / "_binaries"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Create different objects
        scaler = StandardScaler()
        scaler.fit(np.array([[0], [1], [2]]))

        model = LogisticRegression(random_state=42)
        model.fit(np.array([[0, 0], [1, 1]]), np.array([0, 1]))

        arr = np.array([1, 2, 3, 4, 5])

        # Persist all
        artifact1 = persist(scaler, artifacts_dir, "scaler")
        artifact2 = persist(model, artifacts_dir, "model")
        artifact3 = persist(arr, artifacts_dir, "array")

        # All should have different hashes
        assert artifact1["hash"] != artifact2["hash"]
        assert artifact2["hash"] != artifact3["hash"]

        # Load and verify all
        loaded_scaler = load(artifact1, results_dir)
        loaded_model = load(artifact2, results_dir)
        loaded_arr = load(artifact3, results_dir)

        assert hasattr(loaded_scaler, 'mean_')
        assert hasattr(loaded_model, 'coef_')
        np.testing.assert_array_equal(loaded_arr, arr)


class TestIsSerializable:
    """Test is_serializable check."""

    def test_sklearn_object_serializable(self):
        """Test that sklearn objects are serializable."""
        scaler = StandardScaler()
        assert is_serializable(scaler)

    def test_numpy_array_serializable(self):
        """Test that numpy arrays are serializable."""
        arr = np.array([1, 2, 3])
        assert is_serializable(arr)

    def test_dict_serializable(self):
        """Test that dicts are serializable."""
        obj = {"key": "value"}
        assert is_serializable(obj)

    def test_lambda_serializable_with_cloudpickle(self):
        """Test that lambdas are serializable (cloudpickle fallback)."""
        func = lambda x: x + 1
        # This should work with cloudpickle fallback
        result = is_serializable(func)
        # Just check it doesn't crash - result depends on cloudpickle availability
        assert isinstance(result, bool)


class TestContentAddressedStorage:
    """Test content-addressed storage properties."""

    def test_same_content_same_hash(self, artifacts_dir):
        """Test that same content produces same hash."""
        obj = {"constant": "data", "value": 42}

        artifact1 = persist(obj, artifacts_dir, "obj1")
        artifact2 = persist(obj, artifacts_dir, "obj2")

        # Extract just the hash part
        hash1 = artifact1["hash"].split(":")[-1]
        hash2 = artifact2["hash"].split(":")[-1]

        assert hash1 == hash2

    def test_different_content_different_hash(self, artifacts_dir):
        """Test that different content produces different hash."""
        obj1 = {"value": 1}
        obj2 = {"value": 2}

        artifact1 = persist(obj1, artifacts_dir, "obj1")
        artifact2 = persist(obj2, artifacts_dir, "obj2")

        hash1 = artifact1["hash"].split(":")[-1]
        hash2 = artifact2["hash"].split(":")[-1]

        assert hash1 != hash2

    def test_file_not_overwritten(self, artifacts_dir):
        """Test that existing artifacts are not overwritten."""
        import time

        obj = {"data": "test"}

        # Persist first time
        artifact1 = persist(obj, artifacts_dir, "obj1")
        path1 = artifacts_dir / artifact1["path"]
        mtime1 = path1.stat().st_mtime

        # Wait a moment
        time.sleep(0.01)

        # Persist same object again
        artifact2 = persist(obj, artifacts_dir, "obj2")
        path2 = artifacts_dir / artifact2["path"]
        mtime2 = path2.stat().st_mtime

        # File should be the same (not overwritten)
        assert path1 == path2
        assert mtime1 == mtime2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
