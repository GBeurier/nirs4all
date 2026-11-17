"""Tests for data content hashing utility and SpectroDataset.content_hash()."""

import hashlib
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest

from nirs4all.api.studio_lineage import validate_studio_provenance
from nirs4all.data.dataset import SpectroDataset
from nirs4all.utils.hashing import compute_data_hash


class TestComputeDataHash:
    """Tests for the compute_data_hash utility function."""

    def test_deterministic(self):
        """Same array produces the same hash."""
        X = np.random.RandomState(42).rand(100, 200)
        assert compute_data_hash(X) == compute_data_hash(X)

    def test_different_data_different_hash(self):
        """Different array contents produce different hashes."""
        rng = np.random.RandomState(0)
        X1 = rng.rand(100, 200)
        X2 = rng.rand(100, 200)
        assert compute_data_hash(X1) != compute_data_hash(X2)

    def test_same_values_same_hash(self):
        """Two arrays with identical values produce the same hash."""
        X1 = np.ones((50, 100), dtype=np.float64)
        X2 = np.ones((50, 100), dtype=np.float64)
        assert compute_data_hash(X1) == compute_data_hash(X2)

    def test_returns_hex_string(self):
        """Hash is a non-empty hex string."""
        X = np.zeros((10, 5))
        h = compute_data_hash(X)
        assert isinstance(h, str)
        assert len(h) > 0
        # All characters should be valid hex digits
        assert all(c in "0123456789abcdef" for c in h)

    def test_layout_independent(self):
        """C-contiguous and Fortran-contiguous layouts produce the same hash."""
        X_c = np.ascontiguousarray(np.ones((10, 5)))
        X_f = np.asfortranarray(np.ones((10, 5)))
        assert compute_data_hash(X_c) == compute_data_hash(X_f)

    def test_empty_array(self):
        """Empty array produces a valid hash."""
        X = np.empty((0, 10))
        h = compute_data_hash(X)
        assert isinstance(h, str)
        assert len(h) > 0

    def test_3d_array(self):
        """3D arrays hash deterministically."""
        X = np.random.RandomState(7).rand(20, 3, 100)
        assert compute_data_hash(X) == compute_data_hash(X.copy())

class TestSpectroDatasetContentHash:
    """Tests for SpectroDataset.content_hash()."""

    def _make_dataset(self, X: np.ndarray, name: str = "test") -> SpectroDataset:
        """Helper to create a simple dataset with feature data."""
        ds = SpectroDataset(name)
        ds.add_samples(X, {"partition": "train"})
        return ds

    def test_returns_hex_string(self):
        """content_hash returns a non-empty hex digest."""
        ds = self._make_dataset(np.ones((10, 50)))
        h = ds.content_hash()
        assert isinstance(h, str)
        assert len(h) > 0

    def test_deterministic(self):
        """Same dataset returns the same hash on repeated calls."""
        ds = self._make_dataset(np.ones((10, 50)))
        assert ds.content_hash() == ds.content_hash()

    def test_identical_features_same_hash(self):
        """Two datasets with identical features produce the same hash."""
        X = np.random.RandomState(99).rand(30, 100)
        ds1 = self._make_dataset(X.copy())
        ds2 = self._make_dataset(X.copy())
        assert ds1.content_hash() == ds2.content_hash()

    def test_different_features_different_hash(self):
        """Datasets with different feature data produce different hashes."""
        rng = np.random.RandomState(1)
        ds1 = self._make_dataset(rng.rand(30, 100))
        ds2 = self._make_dataset(rng.rand(30, 100))
        assert ds1.content_hash() != ds2.content_hash()

    def test_cache_invalidated_by_add_samples(self):
        """Adding new samples invalidates the cached hash."""
        ds = self._make_dataset(np.ones((10, 50)))
        h1 = ds.content_hash()
        ds.add_samples(np.zeros((5, 50)), {"partition": "train"})
        h2 = ds.content_hash()
        assert h1 != h2

    def test_cache_invalidated_by_replace_features(self):
        """Replacing features invalidates the cached hash."""
        ds = self._make_dataset(np.ones((10, 50)))
        h1 = ds.content_hash()
        ds.replace_features(["raw"], [np.zeros((10, 50))], ["replaced"], source=0)
        h2 = ds.content_hash()
        assert h1 != h2

    def test_source_index_hash(self):
        """content_hash(source_index=i) hashes only that source."""
        X0 = np.ones((10, 50))
        X1 = np.zeros((10, 30))
        ds = SpectroDataset("multi")
        ds.add_samples([X0, X1], {"partition": "train"})
        h0 = ds.content_hash(source_index=0)
        h1 = ds.content_hash(source_index=1)
        assert h0 != h1

    def test_cache_used(self):
        """Second call returns cached value (same object identity not required, but same value)."""
        ds = self._make_dataset(np.ones((10, 50)))
        h1 = ds.content_hash()
        h2 = ds.content_hash()
        assert h1 == h2
        # Verify the internal cache is set
        assert ds._content_hash_cache is not None

class _FakeXxh128:
    """Stand-in for ``xxhash.xxh128``: a 128-bit (32 hex character) digest."""

    def __init__(self, data: bytes = b"") -> None:
        self._digest = hashlib.md5(data)

    def update(self, data: bytes) -> None:
        self._digest.update(data)

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


@pytest.fixture(params=["xxhash_importable", "xxhash_missing"])
def xxhash_environment(request, monkeypatch):
    """Run a test once with ``xxhash`` importable and once with its import failing."""
    if request.param == "xxhash_importable":
        module = types.ModuleType("xxhash")
        module.xxh128 = _FakeXxh128
        monkeypatch.setitem(sys.modules, "xxhash", module)
    else:
        monkeypatch.setitem(sys.modules, "xxhash", None)
    return request.param


class TestContentIdentityIsEnvironmentIndependent:
    """The persisted dataset identity is SHA-256 whatever optional packages are installed."""

    def test_single_source_identity_is_sha256_of_feature_bytes(self, xxhash_environment):
        X = np.random.default_rng(0).normal(size=(8, 4))
        ds = SpectroDataset("identity")
        ds.add_samples(X, {"partition": "train"})
        stored = np.ascontiguousarray(ds._features.sources[0]._storage.array)
        expected = hashlib.sha256(stored.tobytes()).hexdigest()
        assert ds.content_hash() == expected
        assert ds.content_hash(source_index=0) == expected
        assert compute_data_hash(stored) == expected
        assert len(expected) == 64

    def test_multi_source_identity_is_sha256_of_concatenated_source_bytes(self, xxhash_environment):
        rng = np.random.default_rng(1)
        ds = SpectroDataset("identity-multi")
        ds.add_samples([rng.normal(size=(6, 5)), rng.normal(size=(6, 3))], {"partition": "train"})
        hasher = hashlib.sha256()
        for source in ds._features.sources:
            hasher.update(np.ascontiguousarray(source._storage.array).tobytes())
        assert ds.content_hash() == hasher.hexdigest()

    def test_studio_provenance_accepts_the_dataset_identity(self, xxhash_environment):
        ds = SpectroDataset("studio")
        ds.add_samples(np.random.default_rng(2).normal(size=(8, 4)), {"partition": "train"})
        digest = ds.content_hash()
        provenance = validate_studio_provenance({"job_id": "job", "dataset_ids_by_hash": {digest: "dataset-a"}}, "job")
        assert provenance == {"job_id": "job", "dataset_ids_by_hash": {digest: "dataset-a"}}


class TestDatasetMetadataHashing:
    """Tests metadata hashing path for get_dataset_metadata()."""

    def test_get_dataset_metadata_uses_content_hash_without_x_materialization(self):
        """Metadata hash path should not force dataset.x(layout='2d') materialization."""
        ds = SpectroDataset("meta")
        ds.add_samples(np.random.RandomState(123).rand(20, 15), {"partition": "train"})

        expected_hash = ds.content_hash()
        ds._content_hash_cache = None

        with patch.object(ds, "x", side_effect=AssertionError("dataset.x should not be called")):
            meta = ds.get_dataset_metadata(include_y_stats=False)

        assert meta["hash"] == expected_hash
