"""Tests for data content hashing utility and SpectroDataset.content_hash()."""

import numpy as np

from nirs4all.utils.hashing import compute_data_hash
from nirs4all.data.dataset import SpectroDataset


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
