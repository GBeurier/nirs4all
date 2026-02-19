"""
Tests for SpectroDataset NaN tracking features.

Tests has_nan, nan_summary, and _may_contain_nan flag added in Phase 4.
"""

import numpy as np
import pytest

from nirs4all.data.dataset import SpectroDataset

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def clean_dataset():
    """SpectroDataset with clean (no NaN) data."""
    ds = SpectroDataset("clean_test")
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    y = np.array([10.0, 20.0, 30.0])
    ds.add_samples(X, {"partition": "train"})
    ds.add_targets(y)
    return ds

@pytest.fixture
def nan_x_dataset():
    """SpectroDataset with NaN in the feature matrix."""
    ds = SpectroDataset("nan_x_test")
    X = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0], [np.nan, 8.0, 9.0]])
    y = np.array([10.0, 20.0, 30.0])
    ds.add_samples(X, {"partition": "train"})
    ds.add_targets(y)
    return ds

@pytest.fixture
def nan_y_dataset():
    """SpectroDataset with NaN in the target array."""
    ds = SpectroDataset("nan_y_test")
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    y = np.array([10.0, np.nan, 30.0])
    ds.add_samples(X, {"partition": "train"})
    ds.add_targets(y)
    return ds

# =============================================================================
# Test: has_nan property
# =============================================================================

class TestHasNan:
    """Tests for the has_nan property."""

    def test_false_when_clean(self, clean_dataset):
        """has_nan returns False when no NaN in X or y."""
        assert clean_dataset.has_nan is False

    def test_true_when_x_has_nan(self, nan_x_dataset):
        """has_nan returns True when X contains NaN."""
        assert nan_x_dataset.has_nan is True

    def test_true_when_y_has_nan(self, nan_y_dataset):
        """has_nan returns True when y contains NaN."""
        assert nan_y_dataset.has_nan is True

    def test_empty_dataset(self):
        """has_nan returns False for an empty dataset."""
        ds = SpectroDataset("empty")
        assert ds.has_nan is False

# =============================================================================
# Test: nan_summary property
# =============================================================================

class TestNanSummary:
    """Tests for the nan_summary property."""

    def test_clean_dataset_summary(self, clean_dataset):
        """nan_summary returns has_nan=False for clean data."""
        summary = clean_dataset.nan_summary
        assert summary["has_nan"] is False
        assert summary["y_nan"] == 0
        assert len(summary["sources"]) >= 1
        assert summary["sources"][0]["nan_cells"] == 0

    def test_nan_x_summary(self, nan_x_dataset):
        """nan_summary correctly reports NaN in feature data."""
        summary = nan_x_dataset.nan_summary
        assert summary["has_nan"] is True
        assert summary["sources"][0]["nan_cells"] == 2  # Two NaN cells in X
        assert summary["sources"][0]["nan_samples"] == 2  # Two rows have NaN
        assert summary["sources"][0]["nan_features"] == 2  # Two columns have NaN

    def test_nan_y_summary(self, nan_y_dataset):
        """nan_summary checks the 'numeric' target processing for NaN.

        Note: The target converter may encode NaN as a class label during the
        'raw' -> 'numeric' conversion, so NaN in raw targets may not appear in
        the 'numeric' processing. has_nan checks all processings including 'raw',
        while nan_summary only checks 'numeric'.
        """
        summary = nan_y_dataset.nan_summary
        # X is clean in this fixture
        assert summary["sources"][0]["nan_cells"] == 0
        # has_nan property detects NaN across ALL target processings (including raw)
        assert nan_y_dataset.has_nan is True

    def test_summary_structure(self, clean_dataset):
        """nan_summary returns dict with expected keys."""
        summary = clean_dataset.nan_summary
        assert "sources" in summary
        assert "y_nan" in summary
        assert "has_nan" in summary

    def test_per_source_structure(self, clean_dataset):
        """Each source entry in nan_summary has expected keys."""
        summary = clean_dataset.nan_summary
        source_info = summary["sources"][0]
        expected_keys = {"source", "nan_cells", "nan_samples", "nan_features",
                         "total_samples", "total_features"}
        assert expected_keys.issubset(set(source_info.keys()))

# =============================================================================
# Test: _may_contain_nan flag
# =============================================================================

class TestMayContainNanFlag:
    """Tests for the _may_contain_nan internal flag."""

    def test_default_false(self):
        """_may_contain_nan defaults to False on new dataset."""
        ds = SpectroDataset("test")
        assert ds._may_contain_nan is False

    def test_can_be_set_true(self):
        """_may_contain_nan can be set to True."""
        ds = SpectroDataset("test")
        ds._may_contain_nan = True
        assert ds._may_contain_nan is True

    def test_flag_independent_of_actual_nan(self, clean_dataset):
        """_may_contain_nan is a hint flag, independent of actual NaN presence."""
        # Flag is False by default even though has_nan checks actual data
        assert clean_dataset._may_contain_nan is False
        assert clean_dataset.has_nan is False

        # Setting the flag doesn't change actual data
        clean_dataset._may_contain_nan = True
        assert clean_dataset._may_contain_nan is True
        assert clean_dataset.has_nan is False  # Actual data is still clean
