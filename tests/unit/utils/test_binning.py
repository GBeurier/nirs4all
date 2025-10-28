"""Unit tests for BinningCalculator."""
import numpy as np
import pytest
from nirs4all.utils.binning import BinningCalculator


class TestBinContinuousTargets:
    """Test bin_continuous_targets method."""

    def test_basic_quantile_binning(self):
        """Test basic quantile binning with 3 bins."""
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        bin_indices, bin_edges = BinningCalculator.bin_continuous_targets(
            y, bins=3, strategy="quantile"
        )

        assert len(bin_indices) == 10
        assert len(bin_edges) == 4
        assert bin_indices.min() >= 0

    def test_basic_equal_width_binning(self):
        """Test basic equal width binning with 3 bins (default strategy)."""
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # Test with explicit strategy
        bin_indices, bin_edges = BinningCalculator.bin_continuous_targets(
            y, bins=3, strategy="equal_width"
        )
        assert len(bin_indices) == 10
        assert len(bin_edges) == 4

        # Test default strategy is equal_width
        bin_indices_default, bin_edges_default = BinningCalculator.bin_continuous_targets(
            y, bins=3
        )
        assert np.array_equal(bin_indices, bin_indices_default)
        assert np.array_equal(bin_edges, bin_edges_default)

    def test_single_bin(self):
        """Test with single bin."""
        y = np.array([1, 2, 3, 4, 5])
        bin_indices, _ = BinningCalculator.bin_continuous_targets(
            y, bins=1, strategy="quantile"
        )
        assert np.all(bin_indices == 0)

    def test_constant_y(self):
        """Test with constant y values."""
        y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        bin_indices, _ = BinningCalculator.bin_continuous_targets(
            y, bins=3, strategy="quantile"
        )
        assert len(np.unique(bin_indices)) == 1

    def test_nan_raises_error(self):
        """Test that NaN values raise error."""
        y = np.array([1.0, 2.0, np.nan, 4.0])
        with pytest.raises(ValueError, match="NaN"):
            BinningCalculator.bin_continuous_targets(y, bins=3)

    def test_invalid_bins(self):
        """Test invalid bin counts."""
        y = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="bins must be"):
            BinningCalculator.bin_continuous_targets(y, bins=0)
        with pytest.raises(ValueError, match="bins must be"):
            BinningCalculator.bin_continuous_targets(y, bins=1001)

    def test_invalid_strategy(self):
        """Test invalid strategy."""
        y = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="strategy must be"):
            BinningCalculator.bin_continuous_targets(y, bins=3, strategy="invalid")

    def test_empty_y_raises_error(self):
        """Test that empty y raises error."""
        y = np.array([])
        with pytest.raises(ValueError, match="empty"):
            BinningCalculator.bin_continuous_targets(y, bins=3)

    def test_2d_array_flattened(self):
        """Test that 2D arrays are flattened."""
        y = np.array([[1, 2], [3, 4], [5, 6]])
        bin_indices, _ = BinningCalculator.bin_continuous_targets(
            y, bins=2, strategy="quantile"
        )
        assert len(bin_indices) == 6

    def test_bin_edges_monotonic(self):
        """Test that bin edges are monotonically increasing."""
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        for strategy in ["quantile", "equal_width"]:
            _, bin_edges = BinningCalculator.bin_continuous_targets(
                y, bins=5, strategy=strategy
            )
            assert np.all(np.diff(bin_edges) > 0)

    def test_negative_values(self):
        """Test with negative values."""
        y = np.array([-100, -50, 0, 50, 100])
        bin_indices, bin_edges = BinningCalculator.bin_continuous_targets(
            y, bins=3, strategy="quantile"
        )
        assert len(bin_indices) == 5
        assert bin_edges[0] <= -100
        assert bin_edges[-1] >= 100
