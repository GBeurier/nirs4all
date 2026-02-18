"""Tests for NorrisWilliams transform.

Tests cover:
- Function API: norris_williams()
- Transformer API: NorrisWilliams class (fit/transform)
- Parameter validation
- Output shape preservation
- Derivative computation sanity
- sklearn compatibility
"""

import numpy as np
import pytest
from sklearn.base import clone

from nirs4all.operators.transforms.norris_williams import NorrisWilliams, norris_williams


# =============================================================================
# Function API Tests
# =============================================================================


class TestNorrisWilliamsFunction:
    """Test the norris_williams() standalone function."""

    def test_output_shape(self):
        X = np.random.randn(10, 200)
        result = norris_williams(X, gap=5, segment=5, deriv=1)
        assert result.shape == X.shape

    def test_first_derivative(self):
        """First derivative of a linear signal should be approximately constant."""
        X = np.linspace(0, 10, 200).reshape(1, -1)
        result = norris_williams(X, gap=5, segment=1, deriv=1, delta=1.0)
        # Interior points should be approximately constant
        interior = result[0, 20:-20]
        assert np.std(interior) / np.abs(np.mean(interior)) < 0.1

    def test_second_derivative(self):
        X = np.random.randn(5, 200)
        result = norris_williams(X, gap=5, segment=1, deriv=2)
        assert result.shape == X.shape

    def test_segment_smoothing(self):
        """Larger segment should produce smoother output."""
        rng = np.random.RandomState(42)
        X = rng.randn(10, 200)
        result_s1 = norris_williams(X, gap=5, segment=1, deriv=1)
        result_s5 = norris_williams(X, gap=5, segment=5, deriv=1)
        # Smoother output should have lower variance
        assert np.var(result_s5) <= np.var(result_s1) * 1.5  # allow some tolerance

    def test_invalid_deriv(self):
        X = np.random.randn(5, 100)
        with pytest.raises(ValueError, match="deriv must be 1 or 2"):
            norris_williams(X, gap=5, segment=5, deriv=3)

    def test_invalid_segment_even(self):
        X = np.random.randn(5, 100)
        with pytest.raises(ValueError, match="segment must be odd"):
            norris_williams(X, gap=5, segment=4, deriv=1)

    def test_invalid_segment_zero(self):
        X = np.random.randn(5, 100)
        with pytest.raises(ValueError, match="segment must be odd"):
            norris_williams(X, gap=5, segment=0, deriv=1)


# =============================================================================
# Transformer API Tests
# =============================================================================


class TestNorrisWilliamsTransformer:
    """Test the NorrisWilliams sklearn transformer."""

    def test_fit_transform(self):
        X = np.random.randn(10, 200)
        nw = NorrisWilliams(gap=5, segment=5, deriv=1)
        nw.fit(X)
        result = nw.transform(X)
        assert result.shape == X.shape

    def test_fit_returns_self(self):
        X = np.random.randn(10, 100)
        nw = NorrisWilliams()
        result = nw.fit(X)
        assert result is nw

    def test_stateless(self):
        nw = NorrisWilliams()
        assert nw._stateless is True

    def test_webapp_meta(self):
        nw = NorrisWilliams()
        assert hasattr(nw, "_webapp_meta")
        assert nw._webapp_meta["category"] == "derivatives"

    def test_more_tags(self):
        nw = NorrisWilliams()
        tags = nw._more_tags()
        assert tags["allow_nan"] is False

    def test_clone(self):
        nw = NorrisWilliams(gap=7, segment=3, deriv=2, delta=0.5)
        cloned = clone(nw)
        assert cloned.gap == 7
        assert cloned.segment == 3
        assert cloned.deriv == 2
        assert cloned.delta == 0.5
        assert cloned is not nw

    def test_validate_invalid_deriv(self):
        nw = NorrisWilliams(deriv=3)
        with pytest.raises(ValueError):
            nw.fit(np.random.randn(5, 100))

    def test_validate_invalid_segment(self):
        nw = NorrisWilliams(segment=4)
        with pytest.raises(ValueError):
            nw.fit(np.random.randn(5, 100))

    def test_default_params(self):
        nw = NorrisWilliams()
        assert nw.gap == 5
        assert nw.segment == 5
        assert nw.deriv == 1
        assert nw.delta == 1.0

    def test_consistency_with_function(self):
        """Transformer output should match function output."""
        X = np.random.randn(10, 200)
        nw = NorrisWilliams(gap=5, segment=5, deriv=1, delta=1.0)
        nw.fit(X)
        transformer_result = nw.transform(X)
        function_result = norris_williams(X, gap=5, segment=5, deriv=1, delta=1.0)
        np.testing.assert_array_equal(transformer_result, function_result)
