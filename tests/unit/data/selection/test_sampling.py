"""Tests for nirs4all.data.selection.sampling."""

import numpy as np
import pytest

from nirs4all.data.selection.sampling import (
    kmeans_sample,
    random_sample,
    stratified_sample,
)


class TestRandomSample:
    """Tests for random_sample."""

    def test_returns_correct_count(self):
        indices = random_sample(100, 20, seed=42)
        assert len(indices) == 20

    def test_caps_at_n_total(self):
        indices = random_sample(10, 50, seed=42)
        assert len(indices) == 10

    def test_unique_indices(self):
        indices = random_sample(100, 50, seed=42)
        assert len(set(indices)) == 50

    def test_reproducible(self):
        a = random_sample(100, 20, seed=42)
        b = random_sample(100, 20, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = random_sample(100, 20, seed=1)
        b = random_sample(100, 20, seed=2)
        assert not np.array_equal(a, b)


class TestStratifiedSample:
    """Tests for stratified_sample."""

    def test_returns_correct_count(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 10)
        y = rng.rand(200)
        indices = stratified_sample(X, y, 50, seed=42)
        assert len(indices) == 50

    def test_unique_indices(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 10)
        y = rng.rand(200)
        indices = stratified_sample(X, y, 50, seed=42)
        assert len(set(indices)) == 50

    def test_covers_y_range(self):
        """Selected samples should span the full y range reasonably."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 10)
        y = np.linspace(0, 100, 200)
        indices = stratified_sample(X, y, 50, seed=42)
        y_selected = y[indices]
        # Selected samples should cover at least 80% of the y range
        y_range = y_selected.max() - y_selected.min()
        assert y_range > 80

    def test_fallback_with_few_unique_y(self):
        """Falls back to random when y has too few unique values."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5)
        y = np.ones(50)  # All same value
        indices = stratified_sample(X, y, 20, seed=42)
        assert len(indices) == 20

    def test_reproducible(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        y = rng.rand(100)
        a = stratified_sample(X, y, 30, seed=42)
        b = stratified_sample(X, y, 30, seed=42)
        np.testing.assert_array_equal(a, b)


class TestKmeansSample:
    """Tests for kmeans_sample."""

    def test_returns_correct_count(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        indices = kmeans_sample(X, 20, seed=42)
        assert len(indices) == 20

    def test_unique_indices(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        indices = kmeans_sample(X, 20, seed=42)
        assert len(set(indices)) == 20

    def test_caps_at_n_total(self):
        rng = np.random.RandomState(42)
        X = rng.randn(10, 5)
        indices = kmeans_sample(X, 50, seed=42)
        assert len(indices) == 10

    def test_indices_in_range(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        indices = kmeans_sample(X, 20, seed=42)
        assert all(0 <= i < 100 for i in indices)
