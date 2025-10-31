"""
Tests for dataset working with both dict and DataSelector.

This module verifies backward compatibility during context migration.
"""

import pytest
import numpy as np

from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.context import DataSelector


class TestDatasetWithSelector:
    """Tests for dataset.x() and dataset.y() with both formats."""

    def test_x_with_dict_selector(self):
        """Test dataset.x() works with dict selector (legacy)."""
        dataset = SpectroDataset("test")

        # Add data
        X = np.random.rand(100, 50)
        dataset.add_samples(X, {"partition": "train"})

        # Get data with dict selector
        X_train = dataset.x({"partition": "train"})

        assert X_train.shape == (100, 50)
        # Check data is close (allow floating point precision differences)
        np.testing.assert_allclose(X_train, X, rtol=1e-5)

    def test_x_with_data_selector(self):
        """Test dataset.x() works with DataSelector (new)."""
        dataset = SpectroDataset("test")

        # Add data
        X = np.random.rand(100, 50)
        dataset.add_samples(X, {"partition": "train"})

        # Get data with DataSelector
        selector = DataSelector(partition="train")
        X_train = dataset.x(selector)

        assert X_train.shape == (100, 50)
        # Check data is close (allow floating point precision differences)
        np.testing.assert_allclose(X_train, X, rtol=1e-5)

    def test_x_with_none_selector(self):
        """Test dataset.x() works with None selector."""
        dataset = SpectroDataset("test")

        # Add data
        X = np.random.rand(100, 50)
        dataset.add_samples(X, {"partition": "train"})

        # Get all data with None selector
        X_all = dataset.x(None)

        assert X_all.shape == (100, 50)

    def test_y_with_dict_selector(self):
        """Test dataset.y() works with dict selector (legacy)."""
        dataset = SpectroDataset("test")

        # Add data
        X = np.random.rand(100, 50)
        y = np.array([0, 1] * 50)
        dataset.add_samples(X, {"partition": "train"})
        dataset.add_targets(y)

        # Get targets with dict selector
        y_train = dataset.y({"partition": "train"})

        assert len(y_train) == 100
        # Flatten if needed and check values
        y_train_flat = y_train.ravel()
        np.testing.assert_array_equal(y_train_flat, y)

    def test_y_with_data_selector(self):
        """Test dataset.y() works with DataSelector (new)."""
        dataset = SpectroDataset("test")

        # Add data
        X = np.random.rand(100, 50)
        y = np.array([0, 1] * 50)
        dataset.add_samples(X, {"partition": "train"})
        dataset.add_targets(y)

        # Get targets with DataSelector
        selector = DataSelector(partition="train")
        y_train = dataset.y(selector)

        assert len(y_train) == 100
        # Flatten if needed and check values
        y_train_flat = y_train.ravel()
        np.testing.assert_array_equal(y_train_flat, y)

    def test_x_with_processing_in_data_selector(self):
        """Test DataSelector with processing chains."""
        dataset = SpectroDataset("test")

        # Add data
        X = np.random.rand(100, 50)
        dataset.add_samples(X, {"partition": "train"})

        # Create selector with processing
        selector = DataSelector(
            partition="train",
            processing=[["raw"]]
        )
        X_train = dataset.x(selector)

        assert X_train.shape == (100, 50)

    def test_x_with_layout_in_data_selector(self):
        """Test DataSelector layout is used."""
        dataset = SpectroDataset("test")

        # Add data
        X = np.random.rand(100, 50)
        dataset.add_samples(X, {"partition": "train"})

        # Create selector with layout
        selector = DataSelector(partition="train", layout="2d")
        X_train = dataset.x(selector, layout="2d")

        assert X_train.shape == (100, 50)

    def test_augment_samples_with_dict_selector(self):
        """Test augment_samples works with dict selector."""
        dataset = SpectroDataset("test")

        # Add base data
        X = np.random.rand(100, 50)
        dataset.add_samples(X, {"partition": "train"})

        # Augment with dict selector
        X_aug = np.random.rand(100, 50)
        aug_ids = dataset.augment_samples(
            X_aug,
            ["raw", "noise"],
            "aug_v1",
            {"partition": "train"},
            count=1
        )

        assert len(aug_ids) == 100

    def test_augment_samples_with_data_selector(self):
        """Test augment_samples works with DataSelector."""
        dataset = SpectroDataset("test")

        # Add base data
        X = np.random.rand(100, 50)
        dataset.add_samples(X, {"partition": "train"})

        # Augment with DataSelector
        selector = DataSelector(partition="train")
        X_aug = np.random.rand(100, 50)
        aug_ids = dataset.augment_samples(
            X_aug,
            ["raw", "noise"],
            "aug_v1",
            selector,
            count=1
        )

        assert len(aug_ids) == 100

    def test_mixed_dict_and_data_selector(self):
        """Test that dict and DataSelector can be used interchangeably."""
        dataset = SpectroDataset("test")

        # Add data with dict
        X_train = np.random.rand(80, 50)
        dataset.add_samples(X_train, {"partition": "train"})

        X_test = np.random.rand(20, 50)
        dataset.add_samples(X_test, {"partition": "test"})

        # Get with dict
        X1 = dataset.x({"partition": "train"})

        # Get with DataSelector
        selector = DataSelector(partition="train")
        X2 = dataset.x(selector)

        # Should be identical
        np.testing.assert_array_equal(X1, X2)
