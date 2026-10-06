"""
Integration tests for Dataset API augmentation support.

Tests cover:
- x() method with include_augmented parameter
- y() method with augmented sample mapping
- metadata() methods with include_augmented parameter
- Integration with augment_samples()
- Backward compatibility
"""
import numpy as np
import pytest

from nirs4all.data.dataset import SpectroDataset


class TestDatasetXMethod:
    """Tests for Dataset.x() with augmentation support."""

    def test_x_includes_augmented_by_default(self):
        """Test that x() includes augmented samples by default."""
        dataset = SpectroDataset("test")

        # Add base samples
        data = np.random.rand(5, 100)
        dataset.add_samples(data, {"partition": "train"})

        # Add augmented samples
        aug_data = np.random.rand(3, 100)
        dataset.augment_samples(aug_data, ["raw"], "aug_0", {"partition": "train"}, count=[2, 1, 0, 0, 0])

        # Get train data (should include augmented)
        X = dataset.x({"partition": "train"}, layout="2d")
        assert X.shape[0] == 8  # 5 base + 3 augmented

    def test_x_excludes_augmented_when_false(self):
        """Test that x() can exclude augmented samples."""
        dataset = SpectroDataset("test")

        # Add base samples
        data = np.random.rand(5, 100)
        dataset.add_samples(data, {"partition": "train"})

        # Add augmented samples
        aug_data = np.random.rand(3, 100)
        dataset.augment_samples(aug_data, ["raw"], "aug_0", {"partition": "train"}, count=[2, 1, 0, 0, 0])

        # Get train data without augmented
        X = dataset.x({"partition": "train"}, layout="2d", include_augmented=False)
        assert X.shape[0] == 5  # Only base samples

    def test_x_3d_layout_with_augmentation(self):
        """Test that 3D layout works with augmented samples."""
        dataset = SpectroDataset("test")

        # Add base samples
        data = np.random.rand(3, 100)
        dataset.add_samples(data, {"partition": "train"})

        # Add augmented samples
        aug_data = np.random.rand(2, 100)
        dataset.augment_samples(aug_data, ["raw"], "aug_0", {"partition": "train"}, count=[1, 1, 0])

        # Get 3D data
        X = dataset.x({"partition": "train"}, layout="3d")
        assert X.shape[0] == 5  # 3 base + 2 augmented
        assert X.shape[1] == 1  # 1 processing (raw)
        assert X.shape[2] == 100  # features

    def test_x_partition_filtering_with_augmentation(self):
        """Test that partition filtering works correctly with augmented samples."""
        dataset = SpectroDataset("test")

        # Add samples to different partitions
        train_data = np.random.rand(3, 100)
        test_data = np.random.rand(2, 100)
        dataset.add_samples(train_data, {"partition": "train"})
        dataset.add_samples(test_data, {"partition": "test"})

        # Add augmented samples to train
        aug_data = np.random.rand(2, 100)
        dataset.augment_samples(aug_data, ["raw"], "aug_0", {"partition": "train"}, count=[1, 1, 0])

        # Train should include augmented
        X_train = dataset.x({"partition": "train"})
        assert X_train.shape[0] == 5  # 3 base + 2 augmented

        # Test should not include train's augmented
        X_test = dataset.x({"partition": "test"})
        assert X_test.shape[0] == 2  # Only test base samples

class TestDatasetYMethod:
    """Tests for Dataset.y() with augmentation support."""

    def test_y_maps_augmented_to_origin(self):
        """Test that y() correctly maps augmented samples to their origin's y value."""
        dataset = SpectroDataset("test")

        # Add base samples with targets
        data = np.random.rand(5, 100)
        targets = np.array([0, 1, 0, 1, 0])
        dataset.add_samples(data, {"partition": "train"})
        dataset.add_targets(targets)

        # Add augmented samples
        aug_data = np.random.rand(3, 100)
        dataset.augment_samples(aug_data, ["raw"], "aug_0", {"partition": "train"}, count=[2, 1, 0, 0, 0])

        # Get y values (should include augmented mapped to origin)
        y = dataset.y({"partition": "train"})
        assert len(y) == 8  # 5 base + 3 augmented

        # First 2 augmented samples should map to origin 0 (y=0)
        # Next 1 augmented sample should map to origin 1 (y=1)
        # The exact order depends on how x_indices returns them
        # But we can check the counts
        assert np.sum(y == 0) == 5  # 3 original + 2 augmented from sample 0
        assert np.sum(y == 1) == 3  # 2 original + 1 augmented from sample 1

    def test_y_excludes_augmented_when_false(self):
        """Test that y() can exclude augmented samples."""
        dataset = SpectroDataset("test")

        # Add base samples with targets
        data = np.random.rand(5, 100)
        targets = np.array([0, 1, 0, 1, 0])
        dataset.add_samples(data, {"partition": "train"})
        dataset.add_targets(targets)

        # Add augmented samples
        aug_data = np.random.rand(3, 100)
        dataset.augment_samples(aug_data, ["raw"], "aug_0", {"partition": "train"}, count=[2, 1, 0, 0, 0])

        # Get y values without augmented
        y = dataset.y({"partition": "train"}, include_augmented=False)
        assert len(y) == 5  # Only base samples
        # Flatten if 2D
        y_flat = y.flatten() if y.ndim > 1 else y
        np.testing.assert_array_equal(y_flat, targets)

    def test_y_consistency_with_x(self):
        """Test that y values align correctly with x samples."""
        dataset = SpectroDataset("test")

        # Add base samples
        data = np.random.rand(3, 100)
        targets = np.array([10, 20, 30])
        dataset.add_samples(data, {"partition": "train"})
        dataset.add_targets(targets)

        # Add augmented samples
        aug_data = np.random.rand(2, 100)
        dataset.augment_samples(aug_data, ["raw"], "aug_0", {"partition": "train"}, count=[1, 1, 0])

        # Get x and y
        X = dataset.x({"partition": "train"})
        y = dataset.y({"partition": "train"})

        # Should have same length
        assert X.shape[0] == len(y) == 5

class TestDatasetMetadataMethods:
    """Tests for Dataset metadata methods with augmentation support."""

    def test_metadata_includes_augmented_by_default(self):
        """Test that metadata() includes augmented samples by default."""
        dataset = SpectroDataset("test")

        # Add base samples with metadata
        data = np.random.rand(3, 100)
        metadata = np.array([
            ["A", 1],
            ["B", 2],
            ["C", 3]
        ])
        dataset.add_samples(data, {"partition": "train"})
        dataset.add_metadata(metadata, headers=["group", "value"])

        # Augmented metadata follows each sample's origin without storing duplicate rows.
        aug_data = np.random.rand(2, 100)
        dataset.augment_samples(aug_data, ["raw"], "aug_0", {"partition": "train"}, count=[1, 1, 0])

        meta_df = dataset.metadata({"partition": "train"})
        assert len(meta_df) == dataset.x({"partition": "train"}).shape[0] == 5
        assert meta_df.rows() == [("A", "1"), ("B", "2"), ("C", "3"), ("A", "1"), ("B", "2")]

    def test_metadata_excludes_augmented_when_false(self):
        """Test that metadata() can exclude augmented samples."""
        dataset = SpectroDataset("test")

        # Add base samples with metadata
        data = np.random.rand(3, 100)
        metadata = np.array([
            ["A", 1],
            ["B", 2],
            ["C", 3]
        ])
        dataset.add_samples(data, {"partition": "train"})
        dataset.add_metadata(metadata, headers=["group", "value"])

        # Add augmented samples
        aug_data = np.random.rand(2, 100)
        dataset.augment_samples(aug_data, ["raw"], "aug_0", {"partition": "train"}, count=[1, 1, 0])

        # Get metadata without augmented
        meta_df = dataset.metadata({"partition": "train"}, include_augmented=False)
        assert len(meta_df) == 3  # Only base samples

    def test_metadata_column_with_augmentation(self):
        """Test that metadata_column works with augmented samples."""
        dataset = SpectroDataset("test")

        # Add base samples with metadata
        data = np.random.rand(3, 100)
        metadata = np.array([
            ["A", 1],
            ["B", 2],
            ["C", 3]
        ])
        dataset.add_samples(data, {"partition": "train"})
        dataset.add_metadata(metadata, headers=["group", "value"])

        # Add augmented samples
        aug_data = np.random.rand(2, 100)
        dataset.augment_samples(aug_data, ["raw"], "aug_0", {"partition": "train"}, count=[1, 1, 0])

        groups = dataset.metadata_column("group", {"partition": "train"})
        np.testing.assert_array_equal(groups, ["A", "B", "C", "A", "B"])

        # Get column without augmented
        groups_base = dataset.metadata_column("group", {"partition": "train"}, include_augmented=False)
        np.testing.assert_array_equal(groups_base, ["A", "B", "C"])

    def test_metadata_numeric_with_augmentation(self):
        """Test that metadata_numeric works with augmented samples."""
        dataset = SpectroDataset("test")

        # Add base samples with metadata
        data = np.random.rand(4, 100)
        metadata = np.array([
            ["A", 1],
            ["B", 2],
            ["A", 3],
            ["B", 4]
        ])
        dataset.add_samples(data, {"partition": "train"})
        dataset.add_metadata(metadata, headers=["group", "value"])

        # Add augmented samples
        aug_data = np.random.rand(2, 100)
        dataset.augment_samples(aug_data, ["raw"], "aug_0", {"partition": "train"}, count=[1, 1, 0, 0])

        encoded, mapping = dataset.metadata_numeric("group", {"partition": "train"}, include_augmented=False)
        assert len(encoded) == 4
        augmented, augmented_mapping = dataset.metadata_numeric("group", {"partition": "train"})
        np.testing.assert_array_equal(augmented, encoded[[0, 1, 2, 3, 0, 1]])
        assert augmented_mapping == mapping

class TestAugmentSamplesIntegration:
    """Tests for augment_samples integration with enhanced API."""

    def test_augment_samples_basic(self):
        """Test basic augment_samples functionality."""
        dataset = SpectroDataset("test")

        # Add base samples
        data = np.random.rand(3, 100)
        dataset.add_samples(data, {"partition": "train"})

        # Augment samples
        aug_data = np.random.rand(2, 100)
        augmented_ids = dataset.augment_samples(
            aug_data, ["raw"], "aug_test", {"partition": "train"}, count=[1, 1, 0]
        )

        assert len(augmented_ids) == 2

        # Verify augmented samples are included by default
        X = dataset.x({"partition": "train"})
        assert X.shape[0] == 5  # 3 base + 2 augmented

    def test_augment_samples_with_metadata(self):
        """Test that augmented samples inherit metadata from origin."""
        dataset = SpectroDataset("test")

        # Add base samples with metadata
        data = np.random.rand(3, 100)
        metadata = np.array([["A"], ["B"], ["C"]])
        dataset.add_samples(data, {"partition": "train"})
        dataset.add_metadata(metadata, headers=["group"])

        # Augment samples
        aug_data = np.random.rand(2, 100)
        dataset.augment_samples(aug_data, ["raw"], "aug_0", {"partition": "train"}, count=[1, 1, 0])

        meta = dataset.metadata({"partition": "train"})
        assert len(meta) == dataset.x({"partition": "train"}).shape[0] == 5
        assert meta.get_column("group").to_list() == ["A", "B", "C", "A", "B"]
        base_meta = dataset.metadata({"partition": "train"}, include_augmented=False)
        assert base_meta.get_column("group").to_list() == ["A", "B", "C"]

    def test_augment_samples_variable_counts(self):
        """Test augmenting with different counts per sample."""
        dataset = SpectroDataset("test")

        # Add base samples
        data = np.random.rand(4, 100)
        dataset.add_samples(data, {"partition": "train"})

        # Augment with variable counts
        aug_data = np.random.rand(6, 100)  # Total: 3 + 2 + 1 + 0
        dataset.augment_samples(aug_data, ["raw"], "aug_0", {"partition": "train"}, count=[3, 2, 1, 0])

        # Verify total count
        X = dataset.x({"partition": "train"})
        assert X.shape[0] == 10  # 4 base + 6 augmented

class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing code."""

    def test_default_behavior_unchanged(self):
        """Test that default behavior includes augmented samples."""
        dataset = SpectroDataset("test")

        # Add samples
        data = np.random.rand(5, 100)
        targets = np.array([0, 1, 0, 1, 0])
        dataset.add_samples(data, {"partition": "train"})
        dataset.add_targets(targets)

        # Add augmented samples
        aug_data = np.random.rand(2, 100)
        dataset.augment_samples(aug_data, ["raw"], "aug_0", {"partition": "train"}, count=[1, 1, 0, 0, 0])

        # Old API (no include_augmented parameter) should work and include augmented
        X = dataset.x({"partition": "train"})
        y = dataset.y({"partition": "train"})

        assert X.shape[0] == 7  # 5 base + 2 augmented
        assert len(y) == 7

    def test_existing_workflows_unchanged(self):
        """Test that existing workflows continue to work."""
        dataset = SpectroDataset("test")

        # Standard workflow
        train_data = np.random.rand(10, 100)
        test_data = np.random.rand(5, 100)
        train_targets = np.random.randint(0, 2, 10)
        test_targets = np.random.randint(0, 2, 5)

        dataset.add_samples(train_data, {"partition": "train"})
        dataset.add_samples(test_data, {"partition": "test"})
        dataset.add_targets(np.concatenate([train_targets, test_targets]))

        # Get train/test data (should work as before)
        X_train = dataset.x({"partition": "train"})
        y_train = dataset.y({"partition": "train"})
        X_test = dataset.x({"partition": "test"})
        y_test = dataset.y({"partition": "test"})

        assert X_train.shape[0] == 10
        assert len(y_train) == 10
        assert X_test.shape[0] == 5
        assert len(y_test) == 5

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_dataset(self):
        """Test methods on empty dataset."""
        dataset = SpectroDataset("test")

        # Should handle empty dataset gracefully
        # Note: x() will raise ValueError if no features exist
        with pytest.raises(ValueError, match="No features available"):
            X = dataset.x({})

    def test_no_augmented_samples(self):
        """Test that behavior is correct when no augmented samples exist."""
        dataset = SpectroDataset("test")

        # Only base samples
        data = np.random.rand(5, 100)
        targets = np.array([0, 1, 0, 1, 0])
        dataset.add_samples(data, {"partition": "train"})
        dataset.add_targets(targets)

        # Should work the same with or without include_augmented
        X_with = dataset.x({"partition": "train"}, include_augmented=True)
        X_without = dataset.x({"partition": "train"}, include_augmented=False)
        y_with = dataset.y({"partition": "train"}, include_augmented=True)
        y_without = dataset.y({"partition": "train"}, include_augmented=False)

        assert X_with.shape[0] == X_without.shape[0] == 5
        np.testing.assert_array_equal(y_with, y_without)

    def test_all_samples_augmented(self):
        """Test with high augmentation ratio."""
        dataset = SpectroDataset("test")

        # Add base samples
        data = np.random.rand(2, 100)
        dataset.add_samples(data, {"partition": "train"})

        # Add many augmented samples
        aug_data = np.random.rand(10, 100)
        dataset.augment_samples(aug_data, ["raw"], "aug_0", {"partition": "train"}, count=[5, 5])

        # Should handle correctly
        X = dataset.x({"partition": "train"})
        assert X.shape[0] == 12  # 2 base + 10 augmented

        X_base = dataset.x({"partition": "train"}, include_augmented=False)
        assert X_base.shape[0] == 2  # Only base

    def test_multi_source_with_augmentation(self):
        """Test that multi-source datasets work with augmentation."""
        dataset = SpectroDataset("test")

        # Add multi-source data
        data1 = np.random.rand(3, 100)
        data2 = np.random.rand(3, 50)
        dataset.add_samples([data1, data2], {"partition": "train"})

        # Augment
        aug_data1 = np.random.rand(2, 100)
        aug_data2 = np.random.rand(2, 50)
        dataset.augment_samples([aug_data1, aug_data2], ["raw"], "aug_0", {"partition": "train"}, count=[1, 1, 0])

        # Get data
        X = dataset.x({"partition": "train"}, concat_source=False)
        assert isinstance(X, list)
        assert len(X) == 2  # Two sources
        assert X[0].shape[0] == 5  # 3 base + 2 augmented
        assert X[1].shape[0] == 5  # 3 base + 2 augmented
