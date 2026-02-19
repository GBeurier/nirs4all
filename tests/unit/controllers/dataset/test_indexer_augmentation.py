"""
Unit tests for Indexer augmentation methods.

Tests cover:
- get_augmented_for_origins()
- get_origin_for_sample()
- x_indices() with include_augmented parameter
- Two-phase selection behavior
- Leak prevention
"""
import numpy as np
import pytest

from nirs4all.data.indexer import Indexer


class TestGetAugmentedForOrigins:
    """Tests for get_augmented_for_origins method."""

    def test_get_augmented_basic(self):
        """Test basic augmented sample retrieval."""
        indexer = Indexer()

        # Add base samples
        indexer.add_samples(3, partition="train")  # samples 0, 1, 2

        # Add augmented samples
        indexer.add_samples(2, partition="train", origin_indices=[0, 0], augmentation="aug_0")  # samples 3, 4
        indexer.add_samples(1, partition="train", origin_indices=[1], augmentation="aug_1")  # sample 5

        # Get augmented samples for origin 0
        augmented = indexer.get_augmented_for_origins([0])
        assert len(augmented) == 2
        assert set(augmented) == {3, 4}

        # Get augmented samples for origin 1
        augmented = indexer.get_augmented_for_origins([1])
        assert len(augmented) == 1
        assert augmented[0] == 5

        # Get augmented samples for origin 2 (no augmentations)
        augmented = indexer.get_augmented_for_origins([2])
        assert len(augmented) == 0

    def test_get_augmented_multiple_origins(self):
        """Test retrieving augmented samples for multiple origins."""
        indexer = Indexer()

        # Add base samples
        indexer.add_samples(3, partition="train")  # 0, 1, 2

        # Add augmented samples
        indexer.add_samples(2, partition="train", origin_indices=[0, 0], augmentation="aug_0")  # 3, 4
        indexer.add_samples(3, partition="train", origin_indices=[1, 1, 1], augmentation="aug_1")  # 5, 6, 7
        indexer.add_samples(1, partition="train", origin_indices=[2], augmentation="aug_0")  # 8

        # Get augmented for multiple origins
        augmented = indexer.get_augmented_for_origins([0, 1])
        assert len(augmented) == 5
        assert set(augmented) == {3, 4, 5, 6, 7}

        # Get all augmented
        augmented = indexer.get_augmented_for_origins([0, 1, 2])
        assert len(augmented) == 6
        assert set(augmented) == {3, 4, 5, 6, 7, 8}

    def test_get_augmented_empty_input(self):
        """Test that empty input returns empty array."""
        indexer = Indexer()
        indexer.add_samples(3, partition="train")

        augmented = indexer.get_augmented_for_origins([])
        assert len(augmented) == 0
        assert augmented.dtype == np.int32

    def test_get_augmented_no_matches(self):
        """Test that non-existent origins return empty array."""
        indexer = Indexer()
        indexer.add_samples(3, partition="train")

        # No augmented samples exist
        augmented = indexer.get_augmented_for_origins([0, 1, 2])
        assert len(augmented) == 0

    def test_get_augmented_mixed_partitions(self):
        """Test that augmented samples are retrieved regardless of partition."""
        indexer = Indexer()

        # Add base samples
        indexer.add_samples(2, partition="train")  # 0, 1

        # Add augmented samples in different partitions (edge case)
        indexer.add_samples(1, partition="train", origin_indices=[0], augmentation="aug_0")  # 2
        indexer.add_samples(1, partition="test", origin_indices=[0], augmentation="aug_1")  # 3

        # Get augmented for origin 0 (both partitions)
        augmented = indexer.get_augmented_for_origins([0])
        assert len(augmented) == 2
        assert set(augmented) == {2, 3}

class TestGetOriginForSample:
    """Tests for get_origin_for_sample method."""

    def test_get_origin_for_augmented_sample(self):
        """Test getting origin for augmented samples."""
        indexer = Indexer()

        # Add base samples
        indexer.add_samples(3, partition="train")  # 0, 1, 2

        # Add augmented samples
        indexer.add_samples(2, partition="train", origin_indices=[0, 0], augmentation="aug_0")  # 3, 4
        indexer.add_samples(1, partition="train", origin_indices=[1], augmentation="aug_1")  # 5

        # Get origin for augmented samples
        assert indexer.get_origin_for_sample(3) == 0
        assert indexer.get_origin_for_sample(4) == 0
        assert indexer.get_origin_for_sample(5) == 1

    def test_get_origin_for_base_sample(self):
        """Test that base samples return themselves as origin."""
        indexer = Indexer()

        # Add base samples
        indexer.add_samples(3, partition="train")  # 0, 1, 2

        # Base samples should return themselves
        assert indexer.get_origin_for_sample(0) == 0
        assert indexer.get_origin_for_sample(1) == 1
        assert indexer.get_origin_for_sample(2) == 2

    def test_get_origin_for_nonexistent_sample(self):
        """Test that non-existent samples return None."""
        indexer = Indexer()
        indexer.add_samples(3, partition="train")

        # Non-existent sample
        assert indexer.get_origin_for_sample(999) is None

    def test_get_origin_chain(self):
        """Test that origin lookup works correctly for augmented samples."""
        indexer = Indexer()

        # Add base sample
        indexer.add_samples(1, partition="train")  # 0

        # Add multiple levels of augmentation (though not recommended)
        indexer.add_samples(1, partition="train", origin_indices=[0], augmentation="aug_0")  # 1
        indexer.add_samples(1, partition="train", origin_indices=[0], augmentation="aug_1")  # 2

        # All should point back to origin 0
        assert indexer.get_origin_for_sample(1) == 0
        assert indexer.get_origin_for_sample(2) == 0

class TestXIndicesWithAugmentation:
    """Tests for x_indices with include_augmented parameter."""

    def test_x_indices_includes_augmented_by_default(self):
        """Test that augmented samples are included by default."""
        indexer = Indexer()

        # Add base samples
        indexer.add_samples(3, partition="train")  # 0, 1, 2

        # Add augmented samples
        indexer.add_samples(2, partition="train", origin_indices=[0, 0], augmentation="aug_0")  # 3, 4
        indexer.add_samples(1, partition="train", origin_indices=[1], augmentation="aug_1")  # 5

        # Default behavior: include augmented
        indices = indexer.x_indices({"partition": "train"})
        assert len(indices) == 6  # 3 base + 3 augmented
        assert set(indices) == {0, 1, 2, 3, 4, 5}

    def test_x_indices_excludes_augmented_when_false(self):
        """Test that augmented samples can be explicitly excluded."""
        indexer = Indexer()

        # Add base samples
        indexer.add_samples(3, partition="train")  # 0, 1, 2

        # Add augmented samples
        indexer.add_samples(2, partition="train", origin_indices=[0, 0], augmentation="aug_0")  # 3, 4
        indexer.add_samples(1, partition="train", origin_indices=[1], augmentation="aug_1")  # 5

        # Exclude augmented samples
        indices = indexer.x_indices({"partition": "train"}, include_augmented=False)
        assert len(indices) == 3  # Only base samples
        assert set(indices) == {0, 1, 2}

    def test_x_indices_two_phase_selection(self):
        """Test two-phase selection: base samples + their augmented versions."""
        indexer = Indexer()

        # Add base samples in train and test
        indexer.add_samples(3, partition="train")  # 0, 1, 2
        indexer.add_samples(2, partition="test")   # 3, 4

        # Add augmented samples
        indexer.add_samples(2, partition="train", origin_indices=[0, 0], augmentation="aug_0")  # 5, 6
        indexer.add_samples(1, partition="train", origin_indices=[1], augmentation="aug_1")  # 7
        indexer.add_samples(1, partition="test", origin_indices=[3], augmentation="aug_0")  # 8

        # Get train samples (base + augmented)
        train_indices = indexer.x_indices({"partition": "train"})
        assert len(train_indices) == 6  # 3 base + 3 augmented
        assert set(train_indices) == {0, 1, 2, 5, 6, 7}

        # Get test samples (base + augmented)
        test_indices = indexer.x_indices({"partition": "test"})
        assert len(test_indices) == 3  # 2 base + 1 augmented
        assert set(test_indices) == {3, 4, 8}

    def test_x_indices_with_group_filter(self):
        """Test that filtering works correctly with augmented samples."""
        indexer = Indexer()

        # Add base samples with different groups
        indexer.add_samples(2, partition="train", group=0)  # 0, 1
        indexer.add_samples(2, partition="train", group=1)  # 2, 3

        # Add augmented samples
        indexer.add_samples(1, partition="train", group=0, origin_indices=[0], augmentation="aug_0")  # 4
        indexer.add_samples(1, partition="train", group=1, origin_indices=[2], augmentation="aug_0")  # 5

        # Get group 0 samples
        group0_indices = indexer.x_indices({"partition": "train", "group": 0})
        assert len(group0_indices) == 3  # 2 base + 1 augmented
        assert set(group0_indices) == {0, 1, 4}

        # Get group 1 samples
        group1_indices = indexer.x_indices({"partition": "train", "group": 1})
        assert len(group1_indices) == 3  # 2 base + 1 augmented
        assert set(group1_indices) == {2, 3, 5}

    def test_x_indices_empty_selector(self):
        """Test that empty selector returns all samples."""
        indexer = Indexer()

        # Add base and augmented samples
        indexer.add_samples(3, partition="train")  # 0, 1, 2
        indexer.add_samples(2, partition="train", origin_indices=[0, 0], augmentation="aug_0")  # 3, 4

        # Empty selector should return all
        indices = indexer.x_indices({})
        assert len(indices) == 5
        assert set(indices) == {0, 1, 2, 3, 4}

    def test_x_indices_no_augmented_samples(self):
        """Test that behavior is correct when no augmented samples exist."""
        indexer = Indexer()

        # Only base samples
        indexer.add_samples(5, partition="train")  # 0, 1, 2, 3, 4

        # Should work the same with or without include_augmented
        indices_with = indexer.x_indices({"partition": "train"}, include_augmented=True)
        indices_without = indexer.x_indices({"partition": "train"}, include_augmented=False)

        assert len(indices_with) == 5
        assert len(indices_without) == 5
        assert set(indices_with) == set(indices_without) == {0, 1, 2, 3, 4}

    def test_x_indices_only_augmented_samples(self):
        """Test edge case where selector matches only augmented samples."""
        indexer = Indexer()

        # Add base samples
        indexer.add_samples(3, partition="train")  # 0, 1, 2

        # Add augmented samples with specific augmentation ID
        indexer.add_samples(2, partition="train", origin_indices=[0, 1], augmentation="aug_special")  # 3, 4

        # Select by augmentation ID (though unusual, should work)
        indices = indexer.x_indices({"augmentation": "aug_special"}, include_augmented=False)
        # Since augmented samples have origin != None, they won't be selected as base
        assert len(indices) == 0

class TestLeakPrevention:
    """Tests to ensure data leakage prevention across folds."""

    def test_no_leakage_across_partitions(self):
        """Test that augmented samples follow their origin's partition."""
        indexer = Indexer()

        # Add base samples to train and test
        indexer.add_samples(3, partition="train")  # 0, 1, 2
        indexer.add_samples(2, partition="test")   # 3, 4

        # Add augmented samples (should stay with their partition)
        indexer.add_samples(2, partition="train", origin_indices=[0, 1], augmentation="aug_0")  # 5, 6
        indexer.add_samples(1, partition="test", origin_indices=[3], augmentation="aug_0")  # 7

        # Train partition should not include test augmented samples
        train_indices = indexer.x_indices({"partition": "train"})
        assert 7 not in train_indices  # Test augmented sample

        # Test partition should not include train augmented samples
        test_indices = indexer.x_indices({"partition": "test"})
        assert 5 not in test_indices  # Train augmented samples
        assert 6 not in test_indices

    def test_fold_assignment_integrity(self):
        """Test that augmented samples can be traced to their origin."""
        indexer = Indexer()

        # Add base samples
        indexer.add_samples(5, partition="train")  # 0, 1, 2, 3, 4

        # Simulate fold assignment (e.g., fold 0: samples 0,1; fold 1: samples 2,3,4)
        # Add augmented samples for fold 0
        indexer.add_samples(2, partition="train", origin_indices=[0, 1], augmentation="aug_0")  # 5, 6

        # Add augmented samples for fold 1
        indexer.add_samples(3, partition="train", origin_indices=[2, 3, 4], augmentation="aug_0")  # 7, 8, 9

        # Verify origin mapping
        assert indexer.get_origin_for_sample(5) == 0  # fold 0
        assert indexer.get_origin_for_sample(6) == 1  # fold 0
        assert indexer.get_origin_for_sample(7) == 2  # fold 1
        assert indexer.get_origin_for_sample(8) == 3  # fold 1
        assert indexer.get_origin_for_sample(9) == 4  # fold 1

        # Verify that getting base samples excludes augmented
        base_samples = indexer.x_indices({"partition": "train"}, include_augmented=False)
        assert len(base_samples) == 5
        assert set(base_samples) == {0, 1, 2, 3, 4}

class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing code."""

    def test_default_behavior_unchanged(self):
        """Test that default behavior includes augmented samples."""
        indexer = Indexer()

        # Add samples
        indexer.add_samples(3, partition="train")
        indexer.add_samples(2, partition="train", origin_indices=[0, 1], augmentation="aug_0")

        # Old API (no include_augmented parameter) should work
        indices = indexer.x_indices({"partition": "train"})
        assert len(indices) == 5  # Includes augmented by default

    def test_y_indices_unchanged(self):
        """Test that y_indices still works as expected."""
        indexer = Indexer()

        # Add base samples
        indexer.add_samples(3, partition="train")  # 0, 1, 2

        # Add augmented samples
        indexer.add_samples(2, partition="train", origin_indices=[0, 0], augmentation="aug_0")  # 3, 4

        # y_indices should map augmented → origin
        y_indices = indexer.y_indices({"partition": "train"})
        assert len(y_indices) == 5
        # Augmented samples 3 and 4 should map to origin 0
        # But y_indices returns all indices, so we need to check manually

        # Get all sample IDs
        all_samples = indexer.x_indices({"partition": "train"})
        y_vals = indexer.y_indices({"partition": "train"})

        # Create mapping
        sample_to_y = dict(zip(all_samples, y_vals))
        assert sample_to_y[0] == 0  # Base sample maps to itself
        assert sample_to_y[1] == 1
        assert sample_to_y[2] == 2
        assert sample_to_y[3] == 0  # Augmented maps to origin
        assert sample_to_y[4] == 0  # Augmented maps to origin

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_indexer(self):
        """Test methods on empty indexer."""
        indexer = Indexer()

        # Should return empty arrays
        assert len(indexer.get_augmented_for_origins([0, 1, 2])) == 0
        assert indexer.get_origin_for_sample(0) is None
        assert len(indexer.x_indices({})) == 0

    def test_large_augmentation_count(self):
        """Test with many augmented samples per origin."""
        indexer = Indexer()

        # Add base sample
        indexer.add_samples(1, partition="train")  # 0

        # Add many augmented samples
        n_augmentations = 100
        indexer.add_samples(n_augmentations, partition="train", origin_indices=[0] * n_augmentations, augmentation="aug_0")

        # Get augmented samples
        augmented = indexer.get_augmented_for_origins([0])
        assert len(augmented) == n_augmentations

        # All should be returned by x_indices
        all_indices = indexer.x_indices({"partition": "train"})
        assert len(all_indices) == 1 + n_augmentations

    def test_multiple_augmentation_types(self):
        """Test with different augmentation types for same origin."""
        indexer = Indexer()

        # Add base sample
        indexer.add_samples(1, partition="train")  # 0

        # Add multiple augmentation types
        indexer.add_samples(2, partition="train", origin_indices=[0, 0], augmentation="aug_savgol")  # 1, 2
        indexer.add_samples(2, partition="train", origin_indices=[0, 0], augmentation="aug_gaussian")  # 3, 4
        indexer.add_samples(1, partition="train", origin_indices=[0], augmentation="aug_snv")  # 5

        # All should be retrieved
        augmented = indexer.get_augmented_for_origins([0])
        assert len(augmented) == 5
        assert set(augmented) == {1, 2, 3, 4, 5}
