"""
Unit tests for BalancingCalculator.calculate_balanced_counts_value_aware.

Tests cover:
- Value-aware balancing with duplicate values
- Distribution fairness across values and samples
- Comparison between sample-aware and value-aware modes
- Different binning strategies
"""
import numpy as np
import pytest

from nirs4all.controllers.data.balancing import BalancingCalculator


class TestValueAwareBalancing:
    """Tests for value-aware balanced count calculation."""

    def test_basic_value_aware_balancing(self):
        """Test basic value-aware balancing with duplicate y-values."""
        # 3 samples in one class, 2 unique values
        # Sample 0: y=1.0, Sample 1: y=2.0, Sample 2: y=2.0
        base_labels = np.array([0, 0, 0])
        base_indices = np.array([100, 101, 102])
        base_values = np.array([1.0, 2.0, 2.0])

        # All labels/indices (in this simple case, just the base)
        all_labels = np.array([0, 0, 0])
        all_indices = np.array([100, 101, 102])

        counts = BalancingCalculator.calculate_balanced_counts_value_aware(
            base_labels, base_indices, base_values, all_labels, all_indices,
            target_size=10, random_state=42
        )

        # Need 10 - 3 = 7 augmentations total
        total_aug = sum(counts.values())
        assert total_aug == 7

        # With 2 unique values, distribution should be fair
        # 7 / 2 values = 3 base + 1 remainder
        # One value gets 3, other gets 4
        # Then within each value, distribute to samples

        # Check that samples with same value get similar augmentations
        sample_0_aug = counts[100]  # y=1.0
        sample_1_aug = counts[101]  # y=2.0
        sample_2_aug = counts[102]  # y=2.0

        # Samples 1 and 2 (same value) should have fairer distribution
        # than in sample-aware mode
        print(f"Sample 0 (y=1.0): {sample_0_aug} augmentations")
        print(f"Sample 1 (y=2.0): {sample_1_aug} augmentations")
        print(f"Sample 2 (y=2.0): {sample_2_aug} augmentations")

        # The difference should be at most 1 for samples with same value
        diff = abs(sample_1_aug - sample_2_aug)
        assert diff <= 1, f"Samples with same value should have similar aug counts, got diff={diff}"

    def test_value_aware_vs_sample_aware(self):
        """Compare value-aware vs sample-aware balancing."""
        # Scenario: 3 samples, 2 with same value
        # Sample 0: y=18.3999, Sample 1: y=18.5, Sample 2: y=18.5
        base_labels = np.array([0, 0, 0])
        base_indices = np.array([0, 1, 2])
        base_values = np.array([18.3999, 18.5, 18.5])
        all_labels = base_labels
        all_indices = base_indices

        # Sample-aware balancing
        counts_sample = BalancingCalculator.calculate_balanced_counts(
            base_labels, base_indices, all_labels, all_indices,
            target_size=10, random_state=42
        )

        # Value-aware balancing
        counts_value = BalancingCalculator.calculate_balanced_counts_value_aware(
            base_labels, base_indices, base_values, all_labels, all_indices,
            target_size=10, random_state=42
        )

        print(f"\nSample-aware: {dict(counts_sample)}")
        print(f"Value-aware: {dict(counts_value)}")

        # Both should have same total augmentations
        assert sum(counts_sample.values()) == sum(counts_value.values()) == 7

        # In value-aware mode, samples 1 and 2 should be more balanced
        sample_aware_diff = abs(counts_sample[1] - counts_sample[2])
        value_aware_diff = abs(counts_value[1] - counts_value[2])

        print(f"Sample-aware |count[1]-count[2]|: {sample_aware_diff}")
        print(f"Value-aware |count[1]-count[2]|: {value_aware_diff}")

        # Value-aware should produce more balanced distribution for same-value samples
        assert value_aware_diff <= 1

    def test_multiclass_value_aware_balancing(self):
        """Test value-aware balancing with multiple classes."""
        # Class 0: 2 values, 3 samples total
        # Class 1: 2 values, 2 samples total
        base_labels = np.array([0, 0, 0, 1, 1])
        base_indices = np.array([0, 1, 2, 3, 4])
        base_values = np.array([10.0, 10.5, 10.5, 20.0, 20.5])

        all_labels = np.array([0, 0, 0, 1, 1])
        all_indices = np.array([0, 1, 2, 3, 4])

        counts = BalancingCalculator.calculate_balanced_counts_value_aware(
            base_labels, base_indices, base_values, all_labels, all_indices,
            target_size=6, random_state=42
        )

        # Class 0: 6 - 3 = 3 augmentations
        # Class 1: 6 - 2 = 4 augmentations
        class_0_aug = sum(counts[i] for i in [0, 1, 2])
        class_1_aug = sum(counts[i] for i in [3, 4])

        assert class_0_aug == 3
        assert class_1_aug == 4

    def test_value_aware_with_max_factor(self):
        """Test value-aware balancing with ref_percentage."""
        # 5 samples: 3 with class 0, 2 with class 1
        base_labels = np.array([0, 0, 0, 1, 1])
        base_indices = np.array([0, 1, 2, 3, 4])
        base_values = np.array([1.0, 1.5, 1.5, 5.0, 5.5])

        all_labels = base_labels
        all_indices = base_indices

        counts = BalancingCalculator.calculate_balanced_counts_value_aware(
            base_labels, base_indices, base_values, all_labels, all_indices,
            ref_percentage=1.0, random_state=42
        )

        # Class 0 is majority (3), Class 1 needs to reach 3
        # 3 - 2 = 1 augmentation needed for class 1
        class_1_aug = sum(counts[i] for i in [3, 4])
        assert class_1_aug == 1

    def test_value_aware_single_value(self):
        """Test value-aware when all samples have same value."""
        # All samples have same y-value: should behave like sample-aware
        base_labels = np.array([0, 0, 0])
        base_indices = np.array([0, 1, 2])
        base_values = np.array([5.0, 5.0, 5.0])

        all_labels = base_labels
        all_indices = base_indices

        counts_value = BalancingCalculator.calculate_balanced_counts_value_aware(
            base_labels, base_indices, base_values, all_labels, all_indices,
            target_size=6, random_state=42
        )

        # 6 - 3 = 3 augmentations, distributed among 3 samples
        # 3 / 3 = 1 per sample, no remainder
        for i in range(3):
            assert counts_value[i] == 1

    def test_value_aware_all_unique_values(self):
        """Test value-aware when all samples have unique values."""
        # Each sample has unique value: should give each value 1 augmentation group
        base_labels = np.array([0, 0, 0])
        base_indices = np.array([0, 1, 2])
        base_values = np.array([1.0, 2.0, 3.0])

        all_labels = base_labels
        all_indices = base_indices

        counts_value = BalancingCalculator.calculate_balanced_counts_value_aware(
            base_labels, base_indices, base_values, all_labels, all_indices,
            target_size=6, random_state=42
        )

        # 6 - 3 = 3 augmentations across 3 unique values
        # 3 / 3 = 1 per value, and 1 sample per value, so 1 per sample
        for i in range(3):
            assert counts_value[i] == 1

    def test_value_aware_remainder_distribution(self):
        """Test that remainder is randomly distributed across values."""
        # 5 samples in 3 values: need different random distributions
        base_labels = np.array([0] * 5)
        base_indices = np.array([0, 1, 2, 3, 4])
        base_values = np.array([1.0, 1.5, 1.5, 2.0, 2.5])  # 4 unique values

        all_labels = base_labels
        all_indices = base_indices

        # Run twice with different seeds
        counts1 = BalancingCalculator.calculate_balanced_counts_value_aware(
            base_labels, base_indices, base_values, all_labels, all_indices,
            target_size=13, random_state=1
        )

        counts2 = BalancingCalculator.calculate_balanced_counts_value_aware(
            base_labels, base_indices, base_values, all_labels, all_indices,
            target_size=13, random_state=2
        )

        # Total should be same (13 - 5 = 8)
        assert sum(counts1.values()) == sum(counts2.values()) == 8

        # Distribution might differ due to random remainder selection
        # (Though with only 8 and 4 values, it might be deterministic)
        # At minimum, check both are valid
        for counts in [counts1, counts2]:
            assert all(c >= 0 for c in counts.values())

    def test_value_aware_invalid_args(self):
        """Test error handling in value-aware balancing."""
        base_labels = np.array([0, 0])
        base_indices = np.array([0, 1])
        base_values = np.array([1.0, 2.0])
        all_labels = base_labels
        all_indices = base_indices

        # Mismatched lengths
        with pytest.raises(ValueError, match="same length as base_labels"):
            BalancingCalculator.calculate_balanced_counts_value_aware(
                base_labels, base_indices,
                np.array([1.0, 2.0, 3.0]),  # Wrong length
                all_labels, all_indices,
                target_size=10
            )

        # Multiple modes
        with pytest.raises(ValueError, match="Specify exactly one"):
            BalancingCalculator.calculate_balanced_counts_value_aware(
                base_labels, base_indices, base_values,
                all_labels, all_indices,
                target_size=10, max_factor=1.0
            )

    def test_value_aware_user_scenario(self):
        """Test the specific user scenario: y=18.5 overrepresentation."""
        # Original: 1 sample with y=18.3999, 2 samples with y=18.5
        # Target: 10 total samples
        # Problem: y=18.5 gets overrepresented

        base_labels = np.array([0, 0, 0])
        base_indices = np.array([0, 1, 2])
        base_values = np.array([18.3999, 18.5, 18.5])

        all_labels = np.array([0, 0, 0])
        all_indices = np.array([0, 1, 2])

        counts = BalancingCalculator.calculate_balanced_counts_value_aware(
            base_labels, base_indices, base_values,
            all_labels, all_indices,
            target_size=10, random_state=42
        )

        # Total augmentations: 10 - 3 = 7
        assert sum(counts.values()) == 7

        # With value-aware balancing:
        # - 2 unique values (18.3999, 18.5)
        # - 7 / 2 = 3 base + 1 remainder
        # - One value gets 3, other gets 4 augmentations
        # - Value 18.3999: 1 sample gets 3 or 4 augmentations
        # - Value 18.5: 2 samples share 4 or 3 augmentations

        sample_0 = counts[0]  # y=18.3999
        samples_18_5 = sorted([counts[1], counts[2]])  # y=18.5

        print(f"\nSample 0 (y=18.3999): {sample_0} augmentations → {1 + sample_0} total")
        print(f"Sample 1 (y=18.5): {samples_18_5[0]} augmentations → {1 + samples_18_5[0]} total")
        print(f"Sample 2 (y=18.5): {samples_18_5[1]} augmentations → {1 + samples_18_5[1]} total")

        # Check fairness: samples with same y-value should be balanced
        assert abs(samples_18_5[0] - samples_18_5[1]) <= 1

        # Final result should be approximately:
        # Sample 0: 5, Samples 1&2: 2.5 each (approximately)
        # Actual: Sample 0: 5, Samples 1&2: 2, 3 or 3, 2
        final_counts = [1 + sample_0, 1 + samples_18_5[0], 1 + samples_18_5[1]]
        print(f"Final counts: {final_counts} (sum={sum(final_counts)})")
        assert sum(final_counts) == 10
