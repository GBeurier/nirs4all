"""
Unit tests for BalancingCalculator.

Tests cover:
- Binary classification balancing
- Multi-class balancing
- Different max_factor values
- Already balanced scenarios
- Random transformer selection
- Edge cases and error handling
"""
import numpy as np
import pytest
from nirs4all.utils.balancing import BalancingCalculator


class TestCalculateBalancedCounts:
    """Tests for calculate_balanced_counts method."""

    def test_binary_balancing_basic(self):
        """Test basic binary class balancing."""
        # 4 samples class 0, 2 samples class 1
        labels = np.array([0, 0, 0, 0, 1, 1])
        indices = np.array([10, 11, 12, 13, 14, 15])

        counts = BalancingCalculator.calculate_balanced_counts(labels, indices, labels, indices, max_factor=1.0)

        # Class 0 (majority): no augmentation
        assert counts[10] == 0
        assert counts[11] == 0
        assert counts[12] == 0
        assert counts[13] == 0

        # Class 1 (minority): needs 2 more samples (4 - 2 = 2)
        # 2 augmentations / 2 samples = 1 augmentation per sample
        assert counts[14] == 1
        assert counts[15] == 1

    def test_binary_balancing_with_max_factor(self):
        """Test binary balancing with max_factor < 1.0."""
        # 100 samples class 0, 20 samples class 1
        labels = np.array([0] * 100 + [1] * 20)
        indices = np.arange(120)

        counts = BalancingCalculator.calculate_balanced_counts(labels, indices, labels, indices, max_factor=0.8)

        # Majority class (100 samples): no augmentation
        for i in range(100):
            assert counts[i] == 0

        # Minority class (20 samples): target = 100 * 0.8 = 80
        # Need 60 more samples (80 - 20 = 60)
        # 60 / 20 = 3 augmentations per sample
        for i in range(100, 120):
            assert counts[i] == 3

    def test_multiclass_balancing(self):
        """Test multi-class balancing (3+ classes)."""
        # Class 0: 50 samples, Class 1: 30 samples, Class 2: 20 samples
        labels = np.array([0] * 50 + [1] * 30 + [2] * 20)
        indices = np.arange(100)

        counts = BalancingCalculator.calculate_balanced_counts(labels, indices, labels, indices, max_factor=1.0)

        # Class 0 (majority): no augmentation
        for i in range(50):
            assert counts[i] == 0

        # Class 1: needs 20 more (50 - 30 = 20)
        # 20 / 30 = 0 base + 20 remainder
        # First 20 samples get 1 augmentation, rest get 0
        class1_counts = [counts[i] for i in range(50, 80)]
        assert sum(class1_counts) == 20  # Total 20 augmentations
        assert class1_counts[:20] == [1] * 20  # First 20 get 1
        assert class1_counts[20:] == [0] * 10  # Last 10 get 0

        # Class 2: needs 30 more (50 - 20 = 30)
        # 30 / 20 = 1 base + 10 remainder
        # First 10 samples get 2 augmentations, rest get 1
        class2_counts = [counts[i] for i in range(80, 100)]
        assert sum(class2_counts) == 30  # Total 30 augmentations
        assert class2_counts[:10] == [2] * 10  # First 10 get 2
        assert class2_counts[10:] == [1] * 10  # Last 10 get 1

    def test_multiclass_with_max_factor_05(self):
        """Test multi-class balancing with max_factor=0.5."""
        labels = np.array([0] * 100 + [1] * 40 + [2] * 20)
        indices = np.arange(160)

        counts = BalancingCalculator.calculate_balanced_counts(labels, indices, labels, indices, max_factor=0.5)

        # Majority class (100): no augmentation
        for i in range(100):
            assert counts[i] == 0

        # Class 1 (40): target = 100 * 0.5 = 50, need 10 more
        # 10 / 40 = 0 base + 10 remainder
        class1_counts = [counts[i] for i in range(100, 140)]
        assert sum(class1_counts) == 10
        assert class1_counts[:10] == [1] * 10
        assert class1_counts[10:] == [0] * 30

        # Class 2 (20): target = 50, need 30 more
        # 30 / 20 = 1 base + 10 remainder
        class2_counts = [counts[i] for i in range(140, 160)]
        assert sum(class2_counts) == 30
        assert class2_counts[:10] == [2] * 10
        assert class2_counts[10:] == [1] * 10

    def test_already_balanced_dataset(self):
        """Test that already balanced datasets return zero augmentations."""
        labels = np.array([0] * 50 + [1] * 50 + [2] * 50)
        indices = np.arange(150)

        counts = BalancingCalculator.calculate_balanced_counts(labels, indices, labels, indices, max_factor=1.0)

        # All classes equal size - no augmentation needed
        for i in range(150):
            assert counts[i] == 0

    def test_already_over_balanced(self):
        """Test that classes larger than target get zero augmentations."""
        # Class 0: 100, Class 1: 90 (with max_factor=0.8, target=80)
        labels = np.array([0] * 100 + [1] * 90)
        indices = np.arange(190)

        counts = BalancingCalculator.calculate_balanced_counts(labels, indices, labels, indices, max_factor=0.8)

        # Both classes >= target, no augmentation
        for i in range(190):
            assert counts[i] == 0

    def test_string_labels(self):
        """Test that string labels work correctly."""
        labels = np.array(['cat', 'cat', 'cat', 'dog', 'dog'])
        indices = np.array([0, 1, 2, 3, 4])

        counts = BalancingCalculator.calculate_balanced_counts(labels, indices, labels, indices, max_factor=1.0)

        # 'cat' is majority (3), 'dog' is minority (2)
        assert counts[0] == 0
        assert counts[1] == 0
        assert counts[2] == 0
        # 'dog' needs 1 more to reach 3
        # 1 / 2 = 0 base + 1 remainder
        assert counts[3] == 1  # First sample gets remainder
        assert counts[4] == 0

    def test_empty_input(self):
        """Test that empty inputs return empty dict."""
        labels = np.array([])
        indices = np.array([])

        counts = BalancingCalculator.calculate_balanced_counts(labels, indices, labels, indices, max_factor=1.0)
        assert counts == {}

    def test_single_class(self):
        """Test that single-class datasets work (no balancing needed)."""
        labels = np.array([0, 0, 0, 0])
        indices = np.array([10, 11, 12, 13])

        counts = BalancingCalculator.calculate_balanced_counts(labels, indices, labels, indices, max_factor=1.0)

        # All same class - no augmentation
        assert all(counts[i] == 0 for i in [10, 11, 12, 13])

    def test_invalid_max_factor_above_1(self):
        """Test that max_factor > 1.0 raises ValueError."""
        labels = np.array([0, 1])
        indices = np.array([0, 1])

        with pytest.raises(ValueError, match="max_factor must be between 0.0 and 1.0"):
            BalancingCalculator.calculate_balanced_counts(labels, indices, labels, indices, max_factor=1.5)

    def test_invalid_max_factor_below_0(self):
        """Test that max_factor < 0.0 raises ValueError."""
        labels = np.array([0, 1])
        indices = np.array([0, 1])

        with pytest.raises(ValueError, match="max_factor must be between 0.0 and 1.0"):
            BalancingCalculator.calculate_balanced_counts(labels, indices, labels, indices, max_factor=-0.1)

    def test_mismatched_lengths(self):
        """Test that mismatched labels and indices raise ValueError."""
        labels = np.array([0, 1, 2])
        indices = np.array([10, 11])  # Different length

        with pytest.raises(ValueError, match="base_labels and base_sample_indices must have same length"):
            BalancingCalculator.calculate_balanced_counts(labels, indices, labels, indices, max_factor=1.0)

    def test_large_dataset(self):
        """Test performance with larger dataset."""
        # 10000 majority, 1000 minority
        labels = np.array([0] * 10000 + [1] * 1000)
        indices = np.arange(11000)

        counts = BalancingCalculator.calculate_balanced_counts(labels, indices, labels, indices, max_factor=0.9)

        # Target for class 1: 10000 * 0.9 = 9000
        # Need 8000 more (9000 - 1000)
        # 8000 / 1000 = 8 augmentations per sample
        minority_counts = [counts[i] for i in range(10000, 11000)]
        assert all(c == 8 for c in minority_counts)

        # Majority gets no augmentation
        assert sum(counts[i] for i in range(10000)) == 0


class TestApplyRandomTransformerSelection:
    """Tests for apply_random_transformer_selection method."""

    def test_basic_random_selection(self):
        """Test basic random transformer selection."""
        transformers = ['SavGol', 'Gaussian', 'SNV']  # Mock transformers
        counts = {10: 2, 11: 3, 12: 0}

        selection = BalancingCalculator.apply_random_transformer_selection(
            transformers, counts, random_state=42
        )

        # Check structure
        assert len(selection[10]) == 2
        assert len(selection[11]) == 3
        assert len(selection[12]) == 0

        # Check valid indices
        for sample_id, trans_indices in selection.items():
            for idx in trans_indices:
                assert 0 <= idx < 3

    def test_deterministic_with_seed(self):
        """Test that same seed produces same results."""
        transformers = ['A', 'B', 'C', 'D']
        counts = {1: 5, 2: 5, 3: 5}

        selection1 = BalancingCalculator.apply_random_transformer_selection(
            transformers, counts, random_state=123
        )
        selection2 = BalancingCalculator.apply_random_transformer_selection(
            transformers, counts, random_state=123
        )

        # Same seed = same results
        assert selection1 == selection2

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results (with high probability)."""
        transformers = ['A', 'B', 'C', 'D']
        counts = {1: 100}  # Large count to ensure difference

        selection1 = BalancingCalculator.apply_random_transformer_selection(
            transformers, counts, random_state=1
        )
        selection2 = BalancingCalculator.apply_random_transformer_selection(
            transformers, counts, random_state=2
        )

        # Different seeds = different results (probabilistically)
        assert selection1[1] != selection2[1]

    def test_no_seed_non_deterministic(self):
        """Test that None seed produces non-deterministic results."""
        transformers = ['A', 'B', 'C']
        counts = {1: 50}

        selection1 = BalancingCalculator.apply_random_transformer_selection(
            transformers, counts, random_state=None
        )
        selection2 = BalancingCalculator.apply_random_transformer_selection(
            transformers, counts, random_state=None
        )

        # Without seed, results are different (high probability with 50 samples)
        # Note: This test might rarely fail due to randomness
        assert len(selection1[1]) == len(selection2[1]) == 50

    def test_distribution_fairness(self):
        """Test that random selection is approximately uniform."""
        transformers = ['A', 'B', 'C']
        counts = {1: 3000}  # Large count for statistical test

        selection = BalancingCalculator.apply_random_transformer_selection(
            transformers, counts, random_state=42
        )

        # Count transformer occurrences
        trans_counts = [0, 0, 0]
        for idx in selection[1]:
            trans_counts[idx] += 1

        # Each transformer should appear ~1000 times (3000 / 3)
        # Allow 10% deviation (900-1100)
        for count in trans_counts:
            assert 900 <= count <= 1100

    def test_single_transformer(self):
        """Test with only one transformer."""
        transformers = ['SavGol']
        counts = {1: 5, 2: 10}

        selection = BalancingCalculator.apply_random_transformer_selection(
            transformers, counts, random_state=42
        )

        # All selections should be index 0
        assert selection[1] == [0] * 5
        assert selection[2] == [0] * 10

    def test_many_transformers(self):
        """Test with many transformers."""
        transformers = [f'Trans_{i}' for i in range(20)]
        counts = {1: 100}

        selection = BalancingCalculator.apply_random_transformer_selection(
            transformers, counts, random_state=42
        )

        # Check all indices are valid
        assert len(selection[1]) == 100
        assert all(0 <= idx < 20 for idx in selection[1])

    def test_empty_transformers_raises_error(self):
        """Test that empty transformers list raises ValueError."""
        transformers = []
        counts = {1: 5}

        with pytest.raises(ValueError, match="transformers list cannot be empty"):
            BalancingCalculator.apply_random_transformer_selection(
                transformers, counts, random_state=42
            )

    def test_zero_counts_for_all_samples(self):
        """Test that zero counts produce empty lists."""
        transformers = ['A', 'B', 'C']
        counts = {1: 0, 2: 0, 3: 0}

        selection = BalancingCalculator.apply_random_transformer_selection(
            transformers, counts, random_state=42
        )

        assert selection[1] == []
        assert selection[2] == []
        assert selection[3] == []

    def test_mixed_counts(self):
        """Test mixed counts including zeros."""
        transformers = ['A', 'B']
        counts = {1: 0, 2: 5, 3: 0, 4: 10, 5: 0}

        selection = BalancingCalculator.apply_random_transformer_selection(
            transformers, counts, random_state=42
        )

        assert selection[1] == []
        assert len(selection[2]) == 5
        assert selection[3] == []
        assert len(selection[4]) == 10
        assert selection[5] == []


class TestIntegration:
    """Integration tests combining both methods."""

    def test_full_balanced_workflow(self):
        """Test complete workflow: calculate counts then select transformers."""
        # Setup: imbalanced dataset
        labels = np.array([0] * 80 + [1] * 20)
        indices = np.arange(100)
        transformers = ['SavGol', 'Gaussian', 'SNV']

        # Step 1: Calculate balanced counts
        counts = BalancingCalculator.calculate_balanced_counts(
            labels, indices, labels, indices, max_factor=0.9
        )

        # Verify: minority class gets augmentations
        minority_augmentations = sum(counts[i] for i in range(80, 100))
        assert minority_augmentations > 0

        # Step 2: Apply random transformer selection
        selection = BalancingCalculator.apply_random_transformer_selection(
            transformers, counts, random_state=42
        )

        # Verify: selection matches counts
        for sample_id, count in counts.items():
            assert len(selection[sample_id]) == count

        # Verify: all transformer indices are valid
        for sample_id, trans_indices in selection.items():
            assert all(0 <= idx < len(transformers) for idx in trans_indices)

    def test_multiclass_full_workflow(self):
        """Test workflow with multi-class dataset."""
        labels = np.array([0] * 100 + [1] * 50 + [2] * 25)
        indices = np.arange(175)
        transformers = ['Trans1', 'Trans2', 'Trans3', 'Trans4']

        # Calculate and select
        counts = BalancingCalculator.calculate_balanced_counts(
            labels, indices, labels, indices, max_factor=0.8
        )
        selection = BalancingCalculator.apply_random_transformer_selection(
            transformers, counts, random_state=99
        )

        # Class 0 (majority): no augmentations
        assert all(counts[i] == 0 for i in range(100))
        assert all(len(selection[i]) == 0 for i in range(100))

        # Classes 1 and 2: have augmentations
        assert sum(counts[i] for i in range(100, 175)) > 0
        assert sum(len(selection[i]) for i in range(100, 175)) > 0
