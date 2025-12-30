"""
Unit tests for MetadataGenerator class.
"""

import pytest
import numpy as np

from nirs4all.data.synthetic.metadata import (
    MetadataGenerator,
    MetadataGenerationResult,
    generate_sample_metadata,
)


class TestMetadataGeneratorInit:
    """Tests for MetadataGenerator initialization."""

    def test_default_init(self):
        """Test default initialization."""
        gen = MetadataGenerator()
        assert gen.rng is not None

    def test_with_random_state(self):
        """Test initialization with random state."""
        gen = MetadataGenerator(random_state=42)
        assert gen._random_state == 42


class TestBasicGeneration:
    """Tests for basic metadata generation."""

    def test_generate_sample_ids(self):
        """Test generating sample IDs."""
        gen = MetadataGenerator(random_state=42)
        result = gen.generate(n_samples=30)

        assert len(result.sample_ids) == 30
        assert result.sample_ids[0].startswith("S")
        assert len(set(result.sample_ids)) == 30  # All unique

    def test_custom_prefix(self):
        """Test custom sample ID prefix."""
        gen = MetadataGenerator(random_state=42)
        result = gen.generate(n_samples=10, sample_id_prefix="WHEAT")

        assert all(s.startswith("WHEAT") for s in result.sample_ids)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        gen = MetadataGenerator(random_state=42)
        result = gen.generate(n_samples=20)
        d = result.to_dict()

        assert "sample_id" in d
        assert len(d["sample_id"]) == 20


class TestGroupGeneration:
    """Tests for group metadata generation."""

    def test_generate_groups(self):
        """Test generating groups."""
        gen = MetadataGenerator(random_state=42)
        result = gen.generate(n_samples=30, n_groups=3)

        assert result.groups is not None
        assert len(result.groups) == 30
        assert result.group_indices is not None
        assert set(result.group_indices) == {0, 1, 2}

    def test_custom_group_names(self):
        """Test custom group names."""
        gen = MetadataGenerator(random_state=42)
        names = ["Field_A", "Field_B", "Field_C"]
        result = gen.generate(n_samples=30, n_groups=3, group_names=names)

        assert set(result.groups) == set(names)

    def test_group_name_mismatch_raises(self):
        """Test that mismatched group names raise error."""
        gen = MetadataGenerator(random_state=42)
        with pytest.raises(ValueError, match="group_names length"):
            gen.generate(n_samples=20, n_groups=3, group_names=["A", "B"])

    def test_groups_balanced(self):
        """Test that groups are approximately balanced."""
        gen = MetadataGenerator(random_state=42)
        result = gen.generate(n_samples=99, n_groups=3)

        unique, counts = np.unique(result.group_indices, return_counts=True)
        assert len(unique) == 3
        # Each group should have ~33 samples
        assert all(30 <= c <= 36 for c in counts)


class TestRepetitionGeneration:
    """Tests for repetition structure generation."""

    def test_fixed_repetitions(self):
        """Test fixed number of repetitions."""
        gen = MetadataGenerator(random_state=42)
        result = gen.generate(n_samples=40, n_repetitions=2)

        assert result.bio_sample_ids is not None
        assert result.repetitions is not None
        assert result.n_bio_samples == 20  # 40 samples / 2 reps

    def test_variable_repetitions(self):
        """Test variable repetitions."""
        gen = MetadataGenerator(random_state=42)
        result = gen.generate(n_samples=40, n_repetitions=(2, 4))

        assert result.bio_sample_ids is not None
        assert result.n_bio_samples < 40
        # Repetition values should be in range
        assert all(1 <= r <= 4 for r in result.repetitions)

    def test_no_repetitions(self):
        """Test no repetitions (default)."""
        gen = MetadataGenerator(random_state=42)
        result = gen.generate(n_samples=30, n_repetitions=1)

        # No bio_sample_ids when no repetitions
        assert result.bio_sample_ids is None
        assert result.n_bio_samples == 30

    def test_repetitions_with_groups(self):
        """Test repetitions with groups - groups assigned at bio sample level."""
        gen = MetadataGenerator(random_state=42)
        result = gen.generate(
            n_samples=30,
            n_groups=3,
            n_repetitions=2,
        )

        # All reps of same bio sample should have same group
        bio_to_group = {}
        for i in range(len(result.bio_sample_ids)):
            bio_id = result.bio_sample_ids[i]
            group = result.groups[i]
            if bio_id in bio_to_group:
                assert bio_to_group[bio_id] == group
            else:
                bio_to_group[bio_id] = group

    def test_invalid_min_repetitions(self):
        """Test error on invalid min repetitions."""
        gen = MetadataGenerator(random_state=42)
        with pytest.raises(ValueError, match="Minimum repetitions"):
            gen.generate(n_samples=20, n_repetitions=(0, 3))


class TestAdditionalColumns:
    """Tests for additional column generation."""

    def test_callable_column(self):
        """Test callable column generator."""
        gen = MetadataGenerator(random_state=42)

        def custom_gen(n, rng):
            return rng.uniform(0, 1, size=n)

        result = gen.generate(
            n_samples=50,
            additional_columns={"custom": custom_gen}
        )

        assert result.additional_columns is not None
        assert "custom" in result.additional_columns
        assert len(result.additional_columns["custom"]) == 50

    def test_list_column(self):
        """Test sampling from list."""
        gen = MetadataGenerator(random_state=42)
        result = gen.generate(
            n_samples=50,
            additional_columns={"category": ["A", "B", "C"]}
        )

        assert all(v in ["A", "B", "C"] for v in result.additional_columns["category"])

    def test_distribution_column(self):
        """Test distribution-based column."""
        gen = MetadataGenerator(random_state=42)
        result = gen.generate(
            n_samples=50,
            additional_columns={
                "weight": ("uniform", {"low": 0.5, "high": 1.5})
            }
        )

        values = result.additional_columns["weight"]
        assert all(0.5 <= v <= 1.5 for v in values)

    def test_invalid_column_spec(self):
        """Test error on invalid column specification."""
        gen = MetadataGenerator(random_state=42)
        with pytest.raises(ValueError, match="Invalid specification"):
            gen.generate(n_samples=50, additional_columns={"bad": 123})


class TestReproducibility:
    """Tests for reproducibility."""

    def test_same_seed_same_results(self):
        """Test that same seed produces same results."""
        gen1 = MetadataGenerator(random_state=42)
        gen2 = MetadataGenerator(random_state=42)

        result1 = gen1.generate(n_samples=30, n_groups=3, n_repetitions=(2, 4))
        result2 = gen2.generate(n_samples=30, n_groups=3, n_repetitions=(2, 4))

        np.testing.assert_array_equal(result1.sample_ids, result2.sample_ids)
        np.testing.assert_array_equal(result1.groups, result2.groups)

    def test_different_seed_different_results(self):
        """Test that different seeds produce different results."""
        gen1 = MetadataGenerator(random_state=42)
        gen2 = MetadataGenerator(random_state=123)

        result1 = gen1.generate(n_samples=30, n_groups=3)
        result2 = gen2.generate(n_samples=30, n_groups=3)

        # Groups should be shuffled differently
        assert not np.array_equal(result1.group_indices, result2.group_indices)


class TestConvenienceFunction:
    """Tests for generate_sample_metadata convenience function."""

    def test_basic_usage(self):
        """Test basic convenience function usage."""
        metadata = generate_sample_metadata(
            n_samples=30,
            random_state=42,
            n_groups=3,
        )

        assert "sample_id" in metadata
        assert "group" in metadata
        assert len(metadata["sample_id"]) == 30

    def test_with_repetitions(self):
        """Test convenience function with repetitions."""
        metadata = generate_sample_metadata(
            n_samples=30,
            random_state=42,
            n_repetitions=(2, 4),
        )

        assert "bio_sample_id" in metadata
        assert "repetition" in metadata
