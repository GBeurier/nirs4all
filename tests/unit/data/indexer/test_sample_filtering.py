"""
Unit tests for Indexer sample filtering methods.

Tests cover:
- Schema changes (excluded, exclusion_reason columns)
- mark_excluded() method with cascade behavior
- mark_included() method
- x_indices() with include_excluded parameter
- y_indices() with include_excluded parameter
- get_excluded_samples() method
- get_exclusion_summary() method
- reset_exclusions() method
- Integration with augmentation
- Edge cases and backward compatibility
"""
import numpy as np
import polars as pl
import pytest

from nirs4all.data.indexer import Indexer


class TestSchemaChanges:
    """Tests for the new excluded and exclusion_reason columns."""

    def test_indexer_has_excluded_column(self):
        """Test that the indexer schema includes excluded column."""
        indexer = Indexer()
        assert "excluded" in indexer.df.columns

    def test_indexer_has_exclusion_reason_column(self):
        """Test that the indexer schema includes exclusion_reason column."""
        indexer = Indexer()
        assert "exclusion_reason" in indexer.df.columns

    def test_new_samples_default_excluded_false(self):
        """Test that new samples have excluded=False by default."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train")

        # All samples should have excluded=False
        excluded_values = indexer.df.select("excluded").to_series().to_list()
        assert all(v is False for v in excluded_values)

    def test_new_samples_default_exclusion_reason_none(self):
        """Test that new samples have exclusion_reason=None by default."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train")

        # All samples should have exclusion_reason=None
        reason_values = indexer.df.select("exclusion_reason").to_series().to_list()
        assert all(v is None for v in reason_values)

class TestMarkExcluded:
    """Tests for mark_excluded method."""

    def test_mark_excluded_single_sample(self):
        """Test marking a single sample as excluded."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train")  # 0, 1, 2, 3, 4

        n_excluded = indexer.mark_excluded([0], reason="outlier")

        assert n_excluded == 1
        excluded_df = indexer.df.filter(pl.col("excluded"))
        assert len(excluded_df) == 1
        assert excluded_df["sample"][0] == 0
        assert excluded_df["exclusion_reason"][0] == "outlier"

    def test_mark_excluded_multiple_samples(self):
        """Test marking multiple samples as excluded."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train")  # 0, 1, 2, 3, 4

        n_excluded = indexer.mark_excluded([0, 2, 4], reason="low_quality")

        assert n_excluded == 3
        excluded_df = indexer.df.filter(pl.col("excluded"))
        assert len(excluded_df) == 3
        assert set(excluded_df["sample"].to_list()) == {0, 2, 4}

    def test_mark_excluded_without_reason(self):
        """Test marking samples as excluded without providing a reason."""
        indexer = Indexer()
        indexer.add_samples(3, partition="train")

        n_excluded = indexer.mark_excluded([0])

        assert n_excluded == 1
        excluded_df = indexer.df.filter(pl.col("excluded"))
        # exclusion_reason should remain None
        assert excluded_df["exclusion_reason"][0] is None

    def test_mark_excluded_cascade_to_augmented(self):
        """Test that excluding base samples cascades to their augmented versions."""
        indexer = Indexer()
        indexer.add_samples(3, partition="train")  # 0, 1, 2
        indexer.augment_rows([0, 1], 2, "flip")  # 3, 4 for sample 0; 5, 6 for sample 1

        # Exclude sample 0 with cascade (default)
        n_excluded = indexer.mark_excluded([0], reason="outlier", cascade_to_augmented=True)

        # Should exclude sample 0 and its 2 augmented versions
        assert n_excluded == 3
        excluded_samples = indexer.df.filter(pl.col("excluded"))["sample"].to_list()
        assert set(excluded_samples) == {0, 3, 4}

    def test_mark_excluded_no_cascade(self):
        """Test excluding base samples without cascading to augmented."""
        indexer = Indexer()
        indexer.add_samples(3, partition="train")  # 0, 1, 2
        indexer.augment_rows([0, 1], 2, "flip")  # 3, 4 for sample 0; 5, 6 for sample 1

        # Exclude sample 0 without cascade
        n_excluded = indexer.mark_excluded([0], reason="outlier", cascade_to_augmented=False)

        # Should exclude only sample 0
        assert n_excluded == 1
        excluded_samples = indexer.df.filter(pl.col("excluded"))["sample"].to_list()
        assert excluded_samples == [0]

    def test_mark_excluded_with_numpy_array(self):
        """Test marking samples using numpy array as input."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train")

        sample_ids = np.array([0, 1, 2])
        n_excluded = indexer.mark_excluded(sample_ids, reason="test")

        assert n_excluded == 3
        excluded_df = indexer.df.filter(pl.col("excluded"))
        assert len(excluded_df) == 3

    def test_mark_excluded_empty_list(self):
        """Test that marking empty list returns 0."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train")

        n_excluded = indexer.mark_excluded([])
        assert n_excluded == 0

    def test_mark_excluded_idempotent(self):
        """Test that marking already excluded samples is idempotent."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train")

        # Mark twice
        indexer.mark_excluded([0], reason="first")
        indexer.mark_excluded([0], reason="second")

        excluded_df = indexer.df.filter(pl.col("excluded"))
        assert len(excluded_df) == 1
        # Reason should be updated
        assert excluded_df["exclusion_reason"][0] == "second"

class TestMarkIncluded:
    """Tests for mark_included method."""

    def test_mark_included_single_sample(self):
        """Test unmarking a single sample as excluded."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train")
        indexer.mark_excluded([0, 1], reason="outlier")

        n_included = indexer.mark_included([0])

        assert n_included == 1
        excluded_df = indexer.df.filter(pl.col("excluded"))
        assert len(excluded_df) == 1
        assert excluded_df["sample"][0] == 1  # Only sample 1 still excluded

    def test_mark_included_clears_reason(self):
        """Test that marking included clears the exclusion reason."""
        indexer = Indexer()
        indexer.add_samples(3, partition="train")
        indexer.mark_excluded([0], reason="outlier")

        indexer.mark_included([0])

        sample_row = indexer.df.filter(pl.col("sample") == 0)
        assert sample_row["excluded"][0] is False
        assert sample_row["exclusion_reason"][0] is None

    def test_mark_included_all_excluded(self):
        """Test unmarking all excluded samples at once."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train")
        indexer.mark_excluded([0, 1, 2], reason="outlier")

        # Mark all included (no argument)
        n_included = indexer.mark_included()

        assert n_included == 3
        excluded_df = indexer.df.filter(pl.col("excluded"))
        assert len(excluded_df) == 0

    def test_mark_included_cascade_to_augmented(self):
        """Test that including base samples cascades to their augmented versions."""
        indexer = Indexer()
        indexer.add_samples(3, partition="train")  # 0, 1, 2
        indexer.augment_rows([0], 2, "flip")  # 3, 4 for sample 0
        indexer.mark_excluded([0], reason="outlier", cascade_to_augmented=True)  # Excludes 0, 3, 4

        # Include sample 0 with cascade
        n_included = indexer.mark_included([0], cascade_to_augmented=True)

        assert n_included == 3
        excluded_df = indexer.df.filter(pl.col("excluded"))
        assert len(excluded_df) == 0

class TestXIndicesWithExclusion:
    """Tests for x_indices with include_excluded parameter."""

    def test_x_indices_excludes_by_default(self):
        """Test that excluded samples are filtered out by default."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train")  # 0, 1, 2, 3, 4
        indexer.mark_excluded([0, 1], reason="outlier")

        indices = indexer.x_indices({"partition": "train"})

        assert len(indices) == 3
        assert set(indices) == {2, 3, 4}

    def test_x_indices_include_excluded_true(self):
        """Test that excluded samples can be included explicitly."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train")
        indexer.mark_excluded([0, 1], reason="outlier")

        indices = indexer.x_indices({"partition": "train"}, include_excluded=True)

        assert len(indices) == 5
        assert set(indices) == {0, 1, 2, 3, 4}

    def test_x_indices_with_augmentation_and_exclusion(self):
        """Test x_indices with both augmentation and exclusion."""
        indexer = Indexer()
        indexer.add_samples(3, partition="train")  # 0, 1, 2
        indexer.augment_rows([0, 1], 2, "flip")  # 3, 4 for 0; 5, 6 for 1

        # Exclude sample 0 and its augmentations
        indexer.mark_excluded([0], reason="outlier", cascade_to_augmented=True)

        # Should return sample 1, 2 and augmentations of sample 1 (5, 6)
        indices = indexer.x_indices({"partition": "train"})
        assert len(indices) == 4
        assert set(indices) == {1, 2, 5, 6}

    def test_x_indices_base_only_with_exclusion(self):
        """Test x_indices with include_augmented=False and exclusion."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train")
        indexer.augment_rows([0, 1], 2, "flip")
        indexer.mark_excluded([0, 1], reason="outlier")

        # Get base samples only, excluding excluded
        indices = indexer.x_indices({"partition": "train"}, include_augmented=False)

        assert len(indices) == 3
        assert set(indices) == {2, 3, 4}

    def test_x_indices_all_excluded(self):
        """Test x_indices when all samples are excluded."""
        indexer = Indexer()
        indexer.add_samples(3, partition="train")
        indexer.mark_excluded([0, 1, 2], reason="all_bad")

        indices = indexer.x_indices({"partition": "train"})

        assert len(indices) == 0

class TestYIndicesWithExclusion:
    """Tests for y_indices with include_excluded parameter."""

    def test_y_indices_excludes_by_default(self):
        """Test that excluded samples are filtered from y_indices by default."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train")
        indexer.mark_excluded([0, 1], reason="outlier")

        y_idx = indexer.y_indices({"partition": "train"})

        assert len(y_idx) == 3
        assert set(y_idx) == {2, 3, 4}

    def test_y_indices_include_excluded_true(self):
        """Test that excluded samples can be included in y_indices."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train")
        indexer.mark_excluded([0, 1], reason="outlier")

        y_idx = indexer.y_indices({"partition": "train"}, include_excluded=True)

        assert len(y_idx) == 5

    def test_y_indices_x_indices_alignment(self):
        """Test that y_indices and x_indices remain aligned with exclusion."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train")
        indexer.augment_rows([0, 1], 2, "flip")  # 5, 6 for 0; 7, 8 for 1
        indexer.mark_excluded([0], reason="outlier", cascade_to_augmented=True)

        x_idx = indexer.x_indices({"partition": "train"})
        y_idx = indexer.y_indices({"partition": "train"})

        # Lengths should match
        assert len(x_idx) == len(y_idx)

class TestGetExcludedSamples:
    """Tests for get_excluded_samples method."""

    def test_get_excluded_samples_basic(self):
        """Test basic retrieval of excluded samples."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train")
        indexer.mark_excluded([0, 1], reason="outlier")
        indexer.mark_excluded([2], reason="corrupted")

        excluded_df = indexer.get_excluded_samples()

        assert len(excluded_df) == 3
        assert set(excluded_df["sample"].to_list()) == {0, 1, 2}

    def test_get_excluded_samples_with_selector(self):
        """Test get_excluded_samples with partition filter."""
        indexer = Indexer()
        indexer.add_samples(3, partition="train")
        indexer.add_samples(2, partition="test")  # 3, 4
        indexer.mark_excluded([0, 3], reason="outlier")

        # Get only train excluded
        train_excluded = indexer.get_excluded_samples({"partition": "train"})
        assert len(train_excluded) == 1
        assert train_excluded["sample"][0] == 0

        # Get only test excluded
        test_excluded = indexer.get_excluded_samples({"partition": "test"})
        assert len(test_excluded) == 1
        assert test_excluded["sample"][0] == 3

    def test_get_excluded_samples_no_excluded(self):
        """Test get_excluded_samples when no samples are excluded."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train")

        excluded_df = indexer.get_excluded_samples()

        assert len(excluded_df) == 0

    def test_get_excluded_samples_returns_dataframe(self):
        """Test that get_excluded_samples returns a Polars DataFrame."""
        indexer = Indexer()
        indexer.add_samples(3, partition="train")
        indexer.mark_excluded([0], reason="test")

        excluded_df = indexer.get_excluded_samples()

        assert isinstance(excluded_df, pl.DataFrame)
        assert "sample" in excluded_df.columns
        assert "exclusion_reason" in excluded_df.columns

class TestGetExclusionSummary:
    """Tests for get_exclusion_summary method."""

    def test_get_exclusion_summary_basic(self):
        """Test basic exclusion summary."""
        indexer = Indexer()
        indexer.add_samples(10, partition="train")
        indexer.mark_excluded([0, 1], reason="outlier")
        indexer.mark_excluded([2, 3, 4], reason="low_quality")

        summary = indexer.get_exclusion_summary()

        assert summary["total_excluded"] == 5
        assert summary["total_samples"] == 10
        assert summary["exclusion_rate"] == 0.5
        assert summary["by_reason"]["outlier"] == 2
        assert summary["by_reason"]["low_quality"] == 3
        assert summary["by_partition"]["train"] == 5

    def test_get_exclusion_summary_multiple_partitions(self):
        """Test exclusion summary with multiple partitions."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train")
        indexer.add_samples(5, partition="test")  # 5-9
        indexer.mark_excluded([0, 1], reason="outlier")
        indexer.mark_excluded([5, 6, 7], reason="outlier")

        summary = indexer.get_exclusion_summary()

        assert summary["total_excluded"] == 5
        assert summary["by_partition"]["train"] == 2
        assert summary["by_partition"]["test"] == 3

    def test_get_exclusion_summary_no_exclusions(self):
        """Test exclusion summary when no samples are excluded."""
        indexer = Indexer()
        indexer.add_samples(10, partition="train")

        summary = indexer.get_exclusion_summary()

        assert summary["total_excluded"] == 0
        assert summary["exclusion_rate"] == 0.0
        assert summary["by_reason"] == {}
        assert summary["by_partition"] == {}

    def test_get_exclusion_summary_without_reason(self):
        """Test exclusion summary with samples excluded without reason."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train")
        indexer.mark_excluded([0, 1])  # No reason

        summary = indexer.get_exclusion_summary()

        assert summary["total_excluded"] == 2
        assert "unspecified" in summary["by_reason"] or None in summary["by_reason"]

class TestResetExclusions:
    """Tests for reset_exclusions method."""

    def test_reset_all_exclusions(self):
        """Test resetting all exclusions."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train")
        indexer.mark_excluded([0, 1, 2], reason="outlier")

        n_reset = indexer.reset_exclusions()

        assert n_reset == 3
        excluded_df = indexer.df.filter(pl.col("excluded"))
        assert len(excluded_df) == 0

    def test_reset_exclusions_with_selector(self):
        """Test resetting exclusions for specific partition."""
        indexer = Indexer()
        indexer.add_samples(3, partition="train")
        indexer.add_samples(2, partition="test")  # 3, 4
        indexer.mark_excluded([0, 1, 3], reason="outlier")

        # Reset only train exclusions
        n_reset = indexer.reset_exclusions({"partition": "train"})

        assert n_reset == 2  # Only train samples reset
        excluded_df = indexer.df.filter(pl.col("excluded"))
        assert len(excluded_df) == 1
        assert excluded_df["sample"][0] == 3  # Test sample still excluded

class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing code."""

    def test_default_behavior_unchanged(self):
        """Test that default behavior excludes marked samples."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train")
        indexer.mark_excluded([0], reason="outlier")

        # Default behavior should filter excluded samples
        indices = indexer.x_indices({"partition": "train"})
        assert 0 not in indices

    def test_old_api_still_works(self):
        """Test that code not using exclusion still works."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train")
        indexer.augment_rows([0, 1], 2, "flip")

        # Old API without include_excluded parameter
        x_idx = indexer.x_indices({"partition": "train"})
        y_idx = indexer.y_indices({"partition": "train"})

        assert len(x_idx) == 9  # 5 base + 4 augmented
        assert len(y_idx) == 9

    def test_existing_operations_unaffected(self):
        """Test that other operations are unaffected."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train", group=1)

        # Test existing operations
        assert indexer.next_sample_index() == 5
        assert indexer.next_row_index() == 5
        assert 1 in indexer.uniques("group")

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_indexer(self):
        """Test exclusion methods on empty indexer."""
        indexer = Indexer()

        # Should handle gracefully
        assert indexer.mark_excluded([]) == 0
        assert len(indexer.get_excluded_samples()) == 0
        summary = indexer.get_exclusion_summary()
        assert summary["total_samples"] == 0
        assert summary["exclusion_rate"] == 0.0

    def test_exclude_nonexistent_sample(self):
        """Test excluding sample that doesn't exist."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train")

        # Trying to exclude non-existent sample
        n_excluded = indexer.mark_excluded([999], reason="test")

        # Should not crash, may report 1 (update ran) or 0 depending on implementation
        # The important thing is it doesn't break
        assert n_excluded >= 0

    def test_all_samples_excluded_then_included(self):
        """Test excluding and then including all samples."""
        indexer = Indexer()
        indexer.add_samples(5, partition="train")

        # Exclude all
        indexer.mark_excluded([0, 1, 2, 3, 4], reason="test")
        assert len(indexer.x_indices({"partition": "train"})) == 0

        # Include all
        indexer.mark_included()
        assert len(indexer.x_indices({"partition": "train"})) == 5

    def test_exclusion_with_augmentation_chain(self):
        """Test exclusion with complex augmentation scenario."""
        indexer = Indexer()
        indexer.add_samples(2, partition="train")  # 0, 1
        indexer.augment_rows([0], 3, "flip")  # 2, 3, 4
        indexer.augment_rows([1], 2, "rotate")  # 5, 6

        # Exclude sample 0
        indexer.mark_excluded([0], cascade_to_augmented=True)

        # Sample 0 and its augmentations (2, 3, 4) should be excluded
        indices = indexer.x_indices({"partition": "train"})
        assert set(indices) == {1, 5, 6}

    def test_different_reasons_same_samples(self):
        """Test updating exclusion reason."""
        indexer = Indexer()
        indexer.add_samples(3, partition="train")

        indexer.mark_excluded([0], reason="first_reason")
        indexer.mark_excluded([0], reason="updated_reason")

        excluded_df = indexer.get_excluded_samples()
        assert excluded_df["exclusion_reason"][0] == "updated_reason"

class TestQueryBuilderExcludedFilter:
    """Tests for QueryBuilder.build_excluded_filter method."""

    def test_build_excluded_filter_default(self):
        """Test that build_excluded_filter excludes by default."""
        from nirs4all.data._indexer.query_builder import QueryBuilder

        builder = QueryBuilder()
        expr = builder.build_excluded_filter(include_excluded=False)

        # Just verify it returns an expression
        assert expr is not None

    def test_build_excluded_filter_include_all(self):
        """Test that include_excluded=True returns True expression."""
        import polars as pl

        from nirs4all.data._indexer.query_builder import QueryBuilder

        builder = QueryBuilder()
        expr = builder.build_excluded_filter(include_excluded=True)

        # When including excluded, should match all
        # Test with a simple DataFrame
        df = pl.DataFrame({
            "sample": [0, 1, 2],
            "excluded": [True, False, None]
        })

        filtered = df.filter(expr)
        assert len(filtered) == 3

class TestIntegrationWithAugmentation:
    """Integration tests for exclusion with augmentation tracker."""

    def test_excluded_base_sample_augmentations_also_excluded(self):
        """Test that augmentations of excluded base samples are also excluded."""
        indexer = Indexer()
        indexer.add_samples(3, partition="train")  # 0, 1, 2
        indexer.augment_rows([0, 1, 2], 2, "flip")  # 3,4 for 0; 5,6 for 1; 7,8 for 2

        # Exclude sample 0 with cascade
        indexer.mark_excluded([0], cascade_to_augmented=True)

        # x_indices should exclude sample 0 and its augmentations
        indices = indexer.x_indices({"partition": "train"})
        assert 0 not in indices
        assert 3 not in indices  # Augmentation of 0
        assert 4 not in indices  # Augmentation of 0

        # But samples 1, 2 and their augmentations should remain
        assert 1 in indices
        assert 2 in indices
        assert 5 in indices
        assert 6 in indices
        assert 7 in indices
        assert 8 in indices

    def test_exclude_augmented_sample_only(self):
        """Test excluding only specific augmented sample."""
        indexer = Indexer()
        indexer.add_samples(2, partition="train")  # 0, 1
        indexer.augment_rows([0], 3, "flip")  # 2, 3, 4

        # Exclude only one augmented sample
        indexer.mark_excluded([3], cascade_to_augmented=False)

        indices = indexer.x_indices({"partition": "train"})
        assert 0 in indices
        assert 1 in indices
        assert 2 in indices
        assert 3 not in indices  # Excluded
        assert 4 in indices
