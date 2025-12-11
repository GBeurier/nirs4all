"""
Unit tests for SampleFilter base class and CompositeFilter.
"""

import numpy as np
import pytest

from nirs4all.operators.filters.base import SampleFilter, CompositeFilter


class MockKeepAllFilter(SampleFilter):
    """Mock filter that keeps all samples."""

    def get_mask(self, X, y=None):
        return np.ones(len(X), dtype=bool)


class MockKeepNoneFilter(SampleFilter):
    """Mock filter that excludes all samples."""

    def get_mask(self, X, y=None):
        return np.zeros(len(X), dtype=bool)


class MockKeepEvenFilter(SampleFilter):
    """Mock filter that keeps samples at even indices."""

    def get_mask(self, X, y=None):
        mask = np.zeros(len(X), dtype=bool)
        mask[::2] = True
        return mask


class MockKeepOddFilter(SampleFilter):
    """Mock filter that keeps samples at odd indices."""

    def get_mask(self, X, y=None):
        mask = np.zeros(len(X), dtype=bool)
        mask[1::2] = True
        return mask


class TestSampleFilterBase:
    """Tests for SampleFilter base class."""

    def test_abstract_get_mask_raises_error(self):
        """Test that abstract get_mask raises NotImplementedError."""
        # Cannot instantiate abstract class directly
        with pytest.raises(TypeError):
            SampleFilter()

    def test_exclusion_reason_default(self):
        """Test that default exclusion reason is class name."""
        filter_obj = MockKeepAllFilter()
        assert filter_obj.exclusion_reason == "MockKeepAllFilter"

    def test_exclusion_reason_custom(self):
        """Test custom exclusion reason."""
        filter_obj = MockKeepAllFilter(reason="my_reason")
        assert filter_obj.exclusion_reason == "my_reason"

    def test_fit_returns_self(self):
        """Test that default fit returns self."""
        filter_obj = MockKeepAllFilter()
        X = np.random.rand(10, 5)
        result = filter_obj.fit(X)
        assert result is filter_obj

    def test_transform_returns_unchanged(self):
        """Test that transform returns input unchanged."""
        filter_obj = MockKeepAllFilter()
        X = np.random.rand(10, 5)
        X_transformed = filter_obj.transform(X)
        np.testing.assert_array_equal(X, X_transformed)

    def test_get_excluded_indices(self):
        """Test get_excluded_indices helper method."""
        filter_obj = MockKeepEvenFilter()
        X = np.random.rand(10, 5)
        excluded = filter_obj.get_excluded_indices(X)

        # Odd indices should be excluded
        expected = np.array([1, 3, 5, 7, 9])
        np.testing.assert_array_equal(excluded, expected)

    def test_get_kept_indices(self):
        """Test get_kept_indices helper method."""
        filter_obj = MockKeepEvenFilter()
        X = np.random.rand(10, 5)
        kept = filter_obj.get_kept_indices(X)

        # Even indices should be kept
        expected = np.array([0, 2, 4, 6, 8])
        np.testing.assert_array_equal(kept, expected)

    def test_get_filter_stats(self):
        """Test get_filter_stats returns expected statistics."""
        filter_obj = MockKeepEvenFilter()
        X = np.random.rand(10, 5)
        stats = filter_obj.get_filter_stats(X)

        assert stats["n_samples"] == 10
        assert stats["n_kept"] == 5
        assert stats["n_excluded"] == 5
        assert stats["exclusion_rate"] == 0.5
        assert stats["reason"] == "MockKeepEvenFilter"

    def test_get_filter_stats_empty_data(self):
        """Test get_filter_stats with empty data."""
        filter_obj = MockKeepAllFilter()
        X = np.random.rand(0, 5)
        stats = filter_obj.get_filter_stats(X)

        assert stats["n_samples"] == 0
        assert stats["exclusion_rate"] == 0.0


class TestCompositeFilter:
    """Tests for CompositeFilter class."""

    def test_composite_initialization(self):
        """Test CompositeFilter initialization."""
        filter1 = MockKeepEvenFilter()
        filter2 = MockKeepOddFilter()

        composite = CompositeFilter(filters=[filter1, filter2], mode="any")

        assert len(composite.filters) == 2
        assert composite.mode == "any"

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be"):
            CompositeFilter(mode="invalid")

    def test_empty_filters_keeps_all(self):
        """Test that empty filters list keeps all samples."""
        composite = CompositeFilter(filters=[], mode="any")
        X = np.random.rand(10, 5)
        mask = composite.get_mask(X)

        assert mask.sum() == 10

    def test_mode_any_excludes_if_any_flags(self):
        """Test 'any' mode: exclude if ANY filter flags."""
        # Even filter keeps [0,2,4,6,8], Odd filter keeps [1,3,5,7,9]
        # "any" mode with these filters means: exclude if ANY says exclude
        # So: keep only if BOTH say keep -> intersection -> empty!
        filter_even = MockKeepEvenFilter()
        filter_odd = MockKeepOddFilter()

        composite = CompositeFilter(filters=[filter_even, filter_odd], mode="any")
        X = np.random.rand(10, 5)
        mask = composite.get_mask(X)

        # No overlap between even and odd -> nothing kept
        assert mask.sum() == 0

    def test_mode_all_excludes_only_if_all_flag(self):
        """Test 'all' mode: exclude only if ALL filters flag."""
        # Even filter keeps [0,2,4,6,8], Odd filter keeps [1,3,5,7,9]
        # "all" mode: exclude only if ALL say exclude -> union -> all kept!
        filter_even = MockKeepEvenFilter()
        filter_odd = MockKeepOddFilter()

        composite = CompositeFilter(filters=[filter_even, filter_odd], mode="all")
        X = np.random.rand(10, 5)
        mask = composite.get_mask(X)

        # Union of even and odd -> all kept
        assert mask.sum() == 10

    def test_mode_any_with_keep_all_filter(self):
        """Test 'any' mode with a keep-all filter."""
        filter_even = MockKeepEvenFilter()  # Keeps 0,2,4,6,8
        filter_all = MockKeepAllFilter()    # Keeps all

        composite = CompositeFilter(filters=[filter_even, filter_all], mode="any")
        X = np.random.rand(10, 5)
        mask = composite.get_mask(X)

        # "any" -> keep only if ALL keep -> only even indices
        assert mask.sum() == 5
        assert mask[0] == True  # noqa: E712
        assert mask[1] == False  # noqa: E712

    def test_mode_all_with_keep_none_filter(self):
        """Test 'all' mode with a keep-none filter."""
        filter_even = MockKeepEvenFilter()
        filter_none = MockKeepNoneFilter()

        composite = CompositeFilter(filters=[filter_even, filter_none], mode="all")
        X = np.random.rand(10, 5)
        mask = composite.get_mask(X)

        # "all" -> keep if ANY keeps -> even indices kept
        assert mask.sum() == 5

    def test_composite_fit_fits_all_subfilters(self):
        """Test that composite fit calls fit on all sub-filters."""

        class CountingFilter(SampleFilter):
            fit_count = 0

            def fit(self, X, y=None):
                CountingFilter.fit_count += 1
                return self

            def get_mask(self, X, y=None):
                return np.ones(len(X), dtype=bool)

        CountingFilter.fit_count = 0
        filters = [CountingFilter(), CountingFilter(), CountingFilter()]
        composite = CompositeFilter(filters=filters)

        X = np.random.rand(10, 5)
        composite.fit(X)

        assert CountingFilter.fit_count == 3

    def test_composite_exclusion_reason_auto_generated(self):
        """Test auto-generated exclusion reason."""
        filter1 = MockKeepEvenFilter(reason="even")
        filter2 = MockKeepOddFilter(reason="odd")

        composite = CompositeFilter(filters=[filter1, filter2], mode="any")

        assert "composite" in composite.exclusion_reason
        assert "any" in composite.exclusion_reason
        assert "even" in composite.exclusion_reason
        assert "odd" in composite.exclusion_reason

    def test_composite_custom_reason(self):
        """Test custom exclusion reason."""
        composite = CompositeFilter(
            filters=[MockKeepAllFilter()],
            mode="any",
            reason="my_composite"
        )
        assert composite.exclusion_reason == "my_composite"

    def test_composite_get_filter_stats(self):
        """Test get_filter_stats includes per-filter breakdown."""
        filter1 = MockKeepEvenFilter()
        filter2 = MockKeepOddFilter()

        composite = CompositeFilter(filters=[filter1, filter2], mode="all")
        X = np.random.rand(10, 5)
        stats = composite.get_filter_stats(X)

        assert "mode" in stats
        assert stats["mode"] == "all"
        assert "filter_breakdown" in stats
        assert len(stats["filter_breakdown"]) == 2

    def test_single_filter_composite(self):
        """Test composite with single filter behaves like that filter."""
        filter_even = MockKeepEvenFilter()
        composite = CompositeFilter(filters=[filter_even], mode="any")

        X = np.random.rand(10, 5)
        mask_single = filter_even.get_mask(X)
        mask_composite = composite.get_mask(X)

        np.testing.assert_array_equal(mask_single, mask_composite)
