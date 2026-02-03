"""
Comprehensive tests for apply_na_policy() — the centralized NA handling utility.

Tests all 6 canonical policies: auto, abort, remove_sample, remove_feature, replace, ignore.
Also tests edge cases: no-NaN data, unknown policy, NAFillConfig options.
"""

import numpy as np
import pandas as pd
import pytest

from nirs4all.core.exceptions import NAError
from nirs4all.data.loaders.base import apply_na_policy
from nirs4all.data.schema.config import NAFillConfig, NAFillMethod


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def df_with_nan():
    """DataFrame with known NaN positions for deterministic testing."""
    return pd.DataFrame({
        "f1": [1.0, np.nan, 3.0, 4.0, 5.0],
        "f2": [10.0, 20.0, np.nan, 40.0, 50.0],
        "f3": [100.0, 200.0, 300.0, 400.0, 500.0],
    })


@pytest.fixture
def df_clean():
    """DataFrame without any NaN values."""
    return pd.DataFrame({
        "f1": [1.0, 2.0, 3.0],
        "f2": [10.0, 20.0, 30.0],
        "f3": [100.0, 200.0, 300.0],
    })


# =============================================================================
# Test: abort policy
# =============================================================================


class TestAbortPolicy:
    """Tests for na_policy='abort'."""

    def test_abort_raises_on_nan(self, df_with_nan):
        """abort raises NAError when NaN is detected."""
        with pytest.raises(NAError, match="NA values detected"):
            apply_na_policy(df_with_nan, "abort")

    def test_abort_includes_location_info(self, df_with_nan):
        """abort error message includes column and row information."""
        with pytest.raises(NAError, match=r"column.*row"):
            apply_na_policy(df_with_nan, "abort")

    def test_abort_passes_clean_data(self, df_clean):
        """abort returns clean data unchanged."""
        result, report = apply_na_policy(df_clean, "abort")
        pd.testing.assert_frame_equal(result, df_clean)
        assert report["na_detected"] is False


# =============================================================================
# Test: auto policy (resolves to abort)
# =============================================================================


class TestAutoPolicy:
    """Tests for na_policy='auto' (resolves to 'abort')."""

    def test_auto_raises_on_nan(self, df_with_nan):
        """auto resolves to abort and raises NAError on NaN."""
        with pytest.raises(NAError):
            apply_na_policy(df_with_nan, "auto")

    def test_auto_passes_clean_data(self, df_clean):
        """auto returns clean data unchanged."""
        result, report = apply_na_policy(df_clean, "auto")
        pd.testing.assert_frame_equal(result, df_clean)
        assert report["na_detected"] is False


# =============================================================================
# Test: remove_sample policy
# =============================================================================


class TestRemoveSamplePolicy:
    """Tests for na_policy='remove_sample'."""

    def test_drops_correct_rows(self, df_with_nan):
        """remove_sample drops rows that contain NaN."""
        result, report = apply_na_policy(df_with_nan, "remove_sample")
        # Rows 1 (f1=NaN) and 2 (f2=NaN) should be dropped
        assert len(result) == 3
        assert not result.isna().any().any()

    def test_reports_removed_indices(self, df_with_nan):
        """remove_sample reports the indices of removed rows."""
        _, report = apply_na_policy(df_with_nan, "remove_sample")
        assert report["na_detected"] is True
        assert 1 in report["removed_samples"]
        assert 2 in report["removed_samples"]
        assert len(report["removed_samples"]) == 2

    def test_clean_data_unchanged(self, df_clean):
        """remove_sample returns clean data unchanged."""
        result, report = apply_na_policy(df_clean, "remove_sample")
        pd.testing.assert_frame_equal(result, df_clean)
        assert report["removed_samples"] == []


# =============================================================================
# Test: remove_feature policy
# =============================================================================


class TestRemoveFeaturePolicy:
    """Tests for na_policy='remove_feature'."""

    def test_drops_correct_columns(self, df_with_nan):
        """remove_feature drops columns that contain NaN."""
        result, report = apply_na_policy(df_with_nan, "remove_feature")
        # f1 and f2 have NaN, f3 is clean
        assert list(result.columns) == ["f3"]
        assert len(result) == 5  # All rows preserved

    def test_reports_removed_features(self, df_with_nan):
        """remove_feature reports the names of removed columns."""
        _, report = apply_na_policy(df_with_nan, "remove_feature")
        assert report["na_detected"] is True
        assert "f1" in report["removed_features"]
        assert "f2" in report["removed_features"]
        assert "f3" not in report["removed_features"]

    def test_warns_when_many_columns_removed(self):
        """remove_feature warns when >10% of columns are removed."""
        # Create DataFrame where most columns have NaN
        df = pd.DataFrame({
            f"c{i}": [np.nan if i < 8 else 1.0] for i in range(10)
        })
        with pytest.warns(UserWarning, match="exceeds 10%"):
            apply_na_policy(df, "remove_feature")

    def test_clean_data_unchanged(self, df_clean):
        """remove_feature returns clean data unchanged."""
        result, report = apply_na_policy(df_clean, "remove_feature")
        pd.testing.assert_frame_equal(result, df_clean)
        assert report["removed_features"] == []


# =============================================================================
# Test: replace policy
# =============================================================================


class TestReplacePolicy:
    """Tests for na_policy='replace' with various NAFillMethod options."""

    def test_replace_with_value(self, df_with_nan):
        """replace with NAFillMethod.VALUE fills NaN with specified value."""
        config = NAFillConfig(method=NAFillMethod.VALUE, fill_value=-1.0)
        result, report = apply_na_policy(df_with_nan, "replace", config)
        assert result.loc[1, "f1"] == -1.0
        assert result.loc[2, "f2"] == -1.0
        assert report["fill_method"] == "value"

    def test_replace_with_default_config(self, df_with_nan):
        """replace with default NAFillConfig fills with 0.0."""
        result, report = apply_na_policy(df_with_nan, "replace")
        assert result.loc[1, "f1"] == 0.0
        assert result.loc[2, "f2"] == 0.0
        assert report["fill_method"] == "value"

    def test_replace_with_mean_per_column(self, df_with_nan):
        """replace with MEAN per_column=True fills with column means."""
        config = NAFillConfig(method=NAFillMethod.MEAN, per_column=True)
        result, report = apply_na_policy(df_with_nan, "replace", config)
        # f1 mean (excluding NaN): (1+3+4+5)/4 = 3.25
        assert abs(result.loc[1, "f1"] - 3.25) < 1e-10
        # f2 mean (excluding NaN): (10+40+50)/3 ≈ 33.33 — wait, there are 4 non-NaN: (10+20+40+50)/4 = 30.0
        assert abs(result.loc[2, "f2"] - 30.0) < 1e-10
        assert report["fill_method"] == "mean"

    def test_replace_with_mean_global(self, df_with_nan):
        """replace with MEAN per_column=False fills with global mean."""
        config = NAFillConfig(method=NAFillMethod.MEAN, per_column=False)
        result, report = apply_na_policy(df_with_nan, "replace", config)
        # Global mean of all non-NaN values
        all_values = df_with_nan.values.flatten()
        global_mean = np.nanmean(all_values)
        assert abs(result.loc[1, "f1"] - global_mean) < 1e-10
        assert abs(result.loc[2, "f2"] - global_mean) < 1e-10

    def test_replace_with_median_per_column(self, df_with_nan):
        """replace with MEDIAN per_column=True fills with column medians."""
        config = NAFillConfig(method=NAFillMethod.MEDIAN, per_column=True)
        result, report = apply_na_policy(df_with_nan, "replace", config)
        # f1 median (excluding NaN): median(1,3,4,5) = 3.5
        assert abs(result.loc[1, "f1"] - 3.5) < 1e-10
        assert report["fill_method"] == "median"

    def test_replace_with_median_global(self, df_with_nan):
        """replace with MEDIAN per_column=False fills with global median."""
        config = NAFillConfig(method=NAFillMethod.MEDIAN, per_column=False)
        result, report = apply_na_policy(df_with_nan, "replace", config)
        all_values = df_with_nan.values.flatten()
        global_median = float(np.nanmedian(all_values))
        assert abs(result.loc[1, "f1"] - global_median) < 1e-10

    def test_replace_with_forward_fill(self, df_with_nan):
        """replace with FORWARD_FILL fills NaN using forward fill along columns."""
        config = NAFillConfig(method=NAFillMethod.FORWARD_FILL)
        result, report = apply_na_policy(df_with_nan, "replace", config)
        # Forward fill along axis=1 (columns): row 1: f1=NaN -> stays NaN (no left neighbor) or is filled
        # Actually ffill(axis=1): for row 1, f1=NaN has no preceding value... let's check
        # Row 1: [NaN, 20.0, 200.0] -> ffill axis=1 -> [NaN, 20.0, 200.0] (first col can't forward fill)
        # Row 2: [3.0, NaN, 300.0] -> ffill axis=1 -> [3.0, 3.0, 300.0]
        assert result.loc[2, "f2"] == 3.0
        assert report["fill_method"] == "forward_fill"

    def test_replace_with_backward_fill(self, df_with_nan):
        """replace with BACKWARD_FILL fills NaN using backward fill along columns."""
        config = NAFillConfig(method=NAFillMethod.BACKWARD_FILL)
        result, report = apply_na_policy(df_with_nan, "replace", config)
        # Backward fill along axis=1:
        # Row 1: [NaN, 20.0, 200.0] -> bfill axis=1 -> [20.0, 20.0, 200.0]
        assert result.loc[1, "f1"] == 20.0
        # Row 2: [3.0, NaN, 300.0] -> bfill axis=1 -> [3.0, 300.0, 300.0]
        assert result.loc[2, "f2"] == 300.0
        assert report["fill_method"] == "backward_fill"

    def test_replace_clean_data_unchanged(self, df_clean):
        """replace returns clean data unchanged."""
        config = NAFillConfig(method=NAFillMethod.VALUE, fill_value=-1.0)
        result, report = apply_na_policy(df_clean, "replace", config)
        pd.testing.assert_frame_equal(result, df_clean)


# =============================================================================
# Test: ignore policy
# =============================================================================


class TestIgnorePolicy:
    """Tests for na_policy='ignore'."""

    def test_data_returned_unchanged(self, df_with_nan):
        """ignore returns data with NaN values preserved."""
        result, report = apply_na_policy(df_with_nan, "ignore")
        pd.testing.assert_frame_equal(result, df_with_nan)

    def test_na_preserved_flag(self, df_with_nan):
        """ignore sets na_preserved=True in report."""
        _, report = apply_na_policy(df_with_nan, "ignore")
        assert report["na_preserved"] is True
        assert report["na_detected"] is True

    def test_clean_data_unchanged(self, df_clean):
        """ignore returns clean data unchanged."""
        result, report = apply_na_policy(df_clean, "ignore")
        pd.testing.assert_frame_equal(result, df_clean)
        assert report["na_preserved"] is False


# =============================================================================
# Test: clean data (no NaN) for all policies
# =============================================================================


class TestCleanDataAllPolicies:
    """Verify that all policies return clean data unchanged when there are no NaN values."""

    @pytest.mark.parametrize("policy", [
        "auto", "abort", "remove_sample", "remove_feature", "replace", "ignore",
    ])
    def test_no_nan_returns_unchanged(self, df_clean, policy):
        """All policies return data unchanged when no NaN is present."""
        result, report = apply_na_policy(df_clean, policy)
        pd.testing.assert_frame_equal(result, df_clean)
        assert report["na_detected"] is False


# =============================================================================
# Test: unknown policy
# =============================================================================


class TestUnknownPolicy:
    """Tests for invalid/unknown policy values."""

    def test_unknown_policy_raises_value_error(self, df_clean):
        """Unknown policy raises ValueError."""
        with pytest.raises(ValueError, match="Invalid na_policy"):
            apply_na_policy(df_clean, "unknown_policy")

    def test_old_remove_policy_raises(self, df_clean):
        """Old 'remove' policy (dead vocabulary) raises ValueError."""
        with pytest.raises(ValueError, match="Invalid na_policy"):
            apply_na_policy(df_clean, "remove")

    def test_old_drop_policy_raises(self, df_clean):
        """Old 'drop' policy (dead vocabulary) raises ValueError."""
        with pytest.raises(ValueError, match="Invalid na_policy"):
            apply_na_policy(df_clean, "drop")

    def test_old_keep_policy_raises(self, df_clean):
        """Old 'keep' policy (dead vocabulary) raises ValueError."""
        with pytest.raises(ValueError, match="Invalid na_policy"):
            apply_na_policy(df_clean, "keep")

    def test_old_fill_mean_policy_raises(self, df_clean):
        """Old 'fill_mean' policy (dead vocabulary) raises ValueError."""
        with pytest.raises(ValueError, match="Invalid na_policy"):
            apply_na_policy(df_clean, "fill_mean")


# =============================================================================
# Test: report structure
# =============================================================================


class TestReportStructure:
    """Tests for the structure of the na_handling report dict."""

    def test_report_keys_present(self, df_with_nan):
        """Report contains all expected keys."""
        _, report = apply_na_policy(df_with_nan, "remove_sample")
        expected_keys = {
            "strategy", "na_detected", "na_count", "na_samples",
            "na_features", "removed_samples", "removed_features",
            "fill_method", "na_preserved",
        }
        assert expected_keys.issubset(set(report.keys()))

    def test_report_strategy_matches_resolved_policy(self, df_with_nan):
        """Report strategy reflects the resolved policy (auto -> abort)."""
        _, report = apply_na_policy(df_with_nan, "remove_sample")
        assert report["strategy"] == "remove_sample"

    def test_report_na_count_accurate(self, df_with_nan):
        """Report na_count accurately counts total NaN cells."""
        _, report = apply_na_policy(df_with_nan, "remove_sample")
        # df_with_nan has 2 NaN cells (row 1 f1, row 2 f2)
        assert report["na_count"] == 2

    def test_report_na_samples_accurate(self, df_with_nan):
        """Report na_samples accurately counts rows with NaN."""
        _, report = apply_na_policy(df_with_nan, "remove_sample")
        assert report["na_samples"] == 2

    def test_report_na_features_accurate(self, df_with_nan):
        """Report na_features accurately counts columns with NaN."""
        _, report = apply_na_policy(df_with_nan, "remove_sample")
        assert report["na_features"] == 2  # f1 and f2 have NaN
