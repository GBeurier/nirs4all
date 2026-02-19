"""
Tests for SampleLinker class.

Tests key-based sample linking across multiple files.
"""

import numpy as np
import pandas as pd
import pytest

from nirs4all.data.selection import LinkingError, SampleLinker
from nirs4all.data.selection.sample_linker import link_xy, link_xym


@pytest.fixture
def features_df():
    """Create a features DataFrame."""
    return pd.DataFrame({
        "sample_id": [1, 2, 3, 4, 5],
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature_2": [0.1, 0.2, 0.3, 0.4, 0.5],
    })

@pytest.fixture
def targets_df():
    """Create a targets DataFrame with same keys."""
    return pd.DataFrame({
        "sample_id": [1, 2, 3, 4, 5],
        "target": [0, 1, 0, 1, 0],
    })

@pytest.fixture
def metadata_df():
    """Create a metadata DataFrame with same keys."""
    return pd.DataFrame({
        "sample_id": [1, 2, 3, 4, 5],
        "group": ["A", "B", "A", "B", "A"],
        "date": ["2024-01-01"] * 5,
    })

@pytest.fixture
def partial_targets_df():
    """Create a targets DataFrame with fewer keys."""
    return pd.DataFrame({
        "sample_id": [1, 2, 3],  # Only 3 samples
        "target": [0, 1, 0],
    })

@pytest.fixture
def extra_targets_df():
    """Create a targets DataFrame with extra keys."""
    return pd.DataFrame({
        "sample_id": [1, 2, 3, 4, 5, 6, 7],  # 2 extra samples
        "target": [0, 1, 0, 1, 0, 1, 1],
    })

class TestSampleLinkerBasic:
    """Basic sample linking tests."""

    def test_link_matching_keys(self, features_df, targets_df):
        """Test linking DataFrames with matching keys."""
        linker = SampleLinker()
        result = linker.link(
            {"X": features_df, "Y": targets_df},
            link_by="sample_id"
        )

        assert result.sample_count == 5
        assert len(result.matched_keys) == 5
        assert len(result.linked_data["X"]) == 5
        assert len(result.linked_data["Y"]) == 5

        # Key column should be removed by default
        assert "sample_id" not in result.linked_data["X"].columns
        assert "sample_id" not in result.linked_data["Y"].columns

    def test_link_keep_key_column(self, features_df, targets_df):
        """Test keeping the key column in output."""
        linker = SampleLinker()
        result = linker.link(
            {"X": features_df, "Y": targets_df},
            link_by="sample_id",
            keep_key_column=True
        )

        assert "sample_id" in result.linked_data["X"].columns
        assert "sample_id" in result.linked_data["Y"].columns

    def test_link_single_source(self, features_df):
        """Test linking with single source (pass-through)."""
        linker = SampleLinker()
        result = linker.link(
            {"X": features_df},
            link_by="sample_id"
        )

        assert result.sample_count == 5
        assert len(result.linked_data["X"]) == 5

    def test_link_missing_key_column_raises(self, features_df):
        """Test error when key column is missing."""
        linker = SampleLinker()
        df_no_key = features_df.drop(columns=["sample_id"])

        with pytest.raises(LinkingError, match="not found"):
            linker.link(
                {"X": df_no_key},
                link_by="sample_id"
            )

class TestSampleLinkerModes:
    """Tests for different linking modes."""

    def test_inner_mode_default(self, features_df, partial_targets_df):
        """Test inner mode (default) keeps only matching keys."""
        linker = SampleLinker(mode="inner")
        result = linker.link(
            {"X": features_df, "Y": partial_targets_df},
            link_by="sample_id"
        )

        # Only 3 matching keys
        assert result.sample_count == 3
        assert len(result.matched_keys) == 3

    def test_left_mode_keeps_first_source_keys(self, features_df, partial_targets_df):
        """Test left mode keeps all keys from first source."""
        linker = SampleLinker(mode="left")
        result = linker.link(
            {"X": features_df, "Y": partial_targets_df},
            link_by="sample_id"
        )

        # All 5 keys from X
        assert len(result.matched_keys) == 5

    def test_outer_mode_keeps_all_keys(self, features_df, extra_targets_df):
        """Test outer mode keeps all keys from any source."""
        linker = SampleLinker(mode="outer")
        result = linker.link(
            {"X": features_df, "Y": extra_targets_df},
            link_by="sample_id"
        )

        # All 7 keys (5 from X, 2 extra from Y)
        assert len(result.matched_keys) == 7

class TestSampleLinkerMissingKeys:
    """Tests for handling missing keys."""

    def test_missing_keys_warn_by_default(self, features_df, partial_targets_df):
        """Test that missing keys produce warning by default."""
        linker = SampleLinker(mode="left", on_missing="warn")

        with pytest.warns(UserWarning, match="Missing keys"):
            result = linker.link(
                {"X": features_df, "Y": partial_targets_df},
                link_by="sample_id"
            )

        # In left mode, keys 4, 5 are in X but missing from Y
        assert len(result.missing_keys["Y"]) == 2

    def test_missing_keys_error(self, features_df, partial_targets_df):
        """Test that missing keys raise error when configured."""
        linker = SampleLinker(mode="left", on_missing="error")

        with pytest.raises(LinkingError, match="Missing keys"):
            linker.link(
                {"X": features_df, "Y": partial_targets_df},
                link_by="sample_id"
            )

    def test_missing_keys_ignore(self, features_df, partial_targets_df):
        """Test that missing keys are silently ignored."""
        linker = SampleLinker(on_missing="ignore")

        # Should not warn or raise
        result = linker.link(
            {"X": features_df, "Y": partial_targets_df},
            link_by="sample_id"
        )

        assert result.sample_count == 3

class TestSampleLinkerThreeSources:
    """Tests for linking three sources."""

    def test_link_three_sources(self, features_df, targets_df, metadata_df):
        """Test linking three DataFrames."""
        linker = SampleLinker()
        result = linker.link(
            {"X": features_df, "Y": targets_df, "M": metadata_df},
            link_by="sample_id"
        )

        assert result.sample_count == 5
        assert len(result.linked_data) == 3
        assert "X" in result.linked_data
        assert "Y" in result.linked_data
        assert "M" in result.linked_data

    def test_link_three_sources_with_partial(
        self, features_df, partial_targets_df, metadata_df
    ):
        """Test linking three sources with partial match."""
        linker = SampleLinker(mode="inner", on_missing="ignore")
        result = linker.link(
            {"X": features_df, "Y": partial_targets_df, "M": metadata_df},
            link_by="sample_id"
        )

        # Inner join should have only 3 samples
        assert result.sample_count == 3

class TestSampleLinkerAlignment:
    """Tests for row alignment."""

    def test_linked_rows_aligned(self, features_df, targets_df):
        """Test that linked rows are properly aligned."""
        linker = SampleLinker()
        result = linker.link(
            {"X": features_df, "Y": targets_df},
            link_by="sample_id",
            keep_key_column=True
        )

        X = result.linked_data["X"]
        Y = result.linked_data["Y"]

        # Keys should match row by row
        assert list(X["sample_id"]) == list(Y["sample_id"])

    def test_shuffled_keys_aligned(self):
        """Test alignment when keys are in different order."""
        features = pd.DataFrame({
            "sample_id": [5, 3, 1, 4, 2],
            "feature": [5.0, 3.0, 1.0, 4.0, 2.0],
        })
        targets = pd.DataFrame({
            "sample_id": [1, 2, 3, 4, 5],
            "target": [1, 2, 3, 4, 5],
        })

        linker = SampleLinker()
        result = linker.link(
            {"X": features, "Y": targets},
            link_by="sample_id",
            keep_key_column=True
        )

        X = result.linked_data["X"]
        Y = result.linked_data["Y"]

        # Should be sorted by key
        assert list(X["sample_id"]) == list(Y["sample_id"])
        assert list(X["sample_id"]) == [1, 2, 3, 4, 5]

class TestSampleLinkerAligned:
    """Tests for aligned source linking."""

    def test_link_aligned_valid(self, features_df, targets_df):
        """Test linking pre-aligned sources."""
        # Remove key column
        X = features_df.drop(columns=["sample_id"])
        Y = targets_df.drop(columns=["sample_id"])

        linker = SampleLinker()
        result = linker.link_aligned({"X": X, "Y": Y})

        assert len(result["X"]) == 5
        assert len(result["Y"]) == 5

    def test_link_aligned_mismatched_raises(self, features_df, partial_targets_df):
        """Test that mismatched row counts raise error."""
        X = features_df.drop(columns=["sample_id"])
        Y = partial_targets_df.drop(columns=["sample_id"])

        linker = SampleLinker()

        with pytest.raises(LinkingError, match="different row counts"):
            linker.link_aligned({"X": X, "Y": Y})

class TestSampleLinkerIndex:
    """Tests for sample index creation."""

    def test_create_sample_index(self, features_df, partial_targets_df):
        """Test creating sample index."""
        linker = SampleLinker()
        index = linker.create_sample_index(
            {"X": features_df, "Y": partial_targets_df},
            link_by="sample_id"
        )

        assert len(index) == 5
        assert "in_X" in index.columns
        assert "in_Y" in index.columns
        assert "in_all" in index.columns

        # Check values (use bool() to convert numpy booleans)
        assert bool(index.loc[1, "in_X"]) is True
        assert bool(index.loc[1, "in_Y"]) is True
        assert bool(index.loc[4, "in_X"]) is True
        assert bool(index.loc[4, "in_Y"]) is False
        assert bool(index.loc[1, "in_all"]) is True
        assert bool(index.loc[4, "in_all"]) is False

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_link_xy(self, features_df, targets_df):
        """Test link_xy convenience function."""
        X, Y = link_xy(features_df, targets_df, link_by="sample_id")

        assert len(X) == 5
        assert len(Y) == 5
        assert "sample_id" not in X.columns
        assert "sample_id" not in Y.columns

    def test_link_xym(self, features_df, targets_df, metadata_df):
        """Test link_xym convenience function."""
        X, Y, M = link_xym(
            features_df, targets_df, metadata_df,
            link_by="sample_id"
        )

        assert len(X) == 5
        assert len(Y) == 5
        assert len(M) == 5

class TestSampleLinkerReport:
    """Tests for linking report."""

    def test_report_contains_info(self, features_df, targets_df):
        """Test that report contains useful information."""
        linker = SampleLinker()
        result = linker.link(
            {"X": features_df, "Y": targets_df},
            link_by="sample_id"
        )

        assert "mode" in result.report
        assert "original_counts" in result.report
        assert "matched_key_count" in result.report
        assert result.report["matched_key_count"] == 5
