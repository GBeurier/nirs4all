"""
Tests for ColumnSelector class.

Tests column selection by name, index, range, regex pattern, and exclusion.
"""

import numpy as np
import pandas as pd
import pytest

from nirs4all.data.selection import ColumnSelectionError, ColumnSelector


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "sample_name": ["A", "B", "C", "D", "E"],
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature_2": [0.1, 0.2, 0.3, 0.4, 0.5],
        "feature_3": [10, 20, 30, 40, 50],
        "target": [0, 1, 0, 1, 0],
    })

@pytest.fixture
def spectral_df():
    """Create a spectral-like DataFrame with numeric column names."""
    cols = [str(i) for i in range(1000, 2000, 100)]  # 10 wavelength columns
    data = np.random.randn(5, len(cols))
    return pd.DataFrame(data, columns=cols)

class TestColumnSelectorBasic:
    """Basic column selection tests."""

    def test_select_none_returns_all(self, sample_df):
        """Test that None selection returns all columns."""
        selector = ColumnSelector()
        result = selector.select(sample_df, None)

        assert result.indices == list(range(len(sample_df.columns)))
        assert result.names == sample_df.columns.tolist()
        assert result.data.equals(sample_df)

    def test_select_single_index(self, sample_df):
        """Test selecting a single column by index."""
        selector = ColumnSelector()
        result = selector.select(sample_df, 0)

        assert result.indices == [0]
        assert result.names == ["id"]
        assert list(result.data.columns) == ["id"]

    def test_select_negative_index(self, sample_df):
        """Test selecting a column by negative index."""
        selector = ColumnSelector()
        result = selector.select(sample_df, -1)

        assert result.indices == [5]
        assert result.names == ["target"]

    def test_select_invalid_index_raises(self, sample_df):
        """Test that invalid index raises error."""
        selector = ColumnSelector()

        with pytest.raises(ColumnSelectionError, match="out of range"):
            selector.select(sample_df, 10)

    def test_select_single_name(self, sample_df):
        """Test selecting a single column by name."""
        selector = ColumnSelector()
        result = selector.select(sample_df, "feature_1")

        assert result.indices == [2]
        assert result.names == ["feature_1"]

    def test_select_invalid_name_raises(self, sample_df):
        """Test that invalid name raises error."""
        selector = ColumnSelector()

        with pytest.raises(ColumnSelectionError, match="not found"):
            selector.select(sample_df, "nonexistent")

class TestColumnSelectorLists:
    """Tests for list-based column selection."""

    def test_select_by_index_list(self, sample_df):
        """Test selecting multiple columns by index list."""
        selector = ColumnSelector()
        result = selector.select(sample_df, [0, 2, 4])

        assert result.indices == [0, 2, 4]
        assert result.names == ["id", "feature_1", "feature_3"]

    def test_select_by_index_list_with_negative(self, sample_df):
        """Test selecting with negative indices in list."""
        selector = ColumnSelector()
        result = selector.select(sample_df, [0, -1])

        assert result.indices == [0, 5]
        assert result.names == ["id", "target"]

    def test_select_by_name_list(self, sample_df):
        """Test selecting multiple columns by name list."""
        selector = ColumnSelector()
        result = selector.select(sample_df, ["feature_1", "feature_2"])

        assert result.indices == [2, 3]
        assert result.names == ["feature_1", "feature_2"]

    def test_empty_list_raises(self, sample_df):
        """Test that empty list raises error."""
        selector = ColumnSelector()

        with pytest.raises(ColumnSelectionError, match="Empty selection"):
            selector.select(sample_df, [])

class TestColumnSelectorRanges:
    """Tests for range-based column selection."""

    def test_select_range_string(self, sample_df):
        """Test selecting columns by range string."""
        selector = ColumnSelector()
        result = selector.select(sample_df, "2:5")

        assert result.indices == [2, 3, 4]
        assert result.names == ["feature_1", "feature_2", "feature_3"]

    def test_select_range_from_start(self, sample_df):
        """Test selecting columns from start."""
        selector = ColumnSelector()
        result = selector.select(sample_df, ":3")

        assert result.indices == [0, 1, 2]
        assert result.names == ["id", "sample_name", "feature_1"]

    def test_select_range_to_end(self, sample_df):
        """Test selecting columns to end."""
        selector = ColumnSelector()
        result = selector.select(sample_df, "3:")

        assert result.indices == [3, 4, 5]
        assert result.names == ["feature_2", "feature_3", "target"]

    def test_select_range_with_negative(self, sample_df):
        """Test selecting columns with negative range."""
        selector = ColumnSelector()
        result = selector.select(sample_df, "2:-1")

        assert result.indices == [2, 3, 4]
        assert result.names == ["feature_1", "feature_2", "feature_3"]

    def test_select_range_with_step(self, sample_df):
        """Test selecting columns with step."""
        selector = ColumnSelector()
        result = selector.select(sample_df, "0:6:2")

        assert result.indices == [0, 2, 4]
        assert result.names == ["id", "feature_1", "feature_3"]

    def test_select_slice_object(self, sample_df):
        """Test selecting columns with slice object."""
        selector = ColumnSelector()
        result = selector.select(sample_df, slice(1, 4))

        assert result.indices == [1, 2, 3]
        assert result.names == ["sample_name", "feature_1", "feature_2"]

class TestColumnSelectorDict:
    """Tests for dictionary-based column selection."""

    def test_select_by_regex(self, sample_df):
        """Test selecting columns by regex pattern."""
        selector = ColumnSelector()
        result = selector.select(sample_df, {"regex": "^feature_.*"})

        assert result.indices == [2, 3, 4]
        assert result.names == ["feature_1", "feature_2", "feature_3"]

    def test_select_by_startswith(self, sample_df):
        """Test selecting columns by prefix."""
        selector = ColumnSelector()
        result = selector.select(sample_df, {"startswith": "feature"})

        assert result.indices == [2, 3, 4]
        assert result.names == ["feature_1", "feature_2", "feature_3"]

    def test_select_by_endswith(self, sample_df):
        """Test selecting columns by suffix."""
        selector = ColumnSelector()
        result = selector.select(sample_df, {"endswith": "_1"})

        assert result.indices == [2]
        assert result.names == ["feature_1"]

    def test_select_by_contains(self, sample_df):
        """Test selecting columns by substring."""
        selector = ColumnSelector()
        result = selector.select(sample_df, {"contains": "ure"})

        assert result.indices == [2, 3, 4]
        assert result.names == ["feature_1", "feature_2", "feature_3"]

    def test_select_by_exclude(self, sample_df):
        """Test excluding columns."""
        selector = ColumnSelector()
        result = selector.select(sample_df, {"exclude": ["id", "target"]})

        assert result.indices == [1, 2, 3, 4]
        assert result.names == ["sample_name", "feature_1", "feature_2", "feature_3"]

    def test_select_with_include_and_exclude(self, sample_df):
        """Test combining include and exclude."""
        selector = ColumnSelector()
        result = selector.select(sample_df, {
            "include": ["id", "feature_1", "feature_2", "target"],
            "exclude": ["target"]
        })

        assert result.indices == [0, 2, 3]
        assert result.names == ["id", "feature_1", "feature_2"]

    def test_select_by_dtype(self, sample_df):
        """Test selecting columns by dtype."""
        selector = ColumnSelector()
        result = selector.select(sample_df, {"dtype": "float"})

        assert "feature_1" in result.names
        assert "feature_2" in result.names

    def test_invalid_regex_raises(self, sample_df):
        """Test that invalid regex raises error."""
        selector = ColumnSelector()

        with pytest.raises(ColumnSelectionError, match="Invalid regex"):
            selector.select(sample_df, {"regex": "["})

class TestColumnSelectorCaseSensitivity:
    """Tests for case sensitivity options."""

    def test_case_sensitive_by_default(self, sample_df):
        """Test that selection is case-sensitive by default."""
        selector = ColumnSelector(case_sensitive=True)

        with pytest.raises(ColumnSelectionError, match="not found"):
            selector.select(sample_df, "FEATURE_1")

    def test_case_insensitive_option(self, sample_df):
        """Test case-insensitive selection."""
        selector = ColumnSelector(case_sensitive=False)
        result = selector.select(sample_df, "FEATURE_1")

        assert result.indices == [2]
        assert result.names == ["feature_1"]

    def test_case_insensitive_regex(self, sample_df):
        """Test case-insensitive regex."""
        selector = ColumnSelector(case_sensitive=False)
        result = selector.select(sample_df, {"regex": "^FEATURE_.*"})

        assert len(result.indices) == 3

class TestColumnSelectorSpectral:
    """Tests with spectral-like data (numeric column names)."""

    def test_select_wavelength_range(self, spectral_df):
        """Test selecting wavelength columns by range."""
        selector = ColumnSelector()
        result = selector.select(spectral_df, "2:8")

        assert len(result.indices) == 6

    def test_select_all_numeric_columns(self, spectral_df):
        """Test selecting all columns from spectral data."""
        selector = ColumnSelector()
        result = selector.select(spectral_df, None)

        assert len(result.indices) == 10

class TestColumnSelectorParse:
    """Tests for parse_selection method."""

    def test_parse_selection(self):
        """Test parsing selection without DataFrame."""
        selector = ColumnSelector()
        columns = ["id", "feature_1", "feature_2", "target"]

        indices = selector.parse_selection([0, 1, 2], columns)
        assert indices == [0, 1, 2]

        indices = selector.parse_selection("1:-1", columns)
        assert indices == [1, 2]
