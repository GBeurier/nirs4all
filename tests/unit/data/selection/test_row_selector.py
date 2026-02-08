"""
Tests for RowSelector class.

Tests row selection by index, range, percentage, condition, and sampling.
"""

import pytest
import numpy as np
import pandas as pd

from nirs4all.data.selection import RowSelector, RowSelectionError


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "id": list(range(100)),
        "category": ["A", "B", "C", "D"] * 25,
        "value": np.random.randn(100),
        "quality": np.random.rand(100),
        "label": [0, 1] * 50,
    })


@pytest.fixture
def small_df():
    """Create a small DataFrame for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "value": [10.0, 20.0, 30.0, 40.0, 50.0],
        "label": ["A", "B", "A", "B", "A"],
    })


class TestRowSelectorBasic:
    """Basic row selection tests."""

    def test_select_none_returns_all(self, sample_df):
        """Test that None selection returns all rows."""
        selector = RowSelector()
        result = selector.select(sample_df, None)

        assert result.indices == list(range(len(sample_df)))
        assert len(result.data) == len(sample_df)
        assert result.mask.all()

    def test_select_single_index(self, small_df):
        """Test selecting a single row by index."""
        selector = RowSelector()
        result = selector.select(small_df, 0)

        assert result.indices == [0]
        assert len(result.data) == 1
        assert result.data.iloc[0]["id"] == 1

    def test_select_negative_index(self, small_df):
        """Test selecting a row by negative index."""
        selector = RowSelector()
        result = selector.select(small_df, -1)

        assert result.indices == [4]
        assert result.data.iloc[0]["id"] == 5

    def test_select_invalid_index_raises(self, small_df):
        """Test that invalid index raises error."""
        selector = RowSelector()

        with pytest.raises(RowSelectionError, match="out of range"):
            selector.select(small_df, 10)


class TestRowSelectorLists:
    """Tests for list-based row selection."""

    def test_select_by_index_list(self, small_df):
        """Test selecting multiple rows by index list."""
        selector = RowSelector()
        result = selector.select(small_df, [0, 2, 4])

        assert result.indices == [0, 2, 4]
        assert len(result.data) == 3
        assert list(result.data["id"]) == [1, 3, 5]

    def test_select_by_index_list_with_negative(self, small_df):
        """Test selecting with negative indices in list."""
        selector = RowSelector()
        result = selector.select(small_df, [0, -1])

        assert result.indices == [0, 4]
        assert len(result.data) == 2


class TestRowSelectorRanges:
    """Tests for range-based row selection."""

    def test_select_range_string(self, sample_df):
        """Test selecting rows by range string."""
        selector = RowSelector()
        result = selector.select(sample_df, "0:10")

        assert result.indices == list(range(10))
        assert len(result.data) == 10

    def test_select_range_from_start(self, sample_df):
        """Test selecting rows from start."""
        selector = RowSelector()
        result = selector.select(sample_df, ":20")

        assert len(result.data) == 20

    def test_select_range_to_end(self, sample_df):
        """Test selecting rows to end."""
        selector = RowSelector()
        result = selector.select(sample_df, "90:")

        assert len(result.data) == 10

    def test_select_range_with_step(self, sample_df):
        """Test selecting rows with step."""
        selector = RowSelector()
        result = selector.select(sample_df, "0:10:2")

        assert result.indices == [0, 2, 4, 6, 8]
        assert len(result.data) == 5


class TestRowSelectorPercentage:
    """Tests for percentage-based row selection."""

    def test_select_first_percentage(self, sample_df):
        """Test selecting first percentage of rows."""
        selector = RowSelector()
        result = selector.select(sample_df, "0:80%")

        assert len(result.data) == 80

    def test_select_last_percentage(self, sample_df):
        """Test selecting last percentage of rows."""
        selector = RowSelector()
        result = selector.select(sample_df, "80%:100%")

        assert len(result.data) == 20

    def test_select_middle_percentage(self, sample_df):
        """Test selecting middle percentage of rows."""
        selector = RowSelector()
        result = selector.select(sample_df, "20%:80%")

        assert len(result.data) == 60

    def test_select_percentage_with_mixed_format(self, sample_df):
        """Test selecting with mixed index and percentage."""
        selector = RowSelector()
        result = selector.select(sample_df, "0:50%")

        assert len(result.data) == 50

    def test_invalid_percentage_range_raises(self, sample_df):
        """Test that invalid percentage range raises error."""
        selector = RowSelector()

        with pytest.raises(RowSelectionError, match="Invalid percentage range"):
            selector.select(sample_df, "80%:20%")


class TestRowSelectorConditions:
    """Tests for condition-based row selection."""

    def test_select_where_equals(self, small_df):
        """Test selecting rows where column equals value."""
        selector = RowSelector()
        result = selector.select(small_df, {
            "where": {"column": "label", "op": "==", "value": "A"}
        })

        assert len(result.data) == 3
        assert all(result.data["label"] == "A")

    def test_select_where_greater_than(self, small_df):
        """Test selecting rows where column greater than value."""
        selector = RowSelector()
        result = selector.select(small_df, {
            "where": {"column": "value", "op": ">", "value": 30}
        })

        assert len(result.data) == 2
        assert all(result.data["value"] > 30)

    def test_select_where_less_equal(self, small_df):
        """Test selecting rows where column less than or equal."""
        selector = RowSelector()
        result = selector.select(small_df, {
            "where": {"column": "value", "op": "<=", "value": 30}
        })

        assert len(result.data) == 3

    def test_select_where_not_equal(self, small_df):
        """Test selecting rows where column not equals value."""
        selector = RowSelector()
        result = selector.select(small_df, {
            "where": {"column": "label", "op": "!=", "value": "A"}
        })

        assert len(result.data) == 2
        assert all(result.data["label"] == "B")

    def test_select_where_in_list(self, small_df):
        """Test selecting rows where column value in list."""
        selector = RowSelector()
        result = selector.select(small_df, {
            "where": {"column": "id", "op": "in", "value": [1, 3, 5]}
        })

        assert len(result.data) == 3

    def test_select_where_contains(self, small_df):
        """Test selecting rows where column contains substring."""
        # Create df with string values
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Carol", "David"],
            "value": [1, 2, 3, 4],
        })

        selector = RowSelector()
        result = selector.select(df, {
            "where": {"column": "name", "op": "contains", "value": "a"}
        })

        assert len(result.data) == 2  # Carol, David

    def test_select_where_and_conditions(self, sample_df):
        """Test selecting with AND conditions."""
        selector = RowSelector()
        result = selector.select(sample_df, {
            "where": [
                {"column": "category", "op": "==", "value": "A"},
                {"column": "label", "op": "==", "value": 0}
            ]
        })

        assert all(result.data["category"] == "A")
        assert all(result.data["label"] == 0)

    def test_select_where_or_conditions(self, small_df):
        """Test selecting with OR conditions."""
        selector = RowSelector()
        result = selector.select(small_df, {
            "where": {"or": [
                {"column": "id", "op": "==", "value": 1},
                {"column": "id", "op": "==", "value": 5}
            ]}
        })

        assert len(result.data) == 2
        assert set(result.data["id"]) == {1, 5}

    def test_select_where_missing_column_raises(self, small_df):
        """Test that missing column in condition raises error."""
        selector = RowSelector()

        with pytest.raises(RowSelectionError, match="not found"):
            selector.select(small_df, {
                "where": {"column": "nonexistent", "op": "==", "value": 1}
            })

    def test_select_where_invalid_operator_raises(self, small_df):
        """Test that invalid operator raises error."""
        selector = RowSelector()

        with pytest.raises(RowSelectionError, match="Unknown operator"):
            selector.select(small_df, {
                "where": {"column": "value", "op": "invalid", "value": 1}
            })


class TestRowSelectorSampling:
    """Tests for sampling-based row selection."""

    def test_select_sample_n(self, sample_df):
        """Test random sampling N rows."""
        selector = RowSelector()
        result = selector.select(sample_df, {
            "sample": 10,
            "random_state": 42
        })

        assert len(result.data) == 10

    def test_select_sample_fraction(self, sample_df):
        """Test random sampling fraction of rows."""
        selector = RowSelector()
        result = selector.select(sample_df, {
            "sample_frac": 0.2,
            "random_state": 42
        })

        assert len(result.data) == 20

    def test_select_sample_reproducible(self, sample_df):
        """Test that sampling is reproducible with random_state."""
        selector = RowSelector()

        result1 = selector.select(sample_df, {"sample": 10, "random_state": 42})
        result2 = selector.select(sample_df, {"sample": 10, "random_state": 42})

        assert result1.indices == result2.indices

    def test_select_sample_with_condition(self, sample_df):
        """Test sampling after applying condition."""
        selector = RowSelector()
        result = selector.select(sample_df, {
            "where": {"column": "category", "op": "==", "value": "A"},
            "sample": 5,
            "random_state": 42
        })

        assert len(result.data) == 5
        assert all(result.data["category"] == "A")

    def test_select_stratified_sample(self, sample_df):
        """Test stratified sampling."""
        selector = RowSelector()
        result = selector.select(sample_df, {
            "sample": 20,
            "stratify": "category",
            "random_state": 42
        })

        assert len(result.data) == 20
        # Check roughly equal representation
        category_counts = result.data["category"].value_counts()
        assert all(count >= 4 for count in category_counts)

    def test_stratified_sample_does_not_mutate_global_numpy_rng(self):
        """Stratified sampling must avoid changing global NumPy RNG state."""
        selector = RowSelector()
        df = pd.DataFrame(
            {
                "category": ["A", "B", "C", "D"] * 25,
                "value": np.arange(100),
            }
        )

        np.random.seed(2026)
        expected_next = np.random.random()

        np.random.seed(2026)
        _ = selector.select(
            df,
            {
                "sample": 20,
                "stratify": "category",
                "random_state": 42,
            },
        )
        observed_next = np.random.random()

        assert observed_next == expected_next

    def test_select_shuffle(self, sample_df):
        """Test shuffling rows."""
        selector = RowSelector()
        result = selector.select(sample_df, {
            "shuffle": True,
            "random_state": 42
        })

        assert len(result.data) == len(sample_df)
        # Indices should be shuffled
        assert result.indices != list(range(len(sample_df)))


class TestRowSelectorHeadTail:
    """Tests for head/tail row selection."""

    def test_select_head(self, sample_df):
        """Test selecting first N rows."""
        selector = RowSelector()
        result = selector.select(sample_df, {"head": 10})

        assert len(result.data) == 10
        assert result.indices == list(range(10))

    def test_select_tail(self, sample_df):
        """Test selecting last N rows."""
        selector = RowSelector()
        result = selector.select(sample_df, {"tail": 10})

        assert len(result.data) == 10
        assert result.indices == list(range(90, 100))

    def test_select_head_with_condition(self, sample_df):
        """Test head combined with condition."""
        selector = RowSelector()
        result = selector.select(sample_df, {
            "where": {"column": "category", "op": "==", "value": "A"},
            "head": 5
        })

        # Should get first 5 rows matching category A
        assert len(result.data) <= 5
        assert all(result.data["category"] == "A")
