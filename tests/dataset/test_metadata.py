"""Tests for the MetadataBlock class."""
import polars as pl
import pytest
from nirs4all.dataset.metadata import MetadataBlock


class TestMetadataBlockBasics:
    """Test basic MetadataBlock functionality."""

    def test_empty_metadata_block(self):
        """Test empty MetadataBlock state and representation."""
        mb = MetadataBlock()

        assert hasattr(mb, 'table'), "MetadataBlock should have table attribute"
        assert 'MetadataBlock' in repr(mb), "Repr should include class name"

    def test_add_valid_metadata(self):
        """Test adding valid metadata with required sample column."""
        mb = MetadataBlock()
        df = pl.DataFrame({
            'sample': [0, 1, 2],
            'instrument': ['A', 'B', 'A'],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03']
        })

        mb.add_meta(df)

        assert mb.table is not None, "Table should be set after adding metadata"
        assert mb.table.height == 3, "Table should have 3 rows"
        assert 'rows=3' in repr(mb), "Repr should show row count"
        assert set(mb.table.columns) == {'sample', 'instrument', 'date'}, "All columns should be preserved"

    def test_add_metadata_preserves_dtypes(self):
        """Test that metadata dtypes are preserved."""
        mb = MetadataBlock()
        df = pl.DataFrame({
            'sample': [0, 1, 2],
            'int_col': [10, 20, 30],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['x', 'y', 'z'],
            'bool_col': [True, False, True]
        })

        mb.add_meta(df)

        result_df = mb.table
        assert result_df['int_col'].dtype == pl.Int64, "Integer column dtype should be preserved"
        assert result_df['float_col'].dtype == pl.Float64, "Float column dtype should be preserved"
        assert result_df['str_col'].dtype == pl.Utf8, "String column dtype should be preserved"
        assert result_df['bool_col'].dtype == pl.Boolean, "Boolean column dtype should be preserved"

    def test_add_metadata_multiple_calls(self):
        """Test behavior when adding metadata multiple times."""
        mb = MetadataBlock()

        # First add
        df1 = pl.DataFrame({
            'sample': [0, 1],
            'group': ['A', 'B']
        })
        mb.add_meta(df1)
        assert mb.table.height == 2

        # Second add - should overwrite or extend based on implementation
        df2 = pl.DataFrame({
            'sample': [2, 3],
            'group': ['C', 'D'],
            'extra': [10, 20]
        })
        mb.add_meta(df2)

        # Check final state
        final_table = mb.table
        assert final_table is not None, "Table should exist after second add"
        # Exact behavior depends on implementation - just ensure it's reasonable
        assert final_table.height > 0, "Table should have rows after adding metadata"


class TestMetadataBlockFiltering:
    """Test MetadataBlock filtering functionality."""

    @pytest.fixture
    def populated_metadata_block(self):
        """Create a MetadataBlock with test data."""
        mb = MetadataBlock()
        df = pl.DataFrame({
            'sample': [0, 1, 2, 3, 4, 5],
            'group': ['A', 'B', 'A', 'C', 'B', 'A'],
            'value': [10, 20, 30, 40, 50, 60],
            'active': [True, False, True, True, False, True]
        })
        mb.add_meta(df)
        return mb

    def test_filter_by_single_value(self, populated_metadata_block):
        """Test filtering by a single column value."""
        mb = populated_metadata_block

        filtered = mb.meta({'group': 'A'})

        assert isinstance(filtered, pl.DataFrame), "Filtered result should be DataFrame"
        assert filtered.height == 3, "Should match 3 samples with group A"
        assert all(filtered['group'] == 'A'), "All rows should have group A"
        expected_samples = {0, 2, 5}
        actual_samples = set(filtered['sample'].to_list())
        assert actual_samples == expected_samples, f"Should include samples {expected_samples}"

    def test_filter_by_multiple_values(self, populated_metadata_block):
        """Test filtering by multiple column values."""
        mb = populated_metadata_block

        filtered = mb.meta({'group': 'A', 'active': True})

        assert filtered.height == 3, "Should match 3 samples with group A and active True"
        assert all(filtered['group'] == 'A'), "All rows should have group A"
        assert all(filtered['active']), "All rows should be active"
        expected_samples = {0, 2, 5}
        actual_samples = set(filtered['sample'].to_list())
        assert actual_samples == expected_samples, f"Should include samples {expected_samples}"

    def test_filter_by_list_values(self, populated_metadata_block):
        """Test filtering by list of acceptable values."""
        mb = populated_metadata_block

        filtered = mb.meta({'group': ['A', 'C']})

        assert filtered.height == 4, "Should match samples with group A or C"
        group_values = set(filtered['group'].to_list())
        assert group_values == {'A', 'C'}, "Should only include groups A and C"

    def test_filter_by_sample_list(self, populated_metadata_block):
        """Test filtering by specific sample indices."""
        mb = populated_metadata_block

        filtered = mb.meta({'sample': [1, 3, 5]})

        assert filtered.height == 3, "Should match 3 specific samples"
        expected_samples = {1, 3, 5}
        actual_samples = set(filtered['sample'].to_list())
        assert actual_samples == expected_samples, f"Should include exactly samples {expected_samples}"

    def test_filter_no_matches(self, populated_metadata_block):
        """Test filtering with criteria that matches no rows."""
        mb = populated_metadata_block

        filtered = mb.meta({'group': 'nonexistent'})

        assert isinstance(filtered, pl.DataFrame), "Result should still be DataFrame"
        assert filtered.height == 0, "Should have no matching rows"
        assert set(filtered.columns) == set(mb.table.columns), "Should preserve column structure"

    def test_filter_all_rows(self, populated_metadata_block):
        """Test filtering with criteria that matches all rows."""
        mb = populated_metadata_block        # Filter by column that exists in all rows
        all_samples = list(range(6))
        filtered = mb.meta({'sample': all_samples})

        assert filtered.height == 6, "Should match all rows"
        assert filtered.equals(mb.table), "Should be identical to original table"

    def test_filter_complex_criteria(self, populated_metadata_block):
        """Test filtering with complex criteria combinations."""
        mb = populated_metadata_block

        # Multiple criteria with lists
        filtered = mb.meta({
            'group': ['A', 'B'],
            'value': [10, 20, 60]
        })

        # Should match samples where group is A or B AND value is 10, 20, or 60
        assert filtered.height > 0, "Should have some matches"

        for row in filtered.iter_rows(named=True):
            assert row['group'] in ['A', 'B'], "Group should be A or B"
            assert row['value'] in [10, 20, 60], "Value should be in specified list"


class TestMetadataBlockErrorHandling:
    """Test MetadataBlock error handling."""

    def test_add_metadata_missing_sample_column(self):
        """Test error when sample column is missing."""
        mb = MetadataBlock()
        df = pl.DataFrame({
            'id': [0, 1, 2],  # Wrong column name
            'value': [10, 20, 30]
        })

        with pytest.raises(ValueError, match=".*sample.*"):
            mb.add_meta(df)

    def test_add_metadata_sample_column_different_case(self):
        """Test error with sample column in different case."""
        mb = MetadataBlock()
        df = pl.DataFrame({
            'Sample': [0, 1, 2],  # Wrong case
            'value': [10, 20, 30]
        })

        with pytest.raises(ValueError, match=".*sample.*"):
            mb.add_meta(df)

    def test_add_metadata_empty_dataframe(self):
        """Test adding empty DataFrame."""
        mb = MetadataBlock()
        df = pl.DataFrame({'sample': []})  # Empty but with correct column

        # Should handle empty DataFrame gracefully
        try:
            mb.add_meta(df)
            assert mb.table is not None, "Table should be set even for empty DataFrame"
            assert mb.table.height == 0, "Table should have no rows"
        except ValueError:
            # Empty DataFrames might not be allowed, which is acceptable
            pass

    def test_filter_without_table(self):
        """Test filtering when no metadata has been added."""
        mb = MetadataBlock()

        with pytest.raises(ValueError, match=".*metadata.*"):
            mb.meta({'sample': 0})

    def test_filter_with_invalid_column(self):
        """Test filtering with non-existent column."""
        mb = MetadataBlock()
        df = pl.DataFrame({
            'sample': [0, 1, 2],
            'group': ['A', 'B', 'C']
        })
        mb.add_meta(df)

        # Filter by non-existent column should return empty or raise error
        try:
            filtered = mb.meta({'nonexistent_column': 'value'})
            # If it doesn't raise error, should return empty DataFrame
            assert filtered.height == 0, "Non-existent column filter should return no results"
        except (KeyError, ValueError):
            # Raising error for non-existent column is also acceptable
            pass

    def test_add_metadata_with_none_values(self):
        """Test adding metadata with None/null values."""
        mb = MetadataBlock()
        df = pl.DataFrame({
            'sample': [0, 1, 2],
            'nullable_col': [10, None, 30],
            'str_col': ['a', None, 'c']
        })

        mb.add_meta(df)

        result_df = mb.table
        assert result_df.height == 3, "Should preserve all rows including those with nulls"

        # Check null handling
        nullable_values = result_df['nullable_col'].to_list()
        assert nullable_values[1] is None, "Null values should be preserved"

    def test_filter_with_none_values(self):
        """Test filtering when metadata contains None values."""
        mb = MetadataBlock()
        df = pl.DataFrame({
            'sample': [0, 1, 2, 3],
            'group': ['A', None, 'B', 'A'],
            'value': [10, 20, None, 40]
        })
        mb.add_meta(df)

        # Filter excluding None values
        filtered = mb.meta({'group': 'A'})
        assert filtered.height == 2, "Should match 2 samples with group A"

        # Check behavior with None in filter criteria
        try:
            filtered_none = mb.meta({'group': None})
            # If supported, should match the None value
            if filtered_none.height > 0:
                assert all(v is None for v in filtered_none['group'].to_list()), "Should match None values"
        except (TypeError, ValueError):
            # Some implementations might not support None in filter criteria
            pass


class TestMetadataBlockIntegration:
    """Test MetadataBlock integration scenarios."""

    def test_large_metadata_dataset(self):
        """Test with large metadata dataset."""
        mb = MetadataBlock()

        # Create large dataset
        n_samples = 10000
        df = pl.DataFrame({
            'sample': list(range(n_samples)),
            'group': [f'group_{i % 100}' for i in range(n_samples)],
            'batch': [i // 1000 for i in range(n_samples)],
            'value': [i * 0.1 for i in range(n_samples)]
        })

        mb.add_meta(df)

        # Test filtering on large dataset
        filtered = mb.meta({'group': 'group_0'})
        assert filtered.height == 100, "Should find 100 samples for group_0"

        # Test batch filtering
        batch_filtered = mb.meta({'batch': 5})
        assert batch_filtered.height == 1000, "Should find 1000 samples for batch 5"

    def test_metadata_with_complex_dtypes(self):
        """Test metadata with various data types."""
        mb = MetadataBlock()

        df = pl.DataFrame({
            'sample': [0, 1, 2],
            'int8_col': pl.Series([1, 2, 3], dtype=pl.Int8),
            'int32_col': pl.Series([100, 200, 300], dtype=pl.Int32),
            'float32_col': pl.Series([1.1, 2.2, 3.3], dtype=pl.Float32),
            'date_col': pl.Series(['2023-01-01', '2023-01-02', '2023-01-03']).str.strptime(pl.Date),
            'categorical_col': pl.Series(['A', 'B', 'A']).cast(pl.Categorical)
        })

        mb.add_meta(df)

        # Verify dtypes are preserved
        result_df = mb.table
        assert result_df['int8_col'].dtype == pl.Int8, "Int8 dtype should be preserved"
        assert result_df['int32_col'].dtype == pl.Int32, "Int32 dtype should be preserved"
        assert result_df['float32_col'].dtype == pl.Float32, "Float32 dtype should be preserved"
        assert result_df['date_col'].dtype == pl.Date, "Date dtype should be preserved"
        assert result_df['categorical_col'].dtype == pl.Categorical, "Categorical dtype should be preserved"

        # Test filtering with different dtypes
        filtered = mb.meta({'categorical_col': 'A'})
        assert filtered.height == 2, "Should filter categorical columns correctly"
