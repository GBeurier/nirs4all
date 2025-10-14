"""
Unit tests for Metadata class.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nirs4all.dataset.metadata import Metadata


class TestMetadataBasics:
    """Test basic metadata operations."""

    def test_empty_metadata(self):
        """Test empty metadata initialization."""
        meta = Metadata()
        assert meta.df is None
        assert meta.num_rows == 0
        assert meta.columns == []

    def test_add_metadata_numpy(self):
        """Test adding metadata from numpy array."""
        meta = Metadata()
        data = np.array([[1, 'A'], [2, 'B'], [3, 'C']], dtype=object)
        headers = ['id', 'label']

        meta.add_metadata(data, headers)

        assert meta.num_rows == 3
        assert meta.columns == ['id', 'label']

    def test_add_metadata_dataframe(self):
        """Test adding metadata from pandas DataFrame."""
        meta = Metadata()
        df = pd.DataFrame({
            'batch': [1, 1, 2, 2],
            'location': ['A', 'A', 'B', 'B']
        })

        meta.add_metadata(df)

        assert meta.num_rows == 4
        assert set(meta.columns) == {'batch', 'location'}

    def test_add_metadata_incrementally(self):
        """Test adding metadata in multiple calls."""
        meta = Metadata()

        # First batch
        data1 = np.array([[1, 'A'], [2, 'B']], dtype=object)
        meta.add_metadata(data1, ['id', 'label'])
        assert meta.num_rows == 2

        # Second batch
        data2 = np.array([[3, 'C'], [4, 'D']], dtype=object)
        meta.add_metadata(data2, ['id', 'label'])
        assert meta.num_rows == 4

    def test_add_metadata_with_none(self):
        """Test that None data is handled gracefully."""
        meta = Metadata()
        meta.add_metadata(None)
        assert meta.num_rows == 0

    def test_add_metadata_with_empty_array(self):
        """Test that empty array is handled gracefully."""
        meta = Metadata()
        data = np.array([]).reshape(0, 2)
        meta.add_metadata(data, ['col1', 'col2'])
        assert meta.num_rows == 0


class TestMetadataRetrieval:
    """Test metadata retrieval operations."""

    def setup_method(self):
        """Setup metadata for testing."""
        self.meta = Metadata()
        data = pd.DataFrame({
            'batch': [1, 1, 2, 2, 3],
            'location': ['A', 'A', 'B', 'B', 'C'],
            'instrument': ['X', 'X', 'Y', 'Y', 'X']
        })
        self.meta.add_metadata(data)

    def test_get_all_metadata(self):
        """Test getting all metadata without filters."""
        result = self.meta.get()
        assert len(result) == 5
        assert set(result.columns) == {'batch', 'location', 'instrument'}

    def test_get_metadata_with_indices(self):
        """Test getting metadata with specific indices."""
        result = self.meta.get(indices=[0, 2, 4])
        assert len(result) == 3

    def test_get_metadata_with_columns(self):
        """Test getting specific columns."""
        result = self.meta.get(columns=['batch', 'location'])
        assert set(result.columns) == {'batch', 'location'}

    def test_get_column(self):
        """Test getting single column."""
        batch = self.meta.get_column('batch')
        assert len(batch) == 5
        assert list(batch) == [1, 1, 2, 2, 3]

    def test_get_column_with_indices(self):
        """Test getting column with specific indices."""
        batch = self.meta.get_column('batch', indices=[0, 2, 4])
        assert len(batch) == 3
        assert list(batch) == [1, 2, 3]

    def test_get_column_nonexistent(self):
        """Test error when getting nonexistent column."""
        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            self.meta.get_column('nonexistent')


class TestMetadataNumericConversion:
    """Test numeric conversion of metadata."""

    def setup_method(self):
        """Setup metadata with mixed types."""
        self.meta = Metadata()
        data = pd.DataFrame({
            'numeric_col': [1.5, 2.5, 3.5, 4.5],
            'categorical_col': ['A', 'B', 'A', 'C'],
            'int_col': [1, 2, 3, 4]
        })
        self.meta.add_metadata(data)

    def test_to_numeric_already_numeric(self):
        """Test that numeric columns pass through."""
        numeric, info = self.meta.to_numeric('numeric_col')
        assert len(numeric) == 4
        assert info['method'] == 'numeric'
        assert np.allclose(numeric, [1.5, 2.5, 3.5, 4.5])

    def test_to_numeric_label_encoding(self):
        """Test label encoding of categorical column."""
        numeric, info = self.meta.to_numeric('categorical_col', method='label')
        assert len(numeric) == 4
        assert info['method'] == 'label'
        assert 'classes' in info
        assert set(info['classes']) == {'A', 'B', 'C'}

    def test_to_numeric_onehot_encoding(self):
        """Test one-hot encoding of categorical column."""
        numeric, info = self.meta.to_numeric('categorical_col', method='onehot')
        assert numeric.shape == (4, 3)  # 4 samples, 3 classes
        assert info['method'] == 'onehot'
        assert set(info['classes']) == {'A', 'B', 'C'}
        # Check that rows sum to 1
        assert np.allclose(numeric.sum(axis=1), 1.0)

    def test_to_numeric_with_indices(self):
        """Test numeric conversion with specific indices."""
        numeric, info = self.meta.to_numeric('categorical_col', indices=[0, 2], method='label')
        assert len(numeric) == 2

    def test_to_numeric_caching(self):
        """Test that encodings are cached for consistency."""
        # First call
        numeric1, info1 = self.meta.to_numeric('categorical_col', method='label')
        # Second call
        numeric2, info2 = self.meta.to_numeric('categorical_col', method='label')

        # Should be identical
        assert np.array_equal(numeric1, numeric2)
        assert info1 == info2

    def test_to_numeric_invalid_method(self):
        """Test error with invalid method."""
        with pytest.raises(ValueError, match="Unknown method"):
            self.meta.to_numeric('categorical_col', method='invalid')


class TestMetadataModification:
    """Test metadata modification operations."""

    def setup_method(self):
        """Setup metadata for testing."""
        self.meta = Metadata()
        data = pd.DataFrame({
            'batch': [1, 1, 2, 2],
            'status': ['pending', 'pending', 'done', 'done']
        })
        self.meta.add_metadata(data)

    def test_update_metadata(self):
        """Test updating metadata values."""
        self.meta.update_metadata([0, 1], 'status', ['processed', 'processed'])
        status = self.meta.get_column('status')
        assert list(status[:2]) == ['processed', 'processed']
        assert list(status[2:]) == ['done', 'done']

    def test_update_metadata_length_mismatch(self):
        """Test error when indices and values length mismatch."""
        with pytest.raises(ValueError, match="Length mismatch"):
            self.meta.update_metadata([0, 1], 'status', ['processed'])

    def test_add_column(self):
        """Test adding new column."""
        self.meta.add_column('instrument', ['A', 'A', 'B', 'B'])
        assert 'instrument' in self.meta.columns
        assert len(self.meta.columns) == 3

    def test_add_column_wrong_length(self):
        """Test error when adding column with wrong length."""
        with pytest.raises(ValueError, match="Values length"):
            self.meta.add_column('new_col', [1, 2])  # Only 2 values for 4 rows

    def test_add_column_duplicate(self):
        """Test error when adding duplicate column."""
        with pytest.raises(ValueError, match="already exists"):
            self.meta.add_column('batch', [1, 2, 3, 4])


class TestMetadataEdgeCases:
    """Test edge cases and error handling."""

    def test_get_from_empty_metadata(self):
        """Test getting from empty metadata."""
        meta = Metadata()
        result = meta.get()
        assert len(result) == 0

    def test_get_column_from_empty_metadata(self):
        """Test error when getting column from empty metadata."""
        meta = Metadata()
        with pytest.raises(ValueError, match="No metadata available"):
            meta.get_column('col')

    def test_to_numeric_from_empty_metadata(self):
        """Test error when converting from empty metadata."""
        meta = Metadata()
        with pytest.raises(ValueError, match="No metadata available"):
            meta.to_numeric('col')

    def test_update_empty_metadata(self):
        """Test error when updating empty metadata."""
        meta = Metadata()
        with pytest.raises(ValueError, match="No metadata available"):
            meta.update_metadata([0], 'col', ['value'])

    def test_add_column_to_empty_metadata(self):
        """Test error when adding column to empty metadata."""
        meta = Metadata()
        with pytest.raises(ValueError, match="No metadata available"):
            meta.add_column('col', [1, 2, 3])

    def test_repr(self):
        """Test string representation."""
        meta = Metadata()
        assert repr(meta) == "Metadata(empty)"

        data = pd.DataFrame({'col1': [1, 2], 'col2': ['A', 'B']})
        meta.add_metadata(data)
        assert "rows=2" in repr(meta)
        assert "col1" in repr(meta)
