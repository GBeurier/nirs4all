"""
Unit tests for Metadata class.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nirs4all.data.metadata import Metadata


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


# ========================================================================
# Tests merged from test_metadata_loading.py
# ========================================================================

import tempfile
from pathlib import Path
from nirs4all.data.config import DatasetConfigs


class TestMetadataLoading:
    """Test loading datasets with metadata."""

    def setup_method(self):
        """Create temporary CSV files for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create X_train data
        x_train = pd.DataFrame(np.random.rand(10, 5))
        x_train.to_csv(self.temp_path / "X_train.csv", index=False, sep=';')

        # Create Y_train data
        y_train = pd.DataFrame({'target': np.random.rand(10)})
        y_train.to_csv(self.temp_path / "Y_train.csv", index=False, sep=';')

        # Create metadata (M_train)
        m_train = pd.DataFrame({
            'batch': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
            'location': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B', 'C', 'C'],
            'instrument': ['X1', 'X1', 'X1', 'X1', 'X2', 'X2', 'X2', 'X2', 'X1', 'X1']
        })
        m_train.to_csv(self.temp_path / "M_train.csv", index=False, sep=';')

        # Create X_test data
        x_test = pd.DataFrame(np.random.rand(5, 5))
        x_test.to_csv(self.temp_path / "X_test.csv", index=False, sep=';')

        # Create Y_test data
        y_test = pd.DataFrame({'target': np.random.rand(5)})
        y_test.to_csv(self.temp_path / "Y_test.csv", index=False, sep=';')

        # Create metadata (M_test)
        m_test = pd.DataFrame({
            'batch': [3, 3, 4, 4, 4],
            'location': ['C', 'C', 'D', 'D', 'D'],
            'instrument': ['X1', 'X1', 'X2', 'X2', 'X2']
        })
        m_test.to_csv(self.temp_path / "M_test.csv", index=False, sep=';')

    def test_load_dataset_with_metadata_from_folder(self):
        """Test loading dataset with metadata from folder."""
        configs = DatasetConfigs(str(self.temp_path))
        dataset = configs.get_dataset_at(0)

        # Check basic properties
        assert dataset.num_samples == 15  # 10 train + 5 test
        assert dataset.num_features == 5

        # Check metadata was loaded
        assert dataset._metadata.num_rows == 15
        assert set(dataset.metadata_columns) == {'batch', 'location', 'instrument'}

        # Test accessing train metadata
        train_meta = dataset.metadata(selector={"partition": "train"})
        assert len(train_meta) == 10
        assert list(train_meta['batch']) == [1, 1, 1, 1, 2, 2, 2, 2, 3, 3]

        # Test accessing test metadata
        test_meta = dataset.metadata(selector={"partition": "test"})
        assert len(test_meta) == 5
        assert list(test_meta['batch']) == [3, 3, 4, 4, 4]

    def test_load_dataset_with_metadata_from_dict_config(self):
        """Test loading dataset with metadata from dict config."""
        config = {
            'train_x': str(self.temp_path / "X_train.csv"),
            'train_y': str(self.temp_path / "Y_train.csv"),
            'train_group': str(self.temp_path / "M_train.csv"),
            'test_x': str(self.temp_path / "X_test.csv"),
            'test_y': str(self.temp_path / "Y_test.csv"),
            'test_group': str(self.temp_path / "M_test.csv"),
        }

        configs = DatasetConfigs(config)
        dataset = configs.get_dataset_at(0)

        # Check metadata was loaded
        assert dataset._metadata.num_rows == 15
        assert 'batch' in dataset.metadata_columns

    def test_metadata_column_access(self):
        """Test accessing individual metadata columns."""
        configs = DatasetConfigs(str(self.temp_path))
        dataset = configs.get_dataset_at(0)

        # Get train batch column
        train_batch = dataset.metadata_column('batch', selector={"partition": "train"})
        assert len(train_batch) == 10
        assert list(train_batch) == [1, 1, 1, 1, 2, 2, 2, 2, 3, 3]

        # Get all instrument data
        instruments = dataset.metadata_column('instrument')
        assert len(instruments) == 15

    def test_metadata_numeric_encoding(self):
        """Test converting metadata to numeric format."""
        configs = DatasetConfigs(str(self.temp_path))
        dataset = configs.get_dataset_at(0)

        # Label encoding
        location_encoded, info = dataset.metadata_numeric('location', method='label')
        assert len(location_encoded) == 15
        assert info['method'] == 'label'
        assert set(info['classes']) == {'A', 'B', 'C', 'D'}

        # One-hot encoding
        location_onehot, info = dataset.metadata_numeric('location', method='onehot')
        assert location_onehot.shape == (15, 4)  # 15 samples, 4 locations
        assert info['method'] == 'onehot'

    def test_dataset_without_metadata(self):
        """Test that datasets without metadata still work."""
        # Create a simple config without metadata
        temp_dir2 = tempfile.mkdtemp()
        temp_path2 = Path(temp_dir2)

        x_train = pd.DataFrame(np.random.rand(5, 3))
        x_train.to_csv(temp_path2 / "X_train.csv", index=False, sep=';')
        y_train = pd.DataFrame({'target': [1, 2, 3, 4, 5]})
        y_train.to_csv(temp_path2 / "Y_train.csv", index=False, sep=';')

        configs = DatasetConfigs(str(temp_path2))
        dataset = configs.get_dataset_at(0)

        # Should work without metadata
        assert dataset.num_samples == 5
        assert dataset._metadata.num_rows == 0
        assert dataset.metadata_columns == []

    def test_metadata_with_alternative_naming(self):
        """Test that alternative metadata file names are detected."""
        # Create files with alternative names
        temp_dir3 = tempfile.mkdtemp()
        temp_path3 = Path(temp_dir3)

        x_train = pd.DataFrame(np.random.rand(5, 3))
        x_train.to_csv(temp_path3 / "X_train.csv", index=False, sep=';')

        # Use "Metacal" naming
        m_train = pd.DataFrame({'sample_id': [1, 2, 3, 4, 5]})
        m_train.to_csv(temp_path3 / "Metacal.csv", index=False, sep=';')

        configs = DatasetConfigs(str(temp_path3))
        dataset = configs.get_dataset_at(0)

        # Should detect and load metadata
        assert dataset._metadata.num_rows == 5
        assert 'sample_id' in dataset.metadata_columns

    def test_metadata_update_operations(self):
        """Test updating metadata after loading."""
        configs = DatasetConfigs(str(self.temp_path))
        dataset = configs.get_dataset_at(0)

        # Get the indices we want to update (first 2 train samples)
        train_indices = dataset._indexer.x_indices({"partition": "train"})[:2]

        # Update first two train samples
        dataset._metadata.update_metadata(train_indices, 'location', ['Z', 'Z'])

        # Check first two train samples were updated
        train_meta = dataset.metadata(selector={"partition": "train"})
        assert train_meta['location'][0] == 'Z'
        assert train_meta['location'][1] == 'Z'

    def test_add_metadata_column(self):
        """Test adding new metadata column."""
        configs = DatasetConfigs(str(self.temp_path))
        dataset = configs.get_dataset_at(0)

        # Add a new column
        quality_scores = np.random.rand(15)
        dataset.add_metadata_column('quality', quality_scores)

        assert 'quality' in dataset.metadata_columns
        quality_values = dataset.metadata_column('quality')
        assert len(quality_values) == 15


class TestMetadataConfigNormalization:
    """Test config key normalization for metadata."""

    def test_normalize_metadata_keys(self):
        """Test that various metadata key formats are normalized."""
        from nirs4all.data.config_parser import normalize_config_keys

        # Test various formats
        configs_to_test = [
            {'train_metadata': 'path/to/meta.csv'},
            {'metadata_train': 'path/to/meta.csv'},
            {'train_meta': 'path/to/meta.csv'},
            {'meta_train': 'path/to/meta.csv'},
            {'train_m': 'path/to/meta.csv'},
            {'m_train': 'path/to/meta.csv'},
        ]

        for config in configs_to_test:
            normalized = normalize_config_keys(config)
            assert 'train_group' in normalized
            assert normalized['train_group'] == 'path/to/meta.csv'
