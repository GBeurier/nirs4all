"""
Integration tests for metadata loading through DatasetConfigs.
"""

import tempfile
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from nirs4all.dataset.dataset_config import DatasetConfigs


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
        from nirs4all.dataset.dataset_config_parser import normalize_config_keys

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
