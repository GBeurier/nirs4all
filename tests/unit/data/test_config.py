"""Tests for dataset configuration."""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from nirs4all.data.config import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset


class TestDatasetConfig:
    """Test suite for DatasetConfigs."""

    def test_placeholder(self):
        """Placeholder test."""
        # TODO: Add comprehensive DatasetConfigs tests
        pass


class TestDatasetConfigAggregate:
    """Test suite for aggregate parameter in DatasetConfigs."""

    @pytest.fixture
    def sample_data_files(self):
        """Create temporary CSV files for testing."""
        # Create train_x
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as fx:
            fx.write("1000;2000;3000\n")
            fx.write("0.1;0.2;0.3\n")
            fx.write("0.4;0.5;0.6\n")
            fx.write("0.7;0.8;0.9\n")
            x_path = fx.name

        # Create train_y
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as fy:
            fy.write("target\n")
            fy.write("10.5\n")
            fy.write("20.3\n")
            fy.write("15.7\n")
            y_path = fy.name

        # Create train_m (metadata with sample_id)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as fm:
            fm.write("sample_id;batch\n")
            fm.write("S001;A\n")
            fm.write("S001;A\n")
            fm.write("S002;B\n")
            m_path = fm.name

        yield {'x': x_path, 'y': y_path, 'm': m_path}

        # Cleanup
        Path(x_path).unlink()
        Path(y_path).unlink()
        Path(m_path).unlink()

    def test_aggregate_none_by_default(self, sample_data_files):
        """Test that aggregate is None by default."""
        config = {
            'train_x': sample_data_files['x'],
            'train_y': sample_data_files['y'],
            'global_params': {'delimiter': ';', 'has_header': True}
        }

        dataset_config = DatasetConfigs(config)
        dataset = dataset_config.get_dataset_at(0)

        assert dataset.aggregate is None

    def test_aggregate_via_constructor_string(self, sample_data_files):
        """Test setting aggregate via constructor with column name."""
        config = {
            'train_x': sample_data_files['x'],
            'train_y': sample_data_files['y'],
            'global_params': {'delimiter': ';', 'has_header': True}
        }

        dataset_config = DatasetConfigs(config, aggregate='sample_id')
        dataset = dataset_config.get_dataset_at(0)

        assert dataset.aggregate == 'sample_id'

    def test_aggregate_via_constructor_true(self, sample_data_files):
        """Test setting aggregate via constructor with True (aggregate by y)."""
        config = {
            'train_x': sample_data_files['x'],
            'train_y': sample_data_files['y'],
            'global_params': {'delimiter': ';', 'has_header': True}
        }

        dataset_config = DatasetConfigs(config, aggregate=True)
        dataset = dataset_config.get_dataset_at(0)

        assert dataset.aggregate == 'y'

    def test_aggregate_via_config_dict(self, sample_data_files):
        """Test setting aggregate via config dict."""
        config = {
            'train_x': sample_data_files['x'],
            'train_y': sample_data_files['y'],
            'aggregate': 'sample_id',
            'global_params': {'delimiter': ';', 'has_header': True}
        }

        dataset_config = DatasetConfigs(config)
        dataset = dataset_config.get_dataset_at(0)

        assert dataset.aggregate == 'sample_id'

    def test_aggregate_via_config_dict_true(self, sample_data_files):
        """Test setting aggregate=True via config dict."""
        config = {
            'train_x': sample_data_files['x'],
            'train_y': sample_data_files['y'],
            'aggregate': True,
            'global_params': {'delimiter': ';', 'has_header': True}
        }

        dataset_config = DatasetConfigs(config)
        dataset = dataset_config.get_dataset_at(0)

        assert dataset.aggregate == 'y'

    def test_aggregate_constructor_overrides_config(self, sample_data_files):
        """Test that constructor parameter overrides config dict value."""
        config = {
            'train_x': sample_data_files['x'],
            'train_y': sample_data_files['y'],
            'aggregate': 'sample_id',  # Config says sample_id
            'global_params': {'delimiter': ';', 'has_header': True}
        }

        # Constructor parameter overrides to batch
        dataset_config = DatasetConfigs(config, aggregate='batch')
        dataset = dataset_config.get_dataset_at(0)

        assert dataset.aggregate == 'batch'

    def test_aggregate_constructor_true_overrides_config_string(self, sample_data_files):
        """Test that constructor True overrides config dict string."""
        config = {
            'train_x': sample_data_files['x'],
            'train_y': sample_data_files['y'],
            'aggregate': 'sample_id',  # Config says sample_id
            'global_params': {'delimiter': ';', 'has_header': True}
        }

        # Constructor parameter overrides to True (y-based)
        dataset_config = DatasetConfigs(config, aggregate=True)
        dataset = dataset_config.get_dataset_at(0)

        assert dataset.aggregate == 'y'

    def test_aggregate_per_dataset_list(self, sample_data_files):
        """Test per-dataset aggregate settings with list."""
        config1 = {
            'train_x': sample_data_files['x'],
            'train_y': sample_data_files['y'],
            'global_params': {'delimiter': ';', 'has_header': True}
        }
        config2 = {
            'train_x': sample_data_files['x'],
            'train_y': sample_data_files['y'],
            'global_params': {'delimiter': ';', 'has_header': True}
        }

        # Different aggregate for each dataset
        dataset_config = DatasetConfigs([config1, config2], aggregate=['sample_id', 'batch'])

        dataset1 = dataset_config.get_dataset_at(0)
        dataset2 = dataset_config.get_dataset_at(1)

        assert dataset1.aggregate == 'sample_id'
        assert dataset2.aggregate == 'batch'

    def test_aggregate_list_length_mismatch_raises(self, sample_data_files):
        """Test that aggregate list length mismatch raises ValueError."""
        config = {
            'train_x': sample_data_files['x'],
            'train_y': sample_data_files['y'],
            'global_params': {'delimiter': ';', 'has_header': True}
        }

        with pytest.raises(ValueError, match="aggregate list length"):
            DatasetConfigs(config, aggregate=['sample_id', 'batch'])

    def test_aggregate_iter_datasets(self, sample_data_files):
        """Test that aggregate is applied when using iter_datasets()."""
        config = {
            'train_x': sample_data_files['x'],
            'train_y': sample_data_files['y'],
            'aggregate': 'sample_id',
            'global_params': {'delimiter': ';', 'has_header': True}
        }

        dataset_config = DatasetConfigs(config)

        for dataset in dataset_config.iter_datasets():
            assert dataset.aggregate == 'sample_id'

    def test_aggregate_get_datasets(self, sample_data_files):
        """Test that aggregate is applied when using get_datasets()."""
        config = {
            'train_x': sample_data_files['x'],
            'train_y': sample_data_files['y'],
            'aggregate': 'sample_id',
            'global_params': {'delimiter': ';', 'has_header': True}
        }

        dataset_config = DatasetConfigs(config)
        datasets = dataset_config.get_datasets()

        assert len(datasets) == 1
        assert datasets[0].aggregate == 'sample_id'


class TestSpectroDatasetAggregate:
    """Test suite for aggregate property in SpectroDataset."""

    def test_aggregate_default_none(self):
        """Test that aggregate is None by default."""
        dataset = SpectroDataset("test")
        assert dataset.aggregate is None

    def test_set_aggregate_string(self):
        """Test setting aggregate with column name."""
        dataset = SpectroDataset("test")
        dataset.set_aggregate('sample_id')

        assert dataset.aggregate == 'sample_id'

    def test_set_aggregate_true(self):
        """Test setting aggregate with True (y-based)."""
        dataset = SpectroDataset("test")
        dataset.set_aggregate(True)

        assert dataset.aggregate == 'y'

    def test_set_aggregate_none_clears(self):
        """Test that setting aggregate to None clears previous setting."""
        dataset = SpectroDataset("test")
        dataset.set_aggregate('sample_id')
        assert dataset.aggregate == 'sample_id'

        dataset.set_aggregate(None)
        assert dataset.aggregate is None

    def test_set_aggregate_switches_modes(self):
        """Test switching between string and True aggregate modes."""
        dataset = SpectroDataset("test")

        # Start with column-based
        dataset.set_aggregate('sample_id')
        assert dataset.aggregate == 'sample_id'
        assert dataset._aggregate_by_y is False
        assert dataset._aggregate_column == 'sample_id'

        # Switch to y-based
        dataset.set_aggregate(True)
        assert dataset.aggregate == 'y'
        assert dataset._aggregate_by_y is True
        assert dataset._aggregate_column is None

        # Switch back to column-based
        dataset.set_aggregate('batch')
        assert dataset.aggregate == 'batch'
        assert dataset._aggregate_by_y is False
        assert dataset._aggregate_column == 'batch'
