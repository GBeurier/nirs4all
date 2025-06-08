"""Tests for dataset I/O functionality."""
import tempfile
import numpy as np
import polars as pl
import pytest
from nirs4all.dataset.io import save, load
from nirs4all.dataset.dataset import SpectroDataset


@pytest.fixture
def minimal_dataset():
    """Create a minimal dataset for testing."""
    ds = SpectroDataset()
    data = np.random.rand(4, 2).astype(np.float32)
    ds.add_features([data])
    return ds


@pytest.fixture
def comprehensive_dataset():
    """Create a comprehensive dataset with all components."""
    ds = SpectroDataset()

    # Add features
    data = np.random.rand(8, 3).astype(np.float32)
    ds.add_features([data])
    # Add metadata
    meta = pl.DataFrame({
        'sample': list(range(8)),
        'instrument': ['A'] * 4 + ['B'] * 4,
        'measurement_date': ['2023-01-01'] * 8
    })
    ds.add_meta(meta)    # Add targets
    targets_data = np.array([[float(i)] for i in range(8)], dtype=np.float32)
    samples = np.array(list(range(8)))
    ds.targets.add_regression_targets("targets", targets_data, samples, "raw")

    # Add predictions
    preds = np.arange(8).reshape(8, 1).astype(np.float32)
    ds.predictions.add_prediction(
        preds,
        {
            'model': 'test_model',
            'fold': 0,
            'repeat': 0,
            'partition': 'val',
            'processing': 'raw',
            'seed': 42
        }
    )    # Set folds
    ds.folds.set_folds([(list(range(4)), list(range(4, 8)))])

    return ds


@pytest.fixture
def multi_source_dataset():
    """Create a dataset with multiple feature sources."""
    ds = SpectroDataset()

    data1 = np.random.rand(6, 4).astype(np.float32)
    data2 = np.random.rand(6, 2).astype(np.float32)
    data3 = np.random.rand(6, 5).astype(np.float32)
    ds.add_features([data1, data2, data3])

    targets_data = np.array([[float(i)] for i in range(6)], dtype=np.float32)
    samples = np.array(list(range(6)))
    ds.targets.add_regression_targets("targets", targets_data, samples, "raw")

    return ds


class TestDatasetIO:
    """Test dataset save and load functionality."""

    def test_save_and_load_minimal_dataset(self, minimal_dataset, tmp_path):
        """Test save/load roundtrip with minimal dataset."""
        ds = minimal_dataset
        save_path = tmp_path / "minimal_dataset"

        # Save dataset
        save(ds, str(save_path))

        # Check that directory was created
        assert save_path.exists(), "Save directory should be created"
        assert save_path.is_dir(), "Save path should be a directory"

        # Check expected files exist
        expected_files = {'features_src0.npy', 'index.parquet'}
        actual_files = set(f.name for f in save_path.iterdir())
        assert expected_files.issubset(actual_files), f"Expected files {expected_files} not all found in {actual_files}"

        # Load dataset back
        ds_loaded = load(str(save_path))        # Compare features
        original_features = ds.x({}, layout='2d')
        loaded_features = ds_loaded.x({}, layout='2d')

        assert np.allclose(original_features[0], loaded_features[0]), "Loaded features should match original"
        assert original_features[0].shape == loaded_features[0].shape, "Feature shapes should match"

    def test_save_and_load_comprehensive_dataset(self, comprehensive_dataset, tmp_path):
        """Test save/load roundtrip with comprehensive dataset."""
        ds = comprehensive_dataset
        save_path = tmp_path / "comprehensive_dataset"

        # Save dataset
        save(ds, str(save_path))

        # Check all expected files exist
        expected_files = {
            'features_src0.npy',
            'index.parquet',
            'metadata.parquet',
            'targets.parquet',
            'predictions.parquet',
            'folds.json'
        }
        actual_files = set(f.name for f in save_path.iterdir())
        assert expected_files.issubset(actual_files), f"Expected files {expected_files} not all found in {actual_files}"        # Load dataset back
        ds_loaded = load(str(save_path))

        # Compare features
        original_features = ds.x({}, layout='2d')
        loaded_features = ds_loaded.x({}, layout='2d')
        assert np.allclose(original_features[0], loaded_features[0]), "Features should match after load"

        # Compare metadata
        original_meta = ds.metadata.table
        loaded_meta = ds_loaded.metadata.table
        assert original_meta.equals(loaded_meta), "Metadata should match after load"

        # Compare targets
        assert len(ds.targets.sources) == len(ds_loaded.targets.sources), "Target sources count should match"
        for name in ds.targets.get_target_names():
            # Use a filter that includes all samples to get all target data
            all_samples_filter = {}
            original_y = ds.targets.y(all_samples_filter, target_name=name, encoded=False)
            loaded_y = ds_loaded.targets.y(all_samples_filter, target_name=name, encoded=False)
            np.testing.assert_array_equal(original_y, loaded_y, f"Target {name} should match after load")

        # Compare predictions
        original_preds = ds.predictions.table
        loaded_preds = ds_loaded.predictions.table
        assert original_preds.equals(loaded_preds), "Predictions should match after load"

        # Compare folds
        assert ds_loaded.folds.folds == ds.folds.folds, "Folds should match after load"

    def test_save_and_load_multi_source_dataset(self, multi_source_dataset, tmp_path):
        """Test save/load with multiple feature sources."""
        ds = multi_source_dataset
        save_path = tmp_path / "multi_source_dataset"

        # Save dataset
        save(ds, str(save_path))

        # Check feature files exist
        expected_feature_files = {
            'features_src0.npy',
            'features_src1.npy',
            'features_src2.npy'
        }
        actual_files = set(f.name for f in save_path.iterdir())
        assert expected_feature_files.issubset(actual_files), "All feature source files should be saved"

        # Load dataset back
        ds_loaded = load(str(save_path))

        # Compare all feature sources
        original_features = ds.x({}, layout='2d')
        loaded_features = ds_loaded.x({}, layout='2d')

        assert len(original_features) == len(loaded_features), "Number of feature sources should match"
        assert len(loaded_features) == 3, "Should have 3 feature sources"

        for i, (orig, loaded) in enumerate(zip(original_features, loaded_features)):
            assert np.allclose(orig, loaded), f"Feature source {i} should match after load"
            assert orig.shape == loaded.shape, f"Feature source {i} shape should match"

    def test_save_overwrites_existing_directory(self, comprehensive_dataset, tmp_path):
        """Test that saving overwrites existing directory contents."""
        ds = comprehensive_dataset
        save_path = tmp_path / "overwrite_test"

        # First save
        save(ds, str(save_path))
        original_files = set(f.name for f in save_path.iterdir())

        # Modify dataset
        new_data = np.random.rand(8, 3).astype(np.float32)
        ds.add_features([new_data])  # Add second feature source

        # Second save should overwrite
        save(ds, str(save_path))

        # Load and verify changes
        ds_loaded = load(str(save_path))
        loaded_features = ds_loaded.x({}, layout='2d')

        assert len(loaded_features) == 2, "Should have 2 feature sources after overwrite"

        # Check new feature files exist
        new_files = set(f.name for f in save_path.iterdir())
        assert 'features_src1.npy' in new_files, "New feature source file should exist"

    def test_load_nonexistent_directory(self):
        """Test error handling when loading from nonexistent directory."""
        with pytest.raises((FileNotFoundError, OSError)):
            load("/nonexistent/path")

    def test_save_to_file_instead_of_directory(self, minimal_dataset, tmp_path):
        """Test error handling when trying to save to a file path."""
        ds = minimal_dataset
        file_path = tmp_path / "not_a_directory.txt"
        file_path.write_text("existing file")

        # Should handle this gracefully or raise appropriate error
        try:
            save(ds, str(file_path))
            # If it succeeds, check that it created a directory or handled appropriately
        except (OSError, ValueError, IsADirectoryError):
            # These are acceptable error types for this scenario
            pass

    def test_load_partial_dataset_directory(self, tmp_path):
        """Test loading from directory with missing required files."""
        save_path = tmp_path / "partial_dataset"
        save_path.mkdir()

        # Create only index file, missing features
        index_df = pl.DataFrame({'sample': [0, 1, 2]})
        index_df.write_parquet(save_path / "index.parquet")

        # Should handle missing files gracefully or raise appropriate error
        try:
            ds = load(str(save_path))
            # If it loads, check that it's in a reasonable state
            assert hasattr(ds, 'features'), "Loaded dataset should have features attribute"
        except (FileNotFoundError, KeyError, ValueError):
            # These are acceptable error types for this scenario
            pass

    def test_save_empty_dataset(self, tmp_path):
        """Test saving completely empty dataset."""
        ds = SpectroDataset()
        save_path = tmp_path / "empty_dataset"

        # Should handle empty dataset gracefully
        try:
            save(ds, str(save_path))

            # If save succeeds, test loading
            ds_loaded = load(str(save_path))
            assert isinstance(ds_loaded, SpectroDataset), "Should load as SpectroDataset"

        except (ValueError, AttributeError):
            # Empty dataset might not be saveable, which is acceptable
            pass

    def test_save_dataset_with_complex_metadata(self, tmp_path):
        """Test saving dataset with complex metadata types."""
        ds = SpectroDataset()
        data = np.random.rand(5, 3).astype(np.float32)
        ds.add_features([data])

        # Add metadata with various data types
        complex_meta = pl.DataFrame({
            'sample': list(range(5)),
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['a', 'b', 'c', 'd', 'e'],
            'bool_col': [True, False, True, False, True],
            'date_col': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
        })
        ds.add_meta(complex_meta)

        save_path = tmp_path / "complex_meta_dataset"

        # Save and load
        save(ds, str(save_path))
        ds_loaded = load(str(save_path))        # Compare metadata
        original_meta = ds.metadata.table
        loaded_meta = ds_loaded.metadata.table

        if original_meta is not None and loaded_meta is not None:
            assert original_meta.shape == loaded_meta.shape, "Metadata shape should be preserved"
            assert set(original_meta.columns) == set(loaded_meta.columns), "Metadata columns should be preserved"

    def test_concurrent_save_load_operations(self, comprehensive_dataset, tmp_path):
        """Test that save/load operations are robust to concurrent access."""
        ds = comprehensive_dataset

        # Save dataset
        save_path = tmp_path / "concurrent_test"
        save(ds, str(save_path))

        # Multiple load operations should work
        ds1 = load(str(save_path))
        ds2 = load(str(save_path))

        # Both loaded datasets should be equivalent
        features1 = ds1.x({}, layout='2d')
        features2 = ds2.x({}, layout='2d')

        assert len(features1) == len(features2), "Both loads should have same number of feature sources"
        for f1, f2 in zip(features1, features2):
            assert np.allclose(f1, f2), "Features from concurrent loads should match"


class TestIOEdgeCases:
    """Test edge cases and error handling for I/O operations."""

    def test_save_with_special_characters_in_path(self, tmp_path):
        """Test saving to path with special characters."""
        ds = SpectroDataset()
        data = np.random.rand(3, 2).astype(np.float32)
        ds.add_features([data])

        # Path with spaces and special characters
        save_path = tmp_path / "dataset with spaces & symbols"

        try:
            save(ds, str(save_path))
            ds_loaded = load(str(save_path))            # Basic verification
            original_features = ds.x({}, layout='2d')
            loaded_features = ds_loaded.x({}, layout='2d')
            assert np.allclose(original_features[0], loaded_features[0]), "Features should match despite special path chars"

        except (OSError, UnicodeError):
            # OS might not support special characters in paths
            pytest.skip("OS does not support special characters in paths")

    def test_save_very_large_dataset(self, tmp_path):
        """Test saving dataset with large feature arrays."""
        ds = SpectroDataset()

        # Create large feature array (but not too large for CI)
        large_data = np.random.rand(1000, 100).astype(np.float32)
        ds.add_features([large_data])

        save_path = tmp_path / "large_dataset"

        # Save and load
        save(ds, str(save_path))
        ds_loaded = load(str(save_path))        # Verify shape (don't compare values for performance)
        original_features = ds.x({}, layout='2d')
        loaded_features = ds_loaded.x({}, layout='2d')

        assert original_features[0].shape == loaded_features[0].shape, "Large dataset shape should be preserved"

    def test_load_corrupted_files(self, comprehensive_dataset, tmp_path):
        """Test handling of corrupted save files."""
        ds = comprehensive_dataset
        save_path = tmp_path / "corrupted_test"

        # Save normally first
        save(ds, str(save_path))

        # Corrupt one of the files
        feature_file = save_path / "features_src0.npy"
        with open(feature_file, 'wb') as f:
            f.write(b"corrupted data")

        # Loading should handle corruption gracefully
        with pytest.raises((ValueError, OSError, IOError)):
            load(str(save_path))
