import numpy as np
import polars as pl
import pytest
from nirs4all.dataset.dataset import SpectroDataset


@pytest.fixture
def sample_full_dataset(tmp_path):
    """Create a comprehensive SpectroDataset for testing."""
    ds = SpectroDataset()

    # Add features with different shapes
    f1 = np.arange(6).reshape(3, 2).astype(np.float32)
    f2 = (np.arange(6) + 10).reshape(3, 2).astype(np.float32)
    ds.add_features([f1, f2])    # Add targets
    targets_data = np.array([[1], [2], [3]], dtype=np.float32)
    samples = np.array([0, 1, 2])
    ds.targets.add_regression_targets("target", targets_data, samples, "p0")

    # Add metadata
    metadata_df = pl.DataFrame({
        'sample': [0, 1, 2],
        'group': ['a', 'b', 'a'],
        'val': [0.1, 0.2, 0.3]
    })
    ds.add_meta(metadata_df)

    # Set folds
    ds.folds.set_folds([([0, 1], [2])])

    # Add predictions
    preds = np.array([[0.5], [0.6], [0.7]], dtype=np.float32)
    prediction_meta = {
        'model': 'm',
        'fold': 0,
        'repeat': 0,
        'partition': 'val',
        'processing': 'p0',
        'seed': 0
    }
    ds.predictions.add_prediction(preds, prediction_meta)
    return ds


@pytest.fixture
def empty_dataset():
    """Create an empty SpectroDataset for testing edge cases."""
    return SpectroDataset()


class TestSpectroDatasetBasics:
    """Test basic SpectroDataset functionality."""

    def test_empty_dataset_initialization(self, empty_dataset):
        """Test that empty dataset initializes correctly."""
        ds = empty_dataset
        assert len(ds.features.sources) == 0
        assert ds.features.index_df is None
        assert len(ds.targets.sources) == 0  # New API: check sources instead of table
        assert ds.metadata.table is None
        assert len(ds.folds) == 0
        assert ds.predictions.table is None

    def test_repr_format(self, sample_full_dataset):
        """Test that __repr__ contains expected components."""
        ds = sample_full_dataset
        repr_str = repr(ds)
        assert 'SpectroDataset' in repr_str
        assert 'features=' in repr_str
        assert 'targets=' in repr_str
        assert 'metadata=' in repr_str
        assert 'folds=' in repr_str
        assert 'predictions=' in repr_str

    def test_print_summary_content(self, sample_full_dataset, capsys):
        """Test that print_summary produces expected output."""
        ds = sample_full_dataset
        ds.print_summary()
        captured = capsys.readouterr()

        # Check for expected sections
        assert 'SpectroDataset Summary' in captured.out
        assert '📊 Features:' in captured.out
        assert '🎯 Targets:' in captured.out
        assert '📋 Metadata:' in captured.out
        assert '🔄 Folds:' in captured.out
        assert '🔮 Predictions:' in captured.out

        # Check specific content
        assert '3 samples' in captured.out
        assert '2 source(s)' in captured.out
        assert 'Processing versions: [\'p0\']' in captured.out

    def test_empty_dataset_summary(self, empty_dataset, capsys):
        """Test print_summary for empty dataset."""
        ds = empty_dataset
        ds.print_summary()
        captured = capsys.readouterr()

        assert 'No data' in captured.out


class TestSpectroDatasetDataAccess:
    """Test data access methods of SpectroDataset."""

    def test_feature_access_with_filtering(self, sample_full_dataset):
        """Test x() method with various filters and layouts."""
        ds = sample_full_dataset

        # Test with sample filtering and concatenation
        x2d = ds.x({'sample': [0, 2]}, layout='2d', src_concat=True)
        assert isinstance(x2d, tuple)
        assert len(x2d) == 1
        arr = x2d[0]
        assert arr.shape == (2, 4)  # 2 samples, 4 features total (2+2)

        # Verify content matches manual concatenation
        manual = np.hstack([src.array for src in ds.features.sources])[[0, 2]]
        assert np.array_equal(arr, manual)

    def test_target_access_with_filtering(self, sample_full_dataset):
        """Test y() method with filtering."""
        ds = sample_full_dataset
        y = ds.y({'processing': 'p0'})
        assert y.shape == (3, 1)
        expected = np.array([[1], [2], [3]], dtype=np.float32)
        assert np.array_equal(y, expected)

    def test_metadata_access_with_filtering(self, sample_full_dataset):
        """Test meta() method with filtering."""
        ds = sample_full_dataset
        m = ds.meta({'group': 'a'})
        assert isinstance(m, pl.DataFrame)
        assert set(m['sample'].to_list()) == {0, 2}
        assert len(m) == 2

    def test_indexed_features_access(self, sample_full_dataset):
        """Test get_indexed_features method."""
        ds = sample_full_dataset
        features, index_df = ds.get_indexed_features({'sample': [1]}, layout='2d')

        assert isinstance(features, tuple)
        assert len(features) == 2  # Two sources
        assert features[0].shape == (1, 2)  # One sample, 2 features
        assert features[1].shape == (1, 2)  # One sample, 2 features
        assert len(index_df) == 1
        assert index_df['sample'].to_list() == [1]

    def test_indexed_targets_access(self, sample_full_dataset):
        """Test get_indexed_targets method."""
        ds = sample_full_dataset
        target_data = ds.get_indexed_targets({'processing': 'p0'})

        assert isinstance(target_data, list)
        assert len(target_data) == 1
        targets, index_df = target_data[0]
        assert targets.shape == (3, 1)
        assert len(index_df) == 3


class TestSpectroDatasetPersistence:
    """Test save/load functionality of SpectroDataset."""

    def test_save_load_roundtrip(self, sample_full_dataset, tmp_path):
        """Test complete save/load roundtrip preserves data."""
        ds = sample_full_dataset
        path = str(tmp_path / 'dataset')

        # Save via method
        ds.save(path)

        # Load into new dataset
        ds2 = SpectroDataset().load(path)

        # Compare features
        original_features = ds.x({}, src_concat=True)[0]
        loaded_features = ds2.x({}, src_concat=True)[0]
        assert np.allclose(original_features, loaded_features)

        # Compare metadata
        assert ds2.metadata.table is not None
        assert ds2.metadata.table.height == 3        # Compare targets
        assert ds2.targets.y({'processing': 'p0'}).shape == (3, 1)
        original_targets = ds.targets.y({'processing': 'p0'})
        loaded_targets = ds2.targets.y({'processing': 'p0'})
        assert np.array_equal(original_targets, loaded_targets)

        # Compare predictions
        p1 = ds.predictions.prediction({'model': 'm'})
        p2 = ds2.predictions.prediction({'model': 'm'})
        assert np.allclose(p1, p2)

        # Compare folds
        assert len(ds2.folds) == len(ds.folds)
        assert ds2.folds.folds == ds.folds.folds

    def test_save_load_empty_components(self, tmp_path):
        """Test save/load behavior with partially empty dataset."""
        ds = SpectroDataset()

        # Only add features
        data = np.random.rand(5, 3).astype(np.float32)
        ds.add_features([data])

        path = str(tmp_path / 'partial_dataset')
        ds.save(path)

        ds2 = SpectroDataset().load(path)
        assert len(ds2.features.sources) == 1
        assert len(ds2.targets.sources) == 0
        assert ds2.metadata.table is None
        assert len(ds2.folds) == 0
        assert ds2.predictions.table is None


class TestSpectroDatasetEdgeCases:
    """Test edge cases and error conditions."""

    def test_update_targets(self, sample_full_dataset):
        """Test target updating functionality."""
        ds = sample_full_dataset        # Update first two samples with new processing
        new_values = np.array([[10.0], [20.0]], dtype=np.float32)
        sample_indexes = np.array([0, 1])  # Update samples 0 and 1
        ds.update_y(new_values, sample_indexes, 'updated')        # Verify original targets still exist (should return all 3 samples)
        original = ds.y({'processing': 'p0'})
        assert original.shape == (3, 1)

        # Verify updated targets exist (should return 2 samples)
        updated = ds.y({'processing': 'updated'})
        assert updated.shape == (2, 1)
        assert np.array_equal(updated, new_values)

    def test_empty_filter_results(self, sample_full_dataset):
        """Test behavior with filters that return no results."""
        ds = sample_full_dataset

        # Filter that should return empty results
        empty_features = ds.x({'sample': [999]}, layout='2d')
        assert len(empty_features) > 0  # Tuple is returned
        assert all(arr.shape[0] == 0 for arr in empty_features)  # But arrays are empty

        empty_targets = ds.y({'processing': 'nonexistent'})
        assert empty_targets.shape[0] == 0

    def test_invalid_layout_parameter(self, sample_full_dataset):
        """Test behavior with invalid layout parameter."""
        ds = sample_full_dataset

        with pytest.raises((ValueError, KeyError)):
            ds.x({}, layout='invalid_layout')
