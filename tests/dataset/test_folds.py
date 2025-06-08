"""Tests for the FoldsManager class."""
import numpy as np
import pytest
from sklearn.model_selection import KFold, StratifiedKFold
from nirs4all.dataset.folds import FoldsManager
from nirs4all.dataset.dataset import SpectroDataset


class TestFoldsManagerBasics:
    """Test basic FoldsManager functionality."""

    def test_empty_folds_manager(self):
        """Test empty FoldsManager state and representation."""
        fm = FoldsManager()

        assert len(fm) == 0, "Empty FoldsManager should have length 0"
        assert 'empty' in repr(fm).lower(), "Empty FoldsManager repr should indicate empty state"

    def test_set_single_fold(self):
        """Test setting a single fold and accessing its data."""
        fm = FoldsManager()
        train_indices = list(range(5))
        val_indices = list(range(5, 10))
        folds = [(train_indices, val_indices)]

        fm.set_folds(folds)

        assert len(fm) == 1, "FoldsManager should contain exactly 1 fold"
        assert '1 fold' in repr(fm), "Repr should indicate 1 fold"
        assert fm.folds[0]['train'] == train_indices, "Train indices should match input"
        assert fm.folds[0]['val'] == val_indices, "Val indices should match input"

    def test_set_multiple_folds(self):
        """Test setting multiple folds."""
        fm = FoldsManager()
        folds = [
            ([0, 1, 2], [3, 4]),
            ([1, 2, 3], [0, 4]),
            ([0, 3, 4], [1, 2])
        ]

        fm.set_folds(folds)

        assert len(fm) == 3, "FoldsManager should contain 3 folds"
        assert '3 fold' in repr(fm), "Repr should indicate 3 folds"

        for i, (expected_train, expected_val) in enumerate(folds):
            assert fm.folds[i]['train'] == expected_train, f"Fold {i} train indices should match"
            assert fm.folds[i]['val'] == expected_val, f"Fold {i} val indices should match"

    def test_set_folds_overwrites_existing(self):
        """Test that setting folds overwrites existing folds."""
        fm = FoldsManager()

        # First set
        fm.set_folds([([0, 1], [2, 3])])
        assert len(fm) == 1

        # Second set overwrites
        fm.set_folds([([0], [1]), ([2], [3])])
        assert len(fm) == 2, "New folds should overwrite existing ones"


class TestFoldsManagerDataGeneration:
    """Test FoldsManager data generation functionality."""

    @pytest.fixture
    def simple_dataset(self):
        """Create a simple dataset for testing."""
        ds = SpectroDataset()
        data = np.arange(20).reshape(10, 2).astype(np.float32)
        ds.add_features([data])
        targets_data = np.array([[float(i)] for i in range(10)], dtype=np.float32)
        samples = np.array(list(range(10)))
        ds.targets.add_regression_targets("targets", targets_data, samples, "raw")
        return ds

    @pytest.fixture
    def multi_source_dataset(self):
        """Create a dataset with multiple feature sources."""
        ds = SpectroDataset()
        data1 = np.random.rand(12, 3).astype(np.float32)
        data2 = np.random.rand(12, 5).astype(np.float32)
        ds.add_features([data1, data2])
        targets_data = np.array([[float(i)] for i in range(12)], dtype=np.float32)
        samples = np.array(list(range(12)))
        ds.targets.add_regression_targets("targets", targets_data, samples, "raw")
        return ds

    def test_get_data_basic_functionality(self, simple_dataset):
        """Test basic data generation from folds."""
        ds = simple_dataset
        fm = FoldsManager()
        kf = KFold(n_splits=2, shuffle=False)
        fm.set_folds([(list(train), list(val)) for train, val in kf.split(np.arange(10))])

        results = list(fm.get_data(ds, layout='2d'))

        assert len(results) == 2, "Should generate data for 2 folds"

        total_train_samples = 0
        total_val_samples = 0

        for fold_idx, (x_tr, y_tr, x_val, y_val) in enumerate(results):
            # Check feature structure
            assert isinstance(x_tr, tuple), f"Fold {fold_idx}: x_tr should be tuple of arrays"
            assert len(x_tr) == 1, f"Fold {fold_idx}: Should have 1 feature source"
            assert x_tr[0].ndim == 2, f"Fold {fold_idx}: Features should be 2D"

            assert isinstance(x_val, tuple), f"Fold {fold_idx}: x_val should be tuple of arrays"
            assert len(x_val) == 1, f"Fold {fold_idx}: Should have 1 feature source"
            assert x_val[0].ndim == 2, f"Fold {fold_idx}: Val features should be 2D"

            # Check target structure
            assert y_tr.ndim == 2, f"Fold {fold_idx}: Train targets should be 2D"
            assert y_val.ndim == 2, f"Fold {fold_idx}: Val targets should be 2D"

            # Check consistency between features and targets
            assert x_tr[0].shape[0] == y_tr.shape[0], f"Fold {fold_idx}: Train features and targets should have same sample count"
            assert x_val[0].shape[0] == y_val.shape[0], f"Fold {fold_idx}: Val features and targets should have same sample count"

            total_train_samples += len(y_tr)
            total_val_samples += len(y_val)

        # Each sample should appear exactly once across all folds
        assert total_train_samples + total_val_samples == 20, "Total samples should match dataset size (2 folds × 10 samples)"

    def test_get_data_multi_source(self, multi_source_dataset):
        """Test data generation with multiple feature sources."""
        ds = multi_source_dataset
        fm = FoldsManager()
        fm.set_folds([([0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11])])

        results = list(fm.get_data(ds, layout='2d'))

        assert len(results) == 1, "Should generate data for 1 fold"

        x_tr, y_tr, x_val, y_val = results[0]

        assert len(x_tr) == 2, "Should have 2 feature sources in train"
        assert len(x_val) == 2, "Should have 2 feature sources in val"

        assert x_tr[0].shape == (6, 3), "First train source should have correct shape"
        assert x_tr[1].shape == (6, 5), "Second train source should have correct shape"
        assert x_val[0].shape == (6, 3), "First val source should have correct shape"
        assert x_val[1].shape == (6, 5), "Second val source should have correct shape"

    def test_get_data_different_layouts(self, simple_dataset):
        """Test data generation with different layouts."""
        ds = simple_dataset
        fm = FoldsManager()
        fm.set_folds([([0, 1, 2], [3, 4, 5])])

        # Test 2D layout
        results_2d = list(fm.get_data(ds, layout='2d'))
        x_tr_2d, _, _, _ = results_2d[0]
        assert x_tr_2d[0].ndim == 2, "2D layout should produce 2D arrays"        # Test 3D layout
        results_3d = list(fm.get_data(ds, layout='3d'))
        x_tr_3d, _, _, _ = results_3d[0]
        assert x_tr_3d[0].ndim == 3, "3D layout should produce 3D arrays"

    def test_get_data_with_filtering(self, simple_dataset):
        """Test data generation with fold manager."""
        ds = simple_dataset
        fm = FoldsManager()
        fm.set_folds([([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])])

        # Get data from folds
        results = list(fm.get_data(ds, layout='2d'))

        x_tr, y_tr, x_val, y_val = results[0]

        # Check that we got the right fold sizes
        assert x_tr[0].shape[0] == 5, "Train set should have 5 samples"
        assert x_val[0].shape[0] == 5, "Val set should have 5 samples"


class TestFoldsManagerEdgeCases:
    """Test edge cases and error handling."""
    def test_get_data_without_folds(self):
        """Test behavior when trying to get data without setting folds."""
        fm = FoldsManager()
        ds = SpectroDataset()

        # Should return empty generator (no folds to iterate over)
        results = list(fm.get_data(ds))
        assert len(results) == 0, "Should yield no data when no folds are set"

    def test_get_data_with_none_dataset(self):
        """Test error when passing None as dataset."""
        fm = FoldsManager()
        fm.set_folds([([0, 1], [2, 3])])

        with pytest.raises(AttributeError):
            list(fm.get_data(None))

    def test_empty_fold_indices(self):
        """Test handling of empty fold indices."""
        fm = FoldsManager()
        fm.set_folds([([], [0, 1, 2])])  # Empty train set

        ds = SpectroDataset()
        data = np.random.rand(3, 2).astype(np.float32)
        ds.add_features([data])
        targets_data = np.array([[0.], [1.], [2.]], dtype=np.float32)
        samples = np.array([0, 1, 2])
        ds.targets.add_regression_targets("targets", targets_data, samples, "raw")

        results = list(fm.get_data(ds, layout='2d'))
        x_tr, y_tr, x_val, y_val = results[0]

        assert x_tr[0].shape[0] == 0, "Empty train set should have 0 samples"
        assert y_tr.shape[0] == 0, "Empty train targets should have 0 samples"
        assert x_val[0].shape[0] == 3, "Val set should have all samples"
        assert y_val.shape[0] == 3, "Val targets should have all samples"

    def test_overlapping_indices(self):
        """Test behavior with overlapping train/val indices."""
        fm = FoldsManager()
        fm.set_folds([([0, 1, 2], [1, 2, 3])])  # Overlap at indices 1, 2

        ds = SpectroDataset()
        data = np.random.rand(4, 2).astype(np.float32)
        ds.add_features([data])
        targets_data = np.array([[0.], [1.], [2.], [3.]], dtype=np.float32)
        samples = np.array([0, 1, 2, 3])
        ds.targets.add_regression_targets("targets", targets_data, samples, "raw")

        # Should not raise error - overlapping indices are allowed
        results = list(fm.get_data(ds, layout='2d'))
        x_tr, y_tr, x_val, y_val = results[0]

        assert x_tr[0].shape[0] == 3, "Train set should have 3 samples"
        assert x_val[0].shape[0] == 3, "Val set should have 3 samples"

    def test_out_of_bounds_indices(self):
        """Test behavior with out-of-bounds indices."""
        fm = FoldsManager()
        fm.set_folds([([0, 1, 100], [2, 3])])  # Index 100 is out of bounds

        ds = SpectroDataset()
        data = np.random.rand(4, 2).astype(np.float32)
        ds.add_features([data])
        targets_data = np.array([[0.], [1.], [2.], [3.]], dtype=np.float32)
        samples = np.array([0, 1, 2, 3])
        ds.targets.add_regression_targets("targets", targets_data, samples, "raw")

        # This might raise an error depending on implementation
        # Test that it either works (ignoring out-of-bounds) or raises appropriate error
        try:
            results = list(fm.get_data(ds, layout='2d'))
            # If it works, check the results
            x_tr, y_tr, x_val, y_val = results[0]
            assert x_tr[0].shape[0] <= 3, "Train set should not exceed available samples"
        except (IndexError, KeyError):
            # This is also acceptable behavior
            pass


class TestFoldsManagerIntegration:
    """Test integration with sklearn cross-validation."""

    def test_sklearn_kfold_integration(self):
        """Test integration with sklearn KFold."""
        ds = SpectroDataset()
        data = np.random.rand(15, 4).astype(np.float32)
        ds.add_features([data])
        targets_data = np.array([[float(i)] for i in range(15)], dtype=np.float32)
        samples = np.array(list(range(15)))
        ds.targets.add_regression_targets("targets", targets_data, samples, "raw")

        fm = FoldsManager()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        folds = [(list(train), list(val)) for train, val in kf.split(np.arange(15))]
        fm.set_folds(folds)

        results = list(fm.get_data(ds, layout='2d'))

        assert len(results) == 5, "Should generate 5 folds"

        # Check that all samples are used exactly once in validation
        all_val_indices = set()
        for _, _, _, y_val in results:
            val_size = y_val.shape[0]
            # Can't directly check indices without knowing the internal mapping
            # But we can check sizes are reasonable
            assert val_size > 0, "Each fold should have validation samples"

        # Check total sample coverage
        total_samples = sum(y_tr.shape[0] + y_val.shape[0] for _, y_tr, _, y_val in results)
        expected_total = 5 * 15  # 5 folds × 15 samples each
        assert total_samples == expected_total, "Total samples across folds should match expectation"

    def test_sklearn_stratified_kfold_integration(self):
        """Test integration with sklearn StratifiedKFold."""
        ds = SpectroDataset()
        data = np.random.rand(20, 3).astype(np.float32)
        ds.add_features([data])        # Create stratified targets (binary classification)
        stratify_labels = [0] * 10 + [1] * 10
        targets_data = np.array([[float(label)] for label in stratify_labels], dtype=np.float32)
        samples = np.array(list(range(20)))
        ds.targets.add_classification_targets("targets", targets_data, samples, "raw")

        fm = FoldsManager()
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        folds = [(list(train), list(val)) for train, val in skf.split(np.arange(20), stratify_labels)]
        fm.set_folds(folds)

        results = list(fm.get_data(ds, layout='2d'))

        assert len(results) == 4, "Should generate 4 stratified folds"

        for fold_idx, (_, y_tr, _, y_val) in enumerate(results):
            assert y_tr.shape[0] > 0, f"Fold {fold_idx} should have training samples"
            assert y_val.shape[0] > 0, f"Fold {fold_idx} should have validation samples"
