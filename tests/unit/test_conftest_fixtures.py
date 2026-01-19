"""
Tests for pytest fixtures defined in tests/conftest.py.

These tests verify that the core fixtures work correctly,
are reproducible, and provide the expected data structures.
"""

import numpy as np
import pytest
from pathlib import Path


class TestSyntheticBuilderFactory:
    """Tests for synthetic_builder_factory fixture."""

    def test_factory_creates_builder(self, synthetic_builder_factory):
        """Factory should return a builder instance."""
        builder = synthetic_builder_factory(n_samples=50)
        assert builder is not None
        assert hasattr(builder, 'build')
        assert hasattr(builder, 'build_arrays')

    def test_factory_respects_n_samples(self, synthetic_builder_factory):
        """Factory should respect n_samples parameter."""
        X, y = synthetic_builder_factory(n_samples=75).build_arrays()
        assert X.shape[0] == 75
        assert y.shape[0] == 75

    def test_factory_respects_random_state(self, synthetic_builder_factory):
        """Factory should produce reproducible results."""
        X1, y1 = synthetic_builder_factory(n_samples=50, random_state=123).build_arrays()
        X2, y2 = synthetic_builder_factory(n_samples=50, random_state=123).build_arrays()

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_factory_different_seeds_differ(self, synthetic_builder_factory):
        """Different random seeds should produce different data."""
        X1, _ = synthetic_builder_factory(n_samples=50, random_state=1).build_arrays()
        X2, _ = synthetic_builder_factory(n_samples=50, random_state=2).build_arrays()

        assert not np.allclose(X1, X2)


class TestStandardRegressionDataset:
    """Tests for standard_regression_dataset fixture."""

    def test_dataset_not_none(self, standard_regression_dataset):
        """Dataset should not be None."""
        assert standard_regression_dataset is not None

    def test_dataset_has_correct_samples(self, standard_regression_dataset):
        """Dataset should have 200 samples total."""
        assert standard_regression_dataset.num_samples == 200

    def test_dataset_has_partitions(self, standard_regression_dataset):
        """Dataset should have train and test partitions."""
        X_train = standard_regression_dataset.x({"partition": "train"})
        X_test = standard_regression_dataset.x({"partition": "test"})

        # 80/20 split of 200 samples
        assert X_train.shape[0] == 160
        assert X_test.shape[0] == 40

    def test_dataset_targets_in_range(self, standard_regression_dataset):
        """Target values should be in specified range."""
        y = standard_regression_dataset.y({})
        assert y.min() >= 10 - 1e-6  # Small tolerance for float precision
        assert y.max() <= 50 + 1e-6

    def test_dataset_is_reproducible(self, standard_regression_dataset):
        """Same fixture should return identical data across calls."""
        X1 = standard_regression_dataset.x({})
        X2 = standard_regression_dataset.x({})
        np.testing.assert_array_equal(X1, X2)


class TestStandardClassificationDataset:
    """Tests for standard_classification_dataset fixture."""

    def test_dataset_has_correct_samples(self, standard_classification_dataset):
        """Dataset should have 150 samples total."""
        assert standard_classification_dataset.num_samples == 150

    def test_dataset_has_three_classes(self, standard_classification_dataset):
        """Dataset should have 3 distinct classes."""
        y = standard_classification_dataset.y({})
        unique_classes = np.unique(y)
        assert len(unique_classes) == 3

    def test_classes_are_integers(self, standard_classification_dataset):
        """Class labels should be integers."""
        y = standard_classification_dataset.y({})
        # Check if all values are close to integers
        assert np.allclose(y, np.round(y))


class TestStandardBinaryDataset:
    """Tests for standard_binary_dataset fixture."""

    def test_dataset_has_correct_samples(self, standard_binary_dataset):
        """Dataset should have 100 samples total."""
        assert standard_binary_dataset.num_samples == 100

    def test_dataset_has_two_classes(self, standard_binary_dataset):
        """Dataset should have exactly 2 classes."""
        y = standard_binary_dataset.y({})
        unique_classes = np.unique(y)
        assert len(unique_classes) == 2


class TestMultiSourceDataset:
    """Tests for multi_source_dataset fixture."""

    def test_dataset_has_correct_samples(self, multi_source_dataset):
        """Dataset should have 200 samples."""
        assert multi_source_dataset.num_samples == 200

    def test_dataset_has_features(self, multi_source_dataset):
        """Dataset should have features."""
        X = multi_source_dataset.x({})
        assert X is not None
        assert X.shape[0] == 200
        assert X.shape[1] > 0


class TestFreshRegressionDataset:
    """Tests for fresh_regression_dataset fixture."""

    def test_returns_dataset(self, fresh_regression_dataset):
        """Should return a SpectroDataset."""
        assert fresh_regression_dataset is not None
        assert hasattr(fresh_regression_dataset, 'x')
        assert hasattr(fresh_regression_dataset, 'y')

    def test_has_correct_samples(self, fresh_regression_dataset):
        """Should have 100 samples."""
        assert fresh_regression_dataset.num_samples == 100


class TestRegressionArrays:
    """Tests for regression_arrays fixture."""

    def test_returns_tuple(self, regression_arrays):
        """Should return (X, y) tuple."""
        assert isinstance(regression_arrays, tuple)
        assert len(regression_arrays) == 2

    def test_arrays_are_numpy(self, regression_arrays):
        """Both X and y should be numpy arrays."""
        X, y = regression_arrays
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_correct_shapes(self, regression_arrays):
        """X and y should have compatible shapes."""
        X, y = regression_arrays
        assert X.shape[0] == y.shape[0] if y.ndim == 1 else X.shape[0] == y.shape[0]
        assert X.shape[0] == 100  # Default n_samples
        assert X.ndim == 2
        # y can be 1D (single target) or 2D (multi-target or single target column)
        assert y.ndim in [1, 2]


class TestClassificationArrays:
    """Tests for classification_arrays fixture."""

    def test_returns_tuple(self, classification_arrays):
        """Should return (X, y) tuple."""
        assert isinstance(classification_arrays, tuple)
        assert len(classification_arrays) == 2

    def test_y_contains_class_labels(self, classification_arrays):
        """y should contain discrete class labels."""
        X, y = classification_arrays
        unique = np.unique(y)
        assert len(unique) == 3  # 3 classes


class TestSyntheticDatasetFolder:
    """Tests for synthetic_dataset_folder fixture."""

    def test_returns_path(self, synthetic_dataset_folder):
        """Should return a Path object."""
        assert isinstance(synthetic_dataset_folder, Path)

    def test_folder_exists(self, synthetic_dataset_folder):
        """Folder should exist."""
        assert synthetic_dataset_folder.exists()
        assert synthetic_dataset_folder.is_dir()

    def test_contains_xcal_file(self, synthetic_dataset_folder):
        """Folder should contain Xcal file."""
        xcal_files = list(synthetic_dataset_folder.glob("Xcal*"))
        assert len(xcal_files) > 0

    def test_contains_ycal_file(self, synthetic_dataset_folder):
        """Folder should contain Ycal file."""
        ycal_files = list(synthetic_dataset_folder.glob("Ycal*"))
        assert len(ycal_files) > 0

    def test_contains_xval_file(self, synthetic_dataset_folder):
        """Folder should contain Xval file."""
        xval_files = list(synthetic_dataset_folder.glob("Xval*"))
        assert len(xval_files) > 0

    def test_contains_yval_file(self, synthetic_dataset_folder):
        """Folder should contain Yval file."""
        yval_files = list(synthetic_dataset_folder.glob("Yval*"))
        assert len(yval_files) > 0


class TestSyntheticCsvFile:
    """Tests for synthetic_csv_file fixture."""

    def test_returns_path(self, synthetic_csv_file):
        """Should return a Path object."""
        assert isinstance(synthetic_csv_file, Path)

    def test_file_exists(self, synthetic_csv_file):
        """CSV file should exist."""
        assert synthetic_csv_file.exists()
        assert synthetic_csv_file.is_file()

    def test_file_has_csv_extension(self, synthetic_csv_file):
        """File should have .csv extension."""
        assert synthetic_csv_file.suffix == '.csv'


class TestBaseSyntheticData:
    """Tests for base_synthetic_data fixture."""

    def test_returns_tuple_of_three(self, base_synthetic_data):
        """Should return (X, y, wavelengths) tuple."""
        assert isinstance(base_synthetic_data, tuple)
        assert len(base_synthetic_data) == 3

    def test_all_numpy_arrays(self, base_synthetic_data):
        """All elements should be numpy arrays."""
        X, y, wavelengths = base_synthetic_data
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(wavelengths, np.ndarray)

    def test_correct_sample_count(self, base_synthetic_data):
        """Should have 50 samples."""
        X, y, _ = base_synthetic_data
        assert X.shape[0] == 50
        assert y.shape[0] == 50

    def test_wavelength_range(self, base_synthetic_data):
        """Wavelengths should be in 1000-2000nm range."""
        _, _, wavelengths = base_synthetic_data
        assert wavelengths[0] == 1000
        assert wavelengths[-1] == 2000


class TestSampleWavelengths:
    """Tests for sample_wavelengths fixture."""

    def test_returns_array(self, sample_wavelengths):
        """Should return numpy array."""
        assert isinstance(sample_wavelengths, np.ndarray)

    def test_correct_range(self, sample_wavelengths):
        """Should cover 1000-2500nm range."""
        assert sample_wavelengths[0] == 1000
        assert sample_wavelengths[-1] == 2500

    def test_correct_step(self, sample_wavelengths):
        """Should have 2nm step."""
        diff = np.diff(sample_wavelengths)
        assert np.all(diff == 2)


class TestDatasetAllComplexities:
    """Tests for dataset_all_complexities parametrized fixture."""

    def test_dataset_valid(self, dataset_all_complexities):
        """Each complexity level should produce valid dataset."""
        assert dataset_all_complexities is not None
        assert dataset_all_complexities.num_samples == 100


class TestClassificationNClasses:
    """Tests for classification_n_classes parametrized fixture."""

    def test_correct_class_count(self, classification_n_classes):
        """Dataset should have correct number of classes."""
        y = classification_n_classes.y({})
        unique = np.unique(y)
        # Number of classes should be 2, 3, or 5 based on param
        assert len(unique) in [2, 3, 5]


class TestCsvVariationGenerator:
    """Tests for csv_variation_generator fixture."""

    def test_returns_generator(self, csv_variation_generator):
        """Should return CSVVariationGenerator instance."""
        assert csv_variation_generator is not None
        assert hasattr(csv_variation_generator, 'generate_all_variations')
        assert hasattr(csv_variation_generator, 'with_semicolon_delimiter')
        assert hasattr(csv_variation_generator, 'with_comma_delimiter')


class TestCsvFormatFixtures:
    """Tests for CSV format variation fixtures."""

    def test_semicolon_format_exists(self, csv_semicolon_format):
        """Semicolon format folder should exist."""
        assert csv_semicolon_format.exists()
        assert csv_semicolon_format.is_dir()

    def test_comma_format_exists(self, csv_comma_format):
        """Comma format folder should exist."""
        assert csv_comma_format.exists()
        assert csv_comma_format.is_dir()

    def test_tab_format_exists(self, csv_tab_format):
        """Tab format folder should exist."""
        assert csv_tab_format.exists()
        assert csv_tab_format.is_dir()

    def test_no_headers_format_exists(self, csv_no_headers_format):
        """No headers format folder should exist."""
        assert csv_no_headers_format.exists()
        assert csv_no_headers_format.is_dir()

    def test_with_index_format_exists(self, csv_with_index_format):
        """With index format folder should exist."""
        assert csv_with_index_format.exists()
        assert csv_with_index_format.is_dir()


class TestCsvAllVariations:
    """Tests for csv_all_variations fixture."""

    def test_returns_dict(self, csv_all_variations):
        """Should return dictionary of paths."""
        assert isinstance(csv_all_variations, dict)

    def test_has_multiple_variations(self, csv_all_variations):
        """Should have multiple format variations."""
        assert len(csv_all_variations) >= 5

    def test_all_paths_exist(self, csv_all_variations):
        """All variation paths should exist."""
        for name, path in csv_all_variations.items():
            assert path.exists(), f"Path for '{name}' does not exist: {path}"


class TestFixtureReproducibility:
    """Tests to ensure fixtures produce reproducible results."""

    def test_standard_dataset_reproducible_across_tests(
        self,
        standard_regression_dataset
    ):
        """Session-scoped fixtures should be identical across test runs."""
        # This test runs multiple times with the same fixture
        # The fixture should always return the same data
        X = standard_regression_dataset.x({})
        assert X.shape == (200, 1076)  # Expected shape for 350-2500nm, step 2

        # Check first few values are consistent
        first_sample_sum = X[0].sum()
        assert first_sample_sum != 0  # Should have non-zero values


class TestFixtureIntegration:
    """Integration tests for fixtures with nirs4all components."""

    def test_dataset_works_with_x_accessor(self, standard_regression_dataset):
        """Dataset should work with x() accessor."""
        X = standard_regression_dataset.x({})
        assert X is not None
        assert isinstance(X, np.ndarray)

    def test_dataset_works_with_y_accessor(self, standard_regression_dataset):
        """Dataset should work with y() accessor."""
        y = standard_regression_dataset.y({})
        assert y is not None
        assert isinstance(y, np.ndarray)

    def test_dataset_supports_partition_filter(self, standard_regression_dataset):
        """Dataset should support partition filtering."""
        X_train = standard_regression_dataset.x({"partition": "train"})
        X_test = standard_regression_dataset.x({"partition": "test"})

        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert X_train.shape[0] + X_test.shape[0] == 200
