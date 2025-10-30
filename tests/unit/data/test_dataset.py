"""Tests for SpectroDataset class."""

import pytest
import numpy as np
from nirs4all.data.dataset import SpectroDataset
from nirs4all.core.task_type import TaskType


class TestSpectroDatasetInitialization:
    """Test dataset initialization and basic properties."""

    def test_init_default_name(self):
        """Test initialization with default name."""
        dataset = SpectroDataset()
        assert dataset.name == "Unknown_dataset"
        assert dataset.num_samples == 0
        assert dataset.n_sources == 0

    def test_init_custom_name(self):
        """Test initialization with custom name."""
        dataset = SpectroDataset("my_dataset")
        assert dataset.name == "my_dataset"

    def test_init_empty_folds(self):
        """Test that folds are empty on initialization."""
        dataset = SpectroDataset()
        assert dataset.folds == []
        assert dataset.num_folds == 0


class TestFeatureOperations:
    """Test feature-related operations."""

    def test_add_samples_basic(self):
        """Test adding samples to dataset."""
        dataset = SpectroDataset("test")
        data = np.random.rand(10, 100)
        dataset.add_samples(data, {"partition": "train"})

        assert dataset.num_samples == 10
        assert dataset.num_features == 100

    def test_x_retrieval(self):
        """Test retrieving features with x() method."""
        dataset = SpectroDataset("test")
        data = np.random.rand(5, 50)
        dataset.add_samples(data, {"partition": "train"})

        X = dataset.x({"partition": "train"})
        assert X.shape == (5, 50)

    def test_headers_storage(self):
        """Test that headers are stored correctly."""
        dataset = SpectroDataset("test")
        data = np.random.rand(5, 3)
        headers = ["1000.0", "2000.0", "3000.0"]
        dataset.add_samples(data, headers=headers)

        retrieved_headers = dataset.headers(0)
        assert retrieved_headers == headers

    def test_header_unit_storage(self):
        """Test that header units are stored correctly."""
        dataset = SpectroDataset("test")
        data = np.random.rand(5, 3)
        headers = ["1000.0", "2000.0", "3000.0"]
        dataset.add_samples(data, headers=headers, header_unit="nm")

        assert dataset.header_unit(0) == "nm"

    def test_multi_partition_samples(self):
        """Test adding samples to different partitions."""
        dataset = SpectroDataset("test")
        train_data = np.random.rand(10, 50)
        test_data = np.random.rand(5, 50)

        dataset.add_samples(train_data, {"partition": "train"})
        dataset.add_samples(test_data, {"partition": "test"})

        assert dataset.num_samples == 15
        assert dataset.x({"partition": "train"}).shape[0] == 10
        assert dataset.x({"partition": "test"}).shape[0] == 5

    def test_features_processings(self):
        """Test getting processing names."""
        dataset = SpectroDataset("test")
        data = np.random.rand(5, 50)
        dataset.add_samples(data)

        processings = dataset.features_processings(0)
        assert "raw" in processings

    def test_is_multi_source_false(self):
        """Test is_multi_source with single source."""
        dataset = SpectroDataset("test")
        data = np.random.rand(5, 50)
        dataset.add_samples(data)

        assert not dataset.is_multi_source()

    def test_features_sources(self):
        """Test features_sources method."""
        dataset = SpectroDataset("test")
        data = np.random.rand(5, 50)
        dataset.add_samples(data)

        assert dataset.features_sources() == 1


class TestTargetOperations:
    """Test target-related operations."""

    def test_add_targets_classification(self):
        """Test adding classification targets."""
        dataset = SpectroDataset("test")
        data = np.random.rand(10, 50)
        targets = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

        dataset.add_samples(data)
        dataset.add_targets(targets)

        assert dataset.task_type == TaskType.MULTICLASS_CLASSIFICATION
        assert dataset.num_classes == 3
        assert dataset.is_classification
        assert not dataset.is_regression

    def test_add_targets_binary(self):
        """Test adding binary classification targets."""
        dataset = SpectroDataset("test")
        data = np.random.rand(10, 50)
        targets = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        dataset.add_samples(data)
        dataset.add_targets(targets)

        assert dataset.task_type == TaskType.BINARY_CLASSIFICATION
        assert dataset.num_classes == 2

    def test_add_targets_regression(self):
        """Test adding regression targets."""
        dataset = SpectroDataset("test")
        data = np.random.rand(10, 50)
        targets = np.array([1.5, 2.3, 4.1, 3.2, 5.7, 2.8, 4.9, 3.5, 2.1, 4.3])

        dataset.add_samples(data)
        dataset.add_targets(targets)

        assert dataset.task_type == TaskType.REGRESSION
        assert dataset.is_regression
        assert not dataset.is_classification

    def test_y_retrieval(self):
        """Test retrieving targets with y() method."""
        dataset = SpectroDataset("test")
        data = np.random.rand(5, 50)
        targets = np.array([0, 1, 0, 1, 0])

        dataset.add_samples(data, {"partition": "train"})
        dataset.add_targets(targets)

        y = dataset.y({"partition": "train"})
        assert len(y) == 5
        np.testing.assert_array_equal(y.flatten(), targets)


class TestMetadataOperations:
    """Test metadata-related operations."""

    def test_add_metadata(self):
        """Test adding metadata to dataset."""
        dataset = SpectroDataset("test")
        data = np.random.rand(5, 50)
        metadata = np.array([
            ["A", "1"],
            ["B", "2"],
            ["C", "3"],
            ["D", "4"],
            ["E", "5"]
        ])

        dataset.add_samples(data)
        dataset.add_metadata(metadata, headers=["group", "value"])

        assert "group" in dataset.metadata_columns
        assert "value" in dataset.metadata_columns

    def test_metadata_retrieval(self):
        """Test retrieving metadata."""
        dataset = SpectroDataset("test")
        data = np.random.rand(3, 50)
        metadata = np.array([["A"], ["B"], ["C"]])

        dataset.add_samples(data, {"partition": "train"})
        dataset.add_metadata(metadata, headers=["group"])

        meta_df = dataset.metadata({"partition": "train"})
        assert len(meta_df) == 3

    def test_metadata_column(self):
        """Test retrieving single metadata column."""
        dataset = SpectroDataset("test")
        data = np.random.rand(3, 50)
        metadata = np.array([["A"], ["B"], ["C"]])

        dataset.add_samples(data)
        dataset.add_metadata(metadata, headers=["group"])

        groups = dataset.metadata_column("group")
        assert len(groups) == 3

    def test_add_metadata_column(self):
        """Test adding new metadata column."""
        dataset = SpectroDataset("test")
        data = np.random.rand(3, 50)
        metadata = np.array([["A"], ["B"], ["C"]])

        dataset.add_samples(data)
        dataset.add_metadata(metadata, headers=["group"])

        # Add new column
        new_values = ["X", "Y", "Z"]
        dataset.add_metadata_column("category", new_values)

        assert "category" in dataset.metadata_columns


class TestCrossValidationFolds:
    """Test cross-validation fold operations."""

    def test_set_folds(self):
        """Test setting cross-validation folds."""
        dataset = SpectroDataset("test")
        folds = [
            ([0, 1, 2], [3, 4]),
            ([0, 1, 3, 4], [2]),
            ([2, 3, 4], [0, 1])
        ]

        dataset.set_folds(folds)

        assert dataset.num_folds == 3
        assert len(dataset.folds) == 3

    def test_folds_property(self):
        """Test folds property access."""
        dataset = SpectroDataset("test")
        folds = [([0, 1], [2, 3])]
        dataset.set_folds(folds)

        retrieved_folds = dataset.folds
        assert len(retrieved_folds) == 1


class TestDatasetProperties:
    """Test dataset properties and size getters."""

    def test_num_samples_property(self):
        """Test num_samples property."""
        dataset = SpectroDataset("test")
        data = np.random.rand(10, 50)
        dataset.add_samples(data)

        assert dataset.num_samples == 10

    def test_num_features_property(self):
        """Test num_features property."""
        dataset = SpectroDataset("test")
        data = np.random.rand(10, 50)
        dataset.add_samples(data)

        assert dataset.num_features == 50

    def test_n_sources_property(self):
        """Test n_sources property."""
        dataset = SpectroDataset("test")
        data = np.random.rand(10, 50)
        dataset.add_samples(data)

        assert dataset.n_sources == 1


class TestDatasetStringRepresentations:
    """Test string representations and summaries."""

    def test_str_representation(self):
        """Test __str__ method."""
        dataset = SpectroDataset("test_dataset")
        data = np.random.rand(5, 50)
        dataset.add_samples(data)

        str_repr = str(dataset)
        assert "test_dataset" in str_repr
        assert "Dataset:" in str_repr

    def test_str_with_task_type(self):
        """Test __str__ includes task type when targets added."""
        dataset = SpectroDataset("test")
        data = np.random.rand(5, 50)
        targets = np.array([0, 1, 0, 1, 0])

        dataset.add_samples(data)
        dataset.add_targets(targets)

        str_repr = str(dataset)
        assert "BINARY_CLASSIFICATION" in str_repr or "binary" in str_repr.lower()

    def test_print_summary(self, capsys):
        """Test print_summary method."""
        dataset = SpectroDataset("test")
        data = np.random.rand(5, 50)
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        dataset.add_samples(data)
        dataset.add_targets(targets)

        dataset.print_summary()

        captured = capsys.readouterr()
        assert "SpectroDataset Summary" in captured.out
        assert "Task Type" in captured.out


class TestBackwardCompatibility:
    """Test backward compatibility of API."""

    def test_legacy_workflow(self):
        """Test that legacy workflow still works."""
        dataset = SpectroDataset("test")

        # Legacy workflow
        train_data = np.random.rand(10, 50)
        test_data = np.random.rand(5, 50)
        train_targets = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        test_targets = np.array([0, 1, 0, 1, 0])

        dataset.add_samples(train_data, {"partition": "train"})
        dataset.add_samples(test_data, {"partition": "test"})
        dataset.add_targets(np.concatenate([train_targets, test_targets]))

        # Should work as before
        X_train = dataset.x({"partition": "train"})
        y_train = dataset.y({"partition": "train"})
        X_test = dataset.x({"partition": "test"})
        y_test = dataset.y({"partition": "test"})

        assert X_train.shape[0] == 10
        assert len(y_train) == 10
        assert X_test.shape[0] == 5
        assert len(y_test) == 5


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataset_properties(self):
        """Test properties on empty dataset."""
        dataset = SpectroDataset()

        assert dataset.num_samples == 0
        assert dataset.num_folds == 0
        assert dataset.task_type is None

    def test_x_on_empty_dataset(self):
        """Test that x() raises error on empty dataset."""
        dataset = SpectroDataset()

        with pytest.raises(ValueError, match="No features available"):
            dataset.x({})

    def test_metadata_before_samples(self):
        """Test that adding metadata before samples works."""
        dataset = SpectroDataset()
        metadata = np.array([["A"], ["B"]])

        # This should work - metadata can be added independently
        dataset.add_metadata(metadata, headers=["group"])
        assert len(dataset.metadata_columns) == 1
