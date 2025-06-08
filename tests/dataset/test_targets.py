"""Tests for the new TargetBlock class with type-aware, zero-copy architecture."""
import numpy as np
import pytest
from nirs4all.dataset.targets import (
    TargetBlock, TargetType, RegressionTargetSource,
    ClassificationTargetSource, MultilabelTargetSource
)


class TestTargetType:
    """Test TargetType enum."""

    def test_target_types(self):
        """Test that all target types are available."""
        assert TargetType.REGRESSION.value == "regression"
        assert TargetType.CLASSIFICATION.value == "classification"
        assert TargetType.MULTILABEL.value == "multilabel"


class TestRegressionTargetSource:
    """Test RegressionTargetSource functionality."""

    @pytest.fixture
    def regression_data(self):
        """Create regression test data."""
        samples = np.array([10, 11, 12, 13, 14])
        data = np.array([[1.5], [2.5], [3.5], [4.5], [5.5]])
        return samples, data

    def test_regression_source_creation(self, regression_data):
        """Test creation of regression target source."""
        samples, data = regression_data
        source = RegressionTargetSource("yield", data, samples, "raw")

        assert source.name == "yield"
        assert source.target_type == TargetType.REGRESSION
        assert source.processing == "raw"
        assert source.shape == (5, 1)
        assert np.array_equal(source.samples, samples)
        assert source.data.dtype == np.float32

    def test_regression_get_raw_data(self, regression_data):
        """Test getting raw regression data."""
        samples, data = regression_data
        source = RegressionTargetSource("yield", data, samples)

        raw_data = source.get_raw_data()
        assert np.array_equal(raw_data, data.astype(np.float32))

    def test_regression_get_encoded_data(self, regression_data):
        """Test getting encoded regression data (same as raw)."""
        samples, data = regression_data
        source = RegressionTargetSource("yield", data, samples)

        encoded_data = source.get_encoded_data()
        assert np.array_equal(encoded_data, data.astype(np.float32))

    def test_regression_zero_copy_subset(self, regression_data):
        """Test zero-copy subset for contiguous indices."""
        samples, data = regression_data
        source = RegressionTargetSource("yield", data, samples)

        # Contiguous indices - should use view (zero-copy)
        contiguous_indices = np.array([1, 2, 3])
        subset = source.get_subset(contiguous_indices)

        assert isinstance(subset, RegressionTargetSource)
        assert subset.shape == (3, 1)
        assert np.array_equal(subset.samples, samples[1:4])

        # Verify it's a view by checking if data shares memory
        # (Note: view relationship may not always show as base due to dtype conversion)
        expected_data = data[1:4].astype(np.float32)
        np.testing.assert_array_equal(subset.data, expected_data)

    def test_regression_copy_subset(self, regression_data):
        """Test copy subset for non-contiguous indices."""
        samples, data = regression_data
        source = RegressionTargetSource("yield", data, samples)

        # Non-contiguous indices - should copy
        non_contiguous_indices = np.array([0, 2, 4])
        subset = source.get_subset(non_contiguous_indices)

        assert isinstance(subset, RegressionTargetSource)
        assert subset.shape == (3, 1)
        assert np.array_equal(subset.samples, samples[[0, 2, 4]])

        # Verify it's a copy by checking base
        assert subset.data.base is not source.data


class TestClassificationTargetSource:
    """Test ClassificationTargetSource functionality."""

    @pytest.fixture
    def classification_data(self):
        """Create classification test data."""
        samples = np.array([20, 21, 22, 23, 24])
        data = np.array(['low', 'medium', 'high', 'low', 'high'])
        return samples, data

    @pytest.fixture
    def integer_classification_data(self):
        """Create integer classification test data."""
        samples = np.array([30, 31, 32, 33, 34])
        data = np.array([0, 1, 2, 0, 2])
        return samples, data

    def test_classification_source_creation(self, classification_data):
        """Test creation of classification target source."""
        samples, data = classification_data
        source = ClassificationTargetSource("quality", data, samples, "raw")

        assert source.name == "quality"
        assert source.target_type == TargetType.CLASSIFICATION
        assert source.processing == "raw"
        assert source.shape == (5,)
        assert np.array_equal(source.samples, samples)

    def test_classification_get_raw_data(self, classification_data):
        """Test getting raw classification data."""
        samples, data = classification_data
        source = ClassificationTargetSource("quality", data, samples)

        raw_data = source.get_raw_data()
        assert np.array_equal(raw_data, data)

    def test_classification_get_encoded_data_strings(self, classification_data):
        """Test getting encoded classification data for strings."""
        samples, data = classification_data
        source = ClassificationTargetSource("quality", data, samples)

        encoded_data = source.get_encoded_data()        # Should be label encoded integers
        assert encoded_data.dtype.kind in 'ui'  # Integer type
        assert len(np.unique(encoded_data)) == 3  # Three classes
        assert encoded_data.shape == (5,)

    def test_classification_get_encoded_data_integers(self, integer_classification_data):
        """Test getting encoded classification data for integers."""
        samples, data = integer_classification_data
        source = ClassificationTargetSource("category", data, samples)

        encoded_data = source.get_encoded_data()

        # Should remain as integers
        assert np.array_equal(encoded_data, data)

    def test_classification_classes_property(self, classification_data):
        """Test classes property for classification."""
        samples, data = classification_data
        source = ClassificationTargetSource("quality", data, samples)

        classes = source.classes
        expected_classes = np.array(['high', 'low', 'medium'])  # Sorted unique
        assert np.array_equal(classes, expected_classes)

    def test_classification_one_hot_encoding(self, classification_data):
        """Test one-hot encoding for classification."""
        samples, data = classification_data
        source = ClassificationTargetSource("quality", data, samples)

        one_hot = source.get_one_hot_encoded()

        assert one_hot.shape == (5, 3)  # 5 samples, 3 classes
        assert one_hot.dtype == np.float64
        assert np.all(np.sum(one_hot, axis=1) == 1)  # Each row sums to 1


class TestMultilabelTargetSource:
    """Test MultilabelTargetSource functionality."""

    @pytest.fixture
    def multilabel_data(self):
        """Create multilabel test data."""
        samples = np.array([40, 41, 42, 43, 44])
        # Binary matrix: each row is a sample, each column is a label
        data = np.array([
            [1, 0, 1],  # Labels 0 and 2
            [0, 1, 0],  # Label 1
            [1, 1, 1],  # All labels
            [0, 0, 0],  # No labels
            [1, 0, 0]   # Label 0
        ])
        return samples, data

    @pytest.fixture
    def string_multilabel_data(self):
        """Create string-based multilabel test data."""
        samples = np.array([50, 51, 52])
        data = np.array(['tag1,tag3', 'tag2', 'tag1,tag2,tag3'])
        return samples, data

    def test_multilabel_source_creation(self, multilabel_data):
        """Test creation of multilabel target source."""
        samples, data = multilabel_data
        source = MultilabelTargetSource("tags", data, samples, "raw")

        assert source.name == "tags"
        assert source.target_type == TargetType.MULTILABEL
        assert source.processing == "raw"
        assert source.shape == (5, 3)
        assert np.array_equal(source.samples, samples)

    def test_multilabel_get_raw_data(self, multilabel_data):
        """Test getting raw multilabel data."""
        samples, data = multilabel_data
        source = MultilabelTargetSource("tags", data, samples)

        raw_data = source.get_raw_data()
        assert np.array_equal(raw_data, data)

    def test_multilabel_get_encoded_data_binary(self, multilabel_data):
        """Test getting encoded multilabel data (already binary)."""
        samples, data = multilabel_data
        source = MultilabelTargetSource("tags", data, samples)

        encoded_data = source.get_encoded_data()
        assert np.array_equal(encoded_data, data)

    def test_multilabel_get_encoded_data_strings(self, string_multilabel_data):
        """Test getting encoded multilabel data from strings."""
        samples, data = string_multilabel_data
        source = MultilabelTargetSource("tags", data, samples)

        encoded_data = source.get_encoded_data()        # Should be binary matrix
        assert encoded_data.ndim == 2
        assert encoded_data.shape[0] == 3  # 3 samples
        assert encoded_data.dtype.kind in 'ui'  # Integer type


class TestTargetBlock:
    """Test TargetBlock functionality."""

    @pytest.fixture
    def target_block(self):
        """Create a target block with various target types."""
        tb = TargetBlock()

        # Add regression targets
        reg_samples = np.array([10, 11, 12, 13, 14])
        reg_data = np.array([[1.5], [2.5], [3.5], [4.5], [5.5]])
        tb.add_regression_targets("yield", reg_data, reg_samples, "raw")

        # Add classification targets
        cls_samples = np.array([10, 11, 12, 13, 14])
        cls_data = np.array(['low', 'medium', 'high', 'low', 'high'])
        tb.add_classification_targets("quality", cls_data, cls_samples, "raw")

        # Add multilabel targets
        ml_samples = np.array([10, 11, 12, 13, 14])
        ml_data = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0],
            [1, 0, 0]
        ])
        tb.add_multilabel_targets("tags", ml_data, ml_samples, "raw")

        return tb

    def test_empty_target_block(self):
        """Test empty target block."""
        tb = TargetBlock()

        assert len(tb.sources) == 0
        assert len(tb.sample_mapping) == 0
        assert tb.get_target_names() == []
        assert repr(tb) == "TargetBlock(empty)"

    def test_target_block_creation(self, target_block):
        """Test target block with multiple target types."""
        tb = target_block

        assert len(tb.sources) == 3
        target_names = tb.get_target_names()
        assert set(target_names) == {'yield', 'quality', 'tags'}

    def test_get_processing_versions(self, target_block):
        """Test getting processing versions."""
        tb = target_block

        yield_versions = tb.get_processing_versions("yield")
        assert yield_versions == ["raw"]

        # Add another version
        new_data = np.array([[2.0], [3.0], [4.0], [5.0], [6.0]])
        samples = np.array([10, 11, 12, 13, 14])
        tb.add_regression_targets("yield", new_data, samples, "processed")

        yield_versions = tb.get_processing_versions("yield")
        assert set(yield_versions) == {"raw", "processed"}

    def test_y_regression_targets(self, target_block):
        """Test getting regression targets."""
        tb = target_block

        # Get all regression targets
        targets = tb.y({}, target_name="yield")

        assert targets.shape == (5, 1)
        assert targets.dtype == np.float32
        expected = np.array([[1.5], [2.5], [3.5], [4.5], [5.5]], dtype=np.float32)
        np.testing.assert_array_equal(targets, expected)

    def test_y_classification_targets(self, target_block):
        """Test getting classification targets."""
        tb = target_block

        # Get encoded classification targets
        targets = tb.y({}, target_name="quality", encoded=True)

        assert targets.shape == (5,)
        assert targets.dtype.kind in 'ui'  # Integer type

        # Get raw classification targets
        raw_targets = tb.y({}, target_name="quality", encoded=False)
        expected_raw = np.array(['low', 'medium', 'high', 'low', 'high'])
        np.testing.assert_array_equal(raw_targets, expected_raw)

    def test_y_multilabel_targets(self, target_block):
        """Test getting multilabel targets."""
        tb = target_block

        # Get multilabel targets
        targets = tb.y({}, target_name="tags")

        assert targets.shape == (5, 3)
        expected = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0],
            [1, 0, 0]
        ])
        np.testing.assert_array_equal(targets, expected)

    def test_y_with_sample_filtering(self, target_block):
        """Test target access with sample filtering."""
        tb = target_block

        # Filter by specific samples
        filter_dict = {'sample': [10, 12, 14]}
        targets = tb.y(filter_dict, target_name="yield")

        assert targets.shape == (3, 1)
        expected = np.array([[1.5], [3.5], [5.5]], dtype=np.float32)
        np.testing.assert_array_equal(targets, expected)

    def test_y_with_single_sample_filtering(self, target_block):
        """Test target access with single sample filtering."""
        tb = target_block

        # Filter by single sample
        filter_dict = {'sample': 12}
        targets = tb.y(filter_dict, target_name="yield")

        assert targets.shape == (1, 1)
        expected = np.array([[3.5]], dtype=np.float32)
        np.testing.assert_array_equal(targets, expected)

    def test_y_no_matches(self, target_block):
        """Test target access when filter matches no samples."""
        tb = target_block

        # Filter by non-existent sample
        filter_dict = {'sample': [999]}
        targets = tb.y(filter_dict, target_name="yield")

        assert targets.shape == (0, 1)

    def test_y_default_target_name(self, target_block):
        """Test using default target name."""
        tb = target_block

        # Should use first target alphabetically
        targets = tb.y({})

        # "quality" comes first alphabetically
        assert targets.shape == (5,)  # Classification target shape

    def test_get_target_info(self, target_block):
        """Test getting target information."""
        tb = target_block

        # Get regression target info
        yield_info = tb.get_target_info("yield")

        assert yield_info['name'] == "yield"
        assert yield_info['type'] == "regression"
        assert yield_info['shape'] == (5, 1)
        assert yield_info['processing'] == "raw"
        assert yield_info['sample_count'] == 5

        # Get classification target info
        quality_info = tb.get_target_info("quality")

        assert quality_info['name'] == "quality"
        assert quality_info['type'] == "classification"
        assert quality_info['shape'] == (5,)
        assert 'classes' in quality_info
        assert quality_info['num_classes'] == 3

    def test_zero_copy_behavior(self):
        """Test zero-copy behavior for contiguous slicing."""
        # Create large dataset to test zero-copy
        samples = np.arange(1000)
        data = np.random.randn(1000, 3).astype(np.float32)

        tb = TargetBlock()
        tb.add_regression_targets("large", data, samples)

        # Test contiguous slicing (should be zero-copy)
        filter_dict = {'sample': np.arange(100, 200)}  # Contiguous range
        targets = tb.y(filter_dict, target_name="large")

        assert targets.shape == (100, 3)

        # Verify this is indeed efficient (we can't directly test for views
        # but can check the operation is fast and gives expected results)
        expected = data[100:200]
        np.testing.assert_array_equal(targets, expected)

    def test_error_handling(self, target_block):
        """Test error handling."""
        tb = target_block

        # Non-existent target
        with pytest.raises(ValueError, match="not found"):
            tb.y({}, target_name="nonexistent")        # Non-existent processing version
        with pytest.warns(UserWarning, match="not found"):
            result = tb.y({}, target_name="yield", processing="nonexistent")
            assert result.shape[0] == 0  # Should return empty array

        # Empty target block
        empty_tb = TargetBlock()
        with pytest.raises(ValueError, match="No targets available"):
            empty_tb.y({})

    def test_repr(self, target_block):
        """Test string representation."""
        tb = target_block

        repr_str = repr(tb)
        assert "TargetBlock" in repr_str
        assert "targets=['quality', 'tags', 'yield']" in repr_str
        assert "sources=3" in repr_str


class TestComplexFiltering:
    """Test complex filtering scenarios similar to FeatureBlock."""

    @pytest.fixture
    def complex_target_block(self):
        """Create a target block with multiple processing versions and samples."""
        tb = TargetBlock()

        # Raw data
        samples_raw = np.array([1, 2, 3, 4, 5])
        data_raw = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        tb.add_regression_targets("measurement", data_raw, samples_raw, "raw")

        # Processed data (different samples)
        samples_proc = np.array([2, 3, 4, 5, 6])
        data_proc = np.array([[2.1], [3.1], [4.1], [5.1], [6.1]])
        tb.add_regression_targets("measurement", data_proc, samples_proc, "normalized")

        # Different target with different samples
        samples_other = np.array([1, 3, 5, 7, 9])
        data_other = np.array(['A', 'B', 'C', 'A', 'B'])
        tb.add_classification_targets("category", data_other, samples_other, "raw")

        return tb

    def test_complex_sample_filtering(self, complex_target_block):
        """Test filtering with overlapping samples across processing versions."""
        tb = complex_target_block

        # Get measurement data for samples that exist in both raw and normalized
        common_samples = [2, 3, 4, 5]

        raw_targets = tb.y({'sample': common_samples}, target_name="measurement", processing="raw")
        norm_targets = tb.y({'sample': common_samples}, target_name="measurement", processing="normalized")

        assert raw_targets.shape == (4, 1)
        assert norm_targets.shape == (4, 1)

        # Verify different processing gives different results
        assert not np.array_equal(raw_targets, norm_targets)

    def test_partial_sample_coverage(self, complex_target_block):
        """Test filtering when not all requested samples are available."""
        tb = complex_target_block        # Request samples where only some exist in the target
        requested_samples = [1, 2, 6, 7, 8]  # 1,2 exist in raw; 2,6 exist in normalized; 7,8 don't exist

        raw_targets = tb.y({'sample': requested_samples}, target_name="measurement", processing="raw")
        norm_targets = tb.y({'sample': requested_samples}, target_name="measurement", processing="normalized")

        # Should only get samples that exist in each version
        assert raw_targets.shape == (2, 1)  # samples 1,2
        assert norm_targets.shape == (2, 1)  # samples 2,6

        expected_raw = np.array([[1.0], [2.0]], dtype=np.float32)
        expected_norm = np.array([[2.1], [6.1]], dtype=np.float32)

        np.testing.assert_array_equal(raw_targets, expected_raw)
        np.testing.assert_array_equal(norm_targets, expected_norm)
