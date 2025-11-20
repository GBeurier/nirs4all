"""
Tests for the refactored feature components.

This test suite validates the modular architecture of FeatureSource
using component-based design.
"""

import pytest
import numpy as np
from nirs4all.data._features import (
    FeatureSource,
    FeatureLayout,
    HeaderUnit,
    normalize_layout,
    normalize_header_unit,
    ArrayStorage,
    ProcessingManager,
    HeaderManager,
    LayoutTransformer,
)


class TestFeatureConstants:
    """Test enums and normalization functions."""

    def test_layout_enum_values(self):
        """Test that layout enum values match expected strings."""
        assert FeatureLayout.FLAT_2D.value == "2d"
        assert FeatureLayout.FLAT_2D_INTERLEAVED.value == "2d_interleaved"
        assert FeatureLayout.VOLUME_3D.value == "3d"
        assert FeatureLayout.VOLUME_3D_TRANSPOSE.value == "3d_transpose"

    def test_header_unit_enum_values(self):
        """Test that header unit enum values match expected strings."""
        assert HeaderUnit.WAVENUMBER.value == "cm-1"
        assert HeaderUnit.WAVELENGTH.value == "nm"
        assert HeaderUnit.NONE.value == "none"
        assert HeaderUnit.TEXT.value == "text"
        assert HeaderUnit.INDEX.value == "index"

    def test_normalize_layout_from_string(self):
        """Test normalizing string to FeatureLayout enum."""
        assert normalize_layout("2d") == FeatureLayout.FLAT_2D
        assert normalize_layout("3d_transpose") == FeatureLayout.VOLUME_3D_TRANSPOSE

    def test_normalize_layout_from_enum(self):
        """Test normalizing enum to FeatureLayout enum (identity)."""
        layout = FeatureLayout.FLAT_2D
        assert normalize_layout(layout) == FeatureLayout.FLAT_2D

    def test_normalize_layout_invalid(self):
        """Test that invalid layout string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid layout"):
            normalize_layout("invalid_layout")

    def test_normalize_header_unit_from_string(self):
        """Test normalizing string to HeaderUnit enum."""
        assert normalize_header_unit("cm-1") == HeaderUnit.WAVENUMBER
        assert normalize_header_unit("nm") == HeaderUnit.WAVELENGTH

    def test_normalize_header_unit_from_enum(self):
        """Test normalizing enum to HeaderUnit enum (identity)."""
        unit = HeaderUnit.WAVENUMBER
        assert normalize_header_unit(unit) == HeaderUnit.WAVENUMBER

    def test_normalize_header_unit_invalid(self):
        """Test that invalid unit string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid header unit"):
            normalize_header_unit("invalid_unit")


class TestArrayStorage:
    """Test ArrayStorage component."""

    def test_initialization(self):
        """Test ArrayStorage initialization."""
        storage = ArrayStorage()
        assert storage.num_samples == 0
        assert storage.num_processings == 1
        assert storage.num_features == 0

    def test_add_samples(self):
        """Test adding samples to storage."""
        storage = ArrayStorage()
        data = np.random.rand(10, 5)
        storage.add_samples(data)

        assert storage.num_samples == 10
        assert storage.num_features == 5

    def test_add_processing(self):
        """Test adding a new processing dimension."""
        storage = ArrayStorage()
        data1 = np.random.rand(10, 5)
        storage.add_samples(data1)

        data2 = np.random.rand(10, 5)
        idx = storage.add_processing(data2)

        assert idx == 1
        assert storage.num_processings == 2

    def test_resize_features(self):
        """Test resizing feature dimension."""
        storage = ArrayStorage()
        data = np.random.rand(10, 5)
        storage.add_samples(data)

        storage.resize_features(8)
        assert storage.num_features == 8


class TestProcessingManager:
    """Test ProcessingManager component."""

    def test_initialization(self):
        """Test ProcessingManager initialization with default 'raw'."""
        mgr = ProcessingManager()
        assert mgr.num_processings == 1
        assert mgr.processing_ids == ["raw"]

    def test_add_processing(self):
        """Test adding a new processing."""
        mgr = ProcessingManager()
        idx = mgr.add_processing("msc")

        assert idx == 1
        assert mgr.has_processing("msc")
        assert mgr.get_index("msc") == 1

    def test_rename_processing(self):
        """Test renaming a processing."""
        mgr = ProcessingManager()
        mgr.add_processing("old_name")
        mgr.rename_processing("old_name", "new_name")

        assert mgr.has_processing("new_name")
        assert not mgr.has_processing("old_name")

    def test_add_duplicate_raises_error(self):
        """Test that adding duplicate processing raises error."""
        mgr = ProcessingManager()
        mgr.add_processing("msc")

        with pytest.raises(ValueError, match="already exists"):
            mgr.add_processing("msc")


class TestHeaderManager:
    """Test HeaderManager component."""

    def test_initialization(self):
        """Test HeaderManager initialization."""
        mgr = HeaderManager()
        assert mgr.headers is None
        assert mgr.header_unit == "cm-1"

    def test_set_headers_with_unit(self):
        """Test setting headers with unit."""
        mgr = HeaderManager()
        headers = ["1000", "1100", "1200"]
        mgr.set_headers(headers, unit="nm")

        assert mgr.headers == headers
        assert mgr.header_unit == "nm"

    def test_clear_headers(self):
        """Test clearing headers."""
        mgr = HeaderManager()
        mgr.set_headers(["a", "b", "c"])
        mgr.clear_headers()

        assert mgr.headers is None


class TestLayoutTransformer:
    """Test LayoutTransformer component."""

    def test_flat_2d_transform(self):
        """Test transforming to flat 2D layout."""
        data = np.random.rand(10, 3, 5)  # (samples, processings, features)
        result = LayoutTransformer.transform(data, "2d", 3, 5)

        assert result.shape == (10, 15)  # Flattened: 3*5=15

    def test_flat_2d_interleaved_transform(self):
        """Test transforming to interleaved 2D layout."""
        data = np.random.rand(10, 3, 5)
        result = LayoutTransformer.transform(data, "2d_interleaved", 3, 5)

        assert result.shape == (10, 15)

    def test_volume_3d_transform(self):
        """Test transforming to 3D layout (no change)."""
        data = np.random.rand(10, 3, 5)
        result = LayoutTransformer.transform(data, "3d", 3, 5)

        assert result.shape == (10, 3, 5)

    def test_volume_3d_transpose_transform(self):
        """Test transforming to transposed 3D layout."""
        data = np.random.rand(10, 3, 5)
        result = LayoutTransformer.transform(data, "3d_transpose", 3, 5)

        assert result.shape == (10, 5, 3)

    def test_get_empty_array_2d(self):
        """Test creating empty array for 2D layout."""
        result = LayoutTransformer.get_empty_array("2d", 3, 5)
        assert result.shape == (0, 15)

    def test_get_empty_array_3d(self):
        """Test creating empty array for 3D layout."""
        result = LayoutTransformer.get_empty_array("3d", 3, 5)
        assert result.shape == (0, 3, 5)


class TestFeatureSourceIntegration:
    """Integration tests for refactored FeatureSource."""

    def test_basic_workflow(self):
        """Test basic add samples and retrieve workflow."""
        source = FeatureSource()
        data = np.random.rand(10, 5)
        headers = ["f1", "f2", "f3", "f4", "f5"]

        source.add_samples(data, headers=headers)

        assert source.num_samples == 10
        assert source.num_features == 5
        assert source.headers == headers

    def test_update_features_replacement(self):
        """Test replacing existing processing."""
        source = FeatureSource()
        data = np.random.rand(10, 5)
        source.add_samples(data)

        # Replace raw with msc
        msc_data = np.random.rand(10, 5)
        source.update_features(["raw"], [msc_data], ["msc"])

        assert "msc" in source.processing_ids
        assert source.num_processings == 1

    def test_update_features_addition(self):
        """Test adding new processing."""
        source = FeatureSource()
        data = np.random.rand(10, 5)
        source.add_samples(data)

        # Add new processing
        snv_data = np.random.rand(10, 5)
        source.update_features([""], [snv_data], ["snv"])

        assert "snv" in source.processing_ids
        assert source.num_processings == 2

    def test_x_retrieval_different_layouts(self):
        """Test retrieving data in different layouts."""
        source = FeatureSource()
        data = np.random.rand(10, 5)
        source.add_samples(data)

        # Add another processing
        processed = np.random.rand(10, 5)
        source.update_features([""], [processed], ["processed"])

        # Test different layouts
        x_2d = source.x(list(range(10)), "2d")
        assert x_2d.shape == (10, 10)  # 2 processings * 5 features

        x_3d = source.x(list(range(10)), "3d")
        assert x_3d.shape == (10, 2, 5)

        x_3d_t = source.x(list(range(10)), "3d_transpose")
        assert x_3d_t.shape == (10, 5, 2)

    def test_augment_samples(self):
        """Test sample augmentation."""
        source = FeatureSource()
        data = np.random.rand(10, 5)
        source.add_samples(data)

        # Augment first 2 samples, 3 times each
        aug_data = np.random.rand(6, 5)  # 2 samples * 3 augmentations
        source.augment_samples([0, 1], aug_data, ["augmented"], [3, 3])

        assert source.num_samples == 16  # 10 original + 6 augmented
        assert "augmented" in source.processing_ids

    def test_backward_compatibility_string_layouts(self):
        """Test that string layouts still work (backward compatibility)."""
        source = FeatureSource()
        data = np.random.rand(10, 5)
        source.add_samples(data)

        # These should all work with string layouts
        x1 = source.x(list(range(10)), "2d")
        x2 = source.x(list(range(10)), "3d")
        x3 = source.x(list(range(10)), "2d_interleaved")
        x4 = source.x(list(range(10)), "3d_transpose")

        assert x1.shape[0] == 10
        assert x2.shape[0] == 10
        assert x3.shape[0] == 10
        assert x4.shape[0] == 10

    def test_backward_compatibility_string_units(self):
        """Test that string header units still work."""
        source = FeatureSource()
        headers = ["1000", "1100", "1200"]

        # These should all work with string units
        source.set_headers(headers, unit="cm-1")
        assert source.header_unit == "cm-1"

        source.set_headers(headers, unit="nm")
        assert source.header_unit == "nm"

        source.set_headers(headers, unit="none")
        assert source.header_unit == "none"


class TestFeatureSourceEdgeCases:
    """Test edge cases and error conditions."""

    def test_cannot_add_samples_after_processing(self):
        """Test that adding samples after processing raises error."""
        source = FeatureSource()
        data = np.random.rand(10, 5)
        source.add_samples(data)

        # Add processing
        processed = np.random.rand(10, 5)
        source.update_features([""], [processed], ["processed"])

        # Try to add more samples (should fail)
        with pytest.raises(ValueError, match="already has been processed"):
            source.add_samples(np.random.rand(5, 5))

    def test_empty_source_x_retrieval(self):
        """Test retrieving from empty source."""
        source = FeatureSource()
        x = source.x([], "2d")
        assert x.shape[0] == 0

    def test_feature_dimension_mismatch_with_padding_disabled(self):
        """Test that dimension mismatch raises error when padding disabled."""
        source = FeatureSource(padding=False)
        data = np.random.rand(10, 5)
        source.add_samples(data)

        # Try to add processing with different feature dimension
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            source.update_features([""], [np.random.rand(10, 8)], ["wrong_dim"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
