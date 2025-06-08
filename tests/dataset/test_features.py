import numpy as np
import polars as pl
import pytest

from nirs4all.dataset.features import FeatureSource, FeatureBlock


class TestFeatureSource:
    """Test FeatureSource class functionality."""

    def test_valid_initialization(self):
        """Test FeatureSource initialization with valid inputs."""
        # Test different float dtypes
        for dtype in [np.float32, np.float64]:
            arr = np.random.rand(5, 3).astype(dtype)
            src = FeatureSource(arr)
            assert src.n_rows == 5
            assert src.n_dims == 3
            assert src.array is arr  # Zero-copy reference

    def test_invalid_type_validation(self):
        """Test FeatureSource rejects invalid input types."""
        with pytest.raises(TypeError, match="array must be a numpy ndarray"):
            FeatureSource([1, 2, 3])  # type: ignore

        with pytest.raises(TypeError, match="array must be a numpy ndarray"):
            FeatureSource([[1, 2], [3, 4]])  # type: ignore

    def test_invalid_dimensions_validation(self):
        """Test FeatureSource rejects non-2D arrays."""
        # 1D array
        arr1d = np.array([1, 2, 3], dtype=np.float32)
        with pytest.raises(ValueError, match="array must be 2-dimensional"):
            FeatureSource(arr1d)

        # 3D array
        arr3d = np.zeros((2, 3, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="array must be 2-dimensional"):
            FeatureSource(arr3d)

    def test_invalid_dtype_validation(self):
        """Test FeatureSource rejects non-floating point arrays."""
        # Integer array
        arr_int = np.zeros((2, 3), dtype=np.int32)
        with pytest.raises(ValueError, match="array must have a floating point dtype"):
            FeatureSource(arr_int)

        # Boolean array
        arr_bool = np.zeros((2, 3), dtype=bool)
        with pytest.raises(ValueError, match="array must have a floating point dtype"):
            FeatureSource(arr_bool)

    def test_properties_and_repr(self):
        """Test FeatureSource properties and string representation."""
        arr = np.random.rand(7, 11).astype(np.float32)
        src = FeatureSource(arr)

        assert src.array is arr
        assert src.n_rows == 7
        assert src.n_dims == 11

        repr_str = repr(src)
        assert "FeatureSource" in repr_str
        assert "shape=(7, 11)" in repr_str
        assert "float32" in repr_str


class TestFeatureBlock:
    """Test FeatureBlock class functionality."""

    def test_empty_initialization(self):
        """Test FeatureBlock initializes empty correctly."""
        fb = FeatureBlock()
        assert len(fb.sources) == 0
        assert fb.index_df is None
        assert fb.n_samples == 0
        assert "FeatureBlock" in repr(fb)
        assert "sources=0" in repr(fb)

    def test_add_features_single_source(self):
        """Test adding a single feature source."""
        fb = FeatureBlock()
        arr = np.random.rand(10, 5).astype(np.float64)
        fb.add_features([arr])

        assert len(fb.sources) == 1
        assert fb.n_samples == 10
        assert fb.sources[0].array is arr  # Zero-copy

        # Check index creation
        assert fb.index_df is not None
        assert isinstance(fb.index_df, pl.DataFrame)
        assert fb.index_df.height == 10
        assert set(fb.index_df.columns) >= {"sample", "row"}
        assert fb.index_df['sample'].to_list() == list(range(10))

    def test_add_features_multiple_sources(self):
        """Test adding multiple feature sources."""
        fb = FeatureBlock()
        arr1 = np.random.rand(8, 4).astype(np.float32)
        arr2 = np.random.rand(8, 6).astype(np.float64)
        arr3 = np.random.rand(8, 2).astype(np.float32)

        fb.add_features([arr1, arr2, arr3])

        assert len(fb.sources) == 3
        assert fb.n_samples == 8
        assert all(src.n_rows == 8 for src in fb.sources)
        assert [src.n_dims for src in fb.sources] == [4, 6, 2]

    def test_add_features_empty_list(self):
        """Test adding empty list of features."""
        fb = FeatureBlock()
        with pytest.raises(ValueError, match="x_list cannot be empty"):
            fb.add_features([])

    def test_add_features_row_mismatch_in_batch(self):
        """Test row count mismatch within single batch."""
        fb = FeatureBlock()
        arr1 = np.zeros((5, 2), dtype=np.float32)
        arr2 = np.zeros((6, 2), dtype=np.float32)  # Different row count

        with pytest.raises(ValueError, match="Array 1 has .* rows, expected .*"):
            fb.add_features([arr1, arr2])

    def test_add_features_row_mismatch_across_batches(self):
        """Test row count mismatch across multiple add_features calls."""
        fb = FeatureBlock()
        arr1 = np.zeros((5, 2), dtype=np.float32)
        arr2 = np.zeros((6, 2), dtype=np.float32)

        fb.add_features([arr1])
        with pytest.raises(ValueError, match="New arrays have .* rows, existing sources have .*"):
            fb.add_features([arr2])

    def test_layout_2d_single_source(self):
        """Test 2D layout with single source."""
        fb = FeatureBlock()
        data = np.arange(12, dtype=np.float32).reshape(6, 2)
        fb.add_features([data])
        result = fb.x({}, layout='2d')
        assert len(result) == 1
        x2d = result[0]
        assert np.array_equal(x2d, data)
        assert np.may_share_memory(x2d, data)  # Zero-copy

    def test_layout_2d_multiple_sources_no_concat(self):
        """Test 2D layout with multiple sources without concatenation."""
        fb = FeatureBlock()
        a = np.arange(10).reshape(5, 2).astype(np.float32)
        b = np.arange(15).reshape(5, 3).astype(np.float32)
        fb.add_features([a, b])

        result = fb.x({}, layout='2d', src_concat=False)
        assert len(result) == 2
        x_a, x_b = result
        assert np.array_equal(x_a, a)
        assert np.array_equal(x_b, b)

    def test_layout_2d_multiple_sources_with_concat(self):
        """Test 2D layout with multiple sources with concatenation."""
        fb = FeatureBlock()
        a = np.arange(10).reshape(5, 2).astype(np.float32)
        b = np.arange(15).reshape(5, 3).astype(np.float32)
        fb.add_features([a, b])

        result = fb.x({}, layout='2d', src_concat=True)
        assert len(result) == 1
        x_concat = result[0]
        expected = np.hstack([a, b])
        assert x_concat.shape == (5, 5)  # 2 + 3 features
        assert np.array_equal(x_concat, expected)

    def test_layout_3d_and_transpose(self):
        """Test 3D and 3D transpose layouts."""
        fb = FeatureBlock()
        data = np.arange(12, dtype=np.float32).reshape(6, 2)
        fb.add_features([data])

        # 3D layout adds variant dimension
        result = fb.x({}, layout='3d')
        assert len(result) == 1
        x3d = result[0]
        assert x3d.shape == (6, 1, 2)
        assert np.array_equal(x3d[:, 0, :], data)

        # 3D transpose swaps last two dimensions
        result = fb.x({}, layout='3d_transpose')
        assert len(result) == 1
        x3dt = result[0]
        assert x3dt.shape == (6, 2, 1)
        assert np.array_equal(x3dt[:, :, 0], data)

    def test_layout_2d_interlaced(self):
        """Test 2D interlaced layout."""
        fb = FeatureBlock()
        a = np.arange(6).reshape(3, 2).astype(np.float32)  # [[0,1], [2,3], [4,5]]
        b = np.arange(9).reshape(3, 3).astype(np.float32)  # [[0,1,2], [3,4,5], [6,7,8]]
        fb.add_features([a, b])

        result = fb.x({}, layout='2d_interlaced')
        assert len(result) == 2  # For now, same as 2d layout
        xi_a, xi_b = result
        assert xi_a.shape == (3, 2)
        assert xi_b.shape == (3, 3)

        # Verify data matches original sources
        assert np.array_equal(xi_a, a)
        assert np.array_equal(xi_b, b)

    def test_filtering_by_sample_indices(self):
        """Test filtering by sample indices."""
        fb = FeatureBlock()
        data = np.arange(20, dtype=np.float32).reshape(10, 2)
        fb.add_features([data])

        # Filter specific samples
        result = fb.x({'sample': [1, 3, 7]}, layout='2d')
        assert len(result) == 1
        x_filtered = result[0]
        expected = data[[1, 3, 7], :]
        assert x_filtered.shape == (3, 2)
        assert np.array_equal(x_filtered, expected)

    def test_get_indexed_features(self):
        """Test get_indexed_features method."""
        fb = FeatureBlock()
        data = np.random.rand(8, 4).astype(np.float32)
        fb.add_features([data])

        # Verify index_df was created
        assert fb.index_df is not None

        # Add custom column to index for filtering
        fb.index_df = fb.index_df.with_columns(
            pl.when(pl.col("sample") < 4).then(pl.lit("train")).otherwise(pl.lit("test")).alias("split")
        )

        features, index_df = fb.get_indexed_features({'split': 'train'}, layout='2d')

        assert len(features) == 1
        x_train, = features
        assert x_train.shape == (4, 4)  # First 4 samples
        assert len(index_df) == 4
        assert all(index_df['split'] == 'train')
        assert np.array_equal(x_train, data[:4, :])

    def test_invalid_layout(self):
        """Test behavior with invalid layout parameter."""
        fb = FeatureBlock()
        data = np.random.rand(5, 3).astype(np.float32)
        fb.add_features([data])

        with pytest.raises((ValueError, KeyError)):
            fb.x({}, layout='invalid_layout')
