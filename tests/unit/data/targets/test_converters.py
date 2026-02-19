"""Unit tests for data converters."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from nirs4all.data._targets.converters import ColumnWiseTransformer, NumericConverter


class TestNumericConverter:
    """Test suite for NumericConverter class."""

    def test_convert_numeric_data(self):
        """Test conversion of already numeric data."""
        # Use 0-based consecutive integers to avoid re-encoding
        data = np.array([0, 1, 2, 3, 4])

        result, transformer = NumericConverter.convert(data)

        # NumericConverter returns 2D array
        assert result.ndim == 2
        assert np.array_equal(result.flatten(), data.astype(np.float32))
        assert result.dtype == np.float32
        assert transformer is not None

    def test_convert_string_data(self):
        """Test conversion of string data to numeric labels."""
        data = np.array(['cat', 'dog', 'bird', 'cat', 'dog'])

        result, transformer = NumericConverter.convert(data)

        assert result.dtype == np.float32
        assert transformer is not None
        assert hasattr(transformer, 'transform')

        # Check that labels are encoded 0, 1, 2
        unique_values = np.unique(result)
        assert len(unique_values) == 3
        assert np.array_equal(unique_values, np.array([0., 1., 2.]))

    def test_convert_preserves_nan(self):
        """Test that NaN values are preserved during conversion."""
        # Use continuous float data to avoid classification re-encoding
        data = np.array([1.5, 2.7, np.nan, 4.1])

        result, _ = NumericConverter.convert(data)

        # NumericConverter returns 2D, check after flattening
        result_flat = result.flatten()
        assert np.isnan(result_flat[2])
        assert not np.isnan(result_flat[0])
        assert not np.isnan(result_flat[1])
        assert not np.isnan(result_flat[3])

    def test_convert_1d_array(self):
        """Test conversion of 1D array."""
        data = np.array([10, 20, 30])

        result, transformer = NumericConverter.convert(data)

        # NumericConverter always returns 2D
        assert result.ndim == 2
        assert result.shape == (3, 1)

    def test_convert_2d_array(self):
        """Test conversion of 2D array."""
        data = np.array([[1, 2], [3, 4], [5, 6]])

        result, transformer = NumericConverter.convert(data)

        assert result.ndim == 2
        assert result.shape == (3, 2)

    def test_convert_empty_array(self):
        """Test conversion of empty array."""
        data = np.array([])

        result, transformer = NumericConverter.convert(data)

        assert len(result) == 0

class TestColumnWiseTransformer:
    """Test suite for ColumnWiseTransformer class."""

    def test_columnwise_fit_transform(self):
        """Test fitting and transforming column-wise."""
        # Create transformers dict for each column
        scaler1 = StandardScaler()
        scaler2 = StandardScaler()

        # Create data
        data = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.float32)

        # Fit scalers (ColumnWiseTransformer handles reshaping internally)
        scaler1.fit(data[:, 0:1])
        scaler2.fit(data[:, 1:2])

        transformers = {0: scaler1, 1: scaler2}
        col_transformer = ColumnWiseTransformer(transformers)

        # Transform
        result = col_transformer.transform(data)

        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_columnwise_single_column(self):
        """Test with single column (as 2D array)."""
        scaler = StandardScaler()
        data = np.array([[1], [2], [3], [4], [5]], dtype=np.float32)

        scaler.fit(data)
        transformers = {0: scaler}
        col_transformer = ColumnWiseTransformer(transformers)

        # Transform
        result = col_transformer.transform(data)

        assert result.shape == (5, 1)

    def test_columnwise_none_transformer(self):
        """Test with None transformer (identity)."""
        transformers = {0: None}
        col_transformer = ColumnWiseTransformer(transformers)

        data = np.array([1, 2, 3], dtype=np.float32)
        result = col_transformer.transform(data)

        assert np.allclose(result.flatten(), data)

    def test_columnwise_preserves_dtype(self):
        """Test that float32 dtype is preserved."""
        transformers = {0: None}
        col_transformer = ColumnWiseTransformer(transformers)

        data = np.array([1, 2, 3], dtype=np.float32)
        result = col_transformer.transform(data)

        assert result.dtype == np.float32
