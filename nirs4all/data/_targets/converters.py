"""Numeric conversion utilities for target data."""

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer

from .encoders import FlexibleLabelEncoder


class NumericConverter:
    """
    Converts raw target data to numeric representation.

    Handles multiple data types (numeric, string, object, mixed) and applies
    appropriate transformations column-wise. Supports both classification
    labels and regression targets.

    Methods:
        convert(data, existing_transformer): Convert data to numeric format with appropriate transformer

    Examples:
        >>> converter = NumericConverter()
        >>> data = np.array(['cat', 'dog', 'cat', 'bird'])
        >>> numeric, transformer = converter.convert(data)
        >>> numeric
        array([0., 1., 0., 2.])

    See Also:
        FlexibleLabelEncoder: Handles label encoding with unseen labels
    """

    @staticmethod
    def convert(data: np.ndarray,
                existing_transformer: Optional[TransformerMixin] = None
                ) -> Tuple[np.ndarray, TransformerMixin]:
        """
        Convert raw target data to numeric format.

        Analyzes data type and applies appropriate transformation:
        - Already numeric: identity or label encoding for classification
        - String/object: label encoding
        - Mixed columns: column-wise transformation

        Args:
            data (np.ndarray): Raw target data of any dtype
            existing_transformer (TransformerMixin, optional): Reuse existing transformer if provided (for appending data)

        Returns:
            Tuple[np.ndarray, TransformerMixin]: Tuple of (numeric_data, transformer)

        Notes:
        - Preserves NaN values in numeric data
        - Detects classification labels (small set of integer-like values)
        - Creates column-wise transformers for mixed data types
        - Always returns float32 dtype for consistency

        Examples:
        >>> # Numeric regression data
        >>> data = np.array([1.5, 2.3, 3.1])
        >>> numeric, trans = NumericConverter.convert(data)
        >>> numeric.dtype
        dtype('float32')

        >>> # Classification labels
        >>> data = np.array([1, 2, 1, 3])
        >>> numeric, trans = NumericConverter.convert(data)
        >>> numeric
        array([0., 1., 0., 2.])  # Re-encoded to 0-based
        """
        # Reuse existing transformer if provided
        if existing_transformer is not None:
            if hasattr(existing_transformer, 'transform'):
                numeric = existing_transformer.transform(data)
                return numeric.astype(np.float32), existing_transformer

        # Ensure 2D shape
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Check if already numeric
        if np.issubdtype(data.dtype, np.number):
            return NumericConverter._handle_numeric_data(data)

        # Handle non-numeric data column by column
        return NumericConverter._handle_mixed_data(data)

    @staticmethod
    def _handle_numeric_data(data: np.ndarray) -> Tuple[np.ndarray, TransformerMixin]:
        """
        Handle already numeric data.

        Args:
            data (np.ndarray): Numeric data array

        Returns:
            Tuple[np.ndarray, TransformerMixin]: Tuple of (numeric_data, transformer)
        """
        # Check if classification labels needing encoding
        data_flat = data.flatten()
        unique_vals = np.unique(data_flat[~np.isnan(data_flat)])

        # Detect classification: integer-like values, small set, not 0-based
        is_integer_like = np.allclose(unique_vals, np.round(unique_vals), atol=1e-10)
        expected_consecutive = np.arange(len(unique_vals))
        is_classification = (
            is_integer_like
            and len(unique_vals) <= 50
            and not np.array_equal(unique_vals, expected_consecutive)
        )

        if is_classification:
            # Re-encode to 0-based consecutive integers
            encoder = FlexibleLabelEncoder()
            encoded = encoder.fit_transform(data_flat.astype(np.int32))
            return encoded.reshape(data.shape).astype(np.float32), encoder
        else:
            # Identity transformation for regression or already 0-based
            transformer = FunctionTransformer(validate=False)
            transformer.fit(data)
            return data.astype(np.float32), transformer

    @staticmethod
    def _handle_mixed_data(data: np.ndarray) -> Tuple[np.ndarray, TransformerMixin]:
        """
        Handle mixed or non-numeric data column by column.

        Args:
            data (np.ndarray): Mixed or non-numeric data array

        Returns:
            Tuple[np.ndarray, TransformerMixin]: Tuple of (numeric_data, transformer)
        """
        numeric = np.empty_like(data, dtype=np.float32)
        column_transformers: Dict[int, Optional[TransformerMixin]] = {}

        for col in range(data.shape[1]):
            col_data = data[:, col]

            if col_data.dtype.kind in {"U", "S", "O"}:  # String/object types
                encoder = FlexibleLabelEncoder()
                numeric[:, col] = encoder.fit_transform(col_data)
                column_transformers[col] = encoder
            else:
                # Try numeric conversion
                try:
                    numeric[:, col] = col_data.astype(np.float32)
                    column_transformers[col] = None
                except (ValueError, TypeError):
                    # Fallback to encoding
                    encoder = FlexibleLabelEncoder()
                    numeric[:, col] = encoder.fit_transform(col_data.astype(str))
                    column_transformers[col] = encoder

        # Wrap in a transformer that remembers column-wise logic
        transformer = ColumnWiseTransformer(column_transformers)
        return numeric, transformer


class ColumnWiseTransformer(TransformerMixin):
    """
    Applies different transformers to different columns.

    Args:
        column_transformers (dict): Maps column index to transformer (or None for identity)

    Attributes:
        column_transformers (dict): Stored column transformer mapping

    Examples:
    >>> transformers = {0: encoder1, 1: None, 2: encoder2}
    >>> transformer = ColumnWiseTransformer(transformers)
    >>> result = transformer.transform(data)
    """

    def __init__(self, column_transformers: Dict[int, Optional[TransformerMixin]]):
        """
        Initialize with column transformer mapping.

        Args:
            column_transformers (dict): Mapping from column index to transformer
        """
        self.column_transformers = column_transformers

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'ColumnWiseTransformer':
        """
        Fit transformer (no-op, already fitted).

        Args:
            X (np.ndarray): Input data (ignored)
            y (np.ndarray, optional): Target data (ignored)

        Returns:
            ColumnWiseTransformer: This instance
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using column-specific transformers.

        Args:
            X (np.ndarray): Data to transform

        Returns:
            np.ndarray: Transformed data as float32
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        result = np.empty_like(X, dtype=np.float32)
        for col, transformer in self.column_transformers.items():
            if transformer is None:
                result[:, col] = X[:, col].astype(np.float32)
            else:
                # Reshape to 2D for sklearn compatibility
                col_data = X[:, col:col+1]
                transformed = transformer.transform(col_data)
                result[:, col] = transformed.flatten()
        return result

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform data using column-specific transformers.

        Args:
            X (np.ndarray): Data to inverse transform

        Returns:
            np.ndarray: Original representation
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        result = np.empty(X.shape, dtype=object)
        for col, transformer in self.column_transformers.items():
            if transformer is None:
                result[:, col] = X[:, col]
            elif hasattr(transformer, 'inverse_transform'):
                # Reshape to 2D for sklearn compatibility
                col_data = X[:, col:col+1].astype(int)
                inv_transformed = transformer.inverse_transform(col_data)
                result[:, col] = inv_transformed.flatten()
            else:
                result[:, col] = X[:, col]
        return result
