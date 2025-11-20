"""Unit tests for FlexibleLabelEncoder."""

import numpy as np
import pytest

from nirs4all.data._targets.encoders import FlexibleLabelEncoder


class TestFlexibleLabelEncoder:
    """Test suite for FlexibleLabelEncoder class."""

    def test_flexible_encoder_fit_transform(self):
        """Test basic fit and transform functionality."""
        encoder = FlexibleLabelEncoder()
        data = np.array(['cat', 'dog', 'bird', 'cat', 'dog'])

        # Fit and transform
        result = encoder.fit_transform(data)

        # Check results
        assert result.dtype == np.float32
        assert len(encoder.classes_) == 3
        assert set(encoder.classes_) == {'cat', 'dog', 'bird'}

        # Check encoding is 0-based consecutive
        unique_encoded = np.unique(result)
        assert len(unique_encoded) == 3
        assert np.array_equal(unique_encoded, np.array([0., 1., 2.]))

    def test_flexible_encoder_unseen_labels(self):
        """Test handling of unseen labels during transform."""
        encoder = FlexibleLabelEncoder()
        train_data = np.array(['cat', 'dog', 'bird'])

        # Fit on training data
        encoder.fit(train_data)

        # Transform with unseen label
        test_data = np.array(['cat', 'dog', 'bird', 'fish', 'rabbit'])
        result = encoder.transform(test_data)

        # Check that known labels are mapped correctly
        assert result[0] == encoder.class_to_idx['cat']
        assert result[1] == encoder.class_to_idx['dog']
        assert result[2] == encoder.class_to_idx['bird']

        # Check that unseen labels get new indices
        assert result[3] >= len(encoder.classes_)  # fish gets 3
        assert result[4] >= len(encoder.classes_)  # rabbit gets 4
        assert result[3] != result[4]  # Different unseen labels get different indices

    def test_flexible_encoder_nan_handling(self):
        """Test that NaN values are preserved."""
        encoder = FlexibleLabelEncoder()
        data = np.array([1.0, 2.0, 3.0, np.nan, 2.0])

        # Fit and transform
        result = encoder.fit_transform(data)

        # Check that NaN is preserved
        assert np.isnan(result[3])

        # Check that other values are encoded
        assert not np.isnan(result[0])
        assert not np.isnan(result[1])
        assert not np.isnan(result[2])
        assert not np.isnan(result[4])

    def test_flexible_encoder_numeric_labels(self):
        """Test encoding of numeric labels."""
        encoder = FlexibleLabelEncoder()
        data = np.array([1, 2, 3, 1, 2, 3])

        result = encoder.fit_transform(data)

        assert result.dtype == np.float32
        assert len(encoder.classes_) == 3
        assert np.array_equal(np.unique(result), np.array([0., 1., 2.]))

    def test_flexible_encoder_transform_without_fit(self):
        """Test that transform without fit raises error."""
        encoder = FlexibleLabelEncoder()
        data = np.array(['a', 'b', 'c'])

        with pytest.raises(ValueError, match="Encoder must be fitted"):
            encoder.transform(data)

    def test_flexible_encoder_consistent_encoding(self):
        """Test that same labels get same encoding across calls."""
        encoder = FlexibleLabelEncoder()
        train_data = np.array(['x', 'y', 'z'])
        encoder.fit(train_data)

        test_data_1 = np.array(['x', 'y', 'z'])
        test_data_2 = np.array(['z', 'y', 'x'])

        result_1 = encoder.transform(test_data_1)
        result_2 = encoder.transform(test_data_2)

        # Same labels should get same encoding
        assert result_1[0] == result_2[2]  # Both 'x'
        assert result_1[1] == result_2[1]  # Both 'y'
        assert result_1[2] == result_2[0]  # Both 'z'

    def test_flexible_encoder_empty_array(self):
        """Test handling of empty arrays."""
        encoder = FlexibleLabelEncoder()
        data = np.array([])

        result = encoder.fit_transform(data)

        assert len(result) == 0
        assert len(encoder.classes_) == 0

    def test_flexible_encoder_single_class(self):
        """Test encoding with single class."""
        encoder = FlexibleLabelEncoder()
        data = np.array(['a', 'a', 'a'])

        result = encoder.fit_transform(data)

        assert len(encoder.classes_) == 1
        assert np.all(result == 0.0)
