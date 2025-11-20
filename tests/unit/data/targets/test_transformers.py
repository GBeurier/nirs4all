"""Unit tests for TargetTransformer."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from nirs4all.data._targets.processing_chain import ProcessingChain
from nirs4all.data._targets.transformers import TargetTransformer


class TestTargetTransformer:
    """Test suite for TargetTransformer class."""

    def test_transform_forward(self):
        """Test forward transformation through processing chain."""
        chain = ProcessingChain()
        transformer_obj = TargetTransformer(chain)

        scaler1 = StandardScaler()
        scaler2 = StandardScaler()

        # Fit scalers with some data
        data = np.array([[1], [2], [3], [4], [5]], dtype=np.float32)
        scaler1.fit(data)
        scaler2.fit(scaler1.transform(data))

        # Build chain
        chain.add_processing("raw", ancestor=None, transformer=None)
        chain.add_processing("scaled", ancestor="raw", transformer=scaler1)
        chain.add_processing("normalized", ancestor="scaled", transformer=scaler2)

        # Transform from raw to normalized
        test_data = np.array([[3]], dtype=np.float32)

        # Create data storage
        data_storage = {
            "raw": test_data,
            "scaled": scaler1.transform(test_data),
            "normalized": scaler2.transform(scaler1.transform(test_data))
        }

        result = transformer_obj.transform(test_data, "raw", "normalized", data_storage)

        # Should apply both scalers
        expected = scaler2.transform(scaler1.transform(test_data))
        assert np.allclose(result, expected)

    def test_transform_inverse(self):
        """Test inverse transformation through processing chain."""
        chain = ProcessingChain()
        transformer_obj = TargetTransformer(chain)

        scaler1 = StandardScaler()
        scaler2 = StandardScaler()

        # Fit scalers
        data = np.array([[1], [2], [3], [4], [5]], dtype=np.float32)
        scaler1.fit(data)
        scaler2.fit(scaler1.transform(data))

        # Build chain
        chain.add_processing("raw", ancestor=None, transformer=None)
        chain.add_processing("scaled", ancestor="raw", transformer=scaler1)
        chain.add_processing("normalized", ancestor="scaled", transformer=scaler2)

        # Transform from normalized back to raw
        test_data = np.array([[0]], dtype=np.float32)

        # Create data storage
        data_storage = {
            "raw": np.array([[3]], dtype=np.float32),
            "scaled": np.array([[0]], dtype=np.float32),
            "normalized": test_data
        }

        result = transformer_obj.transform(test_data, "normalized", "raw", data_storage)

        # Should apply inverse of both scalers
        expected = scaler1.inverse_transform(scaler2.inverse_transform(test_data))
        assert np.allclose(result, expected)

    def test_transform_no_path(self):
        """Test transform when no path exists."""
        chain = ProcessingChain()
        transformer_obj = TargetTransformer(chain)

        scaler1 = StandardScaler()
        scaler2 = StandardScaler()

        data = np.array([[1], [2], [3]], dtype=np.float32)
        scaler1.fit(data)
        scaler2.fit(data)

        # Create two separate trees
        chain.add_processing("raw1", ancestor=None, transformer=None)
        chain.add_processing("scaled1", ancestor="raw1", transformer=scaler1)
        chain.add_processing("raw2", ancestor=None, transformer=None)
        chain.add_processing("scaled2", ancestor="raw2", transformer=scaler2)

        # No path between separate trees
        test_data = np.array([[1]], dtype=np.float32)
        data_storage = {"scaled1": test_data, "scaled2": test_data}

        with pytest.raises(ValueError, match="No common ancestor"):
            transformer_obj.transform(test_data, "scaled1", "scaled2", data_storage)

    def test_transform_same_state(self):
        """Test transform when source and target are the same."""
        chain = ProcessingChain()
        transformer_obj = TargetTransformer(chain)

        scaler = StandardScaler()

        data = np.array([[1], [2], [3]], dtype=np.float32)
        scaler.fit(data)

        chain.add_processing("raw", ancestor=None, transformer=None)
        chain.add_processing("scaled", ancestor="raw", transformer=scaler)

        # Transform from state to itself
        test_data = np.array([[2]], dtype=np.float32)
        data_storage = {"raw": test_data, "scaled": scaler.transform(test_data)}

        result = transformer_obj.transform(test_data, "raw", "raw", data_storage)

        # Should return unchanged
        assert np.array_equal(result, test_data)

    def test_transform_preserves_shape(self):
        """Test that transformation preserves data shape."""
        chain = ProcessingChain()
        transformer_obj = TargetTransformer(chain)

        scaler = StandardScaler()

        data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        scaler.fit(data)

        chain.add_processing("raw", ancestor=None, transformer=None)
        chain.add_processing("scaled", ancestor="raw", transformer=scaler)

        test_data = np.array([[2, 3]], dtype=np.float32)
        data_storage = {"raw": test_data, "scaled": scaler.transform(test_data)}

        result = transformer_obj.transform(test_data, "raw", "scaled", data_storage)

        assert result.shape == test_data.shape

    def test_transform_roundtrip(self):
        """Test forward and inverse transformation roundtrip."""
        chain = ProcessingChain()
        transformer_obj = TargetTransformer(chain)

        scaler = StandardScaler()

        data = np.array([[1], [2], [3], [4], [5]], dtype=np.float32)
        scaler.fit(data)

        chain.add_processing("raw", ancestor=None, transformer=None)
        chain.add_processing("scaled", ancestor="raw", transformer=scaler)

        # Forward then inverse
        original_data = np.array([[3]], dtype=np.float32)
        scaled_data = scaler.transform(original_data)
        data_storage = {"raw": original_data, "scaled": scaled_data}

        scaled = transformer_obj.transform(original_data, "raw", "scaled", data_storage)
        restored = transformer_obj.transform(scaled, "scaled", "raw", data_storage)

        assert np.allclose(restored, original_data)
