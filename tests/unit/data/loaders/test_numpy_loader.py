"""
Unit tests for the NumPy loader.

Tests loading .npy and .npz files with various configurations.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nirs4all.data.loaders.numpy_loader import NumpyLoader, load_numpy


class TestNumpyLoader:
    """Tests for NumpyLoader class."""

    @pytest.fixture
    def sample_npy_file(self):
        """Create a temporary .npy file."""
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            np.save(f.name, arr)
            yield Path(f.name)
        Path(f.name).unlink()

    @pytest.fixture
    def sample_npz_file(self):
        """Create a temporary .npz file with multiple arrays."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            arr1 = np.array([[1.0, 2.0], [3.0, 4.0]])
            arr2 = np.array([[5.0, 6.0], [7.0, 8.0]])
            np.savez(f.name, X=arr1, Y=arr2)
            yield Path(f.name)
        Path(f.name).unlink()

    @pytest.fixture
    def sample_1d_npy_file(self):
        """Create a temporary .npy file with 1D array."""
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            np.save(f.name, arr)
            yield Path(f.name)
        Path(f.name).unlink()

    def test_supports_npy(self):
        """Test that NumpyLoader supports .npy files."""
        assert NumpyLoader.supports(Path("data.npy"))
        assert NumpyLoader.supports(Path("data.NPY"))

    def test_supports_npz(self):
        """Test that NumpyLoader supports .npz files."""
        assert NumpyLoader.supports(Path("data.npz"))
        assert NumpyLoader.supports(Path("data.NPZ"))

    def test_not_supports_other(self):
        """Test that NumpyLoader doesn't support other formats."""
        assert not NumpyLoader.supports(Path("data.csv"))
        assert not NumpyLoader.supports(Path("data.npy.gz"))

    def test_load_npy_file(self, sample_npy_file):
        """Test loading a .npy file."""
        loader = NumpyLoader()
        result = loader.load(sample_npy_file)

        assert result.success
        assert result.data is not None
        assert result.data.shape == (2, 3)
        np.testing.assert_array_almost_equal(
            result.data.values,
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        )

    def test_load_npz_file_first_array(self, sample_npz_file):
        """Test loading first array from .npz file."""
        loader = NumpyLoader()
        result = loader.load(sample_npz_file)

        assert result.success
        assert result.data is not None
        assert result.data.shape == (2, 2)
        # Should have loaded X array (first alphabetically or first stored)
        assert "X" in result.report.get("key_used", "") or "Y" in result.report.get("key_used", "")

    def test_load_npz_specific_key(self, sample_npz_file):
        """Test loading specific array from .npz file."""
        loader = NumpyLoader()
        result = loader.load(sample_npz_file, key="Y")

        assert result.success
        assert result.report["key_used"] == "Y"
        np.testing.assert_array_almost_equal(
            result.data.values,
            np.array([[5.0, 6.0], [7.0, 8.0]])
        )

    def test_load_npz_invalid_key(self, sample_npz_file):
        """Test loading with invalid key raises error."""
        loader = NumpyLoader()
        result = loader.load(sample_npz_file, key="invalid")

        assert not result.success
        assert "not found" in result.error.lower()

    def test_load_1d_array(self, sample_1d_npy_file):
        """Test that 1D arrays are reshaped to 2D."""
        loader = NumpyLoader()
        result = loader.load(sample_1d_npy_file)

        assert result.success
        assert result.data.shape == (5, 1)

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        loader = NumpyLoader()
        result = loader.load(Path("/nonexistent/file.npy"))

        assert not result.success
        assert "not found" in result.error.lower()

    def test_header_unit_index(self, sample_npy_file):
        """Test that index header_unit generates numeric headers."""
        loader = NumpyLoader()
        result = loader.load(sample_npy_file, header_unit="index")

        assert result.headers == ["0", "1", "2"]
        assert result.header_unit == "index"

    def test_header_unit_other(self, sample_npy_file):
        """Test that other header_unit generates feature_ headers."""
        loader = NumpyLoader()
        result = loader.load(sample_npy_file, header_unit="cm-1")

        assert result.headers == ["feature_0", "feature_1", "feature_2"]

    def test_report_contains_format_details(self, sample_npy_file):
        """Test that report contains format details."""
        loader = NumpyLoader()
        result = loader.load(sample_npy_file)

        assert result.report["format"] == "npy"
        assert "format_details" in result.report

class TestLoadNumpyFunction:
    """Tests for the load_numpy convenience function."""

    def test_load_numpy_returns_tuple(self):
        """Test that load_numpy returns the expected tuple."""
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            arr = np.array([[1.0, 2.0], [3.0, 4.0]])
            np.save(f.name, arr)

            data, report, na_mask, headers, header_unit = load_numpy(f.name)

            assert isinstance(data, pd.DataFrame)
            assert isinstance(report, dict)
            assert data.shape == (2, 2)

        Path(f.name).unlink()
