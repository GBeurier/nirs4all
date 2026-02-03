"""
Unit tests for the MATLAB loader.

Tests loading .mat files with various configurations.
Requires scipy to be installed.
"""

import pytest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from nirs4all.data.loaders.matlab_loader import MatlabLoader, load_matlab


# Check if scipy is available
try:
    import scipy.io
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class TestMatlabLoaderSupports:
    """Tests for MatlabLoader.supports() method."""

    def test_supports_mat(self):
        """Test that MatlabLoader supports .mat files."""
        assert MatlabLoader.supports(Path("data.mat"))
        assert MatlabLoader.supports(Path("data.MAT"))

    def test_not_supports_other(self):
        """Test that MatlabLoader doesn't support other formats."""
        assert not MatlabLoader.supports(Path("data.csv"))
        assert not MatlabLoader.supports(Path("data.npy"))


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
class TestMatlabLoaderLoad:
    """Tests for MatlabLoader.load() method."""

    @pytest.fixture
    def simple_mat_file(self):
        """Create a simple MATLAB file."""
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            scipy.io.savemat(f.name, {"X": X})
            yield Path(f.name)
        Path(f.name).unlink()

    @pytest.fixture
    def multi_var_mat_file(self):
        """Create a MATLAB file with multiple variables."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        Y = np.array([[5.0, 6.0], [7.0, 8.0]])
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            scipy.io.savemat(f.name, {"X": X, "Y": Y})
            yield Path(f.name)
        Path(f.name).unlink()

    @pytest.fixture
    def mat_with_nan(self):
        """Create a MATLAB file with NaN values."""
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            scipy.io.savemat(f.name, {"X": X})
            yield Path(f.name)
        Path(f.name).unlink()

    @pytest.fixture
    def mat_1d_array(self):
        """Create a MATLAB file with 1D array."""
        vec = np.array([1.0, 2.0, 3.0, 4.0])
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            scipy.io.savemat(f.name, {"data": vec})
            yield Path(f.name)
        Path(f.name).unlink()

    def test_load_mat(self, simple_mat_file):
        """Test loading a MATLAB file."""
        loader = MatlabLoader()
        result = loader.load(simple_mat_file)

        assert result.success
        assert result.data is not None
        assert result.data.shape == (2, 3)

    def test_load_specific_variable(self, multi_var_mat_file):
        """Test loading a specific variable."""
        loader = MatlabLoader()
        result = loader.load(multi_var_mat_file, variable="Y")

        assert result.success
        assert result.report["variable_used"] == "Y"
        np.testing.assert_array_almost_equal(
            result.data.values,
            np.array([[5.0, 6.0], [7.0, 8.0]])
        )

    def test_load_invalid_variable(self, simple_mat_file):
        """Test loading with invalid variable name."""
        loader = MatlabLoader()
        result = loader.load(simple_mat_file, variable="invalid")

        assert not result.success
        assert "not found" in result.error.lower()

    def test_load_mat_with_nan(self, mat_with_nan):
        """Test that NaN values are handled."""
        loader = MatlabLoader()
        result = loader.load(mat_with_nan, na_policy="remove_sample")

        assert result.success
        assert result.data.shape == (1, 2)  # One row removed
        assert len(result.report["na_handling"]["removed_samples"]) == 1

    def test_load_1d_array(self, mat_1d_array):
        """Test that 1D arrays are reshaped to 2D."""
        loader = MatlabLoader()
        result = loader.load(mat_1d_array)

        assert result.success
        assert result.data.shape == (4, 1)

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        loader = MatlabLoader()
        result = loader.load(Path("/nonexistent/file.mat"))

        assert not result.success
        assert "not found" in result.error.lower()

    def test_report_contains_variables(self, multi_var_mat_file):
        """Test that report contains available variables."""
        loader = MatlabLoader()
        result = loader.load(multi_var_mat_file)

        assert "X" in result.report["variables_available"]
        assert "Y" in result.report["variables_available"]

    def test_auto_select_x_variable(self, multi_var_mat_file):
        """Test that X variable is auto-selected when available."""
        loader = MatlabLoader()
        result = loader.load(multi_var_mat_file)

        assert result.success
        assert result.report["variable_used"] == "X"


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
class TestLoadMatlabFunction:
    """Tests for the load_matlab convenience function."""

    @pytest.fixture
    def sample_mat(self):
        """Create a sample MATLAB file."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            scipy.io.savemat(f.name, {"X": X})
            yield Path(f.name)
        Path(f.name).unlink()

    def test_load_matlab_returns_tuple(self, sample_mat):
        """Test that load_matlab returns expected tuple."""
        data, report, na_mask, headers, header_unit = load_matlab(sample_mat)

        assert isinstance(data, pd.DataFrame)
        assert isinstance(report, dict)
        assert data.shape == (2, 2)
