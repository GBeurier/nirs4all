"""
Unit tests for the Excel loader.

Tests loading .xlsx and .xls files with various configurations.
Requires openpyxl to be installed for .xlsx files.
"""

import pytest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from nirs4all.data.loaders.excel_loader import ExcelLoader, load_excel


# Check if openpyxl is available
try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False


class TestExcelLoaderSupports:
    """Tests for ExcelLoader.supports() method."""

    def test_supports_xlsx(self):
        """Test that ExcelLoader supports .xlsx files."""
        assert ExcelLoader.supports(Path("data.xlsx"))
        assert ExcelLoader.supports(Path("data.XLSX"))

    def test_supports_xls(self):
        """Test that ExcelLoader supports .xls files."""
        assert ExcelLoader.supports(Path("data.xls"))

    def test_not_supports_other(self):
        """Test that ExcelLoader doesn't support other formats."""
        assert not ExcelLoader.supports(Path("data.csv"))
        assert not ExcelLoader.supports(Path("data.ods"))


@pytest.mark.skipif(not HAS_OPENPYXL, reason="openpyxl not installed")
class TestExcelLoaderLoad:
    """Tests for ExcelLoader.load() method."""

    @pytest.fixture
    def simple_xlsx_file(self):
        """Create a simple Excel file."""
        df = pd.DataFrame({
            "feature_1": [1.0, 2.0, 3.0],
            "feature_2": [4.0, 5.0, 6.0],
        })
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            df.to_excel(f.name, index=False, engine="openpyxl")
            yield Path(f.name)
        Path(f.name).unlink()

    @pytest.fixture
    def multi_sheet_xlsx(self):
        """Create an Excel file with multiple sheets."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            with pd.ExcelWriter(f.name, engine="openpyxl") as writer:
                pd.DataFrame({"a": [1, 2]}).to_excel(writer, sheet_name="Sheet1", index=False)
                pd.DataFrame({"b": [3, 4]}).to_excel(writer, sheet_name="Sheet2", index=False)
            yield Path(f.name)
        Path(f.name).unlink()

    @pytest.fixture
    def xlsx_with_na(self):
        """Create an Excel file with NA values."""
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": [4.0, 5.0, 6.0],
        })
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            df.to_excel(f.name, index=False, engine="openpyxl")
            yield Path(f.name)
        Path(f.name).unlink()

    def test_load_xlsx(self, simple_xlsx_file):
        """Test loading an Excel file."""
        loader = ExcelLoader()
        result = loader.load(simple_xlsx_file)

        assert result.success
        assert result.data is not None
        assert result.data.shape == (3, 2)
        assert list(result.data.columns) == ["feature_1", "feature_2"]

    def test_load_specific_sheet_by_name(self, multi_sheet_xlsx):
        """Test loading a specific sheet by name."""
        loader = ExcelLoader()
        result = loader.load(multi_sheet_xlsx, sheet_name="Sheet2")

        assert result.success
        assert list(result.data.columns) == ["b"]

    def test_load_specific_sheet_by_index(self, multi_sheet_xlsx):
        """Test loading a specific sheet by index."""
        loader = ExcelLoader()
        result = loader.load(multi_sheet_xlsx, sheet_name=1)

        assert result.success
        assert list(result.data.columns) == ["b"]

    def test_load_xlsx_with_na(self, xlsx_with_na):
        """Test that NA values are handled."""
        loader = ExcelLoader()
        result = loader.load(xlsx_with_na)

        assert result.success
        assert result.data.shape == (2, 2)  # One row removed
        assert result.report["na_handling"]["nb_removed_rows"] == 1

    def test_load_with_usecols(self, simple_xlsx_file):
        """Test loading specific columns."""
        loader = ExcelLoader()
        result = loader.load(simple_xlsx_file, usecols=["feature_1"])

        assert result.success
        assert result.data.shape == (3, 1)
        assert list(result.data.columns) == ["feature_1"]

    def test_load_with_skip_rows(self):
        """Test loading with skip_rows parameter."""
        # Create file with header rows to skip
        df = pd.DataFrame({
            "feature_1": [1.0, 2.0],
            "feature_2": [3.0, 4.0],
        })
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            df.to_excel(f.name, index=False, engine="openpyxl")
            path = Path(f.name)

        try:
            loader = ExcelLoader()
            result = loader.load(path, skip_rows=1)

            assert result.success
            # One row should be skipped
            assert result.data.shape[0] == 1
        finally:
            path.unlink()

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        loader = ExcelLoader()
        result = loader.load(Path("/nonexistent/file.xlsx"))

        assert not result.success
        assert "not found" in result.error.lower()

    def test_report_contains_engine(self, simple_xlsx_file):
        """Test that report contains engine info."""
        loader = ExcelLoader()
        result = loader.load(simple_xlsx_file)

        assert result.report["engine"] == "openpyxl"


@pytest.mark.skipif(not HAS_OPENPYXL, reason="openpyxl not installed")
class TestLoadExcelFunction:
    """Tests for the load_excel convenience function."""

    @pytest.fixture
    def sample_xlsx(self):
        """Create a sample Excel file."""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            df.to_excel(f.name, index=False, engine="openpyxl")
            yield Path(f.name)
        Path(f.name).unlink()

    def test_load_excel_returns_tuple(self, sample_xlsx):
        """Test that load_excel returns expected tuple."""
        data, report, na_mask, headers, header_unit = load_excel(sample_xlsx)

        assert isinstance(data, pd.DataFrame)
        assert isinstance(report, dict)
        assert headers == ["a", "b"]
