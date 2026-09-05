"""
Unit tests for the Excel loader.

Tests loading .xlsx and .xls files with various configurations.
Requires openpyxl to be installed for .xlsx files.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
        result = loader.load(xlsx_with_na, na_policy="remove_sample")

        assert result.success
        assert result.data.shape == (2, 2)  # One row removed
        assert len(result.report["na_handling"]["removed_samples"]) == 1

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


@pytest.mark.parametrize("has_header", [True, False])
def test_shared_header_alias_keeps_every_excel_data_row(tmp_path, has_header):
    path = tmp_path / "header_alias.xlsx"
    values = pd.DataFrame({"first": [1.5, 2.5, 3.5], "second": [5.0, 6.0, 7.0]})
    values.to_excel(path, index=False, header=has_header)
    data, report, _, _, _ = load_excel(path, has_header=has_header)
    assert report["error"] is None
    np.testing.assert_array_equal(data.to_numpy(), values.to_numpy())


@pytest.mark.parametrize("options", [
    {"has_header": True, "header": None},
    {"has_header": False, "header": 0},
    {"has_header": "false"},
])
def test_conflicting_header_alias_fails_before_excel_read(tmp_path, monkeypatch, options):
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: pytest.fail("invalid options must fail before read"))
    result = ExcelLoader().load(tmp_path / "unopened.xlsx", **options)
    assert not result.success
    assert "has_header" in result.error


def test_header_alias_preserves_explicit_header_row_sheet_and_columns(tmp_path):
    path = tmp_path / "sheets.xlsx"
    values = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"other": [99]}).to_excel(writer, sheet_name="first", index=False)
        values.to_excel(writer, sheet_name="wanted", index=False, startrow=1)
    result = ExcelLoader().load(path, has_header=True, header=1, sheet_name="wanted", usecols=["b"])
    assert result.success
    np.testing.assert_array_equal(result.data.to_numpy(), values[["b"]].to_numpy())


def test_shared_numeric_and_metadata_options_are_not_forwarded_to_pandas(tmp_path):
    path = tmp_path / "metadata.xlsx"
    values = pd.DataFrame({"subject": ["alpha", "beta"], "measurement": ["1,25", "2,50"]})
    values.to_excel(path, index=False)
    result = ExcelLoader().load(path, data_type="metadata", categorical_mode="preserve",
                                has_header=True, delimiter=";", encoding="utf-8", decimal_separator=",")
    assert result.success
    assert result.data["subject"].tolist() == ["alpha", "beta"]
    np.testing.assert_array_equal(result.data["measurement"].to_numpy(), [1.25, 2.5])
    assert result.report["categorical_mode"] == "preserve"
    assert len(result.report["warnings"]) == 2
