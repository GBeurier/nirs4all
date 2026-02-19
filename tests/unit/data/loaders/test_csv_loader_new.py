"""
Unit tests for the new CSVLoader class.

Tests the refactored CSV loader with the FileLoader interface.
"""

import gzip
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nirs4all.data.loaders.csv_loader_new import CSVLoader, load_csv


class TestCSVLoaderSupports:
    """Tests for CSVLoader.supports() method."""

    def test_supports_csv(self):
        """Test that CSVLoader supports .csv files."""
        assert CSVLoader.supports(Path("data.csv"))
        assert CSVLoader.supports(Path("data.CSV"))

    def test_supports_csv_gz(self):
        """Test that CSVLoader supports .csv.gz files."""
        assert CSVLoader.supports(Path("data.csv.gz"))

    def test_supports_csv_zip(self):
        """Test that CSVLoader supports .csv.zip files."""
        assert CSVLoader.supports(Path("data.csv.zip"))

    def test_not_supports_other(self):
        """Test that CSVLoader doesn't support other formats."""
        assert not CSVLoader.supports(Path("data.xlsx"))
        assert not CSVLoader.supports(Path("data.parquet"))
        assert not CSVLoader.supports(Path("data.npy"))

class TestCSVLoaderLoad:
    """Tests for CSVLoader.load() method."""

    @pytest.fixture
    def simple_csv_file(self):
        """Create a simple CSV file."""
        content = "a;b;c\n1.0;2.0;3.0\n4.0;5.0;6.0"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)
        yield path
        path.unlink()

    @pytest.fixture
    def csv_with_na_file(self):
        """Create a CSV file with NA values."""
        content = "a;b;c\n1.0;2.0;3.0\n4.0;;6.0\n7.0;8.0;9.0"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)
        yield path
        path.unlink()

    @pytest.fixture
    def csv_gz_file(self):
        """Create a gzipped CSV file."""
        content = b"a;b;c\n1.0;2.0;3.0\n4.0;5.0;6.0"
        with tempfile.NamedTemporaryFile(suffix=".csv.gz", delete=False) as f:
            path = Path(f.name)
        with gzip.open(path, "wb") as gz:
            gz.write(content)
        yield path
        path.unlink()

    @pytest.fixture
    def csv_zip_file(self):
        """Create a zipped CSV file."""
        content = "a;b;c\n1.0;2.0;3.0\n4.0;5.0;6.0"
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            path = Path(f.name)
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("data.csv", content)
        yield path
        path.unlink()

    def test_load_simple_csv(self, simple_csv_file):
        """Test loading a simple CSV file."""
        loader = CSVLoader()
        result = loader.load(simple_csv_file)

        assert result.success
        assert result.data is not None
        assert result.data.shape == (2, 3)
        assert list(result.data.columns) == ["a", "b", "c"]

    def test_load_csv_with_na_remove(self, csv_with_na_file):
        """Test loading CSV with NA values using remove policy."""
        loader = CSVLoader()
        result = loader.load(csv_with_na_file, na_policy="remove_sample")

        assert result.success
        assert result.data.shape == (2, 3)  # One row removed
        assert len(result.report["na_handling"]["removed_samples"]) == 1

    def test_load_csv_with_na_abort(self, csv_with_na_file):
        """Test loading CSV with NA values using abort policy."""
        loader = CSVLoader()
        result = loader.load(csv_with_na_file, na_policy="abort")

        assert not result.success
        assert "na_policy is 'abort'" in result.error.lower() or "na" in result.error.lower()

    def test_load_csv_gz(self, csv_gz_file):
        """Test loading a gzipped CSV file."""
        loader = CSVLoader()
        result = loader.load(csv_gz_file)

        assert result.success
        assert result.data.shape == (2, 3)

    def test_load_csv_zip(self, csv_zip_file):
        """Test loading a zipped CSV file."""
        loader = CSVLoader()
        result = loader.load(csv_zip_file)

        assert result.success
        assert result.data.shape == (2, 3)

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        loader = CSVLoader()
        result = loader.load(Path("/nonexistent/file.csv"))

        assert not result.success
        assert result.error is not None

    def test_load_with_custom_delimiter(self):
        """Test loading CSV with custom delimiter."""
        content = "a,b,c\n1.0,2.0,3.0"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
            path = Path(f.name)

        try:
            loader = CSVLoader()
            result = loader.load(path, delimiter=",")

            assert result.success
            assert list(result.data.columns) == ["a", "b", "c"]
        finally:
            path.unlink()

    def test_load_without_header(self):
        """Test loading CSV without header."""
        content = "1.0;2.0;3.0\n4.0;5.0;6.0"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
            path = Path(f.name)

        try:
            loader = CSVLoader()
            result = loader.load(path, has_header=False)

            assert result.success
            assert result.data.shape == (2, 3)
        finally:
            path.unlink()

    def test_header_unit_preserved(self, simple_csv_file):
        """Test that header_unit is preserved in result."""
        loader = CSVLoader()
        result = loader.load(simple_csv_file, header_unit="nm")

        assert result.header_unit == "nm"

    def test_report_contains_detection_params(self, simple_csv_file):
        """Test that report contains detection parameters."""
        loader = CSVLoader()
        result = loader.load(simple_csv_file)

        assert "detection_params" in result.report
        assert result.report["delimiter"] == ";"
        assert result.report["has_header"] is True

class TestLoadCsvFunction:
    """Tests for the load_csv convenience function."""

    def test_load_csv_returns_tuple(self):
        """Test that load_csv returns expected tuple format."""
        content = "a;b\n1.0;2.0"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            data, report, na_mask, headers, header_unit = load_csv(path)

            assert isinstance(data, pd.DataFrame)
            assert isinstance(report, dict)
            assert headers == ["a", "b"]
        finally:
            path.unlink()

class TestCSVLoaderDataTypes:
    """Tests for data type handling in CSVLoader."""

    def test_load_y_data_categorical_auto(self):
        """Test loading Y data with categorical auto-detection."""
        content = "class\nA\nB\nA\nC"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            loader = CSVLoader()
            result = loader.load(path, data_type="y", categorical_mode="auto")

            assert result.success
            assert "categorical_info" in result.report
        finally:
            path.unlink()

    def test_load_x_data_numeric_conversion(self):
        """Test that X data is converted to numeric."""
        content = "a;b\n1.0;2.0\n3.0;text"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            loader = CSVLoader()
            result = loader.load(path, data_type="x", na_policy="remove_sample")

            assert result.success
            # Second row should be removed due to 'text' becoming NaN
            assert len(result.report["na_handling"]["removed_samples"]) == 1
        finally:
            path.unlink()
