"""
Unit tests for the Parquet loader.

Tests loading .parquet files with various configurations.
Requires pyarrow or fastparquet to be installed.
"""

import pytest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from nirs4all.data.loaders.parquet_loader import ParquetLoader, load_parquet


# Skip all tests if no Parquet engine is available
pytestmark = pytest.mark.skipif(
    not any([
        pytest.importorskip("pyarrow", reason="pyarrow not installed"),
    ]),
    reason="No Parquet engine available"
)


@pytest.fixture
def parquet_engine():
    """Get the available Parquet engine."""
    try:
        import pyarrow
        return "pyarrow"
    except ImportError:
        pass
    try:
        import fastparquet
        return "fastparquet"
    except ImportError:
        pass
    pytest.skip("No Parquet engine available")


class TestParquetLoaderSupports:
    """Tests for ParquetLoader.supports() method."""

    def test_supports_parquet(self):
        """Test that ParquetLoader supports .parquet files."""
        assert ParquetLoader.supports(Path("data.parquet"))
        assert ParquetLoader.supports(Path("data.PARQUET"))

    def test_supports_pq(self):
        """Test that ParquetLoader supports .pq files."""
        assert ParquetLoader.supports(Path("data.pq"))

    def test_not_supports_other(self):
        """Test that ParquetLoader doesn't support other formats."""
        assert not ParquetLoader.supports(Path("data.csv"))
        assert not ParquetLoader.supports(Path("data.xlsx"))


class TestParquetLoaderLoad:
    """Tests for ParquetLoader.load() method."""

    @pytest.fixture
    def sample_parquet_file(self, parquet_engine):
        """Create a temporary Parquet file."""
        df = pd.DataFrame({
            "feature_1": [1.0, 2.0, 3.0, 4.0],
            "feature_2": [5.0, 6.0, 7.0, 8.0],
            "feature_3": [9.0, 10.0, 11.0, 12.0],
        })
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, engine=parquet_engine)
            yield Path(f.name)
        Path(f.name).unlink()

    @pytest.fixture
    def parquet_with_na(self, parquet_engine):
        """Create a Parquet file with NA values."""
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": [4.0, 5.0, 6.0],
        })
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, engine=parquet_engine)
            yield Path(f.name)
        Path(f.name).unlink()

    def test_load_parquet(self, sample_parquet_file):
        """Test loading a Parquet file."""
        loader = ParquetLoader()
        result = loader.load(sample_parquet_file)

        assert result.success
        assert result.data is not None
        assert result.data.shape == (4, 3)
        assert list(result.data.columns) == ["feature_1", "feature_2", "feature_3"]

    def test_load_parquet_with_column_selection(self, sample_parquet_file):
        """Test loading specific columns from Parquet."""
        loader = ParquetLoader()
        result = loader.load(sample_parquet_file, columns=["feature_1", "feature_3"])

        assert result.success
        assert result.data.shape == (4, 2)
        assert list(result.data.columns) == ["feature_1", "feature_3"]

    def test_load_parquet_with_na(self, parquet_with_na):
        """Test that NA values are handled."""
        loader = ParquetLoader()
        result = loader.load(parquet_with_na, na_policy="remove_sample")

        assert result.success
        assert result.data.shape == (2, 2)  # One row removed
        assert len(result.report["na_handling"]["removed_samples"]) == 1

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        loader = ParquetLoader()
        result = loader.load(Path("/nonexistent/file.parquet"))

        assert not result.success
        assert "not found" in result.error.lower()

    def test_report_contains_engine(self, sample_parquet_file, parquet_engine):
        """Test that report contains engine info."""
        loader = ParquetLoader()
        result = loader.load(sample_parquet_file)

        assert result.report["engine"] == parquet_engine


class TestLoadParquetFunction:
    """Tests for the load_parquet convenience function."""

    @pytest.fixture
    def sample_parquet(self, parquet_engine):
        """Create a sample Parquet file."""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, engine=parquet_engine)
            yield Path(f.name)
        Path(f.name).unlink()

    def test_load_parquet_returns_tuple(self, sample_parquet):
        """Test that load_parquet returns expected tuple."""
        data, report, na_mask, headers, header_unit = load_parquet(sample_parquet)

        assert isinstance(data, pd.DataFrame)
        assert isinstance(report, dict)
        assert headers == ["a", "b"]
