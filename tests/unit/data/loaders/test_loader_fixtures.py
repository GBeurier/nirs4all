"""
Tests for CSV loader fixtures defined in tests/unit/data/loaders/conftest.py.

These tests verify that the loader-specific fixtures work correctly
and create valid CSV files in various formats.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


class TestLoaderBaseData:
    """Tests for loader_base_data fixture."""

    def test_returns_tuple(self, loader_base_data):
        """Should return (X, y, wavelengths) tuple."""
        assert isinstance(loader_base_data, tuple)
        assert len(loader_base_data) == 3

    def test_correct_sample_count(self, loader_base_data):
        """Should have 50 samples."""
        X, y, wavelengths = loader_base_data
        assert X.shape[0] == 50
        assert y.shape[0] == 50

    def test_correct_wavelength_range(self, loader_base_data):
        """Should have wavelengths from 1000-2000nm."""
        _, _, wavelengths = loader_base_data
        assert wavelengths[0] == 1000
        assert wavelengths[-1] == 2000
        assert len(wavelengths) == 101  # 1000 to 2000, step 10


class TestCsvStandardFormat:
    """Tests for csv_standard_format fixture."""

    def test_folder_exists(self, csv_standard_format):
        """Folder should exist."""
        assert csv_standard_format.exists()
        assert csv_standard_format.is_dir()

    def test_contains_required_files(self, csv_standard_format):
        """Should contain Xcal, Ycal, Xval, Yval files."""
        files = list(csv_standard_format.glob("*.csv"))
        file_names = {f.stem for f in files}
        assert "Xcal" in file_names
        assert "Ycal" in file_names
        assert "Xval" in file_names
        assert "Yval" in file_names

    def test_xcal_has_correct_delimiter(self, csv_standard_format):
        """Xcal should use semicolon delimiter."""
        xcal = csv_standard_format / "Xcal.csv"
        with open(xcal, 'r') as f:
            first_line = f.readline()
        assert ';' in first_line


class TestCsvCommaDelimiter:
    """Tests for csv_comma_delimiter fixture."""

    def test_folder_exists(self, csv_comma_delimiter):
        """Folder should exist."""
        assert csv_comma_delimiter.exists()

    def test_uses_comma_delimiter(self, csv_comma_delimiter):
        """Files should use comma delimiter."""
        xcal = csv_comma_delimiter / "Xcal.csv"
        with open(xcal, 'r') as f:
            first_line = f.readline()
        # Should have commas but not semicolons (except if in data)
        assert ',' in first_line


class TestCsvTabDelimiter:
    """Tests for csv_tab_delimiter fixture."""

    def test_folder_exists(self, csv_tab_delimiter):
        """Folder should exist."""
        assert csv_tab_delimiter.exists()

    def test_uses_tsv_extension(self, csv_tab_delimiter):
        """Files should have .tsv extension."""
        tsv_files = list(csv_tab_delimiter.glob("*.tsv"))
        assert len(tsv_files) >= 4

    def test_uses_tab_delimiter(self, csv_tab_delimiter):
        """Files should use tab delimiter."""
        xcal = csv_tab_delimiter / "Xcal.tsv"
        with open(xcal, 'r') as f:
            first_line = f.readline()
        assert '\t' in first_line


class TestCsvNoHeaders:
    """Tests for csv_no_headers fixture."""

    def test_folder_exists(self, csv_no_headers):
        """Folder should exist."""
        assert csv_no_headers.exists()

    def test_first_line_is_data(self, csv_no_headers):
        """First line should be data, not headers."""
        xcal = csv_no_headers / "Xcal.csv"
        with open(xcal, 'r') as f:
            first_line = f.readline().strip()

        # First line should parse as numbers
        values = first_line.split(';')
        for val in values[:5]:  # Check first 5 values
            try:
                float(val)
            except ValueError:
                pytest.fail(f"First line contains non-numeric value: {val}")


class TestCsvWithIndex:
    """Tests for csv_with_index fixture."""

    def test_folder_exists(self, csv_with_index):
        """Folder should exist."""
        assert csv_with_index.exists()

    def test_has_index_column(self, csv_with_index):
        """Files should have an index column."""
        xcal = csv_with_index / "Xcal.csv"
        df = pd.read_csv(xcal, sep=';')
        # pandas adds Unnamed: 0 for index column
        assert 'Unnamed: 0' in df.columns or df.columns[0].isdigit() is False


class TestCsvSingleFile:
    """Tests for csv_single_file fixture."""

    def test_folder_exists(self, csv_single_file):
        """Folder should exist."""
        assert csv_single_file.exists()

    def test_contains_data_file(self, csv_single_file):
        """Should contain data.csv file."""
        data_file = csv_single_file / "data.csv"
        assert data_file.exists()

    def test_data_file_has_partition_column(self, csv_single_file):
        """data.csv should have partition column."""
        data_file = csv_single_file / "data.csv"
        df = pd.read_csv(data_file, sep=';')
        assert 'partition' in df.columns


class TestCsvFragmented:
    """Tests for csv_fragmented fixture."""

    def test_folder_exists(self, csv_fragmented):
        """Folder should exist."""
        assert csv_fragmented.exists()

    def test_has_train_folder(self, csv_fragmented):
        """Should have train subfolder."""
        train_folder = csv_fragmented / "train"
        assert train_folder.exists()

    def test_train_has_multiple_files(self, csv_fragmented):
        """Train folder should have multiple X files."""
        train_folder = csv_fragmented / "train"
        x_files = list(train_folder.glob("X_part*.csv"))
        assert len(x_files) >= 2  # Should be fragmented


class TestCsvPrecision:
    """Tests for precision variation fixtures."""

    def test_low_precision_folder_exists(self, csv_low_precision):
        """Low precision folder should exist."""
        assert csv_low_precision.exists()

    def test_high_precision_folder_exists(self, csv_high_precision):
        """High precision folder should exist."""
        assert csv_high_precision.exists()

    def test_low_precision_has_two_decimals(self, csv_low_precision):
        """Low precision should have ~2 decimal places."""
        xcal = csv_low_precision / "Xcal.csv"
        df = pd.read_csv(xcal, sep=';')

        # Check a sample value's decimal places
        sample_val = str(df.iloc[0, 0])
        if '.' in sample_val:
            decimals = len(sample_val.split('.')[1])
            assert decimals <= 3  # Allow some rounding variation


class TestCsvWithMissingValues:
    """Tests for csv_with_missing_values fixture."""

    def test_returns_tuple(self, csv_with_missing_values):
        """Should return (path, missing_ratio) tuple."""
        assert isinstance(csv_with_missing_values, tuple)
        assert len(csv_with_missing_values) == 2

    def test_path_exists(self, csv_with_missing_values):
        """Path should exist."""
        path, _ = csv_with_missing_values
        assert path.exists()

    def test_has_missing_values(self, csv_with_missing_values):
        """Data should contain NaN values."""
        path, missing_ratio = csv_with_missing_values
        xcal = path / "Xcal.csv"
        df = pd.read_csv(xcal, sep=';')
        # Should have some NaN values
        assert df.isna().any().any()


class TestCsvWithTextHeaders:
    """Tests for csv_with_text_headers fixture."""

    def test_path_exists(self, csv_with_text_headers):
        """Path should exist."""
        assert csv_with_text_headers.exists()

    def test_has_text_column_names(self, csv_with_text_headers):
        """Column names should be text (feature_0, etc.)."""
        xcal = csv_with_text_headers / "Xcal.csv"
        df = pd.read_csv(xcal, sep=';')
        # Check that columns are named feature_N
        assert df.columns[0].startswith('feature_')


class TestCsvEuropeanDecimals:
    """Tests for csv_european_decimals fixture."""

    def test_path_exists(self, csv_european_decimals):
        """Path should exist."""
        assert csv_european_decimals.exists()

    def test_uses_comma_as_decimal(self, csv_european_decimals):
        """Should use comma as decimal separator."""
        xcal = csv_european_decimals / "Xcal.csv"
        with open(xcal, 'r') as f:
            content = f.read()

        # Should contain commas within numbers (as decimal separators)
        # e.g., "1,234;5,678" format
        lines = content.strip().split('\n')
        if len(lines) > 1:  # Skip header
            data_line = lines[1]
            # Check that values use comma (European decimal format)
            # Values are separated by semicolons
            values = data_line.split(';')
            for val in values[:3]:
                assert ',' in val, f"Expected comma decimal in {val}"


class TestCsvMultiSource:
    """Tests for csv_multi_source fixture."""

    def test_path_exists(self, csv_multi_source):
        """Path should exist."""
        assert csv_multi_source.exists()

    def test_has_multiple_x_files(self, csv_multi_source):
        """Should have multiple Xcal files."""
        xcal_files = list(csv_multi_source.glob("Xcal_*.csv"))
        assert len(xcal_files) == 3  # n_files=3

    def test_has_single_y_file(self, csv_multi_source):
        """Should have single Ycal file."""
        ycal_files = list(csv_multi_source.glob("Ycal*.csv"))
        # Should have exactly one Ycal file (not Ycal_1, Ycal_2, etc.)
        assert any(f.name == "Ycal.csv" for f in ycal_files)


class TestCsvSingleFileAllData:
    """Tests for csv_single_file_all_data fixture."""

    def test_path_exists(self, csv_single_file_all_data):
        """Path should exist."""
        assert csv_single_file_all_data.exists()

    def test_has_data_csv(self, csv_single_file_all_data):
        """Should have data.csv file."""
        data_file = csv_single_file_all_data / "data.csv"
        assert data_file.exists()

    def test_has_metadata_columns(self, csv_single_file_all_data):
        """data.csv should have metadata columns."""
        data_file = csv_single_file_all_data / "data.csv"
        df = pd.read_csv(data_file, sep=';')

        assert 'sample_id' in df.columns
        assert 'partition' in df.columns
        assert 'group' in df.columns
        assert 'target' in df.columns


class TestCsvBasicVariations:
    """Tests for csv_basic_variations parametrized fixture."""

    def test_path_exists(self, csv_basic_variations):
        """Path should exist for all variations."""
        path, info = csv_basic_variations
        assert path.exists()

    def test_returns_format_info(self, csv_basic_variations):
        """Should return format info dictionary."""
        path, info = csv_basic_variations
        assert isinstance(info, dict)
        assert 'delimiter' in info
        assert 'has_header' in info


class TestCsvFileStructures:
    """Tests for csv_file_structures parametrized fixture."""

    def test_path_exists(self, csv_file_structures):
        """Path should exist for all structures."""
        path, structure_type = csv_file_structures
        assert path.exists()

    def test_returns_structure_type(self, csv_file_structures):
        """Should return structure type string."""
        path, structure_type = csv_file_structures
        assert structure_type in ['standard', 'single_file', 'fragmented']


class TestCsvPrecisionLevels:
    """Tests for csv_precision_levels parametrized fixture."""

    def test_path_exists(self, csv_precision_levels):
        """Path should exist for all precision levels."""
        path, precision = csv_precision_levels
        assert path.exists()

    def test_returns_precision_value(self, csv_precision_levels):
        """Should return precision integer."""
        path, precision = csv_precision_levels
        assert precision in [2, 6, 10]


class TestCsvAllLoaderVariations:
    """Tests for csv_all_loader_variations fixture."""

    def test_returns_dict(self, csv_all_loader_variations):
        """Should return dictionary."""
        assert isinstance(csv_all_loader_variations, dict)

    def test_has_expected_variations(self, csv_all_loader_variations):
        """Should have expected variation names."""
        expected = [
            'standard_semicolon',
            'comma_separated',
            'tab_separated',
            'no_headers',
            'with_index',
        ]
        for name in expected:
            assert name in csv_all_loader_variations, f"Missing variation: {name}"

    def test_all_paths_exist(self, csv_all_loader_variations):
        """All paths should exist."""
        for name, path in csv_all_loader_variations.items():
            assert path.exists(), f"Path for {name} does not exist"


class TestLoaderCsvGenerator:
    """Tests for loader_csv_generator fixture."""

    def test_returns_generator(self, loader_csv_generator):
        """Should return CSVVariationGenerator instance."""
        assert loader_csv_generator is not None
        assert hasattr(loader_csv_generator, 'generate_all_variations')

    def test_has_all_methods(self, loader_csv_generator):
        """Should have all expected methods."""
        expected_methods = [
            'with_semicolon_delimiter',
            'with_comma_delimiter',
            'with_tab_delimiter',
            'without_headers',
            'with_row_index',
            'as_single_file',
            'as_fragmented',
            'with_precision',
        ]
        for method in expected_methods:
            assert hasattr(loader_csv_generator, method), f"Missing method: {method}"
