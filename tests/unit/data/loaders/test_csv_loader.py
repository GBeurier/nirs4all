"""
Test CSV loader with header unit parameter - Step 3
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from nirs4all.data.loaders import load_csv


class TestCSVLoaderHeaderUnit:
    """Test header unit parameter in CSV loader"""

    def test_load_csv_returns_header_unit(self):
        """Test that load_csv returns header_unit"""
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("4000;5000;6000\n")
            f.write("1.1;1.2;1.3\n")
            f.write("2.1;2.2;2.3\n")
            temp_path = f.name

        try:
            data, report, na_mask, headers, unit = load_csv(
                temp_path,
                delimiter=';',
                has_header=True
            )

            assert unit == "cm-1"  # Default
            assert headers == ["4000", "5000", "6000"]
            assert data is not None
        finally:
            Path(temp_path).unlink()

    def test_load_csv_custom_header_unit(self):
        """Test loading CSV with custom header_unit"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("780;800;850\n")
            f.write("0.5;0.6;0.7\n")
            temp_path = f.name

        try:
            data, report, na_mask, headers, unit = load_csv(
                temp_path,
                delimiter=';',
                has_header=True,
                header_unit='nm'
            )

            assert unit == "nm"
            assert headers == ["780", "800", "850"]
        finally:
            Path(temp_path).unlink()

    def test_load_csv_no_headers_unit(self):
        """Test loading CSV without headers"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("1.1;1.2;1.3\n")
            f.write("2.1;2.2;2.3\n")
            temp_path = f.name

        try:
            data, report, na_mask, headers, unit = load_csv(
                temp_path,
                delimiter=';',
                has_header=False,
                header_unit='none'
            )

            assert unit == "none"
            # Headers will be auto-generated column indices
            assert len(headers) == 3
        finally:
            Path(temp_path).unlink()

    def test_load_csv_text_headers(self):
        """Test loading CSV with text headers"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("feature_A;feature_B;feature_C\n")
            f.write("1.1;1.2;1.3\n")
            f.write("2.1;2.2;2.3\n")
            temp_path = f.name

        try:
            data, report, na_mask, headers, unit = load_csv(
                temp_path,
                delimiter=';',
                has_header=True,
                header_unit='text'
            )

            assert unit == "text"
            assert headers == ["feature_A", "feature_B", "feature_C"]
        finally:
            Path(temp_path).unlink()

    def test_load_csv_index_unit(self):
        """Test loading CSV with index unit"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("0;1;2\n")
            f.write("1.1;1.2;1.3\n")
            temp_path = f.name

        try:
            data, report, na_mask, headers, unit = load_csv(
                temp_path,
                delimiter=';',
                has_header=True,
                header_unit='index'
            )

            assert unit == "index"
            assert headers == ["0", "1", "2"]
        finally:
            Path(temp_path).unlink()

    def test_load_csv_header_unit_persists_on_error(self):
        """Test that header_unit is returned even when file doesn't exist"""
        data, report, na_mask, headers, unit = load_csv(
            "nonexistent_file.csv",
            header_unit='nm'
        )

        assert unit == "nm"
        assert data is None
        assert 'error' in report

    def test_load_csv_default_header_unit_backward_compatible(self):
        """Test that existing code works without header_unit parameter"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("4000;5000\n")
            f.write("1.1;1.2\n")
            temp_path = f.name

        try:
            # Call without header_unit (should default to cm-1)
            result = load_csv(temp_path, delimiter=';', has_header=True)

            # Should return 5 values
            assert len(result) == 5
            data, report, na_mask, headers, unit = result

            assert unit == "cm-1"
            assert headers == ["4000", "5000"]
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
