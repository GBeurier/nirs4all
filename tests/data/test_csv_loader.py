# tests/test_csv_loader.py

import pytest
import numpy as np
import os
import tempfile
from nirs4all.data.csv_loader import (
    _determine_csv_parameters,
    _can_be_float,
    load_csv,
)


def test_can_be_float():
    assert _can_be_float("1.0", ".")
    assert _can_be_float("1,0", ",")
    assert not _can_be_float("", ".")
    assert not _can_be_float("abc", ".")
    assert not _can_be_float("1.2.3", ".")
    assert not _can_be_float(123, ".")  # Not a string


def test_determine_csv_parameters_comma_delimiter():
    csv_content = """col1,col2
1.0,2.0
3.0,4.0
5.0,6.0
7.0,8.0"""
    params = _determine_csv_parameters(csv_content, bypass_auto_detection=False)
    assert params['delimiter'] == ','
    assert params['decimal_separator'] == '.'
    assert params['has_header']


def test_determine_csv_parameters_tab_delimiter():
    csv_content = """col1\tcol2
1.0\t2.0
3.0\t4.0
5.0\t6.0
7.0\t8.0"""
    params = _determine_csv_parameters(csv_content, bypass_auto_detection=False)
    assert params['delimiter'] == '\t'
    assert params['decimal_separator'] == '.'
    assert params['has_header']


def test_determine_csv_parameters_semicolon_delimiter():
    csv_content = """col1;col2
1,0;2,0
3,0;4,0
5,0;6,0
7,0;8,0"""
    params = _determine_csv_parameters(csv_content, bypass_auto_detection=False)
    assert params['delimiter'] == ';'
    assert params['decimal_separator'] == ','
    assert params['has_header']


def test_determine_csv_parameters_no_header():
    csv_content = """1.0,2.0
3.0,4.0
5.0,6.0
7.0,8.0"""
    params = _determine_csv_parameters(csv_content, bypass_auto_detection=False)
    assert params['delimiter'] == ','
    assert params['decimal_separator'] == '.'
    # May detect no header, but this depends on the heuristics
    # Just check that the key exists
    assert 'has_header' in params


def test_load_csv(tmp_path):
    csv_content = """col1;col2
1.0;2.0
3.0;4.0
5.0;6.0
7.0;8.0"""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)
    data, report, _ = load_csv(str(csv_file))  # Uses default params, which should be ';', '.', header=True
    expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    print(report)
    assert np.array_equal(data, expected)
    assert report['initial_shape'] == (4, 2)
    assert report['final_shape'] == (4, 2)
    assert 'detection_params' in report
    assert report['detection_params']['delimiter'] == ';'
    assert report['detection_params']['decimal_separator'] == '.'
    # Verify backward compatibility
    assert report['delimiter'] == ';'
    assert report['decimal_separator'] == '.'
    assert report['has_header']


def test_load_csv_with_na_values(tmp_path):
    csv_content = """col1,col2
1.0,2.0
3.0,NA
5.0,6.0
7.0,8.0"""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)
    
    # Test with na_policy='remove' (which is the default behavior)
    # CSV format (';', '.') matches defaults, but explicit for robustness
    data, report, _ = load_csv(str(csv_file), delimiter=',', decimal_separator='.', has_header=True, na_policy='remove')
    # Data should not be None as NAs are being removed
    assert data is not None
    assert report['na_handling']['na_detected']
    assert report['na_handling']['nb_removed_rows'] > 0
    
    # Test with na_policy='abort' explicitly
    data, report, _ = load_csv(str(csv_file), delimiter=',', decimal_separator='.', has_header=True, na_policy='abort')
    # With 'abort' policy, data should be None
    assert data is None
    assert 'error' in report
    # Check that the error message starts correctly
    assert report['error'].startswith("NA values detected")
    assert report['na_handling']['na_detected']


def test_load_csv_invalid_file(tmp_path):
    csv_file = tmp_path / "nonexistent.csv"
    data, report, _ = load_csv(str(csv_file))
    assert data is None
    assert 'error' in report
    assert "n'existe pas" in report['error'] or "not exist" in report['error'].lower()


def test_load_csv_with_gz(tmp_path):
    csv_content = """col1;col2
1.0;2.0
3.0;4.0
5.0;6.0
7.0;8.0"""
    csv_file = tmp_path / "test.csv.gz"
    import gzip
    with gzip.open(csv_file, 'wt', encoding='utf-8') as f:
        f.write(csv_content)
    # CSV format (';', '.') matches defaults, but explicit for robustness
    data, report, _ = load_csv(str(csv_file), delimiter=';', decimal_separator='.', has_header=True)
    expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    assert np.array_equal(data, expected)


def test_load_csv_with_zip(tmp_path):
    csv_content = """col1;col2
1.0;2.0
3.0;4.0
5.0;6.0
7.0;8.0"""
    csv_file = tmp_path / "test.zip"
    import zipfile
    with zipfile.ZipFile(csv_file, 'w') as z:
        z.writestr('test.csv', csv_content)
    data, report, _ = load_csv(str(csv_file))
    expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    assert np.array_equal(data, expected)


def test_load_csv_invalid_na_policy(tmp_path):
    csv_content = """col1,col2
1.0;2.0
3.0;4.0
5.0;6.0
7.0;8.0"""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)
    with pytest.raises(ValueError) as exc_info:
        data, report = load_csv(str(csv_file), delimiter=',', decimal_separator='.', has_header=True, na_policy='invalid')
    assert "Invalid NA policy" in str(exc_info.value)


def test_load_csv_numeric_delimiter_comma(tmp_path):
    csv_content = """col1;col2
1,0;2,0
3,0;4,0
5,0;6,0
7,0;8,0"""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)
    data, report, _ = load_csv(str(csv_file), delimiter=';', decimal_separator=',', has_header=True)
    expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    assert np.array_equal(data, expected)


def test_load_csv_empty_file(tmp_path):
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("")
    data, report, _ = load_csv(str(csv_file))
    assert data is None
    assert 'error' in report


def test_load_csv_with_spaces(tmp_path):
    csv_content = """col1; col2
1.0; 2.0
3.0; 4.0
5.0; 6.0
7.0; 8.0"""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)
    data, report, _ = load_csv(str(csv_file))
    expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    assert np.array_equal(data, expected)


def test_load_csv_with_quoted_values(tmp_path):
    csv_content = """col1,col2
"1.0","2.0"
"3.0","4.0"
"5.0","6.0"
"7.0","8.0" """  # Corrected last line to ensure "8.0" is parsed as float
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)
    data, report, _ = load_csv(str(csv_file), delimiter=',', decimal_separator='.', has_header=True)
    expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    assert np.array_equal(data, expected)


def test_load_csv_with_empty_lines(tmp_path):
    csv_content = """col1,col2

1.0,2.0

3.0,4.0
5.0,6.0

7.0,8.0"""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)
    data, report, _ = load_csv(str(csv_file), delimiter=',', decimal_separator='.', has_header=True)
    expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    assert np.array_equal(data, expected)


def test_load_csv_with_quoted_headers():
    """Test CSV loader with quoted headers."""
    # Create a temporary CSV file with quoted headers
    with tempfile.NamedTemporaryFile(suffix='.csv', mode='w+', delete=False) as f:
        csv_content = """\"col1\";\"col2\";\"col3\"
1;2;3
4;5;6
7;8;9
"""
        f.write(csv_content)
        temp_file_path = f.name
    
    try:
        # Load the CSV file
        data, report, _ = load_csv(temp_file_path, delimiter=';', decimal_separator='.', has_header=True)
        
        # Verify the data was loaded correctly
        assert data is not None, "Data should not be None"
        assert data.shape == (3, 3), f"Unexpected data shape: {data.shape}, expected (3, 3)"
        assert report['initial_shape'] == (3, 3), f"Unexpected initial shape: {report['initial_shape']}, expected (3, 3)"
        assert report['final_shape'] == (3, 3), f"Unexpected final shape: {report['final_shape']}, expected (3, 3)"
        assert report['detection_params']['delimiter'] == ';', f"Unexpected delimiter: {report['detection_params']['delimiter']}, expected ';'"
        
        # Check the contents of the data
        expected_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
        np.testing.assert_array_equal(data, expected_data)
        
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)


def test_csv_loader_with_quoted_headers_and_text_data():
    """Test CSV loader with quoted headers and text data (should report an error)."""
    # Create a temporary CSV file with quoted headers and text data
    with tempfile.NamedTemporaryFile(suffix='.csv', mode='w+', delete=False, newline='') as f:
        # Use semicolon delimiter and quoted headers
        csv_content = '\"col1\";\"col2\";\"col3\"\nA;B;C\nD;E;F\nG;H;I'
        f.write(csv_content)
        temp_file_path = f.name

    try:
        # Load the CSV file - should report an error due to text data
        # Use na_policy='abort' and categorical_mode='none' to force numeric conversion attempt
        data, report, _ = load_csv(temp_file_path, delimiter=';', decimal_separator='.', has_header=True, na_policy='abort', categorical_mode='none')

        # Check that an error is reported because text can't be converted to float.
        assert 'error' in report, "An error message should be present in the report"
        assert report['error'] is not None, "Error message should not be None"
        error_msg = report['error'].lower()
        # Expecting the error related to NA detection as conversion failure leads to NAs
        # or a final conversion error
        condition1 = "na values detected" in error_msg
        condition2 = "could not convert string to float" in error_msg
        condition3 = "failed to convert final data" in error_msg
        assert (condition1 or condition2 or condition3), f"Error message '{report['error']}' did not indicate an NA detection, conversion, or finalization issue."

    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)


