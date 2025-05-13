import os
import tempfile
import pandas as pd
import numpy as np
import pytest
from nirs4all.data.csv_loader import load_csv

# Helper function for creating and cleaning up temp files
@pytest.fixture
def temp_csv_file():
    temp_path = None
    try:
        # Create the file path using mkstemp for better control
        fd, temp_path = tempfile.mkstemp(suffix='.csv')
        os.close(fd)  # Close the file descriptor immediately
        yield temp_path  # Provide the path to the test
    finally:
        # Clean up
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except PermissionError:
                print(f"Warning: Could not remove temp file {temp_path} due to PermissionError.")
            except Exception as e:
                print(f"Warning: Error removing temp file {temp_path}: {e}")


def test_basic_categorical_detection(temp_csv_file):
    """Test that string columns are automatically detected as categorical."""
    temp_path = temp_csv_file  # Get path from fixture

    # Create test data with categorical column
    df = pd.DataFrame({
        'num_col': [1.0, 2.0, 3.0, 4.0, 5.0],
        'cat_col': ['red', 'green', 'blue', 'red', 'green']
    })
    df.to_csv(temp_path, index=False)  # Write to the path

    # Load with automatic categorical detection, explicitly stating header exists
    data, report, _ = load_csv(temp_path, categorical_mode='auto', delimiter=",", data_type='y')

    assert data is not None, f"Report: {report}"
    assert data.shape == (5, 2), f"Report: {report}"

    # Verify categorical info was captured
    assert 'categorical_info' in report
    assert 'cat_col' in report['categorical_info']
    assert len(report['categorical_info']['cat_col']['categories']) == 3
    assert set(report['categorical_info']['cat_col']['categories']) == {'red', 'green', 'blue'}

    # Verify categorical column (now index 0) matches the mapping from the report
    actual_categories = report['categorical_info']['cat_col']['categories']
    mapping = {cat: i for i, cat in enumerate(actual_categories)}
    expected_codes = df['cat_col'].map(mapping).values
    np.testing.assert_array_equal(data.iloc[:, 1].values, expected_codes.astype(np.float32))
    np.testing.assert_array_equal(data.iloc[:, 0].values, df['num_col'].values.astype(np.float32))


def test_categorical_mode_options(temp_csv_file):
    """Test different categorical_mode options."""
    temp_path = temp_csv_file

    # Create test data with categorical column
    df = pd.DataFrame({
        'num_col': [1.0, 2.0, 3.0, 4.0, 5.0],
        'cat_col': ['red', 'green', 'blue', 'red', 'green']
    })
    df.to_csv(temp_path, index=False)

    # Test mode='preserve' - should not convert to categorical codes, keep column as NaN
    data_preserve, report_preserve, _ = load_csv(temp_path, categorical_mode='preserve', delimiter=",", data_type='y')
    assert data_preserve is not None, f"Report: {report_preserve}"
    assert 'cat_col' not in report_preserve['categorical_info']
    # With categorical_mode='preserve' and na_policy='remove' (default),
    # the string column becomes NaN and rows are removed.
    assert data_preserve.shape == (0, 2), f"Expected shape (0, 2), got {data_preserve.shape}. Report: {report_preserve}"

    # Test mode='none' - should not detect categorical, treat as numeric (will become NaN)
    data_none, report_none, _ = load_csv(temp_path, categorical_mode='none', delimiter=",", data_type='y')
    assert data_none is not None, f"Report: {report_none}"
    assert len(report_none['categorical_info']) == 0
    # With mode='none' and na_policy='remove', string columns become NaN and rows are removed.
    assert data_none.shape == (0, 2), f"Expected shape (0, 2), got {data_none.shape}. Report: {report_none}"
    # Cannot check for NaNs in specific columns if the array is empty
    # assert np.isnan(data_none[:, 1]).all()  # String values should become NaN


def test_warning_for_ambiguous_detection(temp_csv_file):
    """Test that warnings are issued for ambiguous categorical detection."""
    temp_path = temp_csv_file

    # Create test data with categorical column that has numeric header
    df = pd.DataFrame({
        '1': [1.0, 2.0, 3.0, 4.0, 5.0],
        '2': ['red', 'green', 'blue', 'red', 'green']  # Numeric column name with string data
    })
    df.to_csv(temp_path, index=False)

    # Load with automatic categorical detection, specifying header exists
    data, report, _ = load_csv(temp_path, categorical_mode='auto', has_header=True, delimiter=",", data_type='y')

    assert data is not None, f"Report: {report}"
    assert data.shape == (5, 2), f"Report: {report}"

    # Verify warning was issued for column '2'
    assert 'warnings' in report
    assert len(report['warnings']) > 0, f"No warnings found. Report: {report}"
    assert any("Column '2' detected as categorical but has a numeric header" in warning for warning in report['warnings']), \
           f"Expected warning not found. Warnings: {report['warnings']}"

    # Verify column '2' (categorical, now index 0) was correctly converted using report's mapping
    assert '2' in report['categorical_info']
    actual_categories_col2 = report['categorical_info']['2']['categories']
    mapping_col2 = {cat: i for i, cat in enumerate(actual_categories_col2)}
    expected_codes_col2 = df['2'].map(mapping_col2).values
    # np.testing.assert_array_equal(data[:, 0], expected_codes_col2.astype(np.float32))
    np.testing.assert_array_equal(data.iloc[:, 1].values, expected_codes_col2.astype(np.float32))

    # Verify column '1' (numeric, now index 1) was treated as numeric
    assert '1' not in report['categorical_info']
    # np.testing.assert_array_equal(data[:, 1], df['1'].values.astype(np.float32))
    np.testing.assert_array_equal(data.iloc[:, 0].values, df['1'].values.astype(np.float32))

# Add more tests as needed, e.g., for mixed types, different NA policies, etc.
