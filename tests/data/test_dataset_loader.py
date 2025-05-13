# tests/test_dataset_loader.py

import pytest
import numpy as np
from pathlib import Path
from nirs4all.data.dataset_loader import (
    _merge_params,
    load_XY,
    handle_data,
    get_dataset,
)


def test_merge_params():
    global_params = {'a': 1, 'b': 2}
    handler_params = {'b': 3, 'c': 4}
    local_params = {'c': 5, 'd': 6}
    result = _merge_params(local_params, handler_params, global_params)
    expected = {'a': 1, 'b': 3, 'c': 5, 'd': 6}
    assert result == expected


def test_merge_params_empty():
    global_params = {'a': 1}
    result = _merge_params(None, None, global_params)
    assert result == {'a': 1}
    
    result = _merge_params({'b': 2}, None, None)
    assert result == {'b': 2}


def test_load_XY(tmp_path):
    x_content = """col1,col2
1,2
3,4
5,6
7,8
9,10"""
    y_content = """label
10
20
30
40
50"""
    x_file = tmp_path / "x.csv"
    y_file = tmp_path / "y.csv"
    x_file.write_text(x_content)
    y_file.write_text(y_content)
    x_params = {'categorical_mode': 'auto', 'data_type': 'x', 'delimiter': ','}
    y_params = {}
    # Unpack all four return values
    x, y, _, _ = load_XY(str(x_file), None, x_params, str(y_file), None, y_params)  # Ignore reports
    
    assert x.shape == (5, 2)
    assert y.shape == (5, 1)
    assert np.array_equal(x[:, 0], [1, 3, 5, 7, 9])
    assert np.array_equal(y[:, 0], [10, 20, 30, 40, 50])


def test_load_XY_invalid_x():
    with pytest.raises(ValueError, match="Invalid x definition: x_path is None"):
        # Ignore return values as we expect an error
        load_XY(None, None, {}, None, None, {})


def test_load_XY_invalid_y(tmp_path):
    x_content = """col1,col2
1,2
3,4
5,6
7,8
9,10"""
    x_file = tmp_path / "x.csv"
    x_file.write_text(x_content)
    # Define y_file even if it doesn't exist, as the error is expected during loading
    y_file = tmp_path / "y_nonexistent.csv"
    x_params = {}
    y_params = {}
    # Update the expected error message to match the actual file not found error
    with pytest.raises(ValueError, match="Invalid data: y contains errors: Le fichier n'existe pas"):
        # Ignore return values
        load_XY(str(x_file), None, {}, str(y_file), None, {})


def test_load_XY_y_from_x(tmp_path):
    x_content = """col1,col2,label
1,2,10
3,4,20
5,6,30
7,8,40
9,10,50"""
    x_file = tmp_path / "x.csv"
    x_file.write_text(x_content)
    x_params = {'categorical_mode': 'auto', 'data_type': 'x', 'delimiter': ','}
    y_params = {}
    # Correct y_filter to match expected_x and expected_y
    y_filter = [2]  # Select only the 'label' column (index 2) as y
    # Unpack all four return values
    x, y, _, _ = load_XY(str(x_file), None, x_params, None, y_filter, y_params)  # Ignore reports
    # x should now contain columns 0 and 1
    expected_x = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.float32)
    # y should contain column 2
    expected_y = np.array([[10], [20], [30], [40], [50]], dtype=np.float32)
    assert np.array_equal(x, expected_x)
    assert np.array_equal(y, expected_y)


def test_load_XY_invalid_y_filter(tmp_path):
    x_content = """col1;col2;label
1;2;10
3;4;20
5;6;30
7;8;40
9;10;50"""
    x_file = tmp_path / "x.csv"
    x_file.write_text(x_content)
    x_params = {}
    y_params = {}
    with pytest.raises(ValueError, match="Invalid y definition: y_filter is not a list of integers"):
        # Ignore return values
        load_XY(str(x_file), None, {}, None, ['not_an_int'], {})


def test_load_XY_y_filter_out_of_bounds(tmp_path):
    x_content = """col1;col2;label
1;2;10
3;4;20
5;6;30
7;8;40
9;10;50"""
    x_file = tmp_path / "x.csv"
    x_file.write_text(x_content)
    x_params = {}
    y_params = {}
    with pytest.raises(ValueError, match="Invalid y definition: y_filter contains invalid indices"):
        # Ignore return values
        load_XY(str(x_file), None, {}, None, [10], {})  # Index out of bounds


def test_handle_data(tmp_path):
    x_content = """col1,col2
1,2
3,4
5,6
7,8
9,10"""
    y_content = """label
10
20
30
40
50"""
    x_file = tmp_path / "x.csv"
    y_file = tmp_path / "y.csv"
    x_file.write_text(x_content)
    y_file.write_text(y_content)
    x_params = {'data_type': 'x'}
    y_params = {'data_type': 'y'}
    config = {
        'train_x': str(x_file),
        'train_y': str(y_file),
        'train_x_params': x_params,  # Corrected trailing whitespace
        'train_y_params': y_params,  # Corrected trailing whitespace
        'global_params': {'delimiter': ','}
    }
    # Unpack all four return values
    x, y, _, _ = handle_data(config, 'train')  # Ignore reports
    
    assert x.shape == (5, 2)
    assert y.shape == (5, 1)


def test_handle_data_with_params(tmp_path):
    x_content = """col1,col2
1,2
3,4
5,6
7,8
9,10"""
    y_content = """label
10
20
30
40
50"""
    x_file = tmp_path / "x.csv"
    y_file = tmp_path / "y.csv"
    x_file.write_text(x_content)
    y_file.write_text(y_content)
    config = {
        'train_x': str(x_file),
        'train_y': str(y_file),
        'global_params': {'na_policy': 'remove', 'has_header': True, 'delimiter': ','},
    }
    # Unpack all four return values
    x, y, _, _ = handle_data(config, 'train')  # Ignore reports
    
    assert x.shape == (5, 2)
    assert y.shape == (5, 1)


def test_get_dataset(tmp_path):
    # Mock the parse_config function to return a valid configuration
    import nirs4all.data.dataset_loader as ds_loader
    original_parse_config = ds_loader.parse_config
    
    def mock_parse_config(config):
        # Just return the config dict directly instead of trying to parse it
        return config
    
    # Apply the monkey patch
    ds_loader.parse_config = mock_parse_config
    
    try:
        x_content = """col1;col2
1;2
3;4
5;6
7;8
9;10"""
        y_content = """label
10
20
30
40
50"""
        x_file = tmp_path / "x.csv"
        y_file = tmp_path / "y.csv"
        x_file.write_text(x_content)
        y_file.write_text(y_content)
        data_config = {
            'data_id': 'test_dataset',
            'train_x': str(x_file),
            'train_y': str(y_file),
            'test_x': str(x_file),
            'test_y': str(y_file),
        }
        dataset = get_dataset(data_config)
        
        assert hasattr(dataset, 'x_train')
        assert hasattr(dataset, 'y_train')
        assert hasattr(dataset, 'x_test')
        assert hasattr(dataset, 'y_test')
        
        assert dataset.x_train_().shape == (5, 2)
        assert dataset.y_train.shape == (5, 1)
        assert dataset.x_test_().shape == (5, 2)
        assert dataset.y_test.shape == (5, 1)
    finally:
        # Restore the original function
        ds_loader.parse_config = original_parse_config


def test_get_dataset_error_handling(tmp_path):
    # Test with a partially invalid configuration
    x_content = """col1,col2
1,2
3,4
5,6
7,8
9,10"""
    y_content = """label
10
20
30
40
50"""
    x_file = tmp_path / "x.csv"
    y_file = tmp_path / "y.csv"
    nonexistent_file = tmp_path / "nonexistent.csv"
    
    x_file.write_text(x_content)
    y_file.write_text(y_content)
    
    data_config = {
        'train_x': str(x_file),
        'train_y': str(y_file),
        'test_x': str(nonexistent_file),  # This should cause an error
        'test_y': str(y_file),
    }
    
    with pytest.raises(ValueError):
        get_dataset(data_config)  # Don't need to assign the result


def test_sample_datasets_loading():
    """Integration test to verify all sample datasets can be loaded correctly."""
    # Get the base path to the sample_data directory
    base_path = Path(__file__).parent.parent.parent / "sample_data"
    assert base_path.exists(), f"Sample data directory not found at {base_path}"
    
    # Mock the parse_config function to return the configuration directly
    import nirs4all.data.dataset_loader as ds_loader
    original_parse_config = ds_loader.parse_config
    ds_loader.parse_config = lambda config: config
    
    try:
        # Create configs for all three datasets
        configs = [
            {
                'name': 'binary',
                'config': {
                    'data_id': 'binary',
                    'train_x': str(base_path / "binary" / "Xcal.csv"),
                    'train_y': str(base_path / "binary" / "Ycal.csv"),
                    'test_x': str(base_path / "binary" / "Xval.csv"),
                    'test_y': str(base_path / "binary" / "Yval.csv"),
                    # Explicitly state header and delimiter for binary dataset
                    'global_params': {'has_header': True, 'delimiter': ';'}
                }
            },
            {
                'name': 'classification',
                'config': {
                    'data_id': 'classification',
                    'train_x': str(base_path / "classification" / "Xcal.csv"),
                    'train_y': str(base_path / "classification" / "Ycal.csv"),
                    'test_x': str(base_path / "classification" / "Xval.csv"),
                    'test_y': str(base_path / "classification" / "Yval.csv"),
                    'global_params': {'na_policy': 'remove', 'has_header': True}
                }
            },
            {
                'name': 'regression',
                'config': {
                    'data_id': 'regression',
                    'train_x': str(base_path / "regression" / "Xcal.csv.gz"),
                    'train_y': str(base_path / "regression" / "Ycal.csv.gz"),
                    'test_x': str(base_path / "regression" / "Xval.csv.gz"),
                    'test_y': str(base_path / "regression" / "Yval.csv.gz"),
                    'global_params': {'na_policy': 'remove', 'has_header': True}
                }
            }
        ]
        
        # Test loading each dataset separately
        for dataset_info in configs:
            name = dataset_info['name']
            config = dataset_info['config']
            
            try:
                dataset = get_dataset(config)
                
                # Basic validation that dataset was loaded
                assert dataset.x_train is not None, f"{name} dataset x_train not loaded"
                assert dataset.y_train is not None, f"{name} dataset y_train not loaded"
                assert dataset.x_test is not None, f"{name} dataset x_test not loaded"
                assert dataset.y_test is not None, f"{name} dataset y_test not loaded"
                
                # Print shape information for debugging
                print(f"\\n{name} dataset properties:")
                print(f"  x_train shape: {dataset.x_train_().shape}")
                print(f"  y_train shape: {dataset.y_train.shape}")
                print(f"  x_test shape: {dataset.x_test_().shape}")
                print(f"  y_test shape: {dataset.y_test.shape}")
                
            except ValueError as e:  # Catch specific error
                pytest.fail(f"Failed to load {name} dataset: {str(e)}")
            # Removed overly broad Exception catch

    finally:
        # Restore original parse_config function
        ds_loader.parse_config = original_parse_config


def test_sample_datasets_shapes_and_properties():
    """Integration test to verify sample datasets shapes and properties."""
    # Get the base path to the sample_data directory
    base_path = Path(__file__).parent.parent.parent / "sample_data"
    
    # Mock the parse_config function to return the configuration directly
    import nirs4all.data.dataset_loader as ds_loader
    original_parse_config = ds_loader.parse_config
    ds_loader.parse_config = lambda config: config
    
    try:
        # Test all three datasets with detailed shape checks
        datasets = {
            'binary': {
                'config': {
                    'data_id': 'binary',
                    'train_x': str(base_path / "binary" / "Xcal.csv"),
                    'train_y': str(base_path / "binary" / "Ycal.csv"),
                    'test_x': str(base_path / "binary" / "Xval.csv"),
                    'test_y': str(base_path / "binary" / "Yval.csv"),
                    # Explicitly state header and delimiter for binary dataset
                    'global_params': {'has_header': True, 'delimiter': ';'}
                }
            },
            'classification': {
                'config': {
                    'data_id': 'classification',
                    'train_x': str(base_path / "classification" / "Xcal.csv"),
                    'train_y': str(base_path / "classification" / "Ycal.csv"),
                    'test_x': str(base_path / "classification" / "Xval.csv"),
                    'test_y': str(base_path / "classification" / "Yval.csv"),
                    'global_params': {'has_header': True}  # Explicitly state header exists
                }
            },
            'regression': {
                'config': {
                    'data_id': 'regression',
                    'train_x': str(base_path / "regression" / "Xcal.csv.gz"),
                    'train_y': str(base_path / "regression" / "Ycal.csv.gz"),
                    'test_x': str(base_path / "regression" / "Xval.csv.gz"),
                    'test_y': str(base_path / "regression" / "Yval.csv.gz"),
                    'global_params': {'has_header': True}  # Explicitly state header exists
                }
            }
        }

        for dataset_name, dataset_info in datasets.items():
            try:
                dataset = get_dataset(dataset_info['config'])
                # ... rest of the test ...
                # Check dataset properties
                assert dataset.x_train_() is not None, f"{dataset_name} dataset x_train not loaded"
                assert dataset.y_train is not None, f"{dataset_name} dataset y_train not loaded"
                assert dataset.x_test_() is not None, f"{dataset_name} dataset x_test not loaded"
                assert dataset.y_test is not None, f"{dataset_name} dataset y_test not loaded"
                
                # Check that dimensions match between X and Y data
                assert dataset.x_train_().shape[0] == dataset.y_train.shape[0], \
                    f"{dataset_name} dataset x_train and y_train have mismatched sample counts"
                assert dataset.x_test_().shape[0] == dataset.y_test.shape[0], \
                    f"{dataset_name} dataset x_test and y_test have mismatched sample counts"
                
                # Print dataset information for debugging
                print(f"\\n{dataset_name} dataset properties:")
                print(f"  x_train shape: {dataset.x_train_().shape}")
                print(f"  y_train shape: {dataset.y_train.shape}")
                print(f"  x_test shape: {dataset.x_test_().shape}")
                print(f"  y_test shape: {dataset.y_test.shape}")
                
                # Store shapes for reference
                dataset_info['shapes'] = {
                    'x_train': dataset.x_train_().shape,
                    'y_train': dataset.y_train.shape,
                    'x_test': dataset.x_test_().shape,
                    'y_test': dataset.y_test.shape
                }
            except ValueError as e:  # Catch specific error
                pytest.fail(f"Failed to load {dataset_name} for shape check: {e}")
            # Removed overly broad Exception catch

    finally:
        # Restore original parse_config function
        ds_loader.parse_config = original_parse_config

