import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from nirs4all.data.dataset import Dataset
from nirs4all.data.dataset_loader import get_dataset

def test_dataset_categorical_handling():
    """Test that Dataset properly handles categorical data."""
    # Create a dataset with categorical info
    dataset = Dataset()
    
    # Set up sample data
    dataset._x_train = np.random.rand(1, 5, 1, 2)  # (augmentations, samples, transformations, features)
    dataset._x_test = np.random.rand(1, 3, 1, 2)
    dataset._y_train = np.array([[0], [1], [2], [0], [1]])
    dataset._y_test = np.array([[2], [0], [1]])
    
    # Set up categorical info
    dataset.y_train_categorical_info = {
        '0': {
            'categories': ['red', 'green', 'blue'],
            'original_values': ['red', 'green', 'blue']
        }
    }
    dataset.y_test_categorical_info = dataset.y_train_categorical_info
    
    # Test has_categorical_columns
    assert dataset.has_categorical_columns() is True
    
    # Test inverse_transform_categorical with 1D array
    y_pred = np.array([0, 1, 2])
    y_inverse = dataset.inverse_transform_categorical(y_pred)
    assert list(y_inverse) == ['red', 'green', 'blue']
    
    # Test with 2D array
    y_pred_2d = np.array([[0], [1], [2]])
    y_inverse_2d = dataset.inverse_transform_categorical(y_pred_2d)
    assert y_inverse_2d.shape == (3, 1)
    assert list(y_inverse_2d.flatten()) == ['red', 'green', 'blue']
    
    # Test the full inverse_transform method
    y_pred = np.array([0, 1, 2])
    y_inverse = dataset.inverse_transform(y_pred)
    assert list(y_inverse) == ['red', 'green', 'blue']
    
    # Test handling of out-of-range values
    y_pred_bad = np.array([0, 1, 10])
    y_inverse_bad = dataset.inverse_transform_categorical(y_pred_bad)
    assert y_inverse_bad[2] is None

def test_dataset_loader_with_categorical():
    """Test loading a dataset with categorical columns."""
    # Create temporary CSV files with categorical data
    with tempfile.TemporaryDirectory() as temp_dir:
        x_path = os.path.join(temp_dir, 'x_data.csv')
        y_path = os.path.join(temp_dir, 'y_data.csv')
        
        # Create X data
        x_df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [5.0, 4.0, 3.0, 2.0, 1.0]
        })
        x_df.to_csv(x_path, index=False)
        
        # Create Y data with categorical column
        y_df = pd.DataFrame({
            'target': ['red', 'green', 'blue', 'red', 'green']
        })
        y_df.to_csv(y_path, index=False)
    
        # Create config dictionary
        config = {
            'data_id': 'categorical_test',  # Add data_id
            'train_x': x_path,
            'train_y': y_path,
            'test_x': x_path,  # Using same data for test for simplicity
            'test_y': y_path,
            # Correct parameter keys, assuming categorical applies to y
            'train_y_params': {'categorical_mode': 'auto'},
            'test_y_params': {'categorical_mode': 'auto'},
            # Add global_params, assuming headers exist
            'global_params': {'has_header': True, 'delimiter': ','},
        }

        # Add assertion to verify config type before passing
        assert isinstance(config, dict), f"Config is not a dict: {type(config)}"
        
        # Load dataset
        dataset = get_dataset(config)
        
        # Verify categorical info was captured
        assert dataset.has_categorical_columns() is True
        assert 'target' in dataset.y_train_categorical_info
        
        # Check that y data contains the correct categorical codes
        unique_values = np.unique(dataset.raw_y_train)
        assert len(unique_values) == 3
        assert all(val in [0, 1, 2] for val in unique_values)
        
        # Test inverse transform
        results = dataset.inverse_transform(np.array([0, 1, 2]))
        assert set(results) == {'red', 'green', 'blue'}
