import pytest
import numpy as np
from sklearn.preprocessing import StandardScaler
from nirs4all.core.processor import Processor
from nirs4all.data.dataset import Dataset

def test_processor_preserves_categorical():
    """Test that processor preserves categorical information."""
    # Create a dataset with categorical info
    dataset = Dataset()
    dataset.x_train = np.random.rand(1, 10, 1, 5)
    dataset.x_test = np.random.rand(1, 5, 1, 5)
    dataset._y_train = np.array([[0], [1], [2], [0], [1], [2], [0], [1], [2], [0]])
    dataset._y_test = np.array([[2], [0], [1], [2], [0]])
    
    # Set up categorical info
    dataset.y_train_categorical_info = {
        '0': {
            'categories': ['red', 'green', 'blue'],
            'original_values': ['red', 'green', 'blue']
        }
    }
    dataset.y_test_categorical_info = dataset.y_train_categorical_info
    
    # Create processor with minimal configuration
    processor = Processor(config={'x_pipeline': None, 'y_pipeline': None})
    
    # Process dataset
    processed = processor.process_dataset(dataset)
    
    # Verify categorical info was preserved
    assert processed.has_categorical_columns() is True
    assert processed.y_train_categorical_info == dataset.y_train_categorical_info
    assert processed.y_test_categorical_info == dataset.y_test_categorical_info
    
    # Test inverse transform of predictions
    y_pred = np.array([0, 1, 2])
    y_inverse = processor.inverse_transform_predictions(y_pred, processed)
    assert list(y_inverse) == ['red', 'green', 'blue']

def test_processor_with_transformations():
    """Test that processor preserves categorical information when applying transformations."""
    # Create a dataset with categorical info
    dataset = Dataset()
    dataset.x_train = np.random.rand(1, 10, 1, 5)
    dataset.x_test = np.random.rand(1, 5, 1, 5)
    dataset._y_train = np.array([[0], [1], [2], [0], [1], [2], [0], [1], [2], [0]])
    dataset._y_test = np.array([[2], [0], [1], [2], [0]])
    
    # Set up categorical info
    dataset.y_train_categorical_info = {
        '0': {
            'categories': ['red', 'green', 'blue'],
            'original_values': ['red', 'green', 'blue']
        }
    }
    dataset.y_test_categorical_info = dataset.y_train_categorical_info
    
    # Create processor with a transformation in the pipeline
    processor = Processor(config={'x_pipeline': StandardScaler(), 'y_pipeline': None})
    
    # Process dataset
    processed = processor.process_dataset(dataset)
    
    # Verify categorical info was preserved even with transformations
    assert processed.has_categorical_columns() is True
    assert processed.y_train_categorical_info == dataset.y_train_categorical_info
    assert processed.y_test_categorical_info == dataset.y_test_categorical_info
    
    # Test inverse transform of predictions
    y_pred = np.array([0, 1, 2])
    y_inverse = processor.inverse_transform_predictions(y_pred, processed)
    assert list(y_inverse) == ['red', 'green', 'blue']
