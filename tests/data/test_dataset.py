# tests/test_dataset.py

import pytest
import numpy as np
from nirs4all.data.dataset import Dataset


def test_dataset_initialization():
    dataset = Dataset()
    assert dataset.x_train is None
    assert dataset.y_train is None
    assert dataset.x_test is None
    assert dataset.y_test is None


def test_dataset_x_train_setter_getter():
    dataset = Dataset()
    x_train = np.random.rand(100, 10)
    dataset.x_train = x_train
    assert dataset.x_train.shape == (1, 100, 1, 10)
    assert dataset._x_train.shape == (1, 100, 1, 10)


def test_dataset_x_train_invalid_shape():
    dataset = Dataset()
    x_train = np.random.rand(100)
    with pytest.raises(ValueError):
        dataset.x_train = x_train


def test_dataset_y_train_setter_getter():
    dataset = Dataset()
    x_train = np.random.rand(1, 100, 1, 10)
    dataset._x_train = x_train
    y_train = np.random.rand(100)
    dataset.y_train = y_train
    assert dataset.y_train.shape == (100, 1)  # Adjusted expected shape
    assert dataset._y_train.shape == (100, 1)


def test_dataset_y_train_invalid_shape():
    dataset = Dataset()
    x_train = np.random.rand(1, 100, 1, 10)
    dataset._x_train = x_train
    y_train = np.random.rand(99)
    with pytest.raises(ValueError):
        dataset.y_train = y_train


def test_dataset_filter_x():
    dataset = Dataset()
    x_train = np.random.rand(2, 50, 1, 10)
    dataset._x_train = x_train
    filtered_x = dataset.filter_x(x_train, union_type='concat', indices=[0, 1, 2])
    assert filtered_x.shape == (6, 10)


def test_dataset_filter_y():
    dataset = Dataset()
    x_train = np.random.rand(2, 50, 1, 10)
    y_train = np.random.rand(50, 1)
    dataset._x_train = x_train
    dataset._y_train = y_train
    filtered_y = dataset.filter_y(y_train, indices=[0, 1, 2])
    assert filtered_y.shape == (6, 1)


def test_dataset_fold_data_no_folds():
    dataset = Dataset()
    x_train = np.random.rand(2, 50, 1, 10)
    y_train = np.random.rand(50, 1)
    x_test = np.random.rand(2, 20, 1, 10)
    y_test = np.random.rand(20, 1)
    dataset._x_train = x_train
    dataset._y_train = y_train
    dataset._x_test = x_test
    dataset._y_test = y_test
    folds = list(dataset.fold_data())
    assert len(folds) == 1
    x_train_fold, y_train_fold, x_test_fold, y_test_fold = folds[0]
    assert x_train_fold.shape == (100, 10)
    assert y_train_fold.shape == (100, 1)
    assert x_test_fold.shape == (40, 10)
    assert y_test_fold.shape == (40, 1)


def test_dataset_fold_data_with_folds():
    dataset = Dataset()
    x_train = np.random.rand(2, 100, 1, 10)
    y_train = np.random.rand(100, 1)
    dataset._x_train = x_train
    dataset._y_train = y_train
    dataset._folds = [(np.arange(80), np.arange(80, 100))]
    folds = list(dataset.fold_data())
    assert len(folds) == 1
    x_train_fold, y_train_fold, x_val_fold, y_val_fold = folds[0]
    assert x_train_fold.shape == (160, 10)
    assert y_train_fold.shape == (160, 1)
    assert x_val_fold.shape == (40, 10)
    assert y_val_fold.shape == (40, 1)


def test_dataset_inverse_transform():
    dataset = Dataset()
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    y_train = np.array([[1], [2], [3]])
    scaler.fit(y_train)
    dataset.y_transformer = scaler
    y_pred = np.array([[0], [0.5], [1]])
    y_inverse = dataset.inverse_transform(y_pred)
    expected = scaler.inverse_transform(y_pred)
    assert np.allclose(y_inverse, expected)


def test_dataset_str():
    dataset = Dataset()
    x_train = np.random.rand(2, 50, 1, 10)
    y_train = np.random.rand(50, 1)
    dataset._x_train = x_train
    dataset._y_train = y_train
    s = str(dataset)
    assert "Dataset(" in s


def test_dataset_test_is_indices():
    dataset = Dataset()
    dataset._x_test = np.array([0, 1, 2])
    assert dataset.test_is_indices


def test_dataset_y_test_when_indices():
    dataset = Dataset()
    x_train = np.random.rand(2, 5, 1, 10)
    y_train = np.random.rand(5, 1)
    dataset._x_train = x_train
    dataset._y_train = y_train
    dataset._x_test = np.array([0, 1])
    y_test = dataset.y_test
    assert y_test.shape == (4, 1)


def test_dataset_y_test_setter_when_indices():
    dataset = Dataset()
    dataset._x_test = np.array([0, 1, 2])
    with pytest.raises(ValueError):
        dataset.y_test = np.array([1, 2, 3])


def test_dataset_x_test_setter():
    dataset = Dataset()
    x_test = np.random.rand(100, 10)
    dataset.x_test = x_test
    assert dataset.x_test.shape == (1, 100, 1, 10)


def test_dataset_x_test_invalid_shape():
    dataset = Dataset()
    x_test = np.random.rand(100, 10, 5)  # 3D array, invalid shape
    with pytest.raises(ValueError):
        dataset.x_test = x_test


def test_dataset_property_group_test():
    dataset = Dataset()
    dataset._x_train = np.random.rand(2, 5, 1, 10)
    dataset._x_test = np.random.rand(2, 5, 1, 10)
    dataset._group_train = np.array([1, 2, 3, 4, 5])
    dataset._group_test = np.array([1, 2, 3, 4, 5])
    group_test = dataset.group_test
    assert group_test.shape == (10,)


def test_dataset_y_train_property():
    dataset = Dataset()
    x_train = np.random.rand(2, 5, 1, 10)
    y_train = np.random.rand(5, 1)
    dataset._x_train = x_train
    dataset._y_train = y_train
    y_train_prop = dataset.y_train
    assert y_train_prop.shape == (10, 1)


def test_dataset_invalid_y_train_shape():
    dataset = Dataset()
    x_train = np.random.rand(1, 100, 1, 10)
    dataset._x_train = x_train
    y_train = np.random.rand(100, 2, 2)  # Invalid shape
    with pytest.raises(ValueError):
        dataset.y_train = y_train


def test_dataset_invalid_y_test_shape():
    dataset = Dataset()
    x_test = np.random.rand(1, 50, 1, 10)
    dataset._x_test = x_test
    y_test = np.random.rand(49, 1)
    with pytest.raises(ValueError):
        dataset.y_test = y_test


def test_dataset_with_encoded_categorical_y():
    """Test that the dataset can handle encoded categorical y data."""
    dataset = Dataset()
    x_train = np.random.rand(1, 5, 1, 10)
    dataset._x_train = x_train
    
    # Simulate already-encoded categorical data (as if processed from a CSV with strings)
    y_encoded = np.array([0, 1, 2, 0, 1]).reshape(-1, 1)
    
    # Set the encoded data
    dataset.y_train = y_encoded
    
    # Verify it was correctly set
    assert dataset._y_train.shape == (5, 1)
    assert np.array_equal(dataset._y_train, y_encoded)


def test_dataset_categorical_y_transformer():
    """Test using a categorical encoder as y_transformer."""
    dataset = Dataset()
    from sklearn.preprocessing import LabelEncoder
    
    # Create a class to wrap LabelEncoder to have transform and inverse_transform methods
    class LabelEncoderWrapper:
        def __init__(self, categories):
            self.encoder = LabelEncoder()
            self.encoder.fit(categories)
            self.classes_ = self.encoder.classes_
            
        def transform(self, y):
            return self.encoder.transform(y.flatten()).reshape(-1, 1)
            
        def inverse_transform(self, y):
            return self.encoder.inverse_transform(y.flatten()).reshape(-1, 1)
    
    # Set up the dataset
    x_train = np.random.rand(1, 5, 1, 10)
    dataset._x_train = x_train
    
    # Original categorical data
    categories = np.array(['cat', 'dog', 'bird', 'cat', 'dog'])
    
    # Create and set the transformer
    encoder_wrapper = LabelEncoderWrapper(categories)
    dataset.y_transformer = encoder_wrapper
    
    # Set encoded y_train
    y_encoded = encoder_wrapper.transform(categories)
    dataset._y_train = y_encoded
    
    # Test inverse transform
    y_pred = np.array([[0], [1], [2]])  # Encoded predictions
    y_inverse = dataset.inverse_transform(y_pred)
    
    # Check that it correctly maps back to original categories based on LabelEncoder's sorted classes
    assert y_inverse[0, 0] == 'bird' # 0 corresponds to 'bird'
    assert y_inverse[1, 0] == 'cat'  # 1 corresponds to 'cat'
    assert y_inverse[2, 0] == 'dog'   # 2 corresponds to 'dog'


def test_dataset_categorical_y_multiple_columns():
    """Test handling multiple categorical columns using OneHotEncoder."""
    dataset = Dataset()
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    
    # Set up the dataset
    x_train = np.random.rand(1, 4, 1, 10)
    dataset._x_train = x_train
    
    # Create multi-column categorical data (as would come from a CSV)
    categories = pd.DataFrame({
        'color': ['red', 'blue', 'green', 'red'],
        'size': ['small', 'medium', 'large', 'small']
    })
      # Create a transformer for multiple categorical columns
    encoder = OneHotEncoder(sparse_output=False)
    encoded_array = encoder.fit_transform(categories)
    
    # Set transformed y_train (now one-hot encoded)
    dataset.y_train = encoded_array

    # Verify shape (4 samples, 6 features from one-hot encoding: 3 for color, 3 for size)
    assert dataset._y_train.shape == (4, 6)

    # Set the encoder as transformer
    dataset.y_transformer = encoder

    # Test with new encoded data
    # Example encoding for ['blue', 'medium'] -> [1, 0, 0] for color, [0, 1, 0] for size
    # Assuming OneHotEncoder order: blue, green, red | large, medium, small
    new_encoded = np.array([[1, 0, 0, 0, 1, 0]]) # Shape (1, 6)
    y_inverse = dataset.inverse_transform(new_encoded)

    # Check it returns something with correct shape and values
    assert y_inverse.shape == (1, 2) # Should return 2 columns
    assert y_inverse[0, 0] == 'blue'
    assert y_inverse[0, 1] == 'medium'
