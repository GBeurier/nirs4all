'''
Integration tests for data loading and validity.
These tests verify that sample datasets can be loaded correctly.
'''

import pytest
import os
import sys
import warnings
import numpy as np

# Add parent directory to sys.path to find the nirs4all module
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

parent_dir = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.append(parent_dir)

from nirs4all.core.config import Config
from nirs4all.core.runner import ExperimentRunner
from nirs4all.data.dataset import Dataset  # Import Dataset for type hinting
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

seed = 42  # Seed for reproducibility, though less critical for pure data loading

def check_dataset_validity(ds_obj: Dataset, dataset_name: str):
    '''Checks the validity of the loaded data, assuming ds_obj is a Dataset object.'''
    assert ds_obj is not None, f"Dataset object for {dataset_name} should not be None"

    # --- Train data checks (properties and processed/augmented versions) ---
    assert hasattr(ds_obj, 'x_train'), f"Dataset for {dataset_name} should have x_train property"
    assert ds_obj.x_train is not None, f"x_train property for {dataset_name} should not be None (returns _x_train)"
    assert isinstance(ds_obj.x_train, np.ndarray), f"x_train property for {dataset_name} should be a numpy array"
    assert ds_obj.x_train.shape[0] > 0, f"x_train property (n_augmentations) for {dataset_name} should be > 0"
    assert ds_obj.x_train.shape[1] > 0, f"x_train property (n_samples) for {dataset_name} should be > 0"

    assert hasattr(ds_obj, 'y_train'), f"Dataset for {dataset_name} should have y_train property"
    y_train_prop = ds_obj.y_train  # Access augmented y_train property
    assert y_train_prop is not None, f"y_train property for {dataset_name} should not be None"
    assert isinstance(y_train_prop, np.ndarray), f"y_train property for {dataset_name} should be a numpy array"
    assert y_train_prop.shape[0] > 0, f"y_train property for {dataset_name} should not be empty"
    
    # Adjust for potential leading dimension in x_train
    x_train_samples = ds_obj.x_train.shape[0]
    if ds_obj.x_train.ndim > y_train_prop.ndim and ds_obj.x_train.shape[0] == 1:
        # If x_train is (1, num_samples, ...) and y_train is (num_samples, ...)
        x_train_samples = ds_obj.x_train.shape[1]

    assert x_train_samples == y_train_prop.shape[0], \
        f"x_train samples ({x_train_samples}) and y_train samples ({y_train_prop.shape[0]}) row count mismatch for {dataset_name}"

    x_train_processed_2d = ds_obj.x_train_(union_type='concat')  # (n_aug * n_samples, features)
    assert x_train_processed_2d.shape[0] == y_train_prop.shape[0], \
        f"Processed x_train ({x_train_processed_2d.shape[0]}) and y_train property ({y_train_prop.shape[0]}) row count mismatch for {dataset_name}"

    # --- Test data checks (properties and processed/augmented versions) ---
    # Mapping x_val_processed -> x_test, y_val_processed -> y_test
    assert hasattr(ds_obj, 'x_test'), f"Dataset for {dataset_name} should have x_test property"
    assert ds_obj.x_test is not None, f"x_test property for {dataset_name} should not be None (returns _x_test)"
    assert isinstance(ds_obj.x_test, np.ndarray), f"x_test property for {dataset_name} should be a numpy array"
    
    y_test_prop = ds_obj.y_test  # Access augmented y_test property
    x_test_processed_2d = ds_obj.x_test_(union_type='concat')

    if not ds_obj.test_is_indices:
        assert ds_obj.x_test.shape[0] > 0, f"x_test property (n_augmentations) for {dataset_name} should be > 0"
        assert ds_obj.x_test.shape[1] > 0, f"x_test property (n_samples) for {dataset_name} should be > 0"
        assert y_test_prop is not None, f"y_test property for {dataset_name} should not be None"
        assert isinstance(y_test_prop, np.ndarray), f"y_test property for {dataset_name} should be a numpy array"
        assert y_test_prop.shape[0] > 0, f"y_test property for {dataset_name} should not be empty"
        assert x_test_processed_2d.shape[0] == y_test_prop.shape[0], \
            f"Processed x_test ({x_test_processed_2d.shape[0]}) and y_test property ({y_test_prop.shape[0]}) row count mismatch for {dataset_name}"

        if x_train_processed_2d.shape[0] > 0 and x_test_processed_2d.shape[0] > 0:
            assert x_train_processed_2d.shape[1] == x_test_processed_2d.shape[1], \
                f"Processed x_train features ({x_train_processed_2d.shape[1]}) and processed x_test features ({x_test_processed_2d.shape[1]}) mismatch for {dataset_name}"
    else:  # x_test is indices
        assert ds_obj.x_test.shape[0] > 0, f"x_test (indices) for {dataset_name} should not be empty"

    # --- "Underlying" raw data checks (using _x_train, _y_train, etc. from ds_obj) ---
    # Mapping x_cal -> _x_train, y_cal -> _y_train
    assert hasattr(ds_obj, '_x_train'), f"Dataset for {dataset_name} should have _x_train (raw x_cal)"
    assert ds_obj._x_train is not None, f"_x_train (raw x_cal) for {dataset_name} should not be None"
    assert isinstance(ds_obj._x_train, np.ndarray), f"_x_train (raw x_cal) for {dataset_name} should be a numpy array"
    assert ds_obj._x_train.shape[1] > 0, f"_x_train (raw x_cal) samples for {dataset_name} should be > 0"

    assert hasattr(ds_obj, '_y_train'), f"Dataset for {dataset_name} should have _y_train (raw y_cal)"
    assert ds_obj._y_train is not None, f"_y_train (raw y_cal) for {dataset_name} should not be None"
    assert isinstance(ds_obj._y_train, np.ndarray), f"_y_train (raw y_cal) for {dataset_name} should be a numpy array"
    assert ds_obj._y_train.shape[0] > 0, f"_y_train (raw y_cal) for {dataset_name} should not be empty"
    assert ds_obj._x_train.shape[1] == ds_obj._y_train.shape[0], \
        f"Raw _x_train samples ({ds_obj._x_train.shape[1]}) and raw _y_train samples ({ds_obj._y_train.shape[0]}) mismatch for {dataset_name} (cal)"

    # Mapping x_val -> _x_test, y_val -> _y_test
    assert hasattr(ds_obj, '_x_test'), f"Dataset for {dataset_name} should have _x_test (raw x_val)"
    assert ds_obj._x_test is not None, f"_x_test (raw x_val) for {dataset_name} should not be None"
    assert isinstance(ds_obj._x_test, np.ndarray), f"_x_test (raw x_val) for {dataset_name} should be a numpy array"

    if not ds_obj.test_is_indices:
        assert ds_obj._x_test.shape[1] > 0, f"_x_test (raw x_val) samples for {dataset_name} should be > 0"
        assert hasattr(ds_obj, '_y_test'), f"Dataset for {dataset_name} should have _y_test (raw y_val)"
        assert ds_obj._y_test is not None, f"_y_test (raw y_val) for {dataset_name} should not be None"
        assert isinstance(ds_obj._y_test, np.ndarray), f"_y_test (raw y_val) for {dataset_name} should be a numpy array"
        assert ds_obj._y_test.shape[0] > 0, f"_y_test (raw y_val) for {dataset_name} should not be empty"
        assert ds_obj._x_test.shape[1] == ds_obj._y_test.shape[0], \
            f"Raw _x_test samples ({ds_obj._x_test.shape[1]}) and raw _y_test samples ({ds_obj._y_test.shape[0]}) mismatch for {dataset_name} (val)"

        # Feature count comparison for raw X data
        if ds_obj._x_train.shape[1] > 0 and ds_obj._x_test.shape[1] > 0:  # If there are samples
            # Get feature count from 2D representation of one augmentation slice
            raw_x_train_features = ds_obj.x_train_(union_type='concat', disable_augmentation=True).shape[1]
            raw_x_test_features = ds_obj.x_test_(union_type='concat', disable_augmentation=True).shape[1]
            assert raw_x_train_features == raw_x_test_features, \
                f"Raw x_train feature count ({raw_x_train_features}) and raw x_test feature count ({raw_x_test_features}) mismatch for {dataset_name}"
    else:  # _x_test is indices
        assert ds_obj._x_test.shape[0] > 0, f"_x_test (indices) for {dataset_name} should not be empty (val)"

@pytest.mark.integration
def test_load_binary_data():
    '''Test loading and validity of the binary dataset.'''
    dataset_name = "binary"
    config = Config(dataset=f"sample_data/{dataset_name}",
                    x_pipeline=None, y_pipeline=None, model=None,
                    seed=seed)
    runner = ExperimentRunner(configs=[config], resume_mode="restart")
    results, _, _, _ = runner.run()  # Assuming results[0] is the Dataset object as per traceback
    
    assert results is not None and len(results) > 0, f"Runner should produce results for {dataset_name}"
    dataset_result = results[0]
    check_dataset_validity(dataset_result, dataset_name)

@pytest.mark.integration
def test_load_classification_data():
    '''Test loading and validity of the classification dataset.'''
    dataset_name = "classification"
    config = Config(dataset=f"sample_data/{dataset_name}",
                    x_pipeline=None, y_pipeline=None, model=None,
                    seed=seed)
    runner = ExperimentRunner(configs=[config], resume_mode="restart")
    results, _, _, _ = runner.run()

    assert results is not None and len(results) > 0, f"Runner should produce results for {dataset_name}"
    dataset_result = results[0]
    check_dataset_validity(dataset_result, dataset_name)

@pytest.mark.integration
def test_load_regression_data():
    '''Test loading and validity of the regression dataset.'''
    dataset_name = "regression"
    # Corrected Config instantiation for dictionary with params
    config = Config(dataset={"path": f"sample_data/{dataset_name}", "params": {"has_header": False}},
                    x_pipeline=None, y_pipeline=None, model=None,
                    seed=seed)
    runner = ExperimentRunner(configs=[config], resume_mode="restart")
    results, _, _, _ = runner.run()
    
    assert results is not None and len(results) > 0, f"Runner should produce results for {dataset_name}"
    dataset_result = results[0]
    check_dataset_validity(dataset_result, dataset_name)
