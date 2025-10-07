# dataset_loader.py

import hashlib
import json
import numpy as np
import pandas as pd
from nirs4all.dataset.dataset_config_parser import parse_config
from nirs4all.dataset.dataset import SpectroDataset
from nirs4all.dataset.csv_loader import load_csv
from typing import Dict, Tuple, Union


def create_synthetic_dataset(config: Dict) -> SpectroDataset:
    """
    Create a synthetic SpectroDataset for testing purposes.

    Args:
        config: Dictionary with keys:
            - X: Feature matrix (n_samples, n_features)
            - y: Target values (n_samples,)
            - folds: Number of CV folds
            - train/val/test: Split ratios
            - random_state: Random seed

    Returns:
        SpectroDataset: Synthetic dataset ready for pipeline use
    """
    X = config['X']
    y = config['y']

    # Create synthetic dataset object with a proper string name
    dataset = SpectroDataset(name="synthetic_test_dataset")

    # Split the data into train and test partitions
    # Use the ratios from config, defaulting to 80/20 split
    n_samples = X.shape[0]
    train_ratio = config.get('train', 0.8)
    n_train = int(n_samples * train_ratio)

    # Split indices
    indices = np.arange(n_samples)
    if 'random_state' in config:
        np.random.seed(config['random_state'])
        indices = np.random.permutation(indices)

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    # Split data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    # Add samples with partition information
    if len(X_train) > 0:
        dataset.add_samples(X_train, {"partition": "train"})
    if len(X_test) > 0:
        dataset.add_samples(X_test, {"partition": "test"})

    # Add targets
    if len(y_train) > 0:
        dataset.add_targets(y_train)
    if len(y_test) > 0:
        dataset.add_targets(y_test)

    return dataset


def _merge_params(local_params, handler_params, global_params):
    """
    Merge parameters from local, handler, and global scopes.

    Parameters:
    - local_params (dict): Local parameters specific to the data subset.
    - handler_params (dict): Parameters specific to the handler.
    - global_params (dict): Global parameters that apply to all handlers.

    Returns:
    - dict: Merged parameters with precedence: local > handler > global.
    """
    merged_params = {} if global_params is None else global_params.copy()
    if handler_params is not None:
        merged_params.update(handler_params)
    if local_params is not None:
        merged_params.update(local_params)
    return merged_params


def load_XY(x_path, x_filter, x_params, y_path, y_filter, y_params):
    """
    Load X and Y data from single paths. For multi-source, this will be called multiple times.

    Parameters:
    - x_path (str): Single path to X data file.
    - x_filter: Filter to apply to X data (not implemented yet).
    - x_params (dict): Parameters for loading X data.
    - y_path (str): Path to the Y data file (can be None).
    - y_filter: Filter to apply to Y data (or indices if y_path is None).
    - y_params (dict): Parameters for loading Y data.

    Returns:
    - tuple: (x, y, x_headers) where x and y are numpy arrays and x_headers is list of column names.

    Raises:
    - ValueError: If data is invalid or if there are inconsistencies.
    """
    if x_path is None:
        raise ValueError("Invalid x definition: x_path is None")

    # Set default parameters
    if 'categorical_mode' not in x_params:
        x_params['categorical_mode'] = 'auto'
    if 'data_type' not in x_params:
        x_params['data_type'] = 'x'

    # Load X data
    try:
        x_df, x_report, x_na_mask, x_headers = load_csv(x_path, **x_params)
        if x_report.get("error") is not None or x_df is None:
            raise ValueError(f"Failed to load X data from {x_path}: {x_report.get('error', 'Unknown error')}")
    except Exception as e:
        raise ValueError(f"Error loading X data from {x_path}: {str(e)}")

    if x_filter is not None:
        raise NotImplementedError("Auto-filtering not implemented yet")

    # Load Y data
    if y_path is None and y_filter is None:
        # No Y data to extract - create empty Y array with same number of rows as X
        y_df = pd.DataFrame(index=x_df.index)  # Empty DataFrame with matching index
    elif y_path is None:
        # Y is a subset of X
        if not all(isinstance(i, int) for i in y_filter):
            raise ValueError("Invalid y definition: y_filter is not a list of integers")

        if x_df.shape[1] <= max(y_filter):
            raise ValueError(f"Y filter indices {y_filter} exceed X columns ({x_df.shape[1]})")

        # Extract Y from X and remove Y columns from X
        y_df = x_df.iloc[:, y_filter]
        x_df = x_df.drop(x_df.columns[y_filter], axis=1)
    else:
        # Y is in a separate file
        try:
            y_params_copy = y_params.copy()
            if 'categorical_mode' not in y_params_copy:
                y_params_copy['categorical_mode'] = 'auto'
            if 'data_type' not in y_params_copy:
                y_params_copy['data_type'] = 'y'

            y_df, y_report, y_na_mask, _ = load_csv(y_path, **y_params_copy)
            if y_report.get("error") is not None or y_df is None:
                raise ValueError(f"Failed to load Y data from {y_path}: {y_report.get('error', 'Unknown error')}")
        except Exception as e:
            raise ValueError(f"Error loading Y data from {y_path}: {str(e)}")

        if y_filter is not None:
            if not all(isinstance(i, int) for i in y_filter):
                raise ValueError("Invalid y_filter: must be list of integers")
            if y_df.shape[1] <= max(y_filter):
                raise ValueError(f"Y filter indices {y_filter} exceed Y columns ({y_df.shape[1]})")
            y_df = y_df.iloc[:, y_filter]

    # Ensure same number of rows (only check if Y has data)
    if not y_df.empty and x_df.shape[0] != y_df.shape[0]:
        raise ValueError(f"Row count mismatch: X({x_df.shape[0]}) Y({y_df.shape[0]})")

    # Update x_headers after potential column removal (if Y was extracted from X)
    x_headers = x_df.columns.tolist()

    # Convert to numpy arrays
    try:
        x = x_df.astype(np.float32).values if not x_df.empty else np.empty((0, x_df.shape[1]), dtype=np.float32)
        y = y_df.values if not y_df.empty else np.empty((x_df.shape[0], 0))  # Match X rows but 0 columns
    except Exception as e:
        raise ValueError(f"Error converting data to numpy arrays: {str(e)}")

    return x, y, x_headers


def handle_data(config, t_set):
    """
    Handle data loading for a given dataset type (train, test).
    Supports both single-source and multi-source datasets.

    Parameters:
    - config (dict): Data configuration dictionary.
    - t_set (str): The dataset type ('train', 'test').

    Returns:
    - tuple: (x, y, headers) where x is numpy array or list of arrays, y is numpy array,
             headers is list of column names or list of lists for multi-source
    """
    if config is None:
        raise ValueError(f"Configuration for {t_set} dataset is None")

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config type for {t_set}: {type(config)}")

    # Get paths
    x_path = config.get(f'{t_set}_x')
    y_path = config.get(f'{t_set}_y')
    x_filter = config.get(f'{t_set}_x_filter')
    y_filter = config.get(f'{t_set}_y_filter')

    # Merge parameters
    x_params = _merge_params(config.get(f'{t_set}_x_params'), config.get(f'{t_set}_params'), config.get('global_params'))
    y_params = _merge_params(config.get(f'{t_set}_y_params'), config.get(f'{t_set}_params'), config.get('global_params'))

    # Handle multi-source X data
    if isinstance(x_path, list):
        x_arrays = []
        headers_arrays = []
        y_array = None

        for i, single_x_path in enumerate(x_path):
            try:
                # For multi-source, only the first source should handle Y extraction
                if i == 0:
                    x_single, y_array, x_headers = load_XY(single_x_path, x_filter, x_params, y_path, y_filter, y_params)
                else:
                    # For additional sources, don't extract Y (set y_path=None and y_filter=None)
                    x_single, _, x_headers = load_XY(single_x_path, x_filter, x_params, None, None, y_params)

                x_arrays.append(x_single)
                headers_arrays.append(x_headers)
            except Exception as e:
                raise ValueError(f"Error loading X source {i} from {single_x_path}: {str(e)}")

        return x_arrays, y_array, headers_arrays
    else:
        # Single source
        return load_XY(x_path, x_filter, x_params, y_path, y_filter, y_params)
