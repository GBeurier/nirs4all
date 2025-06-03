"""
Data loading functions for nirs4all configuration schema.
Handles CSV file loading with configurable parameters and data selection.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, Tuple, Optional, List
import glob
import os


class DataLoadingError(Exception):
    """Custom exception for data loading errors."""
    pass


def parse_selector(selector: Union[str, Dict, List], data_length: int) -> np.ndarray:
    """
    Parse a selector (string slice, dict with from/to, or list of indices) into numpy indices.

    Args:
        selector: The selector configuration
        data_length: Length of the data to select from

    Returns:
        numpy array of indices
    """
    if isinstance(selector, str):
        # Handle string slice notation like "10:20", ":50", "10:", "::2"
        parts = selector.split(':')
        if len(parts) == 1:
            # Single number
            idx = int(parts[0]) if parts[0].strip() else 0
            return np.array([idx])

        start = int(parts[0]) if parts[0].strip() else None
        stop = int(parts[1]) if len(parts) > 1 and parts[1].strip() else None
        step = int(parts[2]) if len(parts) > 2 and parts[2].strip() else None

        return np.arange(data_length)[start:stop:step]

    elif isinstance(selector, dict):
        # Handle {"from": x, "to": y} format
        start = selector.get('from', 0)
        stop = selector.get('to', data_length)
        return np.arange(start, stop)

    elif isinstance(selector, list):
        # Handle list of indices
        return np.array(selector)

    else:
        raise DataLoadingError(f"Invalid selector type: {type(selector)}")


def get_data_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and normalize data reading parameters.

    Args:
        config: Configuration dictionary that may contain 'params'

    Returns:
        Dictionary with pandas read_csv parameters
    """
    params = config.get('params', {})

    # Set defaults
    csv_params = {
        'sep': params.get('delimiter', ';'),
        'decimal': params.get('decimal', '.'),
        'header': params.get('header', 0),
    }

    # Handle null header
    if csv_params['header'] is None:
        csv_params['header'] = None

    return csv_params


def handle_na_policy(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    """
    Handle missing values according to the specified policy.

    Args:
        df: DataFrame to process
        policy: NA handling policy ('abort', 'remove', 'ignore', 'replace', 'auto')

    Returns:
        Processed DataFrame
    """
    if policy == 'abort' and df.isnull().any().any():
        raise DataLoadingError("Found NaN values and na_policy is 'abort'")

    elif policy == 'remove':
        df = df.dropna()

    elif policy == 'ignore':
        pass  # Do nothing

    elif policy == 'replace':
        df = df.fillna(0.0)

    elif policy == 'auto':
        # Auto policy: remove rows with any NaN
        if df.isnull().any().any():
            df = df.dropna()

    return df


def load_csv_file(config: Union[str, Dict[str, Any]]) -> np.ndarray:
    """
    Load a single CSV file according to configuration.

    Args:
        config: Either a file path string or a configuration dictionary

    Returns:
        numpy array with the loaded data
    """
    if isinstance(config, str):
        # Simple file path
        file_path = config
        csv_params = get_data_params({})
        selector = None
    else:
        # Configuration dictionary
        file_path = config['path']
        csv_params = get_data_params(config)
        selector = config.get('filter')

    # Check file exists
    if not Path(file_path).exists():
        raise DataLoadingError(f"File not found: {file_path}")

    try:
        # Load CSV
        df = pd.read_csv(file_path, **csv_params)

        # Handle NA policy if specified
        na_policy = config.get('params', {}).get('na_policy', 'auto') if isinstance(config, dict) else 'auto'
        df = handle_na_policy(df, na_policy)

        # Convert to numpy array
        data = df.values.astype(float)

        # Apply selector if specified
        if selector is not None:
            if isinstance(selector, (str, dict)) and ('from' in str(selector) or 'to' in str(selector) or ':' in str(selector)):
                # Row selector
                indices = parse_selector(selector, len(data))
                data = data[indices]
            else:
                # Column selector
                indices = parse_selector(selector, data.shape[1])
                data = data[:, indices]

        return data

    except Exception as e:
        raise DataLoadingError(f"Error loading CSV file {file_path}: {str(e)}")


def load_xy_data(xy_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load X and Y data from XY configuration.

    Args:
        xy_config: XY configuration dictionary with 'X' and 'Y' keys

    Returns:
        Tuple of (X_data, Y_data) as numpy arrays
    """
    # Load X data
    X_data = load_csv_file(xy_config['X'])

    # Load Y data
    y_config = xy_config['Y']

    if isinstance(y_config, (str, dict)) and not isinstance(y_config, list):
        # Y is a file configuration
        try:
            Y_data = load_csv_file(y_config)
        except:
            # If Y fails as file, try as selector on X
            indices = parse_selector(y_config, X_data.shape[1])
            Y_data = X_data[:, indices]
            # Remove Y columns from X
            remaining_cols = np.setdiff1d(np.arange(X_data.shape[1]), indices)
            if len(remaining_cols) > 0:
                X_data = X_data[:, remaining_cols]
    else:
        # Y is a selector for columns in X
        indices = parse_selector(y_config, X_data.shape[1])
        Y_data = X_data[:, indices]        # Remove Y columns from X
        remaining_cols = np.setdiff1d(np.arange(X_data.shape[1]), indices)
        if len(remaining_cols) > 0:
            X_data = X_data[:, remaining_cols]

    # Flatten Y_data if it's a single column
    if Y_data.ndim == 2 and Y_data.shape[1] == 1:
        Y_data = Y_data.flatten()

    # Ensure same number of samples
    min_samples = min(len(X_data), len(Y_data))
    if len(X_data) != len(Y_data):
        print(f"Warning: X has {len(X_data)} samples, Y has {len(Y_data)} samples. "
              f"Truncating to {min_samples} samples.")
        X_data = X_data[:min_samples]
        Y_data = Y_data[:min_samples]

    return X_data, Y_data


def load_folder_data(folder_config: Union[str, Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from a folder containing CSV files.
    Assumes folder contains separate X and Y files or combined XY files.

    Args:
        folder_config: Either folder path string or configuration dictionary

    Returns:
        Tuple of (X_data, Y_data) as numpy arrays
    """
    if isinstance(folder_config, str):
        folder_path = folder_config
        csv_params = get_data_params({})
    else:
        folder_path = folder_config['folder']
        csv_params = get_data_params(folder_config)

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise DataLoadingError(f"Folder not found: {folder_path}")

    # Find CSV files
    csv_files = list(folder.glob("*.csv")) + list(folder.glob("*.csv.gz"))

    if not csv_files:
        raise DataLoadingError(f"No CSV files found in folder: {folder_path}")    # Try to identify X and Y files by name
    x_files = [f for f in csv_files if 'x' in f.name.lower() or 'feature' in f.name.lower()]
    y_files = [f for f in csv_files if 'y' in f.name.lower() or 'target' in f.name.lower() or 'label' in f.name.lower()]

    if x_files and y_files:
        # Load separate X and Y files
        X_data = load_csv_file(str(x_files[0]))
        Y_data = load_csv_file(str(y_files[0]))
    else:
        # Load first file and assume it contains both X and Y
        data = load_csv_file(str(csv_files[0]))
        # Assume last column is Y, rest is X
        X_data = data[:, :-1]
        Y_data = data[:, -1:]

    # Flatten Y_data if it's a single column
    if Y_data.ndim == 2 and Y_data.shape[1] == 1:
        Y_data = Y_data.flatten()

    # Ensure same number of samples
    min_samples = min(len(X_data), len(Y_data))
    if len(X_data) != len(Y_data):
        X_data = X_data[:min_samples]
        Y_data = Y_data[:min_samples]

    return X_data, Y_data


def load_data_from_config(config: Union[str, Dict[str, Any]]) -> Union[Tuple[np.ndarray, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Main function to load data from a nirs4all configuration.

    Args:
        config: Either JSON string or dictionary with nirs4all data configuration

    Returns:
        - If single dataset (folder or XY): Tuple of (X_data, Y_data) as numpy arrays
        - If multiple datasets (train/test/valid): Dictionary with dataset names as keys and (X, Y) tuples as values

    Raises:
        DataLoadingError: If configuration is invalid or data cannot be loaded
    """
    # Parse JSON string if needed
    # if isinstance(config, str):
    #     try:
    #         config = json.loads(config)
    #     except json.JSONDecodeError as e:
    #         raise DataLoadingError(f"Invalid JSON configuration: {e}")


    # Check if this is a DATAFILES_SCHEMA (contains train/test/valid)
    dataset_keys = {'train', 'test', 'valid'}
    found_datasets = False if isinstance(config, str) else dataset_keys.intersection(config.keys())

    if found_datasets:
        print("Loading multiple datasets...")
        # DATAFILES_SCHEMA - load multiple datasets
        result = {}
        for dataset_name in found_datasets:
            dataset_config = config[dataset_name]
            result[dataset_name] = load_xy_data(dataset_config)
            print(f"Loaded {dataset_name}: X{result[dataset_name][0].shape}, Y{result[dataset_name][1].shape}")
        return result

    elif 'X' in config and 'Y' in config:
        print("Loading single XY dataset...")
        # Single XY file structure
        return load_xy_data(config)

    elif 'folder' in config or isinstance(config, str):
        # Folder structure
        print("Loading data from folder structure...")
        return load_folder_data(config)

    else:
        raise DataLoadingError("Invalid data configuration structure")


def load_train_test_data(config: Union[str, Dict[str, Any]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load train/test/validation data from configuration.
    This is now an alias for load_data_from_config when dealing with multiple datasets.

    Args:
        config: Configuration with train/test/valid data specifications

    Returns:
        Dictionary with dataset names ('train', 'test', 'valid') as keys containing (X, Y) tuples

    Raises:
        DataLoadingError: If configuration doesn't contain multiple datasets
    """
    result = load_data_from_config(config)

    if isinstance(result, dict):
        return result
    else:
        raise DataLoadingError("Configuration does not contain multiple datasets (train/test/valid). "
                             "Use load_data_from_config() for single datasets.")


def get_available_datasets(config: Union[str, Dict[str, Any]]) -> List[str]:
    """
    Get list of available datasets in the configuration.

    Args:
        config: Configuration dictionary or JSON string

    Returns:
        List of dataset names ('train', 'test', 'valid', or ['single'] for single datasets)
    """
    if isinstance(config, str):
        config = json.loads(config)

    data_config = config['dataset']
    dataset_keys = {'train', 'test', 'valid'}
    found_datasets = list(dataset_keys.intersection(data_config.keys()))

    if found_datasets:
        return sorted(found_datasets)  # Return in consistent order
    else:
        return ['single']  # Single dataset (folder or XY)


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Single dataset configuration
    single_config = {
        "dataset": {
            "X": "/sample_data/Xcal.csv",
            "Y": {"from": 0, "to": 3},
            "params": {
                "delimiter": ";",
                "decimal": ".",
                "na_policy": "auto"
            }
        }
    }

    # Example 2: Multiple datasets configuration
    multi_config = {
        "dataset": {
            "train": {
                "X": "/sample_data/Xcal.csv",
                "Y": "/sample_data/Ycal.csv",
            },
            "test": {
                "X": "/sample_data/Xval.csv",
                "Y": "/sample_data/Yval.csv",
            },
            # "valid": {
            #     "X": "/path/to/valid_features.csv",
            #     "Y": [0, 1, 2]
            # }
        }
    }

    # Example 3: Folder configuration
    folder_config = {
        "dataset": "/sample_data/"
    }

    try:
        print("Data loader functions ready to use!")

        print("# For single dataset:")
        X, Y = load_data_from_config(single_config)
        print(f"Loaded single dataset: X shape {X.shape}, Y shape {Y.shape}")

        print("\n# For multiple datasets:")
        datasets = load_data_from_config(multi_config)
        for name, (X_data, Y_data) in datasets.items():
            print(f"Loaded {name}: X shape {X_data.shape}, Y shape {Y_data.shape}")

        print("\n# For folder data:")
        X_folder, Y_folder = load_data_from_config(folder_config)
        print(f"Loaded folder data: X shape {X_folder.shape}, Y shape {Y_folder.shape}")

    except Exception as e:
        print(f"Example failed (expected with dummy paths): {e}")