# dataset_loader.py

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from nirs4all.data.config_parser import parse_config
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.loaders.csv_loader import load_csv
from nirs4all.data.signal_type import SignalType, normalize_signal_type
from typing import Any, Dict, List, Tuple, Union, Optional

# Import the new loader system
from nirs4all.data.loaders.base import LoaderRegistry, FormatNotSupportedError


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


def _load_file_with_registry(
    file_path: Union[str, Path],
    header_unit: str = "cm-1",
    data_type: str = "x",
    **params: Any,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any], Optional[pd.Series], List[str], str]:
    """Load a file using the LoaderRegistry for format detection.

    This function provides automatic format detection and loading using
    the registered file loaders. It falls back to CSV loading for unknown formats.

    Args:
        file_path: Path to the file to load.
        header_unit: Unit for headers ('cm-1', 'nm', etc.).
        data_type: Type of data ('x', 'y', or 'metadata').
        **params: Additional loading parameters.

    Returns:
        Tuple of (DataFrame, report, na_mask, headers, header_unit).
    """
    path = Path(file_path) if isinstance(file_path, str) else file_path

    # Try to use the registry for format detection
    try:
        registry = LoaderRegistry.get_instance()
        loader = registry.get_loader(path)
        result = loader.load(
            path,
            header_unit=header_unit,
            data_type=data_type,
            **params,
        )
        return (
            result.data,
            result.report,
            result.na_mask,
            result.headers,
            result.header_unit,
        )
    except FormatNotSupportedError:
        # Fall back to CSV loader for unknown formats
        return load_csv(file_path, header_unit=header_unit, data_type=data_type, **params)
    except Exception as e:
        # On any other error, try CSV as a fallback
        try:
            return load_csv(file_path, header_unit=header_unit, data_type=data_type, **params)
        except Exception:
            # If CSV also fails, re-raise the original error
            raise e


def load_XY(x_path, x_filter, x_params, y_path, y_filter, y_params, m_path=None, m_filter=None, m_params=None):
    """
    Load X, Y, and metadata from single paths. For multi-source, this will be called multiple times.

    Parameters:
    - x_path (str): Single path to X data file.
    - x_filter: Filter to apply to X data (not implemented yet).
    - x_params (dict): Parameters for loading X data, including:
        - header_unit: Unit for headers ("cm-1", "nm", "none", "text", "index")
        - signal_type: Signal type ("absorbance", "reflectance", "reflectance%", etc.)
        - delimiter, decimal_separator, has_header, na_policy, etc.
    - y_path (str): Path to the Y data file (can be None).
    - y_filter: Filter to apply to Y data (or indices if y_path is None).
    - y_params (dict): Parameters for loading Y data.
    - m_path (str): Path to metadata file (can be None).
    - m_filter: Filter to apply to metadata (not implemented yet).
    - m_params (dict): Parameters for loading metadata.

    Returns:
    - tuple: (x, y, m, x_headers, m_headers, x_header_unit, x_signal_type) where:
        - x, y, m are numpy arrays/DataFrames
        - x_headers, m_headers are lists of column names
        - x_header_unit is the unit string for X headers ("cm-1", "nm", "none", "text", "index")
        - x_signal_type is the signal type (SignalType enum or None for auto-detect)

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

    # Extract header_unit from params (default to cm-1)
    x_header_unit = x_params.pop('header_unit', 'cm-1')

    # Extract signal_type from params (default to None for auto-detect)
    x_signal_type_raw = x_params.pop('signal_type', None)
    x_signal_type: Optional[SignalType] = None
    if x_signal_type_raw is not None:
        x_signal_type = normalize_signal_type(x_signal_type_raw)

    # Load X data using format-aware loader
    try:
        x_df, x_report, x_na_mask, x_headers, x_unit = _load_file_with_registry(
            x_path, header_unit=x_header_unit, **x_params
        )
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

            y_df, y_report, y_na_mask, _, _ = _load_file_with_registry(y_path, **y_params_copy)
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

    # Load metadata if provided
    m_df = pd.DataFrame()
    m_headers = []
    if m_path is not None:
        try:
            if m_params is None:
                m_params = {}
            m_params_copy = m_params.copy()
            if 'categorical_mode' not in m_params_copy:
                m_params_copy['categorical_mode'] = 'preserve'  # Keep original types for metadata
            if 'data_type' not in m_params_copy:
                m_params_copy['data_type'] = 'metadata'
            # Use 'remove' policy but we'll ignore the removed rows for metadata
            # (we want to keep all metadata rows even if some columns have NAs)
            if 'na_policy' not in m_params_copy:
                m_params_copy['na_policy'] = 'remove'

            m_df_temp, m_report, m_na_mask, m_headers, _ = _load_file_with_registry(m_path, **m_params_copy)

            # For metadata, we want to keep ALL rows including those with NAs
            # So we reload the data without NA row removal if rows were removed
            if m_report.get('na_handling', {}).get('nb_removed_rows', 0) > 0:
                # Rows were removed - reload with explicit na_filter=False to keep everything
                m_params_no_na_removal = m_params_copy.copy()
                # We can't directly disable NA removal in load_csv, so we use pandas directly
                import csv
                read_csv_kwargs = {
                    'sep': m_report['delimiter'],
                    'decimal': m_report['decimal_separator'],
                    'header': 0 if m_report['has_header'] else None,
                    'na_filter': True,  # Still detect NAs but don't remove them
                    'keep_default_na': True,
                    'engine': 'python',
                }
                m_df = pd.read_csv(m_path, **read_csv_kwargs)
                m_df.columns = m_df.columns.astype(str)
                m_headers = m_df.columns.tolist()
            else:
                m_df = m_df_temp

            if m_report.get("error") is not None or m_df is None:
                raise ValueError(f"Failed to load metadata from {m_path}: {m_report.get('error', 'Unknown error')}")
        except Exception as e:
            raise ValueError(f"Error loading metadata from {m_path}: {str(e)}")

        if m_filter is not None:
            raise NotImplementedError("Metadata filtering not implemented yet")

        # Ensure metadata has same number of rows as X
        if not m_df.empty and x_df.shape[0] != m_df.shape[0]:
            raise ValueError(f"Row count mismatch: X({x_df.shape[0]}) Metadata({m_df.shape[0]})")

    # Update x_headers after potential column removal (if Y was extracted from X)
    x_headers = x_df.columns.tolist()

    # Convert to numpy arrays
    try:
        x = x_df.astype(np.float32).values if not x_df.empty else np.empty((0, x_df.shape[1]), dtype=np.float32)
        y = y_df.values if not y_df.empty else np.empty((x_df.shape[0], 0))  # Match X rows but 0 columns
        # Keep metadata as DataFrame (don't convert to numeric)
        m = m_df if not m_df.empty else None
    except Exception as e:
        raise ValueError(f"Error converting data to numpy arrays: {str(e)}")

    return x, y, m, x_headers, m_headers, x_unit, x_signal_type


def handle_data(config, t_set):
    """
    Handle data loading for a given dataset type (train, test).
    Supports both single-source and multi-source datasets.

    Parameters:
    - config (dict): Data configuration dictionary.
    - t_set (str): The dataset type ('train', 'test').

    Returns:
    - tuple: (x, y, m, x_headers, m_headers, x_header_unit, x_signal_type) where:
        - x is numpy array or list of arrays
        - y is numpy array
        - m is DataFrame or None (metadata)
        - x_headers is list of column names or list of lists for multi-source
        - m_headers is list of metadata column names
        - x_header_unit is string or list of strings for multi-source ("cm-1", "nm", "none", "text", "index")
        - x_signal_type is SignalType or list of SignalType for multi-source (None for auto-detect)
    """
    if config is None:
        raise ValueError(f"Configuration for {t_set} dataset is None")

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config type for {t_set}: {type(config)}")

    # Get paths
    x_path = config.get(f'{t_set}_x')
    y_path = config.get(f'{t_set}_y')
    m_path = config.get(f'{t_set}_group')  # Metadata uses 'group' key

    # Check if we already have numpy arrays (not file paths)
    if isinstance(x_path, np.ndarray):
        # Data is already loaded as numpy arrays
        x_array = x_path
        y_array = y_path if isinstance(y_path, np.ndarray) else None
        m_data = m_path if isinstance(m_path, (pd.DataFrame, np.ndarray)) else None

        # Generate simple headers
        if isinstance(x_array, np.ndarray):
            x_headers = [f"feature_{i}" for i in range(x_array.shape[1] if x_array.ndim > 1 else 1)]
        else:
            x_headers = []

        m_headers = []
        if isinstance(m_data, pd.DataFrame):
            m_headers = list(m_data.columns)
        elif isinstance(m_data, np.ndarray) and m_data.ndim > 1:
            m_headers = [f"meta_{i}" for i in range(m_data.shape[1])]

        # For pre-loaded arrays, use defaults or config values
        from nirs4all.data._features import HeaderUnit
        x_header_unit = HeaderUnit.WAVENUMBER.value

        # Check for signal_type in config params for pre-loaded arrays
        x_params = config.get(f'{t_set}_x_params') or config.get('global_params') or {}
        x_signal_type = None
        if 'signal_type' in x_params:
            x_signal_type = normalize_signal_type(x_params['signal_type'])

        return x_array, y_array, m_data, x_headers, m_headers, x_header_unit, x_signal_type

    x_filter = config.get(f'{t_set}_x_filter')
    y_filter = config.get(f'{t_set}_y_filter')
    m_filter = config.get(f'{t_set}_group_filter')

    # Handle multi-source X data
    if isinstance(x_path, list):
        x_arrays = []
        headers_arrays = []
        header_units = []
        signal_types = []
        y_array = None
        m_data = None
        m_headers = []

        # Check if we have per-source params
        x_params_config = config.get(f'{t_set}_x_params')

        for i, single_x_path in enumerate(x_path):
            # Determine params for this source
            if isinstance(x_params_config, list) and i < len(x_params_config):
                # Per-source params provided
                source_x_params = _merge_params(x_params_config[i], config.get(f'{t_set}_params'), config.get('global_params'))
            elif isinstance(x_params_config, dict):
                # Check if dict contains list of units or signal_types for multi-source
                source_params = x_params_config.copy()

                # Handle header_unit list
                if 'header_unit' in x_params_config and isinstance(x_params_config['header_unit'], list):
                    if i < len(x_params_config['header_unit']):
                        source_params['header_unit'] = x_params_config['header_unit'][i]
                    else:
                        source_params['header_unit'] = "cm-1"

                # Handle signal_type list
                if 'signal_type' in x_params_config and isinstance(x_params_config['signal_type'], list):
                    if i < len(x_params_config['signal_type']):
                        source_params['signal_type'] = x_params_config['signal_type'][i]
                    else:
                        source_params['signal_type'] = None

                source_x_params = _merge_params(source_params, config.get(f'{t_set}_params'), config.get('global_params'))
            else:
                # No params or unsupported format
                source_x_params = _merge_params(None, config.get(f'{t_set}_params'), config.get('global_params'))

            y_params = _merge_params(config.get(f'{t_set}_y_params'), config.get(f'{t_set}_params'), config.get('global_params'))
            m_params = _merge_params(config.get(f'{t_set}_group_params'), config.get(f'{t_set}_params'), config.get('global_params'))

            try:
                # For multi-source, only the first source should handle Y and metadata extraction
                if i == 0:
                    x_single, y_array, m_data, x_headers, m_headers, x_unit, x_sig_type = load_XY(
                        single_x_path, x_filter, source_x_params,
                        y_path, y_filter, y_params,
                        m_path, m_filter, m_params
                    )
                else:
                    # For additional sources, don't extract Y or metadata
                    x_single, _, _, x_headers, _, x_unit, x_sig_type = load_XY(
                        single_x_path, x_filter, source_x_params,
                        None, None, y_params,
                        None, None, None
                    )

                x_arrays.append(x_single)
                headers_arrays.append(x_headers)
                header_units.append(x_unit)
                signal_types.append(x_sig_type)
            except Exception as e:
                raise ValueError(f"Error loading X source {i} from {single_x_path}: {str(e)}")

        return x_arrays, y_array, m_data, headers_arrays, m_headers, header_units, signal_types
    else:
        # Single source
        x_params = _merge_params(config.get(f'{t_set}_x_params'), config.get(f'{t_set}_params'), config.get('global_params'))
        y_params = _merge_params(config.get(f'{t_set}_y_params'), config.get(f'{t_set}_params'), config.get('global_params'))
        m_params = _merge_params(config.get(f'{t_set}_group_params'), config.get(f'{t_set}_params'), config.get('global_params'))
        return load_XY(x_path, x_filter, x_params, y_path, y_filter, y_params, m_path, m_filter, m_params)
