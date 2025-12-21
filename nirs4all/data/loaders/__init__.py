"""
File loaders module for nirs4all.

This module provides a pluggable file loading system supporting multiple file formats
with automatic format detection and configurable loading parameters.

Supported Formats:
    - CSV (.csv, .csv.gz, .csv.zip) - via CSVLoader
    - NumPy (.npy, .npz) - via NumpyLoader
    - Parquet (.parquet, .pq) - via ParquetLoader (requires pyarrow or fastparquet)
    - Excel (.xlsx, .xls) - via ExcelLoader (requires openpyxl/xlrd)
    - MATLAB (.mat) - via MatlabLoader (requires scipy, optionally h5py)
    - Archives (.tar, .tar.gz, .tgz, .zip) - via TarLoader, EnhancedZipLoader

Usage:
    >>> from nirs4all.data.loaders import LoaderRegistry, load_file
    >>>
    >>> # Using the registry
    >>> registry = LoaderRegistry.get_instance()
    >>> result = registry.load("data.csv", delimiter=",")
    >>>
    >>> # Or using the convenience function
    >>> data, report, na_mask, headers, unit = load_file("data.csv")
    >>>
    >>> # Direct loader usage
    >>> from nirs4all.data.loaders import CSVLoader
    >>> loader = CSVLoader()
    >>> result = loader.load(Path("data.csv"))

Adding Custom Loaders:
    >>> from nirs4all.data.loaders import FileLoader, register_loader
    >>>
    >>> @register_loader
    ... class MyLoader(FileLoader):
    ...     supported_extensions = (".myext",)
    ...     name = "My Loader"
    ...
    ...     @classmethod
    ...     def supports(cls, path):
    ...         return path.suffix.lower() == ".myext"
    ...
    ...     def load(self, path, **params):
    ...         # Load implementation
    ...         pass

Backward Compatibility:
    The legacy load_csv function is still available for existing code:
    >>> from nirs4all.data.loaders.csv_loader import load_csv
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pandas as pd

# Base classes and utilities
from .base import (
    ArchiveHandler,
    FileLoadError,
    FileLoader,
    FormatNotSupportedError,
    LoaderError,
    LoaderRegistry,
    LoaderResult,
    register_loader,
)

# Format-specific loaders
# Note: Importing these modules automatically registers them via @register_loader
from .csv_loader_new import CSVLoader, load_csv as load_csv_new
from .numpy_loader import NumpyLoader, load_numpy
from .parquet_loader import ParquetLoader, load_parquet
from .excel_loader import ExcelLoader, load_excel
from .matlab_loader import MatlabLoader, load_matlab
from .archive_loader import TarLoader, EnhancedZipLoader, list_archive_members

# Legacy imports for backward compatibility
from .csv_loader import load_csv


def load_file(
    path: Union[str, Path],
    **params: Any,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any], Optional[pd.Series], List[str], str]:
    """Load a data file with automatic format detection.

    This is the main entry point for loading files. It automatically detects
    the file format and uses the appropriate loader.

    Args:
        path: Path to the file to load.
        **params: Format-specific loading parameters. Common parameters include:
            - header_unit: Unit for headers ('cm-1', 'nm', 'text', etc.)
            - data_type: Type of data ('x', 'y', or 'metadata')
            - delimiter: CSV delimiter
            - sheet_name: Excel sheet to load
            - variable: MATLAB variable name
            - member: Archive member to extract

    Returns:
        Tuple of:
            - DataFrame with loaded data (or None on error)
            - Report dictionary with loading metadata
            - NA mask Series (rows with missing values)
            - List of column headers
            - Header unit string

    Raises:
        FormatNotSupportedError: If no loader supports the file format.

    Example:
        >>> data, report, na_mask, headers, unit = load_file("data.csv")
        >>> if report.get("error"):
        ...     print(f"Error: {report['error']}")
        >>> else:
        ...     print(f"Loaded {data.shape[0]} samples with {data.shape[1]} features")
    """
    registry = LoaderRegistry.get_instance()
    result = registry.load(path, **params)

    return (
        result.data,
        result.report,
        result.na_mask,
        result.headers,
        result.header_unit,
    )


def get_supported_formats() -> Dict[str, List[str]]:
    """Get all supported file formats and their extensions.

    Returns:
        Dictionary mapping loader names to their supported extensions.

    Example:
        >>> formats = get_supported_formats()
        >>> for name, exts in formats.items():
        ...     print(f"{name}: {', '.join(exts)}")
    """
    registry = LoaderRegistry.get_instance()
    result = {}

    for loader_class in registry.get_registered_loaders():
        result[loader_class.name] = list(loader_class.supported_extensions)

    return result


def get_loader_for_file(path: Union[str, Path]) -> FileLoader:
    """Get the appropriate loader for a file.

    Args:
        path: Path to the file.

    Returns:
        Instance of the appropriate FileLoader subclass.

    Raises:
        FormatNotSupportedError: If no loader supports the file format.
    """
    registry = LoaderRegistry.get_instance()
    return registry.get_loader(path)


__all__ = [
    # Base classes
    "FileLoader",
    "LoaderResult",
    "LoaderRegistry",
    "ArchiveHandler",
    # Exceptions
    "LoaderError",
    "FileLoadError",
    "FormatNotSupportedError",
    # Decorator
    "register_loader",
    # Loaders
    "CSVLoader",
    "NumpyLoader",
    "ParquetLoader",
    "ExcelLoader",
    "MatlabLoader",
    "TarLoader",
    "EnhancedZipLoader",
    # Convenience functions
    "load_file",
    "load_csv",
    "load_csv_new",
    "load_numpy",
    "load_parquet",
    "load_excel",
    "load_matlab",
    "list_archive_members",
    "get_supported_formats",
    "get_loader_for_file",
]
