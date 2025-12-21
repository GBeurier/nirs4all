"""
MATLAB file loader implementation.

This module provides the MatlabLoader class for loading MATLAB .mat files,
supporting both older (v4, v6, v7) and newer (v7.3 HDF5) MATLAB file formats.
"""

from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import (
    FileLoadError,
    FileLoader,
    LoaderResult,
    register_loader,
)


def _check_scipy_available() -> bool:
    """Check if scipy.io is available."""
    try:
        import scipy.io
        return True
    except ImportError:
        return False


def _check_h5py_available() -> bool:
    """Check if h5py is available (for v7.3 MAT files)."""
    try:
        import h5py
        return True
    except ImportError:
        return False


@register_loader
class MatlabLoader(FileLoader):
    """Loader for MATLAB .mat files.

    Supports:
    - MATLAB v4, v6, v7 files via scipy.io
    - MATLAB v7.3 (HDF5) files via h5py (if available)

    Parameters:
        variable: Name of the variable to load. If None, auto-detects.
        squeeze_me: Squeeze unit matrix dimensions (default: True).
        struct_as_record: Load MATLAB structs as numpy record arrays (default: False).
        header_unit: Unit for generated headers ('index', 'cm-1', 'nm', etc.)

    Example:
        >>> loader = MatlabLoader()
        >>> result = loader.load(
        ...     Path("data.mat"),
        ...     variable="X",
        ... )
    """

    supported_extensions: ClassVar[Tuple[str, ...]] = (".mat",)
    name: ClassVar[str] = "MATLAB Loader"
    priority: ClassVar[int] = 45

    @classmethod
    def supports(cls, path: Path) -> bool:
        """Check if this loader supports the given file."""
        return path.suffix.lower() in cls.supported_extensions

    def load(
        self,
        path: Path,
        variable: Optional[str] = None,
        squeeze_me: bool = True,
        struct_as_record: bool = False,
        header_unit: str = "index",
        data_type: str = "x",
        **params: Any,
    ) -> LoaderResult:
        """Load data from a MATLAB .mat file.

        Args:
            path: Path to the MATLAB file.
            variable: Name of the variable to load. If None, auto-detects.
            squeeze_me: Squeeze unit matrix dimensions.
            struct_as_record: Load structs as record arrays.
            header_unit: Unit type for generated headers.
            data_type: Type of data ('x', 'y', or 'metadata').
            **params: Additional parameters.

        Returns:
            LoaderResult with the loaded data.
        """
        report: Dict[str, Any] = {
            "file_path": str(path),
            "format": "matlab",
            "mat_version": None,
            "variable_requested": variable,
            "variable_used": None,
            "variables_available": None,
            "initial_shape": None,
            "final_shape": None,
            "na_handling": {
                "strategy": "remove",
                "na_detected": False,
                "nb_removed_rows": 0,
                "removed_rows_indices": [],
            },
            "warnings": [],
            "error": None,
        }

        try:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            # Check if scipy is available
            if not _check_scipy_available():
                report["error"] = (
                    "scipy is required for MATLAB files. Install it with: "
                    "pip install scipy"
                )
                return LoaderResult(report=report, header_unit=header_unit)

            import scipy.io

            # Try loading the file
            try:
                mat_contents = scipy.io.loadmat(
                    path,
                    squeeze_me=squeeze_me,
                    struct_as_record=struct_as_record,
                )
                report["mat_version"] = "v7 or earlier"
            except NotImplementedError:
                # v7.3 files use HDF5 format
                if not _check_h5py_available():
                    report["error"] = (
                        "This is a MATLAB v7.3 (HDF5) file. h5py is required. "
                        "Install it with: pip install h5py"
                    )
                    return LoaderResult(report=report, header_unit=header_unit)

                # Load with h5py
                mat_contents = self._load_v73(path)
                report["mat_version"] = "v7.3 (HDF5)"

            except Exception as e:
                report["error"] = f"Failed to load MATLAB file: {e}"
                return LoaderResult(report=report, header_unit=header_unit)

            # Filter out MATLAB metadata variables
            data_variables = {
                k: v for k, v in mat_contents.items()
                if not k.startswith("__") and isinstance(v, np.ndarray)
            }

            report["variables_available"] = list(data_variables.keys())

            if not data_variables:
                report["error"] = "No data variables found in MATLAB file."
                return LoaderResult(report=report, header_unit=header_unit)

            # Select variable to use
            if variable is not None:
                if variable not in data_variables:
                    report["error"] = (
                        f"Variable '{variable}' not found. "
                        f"Available: {list(data_variables.keys())}"
                    )
                    return LoaderResult(report=report, header_unit=header_unit)
                selected_var = variable
            else:
                # Auto-select: prefer 'X', 'data', or first numeric array
                preferred_names = ["X", "x", "data", "Data", "spectra", "Spectra"]
                selected_var = None

                for name in preferred_names:
                    if name in data_variables:
                        selected_var = name
                        break

                if selected_var is None:
                    # Use first suitable variable
                    for name, arr in data_variables.items():
                        if arr.ndim >= 1 and np.issubdtype(arr.dtype, np.number):
                            selected_var = name
                            break

                if selected_var is None:
                    selected_var = list(data_variables.keys())[0]

                if len(data_variables) > 1:
                    report["warnings"].append(
                        f"Multiple variables available. Using '{selected_var}'. "
                        f"Specify 'variable' to choose a different one."
                    )

            report["variable_used"] = selected_var
            array = data_variables[selected_var]

            # Ensure 2D array
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            elif array.ndim > 2:
                report["warnings"].append(
                    f"Array has {array.ndim} dimensions. Reshaping to 2D."
                )
                array = array.reshape(array.shape[0], -1)

            # Handle MATLAB column-major order if needed
            # (scipy.io.loadmat usually handles this, but check shape)
            if array.shape[0] < array.shape[1] and array.shape[0] < 10:
                # Might be transposed - log a warning
                report["warnings"].append(
                    f"Array shape is {array.shape}. If this seems wrong, "
                    f"the data might need to be transposed."
                )

            report["initial_shape"] = array.shape

            # Convert to float if not already
            try:
                array = array.astype(np.float64)
            except (ValueError, TypeError) as e:
                report["error"] = f"Cannot convert array to numeric: {e}"
                return LoaderResult(report=report, header_unit=header_unit)

            # Generate column headers
            n_cols = array.shape[1]
            if header_unit == "index":
                headers = [str(i) for i in range(n_cols)]
            else:
                headers = [f"feature_{i}" for i in range(n_cols)]

            # Convert to DataFrame
            try:
                data = pd.DataFrame(array, columns=headers)
            except Exception as e:
                report["error"] = f"Failed to convert array to DataFrame: {e}"
                return LoaderResult(report=report, header_unit=header_unit)

            # Handle NA values (NaN, Inf)
            # Replace Inf with NaN for consistent handling
            data.replace([np.inf, -np.inf], np.nan, inplace=True)

            na_mask = data.isna().any(axis=1)
            report["na_handling"]["na_detected"] = bool(na_mask.any())

            if na_mask.any():
                report["na_handling"]["nb_removed_rows"] = int(na_mask.sum())
                report["na_handling"]["removed_rows_indices"] = data.index[na_mask].tolist()
                data = data[~na_mask].copy()

            report["final_shape"] = data.shape

            return LoaderResult(
                data=data,
                report=report,
                na_mask=na_mask,
                headers=headers,
                header_unit=header_unit,
            )

        except FileNotFoundError as e:
            report["error"] = str(e)
            return LoaderResult(report=report, header_unit=header_unit)
        except Exception as e:
            import traceback
            report["error"] = f"Error loading MATLAB file: {e}\n{traceback.format_exc()}"
            return LoaderResult(report=report, header_unit=header_unit)

    def _load_v73(self, path: Path) -> Dict[str, np.ndarray]:
        """Load a MATLAB v7.3 (HDF5) file.

        Args:
            path: Path to the file.

        Returns:
            Dictionary of variable name to numpy array.
        """
        import h5py

        result = {}

        def extract_data(name: str, obj):
            """Recursively extract data from HDF5 groups."""
            if isinstance(obj, h5py.Dataset):
                # Read dataset and transpose (MATLAB stores column-major)
                data = obj[()]
                if isinstance(data, np.ndarray) and data.ndim == 2:
                    data = data.T
                result[name] = data

        with h5py.File(path, "r") as f:
            # Walk through all items
            for key in f.keys():
                if not key.startswith("#"):  # Skip HDF5 refs
                    item = f[key]
                    if isinstance(item, h5py.Dataset):
                        data = item[()]
                        if isinstance(data, np.ndarray) and data.ndim == 2:
                            data = data.T
                        result[key] = data
                    elif isinstance(item, h5py.Group):
                        # For structs, just get the first suitable array
                        for subkey in item.keys():
                            subitem = item[subkey]
                            if isinstance(subitem, h5py.Dataset):
                                data = subitem[()]
                                if isinstance(data, np.ndarray):
                                    if data.ndim == 2:
                                        data = data.T
                                    result[f"{key}/{subkey}"] = data

        return result


def load_matlab(
    path,
    variable: Optional[str] = None,
    squeeze_me: bool = True,
    header_unit: str = "index",
    **params,
):
    """Load a MATLAB .mat file.

    Convenience function for direct use.

    Args:
        path: Path to the MATLAB file.
        variable: Name of the variable to load.
        squeeze_me: Squeeze unit dimensions.
        header_unit: Unit type for headers.
        **params: Additional parameters.

    Returns:
        Tuple of (DataFrame, report, na_mask, headers, header_unit).
    """
    loader = MatlabLoader()
    result = loader.load(
        Path(path),
        variable=variable,
        squeeze_me=squeeze_me,
        header_unit=header_unit,
        **params,
    )

    return (
        result.data,
        result.report,
        result.na_mask,
        result.headers,
        result.header_unit,
    )
