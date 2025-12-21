"""
NumPy file loader implementation.

This module provides the NumpyLoader class for loading NumPy array files,
including .npy (single array) and .npz (multiple arrays) formats.
"""

from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import (
    FileLoadError,
    FileLoader,
    LoaderResult,
    register_loader,
)


@register_loader
class NumpyLoader(FileLoader):
    """Loader for NumPy array files.

    Supports:
    - Single array files (.npy)
    - Multi-array archives (.npz)

    Parameters:
        allow_pickle: Whether to allow loading pickled objects (default: False).
            Setting this to True may pose a security risk with untrusted files.
        key: For .npz files, the key of the array to load.
            If not specified, uses the first array.
        header_unit: Unit for generated headers ('cm-1', 'nm', 'index', etc.)

    Security Note:
        NumPy's allow_pickle=True can execute arbitrary code when loading
        untrusted files. Only enable this for files you trust completely.
    """

    supported_extensions: ClassVar[Tuple[str, ...]] = (".npy", ".npz")
    name: ClassVar[str] = "NumPy Loader"
    priority: ClassVar[int] = 40  # Higher priority than CSV for numpy files

    @classmethod
    def supports(cls, path: Path) -> bool:
        """Check if this loader supports the given file."""
        return path.suffix.lower() in cls.supported_extensions

    def load(
        self,
        path: Path,
        allow_pickle: bool = False,
        key: Optional[str] = None,
        header_unit: str = "index",
        data_type: str = "x",
        **params: Any,
    ) -> LoaderResult:
        """Load data from a NumPy file.

        Args:
            path: Path to the NumPy file.
            allow_pickle: Whether to allow loading pickled objects.
            key: For .npz files, the key of the array to load.
            header_unit: Unit type for generated headers.
            data_type: Type of data ('x', 'y', or 'metadata').
            **params: Additional parameters (ignored).

        Returns:
            LoaderResult with the loaded data as a DataFrame.
        """
        report: Dict[str, Any] = {
            "file_path": str(path),
            "format": "npy" if path.suffix.lower() == ".npy" else "npz",
            "allow_pickle": allow_pickle,
            "key_used": key,
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

            # Load the array
            array = self._load_array(path, allow_pickle, key, report)

            if array is None:
                return LoaderResult(report=report, header_unit=header_unit)

            # Ensure 2D array
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            elif array.ndim > 2:
                report["warnings"].append(
                    f"Array has {array.ndim} dimensions. Reshaping to 2D."
                )
                array = array.reshape(array.shape[0], -1)

            report["initial_shape"] = array.shape

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

            # Handle NA values
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
            report["error"] = f"Error loading NumPy file: {e}\n{traceback.format_exc()}"
            return LoaderResult(report=report, header_unit=header_unit)

    def _load_array(
        self,
        path: Path,
        allow_pickle: bool,
        key: Optional[str],
        report: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        """Load array from .npy or .npz file.

        Args:
            path: Path to the file.
            allow_pickle: Whether to allow pickled objects.
            key: For .npz, the array key to load.
            report: Report dict to update.

        Returns:
            Loaded numpy array or None on error.
        """
        suffix = path.suffix.lower()

        if suffix == ".npy":
            try:
                array = np.load(path, allow_pickle=allow_pickle)
                report["format_details"] = {
                    "type": "npy",
                    "dtype": str(array.dtype),
                }
                return array
            except Exception as e:
                if "allow_pickle" in str(e).lower():
                    report["error"] = (
                        f"Cannot load pickled array: {e}. "
                        f"Set allow_pickle=True if you trust this file."
                    )
                else:
                    report["error"] = f"Failed to load .npy file: {e}"
                return None

        elif suffix == ".npz":
            try:
                npz_file = np.load(path, allow_pickle=allow_pickle)

                available_keys = list(npz_file.keys())
                report["format_details"] = {
                    "type": "npz",
                    "available_keys": available_keys,
                }

                if not available_keys:
                    report["error"] = f"No arrays found in .npz file: {path}"
                    return None

                # Select the array to use
                if key is not None:
                    if key not in available_keys:
                        report["error"] = (
                            f"Key '{key}' not found in .npz file. "
                            f"Available keys: {available_keys}"
                        )
                        return None
                    selected_key = key
                else:
                    # Use first key
                    selected_key = available_keys[0]
                    if len(available_keys) > 1:
                        report["warnings"].append(
                            f"Multiple arrays in .npz file. Using '{selected_key}'. "
                            f"Specify 'key' parameter to choose a specific array."
                        )

                report["key_used"] = selected_key
                array = npz_file[selected_key]
                report["format_details"]["selected_dtype"] = str(array.dtype)

                return array

            except Exception as e:
                if "allow_pickle" in str(e).lower():
                    report["error"] = (
                        f"Cannot load pickled array: {e}. "
                        f"Set allow_pickle=True if you trust this file."
                    )
                else:
                    report["error"] = f"Failed to load .npz file: {e}"
                return None

        else:
            report["error"] = f"Unsupported NumPy format: {suffix}"
            return None


def load_numpy(
    path,
    allow_pickle: bool = False,
    key: Optional[str] = None,
    header_unit: str = "index",
    **params,
):
    """Load a NumPy file.

    Convenience function for backward compatibility.

    Args:
        path: Path to the NumPy file.
        allow_pickle: Whether to allow pickled objects.
        key: For .npz files, the array key to load.
        header_unit: Unit type for generated headers.
        **params: Additional parameters.

    Returns:
        Tuple of (DataFrame, report, na_mask, headers, header_unit).
    """
    loader = NumpyLoader()
    result = loader.load(
        Path(path),
        allow_pickle=allow_pickle,
        key=key,
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
