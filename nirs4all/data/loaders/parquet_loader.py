"""
Parquet file loader implementation.

This module provides the ParquetLoader class for loading Apache Parquet files.
Requires pyarrow or fastparquet as a dependency.
"""

from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import pandas as pd

from .base import (
    FileLoadError,
    FileLoader,
    LoaderResult,
    register_loader,
)


def _check_parquet_available() -> str:
    """Check if a Parquet engine is available.

    Returns:
        Name of the available engine ('pyarrow' or 'fastparquet').

    Raises:
        ImportError: If no Parquet engine is available.
    """
    try:
        import pyarrow
        return "pyarrow"
    except ImportError:
        pass

    try:
        import fastparquet
        return "fastparquet"
    except ImportError:
        pass

    raise ImportError(
        "No Parquet engine available. Install pyarrow or fastparquet: "
        "pip install pyarrow  # or pip install fastparquet"
    )


@register_loader
class ParquetLoader(FileLoader):
    """Loader for Apache Parquet files.

    Requires pyarrow or fastparquet to be installed.

    Supports:
    - Single Parquet files (.parquet, .pq)
    - Partitioned datasets (directory of parquet files)
    - Column selection for efficient loading

    Parameters:
        columns: List of column names to load (default: all columns).
        engine: Parquet engine to use ('auto', 'pyarrow', or 'fastparquet').
        filters: Row group filters for predicate pushdown (pyarrow only).
        header_unit: Unit for headers ('cm-1', 'nm', 'text', etc.)

    Example:
        >>> loader = ParquetLoader()
        >>> result = loader.load(
        ...     Path("data.parquet"),
        ...     columns=["feature_1", "feature_2"],
        ... )
    """

    supported_extensions: ClassVar[Tuple[str, ...]] = (".parquet", ".pq")
    name: ClassVar[str] = "Parquet Loader"
    priority: ClassVar[int] = 35  # Higher priority for Parquet

    @classmethod
    def supports(cls, path: Path) -> bool:
        """Check if this loader supports the given file."""
        # Check file extension
        if path.suffix.lower() in cls.supported_extensions:
            return True

        # Check if it's a directory with parquet files (partitioned dataset)
        if path.is_dir():
            parquet_files = list(path.glob("*.parquet")) + list(path.glob("*.pq"))
            if parquet_files:
                return True
            # Check for nested partitions
            if list(path.glob("**/*.parquet")) or list(path.glob("**/*.pq")):
                return True

        return False

    def load(
        self,
        path: Path,
        columns: Optional[List[str]] = None,
        engine: str = "auto",
        filters: Optional[List] = None,
        header_unit: str = "text",
        data_type: str = "x",
        **params: Any,
    ) -> LoaderResult:
        """Load data from a Parquet file.

        Args:
            path: Path to the Parquet file or directory.
            columns: List of column names to load. If None, loads all columns.
            engine: Parquet engine ('auto', 'pyarrow', or 'fastparquet').
            filters: Row group filters for predicate pushdown (pyarrow only).
            header_unit: Unit type for headers.
            data_type: Type of data ('x', 'y', or 'metadata').
            **params: Additional parameters passed to read_parquet.

        Returns:
            LoaderResult with the loaded data.
        """
        report: Dict[str, Any] = {
            "file_path": str(path),
            "format": "parquet",
            "engine": None,
            "columns_requested": columns,
            "columns_loaded": None,
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
                raise FileNotFoundError(f"File or directory not found: {path}")

            # Determine engine
            if engine == "auto":
                try:
                    engine = _check_parquet_available()
                except ImportError as e:
                    report["error"] = str(e)
                    return LoaderResult(report=report, header_unit=header_unit)

            report["engine"] = engine

            # Build read_parquet kwargs
            read_kwargs: Dict[str, Any] = {
                "engine": engine,
            }

            if columns is not None:
                read_kwargs["columns"] = columns

            # Filters only work with pyarrow
            if filters is not None:
                if engine == "pyarrow":
                    read_kwargs["filters"] = filters
                else:
                    report["warnings"].append(
                        "Filters are only supported with pyarrow engine. Ignoring."
                    )

            # Add any extra params
            read_kwargs.update(params)

            # Load the data
            try:
                data = pd.read_parquet(path, **read_kwargs)
            except ImportError as e:
                report["error"] = f"Parquet engine not available: {e}"
                return LoaderResult(report=report, header_unit=header_unit)
            except Exception as e:
                report["error"] = f"Failed to read Parquet file: {e}"
                return LoaderResult(report=report, header_unit=header_unit)

            report["initial_shape"] = data.shape
            report["columns_loaded"] = data.columns.tolist()

            # Ensure column names are strings
            data.columns = data.columns.astype(str)

            if data.empty:
                report["warnings"].append("Loaded DataFrame is empty.")
                return LoaderResult(
                    data=pd.DataFrame(),
                    report=report,
                    na_mask=pd.Series(dtype=bool),
                    headers=[],
                    header_unit=header_unit,
                )

            # Type conversion for X data
            if data_type == "x":
                for col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        data[col] = pd.to_numeric(data[col], errors="coerce")

            # Handle NA values
            na_mask = data.isna().any(axis=1)
            report["na_handling"]["na_detected"] = bool(na_mask.any())

            if na_mask.any():
                report["na_handling"]["nb_removed_rows"] = int(na_mask.sum())
                report["na_handling"]["removed_rows_indices"] = data.index[na_mask].tolist()
                data = data[~na_mask].copy()

            report["final_shape"] = data.shape
            headers = data.columns.tolist()

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
            report["error"] = f"Error loading Parquet file: {e}\n{traceback.format_exc()}"
            return LoaderResult(report=report, header_unit=header_unit)


def load_parquet(
    path,
    columns: Optional[List[str]] = None,
    engine: str = "auto",
    header_unit: str = "text",
    **params,
):
    """Load a Parquet file.

    Convenience function for direct use.

    Args:
        path: Path to the Parquet file.
        columns: Column names to load.
        engine: Parquet engine to use.
        header_unit: Unit type for headers.
        **params: Additional parameters.

    Returns:
        Tuple of (DataFrame, report, na_mask, headers, header_unit).
    """
    loader = ParquetLoader()
    result = loader.load(
        Path(path),
        columns=columns,
        engine=engine,
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
