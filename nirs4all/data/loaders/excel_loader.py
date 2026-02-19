"""
Excel file loader implementation.

This module provides the ExcelLoader class for loading Excel spreadsheet files,
including .xlsx (modern) and .xls (legacy) formats.
"""

from pathlib import Path
from typing import Any, ClassVar, Optional, Union

import pandas as pd

from nirs4all.core.exceptions import NAError
from nirs4all.data.schema.config import NAFillConfig

from .base import (
    FileLoader,
    FileLoadError,
    LoaderResult,
    apply_na_policy,
    register_loader,
)


def _check_excel_engine(suffix: str) -> str:
    """Check if the appropriate Excel engine is available.

    Args:
        suffix: File extension (.xlsx or .xls).

    Returns:
        Name of the available engine.

    Raises:
        ImportError: If no suitable engine is available.
    """
    if suffix.lower() == ".xlsx":
        try:
            import openpyxl
            return "openpyxl"
        except ImportError as e:
            raise ImportError(
                "openpyxl is required for .xlsx files. Install it with: "
                "pip install openpyxl"
            ) from e
    elif suffix.lower() == ".xls":
        try:
            import xlrd
            return "xlrd"
        except ImportError as e:
            raise ImportError(
                "xlrd is required for .xls files. Install it with: "
                "pip install xlrd"
            ) from e
    else:
        raise ValueError(f"Unsupported Excel format: {suffix}")

@register_loader
class ExcelLoader(FileLoader):
    """Loader for Excel spreadsheet files.

    Supports:
    - Modern Excel files (.xlsx) via openpyxl
    - Legacy Excel files (.xls) via xlrd

    Parameters:
        sheet_name: Sheet name or index to load (default: 0, first sheet).
            Can be a string (sheet name), integer (0-indexed), or None (all sheets).
        header: Row number to use as header (default: 0).
            Use None for no header.
        skip_rows: Number of rows to skip at the beginning.
        skip_footer: Number of rows to skip at the end.
        usecols: Columns to load (can be list of names, indices, or Excel-style range).
        engine: Excel engine to use ('auto', 'openpyxl', or 'xlrd').
        header_unit: Unit for headers ('cm-1', 'nm', 'text', etc.)

    Example:
        >>> loader = ExcelLoader()
        >>> result = loader.load(
        ...     Path("data.xlsx"),
        ...     sheet_name="Sheet1",
        ...     skip_rows=2,
        ... )
    """

    supported_extensions: ClassVar[tuple[str, ...]] = (".xlsx", ".xls")
    name: ClassVar[str] = "Excel Loader"
    priority: ClassVar[int] = 45

    @classmethod
    def supports(cls, path: Path) -> bool:
        """Check if this loader supports the given file."""
        return path.suffix.lower() in cls.supported_extensions

    def load(
        self,
        path: Path,
        sheet_name: str | int | None = 0,
        header: int | None = 0,
        skip_rows: int | None = None,
        skip_footer: int = 0,
        usecols: list[str] | list[int] | str | None = None,
        engine: str = "auto",
        header_unit: str = "text",
        data_type: str = "x",
        na_policy: str = "auto",
        na_fill_config: NAFillConfig | None = None,
        **params: Any,
    ) -> LoaderResult:
        """Load data from an Excel file.

        Args:
            path: Path to the Excel file.
            sheet_name: Sheet to load (name, index, or None for all).
            header: Row number for header (0-indexed), or None.
            skip_rows: Number of rows to skip at start.
            skip_footer: Number of rows to skip at end.
            usecols: Columns to load.
            engine: Excel engine to use.
            header_unit: Unit type for headers.
            data_type: Type of data ('x', 'y', or 'metadata').
            na_policy: How to handle NA values (any NAPolicy value).
            na_fill_config: Fill configuration when na_policy='replace'.
            **params: Additional parameters passed to read_excel.

        Returns:
            LoaderResult with the loaded data.
        """
        report: dict[str, Any] = {
            "file_path": str(path),
            "format": "excel",
            "engine": None,
            "sheet_name": sheet_name,
            "sheets_available": None,
            "initial_shape": None,
            "final_shape": None,
            "na_handling": {},
            "warnings": [],
            "error": None,
        }

        try:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            # Determine engine
            if engine == "auto":
                try:
                    engine = _check_excel_engine(path.suffix)
                except ImportError as e:
                    report["error"] = str(e)
                    return LoaderResult(report=report, header_unit=header_unit)

            report["engine"] = engine

            # Build read_excel kwargs
            read_kwargs: dict[str, Any] = {
                "engine": engine,
                "sheet_name": sheet_name,
                "header": header,
                "skipfooter": skip_footer,
            }

            if skip_rows is not None:
                read_kwargs["skiprows"] = skip_rows

            if usecols is not None:
                read_kwargs["usecols"] = usecols

            # Add any extra params
            read_kwargs.update(params)

            # Load the data
            try:
                result = pd.read_excel(path, **read_kwargs)
            except ImportError as e:
                report["error"] = f"Excel engine not available: {e}"
                return LoaderResult(report=report, header_unit=header_unit)
            except Exception as e:
                report["error"] = f"Failed to read Excel file: {e}"
                return LoaderResult(report=report, header_unit=header_unit)

            # Handle multiple sheets (when sheet_name is None)
            if isinstance(result, dict):
                report["sheets_available"] = list(result.keys())

                if not result:
                    report["error"] = "No sheets found in Excel file."
                    return LoaderResult(report=report, header_unit=header_unit)

                # Use first sheet
                first_sheet = list(result.keys())[0]
                report["warnings"].append(
                    f"Multiple sheets available. Using '{first_sheet}'. "
                    f"Specify 'sheet_name' to choose a different sheet."
                )
                data = result[first_sheet]
                report["sheet_name"] = first_sheet
            else:
                data = result

            report["initial_shape"] = data.shape

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

            # Handle NA values via centralized utility
            na_mask_before = data.isna().any(axis=1)

            try:
                data, na_report = apply_na_policy(data, na_policy, na_fill_config)
            except NAError:
                report["na_handling"] = {"strategy": "abort", "na_detected": True}
                na_mask = data.isna().any(axis=1)
                first_na_row = data.index[na_mask][0]
                first_na_col = data.loc[first_na_row].isna().idxmax()
                report["error"] = (
                    f"NA values detected and na_policy is 'abort'. "
                    f"First NA in column '{first_na_col}' (row: {first_na_row})."
                )
                return LoaderResult(report=report, na_mask=na_mask, header_unit=header_unit)

            report["na_handling"] = na_report

            # Update headers after potential column removal (remove_feature)
            headers = data.columns.tolist() if na_report.get("removed_features") else data.columns.tolist()

            report["final_shape"] = data.shape

            return LoaderResult(
                data=data,
                report=report,
                na_mask=na_mask_before,
                headers=headers,
                header_unit=header_unit,
            )

        except FileNotFoundError as e:
            report["error"] = str(e)
            return LoaderResult(report=report, header_unit=header_unit)
        except Exception as e:
            import traceback
            report["error"] = f"Error loading Excel file: {e}\n{traceback.format_exc()}"
            return LoaderResult(report=report, header_unit=header_unit)

def load_excel(
    path,
    sheet_name: str | int | None = 0,
    header: int | None = 0,
    skip_rows: int | None = None,
    skip_footer: int = 0,
    usecols: list[str] | list[int] | str | None = None,
    engine: str = "auto",
    header_unit: str = "text",
    **params,
):
    """Load an Excel file.

    Convenience function for direct use.

    Args:
        path: Path to the Excel file.
        sheet_name: Sheet to load.
        header: Row number for header.
        skip_rows: Rows to skip at start.
        skip_footer: Rows to skip at end.
        usecols: Columns to load.
        engine: Excel engine to use.
        header_unit: Unit type for headers.
        **params: Additional parameters.

    Returns:
        Tuple of (DataFrame, report, na_mask, headers, header_unit).
    """
    loader = ExcelLoader()
    result = loader.load(
        Path(path),
        sheet_name=sheet_name,
        header=header,
        skip_rows=skip_rows,
        skip_footer=skip_footer,
        usecols=usecols,
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
