"""
CSV file loader implementation.

This module provides the CSVLoader class for loading CSV files,
including support for compressed CSV files (.csv.gz, .csv.zip).
"""

import csv
import io
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import (
    ArchiveHandler,
    FileLoadError,
    FileLoader,
    LoaderResult,
    register_loader,
)


def _can_be_float(value: str, decimal_sep: str) -> bool:
    """Check if a string can be converted to a float.

    Args:
        value: String value to check.
        decimal_sep: Decimal separator to consider.

    Returns:
        True if the value can be converted to float.
    """
    if not isinstance(value, str):
        return False

    value = value.strip()
    if not value:
        return False

    try:
        # Handle scientific notation
        if "e" in value.lower():
            float(value)
            return True

        # Replace decimal separator if needed
        if decimal_sep == ".":
            float(value)
        else:
            float(value.replace(decimal_sep, ".", 1))
        return True
    except ValueError:
        return False


def _detect_delimiter(lines: List[str], possible_delimiters: Optional[List[str]] = None) -> Optional[str]:
    """Detect the delimiter by looking at column count consistency.

    Args:
        lines: Sample lines from the file.
        possible_delimiters: List of delimiters to try.

    Returns:
        Best delimiter candidate or None.
    """
    if possible_delimiters is None:
        possible_delimiters = [";", ",", "\t", "|", " "]

    best_delim = None
    max_consistent_cols = -1
    most_cols_at_max_consistency = 0

    content_for_test = "".join(lines)

    for delim_candidate in possible_delimiters:
        try:
            reader = csv.reader(io.StringIO(content_for_test), delimiter=delim_candidate)
            cols_counts = [len(row) for row in reader if row]

            if not cols_counts:
                continue

            most_frequent_cols = max(set(cols_counts), key=cols_counts.count)
            consistency = sum(1 for count in cols_counts if count == most_frequent_cols)

            if consistency > max_consistent_cols:
                max_consistent_cols = consistency
                most_cols_at_max_consistency = most_frequent_cols
                best_delim = delim_candidate
            elif consistency == max_consistent_cols:
                if most_frequent_cols > most_cols_at_max_consistency:
                    most_cols_at_max_consistency = most_frequent_cols
                    best_delim = delim_candidate
        except (csv.Error, ValueError):
            continue

    return best_delim


def _detect_decimal_and_header(
    parsed_rows: List[List[str]],
    data_type: str = "x",
) -> Tuple[str, bool]:
    """Detect decimal separator and header presence.

    Args:
        parsed_rows: List of parsed rows (split by delimiter).
        data_type: Type of data ('x' or 'y').

    Returns:
        Tuple of (decimal_separator, has_header).
    """
    if not parsed_rows:
        return ".", False

    num_cols = len(parsed_rows[0])
    if num_cols == 0:
        return ".", False

    best_decimal_sep = "."
    best_has_header = False
    max_numeric_score = -1.0

    for decimal_sep in [".", ","]:
        for has_header_option in [False, True]:
            first_data_row_index = 1 if has_header_option else 0
            if len(parsed_rows) <= first_data_row_index:
                current_score = 0.0
            else:
                data_rows = parsed_rows[first_data_row_index:]
                numeric_cells = 0
                total_cells = 0

                for row in data_rows:
                    if abs(len(row) - num_cols) <= 1:
                        for val in row:
                            total_cells += 1
                            if _can_be_float(val, decimal_sep):
                                numeric_cells += 1

                current_score = numeric_cells / total_cells if total_cells else 0.0

            if has_header_option and parsed_rows:
                header_row = parsed_rows[0]
                if len(header_row) == num_cols:
                    header_numeric_cells = sum(_can_be_float(cell, decimal_sep) for cell in header_row)
                    header_score = header_numeric_cells / len(header_row) if header_row else 0.0
                    if current_score > 0.5 and header_score >= current_score:
                        current_score *= 0.5

            if current_score > max_numeric_score + 1e-6:
                max_numeric_score = current_score
                best_decimal_sep = decimal_sep
                best_has_header = has_header_option
            elif abs(current_score - max_numeric_score) < 1e-6:
                if best_decimal_sep == "," and decimal_sep == ".":
                    best_decimal_sep = decimal_sep
                    best_has_header = has_header_option
                elif best_has_header and (not has_header_option):
                    best_decimal_sep = decimal_sep
                    best_has_header = has_header_option

    return best_decimal_sep, best_has_header


def _determine_csv_parameters(
    csv_content: str,
    sample_lines: int = 20,
    data_type: str = "x",
    user_params: Optional[Dict[str, Any]] = None,
    *,
    bypass_auto_detection: bool = True,
) -> Dict[str, Any]:
    """Determine CSV parameters with defaults or auto-detection.

    Args:
        csv_content: CSV file content as string.
        sample_lines: Number of lines to sample for detection.
        data_type: Type of data ('x' or 'y').
        user_params: User-provided parameters (override auto-detection).
        bypass_auto_detection: If True, skip auto-detection and use defaults.

    Returns:
        Dictionary with delimiter, decimal_separator, and has_header.
    """
    if user_params is None:
        user_params = {}

    # Default parameters
    delimiter = user_params.get("delimiter", ";")
    decimal_sep = user_params.get("decimal_separator", ".")
    has_header = user_params.get("has_header", True)

    if not bypass_auto_detection:
        lines = []
        with io.StringIO(csv_content) as f:
            for i, line in enumerate(f):
                if i >= sample_lines:
                    break
                if line.strip():
                    lines.append(line)

        if not lines:
            return {
                "delimiter": delimiter,
                "decimal_separator": decimal_sep,
                "has_header": has_header,
            }

        # Delimiter detection
        if "delimiter" not in user_params:
            detected_delim = _detect_delimiter(lines)
            if detected_delim:
                delimiter = detected_delim

        # Parse sample to detect decimal and header
        sample_data = "".join(lines)
        parsed_rows_reader = csv.reader(io.StringIO(sample_data), delimiter=delimiter)
        parsed_rows = [row for row in parsed_rows_reader if any(cell.strip() for cell in row)]

        if "decimal_separator" not in user_params:
            decimal_sep, _ = _detect_decimal_and_header(parsed_rows, data_type=data_type)

        if "has_header" not in user_params:
            _, has_header = _detect_decimal_and_header(parsed_rows, data_type=data_type)

    return {
        "delimiter": delimiter,
        "decimal_separator": decimal_sep,
        "has_header": has_header,
    }


@register_loader
class CSVLoader(FileLoader):
    """Loader for CSV files.

    Supports:
    - Plain CSV files (.csv)
    - Gzip-compressed CSV files (.csv.gz)
    - Zip-compressed CSV files (.csv.zip)

    Parameters:
        delimiter: Field delimiter (default: ';')
        decimal_separator: Decimal separator (default: '.')
        has_header: Whether first row is header (default: True)
        header_unit: Unit for headers ('cm-1', 'nm', etc.)
        na_policy: How to handle NA values ('remove' or 'abort')
        categorical_mode: How to handle categorical data ('auto', 'preserve', 'none')
        data_type: Type of data being loaded ('x', 'y', or 'metadata')
        encoding: File encoding (default: 'utf-8')
        member: For zip files, specific member to extract
    """

    supported_extensions: ClassVar[Tuple[str, ...]] = (".csv",)
    name: ClassVar[str] = "CSV Loader"
    priority: ClassVar[int] = 50

    @classmethod
    def supports(cls, path: Path) -> bool:
        """Check if this loader supports the given file.

        Supports .csv, .csv.gz, and .csv.zip files.
        """
        name_lower = path.name.lower()

        # Direct CSV
        if path.suffix.lower() == ".csv":
            return True

        # Compressed CSV
        if name_lower.endswith(".csv.gz"):
            return True
        if name_lower.endswith(".csv.zip"):
            return True

        # Check inside archives
        if path.suffix.lower() == ".gz":
            base = path.with_suffix("")
            if base.suffix.lower() == ".csv":
                return True

        if path.suffix.lower() == ".zip":
            # Could check if zip contains CSV, but for now assume yes
            base = path.with_suffix("")
            if base.suffix.lower() == ".csv":
                return True

        return False

    def load(
        self,
        path: Path,
        na_policy: str = "auto",
        data_type: str = "x",
        categorical_mode: str = "auto",
        header_unit: str = "cm-1",
        encoding: str = "utf-8",
        member: Optional[str] = None,
        **user_params: Any,
    ) -> LoaderResult:
        """Load data from a CSV file.

        Args:
            path: Path to the CSV file.
            na_policy: How to handle NA values ('remove', 'abort', or 'auto').
            data_type: Type of data ('x', 'y', or 'metadata').
            categorical_mode: How to handle categorical columns.
            header_unit: Unit type for headers.
            encoding: File encoding.
            member: For zip files, specific member to extract.
            **user_params: Additional CSV parsing parameters.

        Returns:
            LoaderResult with the loaded data.
        """
        if na_policy == "auto":
            na_policy = "remove"

        if na_policy not in ["remove", "abort"]:
            raise ValueError("Invalid NA policy - only 'remove' or 'abort' (or 'auto') are supported.")
        if categorical_mode not in ["auto", "preserve", "none"]:
            raise ValueError("Invalid categorical mode - only 'auto', 'preserve', or 'none' are supported.")

        report: Dict[str, Any] = {
            "file_path": str(path),
            "detection_params": None,
            "delimiter": None,
            "decimal_separator": None,
            "has_header": None,
            "initial_shape": None,
            "final_shape": None,
            "na_handling": {
                "strategy": na_policy,
                "na_detected": False,
                "nb_removed_rows": 0,
                "removed_rows_indices": [],
            },
            "categorical_info": {},
            "warnings": [],
            "error": None,
        }

        try:
            file_path = Path(path)
            if not file_path.exists():
                raise FileNotFoundError(f"Le fichier n'existe pas: {path}")

            # Read file content
            content = self._read_content(file_path, encoding, member)

            if not content.strip():
                raise ValueError("File is empty or could not be read.")

            # Determine CSV parameters
            detection_params = _determine_csv_parameters(
                csv_content=content,
                user_params=user_params,
            )

            delimiter = detection_params["delimiter"]
            decimal_sep = detection_params["decimal_separator"]
            has_header = detection_params["has_header"]

            report["detection_params"] = detection_params
            report["delimiter"] = delimiter
            report["decimal_separator"] = decimal_sep
            report["has_header"] = has_header

            # Load with pandas
            read_csv_kwargs = {
                "sep": delimiter,
                "decimal": decimal_sep,
                "header": 0 if has_header else None,
                "na_filter": True,
                "na_values": ["NA", "N/A", ""],
                "keep_default_na": True,
                "engine": "python",
                "skip_blank_lines": True,
                "quoting": csv.QUOTE_MINIMAL,
            }

            # Add user-provided read_csv args
            for k, v in user_params.items():
                if k not in ["delimiter", "decimal_separator", "has_header"]:
                    read_csv_kwargs[k] = v

            try:
                data = pd.read_csv(io.StringIO(content), **read_csv_kwargs)
            except Exception as e1:
                try:
                    read_csv_kwargs["engine"] = "c"
                    data = pd.read_csv(io.StringIO(content), **read_csv_kwargs)
                except Exception as e2:
                    msg = f"Could not parse CSV. Python engine error: {e1} | C engine error: {e2}"
                    report["error"] = msg
                    return LoaderResult(report=report, header_unit=header_unit)

            report["initial_shape"] = data.shape

            # Ensure column names are strings
            data.columns = data.columns.astype(str)
            report["shape_after_all_na_col_removal"] = data.shape

            if data.empty:
                report["warnings"].append("Data is empty after removing all-NA columns.")
                return LoaderResult(
                    data=pd.DataFrame(),
                    report=report,
                    na_mask=pd.Series(dtype=bool),
                    headers=[],
                    header_unit=header_unit,
                )

            # Handle type conversion
            report["categorical_info"] = {}
            local_categorical_mappings = {}

            if data_type == "y":
                for col in data.columns:
                    original_col_series = data[col].copy()
                    numeric_representation = pd.to_numeric(data[col], errors="coerce")
                    original_is_object = pd.api.types.is_object_dtype(data[col].dtype)

                    if categorical_mode == "auto":
                        should_treat_as_categorical = False
                        if original_is_object:
                            should_treat_as_categorical = True
                        elif not pd.api.types.is_numeric_dtype(data[col].dtype):
                            if numeric_representation.isna().sum() > original_col_series.isna().sum():
                                should_treat_as_categorical = True

                        if should_treat_as_categorical:
                            if col.isdigit() or (col.count(".") == 1 and col.replace(".", "", 1).isdigit()):
                                report["warnings"].append(
                                    f"Column '{col}' detected as categorical but has a numeric header"
                                )

                            codes, categories = pd.factorize(original_col_series.astype(str))
                            data[col] = codes
                            local_categorical_mappings[col] = {"categories": categories.tolist()}
                        else:
                            data[col] = numeric_representation

                    elif categorical_mode == "preserve":
                        data[col] = numeric_representation

                    elif categorical_mode == "none":
                        data[col] = numeric_representation

                report["categorical_info"] = local_categorical_mappings

            elif data_type == "x":
                for col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors="coerce")

            # Handle NA values
            na_mask = data.isna().any(axis=1)
            report["na_handling"]["na_detected_in_rows"] = bool(na_mask.any())

            if report["na_handling"]["na_detected_in_rows"]:
                if na_policy == "abort":
                    first_na_row = data.index[na_mask][0]
                    first_na_col = data.loc[first_na_row].isna().idxmax()
                    error_msg = (
                        f"NA values detected and na_policy is 'abort'. "
                        f"First NA in column '{first_na_col}' (row: {first_na_row})."
                    )
                    report["error"] = error_msg
                    report["na_handling"]["na_detected"] = True
                    return LoaderResult(report=report, na_mask=na_mask, header_unit=header_unit)

                elif na_policy == "remove":
                    report["na_handling"]["na_detected"] = True
                    report["na_handling"]["nb_removed_rows"] = int(na_mask.sum())
                    report["na_handling"]["removed_rows_indices"] = data.index[na_mask].tolist()
                    data = data[~na_mask].copy()

            report["final_shape"] = data.shape
            report["final_column_names"] = data.columns.tolist()
            headers = data.columns.tolist() if not data.empty else []

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
        except ValueError as e:
            report["error"] = f"ValueError during processing: {e}"
            return LoaderResult(report=report, header_unit=header_unit)
        except Exception as e:
            import traceback
            report["error"] = f"Unexpected error in CSVLoader: {e}\n{traceback.format_exc()}"
            return LoaderResult(report=report, header_unit=header_unit)

    def _read_content(
        self,
        path: Path,
        encoding: str = "utf-8",
        member: Optional[str] = None,
    ) -> str:
        """Read file content, handling compression.

        Args:
            path: Path to the file.
            encoding: Text encoding.
            member: For archives, specific member to extract.

        Returns:
            File content as string.
        """
        name_lower = path.name.lower()

        if path.suffix.lower() == ".gz" or name_lower.endswith(".csv.gz"):
            return ArchiveHandler.decompress_gzip(path, encoding)

        elif path.suffix.lower() == ".zip" or name_lower.endswith(".csv.zip"):
            return ArchiveHandler.extract_from_zip(path, member, encoding)

        else:
            # Plain text file
            try:
                with open(path, "r", encoding=encoding, newline="") as f:
                    return f.read()
            except UnicodeDecodeError:
                # Fall back to latin-1
                with open(path, "r", encoding="latin-1", newline="") as f:
                    return f.read()


# Backward compatibility function
def load_csv(
    path,
    na_policy: str = "auto",
    data_type: str = "x",
    categorical_mode: str = "auto",
    header_unit: str = "cm-1",
    **user_params,
):
    """Load a CSV file using the CSVLoader.

    This function maintains backward compatibility with the original load_csv API.

    Args:
        path: Path to the CSV file.
        na_policy: How to handle NA values.
        data_type: Type of data being loaded.
        categorical_mode: How to handle categorical columns.
        header_unit: Unit type for headers.
        **user_params: Additional CSV parsing parameters.

    Returns:
        Tuple of (DataFrame, report, na_mask, headers, header_unit).
    """
    loader = CSVLoader()
    result = loader.load(
        Path(path),
        na_policy=na_policy,
        data_type=data_type,
        categorical_mode=categorical_mode,
        header_unit=header_unit,
        **user_params,
    )

    return (
        result.data,
        result.report,
        result.na_mask,
        result.headers,
        result.header_unit,
    )
