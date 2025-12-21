"""
Auto-detection for file parameters.

This module provides enhanced auto-detection capabilities for CSV files,
including delimiter detection, decimal separator detection, header detection,
and signal type inference from headers.

Phase 8 Implementation - Dataset Configuration Roadmap
Section 8.2: Auto-Detection Improvements
"""

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from nirs4all.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DetectionResult:
    """Result of auto-detection.

    Attributes:
        delimiter: Detected field delimiter.
        decimal_separator: Detected decimal separator.
        has_header: Whether the file has a header row.
        header_unit: Detected unit type for headers.
        signal_type: Detected signal type.
        encoding: Detected file encoding.
        n_columns: Detected number of columns.
        n_rows: Estimated number of rows.
        confidence: Confidence scores for each detected parameter.
        warnings: List of detection warnings.
    """

    delimiter: str = ";"
    decimal_separator: str = "."
    has_header: bool = True
    header_unit: str = "cm-1"
    signal_type: Optional[str] = None
    encoding: str = "utf-8"
    n_columns: int = 0
    n_rows: int = 0
    confidence: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_params(self) -> Dict[str, Any]:
        """Convert to loading parameters dictionary."""
        return {
            "delimiter": self.delimiter,
            "decimal_separator": self.decimal_separator,
            "has_header": self.has_header,
            "header_unit": self.header_unit,
            "signal_type": self.signal_type,
            "encoding": self.encoding,
        }


class AutoDetector:
    """Auto-detect file parameters.

    Provides methods to detect CSV delimiters, decimal separators,
    header presence, header units, and signal types from file content.

    Example:
        ```python
        detector = AutoDetector()
        result = detector.detect("path/to/file.csv")
        print(f"Delimiter: {result.delimiter}")
        print(f"Has header: {result.has_header}")
        print(f"Signal type: {result.signal_type}")
        ```
    """

    # Common delimiters in order of priority
    DELIMITERS = [",", ";", "\t", "|", " "]

    # Patterns for header unit detection
    HEADER_PATTERNS = {
        "nm": [
            r"^\d{3,4}(?:\.\d+)?$",  # 400, 450.5, 1200
            r"^\d{3,4}(?:\.\d+)?nm$",  # 400nm
        ],
        "cm-1": [
            r"^\d{4,5}(?:\.\d+)?$",  # 4000, 10000.5
            r"^\d{4,5}(?:\.\d+)?cm-1$",  # 4000cm-1
            r"^\d{4,5}(?:\.\d+)?wavenumber$",  # Explicit wavenumber
        ],
        "text": [
            r"^[a-zA-Z]",  # Starts with letter
            r"^feature_\d+$",  # feature_1, feature_2
            r"^[xX]_?\d+$",  # X1, x_1
        ],
        "index": [
            r"^\d{1,3}$",  # 1, 10, 100 (small numbers, likely indices)
        ],
    }

    # Signal type patterns in header values
    SIGNAL_TYPE_PATTERNS = {
        "absorbance": [
            r"abs(orbance)?",
            r"log\s*\(?1/[RT]\)?",
            r"A\s*=",
        ],
        "reflectance": [
            r"reflect(ance)?",
            r"^R$",
            r"R\s*%",
        ],
        "transmittance": [
            r"transmit(tance)?",
            r"^T$",
            r"T\s*%",
        ],
    }

    def __init__(
        self,
        sample_lines: int = 50,
        min_confidence: float = 0.6,
    ):
        """Initialize detector.

        Args:
            sample_lines: Number of lines to sample for detection.
            min_confidence: Minimum confidence threshold for detection.
        """
        self.sample_lines = sample_lines
        self.min_confidence = min_confidence

    def detect(
        self,
        source: Union[str, Path, bytes, io.StringIO],
        known_params: Optional[Dict[str, Any]] = None,
    ) -> DetectionResult:
        """Detect file parameters.

        Args:
            source: Path to file, file content as bytes, or StringIO.
            known_params: Optional known parameters to skip detection for.

        Returns:
            DetectionResult with detected parameters.
        """
        result = DetectionResult()
        known_params = known_params or {}

        # Read content
        content, encoding = self._read_content(source)
        result.encoding = known_params.get("encoding", encoding)

        if not content:
            result.warnings.append("Empty file content")
            return result

        # Get sample lines
        lines = self._get_sample_lines(content)
        if not lines:
            result.warnings.append("No data lines found")
            return result

        # Detect delimiter
        if "delimiter" in known_params:
            result.delimiter = known_params["delimiter"]
            result.confidence["delimiter"] = 1.0
        else:
            result.delimiter, conf = self._detect_delimiter(lines)
            result.confidence["delimiter"] = conf

        # Parse with detected delimiter
        parsed_rows = self._parse_lines(lines, result.delimiter)
        if not parsed_rows:
            result.warnings.append("Could not parse lines")
            return result

        result.n_columns = max(len(row) for row in parsed_rows) if parsed_rows else 0
        result.n_rows = len(parsed_rows)

        # Detect decimal separator
        if "decimal_separator" in known_params:
            result.decimal_separator = known_params["decimal_separator"]
            result.confidence["decimal_separator"] = 1.0
        else:
            result.decimal_separator, conf = self._detect_decimal_separator(parsed_rows)
            result.confidence["decimal_separator"] = conf

        # Detect header
        if "has_header" in known_params:
            result.has_header = known_params["has_header"]
            result.confidence["has_header"] = 1.0
        else:
            result.has_header, conf = self._detect_header(
                parsed_rows,
                result.decimal_separator
            )
            result.confidence["has_header"] = conf

        # Detect header unit
        if "header_unit" in known_params:
            result.header_unit = known_params["header_unit"]
            result.confidence["header_unit"] = 1.0
        elif result.has_header and parsed_rows:
            result.header_unit, conf = self._detect_header_unit(parsed_rows[0])
            result.confidence["header_unit"] = conf

        # Detect signal type
        if "signal_type" in known_params:
            result.signal_type = known_params["signal_type"]
            result.confidence["signal_type"] = 1.0
        elif result.has_header and parsed_rows:
            result.signal_type, conf = self._detect_signal_type_from_header(parsed_rows[0])
            result.confidence["signal_type"] = conf
            if result.signal_type is None:
                # Try to infer from data values
                result.signal_type, conf = self._detect_signal_type_from_values(
                    parsed_rows[1:] if result.has_header else parsed_rows,
                    result.decimal_separator
                )
                result.confidence["signal_type"] = conf

        return result

    def _read_content(
        self,
        source: Union[str, Path, bytes, io.StringIO]
    ) -> Tuple[str, str]:
        """Read content from source.

        Args:
            source: Path, bytes, or StringIO.

        Returns:
            Tuple of (content, encoding).
        """
        if isinstance(source, io.StringIO):
            source.seek(0)
            return source.read(), "utf-8"

        if isinstance(source, bytes):
            # Try to detect encoding
            try:
                return source.decode("utf-8"), "utf-8"
            except UnicodeDecodeError:
                try:
                    return source.decode("latin-1"), "latin-1"
                except UnicodeDecodeError:
                    return source.decode("utf-8", errors="replace"), "utf-8"

        path = Path(source)
        if not path.exists():
            return "", "utf-8"

        # Try encodings
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                with open(path, "r", encoding=encoding) as f:
                    return f.read(), encoding
            except UnicodeDecodeError:
                continue

        # Fallback
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), "utf-8"

    def _get_sample_lines(self, content: str) -> List[str]:
        """Get sample lines from content.

        Args:
            content: File content.

        Returns:
            List of sample lines.
        """
        lines = []
        for i, line in enumerate(content.split("\n")):
            if i >= self.sample_lines:
                break
            line = line.strip()
            if line:
                lines.append(line)
        return lines

    def _parse_lines(self, lines: List[str], delimiter: str) -> List[List[str]]:
        """Parse lines with delimiter.

        Args:
            lines: List of lines.
            delimiter: Field delimiter.

        Returns:
            List of parsed rows (lists of fields).
        """
        content = "\n".join(lines)
        reader = csv.reader(io.StringIO(content), delimiter=delimiter)
        return [row for row in reader if any(cell.strip() for cell in row)]

    def _detect_delimiter(self, lines: List[str]) -> Tuple[str, float]:
        """Detect the field delimiter.

        Args:
            lines: Sample lines.

        Returns:
            Tuple of (delimiter, confidence).
        """
        best_delim = ";"
        max_score = 0.0

        for delim in self.DELIMITERS:
            score = self._score_delimiter(lines, delim)
            if score > max_score:
                max_score = score
                best_delim = delim

        confidence = min(max_score / 10, 1.0)  # Normalize score to confidence
        return best_delim, confidence

    def _score_delimiter(self, lines: List[str], delim: str) -> float:
        """Score a delimiter based on consistency.

        Args:
            lines: Sample lines.
            delim: Delimiter to test.

        Returns:
            Score (higher is better).
        """
        if not lines:
            return 0.0

        try:
            reader = csv.reader(io.StringIO("\n".join(lines)), delimiter=delim)
            col_counts = [len(row) for row in reader if row]
        except csv.Error:
            return 0.0

        if not col_counts:
            return 0.0

        # Most common column count
        from collections import Counter
        count_freq = Counter(col_counts)
        most_common_count, frequency = count_freq.most_common(1)[0]

        # Score based on consistency and number of columns
        consistency = frequency / len(col_counts)
        col_bonus = min(most_common_count / 10, 5)  # Prefer more columns, max bonus 5

        # Penalize if only 1 column
        if most_common_count == 1:
            return 0.1

        return consistency * 5 + col_bonus

    def _detect_decimal_separator(
        self,
        parsed_rows: List[List[str]]
    ) -> Tuple[str, float]:
        """Detect the decimal separator.

        Args:
            parsed_rows: Parsed data rows.

        Returns:
            Tuple of (separator, confidence).
        """
        # Count occurrences of . and , in numeric-looking values
        dot_count = 0
        comma_count = 0
        dot_valid = 0
        comma_valid = 0

        for row in parsed_rows[1:]:  # Skip potential header
            for cell in row:
                cell = cell.strip()
                if not cell:
                    continue

                # Check if it looks like a number with dot
                if "." in cell and "," not in cell:
                    try:
                        float(cell)
                        dot_valid += 1
                    except ValueError:
                        pass
                    dot_count += 1

                # Check if it looks like a number with comma
                if "," in cell and "." not in cell:
                    try:
                        float(cell.replace(",", "."))
                        comma_valid += 1
                    except ValueError:
                        pass
                    comma_count += 1

        # Prefer dot as it's more common in scientific data
        if dot_valid >= comma_valid:
            return ".", min((dot_valid + 1) / (max(dot_count, 1) + comma_count + 1), 1.0)
        else:
            return ",", min((comma_valid + 1) / (max(comma_count, 1) + dot_count + 1), 1.0)

    def _detect_header(
        self,
        parsed_rows: List[List[str]],
        decimal_sep: str
    ) -> Tuple[bool, float]:
        """Detect if file has a header row.

        Args:
            parsed_rows: Parsed data rows.
            decimal_sep: Detected decimal separator.

        Returns:
            Tuple of (has_header, confidence).
        """
        if len(parsed_rows) < 2:
            return True, 0.5  # Default to True with low confidence

        first_row = parsed_rows[0]
        data_rows = parsed_rows[1:min(10, len(parsed_rows))]

        # Count numeric values in first row vs data rows
        first_numeric = sum(1 for cell in first_row if self._is_numeric(cell, decimal_sep))
        first_ratio = first_numeric / len(first_row) if first_row else 0

        data_numeric_ratios = []
        for row in data_rows:
            if not row:
                continue
            numeric = sum(1 for cell in row if self._is_numeric(cell, decimal_sep))
            data_numeric_ratios.append(numeric / len(row))

        if not data_numeric_ratios:
            return True, 0.5

        avg_data_ratio = np.mean(data_numeric_ratios)

        # If first row has significantly fewer numeric values, it's likely a header
        if first_ratio < avg_data_ratio - 0.3:
            confidence = min((avg_data_ratio - first_ratio) * 2, 1.0)
            return True, confidence

        # If ratios are similar, probably no header
        if abs(first_ratio - avg_data_ratio) < 0.1:
            return False, 0.7

        return True, 0.5  # Default with uncertainty

    def _is_numeric(self, value: str, decimal_sep: str = ".") -> bool:
        """Check if a string value is numeric.

        Args:
            value: Value to check.
            decimal_sep: Decimal separator.

        Returns:
            True if numeric.
        """
        value = value.strip()
        if not value:
            return False

        # Handle scientific notation
        if "e" in value.lower():
            try:
                float(value.replace(decimal_sep, "."))
                return True
            except ValueError:
                return False

        # Standard numeric
        if decimal_sep == ",":
            value = value.replace(",", ".")

        try:
            float(value)
            return True
        except ValueError:
            return False

    def _detect_header_unit(self, header_row: List[str]) -> Tuple[str, float]:
        """Detect the unit type from header values.

        Args:
            header_row: First row (header) values.

        Returns:
            Tuple of (unit_type, confidence).
        """
        scores = {unit: 0 for unit in self.HEADER_PATTERNS}

        for cell in header_row:
            cell = cell.strip()
            if not cell:
                continue

            for unit, patterns in self.HEADER_PATTERNS.items():
                for pattern in patterns:
                    if re.match(pattern, cell, re.IGNORECASE):
                        scores[unit] += 1
                        break

        if not any(scores.values()):
            return "text", 0.5  # Default

        # Find best match
        best_unit = max(scores, key=scores.get)
        best_score = scores[best_unit]

        # Calculate confidence
        total = sum(scores.values())
        confidence = best_score / total if total > 0 else 0.5

        # Special case: distinguish between nm and cm-1 based on value ranges
        if best_unit in ["nm", "cm-1"]:
            numeric_values = []
            for cell in header_row:
                try:
                    val = float(cell.strip())
                    numeric_values.append(val)
                except ValueError:
                    continue

            if numeric_values:
                min_val = min(numeric_values)
                max_val = max(numeric_values)

                # Typical wavelength range: 350-2500 nm
                # Typical wavenumber range: 400-12500 cm-1
                if 350 <= min_val <= 2500 and max_val <= 2500:
                    return "nm", 0.8
                elif 400 <= min_val <= 12500 and max_val <= 12500 and min_val > 2500:
                    return "cm-1", 0.8

        return best_unit, confidence

    def _detect_signal_type_from_header(
        self,
        header_row: List[str]
    ) -> Tuple[Optional[str], float]:
        """Detect signal type from header content.

        Args:
            header_row: First row (header) values.

        Returns:
            Tuple of (signal_type or None, confidence).
        """
        full_header = " ".join(header_row).lower()

        for signal_type, patterns in self.SIGNAL_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, full_header, re.IGNORECASE):
                    return signal_type, 0.8

        return None, 0.0

    def _detect_signal_type_from_values(
        self,
        data_rows: List[List[str]],
        decimal_sep: str
    ) -> Tuple[Optional[str], float]:
        """Detect signal type from data values.

        Args:
            data_rows: Data rows (excluding header).
            decimal_sep: Decimal separator.

        Returns:
            Tuple of (signal_type or None, confidence).
        """
        # Collect numeric values
        values = []
        for row in data_rows[:20]:  # Sample first 20 rows
            for cell in row:
                try:
                    val = float(cell.strip().replace(decimal_sep, "."))
                    values.append(val)
                except ValueError:
                    continue

        if not values:
            return None, 0.0

        min_val = min(values)
        max_val = max(values)

        # Absorbance: typically 0-3 range
        if 0 <= min_val <= 0.5 and 0.5 <= max_val <= 5:
            return "absorbance", 0.6

        # Reflectance/Transmittance: typically 0-1 or 0-100 range
        if 0 <= min_val <= 0.1 and 0.8 <= max_val <= 1.0:
            return "reflectance", 0.5
        if 0 <= min_val <= 10 and 80 <= max_val <= 100:
            return "reflectance%", 0.5

        return None, 0.0


def detect_file_parameters(
    source: Union[str, Path, bytes],
    known_params: Optional[Dict[str, Any]] = None,
    sample_lines: int = 50,
) -> DetectionResult:
    """Convenience function to detect file parameters.

    Args:
        source: Path to file or file content.
        known_params: Optional known parameters.
        sample_lines: Number of lines to sample.

    Returns:
        DetectionResult with detected parameters.
    """
    detector = AutoDetector(sample_lines=sample_lines)
    return detector.detect(source, known_params)


def detect_signal_type(
    header: Optional[List[str]] = None,
    data: Optional[np.ndarray] = None,
) -> Tuple[Optional[str], float]:
    """Detect signal type from header and/or data.

    Args:
        header: Optional list of header values.
        data: Optional data array.

    Returns:
        Tuple of (signal_type or None, confidence).
    """
    detector = AutoDetector()

    # Try header first
    if header:
        signal_type, conf = detector._detect_signal_type_from_header(header)
        if signal_type and conf >= 0.6:
            return signal_type, conf

    # Try data values
    if data is not None:
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)

        # Absorbance: typically 0-3 range
        if 0 <= min_val <= 0.5 and 0.5 <= max_val <= 5:
            return "absorbance", 0.6

        # Reflectance/Transmittance: typically 0-1 or 0-100 range
        if 0 <= min_val <= 0.1 and 0.8 <= max_val <= 1.0:
            return "reflectance", 0.5
        if 0 <= min_val <= 10 and 80 <= max_val <= 100:
            return "reflectance%", 0.5

    return None, 0.0
