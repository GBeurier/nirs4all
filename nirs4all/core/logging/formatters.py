"""Console and file formatters for nirs4all logging.

This module provides formatters for human-readable console output and
structured file logging, with ASCII and Unicode mode support.
"""

import logging
import sys
from datetime import datetime, timedelta
from typing import Any, Optional

from .events import Status


class Symbols:
    """Symbol system for log output with ASCII/Unicode modes.

    Provides consistent symbols for status indicators, hierarchy markers,
    and other visual elements in log output.
    """

    def __init__(self, use_unicode: bool = True) -> None:
        """Initialize symbols.

        Args:
            use_unicode: If True, use Unicode symbols. If False, use ASCII-only.
        """
        self.use_unicode = use_unicode

    @property
    def starting(self) -> str:
        """Starting/beginning of a phase."""
        return ">"

    @property
    def success(self) -> str:
        """Successful completion."""
        return "[OK]"

    @property
    def progress(self) -> str:
        """In progress."""
        return "*"

    @property
    def skipped(self) -> str:
        """Skipped (cached)."""
        return "-"

    @property
    def warning(self) -> str:
        """Non-fatal issue."""
        return "[!]"

    @property
    def error(self) -> str:
        """Fatal error."""
        return "[X]"

    @property
    def branch(self) -> str:
        """Branch indicator."""
        return "|"

    @property
    def indent(self) -> str:
        """Hierarchy level indent."""
        return "  "

    @property
    def arrow(self) -> str:
        """Flow/result arrow."""
        return "->"

    @property
    def branch_item(self) -> str:
        """Branch item prefix."""
        return "|--"

    @property
    def sub_item(self) -> str:
        """Sub-item prefix."""
        return "+--"

    @property
    def separator_heavy(self) -> str:
        """Heavy separator line (80 chars)."""
        return "=" * 80

    @property
    def separator_light(self) -> str:
        """Light separator line (76 chars)."""
        return "-" * 76

    def get_status_symbol(self, status: Optional[Status]) -> str:
        """Get symbol for a status indicator.

        Args:
            status: Status to get symbol for.

        Returns:
            Appropriate symbol string.
        """
        if status is None:
            return ""

        mapping = {
            Status.STARTING: self.starting,
            Status.SUCCESS: self.success,
            Status.IN_PROGRESS: self.progress,
            Status.SKIPPED: self.skipped,
            Status.WARNING: self.warning,
            Status.ERROR: self.error,
        }
        return mapping.get(status, "")


# Global symbols instance, configured at runtime
_symbols = Symbols(use_unicode=True)


def get_symbols() -> Symbols:
    """Get the global symbols instance."""
    return _symbols


def configure_symbols(use_unicode: bool = True) -> None:
    """Configure the global symbols instance.

    Args:
        use_unicode: If True, use Unicode symbols. If False, use ASCII-only.
    """
    global _symbols
    _symbols = Symbols(use_unicode=use_unicode)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter for nirs4all logs.

    Produces clean, scannable output optimized for terminal viewing with
    proper indentation for hierarchy, status symbols, and optional elapsed time.
    """

    # ANSI color codes
    COLORS = {
        "RESET": "\033[0m",
        "BOLD": "\033[1m",
        "DIM": "\033[2m",
        "RED": "\033[31m",
        "GREEN": "\033[32m",
        "YELLOW": "\033[33m",
        "BLUE": "\033[34m",
        "MAGENTA": "\033[35m",
        "CYAN": "\033[36m",
        "WHITE": "\033[37m",
    }

    LEVEL_COLORS = {
        "DEBUG": "DIM",
        "INFO": "RESET",
        "WARNING": "YELLOW",
        "ERROR": "RED",
        "CRITICAL": "RED",
    }

    def __init__(
        self,
        use_colors: bool = True,
        show_elapsed: bool = False,
        use_unicode: bool = True,
    ) -> None:
        """Initialize console formatter.

        Args:
            use_colors: If True, use ANSI colors for terminal output.
            show_elapsed: If True, show elapsed time since run start.
            use_unicode: If True, use Unicode symbols. If False, use ASCII-only.
        """
        super().__init__()
        self.use_colors = use_colors and self._supports_color()
        self.show_elapsed = show_elapsed
        self.use_unicode = use_unicode
        self.symbols = Symbols(use_unicode=use_unicode)
        self._run_start: Optional[datetime] = None

    def _supports_color(self) -> bool:
        """Check if terminal supports colors."""
        if not hasattr(sys.stdout, "isatty"):
            return False
        if not sys.stdout.isatty():
            return False
        return True

    def set_run_start(self, start_time: datetime) -> None:
        """Set the run start time for elapsed time calculation.

        Args:
            start_time: Datetime when the run started.
        """
        self._run_start = start_time

    def _colorize(self, text: str, color: str) -> str:
        """Apply ANSI color to text.

        Args:
            text: Text to colorize.
            color: Color name from COLORS dict.

        Returns:
            Colorized text or original if colors disabled.
        """
        if not self.use_colors:
            return text
        color_code = self.COLORS.get(color, "")
        reset = self.COLORS["RESET"]
        return f"{color_code}{text}{reset}"

    def _format_elapsed(self) -> str:
        """Format elapsed time since run start.

        Returns:
            Formatted elapsed time string or empty string.
        """
        if not self.show_elapsed or self._run_start is None:
            return ""
        elapsed = datetime.now() - self._run_start
        total_seconds = elapsed.total_seconds()

        if total_seconds < 60:
            return f"[{total_seconds:5.1f}s] "
        elif total_seconds < 3600:
            minutes = int(total_seconds // 60)
            seconds = total_seconds % 60
            return f"[{minutes}m{seconds:04.1f}s] "
        else:
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = total_seconds % 60
            return f"[{hours}h{minutes:02d}m{seconds:02.0f}s] "

    def _get_indent(self, record: logging.LogRecord) -> str:
        """Get indentation based on context depth.

        Args:
            record: Log record with optional context attributes.

        Returns:
            Indentation string.
        """
        depth = getattr(record, "depth", 0)
        branch_depth = getattr(record, "branch_depth", 0)
        total_depth = depth + branch_depth
        return self.symbols.indent * total_depth

    def _format_branch_context(self, record: logging.LogRecord) -> str:
        """Format branch context prefix if in a branch.

        Args:
            record: Log record with optional branch context.

        Returns:
            Branch context prefix or empty string.
        """
        branch_name = getattr(record, "branch_name", None)
        if not branch_name:
            return ""

        branch_index = getattr(record, "branch_index", None)
        total_branches = getattr(record, "total_branches", None)

        if branch_index is not None and total_branches is not None:
            return f"[branch:{branch_name}] "
        return f"[branch:{branch_name}] "

    def _format_source_context(self, record: logging.LogRecord) -> str:
        """Format source context prefix if in a source context.

        Args:
            record: Log record with optional source context.

        Returns:
            Source context prefix or empty string.
        """
        source_name = getattr(record, "source_name", None)
        if not source_name:
            return ""

        source_index = getattr(record, "source_index", None)
        total_sources = getattr(record, "total_sources", None)

        if source_index is not None and total_sources is not None:
            return f"[source:{source_index}/{source_name}] "
        return f"[source:{source_name}] "

    def _format_status(self, record: logging.LogRecord) -> str:
        """Format status symbol for the message.

        Args:
            record: Log record with optional status attribute.

        Returns:
            Status symbol with trailing space or empty string.
        """
        status = getattr(record, "status", None)

        # Infer status from log level if not explicitly set
        if status is None:
            if record.levelno >= logging.ERROR:
                status = Status.ERROR
            elif record.levelno >= logging.WARNING:
                status = Status.WARNING

        if status is None:
            return ""

        symbol = self.symbols.get_status_symbol(status)
        if symbol:
            # Colorize based on status
            color_map = {
                Status.SUCCESS: "GREEN",
                Status.WARNING: "YELLOW",
                Status.ERROR: "RED",
                Status.STARTING: "CYAN",
                Status.IN_PROGRESS: "BLUE",
                Status.SKIPPED: "DIM",
            }
            color = color_map.get(status, "RESET")
            return self._colorize(symbol, color) + " "
        return ""

    def _format_extra_fields(self, record: logging.LogRecord) -> str:
        """Format extra fields as key=value pairs.

        Args:
            record: Log record with optional extra_fields attribute.

        Returns:
            Formatted extra fields or empty string.
        """
        extra_fields = getattr(record, "extra_fields", None)
        if not extra_fields:
            return ""

        parts = []
        for key, value in extra_fields.items():
            if isinstance(value, float):
                parts.append(f"{key}={value:.4g}")
            else:
                parts.append(f"{key}={value}")

        if parts:
            return self._colorize(f" ({', '.join(parts)})", "DIM")
        return ""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record for console output.

        Args:
            record: Log record to format.

        Returns:
            Formatted log message.
        """
        # Get components
        elapsed = self._format_elapsed()
        indent = self._get_indent(record)
        status = self._format_status(record)
        branch_ctx = self._format_branch_context(record)
        source_ctx = self._format_source_context(record)
        extra = self._format_extra_fields(record)

        # Build message
        message = record.getMessage()

        # Apply level color to message
        level_color = self.LEVEL_COLORS.get(record.levelname, "RESET")
        message = self._colorize(message, level_color)

        # Combine all parts
        parts = [elapsed, indent, status, branch_ctx, source_ctx, message, extra]
        return "".join(parts)


class FileFormatter(logging.Formatter):
    """File formatter for human-readable log files.

    Produces timestamped, leveled output suitable for file storage and
    later analysis.
    """

    def __init__(self) -> None:
        """Initialize file formatter."""
        super().__init__(
            fmt="%(asctime)s [%(levelname)-5s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record for file output.

        Args:
            record: Log record to format.

        Returns:
            Formatted log message with timestamp and level.
        """
        # Add context info if present
        extras = []

        branch_name = getattr(record, "branch_name", None)
        if branch_name:
            extras.append(f"branch={branch_name}")

        source_name = getattr(record, "source_name", None)
        if source_name:
            extras.append(f"source={source_name}")

        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields:
            for key, value in extra_fields.items():
                if isinstance(value, float):
                    extras.append(f"{key}={value:.4g}")
                else:
                    extras.append(f"{key}={value}")

        # Append extras to message if present
        if extras:
            record.msg = f"{record.msg} [{', '.join(extras)}]"

        return super().format(record)


class JsonFormatter(logging.Formatter):
    """JSON Lines formatter for machine-readable log files.

    Produces one JSON object per line for easy parsing by log aggregation
    systems like ELK, Loki, etc.
    """

    def __init__(self, run_id: Optional[str] = None) -> None:
        """Initialize JSON formatter.

        Args:
            run_id: Run identifier to include in all log entries.
        """
        super().__init__()
        self.run_id = run_id

    def set_run_id(self, run_id: str) -> None:
        """Set the run ID for log entries.

        Args:
            run_id: Unique run identifier.
        """
        self.run_id = run_id

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON.

        Args:
            record: Log record to format.

        Returns:
            JSON string representation of the log record.
        """
        import json

        log_data: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if self.run_id:
            log_data["run_id"] = self.run_id

        # Add optional context fields
        optional_fields = [
            "phase",
            "event_type",
            "status",
            "branch_name",
            "branch_path",
            "branch_index",
            "total_branches",
            "source_name",
            "source_index",
            "total_sources",
            "duration_ms",
        ]

        for field in optional_fields:
            value = getattr(record, field, None)
            if value is not None:
                # Convert enums to their values
                if hasattr(value, "value"):
                    value = value.value
                log_data[field] = value

        # Add extra fields
        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields:
            log_data.update(extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Human-readable duration string (e.g., "2m 5.9s", "1h 23m 45s").
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def format_number(value: float | int, precision: int = 3) -> str:
    """Format a number for display.

    Uses thousands separators for large integers and appropriate
    precision for floats.

    Args:
        value: Number to format.
        precision: Decimal precision for floats.

    Returns:
        Formatted number string.
    """
    if isinstance(value, int):
        return f"{value:,}"
    else:
        return f"{value:.{precision}g}"


def format_table(
    headers: list[str],
    rows: list[list[str]],
    use_unicode: bool = True,
) -> str:
    """Format a simple ASCII table.

    Args:
        headers: Column headers.
        rows: Table rows (list of lists).
        use_unicode: Unused, kept for API consistency.

    Returns:
        Formatted table string.
    """
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Build separator line
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    # Build header row
    header_row = (
        "|" + "|".join(f" {h:<{w}} " for h, w in zip(headers, widths)) + "|"
    )

    # Build data rows
    data_rows = []
    for row in rows:
        data_row = (
            "|"
            + "|".join(
                f" {str(cell):<{w}} " for cell, w in zip(row, widths)
            )
            + "|"
        )
        data_rows.append(data_row)

    # Combine
    lines = [sep, header_row, sep] + data_rows + [sep]
    return "\n".join(lines)


def format_run_header(
    run_name: str,
    start_time: datetime,
    environment_info: Optional[dict[str, str]] = None,
    reproducibility_info: Optional[dict[str, str]] = None,
    use_unicode: bool = True,
) -> str:
    """Format the run header block.

    Args:
        run_name: Name of the run.
        start_time: Run start timestamp.
        environment_info: Optional environment details (Python version, etc.).
        reproducibility_info: Optional reproducibility info (seed, git hash, etc.).
        use_unicode: If True, use Unicode symbols.

    Returns:
        Formatted header block string.
    """
    symbols = Symbols(use_unicode=use_unicode)
    lines = [
        symbols.separator_heavy,
        f"  nirs4all run: {run_name}",
        f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    if environment_info or reproducibility_info:
        lines.append("  " + symbols.separator_light)

    if environment_info:
        lines.append("  Environment:")
        for key, value in environment_info.items():
            lines.append(f"    {key}: {value}")

    if reproducibility_info:
        lines.append("  Reproducibility:")
        parts = [f"{k}={v}" for k, v in reproducibility_info.items()]
        lines.append(f"    {' | '.join(parts)}")

    lines.append(symbols.separator_heavy)
    return "\n".join(lines)


def format_run_footer(
    status: Status,
    duration_seconds: float,
    best_pipeline: Optional[str] = None,
    metrics: Optional[dict[str, float]] = None,
    use_unicode: bool = True,
) -> str:
    """Format the run footer block.

    Args:
        status: Final run status.
        duration_seconds: Total run duration in seconds.
        best_pipeline: Description of best pipeline (optional).
        metrics: Final metrics dict (optional).
        use_unicode: If True, use Unicode symbols.

    Returns:
        Formatted footer block string.
    """
    symbols = Symbols(use_unicode=use_unicode)
    status_symbol = symbols.get_status_symbol(status)
    duration_str = format_duration(duration_seconds)

    lines = [
        symbols.separator_heavy,
        f"  {status_symbol} Run completed in {duration_str}",
    ]

    if best_pipeline:
        lines.append("")
        lines.append(f"  Best pipeline: {best_pipeline}")

    if metrics:
        metrics_str = "  ".join(
            f"{k}={format_number(v)}" for k, v in metrics.items()
        )
        lines.append(f"  Metrics: {metrics_str}")

    lines.append(symbols.separator_heavy)
    return "\n".join(lines)
