"""Configuration and initialization for nirs4all logging.

This module provides the main configuration interface for the logging system,
including the configure_logging() function and related utilities.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Union

from .context import LogContext, get_current_state, inject_context
from .events import Phase, Status
from .formatters import (
    ConsoleFormatter,
    FileFormatter,
    JsonFormatter,
    Symbols,
    configure_symbols,
    format_duration,
    format_run_footer,
    format_run_header,
)
from .handlers import RotatingRunFileHandler, ThrottledHandler


# Custom log level for TRACE (below DEBUG)
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


class LogConfig:
    """Global logging configuration."""

    def __init__(self) -> None:
        """Initialize with defaults."""
        self.verbose: int = 1
        self.log_file: bool = False
        self.log_dir: Optional[Path] = None
        self.log_format: str = "pretty"
        self.use_unicode: bool = True
        self.use_colors: bool = True
        self.show_elapsed: bool = False
        self.show_progress: bool = True
        self.show_progress_bar: bool = True
        self.json_output: bool = False
        self.run_id: Optional[str] = None

        # File rotation settings
        self.max_log_runs: int = 100
        self.max_log_age_days: Optional[int] = 30
        self.max_log_bytes: Optional[int] = None
        self.compress_logs: bool = True

        # Handlers
        self._console_handler: Optional[logging.Handler] = None
        self._file_handler: Optional[RotatingRunFileHandler] = None
        self._throttle_handler: Optional[ThrottledHandler] = None

        # Configured flag
        self._configured: bool = False


# Global configuration instance
_config = LogConfig()


def _get_level_from_verbose(verbose: int) -> int:
    """Convert verbose level to logging level.

    Args:
        verbose: Verbosity level (0-3).

    Returns:
        Corresponding logging level.
    """
    mapping = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
        3: TRACE,
    }
    return mapping.get(verbose, logging.INFO)


def _detect_color_support() -> bool:
    """Detect if terminal supports ANSI colors.

    Returns:
        True if colors are supported.
    """
    # Check for NO_COLOR environment variable (https://no-color.org/)
    if os.environ.get("NO_COLOR"):
        return False

    # Check for nirs4all-specific override
    if os.environ.get("NIRS4ALL_NO_COLOR", "0") == "1":
        return False

    # Check if stdout is a TTY
    if not hasattr(sys.stdout, "isatty"):
        return False
    if not sys.stdout.isatty():
        return False

    # Check for dumb terminal
    term = os.environ.get("TERM", "")
    if term == "dumb":
        return False

    return True


def _detect_unicode_support() -> bool:
    """Detect if terminal supports Unicode.

    Returns:
        True if Unicode is supported.
    """
    # Check for ASCII-only override
    if os.environ.get("NIRS4ALL_ASCII_ONLY", "0") == "1":
        return False

    # Check stdout encoding
    try:
        encoding = sys.stdout.encoding
        if encoding and encoding.lower() in ("utf-8", "utf8"):
            return True
    except AttributeError:
        pass

    # Check locale
    import locale

    try:
        loc = locale.getlocale()
        if loc and loc[1] and "utf" in loc[1].lower():
            return True
    except (AttributeError, ValueError):
        pass

    return False


def configure_logging(
    verbose: int = 1,
    log_file: bool = False,
    log_dir: Optional[Union[str, Path]] = None,
    log_format: str = "pretty",
    use_unicode: Optional[bool] = None,
    use_colors: Optional[bool] = None,
    show_elapsed: bool = False,
    show_progress: bool = True,
    show_progress_bar: bool = True,
    json_output: bool = False,
    run_id: Optional[str] = None,
    max_log_runs: int = 100,
    max_log_age_days: Optional[int] = 30,
    max_log_bytes: Optional[int] = None,
    compress_logs: bool = True,
) -> None:
    """Configure the nirs4all logging system.

    This function should be called once at application startup to set up
    logging. Subsequent calls will reconfigure the logging system.

    Args:
        verbose: Verbosity level (0=WARNING, 1=INFO, 2=DEBUG, 3=TRACE).
        log_file: If True, write logs to workspace/logs/ directory.
        log_dir: Directory for log files (used if log_file=True).
        log_format: Output format: "pretty", "minimal", or "json".
        use_unicode: Use Unicode symbols (auto-detected if None).
        use_colors: Use ANSI colors (auto-detected if None).
        show_elapsed: Show elapsed time since run start.
        show_progress: Show progress updates for long operations.
        show_progress_bar: Show TTY-aware progress bars.
        json_output: Also write JSON Lines log file.
        run_id: Override run ID (auto-generated if not provided).
        max_log_runs: Maximum number of run logs to keep (count-based rotation).
        max_log_age_days: Maximum age of logs in days (None to disable).
        max_log_bytes: Maximum log file size before rotation (None to disable).
        compress_logs: Whether to gzip rotated log files.
    """
    global _config

    # Apply environment variable overrides
    if os.environ.get("NIRS4ALL_LOG_LEVEL"):
        level_name = os.environ["NIRS4ALL_LOG_LEVEL"].upper()
        level_map = {"WARNING": 0, "INFO": 1, "DEBUG": 2, "TRACE": 3}
        verbose = level_map.get(level_name, verbose)

    # Auto-detect capabilities if not specified
    if use_unicode is None:
        use_unicode = _detect_unicode_support()
    if use_colors is None:
        use_colors = _detect_color_support()

    # Store configuration
    _config.verbose = verbose
    _config.log_file = log_file
    _config.log_dir = Path(log_dir) if log_dir else None
    _config.log_format = log_format
    _config.use_unicode = use_unicode
    _config.use_colors = use_colors
    _config.show_elapsed = show_elapsed
    _config.show_progress = show_progress
    _config.show_progress_bar = show_progress_bar
    _config.json_output = json_output
    _config.run_id = run_id
    _config.max_log_runs = max_log_runs
    _config.max_log_age_days = max_log_age_days
    _config.max_log_bytes = max_log_bytes
    _config.compress_logs = compress_logs

    # Configure global symbols
    configure_symbols(use_unicode=use_unicode)

    # Configure progress bar settings
    from .progress import configure_progress
    configure_progress(
        use_unicode=use_unicode,
        use_colors=use_colors,
    )

    # Get the root nirs4all logger
    root_logger = logging.getLogger("nirs4all")

    # Clear existing handlers
    root_logger.handlers.clear()

    # Set level
    log_level = _get_level_from_verbose(verbose)
    root_logger.setLevel(log_level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Create formatter based on format
    if log_format == "json":
        console_formatter = JsonFormatter(run_id=run_id)
    elif log_format == "minimal":
        console_formatter = ConsoleFormatter(
            use_colors=False,
            show_elapsed=False,
            use_unicode=False,
        )
    else:  # "pretty"
        console_formatter = ConsoleFormatter(
            use_colors=use_colors,
            show_elapsed=show_elapsed,
            use_unicode=use_unicode,
        )

    console_handler.setFormatter(console_formatter)

    # Wrap with throttle handler for progress messages
    if show_progress:
        _config._throttle_handler = ThrottledHandler(console_handler)
        root_logger.addHandler(_config._throttle_handler)
    else:
        root_logger.addHandler(console_handler)

    _config._console_handler = console_handler

    # Configure file handler if enabled
    if log_file and log_dir:
        from .context import _generate_run_id

        actual_run_id = run_id or _generate_run_id()

        file_handler = RotatingRunFileHandler(
            log_dir=Path(log_dir),
            run_id=actual_run_id,
            max_runs=max_log_runs,
            max_age_days=max_log_age_days,
            max_bytes=max_log_bytes,
            compress_rotated=compress_logs,
            json_output=json_output,
        )
        file_handler.setLevel(logging.DEBUG)  # Capture all levels to file
        file_handler.setFormatter(FileFormatter())
        root_logger.addHandler(file_handler)
        _config._file_handler = file_handler

    # Prevent propagation to root logger
    root_logger.propagate = False

    _config._configured = True


def get_config() -> LogConfig:
    """Get the current logging configuration.

    Returns:
        Current LogConfig instance.
    """
    return _config


def is_configured() -> bool:
    """Check if logging has been configured.

    Returns:
        True if configure_logging() has been called.
    """
    return _config._configured


def reset_logging() -> None:
    """Reset logging configuration to defaults.

    This clears all handlers and resets the configuration. Useful for testing.
    Ensures file handlers are properly closed to avoid issues on Windows where
    open file handles prevent file deletion.
    """
    global _config

    # Close stored handlers explicitly first (before removing from logger)
    # This ensures file handles are released on Windows
    if _config._file_handler is not None:
        try:
            _config._file_handler.flush()
            _config._file_handler.close()
        except Exception:
            pass
        _config._file_handler = None

    if _config._throttle_handler is not None:
        try:
            _config._throttle_handler.close()
        except Exception:
            pass
        _config._throttle_handler = None

    if _config._console_handler is not None:
        try:
            _config._console_handler.close()
        except Exception:
            pass
        _config._console_handler = None

    # Clear handlers from root logger
    root_logger = logging.getLogger("nirs4all")
    for handler in root_logger.handlers[:]:
        try:
            handler.close()
        except Exception:
            pass
        root_logger.removeHandler(handler)

    # Reset configuration
    _config = LogConfig()


class Nirs4allLogger(logging.Logger):
    """Extended logger with nirs4all-specific methods.

    Provides convenience methods for structured logging with status indicators,
    progress reporting, and context-aware logging.
    """

    def __init__(self, name: str, level: int = logging.NOTSET) -> None:
        """Initialize the logger.

        Args:
            name: Logger name.
            level: Logging level.
        """
        super().__init__(name, level)

    def _log_with_context(
        self,
        level: int,
        msg: str,
        status: Optional[Status] = None,
        extra_fields: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Log a message with context injection.

        Args:
            level: Logging level.
            msg: Log message.
            status: Optional status indicator.
            extra_fields: Additional fields to include in the log.
            **kwargs: Additional logging kwargs (exc_info, etc.).
        """
        extra = kwargs.pop("extra", {})

        if status:
            extra["status"] = status

        if extra_fields:
            extra["extra_fields"] = extra_fields

        # Let the log record be created normally
        super().log(level, msg, extra=extra, **kwargs)

    def makeRecord(
        self,
        name: str,
        level: int,
        fn: str,
        lno: int,
        msg: object,
        args: tuple,
        exc_info: Any,
        func: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
        sinfo: Optional[str] = None,
    ) -> logging.LogRecord:
        """Create a LogRecord with context injection.

        Overrides the standard makeRecord to inject run context.
        """
        record = super().makeRecord(
            name, level, fn, lno, msg, args, exc_info, func, extra, sinfo
        )
        return inject_context(record)

    def trace(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log at TRACE level (below DEBUG).

        Args:
            msg: Log message.
            *args: Message formatting args.
            **kwargs: Additional logging kwargs.
        """
        if self.isEnabledFor(TRACE):
            self._log(TRACE, msg, args, **kwargs)

    def success(
        self, msg: str, *args: Any, extra_fields: Optional[dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Log a success message.

        Args:
            msg: Log message.
            *args: Message formatting args.
            extra_fields: Additional context fields.
            **kwargs: Additional logging kwargs.
        """
        self._log_with_context(
            logging.INFO, msg, status=Status.SUCCESS, extra_fields=extra_fields, **kwargs
        )

    def starting(
        self, msg: str, *args: Any, extra_fields: Optional[dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Log a starting/beginning message.

        Args:
            msg: Log message.
            *args: Message formatting args.
            extra_fields: Additional context fields.
            **kwargs: Additional logging kwargs.
        """
        self._log_with_context(
            logging.INFO, msg, status=Status.STARTING, extra_fields=extra_fields, **kwargs
        )

    def progress(
        self,
        operation: str,
        current: int,
        total: int,
        best_score: Optional[float] = None,
        is_new_best: bool = False,
        extra_fields: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Log a progress update.

        Progress messages are automatically throttled to avoid flooding.

        Args:
            operation: Name of the operation in progress.
            current: Current step number.
            total: Total number of steps.
            best_score: Current best score (optional).
            is_new_best: If True, this update reports a new best result.
            extra_fields: Additional context fields.
            **kwargs: Additional logging kwargs.
        """
        percentage = (current / total * 100) if total > 0 else 0

        msg_parts = [f"Progress: {current}/{total} ({percentage:.0f}%)"]
        if best_score is not None:
            msg_parts.append(f"best: {best_score:.4g}")

        msg = " -- ".join(msg_parts)

        extra = kwargs.pop("extra", {})
        extra["is_progress"] = True
        extra["percentage"] = percentage
        extra["is_new_best"] = is_new_best
        extra["status"] = Status.IN_PROGRESS

        if extra_fields:
            extra["extra_fields"] = extra_fields

        self.info(msg, extra=extra, **kwargs)

    def metric(
        self,
        name: str,
        value: float,
        scope: str = "cv",
        fold: Optional[int] = None,
        pipeline: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a metric value.

        Args:
            name: Metric name (e.g., "RMSE", "R2").
            value: Metric value.
            scope: Metric scope ("cv", "train", "test", "fold").
            fold: Fold number (for per-fold metrics).
            pipeline: Pipeline identifier.
            **kwargs: Additional logging kwargs.
        """
        extra_fields = {"scope": scope}
        if fold is not None:
            extra_fields["fold"] = fold
        if pipeline:
            extra_fields["pipeline"] = pipeline

        self._log_with_context(
            logging.DEBUG,
            f"{name}={value:.4g}",
            extra_fields=extra_fields,
            **kwargs,
        )

    def artifact(
        self, artifact_type: str, path: Union[str, Path], size_bytes: Optional[int] = None, **kwargs: Any
    ) -> None:
        """Log an artifact save.

        Args:
            artifact_type: Type of artifact ("model", "report", "predictions").
            path: Path where artifact was saved.
            size_bytes: Size of artifact in bytes.
            **kwargs: Additional logging kwargs.
        """
        extra_fields: dict[str, Any] = {"artifact_type": artifact_type}
        if size_bytes:
            extra_fields["size_bytes"] = size_bytes

        self._log_with_context(
            logging.INFO,
            f"{artifact_type.capitalize()} saved: {path}",
            status=Status.SUCCESS,
            extra_fields=extra_fields,
            **kwargs,
        )

    def phase_start(self, phase: Phase, **extra_fields: Any) -> None:
        """Log the start of a workflow phase.

        Args:
            phase: The phase being started.
            **extra_fields: Additional context fields.
        """
        phase_names = {
            Phase.INIT: "Initializing",
            Phase.DATA: "Loading data",
            Phase.SPLIT: "Building cross-validation splits",
            Phase.GENERATE: "Generating pipeline candidates",
            Phase.BRANCH: "Processing branches",
            Phase.SOURCE: "Processing sources",
            Phase.EVALUATE: "Evaluating pipelines",
            Phase.STACK: "Building stacking ensemble",
            Phase.TRAIN: "Training best model",
            Phase.PREDICT: "Generating predictions",
            Phase.EXPORT: "Saving artifacts",
            Phase.COMPLETE: "Completing run",
        }

        msg = phase_names.get(phase, f"Starting {phase.value}")
        self.starting(f"{msg}...", extra_fields=extra_fields or None)

    def phase_complete(
        self, phase: Phase, duration_seconds: Optional[float] = None, **extra_fields: Any
    ) -> None:
        """Log the completion of a workflow phase.

        Args:
            phase: The phase that completed.
            duration_seconds: Phase duration in seconds.
            **extra_fields: Additional context fields.
        """
        parts = [f"{phase.value} complete"]
        if duration_seconds is not None:
            parts.append(f"({format_duration(duration_seconds)})")

        fields = dict(extra_fields) if extra_fields else None
        if duration_seconds is not None:
            fields = fields or {}
            fields["duration_s"] = duration_seconds

        self.success(" ".join(parts), extra_fields=fields)

    def run_header(
        self,
        run_name: str,
        environment_info: Optional[dict[str, str]] = None,
        reproducibility_info: Optional[dict[str, str]] = None,
    ) -> None:
        """Log the run header block.

        Args:
            run_name: Name of the run.
            environment_info: Environment details (Python version, etc.).
            reproducibility_info: Reproducibility info (seed, git hash, etc.).
        """
        header = format_run_header(
            run_name=run_name,
            start_time=datetime.now(),
            environment_info=environment_info,
            reproducibility_info=reproducibility_info,
            use_unicode=_config.use_unicode,
        )
        # Print header directly to avoid formatting
        print(header)

    def run_footer(
        self,
        status: Status,
        duration_seconds: float,
        best_pipeline: Optional[str] = None,
        metrics: Optional[dict[str, float]] = None,
    ) -> None:
        """Log the run footer block.

        Args:
            status: Final run status.
            duration_seconds: Total run duration.
            best_pipeline: Best pipeline description.
            metrics: Final metrics.
        """
        footer = format_run_footer(
            status=status,
            duration_seconds=duration_seconds,
            best_pipeline=best_pipeline,
            metrics=metrics,
            use_unicode=_config.use_unicode,
        )
        # Print footer directly to avoid formatting
        print(footer)


# Replace default logger class
logging.setLoggerClass(Nirs4allLogger)


def get_logger(name: str) -> Nirs4allLogger:
    """Get a logger for the specified module.

    This is the primary interface for obtaining a logger in nirs4all modules.
    The logger is automatically configured with context injection and
    nirs4all-specific methods.

    Args:
        name: Logger name, typically __name__.

    Returns:
        Configured Nirs4allLogger instance.

    Example:
        >>> from nirs4all.core.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing data")
    """
    # Ensure name is under nirs4all namespace
    if not name.startswith("nirs4all"):
        name = f"nirs4all.{name}"

    logger = logging.getLogger(name)

    # Ensure it's our custom logger class
    if not isinstance(logger, Nirs4allLogger):
        # This can happen if logging was already configured with standard class
        # Convert it by setting the class (logger retains handlers)
        logger.__class__ = Nirs4allLogger

    return logger  # type: ignore
