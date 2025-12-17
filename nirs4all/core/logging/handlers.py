"""Custom log handlers for nirs4all logging.

This module provides specialized handlers for throttled progress updates,
file rotation, and other custom logging behaviors.
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional


class ThrottledHandler(logging.Handler):
    """Handler that throttles progress messages to avoid flooding.

    Uses time-based and percentage-based throttling to limit progress
    updates while still reporting important milestones.
    """

    # Milestone percentages to always report
    MILESTONES = {10, 25, 50, 75, 90, 100}

    def __init__(
        self,
        base_handler: logging.Handler,
        min_interval: float = 5.0,
    ) -> None:
        """Initialize throttled handler.

        Args:
            base_handler: Handler to forward non-throttled messages to.
            min_interval: Minimum seconds between progress updates.
        """
        super().__init__()
        self.base_handler = base_handler
        self.min_interval = min_interval
        self._last_progress_time: float = 0
        self._last_percentage: int = -1
        self._lock = Lock()

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record, throttling progress messages.

        Args:
            record: Log record to potentially emit.
        """
        # Check if this is a progress message
        is_progress = getattr(record, "is_progress", False)

        if not is_progress:
            # Non-progress messages pass through immediately
            self.base_handler.emit(record)
            return

        # Throttle progress messages
        with self._lock:
            current_time = time.time()
            percentage = getattr(record, "percentage", None)
            is_best = getattr(record, "is_new_best", False)

            should_emit = False

            # Always emit if it's a new best result
            if is_best:
                should_emit = True

            # Always emit milestones
            elif percentage is not None and int(percentage) in self.MILESTONES:
                if int(percentage) != self._last_percentage:
                    should_emit = True
                    self._last_percentage = int(percentage)

            # Time-based throttle
            elif current_time - self._last_progress_time >= self.min_interval:
                should_emit = True

            if should_emit:
                self._last_progress_time = current_time
                self.base_handler.emit(record)

    def reset(self) -> None:
        """Reset throttle state for a new operation."""
        with self._lock:
            self._last_progress_time = 0
            self._last_percentage = -1


class RotatingRunFileHandler(logging.Handler):
    """Handler that writes logs to run-specific files with rotation.

    Creates a new log file for each run, with optional rotation to
    limit total log storage.
    """

    def __init__(
        self,
        log_dir: Path,
        run_id: str,
        max_runs: int = 100,
        json_output: bool = False,
    ) -> None:
        """Initialize rotating file handler.

        Args:
            log_dir: Directory for log files.
            run_id: Unique run identifier.
            max_runs: Maximum number of run logs to keep.
            json_output: If True, also write JSON Lines file.
        """
        super().__init__()
        self.log_dir = Path(log_dir)
        self.run_id = run_id
        self.max_runs = max_runs
        self.json_output = json_output

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create file handlers
        self._log_file = self.log_dir / f"{run_id}.log"
        self._log_handle = open(self._log_file, "w", encoding="utf-8")

        if json_output:
            self._json_file = self.log_dir / f"{run_id}.jsonl"
            self._json_handle = open(self._json_file, "w", encoding="utf-8")
        else:
            self._json_handle = None

        # Rotate old logs
        self._rotate_logs()

    def _rotate_logs(self) -> None:
        """Remove oldest log files if over limit."""
        log_files = sorted(
            self.log_dir.glob("*.log"),
            key=lambda p: p.stat().st_mtime,
        )

        # Keep only the most recent logs (excluding current)
        files_to_remove = log_files[: -(self.max_runs)]
        for log_file in files_to_remove:
            try:
                log_file.unlink()
                # Also remove corresponding JSON file if exists
                json_file = log_file.with_suffix(".jsonl")
                if json_file.exists():
                    json_file.unlink()
            except OSError:
                pass

    def emit(self, record: logging.LogRecord) -> None:
        """Write log record to file(s).

        Args:
            record: Log record to write.
        """
        try:
            # Format for human-readable log
            msg = self.format(record)
            self._log_handle.write(msg + "\n")
            self._log_handle.flush()

            # Format for JSON log if enabled
            if self._json_handle is not None:
                from .formatters import JsonFormatter

                json_formatter = JsonFormatter(run_id=self.run_id)
                json_msg = json_formatter.format(record)
                self._json_handle.write(json_msg + "\n")
                self._json_handle.flush()

        except Exception:
            self.handleError(record)

    def close(self) -> None:
        """Close file handles."""
        try:
            self._log_handle.close()
            if self._json_handle is not None:
                self._json_handle.close()
        except Exception:
            pass
        super().close()


class NullHandler(logging.Handler):
    """Handler that discards all log records.

    Used when logging should be completely silent.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Discard the log record.

        Args:
            record: Log record (ignored).
        """
        pass


class BufferedHandler(logging.Handler):
    """Handler that buffers log records for batch processing.

    Useful for collecting logs during a phase and outputting them
    together, e.g., for branch comparison summaries.
    """

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize buffered handler.

        Args:
            max_size: Maximum number of records to buffer.
        """
        super().__init__()
        self.max_size = max_size
        self._buffer: list[logging.LogRecord] = []
        self._lock = Lock()

    def emit(self, record: logging.LogRecord) -> None:
        """Buffer the log record.

        Args:
            record: Log record to buffer.
        """
        with self._lock:
            if len(self._buffer) < self.max_size:
                self._buffer.append(record)

    def flush_to(self, handler: logging.Handler) -> None:
        """Flush buffered records to another handler.

        Args:
            handler: Handler to send buffered records to.
        """
        with self._lock:
            for record in self._buffer:
                handler.emit(record)
            self._buffer.clear()

    def get_records(self) -> list[logging.LogRecord]:
        """Get buffered records.

        Returns:
            List of buffered log records.
        """
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
