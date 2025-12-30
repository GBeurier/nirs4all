"""Custom log handlers for nirs4all logging.

This module provides specialized handlers for throttled progress updates,
file rotation, and other custom logging behaviors.
"""

from __future__ import annotations

import gzip
import logging
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Optional


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

    def close(self) -> None:
        """Close this handler and the base handler."""
        try:
            if self.base_handler is not None:
                self.base_handler.close()
        except Exception:
            pass
        super().close()


class RotatingRunFileHandler(logging.Handler):
    """Handler that writes logs to run-specific files with rotation.

    Creates a new log file for each run, with optional rotation to
    limit total log storage. Supports both count-based and age-based
    rotation policies.

    Features:
        - Separate log files per run
        - Count-based rotation (max_runs)
        - Age-based rotation (max_age_days)
        - Size-based rotation (max_bytes)
        - Optional gzip compression for rotated logs
        - Optional JSON Lines output
    """

    def __init__(
        self,
        log_dir: Path,
        run_id: str,
        max_runs: int = 100,
        max_age_days: Optional[int] = 30,
        max_bytes: Optional[int] = None,
        compress_rotated: bool = True,
        json_output: bool = False,
    ) -> None:
        """Initialize rotating file handler.

        Args:
            log_dir: Directory for log files.
            run_id: Unique run identifier.
            max_runs: Maximum number of run logs to keep.
            max_age_days: Maximum age of logs in days (None to disable).
            max_bytes: Maximum size of a single log file before rotation (None to disable).
            compress_rotated: Whether to gzip old log files.
            json_output: If True, also write JSON Lines file.
        """
        super().__init__()
        self.log_dir = Path(log_dir)
        self.run_id = run_id
        self.max_runs = max_runs
        self.max_age_days = max_age_days
        self.max_bytes = max_bytes
        self.compress_rotated = compress_rotated
        self.json_output = json_output

        self._lock = Lock()
        self._rotation_counter = 0

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create file handlers
        self._log_file = self.log_dir / f"{run_id}.log"
        self._log_handle = open(self._log_file, "w", encoding="utf-8")
        self._current_size = 0

        if json_output:
            self._json_file = self.log_dir / f"{run_id}.jsonl"
            self._json_handle = open(self._json_file, "w", encoding="utf-8")
        else:
            self._json_handle = None
            self._json_file = None

        # Rotate old logs
        self._rotate_logs()

    def _rotate_logs(self) -> None:
        """Remove oldest log files if over limit.

        Applies both count-based and age-based rotation policies.
        """
        now = datetime.now()

        # Get all log files
        log_files = list(self.log_dir.glob("*.log"))
        log_files.extend(self.log_dir.glob("*.log.gz"))

        # Sort by modification time (oldest first)
        log_files = sorted(
            log_files,
            key=lambda p: p.stat().st_mtime,
        )

        files_to_remove: list[Path] = []

        # Age-based rotation
        if self.max_age_days is not None:
            cutoff = now - timedelta(days=self.max_age_days)
            for log_file in log_files:
                try:
                    mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if mtime < cutoff:
                        files_to_remove.append(log_file)
                except OSError:
                    pass

        # Count-based rotation (excluding current file)
        current_files = [f for f in log_files if f not in files_to_remove]
        if len(current_files) >= self.max_runs:
            # Remove oldest files to get under limit
            excess = len(current_files) - self.max_runs + 1  # +1 for current
            files_to_remove.extend(current_files[:excess])

        # Remove files and their JSON companions
        for log_file in set(files_to_remove):
            try:
                log_file.unlink()
                # Also remove corresponding JSON file if exists
                json_file = log_file.with_suffix(".jsonl")
                if json_file.exists():
                    json_file.unlink()
                json_gz = log_file.with_suffix(".jsonl.gz")
                if json_gz.exists():
                    json_gz.unlink()
            except OSError:
                pass

    def _rotate_current_file(self) -> None:
        """Rotate the current log file when size limit is reached."""
        if self._log_handle is None:
            return

        with self._lock:
            # Close current file
            self._log_handle.close()

            # Rename with rotation counter
            self._rotation_counter += 1
            rotated_name = self._log_file.with_suffix(f".{self._rotation_counter}.log")
            self._log_file.rename(rotated_name)

            # Compress if enabled
            if self.compress_rotated:
                self._compress_file(rotated_name)

            # Open new file
            self._log_handle = open(self._log_file, "w", encoding="utf-8")
            self._current_size = 0

    def _compress_file(self, file_path: Path) -> None:
        """Compress a file with gzip.

        Args:
            file_path: Path to file to compress.
        """
        try:
            gz_path = file_path.with_suffix(file_path.suffix + ".gz")
            with open(file_path, "rb") as f_in:
                with gzip.open(gz_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            file_path.unlink()
        except Exception:
            pass  # Keep uncompressed if compression fails

    def emit(self, record: logging.LogRecord) -> None:
        """Write log record to file(s).

        Args:
            record: Log record to write.
        """
        try:
            # Format for human-readable log
            msg = self.format(record)

            with self._lock:
                # Check size limit before writing
                msg_bytes = len(msg.encode("utf-8")) + 1  # +1 for newline
                if self.max_bytes and self._current_size + msg_bytes > self.max_bytes:
                    self._rotate_current_file()

                self._log_handle.write(msg + "\n")
                self._log_handle.flush()
                self._current_size += msg_bytes

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
        with self._lock:
            try:
                if self._log_handle is not None:
                    self._log_handle.close()
                    self._log_handle = None
                if self._json_handle is not None:
                    self._json_handle.close()
                    self._json_handle = None
            except Exception:
                pass
        super().close()

    def get_log_file_path(self) -> Path:
        """Get the path to the current log file.

        Returns:
            Path to the current log file.
        """
        return self._log_file

    def get_json_file_path(self) -> Optional[Path]:
        """Get the path to the current JSON log file.

        Returns:
            Path to JSON log file, or None if not enabled.
        """
        return self._json_file


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
