"""Progress bar support for nirs4all logging.

This module provides TTY-aware progress bars for tracking long-running operations
at multiple levels: run, pipeline, evaluation, and training.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Callable, Iterator, Optional, Sequence, TypeVar

from .formatters import Symbols, format_duration, get_symbols

T = TypeVar("T")


def _is_tty() -> bool:
    """Check if stdout is a terminal."""
    try:
        return sys.stdout.isatty()
    except AttributeError:
        return False


def _supports_ansi() -> bool:
    """Check if terminal supports ANSI escape codes."""
    if not _is_tty():
        return False

    import os

    # Windows check
    if sys.platform == "win32":
        # Check for Windows Terminal or ConEmu
        if os.environ.get("WT_SESSION") or os.environ.get("ConEmuANSI") == "ON":
            return True
        # Enable ANSI on Windows 10+
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(
                kernel32.GetStdHandle(-11), 7
            )  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
            return True
        except Exception:
            return False

    # Unix-like systems generally support ANSI
    term = os.environ.get("TERM", "")
    return term != "dumb"


# ANSI escape codes
CLEAR_LINE = "\033[2K"
MOVE_TO_START = "\r"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
MOVE_UP = "\033[{n}A"


@dataclass
class ProgressConfig:
    """Configuration for progress bar display.

    Attributes:
        bar_width: Width of the progress bar in characters.
        show_percentage: Whether to show percentage.
        show_count: Whether to show current/total count.
        show_elapsed: Whether to show elapsed time.
        show_eta: Whether to show estimated time remaining.
        show_rate: Whether to show items per second.
        refresh_rate: Minimum seconds between updates.
        use_unicode: Use Unicode box-drawing characters.
        use_colors: Use ANSI colors.
    """
    bar_width: int = 30
    show_percentage: bool = True
    show_count: bool = True
    show_elapsed: bool = True
    show_eta: bool = True
    show_rate: bool = False
    refresh_rate: float = 0.1
    use_unicode: bool = True
    use_colors: bool = True


# Global progress config
_progress_config = ProgressConfig()


def configure_progress(
    bar_width: int = 30,
    show_percentage: bool = True,
    show_count: bool = True,
    show_elapsed: bool = True,
    show_eta: bool = True,
    show_rate: bool = False,
    refresh_rate: float = 0.1,
    use_unicode: bool = True,
    use_colors: bool = True,
) -> None:
    """Configure global progress bar settings.

    Args:
        bar_width: Width of progress bar in characters.
        show_percentage: Show percentage completion.
        show_count: Show current/total count.
        show_elapsed: Show elapsed time.
        show_eta: Show estimated time remaining.
        show_rate: Show items per second.
        refresh_rate: Minimum seconds between display updates.
        use_unicode: Use Unicode characters for bar.
        use_colors: Use ANSI colors.
    """
    global _progress_config
    _progress_config = ProgressConfig(
        bar_width=bar_width,
        show_percentage=show_percentage,
        show_count=show_count,
        show_elapsed=show_elapsed,
        show_eta=show_eta,
        show_rate=show_rate,
        refresh_rate=refresh_rate,
        use_unicode=use_unicode,
        use_colors=use_colors,
    )


class ProgressBar:
    """TTY-aware progress bar for tracking iterations.

    Provides a clean progress display that automatically adapts to terminal
    capabilities. Falls back to periodic line-based updates when not in a TTY.

    Example:
        >>> with ProgressBar(total=100, description="Processing") as pbar:
        ...     for i in range(100):
        ...         # do work
        ...         pbar.update(1)

        >>> # Or with iterator
        >>> for item in ProgressBar.wrap(items, description="Processing"):
        ...     # do work with item
    """

    # Bar characters
    FILLED_UNICODE = "█"
    EMPTY_UNICODE = "░"
    FILLED_ASCII = "#"
    EMPTY_ASCII = "-"

    # Colors
    COLOR_RESET = "\033[0m"
    COLOR_CYAN = "\033[36m"
    COLOR_GREEN = "\033[32m"
    COLOR_YELLOW = "\033[33m"
    COLOR_DIM = "\033[2m"

    def __init__(
        self,
        total: int,
        description: str = "",
        config: Optional[ProgressConfig] = None,
        leave: bool = True,
        disable: bool = False,
        unit: str = "it",
        file: Any = None,
        initial: int = 0,
        ncols: Optional[int] = None,
    ) -> None:
        """Initialize progress bar.

        Args:
            total: Total number of items.
            description: Description text shown before bar.
            config: Progress bar configuration (uses global if None).
            leave: Whether to leave bar visible after completion.
            disable: Whether to disable all output.
            unit: Unit name for items (e.g., "it", "samples", "pipelines").
            file: File to write to (default: stdout).
            initial: Initial count value.
            ncols: Number of columns for bar (auto-detect if None).
        """
        self.total = total
        self.description = description
        self.config = config or _progress_config
        self.leave = leave
        self.disable = disable
        self.unit = unit
        self.file = file or sys.stdout
        self.current = initial
        self.ncols = ncols

        # State
        self._start_time: Optional[float] = None
        self._last_update_time: float = 0
        self._last_print_time: float = 0
        self._lock = Lock()
        self._closed = False

        # Terminal capabilities
        self._is_tty = _is_tty() and hasattr(self.file, 'isatty') and self.file.isatty()
        self._supports_ansi = _supports_ansi() if self._is_tty else False

        # Best score tracking (for ML pipelines)
        self._best_score: Optional[float] = None
        self._best_label: str = "best"

    def __enter__(self) -> "ProgressBar":
        """Enter context manager."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        self.close()

    def start(self) -> None:
        """Start the progress bar."""
        if self.disable:
            return

        self._start_time = time.time()
        self._last_update_time = self._start_time

        if self._supports_ansi:
            # Hide cursor during progress
            self._write(HIDE_CURSOR)

        self._display()

    def update(
        self,
        n: int = 1,
        best_score: Optional[float] = None,
        best_label: str = "best",
    ) -> None:
        """Update progress by n items.

        Args:
            n: Number of items completed.
            best_score: Current best score (for ML tracking).
            best_label: Label for best score display.
        """
        with self._lock:
            self.current += n

            if best_score is not None:
                self._best_score = best_score
                self._best_label = best_label

            # Skip display if disabled
            if self.disable:
                return

            # Throttle updates
            now = time.time()
            if now - self._last_print_time >= self.config.refresh_rate:
                self._display()
                self._last_print_time = now

    def set_description(self, description: str) -> None:
        """Update description text.

        Args:
            description: New description text.
        """
        self.description = description
        if not self.disable:
            self._display()

    def set_postfix(self, **kwargs: Any) -> None:
        """Set postfix values (shown after bar).

        Args:
            **kwargs: Key-value pairs to display.
        """
        # Store postfix for display
        self._postfix = kwargs
        if not self.disable:
            self._display()

    def close(self) -> None:
        """Close the progress bar."""
        if self._closed or self.disable:
            return

        with self._lock:
            self._closed = True

            # Final display
            self._display(force=True)

            if self._supports_ansi:
                # Show cursor
                self._write(SHOW_CURSOR)

            if self.leave:
                self._write("\n")
            else:
                # Clear the line
                self._write(CLEAR_LINE + MOVE_TO_START)

            self.file.flush()

    def _write(self, text: str) -> None:
        """Write text to output file."""
        try:
            self.file.write(text)
            self.file.flush()
        except Exception:
            pass

    def _get_bar(self) -> tuple[str, float]:
        """Generate the progress bar string and percentage.

        Returns:
            Tuple of (bar_string, percentage).
        """
        if self.total == 0:
            percentage = 100.0
            filled = self.config.bar_width
        else:
            percentage = min(100.0, self.current / self.total * 100)
            filled = int(self.config.bar_width * self.current / self.total)

        empty = self.config.bar_width - filled

        if self.config.use_unicode:
            bar = self.FILLED_UNICODE * filled + self.EMPTY_UNICODE * empty
        else:
            bar = self.FILLED_ASCII * filled + self.EMPTY_ASCII * empty

        return bar, percentage

    def _get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def _get_eta(self) -> Optional[float]:
        """Get estimated time remaining in seconds."""
        elapsed = self._get_elapsed()
        if elapsed <= 0 or self.current <= 0:
            return None

        rate = self.current / elapsed
        remaining = self.total - self.current

        if rate > 0:
            return remaining / rate
        return None

    def _format_time(self, seconds: float) -> str:
        """Format time as human-readable string."""
        return format_duration(seconds)

    def _display(self, force: bool = False) -> None:
        """Display the progress bar."""
        if self.disable:
            return

        bar, percentage = self._get_bar()
        elapsed = self._get_elapsed()
        eta = self._get_eta()

        # Build components
        parts = []

        # Description
        if self.description:
            parts.append(self.description)

        # Colorize bar if supported
        if self.config.use_colors and self._supports_ansi:
            if percentage >= 100:
                bar_str = f"{self.COLOR_GREEN}|{bar}|{self.COLOR_RESET}"
            elif percentage >= 50:
                bar_str = f"{self.COLOR_CYAN}|{bar}|{self.COLOR_RESET}"
            else:
                bar_str = f"|{bar}|"
        else:
            bar_str = f"|{bar}|"
        parts.append(bar_str)

        # Percentage
        if self.config.show_percentage:
            parts.append(f"{percentage:5.1f}%")

        # Count
        if self.config.show_count:
            parts.append(f"{self.current}/{self.total}")

        # Elapsed
        if self.config.show_elapsed:
            elapsed_str = self._format_time(elapsed)
            parts.append(f"[{elapsed_str}")

            # ETA
            if self.config.show_eta and eta is not None and self.current < self.total:
                eta_str = self._format_time(eta)
                parts[-1] += f"<{eta_str}"

            parts[-1] += "]"

        # Best score
        if self._best_score is not None:
            if self.config.use_colors and self._supports_ansi:
                parts.append(f"{self.COLOR_YELLOW}{self._best_label}: {self._best_score:.4g}{self.COLOR_RESET}")
            else:
                parts.append(f"{self._best_label}: {self._best_score:.4g}")

        # Postfix
        if hasattr(self, "_postfix") and self._postfix:
            postfix_parts = []
            for k, v in self._postfix.items():
                if isinstance(v, float):
                    postfix_parts.append(f"{k}={v:.4g}")
                else:
                    postfix_parts.append(f"{k}={v}")
            if postfix_parts:
                parts.append(", ".join(postfix_parts))

        line = " ".join(parts)

        # Output
        if self._supports_ansi:
            # Overwrite current line
            self._write(CLEAR_LINE + MOVE_TO_START + line)
        else:
            # Non-TTY: print periodic updates
            now = time.time()
            # Print at milestones or when forced
            should_print = force or (
                self.current == self.total or
                int(percentage) % 25 == 0 and int(percentage) != getattr(self, "_last_milestone", -1)
            )
            if should_print:
                self._last_milestone = int(percentage)
                self._write(f"  * {line}\n")

    @classmethod
    def wrap(
        cls,
        iterable: Sequence[T],
        description: str = "",
        **kwargs: Any,
    ) -> Iterator[T]:
        """Wrap an iterable with a progress bar.

        Args:
            iterable: Iterable to wrap.
            description: Progress bar description.
            **kwargs: Additional ProgressBar arguments.

        Yields:
            Items from the iterable.
        """
        total = len(iterable)
        with cls(total=total, description=description, **kwargs) as pbar:
            for item in iterable:
                yield item
                pbar.update(1)


class MultiLevelProgress:
    """Multi-level progress tracking for nested operations.

    Supports tracking progress at multiple levels:
    - Run level (overall run progress)
    - Pipeline level (pipeline evaluation within run)
    - Fold level (CV folds within pipeline)
    - Training level (epochs/batches within training)

    Example:
        >>> progress = MultiLevelProgress(
        ...     run_total=5,
        ...     run_description="Evaluating datasets"
        ... )
        >>>
        >>> with progress.run_level() as run_pbar:
        ...     for dataset in datasets:
        ...         with progress.pipeline_level(total=10) as pipe_pbar:
        ...             for pipeline in pipelines:
        ...                 # evaluate
        ...                 pipe_pbar.update(1)
        ...         run_pbar.update(1)
    """

    def __init__(
        self,
        run_total: Optional[int] = None,
        run_description: str = "Run progress",
        config: Optional[ProgressConfig] = None,
        disable: bool = False,
    ) -> None:
        """Initialize multi-level progress.

        Args:
            run_total: Total items for run level.
            run_description: Description for run level.
            config: Progress configuration.
            disable: Disable all progress display.
        """
        self.config = config or _progress_config
        self.disable = disable

        self._run_total = run_total
        self._run_description = run_description

        # Active bars
        self._run_bar: Optional[ProgressBar] = None
        self._pipeline_bar: Optional[ProgressBar] = None
        self._fold_bar: Optional[ProgressBar] = None
        self._training_bar: Optional[ProgressBar] = None

        self._lock = Lock()

    def run_level(
        self,
        total: Optional[int] = None,
        description: Optional[str] = None,
    ) -> ProgressBar:
        """Get run-level progress bar.

        Args:
            total: Override total (uses init value if None).
            description: Override description.

        Returns:
            ProgressBar for run level.
        """
        with self._lock:
            if self._run_bar is not None:
                self._run_bar.close()

            self._run_bar = ProgressBar(
                total=total or self._run_total or 1,
                description=description or self._run_description,
                config=self.config,
                disable=self.disable,
                unit="datasets",
            )
            return self._run_bar

    def pipeline_level(
        self,
        total: int,
        description: str = "Evaluating pipelines",
    ) -> ProgressBar:
        """Get pipeline-level progress bar.

        Args:
            total: Total number of pipelines.
            description: Description text.

        Returns:
            ProgressBar for pipeline level.
        """
        with self._lock:
            if self._pipeline_bar is not None:
                self._pipeline_bar.close()

            self._pipeline_bar = ProgressBar(
                total=total,
                description=f"  {description}",  # Indent under run
                config=self.config,
                disable=self.disable,
                unit="pipelines",
            )
            return self._pipeline_bar

    def fold_level(
        self,
        total: int,
        description: str = "Cross-validation",
    ) -> ProgressBar:
        """Get fold-level progress bar.

        Args:
            total: Total number of folds.
            description: Description text.

        Returns:
            ProgressBar for fold level.
        """
        with self._lock:
            if self._fold_bar is not None:
                self._fold_bar.close()

            self._fold_bar = ProgressBar(
                total=total,
                description=f"    {description}",  # Double indent
                config=self.config,
                disable=self.disable,
                unit="folds",
            )
            return self._fold_bar

    def training_level(
        self,
        total: int,
        description: str = "Training",
    ) -> ProgressBar:
        """Get training-level progress bar.

        Args:
            total: Total epochs or batches.
            description: Description text.

        Returns:
            ProgressBar for training level.
        """
        with self._lock:
            if self._training_bar is not None:
                self._training_bar.close()

            self._training_bar = ProgressBar(
                total=total,
                description=f"      {description}",  # Triple indent
                config=self.config,
                disable=self.disable,
                unit="epochs",
            )
            return self._training_bar

    def close_all(self) -> None:
        """Close all active progress bars."""
        with self._lock:
            for bar in [self._training_bar, self._fold_bar,
                       self._pipeline_bar, self._run_bar]:
                if bar is not None:
                    bar.close()

            self._run_bar = None
            self._pipeline_bar = None
            self._fold_bar = None
            self._training_bar = None


class EvaluationProgress:
    """Specialized progress tracker for pipeline evaluation.

    Provides ML-specific tracking with best score updates and
    automatic milestone reporting.

    Example:
        >>> progress = EvaluationProgress(
        ...     total_pipelines=42,
        ...     metric_name="RMSE",
        ...     higher_is_better=False
        ... )
        >>>
        >>> with progress:
        ...     for pipeline in pipelines:
        ...         score = evaluate(pipeline)
        ...         progress.update(score=score, pipeline_name="SavGol+PLS")
    """

    def __init__(
        self,
        total_pipelines: int,
        metric_name: str = "score",
        higher_is_better: bool = False,
        description: str = "Evaluating pipelines",
        config: Optional[ProgressConfig] = None,
        disable: bool = False,
    ) -> None:
        """Initialize evaluation progress.

        Args:
            total_pipelines: Total number of pipelines to evaluate.
            metric_name: Name of the metric being optimized.
            higher_is_better: True if higher metric values are better.
            description: Progress bar description.
            config: Progress configuration.
            disable: Disable progress display.
        """
        self.total_pipelines = total_pipelines
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.description = description
        self.config = config or _progress_config
        self.disable = disable

        self._best_score: Optional[float] = None
        self._best_pipeline: Optional[str] = None
        self._completed = 0
        self._bar: Optional[ProgressBar] = None

    def __enter__(self) -> "EvaluationProgress":
        """Enter context manager."""
        self._bar = ProgressBar(
            total=self.total_pipelines,
            description=self.description,
            config=self.config,
            disable=self.disable,
            unit="pipelines",
        )
        self._bar.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        if self._bar is not None:
            self._bar.close()

    def update(
        self,
        score: Optional[float] = None,
        pipeline_name: Optional[str] = None,
        n: int = 1,
    ) -> bool:
        """Update progress with optional score.

        Args:
            score: Score for the completed pipeline.
            pipeline_name: Name of the pipeline.
            n: Number of pipelines completed.

        Returns:
            True if this was a new best score.
        """
        self._completed += n
        is_new_best = False

        if score is not None:
            # Check if this is a new best
            if self._best_score is None:
                is_new_best = True
            elif self.higher_is_better:
                is_new_best = score > self._best_score
            else:
                is_new_best = score < self._best_score

            if is_new_best:
                self._best_score = score
                self._best_pipeline = pipeline_name

        if self._bar is not None:
            self._bar.update(
                n=n,
                best_score=self._best_score,
                best_label=f"best {self.metric_name}",
            )

        return is_new_best

    @property
    def best_score(self) -> Optional[float]:
        """Get the current best score."""
        return self._best_score

    @property
    def best_pipeline(self) -> Optional[str]:
        """Get the name of the best pipeline."""
        return self._best_pipeline


class SpinnerProgress:
    """Spinner for indeterminate progress indication.

    Use when the total number of items is unknown.

    Example:
        >>> with SpinnerProgress("Loading data") as spinner:
        ...     data = load_large_dataset()
        ...     spinner.update("Parsing...")
        ...     parsed = parse(data)
    """

    FRAMES_UNICODE = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    FRAMES_ASCII = ["|", "/", "-", "\\"]

    def __init__(
        self,
        description: str = "Processing",
        use_unicode: bool = True,
        use_colors: bool = True,
        disable: bool = False,
    ) -> None:
        """Initialize spinner.

        Args:
            description: Description text.
            use_unicode: Use Unicode spinner characters.
            use_colors: Use ANSI colors.
            disable: Disable spinner.
        """
        self.description = description
        self.use_unicode = use_unicode
        self.use_colors = use_colors
        self.disable = disable

        self._frame = 0
        self._start_time: Optional[float] = None
        self._running = False
        self._supports_ansi = _supports_ansi()

        if use_unicode:
            self._frames = self.FRAMES_UNICODE
        else:
            self._frames = self.FRAMES_ASCII

    def __enter__(self) -> "SpinnerProgress":
        """Enter context manager."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        self.stop(success=exc_type is None)

    def start(self) -> None:
        """Start the spinner."""
        if self.disable:
            return

        self._start_time = time.time()
        self._running = True

        if self._supports_ansi:
            sys.stdout.write(HIDE_CURSOR)

        self._display()

    def update(self, description: Optional[str] = None) -> None:
        """Update spinner with new description.

        Args:
            description: New description text.
        """
        if self.disable or not self._running:
            return

        if description is not None:
            self.description = description

        self._frame = (self._frame + 1) % len(self._frames)
        self._display()

    def stop(self, success: bool = True) -> None:
        """Stop the spinner.

        Args:
            success: Whether operation completed successfully.
        """
        if self.disable or not self._running:
            return

        self._running = False

        elapsed = time.time() - (self._start_time or time.time())
        elapsed_str = format_duration(elapsed)

        if self._supports_ansi:
            sys.stdout.write(CLEAR_LINE + MOVE_TO_START)
            sys.stdout.write(SHOW_CURSOR)

        # Final message
        if success:
            symbol = "[OK]"
            color = "\033[32m" if self.use_colors and self._supports_ansi else ""
        else:
            symbol = "[X]"
            color = "\033[31m" if self.use_colors and self._supports_ansi else ""

        reset = "\033[0m" if color else ""
        print(f"{color}{symbol}{reset} {self.description} ({elapsed_str})")

    def _display(self) -> None:
        """Display current spinner state."""
        if self.disable:
            return

        frame = self._frames[self._frame]

        elapsed = time.time() - (self._start_time or time.time())
        elapsed_str = format_duration(elapsed)

        if self._supports_ansi:
            line = f"{frame} {self.description} [{elapsed_str}]"
            sys.stdout.write(CLEAR_LINE + MOVE_TO_START + line)
            sys.stdout.flush()
        # Non-TTY: don't spam output, just show on completion


# Convenience functions
def progress_bar(
    total: int,
    description: str = "",
    **kwargs: Any,
) -> ProgressBar:
    """Create a progress bar.

    Args:
        total: Total number of items.
        description: Description text.
        **kwargs: Additional ProgressBar arguments.

    Returns:
        ProgressBar instance.
    """
    return ProgressBar(total=total, description=description, **kwargs)


def evaluation_progress(
    total_pipelines: int,
    metric_name: str = "score",
    higher_is_better: bool = False,
    **kwargs: Any,
) -> EvaluationProgress:
    """Create an evaluation progress tracker.

    Args:
        total_pipelines: Total pipelines to evaluate.
        metric_name: Name of optimization metric.
        higher_is_better: True if higher is better.
        **kwargs: Additional arguments.

    Returns:
        EvaluationProgress instance.
    """
    return EvaluationProgress(
        total_pipelines=total_pipelines,
        metric_name=metric_name,
        higher_is_better=higher_is_better,
        **kwargs,
    )


def spinner(description: str = "Processing", **kwargs: Any) -> SpinnerProgress:
    """Create a spinner for indeterminate progress.

    Args:
        description: Description text.
        **kwargs: Additional SpinnerProgress arguments.

    Returns:
        SpinnerProgress instance.
    """
    return SpinnerProgress(description=description, **kwargs)
