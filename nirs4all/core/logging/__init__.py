"""Unified logging system for nirs4all.

This module provides a structured, configurable logging infrastructure
that replaces all print() statements with proper logging. It supports:

- Human-readable console output optimized for scientists
- Machine-parseable file logging for automation
- Context tracking for runs, phases, branches, and sources
- ASCII-safe output for HPC/cluster environments
- Progress throttling to avoid terminal flooding
- TTY-aware progress bars with multi-level support

Usage:
    >>> from nirs4all.core.logging import get_logger, configure_logging, LogContext
    >>>
    >>> # Configure at application startup
    >>> configure_logging(verbose=1, use_unicode=True)
    >>>
    >>> # Get logger in each module
    >>> logger = get_logger(__name__)
    >>>
    >>> # Use context for run tracking
    >>> with LogContext(run_id="my-experiment"):
    ...     logger.info("Starting analysis")
    ...     with LogContext.branch("snv", index=0, total=4):
    ...         logger.success("SNV preprocessing complete")

Progress Bars:
    >>> from nirs4all.core.logging import ProgressBar, EvaluationProgress
    >>>
    >>> # Simple progress bar
    >>> with ProgressBar(total=100, description="Processing") as pbar:
    ...     for i in range(100):
    ...         pbar.update(1)
    >>>
    >>> # ML-specific evaluation progress
    >>> with EvaluationProgress(total_pipelines=42, metric_name="RMSE") as progress:
    ...     for pipeline in pipelines:
    ...         score = evaluate(pipeline)
    ...         progress.update(score=score)

See Also:
    - :mod:`nirs4all.core.logging.config` for configuration details
    - :mod:`nirs4all.core.logging.context` for context management
    - :mod:`nirs4all.core.logging.formatters` for output formatting
    - :mod:`nirs4all.core.logging.progress` for progress bars
"""

from .config import (
    TRACE,
    Nirs4allLogger,
    configure_logging,
    get_config,
    get_logger,
    is_configured,
    reset_logging,
)
from .context import (
    BranchContext,
    LogContext,
    RunState,
    SourceContext,
    StackContext,
    get_current_state,
    get_run_id,
)
from .events import (
    EventType,
    LogEvent,
    Phase,
    Status,
)
from .formatters import (
    ConsoleFormatter,
    FileFormatter,
    JsonFormatter,
    Symbols,
    format_duration,
    format_number,
    format_run_footer,
    format_run_header,
    format_table,
    get_symbols,
)
from .handlers import (
    BufferedHandler,
    NullHandler,
    RotatingRunFileHandler,
    ThrottledHandler,
)
from .progress import (
    EvaluationProgress,
    MultiLevelProgress,
    ProgressBar,
    ProgressConfig,
    SpinnerProgress,
    configure_progress,
    evaluation_progress,
    progress_bar,
    spinner,
)

__all__ = [
    # Main API
    "get_logger",
    "configure_logging",
    "LogContext",
    # Configuration
    "get_config",
    "is_configured",
    "reset_logging",
    "TRACE",
    "Nirs4allLogger",
    # Context
    "get_current_state",
    "get_run_id",
    "RunState",
    "BranchContext",
    "SourceContext",
    "StackContext",
    # Events
    "Phase",
    "EventType",
    "Status",
    "LogEvent",
    # Formatters
    "Symbols",
    "get_symbols",
    "ConsoleFormatter",
    "FileFormatter",
    "JsonFormatter",
    "format_duration",
    "format_number",
    "format_table",
    "format_run_header",
    "format_run_footer",
    # Handlers
    "ThrottledHandler",
    "RotatingRunFileHandler",
    "BufferedHandler",
    "NullHandler",
    # Progress bars
    "ProgressBar",
    "ProgressConfig",
    "EvaluationProgress",
    "MultiLevelProgress",
    "SpinnerProgress",
    "configure_progress",
    "progress_bar",
    "evaluation_progress",
    "spinner",
]
