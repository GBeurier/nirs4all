"""Run context management for nirs4all logging.

This module provides context managers for tracking run state, phases,
branches, sources, and stacking operations in the logging system.
"""

from __future__ import annotations

import logging
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generator, Optional

from .events import Phase, Status


@dataclass
class BranchContext:
    """Context for a branch in the pipeline.

    Attributes:
        name: Branch name.
        path: Full branch path (for nested branches).
        index: Branch index (0-based).
        total: Total number of branches at this level.
        parent: Parent branch name (for nested branches).
        depth: Nesting depth (0 for top-level).
    """

    name: str
    path: list[str] = field(default_factory=list)
    index: Optional[int] = None
    total: Optional[int] = None
    parent: Optional[str] = None
    depth: int = 0


@dataclass
class SourceContext:
    """Context for a source in multi-source pipelines.

    Attributes:
        name: Source name.
        index: Source index (0-based).
        total: Total number of sources.
    """

    name: str
    index: Optional[int] = None
    total: Optional[int] = None


@dataclass
class StackContext:
    """Context for stacking operations.

    Attributes:
        n_branches: Number of branches being stacked.
        meta_model: Meta-model name/description.
        branch_sources: List of branch names being stacked.
    """

    n_branches: int
    meta_model: Optional[str] = None
    branch_sources: list[str] = field(default_factory=list)


@dataclass
class RunState:
    """State for a single run.

    Attributes:
        run_id: Unique run identifier.
        run_name: Human-readable run name.
        project: Project name (optional).
        start_time: Run start timestamp.
        current_phase: Current workflow phase.
        branch_stack: Stack of branch contexts (for nesting).
        source_context: Current source context (if any).
        stack_context: Current stacking context (if any).
        extra: Additional run-level metadata.
    """

    run_id: str
    run_name: Optional[str] = None
    project: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    current_phase: Optional[Phase] = None
    branch_stack: list[BranchContext] = field(default_factory=list)
    source_context: Optional[SourceContext] = None
    stack_context: Optional[StackContext] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def current_branch(self) -> Optional[BranchContext]:
        """Get the current (innermost) branch context."""
        return self.branch_stack[-1] if self.branch_stack else None

    @property
    def branch_depth(self) -> int:
        """Get the current branch nesting depth."""
        return len(self.branch_stack)

    @property
    def branch_path(self) -> list[str]:
        """Get the full branch path."""
        return [b.name for b in self.branch_stack]


class _ContextStorage(threading.local):
    """Thread-local storage for run context."""

    def __init__(self) -> None:
        super().__init__()
        self.run_state: Optional[RunState] = None


# Global context storage
_context = _ContextStorage()


def get_current_state() -> Optional[RunState]:
    """Get the current run state.

    Returns:
        Current RunState or None if not in a run context.
    """
    return _context.run_state


def get_run_id() -> Optional[str]:
    """Get the current run ID.

    Returns:
        Current run ID or None if not in a run context.
    """
    state = get_current_state()
    return state.run_id if state else None


def _generate_run_id() -> str:
    """Generate a unique run ID.

    Returns:
        Run ID in format "R-YYYYMMDD-HHMMSS-XXXX".
    """
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:4]
    return f"R-{timestamp}-{suffix}"


class LogContext:
    """Context manager for run-level logging context.

    Provides a context manager that sets up run state and can be used
    to track the current run, phase, branches, sources, etc.

    Example:
        >>> with LogContext(run_id="my-experiment", project="protein"):
        ...     logger.info("Starting analysis")
        ...     with LogContext.branch("snv", index=0, total=4):
        ...         logger.info("Processing SNV branch")
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        run_name: Optional[str] = None,
        project: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Initialize log context.

        Args:
            run_id: Unique run identifier (auto-generated if not provided).
            run_name: Human-readable run name.
            project: Project name.
            **extra: Additional run-level metadata.
        """
        self.run_id = run_id or _generate_run_id()
        self.run_name = run_name or self.run_id
        self.project = project
        self.extra = extra
        self._previous_state: Optional[RunState] = None

    def __enter__(self) -> "LogContext":
        """Enter the context, setting up run state."""
        self._previous_state = _context.run_state
        _context.run_state = RunState(
            run_id=self.run_id,
            run_name=self.run_name,
            project=self.project,
            start_time=datetime.now(),
            extra=self.extra,
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context, restoring previous state."""
        _context.run_state = self._previous_state

    @staticmethod
    @contextmanager
    def phase(phase: Phase) -> Generator[None, None, None]:
        """Context manager for tracking the current phase.

        Args:
            phase: The phase being entered.

        Yields:
            None
        """
        state = get_current_state()
        if state is None:
            yield
            return

        previous_phase = state.current_phase
        state.current_phase = phase
        try:
            yield
        finally:
            state.current_phase = previous_phase

    @staticmethod
    @contextmanager
    def branch(
        name: str,
        index: Optional[int] = None,
        total: Optional[int] = None,
        parent: Optional[str] = None,
    ) -> Generator[BranchContext, None, None]:
        """Context manager for tracking branch execution.

        Args:
            name: Branch name.
            index: Branch index (0-based).
            total: Total number of branches at this level.
            parent: Parent branch name (for nested branches).

        Yields:
            BranchContext for the current branch.
        """
        state = get_current_state()
        if state is None:
            # Create a temporary context for the branch
            ctx = BranchContext(
                name=name,
                path=[name],
                index=index,
                total=total,
                parent=parent,
                depth=0,
            )
            yield ctx
            return

        # Build branch path from parent branches
        path = state.branch_path + [name]
        depth = state.branch_depth

        branch_ctx = BranchContext(
            name=name,
            path=path,
            index=index,
            total=total,
            parent=parent,
            depth=depth,
        )

        state.branch_stack.append(branch_ctx)
        try:
            yield branch_ctx
        finally:
            state.branch_stack.pop()

    @staticmethod
    @contextmanager
    def source(
        name: str,
        index: Optional[int] = None,
        total: Optional[int] = None,
    ) -> Generator[SourceContext, None, None]:
        """Context manager for tracking source processing in multi-source pipelines.

        Args:
            name: Source name.
            index: Source index (0-based).
            total: Total number of sources.

        Yields:
            SourceContext for the current source.
        """
        state = get_current_state()
        source_ctx = SourceContext(name=name, index=index, total=total)

        if state is None:
            yield source_ctx
            return

        previous_source = state.source_context
        state.source_context = source_ctx
        try:
            yield source_ctx
        finally:
            state.source_context = previous_source

    @staticmethod
    @contextmanager
    def stack(
        n_branches: int,
        meta_model: Optional[str] = None,
        branch_sources: Optional[list[str]] = None,
    ) -> Generator[StackContext, None, None]:
        """Context manager for tracking stacking operations.

        Args:
            n_branches: Number of branches being stacked.
            meta_model: Meta-model name/description.
            branch_sources: List of branch names being stacked.

        Yields:
            StackContext for the stacking operation.
        """
        state = get_current_state()
        stack_ctx = StackContext(
            n_branches=n_branches,
            meta_model=meta_model,
            branch_sources=branch_sources or [],
        )

        if state is None:
            yield stack_ctx
            return

        previous_stack = state.stack_context
        state.stack_context = stack_ctx
        try:
            yield stack_ctx
        finally:
            state.stack_context = previous_stack


def inject_context(record: logging.LogRecord) -> logging.LogRecord:
    """Inject current context into a log record.

    This is called by the logger to add context fields to each log record.

    Args:
        record: Log record to inject context into.

    Returns:
        Modified log record with context fields.
    """
    state = get_current_state()

    if state is None:
        return record

    # Add run context
    record.run_id = state.run_id  # type: ignore
    record.run_name = state.run_name  # type: ignore

    if state.current_phase:
        record.phase = state.current_phase  # type: ignore

    # Add branch context
    if state.current_branch:
        branch = state.current_branch
        record.branch_name = branch.name  # type: ignore
        record.branch_path = branch.path  # type: ignore
        record.branch_index = branch.index  # type: ignore
        record.total_branches = branch.total  # type: ignore
        record.branch_depth = branch.depth  # type: ignore

    # Add source context
    if state.source_context:
        source = state.source_context
        record.source_name = source.name  # type: ignore
        record.source_index = source.index  # type: ignore
        record.total_sources = source.total  # type: ignore

    # Add stack context
    if state.stack_context:
        stack = state.stack_context
        record.stack_n_branches = stack.n_branches  # type: ignore
        record.stack_meta_model = stack.meta_model  # type: ignore
        record.stack_branch_sources = stack.branch_sources  # type: ignore

    return record
