"""
Session context manager for nirs4all API.

A Session maintains shared resources across multiple nirs4all operations,
including a reusable PipelineRunner instance, consistent workspace paths,
and shared logging configuration.

Example:
    >>> with nirs4all.session(verbose=2, save_artifacts=True) as s:
    ...     r1 = nirs4all.run(pipeline1, data1, session=s)
    ...     r2 = nirs4all.run(pipeline2, data2, session=s)
    ...     # Both runs share workspace and configuration

Note:
    This is a Phase 0 stub. Full implementation in Phase 3.
"""

from contextlib import contextmanager
from typing import Optional, Generator, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from nirs4all.pipeline import PipelineRunner


class Session:
    """Execution session for resource reuse across multiple operations.

    A session maintains:
    - Shared PipelineRunner instance (created lazily on first use)
    - Consistent workspace path across runs
    - Shared logging configuration
    - Potential for cached/fitted transformers (future)

    Attributes:
        runner: The shared PipelineRunner instance (created on first use).
        workspace_path: Path to the shared workspace directory.

    Example:
        >>> with nirs4all.session(verbose=1) as s:
        ...     result1 = nirs4all.run(pipeline1, data1, session=s)
        ...     result2 = nirs4all.run(pipeline2, data2, session=s)

    Note:
        This is a Phase 0 stub. Full implementation in Phase 3.
    """

    def __init__(self, **runner_kwargs: Any) -> None:
        """Initialize a session with PipelineRunner configuration.

        Args:
            **runner_kwargs: Arguments to pass to PipelineRunner when created.
                Common options: verbose, save_artifacts, workspace_path,
                random_state, plots_visible, etc.
        """
        self._runner_kwargs = runner_kwargs
        self._runner: Optional["PipelineRunner"] = None

    @property
    def runner(self) -> "PipelineRunner":
        """Get or create the shared PipelineRunner instance.

        The runner is created lazily on first access.

        Returns:
            The shared PipelineRunner instance.
        """
        if self._runner is None:
            from nirs4all.pipeline import PipelineRunner
            self._runner = PipelineRunner(**self._runner_kwargs)
        return self._runner

    @property
    def workspace_path(self) -> Optional[Any]:
        """Get the workspace path from the runner.

        Returns:
            Path to the workspace directory, or None if runner not created.
        """
        if self._runner is not None:
            return getattr(self._runner, 'workspace_path', None)
        return self._runner_kwargs.get('workspace_path')

    def close(self) -> None:
        """Clean up session resources.

        Called automatically when exiting a context manager block.
        """
        # Future: cleanup cached transformers, close log files, etc.
        self._runner = None

    def __enter__(self) -> "Session":
        """Enter the session context."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the session context and clean up resources."""
        self.close()

    def __repr__(self) -> str:
        """Return string representation of session."""
        status = "active" if self._runner is not None else "idle"
        return f"Session({status}, kwargs={list(self._runner_kwargs.keys())})"


@contextmanager
def session(**kwargs: Any) -> Generator[Session, None, None]:
    """Create an execution session context manager.

    This is a convenience function that creates a Session and yields it
    within a context manager block.

    Args:
        **kwargs: Arguments passed to Session (and ultimately PipelineRunner).
            Common options:
            - verbose (int): Verbosity level (0-3). Default: 1
            - save_artifacts (bool): Save model artifacts. Default: True
            - workspace_path (str|Path): Workspace directory.
            - random_state (int): Random seed for reproducibility.

    Yields:
        Session: The active session for use within the block.

    Example:
        >>> with nirs4all.session(verbose=2, save_artifacts=True) as s:
        ...     r1 = nirs4all.run(pipeline1, data1, session=s)
        ...     r2 = nirs4all.run(pipeline2, data2, session=s)
        ...     print(f"PLS: {r1.best_score:.4f}, RF: {r2.best_score:.4f}")

    Note:
        This is a Phase 0 stub. Full implementation in Phase 3.
    """
    s = Session(**kwargs)
    try:
        yield s
    finally:
        s.close()
