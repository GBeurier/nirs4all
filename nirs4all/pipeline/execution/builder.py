"""Executor builder for creating configured PipelineExecutor instances.

This module provides a builder pattern for constructing PipelineExecutor instances
with all necessary dependencies and configuration.
"""
from pathlib import Path
from typing import Any, Optional

from nirs4all.pipeline.execution.executor import PipelineExecutor
from nirs4all.pipeline.steps.parser import StepParser
from nirs4all.pipeline.steps.router import ControllerRouter
from nirs4all.pipeline.steps.step_runner import StepRunner


class ExecutorBuilder:
    """Builder for creating PipelineExecutor instances.

    Provides a fluent interface for configuring and building executors with
    all necessary dependencies. Constructs a WorkspaceStore-backed executor.

    Example:
        >>> builder = ExecutorBuilder()
        >>> executor = (builder
        ...     .with_workspace(workspace_path)
        ...     .with_mode("train")
        ...     .with_verbose(1)
        ...     .build())
    """

    def __init__(self) -> None:
        """Initialize builder with default values."""
        # Required parameters
        self._workspace: Optional[Path] = None

        # Optional parameters with defaults
        self._verbose: int = 0
        self._mode: str = "train"
        self._save_artifacts: bool = True
        self._save_charts: bool = True
        self._continue_on_error: bool = False
        self._show_spinner: bool = True
        self._plots_visible: bool = False
        self._artifact_loader: Any = None
        self._artifact_registry: Any = None
        self._store: Any = None  # WorkspaceStore

        # Components (will be created if not provided)
        self._step_runner: Optional[StepRunner] = None

    def with_workspace(self, workspace: Path) -> "ExecutorBuilder":
        """Set workspace root path for artifact storage.

        Args:
            workspace: Workspace root path

        Returns:
            Self for method chaining
        """
        self._workspace = workspace
        return self

    def with_verbose(self, verbose: int) -> "ExecutorBuilder":
        """Set verbosity level.

        Args:
            verbose: Verbosity level (0=quiet, 1=info, 2=debug)

        Returns:
            Self for method chaining
        """
        self._verbose = verbose
        return self

    def with_mode(self, mode: str) -> "ExecutorBuilder":
        """Set execution mode.

        Args:
            mode: Execution mode ('train', 'predict', 'explain')

        Returns:
            Self for method chaining
        """
        self._mode = mode
        return self

    def with_save_artifacts(self, save_artifacts: bool) -> "ExecutorBuilder":
        """Set whether to save binary artifacts (models, transformers).

        Args:
            save_artifacts: Whether to save artifacts

        Returns:
            Self for method chaining
        """
        self._save_artifacts = save_artifacts
        return self

    def with_save_charts(self, save_charts: bool) -> "ExecutorBuilder":
        """Set whether to save charts and visual outputs.

        Args:
            save_charts: Whether to save charts

        Returns:
            Self for method chaining
        """
        self._save_charts = save_charts
        return self

    def with_continue_on_error(self, continue_on_error: bool) -> "ExecutorBuilder":
        """Set whether to continue execution on errors.

        Args:
            continue_on_error: Whether to continue on errors

        Returns:
            Self for method chaining
        """
        self._continue_on_error = continue_on_error
        return self

    def with_show_spinner(self, show_spinner: bool) -> "ExecutorBuilder":
        """Set whether to show progress spinners.

        Args:
            show_spinner: Whether to show spinners

        Returns:
            Self for method chaining
        """
        self._show_spinner = show_spinner
        return self

    def with_plots_visible(self, plots_visible: bool) -> "ExecutorBuilder":
        """Set whether to display plots.

        Args:
            plots_visible: Whether to display plots

        Returns:
            Self for method chaining
        """
        self._plots_visible = plots_visible
        return self

    def with_artifact_loader(self, artifact_loader: Any) -> "ExecutorBuilder":
        """Set artifact loader for predict/explain modes.

        Args:
            artifact_loader: ArtifactLoader instance

        Returns:
            Self for method chaining
        """
        self._artifact_loader = artifact_loader
        return self

    def with_artifact_registry(self, artifact_registry: Any) -> "ExecutorBuilder":
        """Set artifact registry for train mode.

        Args:
            artifact_registry: ArtifactRegistry instance

        Returns:
            Self for method chaining
        """
        self._artifact_registry = artifact_registry
        return self

    def with_store(self, store: Any) -> "ExecutorBuilder":
        """Set the WorkspaceStore for DuckDB-backed persistence.

        Args:
            store: WorkspaceStore instance

        Returns:
            Self for method chaining
        """
        self._store = store
        return self

    def with_step_runner(self, step_runner: StepRunner) -> "ExecutorBuilder":
        """Set custom step runner.

        Args:
            step_runner: StepRunner instance

        Returns:
            Self for method chaining
        """
        self._step_runner = step_runner
        return self

    def build(self) -> PipelineExecutor:
        """Build the configured PipelineExecutor instance.

        Creates all necessary components if not already provided, then constructs
        and returns a fully configured PipelineExecutor.

        Returns:
            Configured PipelineExecutor instance

        Raises:
            ValueError: If workspace is not set
        """
        if self._workspace is None:
            raise ValueError("workspace must be set before building executor")

        # Create step runner if not provided
        if self._step_runner is None:
            self._step_runner = StepRunner(
                parser=StepParser(),
                router=ControllerRouter(),
                verbose=self._verbose,
                mode=self._mode,
                show_spinner=self._show_spinner,
                plots_visible=self._plots_visible
            )

        # Build and return executor
        return PipelineExecutor(
            step_runner=self._step_runner,
            verbose=self._verbose,
            mode=self._mode,
            continue_on_error=self._continue_on_error,
            store=self._store,
            save_artifacts=self._save_artifacts,
            artifact_loader=self._artifact_loader,
            artifact_registry=self._artifact_registry
        )

    @property
    def workspace(self) -> Optional[Path]:
        """Get the workspace path."""
        return self._workspace

    @property
    def artifact_registry(self) -> Any:
        """Get the artifact registry."""
        return self._artifact_registry
