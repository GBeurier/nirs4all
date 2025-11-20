"""Executor builder for creating configured PipelineExecutor instances.

This module provides a builder pattern for constructing PipelineExecutor instances
with all necessary dependencies and configuration.
"""
from pathlib import Path
from typing import Any, Optional

from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.storage.artifacts.manager import ArtifactManager
from nirs4all.pipeline.execution.executor import PipelineExecutor
from nirs4all.pipeline.storage.io import SimulationSaver
from nirs4all.pipeline.storage.manifest_manager import ManifestManager
from nirs4all.pipeline.steps.parser import StepParser
from nirs4all.pipeline.steps.router import ControllerRouter
from nirs4all.pipeline.steps.step_runner import StepRunner


class ExecutorBuilder:
    """Builder for creating PipelineExecutor instances.

    Provides a fluent interface for configuring and building executors with
    all necessary dependencies. Encapsulates the complex setup logic previously
    duplicated between Orchestrator and other components.

    Example:
        >>> builder = ExecutorBuilder()
        >>> executor = (builder
        ...     .with_run_directory(run_dir)
        ...     .with_mode("train")
        ...     .with_verbose(1)
        ...     .build())
    """

    def __init__(self):
        """Initialize builder with default values."""
        # Required parameters
        self._run_directory: Optional[Path] = None
        self._dataset: Optional[SpectroDataset] = None

        # Optional parameters with defaults
        self._verbose: int = 0
        self._mode: str = "train"
        self._save_files: bool = True
        self._continue_on_error: bool = False
        self._show_spinner: bool = True
        self._binary_loader: Any = None
        self._runner: Any = None

        # Components (will be created if not provided)
        self._saver: Optional[SimulationSaver] = None
        self._manifest_manager: Optional[ManifestManager] = None
        self._artifact_manager: Optional[ArtifactManager] = None
        self._step_runner: Optional[StepRunner] = None

    def with_run_directory(self, run_directory: Path) -> 'ExecutorBuilder':
        """Set the run directory for this execution.

        Args:
            run_directory: Path to the run directory where outputs will be saved

        Returns:
            Self for method chaining
        """
        self._run_directory = run_directory
        return self

    def with_dataset(self, dataset: SpectroDataset) -> 'ExecutorBuilder':
        """Set the dataset for this execution.

        Args:
            dataset: Dataset to process

        Returns:
            Self for method chaining
        """
        self._dataset = dataset
        return self

    def with_verbose(self, verbose: int) -> 'ExecutorBuilder':
        """Set verbosity level.

        Args:
            verbose: Verbosity level (0=quiet, 1=info, 2=debug)

        Returns:
            Self for method chaining
        """
        self._verbose = verbose
        return self

    def with_mode(self, mode: str) -> 'ExecutorBuilder':
        """Set execution mode.

        Args:
            mode: Execution mode ('train', 'predict', 'explain')

        Returns:
            Self for method chaining
        """
        self._mode = mode
        return self

    def with_save_files(self, save_files: bool) -> 'ExecutorBuilder':
        """Set whether to save output files.

        Args:
            save_files: Whether to save files

        Returns:
            Self for method chaining
        """
        self._save_files = save_files
        return self

    def with_continue_on_error(self, continue_on_error: bool) -> 'ExecutorBuilder':
        """Set whether to continue execution on errors.

        Args:
            continue_on_error: Whether to continue on errors

        Returns:
            Self for method chaining
        """
        self._continue_on_error = continue_on_error
        return self

    def with_show_spinner(self, show_spinner: bool) -> 'ExecutorBuilder':
        """Set whether to show progress spinners.

        Args:
            show_spinner: Whether to show spinners

        Returns:
            Self for method chaining
        """
        self._show_spinner = show_spinner
        return self

    def with_binary_loader(self, binary_loader: Any) -> 'ExecutorBuilder':
        """Set binary loader for predict/explain modes.

        Args:
            binary_loader: BinaryLoader instance

        Returns:
            Self for method chaining
        """
        self._binary_loader = binary_loader
        return self

    def with_runner(self, runner: Any) -> 'ExecutorBuilder':
        """Set runner reference for backward compatibility.

        Args:
            runner: PipelineRunner instance

        Returns:
            Self for method chaining
        """
        self._runner = runner
        return self

    def with_saver(self, saver: SimulationSaver) -> 'ExecutorBuilder':
        """Set custom simulation saver.

        Args:
            saver: SimulationSaver instance

        Returns:
            Self for method chaining
        """
        self._saver = saver
        return self

    def with_manifest_manager(self, manifest_manager: ManifestManager) -> 'ExecutorBuilder':
        """Set custom manifest manager.

        Args:
            manifest_manager: ManifestManager instance

        Returns:
            Self for method chaining
        """
        self._manifest_manager = manifest_manager
        return self

    def with_artifact_manager(self, artifact_manager: ArtifactManager) -> 'ExecutorBuilder':
        """Set custom artifact manager.

        Args:
            artifact_manager: ArtifactManager instance

        Returns:
            Self for method chaining
        """
        self._artifact_manager = artifact_manager
        return self

    def with_step_runner(self, step_runner: StepRunner) -> 'ExecutorBuilder':
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
            ValueError: If run_directory is not set
        """
        if self._run_directory is None:
            raise ValueError("run_directory must be set before building executor")

        # Create saver if not provided
        if self._saver is None:
            self._saver = SimulationSaver(
                self._run_directory,
                save_files=self._save_files
            )

        # Create manifest manager if not provided
        if self._manifest_manager is None:
            self._manifest_manager = ManifestManager(self._run_directory)

        # Create artifact manager if not provided
        if self._artifact_manager is None:
            artifacts_dir = self._run_directory / "_binaries"
            self._artifact_manager = ArtifactManager(
                artifacts_dir,
                manifest_manager=self._manifest_manager
            )

        # Create step runner if not provided
        if self._step_runner is None:
            self._step_runner = StepRunner(
                parser=StepParser(),
                router=ControllerRouter(),
                verbose=self._verbose,
                mode=self._mode,
                show_spinner=self._show_spinner
            )

        # Update runner attributes if runner is provided (for compatibility)
        if self._runner:
            self._runner.saver = self._saver
            self._runner.manifest_manager = self._manifest_manager

        # Build and return executor
        return PipelineExecutor(
            step_runner=self._step_runner,
            artifact_manager=self._artifact_manager,
            manifest_manager=self._manifest_manager,
            verbose=self._verbose,
            mode=self._mode,
            continue_on_error=self._continue_on_error,
            saver=self._saver,
            binary_loader=self._binary_loader
        )
