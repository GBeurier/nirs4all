"""
Session context manager for nirs4all API.

A Session maintains shared resources across multiple nirs4all operations,
including a reusable PipelineRunner instance, consistent workspace paths,
and shared logging configuration.

Two usage patterns:

1. Context manager for shared runner (resource reuse):
    >>> with nirs4all.session(verbose=2, save_artifacts=True) as s:
    ...     r1 = nirs4all.run(pipeline1, data1, session=s)
    ...     r2 = nirs4all.run(pipeline2, data2, session=s)
    ...     # Both runs share workspace and configuration

2. Stateful session with pipeline (training workflow):
    >>> session = nirs4all.Session(pipeline=pipeline, name="MyModel")
    >>> result = session.run(dataset)
    >>> predictions = session.predict(new_data)
    >>> session.save("model.n4a")
"""

from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Union, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from nirs4all.pipeline import PipelineRunner
    from nirs4all.api.result import RunResult, PredictResult


class Session:
    """Execution session for resource reuse and stateful pipeline management.

    A session can be used in two modes:

    1. **Resource sharing mode** (no pipeline):
       Share a PipelineRunner across multiple nirs4all.run() calls.

    2. **Stateful pipeline mode** (with pipeline):
       Manage a single pipeline's lifecycle: train, predict, save, load.

    Attributes:
        name: Session/pipeline name for identification.
        pipeline: Pipeline definition (if in stateful mode).
        status: Current session status ('initialized', 'trained', 'error').
        is_trained: Whether the pipeline has been trained.
        runner: The shared PipelineRunner instance.
        workspace_path: Path to the workspace directory.

    Example (resource sharing):
        >>> with nirs4all.session(verbose=1) as s:
        ...     result1 = nirs4all.run(pipeline1, data1, session=s)
        ...     result2 = nirs4all.run(pipeline2, data2, session=s)

    Example (stateful pipeline):
        >>> session = nirs4all.Session(pipeline=pipeline, name="MyModel")
        >>> result = session.run("sample_data/regression")
        >>> predictions = session.predict(new_data)
        >>> session.save("exports/my_model.n4a")
    """

    def __init__(
        self,
        pipeline: Optional[List[Any]] = None,
        name: str = "",
        **runner_kwargs: Any
    ) -> None:
        """Initialize a session.

        Args:
            pipeline: Optional pipeline definition for stateful mode.
                If provided, enables run(), predict(), save() methods.
            name: Name for the session/pipeline.
            **runner_kwargs: Arguments to pass to PipelineRunner.
                Common options: verbose, save_artifacts, workspace_path,
                random_state, plots_visible, etc.
        """
        self._pipeline = pipeline
        self._name = name or "Session"
        self._runner_kwargs = runner_kwargs
        self._runner: Optional["PipelineRunner"] = None
        self._status = "initialized"
        self._last_result: Optional["RunResult"] = None
        self._run_history: List[Dict[str, Any]] = []
        self._bundle_path: Optional[Path] = None  # Set when loading from bundle

    @property
    def name(self) -> str:
        """Get session name."""
        return self._name

    @property
    def pipeline(self) -> Optional[List[Any]]:
        """Get pipeline definition."""
        return self._pipeline

    @property
    def status(self) -> str:
        """Get current session status.

        Returns:
            One of: 'initialized', 'trained', 'error'
        """
        return self._status

    @property
    def is_trained(self) -> bool:
        """Check if pipeline has been trained or loaded from a bundle."""
        if self._status != "trained":
            return False
        # Trained if we have a result from training or a loaded bundle
        return self._last_result is not None or self._bundle_path is not None

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
    def workspace_path(self) -> Optional[Path]:
        """Get the workspace path from the runner.

        Returns:
            Path to the workspace directory, or None if runner not created.
        """
        if self._runner is not None:
            return getattr(self._runner, 'workspace_path', None)
        return self._runner_kwargs.get('workspace_path')

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Get run history for this session."""
        return self._run_history

    def run(
        self,
        dataset: Union[str, Path, Any],
        *,
        plots_visible: bool = False,
        **kwargs: Any
    ) -> "RunResult":
        """Train the session's pipeline on a dataset.

        Args:
            dataset: Dataset to train on. Can be:
                - Path to data folder: "sample_data/regression"
                - Numpy arrays: (X, y)
                - Dict: {"X": X, "y": y}
            plots_visible: Whether to show plots during training.
            **kwargs: Additional arguments passed to runner.run().

        Returns:
            RunResult with predictions and metrics.

        Raises:
            ValueError: If no pipeline was provided to the session.
        """
        if self._pipeline is None:
            raise ValueError(
                "No pipeline defined for this session. "
                "Either pass pipeline= to Session() or use nirs4all.run() directly."
            )

        from nirs4all.api.result import RunResult
        from nirs4all.pipeline import PipelineConfigs
        from nirs4all.data import DatasetConfigs

        try:
            # Build configs
            pipeline_config = PipelineConfigs(self._pipeline, self._name)

            # Handle dataset
            if isinstance(dataset, DatasetConfigs):
                dataset_config = dataset
            elif isinstance(dataset, (str, Path)):
                dataset_config = DatasetConfigs(str(dataset))
            else:
                dataset_config = DatasetConfigs(dataset)

            # Update runner's plots_visible if specified
            if plots_visible is not None:
                self.runner.plots_visible = plots_visible
                if hasattr(self.runner, 'orchestrator') and self.runner.orchestrator:
                    self.runner.orchestrator.plots_visible = plots_visible

            # Filter kwargs for runner.run() - only pass valid parameters
            valid_run_params = {'pipeline_name', 'dataset_name', 'max_generation_count'}
            run_kwargs = {k: v for k, v in kwargs.items() if k in valid_run_params}

            # Run pipeline
            predictions, per_dataset = self.runner.run(
                pipeline_config,
                dataset_config,
                **run_kwargs
            )

            self._last_result = RunResult(
                predictions=predictions,
                per_dataset=per_dataset,
                _runner=self.runner
            )
            self._status = "trained"

            # Record in history
            self._run_history.append({
                'dataset': str(dataset),
                'best_score': self._last_result.best_score,
                'num_predictions': self._last_result.num_predictions
            })

            return self._last_result

        except Exception as e:
            self._status = "error"
            raise

    def predict(
        self,
        dataset: Union[str, Path, Any],
        **kwargs: Any
    ) -> "PredictResult":
        """Make predictions using the trained pipeline.

        Args:
            dataset: Data to predict on. Can be:
                - Path to data folder
                - Numpy array X
                - Dict with 'X' key
            **kwargs: Additional arguments for prediction.

        Returns:
            PredictResult with predictions.

        Raises:
            ValueError: If session has not been trained.
        """
        if not self.is_trained:
            raise ValueError(
                "Session must be trained before prediction. "
                "Call session.run(dataset) first."
            )

        from nirs4all.api.result import PredictResult
        import numpy as np

        # Handle dataset
        if isinstance(dataset, (str, Path)):
            from nirs4all.data import DatasetConfigs
            dataset_config = DatasetConfigs(str(dataset))
        else:
            dataset_config = dataset

        # Determine prediction source: bundle path or trained model
        if self._bundle_path is not None:
            # Use bundle file for loaded sessions
            prediction_obj = str(self._bundle_path)
            model_name = self._name
        elif self._last_result is not None:
            # Use best model from training
            best = self._last_result.best
            if not best:
                raise ValueError("No trained model available for prediction.")
            prediction_obj = best
            model_name = best.get('model_name', '')
        else:
            raise ValueError("No trained model available for prediction.")

        y_pred, predictions = self.runner.predict(
            prediction_obj=prediction_obj,
            dataset=dataset_config,
            **kwargs
        )

        return PredictResult(
            y_pred=np.asarray(y_pred).flatten() if y_pred is not None else np.array([]),
            metadata=predictions.__dict__ if hasattr(predictions, '__dict__') else {},
            model_name=model_name
        )

    def retrain(
        self,
        dataset: Union[str, Path, Any],
        mode: str = "full",
        **kwargs: Any
    ) -> "RunResult":
        """Retrain the pipeline on new data.

        Args:
            dataset: New dataset to train on.
            mode: Retrain mode ('full', 'transfer', 'finetune').
            **kwargs: Additional arguments for retraining.

        Returns:
            RunResult from retraining.

        Raises:
            ValueError: If session has not been trained.
        """
        if not self.is_trained:
            raise ValueError(
                "Session must be trained before retraining. "
                "Call session.run(dataset) first."
            )

        from nirs4all.api.result import RunResult
        from nirs4all.data import DatasetConfigs

        # Determine source: bundle path or trained model
        if self._bundle_path is not None:
            # Use bundle file for loaded sessions
            source = str(self._bundle_path)
        elif self._last_result is not None:
            best = self._last_result.best
            if not best:
                raise ValueError("No trained model available for retraining.")
            source = best
        else:
            raise ValueError("No trained model available for retraining.")

        # Handle dataset
        if isinstance(dataset, (str, Path)):
            dataset_config = DatasetConfigs(str(dataset))
        else:
            dataset_config = dataset

        predictions, per_dataset = self.runner.retrain(
            source=source,
            dataset=dataset_config,
            mode=mode,
            **kwargs
        )

        self._last_result = RunResult(
            predictions=predictions,
            per_dataset=per_dataset,
            _runner=self.runner
        )

        # Record in history
        self._run_history.append({
            'dataset': str(dataset),
            'mode': mode,
            'best_score': self._last_result.best_score,
            'num_predictions': self._last_result.num_predictions
        })

        return self._last_result

    def save(self, path: Union[str, Path]) -> Path:
        """Save the trained session to a bundle file.

        Args:
            path: Output path for the .n4a bundle file.

        Returns:
            Path to the saved bundle file.

        Raises:
            ValueError: If session has not been trained.
        """
        if not self.is_trained or self._last_result is None:
            raise ValueError(
                "Session must be trained before saving. "
                "Call session.run(dataset) first."
            )

        return self._last_result.export(path)

    def close(self) -> None:
        """Clean up session resources.

        Called automatically when exiting a context manager block.
        """
        self._runner = None

    def __enter__(self) -> "Session":
        """Enter the session context."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the session context and clean up resources."""
        self.close()

    def __repr__(self) -> str:
        """Return string representation of session."""
        if self._pipeline is not None:
            return f"Session(name='{self._name}', status='{self._status}', steps={len(self._pipeline)})"
        else:
            status = "active" if self._runner is not None else "idle"
            return f"Session({status}, kwargs={list(self._runner_kwargs.keys())})"


def load_session(path: Union[str, Path]) -> Session:
    """Load a session from a saved bundle file.

    Args:
        path: Path to .n4a bundle file.

    Returns:
        Session ready for prediction.

    Example:
        >>> session = nirs4all.load_session("exports/model.n4a")
        >>> predictions = session.predict(new_data)
    """
    from nirs4all.pipeline.bundle import BundleLoader
    from nirs4all.api.result import RunResult

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Bundle not found: {path}")

    # Load the bundle to get pipeline info
    loader = BundleLoader(path)
    pipeline_config = loader.pipeline_config

    # pipeline_config is a dict with 'steps', 'name', etc.
    steps = pipeline_config.get('steps', []) if isinstance(pipeline_config, dict) else []
    name = pipeline_config.get('name', path.stem) if isinstance(pipeline_config, dict) else path.stem

    # Create session with loaded pipeline
    session = Session(
        pipeline=steps,
        name=name
    )

    # Mark as trained by creating a mock result
    # The actual prediction will use the bundle directly
    session._status = "trained"
    session._bundle_path = path

    return session


@contextmanager
def session(
    pipeline: Optional[List[Any]] = None,
    name: str = "",
    **kwargs: Any
) -> Generator[Session, None, None]:
    """Create an execution session context manager.

    This is a convenience function that creates a Session and yields it
    within a context manager block.

    Args:
        pipeline: Optional pipeline definition for stateful mode.
        name: Name for the session/pipeline.
        **kwargs: Arguments passed to Session (and ultimately PipelineRunner).
            Common options:
            - verbose (int): Verbosity level (0-3). Default: 1
            - save_artifacts (bool): Save model artifacts. Default: True
            - workspace_path (str|Path): Workspace directory.
            - random_state (int): Random seed for reproducibility.

    Yields:
        Session: The active session for use within the block.

    Example (resource sharing):
        >>> with nirs4all.session(verbose=2, save_artifacts=True) as s:
        ...     r1 = nirs4all.run(pipeline1, data1, session=s)
        ...     r2 = nirs4all.run(pipeline2, data2, session=s)
        ...     print(f"PLS: {r1.best_score:.4f}, RF: {r2.best_score:.4f}")

    Example (stateful pipeline):
        >>> with nirs4all.session(pipeline=my_pipeline, name="Demo") as s:
        ...     result = s.run("sample_data/regression")
        ...     print(f"Best score: {result.best_score:.4f}")
    """
    s = Session(pipeline=pipeline, name=name, **kwargs)
    try:
        yield s
    finally:
        s.close()
