"""Pipeline predictor - Handles prediction mode execution.

This module provides the Predictor class for running predictions using trained
pipelines on new datasets.

Supports two prediction paths:
    - Bundle path: ``.n4a`` bundles loaded via BundleLoader (self-contained).
    - Resolver path: Prediction dicts / folders resolved via PredictionResolver,
      with optional minimal pipeline extraction from execution traces.
"""
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from nirs4all.core.logging import Nirs4allLogger, get_logger

if TYPE_CHECKING:
    from nirs4all.pipeline.resolver import ResolvedPrediction
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.pipeline.trace import ExecutionTrace, MinimalPipeline
from nirs4all.data.config import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.config.context import (
    DataSelector,
    ExecutionContext,
    LoaderArtifactProvider,
    PipelineState,
    RuntimeContext,
    StepMetadata,
)
from nirs4all.pipeline.execution.builder import ExecutorBuilder
from nirs4all.pipeline.storage.artifacts.artifact_loader import ArtifactLoader

logger: Nirs4allLogger = get_logger(__name__)

@dataclass(frozen=True)
class _PredictionStrategy:
    """Plan for executing one prediction replay strategy."""

    name: str
    mode: str  # "minimal" or "full"
    steps: list[Any]
    minimal_pipeline: Any | None
    artifact_provider_factory: Callable[[], Any]

class Predictor:
    """Handles prediction using trained pipelines.

    This class manages the prediction workflow: loading saved models,
    replaying pipeline configurations, and generating predictions on new data.

    When ``use_minimal_pipeline=True`` (default), the predictor will:
    1. Check if an execution trace is available for the prediction.
    2. Extract the minimal pipeline (only required steps) from the trace.
    3. Execute only those steps, significantly reducing prediction time.

    Attributes:
        runner: Parent PipelineRunner instance.
        pipeline_uid: Unique identifier for the pipeline.
        artifact_loader: Loader for trained model artifacts.
        config_path: Path to the pipeline configuration.
        target_model: Metadata for the target model.
        use_minimal_pipeline: Whether to use minimal pipeline execution.
    """

    def __init__(self, runner: "PipelineRunner", use_minimal_pipeline: bool = True):
        """Initialize predictor.

        Args:
            runner: Parent PipelineRunner instance.
            use_minimal_pipeline: If True, use minimal pipeline execution when
                execution traces are available.
        """
        self.runner = runner
        self.pipeline_uid: str | None = None
        self.artifact_loader: ArtifactLoader | None = None
        self.config_path: str | None = None
        self.target_model: dict[str, Any] | None = None
        self.use_minimal_pipeline = use_minimal_pipeline
        self._execution_trace: ExecutionTrace | None = None
        self._minimal_pipeline: MinimalPipeline | None = None
        self._resolved: ResolvedPrediction | None = None

    def predict(
        self,
        prediction_obj: dict[str, Any] | str,
        dataset: DatasetConfigs | SpectroDataset | np.ndarray | tuple[np.ndarray, ...] | dict | list[dict] | str | list[str],
        dataset_name: str = "prediction_dataset",
        all_predictions: bool = False,
        verbose: int = 0,
    ) -> tuple[np.ndarray, Predictions] | tuple[dict[str, Any], Predictions]:
        """Run prediction using a saved model on new dataset.

        Args:
            prediction_obj: Model identifier (dict with config_path or prediction ID).
            dataset: New dataset to predict on.
            dataset_name: Name for the dataset.
            all_predictions: If True, return all predictions; if False, return single best.
            verbose: Verbosity level.

        Returns:
            If ``all_predictions=False``: ``(y_pred, predictions)``
            If ``all_predictions=True``: ``(predictions_dict, predictions)``
        """
        logger.starting("Starting Nirs4all prediction(s)")

        # Handle bundle files (.n4a) directly via BundleLoader
        if isinstance(prediction_obj, str) and (prediction_obj.endswith(".n4a") or prediction_obj.endswith(".n4a.py")):
            return self._predict_from_bundle(prediction_obj, dataset, dataset_name, all_predictions, verbose)

        # Normalize dataset
        dataset_config = self.runner.orchestrator._normalize_dataset(dataset, dataset_name)

        # Setup prediction mode
        self.runner.mode = "predict"
        self.runner.verbose = verbose

        # Resolve prediction source via PredictionResolver
        self._resolve_prediction(prediction_obj, verbose)

        # Try minimal pipeline execution if trace is available
        if self.use_minimal_pipeline and self._execution_trace is not None:
            return self._predict_with_minimal_pipeline(dataset_config, all_predictions, verbose)

        # Fallback: Full pipeline execution
        return self._predict_full_pipeline(dataset_config, all_predictions, verbose)

    # -----------------------------------------------------------------
    # Resolution
    # -----------------------------------------------------------------

    def _resolve_prediction(self, prediction_obj: dict[str, Any] | str, verbose: int) -> None:
        """Resolve a prediction object to executable components.

        Extracts ``config_path``, ``target_model``, ``artifact_loader`` and
        optionally an ``_execution_trace`` from the prediction source.

        Args:
            prediction_obj: Prediction dict or string identifier.
            verbose: Verbosity level.
        """
        from nirs4all.pipeline.resolver import PredictionResolver

        resolver = PredictionResolver(
            workspace_path=self.runner.workspace_path,
            runs_dir=self.runner.workspace_path,
            store=self.runner.store,
        )
        resolved = resolver.resolve(prediction_obj, verbose=verbose)
        self._resolved = resolved

        # Extract target model metadata
        target_model = {k: v for k, v in prediction_obj.items() if k not in ("y_pred", "y_true", "X")} if isinstance(prediction_obj, dict) else dict(resolved.target_model) if resolved.target_model else {}

        if not target_model or "model_name" not in target_model:
            raise ValueError(
                "No model information found in prediction. "
                "Please provide a prediction dict with 'model_name' and 'pipeline_uid' keys."
            )

        self.target_model = target_model
        self.runner.target_model = target_model
        self.pipeline_uid = resolved.pipeline_uid
        self.config_path = resolved.pipeline_uid
        self.artifact_loader = None

        # Create artifact loader from resolved artifact provider
        if resolved.artifact_provider and hasattr(resolved.artifact_provider, "artifact_loader"):
            self.artifact_loader = resolved.artifact_provider.artifact_loader

        # Cache execution trace
        self._execution_trace = resolved.trace

    # -----------------------------------------------------------------
    # Minimal pipeline path
    # -----------------------------------------------------------------

    def _predict_with_minimal_pipeline(
        self,
        dataset_config: DatasetConfigs,
        all_predictions: bool,
        verbose: int,
    ) -> tuple[np.ndarray, Predictions] | tuple[dict[str, Any], Predictions]:
        """Execute prediction using minimal pipeline extracted from trace.

        Args:
            dataset_config: Dataset configuration.
            all_predictions: Whether to return all predictions.
            verbose: Verbosity level.

        Returns:
            Same as ``predict()`` method.
        """
        logger.info("Using minimal pipeline execution")
        strategy = self._make_minimal_strategy()
        return self._run_prediction_strategy(
            strategy=strategy,
            dataset_config=dataset_config,
            all_predictions=all_predictions,
            verbose=verbose,
        )

    # -----------------------------------------------------------------
    # Full pipeline path
    # -----------------------------------------------------------------

    def _predict_full_pipeline(
        self,
        dataset_config: DatasetConfigs,
        all_predictions: bool,
        verbose: int,
    ) -> tuple[np.ndarray, Predictions] | tuple[dict[str, Any], Predictions]:
        """Execute prediction using full pipeline replay.

        Args:
            dataset_config: Dataset configuration.
            all_predictions: Whether to return all predictions.
            verbose: Verbosity level.

        Returns:
            Same as ``predict()`` method.
        """
        strategy = self._make_full_strategy()
        return self._run_prediction_strategy(
            strategy=strategy,
            dataset_config=dataset_config,
            all_predictions=all_predictions,
            verbose=verbose,
        )

    def _make_minimal_strategy(self) -> _PredictionStrategy:
        """Build the minimal-pipeline replay strategy plan."""
        from nirs4all.pipeline.minimal_predictor import MinimalArtifactProvider
        from nirs4all.pipeline.trace import TraceBasedExtractor

        assert self._resolved is not None
        assert self._execution_trace is not None

        extractor = TraceBasedExtractor()
        full_steps = list(self._resolved.minimal_pipeline)

        target_branch_name = self.target_model.get("branch_name") if self.target_model else None
        target_branch_id = self.target_model.get("branch_id") if self.target_model else None
        target_branch_path = self.target_model.get("branch_path") if self.target_model else None

        if target_branch_name:
            minimal_pipeline = extractor.extract_for_branch_name(
                trace=self._execution_trace,
                branch_name=target_branch_name,
                full_pipeline=full_steps,
            )
        elif target_branch_path:
            minimal_pipeline = extractor.extract_for_branch(
                trace=self._execution_trace,
                branch_path=target_branch_path,
                full_pipeline=full_steps,
            )
        elif target_branch_id is not None:
            minimal_pipeline = extractor.extract_for_branch(
                trace=self._execution_trace,
                branch_path=[target_branch_id],
                full_pipeline=full_steps,
            )
        else:
            minimal_pipeline = extractor.extract(
                trace=self._execution_trace,
                full_pipeline=full_steps,
                up_to_model=True,
            )

        self._minimal_pipeline = minimal_pipeline
        logger.debug(
            f"Minimal pipeline: {minimal_pipeline.get_step_count()} steps "
            f"(from {len(full_steps)} total)"
        )

        target_sub_index = None
        target_model_name = None
        if self.target_model:
            model_artifact_id = self.target_model.get("model_artifact_id")
            if model_artifact_id:
                target_sub_index = self._get_substep_from_artifact(model_artifact_id)
            else:
                target_model_name = self.target_model.get("model_name")

        def _artifact_provider_factory() -> Any:
            return MinimalArtifactProvider(
                minimal_pipeline=minimal_pipeline,
                artifact_loader=self.artifact_loader,
                target_sub_index=target_sub_index,
                target_model_name=target_model_name,
            )

        return _PredictionStrategy(
            name="minimal",
            mode="minimal",
            steps=[step.step_config for step in minimal_pipeline.steps],
            minimal_pipeline=minimal_pipeline,
            artifact_provider_factory=_artifact_provider_factory,
        )

    def _make_full_strategy(self) -> _PredictionStrategy:
        """Build the full-pipeline replay strategy plan."""
        assert self._resolved is not None
        resolved = self._resolved

        def _artifact_provider_factory() -> Any:
            artifact_provider = resolved.artifact_provider
            if artifact_provider is None and self.artifact_loader:
                artifact_provider = LoaderArtifactProvider(loader=self.artifact_loader)
            return artifact_provider

        return _PredictionStrategy(
            name="full",
            mode="full",
            steps=list(resolved.minimal_pipeline),
            minimal_pipeline=None,
            artifact_provider_factory=_artifact_provider_factory,
        )

    def _run_prediction_strategy(
        self,
        strategy: _PredictionStrategy,
        dataset_config: DatasetConfigs,
        all_predictions: bool,
        verbose: int,
    ) -> tuple[np.ndarray, Predictions] | tuple[dict[str, Any], Predictions]:
        """Execute prediction replay using a strategy plan."""
        run_predictions = Predictions()

        for config, name in dataset_config.configs:
            dataset_obj = dataset_config.get_dataset(config, name)
            config_predictions = Predictions()

            selector = self._build_prediction_selector(dataset_obj)
            context = self._build_prediction_context(selector)
            executor = self._build_prediction_executor(verbose)

            runtime_context = self._build_prediction_runtime_context(
                executor=executor,
                artifact_provider=strategy.artifact_provider_factory(),
            )

            if strategy.mode == "minimal":
                executor.execute_minimal(
                    steps=strategy.steps,
                    minimal_pipeline=strategy.minimal_pipeline,
                    dataset=dataset_obj,
                    context=context,
                    runtime_context=runtime_context,
                    prediction_store=config_predictions,
                )
            else:
                executor.execute(
                    strategy.steps,
                    "prediction",
                    dataset_obj,
                    context,
                    runtime_context,
                    config_predictions,
                )

            run_predictions.merge_predictions(config_predictions)

        return self._process_prediction_results(run_predictions, all_predictions)

    def _build_prediction_selector(self, dataset_obj: SpectroDataset) -> DataSelector:
        """Build prediction selector with optional branch filters."""
        selector = DataSelector(
            partition="all",
            processing=[["raw"]] * dataset_obj.features_sources(),
            layout="2d",
            concat_source=True,
        )

        target_branch_id = self.target_model.get("branch_id") if self.target_model else None
        target_branch_name = self.target_model.get("branch_name") if self.target_model else None
        target_branch_path = self.target_model.get("branch_path") if self.target_model else None

        if target_branch_id is not None or target_branch_name:
            bp = target_branch_path or ([target_branch_id] if target_branch_id is not None else None)
            selector = selector.with_branch(
                branch_id=target_branch_id,
                branch_name=target_branch_name,
                branch_path=bp,
            )

        return selector

    @staticmethod
    def _build_prediction_context(selector: DataSelector) -> ExecutionContext:
        """Build execution context for prediction replay."""
        return ExecutionContext(
            selector=selector,
            state=PipelineState(y_processing="numeric", step_number=0, mode="predict"),
            metadata=StepMetadata(),
        )

    def _build_prediction_executor(self, verbose: int) -> Any:
        """Build configured executor used by both replay strategies."""
        return (
            ExecutorBuilder()
            .with_workspace(self.runner.workspace_path)
            .with_verbose(verbose)
            .with_mode("predict")
            .with_save_artifacts(False)
            .with_save_charts(self.runner.save_charts)
            .with_continue_on_error(self.runner.continue_on_error)
            .with_show_spinner(self.runner.show_spinner)
            .with_plots_visible(self.runner.plots_visible)
            .with_artifact_loader(self.artifact_loader)
            .build()
        )

    def _build_prediction_runtime_context(self, executor: Any, artifact_provider: Any) -> RuntimeContext:
        """Build runtime context shared by full and minimal prediction replay."""
        return RuntimeContext(
            artifact_loader=self.artifact_loader,
            artifact_provider=artifact_provider,
            step_runner=executor.step_runner,
            target_model=self.target_model,
            explainer=self.runner.explainer,
        )

    # -----------------------------------------------------------------
    # Result processing
    # -----------------------------------------------------------------

    def _process_prediction_results(
        self,
        run_predictions: Predictions,
        all_predictions: bool,
    ) -> tuple[np.ndarray, Predictions] | tuple[dict[str, Any], Predictions]:
        """Process prediction results and return in requested format.

        Args:
            run_predictions: Predictions object with results.
            all_predictions: Whether to return all predictions.

        Returns:
            Formatted prediction results.
        """
        if all_predictions:
            res: dict[str, Any] = {}
            for pred in run_predictions.to_dicts():
                ds = pred["dataset_name"]
                if ds not in res:
                    res[ds] = {}
                res[ds][pred["id"]] = pred["y_pred"]
            return res, run_predictions

        # Get single prediction matching target model
        tm = self.target_model or {}
        target_fold_id = tm.get("fold_id", None)

        is_aggregated_fold = target_fold_id in ("avg", "w_avg")
        is_refit_fold = target_fold_id == "final"

        filter_kwargs: dict[str, Any] = {
            "model_name": tm.get("model_name"),
            "step_idx": tm.get("step_idx"),
        }

        if not is_aggregated_fold and not is_refit_fold:
            filter_kwargs["fold_id"] = target_fold_id

        target_branch_id = tm.get("branch_id")
        if target_branch_id is not None:
            filter_kwargs["branch_id"] = target_branch_id

        candidates = run_predictions.filter_predictions(**filter_kwargs)
        non_empty = [p for p in candidates if len(p["y_pred"]) > 0]

        single_pred = self._aggregate_fold_predictions(non_empty, str(target_fold_id)) if is_aggregated_fold and len(non_empty) > 1 else non_empty[0] if non_empty else (candidates[0] if candidates else None)

        if single_pred is None:
            raise ValueError("No matching prediction found for the specified model criteria.")

        logger.success(f"Predicted with: {single_pred['model_name']} [{single_pred['id']}]")

        return single_pred["y_pred"], run_predictions

    def _aggregate_fold_predictions(
        self,
        non_empty: list[dict[str, Any]],
        target_fold_id: str,
    ) -> dict[str, Any] | None:
        """Aggregate predictions across folds.

        Args:
            non_empty: Non-empty prediction entries.
            target_fold_id: Target fold id ('avg' or 'w_avg').

        Returns:
            Aggregated prediction dict, or first candidate.
        """
        tm = self.target_model or {}
        fold_weights = tm.get("weights")

        fold_preds: dict[Any, np.ndarray] = {}
        for p in non_empty:
            fid = p.get("fold_id")
            if fid not in ("avg", "w_avg") and fid is not None and fid not in fold_preds:
                fold_preds[fid] = p["y_pred"]

        if not fold_preds:
            return non_empty[0] if non_empty else None

        sorted_folds = sorted(fold_preds.keys())
        y_arrays = [fold_preds[fid] for fid in sorted_folds]

        has_valid_weights = (
            target_fold_id == "w_avg"
            and fold_weights is not None
            and len(fold_weights) >= len(sorted_folds)
        )

        if has_valid_weights and fold_weights is not None:
            wl = [fold_weights.get(fid, 1.0) for fid in sorted_folds] if isinstance(fold_weights, dict) else [fold_weights[i] for i in range(len(sorted_folds))]
            total_w = sum(wl)
            y_pred = sum(w * np.array(y) for w, y in zip(wl, y_arrays, strict=False)) / total_w
        else:
            y_pred = np.mean(y_arrays, axis=0)

        single_pred = non_empty[0].copy()
        single_pred["y_pred"] = y_pred
        single_pred["fold_id"] = target_fold_id
        return single_pred

    # -----------------------------------------------------------------
    # Bundle path
    # -----------------------------------------------------------------

    def _predict_from_bundle(
        self,
        bundle_path: str,
        dataset: DatasetConfigs | SpectroDataset | np.ndarray | tuple[np.ndarray, ...] | dict | list[dict] | str | list[str],
        dataset_name: str,
        all_predictions: bool,
        verbose: int,
    ) -> tuple[np.ndarray, Predictions] | tuple[dict[str, Any], Predictions]:
        """Predict from an exported bundle file.

        Args:
            bundle_path: Path to ``.n4a`` bundle file.
            dataset: Dataset to predict on.
            dataset_name: Name for the dataset.
            all_predictions: Whether to return all predictions.
            verbose: Verbosity level.

        Returns:
            Predictions.
        """
        from nirs4all.pipeline.bundle import BundleLoader

        logger.debug(f"Loading bundle: {bundle_path}")

        loader = BundleLoader(bundle_path)

        dataset_config = self.runner.orchestrator._normalize_dataset(dataset, dataset_name)
        X_data: np.ndarray | None = None

        for data_config, name in dataset_config.configs:
            dataset_obj = dataset_config.get_dataset(data_config, name)
            raw = dataset_obj.x({})
            X_data = np.concatenate(raw, axis=1) if isinstance(raw, list) else raw
            break

        if X_data is None:
            raise ValueError("No data found in dataset for prediction")

        y_pred = loader.predict(X_data)

        model_name = "bundle_model"
        if loader.metadata:
            model_name = loader.metadata.original_manifest.get("name", "bundle_model")

        run_predictions = Predictions()
        run_predictions.add_prediction(
            dataset_name=dataset_name,
            model_name=model_name,
            y_pred=y_pred,
            y_true=np.array([]),
            partition="test",
            step_idx=loader.metadata.model_step_index if loader.metadata and loader.metadata.model_step_index is not None else 0,
            config_path=str(bundle_path),
            fold_id="all",
        )

        logger.success(f"Predicted with bundle: {model_name}")

        if all_predictions:
            return {"bundle": {"prediction": y_pred}}, run_predictions

        return y_pred, run_predictions

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _get_substep_from_artifact(self, artifact_id: str) -> int | None:
        """Get substep_index from artifact record.

        Args:
            artifact_id: Artifact ID to look up.

        Returns:
            substep_index or None if not found.
        """
        if self.artifact_loader is None:
            return None
        try:
            record = self.artifact_loader.get_record(artifact_id)
            if record is not None:
                return record.substep_index
        except (KeyError, AttributeError):
            pass
        return None
