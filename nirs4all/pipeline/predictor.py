"""Pipeline predictor - Handles prediction mode execution.

This module provides the Predictor class for running predictions using trained
pipelines on new datasets.

Supports two prediction paths:
    - Bundle path: ``.n4a`` bundles loaded via BundleLoader (self-contained).
    - Resolver path: Prediction dicts / folders resolved via PredictionResolver,
      with optional minimal pipeline extraction from execution traces.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from nirs4all.core.logging import get_logger
from nirs4all.data.config import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.storage.artifacts.artifact_loader import ArtifactLoader
from nirs4all.pipeline.config.context import ExecutionContext, DataSelector, PipelineState, StepMetadata
from nirs4all.pipeline.execution.builder import ExecutorBuilder

logger = get_logger(__name__)


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
        self.pipeline_uid: Optional[str] = None
        self.artifact_loader: Optional[ArtifactLoader] = None
        self.config_path: Optional[str] = None
        self.target_model: Optional[Dict[str, Any]] = None
        self.use_minimal_pipeline = use_minimal_pipeline
        self._execution_trace = None  # Cached execution trace
        self._minimal_pipeline = None  # Cached minimal pipeline
        self._resolved = None  # Cached ResolvedPrediction

    def predict(
        self,
        prediction_obj: Union[Dict[str, Any], str],
        dataset: Union[DatasetConfigs, SpectroDataset, np.ndarray, Tuple[np.ndarray, ...], Dict, List[Dict], str, List[str]],
        dataset_name: str = "prediction_dataset",
        all_predictions: bool = False,
        verbose: int = 0,
    ) -> Union[Tuple[np.ndarray, Predictions], Tuple[Dict[str, Any], Predictions]]:
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

    def _resolve_prediction(self, prediction_obj: Union[Dict[str, Any], str], verbose: int) -> None:
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
        )
        self._resolved = resolver.resolve(prediction_obj, verbose=verbose)

        # Extract target model metadata
        if isinstance(prediction_obj, dict):
            target_model = {k: v for k, v in prediction_obj.items() if k not in ("y_pred", "y_true", "X")}
        else:
            target_model = dict(self._resolved.target_model) if self._resolved.target_model else {}

        if not target_model or "model_name" not in target_model:
            raise ValueError(
                "No model information found in prediction. "
                "Please provide a prediction dict with 'model_name' and 'pipeline_uid' keys."
            )

        self.target_model = target_model
        self.runner.target_model = target_model
        self.pipeline_uid = self._resolved.pipeline_uid
        self.config_path = self._resolved.pipeline_uid
        self.artifact_loader = None

        # Create artifact loader from resolved artifact provider
        if self._resolved.artifact_provider and hasattr(self._resolved.artifact_provider, "artifact_loader"):
            self.artifact_loader = self._resolved.artifact_provider.artifact_loader

        # Cache execution trace
        self._execution_trace = self._resolved.trace

    # -----------------------------------------------------------------
    # Minimal pipeline path
    # -----------------------------------------------------------------

    def _predict_with_minimal_pipeline(
        self,
        dataset_config: DatasetConfigs,
        all_predictions: bool,
        verbose: int,
    ) -> Union[Tuple[np.ndarray, Predictions], Tuple[Dict[str, Any], Predictions]]:
        """Execute prediction using minimal pipeline extracted from trace.

        Args:
            dataset_config: Dataset configuration.
            all_predictions: Whether to return all predictions.
            verbose: Verbosity level.

        Returns:
            Same as ``predict()`` method.
        """
        from nirs4all.pipeline.trace import TraceBasedExtractor
        from nirs4all.pipeline.minimal_predictor import MinimalArtifactProvider
        from nirs4all.pipeline.config.context import RuntimeContext

        logger.info("Using minimal pipeline execution")

        extractor = TraceBasedExtractor()
        full_steps = list(self._resolved.minimal_pipeline)

        target_branch_name = self.target_model.get("branch_name") if self.target_model else None
        target_branch_id = self.target_model.get("branch_id") if self.target_model else None
        target_branch_path = self.target_model.get("branch_path") if self.target_model else None

        if target_branch_name:
            minimal_pipeline = extractor.extract_for_branch_name(
                trace=self._execution_trace, branch_name=target_branch_name, full_pipeline=full_steps,
            )
        elif target_branch_path:
            minimal_pipeline = extractor.extract_for_branch(
                trace=self._execution_trace, branch_path=target_branch_path, full_pipeline=full_steps,
            )
        elif target_branch_id is not None:
            minimal_pipeline = extractor.extract_for_branch(
                trace=self._execution_trace, branch_path=[target_branch_id], full_pipeline=full_steps,
            )
        else:
            minimal_pipeline = extractor.extract(
                trace=self._execution_trace, full_pipeline=full_steps, up_to_model=True,
            )

        self._minimal_pipeline = minimal_pipeline

        logger.debug(
            f"Minimal pipeline: {minimal_pipeline.get_step_count()} steps "
            f"(from {len(full_steps)} total)"
        )

        run_predictions = Predictions()

        for config, name in dataset_config.configs:
            dataset_obj = dataset_config.get_dataset(config, name)
            config_predictions = Predictions()

            selector = DataSelector(
                partition=None,
                processing=[["raw"]] * dataset_obj.features_sources(),
                layout="2d",
                concat_source=True,
            )

            if target_branch_id is not None or target_branch_name:
                bp = target_branch_path or ([target_branch_id] if target_branch_id is not None else None)
                selector = selector.with_branch(branch_id=target_branch_id, branch_name=target_branch_name, branch_path=bp)

            context = ExecutionContext(
                selector=selector,
                state=PipelineState(y_processing="numeric", step_number=0, mode="predict"),
                metadata=StepMetadata(),
            )

            executor = (
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

            # Determine substep target
            target_sub_index = None
            target_model_name = None
            if self.target_model:
                model_artifact_id = self.target_model.get("model_artifact_id")
                if model_artifact_id:
                    target_sub_index = self._get_substep_from_artifact(model_artifact_id)
                else:
                    target_model_name = self.target_model.get("model_name")

            artifact_provider = MinimalArtifactProvider(
                minimal_pipeline=minimal_pipeline,
                artifact_loader=self.artifact_loader,
                target_sub_index=target_sub_index,
                target_model_name=target_model_name,
            )

            runtime_context = RuntimeContext(
                artifact_loader=self.artifact_loader,
                artifact_provider=artifact_provider,
                step_runner=executor.step_runner,
                target_model=self.target_model,
                explainer=self.runner.explainer,
            )

            steps = [step.step_config for step in minimal_pipeline.steps]

            executor.execute_minimal(
                steps=steps,
                minimal_pipeline=minimal_pipeline,
                dataset=dataset_obj,
                context=context,
                runtime_context=runtime_context,
                prediction_store=config_predictions,
            )

            run_predictions.merge_predictions(config_predictions)

        return self._process_prediction_results(run_predictions, all_predictions)

    # -----------------------------------------------------------------
    # Full pipeline path
    # -----------------------------------------------------------------

    def _predict_full_pipeline(
        self,
        dataset_config: DatasetConfigs,
        all_predictions: bool,
        verbose: int,
    ) -> Union[Tuple[np.ndarray, Predictions], Tuple[Dict[str, Any], Predictions]]:
        """Execute prediction using full pipeline replay.

        Args:
            dataset_config: Dataset configuration.
            all_predictions: Whether to return all predictions.
            verbose: Verbosity level.

        Returns:
            Same as ``predict()`` method.
        """
        from nirs4all.pipeline.config.context import RuntimeContext, LoaderArtifactProvider

        steps = list(self._resolved.minimal_pipeline)

        run_predictions = Predictions()
        for config, name in dataset_config.configs:
            dataset_obj = dataset_config.get_dataset(config, name)
            config_predictions = Predictions()

            context = ExecutionContext(
                selector=DataSelector(
                    partition=None,
                    processing=[["raw"]] * dataset_obj.features_sources(),
                    layout="2d",
                    concat_source=True,
                ),
                state=PipelineState(y_processing="numeric", step_number=0, mode="predict"),
                metadata=StepMetadata(),
            )

            executor = (
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

            artifact_provider = self._resolved.artifact_provider
            if artifact_provider is None and self.artifact_loader:
                artifact_provider = LoaderArtifactProvider(loader=self.artifact_loader)

            runtime_context = RuntimeContext(
                artifact_loader=self.artifact_loader,
                artifact_provider=artifact_provider,
                step_runner=executor.step_runner,
                target_model=self.target_model,
                explainer=self.runner.explainer,
            )

            executor.execute(steps, "prediction", dataset_obj, context, runtime_context, config_predictions)
            run_predictions.merge_predictions(config_predictions)

        return self._process_prediction_results(run_predictions, all_predictions)

    # -----------------------------------------------------------------
    # Result processing
    # -----------------------------------------------------------------

    def _process_prediction_results(
        self,
        run_predictions: Predictions,
        all_predictions: bool,
    ) -> Union[Tuple[np.ndarray, Predictions], Tuple[Dict[str, Any], Predictions]]:
        """Process prediction results and return in requested format.

        Args:
            run_predictions: Predictions object with results.
            all_predictions: Whether to return all predictions.

        Returns:
            Formatted prediction results.
        """
        if all_predictions:
            res: Dict[str, Any] = {}
            for pred in run_predictions.to_dicts():
                ds = pred["dataset_name"]
                if ds not in res:
                    res[ds] = {}
                res[ds][pred["id"]] = pred["y_pred"]
            return res, run_predictions

        # Get single prediction matching target model
        target_fold_id = self.target_model.get("fold_id", None)

        is_aggregated_fold = target_fold_id in ("avg", "w_avg")

        filter_kwargs: Dict[str, Any] = {
            "model_name": self.target_model.get("model_name"),
            "step_idx": self.target_model.get("step_idx"),
        }

        if not is_aggregated_fold:
            filter_kwargs["fold_id"] = target_fold_id

        target_branch_id = self.target_model.get("branch_id")
        if target_branch_id is not None:
            filter_kwargs["branch_id"] = target_branch_id

        candidates = run_predictions.filter_predictions(**filter_kwargs)
        non_empty = [p for p in candidates if len(p["y_pred"]) > 0]

        if is_aggregated_fold and len(non_empty) > 1:
            single_pred = self._aggregate_fold_predictions(non_empty, target_fold_id)
        else:
            single_pred = non_empty[0] if non_empty else (candidates[0] if candidates else None)

        if single_pred is None:
            raise ValueError("No matching prediction found for the specified model criteria.")

        logger.success(f"Predicted with: {single_pred['model_name']} [{single_pred['id']}]")

        return single_pred["y_pred"], run_predictions

    def _aggregate_fold_predictions(
        self,
        non_empty: List[Dict[str, Any]],
        target_fold_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Aggregate predictions across folds.

        Args:
            non_empty: Non-empty prediction entries.
            target_fold_id: Target fold id ('avg' or 'w_avg').

        Returns:
            Aggregated prediction dict, or first candidate.
        """
        fold_weights = self.target_model.get("weights")

        fold_preds: Dict[Any, np.ndarray] = {}
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

        if has_valid_weights:
            if isinstance(fold_weights, dict):
                wl = [fold_weights.get(fid, 1.0) for fid in sorted_folds]
            else:
                wl = [fold_weights[i] for i in range(len(sorted_folds))]
            total_w = sum(wl)
            y_pred = sum(w * np.array(y) for w, y in zip(wl, y_arrays)) / total_w
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
        dataset: Union[DatasetConfigs, SpectroDataset, np.ndarray, Tuple[np.ndarray, ...], Dict, List[Dict], str, List[str]],
        dataset_name: str,
        all_predictions: bool,
        verbose: int,
    ) -> Union[Tuple[np.ndarray, Predictions], Tuple[Dict[str, Any], Predictions]]:
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
        X_data = None

        for data_config, name in dataset_config.configs:
            dataset_obj = dataset_config.get_dataset(data_config, name)
            X_data = dataset_obj.x({})
            break

        if X_data is None:
            raise ValueError("No data found in dataset for prediction")

        y_pred = loader.predict(X_data)

        model_name = "bundle_model"
        if loader.metadata:
            model_name = loader.metadata.original_manifest.get("name", "bundle_model")

        run_predictions = Predictions()
        prediction_entry = {
            "model_name": model_name,
            "id": "bundle_prediction",
            "y_pred": y_pred,
            "y_true": np.array([]),
            "partition": "test",
            "step_idx": loader.metadata.model_step_index if loader.metadata else 0,
            "dataset_name": dataset_name,
            "config_path": str(bundle_path),
            "fold_id": "all",
        }
        run_predictions.add_prediction(prediction_entry)

        logger.success(f"Predicted with bundle: {model_name}")

        if all_predictions:
            return {"bundle": {"prediction": y_pred}}, run_predictions

        return y_pred, run_predictions

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _get_substep_from_artifact(self, artifact_id: str) -> Optional[int]:
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
