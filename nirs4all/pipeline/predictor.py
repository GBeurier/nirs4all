"""Pipeline predictor - Handles prediction mode execution.

This module provides the Predictor class for running predictions using trained
pipelines on new datasets.

Phase 5 Enhancement:
    The Predictor now supports minimal pipeline execution via TraceBasedExtractor.
    When an execution trace is available (from Phase 2+), the predictor can extract
    and run only the required steps, significantly improving prediction speed for
    complex pipelines.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from nirs4all.data.config import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.storage.artifacts.artifact_loader import ArtifactLoader
from nirs4all.pipeline.config.context import ExecutionContext, DataSelector, PipelineState, StepMetadata
from nirs4all.pipeline.execution.builder import ExecutorBuilder
from nirs4all.pipeline.storage.io import SimulationSaver
from nirs4all.pipeline.storage.manifest_manager import ManifestManager


class Predictor:
    """Handles prediction using trained pipelines.

    This class manages the prediction workflow: loading saved models,
    replaying pipeline configurations, and generating predictions on new data.

    Phase 5 Enhancement:
        When use_minimal_pipeline=True (default), the predictor will:
        1. Check if an execution trace is available for the prediction
        2. Extract the minimal pipeline (only required steps) from the trace
        3. Execute only those steps, significantly reducing prediction time

        This is especially beneficial for complex pipelines with multiple
        preprocessing options, branches, or steps that aren't needed for
        the specific model being predicted.

    Attributes:
        runner: Parent PipelineRunner instance
        saver: File saver for managing outputs
        manifest_manager: Manager for pipeline manifests
        pipeline_uid: Unique identifier for the pipeline
        artifact_loader: Loader for trained model artifacts
        config_path: Path to the pipeline configuration
        target_model: Metadata for the target model
        use_minimal_pipeline: Whether to use minimal pipeline execution (Phase 5)
    """

    def __init__(self, runner: 'PipelineRunner', use_minimal_pipeline: bool = True):
        """Initialize predictor.

        Args:
            runner: Parent PipelineRunner instance
            use_minimal_pipeline: If True, use minimal pipeline execution when
                                  execution traces are available (Phase 5)
        """
        self.runner = runner
        self.saver: Optional[SimulationSaver] = None
        self.manifest_manager: Optional[ManifestManager] = None
        self.pipeline_uid: Optional[str] = None
        self.artifact_loader: Optional[ArtifactLoader] = None
        self.config_path: Optional[str] = None
        self.target_model: Optional[Dict[str, Any]] = None
        self.use_minimal_pipeline = use_minimal_pipeline
        self._execution_trace = None  # Cached execution trace
        self._minimal_pipeline = None  # Cached minimal pipeline

    def predict(
        self,
        prediction_obj: Union[Dict[str, Any], str],
        dataset: Union[DatasetConfigs, SpectroDataset, np.ndarray, Tuple[np.ndarray, ...], Dict, List[Dict], str, List[str]],
        dataset_name: str = "prediction_dataset",
        all_predictions: bool = False,
        verbose: int = 0
    ) -> Union[Tuple[np.ndarray, Predictions], Tuple[Dict[str, Any], Predictions]]:
        """Run prediction using a saved model on new dataset.

        Phase 5 Enhancement:
            When use_minimal_pipeline=True and an execution trace is available,
            this method will use TraceBasedExtractor to extract and execute only
            the required steps, improving prediction speed.

        Args:
            prediction_obj: Model identifier (dict with config_path or prediction ID)
            dataset: New dataset to predict on
            dataset_name: Name for the dataset
            all_predictions: If True, return all predictions; if False, return single best
            verbose: Verbosity level

        Returns:
            If all_predictions=False: (y_pred, predictions)
            If all_predictions=True: (predictions_dict, predictions)

        Example:
            >>> predictor = Predictor(runner)
            >>> y_pred, preds = predictor.predict(
            ...     {"config_path": "0001_abc123"},
            ...     X_new
            ... )
        """
        from nirs4all.utils.emoji import ROCKET, CHECK, SEARCH, BOLT

        print("=" * 120)
        print(f"\033[94m{ROCKET}Starting Nirs4all prediction(s)\033[0m")
        print("=" * 120)

        # Normalize dataset
        dataset_config = self.runner.orchestrator._normalize_dataset(
            dataset, dataset_name
        )

        # Setup prediction mode
        self.runner.mode = "predict"
        self.runner.verbose = verbose

        # Initialize saver for prediction mode
        run_dir = self._get_run_dir_from_prediction(prediction_obj)
        self.saver = SimulationSaver(run_dir, save_files=self.runner.save_files)
        self.manifest_manager = ManifestManager(run_dir)

        # Load pipeline steps and try to get execution trace
        steps = self._prepare_replay(prediction_obj, dataset_config, verbose)

        # Phase 5: Try minimal pipeline execution if trace is available
        if self.use_minimal_pipeline and self._execution_trace is not None:
            return self._predict_with_minimal_pipeline(
                dataset_config, run_dir, all_predictions, verbose
            )

        # Fallback: Full pipeline execution (original behavior)
        return self._predict_full_pipeline(
            steps, dataset_config, run_dir, all_predictions, verbose
        )

    def _predict_with_minimal_pipeline(
        self,
        dataset_config: DatasetConfigs,
        run_dir: Path,
        all_predictions: bool,
        verbose: int
    ) -> Union[Tuple[np.ndarray, Predictions], Tuple[Dict[str, Any], Predictions]]:
        """Execute prediction using minimal pipeline from trace (Phase 5).

        This method extracts and executes only the required steps from the
        execution trace, significantly improving prediction speed for complex
        pipelines.

        Args:
            dataset_config: Dataset configuration
            run_dir: Run directory path
            all_predictions: Whether to return all predictions
            verbose: Verbosity level

        Returns:
            Same as predict() method
        """
        from nirs4all.utils.emoji import CHECK, BOLT
        from nirs4all.pipeline.trace import TraceBasedExtractor
        from nirs4all.pipeline.minimal_predictor import MinimalPredictor, MinimalArtifactProvider
        from nirs4all.pipeline.config.context import RuntimeContext

        print(f"{BOLT} Using minimal pipeline execution (Phase 5)")

        # Extract minimal pipeline from trace
        extractor = TraceBasedExtractor()

        # Load full pipeline steps for step configs
        full_steps = self._load_pipeline_steps()

        # Check if we need branch-specific extraction
        # Prefer branch_name for matching (more reliable for nested branches)
        target_branch_name = self.target_model.get('branch_name') if self.target_model else None
        target_branch_id = self.target_model.get('branch_id') if self.target_model else None
        target_branch_path = self.target_model.get('branch_path') if self.target_model else None

        if target_branch_name:
            # Use branch name for extraction (most reliable for nested branches)
            minimal_pipeline = extractor.extract_for_branch_name(
                trace=self._execution_trace,
                branch_name=target_branch_name,
                full_pipeline=full_steps
            )
            if verbose > 0:
                print(f"  Extracting for branch: {target_branch_name}")
        elif target_branch_path:
            # Use explicit branch path
            minimal_pipeline = extractor.extract_for_branch(
                trace=self._execution_trace,
                branch_path=target_branch_path,
                full_pipeline=full_steps
            )
            if verbose > 0:
                print(f"  Extracting for branch path: {target_branch_path}")
        elif target_branch_id is not None:
            # Fallback: convert branch_id to path (only works for single-level branches)
            branch_path = [target_branch_id]
            minimal_pipeline = extractor.extract_for_branch(
                trace=self._execution_trace,
                branch_path=branch_path,
                full_pipeline=full_steps
            )
            if verbose > 0:
                print(f"  Extracting for branch_id: {target_branch_id}")
        else:
            # No branch filter, extract all steps up to model
            minimal_pipeline = extractor.extract(
                trace=self._execution_trace,
                full_pipeline=full_steps,
                up_to_model=True
            )

        self._minimal_pipeline = minimal_pipeline

        if verbose > 0:
            print(f"  Minimal pipeline: {minimal_pipeline.get_step_count()} steps "
                  f"(from {len(full_steps)} total)")
            print(f"  Artifacts: {len(minimal_pipeline.get_artifact_ids())}")

        # Execute prediction using MinimalPredictor
        run_predictions = Predictions()

        for config, name in dataset_config.configs:
            dataset_obj = dataset_config.get_dataset(config, name)
            config_predictions = Predictions()

            # Initialize context
            context = ExecutionContext(
                selector=DataSelector(
                    partition=None,
                    processing=[["raw"]] * dataset_obj.features_sources(),
                    layout="2d",
                    concat_source=True
                ),
                state=PipelineState(y_processing="numeric", step_number=0, mode="predict"),
                metadata=StepMetadata()
            )

            # Build executor
            executor = (ExecutorBuilder()
                .with_run_directory(run_dir)
                .with_verbose(verbose)
                .with_mode("predict")
                .with_save_files(self.runner.save_files)
                .with_continue_on_error(self.runner.continue_on_error)
                .with_show_spinner(self.runner.show_spinner)
                .with_plots_visible(self.runner.plots_visible)
                .with_artifact_loader(self.artifact_loader)
                .with_saver(self.saver)
                .with_manifest_manager(self.manifest_manager)
                .build())

            # Create artifact provider from minimal pipeline
            artifact_provider = MinimalArtifactProvider(
                minimal_pipeline=minimal_pipeline,
                artifact_loader=self.artifact_loader
            )

            # Create RuntimeContext with artifact_provider
            runtime_context = RuntimeContext(
                saver=self.saver,
                manifest_manager=self.manifest_manager,
                artifact_loader=self.artifact_loader,
                artifact_provider=artifact_provider,
                step_runner=executor.step_runner,
                target_model=self.target_model,
                explainer=self.runner.explainer
            )

            # Extract step configs from minimal pipeline
            steps = [step.step_config for step in minimal_pipeline.steps]

            # Execute minimal pipeline
            executor.execute_minimal(
                steps=steps,
                minimal_pipeline=minimal_pipeline,
                dataset=dataset_obj,
                context=context,
                runtime_context=runtime_context,
                prediction_store=config_predictions
            )

            run_predictions.merge_predictions(config_predictions)

        # Process results (same as full pipeline)
        return self._process_prediction_results(run_predictions, all_predictions)

    def _predict_full_pipeline(
        self,
        steps: List[Any],
        dataset_config: DatasetConfigs,
        run_dir: Path,
        all_predictions: bool,
        verbose: int
    ) -> Union[Tuple[np.ndarray, Predictions], Tuple[Dict[str, Any], Predictions]]:
        """Execute prediction using full pipeline (original behavior).

        Args:
            steps: Full pipeline steps
            dataset_config: Dataset configuration
            run_dir: Run directory path
            all_predictions: Whether to return all predictions
            verbose: Verbosity level

        Returns:
            Same as predict() method
        """
        from nirs4all.pipeline.config.context import RuntimeContext, LoaderArtifactProvider

        # Execute pipeline on dataset
        run_predictions = Predictions()
        for config, name in dataset_config.configs:
            dataset_obj = dataset_config.get_dataset(config, name)
            config_predictions = Predictions()

            # Initialize context
            context = ExecutionContext(
                selector=DataSelector(
                    partition=None,
                    processing=[["raw"]] * dataset_obj.features_sources(),
                    layout="2d",
                    concat_source=True
                ),
                state=PipelineState(y_processing="numeric", step_number=0, mode="predict"),
                metadata=StepMetadata()
            )

            # Build executor using ExecutorBuilder
            executor = (ExecutorBuilder()
                .with_run_directory(run_dir)
                .with_verbose(verbose)
                .with_mode("predict")
                .with_save_files(self.runner.save_files)
                .with_continue_on_error(self.runner.continue_on_error)
                .with_show_spinner(self.runner.show_spinner)
                .with_plots_visible(self.runner.plots_visible)
                .with_artifact_loader(self.artifact_loader)
                .with_saver(self.saver)
                .with_manifest_manager(self.manifest_manager)
                .build())

            # Create artifact_provider from artifact_loader for controller-agnostic artifact injection
            artifact_provider = None
            if self.artifact_loader:
                artifact_provider = LoaderArtifactProvider(loader=self.artifact_loader)

            runtime_context = RuntimeContext(
                saver=self.saver,
                manifest_manager=self.manifest_manager,
                artifact_loader=self.artifact_loader,
                artifact_provider=artifact_provider,
                step_runner=executor.step_runner,
                target_model=self.target_model,
                explainer=self.runner.explainer
            )

            executor.execute(steps, "prediction", dataset_obj, context, runtime_context, config_predictions)
            run_predictions.merge_predictions(config_predictions)

        return self._process_prediction_results(run_predictions, all_predictions)

    def _process_prediction_results(
        self,
        run_predictions: Predictions,
        all_predictions: bool
    ) -> Union[Tuple[np.ndarray, Predictions], Tuple[Dict[str, Any], Predictions]]:
        """Process prediction results and return in requested format.

        Args:
            run_predictions: Predictions object with results
            all_predictions: Whether to return all predictions

        Returns:
            Formatted prediction results
        """
        from nirs4all.utils.emoji import CHECK

        if all_predictions:
            res = {}
            for pred in run_predictions.to_dicts():
                if pred['dataset_name'] not in res:
                    res[pred['dataset_name']] = {}
                res[pred['dataset_name']][pred['id']] = pred['y_pred']
            return res, run_predictions

        # Get single prediction matching target model
        filter_kwargs = {
            'model_name': self.target_model.get('model_name', None),
            'step_idx': self.target_model.get('step_idx', None),
            'fold_id': self.target_model.get('fold_id', None)
        }

        # Add branch filtering if target model has branch info
        target_branch_id = self.target_model.get('branch_id')
        if target_branch_id is not None:
            filter_kwargs['branch_id'] = target_branch_id

        candidates = run_predictions.filter_predictions(**filter_kwargs)

        # Prefer predictions with non-empty y_pred (in predict mode, train partition may be empty)
        non_empty = [p for p in candidates if len(p['y_pred']) > 0]
        single_pred = non_empty[0] if non_empty else (candidates[0] if candidates else None)

        if single_pred is None:
            raise ValueError("No matching prediction found for the specified model criteria.")

        print(f"{CHECK}Predicted with: {single_pred['model_name']} [{single_pred['id']}]")
        filename = f"Predict_[{single_pred['id']}].csv"
        y_pred = single_pred["y_pred"]
        prediction_path = self.saver.base_path / filename
        Predictions.save_predictions_to_csv(y_pred=y_pred, filepath=prediction_path)

        return single_pred["y_pred"], run_predictions

    def _load_pipeline_steps(self) -> List[Any]:
        """Load full pipeline steps from pipeline.json.

        Returns:
            List of pipeline steps
        """
        import json

        pipeline_dir_name = Path(self.config_path).parts[-1] if '/' in self.config_path or '\\' in self.config_path else self.config_path
        config_dir = self.saver.base_path / pipeline_dir_name
        pipeline_json = config_dir / "pipeline.json"

        if not pipeline_json.exists():
            return []

        with open(pipeline_json, 'r', encoding='utf-8') as f:
            pipeline_data = json.load(f)

        if isinstance(pipeline_data, dict) and "steps" in pipeline_data:
            return pipeline_data["steps"]
        return pipeline_data if isinstance(pipeline_data, list) else []

    def _get_run_dir_from_prediction(self, prediction_obj: Union[Dict[str, Any], str]) -> Path:
        """Get run directory from prediction object.

        Args:
            prediction_obj: Model identifier

        Returns:
            Path to run directory

        Raises:
            ValueError: If no run directory can be found
        """
        if isinstance(prediction_obj, dict):
            if 'run_dir' in prediction_obj:
                return Path(prediction_obj['run_dir'])
            elif 'config_path' in prediction_obj:
                config_path = prediction_obj['config_path']
                dataset_name = Path(config_path).parts[0]
                matching_dirs = sorted(
                    self.runner.orchestrator.runs_dir.glob(f"*_{dataset_name}"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
                if matching_dirs:
                    return matching_dirs[0]
                raise ValueError(f"No run directory found for dataset: {dataset_name}")

        # Fallback: use most recent run
        run_dirs = sorted(
            self.runner.orchestrator.runs_dir.glob("*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if run_dirs:
            return run_dirs[0]
        raise ValueError("No run directories found")

    def _prepare_replay(
        self,
        selection_obj: Union[Dict[str, Any], str],
        dataset_config: DatasetConfigs,
        verbose: int = 0
    ) -> List[Any]:
        """Prepare pipeline replay from saved configuration.

        Phase 5 Enhancement:
            Also attempts to load an execution trace for minimal pipeline execution.
            If trace_id is in the prediction, loads that specific trace.
            Otherwise, loads the latest trace for the pipeline.

        Args:
            selection_obj: Model selection criteria
            dataset_config: Dataset configuration
            verbose: Verbosity level

        Returns:
            List of pipeline steps to execute

        Raises:
            ValueError: If pipeline_uid is missing or invalid
            FileNotFoundError: If pipeline configuration or manifest not found
        """
        from nirs4all.utils.emoji import SEARCH
        import json

        # Get configuration path and target model
        config_path, target_model = self.saver.get_predict_targets(selection_obj)
        target_model.pop("y_pred", None)
        target_model.pop("y_true", None)

        self.config_path = config_path
        self.target_model = target_model
        self.runner.target_model = target_model  # Set on runner for controller access

        pipeline_uid = target_model.get('pipeline_uid')
        if not pipeline_uid:
            raise ValueError(
                "No pipeline_uid found in prediction metadata. "
                "This prediction was created with an older version of nirs4all. "
                "Please retrain the model."
            )

        self.pipeline_uid = pipeline_uid

        # Load pipeline configuration
        pipeline_dir_name = Path(config_path).parts[-1] if '/' in config_path or '\\' in config_path else config_path
        config_dir = self.saver.base_path / pipeline_dir_name
        pipeline_json = config_dir / "pipeline.json"

        if verbose > 0:
            print(f"{SEARCH}Loading {pipeline_json}")

        if not pipeline_json.exists():
            raise FileNotFoundError(f"Pipeline not found: {pipeline_json}")

        with open(pipeline_json, 'r', encoding='utf-8') as f:
            pipeline_data = json.load(f)

        steps = pipeline_data["steps"] if isinstance(pipeline_data, dict) and "steps" in pipeline_data else pipeline_data

        # Load binaries from manifest
        manifest_path = self.saver.base_path / pipeline_uid / "manifest.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                f"Pipeline UID: {pipeline_uid}\n"
                f"The model artifacts may have been deleted or moved."
            )

        print(f"{SEARCH}Loading from manifest: {pipeline_uid}")
        manifest = self.manifest_manager.load_manifest(pipeline_uid)
        self.artifact_loader = ArtifactLoader.from_manifest(manifest, self.saver.base_path)

        # Phase 5: Try to load execution trace for minimal pipeline execution
        self._execution_trace = None
        if self.use_minimal_pipeline:
            trace_id = target_model.get('trace_id')
            if trace_id:
                # Load specific trace referenced by prediction
                self._execution_trace = self.manifest_manager.load_execution_trace(
                    pipeline_uid, trace_id
                )
                if verbose > 0 and self._execution_trace:
                    print(f"{SEARCH}Loaded execution trace: {trace_id}")
            else:
                # Try to load latest trace for this pipeline
                self._execution_trace = self.manifest_manager.get_latest_execution_trace(
                    pipeline_uid
                )
                if verbose > 0 and self._execution_trace:
                    print(f"{SEARCH}Loaded latest execution trace: {self._execution_trace.trace_id}")

            if verbose > 0 and not self._execution_trace:
                print("  No execution trace available, using full pipeline replay")

        return steps
