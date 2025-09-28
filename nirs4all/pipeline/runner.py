"""
PipelineRunner - Simplified execution engine with branch-based context

Features:
- Simple branch-based context
- Direct operation execution
- Basic branching support
- Simplified data selection
"""
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import json
import warnings

from joblib import Parallel, delayed, parallel_backend
from nirs4all.dataset.predictions import Predictions

# from nirs4all.controllers.controller import OperatorController

from nirs4all.pipeline.serialization import deserialize_component
from nirs4all.pipeline.history import PipelineHistory
from nirs4all.pipeline.config import PipelineConfigs
from nirs4all.pipeline.io import SimulationSaver
from nirs4all.dataset.dataset import SpectroDataset
from nirs4all.dataset.dataset_config import DatasetConfigs
from nirs4all.controllers.registry import CONTROLLER_REGISTRY
from nirs4all.pipeline.binary_loader import BinaryLoader
from nirs4all.utils.spinner import spinner_context



class PipelineRunner:
    """PipelineRunner - Executes a pipeline with enhanced context management and DatasetView support."""

    ##TODO operators should not be located in workflow and serialization but only in registry (basically hardcode of class, _runtime_instance and so, dynamic loading for the rest)
    ##TODO handle the models defined as a class
    WORKFLOW_OPERATORS = ["sample_augmentation", "feature_augmentation", "branch", "dispatch", "model", "stack",
                          "scope", "cluster", "merge", "uncluster", "unscope", "chart_2d", "chart_3d", "fold_chart",
                          "model", "y_processing", "y_chart"]
    SERIALIZATION_OPERATORS = ["class", "function", "module", "object", "pipeline", "instance"]

    def __init__(self, ##TODO add resume / overwrite support / realtime viz
                 max_workers: Optional[int] = None,
                 continue_on_error: bool = False,
                 backend: str = 'threading',
                 verbose: int = 0,
                 parallel: bool = False,
                 results_path: Optional[str] = None,
                 save_binaries: bool = True,
                 mode: str = "train",
                 load_existing_predictions: bool = True,
                 show_spinner: bool = True):

        self.max_workers = max_workers or -1  # -1 means use all available cores
        self.continue_on_error = continue_on_error
        self.backend = backend
        self.verbose = verbose
        self.history = PipelineHistory()
        self.parallel = parallel
        self.step_number = 0  # Initialize step number for tracking
        self.substep_number = -1  # Initialize sub-step number for tracking
        self.saver = SimulationSaver(results_path)
        self.operation_count = 0
        self.save_binaries = save_binaries
        self.mode = mode
        self.load_existing_predictions = load_existing_predictions
        self.step_binaries: Dict[str, List[str]] = {}  # Track step-to-binary mapping
        self.binary_loader: Optional[BinaryLoader] = None
        self.show_spinner = show_spinner

    def next_op(self) -> int:
        """Get the next operation ID."""
        self.operation_count += 1
        return self.operation_count

    def _display_best_for_config(self, dataset: SpectroDataset, config_name: str) -> None:
        """Display best score for this specific config."""
        try:
            from nirs4all.utils.model_utils import ModelUtils

            # Get all predictions for this config
            all_keys = dataset._predictions.list_keys()
            config_predictions = [key for key in all_keys if config_name in key]

            if not config_predictions:
                return

            best_score = None
            best_model = None
            higher_is_better = False

            for key in config_predictions:
                parts = key.split('_')
                if len(parts) >= 4:
                    pred_dataset_name = parts[0]
                    pipeline_name = '_'.join(parts[1:-2])
                    model_name = parts[-2]
                    partition_name = parts[-1]

                    pred_data = dataset._predictions.get_prediction_data(
                        pred_dataset_name, pipeline_name, model_name, partition_name
                    )

                    if pred_data and 'y_true' in pred_data and 'y_pred' in pred_data:
                        task_type = ModelUtils.detect_task_type(pred_data['y_true'])
                        scores = ModelUtils.calculate_scores(pred_data['y_true'], pred_data['y_pred'], task_type)
                        best_metric, metric_higher_is_better = ModelUtils.get_best_score_metric(task_type)
                        score = scores.get(best_metric)

                        higher_is_better = metric_higher_is_better

                        if score is not None:
                            if best_score is None or (
                                (higher_is_better and score > best_score) or
                                (not higher_is_better and score < best_score)
                            ):
                                best_score = score
                                best_model = model_name

            if best_score is not None and best_model is not None:
                direction = "↑" if higher_is_better else "↓"
                print(f"🏆 Best for config: {best_model} ({config_name}) - mse={best_score:.4f}{direction}")

        except Exception as e:
            print(f"⚠️ Could not display best for config: {e}")

    def run(self, pipeline_configs: PipelineConfigs, dataset_configs: DatasetConfigs) -> List[Tuple[SpectroDataset, PipelineHistory, Any]]:
        """Run pipeline configurations on dataset configurations."""
        results = []

        # Get datasets from DatasetConfigs
        for d_config in dataset_configs.data_configs:
            print("=" * 200)
            existing_predictions = 0
            prediction_db = None
            prediction_run = Predictions()
            dataset_name = "unknown_dataset"
            for i, (steps, config_name) in enumerate(zip(pipeline_configs.steps, pipeline_configs.names)):
                dataset = dataset_configs.get_dataset(d_config)
                dataset_name = dataset.name

                if i == 0 and self.load_existing_predictions:
                    prediction_db = Predictions.load_dataset_predictions(dataset, self.saver)
                    existing_predictions = len(prediction_db._predictions.keys()) if prediction_db is not None else 0

                self._run_single(steps, config_name, dataset)
                # results.append(result)

                if prediction_db is not None:
                    prediction_db.merge_predictions(dataset._predictions)
                    prediction_run.merge_predictions(dataset._predictions)
            if prediction_db is not None:
                prediction_db.display_best_scores_summary(dataset_name, existing_predictions)

            prediction_db.save_to_file(str(self.saver.base_path / dataset_name / f"{dataset_name}_predictions.json"))
            results.append((prediction_db, prediction_run))

        return results

    def _run_single(self, steps: List[Any], config_name: str, dataset: SpectroDataset) -> SpectroDataset:
        """Run a single pipeline configuration on a single dataset."""
        # Reset runner state for each run
        self.history = PipelineHistory()
        self.step_number = 0
        self.substep_number = -1
        self.operation_count = 0
        self.step_binaries = {}

        print("=" * 200)
        print(f"\033[94m🚀 Starting pipeline {config_name} on dataset {dataset.name}\033[0m")
        print("-" * 200)

        storage_path = self.saver.register(dataset.name, config_name)
        self.saver.save_json("pipeline.json", PipelineConfigs.serializable_steps(steps))

        # Initialize context
        context = {"processing": [["raw"]] * dataset.features_sources(), "y": "numeric"}

        try:
            self.run_steps(steps, dataset, context, execution="sequential")

            # Save enhanced configuration with metadata if saving binaries
            enhanced_config = {
                "steps": PipelineConfigs.serializable_steps(steps),
                "execution_metadata": {
                    "step_binaries": self.step_binaries,
                    "created_at": datetime.now().isoformat(),
                    "pipeline_version": "1.0",
                    "mode": self.mode
                }
            }
            self.saver.save_json("pipeline.json", enhanced_config)

            # Display best score for this specific config
            if hasattr(dataset, '_predictions') and dataset._predictions:
                self._display_best_for_config(dataset, config_name)

            print(f"\033[94m✅ Pipeline {config_name} completed successfully on dataset {dataset.name}\033[0m")

        except Exception as e:
            print(f"\033[91m❌ Pipeline {config_name} on dataset {dataset.name} failed: \n{str(e)}\033[0m")
            import traceback
            traceback.print_exc()
            raise

        return dataset

    def run_steps(self, steps: List[Any], dataset: SpectroDataset, context: Union[List[Dict[str, Any]], Dict[str, Any]], execution: str = "sequential", is_substep: bool = False) -> Dict[str, Any]:
        """Run a list of steps with enhanced context management and DatasetView support."""
        if not isinstance(steps, list):
            steps = [steps]
        # print(f"\033[94m🔄 Running {len(steps)} steps in {execution} mode\033[0m")

        if execution == "sequential":
            if isinstance(context, list) and len(context) == len(steps):
                # print("🔄 Running steps sequentially with separate contexts")
                for step, ctx in zip(steps, context):
                    self.run_step(step, dataset, ctx, is_substep=is_substep)
                return context[-1]
            elif isinstance(context, dict):
                # print("🔄 Running steps sequentially with shared context")
                for step in steps:
                    context = self.run_step(step, dataset, context, is_substep=is_substep)
                    # print(f"🔹 Updated context after step: {context}")
                self.substep_number = -1  # Reset sub-step number after sequential execution
                return context

        elif execution == "parallel" and self.parallel:
            # print(f"🔄 Running steps in parallel with {self.max_workers} workers")
            with parallel_backend(self.backend, n_jobs=self.max_workers):
                Parallel()(delayed(self.run_step)(step, dataset, context) for step, context in zip(steps, context))

    def run_step(self, step: Any, dataset: SpectroDataset, context: Dict[str, Any], *, is_substep: bool = False):
        """
        Run a single pipeline step with enhanced context management and DatasetView support.
        """
        before_dataset_str = str(dataset)

        step_description = str(step)  # Simple description for now
        if is_substep:
            self.substep_number += 1
            print(f"\033[96m   ▶ Sub-step {self.step_number}.{self.substep_number}: {step_description}\033[0m")
        else:
            self.step_number += 1
            self.substep_number = 0  # Reset substep counter for new main step
            print(f"\033[92m🔷 Step {self.step_number}: {step_description}\033[0m")
        # print(f"🔹 Current context: {context}")
        # print(f"🔹 Step config: {step}")

        if step is None:
            print("🔷 No operation defined for this step, skipping.")
            return context

        # Start step tracking
        # step_execution = self.history.start_step(
        #     step_number=self.current_step,
        #     step_description=step_description,
        #     step_config=step
        # )

        try:
            operator, controller = None, None
            if isinstance(step, dict):
                if key := next((k for k in step if k in self.WORKFLOW_OPERATORS), None):
                    # print(f"📋 Workflow operation: {key}")
                    if 'class' in step[key]:
                        if '_runtime_instance' in step[key]:
                            operator = step[key]['_runtime_instance']
                        else:
                            operator = deserialize_component(step[key])
                        controller = self._select_controller(step, keyword=key, operator=operator)
                    else:
                        controller = self._select_controller(step, keyword=key)
                elif key := next((k for k in step if k in self.SERIALIZATION_OPERATORS), None):
                    # print(f"📦 Deserializing dict operation: {key}")
                    if '_runtime_instance' in step:
                        operator = step['_runtime_instance']
                    else:
                        operator = deserialize_component(step)
                    controller = self._select_controller(step, operator=operator)
                else:
                    # print(f"🔗 Running dict operation: {step}")
                    controller = self._select_controller(step)
            elif isinstance(step, list):
                # print(f"🔗 Sub-pipeline with {len(step)} steps")
                return self.run_steps(step, dataset, context, execution="sequential", is_substep=True)

            elif isinstance(step, str):
                if key := next((s for s in step.split() if s in self.WORKFLOW_OPERATORS), None):
                    # print(f"📋 Workflow operation: {key}")
                    controller = self._select_controller(key, keyword=key)
                else:
                    # print(f"📦 Deserializing str operation: {step}")
                    operator = deserialize_component(step)
                    controller = self._select_controller(step, operator=operator, keyword=step)

            else:
                # print(f"🔍 Unknown step type: {type(step).__name__}, executing as operation")
                controller = self._select_controller(step)

            if controller is not None:
                # Check if controller supports prediction mode
                if self.mode == "predict" and not controller.supports_prediction_mode():
                    print(f"🔄 Skipping step {self.step_number} in prediction mode")
                    return context

                # Load binaries if in prediction mode
                loaded_binaries = None
                if self.mode == "predict" and self.binary_loader is not None:
                    loaded_binaries = self.binary_loader.get_binaries_for_step(
                        self.step_number, self.substep_number
                    )

                # print(f"🔄 Selected controller: {controller.__class__.__name__}")
                context["step_id"] = self.step_number
                return self._execute_controller(
                    controller, step, operator, dataset, context, -1, loaded_binaries
                )


            # self.history.complete_step(step_execution.step_id)

        except Exception as e:
            # Fail step
            # self.history.fail_step(step_execution.step_id, str(e))
            import traceback
            traceback.print_exc()
            if self.continue_on_error:
                print(f"⚠️ Step failed but continuing: {str(e)}")
            else:
                raise RuntimeError(f"Pipeline step failed: {str(e)}") from e

        finally:
            if not is_substep:
                print("-" * 200)
                # after_dataset_str = str(dataset)
                # # print(before_dataset_str)
                # if before_dataset_str != after_dataset_str:
                #     print(f"\033[97mUpdate: {after_dataset_str}\033[0m")
                #     print("-" * 200)

    def _select_controller(self, step: Any, operator: Any = None, keyword: str = ""):
        matches = [cls for cls in CONTROLLER_REGISTRY if cls.matches(step, operator, keyword)]
        if not matches:
            raise TypeError(f"No matching controller found for {step}. Available controllers: {[cls.__name__ for cls in CONTROLLER_REGISTRY]}")
        matches.sort(key=lambda c: c.priority)
        return matches[0]()

    def _execute_controller(  # TODO Choose one option for multi-source datasets and parrallel execution
        self,
        controller: Any,
        step: Any,
        operator: Any,
        dataset: SpectroDataset,
        context: Dict[str, Any],
        source: Union[int, List[int]] = -1,
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None
    ):
        """Execute the controller for the given step and operator."""
        operator_name = operator.__class__.__name__ if operator is not None else ""
        controller_name = controller.__class__.__name__

        if operator is not None:
            print(f"🔹 Executing controller {controller_name} with operator {operator_name}")
        else:
            print(f"🔹 Executing controller {controller_name} without operator")

        # Store previous predictions count to detect if new predictions were added
        prev_prediction_count = len(dataset._predictions) if hasattr(dataset, '_predictions') else 0

        # Determine if we need a spinner (for model controllers and other long operations)
        is_model_controller = 'model' in controller_name.lower()
        needs_spinner = is_model_controller or any(keyword in controller_name.lower()
                                                   for keyword in ['transform', 'preprocess', 'augment'])

        # Execute with spinner if needed
        if needs_spinner and self.show_spinner and self.verbose == 0:  # Only show spinner when not verbose
            # Create spinner message
            spinner_message = f"🔄 {controller_name.replace('Controller', '')}"
            if operator_name:
                spinner_message += f" ({operator_name})"

            # Use braille spinner characters for smooth animation
            with spinner_context(spinner_message, chars="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", speed=0.08):
                context, binaries = controller.execute(
                    step,
                    operator,
                    dataset,
                    context,
                    self,
                    source,
                    self.mode,
                    loaded_binaries
                )
        else:
            # Execute without spinner
            context, binaries = controller.execute(
                step,
                operator,
                dataset,
                context,
                self,
                source,
                self.mode,
                loaded_binaries
            )

        # Always show final score for model controllers when verbose=0
        is_model_controller = 'model' in controller_name.lower()

        # Save binaries if in training mode and saving is enabled
        if self.mode == "train" and self.save_binaries and binaries:
            # Track binaries for this step with correct naming
            step_id = f"{self.step_number}_{self.substep_number}"

            # Store the actual filenames that will be saved (with step prefixes)
            actual_filenames = []
            for binary_name, _ in binaries:
                # Construct the actual saved filename (same logic as in io.py)
                prefixed_name = str(self.step_number)
                if self.substep_number > 0:
                    prefixed_name += "_" + str(self.substep_number)
                prefixed_name += "_" + str(binary_name)
                actual_filenames.append(prefixed_name)

            self.step_binaries[step_id] = actual_filenames
            self.saver.save_binaries(self.step_number, self.substep_number, binaries, self.save_binaries)

        return context

        # if controller.use_multi_source():
        #     if not dataset.is_multi_source():
        #         source = 0
        #     else:
        #         source = [i for i in range(dataset.n_sources)]
        #         operator = [operator]
        #         for _ in range(len(source) - len(operator)):
        #             op = deserialize_component(step)
        #             print(f"🔄 Adding operator {op} for additional source")
        #             operator.append(op)

        # if isinstance(operator, list) and self.parallel:
        #     print(f"🔄 Running operators in parallel with {self.max_workers} workers")
        #     with parallel_backend(self.backend, n_jobs=self.max_workers):
        #         Parallel()(delayed(controller.execute)(step, op, dataset, context, self, src) for op, src in zip(operator, source))
        #     return context
        # else:
        #     print(f"🔄 Running single operator {operator} for step: {step}, source: {source}")
            # return controller.execute(step, operator, dataset, context, self, source)

    @staticmethod
    def predict(
        path: Union[str, Path],
        dataset: SpectroDataset,
        verbose: int = 0
    ) -> Tuple[SpectroDataset, Dict[str, Any]]:
        """
        Load a saved pipeline and run it in prediction mode.

        Args:
            path: Path to saved pipeline directory
            dataset: Dataset to make predictions on
            verbose: Verbosity level

        Returns:
            Tuple of (updated_dataset, final_context) with predictions stored in dataset

        Raises:
            FileNotFoundError: If pipeline directory or required files don't exist
            ValueError: If pipeline configuration is invalid
            RuntimeError: If prediction execution fails
        """
        path = Path(path)

        # Validate pipeline path
        if not path.exists():
            raise FileNotFoundError(f"Pipeline directory does not exist: {path}")

        pipeline_json_path = path / "pipeline.json"
        if not pipeline_json_path.exists():
            raise FileNotFoundError(f"Pipeline configuration not found: {pipeline_json_path}")

        # Load pipeline configuration
        try:
            with open(pipeline_json_path, 'r') as f:
                pipeline_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise ValueError(f"Failed to load pipeline configuration: {e}")

        # Check for binary metadata
        if "execution_metadata" not in pipeline_data:
            warnings.warn(
                f"Pipeline at {path} was saved without binary metadata. "
                "This pipeline may not work properly in prediction mode. "
                "Consider re-running the pipeline with save_binaries=True to enable full prediction support.",
                UserWarning
            )

        # Extract steps - handle both old and new format
        if "steps" in pipeline_data:
            steps = pipeline_data["steps"]
        else:
            # Old format - pipeline_data is the steps directly
            steps = pipeline_data

        # Create binary loader
        binary_loader = BinaryLoader(path)

        # Create prediction runner
        runner = PipelineRunner(
            verbose=verbose,
            save_binaries=False,  # Don't save binaries during prediction
            mode="predict"
        )
        runner.binary_loader = binary_loader

        # Create config and run pipeline
        config = PipelineConfigs(steps)
        config.name = f"prediction_{dataset.name}"

        if verbose > 0:
            print(f"🔮 Starting prediction mode for pipeline on dataset {dataset.name}")
            cache_info = binary_loader.get_cache_info()
            print(f"📦 Available binaries for {cache_info['total_available_binaries']} operations across {len(cache_info['available_steps'])} steps")

        try:
            result_dataset, history, pipeline = runner.run(config, dataset)

            # Extract final context
            final_context = {"processing": [["prediction"]] * dataset.features_sources(), "y": "prediction"}

            if verbose > 0:
                print(f"✅ Prediction completed successfully")

            return result_dataset, final_context

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}") from e
