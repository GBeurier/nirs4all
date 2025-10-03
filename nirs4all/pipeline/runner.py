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
import os

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
from nirs4all.dataset.prediction_helpers import PredictionHelpers
from nirs4all.utils.spinner import spinner_context
from nirs4all.utils.tab_report_manager import TabReportManager



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
                 save_files: bool = True,
                 mode: str = "train",
                 load_existing_predictions: bool = True,
                 show_spinner: bool = True,
                 enable_tab_reports: bool = True):

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
        self.save_files = save_files
        self.mode = mode
        self.load_existing_predictions = load_existing_predictions
        self.step_binaries: Dict[str, List[str]] = {}  # Track step-to-binary mapping
        self.binary_loader: Optional[BinaryLoader] = None
        self.show_spinner = show_spinner
        self.enable_tab_reports = enable_tab_reports
        self.prediction_metadata: Optional[Dict[str, Any]] = None

    def run(self, pipeline_configs: PipelineConfigs, dataset_configs: DatasetConfigs) -> Any:
        """Run pipeline configurations on dataset configurations."""

        nb_combinations = len(pipeline_configs.steps) * len(dataset_configs.configs)
        print(f"🚀 Starting pipeline run with {len(pipeline_configs.steps)} pipeline configuration(s) on {len(dataset_configs.configs)} dataset configuration(s) ({nb_combinations} total runs).")

        datasets_predictions = {}
        run_predictions = Predictions()

        # Get datasets from DatasetConfigs
        for config, name in dataset_configs.configs:
            print("=" * 120)

            dataset_prediction_path = self.saver.base_path / name / "predictions.json"
            global_dataset_predictions = Predictions.load_from_file_cls(dataset_prediction_path)
            run_dataset_predictions = Predictions()

            for i, (steps, config_name) in enumerate(zip(pipeline_configs.steps, pipeline_configs.names)):
                dataset = dataset_configs.get_dataset(config, name)
                dataset_name = name

                if self.verbose > 0:
                    print(dataset)

                config_predictions = Predictions()
                self._run_single(steps, config_name, dataset, config_predictions)

                # Merge new predictions into stores
                if config_predictions.num_predictions > 0:
                    global_dataset_predictions.merge_predictions(config_predictions)
                    run_dataset_predictions.merge_predictions(config_predictions)
                    run_predictions.merge_predictions(config_predictions)

            ### Print best results for this dataset
            if run_dataset_predictions.num_predictions > 0:
                best = run_dataset_predictions.get_best()
                print(f"🏆 Run best for dataset '{name}': {Predictions.pred_long_string(best)}")
                best_by_partition = run_dataset_predictions.get_entry_partitions(best)
                str_desc, csv_file = TabReportManager.generate_best_score_tab_report(best_by_partition)
                print(str_desc)
                if csv_file:
                    filename = f"{best['step_idx']}_{best['model_name']}_{best['op_counter']}.csv"
                    print(filename)
                    self.saver.save_file(filename, csv_file, into_dataset=True)
                print("-" * 120)

            if global_dataset_predictions.num_predictions > 0:
                global_dataset_predictions.save_to_file(dataset_prediction_path)
                best_overall = global_dataset_predictions.get_best()
                print(f"🏆 Best Overall for dataset '{name}': {Predictions.pred_long_string(best_overall)}")
                overall_best_by_partition = global_dataset_predictions.get_entry_partitions(best_overall)
                str_desc, csv_file = TabReportManager.generate_best_score_tab_report(overall_best_by_partition)
                print(str_desc)
                if csv_file:
                    filename = f"Best_{best_overall['step_idx']}_{best_overall['model_name']}_{best_overall['op_counter']}.csv"
                    print(filename)
                    self.saver.save_file(filename, csv_file)

            print("=" * 120)
            print("=" * 120)

            # if self.enable_tab_reports:
                # PredictionHelpers.generate_best_score_tab_report(global_dataset_predictions, dataset_name, str(self.saver.base_path / dataset_name), True, dataset)

            # Generate best score tab report
            datasets_predictions[dataset_name] = {
                "global_predictions": global_dataset_predictions,
                "run_predictions": run_dataset_predictions,
                "dataset": dataset,
                "dataset_name": dataset_name
            }

        return run_predictions, datasets_predictions







    @staticmethod
    def predict_from_pred(prediction_obj: Dict[str, Any], dataset_config: DatasetConfigs, verbose: int = 0, output_path: Optional[str] = None) -> Dict[str, Any]:

        # 1. Extract paths and load pipeline configuration
        config_path = prediction_obj['config_path']
        pipeline_config_file = Path(f"results/{config_path}/pipeline.json")
        metadata_file = Path(f"results/{config_path}/metadata.json")

        # 2. Load pipeline steps and metadata
        pipeline_steps = json.load(open(pipeline_config_file))
        metadata = json.load(open(metadata_file))
        pipeline_steps = json.load(open(pipeline_config_file))
        ################ GET BINARIES LINKS FOR THE STEPS ################

        # 3. Create prediction runner with binary resolution capability
        runner = PipelineRunner(mode="predict", verbose=verbose, save_files=False)
        runner.prediction_metadata = metadata
        runner.target_model_info = prediction_obj  # For model-specific execution
        # runner.config_path = config_path

        # 4. Load dataset and execute pipeline
        # for config, name in dataset_configs.configs:
        #     print("*" * 120)

        #     for i, (steps, config_name) in enumerate(zip(pipeline_configs.steps, pipeline_configs.names)):
        #         dataset = dataset_configs.get_dataset(config, name)
        #         dataset_name = name


        dataset = dataset_config.get_dataset_at(0)
        prediction_store = Predictions()

        # 5. Run pipeline in prediction mode
        final_dataset = runner._run_single(pipeline_steps, "prediction", dataset, prediction_store)

        # 6. Extract predictions from prediction store
        # return runner._extract_prediction_results(prediction_store, prediction_obj)



    def _run_single(self, steps: List[Any], config_name: str, dataset: SpectroDataset, config_predictions: 'Predictions') -> SpectroDataset:
        """Run a single pipeline configuration on a single dataset with external prediction store."""
        # Reset runner state for each run
        # self.history = PipelineHistory()
        self.step_number = 0
        self.substep_number = -1
        self.operation_count = 0
        self.step_binaries = {}

        print("=" * 120)
        print(f"\033[94m🚀 Starting pipeline {config_name} on dataset {dataset.name}\033[0m")
        print("-" * 120)

        self.saver.register(dataset.name, config_name)
        self.saver.save_json("pipeline.json", PipelineConfigs.serializable_steps(steps))

        # Initialize context
        context = {"processing": [["raw"]] * dataset.features_sources(), "y": "numeric"}

        try:
            self.run_steps(steps, dataset, context, execution="sequential", prediction_store=config_predictions)
            self.saver.save_json("pipeline.json", PipelineConfigs.serializable_steps(steps))

            if config_predictions.num_predictions > 0:
                pipeline_best = config_predictions.get_best()
                print(f"🥇 Pipeline Best: {Predictions.pred_short_string(pipeline_best)}")
                print(f"\033[94m🏁 Pipeline {config_name} completed successfully on dataset {dataset.name}\033[0m")
                print("=" * 120)

        except Exception as e:
            print(f"\033[91m❌ Pipeline {config_name} on dataset {dataset.name} failed: \n{str(e)}\033[0m")
            import traceback
            traceback.print_exc()
            raise

        return dataset

    def run_steps(self, steps: List[Any], dataset: SpectroDataset, context: Union[List[Dict[str, Any]], Dict[str, Any]], execution: str = "sequential", prediction_store: Optional['Predictions'] = None, is_substep: bool = False) -> Dict[str, Any]:
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
                    context = self.run_step(step, dataset, context, prediction_store, is_substep=is_substep)
                    # print(f"🔹 Updated context after step: {context}")
                self.substep_number = -1  # Reset sub-step number after sequential execution
                return context

        elif execution == "parallel" and self.parallel:
            # print(f"🔄 Running steps in parallel with {self.max_workers} workers")
            with parallel_backend(self.backend, n_jobs=self.max_workers):
                Parallel()(delayed(self.run_step)(step, dataset, context, prediction_store) for step, context in zip(steps, context))

    def run_step(self, step: Any, dataset: SpectroDataset, context: Dict[str, Any], prediction_store: Optional['Predictions'] = None, *, is_substep: bool = False):
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
            self.operation_count = 0
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
                    # print(f"🔄 Skipping step {self.step_number} in prediction mode")
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
                    controller, step, operator, dataset, context, prediction_store, -1, loaded_binaries
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
                print("-" * 120)
                after_dataset_str = str(dataset)
                # print(before_dataset_str)
                if before_dataset_str != after_dataset_str and self.verbose > 0:
                    print(f"\033[97mUpdate: {after_dataset_str}\033[0m")
                    print("-" * 120)

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
        prediction_store: Optional['Predictions'] = None,
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

        # Prediction counting is now handled by config_predictions externally
        # prev_prediction_count = len(config_predictions) if config_predictions else 0

        # Determine if we need a spinner (for model controllers and other long operations)
        is_model_controller = 'model' in controller_name.lower()
        # needs_spinner = is_model_controller
        needs_spinner = False ####TODO DEBUG spinner \r\n

        # Execute with spinner if needed
        if needs_spinner and self.show_spinner and self.verbose == 0:  # Only show spinner when not verbose
            # Create and print the initial message
            controller_display_name = controller_name.replace('Controller', '')
            initial_message = f"🔄 {controller_name} executes {controller_display_name}"

            # Only show test data shape for model controllers
            if is_model_controller:
                y_test_shape = dataset.y({"partition": "test"}).shape
                initial_message += f" (test: {y_test_shape})"

            if operator_name:
                initial_message += f" ({operator_name})"

            # Use spinner context manager for long operations
            with spinner_context(initial_message):
                context, binaries = controller.execute(
                    step,
                    operator,
                    dataset,
                    context,
                    self,
                    source,
                    self.mode,
                    loaded_binaries,
                    prediction_store
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
                loaded_binaries,
                prediction_store
            )

        # Always show final score for model controllers when verbose=0
        is_model_controller = 'model' in controller_name.lower()
        # print("🔹 Controller execution completed")
        # Save binaries if in training mode and saving is enabled
        if self.mode == "train" and self.save_files and binaries:
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
            self.saver.save_files(self.step_number, self.substep_number, binaries, self.save_files)

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
    def _detect_prediction_source_type(source: Union[str, Path, Dict], verbose: int = 0) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Detect source type and resolve to list of model/pipeline information.

        Returns:
            Tuple of (source_type, resolved_paths) where resolved_paths is a list of
            dictionaries containing 'path', 'pipeline_path', 'model_path', etc.
        """
        if isinstance(source, dict):
            # prediction_model dictionary
            return "prediction_model", [source]

        source_path = Path(source)

        # Check if it's a direct model file
        if source_path.suffix == '.pkl':
            # Check if the exact path exists
            if source_path.is_file():
                return "model_path", [{'model_path': str(source_path)}]
            else:
                # File doesn't exist exactly - try to find it with glob pattern
                # This handles cases where the metadata has the model name but
                # the actual file has a step prefix (e.g., "4_ModelName.pkl")
                parent_dir = source_path.parent
                model_name = source_path.stem  # Without .pkl extension

                if parent_dir.exists():
                    # Try different patterns to find the actual model file
                    patterns = [
                        f"*{model_name}*.pkl",  # With any prefix/suffix
                        f"{model_name}.pkl",    # Exact name
                        f"*{model_name.split('_')[-1]}*.pkl"  # Last part of name
                    ]

                    for pattern in patterns:
                        matching_files = list(parent_dir.glob(pattern))
                        if matching_files:
                            # Return the first match
                            actual_model_path = matching_files[0]
                            return "model_path", [{'model_path': str(actual_model_path)}]

                # If no model file found, but directory exists with pipeline.json, treat as config
                if parent_dir.exists() and (parent_dir / "pipeline.json").exists():
                    return "config_path", [{'config_path': str(parent_dir)}]

        # Check if it's a config directory with pipeline.json
        if source_path.is_dir() and (source_path / "pipeline.json").exists():
            return "config_path", [{'config_path': str(source_path)}]

        # Check if it's a config_id or dataset_name - search results directory
        results_dir = Path("results")
        if results_dir.exists():
            found_paths = PipelineRunner._search_results(str(source), verbose)
            if found_paths:
                if any('config' in str(p['config_path']).lower() for p in found_paths):
                    return "config_id", found_paths
                else:
                    return "dataset_name", found_paths

        # If nothing found, try as config_path anyway
        return "config_path", [{'config_path': str(source)}]

    @staticmethod
    def _search_results(search_term: str, verbose: int = 0) -> List[Dict[str, Any]]:
        """Search results directory for matching configs/datasets."""
        results_dir = Path("results")
        found_configs = []

        if not results_dir.exists():
            return []

        # Search for matching directories
        for dataset_dir in results_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            # Check if dataset name matches
            if search_term.lower() in dataset_dir.name.lower():
                # This is a dataset match - find configs within it
                for config_dir in dataset_dir.iterdir():
                    if config_dir.is_dir() and (config_dir / "pipeline.json").exists():
                        found_configs.append({
                            'config_path': str(config_dir),
                            'dataset_name': dataset_dir.name,
                            'config_name': config_dir.name
                        })
            else:
                # Check for config matches within this dataset
                for config_dir in dataset_dir.iterdir():
                    if (config_dir.is_dir() and
                        search_term.lower() in config_dir.name.lower() and
                        (config_dir / "pipeline.json").exists()):
                        found_configs.append({
                            'config_path': str(config_dir),
                            'dataset_name': dataset_dir.name,
                            'config_name': config_dir.name
                        })

        return found_configs



    def next_op(self) -> int:
        """Get the next operation ID."""
        self.operation_count += 1
        return self.operation_count


    def _resolve_binaries_for_step(self, step_number: int) -> List[Tuple[str, Any]]:
        """
        Resolve and load binary files for a specific step using enhanced metadata.

        Uses step-to-binary mapping stored during training to eliminate counter guessing.
        """
        step_key = str(step_number)

        # Get binary filenames for this step from enhanced metadata
        binary_filenames = self.prediction_metadata.get('step_binaries', {}).get(step_key, [])

        if not binary_filenames:
            return []  # No binaries for this step

        # Load all binaries for this step
        loaded_binaries = []
        for filename in binary_filenames:
            binary_path = Path(f"results/{self.config_path}/{filename}")
            if binary_path.exists():
                with open(binary_path, 'rb') as f:
                    loaded_obj = pickle.load(f)
                loaded_binaries.append((filename, loaded_obj))
            else:
                print(f"⚠️ Binary file not found: {filename}")

        return loaded_binaries

    def _resolve_target_model_binary(self) -> Optional[Tuple[str, Any]]:
        """
        Load specific target model binary for model-specific prediction (Q4 use case).
        """
        if not hasattr(self, 'target_model_info'):
            return None

        model_path = self.target_model_info.get('model_path', '')
        if model_path and Path(model_path).exists():
            with open(model_path, 'rb') as f:
                loaded_obj = pickle.load(f)
            return (Path(model_path).name, loaded_obj)

        return None





#  @staticmethod
#     def _predict_single_model(path_info: Dict[str, Any], dataset: DatasetConfigs, verbose: int = 0) -> Dict[str, Any]:
#         """Load and run prediction for a single model/config."""

#         # Check if this is a virtual model (avg/w-avg)
#         if 'metadata' in path_info:
#             metadata = path_info['metadata']
#             if metadata.get('is_virtual_model', False):
#                 return PipelineRunner._predict_virtual_model(path_info, dataset, verbose)

#         # Handle different path types for real models
#         if 'model_path' in path_info:
#             # Direct model file - need to find associated pipeline
#             model_path = Path(path_info['model_path'])

#             # Check if model_path is actually a file path
#             if model_path.suffix == '.pkl':
#                 config_dir = model_path.parent
#             else:
#                 # If model_path doesn't end with .pkl, treat it as config directory
#                 config_dir = model_path

#             pipeline_json = config_dir / "pipeline.json"

#             if not pipeline_json.exists():
#                 raise FileNotFoundError(f"Pipeline configuration not found: {pipeline_json}")

#         elif 'config_path' in path_info:
#             # Config directory
#             config_dir = Path(path_info['config_path'])
#             pipeline_json = config_dir / "pipeline.json"

#         elif 'pipeline_path' in path_info:
#             # From prediction_model dict
#             config_dir = Path(path_info['pipeline_path'])
#             pipeline_json = config_dir / "pipeline.json"

#         else:
#             raise ValueError(f"Unknown path info format: {path_info}")

#         # Load pipeline configuration
#         try:
#             with open(pipeline_json, 'r') as f:
#                 pipeline_data = json.load(f)
#         except (json.JSONDecodeError, FileNotFoundError) as e:
#             raise ValueError(f"Failed to load pipeline configuration: {e}")

#         # Extract steps
#         if "steps" in pipeline_data:
#             steps = pipeline_data["steps"]
#         else:
#             steps = pipeline_data

#         # Create binary loader
#         binary_loader = BinaryLoader(config_dir)

#         # Create prediction runner
#         runner = PipelineRunner(
#             verbose=verbose,
#             save_files=False,
#             mode="predict"
#         )
#         runner.binary_loader = binary_loader

#         # Create dataset instance
#         dataset_instance = dataset.configs[0][0] if dataset.configs else None
#         if dataset_instance is None:
#             raise ValueError("No dataset configuration provided")

#         # Create config and run pipeline
#         config = PipelineConfigs(steps)

#         if verbose > 0:
#             cache_info = binary_loader.get_cache_info()
#             print(f"📦 Available binaries for {cache_info['total_available_binaries']} operations across {len(cache_info['available_steps'])} steps")

#         try:
#             predictions, results = runner.run(config, dataset)
#             return {
#                 'predictions': predictions,
#                 'results': results,
#                 'config_path': str(config_dir),
#                 'path_info': path_info
#             }

#         except Exception as e:
#             if verbose > 0:
#                 print(f"⚠️ Prediction failed for {config_dir}: {e}")
#             return None

#     @staticmethod
#     def _predict_virtual_model(path_info: Dict[str, Any], dataset: DatasetConfigs, verbose: int = 0) -> Dict[str, Any]:
#         """Handle prediction for virtual models (avg/w-avg) by loading constituent models."""

#         metadata = path_info['metadata']
#         virtual_type = metadata.get('virtual_type', 'unknown')
#         averaging_method = metadata.get('averaging_method', 'equal')
#         constituent_models = metadata.get('constituent_models', [])
#         weights = metadata.get('weights', [])

#         if verbose > 0:
#             print(f"🔮 Processing virtual model: {virtual_type} with {len(constituent_models)} constituent models")
#             print(f"📊 Averaging method: {averaging_method}")

#         if not constituent_models:
#             raise ValueError("Virtual model has no constituent models defined")

#         # Get config directory from path_info
#         if 'config_path' in path_info:
#             config_dir = Path(path_info['config_path'])
#         elif 'path' in path_info:
#             # This is the field from prediction_model dictionary
#             config_dir = Path(path_info['path'])
#         elif 'pipeline_path' in path_info:
#             config_dir = Path(path_info['pipeline_path'])
#         elif 'config_path' in metadata:
#             config_dir = Path(metadata['config_path'])
#         elif 'pipeline_path' in metadata:
#             config_dir = Path(metadata['pipeline_path'])
#         else:
#             # Try to extract from model_path or other metadata
#             model_path = path_info.get('model_path', '') or metadata.get('model_path', '')
#             if model_path:
#                 config_dir = Path(model_path).parent if model_path else Path('.')
#             else:
#                 config_dir = Path('.')

#         constituent_predictions = []

#         # Load each constituent model and predict
#         for i, model_uuid in enumerate(constituent_models):
#             # Find the actual model file using glob pattern to handle step prefixes
#             model_file_patterns = [
#                 f"*{model_uuid}*.pkl",  # First try with full UUID
#                 f"*{model_uuid.split('_')[0]}*.pkl"  # Then try with just the model name part
#             ]

#             model_files = []
#             for pattern in model_file_patterns:
#                 model_files = list(config_dir.glob(pattern))
#                 if model_files:
#                     break

#             if not model_files:
#                 if verbose > 0:
#                     print(f"⚠️ Model file not found for {model_uuid} in {config_dir}")
#                     print(f"    Tried patterns: {model_file_patterns}")
#                 continue

#             model_path = model_files[0]  # Take first match

#             # Create path_info for this constituent model
#             constituent_path_info = {
#                 'model_path': str(model_path),
#                 'config_path': str(config_dir)
#             }

#             if verbose > 0:
#                 print(f"🔄 Loading constituent model {i+1}/{len(constituent_models)}: {model_path.name}")

#             # Predict with this constituent model
#             try:
#                 pred_result = PipelineRunner._predict_single_model(constituent_path_info, dataset, 0)  # Reduce verbosity for constituents
#                 if pred_result:
#                     constituent_predictions.append(pred_result)
#             except Exception as e:
#                 if verbose > 0:
#                     print(f"⚠️ Failed to predict with constituent model {model_uuid}: {e}")
#                 continue

#         if not constituent_predictions:
#             raise RuntimeError("No constituent models could be loaded for virtual model prediction")

#         # Combine predictions using the saved weights
#         if averaging_method == 'weighted' and weights:
#             # Use the original weights saved during training
#             if len(weights) != len(constituent_predictions):
#                 if verbose > 0:
#                     print(f"⚠️ Weight mismatch: {len(weights)} weights vs {len(constituent_predictions)} predictions, using equal weights")
#                 weights = [1.0 / len(constituent_predictions)] * len(constituent_predictions)
#         else:
#             # Equal weighting
#             weights = [1.0 / len(constituent_predictions)] * len(constituent_predictions)

#         # Combine the predictions
#         combined_result = PipelineRunner._combine_predictions_with_weights(
#             constituent_predictions, weights, verbose
#         )

#         # Add virtual model metadata to result
#         combined_result['virtual_model_info'] = {
#             'virtual_type': virtual_type,
#             'averaging_method': averaging_method,
#             'num_constituents': len(constituent_predictions),
#             'weights': weights
#         }

#         if verbose > 0:
#             print(f"✅ Virtual model prediction completed using {len(constituent_predictions)} models")

#         return combined_result

#     @staticmethod
#     def _combine_predictions_with_weights(all_predictions: List[Dict], weights: List[float], verbose: int = 0) -> Dict[str, Any]:
#         """Combine predictions from multiple models using specified weights."""
#         if verbose > 0:
#             print(f"🔗 Combining {len(all_predictions)} predictions with weights: {[f'{w:.3f}' for w in weights]}")

#         # For now, return the first prediction (basic implementation)
#         # In a full implementation, you would:
#         # 1. Extract prediction arrays from each result
#         # 2. Compute weighted average: sum(weight_i * prediction_i)
#         # 3. Combine metadata appropriately

#         combined_result = all_predictions[0].copy()
#         combined_result['combination_method'] = 'weighted_average'
#         combined_result['constituent_count'] = len(all_predictions)
#         combined_result['weights'] = weights

#         return combined_result

#     @staticmethod
#     def _combine_predictions(all_predictions: List[Dict], ensemble_method: str, verbose: int = 0) -> Dict[str, Any]:
#         """Combine predictions from multiple models."""
#         if ensemble_method == "average":
#             # Simple average of predictions
#             # This is a simplified implementation - in practice you'd want to
#             # properly combine the prediction arrays
#             if verbose > 0:
#                 print(f"🔗 Combining {len(all_predictions)} predictions using {ensemble_method}")

#             # Return the first prediction for now - full ensemble logic would go here
#             return all_predictions[0]

#         elif ensemble_method == "weighted_average":
#             # Weighted average based on model performance
#             # Implementation would go here
#             return all_predictions[0]

#         else:
#             # Default to first prediction
#             return all_predictions[0]