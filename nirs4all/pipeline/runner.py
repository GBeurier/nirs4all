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

# from nirs4all.controllers.controller import OperatorController

from nirs4all.pipeline.serialization import deserialize_component
from nirs4all.pipeline.history import PipelineHistory
from nirs4all.pipeline.config import PipelineConfigs
from nirs4all.pipeline.io import SimulationSaver
from nirs4all.dataset.dataset import SpectroDataset
from nirs4all.dataset.dataset_config import DatasetConfigs
from nirs4all.controllers.registry import CONTROLLER_REGISTRY
from nirs4all.pipeline.binary_loader import BinaryLoader



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
                 mode: str = "train"):

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
        self.step_binaries: Dict[str, List[str]] = {}  # Track step-to-binary mapping
        self.binary_loader: Optional[BinaryLoader] = None

    def next_op(self) -> int:
        """Get the next operation ID."""
        self.operation_count += 1
        return self.operation_count

    def run(self, pipeline_configs: PipelineConfigs, dataset_configs: DatasetConfigs) -> List[Tuple[SpectroDataset, PipelineHistory, Any]]:
        """Run pipeline configurations on dataset configurations."""
        results = []

        # Get datasets from DatasetConfigs
        for d_config in dataset_configs.data_configs:
            print("=" * 200)
            for steps, config_name in zip(pipeline_configs.steps, pipeline_configs.names):
                dataset = dataset_configs.get_dataset(d_config) ## TODO put in upper loop when cache and context will be ready for that
                result = self._run_single(steps, config_name, dataset)
                results.append(result)

        return results

    def _run_single(self, steps: List[Any], config_name: str, dataset: SpectroDataset) -> Tuple[SpectroDataset, PipelineHistory, Any]:
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

            print(f"\033[94m✅ Pipeline {config_name} completed successfully on dataset {dataset.name}\033[0m")

        except Exception as e:
            print(f"\033[91m❌ Pipeline {config_name} on dataset {dataset.name} failed: \n{str(e)}\033[0m")
            import traceback
            traceback.print_exc()
            raise

        return dataset, self.history, None  # TODO remove None and return the actual pipeline object

    def run_steps(self, steps: List[Any], dataset: SpectroDataset, context: Union[List[Dict[str, Any]], Dict[str, Any]], execution: str = "sequential", is_substep: bool = False) -> Dict[str, Any]: ##TODO distinguish parallel and sequential contexts from parrallel and sequential execution
        """Run a list of steps with enhanced context management and DatasetView support."""
        if not isinstance(steps, list):
            steps = [steps]
        print(f"\033[94m🔄 Running {len(steps)} steps in {execution} mode\033[0m")

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
        if is_substep or self.substep_number > 0:
            self.substep_number += 1
            print(f"\033[96m   ▶ Sub-step {self.step_number}.{self.substep_number}: {step_description}\033[0m")
        else:
            self.substep_number = 0
            self.step_number += 1
        # print(f"🔷 Step {self.step_number}: {step_description}")
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
