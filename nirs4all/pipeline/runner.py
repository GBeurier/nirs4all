"""
PipelineRunner - Simplified execution engine with branch-based context

Features:
- Simple branch-based context
- Direct operation execution
- Basic branching support
- Simplified data selection
"""
from typing import Any, Dict, List, Optional, Tuple, Union

from joblib import Parallel, delayed, parallel_backend

# from nirs4all.controllers.controller import OperatorController

from nirs4all.pipeline.serialization import deserialize_component
from nirs4all.pipeline.history import PipelineHistory
from nirs4all.pipeline.config import PipelineConfig
# from .operation import PipelineOperation
from nirs4all.pipeline.pipeline import Pipeline
from nirs4all.pipeline.io import SimulationSaver
from nirs4all.dataset.dataset import SpectroDataset
from nirs4all.controllers.registry import CONTROLLER_REGISTRY



class PipelineRunner:
    """PipelineRunner - Executes a pipeline with enhanced context management and DatasetView support."""

    WORKFLOW_OPERATORS = ["sample_augmentation", "feature_augmentation", "branch", "dispatch", "model", "stack", "scope", "cluster", "merge", "uncluster", "unscope", "chart_2d", "chart_3d", "fold_chart"]
    SERIALIZATION_OPERATORS = ["class", "function", "module", "object", "pipeline", "instance"]

    def __init__(self, ##TODO add resume / overwrite support / realtime viz
                 max_workers: Optional[int] = None,
                 continue_on_error: bool = False,
                 backend: str = 'threading',
                 verbose: int = 0,
                 parallel: bool = False,
                 results_path: Optional[str] = None):

        self.max_workers = max_workers or -1  # -1 means use all available cores
        self.continue_on_error = continue_on_error
        self.backend = backend
        self.verbose = verbose
        self.history = PipelineHistory()
        self.pipeline = Pipeline()
        self.parallel = parallel
        self.step_number = 0  # Initialize step number for tracking
        self.substep_number = -1  # Initialize sub-step number for tracking
        self.saver = SimulationSaver(results_path)
        self.operation_count = 0

    def next_op(self) -> int:
        """Get the next operation ID."""
        self.operation_count += 1
        return self.operation_count

    def run(self, config: PipelineConfig, dataset: SpectroDataset) -> Tuple[SpectroDataset, PipelineHistory, Pipeline]:
        """Run the pipeline with the given configuration and dataset."""
        print("=" * 200)
        print(f"\033[94m🚀 Starting pipeline {config.name} on dataset {dataset.name}\033[0m")
        print("-" * 200)
        self.saver.register(dataset.name, config.name)
        self.saver.save_json("pipeline.json", config.serializable_steps())
        # context = {"branch": 0, "processing": "raw", "y": "numeric"}
        context = {"processing": [["raw"]] * dataset.features_sources()}

        try:
            self.run_steps(config.steps, dataset, context, execution="sequential")
            # self.history.complete_execution()
            print(f"\033[94m✅ Pipeline {config.name} completed successfully on dataset {dataset.name}\033[0m")

        except Exception as e:
            # self.history.fail_execution(str(e))
            print(f"\033[91m❌ Pipeline {config.name} on dataset {dataset.name} failed: \n{str(e)}\033[0m")
            raise

        return dataset, self.history, self.pipeline

    def run_steps(self, steps: List[Any], dataset: SpectroDataset, context: Union[List[Dict[str, Any]], Dict[str, Any]], execution: str = "sequential", is_substep: bool = False) -> Dict[str, Any]:
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

        step_description = self._get_step_description(step)
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
                # print(f"🔄 Selected controller: {controller.__class__.__name__}")
                context["step_id"] = self.step_number
                return self._execute_controller(controller, step, operator, dataset, context)


            # self.history.complete_step(step_execution.step_id)

        except (RuntimeError, ValueError, TypeError, ImportError, KeyError, AttributeError, IndexError) as e:
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
                after_dataset_str = str(dataset)
                # print(before_dataset_str)
                if before_dataset_str != after_dataset_str:
                    print(f"\033[97mUpdate: {after_dataset_str}\033[0m")
                    print("-" * 200)

    def _select_controller(self, step: Any, operator: Any = None, keyword: str = ""):
        matches = [cls for cls in CONTROLLER_REGISTRY if cls.matches(step, operator, keyword)]
        if not matches:
            raise TypeError(f"No matching controller found for {step}. Available controllers: {[cls.__name__ for cls in CONTROLLER_REGISTRY]}")
        matches.sort(key=lambda c: c.priority)
        return matches[0]()

    def _execute_controller( ## TODO Choose one option for multi-source datasets and parrallel execution
        self,
        controller: Any,
        step: Any,
        operator: Any,
        dataset: SpectroDataset,
        context: Dict[str, Any],
        source: Union[int, List[int]] = -1
    ):
        """Execute the controller for the given step and operator."""
        operator_name = operator.__class__.__name__ if operator else ""
        controller_name = controller.__class__.__name__


        if operator:
            print(f"🔹 Executing controller {controller_name} with operator {operator_name}")
        else:
            print(f"🔹 Executing controller {controller_name} without operator")

        context, binaries = controller.execute(
            step,
            operator,
            dataset,
            context,
            self,
            source
        )

        self.saver.save_binaries(self.step_number, self.substep_number, binaries)
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

    # Helper method to get a human-readable description of a step
    def _get_step_description(self, step: Any) -> str:
        """Get a human-readable description of a step"""
        if step is None:
            return "No operation"
        if isinstance(step, dict):
            if len(step) == 1:
                key = next(iter(step.keys()))
                return f"{key}"
            elif "class" in step:
                key = f"{step['class'].split('.')[-1]}"
                if "params" in step:
                    params_str = ", ".join(f"{k}={v}" for k, v in step["params"].items())
                    return f"{key}({params_str})"
            else:
                return f"Dict with {len(step)} keys"
        elif isinstance(step, list):
            return f"Sub-pipeline ({len(step)} steps)"
        elif isinstance(step, str):
            return step
        else:
            return str(type(step).__name__)
