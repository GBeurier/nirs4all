"""
PipelineRunner - Simplified execution engine with branch-based context

Features:
- Simple branch-based context
- Direct operation execution
- Basic branching support
- Simplified data selection
"""
from typing import Any, Dict, List, Optional, Tuple

from joblib import Parallel, delayed, parallel_backend

from .serialization import deserialize_component
from nirs4all.spectra.spectra_dataset import SpectraDataset
from .history import PipelineHistory
from .config import PipelineConfig
from .operation import PipelineOperation
from .pipeline import Pipeline


class PipelineRunner:
    """PipelineRunner - Executes a pipeline with enhanced context management and DatasetView support."""

    WORKFLOW_OPERATORS = ["sample_augmentation", "feature_augmentation", "branch", "dispatch", "model", "stack", "scope", "cluster", "merge", "uncluster", "unscope"]
    SERIALIZATION_OPERATORS = ["class", "function", "module", "object", "pipeline", "instance"]

    def __init__(self, max_workers: Optional[int] = None, continue_on_error: bool = False, backend: str = 'threading', verbose: int = 0):
        self.max_workers = max_workers or -1  # -1 means use all available cores
        self.continue_on_error = continue_on_error
        self.backend = backend
        self.verbose = verbose
        self.history = PipelineHistory()

        # serialization datamodel
        self.pipeline = Pipeline()

    def run(self, config: PipelineConfig, dataset: SpectraDataset) -> Tuple[SpectraDataset, PipelineHistory, Pipeline]:
        """Run the pipeline with the given configuration and dataset."""

        print("🚀 Starting Pipeline Runner")
        context = {"dataset": {"branch": 0}}

        try:
            self.run_steps(config.steps, dataset, context)
            # self.history.complete_execution()
            print("✅ Pipeline completed successfully")

        except Exception as e:
            # self.history.fail_execution(str(e))
            print(f"❌ Pipeline failed: {str(e)}")
            raise

        return dataset, self.history, self.pipeline

    def run_steps(self, steps: List[Any], dataset: SpectraDataset, context: Dict[str, Any], execution: str = "sequential") -> None:
        """Run a list of steps with enhanced context management and DatasetView support."""
        if not isinstance(steps, list):
            steps = [steps]
        print(f"🔄 Running {len(steps)} steps in {execution} mode")

        if execution == "sequential":
            for step in steps:
                self._run_step(step, dataset, context)
        elif execution == "parallel":
            with parallel_backend(self.backend, n_jobs=self.max_workers):
                Parallel()(delayed(self._run_step)(step, dataset, context) for step in steps)

    def _run_step(self, step: Any, dataset: SpectraDataset, context: Dict[str, Any]):
        """
        Run a single pipeline step with enhanced context management and DatasetView support.
        """
        step_description = self._get_step_description(step)
        print(f"🔹 Step {step_description}")
        print(f"🔹 Current context: {context}")
        print(f"🔹 Step config: {step}")

        # Start step tracking
        # step_execution = self.history.start_step(
        #     step_number=self.current_step,
        #     step_description=step_description,
        #     step_config=step
        # )

        try:
            if isinstance(step, dict):
                if key := next((k for k in step if k in self.WORKFLOW_OPERATORS), None):
                    print(f"📋 Workflow operation: {key}")
                    operation = PipelineOperation(step=step, keyword=key)
                    operation.execute(dataset, context, self)
                elif key := next((k for k in step if k in self.SERIALIZATION_OPERATORS), None):
                    print(f"📦 Deserializing operation: {key}")
                    if '_runtime_instance' in step:
                        operator = step['_runtime_instance']
                    else:
                        operator = deserialize_component(step)
                    operation = PipelineOperation(step=step, operator=operator)
                    operation.execute(dataset, context, self)

            elif isinstance(step, list):
                print(f"🔗 Sub-pipeline with {len(step)} steps")
                self.run_steps(step, dataset, context, execution="sequential")

            elif isinstance(step, str):
                if step := next((s for s in step.split() if s in self.WORKFLOW_OPERATORS), None):
                    print(f"📋 Workflow operation: {step}")
                    operation = PipelineOperation(step=step)
                    operation.execute(dataset, context, self)
                else:
                    print(f"📦 Deserializing operation: {step}")
                    operator = deserialize_component(step)
                    operation = PipelineOperation(step=step, operator=operator)
                    operation.execute(dataset, context, self)

            else:
                print(f"🔍 Unknown step type: {type(step).__name__}, executing as operation")
                operation = PipelineOperation(step=step)
                operation.execute(dataset, context, self)

            # self.history.complete_step(step_execution.step_id)

        except (RuntimeError, ValueError, TypeError, ImportError, KeyError, AttributeError, IndexError) as e:
            # Fail step
            # self.history.fail_step(step_execution.step_id, str(e))

            if self.continue_on_error:
                print(f"⚠️ Step failed but continuing: {str(e)}")
            else:
                raise RuntimeError(f"Pipeline step failed: {str(e)}") from e

        finally:
            print("-" * 200)
            print(f"Step completed: {step_description}")
            print("Dataset state after step:")
            print(dataset)
            print("-" * 200)

    def _get_step_description(self, step: Any) -> str:
        """Get a human-readable description of a step"""
        if isinstance(step, dict):
            if len(step) == 1:
                key = next(iter(step.keys()))
                return f"{key}"
            else:
                return f"Dict with {len(step)} keys"
        elif isinstance(step, list):
            return f"Sub-pipeline ({len(step)} steps)"
        elif isinstance(step, str):
            return step
        else:
            return str(type(step).__name__)
