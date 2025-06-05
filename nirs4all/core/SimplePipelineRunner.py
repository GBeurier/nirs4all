"""
SimplePipelineRunner - Simplified execution engine with basic branch context

Features:
- Simple dict-based context with branch info
- Direct operation execution
- No complex scoping or data selectors
"""
import json
from typing import Any, Dict, List, Union, Tuple

from SpectraDataset import SpectraDataset
from PipelineBuilder import PipelineBuilder
from PipelineHistory import PipelineHistory


class SimplePipelineRunner:
    def __init__(self, continue_on_error: bool = False, verbose: int = 0):
        self.continue_on_error = continue_on_error
        self.verbose = verbose

        self.builder = PipelineBuilder()
        self.history = PipelineHistory()
        self.current_step = 0

        # Simplified context - just a dict with current branch
        self.context = {"branch": 0}

    def run(self, config: Union[Dict, str, List], dataset: SpectraDataset) -> Tuple[SpectraDataset, Any, PipelineHistory]:
        print("🚀 Starting Simple Pipeline Runner")

        # Reset pipeline state
        if self.current_step > 0:
            print("  ⚠️ Warning: Previous run detected, resetting step count")
            self.current_step = 0
            self.context = {"branch": 0}

        # Handle different config types
        if isinstance(config, str):
            if config.endswith('.json'):
                with open(config, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            elif config.endswith('.yaml') or config.endswith('.yml'):
                import yaml
                with open(config, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            else:
                raise ValueError(f"Cannot handle config file: {config}")

        # Extract pipeline steps
        if isinstance(config, dict):
            steps = config.get("pipeline", [])
        elif isinstance(config, list):
            steps = config
        else:
            raise ValueError("Pipeline configuration must be a dict with 'pipeline' key or a list of steps")

        if not isinstance(steps, list):
            raise ValueError("Pipeline steps must be a list")

        # Start pipeline execution tracking
        self.history.start_execution({"pipeline": steps})

        try:
            for step in steps:
                self._run_step(step, dataset)

            # Complete pipeline execution
            self.history.complete_execution()
            print("✅ Pipeline completed successfully")

        except Exception as e:
            self.history.fail_execution(str(e))
            print(f"❌ Pipeline failed: {str(e)}")
            raise

        return dataset, None, self.history

    def _run_step(self, step: Any, dataset: SpectraDataset, prefix: str = ""):
        """Run a single pipeline step with simplified context"""
        self.current_step += 1
        step_description = self._get_step_description(step)
        print(f"{prefix}🔹 Step {self.current_step}: {step_description}")
        print(f"{prefix}🔹 Current context: {self.context}")

        # Start step tracking
        step_execution = self.history.start_step(
            step_number=self.current_step,
            step_description=step_description,
            step_config=step
        )

        try:
            # Handle different step types
            if isinstance(step, dict):
                # Check for branch control
                if "branch" in step:
                    self._run_branch(step["branch"], dataset, prefix + "  ")
                else:
                    # Direct operation
                    operation = self.builder.build_operation(step)
                    self._execute_operation(operation, dataset, prefix + "  ")

            elif isinstance(step, list):
                # Sub-pipeline
                print(f"{prefix}  📁 Sub-pipeline with {len(step)} steps")
                for sub_step in step:
                    self._run_step(sub_step, dataset, prefix + "    ")

            elif isinstance(step, str):
                # String preset or operation
                operation = self.builder.build_operation(step)
                self._execute_operation(operation, dataset, prefix + "  ")

            else:
                # Direct operation object
                operation = self.builder.build_operation(step)
                self._execute_operation(operation, dataset, prefix + "  ")

            # Complete step successfully
            self.history.complete_step(step_execution.step_id)

        except Exception as e:
            # Fail step
            self.history.fail_step(step_execution.step_id, str(e))

            if self.continue_on_error:
                print(f"{prefix}  ⚠️ Step failed but continuing: {str(e)}")
            else:
                print(f"{prefix}  ❌ Step failed: {str(e)}")
                raise

    def _execute_operation(self, operation: Any, dataset: SpectraDataset, prefix: str):
        """Execute a single operation with simplified context"""
        print(f"{prefix}🔧 Executing operation: {operation.get_name()}")

        # Execute the operation with simple context
        operation.execute(dataset, self.context)

        print(f"{prefix}✅ Operation completed: {operation.get_name()}")

    def _run_branch(self, branch_config: Any, dataset: SpectraDataset, prefix: str):
        """Handle branching - simple implementation"""
        print(f"{prefix}🌿 Branching operation")

        if isinstance(branch_config, list):
            # Multiple branches
            for i, branch_steps in enumerate(branch_config):
                print(f"{prefix}  🌱 Branch {i}")
                # Save current branch
                old_branch = self.context["branch"]
                # Set new branch
                self.context["branch"] = i

                try:
                    # Execute branch steps
                    if isinstance(branch_steps, list):
                        for step in branch_steps:
                            self._run_step(step, dataset, prefix + "    ")
                    else:
                        self._run_step(branch_steps, dataset, prefix + "    ")
                finally:
                    # Restore branch
                    self.context["branch"] = old_branch

        else:
            # Single branch
            print(f"{prefix}  🌱 Single branch")
            self._run_step(branch_config, dataset, prefix + "    ")

    def _get_step_description(self, step: Any) -> str:
        """Get a human-readable description of a step"""
        if isinstance(step, dict):
            if "branch" in step:
                return "Branch operation"
            elif hasattr(step, 'get') and "class" in step:
                return f"Operation: {step['class']}"
            else:
                return f"Dict operation: {list(step.keys())}"
        elif isinstance(step, list):
            return f"Sub-pipeline ({len(step)} steps)"
        elif isinstance(step, str):
            return f"String operation: {step}"
        else:
            return f"Object: {type(step).__name__}"
