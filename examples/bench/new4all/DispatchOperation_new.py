"""
DispatchOperation - Parallel and branched execution of pipeline operations
"""
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import copy
from PipelineOperation import PipelineOperation
from SpectraDataset import SpectraDataset
from PipelineContext import PipelineContext


class DispatchOperation(PipelineOperation):
    """Operation for parallel or branched execution of operations"""

    def __init__(self,
                 operations: List[Any],
                 dispatch_strategy: str = "parallel",
                 max_workers: Optional[int] = None,
                 merge_results: bool = True,
                 execution_mode: str = "thread"):
        """
        Initialize dispatch operation

        Parameters:
        -----------
        operations : List[Any]
            List of operations or sub-pipelines to execute
            Can contain:
            - Single PipelineOperation objects
            - Lists of PipelineOperation objects (sub-pipelines)
        dispatch_strategy : str
            Execution strategy: "parallel", "sequential", "conditional"
        max_workers : Optional[int]
            Maximum number of parallel workers
        merge_results : bool
            Whether to merge results from parallel operations
        execution_mode : str
            Execution mode: "thread", "process"
        """
        super().__init__()
        self.operations = operations
        self.dispatch_strategy = dispatch_strategy
        self.max_workers = max_workers
        self.merge_results = merge_results
        self.execution_mode = execution_mode

        # Results storage
        self.operation_results = []
        self.execution_summary = {}

    def execute(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Execute dispatch operation"""
        if not self.can_execute(dataset, context):
            raise ValueError("Cannot execute dispatch - no operations available")

        if self.dispatch_strategy == "parallel":
            self.execute_parallel(dataset, context)
        elif self.dispatch_strategy == "sequential":
            self.execute_sequential(dataset, context)
        elif self.dispatch_strategy == "conditional":
            self.execute_conditional(dataset, context)
        else:
            raise ValueError(f"Unknown dispatch strategy: {self.dispatch_strategy}")

        # Store execution summary in context (if supported)
        if hasattr(context, 'dispatch_results'):
            context.dispatch_results = {
                'strategy': self.dispatch_strategy,
                'n_operations': len(self.operations),
                'results': self.operation_results,
                'summary': self.execution_summary
            }

        print(f"Dispatch operation completed: {len(self.operations)} operations using {self.dispatch_strategy} strategy")

    def can_execute(self, dataset: SpectraDataset, context: PipelineContext) -> bool:
        """Check if dispatch can be executed"""
        return len(self.operations) > 0

    def get_name(self) -> str:
        """Get operation name"""
        return f"DispatchOperation({self.dispatch_strategy})"

    def execute_parallel(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Execute operations in parallel"""
        if self.execution_mode == "thread":
            executor_class = ThreadPoolExecutor
        elif self.execution_mode == "process":
            executor_class = ProcessPoolExecutor
        else:
            raise ValueError(f"Unknown execution mode: {self.execution_mode}")

        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all branches
            future_to_branch = {}

            for i, branch in enumerate(self.operations):
                # Create copies of dataset and context for each branch
                dataset_copy = self.copy_dataset(dataset)
                context_copy = self.copy_context(context)
                # Add branch identifier if context supports it
                if hasattr(context_copy, '__dict__'):
                    context_copy.branch_id = i

                if self.execution_mode == "thread":
                    future = executor.submit(self.execute_single_branch,
                                           branch, dataset_copy, context_copy, i)
                else:
                    future = executor.submit(self.execute_single_branch_safe,
                                           branch, dataset_copy, context_copy, i)

                future_to_branch[future] = (branch, i)

            # Collect results
            self.operation_results = [None] * len(self.operations)

            for future in as_completed(future_to_branch):
                branch, branch_index = future_to_branch[future]

                try:
                    result = future.result()
                    self.operation_results[branch_index] = result
                    branch_name = f"Branch {branch_index}"
                    if isinstance(branch, list):
                        branch_name += f" ({len(branch)} operations)"
                    else:
                        branch_name += f" (single operation)"
                    print(f"{branch_name} completed successfully")

                except Exception as e:
                    error_result = {
                        'branch_index': branch_index,
                        'branch_type': 'list' if isinstance(branch, list) else 'single',
                        'error': str(e),
                        'success': False
                    }
                    self.operation_results[branch_index] = error_result
                    print(f"Branch {branch_index} failed: {e}")

        # Merge results if requested
        if self.merge_results:
            self.merge_parallel_results(dataset, context)

    def execute_sequential(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Execute operations sequentially"""
        self.operation_results = []

        for i, branch in enumerate(self.operations):
            try:
                # Execute branch (operation or sub-pipeline)
                result = self.execute_single_branch(branch, dataset, context, i)
                self.operation_results.append(result)

                print(f"Branch {i} completed successfully")

            except Exception as e:
                result = {
                    'branch_index': i,
                    'branch_type': 'list' if isinstance(branch, list) else 'single',
                    'error': str(e),
                    'success': False
                }
                self.operation_results.append(result)

                print(f"Branch {i} failed: {e}")

                # Decide whether to continue or stop on error
                if not getattr(context, 'continue_on_error', False):
                    break

    def execute_conditional(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Execute operations based on conditions"""
        self.operation_results = []

        for i, branch in enumerate(self.operations):
            # Check if operation should be executed based on condition
            should_execute = self.check_execution_condition(branch, dataset, context)

            if should_execute:
                try:
                    result = self.execute_single_branch(branch, dataset, context, i)
                    result['executed'] = True
                    self.operation_results.append(result)

                    print(f"Branch {i} executed successfully")

                except Exception as e:
                    result = {
                        'branch_index': i,
                        'branch_type': 'list' if isinstance(branch, list) else 'single',
                        'executed': True,
                        'error': str(e),
                        'success': False
                    }
                    self.operation_results.append(result)

                    print(f"Branch {i} failed: {e}")
            else:
                result = {
                    'branch_index': i,
                    'branch_type': 'list' if isinstance(branch, list) else 'single',
                    'executed': False,
                    'reason': 'Condition not met'
                }
                self.operation_results.append(result)

                print(f"Branch {i} skipped: condition not met")

    def execute_single_branch(self, branch: Any,
                             dataset: SpectraDataset, context: PipelineContext,
                             branch_index: int) -> Dict[str, Any]:
        """Execute a single branch (operation or sub-pipeline) and return result"""
        try:
            if isinstance(branch, list):
                # Execute as sub-pipeline
                return self.execute_sub_pipeline(branch, dataset, context, branch_index)
            else:
                # Execute as single operation
                return self.execute_single_operation(branch, dataset, context, branch_index)

        except Exception as e:
            return {
                'branch_index': branch_index,
                'branch_type': 'list' if isinstance(branch, list) else 'single',
                'error': str(e),
                'success': False
            }

    def execute_sub_pipeline(self, operations: List[Any],
                           dataset: SpectraDataset, context: PipelineContext,
                           branch_index: int) -> Dict[str, Any]:
        """Execute a list of operations as a sub-pipeline"""
        executed_operations = []

        for i, operation in enumerate(operations):
            try:
                # Check if operation can execute (basic check)
                if hasattr(operation, 'can_execute'):
                    if callable(getattr(operation, 'can_execute', None)):
                        if not operation.can_execute(dataset, context):
                            print(f"Branch {branch_index} - Operation {i} skipped: cannot execute")
                            continue

                # Execute operation
                operation.execute(dataset, context)

                executed_operations.append({
                    'operation_index': i,
                    'operation_name': getattr(operation, '__class__', type(operation)).__name__,
                    'success': True
                })

                print(f"Branch {branch_index} - Operation {i} completed successfully")

            except Exception as e:
                executed_operations.append({
                    'operation_index': i,
                    'operation_name': getattr(operation, '__class__', type(operation)).__name__,
                    'error': str(e),
                    'success': False
                })
                print(f"Branch {branch_index} - Operation {i} failed: {e}")
                # Continue with next operation

        return {
            'branch_index': branch_index,
            'branch_type': 'pipeline',
            'operations_executed': len(executed_operations),
            'operations_successful': len([op for op in executed_operations if op.get('success', False)]),
            'dataset': dataset,
            'context': context,
            'operation_results': executed_operations,
            'success': True
        }

    def execute_single_operation(self, operation: Any,
                                dataset: SpectraDataset, context: PipelineContext,
                                operation_index: int) -> Dict[str, Any]:
        """Execute a single operation and return result"""
        try:
            operation.execute(dataset, context)

            return {
                'operation_index': operation_index,
                'operation_name': getattr(operation, '__class__', type(operation)).__name__,
                'dataset': dataset,
                'context': context,
                'success': True
            }

        except Exception as e:
            return {
                'operation_index': operation_index,
                'operation_name': getattr(operation, '__class__', type(operation)).__name__,
                'error': str(e),
                'success': False
            }

    def execute_single_branch_safe(self, branch: Any,
                                  dataset: SpectraDataset, context: PipelineContext,
                                  branch_index: int) -> Dict[str, Any]:
        """Execute branch safely for process execution"""
        return self.execute_single_branch(branch, dataset, context, branch_index)

    def copy_dataset(self, dataset: SpectraDataset) -> SpectraDataset:
        """Create a deep copy of dataset"""
        return copy.deepcopy(dataset)

    def copy_context(self, context: PipelineContext) -> PipelineContext:
        """Create a deep copy of context"""
        return copy.deepcopy(context)

    def check_execution_condition(self, operation: Any,
                                dataset: SpectraDataset, context: PipelineContext) -> bool:
        """Check if operation should be executed based on conditions"""
        # Default implementation - execute if operation can execute
        if isinstance(operation, list):
            # For sub-pipelines, check if at least one operation can execute
            return len(operation) > 0
        else:
            # For single operations, check can_execute if available
            if hasattr(operation, 'can_execute') and callable(getattr(operation, 'can_execute', None)):
                return operation.can_execute(dataset, context)
            return True

    def merge_parallel_results(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Merge results from parallel execution"""
        # This is a complex operation that depends on the specific operations
        # For now, we'll implement a basic merge strategy

        successful_results = [r for r in self.operation_results
                             if r and r.get('success', False) and 'dataset' in r]

        if not successful_results:
            print("No successful operations to merge")
            return

        # Skip complex merging for now - individual operation results are preserved
        print(f"Parallel execution completed: {len(successful_results)} successful operations")
        print("Note: Individual model predictions are preserved in their respective contexts")

        # For testing purposes, we can collect some summary info
        prediction_counts = 0
        for result in successful_results:
            if 'context' in result and hasattr(result['context'], 'predictions'):
                prediction_counts += 1

        print(f"Total prediction sets generated: {prediction_counts}")


class DispatchStrategy:
    """Strategy for common dispatch scenarios"""

    @classmethod
    def parallel_transformations(cls, operations: List[Any],
                               max_workers: Optional[int] = None) -> DispatchOperation:
        """Execute transformations in parallel"""
        return DispatchOperation(
            operations=operations,
            dispatch_strategy="parallel",
            max_workers=max_workers,
            merge_results=True,
            execution_mode="thread"
        )

    @classmethod
    def model_ensemble(cls, model_operations: List[Any]) -> DispatchOperation:
        """Execute multiple models in parallel for ensemble"""
        return DispatchOperation(
            operations=model_operations,
            dispatch_strategy="parallel",
            merge_results=True,
            execution_mode="thread"
        )

    @classmethod
    def conditional_pipeline(cls, operations: List[Any]) -> DispatchOperation:
        """Execute operations based on conditions"""
        return DispatchOperation(
            operations=operations,
            dispatch_strategy="conditional",
            merge_results=False
        )

    @classmethod
    def pipeline_fork(cls, operations: List[Any]) -> DispatchOperation:
        """Fork pipeline into multiple branches"""
        return DispatchOperation(
            operations=operations,
            dispatch_strategy="parallel",
            merge_results=False,
            execution_mode="thread"
        )
