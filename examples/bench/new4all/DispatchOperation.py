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
                 operations: List[PipelineOperation],
                 dispatch_strategy: str = "parallel",
                 max_workers: int = None,
                 merge_results: bool = True,
                 execution_mode: str = "thread"):
        """
        Initialize dispatch operation

        Parameters:
        -----------
        operations : List[PipelineOperation]
            List of operations to execute
        dispatch_strategy : str
            Execution strategy: "parallel", "sequential", "conditional"
        max_workers : int
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

        # Store execution summary in context
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
            # Submit all operations
            future_to_operation = {}

            for i, operation in enumerate(self.operations):
                # Create copies of dataset and context for each operation
                dataset_copy = self.copy_dataset(dataset)
                context_copy = self.copy_context(context)

                if self.execution_mode == "thread":
                    # For thread execution, we can pass objects directly
                    future = executor.submit(self.execute_single_operation,
                                           operation, dataset_copy, context_copy, i)
                else:
                    # For process execution, we need to be more careful with serialization
                    future = executor.submit(self.execute_single_operation_safe,
                                           operation, dataset_copy, context_copy, i)

                future_to_operation[future] = (operation, i)

            # Collect results
            self.operation_results = [None] * len(self.operations)

            for future in as_completed(future_to_operation):
                operation, operation_index = future_to_operation[future]

                try:
                    result = future.result()
                    self.operation_results[operation_index] = result
                    print(f"Operation {operation_index} ({operation.get_name()}) completed successfully")

                except Exception as e:
                    error_result = {
                        'operation_index': operation_index,
                        'operation_name': operation.get_name(),
                        'error': str(e),
                        'success': False
                    }
                    self.operation_results[operation_index] = error_result
                    print(f"Operation {operation_index} failed: {e}")

        # Merge results if requested
        if self.merge_results:
            self.merge_parallel_results(dataset, context)

    def execute_sequential(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Execute operations sequentially"""
        self.operation_results = []

        for i, operation in enumerate(self.operations):
            try:
                # Execute operation directly on dataset and context
                operation.execute(dataset, context)

                result = {
                    'operation_index': i,
                    'operation_name': operation.get_name(),
                    'success': True
                }
                self.operation_results.append(result)

                print(f"Operation {i} ({operation.get_name()}) completed successfully")

            except Exception as e:
                result = {
                    'operation_index': i,
                    'operation_name': operation.get_name(),
                    'error': str(e),
                    'success': False
                }
                self.operation_results.append(result)

                print(f"Operation {i} failed: {e}")

                # Decide whether to continue or stop on error
                if not getattr(context, 'continue_on_error', False):
                    break

    def execute_conditional(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Execute operations based on conditions"""
        self.operation_results = []

        for i, operation in enumerate(self.operations):
            # Check if operation should be executed based on condition
            should_execute = self.check_execution_condition(operation, dataset, context)

            if should_execute:
                try:
                    operation.execute(dataset, context)

                    result = {
                        'operation_index': i,
                        'operation_name': operation.get_name(),
                        'executed': True,
                        'success': True
                    }
                    self.operation_results.append(result)

                    print(f"Operation {i} ({operation.get_name()}) executed successfully")

                except Exception as e:
                    result = {
                        'operation_index': i,
                        'operation_name': operation.get_name(),
                        'executed': True,
                        'error': str(e),
                        'success': False
                    }
                    self.operation_results.append(result)

                    print(f"Operation {i} failed: {e}")
            else:
                result = {
                    'operation_index': i,
                    'operation_name': operation.get_name(),
                    'executed': False,
                    'reason': 'Condition not met'
                }
                self.operation_results.append(result)

                print(f"Operation {i} skipped: condition not met")

    def execute_single_operation(self, operation: PipelineOperation,
                                dataset: SpectraDataset, context: PipelineContext,
                                operation_index: int) -> Dict[str, Any]:
        """Execute a single operation and return result"""
        try:
            operation.execute(dataset, context)

            return {
                'operation_index': operation_index,
                'operation_name': operation.get_name(),
                'dataset': dataset,
                'context': context,
                'success': True
            }

        except Exception as e:
            return {
                'operation_index': operation_index,
                'operation_name': operation.get_name(),
                'error': str(e),
                'success': False
            }

    def execute_single_operation_safe(self, operation: PipelineOperation,
                                    dataset: SpectraDataset, context: PipelineContext,
                                    operation_index: int) -> Dict[str, Any]:
        """Execute operation safely for process execution"""
        # This is a simplified version for process execution
        # In practice, you'd need to handle serialization properly
        return self.execute_single_operation(operation, dataset, context, operation_index)

    def copy_dataset(self, dataset: SpectraDataset) -> SpectraDataset:
        """Create a deep copy of dataset"""
        return copy.deepcopy(dataset)

    def copy_context(self, context: PipelineContext) -> PipelineContext:
        """Create a deep copy of context"""
        return copy.deepcopy(context)

    def check_execution_condition(self, operation: PipelineOperation,
                                dataset: SpectraDataset, context: PipelineContext) -> bool:
        """Check if operation should be executed based on conditions"""
        # Default implementation - execute if operation can execute
        return operation.can_execute(dataset, context)

    def merge_parallel_results(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Merge results from parallel execution"""
        # This is a complex operation that depends on the specific operations
        # For now, we'll implement a basic merge strategy

        successful_results = [r for r in self.operation_results
                             if r and r.get('success', False) and 'dataset' in r]

        if not successful_results:
            print("No successful operations to merge")
            return

        # Skip complex merging for now - the main dataset doesn't have X attribute
        # Individual operation results are preserved and accessible
        print(f"Parallel execution completed: {len(successful_results)} successful operations")
        print("Note: Individual model predictions are preserved in their respective contexts")

        # For testing purposes, we can collect some summary info
        prediction_counts = 0
        for result in successful_results:
            if 'context' in result and hasattr(result['context'], 'predictions'):
                pred_dict = result['context'].predictions
                if pred_dict:
                    prediction_counts += len(pred_dict)

        print(f"Total prediction sets generated: {prediction_counts}")


class DispatchStrategy:
    """Strategy for common dispatch scenarios"""

    @classmethod
    def parallel_transformations(cls, operations: List[PipelineOperation],
                               max_workers: int = None) -> DispatchOperation:
        """Execute transformations in parallel"""
        return DispatchOperation(
            operations=operations,
            dispatch_strategy="parallel",
            max_workers=max_workers,
            merge_results=True,
            execution_mode="thread"
        )

    @classmethod
    def model_ensemble(cls, model_operations: List[PipelineOperation]) -> DispatchOperation:
        """Execute multiple models in parallel for ensemble"""
        return DispatchOperation(
            operations=model_operations,
            dispatch_strategy="parallel",
            merge_results=True,
            execution_mode="thread"
        )

    @classmethod
    def conditional_pipeline(cls, operations: List[PipelineOperation]) -> DispatchOperation:
        """Execute operations based on conditions"""
        return DispatchOperation(
            operations=operations,
            dispatch_strategy="conditional",
            merge_results=False
        )

    @classmethod
    def pipeline_fork(cls, operations: List[PipelineOperation]) -> DispatchOperation:
        """Fork pipeline into multiple branches"""
        return DispatchOperation(
            operations=operations,
            dispatch_strategy="parallel",
            merge_results=False,
            execution_mode="thread"
        )
