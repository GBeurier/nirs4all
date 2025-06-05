"""
Subpipeline Operation - Executes a sequence of operations on the same dataset scope

This operation:
1. Takes a list of operations to execute sequentially
2. Maintains the same data scope throughout the subpipeline
3. Allows for complex multi-step transformations
4. Can be nested within larger pipelines
"""
from typing import List, Optional, Any, Dict

from PipelineOperation import PipelineOperation
from SpectraDataset import SpectraDataset
from PipelineContext import PipelineContext


class OperationSubpipeline(PipelineOperation):
    """Pipeline operation that executes a sequence of operations sequentially"""

    def __init__(self,
                 operations: List[PipelineOperation],
                 maintain_scope: bool = True,
                 operation_name: Optional[str] = None):
        """
        Initialize subpipeline operation

        Args:
            operations: list of operations to execute sequentially
            maintain_scope: whether to maintain the same data scope throughout
            operation_name: custom name for this operation
        """
        self.operations = operations
        self.maintain_scope = maintain_scope
        self.operation_name = operation_name
        self.execution_results = []  # Store results from each operation

    def execute(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Execute subpipeline: run operations sequentially on same scope"""
        print(f"🔄 Executing {self.get_name()}")
        print(f"  📋 Subpipeline contains {len(self.operations)} operations")

        # Save initial scope if maintaining scope
        initial_filters = context.current_filters.copy() if self.maintain_scope else {}

        # Execute each operation in sequence
        for i, operation in enumerate(self.operations):
            print(f"  🔗 Step {i+1}/{len(self.operations)}: {operation.get_name()}")

            # Restore scope if maintaining scope
            if self.maintain_scope:
                context.current_filters = initial_filters.copy()

            # Check if operation can execute
            if not operation.can_execute(dataset, context):
                print(f"    ⚠️ Skipping {operation.get_name()} - cannot execute")
                continue

            # Execute operation
            try:
                operation.execute(dataset, context)
                self.execution_results.append({
                    'operation': operation.get_name(),
                    'status': 'success',
                    'step': i + 1
                })
                print(f"    ✅ {operation.get_name()} completed successfully")

            except Exception as e:
                error_msg = f"Failed to execute {operation.get_name()}: {str(e)}"
                print(f"    ❌ {error_msg}")

                self.execution_results.append({
                    'operation': operation.get_name(),
                    'status': 'failed',
                    'error': str(e),
                    'step': i + 1
                })

                # Decide whether to continue or stop
                # For now, we'll continue with remaining operations
                continue

        print(f"  ✅ Subpipeline execution complete")
        self._report_results()

    def _report_results(self) -> None:
        """Report execution results"""
        successful = sum(1 for r in self.execution_results if r['status'] == 'success')
        failed = sum(1 for r in self.execution_results if r['status'] == 'failed')

        print(f"  📊 Execution summary: {successful} successful, {failed} failed")

        if failed > 0:
            print(f"  ❌ Failed operations:")
            for result in self.execution_results:
                if result['status'] == 'failed':
                    print(f"    Step {result['step']}: {result['operation']} - {result['error']}")

    def get_execution_results(self) -> List[Dict[str, Any]]:
        """Get detailed execution results"""
        return self.execution_results.copy()

    def get_successful_operations(self) -> List[str]:
        """Get list of successfully executed operation names"""
        return [r['operation'] for r in self.execution_results if r['status'] == 'success']

    def get_failed_operations(self) -> List[str]:
        """Get list of failed operation names"""
        return [r['operation'] for r in self.execution_results if r['status'] == 'failed']

    def add_operation(self, operation: PipelineOperation) -> None:
        """Add operation to the end of the subpipeline"""
        self.operations.append(operation)

    def insert_operation(self, index: int, operation: PipelineOperation) -> None:
        """Insert operation at specific index in the subpipeline"""
        self.operations.insert(index, operation)

    def remove_operation(self, index: int) -> PipelineOperation:
        """Remove and return operation at specific index"""
        return self.operations.pop(index)

    def get_name(self) -> str:
        """Get operation name"""
        if self.operation_name:
            return self.operation_name

        if not self.operations:
            return "Subpipeline(empty)"

        # Create abbreviated name from first and last operations
        if len(self.operations) == 1:
            return f"Subpipeline({self.operations[0].get_name()})"
        elif len(self.operations) <= 3:
            op_names = " → ".join([op.get_name() for op in self.operations])
            return f"Subpipeline({op_names})"
        else:
            first_op = self.operations[0].get_name()
            last_op = self.operations[-1].get_name()
            return f"Subpipeline({first_op} → ... → {last_op}[{len(self.operations)}])"

    def can_execute(self, dataset: SpectraDataset, context: PipelineContext) -> bool:
        """Check if subpipeline can be executed"""
        if not self.operations:
            return False

        # Check if at least one operation can execute
        return any(op.can_execute(dataset, context) for op in self.operations)

    def __len__(self) -> int:
        """Return number of operations in subpipeline"""
        return len(self.operations)

    def __getitem__(self, index: int) -> PipelineOperation:
        """Get operation at index"""
        return self.operations[index]

    def __iter__(self):
        """Iterate over operations"""
        return iter(self.operations)
