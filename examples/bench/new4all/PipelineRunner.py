from typing import List
from .PipelineContext import PipelineContext
from .PipelineOperation import PipelineOperation
from .SpectraDataset import SpectraDataset

class PipelineRunner:
    """Simplified, efficient pipeline runner."""

    def __init__(self):
        self.context = PipelineContext()

    def run(self, operations: List[PipelineOperation], dataset: SpectraDataset):
        """Execute pipeline operations."""
        for operation in operations:
            print(f"Executing: {operation.get_name()}")
            try:
                operation.execute(dataset, self.context)
                self.context.processing_history.append(operation.get_name())
            except Exception as e:
                print(f"Error in {operation.get_name()}: {e}")
                raise

        print("Pipeline completed successfully")
        return self.context.predictions