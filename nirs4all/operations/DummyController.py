"""DummyController.py - A dummy controller for testing purposes in the nirs4all pipeline."""

from typing import Any, Dict, TYPE_CHECKING

from .OperatorController import OperatorController, register_controller

if TYPE_CHECKING:
    from nirs4all.pipeline.PipelineRunner import PipelineRunner
    from nirs4all.spectra.SpectraDataset import SpectraDataset

@register_controller
class DummyController(OperatorController):
    """Dummy controller for testing purposes."""

    priority = 1000  # Lower priority than other controllers

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Check if the operator matches the step and keyword."""
        return True  # Always matches for testing

    @classmethod
    def execute(
        cls,
        step: Any,
        dataset: 'SpectraDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner'
    ):
        """Run the operator with the given parameters and context."""
        print(f"Executing dummy operation for step: {step}, keyword: {context.get('keyword', '')}")
        return dataset  # Return the dataset unchanged