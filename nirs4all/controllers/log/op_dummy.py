"""DummyController.py - A dummy controller for testing purposes in the nirs4all pipeline."""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset

@register_controller
class DummyController(OperatorController):
    """Dummy controller for testing purposes."""

    priority = 1000  # Lower priority than other controllers

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Check if the operator matches the step and keyword."""
        # Only match if explicitly requested for testing
        # This prevents the dummy controller from interfering with real operations
        return keyword == "dummy" or (isinstance(step, str) and step == "dummy")

    @classmethod
    def use_multi_source(cls) -> bool:
        """Check if the operator supports multi-source datasets."""
        return False

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Dummy controller supports prediction mode."""
        return True

    def execute(
        self,
        step: Any,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, bytes]]] = None
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Run the operator with the given parameters and context."""
        # print a explosion character
        print(f"💥 Executing dummy operation for step: {step}, keyword: {context.get('keyword', '')}, source: {source}")

        return context, []

