from typing import Any, Dict

from sklearn.base import TransformerMixin

from nirs4all.core.PipelineOperatorWrapper import PipelineOperatorWrapper
from nirs4all.core import register_wrapper


@register_wrapper
class OpTransformerMixin(PipelineOperatorWrapper):
    """Mixin class for pipeline operators that transform data."""

    @classmethod
    def matches(cls, op: Any | None, keyword: str | None) -> bool:
        """Check if the operator is a transformer."""
        return isinstance(op, TransformerMixin)

    def execute(self, operator: Any | None, dataset: Any, context: Dict[str, Any], params: Dict[str, Any]):
        """Run the transformation on the dataset."""
        if operator is None:
            raise ValueError("Operator cannot be None for transformation.")

