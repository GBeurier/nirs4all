from typing import Any, Dict, TYPE_CHECKING

from sklearn.base import TransformerMixin

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.spectra.spectra_dataset import SpectroDataset


@register_controller
class TransformerMixinController(OperatorController):
    priority = 10

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        return isinstance(operator, TransformerMixin)

    @classmethod
    def use_multi_source(cls) -> bool:
        """Check if the operator supports multi-source datasets."""
        return True

    def execute(
        self,
        step: Any,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        source: int = -1
    ):
        print(f"Executing transformer operation for step: {step}, keyword: {context.get('keyword', '')}, source: {source}")

        # Apply the transformer to the dataset
        try:
            fit_data = dataset.x({"partition": "train"}, "2d", source=source)
            operator.fit(fit_data)
            transformed_data = dataset.x({}, "2d", source=source)
            transformed_data = operator.transform(transformed_data)
            dataset.set_x({}, transformed_data, layout="2d", source=source)
            print(f"✅ Successfully applied {operator.__class__.__name__} transformation")

        except Exception as e:
            print(f"❌ Error applying transformation: {e}")
            raise