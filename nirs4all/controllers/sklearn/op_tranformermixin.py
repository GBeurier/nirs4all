from typing import Any, Dict, TYPE_CHECKING

from sklearn.base import TransformerMixin

from nirs4all.operations.operator_controller import OperatorController
from nirs4all.operations.operator_registry import register_controller

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.spectra.spectra_dataset import SpectraDataset


@register_controller
class TransformerMixinController(OperatorController):
    priority = 10  # Lower priority than other controllers

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        return isinstance(operator, TransformerMixin)

    def execute(
        self,
        step: Any,
        operator: Any,
        dataset: 'SpectraDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner'
    ):
        """Run the operator with the given parameters and context."""
        print(f"Executing transformer operation for step: {step}, keyword: {context.get('keyword', '')}")

        # # Apply the transformer to the dataset
        # try:
        #     # Get the current features from the dataset
        #     if hasattr(dataset, 'features') and dataset.features is not None:
        #         # Transform each source in the features
        #         transformed_sources = []
        #         for source in dataset.features.sources:
        #             if hasattr(operator, 'fit_transform'):
        #                 transformed_source = operator.fit_transform(source)
        #             else:
        #                 # Fallback to fit then transform
        #                 transformed_source = operator.fit(source).transform(source)
        #             transformed_sources.append(transformed_source)

        #         # Create new SpectraFeatures with transformed data
        #         from nirs4all.spectra.spectra_features import SpectraFeatures
        #         dataset.features = SpectraFeatures(transformed_sources)
        #         print(f"✅ Successfully applied {operator.__class__.__name__} transformation")
        #     else:
        #         print("⚠️ No features found in dataset to transform")

        # except Exception as e:
        #     print(f"❌ Error applying transformation: {e}")
        #     raise

        # return dataset