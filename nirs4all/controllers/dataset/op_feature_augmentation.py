from typing import Any, Dict, TYPE_CHECKING

from sklearn.base import TransformerMixin

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.spectra.spectra_dataset import SpectroDataset


@register_controller
class FeatureAugmentationController(OperatorController):
    priority = 10

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        return keyword == "feature_augmentation"

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
        print(f"Executing feature augmentation for step: {step}, keyword: {context.get('keyword', '')}, source: {source}")

        # Apply the transformer to the dataset
        try:
            x_source = dataset.x(context, "2d", source=source)
            contexts = []
            steps = []
            for i, operation in enumerate(step["feature_augmentation"]):
                if operation is None:
                    contexts.append(context)
                    steps.append(None)
                    continue
                local_context = context.copy()
                local_context["processing"] = f"sample_augmentation_{i}"
                dataset.add_features(local_context, x_source.copy())
                contexts.append(local_context)
                steps.append(operation)

            runner.run_steps(steps, dataset, contexts, execution="parallel")
            res_context = context.copy()
            res_context["processing"] = [c["processing"] for c in contexts]
            return res_context

        except Exception as e:
            print(f"❌ Error applying transformation: {e}")
            raise