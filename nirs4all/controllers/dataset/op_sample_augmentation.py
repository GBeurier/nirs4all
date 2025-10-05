from typing import Any, Dict, TYPE_CHECKING

from sklearn.base import TransformerMixin

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.spectra.spectra_dataset import SpectroDataset


@register_controller
class SampleAugmentationController(OperatorController):
    priority = 10

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        return keyword == "sample_augmentation"

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
        print(f"Executing sample augmentation for step: {step}, keyword: {context.get('keyword', '')}, source: {source}")

        # Apply the transformer to the dataset
        try:
            contexts = []
            steps = []
            for i, operation in enumerate(step["sample_augmentation"]):
                augmentation_id=f"aug_{i}"
                if operation is None:
                    contexts.append(context)
                    steps.append(None)
                    continue
                local_context = context.copy()
                local_context["origin"] = None
                local_context["partition"] = "train"
                dataset.augment_samples(local_context, 1, augmentation_id=augmentation_id)
                local_context = context.copy()
                local_context["augmentation"] = augmentation_id
                contexts.append(local_context)
                steps.append(operation)

            runner.run_steps(steps, dataset, contexts, execution="sequential")
            return context

        except Exception as e:
            print(f"❌ Error applying transformation: {e}")
            raise