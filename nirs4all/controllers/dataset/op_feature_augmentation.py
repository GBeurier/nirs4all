from typing import Any, Dict, TYPE_CHECKING

from sklearn.base import TransformerMixin

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.spectra.spectra_dataset import SpectroDataset
import copy


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

    def execute( ## TODO reup parralelization
        self,
        step: Any,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        source: int = -1
    ):
        print(f"Executing feature augmentation for step: {step}, keyword: {context.get('keyword', '')}, source: {source}")
        try:
            initial_context = copy.deepcopy(context)
            # Faire une deepcopy à chaque utilisation pour éviter les modifications
            original_source_processings = copy.deepcopy(initial_context["processing"])

            for i, operation in enumerate(step["feature_augmentation"]):
                # Recréer source_processings à chaque itération pour éviter les mutations
                source_processings = copy.deepcopy(original_source_processings)
                local_context = copy.deepcopy(initial_context)

                if i == 0 and operation is None:
                    print("Skipping no-op augmentation")
                    continue
                if i > 0:
                    local_context["add_feature"] = True

                # Assigner une nouvelle copie à chaque fois
                local_context["processing"] = copy.deepcopy(source_processings)
                runner.run_step(operation, dataset, local_context, is_substep=True)

        except Exception as e:
            print(f"❌ Error applying feature augmentation: {e}")
            raise

        return context, []