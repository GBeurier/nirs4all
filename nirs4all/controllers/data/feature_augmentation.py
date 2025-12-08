from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING

from sklearn.base import TransformerMixin

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.utils.emoji import CROSS

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.spectra.spectra_dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext
import copy


@register_controller
class FeatureAugmentationController(OperatorController):
    priority = 10

    @staticmethod
    def normalize_generator_spec(spec: Any) -> Any:
        """Normalize generator spec for feature_augmentation context.

        In feature_augmentation context, multi-selection should use combinations
        by default since the order of parallel feature channels doesn't matter.
        Translates legacy 'size' to 'pick' for explicit semantics.

        Args:
            spec: Generator specification (may contain _or_, size, pick, arrange).

        Returns:
            Normalized spec with 'size' converted to 'pick' if needed.
        """
        if not isinstance(spec, dict):
            return spec

        # If explicit pick/arrange specified, honor it
        if "pick" in spec or "arrange" in spec:
            return spec

        # Convert legacy size to pick (combinations) for feature_augmentation
        if "size" in spec and "_or_" in spec:
            result = dict(spec)
            result["pick"] = result.pop("size")
            return result

        return spec

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        return keyword == "feature_augmentation"

    @classmethod
    def use_multi_source(cls) -> bool:
        """Check if the operator supports multi-source datasets."""
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Feature augmentation should NOT execute during prediction mode - transformations are already applied and saved."""
        return True

    def execute(  # TODO reup parralelization
        self,
        step_info: 'ParsedStep',
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        runtime_context: 'RuntimeContext',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple['ExecutionContext', List[Tuple[str, bytes]]]:
        # print(f"Executing feature augmentation for step: {step}, keyword: {context.metadata.keyword}, source: {source}, mode: {mode}")
        op = step_info.operator

        try:
            initial_context = context.copy()
            # Faire une deepcopy à chaque utilisation pour éviter les modifications
            original_source_processings = copy.deepcopy(initial_context.selector.processing)
            all_artifacts = []

            for i, operation in enumerate(step_info.original_step["feature_augmentation"]):
                # Recréer source_processings à chaque itération pour éviter les mutations
                source_processings = copy.deepcopy(original_source_processings)
                local_context = initial_context.copy()
                # print(f"Applying feature augmentation operation {i + 1}/{len(step_info.original_step['feature_augmentation'])}: {operation}")
                # if i == 0 and operation is None:
                #     print("Skipping no-op feature augmentation")
                #     continue
                # if i > 0:
                local_context = local_context.with_metadata(add_feature=True)

                # Assigner une nouvelle copie à chaque fois
                local_context = local_context.with_processing(copy.deepcopy(source_processings))

                # Run substep and collect artifacts
                if runtime_context.step_runner:
                    runtime_context.substep_number += 1
                    result = runtime_context.step_runner.execute(
                        operation, dataset, local_context, runtime_context,
                        loaded_binaries=loaded_binaries, prediction_store=prediction_store
                    )
                    updated_context = result.updated_context
                    substep_artifacts = result.artifacts
                    all_artifacts.extend(substep_artifacts)

            new_processing = []
            for sdx in range(dataset.n_sources):
                processing_ids = dataset.features_processings(sdx)
                new_processing.append(processing_ids)
            context = context.with_processing(new_processing)
            return context, all_artifacts

        except Exception as e:
            print(f"{CROSS} Error applying feature augmentation: {e}")
            raise
