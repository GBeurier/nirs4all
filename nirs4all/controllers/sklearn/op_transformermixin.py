from typing import Any, Dict, TYPE_CHECKING

from sklearn.base import TransformerMixin

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.spectra.spectra_dataset import SpectroDataset

import numpy as np

## TODO add parrallel support for multi-source datasets and multi-processing datasets

@register_controller
class TransformerMixinController(OperatorController):
    priority = 10

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        return isinstance(operator, TransformerMixin) or issubclass(operator.__class__, TransformerMixin)

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
        # Apply the transformer to the dataset
        try:
            operator_id = operator.__class__.__name__[0:6] + f"_{context.get('step_id', 'unknown')}"

            train_context = context.copy()
            # print("🔄 Preparing training context for operator:", operator_id)
            # print(f"🔄 Current context: {context}")
            train_context["partition"] = "train"

            train_data = dataset.x(train_context, "3d", concat_source=False)
            all_data = dataset.x(context, "3d", concat_source=False)

            # for train



            fit_data = dataset.x(train_context, "2d", )

            print(f"🔄 Fitting operator {operator_id} with data shape: {fit_data.shape}")

            operator.fit(fit_data)

            transformed_data = dataset.x(context, "2d", source=source)

            # print(f"🔄 Transforming data with operator {operator_id} with data shape: {transformed_data.shape}")

            # transformed_data = operator.transform(transformed_data)

            # print(f"✅ Transformation complete, transformed data shape: {transformed_data.shape}")

            # processing_update = {}
            # if "processing" in context:
            #     if isinstance(context["processing"], list):
            #         processing_update["processing"] = [p + f"_{operator_id}" for p in context["processing"]]
            #     else:
            #         processing_update["processing"] = context["processing"] + f"_{operator_id}"
            # if "augmentation" in context:
            #     if isinstance(context["augmentation"], list):
            #         processing_update["augmentation"] = [p + f"_{operator_id}" for p in context["augmentation"]]
            #     else:
            #         processing_update["augmentation"] = context["augmentation"] + f"_{operator_id}"
            #     if "processing" in processing_update:
            #         del processing_update["processing"]

            # print(np.mean(transformed_data, axis=0))
            # print(f"from {dataset.x(context, '2d', source=source).shape} {context} to {transformed_data.shape}, filter_update={processing_update}")

            # dataset.set_x(context, transformed_data, layout="2d", filter_update=processing_update, source=source)

            print(f"✅ Successfully applied {operator_id} transformation")

        except Exception as e:
            print(f"❌ Error applying transformation: {e}")
            raise

        if "processing" in processing_update:
            context["processing"] = processing_update["processing"]
        return context
