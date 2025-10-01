from typing import Any, Dict, TYPE_CHECKING

from sklearn.base import TransformerMixin

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset

import numpy as np

@register_controller
class YTransformerMixinController(OperatorController):
    """
    Controller for applying sklearn TransformerMixin operators to targets (y) instead of features (X).

    Triggered by the "y_processing" keyword and applies transformations to target data,
    fitting on train targets and transforming all target data.
    """
    priority = 5

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Match if keyword is 'y_processing' and operator is a TransformerMixin."""
        # print(">>>> Checking YTransformerMixinController match...")
        # print(f"Keyword: {keyword}, Operator: {operator}, Is TransformerMixin: {isinstance(operator, TransformerMixin) or issubclass(operator.__class__, TransformerMixin)}")
        return (keyword == "y_processing" and
                (isinstance(operator, TransformerMixin) or issubclass(operator.__class__, TransformerMixin)))

    @classmethod
    def use_multi_source(cls) -> bool:
        """Check if the operator supports multi-source datasets."""
        return False  # Target processing doesn't depend on multiple sources

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Y transformers should not execute during prediction mode."""
        return False

    def execute(
        self,
        step: Any,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Any = None,
        prediction_store: Any = None
    ):
        """
        Execute transformer on dataset targets, fitting on train targets and transforming all targets.
        Skips execution in prediction mode.

        Args:
            step: Pipeline step configuration
            operator: sklearn TransformerMixin to apply to targets
            dataset: Dataset containing targets to transform
            context: Pipeline context with partition information
            runner: Pipeline runner instance
            source: Source index (not used for target processing)
            mode: Execution mode ("train" or "predict")
            loaded_binaries: Pre-loaded binaries (unused)

        Returns:
            Tuple of (updated_context, fitted_transformers_list)
        """
        # Skip execution in prediction mode
        if mode == "predict":
            return context, []
        import pickle
        from sklearn.base import clone

        operator_name = operator.__class__.__name__

        # Get current y processing from context, default to "numeric"
        current_y_processing = context.get("y", "numeric")

        # Get train and all targets
        train_context = context.copy()
        train_context["partition"] = "train"
        train_data = dataset.y(train_context)
        all_data = dataset.y(context)

        # Clone and fit the transformer on training targets
        transformer = clone(operator)
        transformer.fit(train_data)

        # Transform all targets
        transformed_targets = transformer.transform(all_data)

        # Create new processing name
        new_processing_name = f"{current_y_processing}_{operator_name}{runner.next_op()}"

        # Add the processed targets to the dataset
        dataset.add_processed_targets(
            processing_name=new_processing_name,
            targets=transformed_targets,
            ancestor_processing=current_y_processing,
            transformer=transformer
        )

        # Update context to use the new y processing
        updated_context = context.copy()
        updated_context["y"] = new_processing_name

        # Serialize fitted transformer for potential reuse
        transformer_binary = pickle.dumps(transformer)
        fitted_transformers = [(f"{operator_name}_{new_processing_name}.pkl", transformer_binary)]

        # print(f"✅ Successfully applied {operator_name} to targets: {current_y_processing} → {new_processing_name}")
        # print(f"   Train shape: {train_targets.shape} → {transformer.transform(train_targets).shape}")
        # print(f"   All shape: {all_targets.shape} → {transformed_targets.shape}")

        return updated_context, fitted_transformers