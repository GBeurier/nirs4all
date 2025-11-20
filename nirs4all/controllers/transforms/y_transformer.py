from typing import Any, Dict, TYPE_CHECKING

from sklearn.base import TransformerMixin

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.pipeline.config.context import ExecutionContext

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.steps.parser import ParsedStep

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
        # print(f">>>> Checking YTransformerMixinController match...")
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
        return True

    def execute(
        self,
        step_info: 'ParsedStep',
        dataset: 'SpectroDataset',
        context: ExecutionContext,
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
            step_info: Parsed step containing operator and metadata
            dataset: Dataset to operate on
            dataset: Dataset containing targets to transform
            context: Pipeline context with partition information
            runner: Pipeline runner instance
            source: Source index (not used for target processing)
            mode: Execution mode ("train" or "predict")
            loaded_binaries: Pre-loaded binaries (unused)

        Returns:
            Tuple of (updated_context, fitted_transformers_list)
        """
        # Extract operator for compatibility with existing code
        operator = step_info.operator

        # Skip execution in prediction mode
        from sklearn.base import clone

        # Naming for the new processing
        operator_name = operator.__class__.__name__
        current_y_processing = context.state.y_processing
        new_processing_name = f"{current_y_processing}_{operator_name}{runner.next_op()}"

        if (mode == "predict" or mode == "explain") and loaded_binaries:
            transformer = loaded_binaries[0][1] if loaded_binaries else operator
            # print(f"🔄 Using pre-loaded transformer for prediction: {transformer}")
            dataset._targets.add_processed_targets(
                processing_name=new_processing_name,
                targets=np.array([]),
                ancestor=current_y_processing,
                transformer=transformer,
                mode=mode
            )
            updated_context = context.with_y(new_processing_name)
            # print(f">>>>>>> Registered {transformer}")
            # try:
            #     print(transformer.data_min_, transformer.data_max_)
            # except AttributeError:
            #     print("Transformer does not have data_min_ or data_max_ attributes")
            return updated_context, []

        # Get train and all targets
        train_context = context.with_partition("train")

        train_y_selector = dict(train_context.selector)
        train_y_selector['y'] = train_context.state.y_processing
        train_data = dataset.y(train_y_selector)

        all_y_selector = dict(context.selector)
        all_y_selector['y'] = context.state.y_processing
        all_data = dataset.y(all_y_selector)

        # Clone and fit the transformer on training targets
        transformer = clone(operator)
        transformer.fit(train_data)

        # Transform all targets
        transformed_targets = transformer.transform(all_data)


        # Add the processed targets to the dataset
        dataset.add_processed_targets(
            processing_name=new_processing_name,
            targets=transformed_targets,
            ancestor_processing=current_y_processing,
            transformer=transformer
        )
        # print(f">>>>>>> Registered {transformer}")
        # try:
        #     print(transformer.data_min_, transformer.data_max_)
        # except AttributeError:
        #     print("Transformer does not have data_min_ or data_max_ attributes")
        # Update context to use the new y processing
        updated_context = context.with_y(new_processing_name)

        # Persist fitted transformer using new serializer
        if mode == "train":
            artifact = runner.saver.persist_artifact(
                step_number=runner.step_number,
                name=f"y_{operator_name}",
                obj=transformer,
                format_hint='sklearn'
            )
            fitted_transformers = [artifact]
            return updated_context, fitted_transformers

        # print(f"✅ Successfully applied {operator_name} to targets: {current_y_processing} → {new_processing_name}")
        # print(f"   Train shape: {train_targets.shape} → {transformer.transform(train_targets).shape}")
        # print(f"   All shape: {all_targets.shape} → {transformed_targets.shape}")

        return updated_context, []

