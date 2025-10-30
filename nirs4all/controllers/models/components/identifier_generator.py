"""
Model Identifier Generator - Generate consistent model identifiers

This component centralizes all model naming and identification logic.
Extracted from launch_training() lines 329-345 to improve maintainability.

Generates:
    - classname: from model config or instance.__class__.__name__
    - name: custom name from config or classname
    - model_id: name + operation counter (unique for run)
    - display_name: model_id with fold suffix if applicable
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner


@dataclass
class ModelIdentifiers:
    """Container for all model identifiers."""

    classname: str  # Class name of the model (e.g., "RandomForestRegressor")
    name: str  # User-provided name or classname
    model_id: str  # name + operation counter (e.g., "MyModel_10")
    display_name: str  # model_id with fold suffix (e.g., "MyModel_10_fold0")
    operation_counter: int  # Operation counter from runner
    step_id: int  # Pipeline step index
    fold_idx: Optional[int]  # Fold index if applicable


class ModelIdentifierGenerator:
    """Generates consistent model identifiers for training and persistence.

    This component extracts and centralizes all the naming logic that was
    previously scattered in launch_training().

    Example:
        >>> generator = ModelIdentifierGenerator()
        >>> identifiers = generator.generate(
        ...     model_config={'name': 'MyPLS', 'class': 'sklearn.cross_decomposition.PLSRegression'},
        ...     runner=runner,
        ...     context={'step_id': 5},
        ...     fold_idx=0
        ... )
        >>> identifiers.model_id
        'MyPLS_10'
        >>> identifiers.display_name
        'MyPLS_10_fold0'
    """

    def __init__(self, helper=None):
        """Initialize identifier generator.

        Args:
            helper: ModelControllerHelper instance for extracting names from config.
                   If None, will be created internally.
        """
        if helper is None:
            from ..helper import ModelControllerHelper
            helper = ModelControllerHelper()
        self.helper = helper

    def generate(
        self,
        model_config: Dict[str, Any],
        runner: 'PipelineRunner',
        context: Dict[str, Any],
        fold_idx: Optional[int] = None
    ) -> ModelIdentifiers:
        """Generate all model identifiers from configuration and context.

        Args:
            model_config: Model configuration dictionary
            runner: Pipeline runner for operation counter
            context: Execution context with step_id
            fold_idx: Optional fold index for cross-validation

        Returns:
            ModelIdentifiers: Container with all generated identifiers
        """
        # Extract base information
        classname = self.helper.extract_classname_from_config(model_config)
        name = self.helper.extract_core_name(model_config)

        # Get operation counter and step info
        operation_counter = runner.next_op()
        step_id = context.get('step_id', 0)

        # Build model_id and display_name
        model_id = f"{name}_{operation_counter}"
        display_name = model_id
        if fold_idx is not None:
            display_name += f"_fold{fold_idx}"

        return ModelIdentifiers(
            classname=classname,
            name=name,
            model_id=model_id,
            display_name=display_name,
            operation_counter=operation_counter,
            step_id=step_id,
            fold_idx=fold_idx
        )

    def generate_binary_key(
        self,
        model_id: str,
        fold_idx: Optional[int] = None
    ) -> str:
        """Generate the binary storage key for a model.

        Args:
            model_id: Base model identifier (e.g., "MyModel_10")
            fold_idx: Optional fold index

        Returns:
            Binary key string (e.g., "MyModel_10" or "MyModel_10_fold0")
        """
        if fold_idx is not None:
            return f"{model_id}_fold{fold_idx}"
        return model_id
