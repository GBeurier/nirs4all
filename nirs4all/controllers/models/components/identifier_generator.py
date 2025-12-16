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
    from nirs4all.pipeline.config.context import ExecutionContext


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
            helper: Deprecated parameter, kept for backwards compatibility.
        """
        # Helper is deprecated - methods are now implemented directly in this class

    def extract_core_name(self, model_config: Dict[str, Any]) -> str:
        """Extract core name from model configuration.

        User-provided name or class name. This is the base name provided by
        the user or derived from the class.

        Args:
            model_config: Model configuration dictionary.

        Returns:
            str: Core name extracted from config.
        """
        if isinstance(model_config, dict):
            if 'name' in model_config:
                return model_config['name']
            elif 'function' in model_config:
                # Handle function-based models (like TensorFlow functions)
                function_path = model_config['function']
                if isinstance(function_path, str):
                    # Extract function name from path
                    return function_path.split('.')[-1]
                else:
                    return str(function_path)
            elif 'class' in model_config:
                class_path = model_config['class']
                return class_path.split('.')[-1]  # Get class name from full path
            elif 'model_instance' in model_config:
                return self._get_model_class_name(model_config['model_instance'])
            elif 'model' in model_config:
                # Handle nested model structure
                model_obj = model_config['model']
                if isinstance(model_obj, dict):
                    # Handle dict wrapper from deserialize_component for @framework functions
                    if 'func' in model_obj:
                        func = model_obj['func']
                        if callable(func) and hasattr(func, '__name__'):
                            return func.__name__
                    if 'function' in model_obj:
                        function_path = model_obj['function']
                        return function_path.split('.')[-1] if isinstance(function_path, str) else str(function_path)
                    elif 'class' in model_obj:
                        return model_obj['class'].split('.')[-1]
                else:
                    return self._get_model_class_name(model_obj)

        # Fallback for other types
        return self._get_model_class_name(model_config)

    def extract_classname_from_config(self, model_config: Dict[str, Any]) -> str:
        """Extract classname from model configuration.

        Based on the model declared in config or instance.__class__.__name__ or function name.

        Args:
            model_config: Model configuration dictionary.

        Returns:
            str: Class name of the model.
        """
        # Extract model instance
        model_instance = self._get_model_instance_from_config(model_config)

        if model_instance is not None:
            # Handle dict wrapper from deserialize_component for @framework functions
            if isinstance(model_instance, dict) and 'func' in model_instance:
                func = model_instance['func']
                if callable(func) and hasattr(func, '__name__'):
                    return func.__name__
            # Handle functions
            if callable(model_instance) and hasattr(model_instance, '__name__'):
                return model_instance.__name__
            # Handle classes and instances
            elif hasattr(model_instance, '__class__'):
                return model_instance.__class__.__name__
            else:
                return str(type(model_instance).__name__)

        return "unknown_model"

    def _get_model_instance_from_config(self, model_config: Dict[str, Any]) -> Any:
        """Helper to extract model instance from various config formats.

        Args:
            model_config: Model configuration dictionary.

        Returns:
            Model instance or None.
        """
        if isinstance(model_config, dict):
            # Direct model_instance
            if 'model_instance' in model_config:
                return model_config['model_instance']
            # Nested model structure
            elif 'model' in model_config:
                model_obj = model_config['model']
                if isinstance(model_obj, dict):
                    if 'model' in model_obj:
                        return model_obj['model']
                    else:
                        return model_obj
                else:
                    return model_obj
        else:
            return model_config

        return None

    def _get_model_class_name(self, model: Any) -> str:
        """Get the class name of a model.

        Args:
            model: Model object, class, function, or string representation.

        Returns:
            str: Class or function name.
        """
        import inspect

        # Handle dict wrapper from deserialize_component for @framework functions
        if isinstance(model, dict) and 'func' in model:
            func = model['func']
            if callable(func) and hasattr(func, '__name__'):
                return func.__name__

        if inspect.isclass(model):
            return f"{model.__qualname__}"

        if inspect.isfunction(model) or inspect.isbuiltin(model):
            return f"{model.__name__}"

        # Handle string representation of functions/classes from deserialization
        if isinstance(model, str):
            if model.startswith("<function ") and " at 0x" in model:
                # Extract function name
                return model.split("<function ")[1].split(" at ")[0]
            elif model.startswith("<class '") and "' at 0x" in model:
                # Extract class name
                return model.split("<class '")[1].split("' at ")[0].split(".")[-1]

        else:
            return str(type(model).__name__)

    def generate(
        self,
        model_config: Dict[str, Any],
        runner: 'PipelineRunner',
        context: 'ExecutionContext',
        fold_idx: Optional[int] = None
    ) -> ModelIdentifiers:
        """Generate all model identifiers from configuration and context.

        Args:
            model_config: Model configuration dictionary
            runner: Pipeline runner for operation counter
            context: Execution context with step_number
            fold_idx: Optional fold index for cross-validation

        Returns:
            ModelIdentifiers: Container with all generated identifiers
        """
        # Extract base information
        classname = self.extract_classname_from_config(model_config)
        name = self.extract_core_name(model_config)

        # Get operation counter and step info
        operation_counter = runner.next_op()
        # Use step_number (int) not step_id (str) for the numeric step index
        step_id = context.state.step_number

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
