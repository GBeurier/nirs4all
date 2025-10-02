"""
Model Controller Helper - Utilities for model instantiation, serialization and naming

This module contains utilities for model naming, cloning, instantiation
and serialization operations that are specific to the controller layer.
Score calculation and other general utilities have been moved to utils.model_utils.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np
import copy
import inspect

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner

try:
    from sklearn.base import clone as sklearn_clone
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ModelControllerHelper:
    """
    Model controller helper for naming, cloning, and instantiation.

    Centralizes model-related utility functions specific to the controller layer.
    General scoring and evaluation utilities are in utils.model_utils.
    """

    def __init__(self):
        pass

    def extract_core_name(self, model_config: Dict[str, Any]) -> str:
        """
        Extract Core Name: User-provided name or class name.
        This is the base name provided by the user or derived from the class.
        """
        if isinstance(model_config, dict):
            if 'name' in model_config:
                return model_config['name']
            elif 'model_instance' in model_config:
                # Handle extracted model config from _extract_model_config
                return self.get_model_class_name(model_config['model_instance'])
            elif 'function' in model_config:
                # Handle function-based models (like TensorFlow functions)
                function_path = model_config['function']
                if isinstance(function_path, str):
                    # Extract function name from path (e.g., 'nirs4all.operators.models.cirad_tf.nicon' -> 'nicon')
                    return function_path.split('.')[-1]
                else:
                    return str(function_path)
            elif 'class' in model_config:
                class_path = model_config['class']
                return class_path.split('.')[-1]  # Get class name from full path
            elif '_runtime_instance' in model_config:
                return self.get_model_class_name(model_config['_runtime_instance'])
            elif 'model' in model_config:
                # Handle nested model structure
                model_obj = model_config['model']
                if isinstance(model_obj, dict):
                    if 'function' in model_obj:
                        # Handle nested function models
                        print(">>>> model_obj:", model_obj)
                        function_path = model_obj['function']
                        return function_path.split('.')[-1] if isinstance(function_path, str) else str(function_path)
                    elif '_runtime_instance' in model_obj:
                        return self.get_model_class_name(model_obj['_runtime_instance'])
                    elif 'class' in model_obj:
                        return model_obj['class'].split('.')[-1]
                else:
                    return self.get_model_class_name(model_obj)

        # Fallback for other types
        return self.get_model_class_name(model_config)


    def clone_model(self, model: Any) -> Any:
        """
        Clone model using appropriate method for the framework.

        Uses framework-specific cloning when available, falls back to copy.deepcopy.
        """

        # Try sklearn clone first
        if SKLEARN_AVAILABLE:
            try:
                from sklearn.base import BaseEstimator
                if isinstance(model, BaseEstimator):
                    return sklearn_clone(model)
            except Exception:
                pass

        # Try TensorFlow/Keras cloning
        try:
            if hasattr(model, '_get_trainable_state'):  # Keras model
                # For Keras models, we need to rebuild
                if hasattr(model, 'get_config') and hasattr(model.__class__, 'from_config'):
                    # Use Any to bypass typing issues with dynamic model types
                    model_config = getattr(model, 'get_config')()
                    model_class = model.__class__
                    cloned_model = getattr(model_class, 'from_config')(model_config)
                    return cloned_model
        except Exception:
            pass

        # # Try PyTorch cloning
        # try:
        #     if hasattr(model, 'state_dict'):  # PyTorch model
        #         import torch
        #         cloned_model = copy.deepcopy(model)
        #         return cloned_model
        # except Exception:
        #     pass

        # Fallback to deep copy
        try:
            return copy.deepcopy(model)
        except Exception as e:
            print(f"⚠️ Could not clone model: {e}")
            return model  # Return original if cloning fails

    def get_model_class_name(self, model: Any) -> str:
        """Get the class name of a model."""
        if inspect.isclass(model):
            return f"{model.__qualname__}"

        if inspect.isfunction(model) or inspect.isbuiltin(model):
            return f"{model.__name__}"

        else:
            return str(type(model).__name__)

    def extract_classname_from_config(self, model_config: Dict[str, Any]) -> str:
        """
        Extract the classname based on the model declared in config or instance.__class__.__name__ or function name.
        """

        # Extract model instance
        model_instance = self._get_model_instance_from_config(model_config)

        if model_instance is not None:
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
        """
        Helper to extract model instance from various config formats.
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
                    elif '_runtime_instance' in model_obj:
                        return model_obj['_runtime_instance']
                    else:
                        return model_obj
                else:
                    return model_obj
        else:
            return model_config

        return None

    def sanitize_model_name(self, name: str) -> str:
        """
        Sanitize model name for use in file paths and identifiers.
        """
        # Replace problematic characters
        sanitized = name.replace('(', '_').replace(')', '_')
        sanitized = sanitized.replace('=', '_').replace(',', '_')
        sanitized = sanitized.replace(' ', '_').replace('/', '_')
        sanitized = sanitized.replace('\\', '_').replace(':', '_')

        # Remove multiple underscores
        while '__' in sanitized:
            sanitized = sanitized.replace('__', '_')

        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')

        return sanitized

    def create_model_identifiers(
        self,
        model_config: Dict[str, Any],
        runner: 'PipelineRunner',
        step: int,
        config_id: str,
        fold_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create all model identifiers according to user specifications:

        - classname: based on model declared in config or instance.__class__.__name__ or function
        - name: either custom name defined in config if exists or the classname
        - model_id: name + operation counter (unique for run)
        - model_uuid: model_id + fold + step + config_id (unique in predictions)
        """

        # Extract base info
        classname = self.extract_classname_from_config(model_config)
        name = self.extract_name_from_config(model_config)
        custom_name = model_config.get('name') if isinstance(model_config, dict) else None

        # Create IDs
        model_id = self.create_model_id(name, runner)
        model_uuid = self.create_model_uuid(model_id, runner, step, config_id, fold_idx)

        # Create display name for printing
        display_name = model_id
        if fold_idx is not None:
            display_name += f"_fold{fold_idx}"

        return {
            'classname': classname,
            'name': name,
            'model_id': model_id,
            'model_uuid': model_uuid,
            'custom_name': custom_name or '',
            'display_name': display_name
        }

    def is_model_serializable(self, model: Any) -> bool:
        """
        Check if a model can be serialized with pickle.
        """
        try:
            import pickle
            pickle.dumps(model)
            return True
        except Exception:
            return False

    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """
        Get comprehensive information about a model.
        """
        info = {
            'class_name': self.get_model_class_name(model),
            'module': getattr(model.__class__, '__module__', 'unknown'),
            'serializable': self.is_model_serializable(model),
            'has_fit': hasattr(model, 'fit'),
            'has_predict': hasattr(model, 'predict'),
            'has_get_params': hasattr(model, 'get_params'),
            'has_set_params': hasattr(model, 'set_params'),
        }

        # Try to get parameter info
        if hasattr(model, 'get_params'):
            try:
                info['params'] = model.get_params()
            except Exception:
                info['params'] = {}
        else:
            info['params'] = {}

        return info

    def validate_model(self, model: Any) -> List[str]:
        """
        Validate that a model has the required interface.

        Returns a list of validation errors (empty if valid).
        """
        errors = []

        if not hasattr(model, 'fit'):
            errors.append("Model must have a 'fit' method")

        if not hasattr(model, 'predict'):
            errors.append("Model must have a 'predict' method")

        return errors