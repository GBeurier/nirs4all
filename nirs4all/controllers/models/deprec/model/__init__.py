"""
Model Management - Handles model instantiation, cloning, and framework operations

This module provides a clean interface for managing machine learning models
across different frameworks (sklearn, tensorflow, pytorch, etc.).
"""

from typing import Any, Dict, Optional, Union
import inspect
from copy import deepcopy
import pickle


class ModelManager:
    """
    Handles model instantiation, cloning, and framework detection.

    This class centralizes model management operations that were previously
    scattered across the monolithic BaseModelController.
    """

    def __init__(self):
        """Initialize the model manager."""
        pass

    def create_model(self, config: Dict[str, Any]) -> Any:
        """
        Create a model instance from configuration.

        Args:
            config: Model configuration dictionary

        Returns:
            Any: Model instance
        """
        if 'model_instance' in config:
            model = config['model_instance']
            # If it's still a string or serialized format, try to get the runtime instance
            if isinstance(model, dict) and '_runtime_instance' in model:
                return model['_runtime_instance']
            elif hasattr(model, '_runtime_instance'):
                return model._runtime_instance  # type: ignore
            return model
        else:
            # This should be implemented by subclasses for specific frameworks
            raise NotImplementedError("Model instantiation must be implemented by framework-specific managers")

    def clone_model(self, model: Any) -> Any:
        """
        Clone a model using framework-specific cloning methods.

        Args:
            model: Model to clone

        Returns:
            Any: Cloned model
        """
        framework = self._detect_framework(model)

        if framework == 'sklearn':
            return self._clone_sklearn_model(model)
        elif framework == 'tensorflow':
            return self._clone_tensorflow_model(model)
        elif framework == 'pytorch':
            return self._clone_pytorch_model(model)
        else:
            # Fallback to deepcopy
            return deepcopy(model)

    def apply_parameters(self, model: Any, params: Dict[str, Any]) -> Any:
        """
        Apply parameters to a model.

        Args:
            model: Model instance
            params: Parameters to apply

        Returns:
            Any: Model with applied parameters
        """
        if hasattr(model, 'set_params') and params:
            try:
                model.set_params(**params)
            except Exception as e:
                print(f"⚠️ Could not apply parameters {params}: {e}")
        return model

    def _clone_sklearn_model(self, model: Any) -> Any:
        """Clone a scikit-learn model."""
        try:
            from sklearn.base import clone
            return clone(model)
        except ImportError:
            return deepcopy(model)

    def _clone_tensorflow_model(self, model: Any) -> Any:
        """Clone a TensorFlow/Keras model."""
        try:
            from tensorflow.keras.models import clone_model
            return clone_model(model)
        except TypeError as e:
            if "Could not locate function" in str(e):
                # The model contains non-serializable functions, return as is
                # This happens with models created from functions like nicon
                return model
            else:
                raise
        except ImportError:
            return deepcopy(model)

    def _clone_pytorch_model(self, model: Any) -> Any:
        """Clone a PyTorch model."""
        try:
            import torch
            return deepcopy(model)  # PyTorch models typically use deepcopy
        except ImportError:
            return deepcopy(model)

    def _detect_framework(self, model: Any) -> str:
        """
        Detect the framework from the model instance.

        Args:
            model: Model instance

        Returns:
            str: Framework name
        """
        if hasattr(model, 'framework'):
            return model.framework

        if inspect.isclass(model):
            model_desc = f"{model.__module__}.{model.__name__}"
        else:
            model_desc = f"{model.__class__.__module__}.{model.__class__.__name__}"

        if 'tensorflow' in model_desc or 'keras' in model_desc:
            return 'tensorflow'
        elif 'torch' in model_desc:
            return 'pytorch'
        elif 'sklearn' in model_desc:
            return 'sklearn'
        else:
            return 'unknown'

    def serialize_model(self, model: Any) -> bytes:
        """
        Serialize a model to bytes.

        Args:
            model: Model instance

        Returns:
            bytes: Serialized model
        """
        try:
            return pickle.dumps(model)
        except Exception as e:
            raise RuntimeError(f"Could not serialize model: {e}") from e

    def deserialize_model(self, model_bytes: bytes) -> Any:
        """
        Deserialize a model from bytes.

        Args:
            model_bytes: Serialized model bytes

        Returns:
            Any: Deserialized model
        """
        try:
            return pickle.loads(model_bytes)
        except Exception as e:
            raise RuntimeError(f"Could not deserialize model: {e}") from e

    def get_model_name(self, model: Any, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Get a descriptive name for the model.

        Args:
            model: Model instance
            config: Optional model configuration

        Returns:
            str: Model name
        """
        # Check for custom name in config
        if config and 'name' in config:
            return config['name']

        # Try to get from model class name
        if hasattr(model, '__class__'):
            return model.__class__.__name__

        return "UnknownModel"

    def get_base_model_name(self, config: Dict[str, Any], trained_model: Any = None) -> str:
        """
        Extract the base model name from configuration or trained model.

        Args:
            config: Model configuration
            trained_model: Optional trained model instance

        Returns:
            str: Base model name
        """
        base_name = None

        # First priority: Check for custom name in configuration
        if isinstance(config, dict) and 'name' in config:
            base_name = config['name']

        # Second priority: Extract from trained model
        if not base_name and trained_model:
            base_name = self.get_model_name(trained_model)

        # Clean up the name (remove step/fold suffixes)
        if base_name:
            if '_step' in base_name:
                base_name = base_name.split('_step')[0]
            if '_fold' in base_name:
                base_name = base_name.split('_fold')[0]

        return base_name or "UnknownModel"
