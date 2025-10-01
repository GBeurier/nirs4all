"""
Model Utilities - Simple utilities for model operations

This module contains all the utility functions for model naming, cloning,
score calculation and other model operations that were scattered throughout
the original controller. This keeps the main controller clean and focused.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np
import copy

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner

try:
    from sklearn.base import clone as sklearn_clone
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ModelUtils:
    """
    Simple model utilities for naming, cloning, and scoring.

    Centralizes all model-related utility functions that were
    spread across the original monolithic controller.
    """

    def __init__(self):
        pass

    def create_model_id(self, name: str, runner: 'PipelineRunner') -> str:
        """
        Create model_id: name + operation counter (unique for a run).
        This is the UNIQUE ID of the model in the run.
        """
        op_counter = runner.next_op()
        model_id = f"{name}_{op_counter}"
        return model_id

    def create_model_uuid(
        self,
        model_id: str,
        runner: 'PipelineRunner',
        step: int,
        config_id: str,
        fold_idx: Optional[int] = None
    ) -> str:
        """
        Create model_uuid: model_id + fold + step + config_id.
        This is the unique ID in predictions.
        """

        # Build UUID parts
        uuid_parts = [model_id]

        if fold_idx is not None:
            uuid_parts.append(f"fold{fold_idx}")

        uuid_parts.append(f"step{step}")
        uuid_parts.append(config_id)

        model_uuid = "_".join(uuid_parts)
        return model_uuid

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
                model_config = model.get_config()
                model_class = model.__class__
                cloned_model = model_class.from_config(model_config)
                return cloned_model
        except Exception:
            pass

        # Try PyTorch cloning
        try:
            if hasattr(model, 'state_dict'):  # PyTorch model
                import torch
                cloned_model = copy.deepcopy(model)
                return cloned_model
        except Exception:
            pass

        # Fallback to deep copy
        try:
            return copy.deepcopy(model)
        except Exception as e:
            print(f"⚠️ Could not clone model: {e}")
            return model  # Return original if cloning fails

    def get_model_class_name(self, model: Any) -> str:
        """Get the class name of a model."""
        if hasattr(model, '__class__'):
            return model.__class__.__name__
        else:
            return str(type(model).__name__)

    def calculate_scores(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str = "auto"
    ) -> Dict[str, float]:
        """
        Calculate scores for the predictions.

        Automatically detects regression vs classification and calculates
        appropriate metrics.
        """

        if not SKLEARN_AVAILABLE:
            return {}

        # Ensure arrays are numpy and flatten
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        # Auto-detect task type if needed
        if task_type == "auto":
            task_type = self._detect_task_type(y_true)

        scores = {}

        try:
            if task_type == "regression":
                # Regression metrics
                scores['mse'] = mean_squared_error(y_true, y_pred)
                scores['rmse'] = np.sqrt(scores['mse'])
                scores['mae'] = mean_absolute_error(y_true, y_pred)
                scores['r2'] = r2_score(y_true, y_pred)

            elif task_type == "classification":
                # Classification metrics
                scores['accuracy'] = accuracy_score(y_true, y_pred)
                scores['f1'] = f1_score(y_true, y_pred, average='weighted')
                scores['precision'] = precision_score(y_true, y_pred, average='weighted')
                scores['recall'] = recall_score(y_true, y_pred, average='weighted')

        except Exception as e:
            print(f"⚠️ Error calculating scores: {e}")

        return scores

    def _detect_task_type(self, y: np.ndarray) -> str:
        """
        Detect if this is a regression or classification task.
        """
        y = np.asarray(y).flatten()

        # Check if all values are integers and within a reasonable range for classification
        if np.all(y == y.astype(int)) and len(np.unique(y)) < 50:
            return "classification"
        else:
            return "regression"

    def get_best_metric_for_task(self, task_type: str) -> tuple[str, bool]:
        """
        Get the best metric for a given task type.

        Returns:
            tuple: (metric_name, higher_is_better)
        """
        if task_type == "regression":
            return "rmse", False  # Lower RMSE is better
        elif task_type == "classification":
            return "accuracy", True  # Higher accuracy is better
        else:
            return "rmse", False  # Default

    def format_scores(self, scores: Dict[str, float]) -> str:
        """Format scores dictionary into a readable string."""
        if not scores:
            return "no scores"

        formatted = []
        for metric, value in scores.items():
            if isinstance(value, (int, float)):
                formatted.append(f"{metric}={value:.4f}")
            else:
                formatted.append(f"{metric}={value}")

        return ", ".join(formatted)

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

    def extract_name_from_config(self, model_config: Dict[str, Any]) -> str:
        """
        Extract the name: either custom name defined in config if exists or the classname.
        """

        # Check for custom name first
        if isinstance(model_config, dict) and 'name' in model_config:
            return model_config['name']

        # Fallback to classname
        return self.extract_classname_from_config(model_config)

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