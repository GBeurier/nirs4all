"""
Model Utilities - Simple utilities for model operations

This module contains all the utility functions for model naming, cloning,
score calculation and other model operations that were scattered throughout
the original controller. This keeps the main controller clean and focused.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np
import copy
import inspect

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

    def extract_core_name(self, model_config: Dict[str, Any]) -> str:
        """
        Extract Core Name: User-provided name or class name.
        This is the base name provided by the user or derived from the class.
        """
        print(">>>> model_config:", model_config)
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

    def create_simple_name(self, core_name: str, fold_idx: Optional[int] = None) -> str:
        """
        Create Simple Name: Core Name + fold number if applicable.
        Used for printing results when there's no ambiguity on config and step.
        Examples: 'PLSRegression', 'PLSRegression_fold_0'
        """
        if fold_idx is not None:
            return f"{core_name}_fold_{fold_idx}"
        return core_name

    def create_display_name_for_fold(self, local_name: str, fold_idx: Optional[int] = None) -> str:
        """
        Create display name for fold training: Local Name + Fold:X format.
        Used for printing during fold training.
        Examples: 'PLSRegression_17 Fold:0', 'nicon_25 Fold:1'
        """
        if fold_idx is not None:
            return f"{local_name} Fold:{fold_idx}"
        return local_name

    def create_local_name(self, core_name: str, op_counter: int) -> str:
        """
        Create Local Name: Core Name + operation counter.
        Unique within a config context. Used as base for saving models.
        Examples: 'PLSRegression_17', 'MyCustomModel_5'
        """
        return f"{core_name}_{op_counter}"

    def create_model_file_name(self, local_name: str, step: int) -> str:
        """
        Create Model File Name: Step + Local Name + .pkl
        Used for saving model files.
        Examples: '3_PLSRegression_12.pkl', '4_MyCustomModel_5.pkl'
        """
        return f"{step}_{local_name}.pkl"

    def create_uuid(self, local_name: str, step: int, fold_idx: Optional[int],
                    dataset_name: str, config_id: str) -> str:
        """
        Create UUID: Global unique identifier for database keys.
        Format: Step_LocalName_fold{X}_step{Y}_{dataset}_{config}
        Examples: '4_PLSRegression_3_fold0_step0_regression_config_7dbfba05'
        """
        uuid_parts = [f"{step}_{local_name}"]

        if fold_idx is not None:
            uuid_parts.append(f"fold{fold_idx}")

        uuid_parts.extend([f"step{step}", dataset_name, config_id])
        return "_".join(uuid_parts)

    def get_model_names(self, model_config: Dict[str, Any], model: Any, runner: 'PipelineRunner',
                        step: int, fold_idx: Optional[int] = None,
                        dataset_name: str = "unknown") -> Dict[str, str]:
        """
        Generate all model names at once for consistency.
        Returns a dictionary with all naming variants.
        """
        # Extract core name
        core_name = self.extract_core_name(model_config)
        print(">>>> Core Name:", core_name)

        # Get operation counter
        op_counter = runner.next_op()

        # Get config ID
        config_id = getattr(runner.saver, 'pipeline_name', 'unknown') if hasattr(runner, 'saver') else 'unknown'
        print(">>>> Config ID:", config_id)

        # Generate all names
        simple_name = self.create_simple_name(core_name, fold_idx)
        local_name = self.create_local_name(core_name, op_counter)
        model_file_name = self.create_model_file_name(local_name, step)
        uuid = self.create_uuid(local_name, step, fold_idx, dataset_name, config_id)

        return {
            'core_name': core_name,
            'simple_name': simple_name,
            'local_name': local_name,
            'model_file_name': model_file_name,
            'uuid': uuid,
            'op_counter': op_counter
        }

    # Backward compatibility methods for legacy code
    def extract_name_from_config(self, model_config: Dict[str, Any]) -> str:
        """Legacy method - returns core_name for backward compatibility."""
        return self.extract_core_name(model_config)

    def create_model_id(self, name: str, runner: 'PipelineRunner') -> str:
        """Legacy method - returns local_name for backward compatibility."""
        op_counter = runner.next_op()
        return self.create_local_name(name, op_counter)

    def create_model_uuid(self, model_id: str, runner: 'PipelineRunner',
                          step: int, config_id: str, fold_idx: Optional[int] = None) -> str:
        """Legacy method - creates UUID from existing model_id."""
        # Extract parts from legacy model_id format
        parts = model_id.split('_')
        if len(parts) >= 2:
            core_name = '_'.join(parts[:-1])  # Everything except last part
            op_counter = int(parts[-1])  # Last part is op_counter
            local_name = self.create_local_name(core_name, op_counter)
        else:
            local_name = model_id

        dataset_name = "unknown"  # Default for legacy calls
        return self.create_uuid(local_name, step, fold_idx, dataset_name, config_id)

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
        if inspect.isclass(model):
            return f"{model.__qualname__}"

        if inspect.isfunction(model) or inspect.isbuiltin(model):
            return f"{model.__name__}"

        # if hasattr(model, '__class__'):
            # return model.__class__.__name__
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