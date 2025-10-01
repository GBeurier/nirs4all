"""
Sklearn Model Controller - Concrete implementation for scikit-learn models

This controller extends the AbstractModelController to provide scikit-learn
specific model instantiation, training, and prediction methods.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from nirs4all.controllers.models.base_model_controller import BaseModelController
from nirs4all.controllers.registry import register_controller
from nirs4all.utils.model_utils import TaskType


@register_controller
class SklearnModelController(BaseModelController):
    """
    Concrete model controller for scikit-learn compatible models.

    This controller handles the framework-specific operations for scikit-learn
    models while using the modular architecture for data management, CV strategies,
    prediction handling, and results storage.
    """

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Check if this controller matches the given step and operator."""
        # Check if step contains a model key with sklearn object
        if isinstance(step, dict) and 'model' in step:
            model = step['model']

            # Handle nested model format: {'model': {'name': 'X', 'model': PLSRegression()}}
            if isinstance(model, dict) and 'model' in model:
                model = model['model']

                # Handle even deeper nesting: {'model': {'name': 'X', 'model': {'_runtime_instance': PLSRegression()}}}
                if isinstance(model, dict) and '_runtime_instance' in model:
                    model = model['_runtime_instance']

            # Handle serialized model format
            if isinstance(model, dict) and '_runtime_instance' in model:
                model = model['_runtime_instance']

            if hasattr(model, '__module__') and 'sklearn' in model.__module__:
                return True

        # Check direct sklearn objects
        if hasattr(step, '__module__') and 'sklearn' in step.__module__:
            return True

        # Check operator if provided
        if operator is not None and hasattr(operator, '__module__') and 'sklearn' in operator.__module__:
            return True

        return False

    @classmethod
    def use_multi_source(cls) -> bool:
        """Check if the operator supports multi-source datasets."""
        return False

    @classmethod
    def get_supported_frameworks(cls) -> list[str]:
        """Get the frameworks supported by this controller."""
        return ['sklearn']

    def __init__(self):
        """Initialize the scikit-learn model controller."""
        super().__init__()

    def _get_model_instance(self, model_config: Dict[str, Any]) -> Any:
        """
        Create a model instance from configuration.

        Args:
            model_config: Model configuration dictionary

        Returns:
            Instantiated model object
        """
        # Check if we already have a model instance (from the refactored AbstractModelController)
        if 'model_instance' in model_config:
            return model_config['model_instance']

        # Fallback to the original class-based instantiation for backward compatibility
        model_class_path = model_config.get('model_class')
        if not model_class_path:
            raise ValueError("model_class or model_instance must be specified in model configuration")

        # Import the model class dynamically
        try:
            module_path, class_name = model_class_path.rsplit('.', 1)
            import importlib
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Could not import model class {model_class_path}: {e}")

        # Get model parameters
        model_params = model_config.get('model_params', {})

        # Instantiate the model
        try:
            model = model_class(**model_params)
            return model
        except Exception as e:
            raise ValueError(f"Could not instantiate model {model_class_path}: {e}")

    def _train_model(
        self,
        model: Any,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        train_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Train the scikit-learn model.

        Args:
            model: Model instance to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            train_params: Training parameters

        Returns:
            Trained model
        """
        train_params = train_params or {}

        # Scikit-learn models use fit method
        if hasattr(model, 'fit'):
            # Prepare training data
            X_train_prep, y_train_prep = self._prepare_data(X_train, y_train, {})

            # Fit the model
            if X_val is not None and y_val is not None and hasattr(model, 'fit') and 'early_stopping' in train_params:
                # Some models support validation data
                X_val_prep, y_val_prep = self._prepare_data(X_val, y_val, {})
                model.fit(X_train_prep, y_train_prep, eval_set=[(X_val_prep, y_val_prep)], **train_params)
            else:
                # Filter out parameters that the model doesn't support
                supported_params = {}
                if hasattr(model, 'fit'):
                    import inspect
                    fit_signature = inspect.signature(model.fit)
                    for param_name, param_value in train_params.items():
                        if param_name in fit_signature.parameters:
                            supported_params[param_name] = param_value
                else:
                    supported_params = train_params

                model.fit(X_train_prep, y_train_prep, **supported_params)

            return model
        else:
            raise ValueError("Model does not have a fit method")

    def _predict_model(self, model: Any, X: Any) -> np.ndarray:
        """
        Generate predictions using the scikit-learn model.

        Args:
            model: Trained model
            X: Input features

        Returns:
            Predictions as numpy array
        """
        if hasattr(model, 'predict'):
            # Prepare input data
            X_prep, _ = self._prepare_data(X, np.zeros(len(X)), {})  # Dummy y for interface

            # Generate predictions
            predictions = model.predict(X_prep)

            # Ensure predictions are numpy array
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)

            return predictions
        else:
            raise ValueError("Model does not have a predict method")

    def _prepare_data(self, X: Any, y: Any, context: Dict[str, Any]) -> Tuple[Any, Any]:
        """
        Prepare data in scikit-learn format.

        Args:
            X: Input features
            y: Target values
            context: Pipeline context

        Returns:
            Tuple of (X_prepared, y_prepared)
        """
        # For scikit-learn, we typically just ensure X and y are in the right format
        # More complex preprocessing would be handled by separate transformers

        # Ensure X is 2D
        if hasattr(X, 'shape') and len(X.shape) == 1:
            X = X.reshape(-1, 1)
        elif not hasattr(X, 'shape'):
            X = np.array(X).reshape(-1, 1)

        # Ensure y is 1D for most sklearn models
        if hasattr(y, 'shape') and len(y.shape) > 1:
            if y.shape[1] == 1:
                y = y.ravel()
        elif not hasattr(y, 'shape'):
            y = np.array(y)

        return X, y

    def get_preferred_layout(self) -> str:
        """
        Get the preferred data layout for scikit-learn models.

        Returns:
            Layout string ('2d' for sklearn)
        """
        return '2d'

    def _detect_task_type(self, y: np.ndarray) -> TaskType:
        """
        Detect the task type from target values.

        Args:
            y: Target values

        Returns:
            TaskType enum value
        """
        from nirs4all.utils.model_utils import ModelUtils
        return ModelUtils.detect_task_type(y)

    def _evaluate_model(self, model: Any, X_val: Any, y_val: Any) -> float:
        """
        Evaluate model performance (return score to minimize).

        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Score to minimize (lower is better)
        """
        from sklearn.metrics import mean_squared_error

        predictions = self._predict_model(model, X_val)
        task_type = self._detect_task_type(y_val)

        if task_type == TaskType.REGRESSION:
            # For regression, return MSE (lower is better)
            return mean_squared_error(y_val, predictions)
        else:
            # For classification, could return 1 - accuracy or similar
            # For now, use MSE as a general metric
            return mean_squared_error(y_val, predictions)
