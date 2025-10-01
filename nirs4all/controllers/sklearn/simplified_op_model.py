"""
Simplified Sklearn Model Controller - Inherits from BaseModelController

This is a much simpler version of the sklearn controller that uses
the new BaseModelController architecture. It only needs to implement
the framework-specific methods.
"""

from typing import Any, Dict, Tuple, Optional
import numpy as np
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

from ..models.base_model_controller import BaseModelController
from nirs4all.controllers.registry import register_controller


# Do not register this controller to avoid conflicts
# @register_controller
class SimplifiedSklearnController(BaseModelController):
    """Simplified Sklearn controller using the new BaseModelController."""

    priority = 5  # Higher priority than generic transformers

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Match sklearn estimators and model dictionaries with sklearn models."""
        # Check if step contains a model key with sklearn object
        if isinstance(step, dict) and 'model' in step:
            model = step['model']
            # Handle nested model format
            if isinstance(model, dict):
                if 'model' in model:
                    model = model['model']
                elif '_runtime_instance' in model:
                    model = model['_runtime_instance']

            if isinstance(model, BaseEstimator):
                return is_regressor(model) or is_classifier(model) or hasattr(model, 'predict')

        # Check direct sklearn objects
        if isinstance(step, BaseEstimator):
            return is_regressor(step) or is_classifier(step) or hasattr(step, 'predict')

        # Check operator if provided
        if operator is not None and isinstance(operator, BaseEstimator):
            return is_regressor(operator) or is_classifier(operator) or hasattr(operator, 'predict')

        return False

    def _get_model_instance(self, model_config: Dict[str, Any]) -> BaseEstimator:
        """Create sklearn model instance from configuration."""
        if 'model_instance' in model_config:
            model = model_config['model_instance']
            if isinstance(model, BaseEstimator):
                return model

        # If we have a model class and parameters, instantiate it
        if 'model_class' in model_config:
            model_class = model_config['model_class']
            model_params = model_config.get('model_params', {})
            return model_class(**model_params)

        raise ValueError("Could not create model instance from configuration")

    def _train_model(
        self,
        model: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> BaseEstimator:
        """Train sklearn model."""
        verbose = kwargs.get('verbose', 0)

        # Apply any training parameters
        train_params = {k: v for k, v in kwargs.items() if k != 'verbose'}
        if train_params:
            # Filter out parameters that don't exist in the model
            valid_params = {}
            model_params = model.get_params()
            for key, value in train_params.items():
                if key in model_params:
                    valid_params[key] = value

            if valid_params:
                model.set_params(**valid_params)

        # Fit the model
        model.fit(X_train, y_train.ravel())  # Ensure y is 1D for sklearn

        # Print training info if verbose
        if verbose > 0:
            task_type = self.model_utils._detect_task_type(y_train)
            y_train_pred = self._predict_model(model, X_train)
            train_scores = self.model_utils.calculate_scores(y_train, y_train_pred, task_type)

            if train_scores:
                best_metric, higher_is_better = self.model_utils.get_best_metric_for_task(task_type)
                best_score = train_scores.get(best_metric)
                if best_score is not None:
                    direction = "↑" if higher_is_better else "↓"
                    score_str = self.model_utils.format_scores(train_scores)
                    print(f"✅ {model.__class__.__name__} - train: {best_metric}={best_score:.4f} {direction} ({score_str})")

        return model

    def _predict_model(self, model: BaseEstimator, X: np.ndarray) -> np.ndarray:
        """Generate predictions with sklearn model."""
        predictions = model.predict(X)

        # Ensure predictions are in the correct shape
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        return predictions

    def _prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        context: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for sklearn (ensure 2D X and 2D y for consistency)."""
        if X is None or y is None:
            return None, None

        # Ensure X is 2D
        if X.ndim > 2:
            # Flatten extra dimensions
            X = X.reshape(X.shape[0], -1)
        elif X.ndim == 1:
            X = X.reshape(-1, 1)

        # Ensure y is 2D for consistency with predictions
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim > 2:
            y = y.reshape(y.shape[0], -1)

        return X, y

    def _evaluate_model(self, model: BaseEstimator, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Evaluate sklearn model for optimization (returns score to minimize)."""
        # Ensure y_val is 1D for sklearn functions
        y_val_1d = y_val.ravel() if y_val.ndim > 1 else y_val

        try:
            # Use cross-validation for evaluation
            if is_classifier(model):
                # For classifiers, use negative accuracy (to minimize)
                scores = cross_val_score(model, X_val, y_val_1d, cv=3, scoring='accuracy')
                return -np.mean(scores)  # Negative because we want to minimize
            elif is_regressor(model):
                # For regressors, use negative MSE (to minimize)
                scores = cross_val_score(model, X_val, y_val_1d, cv=3, scoring='neg_mean_squared_error')
                return -np.mean(scores)  # Already negative, so negate to get positive MSE
            else:
                # Default: use model's score method if available
                if hasattr(model, 'score'):
                    score = model.score(X_val, y_val_1d)
                    return -score  # Negative to minimize
                else:
                    # Fallback: MSE for any model
                    y_pred = model.predict(X_val)
                    return mean_squared_error(y_val_1d, y_pred)

        except Exception as e:
            print(f"⚠️ Error in model evaluation: {e}")
            # Fallback evaluation
            try:
                y_pred = model.predict(X_val)
                return mean_squared_error(y_val_1d, y_pred)
            except Exception:
                return float('inf')  # Return worst possible score

    def _sample_hyperparameters(self, trial, finetune_params: Dict[str, Any]) -> Dict[str, Any]:
        """Sample hyperparameters specific to sklearn models."""
        params = {}

        # Get model-specific parameters from finetune config
        model_params = finetune_params.get('model_params', {})

        for param_name, param_config in model_params.items():
            if isinstance(param_config, dict):
                param_type = param_config.get('type', 'float')

                if param_type == 'int':
                    low = param_config.get('low', 1)
                    high = param_config.get('high', 100)
                    params[param_name] = trial.suggest_int(param_name, low, high)

                elif param_type == 'float':
                    low = param_config.get('low', 1e-5)
                    high = param_config.get('high', 1.0)
                    log = param_config.get('log', False)
                    params[param_name] = trial.suggest_float(param_name, low, high, log=log)

                elif param_type == 'categorical':
                    choices = param_config.get('choices', [])
                    if choices:
                        params[param_name] = trial.suggest_categorical(param_name, choices)

        return params

    def get_preferred_layout(self) -> str:
        """Return the preferred data layout for sklearn models."""
        return "2d"