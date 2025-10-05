"""
Sklearn Model Controller - Controller for scikit-learn models

This controller handles sklearn models with support for:
- Training on 2D data (samples x features)
- Cross-validation and hyperparameter tuning with Optuna
- Model persistence and prediction storage
- Integration with the nirs4all pipeline

Matches any sklearn model object (estimators with fit/predict methods).
"""

from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.base import is_classifier, is_regressor

from ..models.base_model_controller import BaseModelController
from nirs4all.controllers.registry import register_controller
from nirs4all.utils.model_utils import ModelUtils
from nirs4all.utils.model_builder import ModelBuilderFactory

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset


@register_controller
class SklearnModelController(BaseModelController):
    """Controller for scikit-learn models."""

    priority = 6  # Higher priority than TransformerMixin (10) to win matching

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Match sklearn estimators and model dictionaries with sklearn models."""
        # Check if step contains a model key with sklearn object
        if isinstance(step, dict) and 'model' in step:
            model = step['model']
            # Handle serialized model format
            if isinstance(model, dict) and '_runtime_instance' in model:
                model = model['_runtime_instance']

            if isinstance(model, BaseEstimator):
                # Prioritize supervised models (need both X and y) over transformers
                from sklearn.base import is_regressor, is_classifier
                return is_regressor(model) or is_classifier(model) or hasattr(model, 'predict')

        # Check direct sklearn objects
        if isinstance(step, BaseEstimator):
            from sklearn.base import is_regressor, is_classifier
            return is_regressor(step) or is_classifier(step) or hasattr(step, 'predict')

        # Check operator if provided
        if operator is not None and isinstance(operator, BaseEstimator):
            from sklearn.base import is_regressor, is_classifier
            return is_regressor(operator) or is_classifier(operator) or hasattr(operator, 'predict')

        return False

    def _get_model_instance(self, dataset: 'SpectroDataset', model_config: Dict[str, Any], force_params: Optional[Dict[str, Any]] = None) -> BaseEstimator:
        """Create sklearn model instance from configuration."""
        # print("Creating sklearn model instance from configuration...")
        # print(model_config, force_params)
        # If we have a model class and parameters, instantiate it
        # ModelBuilder.build_single_model(model_config['model']['class'], mo)
        if 'model_instance' in model_config and force_params is None:
            model = model_config['model_instance']
            if isinstance(model, BaseEstimator):
                return model

        if 'model' in model_config and 'class' in model_config['model']:
            model_class = model_config['model']['class']
            model_params = model_config.get('model_params', {})
            if force_params:
                model_params.update(force_params)
            # return model_class(**model_params)
            model = ModelBuilderFactory.build_single_model(model_class, dataset, model_params)
            # print("Created model:", model)
            # print(model.n_components if hasattr(model, 'n_components') else "No n_components")
            return model



        raise ValueError("Could not create model instance from configuration")

    def _train_model(
        self,
        model: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        train_params: Optional[Dict[str, Any]] = None,
    ) -> BaseEstimator:
        """Train sklearn model with score tracking."""

        if train_params is None:
            train_params = {}

        verbose = train_params.get('verbose', 0)

        # if verbose > 1 and train_params:
            # print(f"🔧 Training {model.__class__.__name__} with params: {train_params}")
        # elif verbose > 1:
            # print(f"🔧 Training {model.__class__.__name__}")        # Model is already cloned in base class, just use it directly
        trained_model = model

        # Set additional parameters if provided
        if train_params:
            # Filter out parameters that don't exist in the model
            valid_params = {}
            model_params = trained_model.get_params()
            for key, value in train_params.items():
                if key in model_params:
                    valid_params[key] = value
                # else:
                    # print(f"⚠️ Parameter {key} not found in model {model.__class__.__name__}")

            if valid_params:
                trained_model.set_params(**valid_params)

        # Fit the model
        trained_model.fit(X_train, y_train.ravel())  # Ensure y is 1D for sklearn

        # Always calculate and display final test scores, regardless of verbose level
        # But control the detail level based on verbose

        if verbose > 1:
            task_type = self._detect_task_type(y_train)
            # Show detailed training scores at verbose > 1
            y_train_pred = self._predict_model(trained_model, X_train)
            train_scores = self._calculate_and_print_scores(
                y_train, y_train_pred, task_type, "train",
                trained_model.__class__.__name__, show_detailed_scores=False
            )
            # Display concise training summary
            if train_scores:
                best_metric, higher_is_better = ModelUtils.get_best_score_metric(task_type)
                best_score = train_scores.get(best_metric)
                if best_score is not None:
                    direction = "↑" if higher_is_better else "↓"
                    all_scores_str = ModelUtils.format_scores(train_scores)
                    # print(f"✅ {trained_model.__class__.__name__} - train: {best_metric}={best_score:.4f} {direction} ({all_scores_str})")

            # Validation scores if available
            if X_val is not None and y_val is not None:
                y_val_pred = self._predict_model(trained_model, X_val)
                val_scores = self._calculate_and_print_scores(
                    y_val, y_val_pred, task_type, "validation",
                    trained_model.__class__.__name__, show_detailed_scores=False
                )
                # Display concise validation summary
                if val_scores:
                    best_metric, higher_is_better = ModelUtils.get_best_score_metric(task_type)
                    best_score = val_scores.get(best_metric)
                    if best_score is not None:
                        direction = "↑" if higher_is_better else "↓"
                        all_scores_str = ModelUtils.format_scores(val_scores)
                        # print(f"✅ {trained_model.__class__.__name__} - validation: {best_metric}={best_score:.4f} {direction} ({all_scores_str})")

        return trained_model

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
        if X is None:
            return None, None

        # Ensure X is 2D
        if X.ndim > 2:
            # Flatten extra dimensions
            X = X.reshape(X.shape[0], -1)
        elif X.ndim == 1:
            X = X.reshape(-1, 1)

        # Handle y (can be None for prediction-only scenarios)
        if y is not None:
            # Ensure y is 2D for consistency with predictions
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            elif y.ndim > 2:
                y = y.reshape(y.shape[0], -1)

        return X, y

    def _evaluate_model(self, model: BaseEstimator, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Evaluate sklearn model using cross-validation."""
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

    def get_preferred_layout(self) -> str:
        """Return the preferred data layout for sklearn models."""
        return "2d"

    def _sample_hyperparameters(self, trial, finetune_params: Dict[str, Any]) -> Dict[str, Any]:
        """Sample hyperparameters specific to sklearn models."""
        params = super()._sample_hyperparameters(trial, finetune_params)

        # Add sklearn-specific parameter handling if needed
        # For example, handle special cases like random_state preservation

        return params

    def execute(
        self,
        step: Any,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, bytes]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute sklearn model controller with score management."""
        # Set layout preference for sklearn models
        context = context.copy()
        context['layout'] = self.get_preferred_layout()

        # Call parent execute method
        return super().execute(step, operator, dataset, context, runner, source, mode, loaded_binaries, prediction_store)
