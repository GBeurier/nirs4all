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

from ..models.base_model import BaseModelController
from nirs4all.controllers.registry import register_controller
from nirs4all.utils.emoji import ARROW_UP, ARROW_DOWN
from .utilities import ModelControllerUtils as ModelUtils
from .factory import ModelFactory

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext


@register_controller
class SklearnModelController(BaseModelController):
    """Controller for scikit-learn models.

    This controller handles sklearn models with support for training on 2D data,
    cross-validation, hyperparameter tuning with Optuna, model persistence,
    and integration with the nirs4all pipeline.

    Attributes:
        priority (int): Controller priority (6) - higher than TransformerMixin to
            prioritize supervised models over transformers.
    """

    priority = 6  # Higher priority than TransformerMixin (10) to win matching

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Match sklearn estimators and model dictionaries with sklearn models.

        Prioritizes supervised models (regressors and classifiers) over transformers
        by checking for predict methods and using sklearn's is_regressor/is_classifier.

        Args:
            step (Any): Pipeline step to check, can be a dict with 'model' key or
                BaseEstimator instance.
            operator (Any): Optional operator object to check if it's a BaseEstimator.
            keyword (str): Pipeline keyword (unused in this implementation).

        Returns:
            bool: True if the step matches a sklearn estimator (regressor, classifier,
                or has predict method), False otherwise.
        """
        # Check if step contains a model key with sklearn object
        if isinstance(step, dict) and 'model' in step:
            model = step['model']

            if isinstance(model, BaseEstimator):
                # Prioritize supervised models (need both X and y) over transformers
                from sklearn.base import is_regressor, is_classifier
                return is_regressor(model) or is_classifier(model) or hasattr(model, 'predict')

            # Handle dictionary config for model
            if isinstance(model, dict) and 'class' in model:
                class_name = model['class']
                if isinstance(class_name, str) and 'sklearn' in class_name:
                    return True

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
        """Create sklearn model instance from configuration.

        Handles multiple configuration formats:
        - Direct model_instance (class or instance)
        - New serialization format with 'function', 'class', or 'import' keys
        - Legacy format with nested 'model' dict containing 'class' key

        Args:
            dataset (SpectroDataset): Dataset for context-aware parameter building.
            model_config (Dict[str, Any]): Model configuration containing model class,
                instance, or serialization info with optional params.
            force_params (Optional[Dict[str, Any]]): Parameters to override or merge
                with existing model parameters. Defaults to None.

        Returns:
            BaseEstimator: Instantiated sklearn model with configured parameters.

        Raises:
            ValueError: If model instance cannot be created from the configuration.
        """
        # If we have a model_instance (class or instance) and force_params, we need to rebuild with new params
        if 'model_instance' in model_config:
            model = model_config['model_instance']

            # If no force_params and it's already an instance, just return it
            if force_params is None and isinstance(model, BaseEstimator):
                return model

            # If we have force_params, we need to get the class and rebuild
            if force_params:
                # Get the model class (either from instance or if it's already a class)
                if isinstance(model, type):
                    model_class = model
                else:
                    model_class = type(model)

                # Rebuild with force_params
                return ModelFactory.build_single_model(model_class, dataset, force_params)

        # Handle new serialization formats: {'function': ..., 'params': ...} or {'class': ..., 'params': ...}
        if any(key in model_config for key in ('function', 'class', 'import')):
            params = model_config.get('params', {})
            if force_params:
                params.update(force_params)
            return ModelFactory.build_single_model(model_config, dataset, params)

        # Handle old format: model_config['model']['class']
        if 'model' in model_config and 'class' in model_config['model']:
            model_class = model_config['model']['class']
            model_params = model_config.get('model_params', {})
            if force_params:
                model_params.update(force_params)
            model = ModelFactory.build_single_model(model_class, dataset, model_params)
            return model

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
        """Train sklearn model with score tracking.

        Trains the model on training data, validates parameters against model's
        available parameters, and optionally calculates training and validation
        scores based on verbosity level.

        Args:
            model (BaseEstimator): Sklearn model instance to train (already cloned).
            X_train (np.ndarray): Training features, shape (n_samples, n_features).
            y_train (np.ndarray): Training targets, shape (n_samples, n_targets).
            X_val (Optional[np.ndarray]): Validation features for score calculation.
                Defaults to None.
            y_val (Optional[np.ndarray]): Validation targets for score calculation.
                Defaults to None.
            **kwargs: Training parameters including 'verbose' level for output control
                and 'task_type' for metric calculation.

        Returns:
            BaseEstimator: Trained sklearn model instance.

        Note:
            - y_train is automatically raveled to 1D for sklearn compatibility
            - Only valid model parameters are applied from kwargs
            - Training and validation scores are displayed when verbose > 1
        """

        train_params = kwargs
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
                    # print(f"{WARNING}Parameter {key} not found in model {model.__class__.__name__}")

            if valid_params:
                trained_model.set_params(**valid_params)

        # Fit the model
        trained_model.fit(X_train, y_train.ravel())  # Ensure y is 1D for sklearn

        # Always calculate and display final test scores, regardless of verbose level
        # But control the detail level based on verbose

        if verbose > 1:
            # Get task_type from train_params (passed by base controller)
            task_type = kwargs.get('task_type')
            if task_type is None:
                raise ValueError("task_type must be provided in train_params")

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
                    direction = ARROW_UP if higher_is_better else ARROW_DOWN
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
                        direction = ARROW_UP if higher_is_better else ARROW_DOWN
                        all_scores_str = ModelUtils.format_scores(val_scores)
                        # print(f"✅ {trained_model.__class__.__name__} - validation: {best_metric}={best_score:.4f} {direction} ({all_scores_str})")

        return trained_model

    def _predict_model(self, model: BaseEstimator, X: np.ndarray) -> np.ndarray:
        """Generate predictions with sklearn model.

        Args:
            model (BaseEstimator): Trained sklearn model instance.
            X (np.ndarray): Input features for prediction, shape (n_samples, n_features).

        Returns:
            np.ndarray: Model predictions, reshaped to (n_samples, n_outputs) format
                for consistency with pipeline expectations.
        """
        predictions = model.predict(X)

        # Ensure predictions are in the correct shape
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        return predictions

    def _prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        context: 'ExecutionContext'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for sklearn (ensure 2D X and 2D y for consistency).

        Reshapes input data to ensure proper dimensionality for sklearn models:
        - X is reshaped to 2D (n_samples, n_features)
        - y is reshaped to 2D (n_samples, n_targets) for consistency

        Args:
            X (np.ndarray): Input features, can be 1D, 2D, or higher dimensional.
            y (np.ndarray): Target values, can be None for prediction-only scenarios.
            context (ExecutionContext): Pipeline context (unused in this implementation).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Prepared (X, y) arrays in proper format,
                or (None, None) if X is None.

        Note:
            - Extra dimensions in X are flattened to (n_samples, n_features)
            - y can be None for prediction-only scenarios
        """
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
        """Evaluate sklearn model using cross-validation.

        Uses task-appropriate metrics:
        - Classifiers: negative accuracy (for minimization)
        - Regressors: negative MSE (for minimization)
        - Others: model's score method or fallback to MSE

        Args:
            model (BaseEstimator): Sklearn model to evaluate.
            X_val (np.ndarray): Validation features, shape (n_samples, n_features).
            y_val (np.ndarray): Validation targets, shape (n_samples, n_targets).

        Returns:
            float: Evaluation score (negative for maximization metrics to support
                minimization-based optimization). Returns inf on error.

        Note:
            - Uses 3-fold cross-validation
            - y_val is automatically raveled to 1D for sklearn compatibility
            - Fallback to MSE if cross-validation fails
        """
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
            print(f"{WARNING}Error in model evaluation: {e}")
            # Fallback evaluation
            try:
                y_pred = model.predict(X_val)
                return mean_squared_error(y_val_1d, y_pred)
            except Exception:
                return float('inf')  # Return worst possible score

    def get_preferred_layout(self) -> str:
        """Return the preferred data layout for sklearn models.

        Returns:
            str: Data layout preference, always '2d' for sklearn models which
                expect (n_samples, n_features) input format.
        """
        return "2d"

    def _clone_model(self, model: BaseEstimator) -> BaseEstimator:
        """Clone sklearn model using sklearn's clone function.

        Uses sklearn.base.clone() which creates a new instance with the same
        parameters but without fitted attributes. This is the recommended way
        to clone sklearn estimators.

        Args:
            model (BaseEstimator): Sklearn model instance to clone.

        Returns:
            BaseEstimator: Cloned sklearn model with same parameters but fresh state.

        Raises:
            RuntimeError: If sklearn is not available.
        """
        try:
            from sklearn.base import clone as sklearn_clone
            return sklearn_clone(model)
        except ImportError:
            raise RuntimeError("sklearn is required to clone sklearn models")

    def _sample_hyperparameters(self, trial, finetune_params: Dict[str, Any]) -> Dict[str, Any]:
        """Sample hyperparameters specific to sklearn models.

        Extends base hyperparameter sampling with sklearn-specific handling.
        Currently delegates to parent implementation but provides extension point
        for sklearn-specific cases like random_state preservation.

        Args:
            trial: Optuna trial object for hyperparameter sampling.
            finetune_params (Dict[str, Any]): Hyperparameter search space configuration.

        Returns:
            Dict[str, Any]: Sampled hyperparameters for model instantiation.
        """
        params = super()._sample_hyperparameters(trial, finetune_params)

        # Add sklearn-specific parameter handling if needed
        # For example, handle special cases like random_state preservation

        return params

    def execute(
        self,
        step_info: 'ParsedStep',
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        runtime_context: 'RuntimeContext',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, bytes]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple['ExecutionContext', List[Tuple[str, bytes]]]:
        """Execute sklearn model controller with score management.

        Main entry point for sklearn model execution in the pipeline. Sets the
        preferred data layout to '2d' and delegates to parent execute method.

        Args:
            step_info: Parsed step containing model configuration and operator.
            dataset (SpectroDataset): Dataset containing features and targets.
            context (ExecutionContext): Pipeline execution context with state info.
            runtime_context (RuntimeContext): Runtime context managing execution state.
            source (int): Source index for multi-source pipelines. Defaults to -1.
            mode (str): Execution mode ('train' or 'predict'). Defaults to 'train'.
            loaded_binaries (Optional[List[Tuple[str, bytes]]]): Pre-loaded model
                binaries for prediction mode. Defaults to None.
            prediction_store (Optional[Any]): Store for managing predictions.
                Defaults to None.

        Returns:
            Tuple[ExecutionContext, List[Tuple[str, bytes]]]: Updated context and
                list of model binaries (name, serialized_model) for persistence.

        Note:
            - Automatically sets context['layout'] = '2d' for sklearn compatibility
            - Inherits full training, evaluation, and prediction logic from BaseModelController
        """
        # Set layout preference for sklearn models
        context = context.with_layout(self.get_preferred_layout())

        # Call parent execute method
        return super().execute(step_info, dataset, context, runtime_context, source, mode, loaded_binaries, prediction_store)


