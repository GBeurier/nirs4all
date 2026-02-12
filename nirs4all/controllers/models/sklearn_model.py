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
import copy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mean_squared_error
from sklearn.base import is_classifier, is_regressor

from ..models.base_model import BaseModelController
from nirs4all.controllers.registry import register_controller
from nirs4all.core.logging import get_logger
from .utilities import ModelControllerUtils as ModelUtils
from .factory import ModelFactory

logger = get_logger(__name__)


def _reset_gpu_memory() -> bool:
    """Reset GPU memory using PyTorch CUDA.

    Clears cached memory and synchronizes GPU operations.
    Useful for preventing memory leaks with GPU-based models like CatBoost.

    Returns:
        bool: True if GPU was successfully reset, False if torch/CUDA unavailable.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return True
    except ImportError:
        pass
    return False

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
        # Check if it is an explicit model configuration
        is_explicit_model = isinstance(step, dict) and 'model' in step

        # Extract the model object/config
        model = step.get('model') if is_explicit_model else step

        # Use Factory to detect framework
        framework = ModelFactory.detect_framework(model)

        # If framework is unknown but we have an operator, try to detect from operator
        if framework == 'unknown' and operator is not None:
            framework = ModelFactory.detect_framework(operator)

        # 1. Safety Net: Explicitly reject other specific frameworks
        # This prevents the controller from "stealing" models if priorities are messed up later
        if framework in ['tensorflow', 'pytorch', 'jax']:
            return False

        # 2. Explicitly accept known libraries that follow sklearn API
        # But for sklearn, we must be strict to avoid transformers
        if framework == 'sklearn':
             # If it's a raw object, it MUST have predict
             if not is_explicit_model:
                 if hasattr(model, 'predict'):
                     return True
                 # Also check the operator if available (handles dict steps where operator is instantiated)
                 if operator is not None and hasattr(operator, 'predict'):
                     return True
                 return False
             # If it's explicit {"model": ...}, we accept it
             return True

        # 3. For other frameworks (xgboost, etc), accept them
        if framework in ['xgboost', 'lightgbm', 'catboost']:
            return True

        # 4. Accept generic objects with fit/predict methods (Duck Typing)
        if hasattr(model, 'fit') and hasattr(model, 'predict'):
            return True

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

            # If no force_params (None or empty dict), return the instance directly
            if not force_params:
                return model

            # We have force_params with actual values — rebuild with new params
            # For instances, pass the instance itself to preserve structural parameters (e.g., estimators in StackingRegressor)
            # For classes, pass the class
            if isinstance(model, type):
                model_class = model
                return ModelFactory.build_single_model(model_class, dataset, force_params)
            else:
                # Pass instance to preserve structural parameters through _from_instance path
                return ModelFactory.build_single_model(model, dataset, force_params)

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

        # Fallback: if model_config itself is a model instance (e.g. XGBRegressor)
        # This happens when we pass {"model": XGBRegressor()} and _extract_model_config returns it wrapped or unwrapped
        # Actually, _extract_model_config returns {'model_instance': ...} usually.
        # But if something slipped through, we can try to use it directly.

        # Check if model_config has 'model' key which is an instance
        if 'model' in model_config:
             model = model_config['model']
             if force_params:
                 return ModelFactory.build_single_model(model, dataset, force_params)
             return model

        raise ValueError(f"Could not create model instance from configuration: {model_config.keys()}")

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

        train_params = kwargs.copy()

        # Extract controller-specific parameters that shouldn't be passed to the model
        # task_type is injected by launch_training and conflicts with CatBoost's task_type (CPU/GPU)
        task_type = train_params.pop('task_type', None)
        # verbose controls controller output, we don't want to force it on the model
        verbose = train_params.pop('verbose', 0)
        # reset_gpu: if True, reset GPU memory before/after training (helps with CatBoost GPU memory leaks)
        reset_gpu = train_params.pop('reset_gpu', False)

        # Reset GPU memory BEFORE training to ensure clean state
        if reset_gpu:
            if _reset_gpu_memory():
                if verbose > 1:
                    logger.debug(f"GPU memory cleared before training {model.__class__.__name__}")
            elif verbose > 0:
                logger.warning("reset_gpu=True but PyTorch/CUDA not available")

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
        # Handle y shape: sklearn expects (n_samples,) for single-output, (n_samples, n_outputs) for multi-output
        if y_train.ndim == 2 and y_train.shape[1] == 1:
            y_fit = y_train.ravel()  # Single output: flatten to 1D
        else:
            y_fit = y_train  # Multi-output: keep as 2D
        trained_model.fit(X_train, y_fit)

        # Reset GPU memory AFTER training as well to free model's training buffers
        if reset_gpu:
            _reset_gpu_memory()

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
                    from nirs4all.core.logging.formatters import get_symbols
                    direction = get_symbols().direction(higher_is_better)
                    all_scores_str = ModelUtils.format_scores(train_scores)
                    # Commented out - using logger instead when needed

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
                        from nirs4all.core.logging.formatters import get_symbols
                        direction = get_symbols().direction(higher_is_better)
                        all_scores_str = ModelUtils.format_scores(val_scores)
                        # Commented out - using logger instead when needed

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

    def _predict_proba_model(self, model: BaseEstimator, X: np.ndarray) -> Optional[np.ndarray]:
        """Get class probabilities for sklearn classification models.

        Supports all sklearn classifiers with predict_proba, plus XGBoost,
        LightGBM, and CatBoost classifiers.

        Args:
            model (BaseEstimator): Trained sklearn classifier.
            X (np.ndarray): Input features, shape (n_samples, n_features).

        Returns:
            np.ndarray: Class probabilities, shape (n_samples, n_classes),
                or None if model doesn't support probability predictions.
        """
        if not hasattr(model, 'predict_proba'):
            return None

        try:
            proba = model.predict_proba(X)

            # Ensure 2D array (some models return 1D for binary classification)
            if proba.ndim == 1:
                proba = np.column_stack([1 - proba, proba])

            return proba
        except Exception:
            # Some models may have predict_proba but fail (e.g., SVC without probability=True)
            return None

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

    def _evaluate_model(self, model: BaseEstimator, X_val: np.ndarray, y_val: np.ndarray, metric: Optional[str] = None, direction: str = "minimize") -> float:
        """Evaluate sklearn model on validation data using direct prediction.

        When ``metric`` is provided, uses ``nirs4all.core.metrics.eval()`` for
        unified metric computation.  Otherwise falls back to legacy behavior
        (MSE for regressors, -balanced_accuracy for classifiers).

        Args:
            model: Trained sklearn model to evaluate.
            X_val: Validation features, shape (n_samples, n_features).
            y_val: Validation targets, shape (n_samples, n_targets).
            metric: Optional metric name (e.g. 'rmse', 'r2', 'accuracy').
            direction: Optimization direction ('minimize' or 'maximize').

        Returns:
            float: Evaluation score. Returns inf on error.
        """
        y_val_1d = y_val.ravel() if y_val.ndim > 1 else y_val

        # Check for NaN in inputs before prediction
        if np.isnan(X_val).any():
            model_name = type(model).__name__
            nan_count = np.isnan(X_val).sum()
            nan_ratio = nan_count / X_val.size
            logger.warning(f"NaN in X_val for {model_name}: {nan_count} NaN values ({nan_ratio:.2%}). Model params: {model.get_params() if hasattr(model, 'get_params') else 'N/A'}")
            return float('inf')

        if np.isnan(y_val).any():
            logger.warning(f"NaN in y_val: {np.isnan(y_val).sum()} NaN values")
            return float('inf')

        try:
            y_pred = model.predict(X_val)
            if y_pred.ndim > 1:
                y_pred = y_pred.ravel()

            if metric is not None:
                from nirs4all.core import metrics as evaluator_mod
                return evaluator_mod.eval(y_val_1d, y_pred, metric)

            # Legacy behavior when no metric specified
            is_clf = is_classifier(model) or isinstance(model, ClassifierMixin)

            if is_clf:
                from sklearn.metrics import balanced_accuracy_score
                return -balanced_accuracy_score(y_val_1d, y_pred)
            else:
                return mean_squared_error(y_val_1d, y_pred)

        except Exception as e:
            logger.warning(f"Error in model evaluation: {e}")
            return float('inf')

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

        For meta-estimators (StackingRegressor, VotingClassifier, etc.) and
        boosting libraries (XGBoost, LightGBM, CatBoost), deepcopy is used
        to preserve all nested structures properly.

        Args:
            model (BaseEstimator): Sklearn model instance to clone.

        Returns:
            BaseEstimator: Cloned sklearn model with same parameters but fresh state.

        Raises:
            RuntimeError: If sklearn is not available.
        """
        # For boosting libraries, deepcopy might be safer to preserve all params
        # especially if they don't perfectly adhere to sklearn API
        framework = ModelFactory.detect_framework(model)
        if framework in ['xgboost', 'lightgbm', 'catboost']:
            return copy.deepcopy(model)

        # Meta-estimators (stacking/voting) need deepcopy to preserve nested estimators
        if ModelFactory.is_meta_estimator(model):
            return copy.deepcopy(model)

        try:
            from sklearn.base import clone as sklearn_clone
            return sklearn_clone(model)
        except ImportError:
            raise RuntimeError("sklearn is required to clone sklearn models")

    def _apply_warm_start(
        self,
        model: Any,
        source_model: Any,
        runtime_context: Any,
    ) -> Any:
        """Apply warm-start for sklearn models.

        If the model supports the ``warm_start`` parameter (e.g.,
        GradientBoostingRegressor, RandomForestRegressor), sets it to True
        and copies the fitted state from the source model.

        Args:
            model: Fresh sklearn model to warm-start.
            source_model: Trained fold model with fitted state.
            runtime_context: Runtime context.

        Returns:
            Model with warm_start enabled and fitted state copied.
        """
        if hasattr(model, 'warm_start'):
            model.warm_start = True
            # Copy fitted attributes from source_model
            for attr in dir(source_model):
                if attr.endswith('_') and not attr.startswith('__'):
                    try:
                        setattr(model, attr, getattr(source_model, attr))
                    except (AttributeError, TypeError):
                        pass
            return model
        return model

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
            - Respects force_layout if specified in step configuration
        """
        # Set layout preference for sklearn models (force_layout overrides preferred)
        context = context.with_layout(self.get_effective_layout(step_info))

        # Call parent execute method
        return super().execute(step_info, dataset, context, runtime_context, source, mode, loaded_binaries, prediction_store)


