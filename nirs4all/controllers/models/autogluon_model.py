"""
AutoGluon Model Controller - Controller for AutoGluon TabularPredictor

This controller handles AutoGluon TabularPredictor with support for:
- Automatic model selection and ensembling
- Training on tabular data (samples x features)
- Model persistence and prediction storage
- Integration with the nirs4all pipeline

AutoGluon differs from sklearn models in that:
- It trains an ensemble of models automatically
- It uses DataFrames internally, not numpy arrays
- It manages its own model directory for persistence
- It has its own hyperparameter tuning (no need for Optuna)

Lazy loading pattern: AutoGluon is only imported when actually needed
for training or prediction, not at module import time.
"""

from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING
import numpy as np
import pandas as pd
import copy
import tempfile
import shutil
import os

from ..models.base_model import BaseModelController
from nirs4all.controllers.registry import register_controller
from nirs4all.core.logging import get_logger
from nirs4all.utils.backend import is_available, require_backend, BackendNotAvailableError

logger = get_logger(__name__)

from nirs4all.pipeline.steps.parser import ParsedStep
from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
from nirs4all.pipeline.storage.artifacts.artifact_persistence import ArtifactMeta

# Fast availability check at module level - no imports
AUTOGLUON_AVAILABLE = is_available('autogluon')

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.data.dataset import SpectroDataset
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        pass


# Lazy-loaded module cache
_ag_modules: Dict[str, Any] = {}


def _get_tabular_predictor():
    """Lazy load AutoGluon TabularPredictor with caching."""
    if 'TabularPredictor' not in _ag_modules:
        require_backend('autogluon', feature='AutoGluon AutoML')
        from autogluon.tabular import TabularPredictor
        _ag_modules['TabularPredictor'] = TabularPredictor
    return _ag_modules['TabularPredictor']


def _is_autogluon_predictor(obj: Any) -> bool:
    """Check if an object is an AutoGluon TabularPredictor.

    Uses module introspection first to avoid importing AutoGluon
    for non-AutoGluon objects.

    Args:
        obj: Object to check.

    Returns:
        bool: True if object is a TabularPredictor instance or class.
    """
    if not AUTOGLUON_AVAILABLE:
        return False

    if obj is None:
        return False

    # Check if it's a dict config with autogluon reference (no import needed)
    if isinstance(obj, dict):
        if 'framework' in obj and obj['framework'] == 'autogluon':
            return True
        if 'class' in obj and 'autogluon' in str(obj['class']):
            return True

    # Check by module name for instances (no import needed)
    if hasattr(obj, '__class__'):
        module = obj.__class__.__module__
        if 'autogluon' in module:
            return True

    # If we need to check with isinstance, load TabularPredictor
    try:
        TabularPredictor = _get_tabular_predictor()

        # Check instance
        if isinstance(obj, TabularPredictor):
            return True

        # Check class
        if obj is TabularPredictor:
            return True
    except (ImportError, BackendNotAvailableError):
        pass

    return False


@register_controller
class AutoGluonModelController(BaseModelController):
    """Controller for AutoGluon TabularPredictor.

    This controller handles AutoGluon models with automatic model selection,
    ensembling, and integration with the nirs4all pipeline.

    AutoGluon automatically:
    - Trains multiple models (LightGBM, CatBoost, XGBoost, Neural Networks, etc.)
    - Performs cross-validation
    - Creates weighted ensembles
    - Handles hyperparameter tuning internally

    Uses lazy loading - AutoGluon is only imported when training starts.

    Attributes:
        priority (int): Controller priority (5) - higher than sklearn (6) to
            prioritize AutoGluon when explicitly requested.
    """

    priority = 5  # Higher priority than sklearn to catch autogluon configs

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Match AutoGluon TabularPredictor configurations.

        Args:
            step (Any): Pipeline step to check.
            operator (Any): Optional operator object.
            keyword (str): Pipeline keyword (unused).

        Returns:
            bool: True if the step matches an AutoGluon configuration.
        """
        if not AUTOGLUON_AVAILABLE:
            return False

        # Check if step is an explicit AutoGluon config
        if isinstance(step, dict):
            # Check for explicit framework key
            if step.get('framework') == 'autogluon':
                return True

            # Check if 'model' contains autogluon reference
            model = step.get('model')
            if _is_autogluon_predictor(model):
                return True

            # Check for class path containing autogluon
            if 'class' in step and 'autogluon' in str(step.get('class', '')):
                return True

        # Check if operator is AutoGluon predictor
        if _is_autogluon_predictor(operator):
            return True

        # Check if step itself is AutoGluon predictor
        if _is_autogluon_predictor(step):
            return True

        return False

    def _get_model_instance(
        self,
        dataset: 'SpectroDataset',
        model_config: Dict[str, Any],
        force_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Create AutoGluon TabularPredictor instance from configuration.

        AutoGluon predictor is created with a temporary path and configured
        based on the task type and user parameters.

        Args:
            dataset (SpectroDataset): Dataset for context-aware configuration.
            model_config (Dict[str, Any]): Model configuration.
            force_params (Optional[Dict[str, Any]]): Parameters to override.

        Returns:
            TabularPredictor: Configured AutoGluon predictor (not yet fitted).
        """
        require_backend('autogluon', feature='AutoGluon AutoML')

        # Get parameters from config
        params = model_config.get('params', {}).copy()
        if force_params:
            params.update(force_params)

        # Extract random_state if provided
        random_state = params.pop('random_state', None)

        # Determine problem type from dataset
        problem_type = None
        if dataset.task_type:
            if dataset.task_type.is_classification:
                if dataset.task_type.value == 'binary_classification':
                    problem_type = 'binary'
                else:
                    problem_type = 'multiclass'
            else:
                problem_type = 'regression'

        # Create temporary directory for AutoGluon models
        # This will be managed by nirs4all's artifact system
        temp_dir = tempfile.mkdtemp(prefix='autogluon_')

        # Create predictor with label placeholder (will be set during fit)
        predictor_params = {
            'label': '__target__',  # Placeholder, actual column name set in _train_model
            'path': temp_dir,
            'problem_type': problem_type,
            'verbosity': params.get('verbosity', 0),
        }

        # Add optional parameters if provided
        if 'eval_metric' in params:
            predictor_params['eval_metric'] = params['eval_metric']

        # Store fit parameters separately for use in _train_model
        self._fit_params = {
            k: v for k, v in params.items()
            if k not in predictor_params
        }

        # We don't create the predictor here, as it needs to be created with
        # the actual label column name. Return a config dict instead.
        return {
            'predictor_params': predictor_params,
            'fit_params': self._fit_params,
            'temp_dir': temp_dir,
            'random_state': random_state
        }

    def _train_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Any:
        """Train AutoGluon TabularPredictor.

        AutoGluon handles cross-validation, model selection, and ensembling
        internally. This method creates a DataFrame from the numpy arrays
        and calls TabularPredictor.fit().

        Args:
            model: Model config dict from _get_model_instance.
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training targets.
            X_val (Optional[np.ndarray]): Validation features (used as tuning_data).
            y_val (Optional[np.ndarray]): Validation targets.
            **kwargs: Additional training parameters.

        Returns:
            TabularPredictor: Trained AutoGluon predictor.
        """
        TabularPredictor = _get_tabular_predictor()

        # Extract controller-specific parameters
        verbose = kwargs.pop('verbose', 0)
        task_type = kwargs.pop('task_type', None)

        # Get config from model
        predictor_params = model['predictor_params'].copy()
        fit_params = model.get('fit_params', {}).copy()
        random_state = model.get('random_state', None)

        # Create DataFrame with features and target
        label_col = '__target__'
        predictor_params['label'] = label_col

        # Convert to DataFrame
        train_df = pd.DataFrame(X_train)
        train_df[label_col] = y_train.ravel() if y_train.ndim > 1 else y_train

        # Create predictor
        predictor = TabularPredictor(**predictor_params)

        # Prepare fit parameters
        fit_kwargs = {}

        # Add validation data if available
        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_df = pd.DataFrame(X_val)
            val_df[label_col] = y_val.ravel() if y_val.ndim > 1 else y_val
            fit_kwargs['tuning_data'] = val_df

        # Add time_limit if specified
        if 'time_limit' in fit_params:
            fit_kwargs['time_limit'] = fit_params.pop('time_limit')

        # Add presets if specified
        if 'presets' in fit_params:
            fit_kwargs['presets'] = fit_params.pop('presets')
        else:
            # Default to 'best_quality' for good results, 'medium_quality' for speed
            fit_kwargs['presets'] = 'medium_quality'

        # Add hyperparameters if specified
        if 'hyperparameters' in fit_params:
            fit_kwargs['hyperparameters'] = fit_params.pop('hyperparameters')

        # Add num_bag_folds if specified (for bagging)
        if 'num_bag_folds' in fit_params:
            fit_kwargs['num_bag_folds'] = fit_params.pop('num_bag_folds')

        # Add random_state via ag_args_fit for reproducibility
        # AutoGluon propagates random seeds to models through ag_args_fit
        if random_state is not None:
            ag_args_fit = fit_kwargs.get('ag_args_fit', {})
            ag_args_fit['random_seed'] = random_state
            fit_kwargs['ag_args_fit'] = ag_args_fit

        # Add remaining fit params
        fit_kwargs.update(fit_params)
        fit_kwargs.update(kwargs)

        # Fit the predictor
        if verbose > 0:
            logger.starting("Training AutoGluon TabularPredictor...")
            logger.info(f"Presets: {fit_kwargs.get('presets', 'default')}")
            logger.info(f"Time limit: {fit_kwargs.get('time_limit', 'None')}")

        predictor.fit(train_df, **fit_kwargs)

        if verbose > 0:
            # Print leaderboard
            logger.info("AutoGluon Model Leaderboard:")
            try:
                leaderboard = predictor.leaderboard(silent=True)
                logger.info(leaderboard.to_string())
            except Exception:
                pass

        return predictor

    def _predict_model(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Generate predictions with AutoGluon predictor.

        Args:
            model (TabularPredictor): Trained AutoGluon predictor.
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Model predictions, shape (n_samples, 1).
        """
        require_backend('autogluon', feature='AutoGluon prediction')

        # Convert to DataFrame
        test_df = pd.DataFrame(X)

        # Get predictions
        predictions = model.predict(test_df)

        # Convert to numpy and reshape
        predictions = np.array(predictions)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        return predictions

    def _predict_proba_model(self, model: Any, X: np.ndarray) -> Optional[np.ndarray]:
        """Get class probabilities for AutoGluon classification models.

        Args:
            model (TabularPredictor): Trained AutoGluon predictor.
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Class probabilities, or None if not classification.
        """
        if not AUTOGLUON_AVAILABLE:
            return None

        if not hasattr(model, 'can_predict_proba') or not model.can_predict_proba:
            return None

        try:
            test_df = pd.DataFrame(X)
            proba = model.predict_proba(test_df)

            # Convert to numpy
            proba = np.array(proba)

            return proba
        except Exception:
            return None

    def _prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        context: 'ExecutionContext'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for AutoGluon (ensure 2D arrays).

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
            context (ExecutionContext): Pipeline context.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Prepared (X, y) arrays.
        """
        if X is None:
            return None, None

        # Ensure X is 2D
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        elif X.ndim == 1:
            X = X.reshape(-1, 1)

        # Handle y
        if y is not None:
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            elif y.ndim > 2:
                y = y.reshape(y.shape[0], -1)

        return X, y

    def _evaluate_model(
        self,
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """Evaluate AutoGluon model using its internal evaluation.

        Args:
            model (TabularPredictor): AutoGluon predictor.
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation targets.

        Returns:
            float: Evaluation score (negative for maximization metrics).
        """
        if not AUTOGLUON_AVAILABLE:
            return float('inf')

        try:
            TabularPredictor = _get_tabular_predictor()

            # Create validation DataFrame
            label_col = '__target__'
            val_df = pd.DataFrame(X_val)
            val_df[label_col] = y_val.ravel() if y_val.ndim > 1 else y_val

            # Evaluate using AutoGluon's evaluate method
            eval_result = model.evaluate(val_df, silent=True)

            # Get the main metric score
            if isinstance(eval_result, dict):
                # Get the primary metric value
                primary_metric = model.eval_metric.name if hasattr(model, 'eval_metric') else None
                if primary_metric and primary_metric in eval_result:
                    score = eval_result[primary_metric]
                else:
                    # Use first metric
                    score = list(eval_result.values())[0]
            else:
                score = float(eval_result)

            # AutoGluon metrics are typically higher-is-better
            # Return negative for minimization-based optimization
            return -score

        except Exception as e:
            logger.warning(f"Error in AutoGluon evaluation: {e}")
            return float('inf')

    def get_preferred_layout(self) -> str:
        """Return the preferred data layout for AutoGluon.

        Returns:
            str: Data layout preference, '2d' for AutoGluon.
        """
        return "2d"

    def _clone_model(self, model: Any) -> Any:
        """Clone AutoGluon model configuration.

        For AutoGluon, we clone the configuration dict since the actual
        predictor needs to be created fresh for each fold.

        Args:
            model: Model config dict or TabularPredictor.

        Returns:
            Cloned configuration or deep copy.
        """
        if isinstance(model, dict):
            return copy.deepcopy(model)

        # For a fitted predictor, we can clone it using AutoGluon's method
        if AUTOGLUON_AVAILABLE:
            try:
                TabularPredictor = _get_tabular_predictor()
                if isinstance(model, TabularPredictor):
                    # Create a new temp directory for the clone
                    temp_dir = tempfile.mkdtemp(prefix='autogluon_clone_')
                    return model.clone(path=temp_dir)
            except (TypeError, BackendNotAvailableError):
                pass

        return copy.deepcopy(model)

    def _sample_hyperparameters(
        self,
        trial,
        finetune_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sample hyperparameters for AutoGluon.

        AutoGluon has its own internal hyperparameter tuning, so this method
        samples high-level configuration parameters like time_limit and presets.

        Args:
            trial: Optuna trial object.
            finetune_params (Dict[str, Any]): Hyperparameter search space.

        Returns:
            Dict[str, Any]: Sampled configuration parameters.
        """
        params = {}

        # Sample presets
        if 'presets' in finetune_params:
            preset_config = finetune_params['presets']
            if isinstance(preset_config, list):
                params['presets'] = trial.suggest_categorical('presets', preset_config)
            else:
                params['presets'] = preset_config

        # Sample time_limit
        if 'time_limit' in finetune_params:
            tl_config = finetune_params['time_limit']
            if isinstance(tl_config, tuple) and len(tl_config) == 3:
                low, high = tl_config[1], tl_config[2]
                params['time_limit'] = trial.suggest_int('time_limit', low, high)
            else:
                params['time_limit'] = tl_config

        # Sample num_bag_folds
        if 'num_bag_folds' in finetune_params:
            nbf_config = finetune_params['num_bag_folds']
            if isinstance(nbf_config, tuple) and len(nbf_config) == 3:
                low, high = nbf_config[1], nbf_config[2]
                params['num_bag_folds'] = trial.suggest_int('num_bag_folds', low, high)
            else:
                params['num_bag_folds'] = nbf_config

        # Sample random_state (typically fixed, but can be searched)
        if 'random_state' in finetune_params:
            rs_config = finetune_params['random_state']
            if isinstance(rs_config, tuple) and len(rs_config) == 3:
                low, high = rs_config[1], rs_config[2]
                params['random_state'] = trial.suggest_int('random_state', low, high)
            elif isinstance(rs_config, list):
                params['random_state'] = trial.suggest_categorical(
                    'random_state', rs_config
                )
            else:
                params['random_state'] = rs_config

        return params

    def save_model(self, model: Any, filepath: str) -> None:
        """Save AutoGluon model to disk.

        AutoGluon models are saved as directories. This method moves the
        model's directory to the specified filepath.

        Args:
            model (TabularPredictor): Trained AutoGluon predictor.
            filepath (str): Target path for saving.
        """
        require_backend('autogluon', feature='AutoGluon model saving')

        # AutoGluon saves to a directory
        # Remove .pkl extension if present
        if filepath.endswith('.pkl'):
            filepath = filepath[:-4]

        # Save the predictor
        model.save(filepath)

    def load_model(self, filepath: str) -> Any:
        """Load AutoGluon model from disk.

        Args:
            filepath (str): Path to the saved model directory.

        Returns:
            TabularPredictor: Loaded AutoGluon predictor.
        """
        TabularPredictor = _get_tabular_predictor()

        # Remove .pkl extension if present
        if filepath.endswith('.pkl'):
            filepath = filepath[:-4]

        return TabularPredictor.load(filepath)

    def execute(
        self,
        step_info: ParsedStep,
        dataset: 'SpectroDataset',
        context: ExecutionContext,
        runtime_context: RuntimeContext,
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, bytes]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple[ExecutionContext, List[ArtifactMeta]]:
        """Execute AutoGluon model controller.

        Main entry point for AutoGluon model execution in the pipeline.

        Args:
            step_info: Parsed step containing model configuration.
            dataset (SpectroDataset): Dataset containing features and targets.
            context (ExecutionContext): Pipeline execution context.
            runtime_context (RuntimeContext): Runtime context.
            source (int): Source index. Defaults to -1.
            mode (str): Execution mode. Defaults to 'train'.
            loaded_binaries: Pre-loaded model binaries for prediction.
            prediction_store: Store for managing predictions.

        Returns:
            Tuple[ExecutionContext, List[ArtifactMeta]]: Updated context
                and list of model binaries.
        """
        # Set layout preference (force_layout overrides preferred)
        context = context.with_layout(self.get_effective_layout(step_info))

        # Call parent execute method
        return super().execute(
            step_info, dataset, context, runtime_context,
            source, mode, loaded_binaries, prediction_store
        )
