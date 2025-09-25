"""
Base Model Controller - Abstract controller for machine learning models

This module provides the base abstract controller that handles the three main modes:
- Training: Fit model on training data
- Fine-tuning: Hyperparameter optimization using Optuna
- Prediction: Generate predictions and store results

All model controllers (sklearn, tensorflow, pytorch) inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, Union, TYPE_CHECKING
import pickle
import numpy as np
from enum import Enum

from nirs4all.dataset.helpers import Layout
from nirs4all.controllers.controller import OperatorController

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset


class ModelMode(Enum):
    """Enumeration of model execution modes."""
    TRAIN = "train"
    FINETUNE = "finetune"
    PREDICT = "predict"


class CVMode(Enum):
    """Enumeration of cross-validation finetuning strategies."""
    SIMPLE = "simple"  # Finetune on full train data, then train on folds
    PER_FOLD = "per_fold"  # Finetune on each fold individually
    NESTED = "nested"  # Inner folds for finetuning, outer folds for training


class ParamStrategy(Enum):
    """Parameter aggregation strategies for cross-validation."""
    GLOBAL_BEST = "global_best"  # Use single best params for all folds
    PER_FOLD_BEST = "per_fold_best"  # Use best params per fold
    WEIGHTED_AVERAGE = "weighted_average"  # Average params weighted by performance


class BaseModelController(OperatorController, ABC):
    """
    Abstract base controller for machine learning models.

    Handles the common pipeline for:
    1. Data preparation (train/test splits, cross-validation)
    2. Model training with configurable parameters
    3. Hyperparameter tuning with Optuna (optional)
    4. Prediction and result storage

    Each framework-specific controller must implement:
    - _get_model_instance: Create model from configuration
    - _train_model: Framework-specific training logic
    - _predict_model: Framework-specific prediction logic
    - _prepare_data: Framework-specific data formatting
    """

    priority = 15  # Higher priority than transformers, lower than data operations

    @classmethod
    def use_multi_source(cls) -> bool:
        """Models can handle multi-source datasets."""
        return True

    @abstractmethod
    def _get_model_instance(self, model_config: Dict[str, Any]) -> Any:
        """Create a model instance from configuration."""
        pass

    @abstractmethod
    def _train_model(
        self,
        model: Any,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        train_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Train the model with framework-specific logic."""
        pass

    @abstractmethod
    def _predict_model(self, model: Any, X: Any) -> np.ndarray:
        """Generate predictions with framework-specific logic."""
        pass

    @abstractmethod
    def _prepare_data(self, X: Any, y: Any, context: Dict[str, Any]) -> Tuple[Any, Any]:
        """Prepare data in framework-specific format (2D, 3D, tensors, etc.)."""
        pass

    @abstractmethod
    def get_preferred_layout(self) -> str:
        """Return the preferred data layout for this model type ('2d' or '3d')."""
        pass

    def execute(
        self,
        step: Any,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        source: int = -1
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """
        Execute model controller in one of three modes: train, finetune, or predict.
        Handles both single training and cross-validation with multiple folds.

        Args:
            step: Pipeline step configuration
            operator: Model operator (unused, model comes from step config)
            dataset: Dataset containing features and targets
            context: Pipeline context with processing state
            runner: Pipeline runner instance
            source: Data source index

        Returns:
            Tuple of (updated_context, binaries_list)
        """
        # print(f"🤖 Executing model controller: {self.__class__.__name__}")

        # Extract model configuration from step
        model_config = self._extract_model_config(step)
        train_params = model_config.get('train_params', {})
        finetune_params = model_config.get('finetune_params', {})

        # Determine execution mode
        mode = self._determine_mode(model_config)
        # print(f"🎯 Model mode: {mode.value}")

        # Prepare data (can return single tuple or list of tuples for folds)
        data_splits = self._prepare_train_test_data(dataset, context)

        # Check if we have folds (list of tuples) or single split (single tuple)
        if isinstance(data_splits, list):
            # Cross-validation mode: determine finetuning strategy
            if mode == ModelMode.FINETUNE and finetune_params:
                cv_mode = self._determine_cv_mode(finetune_params)

                if cv_mode == CVMode.NESTED:
                    return self._execute_nested_cv(
                        model_config, data_splits, train_params, finetune_params,
                        context, runner, dataset
                    )
                elif cv_mode == CVMode.PER_FOLD:
                    return self._execute_per_fold_cv(
                        model_config, data_splits, train_params, finetune_params,
                        context, runner, dataset
                    )
                else:  # CVMode.SIMPLE or default
                    # First finetune on all training data to get best params
                    return self._execute_simple_cv_finetune(
                        model_config, data_splits, train_params, finetune_params,
                        context, runner, dataset
                    )
            else:
                # Standard cross-validation without finetuning
                return self._execute_cross_validation(
                    model_config, data_splits, train_params, finetune_params,
                    mode, context, runner, dataset
                )
        else:
            # Single training mode: no folds
            X_train, y_train, X_test, y_test = data_splits
            if mode == ModelMode.FINETUNE and finetune_params:
                return self._execute_finetune(
                    model_config, X_train, y_train, X_test, y_test,
                    train_params, finetune_params, context, runner
                )
            else:
                return self._execute_train(
                    model_config, X_train, y_train, X_test, y_test,
                    train_params, context, runner
                )

    def _extract_model_config(self, step: Any) -> Dict[str, Any]:
        """Extract model configuration from step."""
        if isinstance(step, dict):
            if 'model' in step:
                # Step format: {"model": model_obj, "train_params": {...}, "finetune_params": {...}}
                config = step.copy()
                model_obj = step['model']

                # Handle serialized model format
                if isinstance(model_obj, dict) and '_runtime_instance' in model_obj:
                    config['model_instance'] = model_obj['_runtime_instance']
                elif hasattr(model_obj, '_runtime_instance'):
                    config['model_instance'] = model_obj._runtime_instance
                else:
                    config['model_instance'] = model_obj
                return config
            else:
                # Direct model object
                return {'model_instance': step}
        else:
            # Direct model object
            return {'model_instance': step}

    def _determine_mode(self, model_config: Dict[str, Any]) -> ModelMode:
        """Determine execution mode based on configuration."""
        if 'finetune_params' in model_config and model_config['finetune_params']:
            return ModelMode.FINETUNE
        else:
            return ModelMode.TRAIN

    def _determine_cv_mode(self, finetune_params: Dict[str, Any]) -> CVMode:
        """Determine the cross-validation finetuning mode from parameters."""
        cv_mode_str = finetune_params.get('cv_mode', 'simple')

        # Check for nested CV indicators
        if (finetune_params.get('inner_cv') is not None or
            finetune_params.get('outer_cv') is not None or
            cv_mode_str == 'nested'):
            return CVMode.NESTED
        elif cv_mode_str == 'per_fold':
            return CVMode.PER_FOLD
        else:
            return CVMode.SIMPLE

    def _create_inner_folds(self, X_train: Any, y_train: Any, inner_cv: Any) -> List[Tuple[Any, Any, Any, Any]]:
        """Create inner cross-validation folds for nested CV."""
        try:
            from sklearn.model_selection import KFold, StratifiedKFold
        except ImportError:
            raise ImportError("scikit-learn is required for nested cross-validation")

        # Default to 3-fold if not specified
        if inner_cv is None:
            inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        elif isinstance(inner_cv, int):
            inner_cv = KFold(n_splits=inner_cv, shuffle=True, random_state=42)

        inner_folds = []
        for train_idx, val_idx in inner_cv.split(X_train, y_train):
            # Use numpy indexing if possible
            if hasattr(X_train, '__getitem__') and hasattr(X_train, 'shape'):
                X_inner_train = X_train[train_idx]
                X_inner_val = X_train[val_idx]
            else:
                X_inner_train = X_train
                X_inner_val = X_train

            if hasattr(y_train, '__getitem__') and hasattr(y_train, 'shape'):
                y_inner_train = y_train[train_idx]
                y_inner_val = y_train[val_idx]
            else:
                y_inner_train = y_train
                y_inner_val = y_train

            inner_folds.append((X_inner_train, y_inner_train, X_inner_val, y_inner_val))

        return inner_folds

    def _execute_simple_cv_finetune(
        self,
        model_config: Dict[str, Any],
        data_splits: List[Tuple[Any, Any, Any, Any]],
        train_params: Dict[str, Any],
        finetune_params: Dict[str, Any],
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        dataset: 'SpectroDataset'
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute simple CV: finetune on full training data, then train on folds."""
        verbose = finetune_params.get('verbose', train_params.get('verbose', 0))

        if verbose > 0:
            print("🔍 Simple CV: Finetuning on full training data...")

        # Combine all training data for finetuning
        all_X_train = []
        all_y_train = []
        for X_train, y_train, _, _ in data_splits:
            all_X_train.append(X_train)
            all_y_train.append(y_train)

        # Concatenate all training data
        import numpy as np
        combined_X_train = np.concatenate(all_X_train, axis=0)
        combined_y_train = np.concatenate(all_y_train, axis=0)

        # Use first fold's test data for finetuning evaluation (could be improved)
        _, _, X_test_sample, y_test_sample = data_splits[0]

        # Execute finetuning to get best parameters
        _, finetune_binaries = self._execute_finetune(
            model_config, combined_X_train, combined_y_train,
            X_test_sample, y_test_sample, train_params, finetune_params,
            context, runner
        )

        # Extract best parameters from the finetuning process
        best_params = getattr(self, '_last_best_params', {})

        if verbose > 0:
            print(f"🏆 Best parameters found: {best_params}")
            print(f"🔄 Training {len(data_splits)} fold models with best parameters...")

        # Now train models on each fold with the best parameters
        all_binaries = []
        all_binaries.extend(finetune_binaries)

        for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(data_splits):
            # Create model with best parameters
            base_model = self._get_model_from_config(model_config)
            model = self._clone_model(base_model)

            # Apply best parameters
            if hasattr(model, 'set_params') and best_params:
                try:
                    model.set_params(**best_params)
                except Exception as e:
                    if verbose > 0:
                        print(f"⚠️ Could not apply best parameters to fold {fold_idx+1}: {e}")

            # Train on this fold
            fold_context, fold_binaries = self._execute_train(
                model_config, X_train, y_train, X_test, y_test,
                train_params, context, runner, fold_idx
            )

            # Add fold suffix to binary names
            fold_binaries_renamed = []
            for name, binary in fold_binaries:
                name_parts = name.rsplit('.', 1)
                if len(name_parts) == 2:
                    new_name = f"{name_parts[0]}_simple_cv_fold{fold_idx+1}.{name_parts[1]}"
                else:
                    new_name = f"{name}_simple_cv_fold{fold_idx+1}"
                fold_binaries_renamed.append((new_name, binary))

            all_binaries.extend(fold_binaries_renamed)

        if verbose > 0:
            print("✅ Simple CV completed successfully")

        return context, all_binaries

    def _execute_per_fold_cv(
        self,
        model_config: Dict[str, Any],
        data_splits: List[Tuple[Any, Any, Any, Any]],
        train_params: Dict[str, Any],
        finetune_params: Dict[str, Any],
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        dataset: 'SpectroDataset'
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute per-fold CV: finetune on each fold individually."""
        verbose = finetune_params.get('verbose', train_params.get('verbose', 0))
        param_strategy = ParamStrategy(finetune_params.get('param_strategy', 'per_fold_best'))

        if verbose > 0:
            print(f"🔍 Per-fold CV: Finetuning on each fold with {param_strategy.value} strategy...")

        all_binaries = []
        all_best_params = []

        # Finetune on each fold
        for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(data_splits):
            if verbose > 0:
                print(f"🎛️ Finetuning fold {fold_idx+1}/{len(data_splits)}...")

            # Execute finetuning for this fold
            fold_context, fold_binaries = self._execute_finetune(
                model_config, X_train, y_train, X_test, y_test,
                train_params, finetune_params, context, runner, fold_idx
            )

            # Store best parameters for this fold
            fold_best_params = getattr(self, '_last_best_params', {})
            all_best_params.append(fold_best_params)

            # Add fold suffix to binary names
            fold_binaries_renamed = []
            for name, binary in fold_binaries:
                name_parts = name.rsplit('.', 1)
                if len(name_parts) == 2:
                    new_name = f"{name_parts[0]}_per_fold_cv_fold{fold_idx+1}.{name_parts[1]}"
                else:
                    new_name = f"{name}_per_fold_cv_fold{fold_idx+1}"
                fold_binaries_renamed.append((new_name, binary))

            all_binaries.extend(fold_binaries_renamed)

        # Handle parameter aggregation strategy
        if param_strategy == ParamStrategy.GLOBAL_BEST:
            # Use the best performing parameters across all folds
            global_best_params = self._select_global_best_params(all_best_params, data_splits)
            if verbose > 0:
                print(f"🏆 Global best parameters: {global_best_params}")

        if verbose > 0:
            print("✅ Per-fold CV completed successfully")

        return context, all_binaries

    def _execute_nested_cv(
        self,
        model_config: Dict[str, Any],
        outer_folds: List[Tuple[Any, Any, Any, Any]],
        train_params: Dict[str, Any],
        finetune_params: Dict[str, Any],
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        dataset: 'SpectroDataset'
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute nested CV: inner folds for finetuning, outer folds for training."""
        verbose = finetune_params.get('verbose', train_params.get('verbose', 0))
        param_strategy = ParamStrategy(finetune_params.get('param_strategy', 'per_fold_best'))
        inner_cv = finetune_params.get('inner_cv', 3)

        if verbose > 0:
            print(f"🔍 Nested CV: {len(outer_folds)} outer folds with inner CV finetuning...")
            print(f"📊 Parameter strategy: {param_strategy.value}")

        all_binaries = []
        all_fold_results = []

        for outer_idx, (X_outer_train, y_outer_train, X_outer_test, y_outer_test) in enumerate(outer_folds):
            if verbose > 0:
                print(f"🏋️ Outer fold {outer_idx+1}/{len(outer_folds)}...")

            # Create inner folds for finetuning
            inner_folds = self._create_inner_folds(X_outer_train, y_outer_train, inner_cv)

            if verbose > 1:
                print(f"  📋 Created {len(inner_folds)} inner folds for finetuning")

            # Finetune using inner folds to find best parameters for this outer fold
            fold_best_params = self._finetune_on_inner_folds(
                model_config, inner_folds, train_params, finetune_params, context, verbose
            )

            if verbose > 1:
                print(f"  🏆 Best params for outer fold {outer_idx+1}: {fold_best_params}")

            # Train final model on full outer training data with best parameters
            base_model = self._get_model_from_config(model_config)
            final_model = self._clone_model(base_model)

            # Apply best parameters
            if hasattr(final_model, 'set_params') and fold_best_params:
                try:
                    final_model.set_params(**fold_best_params)
                except Exception as e:
                    if verbose > 0:
                        print(f"⚠️ Could not apply parameters to outer fold {outer_idx+1}: {e}")

            # Prepare data and train
            X_train_prep, y_train_prep = self._prepare_data(X_outer_train, y_outer_train, context)
            X_test_prep, _ = self._prepare_data(X_outer_test, y_outer_test, context)

            trained_model = self._train_model(
                final_model, X_train_prep, y_train_prep, train_params=train_params
            )

            # Generate predictions
            y_pred = self._predict_model(trained_model, X_test_prep)

            # Store results for this fold
            fold_binaries = self._store_results(
                trained_model, y_pred, y_outer_test, runner, f"nested_cv_outer_fold{outer_idx+1}"
            )

            # Store fold results for potential parameter aggregation
            fold_score = self._evaluate_model(trained_model, X_test_prep, y_outer_test)
            all_fold_results.append({
                'fold_idx': outer_idx,
                'best_params': fold_best_params,
                'score': fold_score,
                'binaries': fold_binaries
            })

            all_binaries.extend(fold_binaries)

        # Handle parameter aggregation across outer folds
        if param_strategy == ParamStrategy.WEIGHTED_AVERAGE:
            self._compute_weighted_average_params(all_fold_results, verbose)

        if verbose > 0:
            print("✅ Nested CV completed successfully")

        return context, all_binaries

    def _finetune_on_inner_folds(
        self,
        model_config: Dict[str, Any],
        inner_folds: List[Tuple[Any, Any, Any, Any]],
        train_params: Dict[str, Any],
        finetune_params: Dict[str, Any],
        context: Dict[str, Any],
        verbose: int = 0
    ) -> Dict[str, Any]:
        """Finetune using inner folds and return best parameters."""
        try:
            import optuna
        except ImportError:
            print("⚠️ Optuna not available for nested CV")
            return {}

        best_params = {}
        best_score = float('inf')

        def objective(trial):
            nonlocal best_params, best_score

            # Sample hyperparameters
            trial_params = self._sample_hyperparameters(trial, finetune_params)

            # Cross-validate on inner folds
            fold_scores = []
            for X_train, y_train, X_val, y_val in inner_folds:
                # Create and configure model
                base_model = self._get_model_from_config(model_config)
                model = self._clone_model(base_model)

                if hasattr(model, 'set_params') and trial_params:
                    try:
                        model.set_params(**trial_params)
                    except Exception:
                        pass

                # Prepare data and train
                X_train_prep, y_train_prep = self._prepare_data(X_train, y_train, context)
                X_val_prep, y_val_prep = self._prepare_data(X_val, y_val, context)

                # Use silent training for inner CV
                inner_train_params = finetune_params.get('train_params', train_params.copy())
                inner_train_params['verbose'] = 0

                trained_model = self._train_model(
                    model, X_train_prep, y_train_prep, train_params=inner_train_params
                )

                # Evaluate
                score = self._evaluate_model(trained_model, X_val_prep, y_val_prep)
                fold_scores.append(score)

            # Average score across inner folds
            avg_score = np.mean(fold_scores)

            if avg_score < best_score:
                best_score = avg_score
                best_params = trial_params.copy()

            return avg_score

        # Configure and run optimization
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="minimize")
        n_trials = finetune_params.get('n_trials', 10)

        if verbose > 2:
            print(f"    🎯 Running {n_trials} inner CV trials...")

        study.optimize(objective, n_trials=n_trials)

        return best_params

    def _select_global_best_params(
        self,
        all_best_params: List[Dict[str, Any]],
        data_splits: List[Tuple[Any, Any, Any, Any]]
    ) -> Dict[str, Any]:
        """Select the globally best parameters from all folds."""
        # For now, return the first set of parameters
        # This could be improved by tracking scores and selecting the best performing set
        if all_best_params:
            return all_best_params[0]
        return {}

    def _compute_weighted_average_params(
        self,
        fold_results: List[Dict[str, Any]],
        verbose: int = 0
    ) -> Dict[str, Any]:
        """Compute weighted average parameters based on fold performance."""
        if verbose > 0:
            print("📊 Computing weighted average parameters...")

        # For numerical parameters, compute weighted average
        # For categorical parameters, select most frequent
        # This is a placeholder implementation
        return {}

    def _prepare_train_test_data(
        self,
        dataset: 'SpectroDataset',
        context: Dict[str, Any]
    ) -> Union[Tuple[Any, Any, Any, Any], List[Tuple[Any, Any, Any, Any]]]:
        """
        Prepare training and test data from dataset, handling cross-validation folds.

        If dataset has folds, returns a list of (X_train, y_train, X_val, y_val) tuples, one per fold.
        If no folds, returns a single tuple (X_train, y_train, X_test, y_test).

        Returns:
            Union[Tuple[Any, Any, Any, Any], List[Tuple[Any, Any, Any, Any]]]
        """
        # Get the preferred layout for this model type
        layout_str = self.get_preferred_layout()
        layout: Layout = layout_str  # type: ignore

        # Check if dataset has folds
        if hasattr(dataset, 'num_folds') and dataset.num_folds > 0:
            # Prepare fold-based train/validation splits
            folds_data = []

            # Get all training data first
            train_context = context.copy()
            train_context["partition"] = "train"
            X_all_train = dataset.x(train_context, layout, concat_source=True)
            y_all_train = dataset.y(train_context)

            # For each fold, create train/validation splits
            for fold_idx, (train_indices, val_indices) in enumerate(dataset.folds):
                X_train_fold = X_all_train[train_indices]
                y_train_fold = y_all_train[train_indices]
                X_val_fold = X_all_train[val_indices]
                y_val_fold = y_all_train[val_indices]

                folds_data.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))
                # print(f"📊 Fold {fold_idx+1}: Train {X_train_fold.shape}, Val {X_val_fold.shape}")

            return folds_data
        else:
            # No folds: use standard train/test split
            train_context = context.copy()
            train_context["partition"] = "train"
            X_train = dataset.x(train_context, layout, concat_source=True)
            y_train = dataset.y(train_context)

            test_context = context.copy()
            test_context["partition"] = "test"
            X_test = dataset.x(test_context, layout, concat_source=True)
            y_test = dataset.y(test_context)

            # print(f"📊 No folds - Train: X{X_train.shape}, y{y_train.shape} | Test: X{X_test.shape}, y{y_test.shape}")
            return X_train, y_train, X_test, y_test

    def _execute_cross_validation(
        self,
        model_config: Dict[str, Any],
        data_splits: List[Tuple[Any, Any, Any, Any]],
        train_params: Dict[str, Any],
        finetune_params: Dict[str, Any],
        mode: ModelMode,
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        dataset: 'SpectroDataset'
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute cross-validation training, creating one model per fold."""
        verbose = train_params.get('verbose', 0)
        all_binaries = []

        # if verbose > 0:
        #     print(f"🔄 Cross-validation: training {len(data_splits)} fold models...")

        # Get test data for final evaluation (after training all folds)
        test_context = context.copy()
        test_context["partition"] = "test"
        layout_str = self.get_preferred_layout()
        layout: Layout = layout_str  # type: ignore
        X_test = dataset.x(test_context, layout, concat_source=True)
        y_test = dataset.y(test_context)

        for fold_idx, (X_train, y_train, X_val, y_val) in enumerate(data_splits):
            # if verbose > 0:
            #     print(f"🏋️ Training fold {fold_idx+1}/{len(data_splits)}...")

            if mode == ModelMode.FINETUNE and finetune_params:
                # Fine-tune for this fold
                fold_context, fold_binaries = self._execute_finetune(
                    model_config, X_train, y_train, X_val, y_val,
                    train_params, finetune_params, context, runner, fold_idx
                )
            else:
                # Train for this fold
                fold_context, fold_binaries = self._execute_train(
                    model_config, X_train, y_train, X_val, y_val,
                    train_params, context, runner, fold_idx
                )

            # Add fold suffix to binary names
            fold_binaries_renamed = []
            for name, binary in fold_binaries:
                name_parts = name.rsplit('.', 1)
                if len(name_parts) == 2:
                    new_name = f"{name_parts[0]}_fold{fold_idx+1}.{name_parts[1]}"
                else:
                    new_name = f"{name}_fold{fold_idx+1}"
                fold_binaries_renamed.append((new_name, binary))

            all_binaries.extend(fold_binaries_renamed)

        # if verbose > 0:
        #     print("✅ Cross-validation completed successfully")

        return context, all_binaries

    def _execute_train(
        self,
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        train_params: Dict[str, Any],
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        fold_idx: Optional[int] = None
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute training mode."""
        verbose = train_params.get('verbose', 0)

        # if verbose > 0:
            # print("🏋️ Training model...")

        # Get model instance and clone it to avoid modifying original
        base_model = self._get_model_from_config(model_config)
        model = self._clone_model(base_model)

        # Prepare data in framework-specific format
        X_train_prep, y_train_prep = self._prepare_data(X_train, y_train, context)
        X_test_prep, _ = self._prepare_data(X_test, y_test, context)

        print(X_train_prep.shape, y_train_prep.shape, X_test_prep.shape, y_test.shape)

        # Train model
        trained_model = self._train_model(
            model, X_train_prep, y_train_prep,
            train_params=train_params
        )

        # Generate predictions
        y_pred = self._predict_model(trained_model, X_test_prep)

        # Store results and serialize model
        binaries = self._store_results(trained_model, y_pred, y_test, runner, "trained")

        # if verbose > 0:
            # print("✅ Training completed successfully")
        return context, binaries

    def _execute_finetune(
        self,
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        train_params: Dict[str, Any],
        finetune_params: Dict[str, Any],
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        fold_idx: Optional[int] = None
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute fine-tuning mode with Optuna."""
        # Check verbose setting from finetune_params or train_params (0=silent, 1=basic, 2=detailed)
        verbose = finetune_params.get('verbose', train_params.get('verbose', 0))

        # if verbose > 0:
            # print("🎛️ Fine-tuning model with Optuna...")

        try:
            import optuna
        except ImportError:
            print("⚠️ Optuna not available, falling back to training mode")
            return self._execute_train(
                model_config, X_train, y_train, X_test, y_test,
                train_params, context, runner
            )        # Prepare data
        X_train_prep, y_train_prep = self._prepare_data(X_train, y_train, context)
        X_test_prep, _ = self._prepare_data(X_test, y_test, context)

        best_model = None
        best_score = float('inf')
        best_params = {}

        def objective(trial):
            nonlocal best_model, best_score, best_params

            # Sample hyperparameters
            trial_params = self._sample_hyperparameters(trial, finetune_params)

            # Create model with trial parameters - get base model and clone it
            base_model = self._get_model_from_config(model_config)
            model = self._clone_model(base_model)

            # Apply trial parameters to the model if possible
            if hasattr(model, 'set_params') and trial_params:
                try:
                    model.set_params(**trial_params)
                except Exception as e:
                    print(f"⚠️ Could not set parameters {trial_params}: {e}")

            # Train model - use train_params from finetune_params if available, otherwise use top-level train_params
            trial_train_params = finetune_params.get('train_params', train_params.copy())
            # Ensure training is silent during trials unless explicitly verbose
            if 'verbose' not in trial_train_params:
                trial_train_params['verbose'] = 0  # Silent during trials

            trained_model = self._train_model(
                model, X_train_prep, y_train_prep,
                train_params=trial_train_params
            )            # Evaluate model (use validation split or cross-validation)
            score = self._evaluate_model(trained_model, X_train_prep, y_train_prep)

            # Keep track of best model
            if score < best_score:
                best_score = score
                best_model = trained_model
                best_params = trial_params.copy()

            return score

        # Configure Optuna logging
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Run optimization with configurable search method
        search_method = finetune_params.get('approach', 'auto')
        n_trials = finetune_params.get('n_trials', 10)

        # Auto-determine search method if not specified
        if search_method == 'auto':
            is_grid_suitable = self._is_grid_search_suitable(finetune_params)
            search_method = 'grid' if is_grid_suitable else 'random'

        # Only use grid search if explicitly requested or auto-determined AND we have categorical params
        has_categorical = any(isinstance(v, list) for k, v in finetune_params.items()
                              if k not in ['n_trials', 'approach'])
        use_grid = (search_method == 'grid' and has_categorical and
                    finetune_params.get('approach', 'auto') != 'random')

        if use_grid:
            # Use GridSampler for exhaustive search
            import optuna.samplers
            search_space = self._create_grid_search_space(finetune_params)
            if search_space:
                sampler = optuna.samplers.GridSampler(search_space)
                study = optuna.create_study(direction="minimize", sampler=sampler)
                # Calculate total combinations for grid search
                n_trials = 1
                for param_values in search_space.values():
                    n_trials *= len(param_values)
                # if verbose:
                    # print(f"🔍 Starting grid search with {n_trials} combinations")
            else:
                # Fallback to random if no categorical parameters found
                study = optuna.create_study(direction="minimize")
                # if verbose:
                    # print(f"🎯 Starting random search with {n_trials} trials (no categorical params for grid)")
        else:
            # Use random/TPE sampler
            study = optuna.create_study(direction="minimize")
            # if verbose:
                # print(f"🎯 Starting {search_method} search with {n_trials} trials")

        # Show concise message about optimization
        print(f"🔍 Optimizing {len(finetune_params.get('model_params', {}))} parameters with {search_method} search ({n_trials} trials)...")
        print(X_train_prep.shape, y_train_prep.shape, X_test_prep.shape, y_test.shape)
        study.optimize(objective, n_trials=n_trials)

        # Store best parameters for access by nested CV methods
        self._last_best_params = best_params

        # if verbose:
            # print(f"🏆 Best parameters: {best_params}, Best score: {best_score:.4f}")
        # Retrain best model with top-level train_params (final training parameters)
        if train_params != finetune_params.get('train_params', train_params):
            # print(f"🏆 Training with best parameters: {best_params}")
            # if verbose > 1:
                # print("🔄 Using final training parameters for best model...")

            base_model = self._get_model_from_config(model_config)
            final_model = self._clone_model(base_model)

            # Apply best parameters to the final model
            if hasattr(final_model, 'set_params') and best_params:
                try:
                    final_model.set_params(**best_params)
                except Exception as e:
                    print(f"⚠️ Could not set best parameters {best_params}: {e}")


            # Train with final train_params
            final_trained_model = self._train_model(
                final_model, X_train_prep, y_train_prep,
                train_params=train_params
            )
            best_model = final_trained_model
        else:
            # Just show the best parameters found
            print(f"🏆 Training with best parameters: {best_params}")


        # Generate final predictions with best model
        y_pred = self._predict_model(best_model, X_test_prep)

        # Store results
        binaries = self._store_results(best_model, y_pred, y_test, runner, "finetuned")

        # if verbose > 0:
            # print("✅ Fine-tuning completed successfully")
        return context, binaries

    def _get_model_from_config(self, model_config: Dict[str, Any]) -> Any:
        """Get model instance from configuration."""
        if 'model_instance' in model_config:
            model = model_config['model_instance']
            # If it's still a string or serialized format, try to get the runtime instance
            if isinstance(model, dict) and '_runtime_instance' in model:
                return model['_runtime_instance']
            elif hasattr(model, '_runtime_instance'):
                return model._runtime_instance  # noqa: SLF001
            return model
        else:
            return self._get_model_instance(model_config)

    def _sample_hyperparameters(self, trial, finetune_params: Dict[str, Any]) -> Dict[str, Any]:
        """Sample hyperparameters for Optuna trial."""
        params = {}

        # Get model parameters to optimize - support both nested and flat structure
        model_params = finetune_params.get('model_params', {})

        # If no model_params found, look for parameters directly in finetune_params (legacy support)
        if not model_params:
            model_params = {k: v for k, v in finetune_params.items()
                            if k not in ['n_trials', 'approach', 'train_params']}

        for param_name, param_config in model_params.items():
            if isinstance(param_config, list):
                # Categorical parameter
                params[param_name] = trial.suggest_categorical(param_name, param_config)
            elif isinstance(param_config, tuple) and len(param_config) == 3:
                # Tuple format: ('type', min, max)
                param_type, min_val, max_val = param_config
                if param_type == 'int':
                    params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                elif param_type == 'float':
                    params[param_name] = trial.suggest_float(param_name, float(min_val), float(max_val))
            elif isinstance(param_config, tuple) and len(param_config) == 2:
                # Range parameter - determine type from values
                min_val, max_val = param_config
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                else:
                    params[param_name] = trial.suggest_float(param_name, float(min_val), float(max_val))
            elif isinstance(param_config, dict):
                # Complex parameter configuration
                param_type = param_config.get('type', 'categorical')
                if param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
                elif param_type == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_config['min'], param_config['max'])
                elif param_type == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_config['min'], param_config['max'])

        return params

    def _is_grid_search_suitable(self, finetune_params: Dict[str, Any]) -> bool:
        """Check if grid search is suitable (all parameters are categorical)."""
        # Get model parameters to check - support both nested and flat structure
        model_params = finetune_params.get('model_params', {})

        # If no model_params found, look for parameters directly in finetune_params (legacy support)
        if not model_params:
            model_params = {k: v for k, v in finetune_params.items()
                            if k not in ['n_trials', 'approach', 'train_params']}

        for param_name, param_config in model_params.items():
            # Only categorical (list) parameters are suitable for grid search
            # Tuple parameters (ranges) need random/TPE sampling
            if not isinstance(param_config, list):
                return False
        return True

    def _create_grid_search_space(self, finetune_params: Dict[str, Any]) -> Dict[str, List]:
        """Create grid search space for categorical parameters only."""
        # Get model parameters to check - support both nested and flat structure
        model_params = finetune_params.get('model_params', {})

        # If no model_params found, look for parameters directly in finetune_params (legacy support)
        if not model_params:
            model_params = {k: v for k, v in finetune_params.items()
                            if k not in ['n_trials', 'approach', 'train_params']}

        search_space = {}
        for param_name, param_config in model_params.items():
            # Only include list parameters (categorical) in grid search
            # Explicitly ignore tuples which are range specifications
            if isinstance(param_config, list):
                search_space[param_name] = param_config
            # Ignore tuples like ('int', min, max) and ('float', min, max)
            elif isinstance(param_config, tuple):
                pass  # Skip range parameters
        return search_space

    @abstractmethod
    def _evaluate_model(self, model: Any, X_val: Any, y_val: Any) -> float:
        """Evaluate model performance (return score to minimize)."""
        pass

    def _clone_model(self, model: Any) -> Any:
        """Clone a model using framework-specific cloning methods."""
        framework = self._detect_framework(model)

        if framework == 'sklearn':
            from sklearn.base import clone
            return clone(model)
        elif framework == 'tensorflow':
            try:
                from tensorflow.keras.models import clone_model
                return clone_model(model)
            except ImportError:
                pass

        # Fallback to deepcopy
        from copy import deepcopy
        return deepcopy(model)

    def _detect_framework(self, model: Any) -> str:
        """Detect the framework from the model instance."""
        if hasattr(model, 'framework'):
            return model.framework

        import inspect
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

    def _store_results(
        self,
        model: Any,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        runner: 'PipelineRunner',
        model_type: str = "model"
    ) -> List[Tuple[str, bytes]]:
        """Store model and prediction results as binaries."""
        binaries = []

        # Serialize trained model
        try:
            model_binary = pickle.dumps(model)
            model_filename = f"{model_type}_{model.__class__.__name__}_{runner.next_op()}.pkl"
            binaries.append((model_filename, model_binary))
        except Exception as e:
            print(f"⚠️ Could not serialize model: {e}")

        # Store predictions as CSV
        try:
            predictions_csv = "y_true,y_pred\n"
            for true_val, pred_val in zip(y_true.flatten(), y_pred.flatten()):
                predictions_csv += f"{true_val},{pred_val}\n"

            pred_filename = f"predictions_{model_type}_{runner.next_op()}.csv"
            binaries.append((pred_filename, predictions_csv.encode('utf-8')))
        except Exception as e:
            print(f"⚠️ Could not store predictions: {e}")

        return binaries
