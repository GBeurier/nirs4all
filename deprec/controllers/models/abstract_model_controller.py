"""
Abstract Model Controller - Refactored, modular model controller

This is a complete refactoring of the monolithic BaseModelController into a clean,
modular architecture. All external signatures are preserved while internal
implementation uses composition and strategy patterns.

Key improvements:
- Single Responsibility: Each module handles one concern
- Testability: Small, focused classes are easy to unit test
- Extensibility: New CV strategies and frameworks can be added easily
- Maintainability: Clear separation of concerns and reduced coupling
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, Union, TYPE_CHECKING
import numpy as np

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.models.config import ModelConfig, CVConfig, FinetuneConfig
from nirs4all.controllers.models.data import DataManager
from nirs4all.controllers.models.model import ModelManager
from nirs4all.controllers.models.cv.factory import CVStrategyFactory
from nirs4all.controllers.models.prediction import PredictionManager
from nirs4all.controllers.models.results import ResultManager
from nirs4all.controllers.models.optuna_manager import OptunaManager
from nirs4all.controllers.models.enums import ModelMode
from nirs4all.utils.model_utils import ModelUtils, TaskType
from nirs4all.controllers.models.model_naming import ModelNamingManager
from nirs4all.controllers.models.cv_averaging import CVAveragingManager

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset


class AbstractModelController(OperatorController, ABC):
    """
    Abstract base controller for machine learning models - Refactored Version.

    This refactored controller uses composition and strategy patterns to provide
    a clean, modular architecture while maintaining all external interfaces.

    Key components:
    - DataManager: Handles data preparation and splitting
    - ModelManager: Manages model instantiation and framework operations
    - CV Strategies: Pluggable cross-validation approaches
    - PredictionManager: Handles prediction generation and storage
    - ResultManager: Manages result serialization and storage
    """

    priority = 15  # Higher priority than transformers, lower than data operations

    def __init__(self):
        """Initialize the refactored model controller with all components."""
        super().__init__()

        # Core components using composition
        self.data_manager = DataManager()
        self.model_manager = ModelManager()
        self.prediction_manager = PredictionManager()
        self.result_manager = ResultManager()
        self.cv_factory = CVStrategyFactory()
        self.optuna_manager = OptunaManager()

        # New components for consistent naming and averaging
        self.naming_manager = ModelNamingManager()
        self.averaging_manager = CVAveragingManager(self.naming_manager)

    @classmethod
    def use_multi_source(cls) -> bool:
        """Models can handle multi-source datasets."""
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Model controllers support prediction mode."""
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
        """Prepare data in framework-specific format."""
        pass

    @abstractmethod
    def get_preferred_layout(self) -> str:
        """Get the preferred data layout for this model type."""
        pass

    def execute(
        self,
        step: Any,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, bytes]]] = None
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """
        Execute model controller in one of three modes: train, finetune, or predict.

        This method maintains the same external signature as the original while
        using the new modular architecture internally.

        Args:
            step: Pipeline step configuration
            operator: Model operator (unused, model comes from step config)
            dataset: Dataset containing features and targets
            context: Pipeline context with processing state
            runner: Pipeline runner instance
            source: Data source index
            mode: Execution mode ("train" or "predict")
            loaded_binaries: Pre-loaded binary objects for prediction mode

        Returns:
            Tuple of (updated_context, binaries_list)
        """
        # In prediction mode, use loaded model for prediction
        if mode == "predict":
            return self._execute_prediction_mode(
                step, operator, dataset, context, runner, loaded_binaries
            )

        # Training/finetuning mode - using new modular architecture
        return self._execute_training_mode_modular(
            step, operator, dataset, context, runner
        )


    def _execute_training_mode_modular(
        self,
        step: Any,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner'
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute training/finetuning mode using the modular architecture."""
        # Extract and validate configuration
        model_config = self._extract_model_config(step, operator)
        cv_config = self._extract_cv_config(model_config)

        print(f"[DEBUG] Model config: {model_config}")  # Debugging output to verify model config
        print(f"[DEBUG] CV config: {cv_config}")      # Debugging output to verify CV config

        # Prepare data splits
        data_splits = self.data_manager.prepare_train_test_data(dataset, context)

        # Choose execution path based on data structure
        if isinstance(data_splits, list):
            # Cross-validation mode
            return self._execute_cross_validation_modular(
                model_config, cv_config, data_splits, context, runner, dataset
            )
        else:
            # Single training mode
            return self._execute_single_training_modular(
                model_config, data_splits, context, runner, dataset
            )

    def _execute_cross_validation_modular(
        self,
        model_config: Dict[str, Any],
        cv_config: CVConfig,
        data_splits: List,
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        dataset: 'SpectroDataset'
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute cross-validation using the strategy pattern."""
        verbose = model_config.get('train_params', {}).get('verbose', 0)

        # Create and execute the appropriate CV strategy
        strategy = self.cv_factory.create_strategy(cv_config.mode)

        # Set up the execution context for the strategy
        from nirs4all.controllers.models.cv.base import CVExecutionContext
        finetune_params = model_config.get('finetune_params')
        finetune_config = FinetuneConfig.from_dict(finetune_params) if finetune_params else None

        execution_context = CVExecutionContext(
            model_config=model_config,
            data_splits=data_splits,
            train_params=model_config.get('train_params', {}),
            cv_config=cv_config,
            runner=runner,
            dataset=dataset,
            controller=self,
            finetune_config=finetune_config
        )

        # Execute the strategy
        result = strategy.execute(execution_context)

        # After CV training, generate average and weighted average predictions
        if len(data_splits) > 1:  # Only for multi-fold CV
            try:
                avg_binaries, avg_metadata = self.averaging_manager.generate_average_predictions(
                    dataset=dataset,
                    model_config=model_config,
                    runner=runner,
                    context=context,
                    fold_count=len(data_splits),
                    verbose=verbose
                )
                result.binaries.extend(avg_binaries)
            except Exception as e:
                if verbose > 1:
                    print(f"⚠️ Could not generate average predictions: {e}")

        return result.context, result.binaries

    def _execute_single_training_modular(
        self,
        model_config: Dict[str, Any],
        data_split,
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        dataset: 'SpectroDataset'
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute single training mode using modular components."""
        # Determine execution mode
        mode = self._determine_mode(model_config)
        assert mode is not None, "Mode should never be None"

        if mode.name == "FINETUNE" and model_config.get('finetune_params'):
            return self._execute_finetune_modular(
                model_config, data_split.X_train, data_split.y_train,
                data_split.X_val, data_split.y_val, data_split.X_test, data_split.y_test,
                model_config.get('train_params', {}),
                FinetuneConfig.from_dict(model_config['finetune_params']),
                context, runner, dataset
            )
        else:
            return self._execute_train_modular(
                model_config, data_split.X_train, data_split.y_train,
                data_split.X_val, data_split.y_val, data_split.X_test, data_split.y_test,
                model_config.get('train_params', {}), context, runner, dataset
            )

    def _execute_train_modular(
        self,
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        X_test: Any,
        y_test: Any,
        train_params: Dict[str, Any],
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        dataset: 'SpectroDataset',
        fold_idx: Optional[int] = None
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute training using modular components."""
        verbose = train_params.get('verbose', 0)

        if verbose > 1:
            print("🏋️ Training model...")

        # Get model instance and prepare data
        base_model = self._get_model_instance(model_config)
        model = self.model_manager.clone_model(base_model)

        X_train_prep, y_train_prep = self._prepare_data(X_train, y_train, context)
        X_val_prep, y_val_prep = self._prepare_data(X_val, y_val, context) if X_val is not None else (None, None)
        X_test_prep, y_test_prep = self._prepare_data(X_test, y_test, context)

        # Train model
        trained_model = self._train_model(
            model, X_train_prep, y_train_prep, X_val=X_val_prep, y_val=y_val_prep,
            train_params=train_params
        )

        # Generate predictions
        y_pred_train = self.prediction_manager.generate_predictions(trained_model, X_train_prep)
        y_pred_test = self.prediction_manager.generate_predictions(trained_model, X_test_prep)
        y_pred_val = self.prediction_manager.generate_predictions(trained_model, X_val_prep) if X_val_prep is not None else None

        # Debugging: show fold and naming information to trace duplicate fold indices
        try:
            runner_verbose = getattr(runner, 'verbose', 0)
            effective_verbose = max(verbose, runner_verbose)
            if effective_verbose > 1:
                # base name (no operation counter) and sizes
                base_name = self.model_manager.get_base_model_name({}, trained_model)
                y_test_len = len(y_test) if y_test is not None else 0
                print(f"DEBUG: _execute_train_modular: fold_idx={fold_idx}, base_model={base_name}, y_test_len={y_test_len}, runner_verbose={runner_verbose}")
        except Exception:
            pass

        # Store predictions
        self._store_predictions_modular(
            trained_model, y_train, y_pred_train, y_val, y_pred_val,
            y_test, y_pred_test, runner, dataset, context, fold_idx
        )

        # Create result binaries
        binaries = self.result_manager.create_result_binaries(
            trained_model, y_pred_test, y_test, runner, "trained"
        )

        # Display results
        self._display_training_results(y_test, y_pred_test, trained_model, runner, fold_idx)

        # Save predictions to results folder
        self.result_manager.save_to_results_folder(dataset._predictions, dataset, runner)

        return context, binaries

    def _execute_finetune_modular(
        self,
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        X_test: Any,
        y_test: Any,
        train_params: Dict[str, Any],
        finetune_config: FinetuneConfig,
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        dataset: 'SpectroDataset',
        fold_idx: Optional[int] = None
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute finetuning using Optuna hyperparameter optimization."""
        verbose = train_params.get('verbose', 0)

        if verbose > 1:
            print("🎯 Starting hyperparameter optimization with Optuna...")

        try:
            # Create objective function for Optuna
            objective_function = self.optuna_manager.create_objective_function(
                model_config, X_train, y_train, X_val, y_val, self, train_params
            )

            # Run hyperparameter optimization
            finetune_dict = {
                'n_trials': finetune_config.n_trials,
                'approach': finetune_config.approach,
                'model_params': finetune_config.model_params,
                'train_params': finetune_config.train_params,
                'verbose': finetune_config.verbose
            }
            best_params, best_score = self.optuna_manager.optimize_hyperparameters(
                objective_function, finetune_dict, verbose
            )

            if verbose > 1:
                print(f"🏆 Optimization completed. Best score: {best_score:.4f}")

            # Train final model with best parameters
            if verbose > 1:
                print("🏋️ Training final model with optimized parameters...")

            # Create model with best parameters
            base_model = self._get_model_instance(model_config)
            final_model = self.model_manager.clone_model(base_model)

            # Apply best parameters
            if hasattr(final_model, 'set_params') and best_params:
                final_model.set_params(**best_params)

            # Prepare data
            X_train_prep, y_train_prep = self._prepare_data(X_train, y_train, context)
            X_val_prep, y_val_prep = self._prepare_data(X_val, y_val, context)
            X_test_prep, y_test_prep = self._prepare_data(X_test, y_test, context)

            # Train final model
            trained_model = self._train_model(
                final_model, X_train_prep, y_train_prep,
                X_val=X_val_prep, y_val=y_val_prep,
                train_params=train_params
            )

            # Generate predictions
            y_pred_val = self.prediction_manager.generate_predictions(trained_model, X_val_prep)
            y_pred_train = self.prediction_manager.generate_predictions(trained_model, X_train_prep)
            y_pred_test = self.prediction_manager.generate_predictions(trained_model, X_test_prep)

            # Store predictions
            self._store_predictions_modular(
                trained_model, y_train, y_pred_train, y_val, y_pred_val,
                y_test, y_pred_test, runner, dataset, context, fold_idx
            )

            # Create result binaries
            binaries = self.result_manager.create_result_binaries(
                trained_model, y_pred_test, y_test, runner, "finetuned"
            )

            # Display results
            self._display_training_results(y_test, y_pred_test, trained_model, runner, fold_idx)

            # Save predictions to results folder
            self.result_manager.save_to_results_folder(dataset._predictions, dataset, runner)

            return context, binaries

        except ImportError:
            # Optuna not available, fall back to regular training
            if verbose > 1:
                print("⚠️ Optuna not available, falling back to regular training...")
            return self._execute_train_modular(
                model_config, X_train, y_train, X_val, y_val, X_test, y_test,
                train_params, context, runner, dataset, fold_idx
            )
        except Exception as e:
            # Any other error, fall back to regular training
            if verbose > 1:
                print(f"⚠️ Optuna optimization failed ({e}), falling back to regular training...")
            return self._execute_train_modular(
                model_config, X_train, y_train, X_val, y_val, X_test, y_test,
                train_params, context, runner, dataset, fold_idx
            )

    def _execute_prediction_mode(
        self,
        step: Any,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        loaded_binaries: Optional[List[Tuple[str, bytes]]]
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute prediction mode using loaded model."""
        if not loaded_binaries:
            raise ValueError("No loaded binaries provided for prediction mode")

        # Find the model binary
        model_binary = None
        for name, binary in loaded_binaries:
            if name.endswith('.pkl') and ('model' in name.lower() or 'trained' in name.lower() or 'finetuned' in name.lower()):
                model_binary = binary
                break

        if model_binary is None:
            raise ValueError("No model binary found in loaded binaries")

        # Load the trained model
        import pickle
        if isinstance(model_binary, bytes):
            trained_model = pickle.loads(model_binary)
        else:
            trained_model = model_binary

        # Prepare prediction data
        layout_str = self.get_preferred_layout()
        prediction_context = context.copy()

        # Try to use test partition, fallback to train if test doesn't exist
        # Try to use test partition, fallback to train if test doesn't exist, then to all data
        available_partitions = dataset._indexer.uniques('partition') if hasattr(dataset, '_indexer') else []

        if "test" in available_partitions:
            prediction_context["partition"] = "test"
        elif "train" in available_partitions:
            prediction_context["partition"] = "train"
        else:
            prediction_context.pop("partition", None)

        # Get prediction data
        test_data = self.data_manager.get_test_data(dataset, prediction_context)
        X_pred_prep, _ = self._prepare_data(test_data.X_test, test_data.y_test, context)

        # Generate predictions
        y_pred = self.prediction_manager.generate_predictions(trained_model, X_pred_prep)

        # Store predictions if y data is available
        try:
            if test_data.y_test is not None and len(test_data.y_test) > 0:
                base_model_name, instance_name, pipeline_path, custom_model_name = self._get_model_names(step, runner, None)
                self.prediction_manager.store_predictions(
                    dataset=getattr(runner.saver, 'dataset_name', 'unknown') or 'unknown',
                    pipeline=getattr(runner.saver, 'pipeline_name', 'unknown') or 'unknown',
                    pipeline_path=pipeline_path,
                    model=base_model_name,
                    real_model=instance_name,
                    partition="prediction",
                    y_true=test_data.y_test,
                    y_pred=y_pred,
                    fold_idx=None,
                    context=context,
                    dataset_obj=dataset,
                    custom_model_name=custom_model_name
                )
        except Exception:
            pass  # No y data available for prediction

        # Create prediction CSV binary
        try:
            predictions_csv = self.result_manager.create_prediction_csv(test_data.y_test or np.array([]), y_pred)
            pred_filename = f"predictions_predict_{runner.next_op()}.csv"
            binaries = [(pred_filename, predictions_csv.encode('utf-8'))]
        except Exception as e:
            print(f"⚠️ Could not store predictions: {e}")
            binaries = []

        return context, binaries

    def _store_predictions_modular(
        self,
        trained_model: Any,
        y_train: Any,
        y_pred_train: Any,
        y_val: Any,
        y_pred_val: Any,
        y_test: Any,
        y_pred_test: Any,
        runner: 'PipelineRunner',
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        fold_idx: Optional[int] = None
    ) -> None:
        """Store predictions using the modular prediction manager."""
        # Use naming manager for consistent model identifiers
        model_config = {'model_instance': trained_model}  # Simple config for trained model
        identifiers = self.naming_manager.create_model_identifiers(
            model_config, runner, fold_idx=fold_idx
        )

        dataset_name = getattr(runner.saver, 'dataset_name', 'unknown') or 'unknown'
        pipeline_name = getattr(runner.saver, 'pipeline_name', 'unknown') or 'unknown'
        pipeline_path = str(runner.saver.current_path) if runner.saver.current_path else ""

        # Store predictions for each partition
        if fold_idx is not None:
            # Fold-based storage - use val partition for test data in CV
            self.prediction_manager.store_predictions(
                dataset=dataset_name, pipeline=pipeline_name, pipeline_path=pipeline_path,
                model=identifiers.classname, real_model=identifiers.model_uuid, partition="val",
                y_true=y_test, y_pred=y_pred_test, fold_idx=fold_idx,
                context=context, dataset_obj=dataset, custom_model_name=identifiers.custom_name
            )
            self.prediction_manager.store_predictions(
                dataset=dataset_name, pipeline=pipeline_name, pipeline_path=pipeline_path,
                model=identifiers.classname, real_model=identifiers.model_uuid, partition="train",
                y_true=y_train, y_pred=y_pred_train, fold_idx=fold_idx,
                context=context, dataset_obj=dataset, custom_model_name=identifiers.custom_name
            )
        else:
            # Global storage
            self.prediction_manager.store_predictions(
                dataset=dataset_name, pipeline=pipeline_name, pipeline_path=pipeline_path,
                model=identifiers.classname, real_model=identifiers.model_uuid, partition="test",
                y_true=y_test, y_pred=y_pred_test, fold_idx=fold_idx,
                context=context, dataset_obj=dataset, custom_model_name=identifiers.custom_name
            )

    def _display_training_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        trained_model: Any,
        runner: 'PipelineRunner',
        fold_idx: Optional[int] = None
    ) -> None:
        """Display training results with scores."""
        task_type = self._detect_task_type(y_true)

        # Use naming manager for consistent display names
        model_config = {'model_instance': trained_model}  # Simple config for display
        identifiers = self.naming_manager.create_model_identifiers(
            model_config, runner, fold_idx=fold_idx
        )
        display_name = identifiers.model_id

        test_scores = self._calculate_and_print_scores(
            y_true, y_pred, task_type, "test", display_name,
            show_detailed_scores=False
        )

        # Display concise final summary
        best_metric, higher_is_better = ModelUtils.get_best_score_metric(task_type)
        best_score = test_scores.get(best_metric)
        if best_score is not None:
            direction = "↑" if higher_is_better else "↓"
            score_display = f"{best_metric}={best_score:.4f}{direction}"

            dataset_info = f"(train:{len(y_true)})" if fold_idx is None else f"(fold:{fold_idx})"

            other_scores = {k: v for k, v in test_scores.items() if k != best_metric}
            if other_scores:
                other_scores_str = ModelUtils.format_scores(other_scores)
                print(f"✅ {display_name} - test: {score_display} ({other_scores_str}) {dataset_info}")
            else:
                print(f"✅ {display_name} - test: {score_display} {dataset_info}")
        else:
            dataset_info = f"(train:{len(y_true)})" if fold_idx is None else f"(fold:{fold_idx})"
            print(f"✅ Model {display_name} completed successfully {dataset_info}")

    # Legacy compatibility methods - these maintain the original interface

    def _extract_model_config(self, step: Any, operator: Any = None) -> Dict[str, Any]:
        """Extract model configuration from step (legacy compatibility)."""
        # If we have a deserialized operator, use it directly
        if operator is not None:
            if isinstance(step, dict):
                config = step.copy()
                config['model_instance'] = operator
                return config
            else:
                return {'model_instance': operator}

        # Fall back to original logic for backward compatibility
        if isinstance(step, dict):
            if 'model' in step:
                config = step.copy()
                model_obj = step['model']

                # Handle nested model format: {'model': {'name': 'X', 'model': PLSRegression()}}
                if isinstance(model_obj, dict):
                    if 'model' in model_obj:
                        # Extract the actual model from nested structure
                        actual_model = model_obj['model']
                        if isinstance(actual_model, dict) and '_runtime_instance' in actual_model:
                            config['model_instance'] = actual_model['_runtime_instance']
                        else:
                            config['model_instance'] = actual_model
                        # Preserve custom name if provided
                        if 'name' in model_obj:
                            config['name'] = model_obj['name']
                    elif '_runtime_instance' in model_obj:
                        config['model_instance'] = model_obj['_runtime_instance']
                    else:
                        # Dict without 'model' key, treat as serialized model
                        config['model_instance'] = model_obj
                elif hasattr(model_obj, '_runtime_instance'):
                    config['model_instance'] = model_obj._runtime_instance  # type: ignore
                else:
                    config['model_instance'] = model_obj
                return config
            else:
                return {'model_instance': step}
        else:
            return {'model_instance': step}

    def _extract_cv_config(self, model_config: Dict[str, Any]) -> CVConfig:
        """Extract CV configuration from model config."""
        finetune_params = model_config.get('finetune_params', {})
        return CVConfig.from_dict(finetune_params)

    def _determine_mode(self, model_config: Dict[str, Any]):
        """Determine execution mode (legacy compatibility)."""
        if 'finetune_params' in model_config and model_config['finetune_params']:
            return ModelMode.FINETUNE
        else:
            return ModelMode.TRAIN

    def _detect_task_type(self, y: np.ndarray) -> TaskType:
        """Detect task type from target values."""
        return ModelUtils.detect_task_type(y)

    def _calculate_and_print_scores(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: TaskType,
        partition: str = "test",
        model_name: str = "model",
        metrics: Optional[List[str]] = None,
        show_detailed_scores: bool = True
    ) -> Dict[str, float]:
        """Calculate scores and print them (legacy compatibility)."""
        scores = ModelUtils.calculate_scores(y_true, y_pred, task_type, metrics)

        if scores and show_detailed_scores:
            print(f"📊 {model_name} {partition} scores: {ModelUtils.format_scores(scores)}")

        return scores

    def _get_base_model_name(self, step: Any, trained_model: Any = None) -> str:
        """Extract the base model name (legacy compatibility)."""
        return self.model_manager.get_base_model_name(step, trained_model)

    def _get_instance_name(self, base_name: str, runner) -> str:
        """Generate the instance name (legacy compatibility)."""
        unique_index = runner.next_op()
        return f"{base_name}_{unique_index}"

    def _get_informative_name(
        self, instance_name: str, fold_idx: Optional[int] = None,
        is_avg: bool = False, is_weighted_avg: bool = False
    ) -> str:
        """Generate the informative name (legacy compatibility)."""
        if is_weighted_avg:
            return f"{instance_name}_w_avg"
        elif is_avg:
            return f"{instance_name}_avg"
        elif fold_idx is not None:
            return f"{instance_name}_fold{fold_idx}"
        else:
            return instance_name

    def _get_model_names(self, model_config: Dict[str, Any], runner, fold_idx: Optional[int] = None):
        """Generate proper model names (legacy compatibility)."""
        base_model_name = self._get_base_model_name(model_config)
        instance_name = self._get_instance_name(base_model_name, runner)
        pipeline_path = str(runner.saver.current_path) if runner.saver.current_path else ""
        custom_model_name = model_config.get('name', None)
        return base_model_name, instance_name, pipeline_path, custom_model_name
