"""
Global Average CV Strategy - Optimize parameters across all folds simultaneously

This strategy implements global average optimization where parameters are
optimized by averaging performance across all folds.
"""

from typing import Dict, Any, List, Tuple
from nirs4all.controllers.models.cv.base import CVStrategy, CVExecutionContext, CVResult


class GlobalAverageCVStrategy(CVStrategy):
    """
    Global average CV strategy: optimize parameters across all folds simultaneously.

    This strategy evaluates hyperparameter combinations on all folds and selects
    the parameters that perform best on average across all folds.
    """

    def execute(self, context: CVExecutionContext) -> CVResult:
        """
        Execute global average CV: optimize parameters across all folds simultaneously.

        Args:
            context: CV execution context

        Returns:
            CVResult: Results of the global average CV execution
        """
        verbose = context.train_params.get('verbose', 0)
        all_binaries = []
        best_params = {}

        if verbose > 1:
            print("🌍 Global Average CV: Optimizing parameters across all folds...")

        if context.finetune_config is not None:
            # Phase 1: Global optimization on combined data
            combined_X_train, combined_y_train = self._combine_all_training_data(context.data_splits)
            X_val_sample, y_val_sample = self._get_validation_sample(context.data_splits)

            if verbose > 1:
                print(f"🔬 Combined training data shape: {combined_X_train.shape}")

            # Create a combined data split for optimization
            from nirs4all.controllers.models.config import DataSplit
            combined_data_split = DataSplit(
                X_train=combined_X_train,
                y_train=combined_y_train,
                X_val=X_val_sample,
                y_val=y_val_sample
            )

            # Create temporary context for global optimization
            global_execution_context = CVExecutionContext(
                model_config=context.model_config,
                data_splits=[combined_data_split],
                train_params=context.train_params,
                cv_config=context.cv_config,
                runner=context.runner,
                dataset=context.dataset,
                controller=context.controller,
                finetune_config=context.finetune_config
            )

            # Execute finetuning on the combined data
            _, finetune_binaries = self._execute_finetune(global_execution_context, fold_idx=0)
            all_binaries.extend(finetune_binaries)

            # Extract best parameters
            best_params = getattr(context.controller, '_last_best_params', {})

            if verbose > 1:
                print(f"🏆 Global optimization completed. Best parameters: {best_params}")

        # Phase 2: Train models on each fold with globally optimal parameters
        if verbose > 1:
            fold_count = len(context.data_splits)
            print(f"🔄 Training {fold_count} fold models with {'globally optimal' if best_params else 'default'} parameters...")

        for fold_idx, _ in enumerate(context.data_splits):
            if verbose > 1:
                print(f"📈 Training fold {fold_idx + 1}/{len(context.data_splits)}")

            # Create fold context with best parameters
            fold_context = self._create_fold_context(context, best_params)
            _, fold_binaries = self._execute_train(fold_context, fold_idx)

            # Rename binaries to include fold information
            fold_binaries_renamed = self._rename_fold_binaries(fold_binaries, fold_idx)
            all_binaries.extend(fold_binaries_renamed)

        if verbose > 1:
            print("✅ Global Average CV completed successfully")

        return CVResult(
            context=context.cv_config.__dict__,
            binaries=all_binaries,
            best_params=best_params
        )

    def _combine_all_training_data(self, data_splits: List) -> tuple[Any, Any]:
        """Combine training data from all folds."""
        import numpy as np
        all_X_train = []
        all_y_train = []

        for data_split in data_splits:
            all_X_train.append(data_split.X_train)
            all_y_train.append(data_split.y_train)

        combined_X_train = np.concatenate(all_X_train, axis=0)
        combined_y_train = np.concatenate(all_y_train, axis=0)

        return combined_X_train, combined_y_train

    def _get_validation_sample(self, data_splits: List) -> tuple[Any, Any]:
        """Get validation sample from all folds."""
        import numpy as np
        # Use validation data from all folds combined
        all_X_val = []
        all_y_val = []

        for data_split in data_splits:
            if data_split.X_val is not None and data_split.y_val is not None:
                all_X_val.append(data_split.X_val)
                all_y_val.append(data_split.y_val)

        if all_X_val:
            X_val_sample = np.concatenate(all_X_val, axis=0)
            y_val_sample = np.concatenate(all_y_val, axis=0)
        else:
            # Fallback to first fold's validation data
            X_val_sample = data_splits[0].X_val
            y_val_sample = data_splits[0].y_val

        return X_val_sample, y_val_sample

    def _create_fold_context(self, context: CVExecutionContext, best_params: Dict[str, Any]) -> CVExecutionContext:
        """Create a context for fold training with best parameters applied."""
        # Create a new model config with best parameters applied
        fold_model_config = context.model_config.copy()

        # Apply best parameters to the model if available
        if best_params and 'model_instance' in fold_model_config:
            model = fold_model_config['model_instance']
            if hasattr(model, 'set_params'):
                try:
                    model = context.controller.model_manager.clone_model(model)
                    model.set_params(**best_params)
                    fold_model_config['model_instance'] = model
                except (ValueError, TypeError) as e:
                    # If parameter application fails, use original model
                    pass

        # Return context with updated model config
        return CVExecutionContext(
            model_config=fold_model_config,
            data_splits=context.data_splits,
            train_params=context.train_params,
            cv_config=context.cv_config,
            runner=context.runner,
            dataset=context.dataset,
            controller=context.controller,
            finetune_config=None  # No finetuning for individual folds
        )

    def _rename_fold_binaries(self, fold_binaries: List[Tuple[str, bytes]], fold_idx: int) -> List[Tuple[str, bytes]]:
        """Rename binaries to include fold information."""
        fold_binaries_renamed = []
        for name, binary in fold_binaries:
            name_parts = name.rsplit('.', 1)
            if len(name_parts) == 2:
                new_name = f"{name_parts[0]}_global_avg_fold{fold_idx}.{name_parts[1]}"
            else:
                new_name = f"{name}_global_avg_fold{fold_idx}"
            fold_binaries_renamed.append((new_name, binary))
        return fold_binaries_renamed
