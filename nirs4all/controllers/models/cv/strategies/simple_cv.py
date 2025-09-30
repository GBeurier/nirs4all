"""
Simple CV Strategy - Finetune on full training data, then train on folds

This strategy implements the simple cross-validation approach where hyperparameters
are optimized on the full training dataset, then final models are trained on each fold.
"""

from typing import Dict, Any, List, Tuple
import numpy as np

from nirs4all.controllers.models.cv.base import CVStrategy, CVExecutionContext, CVResult


class SimpleCVStrategy(CVStrategy):
    """
    Simple CV strategy: finetune on full data, then train on folds.

    This is the most straightforward CV approach:
    1. Combine all training data for hyperparameter optimization
    2. Train final models on each fold using optimized parameters
    """

    def execute(self, context: CVExecutionContext) -> CVResult:
        """
        Execute simple CV: finetune on full training data, then train on folds.

        Args:
            context: CV execution context

        Returns:
            CVResult: Results of the CV execution
        """
        verbose = context.train_params.get('verbose', 0)

        all_binaries = []

        if context.finetune_config is not None:
            # Finetuning mode
            if verbose > 0:
                print("🔍 Simple CV: Finetuning on full training data...")

            # Combine all training data for finetuning
            combined_X_train, combined_y_train = self._combine_training_data(context.data_splits)

            # Use first fold's validation data for finetuning evaluation
            X_val_sample, y_val_sample = context.data_splits[0].X_val, context.data_splits[0].y_val

            # Execute finetuning to get best parameters
            # For simple CV, we need to create a temporary data split for finetuning
            from nirs4all.controllers.models.config import DataSplit
            finetune_data_split = DataSplit(
                X_train=combined_X_train,
                y_train=combined_y_train,
                X_val=X_val_sample,
                y_val=y_val_sample,
                X_test=context.data_splits[0].X_test,  # Use first fold's test data
                y_test=context.data_splits[0].y_test
            )

            # Create temporary context for finetuning
            finetune_execution_context = CVExecutionContext(
                model_config=context.model_config,
                data_splits=[finetune_data_split],
                train_params=context.train_params,
                cv_config=context.cv_config,
                runner=context.runner,
                dataset=context.dataset,
                controller=context.controller,
                finetune_config=context.finetune_config
            )

            _, finetune_binaries = self._execute_finetune(finetune_execution_context, fold_idx=0)
            all_binaries.extend(finetune_binaries)

            # Extract best parameters from the finetuning process
            best_params = getattr(self, '_last_best_params', {})

            if verbose > 0:
                print(f"🏆 Best parameters found: {best_params}")
                print(f"🔄 Training {len(context.data_splits)} fold models with best parameters...")

            # Now train models on each fold with the best parameters
            for fold_idx, _ in enumerate(context.data_splits):
                # Create model with best parameters
                model = self._get_model_instance(context)
                if hasattr(model, 'set_params') and best_params:
                    try:
                        model.set_params(**best_params)
                    except (ValueError, TypeError) as e:
                        if verbose > 0:
                            print(f"⚠️ Could not apply best parameters to fold {fold_idx+1}: {e}")

                # Train on this fold
                _, fold_binaries = self._execute_train(context, fold_idx)

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
        else:
            # No finetuning, just train on each fold
            if verbose > 0:
                print(f"🔄 Training {len(context.data_splits)} fold models...")

            for fold_idx, _ in enumerate(context.data_splits):
                _, fold_binaries = self._execute_train(context, fold_idx)
                all_binaries.extend(fold_binaries)

        if verbose > 0:
            print("✅ Simple CV completed successfully")

        return CVResult(
            context=context.cv_config.__dict__,
            binaries=all_binaries,
            best_params=getattr(self, '_last_best_params', {}) if context.finetune_config is not None else {}
        )

    def _combine_training_data(self, data_splits: List) -> Tuple[Any, Any]:
        """Combine training data from all folds."""
        all_X_train = []
        all_y_train = []

        for data_split in data_splits:
            all_X_train.append(data_split.X_train)
            all_y_train.append(data_split.y_train)

        # Concatenate all training data
        combined_X_train = np.concatenate(all_X_train, axis=0)
        combined_y_train = np.concatenate(all_y_train, axis=0)

        return combined_X_train, combined_y_train
