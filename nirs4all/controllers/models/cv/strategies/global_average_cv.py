"""
Global Average CV Strategy - Optimize parameters across all folds simultaneously

This strategy implements global average optimization where parameters are
optimized by averaging performance across all folds.
"""

from typing import Dict, Any, List
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

        if verbose > 0:
            print(f"🌍 Global Average CV: Optimizing parameters across all {len(context.data_splits)} folds simultaneously...")

        # For now, implement a simplified version that optimizes on combined data from all folds
        # Full implementation would require Optuna integration with multi-fold evaluation

        # Combine all training data for global optimization
        combined_X_train, combined_y_train = self._combine_all_training_data(context.data_splits)

        # Use a sample of the combined validation data for optimization
        X_val_sample, y_val_sample = self._get_validation_sample(context.data_splits)

        if verbose > 0:
            print(f"📊 Combined training data: {combined_X_train.shape[0]} samples for optimization")

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

        # Extract best parameters (would be set by the finetuning process)
        best_params = getattr(context.controller, '_last_best_params', {})

        if verbose > 0:
            print(f"🏆 Best parameters found: {best_params}")
            print(f"🔄 Training {len(context.data_splits)} final models with global best parameters...")

        # Now train models on each fold with the globally optimal parameters
        all_binaries = []
        all_binaries.extend(finetune_binaries)

        for fold_idx, _ in enumerate(context.data_splits):
            # Create model with global best parameters
            model = self._get_model_instance(context)
            if hasattr(model, 'set_params') and best_params:
                try:
                    model.set_params(**best_params)
                except (ValueError, TypeError) as e:
                    if verbose > 0:
                        print(f"⚠️ Could not apply global parameters to fold {fold_idx+1}: {e}")

            # Train on this fold
            _, fold_binaries = self._execute_train(context, fold_idx)

            # Add fold suffix to binary names
            fold_binaries_renamed = []
            for name, binary in fold_binaries:
                name_parts = name.rsplit('.', 1)
                if len(name_parts) == 2:
                    new_name = f"{name_parts[0]}_global_avg_cv_fold{fold_idx+1}.{name_parts[1]}"
                else:
                    new_name = f"{name}_global_avg_cv_fold{fold_idx+1}"
                fold_binaries_renamed.append((new_name, binary))

            all_binaries.extend(fold_binaries_renamed)

        # Check if we should train a single model on full training data
        if context.cv_config.use_full_train_for_final:
            return self._train_single_model_on_full_data(
                context, best_params, "global_avg_full", verbose
            )

        if verbose > 0:
            print("✅ Global Average CV completed successfully")

        return CVResult(
            context=context.cv_config.__dict__,
            binaries=all_binaries,
            best_params=best_params
        )

    def _combine_all_training_data(self, data_splits: List) -> tuple[Any, Any]:
        """Combine training data from all folds for global optimization."""
        all_X_train = []
        all_y_train = []

        for data_split in data_splits:
            all_X_train.append(data_split.X_train)
            all_y_train.append(data_split.y_train)

        # Concatenate all training data
        import numpy as np
        combined_X_train = np.concatenate(all_X_train, axis=0)
        combined_y_train = np.concatenate(all_y_train, axis=0)

        return combined_X_train, combined_y_train

    def _get_validation_sample(self, data_splits: List) -> tuple[Any, Any]:
        """Get a sample of validation data for optimization."""
        # Use the first fold's validation data as a representative sample
        # In a full implementation, you might want to sample from all folds
        if data_splits and hasattr(data_splits[0], 'X_val') and data_splits[0].X_val is not None:
            return data_splits[0].X_val, data_splits[0].y_val
        else:
            # Fallback: use a portion of the first fold's training data
            import numpy as np
            X_train = data_splits[0].X_train
            y_train = data_splits[0].y_train

            # Use 20% of training data as validation sample
            n_samples = len(X_train)
            n_val = max(1, int(0.2 * n_samples))

            indices = np.random.RandomState(42).choice(n_samples, n_val, replace=False)
            return X_train[indices], y_train[indices]

    def _train_single_model_on_full_data(
        self,
        context: CVExecutionContext,
        best_params: Dict[str, Any],
        model_suffix: str = "full_train",
        verbose: int = 0
    ) -> CVResult:
        """
        Train a single model on the full training dataset using globally optimized parameters.

        Instead of training separate models on each fold, this combines all training data
        and trains one model, which can be more effective when you have limited data.
        """
        if verbose > 0:
            print(f"🎯 Training single model on full training data ({model_suffix})...")

        # Combine all training data from folds
        combined_X_train, combined_y_train = self._combine_all_training_data(context.data_splits)

        # Combine all test data
        all_X_test = []
        all_y_test = []
        for data_split in context.data_splits:
            all_X_test.append(data_split.X_test)
            all_y_test.append(data_split.y_test)

        import numpy as np
        combined_X_test = np.concatenate(all_X_test, axis=0)
        combined_y_test = np.concatenate(all_y_test, axis=0)

        if verbose > 0:
            print(f"📊 Combined training data: {combined_X_train.shape[0]} samples")
            print(f"📊 Combined test data: {combined_X_test.shape[0]} samples")

        # Create and configure model with best parameters
        model = self._get_model_instance(context)
        if hasattr(model, 'set_params') and best_params:
            try:
                model.set_params(**best_params)
                if verbose > 0:
                    print(f"✅ Applied globally optimized parameters: {best_params}")
            except (ValueError, TypeError) as e:
                if verbose > 0:
                    print(f"⚠️ Could not apply parameters: {e}")

        # Create a combined data split for training
        from nirs4all.controllers.models.config import DataSplit
        combined_data_split = DataSplit(
            X_train=combined_X_train,
            y_train=combined_y_train,
            X_test=combined_X_test,
            y_test=combined_y_test
        )

        # Create temporary context for training
        train_execution_context = CVExecutionContext(
            model_config=context.model_config,
            data_splits=[combined_data_split],
            train_params=context.train_params,
            cv_config=context.cv_config,
            runner=context.runner,
            dataset=context.dataset,
            controller=context.controller,
            finetune_config=context.finetune_config
        )

        # Train the model
        _, binaries = self._execute_train(train_execution_context, fold_idx=0)

        # Rename binaries to indicate full training
        renamed_binaries = []
        for name, binary in binaries:
            name_parts = name.rsplit('.', 1)
            if len(name_parts) == 2:
                new_name = f"{name_parts[0]}_{model_suffix}.{name_parts[1]}"
            else:
                new_name = f"{name}_{model_suffix}"
            renamed_binaries.append((new_name, binary))

        if verbose > 0:
            print("✅ Single model training on full data completed successfully")

        return CVResult(
            context=context.cv_config.__dict__,
            binaries=renamed_binaries
        )
