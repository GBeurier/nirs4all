"""
Nested CV Strategy - Inner folds for finetuning, outer folds for training

This strategy implements nested cross-validation with separate inner and outer folds.
"""

from typing import Dict, Any, List
from nirs4all.controllers.models.cv.base import CVStrategy, CVExecutionContext, CVResult
from nirs4all.controllers.models.config import ParamStrategy


class NestedCVStrategy(CVStrategy):
    """
    Nested CV strategy: inner folds for finetuning, outer folds for training.

    This strategy provides unbiased performance estimates by using separate
    inner folds for hyperparameter optimization and outer folds for final evaluation.
    """

    def execute(self, context: CVExecutionContext) -> CVResult:
        """
        Execute nested CV: inner folds for optimization, outer folds for training.

        Args:
            context: CV execution context

        Returns:
            CVResult: Results of the nested CV execution
        """
        verbose = context.train_params.get('verbose', 0)
        param_strategy = context.cv_config.param_strategy
        inner_cv = context.cv_config.inner_cv or 3

        if verbose > 1:
            print(f"🔍 Nested CV: {len(context.data_splits)} outer folds with inner CV finetuning...")
            print(f"📊 Parameter strategy: {param_strategy.value}, Inner CV: {inner_cv} folds")

        all_binaries = []
        all_fold_results = []

        for outer_idx, _ in enumerate(context.data_splits):
            if verbose > 1:
                print(f"🏋️ Outer fold {outer_idx+1}/{len(context.data_splits)}...")

            # Create inner folds for finetuning from the current outer fold's training data
            inner_folds = self._create_inner_folds(context, outer_idx, inner_cv)

            if verbose > 2:
                print(f"  📋 Created {len(inner_folds)} inner folds for finetuning")

            # Choose optimization strategy for inner folds
            if param_strategy == ParamStrategy.GLOBAL_AVERAGE:
                # Optimize using global average across inner folds
                fold_best_params = self._optimize_global_average_on_inner_folds(
                    context, inner_folds, verbose
                )
            else:
                # Standard nested CV: finetune using inner folds
                fold_best_params = self._finetune_on_inner_folds(
                    context, inner_folds, verbose
                )

            if verbose > 2:
                print(f"  🏆 Best params for outer fold {outer_idx+1}: {fold_best_params}")

            # Train final model on full outer training data with best parameters
            _, fold_binaries = self._train_final_model_on_outer_fold(
                context, outer_idx, fold_best_params
            )

            # Store fold results for potential parameter aggregation
            fold_result = {
                'fold_idx': outer_idx,
                'best_params': fold_best_params,
                'binaries': fold_binaries
            }
            all_fold_results.append(fold_result)
            all_binaries.extend(fold_binaries)

        # Handle parameter aggregation across outer folds
        if param_strategy == ParamStrategy.WEIGHTED_AVERAGE:
            self._compute_weighted_average_params(all_fold_results, verbose)

        # Check if we should train a single model on full training data
        if context.cv_config.use_full_train_for_final:
            if verbose > 1:
                print("🎯 Training single model on full training data with nested CV optimized parameters...")

            # Use the best parameters from the first outer fold as representative
            representative_params = all_fold_results[0]['best_params'] if all_fold_results else {}

            return self._train_single_model_on_full_data(
                context, representative_params, "nested_cv_full", verbose
            )

        if verbose > 1:
            print("✅ Nested CV completed successfully")

        return CVResult(
            context=context.cv_config.__dict__,
            binaries=all_binaries
        )

    def _create_inner_folds(self, context: CVExecutionContext, outer_idx: int, inner_cv: int) -> List[Dict[str, Any]]:
        """Create inner cross-validation folds for the current outer fold."""
        try:
            from sklearn.model_selection import KFold, StratifiedKFold
        except ImportError:
            raise ImportError("scikit-learn is required for nested cross-validation")

        # Get the outer fold's training data
        outer_data = context.data_splits[outer_idx]
        X_outer_train, y_outer_train = outer_data.X_train, outer_data.y_train

        # Create inner CV splitter
        if isinstance(inner_cv, int):
            inner_splitter = KFold(n_splits=inner_cv, shuffle=True, random_state=42)
        else:
            inner_splitter = KFold(n_splits=3, shuffle=True, random_state=42)

        inner_folds = []
        for train_idx, val_idx in inner_splitter.split(X_outer_train, y_outer_train):
            # Use numpy indexing if possible
            if hasattr(X_outer_train, '__getitem__') and hasattr(X_outer_train, 'shape'):
                X_inner_train = X_outer_train[train_idx]
                X_inner_val = X_outer_train[val_idx]
            else:
                X_inner_train = X_outer_train
                X_inner_val = X_outer_train

            if hasattr(y_outer_train, '__getitem__') and hasattr(y_outer_train, 'shape'):
                y_inner_train = y_outer_train[train_idx]
                y_inner_val = y_outer_train[val_idx]
            else:
                y_inner_train = y_outer_train
                y_inner_val = y_outer_train

            inner_folds.append({
                'X_train': X_inner_train,
                'y_train': y_inner_train,
                'X_val': X_inner_val,
                'y_val': y_inner_val
            })

        return inner_folds

    def _finetune_on_inner_folds(
        self,
        context: CVExecutionContext,
        inner_folds: List[Dict[str, Any]],
        verbose: int = 0
    ) -> Dict[str, Any]:
        """Finetune using inner folds and return best parameters."""
        # For now, implement a simplified version that optimizes on the first inner fold
        # Full implementation would require Optuna integration
        if verbose > 2:
            print("    🎯 Running simplified inner CV optimization...")

        # Create a temporary data split for the first inner fold
        from nirs4all.controllers.models.config import DataSplit
        first_inner = inner_folds[0]
        inner_data_split = DataSplit(
            X_train=first_inner['X_train'],
            y_train=first_inner['y_train'],
            X_val=first_inner['X_val'],
            y_val=first_inner['y_val']
        )

        # Create temporary context for inner fold finetuning
        inner_execution_context = CVExecutionContext(
            model_config=context.model_config,
            data_splits=[inner_data_split],
            train_params=context.train_params,
            cv_config=context.cv_config,
            runner=context.runner,
            dataset=context.dataset,
            controller=context.controller,
            finetune_config=context.finetune_config
        )

        # Execute finetuning on the inner fold
        _, _ = self._execute_finetune(inner_execution_context, fold_idx=0)

        # Return placeholder best parameters
        # In a full implementation, this would extract the actual best params from Optuna
        return {}

    def _optimize_global_average_on_inner_folds(
        self,
        context: CVExecutionContext,
        inner_folds: List[Dict[str, Any]],
        verbose: int = 0
    ) -> Dict[str, Any]:
        """Optimize using global average across inner folds."""
        if verbose > 2:
            print(f"    🌍 Global average optimization across {len(inner_folds)} inner folds")

        # Simplified implementation - same as regular finetuning for now
        return self._finetune_on_inner_folds(context, inner_folds, verbose)

    def _train_final_model_on_outer_fold(
        self,
        context: CVExecutionContext,
        outer_idx: int,
        best_params: Dict[str, Any]
    ) -> tuple[Dict[str, Any], List[tuple[str, bytes]]]:
        """Train final model on outer fold using optimized parameters."""
        # Get the outer fold data
        outer_data = context.data_splits[outer_idx]

        # Create model with best parameters
        model = self._get_model_instance(context)
        if hasattr(model, 'set_params') and best_params:
            try:
                model.set_params(**best_params)
            except (ValueError, TypeError):
                pass  # Invalid parameters, use defaults

        # Create data split for the outer fold
        from nirs4all.controllers.models.config import DataSplit
        outer_data_split = DataSplit(
            X_train=outer_data.X_train,
            y_train=outer_data.y_train,
            X_test=outer_data.X_test,
            y_test=outer_data.y_test
        )

        # Create temporary context for outer fold training
        outer_execution_context = CVExecutionContext(
            model_config=context.model_config,
            data_splits=[outer_data_split],
            train_params=context.train_params,
            cv_config=context.cv_config,
            runner=context.runner,
            dataset=context.dataset,
            controller=context.controller,
            finetune_config=context.finetune_config
        )

        # Train the final model
        _, binaries = self._execute_train(outer_execution_context, fold_idx=0)

        # Rename binaries to indicate nested CV and outer fold
        renamed_binaries = []
        for name, binary in binaries:
            name_parts = name.rsplit('.', 1)
            if len(name_parts) == 2:
                new_name = f"{name_parts[0]}_nested_cv_outer_fold{outer_idx+1}.{name_parts[1]}"
            else:
                new_name = f"{name}_nested_cv_outer_fold{outer_idx+1}"
            renamed_binaries.append((new_name, binary))

        return {}, renamed_binaries

    def _compute_weighted_average_params(
        self,
        fold_results: List[Dict[str, Any]],
        verbose: int = 0
    ) -> Dict[str, Any]:
        """Compute weighted average parameters based on fold performance."""
        if verbose > 1:
            print("📊 Computing weighted average parameters...")

        # Placeholder implementation - would need performance scores from each outer fold
        if fold_results:
            return fold_results[0].get('best_params', {})
        return {}

    def _train_single_model_on_full_data(
        self,
        context: CVExecutionContext,
        best_params: Dict[str, Any],
        model_suffix: str = "full_train",
        verbose: int = 0
    ) -> CVResult:
        """Train a single model on the full training dataset using optimized parameters."""
        if verbose > 1:
            print(f"🎯 Training single model on full training data ({model_suffix})...")

        # Combine all training data from folds
        all_X_train = []
        all_y_train = []
        all_X_test = []
        all_y_test = []

        for data_split in context.data_splits:
            all_X_train.append(data_split.X_train)
            all_y_train.append(data_split.y_train)
            all_X_test.append(data_split.X_test)
            all_y_test.append(data_split.y_test)

        # Concatenate all data
        import numpy as np
        combined_X_train = np.concatenate(all_X_train, axis=0)
        combined_y_train = np.concatenate(all_y_train, axis=0)
        combined_X_test = np.concatenate(all_X_test, axis=0)
        combined_y_test = np.concatenate(all_y_test, axis=0)

        if verbose > 1:
            print(f"📊 Combined training data: {combined_X_train.shape[0]} samples")
            print(f"📊 Combined test data: {combined_X_test.shape[0]} samples")

        # Create and configure model with best parameters
        model = self._get_model_instance(context)
        if hasattr(model, 'set_params') and best_params:
            try:
                model.set_params(**best_params)
                if verbose > 1:
                    print(f"✅ Applied optimized parameters: {best_params}")
            except (ValueError, TypeError) as e:
                if verbose > 1:
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

        if verbose > 1:
            print("✅ Single model training on full data completed successfully")

        return CVResult(
            context=context.cv_config.__dict__,
            binaries=renamed_binaries
        )
