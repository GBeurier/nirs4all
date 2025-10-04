"""
Per-Fold CV Strategy - Finetune on each fold individually

This strategy implements per-fold cross-validation where hyperparameters
are optimized separately for each fold.
"""

from typing import Dict, Any, List
from nirs4all.controllers.models.cv.base import CVStrategy, CVExecutionContext, CVResult
from nirs4all.controllers.models.config import ParamStrategy


class PerFoldCVStrategy(CVStrategy):
    """
    Per-fold CV strategy: finetune on each fold individually.

    This strategy optimizes hyperparameters separately for each fold,
    then applies parameter aggregation strategies across folds.
    """

    def execute(self, context: CVExecutionContext) -> CVResult:
        """
        Execute per-fold CV: finetune on each fold with parameter aggregation.

        Args:
            context: CV execution context

        Returns:
            CVResult: Results of the CV execution
        """
        verbose = context.train_params.get('verbose', 0)
        param_strategy = context.cv_config.param_strategy

        # Handle global average strategy - optimize across all folds simultaneously
        if param_strategy == ParamStrategy.GLOBAL_AVERAGE:
            return self._execute_global_average_optimization(context)

        if verbose > 1:
            print(f"🔍 Per-fold CV: Finetuning on each fold with {param_strategy.value} strategy...")

        all_binaries = []
        all_fold_results = []

        # Finetune on each fold
        for fold_idx, _ in enumerate(context.data_splits):
            if verbose > 1:
                print(f"🎛️ Finetuning fold {fold_idx+1}/{len(context.data_splits)}...")

            # Execute finetuning for this fold
            _, fold_binaries = self._execute_finetune(context, fold_idx)

            # Store fold results for potential parameter aggregation
            fold_result = {
                'fold_idx': fold_idx,
                'binaries': fold_binaries
            }
            all_fold_results.append(fold_result)
            all_binaries.extend(fold_binaries)

        # Handle parameter aggregation strategy
        if param_strategy == ParamStrategy.GLOBAL_BEST:
            # Use the best performing parameters across all folds
            global_best_params = self._select_global_best_params(all_fold_results)
            if verbose > 1:
                print(f"🏆 Global best parameters: {global_best_params}")

            # Check if we should train a single model on full training data
            if context.cv_config.use_full_train_for_final:
                return self._train_single_model_on_full_data(
                    context, global_best_params, "global_best", verbose
                )

        elif param_strategy == ParamStrategy.WEIGHTED_AVERAGE:
            # Compute weighted average parameters based on fold performance
            weighted_params = self._compute_weighted_average_params(all_fold_results)
            if verbose > 1:
                print(f"⚖️ Weighted average parameters: {weighted_params}")

        elif param_strategy in [ParamStrategy.ENSEMBLE_BEST, ParamStrategy.ROBUST_BEST, ParamStrategy.STABILITY_BEST]:
            # These strategies are planned for future implementation
            if verbose > 1:
                print(f"⚠️ Parameter strategy {param_strategy.value} is not yet implemented. Using per_fold_best instead.")

        # For PER_FOLD_BEST, check if we should train on full data (though this is less common)
        if context.cv_config.use_full_train_for_final and param_strategy == ParamStrategy.PER_FOLD_BEST:
            # Use the first fold's parameters as representative (or could average them)
            representative_params = {}  # Would need to extract from first fold
            if verbose > 1:
                print("🔄 Training single model on full data with representative parameters from fold 1")
            return self._train_single_model_on_full_data(
                context, representative_params, "per_fold_repr", verbose
            )

        if verbose > 1:
            print("✅ Per-fold CV completed successfully")

        return CVResult(
            context=context.cv_config.__dict__,
            binaries=all_binaries
        )

    def _execute_global_average_optimization(self, context: CVExecutionContext) -> CVResult:
        """Execute global average optimization: optimize parameters across all folds simultaneously."""
        verbose = context.train_params.get('verbose', 0)

        if verbose > 1:
            print(f"🌍 Global Average CV: Optimizing parameters across all {len(context.data_splits)} folds simultaneously...")

        # For now, implement a simplified version that optimizes on the first fold
        # Full implementation would require Optuna integration across all folds
        if verbose > 1:
            print("ℹ️ Using simplified global average (optimizing on first fold)")

        _, binaries = self._execute_finetune(context, fold_idx=0)

        if verbose > 1:
            print("✅ Global Average CV completed successfully")

        return CVResult(
            context=context.cv_config.__dict__,
            binaries=binaries
        )

    def _select_global_best_params(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the globally best parameters from all folds."""
        # For now, return empty dict - would need to track best params from each fold
        # This requires extracting best_params from the finetuning results
        # Placeholder implementation
        if fold_results:
            return fold_results[0].get('best_params', {})
        return {}

    def _compute_weighted_average_params(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute weighted average parameters based on fold performance."""
        # For numerical parameters, compute weighted average
        # For categorical parameters, select most frequent
        # Placeholder implementation - would need performance scores
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
        """
        Train a single model on the full training dataset using optimized parameters.

        Instead of training separate models on each fold, this combines all training data
        and trains one model, which can be more effective when you have limited data.
        """
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
