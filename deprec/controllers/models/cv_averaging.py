"""
CV Averaging Manager - Handles generation of average and weighted average predictions

This module provides functionality to generate average and weighted average
predictions after cross-validation training is complete.
"""

from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from nirs4all.dataset.dataset import SpectroDataset
    from nirs4all.pipeline.runner import PipelineRunner

from nirs4all.controllers.models.model_naming import ModelNamingManager, ModelIdentifiers


class CVAveragingManager:
    """
    Manages generation of average and weighted average predictions after CV training.

    This separates the averaging logic from the CV strategies and ensures
    consistent generation of average predictions.
    """

    def __init__(self, naming_manager: ModelNamingManager):
        """Initialize with naming manager for consistent IDs."""
        self.naming_manager = naming_manager

    def generate_average_predictions(
        self,
        dataset: 'SpectroDataset',
        model_config: Dict[str, Any],
        runner: 'PipelineRunner',
        context: Dict[str, Any],
        fold_count: int,
        verbose: int = 0
    ) -> Tuple[List[Tuple[str, bytes]], List[Dict[str, Any]]]:
        """
        Generate average and weighted average predictions from fold results.

        Args:
            dataset: Dataset object
            model_config: Model configuration
            runner: Pipeline runner
            context: Pipeline context
            fold_count: Number of folds trained
            verbose: Verbosity level

        Returns:
            Tuple of (binaries, prediction_metadata)
        """
        if verbose > 1:
            print("📊 Generating average and weighted average predictions...")

        # Get base model identifiers for averaging
        base_identifiers = self.naming_manager.create_model_identifiers(
            model_config, runner, fold_idx=None
        )

        # Generate average predictions
        avg_identifiers = self.naming_manager.create_model_identifiers(
            model_config, runner, fold_idx=None, is_avg=True
        )

        # Generate weighted average predictions
        w_avg_identifiers = self.naming_manager.create_model_identifiers(
            model_config, runner, fold_idx=None, is_weighted_avg=True
        )

        binaries = []
        prediction_metadata = []

        # Calculate average predictions using existing prediction helpers
        dataset_name = getattr(runner.saver, 'dataset_name', 'unknown')
        pipeline_name = getattr(runner.saver, 'pipeline_name', 'unknown')

        try:
            # Import the prediction helpers
            from nirs4all.dataset.prediction_helpers import PredictionHelpers

            # Get access to the predictions dictionary
            predictions_dict = getattr(dataset._predictions, '_predictions', {})

            # Generate average predictions using static methods
            avg_result, _ = PredictionHelpers.calculate_average_predictions(
                predictions=predictions_dict,
                dataset=dataset_name,
                pipeline=pipeline_name,
                model=base_identifiers.classname,
                partition="val",  # Use validation partition for fold predictions
                run_path=""
            )

            if avg_result and verbose > 1:
                print(f"✅ Generated average predictions for {avg_identifiers.model_id}")

            # Generate weighted average predictions
            w_avg_result, _ = PredictionHelpers.calculate_weighted_average_predictions(
                predictions=predictions_dict,
                dataset=dataset_name,
                pipeline=pipeline_name,
                model=base_identifiers.classname,
                test_partition="val",  # Use validation partition for CV
                val_partition="train",  # Use training partition for weighting
                metric='rmse',
                run_path=""
            )

            if w_avg_result and verbose > 1:
                print(f"✅ Generated weighted average predictions for {w_avg_identifiers.model_id}")

        except (AttributeError, KeyError, ValueError) as e:
            if verbose > 1:
                print(f"⚠️ Could not generate average predictions: {e}")

        return binaries, prediction_metadata

    def _create_avg_binary(
        self,
        identifiers: ModelIdentifiers,
        predictions: Dict[str, Any]
    ) -> Tuple[str, bytes]:
        """Create a binary representation of average predictions."""
        import pickle
        filename = f"{identifiers.model_id}.pkl"
        binary_data = pickle.dumps(predictions)
        return filename, binary_data
