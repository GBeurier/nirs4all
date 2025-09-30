"""
Prediction Management - Handles prediction generation and storage

This module provides a clean interface for generating predictions and storing
them in the dataset, replacing the scattered prediction logic in the original controller.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset


class PredictionManager:
    """
    Handles prediction generation and storage operations.

    This class centralizes prediction-related operations that were previously
    scattered across the monolithic BaseModelController.
    """

    def __init__(self):
        """Initialize the prediction manager."""

    def generate_predictions(
        self,
        model: Any,
        X_data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Generate predictions using a trained model.

        Args:
            model: Trained model instance
            X_data: Input data for prediction
            context: Optional context information

        Returns:
            np.ndarray: Model predictions
        """
        # This should be implemented by framework-specific subclasses
        # For now, assume the model has a predict method
        if hasattr(model, 'predict'):
            predictions = model.predict(X_data)
            # Ensure predictions are 2D for consistency
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            return predictions
        else:
            raise NotImplementedError("Model prediction method not implemented")

    def store_predictions(
        self,
        dataset: str,
        pipeline: str,
        pipeline_path: str,
        model: str,
        real_model: str,
        partition: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fold_idx: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        dataset_obj: Optional['SpectroDataset'] = None,
        custom_model_name: Optional[str] = None
    ) -> None:
        """
        Store predictions in the dataset's prediction storage.

        Args:
            dataset: Dataset name
            pipeline: Pipeline name
            pipeline_path: Path for loading pipeline
            model: Base model class name
            real_model: Full model identifier with step/fold info
            partition: Partition name ('train', 'val', 'test')
            y_true: True values
            y_pred: Predicted values
            fold_idx: Fold index (0,1,2) or "avg", "weighted", None
            context: Pipeline context with processing information
            dataset_obj: Optional dataset object to store predictions in
            custom_model_name: Optional custom model name
        """
        if dataset_obj is None:
            return

        # Get current y processing from context
        current_y_processing = context.get('y', 'numeric') if context else 'numeric'

        # Inverse transform predictions to original space if possible
        y_true_transformed, y_pred_transformed = self._transform_predictions(
            y_true, y_pred, dataset_obj, current_y_processing
        )

        # Store predictions with new schema
        dataset_obj._predictions.add_prediction(  # type: ignore
            dataset=dataset,
            pipeline=pipeline,
            pipeline_path=pipeline_path,
            model=model,
            real_model=real_model,
            partition=partition,
            y_true=y_true_transformed,
            y_pred=y_pred_transformed,
            fold_idx=fold_idx,
            metadata={
                'y_processing': current_y_processing,
                'model_type': model,
                'real_model': real_model,
                'partition': partition,
                'n_samples': dataset_obj.num_samples,
                'n_features': dataset_obj.num_features
            },
            custom_model_name=custom_model_name
        )

    def _transform_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_obj: 'SpectroDataset',
        current_y_processing: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform predictions back to original space if needed.

        Args:
            y_true: True values
            y_pred: Predicted values
            dataset_obj: Dataset object
            current_y_processing: Current processing type

        Returns:
            tuple: Transformed (y_true, y_pred)
        """
        try:
            if hasattr(dataset_obj, '_targets') and current_y_processing != 'numeric':
                # Transform both y_true and y_pred back to numeric space
                y_true_transformed = dataset_obj._targets.transform_predictions(  # type: ignore
                    y_true, from_processing=current_y_processing, to_processing='numeric'
                )
                y_pred_transformed = dataset_obj._targets.transform_predictions(  # type: ignore
                    y_pred, from_processing=current_y_processing, to_processing='numeric'
                )
                return y_true_transformed, y_pred_transformed
            else:
                return y_true, y_pred
        except (AttributeError, ValueError):
            # If transformation fails, return original values
            return y_true, y_pred
