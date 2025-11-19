"""
Prediction Transformer - Handle scaling/unscaling of predictions

This component handles the transformation of predictions between different
target spaces (scaled/unscaled, numeric/transformed).

Extracted from launch_training() lines 427-447 and _create_fold_averages()
to eliminate duplicate logic.
"""

from typing import Any, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext


class PredictionTransformer:
    """Transforms predictions between scaled and unscaled spaces.

    Handles:
        - Classification tasks: Keep predictions in transformed space
        - Regression tasks: Transform predictions back to numeric space
        - Respects current y_processing from context

    Example:
        >>> transformer = PredictionTransformer()
        >>> y_pred_unscaled = transformer.transform_to_unscaled(
        ...     y_pred_scaled,
        ...     dataset,
        ...     context
        ... )
    """

    def transform_to_unscaled(
        self,
        predictions_scaled: np.ndarray,
        dataset: 'SpectroDataset',
        context: Optional['ExecutionContext'] = None
    ) -> np.ndarray:
        """Transform predictions from scaled/processed space to unscaled/numeric space.

        Args:
            predictions_scaled: Predictions in scaled/processed space
            dataset: Dataset with task type and target transformation info
            context: Execution context with y processing info

        Returns:
            Predictions in unscaled/numeric space
        """
        if predictions_scaled.shape[0] == 0:
            return predictions_scaled

        # Get current y processing mode
        current_y_processing = context.state.y_processing if context else 'numeric'

        # For classification tasks, keep predictions in transformed space
        if dataset.task_type and 'classification' in dataset.task_type:
            return predictions_scaled

        # For regression, transform back to numeric if currently processed
        if current_y_processing != 'numeric':
            return dataset._targets.transform_predictions(  # noqa: SLF001
                predictions_scaled,
                current_y_processing,
                'numeric'
            )

        # Already in numeric space
        return predictions_scaled

    def transform_batch_to_unscaled(
        self,
        predictions_dict: dict,
        dataset: 'SpectroDataset',
        context: Optional['ExecutionContext'] = None
    ) -> dict:
        """Transform a dictionary of predictions to unscaled space.

        Args:
            predictions_dict: Dictionary with keys like 'train', 'val', 'test'
                            and values as prediction arrays
            dataset: Dataset with transformation info
            context: Execution context

        Returns:
            Dictionary with same keys but unscaled predictions
        """
        result = {}
        for key, predictions in predictions_dict.items():
            result[key] = self.transform_to_unscaled(predictions, dataset, context)
        return result
