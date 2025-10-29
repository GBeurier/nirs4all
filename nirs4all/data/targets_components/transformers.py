"""Prediction transformation utilities."""

from typing import Dict

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder

from .processing_chain import ProcessingChain


class TargetTransformer:
    """
    Transforms predictions between different processing states.

    Uses processing chain information to apply forward and inverse
    transformations along the ancestry path.

    Attributes:
        processing_chain (ProcessingChain): Reference to processing chain for ancestry and transformer lookup

    Methods:
        transform(predictions, from_processing, to_processing, data_storage): Transform predictions between processing states

    Examples:
    >>> transformer = TargetTransformer(processing_chain)
    >>> predictions = model.predict(X_test)  # In 'scaled' space
    >>> numeric_preds = transformer.transform(
    ...     predictions,
    ...     from_processing='scaled',
    ...     to_processing='numeric',
    ...     data_storage=targets._data
    ... )
    """

    def __init__(self, processing_chain: ProcessingChain):
        """
        Initialize transformer with processing chain reference.

        Args:
            processing_chain (ProcessingChain): Processing chain to use for ancestry and transformers
        """
        self.processing_chain = processing_chain

    def transform(self,
                 predictions: np.ndarray,
                 from_processing: str,
                 to_processing: str,
                 data_storage: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Transform predictions from one processing state to another.

        Applies inverse transformations when going up the ancestry chain
        and forward transformations when going down.

        Args:
            predictions (np.ndarray): Prediction array to transform
            from_processing (str): Current processing state of predictions
            to_processing (str): Target processing state
            data_storage (dict): Storage dictionary mapping processing names to data arrays

        Returns:
            np.ndarray: Transformed predictions in target processing state

        Raises:
            ValueError: If processing names don't exist
            ValueError: If no path exists between processings
            ValueError: If transformation fails

        Notes:
        - Empty predictions return empty array
        - Handles LabelEncoder specially (converts to int32)
        - Uses cached ancestry for efficiency

        Examples:
        >>> # Transform from model's output space back to numeric
        >>> preds_numeric = transformer.transform(
        ...     model_predictions,
        ...     'minmax-scaled',
        ...     'numeric',
        ...     targets._data
        ... )
        """
        if from_processing == to_processing:
            return predictions.copy()

        if predictions.shape[0] == 0:
            return predictions.copy()

        # Get transformation path
        path, direction = self.processing_chain.get_path(
            from_processing, to_processing
        )

        current = predictions.copy()

        if direction == 'inverse':
            # Walk backward from from_processing to to_processing
            for i in range(len(path) - 1):
                current_proc = path[i]
                transformer = self.processing_chain.get_transformer(current_proc)

                if transformer is not None and hasattr(transformer, 'inverse_transform'):
                    current = self._apply_inverse_transform(current, transformer)
                else:
                    raise ValueError(
                        f"No inverse transformer for processing '{current_proc}'"
                    )

        elif direction == 'forward':
            # Walk forward from from_processing to to_processing
            for i in range(len(path) - 1):
                next_proc = path[i + 1]
                transformer = self.processing_chain.get_transformer(next_proc)

                if transformer is not None and hasattr(transformer, 'transform'):
                    current = self._apply_transform(current, transformer)
                else:
                    raise ValueError(
                        f"No forward transformer for processing '{next_proc}'"
                    )

        elif direction == 'mixed':
            # Handle mixed direction (through common ancestor)
            # Find common ancestor in path
            from_ancestry = self.processing_chain.get_ancestry(from_processing)
            to_ancestry = self.processing_chain.get_ancestry(to_processing)

            common_ancestor = None
            for anc in reversed(from_ancestry):
                if anc in to_ancestry:
                    common_ancestor = anc
                    break

            # Inverse transform to common ancestor
            common_idx = path.index(common_ancestor)
            for i in range(common_idx):
                current_proc = path[i]
                transformer = self.processing_chain.get_transformer(current_proc)

                if transformer and hasattr(transformer, 'inverse_transform'):
                    current = self._apply_inverse_transform(current, transformer)

            # Forward transform to target
            for i in range(common_idx, len(path) - 1):
                next_proc = path[i + 1]
                transformer = self.processing_chain.get_transformer(next_proc)

                if transformer and hasattr(transformer, 'transform'):
                    current = self._apply_transform(current, transformer)

        return current

    @staticmethod
    def _apply_transform(data: np.ndarray,
                        transformer: TransformerMixin) -> np.ndarray:
        """
        Apply forward transformation with error handling.

        Args:
            data (np.ndarray): Data to transform
            transformer (TransformerMixin): Transformer to apply

        Returns:
            np.ndarray: Transformed data

        Raises:
            ValueError: If transformation fails
        """
        try:
            return transformer.transform(data)  # type: ignore
        except Exception as e:
            raise ValueError(
                f"Forward transform failed with {type(transformer).__name__}: {e}"
            ) from e

    @staticmethod
    def _apply_inverse_transform(data: np.ndarray,
                                 transformer: TransformerMixin) -> np.ndarray:
        """
        Apply inverse transformation with error handling.

        Handles special cases like LabelEncoder which requires int32 input.

        Args:
            data (np.ndarray): Data to inverse transform
            transformer (TransformerMixin): Transformer to apply

        Returns:
            np.ndarray: Inverse transformed data

        Raises:
            ValueError: If transformation fails
        """
        try:
            # LabelEncoder requires integer input for inverse_transform
            if isinstance(transformer, LabelEncoder):
                return transformer.inverse_transform(data.astype(np.int32))  # type: ignore
            else:
                return transformer.inverse_transform(data)  # type: ignore
        except Exception as e:
            raise ValueError(
                f"Inverse transform failed with {type(transformer).__name__}: {e}"
            ) from e
