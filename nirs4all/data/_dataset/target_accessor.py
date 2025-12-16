"""
Target accessor for managing target data operations.

This module provides a dedicated interface for all target-related
operations, including target retrieval, processing, and task type management.
"""

import numpy as np
from typing import Optional, List
from sklearn.base import TransformerMixin

from nirs4all.data.types import Selector
from nirs4all.data.indexer import Indexer
from nirs4all.data.targets import Targets
from nirs4all.core.task_type import TaskType


def _selector_to_dict(selector: Optional[Selector]) -> dict:
    """
    Convert selector to dict format for internal use.

    Handles both legacy dict format and new DataSelector format.

    Args:
        selector: Selector in any format

    Returns:
        Dict representation of selector
    """
    if selector is None:
        return {}

    # Handle ExecutionContext (duck typing to avoid circular import)
    if hasattr(selector, "selector") and hasattr(selector, "state"):
        d = dict(selector.selector)
        # Only use state.y_processing if y is not already defined in selector
        if "y" not in d and hasattr(selector.state, "y_processing"):
            d["y"] = selector.state.y_processing
        return d

    return dict(selector)


class TargetAccessor:
    """
    Accessor for target data operations.

    Manages target values with processing chains, task type detection,
    and prediction transformations.

    Attributes:
        num_samples (int): Number of samples with targets
        num_classes (int): Number of unique classes (classification only)
        task_type (TaskType): Detected or manually set task type
        processing_ids (List[str]): Available target processing versions

    Examples:
        >>> # Used internally by SpectroDataset, but accessible as:
        >>> # dataset.targets.task_type
        >>> # dataset.targets.num_classes
    """

    def __init__(self, indexer: Indexer, targets_block: Targets):
        """
        Initialize target accessor.

        Args:
            indexer: Sample index manager for filtering
            targets_block: Underlying target storage
        """
        self._indexer = indexer
        self._block = targets_block

    def y(self,
          selector: Optional[Selector] = None,
          include_augmented: bool = True,
          include_excluded: bool = False) -> np.ndarray:
        """
        Get target values with filtering.

        Automatically maps augmented samples to their origin's y values,
        preventing data leakage in cross-validation.

        Args:
            selector: Filter criteria (same as features.x):
                - partition: "train", "test", "val"
                - group: group identifier
                - branch: branch identifier
                - fold: fold number
                - y: processing version (default "numeric")
            include_augmented: If True, include augmented versions of selected samples.
                             Augmented samples are automatically mapped to their origin's y value.
            include_excluded: If True, include samples marked as excluded.
                            If False (default), exclude samples marked as excluded=True.
                            Use True when transforming ALL targets (e.g., y_processing).

        Returns:
            Target array of shape (n_samples, n_targets)

        Examples:
            >>> # Get all train targets (base + augmented)
            >>> y_train = dataset.y({"partition": "train"})
            >>> # Get only base train targets (for splitting)
            >>> y_base = dataset.y({"partition": "train"}, include_augmented=False)
            >>> # Get specific processing
            >>> y_scaled = dataset.y(
            ...     {"partition": "train", "y": "scaled"},
            ...     include_augmented=True
            ... )
            >>> # Get all targets including excluded (for y_processing)
            >>> y_all = dataset.y({"partition": "train"}, include_excluded=True)
        """
        selector_dict = _selector_to_dict(selector)

        if include_augmented:
            x_indices = self._indexer.x_indices(
                selector_dict, include_augmented=True, include_excluded=include_excluded
            )
            # Map each sample to its y index (augmented → origin)
            y_indices = np.array([
                self._indexer.get_origin_for_sample(int(sample_id))
                for sample_id in x_indices
            ], dtype=np.int32)
        else:
            y_indices = self._indexer.x_indices(
                selector_dict, include_augmented=False, include_excluded=include_excluded
            )

        processing = selector_dict.get("y", "numeric") if selector_dict else "numeric"
        return self._block.y(y_indices, processing)

    def add_targets(self, y: np.ndarray) -> None:
        """
        Add target samples to the dataset.

        Automatically detects task type (regression, binary, multiclass)
        and creates "raw" and "numeric" processing versions.

        Args:
            y: Target values as 1D or 2D array

        Examples:
            >>> # Classification
            >>> y_train = np.array([0, 1, 2, 0, 1, 2])
            >>> dataset.add_targets(y_train)
            >>> print(dataset.task_type)
            TaskType.MULTICLASS_CLASSIFICATION
            >>> # Regression
            >>> y_train = np.array([1.5, 2.3, 4.1, 3.2])
            >>> dataset.add_targets(y_train)
            >>> print(dataset.task_type)
            TaskType.REGRESSION
        """
        self._block.add_targets(y)

    def add_processed_targets(self,
                              processing_name: str,
                              targets: np.ndarray,
                              ancestor_processing: str = "numeric",
                              transformer: Optional[TransformerMixin] = None) -> None:
        """
        Add processed target version (e.g., scaled, encoded).

        Args:
            processing_name: Unique name for this processing
            targets: Processed target values
            ancestor_processing: Parent processing name (for chain tracking)
            transformer: Scikit-learn transformer used (for inverse transform)

        Examples:
            >>> # Add scaled targets for neural networks
            >>> from sklearn.preprocessing import StandardScaler
            >>> scaler = StandardScaler()
            >>> y_train = dataset.y({"partition": "train"})
            >>> y_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            >>> dataset.add_processed_targets(
            ...     "scaled",
            ...     y_scaled,
            ...     "numeric",
            ...     scaler
            ... )
        """
        self._block.add_processed_targets(
            processing_name, targets, ancestor_processing, transformer
        )

    def transform_predictions(self,
                              predictions: np.ndarray,
                              from_processing: str,
                              to_processing: str) -> np.ndarray:
        """
        Transform predictions between processing states.

        Useful for converting model predictions back to original scale.

        Args:
            predictions: Model predictions
            from_processing: Source processing state
            to_processing: Target processing state

        Returns:
            Transformed predictions

        Examples:
            >>> # Model trained on scaled targets
            >>> y_pred_scaled = model.predict(X_test)
            >>> # Transform back to numeric
            >>> y_pred_numeric = dataset.transform_predictions(
            ...     y_pred_scaled,
            ...     from_processing="scaled",
            ...     to_processing="numeric"
            ... )
        """
        return self._block.transform_predictions(
            predictions, from_processing, to_processing
        )

    @property
    def task_type(self) -> Optional[TaskType]:
        """
        Get detected task type.

        Returns:
            TaskType enum or None if no targets added
        """
        return self._block.task_type

    @property
    def num_classes(self) -> int:
        """Number of unique classes (for classification tasks)."""
        return self._block.num_classes

    @property
    def num_samples(self) -> int:
        """Number of samples with target values."""
        return self._block.num_samples

    @property
    def processing_ids(self) -> List[str]:
        """List of available target processing versions."""
        return self._block.processing_ids
