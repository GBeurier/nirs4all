"""
Folds management for cross-validation splits.

This module handles cross-validation fold definitions and provides
data generators that yield train/validation splits using the dataset blocks.
"""

from typing import List, Dict, Tuple, Any, Generator, Iterable

import numpy as np


class FoldsManager:
    """
    Manager for cross-validation folds.

    Stores fold definitions and provides generators for train/validation splits.
    """

    def __init__(self):
        """Initialize empty FoldsManager."""
        self.folds: List[Dict[str, List[int]]] = []

    def set_folds(self, folds_iterable: Iterable[Tuple[List[int], List[int]]]) -> None:
        """
        Set folds from an iterable of (train_idx, val_idx) tuples.

        Args:
            folds_iterable: Iterable yielding (train_indices, validation_indices) tuples
        """
        self.folds = []
        for train_idx, val_idx in folds_iterable:
            self.folds.append({
                "train": list(train_idx),
                "val": list(val_idx)
            })

    def get_data(self, dataset: Any, layout: str = "2d") -> Generator[Tuple[Tuple[np.ndarray, ...], np.ndarray, Tuple[np.ndarray, ...], np.ndarray], None, None]:
        """
        Generator yielding x_train, y_train, x_val, y_val for each fold.

        Args:
            dataset: SpectroDataset instance
            layout: Layout for features ("2d", "2d_interlaced", "3d", "3d_transpose")

        Yields:
            Tuple of (x_train, y_train, x_val, y_val) for each fold
        """
        for fold in self.folds:
            train_indices = fold["train"]
            val_indices = fold["val"]

            # Get training data
            train_filter = {"sample": train_indices}
            x_train = dataset.x(train_filter, layout=layout)
            y_train = dataset.y(train_filter)

            # Get validation data
            val_filter = {"sample": val_indices}
            x_val = dataset.x(val_filter, layout=layout)
            y_val = dataset.y(val_filter)

            yield x_train, y_train, x_val, y_val

    def __len__(self) -> int:
        """Return number of folds."""
        return len(self.folds)

    def __repr__(self) -> str:
        if not self.folds:
            return "FoldsManager(empty)"
        return f"FoldsManager({len(self.folds)} folds)"
