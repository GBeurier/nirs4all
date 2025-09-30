"""
Data Management - Handles data preparation and splitting for model training

This module provides a clean interface for preparing training, validation,
and test data splits, including cross-validation fold creation.
"""

from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
import numpy as np

from nirs4all.controllers.models.config import DataSplit

if TYPE_CHECKING:
    from nirs4all.dataset.dataset import SpectroDataset


class DataManager:
    """
    Handles all data preparation and splitting operations.

    This class centralizes data management logic that was previously scattered
    across the monolithic BaseModelController.
    """

    def __init__(self):
        """Initialize the data manager."""

    def prepare_train_test_data(
        self,
        dataset: 'SpectroDataset',
        context: Dict[str, Any]
    ) -> Union[DataSplit, List[DataSplit]]:
        """
        Prepare training and test data from dataset, handling cross-validation folds.

        If dataset has folds, returns a list of (X_train, y_train, X_val, y_val) tuples, one per fold.
        If no folds, returns a single DataSplit.

        Args:
            dataset: Dataset containing features and targets
            context: Pipeline context with processing state

        Returns:
            Union[DataSplit, List[DataSplit]]: Single split or list of fold splits
        """
        # Get the preferred layout for this model type
        layout_str = self._get_layout_from_context(context)
        layout = layout_str  # type: ignore

        # Check if dataset has folds
        if hasattr(dataset, 'num_folds') and dataset.num_folds > 0:
            # Prepare fold-based train/validation splits
            folds_data = self._create_fold_splits(dataset, context, layout)
            return folds_data
        else:
            # No folds: use standard train/test split
            return self._create_single_split(dataset, context, layout)

    def _get_layout_from_context(self, context: Dict[str, Any]) -> Any:
        """Extract layout preference from context."""
        # This would be set by the controller based on the model type
        # For now, return a default layout string
        return context.get('layout', '2d')

    def _create_fold_splits(
        self,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        layout: Any
    ) -> List[DataSplit]:
        """
        Create cross-validation fold splits.

        Args:
            dataset: Dataset with folds
            context: Pipeline context
            layout: Data layout specification

        Returns:
            List[DataSplit]: List of fold data splits
        """
        folds_data = []

        # Get all training data first
        train_context = context.copy()
        train_context["partition"] = "train"
        X_all_train = dataset.x(train_context, layout, concat_source=True)
        y_all_train = dataset.y(train_context)

        # Get test data
        test_context = context.copy()
        test_context["partition"] = "test"
        X_test = dataset.x(test_context, layout, concat_source=True)
        y_test = dataset.y(test_context)

        # For each fold, create train/validation splits
        for fold_idx, (train_indices, val_indices) in enumerate(dataset.folds):
            # Convert indices to numpy arrays for proper indexing
            train_indices = np.array(train_indices)
            val_indices = np.array(val_indices)

            X_train_fold = X_all_train[train_indices]
            y_train_fold = y_all_train[train_indices]
            X_val_fold = X_all_train[val_indices]
            y_val_fold = y_all_train[val_indices]

            fold_split = DataSplit(
                X_train=X_train_fold,
                y_train=y_train_fold,
                X_val=X_val_fold,
                y_val=y_val_fold,
                X_test=X_test,
                y_test=y_test,
                fold_idx=fold_idx
            )
            folds_data.append(fold_split)

        return folds_data

    def _create_single_split(
        self,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        layout: Any
    ) -> DataSplit:
        """
        Create a single train/test split.

        Args:
            dataset: Dataset without folds
            context: Pipeline context
            layout: Data layout specification

        Returns:
            DataSplit: Single data split
        """
        train_context = context.copy()
        train_context["partition"] = "train"
        X_train = dataset.x(train_context, layout, concat_source=True)
        y_train = dataset.y(train_context)

        test_context = context.copy()
        test_context["partition"] = "test"
        X_test = dataset.x(test_context, layout, concat_source=True)
        y_test = dataset.y(test_context)

        return DataSplit(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

    def create_inner_folds(
        self,
        X_train: Any,
        y_train: Any,
        inner_cv: Optional[Union[int, Any]] = None
    ) -> List[DataSplit]:
        """
        Create inner cross-validation folds for nested CV.

        Args:
            X_train: Training features
            y_train: Training targets
            inner_cv: Inner CV specification (int or cv object)

        Returns:
            List[DataSplit]: Inner fold splits
        """
        try:
            from sklearn.model_selection import KFold
        except ImportError as exc:
            raise ImportError("scikit-learn is required for nested cross-validation") from exc

        # Default to 3-fold if not specified
        if inner_cv is None:
            inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        elif isinstance(inner_cv, int):
            inner_cv = KFold(n_splits=inner_cv, shuffle=True, random_state=42)

        inner_folds = []
        for train_idx, val_idx in inner_cv.split(X_train, y_train):
            # Use numpy indexing if possible
            if hasattr(X_train, '__getitem__') and hasattr(X_train, 'shape'):
                X_inner_train = X_train[train_idx]
                X_inner_val = X_train[val_idx]
            else:
                X_inner_train = X_train
                X_inner_val = X_train

            if hasattr(y_train, '__getitem__') and hasattr(y_train, 'shape'):
                y_inner_train = y_train[train_idx]
                y_inner_val = y_train[val_idx]
            else:
                y_inner_train = y_train
                y_inner_val = y_train

            inner_fold = DataSplit(
                X_train=X_inner_train,
                y_train=y_inner_train,
                X_val=X_inner_val,
                y_val=y_inner_val
            )
            inner_folds.append(inner_fold)

        return inner_folds

    def get_test_data(
        self,
        dataset: 'SpectroDataset',
        context: Dict[str, Any]
    ) -> DataSplit:
        """
        Get test data for evaluation.

        Args:
            dataset: Dataset object
            context: Pipeline context

        Returns:
            DataSplit: Test data split
        """
        layout_str = self._get_layout_from_context(context)
        layout = layout_str  # type: ignore

        test_context = context.copy()
        test_context["partition"] = "test"
        X_test = dataset.x(test_context, layout, concat_source=True)
        y_test = dataset.y(test_context)

        return DataSplit(
            X_train=np.array([]),  # Empty train data for test-only split
            y_train=np.array([]),
            X_test=X_test,
            y_test=y_test
        )
