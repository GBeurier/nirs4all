"""
Predictions management for SpectroDataset.

This module contains PredictionBlock for storing and managing model predictions
with metadata about models, folds, processing steps, etc.
"""

from typing import Dict, Any, List, Callable, Optional

import numpy as np
import polars as pl


class PredictionBlock:
    """
    Block for storing and managing model predictions.

    Uses Polars DataFrame with columns: sample, model, fold, repeat,
    partition, processing, seed, preds (list of predictions).
    """

    def __init__(self):
        """Initialize empty PredictionBlock."""
        self.table: Optional[pl.DataFrame] = None

    def add_prediction(self, np_arr: np.ndarray, meta_dict: Dict[str, Any]) -> None:
        """
        Add prediction data with metadata.

        Args:
            np_arr: Numpy array of predictions
            meta_dict: Dictionary with metadata (model, fold, repeat, partition, processing, seed)
        """
        # Required metadata fields
        required_fields = ["model", "fold", "repeat", "partition", "processing", "seed"]
        for field in required_fields:
            if field not in meta_dict:
                raise ValueError(f"Missing required metadata field: {field}")

        # Convert predictions to list format for Polars
        preds_list = [row.tolist() for row in np_arr]

        # Create sample indices (assuming sequential from 0)
        n_samples = len(np_arr)
        samples = list(range(n_samples))

        # Create new rows DataFrame
        new_rows = pl.DataFrame({
            "sample": samples,
            "model": [meta_dict["model"]] * n_samples,
            "fold": [meta_dict["fold"]] * n_samples,
            "repeat": [meta_dict["repeat"]] * n_samples,
            "partition": [meta_dict["partition"]] * n_samples,
            "processing": [meta_dict["processing"]] * n_samples,
            "seed": [meta_dict["seed"]] * n_samples,
            "preds": preds_list
        })

        if self.table is None:
            self.table = new_rows
        else:
            self.table = pl.concat([self.table, new_rows], how="vertical")

    def prediction(self, filter_dict: Dict[str, Any]) -> np.ndarray:
        """
        Fetch stacked array view of predictions with filtering.

        Args:
            filter_dict: Dictionary of column: value pairs for filtering

        Returns:
            Stacked numpy array of predictions
        """
        if self.table is None:
            raise ValueError("No predictions available")

        # Apply filter
        filtered_df = self.table
        for column, value in filter_dict.items():
            if column not in self.table.columns:
                raise ValueError(f"Column '{column}' not found in predictions")

            # Handle list values with 'is_in' for multiple matches
            if isinstance(value, (list, tuple, np.ndarray)):
                filtered_df = filtered_df.filter(pl.col(column).is_in(value))
            else:
                filtered_df = filtered_df.filter(pl.col(column) == value)        # Extract and stack prediction arrays
        pred_lists = filtered_df.select("preds").to_numpy().flatten()
        if len(pred_lists) == 0:
            # Determine number of output columns from existing data
            if self.table.height > 0:
                # Get the first prediction to determine shape
                first_pred = self.table.select("preds").limit(1).to_numpy().flatten()[0]
                n_cols = len(first_pred)
                return np.empty((0, n_cols), dtype=np.float32)
            else:
                return np.array([])

        # Convert lists back to arrays and stack
        pred_arrays = [np.array(pred_list) for pred_list in pred_lists]
        predictions = np.vstack(pred_arrays)
        return predictions

    def inverse_transform_prediction(self, transformers: List[Callable[[np.ndarray], np.ndarray]], filter_dict: Dict[str, Any]) -> np.ndarray:
        """
        Apply inverse transformers to predictions and return transformed view.

        Args:
            transformers: List of functions that take and return numpy arrays
            filter_dict: Dictionary of column: value pairs for filtering

        Returns:
            Transformed numpy array of predictions
        """
        # Get the predictions
        preds = self.prediction(filter_dict)

        # Apply transformers in sequence
        transformed = preds
        for transformer in transformers:
            transformed = transformer(transformed)

        return transformed

    def __repr__(self) -> str:
        if self.table is None:
            return "PredictionBlock(empty)"
        return f"PredictionBlock(rows={len(self.table)})"
