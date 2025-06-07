"""
Main SpectroDataset orchestrator class.

This module contains the main facade that coordinates all dataset blocks
and provides the primary public API for users.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from .features import FeatureBlock
from .targets import TargetBlock
from .metadata import MetadataBlock
from .folds import FoldsManager
from .predictions import PredictionBlock


class SpectroDataset:
    """
    Main dataset orchestrator that manages feature, target, metadata,
    fold, and prediction blocks.
    """
    def __init__(self):
        """Initialize an empty SpectroDataset."""
        self.features = FeatureBlock()
        self.targets = TargetBlock()
        self.metadata = MetadataBlock()
        self.folds = FoldsManager()
        self.predictions = PredictionBlock()

    def add_features(self, x_list: List[np.ndarray]) -> None:
        """
        Add feature arrays to the dataset.

        Args:
            x_list: List of 2D numpy arrays representing different feature sources
        """
        self.features.add_features(x_list)

    def x(self, filter_dict: Dict[str, Any], layout: str = "2d", src_concat: bool = False) -> Tuple[np.ndarray, ...]:
        """
        Get feature arrays with specified layout and filtering.

        Args:
            filter_dict: Dictionary of column: value pairs for filtering
            layout: Layout type ("2d", "2d_interlaced", "3d", "3d_transpose")
            src_concat: Whether to concatenate sources along axis=1

        Returns:
            Tuple of numpy arrays (zero-copy views)
        """
        return self.features.x(filter_dict, layout, src_concat)

    def get_indexed_features(self, filter_dict: Dict[str, Any], layout: str = "2d",
                           src_concat: bool = False) -> Tuple[Tuple[np.ndarray, ...], Any]:
        """
        Get feature arrays and corresponding index DataFrame.

        Args:
            filter_dict: Dictionary of column: value pairs for filtering
            layout: Layout type ("2d", "2d_interlaced", "3d", "3d_transpose")
            src_concat: Whether to concatenate sources along axis=1

        Returns:
            Tuple of (feature arrays, filtered index DataFrame)
        """
        return self.features.get_indexed_features(filter_dict, layout, src_concat)

    def add_targets(self, y_df, meta_df=None):
        """
        Add target data to the dataset.

        Args:
            y_df: DataFrame with target data
            meta_df: Optional metadata DataFrame
        """
        self.targets.add_targets(y_df, meta_df)

    def y(self, filter_dict, processed=True):
        """
        Get target arrays with filtering.

        Args:
            filter_dict: Dictionary of column: value pairs for filtering
            processed: Whether to return processed targets

        Returns:
            Numpy array of targets
        """
        return self.targets.y(filter_dict, processed)

    def get_indexed_targets(self, filter_dict):
        """
        Get targets with their index information.

        Args:
            filter_dict: Dictionary of column: value pairs for filtering

        Returns:
            List of (target_array, index_dataframe) tuples
        """
        return self.targets.get_indexed_targets(filter_dict)

    def update_y(self, new_values, indexes, processing_id):
        """
        Update target values with new processing version.

        Args:
            new_values: New target values array
            indexes: Index DataFrame indicating which samples to update
            processing_id: Processing identifier for this version
        """
        self.targets.update_y(new_values, indexes, processing_id)

    def add_meta(self, meta_df):
        """
        Add metadata to the dataset.

        Args:
            meta_df: DataFrame with metadata
        """
        self.metadata.add_meta(meta_df)

    def meta(self, filter_dict):
        """
        Get metadata with filtering.

        Args:
            filter_dict: Dictionary of column: value pairs for filtering

        Returns:
            Filtered metadata DataFrame
        """
        return self.metadata.meta(filter_dict)

    def save(self, path: str) -> None:
        """
        Save the dataset to disk.

        Args:
            path: Directory path where to save the dataset
        """
        from . import io
        io.save(self, path)

    def load(self, path: str) -> "SpectroDataset":
        """
        Load a dataset from disk.

        Args:
            path: Directory path containing the saved dataset

        Returns:
            Loaded SpectroDataset instance
        """
        from . import io
        return io.load(path)

    def print_summary(self) -> None:
        """
        Print a comprehensive summary of the dataset.

        Shows counts, dimensions, number of sources, target versions, predictions, etc.
        """
        print("=== SpectroDataset Summary ===")
        print()

        # Features summary
        if self.features.sources:
            total_samples = self.features.n_samples
            n_sources = len(self.features.sources)
            feature_dims = [src.n_dims for src in self.features.sources]
            print(f"📊 Features: {total_samples} samples, {n_sources} source(s)")
            for i, dims in enumerate(feature_dims):
                rows = self.features.sources[i].n_rows
                print(f"   Source {i}: {rows} rows × {dims} dims")
        else:
            print("📊 Features: No data")
        print()

        # Targets summary
        if self.targets.table is not None:
            n_targets = len(self.targets.table)
            processing_versions = self.targets.table.select("processing").unique().to_series().to_list()
            print(f"🎯 Targets: {n_targets} entries")
            print(f"   Processing versions: {processing_versions}")
        else:
            print("🎯 Targets: No data")
        print()

        # Metadata summary
        if self.metadata.table is not None:
            n_meta = len(self.metadata.table)
            meta_cols = self.metadata.table.columns
            print(f"📋 Metadata: {n_meta} entries")
            print(f"   Columns: {meta_cols}")
        else:
            print("📋 Metadata: No data")
        print()

        # Folds summary
        if self.folds.folds:
            n_folds = len(self.folds.folds)
            fold_sizes = [(len(f["train"]), len(f["val"])) for f in self.folds.folds]
            print(f"🔄 Folds: {n_folds} fold(s)")
            for i, (train_size, val_size) in enumerate(fold_sizes):
                print(f"   Fold {i}: {train_size} train, {val_size} val")
        else:
            print("🔄 Folds: No data")
        print()

        # Predictions summary
        if self.predictions.table is not None:
            n_preds = len(self.predictions.table)
            models = self.predictions.table.select("model").unique().to_series().to_list()
            partitions = self.predictions.table.select("partition").unique().to_series().to_list()
            print(f"🔮 Predictions: {n_preds} entries")
            print(f"   Models: {models}")
            print(f"   Partitions: {partitions}")
        else:
            print("🔮 Predictions: No data")
        print()

        print("=" * 30)

    def __repr__(self):
        return f"SpectroDataset(features={self.features}, targets={self.targets}, metadata={self.metadata}, folds={self.folds}, predictions={self.predictions})"
