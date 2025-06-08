"""
Main SpectroDataset orchestrator class.

This module contains the main facade that coordinates all dataset blocks
and provides the primary public API for users.
"""

from typing import List, Dict, Any, Tuple

import numpy as np

from nirs4all.dataset.features import FeatureBlock
from nirs4all.dataset.targets import TargetBlock
from nirs4all.dataset.metadata import MetadataBlock
from nirs4all.dataset.folds import FoldsManager
from nirs4all.dataset.predictions import PredictionBlock


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

    def get_indexed_features(self, filter_dict: Dict[str, Any], layout: str = "2d", src_concat: bool = False) -> Tuple[Tuple[np.ndarray, ...], Any]:
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

    def add_targets(self, y_df):
        """
        Add target data to the dataset.

        Args:
            y_df: DataFrame with target data (should have columns: sample, targets, processing)        """
        # Convert polars DataFrame to numpy arrays for the new API
        import polars as pl

        if not isinstance(y_df, pl.DataFrame):
            raise ValueError("y_df must be a Polars DataFrame")

        if 'sample' not in y_df.columns or 'targets' not in y_df.columns:
            raise ValueError("y_df must have 'sample' and 'targets' columns")

        # Extract data
        samples = y_df['sample'].to_numpy()
        targets_list = y_df['targets'].to_list()
        processing = y_df.get_column('processing').to_list()[0] if 'processing' in y_df.columns else "raw"

        # Convert targets list to numpy array
        # Determine if we have regression or classification targets
        first_target = targets_list[0]
        if isinstance(first_target, (list, tuple)) and len(first_target) == 1:
            # Single target regression
            target_data = np.array([t[0] for t in targets_list], dtype=np.float32).reshape(-1, 1)
            # Determine if it's classification or regression based on data type
            if all(isinstance(targets_list[i][0], (int, np.integer)) for i in range(len(targets_list))):
                self.targets.add_classification_targets("target", target_data.flatten(), samples, processing)
            else:
                self.targets.add_regression_targets("target", target_data, samples, processing)
        else:
            # Multi-target or multilabel
            target_data = np.array(targets_list, dtype=np.float32)
            if target_data.shape[1] == 1:
                # Single regression target
                self.targets.add_regression_targets("target", target_data, samples, processing)
            else:
                # Multi-dimensional - assume regression for now
                self.targets.add_regression_targets("target", target_data, samples, processing)

    def y(self, filter_dict, processed=True):
        """
        Get target arrays with filtering.

        Args:
            filter_dict: Dictionary of column: value pairs for filtering
            processed: Whether to return processed targets

        Returns:
            Numpy array of targets        """
        # Use the new API - pass processed as encoded parameter
        return self.targets.y(filter_dict, encoded=processed)

    def get_indexed_targets(self, filter_dict):
        """
        Get targets with their index information.

        Args:
            filter_dict: Dictionary of column: value pairs for filtering

        Returns:
            List of tuples: [(target_array, index_dataframe), ...]
        """        # Get target data
        target_data = self.targets.y(filter_dict)

        # Get sample indices - extract from samples in the first target source
        if not self.targets.sources:
            return []

        # Get the first source to extract sample info
        first_source_key = list(self.targets.sources.keys())[0]
        first_source = self.targets.sources[first_source_key]

        # Apply sample filtering if specified
        if 'sample' in filter_dict:
            sample_filter = filter_dict['sample']
            if isinstance(sample_filter, (list, tuple, np.ndarray)):
                mask = np.isin(first_source.samples, sample_filter)
                sample_indices = first_source.samples[mask]
            else:
                mask = first_source.samples == sample_filter
                sample_indices = first_source.samples[mask]
        else:
            sample_indices = first_source.samples

        # Create index DataFrame
        import polars as pl
        index_df = pl.DataFrame({'sample': sample_indices})

        # Return as list of tuples for consistency with features API
        return [(target_data, index_df)]

    def update_y(self, new_values, indexes, processing_id):
        """
        Update target values with new processing version.

        Args:
            new_values: New target values array
            indexes: Index DataFrame indicating which samples to update
            processing_id: Processing identifier for this version
        """
        # For the new API, we need to add a new target source with the updated values
        # This is a simplified implementation - in a full implementation, you might want
        # to support more sophisticated updating        # For the new API, we need to add a new target source with the updated values
        # This is a simplified implementation - in a full implementation, you might want
        # to support more sophisticated updating
        if hasattr(indexes, '__len__') and not isinstance(indexes, np.ndarray):
            # Assume it's a DataFrame or similar
            if hasattr(indexes, 'to_numpy'):
                samples = indexes.to_numpy().flatten()
            elif hasattr(indexes, '__getitem__') and 'sample' in indexes:
                samples = indexes['sample'].to_numpy()
            else:
                samples = np.array([indexes])
        else:
            samples = indexes if isinstance(indexes, np.ndarray) else np.array([indexes])

        # Add as a new regression target with the processing_id
        self.targets.add_regression_targets(
            name="target",
            data=new_values.reshape(-1, 1) if new_values.ndim == 1 else new_values,
            samples=samples,
            processing=processing_id
        )

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
        print()        # Features summary
        if self.features.sources:
            total_samples = self.features.n_samples
            n_sources = len(self.features.sources)
            feature_dims = [src.n_dims for src in self.features.sources]
            print(f"📊 Features: {total_samples} samples, {n_sources} source(s)")
            for i, dims in enumerate(feature_dims):
                rows = self.features.sources[i].n_rows
                print(f"   Source {i}: {rows} rows x {dims} dims")
        else:
            print("📊 Features: No data")
        print()        # Targets summary
        if self.targets.sources:
            n_targets = len(self.targets.sources)
            target_names = self.targets.get_target_names()
            processing_versions = []
            for name in target_names:
                versions = self.targets.get_processing_versions(name)
                processing_versions.extend(versions)
            processing_versions = list(set(processing_versions))
            print(f"🎯 Targets: {n_targets} source(s)")
            print(f"   Target names: {target_names}")
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
