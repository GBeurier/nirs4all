"""
SpectraDataset.py
This module defines the SpectraDataset class, which manages spectral data, features, targets, and results.
"""

from datetime import datetime
import json
from typing import Any, Union, List, Optional, Dict, TYPE_CHECKING
import numpy as np
import polars as pl
import yaml

from SpectraFeatures import SpectraFeatures
from SpectraTargets import SpectraTargets
from CsvLoader import load_data_from_config

if TYPE_CHECKING:
    from DatasetView import DatasetView


class SpectraDataset:
    """Main dataset class with efficient operations and clear interface."""

    def __init__(self, float64: bool = True, task_type: str = "auto"):
        self.float64 = float64        # Core data
        self.features: Optional[SpectraFeatures] = None

        # Enhanced index schema for complex pipeline operations
        self.indices = pl.DataFrame({
            # Core identification
            "row": pl.Series([], dtype=pl.Int64),          # Original row index
            "sample": pl.Series([], dtype=pl.Int64),       # Sample identifier

            # Source and origin tracking
            "origin": pl.Series([], dtype=pl.Int64),       # Original source identifier

            # Data partitioning
            "partition": pl.Series([], dtype=pl.Utf8),     # train/val/test/etc.
            "group": pl.Series([], dtype=pl.Int64),        # Group identifier for splits

            # Pipeline execution context
            "branch": pl.Series([], dtype=pl.Int64),       # Pipeline branch identifier
            "processing": pl.Series([], dtype=pl.Utf8),    # Processing level/stage

            # # Advanced features for complex operations
            # "cluster": pl.Series([], dtype=pl.Int64),      # Cluster assignment
            # "centroid": pl.Series([], dtype=pl.Boolean),   # Centroid designation
            # "weight": pl.Series([], dtype=pl.Float64),     # Sample weight

            # Temporal and versioning
            # "timestamp": pl.Series([], dtype=pl.Datetime),  # Processing timestamp
            # "version": pl.Series([], dtype=pl.Int64),      # Data version
        })

        # Target management
        self.target_manager = SpectraTargets(task_type=task_type)        # Results and folds management
        # Initialize with empty DataFrame but with proper schema and one dummy row to ensure correct types
        dummy_data = {
            "sample": [0],
            "seed": [0],
            "branch": [0],
            "model": [""],
            "fold": [0],
            "stack_index": [0],
            "prediction": [0.0],
            "datetime": [datetime.now()],
            "partition": [""],
            "prediction_type": [""],
        }
        self.results = pl.DataFrame(dummy_data)
        # Remove the dummy row to get an empty DataFrame with proper schema
        self.results = self.results.filter(pl.col("sample") == -1)  # This will create empty DataFrame

        self.folds = []  # List of fold definitions        # Counters
        self._next_row = 0
        self._next_sample = 0

    def __len__(self) -> int:
        return len(self.indices)

    def __repr__(self) -> str | tuple[Any, ...]:
        text = "\n"
        if self.features is not None:
            for i, source in enumerate(self.features.sources):
                text += f"Source {i}: "
                if isinstance(source, np.ndarray):
                    text += f"{source.shape[0]}x{source.shape[1]} "
                    feature_mean = np.mean(source, axis=0)
                    text += f"Mean: {feature_mean.mean():.2f}, Std: {feature_mean.std():.2f}\n"

        if not text:
            text = "Empty Dataset"

        text += "\n"
        text += f"Samples: {self._next_sample}, Rows: {self._next_row}, Features: {len(self.features.sources) if self.features else 0}\n"
        text += f"Partitions: {self.indices['partition'].unique().to_list()}\n"
        for partition in self.indices['partition'].unique():
            text += f"  {partition}: {len(self.indices.filter(pl.col('partition') == partition))} samples\n"
        text += f"Groups: {self.indices['group'].unique().to_list()} - "
        text += f"Branches: {self.indices['branch'].unique().to_list()} - "
        text += f"Processing: {self.indices['processing'].unique().to_list()}\n"
        text += f"Targets: {self.target_manager.get_info()}\n"
        text += f"Results: {self.get_results_summary()}\n"
        return text

    def copy(self):
        """Create a deep copy of the dataset"""
        import copy
        new_dataset = SpectraDataset(float64=self.float64)
        new_dataset.features = copy.deepcopy(self.features)
        new_dataset.indices = self.indices.clone()
        new_dataset.target_manager = copy.deepcopy(self.target_manager)
        new_dataset.results = self.results.clone()
        new_dataset._next_sample = self._next_sample
        new_dataset._next_row = self._next_row
        return new_dataset

    def add_data(self,
                 features: Union[np.ndarray, List[np.ndarray]],
                 targets: Optional[np.ndarray] = None,
                 partition: str = "train",
                 group: int = 0,
                 branch: int = 0,
                 processing: str = "raw",
                 origin: Optional[Union[int, List[int]]] = None,
                 sample_ids: Optional[List[int]] = None) -> List[int]:
        """Add new data and return sample IDs."""

        if isinstance(features, np.ndarray):
            features = [features]

        n_samples = len(features[0])

        # Generate sample IDs
        if sample_ids is not None:
            sample_ids = sample_ids
        else:
            sample_ids = list(range(self._next_sample, self._next_sample + n_samples))
        row_ids = list(range(self._next_row, self._next_row + n_samples))

        # Add features
        if self.features is None:
            self.features = SpectraFeatures(features)
        else:
            self.features.append(features)

        # Add indices
        if origin is None:
            origin = sample_ids
        elif isinstance(origin, int):
            origin = [origin] * n_samples

        new_indices = pl.DataFrame({
            "row": row_ids,
            "sample": sample_ids,
            "origin": origin,
            "partition": [partition] * n_samples,
            "group": [group] * n_samples,
            "branch": [branch] * n_samples,
            "processing": [processing] * n_samples,
        })

        self.indices = pl.concat([self.indices, new_indices])        # Add targets if provided
        if targets is not None:
            self.target_manager.add_targets(sample_ids, targets)

        # Update counters
        self._next_row += n_samples
        self._next_sample += n_samples

        return sample_ids

    def sample_augmentation(self,
                            partition: str = "train",
                            n_copies: int = 1,
                            processing_tag: str = "augmented",
                            group_filter: Optional[int] = None) -> List[int]:
        """
        Sample augmentation: Copy train samples to create new rows with new sample IDs.

        Args:
            partition: Source partition to augment (default: "train")
            n_copies: Number of copies to create for each original sample
            processing_tag: Processing tag for augmented samples
            group_filter: Optional group filter to augment only specific groups

        Returns:
            List of new sample IDs created
        """
        # Get original train samples (origin == sample_id and partition == train)
        base_filter = (pl.col("origin") == pl.col("sample")) & (pl.col("partition") == partition)
        if group_filter is not None:
            base_filter = base_filter & (pl.col("group") == group_filter)

        original_indices = self.indices.filter(base_filter)

        if len(original_indices) == 0:
            return []

        original_row_ids = original_indices["row"].to_list()
        original_sample_ids = original_indices["sample"].to_list()

        new_sample_ids = []

        for copy_idx in range(n_copies):
            # Create new sample IDs
            copy_sample_ids = list(range(self._next_sample, self._next_sample + len(original_sample_ids)))
            new_sample_ids.extend(copy_sample_ids)

            # Create new row IDs
            copy_row_ids = list(range(self._next_row, self._next_row + len(original_sample_ids)))

            # Copy features for all sources
            if self.features is not None:
                original_features = []
                for source_idx in range(len(self.features.sources)):
                    source_features = self.features.get_source(source_idx, np.array(original_row_ids))
                    original_features.append(source_features)
                self.features.append(original_features)
              # Create new indices preserving origin but with new sample IDs
            new_indices_data = []
            for i, (orig_row, orig_sample) in enumerate(zip(original_row_ids, original_sample_ids)):
                orig_data = original_indices.filter(pl.col("row") == orig_row).row(0, named=True)
                new_indices_data.append({
                    "row": copy_row_ids[i],
                    "sample": copy_sample_ids[i],
                    "origin": orig_data["origin"],  # Keep original origin
                    "partition": orig_data["partition"],
                    "group": orig_data["group"],
                    "branch": orig_data["branch"],
                    "processing": processing_tag,
                })

            new_indices = pl.DataFrame(new_indices_data)
            self.indices = pl.concat([self.indices, new_indices])

            # Update counters
            self._next_row += len(original_sample_ids)
            self._next_sample += len(original_sample_ids)

        return new_sample_ids

    def sample_augmentation_by_indices(self,
                                       sample_indices_to_augment: List[int],
                                       processing_tag: str = "balanced_augmented",
                                       partition: Optional[str] = None) -> List[int]:
        """
        Sample augmentation by specific indices: Create copies of specified samples.

        This method allows for flexible sample balancing by specifying exactly which
        samples to augment. Sample indices can appear multiple times in the list to
        create multiple copies for class balancing.

        Args:
            sample_indices_to_augment: List of sample indices to augment. Can contain
                                     duplicates to create multiple copies of the same sample.
            processing_tag: Processing tag for augmented samples
            partition: Optional partition filter - if provided, only augment samples
                      from this partition. If None, augment from any partition.

        Returns:
            List of new sample IDs created

        Example:
            # Balance classes by augmenting minority class samples multiple times
            minority_samples = [1, 5, 10]  # Sample IDs of minority class
            # Augment each minority sample 3 times for balancing
            indices_to_augment = minority_samples * 3
            new_ids = dataset.sample_augmentation_by_indices(indices_to_augment)
        """
        if not sample_indices_to_augment:
            return []

        new_sample_ids = []

        # Process each sample index in the list (allowing duplicates)
        for sample_id_to_augment in sample_indices_to_augment:
            # Find the original sample row
            base_filter = (pl.col("sample") == sample_id_to_augment)
            if partition is not None:
                base_filter = base_filter & (pl.col("partition") == partition)

            # Get the first matching row for this sample (should use most recent processing)
            matching_rows = self.indices.filter(base_filter)
            if len(matching_rows) == 0:
                print(f"Warning: Sample ID {sample_id_to_augment} not found" +
                      (f" in partition '{partition}'" if partition else ""))
                continue

            # Use the most recent row (last processing) for this sample
            original_row_data = matching_rows.row(-1, named=True)
            original_row_id = original_row_data["row"]

            # Create new sample and row IDs
            new_sample_id = self._next_sample
            new_row_id = self._next_row
            new_sample_ids.append(new_sample_id)

            # Copy features from original row
            if self.features is not None:
                original_features = []
                for source_idx in range(len(self.features.sources)):
                    source_features = self.features.get_source(source_idx, np.array([original_row_id]))
                    original_features.append(source_features)
                self.features.append(original_features)

            # Create new index entry
            new_index_data = {
                "row": new_row_id,
                "sample": new_sample_id,
                "origin": original_row_data["origin"],  # Keep original origin
                "partition": original_row_data["partition"],
                "group": original_row_data["group"],
                "branch": original_row_data["branch"],
                "processing": processing_tag,
            }

            new_indices = pl.DataFrame([new_index_data])
            self.indices = pl.concat([self.indices, new_indices])

            # Copy targets if they exist
            try:
                original_targets = self.target_manager.get_targets([sample_id_to_augment])
                if original_targets is not None and len(original_targets) > 0:
                    self.target_manager.add_targets([new_sample_id], original_targets)
            except (AttributeError, ValueError, IndexError):
                # No targets to copy
                pass

            # Update counters
            self._next_row += 1
            self._next_sample += 1

        return new_sample_ids

    def feature_augmentation(self, processing_tag: str = "feat_augmented") -> None:
        """
        Feature augmentation: For all samples, create copies with different processing.

        This creates new rows for each existing sample with the same sample_id and origin,
        but different processing tags. The features are initially copied but can be
        modified by transformers later.

        Args:
            processing_tag: New processing tag for augmented features
        """
        if self.features is None or len(self.indices) == 0:
            return

        # Get all current samples
        current_indices = self.indices.to_pandas()

        new_indices_data = []
        new_features_data = []

        # For each source, copy all features
        for source_idx in range(len(self.features.sources)):
            source_features = self.features.sources[source_idx].copy()
            new_features_data.append(source_features)

        # Create new rows with same sample IDs but different processing
        for _, row in current_indices.iterrows():
            new_row_id = self._next_row
            self._next_row += 1

            new_indices_data.append({
                "row": new_row_id,
                "sample": row["sample"],
                "origin": row["origin"],
                "partition": row["partition"],
                "group": row["group"],
                "branch": row["branch"],
                "processing": processing_tag,
            })

        # Add new features
        self.features.append(new_features_data)

        # Add new indices
        new_indices = pl.DataFrame(new_indices_data)
        self.indices = pl.concat([self.indices, new_indices])

    def branch_dataset(self, n_branches: int) -> None:
        """
        Branching: Copy all train data for each branch with different branch IDs.

        Args:
            n_branches: Number of branches to create
        """
        if len(self.indices) == 0:
            return

        # Get all train data
        train_indices = self.indices.filter(pl.col("partition") == "train")

        if len(train_indices) == 0:
            return

        train_row_ids = train_indices["row"].to_list()

        for branch_id in range(1, n_branches):  # Branch 0 already exists
            # Create new row IDs
            new_row_ids = list(range(self._next_row, self._next_row + len(train_row_ids)))

            # Copy features for all sources
            if self.features is not None:
                branch_features = []
                for source_idx in range(len(self.features.sources)):
                    source_features = self.features.get_source(source_idx, np.array(train_row_ids))
                    branch_features.append(source_features)
                self.features.append(branch_features)

            # Create new indices with different branch ID
            new_indices_data = []
            for i, old_row_id in enumerate(train_row_ids):
                orig_data = train_indices.filter(pl.col("row") == old_row_id).row(0, named=True)
                new_indices_data.append({
                    "row": new_row_ids[i],
                    "sample": orig_data["sample"],
                    "origin": orig_data["origin"],
                    "partition": orig_data["partition"],
                    "group": orig_data["group"],
                    "branch": branch_id,
                    "processing": orig_data["processing"],
                })

            new_indices = pl.DataFrame(new_indices_data)
            self.indices = pl.concat([self.indices, new_indices])

            # Update counter
            self._next_row += len(train_row_ids)

    def get_features_2d(self,
                        filters: Optional[Dict] = None,
                        concatenate_sources: bool = True,
                        concatenate_processing: bool = True) -> np.ndarray:
        """
        Get features in 2D format with various concatenation options.

        Args:
            filters: Dictionary of filters to apply
            concatenate_sources: Whether to concatenate different sources
            concatenate_processing: Whether to concatenate different processing levels

        Returns:
            2D numpy array of features
        """
        # Apply filters
        if filters:
            indices = self.indices
            for key, value in filters.items():
                if isinstance(value, list):
                    indices = indices.filter(pl.col(key).is_in(value))
                else:
                    indices = indices.filter(pl.col(key) == value)
        else:
            indices = self.indices

        if len(indices) == 0:
            return np.array([]).reshape(0, 0)

        if self.features is None:
            return np.array([]).reshape(0, 0)

        row_ids = indices["row"].to_numpy()

        if concatenate_sources and concatenate_processing:
            # Simple concatenation of all features
            result = self.features.get_by_rows(row_ids, concatenate=True)
            if isinstance(result, list):
                return np.concatenate(result, axis=1)
            return result
        else:
            # More complex logic for different concatenation strategies
            all_features = []
            for source_idx in range(len(self.features.sources)):
                source_features = self.features.get_source(source_idx, row_ids)
                all_features.append(source_features)

            if concatenate_sources:
                return np.concatenate(all_features, axis=1)
            else:
                # Return concatenated array for now
                return np.concatenate(all_features, axis=1)

    def get_features_3d(self,
                        filters: Optional[Dict] = None,
                        axis_order: str = "samples_features_processing") -> np.ndarray:
        """
        Get features in 3D format for deep learning.

        Args:
            filters: Dictionary of filters to apply
            axis_order: Order of axes - "samples_features_processing" or "samples_processing_features"

        Returns:
            3D numpy array of features
        """
        # This is a placeholder for 3D feature extraction
        # Implementation would depend on how processing levels are organized
        indices = self.indices
        if filters:
            for key, value in filters.items():
                if isinstance(value, list):
                    indices = indices.filter(pl.col(key).is_in(value))
                else:
                    indices = indices.filter(pl.col(key) == value)

        if len(indices) == 0 or self.features is None:
            return np.array([]).reshape(0, 0, 0)

        # Group by sample and processing to create 3D structure
        # This is a simplified implementation
        row_ids = indices["row"].to_numpy()
        result = self.features.get_by_rows(row_ids, concatenate=True)

        if isinstance(result, list):
            features_2d = np.concatenate(result, axis=1)
        else:
            features_2d = result

        # For now, just add a dimension
        if axis_order == "samples_features_processing":
            return features_2d.reshape(features_2d.shape[0], features_2d.shape[1], 1)
        else:
            return features_2d.reshape(features_2d.shape[0], 1, features_2d.shape[1])

    def add_feature_augmentation(self,
                                 sample_ids: List[int],
                                 features: np.ndarray,
                                 processing_tag: str,
                                 source_partition: str = "train") -> None:
        """
        DEPRECATED: Use feature_augmentation() instead.

        Add feature augmentation to existing samples.

        This creates new feature sources with the same sample IDs but different processing tags.
        Used for feature augmentation where we want to add new features to existing samples.

        Args:
            sample_ids: List of existing sample IDs to augment
            features: New features array [n_samples, n_features]
            processing_tag: Unique tag for this augmentation
            source_partition: Partition these samples came from
        """
        n_samples = len(sample_ids)

        if features.shape[0] != n_samples:
            raise ValueError(f"Features shape {features.shape[0]} doesn't match sample_ids length {n_samples}")

        # Add features as new source
        if self.features is None:
            self.features = SpectraFeatures([features])
        else:
            self.features.append([features])

        # Create new rows with same sample IDs but different processing
        row_ids = list(range(self._next_row, self._next_row + n_samples))

        # Get existing data for these samples to preserve partition, group, branch info
        existing_data = self.indices.filter(pl.col("sample").is_in(sample_ids))

        if len(existing_data) == 0:
            raise ValueError(f"No existing data found for sample_ids: {sample_ids}")

        # Take the first occurrence of each sample to get base metadata
        base_data = existing_data.group_by("sample").first().sort("sample")

        new_indices = pl.DataFrame({
            "row": row_ids,
            "sample": sample_ids,
            "origin": base_data["origin"].to_list(),
            "partition": base_data["partition"].to_list(),
            "group": base_data["group"].to_list(),
            "branch": base_data["branch"].to_list(),
            "processing": [processing_tag] * n_samples,
        })

        self.indices = pl.concat([self.indices, new_indices])        # Update row counter
        self._next_row += n_samples

        print(f"  📊 Added {n_samples} samples with {features.shape[1]} features (processing: {processing_tag})")

    def select(self, **filters) -> 'DatasetView':
        """Create an efficient view of the dataset with filters applied."""
        # Import here to avoid circular import
        try:
            from .DatasetView import DatasetView
        except ImportError:
            try:
                from DatasetView import DatasetView
            except ImportError:
                # Last resort: assume DatasetView is available in globals
                import sys
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.insert(0, current_dir)
                from DatasetView import DatasetView
        return DatasetView(self, filters)

    def get_features(self,
                     row_indices: np.ndarray,
                     source_indices: Optional[Union[int, List[int]]] = None,
                     concatenate: bool = False) -> Union[np.ndarray, List[np.ndarray]]:
        """Direct feature access by row indices."""
        if self.features is None:
            return np.array([])
        return self.features.get_by_rows(row_indices, source_indices, concatenate)

    def get_targets(self, sample_ids: List[int],
                    representation: str = "auto",
                    transformer_key: Optional[str] = None) -> np.ndarray:
        """Get targets for specific samples using SpectraTargets."""
        return self.target_manager.get_targets(sample_ids, representation, transformer_key)

    def update_features(self, row_indices: np.ndarray,
                        new_features: Union[np.ndarray, List[np.ndarray]],
                        source_indices: Optional[Union[int, List[int]]] = None):
        """Update features in-place."""
        if self.features is not None:
            self.features.update_rows(row_indices, new_features, source_indices)

    def update_processing(self, sample_ids: List[int], processing_tag: str):
        """Update processing tags for samples."""
        mask = pl.col("sample").is_in(sample_ids)
        self.indices = self.indices.with_columns(
            pl.when(mask).then(pl.lit(processing_tag)).otherwise(pl.col("processing")).alias("processing")
        )    # Target management methods

    def fit_transform_targets(self, sample_ids, transformers, representation="auto", transformer_key="default"):
        """Fit target transformers and return transformed targets."""
        return self.target_manager.fit_transform_targets(
            sample_ids, transformers, representation, transformer_key)

    def inverse_transform_predictions(self, predictions, representation="auto", transformer_key="default", to_original=True):
        """Inverse transform predictions back to original format."""
        return self.target_manager.inverse_transform_predictions(
            predictions, representation, transformer_key, to_original)

    def get_target_info(self) -> Dict[str, Any]:
        """Get information about targets."""
        return self.target_manager.get_info()

    @property
    def task_type(self) -> str:
        """Get the task type."""
        return self.target_manager.task_type

    @property
    def n_classes(self) -> int:
        """Get number of classes for classification tasks."""
        return self.target_manager.n_classes_

    @property
    def classes_(self) -> Optional[np.ndarray]:
        """Get class labels for classification tasks."""
        return self.target_manager.classes_

    @property
    def is_binary(self) -> bool:
        """Check if this is a binary classification task."""
        return self.target_manager.is_binary

    # Results management methods
    def add_predictions(self,
                       sample_ids: List[int],
                       predictions: np.ndarray,
                       model_name: str,
                       partition: str = "test",
                       fold: int = -1,
                       seed: int = 42,
                       branch: int = 0,
                       stack_index: int = 0,
                       prediction_type: str = "raw") -> None:
        """Add predictions to the results DataFrame."""
        # Create datetime for this prediction batch
        prediction_time = datetime.now()        # Prepare data for results DataFrame with proper types
        results_data = {
            "sample": [int(sid) for sid in sample_ids],  # Ensure int type, polars will convert to Int64
            "seed": [int(seed)] * len(sample_ids),  # Ensure int type, polars will convert to Int64
            "branch": [int(branch)] * len(sample_ids),  # Ensure int type, polars will convert to Int64
            "model": [str(model_name)] * len(sample_ids),
            "fold": [int(fold)] * len(sample_ids),  # Ensure int type, polars will convert to Int64
            "stack_index": [int(stack_index)] * len(sample_ids),  # Ensure int type, polars will convert to Int64
            "prediction": predictions.flatten().astype(float),  # Ensure float64
            "datetime": [prediction_time] * len(sample_ids),
            "partition": [str(partition)] * len(sample_ids),
            "prediction_type": [str(prediction_type)] * len(sample_ids),
        }

        # Define schema locally to ensure proper typing
        schema = {
            "sample": pl.Int64,
            "seed": pl.Int64,
            "branch": pl.Int64,
            "model": pl.Utf8,
            "fold": pl.Int64,
            "stack_index": pl.Int64,
            "prediction": pl.Float64,
            "datetime": pl.Datetime,
            "partition": pl.Utf8,
            "prediction_type": pl.Utf8,
        }
        new_results = pl.DataFrame(results_data, schema=schema)

        self.results = pl.concat([self.results, new_results])

    def get_predictions(self,
                       sample_ids: Optional[List[int]] = None,
                       model: Optional[str] = None,
                       fold: Optional[int] = None,
                       partition: Optional[str] = None,
                       prediction_type: Optional[str] = None,
                       as_dict: bool = False) -> Union[pl.DataFrame, Dict[str, np.ndarray]]:
        """Get predictions with optional filtering."""

        filtered = self.results        # Apply filters
        if sample_ids is not None:
            filtered = filtered.filter(pl.col("sample").is_in(sample_ids))
        if model is not None:
            filtered = filtered.filter(pl.col("model") == model)
        if fold is not None:
            filtered = filtered.filter(pl.col("fold") == fold)
        if partition is not None:
            filtered = filtered.filter(pl.col("partition") == partition)
        if prediction_type is not None:
            filtered = filtered.filter(pl.col("prediction_type") == prediction_type)

        if as_dict:
            return {
                "sample_ids": filtered["sample"].to_numpy(),
                "predictions": filtered["prediction"].to_numpy(),
                "model": filtered["model"].to_numpy(),
                "fold": filtered["fold"].to_numpy(),
                "partition": filtered["partition"].to_numpy(),
                "prediction_type": filtered["prediction_type"].to_numpy(),
            }

        return filtered

    def get_fold_predictions(self,
                           model_name: str,
                           aggregation: str = "mean",
                           partition: str = "test",
                           prediction_type: str = "raw") -> Dict[str, np.ndarray]:
        """Get aggregated predictions across folds."""

        # Get all predictions for this model (returns DataFrame)
        fold_preds = self.get_predictions(
            model=model_name,
            partition=partition,
            prediction_type=prediction_type,
            as_dict=False  # Ensure we get DataFrame
        )

        if len(fold_preds) == 0:
            return {"sample_ids": np.array([]), "predictions": np.array([])}

        if aggregation == "mean":
            # Simple mean across folds
            assert isinstance(fold_preds, pl.DataFrame), "Expected DataFrame"
            grouped = fold_preds.group_by("sample").agg([
                pl.col("prediction").mean().alias("mean_prediction")
            ])
            return {
                "sample_ids": grouped["sample"].to_numpy(),
                "predictions": grouped["mean_prediction"].to_numpy()
            }

        elif aggregation == "weighted":
            # For now, simple mean (would need loss information for true weighting)
            # TODO: Implement loss-weighted aggregation when loss tracking is added
            return self.get_fold_predictions(model_name, "mean", partition, prediction_type)

        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

    def get_reconstructed_train_predictions(self, model_name: str) -> Dict[str, np.ndarray]:
        """Get out-of-fold predictions for training samples (useful for stacking)."""        # Get all fold predictions for training partition (returns DataFrame)
        fold_preds = self.get_predictions(
            model=model_name,
            partition="train",
            prediction_type="raw",
            as_dict=False  # Ensure we get DataFrame
        )

        if len(fold_preds) == 0:
            return {"sample_ids": np.array([]), "predictions": np.array([])}        # For each sample, we want the prediction from the fold where it was NOT in training
        # This requires fold information to be properly stored
        # For now, return the available predictions
        assert isinstance(fold_preds, pl.DataFrame), "Expected DataFrame"
        unique_samples = fold_preds.group_by("sample").agg([
            pl.col("prediction").first().alias("oof_prediction")
        ])

        return {
            "sample_ids": unique_samples["sample"].to_numpy(),
            "predictions": unique_samples["oof_prediction"].to_numpy()
        }

    # Fold management methods
    def add_folds(self, fold_definitions: List[Dict[str, Any]]) -> None:
        """Add fold definitions to the dataset."""
        self.folds = fold_definitions

    def get_fold(self, fold_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific fold definition."""
        for fold in self.folds:
            if fold.get("fold_id") == fold_id:
                return fold
        return None

    def iter_folds(self):
        """Iterate over all folds."""
        for fold in self.folds:
            yield fold

    def clear_results(self, model: Optional[str] = None) -> None:
        """Clear results, optionally for a specific model."""
        if model is not None:
            self.results = self.results.filter(pl.col("model") != model)
        else:
            # Define schema to match the initialization schema
            results_schema = {
                "sample": pl.Int64,
                "seed": pl.Int64,
                "branch": pl.Int64,
                "model": pl.Utf8,
                "fold": pl.Int64,
                "stack_index": pl.Int64,
                "prediction": pl.Float64,
                "datetime": pl.Datetime,
                "partition": pl.Utf8,
                "prediction_type": pl.Utf8,
            }
            self.results = pl.DataFrame(schema=results_schema)

    def get_results_summary(self) -> Dict[str, Any]:
        """Get a summary of stored results."""
        if len(self.results) == 0:
            return {"n_predictions": 0, "models": [], "partitions": [], "folds": []}

        return {
            "n_predictions": len(self.results),
            "models": self.results["model"].unique().to_list(),
            "partitions": self.results["partition"].unique().to_list(),
            "folds": sorted(self.results["fold"].unique().to_list()),
            "prediction_types": self.results["prediction_type"].unique().to_list(),
        }

    @staticmethod
    def from_config(config):

        if isinstance(config, str):
            if config.endswith(".json"):
                with open(config, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            elif config.endswith(".yaml") or config.endswith(".yml"):
                with open(config, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
        print(config)
        data = load_data_from_config(config["dataset"])
        dataset = SpectraDataset()

        if isinstance(data, tuple):
            features, targets = data
            dataset.add_data(
                features=features,
                targets=targets,
                partition="train",
            )
        else:
            for name, (X_data, Y_data) in data.items():
                dataset.add_data(
                    features=X_data,
                    targets=Y_data,
                    partition=name,
                )

        return dataset

    def merge_sources(self, merge_config):
        """Merge data sources according to configuration."""
        print(f"[MOCK] Merging sources with config: {merge_config}")
        return self

    def get_sample_indices_by_class(self,
                                    partition: str = "train",
                                    processing: Optional[str] = None) -> Dict[Any, List[int]]:
        """
        Get sample indices grouped by class/target values for class balancing.

        Args:
            partition: Partition to get samples from
            processing: Optional processing filter

        Returns:
            Dictionary mapping class values to lists of sample indices

        Example:
            # Get samples by class for balancing
            class_samples = dataset.get_sample_indices_by_class("train")
            # Result: {'class_A': [1, 3, 5], 'class_B': [2, 4, 6, 7, 8]}

            # Balance by augmenting minority class
            min_class_size = min(len(samples) for samples in class_samples.values())
            max_class_size = max(len(samples) for samples in class_samples.values())

            indices_to_augment = []
            for class_val, sample_ids in class_samples.items():
                if len(sample_ids) < max_class_size:
                    # Augment minority class samples
                    n_needed = max_class_size - len(sample_ids)
                    augment_samples = np.random.choice(sample_ids, n_needed, replace=True)
                    indices_to_augment.extend(augment_samples)

            dataset.sample_augmentation_by_indices(indices_to_augment)
        """
        # Filter indices by partition and processing
        base_filter = pl.col("partition") == partition
        if processing is not None:
            base_filter = base_filter & (pl.col("processing") == processing)

        filtered_indices = self.indices.filter(base_filter)
        sample_ids = filtered_indices["sample"].unique().to_list()

        if not sample_ids:
            return {}

        # Get targets for these samples
        try:
            targets = self.target_manager.get_targets(sample_ids)
            if targets is None:
                return {}
                  # Group sample indices by target value
            class_to_samples = {}
            for sample_id, target in zip(sample_ids, targets):
                if target not in class_to_samples:
                    class_to_samples[target] = []
                class_to_samples[target].append(sample_id)

            return class_to_samples

        except (AttributeError, ValueError, IndexError):
            # No targets available
            return {}