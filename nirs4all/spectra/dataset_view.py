"""
DatasetView - Enhanced scoped data accessor for pipeline operations

Provides filtered views of SpectraDataset based on pipeline context.
Handles complex indexing, data selection logic, and multiple data representations.
"""
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import polars as pl
from abc import ABC, abstractmethod


class DatasetView:
    """
    Enhanced scoped view of SpectraDataset that provides filtered access to data.

    This is the key component that manages pipeline context and determines
    which subset of data operations should work on.

    Handles:
    - Complex filtering and query logic
    - 2D/3D data representations for deep learning
    - Source-specific data access
    - Processing-aware data selection    - Centroid and clustering logic
    - Augmentation-aware indexing
    """

    def __init__(self, dataset: 'SpectraDataset', filters: Optional[Dict[str, Any]] = None,
                 meta_filters: Optional[Dict[str, Any]] = None):
        """
        Create a filtered view of the dataset.

        Args:
            dataset: The source SpectraDataset
            filters: Dictionary of filters to apply. Enhanced to support:
                - Range filters: {"column": {"min": value, "max": value}}
                - Complex queries: {"column": {"operator": "contains", "value": pattern}}
                - Logical operators: {"_and": [filter1, filter2], "_or": [filter1, filter2]}
                - Nested conditions: {"_not": {"column": value}}
                - Scope stack: {"_inherit_scope": True}
            meta_filters: Post-processing filters (limit, sample, offset, etc.)
        """
        self.dataset = dataset
        self.filters = filters or {}
        self.meta_filters = meta_filters or {}
        self._cached_indices = None
        self._cached_sample_ids = None

        # Advanced filtering configuration
        self.filter_mode = self.filters.get("_filter_mode", "strict")  # strict, lenient, fuzzy
        self.missing_column_strategy = self.filters.get("_missing_columns", "ignore")  # ignore, error, create_default
        self.scope_inheritance = self.filters.get("_inherit_scope", False)

    def _apply_complex_filter(self, indices: pl.DataFrame, key: str, value: Any) -> pl.DataFrame:
        """Apply complex filtering logic with advanced query support."""

        # Handle special filter keys
        if key.startswith("_"):
            return self._apply_meta_filter(indices, key, value)

        # Check if column exists
        if key not in indices.columns:
            if self.missing_column_strategy == "error":
                raise ValueError(f"Filter column '{key}' not found in dataset indices")
            elif self.missing_column_strategy == "ignore":
                return indices
            elif self.missing_column_strategy == "create_default":
                # Add default column
                indices = indices.with_columns(pl.lit(None).alias(key))

        # Handle different value types for advanced filtering
        if isinstance(value, dict):
            return self._apply_dict_filter(indices, key, value)
        elif isinstance(value, list):
            return indices.filter(pl.col(key).is_in(value))
        elif value is None:
            return indices.filter(pl.col(key).is_null())
        else:
            return indices.filter(pl.col(key) == value)

    def _apply_dict_filter(self, indices: pl.DataFrame, key: str, filter_dict: Dict[str, Any]) -> pl.DataFrame:
        """Apply dictionary-based complex filters."""

        # Range filters
        if "min" in filter_dict or "max" in filter_dict:
            conditions = []
            if "min" in filter_dict:
                conditions.append(pl.col(key) >= filter_dict["min"])
            if "max" in filter_dict:
                conditions.append(pl.col(key) <= filter_dict["max"])

            if len(conditions) == 1:
                return indices.filter(conditions[0])
            else:
                return indices.filter(pl.all_horizontal(conditions))

        # Operator-based filters
        if "operator" in filter_dict:
            operator = filter_dict["operator"]
            filter_value = filter_dict["value"]

            if operator == "contains":
                return indices.filter(pl.col(key).str.contains(filter_value))
            elif operator == "starts_with":
                return indices.filter(pl.col(key).str.starts_with(filter_value))
            elif operator == "ends_with":
                return indices.filter(pl.col(key).str.ends_with(filter_value))
            elif operator == "regex":
                return indices.filter(pl.col(key).str.contains(filter_value, literal=False))
            elif operator == "not_equal":
                return indices.filter(pl.col(key) != filter_value)
            elif operator == "greater_than":
                return indices.filter(pl.col(key) > filter_value)
            elif operator == "less_than":
                return indices.filter(pl.col(key) < filter_value)
            elif operator == "in_range":
                return indices.filter(pl.col(key).is_between(filter_value[0], filter_value[1]))
            else:
                raise ValueError(f"Unknown operator: {operator}")

        # Set-based filters
        if "in" in filter_dict:
            return indices.filter(pl.col(key).is_in(filter_dict["in"]))
        if "not_in" in filter_dict:
            return indices.filter(~pl.col(key).is_in(filter_dict["not_in"]))

        return indices

    def _apply_meta_filter(self, indices: pl.DataFrame, key: str, value: Any) -> pl.DataFrame:
        """Apply meta filters (logical operators, special conditions)."""

        if key == "_and":
            # Apply all filters (AND logic)
            for sub_filter in value:
                for sub_key, sub_value in sub_filter.items():
                    indices = self._apply_complex_filter(indices, sub_key, sub_value)
            return indices

        elif key == "_or":
            # Apply any filter (OR logic)
            or_conditions = []
            for sub_filter in value:
                temp_indices = indices
                for sub_key, sub_value in sub_filter.items():
                    temp_indices = self._apply_complex_filter(temp_indices, sub_key, sub_value)
                # Get the difference to find matching rows
                matching_rows = temp_indices.select("row").to_series()
                or_conditions.append(pl.col("row").is_in(matching_rows))

            if or_conditions:
                return indices.filter(pl.any_horizontal(or_conditions))
            return indices

        elif key == "_not":
            # Apply negation
            temp_indices = indices
            for sub_key, sub_value in value.items():
                temp_indices = self._apply_complex_filter(temp_indices, sub_key, sub_value)
            # Get rows that don't match
            excluded_rows = temp_indices.select("row").to_series()
            return indices.filter(~pl.col("row").is_in(excluded_rows))

        elif key == "_sample":
            # Random sampling
            if isinstance(value, int):
                return indices.sample(value)
            elif isinstance(value, float):
                return indices.sample(fraction=value)

        elif key == "_limit":
            # Limit number of results
            return indices.head(value)

        elif key == "_offset":
            # Skip first N results
            return indices.slice(value)

        return indices

    def _get_filtered_indices(self) -> pl.DataFrame:
        """Get indices that match the current filters with enhanced logic."""
        if self._cached_indices is None:
            indices = self.dataset.indices

            # Apply scope inheritance if enabled
            if self.scope_inheritance and hasattr(self, '_parent_context'):
                parent_filters = getattr(self._parent_context, 'current_filters', {})
                combined_filters = {**parent_filters, **self.filters}
            else:
                combined_filters = self.filters

            # Apply each filter with complex logic
            for key, value in combined_filters.items():
                indices = self._apply_complex_filter(indices, key, value)

            # Apply meta filters (post-processing)
            for key, value in self.meta_filters.items():
                if key == "limit":
                    indices = indices.head(value)
                elif key == "offset":
                    indices = indices.slice(value)
                elif key == "sample":
                    if isinstance(value, int):
                        indices = indices.sample(value)
                    elif isinstance(value, float):
                        indices = indices.sample(fraction=value)

            self._cached_indices = indices

        return self._cached_indices

    def _clear_cache(self):
        """Clear cached data when filters change."""
        self._cached_indices = None
        self._cached_sample_ids = None

    def with_filters(self, **additional_filters) -> 'DatasetView':
        """Create a new view with additional filters."""
        combined_filters = {**self.filters, **additional_filters}
        return DatasetView(self.dataset, combined_filters)

    def get_sample_ids(self) -> List[int]:
        """Get sample IDs that match the current scope."""
        if self._cached_sample_ids is None:
            indices = self._get_filtered_indices()
            self._cached_sample_ids = indices.select("sample").to_series().to_list()
        return self._cached_sample_ids

    def get_row_indices(self) -> np.ndarray:
        """Get row indices for direct dataset access."""
        indices = self._get_filtered_indices()
        return indices.select("row").to_series().to_numpy()

    def get_features(self,
                    representation: str = "2d_concatenated",
                    source_indices: Optional[Union[int, List[int]]] = None,
                    processing_filter: Optional[str] = None) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Get features with specified representation and filtering.

        Args:
            representation: "2d_concatenated", "2d_separate", "3d_processing", "3d_sources"
            source_indices: Specific sources to include (overrides filter)
            processing_filter: Specific processing to include

        Returns:
            Features in requested format:
            - 2d_concatenated: (n_samples, total_features) - all sources and processing concatenated
            - 2d_separate: List[np.ndarray] - separate arrays per source
            - 3d_processing: (n_samples, n_features, n_processing) - processing as 3rd dimension
            - 3d_sources: (n_samples, n_sources, n_features) - sources as 3rd dimension
        """
        # Get base row indices
        row_indices = self.get_row_indices()

        if len(row_indices) == 0:
            return np.array([]).reshape(0, 0)

        # Handle source filtering
        active_sources = source_indices
        if active_sources is None:
            active_sources = self.filters.get("active_sources")

        # Get features from dataset
        if representation == "2d_concatenated":
            return self._get_2d_concatenated_features(row_indices, active_sources, processing_filter)
        elif representation == "2d_separate":
            return self._get_2d_separate_features(row_indices, active_sources, processing_filter)
        elif representation == "3d_processing":
            return self._get_3d_processing_features(row_indices, active_sources, processing_filter)
        elif representation == "3d_sources":
            return self._get_3d_sources_features(row_indices, active_sources, processing_filter)
        else:
            raise ValueError(f"Unknown representation: {representation}")

    def _get_2d_concatenated_features(self, row_indices: np.ndarray,
                                    source_indices: Optional[List[int]],
                                    processing_filter: Optional[str]) -> np.ndarray:
        """Get 2D concatenated features: all sources and processing concatenated."""
        # Handle source merge mode
        if self.filters.get("source_merge_mode", False):
            # Force concatenation of all sources
            source_indices = None

        # Get features directly from dataset with concatenation
        features = self.dataset.get_features(row_indices, source_indices, concatenate=True)

        # Handle processing filtering if needed
        if processing_filter:
            features = self._filter_by_processing(features, row_indices, processing_filter)

        return features

    def _get_2d_separate_features(self, row_indices: np.ndarray,
                                source_indices: Optional[List[int]],
                                processing_filter: Optional[str]) -> List[np.ndarray]:
        """Get 2D separate features: list of arrays per source."""
        features = self.dataset.get_features(row_indices, source_indices, concatenate=False)

        if processing_filter:
            features = self._filter_by_processing(features, row_indices, processing_filter)

        return features if isinstance(features, list) else [features]

    def _get_3d_processing_features(self, row_indices: np.ndarray,
                                  source_indices: Optional[List[int]],
                                  processing_filter: Optional[str]) -> np.ndarray:
        """Get 3D features with processing as 3rd dimension."""
        # This is complex - need to group by sample and stack different processing
        indices_df = self._get_filtered_indices()

        # Group by sample to get different processing for same samples
        sample_groups = indices_df.filter(pl.col("row").is_in(row_indices.tolist()))

        if len(sample_groups) == 0:
            return np.array([]).reshape(0, 0, 0)

        # Get unique samples and processing types
        unique_samples = sample_groups.select("sample").unique().to_series().to_list()
        unique_processing = sample_groups.select("processing").unique().to_series().to_list()

        if processing_filter:
            unique_processing = [p for p in unique_processing if p == processing_filter]

        if not unique_processing:
            return np.array([]).reshape(0, 0, 0)

        # Build 3D array
        feature_arrays = []
        for sample_id in unique_samples:
            sample_processing_features = []
            for processing in unique_processing:
                # Get features for this sample and processing
                sample_rows = sample_groups.filter(
                    (pl.col("sample") == sample_id) &
                    (pl.col("processing") == processing)
                ).select("row").to_series().to_numpy()

                if len(sample_rows) > 0:
                    sample_features = self.dataset.get_features(sample_rows, source_indices, concatenate=True)
                    if len(sample_features) > 0:
                        sample_processing_features.append(sample_features[0])  # Take first (should be only one)
                    else:
                        # Placeholder for missing processing
                        if feature_arrays:
                            sample_processing_features.append(np.zeros_like(feature_arrays[0][0]))
                        else:
                            sample_processing_features.append(np.array([]))
                else:
                    # No data for this processing, add placeholder
                    if feature_arrays:
                        sample_processing_features.append(np.zeros_like(feature_arrays[0][0]))
                    else:
                        sample_processing_features.append(np.array([]))

            if sample_processing_features:
                feature_arrays.append(sample_processing_features)

        if not feature_arrays:
            return np.array([]).reshape(0, 0, 0)

        # Convert to 3D numpy array
        try:
            return np.array(feature_arrays)  # Shape: (n_samples, n_processing, n_features)
        except ValueError:
            # Handle mismatched shapes by padding
            return self._pad_and_stack_3d(feature_arrays)

    def _get_3d_sources_features(self, row_indices: np.ndarray,
                               source_indices: Optional[List[int]],
                               processing_filter: Optional[str]) -> np.ndarray:
        """Get 3D features with sources as 3rd dimension."""
        # Get separate features per source
        separate_features = self._get_2d_separate_features(row_indices, source_indices, processing_filter)

        if not separate_features:
            return np.array([]).reshape(0, 0, 0)

        # Convert to 3D array: (n_samples, n_sources, n_features)
        try:
            # Transpose to get sources as middle dimension
            stacked = np.stack(separate_features, axis=1)  # (n_samples, n_sources, n_features)
            return stacked
        except ValueError:
            # Handle mismatched feature counts by padding
            return self._pad_and_stack_sources(separate_features)

    def _filter_by_processing(self, features: Union[np.ndarray, List[np.ndarray]],
                            row_indices: np.ndarray,
                            processing_filter: str) -> Union[np.ndarray, List[np.ndarray]]:
        """Filter features by processing type."""
        # Get the indices that match the processing filter
        indices_df = self._get_filtered_indices()
        filtered_rows = indices_df.filter(
            (pl.col("row").is_in(row_indices.tolist())) &
            (pl.col("processing") == processing_filter)
        ).select("row").to_series().to_numpy()

        if len(filtered_rows) == 0:
            if isinstance(features, list):
                return [np.array([]).reshape(0, f.shape[1] if len(f.shape) > 1 else 0) for f in features]
            else:
                return np.array([]).reshape(0, features.shape[1] if len(features.shape) > 1 else 0)

        # Map filtered rows back to indices in the features array
        row_mapping = {row: i for i, row in enumerate(row_indices)}
        feature_indices = [row_mapping[row] for row in filtered_rows if row in row_mapping]

        if isinstance(features, list):
            return [f[feature_indices] for f in features]
        else:
            return features[feature_indices]

    def _pad_and_stack_3d(self, feature_arrays: List[List[np.ndarray]]) -> np.ndarray:
        """Pad arrays to same shape and stack into 3D."""
        if not feature_arrays:
            return np.array([]).reshape(0, 0, 0)

        # Find maximum dimensions
        max_processing = max(len(sample_features) for sample_features in feature_arrays)
        max_features = 0
        for sample_features in feature_arrays:
            for features in sample_features:
                if len(features.shape) > 0:
                    max_features = max(max_features, features.shape[-1])

        if max_features == 0:
            return np.array([]).reshape(0, 0, 0)

        # Pad and stack
        padded_arrays = []
        for sample_features in feature_arrays:
            padded_sample = []
            for i in range(max_processing):
                if i < len(sample_features):
                    features = sample_features[i]
                    if len(features.shape) == 0 or features.shape[-1] < max_features:
                        # Pad features
                        padded = np.zeros(max_features)
                        if len(features.shape) > 0:
                            padded[:features.shape[-1]] = features
                        padded_sample.append(padded)
                    else:
                        padded_sample.append(features)
                else:
                    # Add zero padding for missing processing
                    padded_sample.append(np.zeros(max_features))
            padded_arrays.append(padded_sample)

        return np.array(padded_arrays)

    def _pad_and_stack_sources(self, source_features: List[np.ndarray]) -> np.ndarray:
        """Pad source arrays to same feature count and stack."""
        if not source_features:
            return np.array([]).reshape(0, 0, 0)

        # Find maximum feature count
        max_features = max(f.shape[1] if len(f.shape) > 1 else 0 for f in source_features)

        if max_features == 0:
            return np.array([]).reshape(0, len(source_features), 0)

        # Pad each source to max_features
        padded_sources = []
        for features in source_features:
            if len(features.shape) < 2:
                padded = np.zeros((len(self.get_sample_ids()), max_features))
            elif features.shape[1] < max_features:
                padded = np.zeros((features.shape[0], max_features))
                padded[:, :features.shape[1]] = features
            else:
                padded = features
            padded_sources.append(padded)

        # Stack along source dimension
        return np.stack(padded_sources, axis=1)  # (n_samples, n_sources, n_features)

    def get_targets(self,
                   representation: str = "auto",
                   transformer_key: Optional[str] = None) -> np.ndarray:
        """Get targets for the filtered samples."""
        sample_ids = self.get_sample_ids()
        return self.dataset.get_targets(sample_ids, representation, transformer_key)

    def get_partition_split(self, partition: str) -> 'DatasetView':
        """Get view filtered to specific partition."""
        return self.with_filters(partition=partition)

    def get_group_split(self, group: Union[int, List[int]]) -> 'DatasetView':
        """Get view filtered to specific group(s)."""
        return self.with_filters(group=group)

    def get_source_split(self, source_indices: Union[int, List[int]]) -> 'DatasetView':
        """Get view filtered to specific source(s)."""
        return self.with_filters(active_sources=source_indices)

    def get_processing_split(self, processing: str) -> 'DatasetView':
        """Get view filtered to specific processing."""
        return self.with_filters(processing=processing)

    def get_branch_split(self, branch: int) -> 'DatasetView':
        """Get view filtered to specific branch."""
        return self.with_filters(branch=branch)

    def get_origin_samples(self) -> 'DatasetView':
        """Get view with only original samples (origin == sample)."""
        indices = self._get_filtered_indices()
        origin_samples = indices.filter(pl.col("origin") == pl.col("sample"))
        origin_sample_ids = origin_samples.select("sample").to_series().to_list()
        return self.with_filters(sample=origin_sample_ids)

    def get_augmented_samples(self) -> 'DatasetView':
        """Get view with only augmented samples (origin != sample)."""
        indices = self._get_filtered_indices()
        augmented_samples = indices.filter(pl.col("origin") != pl.col("sample"))
        augmented_sample_ids = augmented_samples.select("sample").to_series().to_list()
        return self.with_filters(sample=augmented_sample_ids)

    def __len__(self) -> int:
        """Number of samples in the view."""
        return len(self.get_sample_ids())

    def __repr__(self) -> str:
        return f"DatasetView({len(self)} samples, filters={self.filters})"


class DataSelector:
    """
    Context-aware data selector that works with DatasetView.

    Provides default scoping rules and creates appropriate DatasetViews
    for different operation types.
    """

    def __init__(self):
        # Import here to avoid circular imports
        from nirs4all.core.spectra.DataSelector import (
            StandardTransformerRule, ClusterRule, ModelRule,
            FoldRule, SplitRule, AugmentationRule
        )

        self.rules = {
            "transformer": StandardTransformerRule(),
            "cluster": ClusterRule(),
            "model": ModelRule(),
            "fold": FoldRule(),
            "split": SplitRule(),
            "sample_augmentation": AugmentationRule("train"),
            "feature_augmentation": AugmentationRule("train"),
        }

    def get_fit_view(self, dataset: 'SpectraDataset', operation: Any,
                    context: 'PipelineContext') -> 'DatasetView':
        """Get DatasetView for fitting the operation."""
        filters = self._get_fit_filters(operation, context)
        return DatasetView(dataset, filters)

    def get_transform_view(self, dataset: 'SpectraDataset', operation: Any,
                          context: 'PipelineContext') -> 'DatasetView':
        """Get DatasetView for transforming with the operation."""
        filters = self._get_transform_filters(operation, context)
        return DatasetView(dataset, filters)

    def get_predict_view(self, dataset: 'SpectraDataset', operation: Any,
                        context: 'PipelineContext') -> 'DatasetView':
        """Get DatasetView for prediction with the operation."""
        filters = self._get_predict_filters(operation, context)
        return DatasetView(dataset, filters)

    def _get_fit_filters(self, operation: Any, context: 'PipelineContext') -> Dict[str, Any]:
        """Get filters for fitting - delegated to imported DataSelector."""
        # This will be implemented when we integrate with the DataSelector
        return context.get_effective_filters()

    def _get_transform_filters(self, operation: Any, context: 'PipelineContext') -> Dict[str, Any]:
        """Get filters for transforming - delegated to imported DataSelector."""
        return context.get_effective_filters()

    def _get_predict_filters(self, operation: Any, context: 'PipelineContext') -> Dict[str, Any]:
        """Get filters for prediction - delegated to imported DataSelector."""
        return context.get_effective_filters()
        indices = self._get_filtered_indices()
        return indices.select("sample").to_series().to_list()

    def get_row_indices(self) -> np.ndarray:
        """Get row indices for feature extraction."""
        indices = self._get_filtered_indices()
        return indices.select("row").to_series().to_numpy()

    def get_features(self,
                    source_indices: Optional[Union[int, List[int]]] = None,
                    concatenate: bool = True,
                    representation: str = "2d") -> Union[np.ndarray, List[np.ndarray]]:
        """
        Get features for the scoped samples.

        Args:
            source_indices: Which feature sources to include
            concatenate: Whether to concatenate sources
            representation: "2d" (flat), "3d" (with processing dimension), "source_separate"
        """
        row_indices = self.get_row_indices()

        if len(row_indices) == 0:
            return np.array([])

        # Get features from dataset
        features = self.dataset.get_features(
            row_indices=row_indices,
            source_indices=source_indices,
            concatenate=concatenate
        )

        # Handle different representations
        if representation == "3d":
            return self._to_3d_representation(features, row_indices)
        elif representation == "source_separate":
            return self._to_source_separate(features, row_indices)
        else:
            return features

    def get_targets(self,
                   representation: str = "auto",
                   transformer_key: Optional[str] = None) -> np.ndarray:
        """Get targets for the scoped samples."""
        sample_ids = self.get_sample_ids()
        return self.dataset.get_targets(sample_ids, representation, transformer_key)

    def get_unique_processings(self) -> List[str]:
        """Get unique processing hashes in this view."""
        indices = self._get_filtered_indices()
        return indices.select("processing").unique().to_series().to_list()

    def get_unique_sources(self) -> List[int]:
        """Get unique source IDs in this view."""
        indices = self._get_filtered_indices()
        if "source_id" in indices.columns:
            return indices.select("source_id").unique().to_series().to_list()
        return [0]  # Default single source

    def get_groups(self) -> List[int]:
        """Get unique group IDs in this view."""
        indices = self._get_filtered_indices()
        return indices.select("group").unique().to_series().to_list()

    def get_partitions(self) -> List[str]:
        """Get unique partitions in this view."""
        indices = self._get_filtered_indices()
        return indices.select("partition").unique().to_series().to_list()

    def split_by_partition(self) -> Dict[str, 'DatasetView']:
        """Split view by partition, returning a DatasetView for each."""
        partitions = self.get_partitions()
        views = {}

        for partition in partitions:
            partition_filters = self.filters.copy()
            partition_filters['partition'] = partition
            views[partition] = DatasetView(self.dataset, partition_filters)

        return views

    def split_by_group(self) -> Dict[int, 'DatasetView']:
        """Split view by group, returning a DatasetView for each."""
        groups = self.get_groups()
        views = {}

        for group in groups:
            group_filters = self.filters.copy()
            group_filters['group'] = group
            views[group] = DatasetView(self.dataset, group_filters)

        return views

    def split_by_source(self) -> Dict[int, 'DatasetView']:
        """Split view by feature source."""
        sources = self.get_unique_sources()
        views = {}

        for source in sources:
            source_filters = self.filters.copy()
            source_filters['source_id'] = source
            views[source] = DatasetView(self.dataset, source_filters)

        return views

    def _to_3d_representation(self, features: np.ndarray, row_indices: np.ndarray) -> np.ndarray:
        """Convert features to 3D representation (samples, sources/processings, features)."""
        # Group by processing and stack
        indices = self._get_filtered_indices()
        processings = self.get_unique_processings()

        if len(processings) == 1:
            return features[np.newaxis, :, :] if features.ndim == 2 else features

        # Stack features by processing
        stacked_features = []
        for processing in processings:
            proc_mask = indices.filter(pl.col("processing") == processing)
            proc_rows = proc_mask.select("row").to_series().to_numpy()
            proc_features = self.dataset.get_features(proc_rows, concatenate=True)
            stacked_features.append(proc_features)

        return np.stack(stacked_features, axis=1)

    def _to_source_separate(self, features: Union[np.ndarray, List[np.ndarray]],
                           row_indices: np.ndarray) -> List[np.ndarray]:
        """Ensure features are returned as separate sources."""
        if isinstance(features, list):
            return features
        else:
            # If concatenated, we need to split back - this requires source info
            sources = self.get_unique_sources()
            if len(sources) == 1:
                return [features]
            else:
                # TODO: Implement source splitting based on feature ranges
                return [features]  # Fallback

    def copy_with_filters(self, additional_filters: Dict[str, Any]) -> 'DatasetView':
        """Create a new view with additional filters."""
        new_filters = self.filters.copy()
        new_filters.update(additional_filters)
        return DatasetView(self.dataset, new_filters)

    def __len__(self) -> int:
        """Number of samples in this view."""
        return len(self._get_filtered_indices())

    def __repr__(self) -> str:
        return f"DatasetView(filters={self.filters}, samples={len(self)})"


class DataSelector:
    """
    Helper class to determine the correct data scope for different operation types.

    This implements the scoping rules described in the requirements.
    """

    @staticmethod
    def get_training_scope(dataset: 'SpectraDataset',
                          context: 'PipelineContext') -> DatasetView:
        """Get the scope for training operations (fit operations)."""
        filters = context.current_filters.copy()
        filters['partition'] = 'train'
        return DatasetView(dataset, filters)

    @staticmethod
    def get_transform_scope(dataset: 'SpectraDataset',
                           context: 'PipelineContext') -> DatasetView:
        """Get the scope for transform operations (applied to all data)."""
        filters = context.current_filters.copy()
        # Transform operations apply to all partitions by default
        return DatasetView(dataset, filters)

    @staticmethod
    def get_prediction_scope(dataset: 'SpectraDataset',
                            context: 'PipelineContext',
                            partition: str = 'test') -> DatasetView:
        """Get the scope for prediction operations."""
        filters = context.current_filters.copy()
        filters['partition'] = partition
        return DatasetView(dataset, filters)

    @staticmethod
    def get_validation_scope(dataset: 'SpectraDataset',
                            context: 'PipelineContext',
                            fold: Optional[int] = None) -> DatasetView:
        """Get the scope for validation operations."""
        filters = context.current_filters.copy()
        filters['partition'] = 'train'  # Validation is typically from train partition

        if fold is not None:
            # TODO: Implement fold-based filtering
            pass

        return DatasetView(dataset, filters)

    @staticmethod
    def get_cluster_scope(dataset: 'SpectraDataset',
                         context: 'PipelineContext') -> DatasetView:
        """Get the scope for clustering operations."""
        filters = context.current_filters.copy()
        filters['partition'] = 'train'  # Clustering typically on train data
        return DatasetView(dataset, filters)
