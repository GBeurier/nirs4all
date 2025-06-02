"""
MergeSourcesOperation - Combines multiple data sources into a single feature matrix
"""
import numpy as np
from typing import Dict, List
from PipelineOperation import PipelineOperation
from SpectraDataset import SpectraDataset
from PipelineContext import PipelineContext


class MergeSourcesOperation(PipelineOperation):
    """Operation for merging multiple data sources into a single matrix"""

    def __init__(self, merge_strategy: str = "concatenate", axis: int = 1):
        """
        Initialize merge sources operation

        Parameters:
        -----------
        merge_strategy : str
            Strategy for merging sources: "concatenate", "average", "weighted"
        axis : int
            Axis along which to merge (1 for features)
        """
        super().__init__()
        self.merge_strategy = merge_strategy
        self.axis = axis

    def execute(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Execute the merge operation"""
        if not self.can_execute(dataset, context):
            raise ValueError("Cannot execute merge operation - insufficient sources")

        if dataset.features is None:
            raise ValueError("Dataset has no features to merge")

        # Get number of sources
        n_sources = len(dataset.features.sources)

        if n_sources <= 1:
            print(f"Only {n_sources} source(s) available, no merge needed")
            return

        # Get all source data
        source_data = []
        for i in range(n_sources):
            source_data.append(dataset.features.get_source(i))

        # Perform merge based on strategy
        if self.merge_strategy == "concatenate":
            X_merged = self.merge_horizontal_sources(source_data)
        elif self.merge_strategy == "average":
            X_merged = self.merge_average_sources(source_data)
        elif self.merge_strategy == "weighted":
            X_merged = self.merge_weighted_sources(source_data)
        else:
            raise ValueError(f"Unknown merge strategy: {self.merge_strategy}")

        # Replace all sources with the merged result
        dataset.features.replace_with_merged(X_merged)

        print(f"Merged {n_sources} sources using {self.merge_strategy} strategy - replaced with single merged source")

    def can_execute(self, dataset: SpectraDataset, context: PipelineContext) -> bool:
        """Check if merge operation can be executed"""
        return len(dataset) > 0 and dataset.features is not None and len(dataset.features.sources) > 1

    def get_name(self) -> str:
        """Get operation name"""
        return f"MergeSourcesOperation({self.merge_strategy})"

    def merge_horizontal(self, X_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate sources horizontally (along feature axis)"""
        # Ensure all arrays have same number of samples
        n_samples = None
        arrays_to_merge = []

        for source_name, X_source in X_dict.items():
            if n_samples is None:
                n_samples = X_source.shape[0]
            elif X_source.shape[0] != n_samples:
                raise ValueError(f"Source '{source_name}' has {X_source.shape[0]} samples, expected {n_samples}")
            arrays_to_merge.append(X_source)

        # Concatenate along feature axis
        return np.concatenate(arrays_to_merge, axis=self.axis)

    def merge_horizontal_sources(self, source_data: List[np.ndarray]) -> np.ndarray:
        """Concatenate sources horizontally (along feature axis)"""
        # Ensure all arrays have same number of samples
        n_samples = source_data[0].shape[0]
        for i, source in enumerate(source_data[1:], 1):
            if source.shape[0] != n_samples:
                raise ValueError(f"Source {i} has {source.shape[0]} samples, expected {n_samples}")

        # Concatenate along feature axis
        return np.concatenate(source_data, axis=self.axis)

    def merge_average_sources(self, source_data: List[np.ndarray]) -> np.ndarray:
        """Average sources element-wise (requires same shape)"""
        # Check that all arrays have the same shape
        reference_shape = source_data[0].shape
        for i, arr in enumerate(source_data[1:], 1):
            if arr.shape != reference_shape:
                raise ValueError(f"Array {i} has shape {arr.shape}, expected {reference_shape}")

        # Compute element-wise average
        return np.mean(np.stack(source_data, axis=0), axis=0)

    def merge_weighted_sources(self, source_data: List[np.ndarray], weights: List[float] = None) -> np.ndarray:
        """Weighted average of sources (requires same shape)"""
        # Default to equal weights if not provided
        if weights is None:
            weights = [1.0] * len(source_data)

        # Check that all arrays have the same shape
        reference_shape = source_data[0].shape
        for i, arr in enumerate(source_data[1:], 1):
            if arr.shape != reference_shape:
                raise ValueError(f"Array {i} has shape {arr.shape}, expected {reference_shape}")

        # Normalize weights
        weight_sum = sum(weights)
        normalized_weights = [w / weight_sum for w in weights]

        # Compute weighted average
        weighted_arrays = [w * arr for w, arr in zip(normalized_weights, source_data)]
        return np.sum(np.stack(weighted_arrays, axis=0), axis=0)

    def merge_average(self, X_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Average sources element-wise (requires same shape)"""
        arrays = list(X_dict.values())

        # Check that all arrays have the same shape
        reference_shape = arrays[0].shape
        for i, arr in enumerate(arrays[1:], 1):
            if arr.shape != reference_shape:
                raise ValueError(f"Array {i} has shape {arr.shape}, expected {reference_shape}")

        # Compute element-wise average
        return np.mean(np.stack(arrays, axis=0), axis=0)

    def merge_weighted(self, X_dict: Dict[str, np.ndarray], weights: Dict[str, float] = None) -> np.ndarray:
        """Weighted average of sources (requires same shape)"""
        arrays = list(X_dict.values())
        source_names = list(X_dict.keys())

        # Default to equal weights if not provided
        if weights is None:
            weights = {name: 1.0 for name in source_names}

        # Check that all arrays have the same shape
        reference_shape = arrays[0].shape
        for i, arr in enumerate(arrays[1:], 1):
            if arr.shape != reference_shape:
                raise ValueError(f"Array {i} has shape {arr.shape}, expected {reference_shape}")

        # Normalize weights
        weight_sum = sum(weights.get(name, 1.0) for name in source_names)
        normalized_weights = [weights.get(name, 1.0) / weight_sum for name in source_names]

        # Compute weighted average
        weighted_arrays = [w * arr for w, arr in zip(normalized_weights, arrays)]
        return np.sum(np.stack(weighted_arrays, axis=0), axis=0)

    def update_feature_names(self, dataset: SpectraDataset) -> None:
        """Update feature names after merging (placeholder for future feature metadata)"""
        # This would update feature metadata if we had it in the dataset
        # For now, just log the action
        print("Feature names updated after merge operation")
