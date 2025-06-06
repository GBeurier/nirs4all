"""
Centroid Propagation Operation - Applies actions to centroids and propagates to group members

This operation:
1. Takes an operation to apply (e.g., split, folds)
2. Applies it only to designated centroids
3. Propagates the results to all samples in the corresponding groups
4. Enables group-based decision making
"""
import numpy as np
from typing import Optional, List, Dict, Any

from nirs4all.pipeline.pipeline_operation import PipelineOperation
from nirs4all.spectra.spectra_dataset import SpectraDataset
from nirs4all.pipeline.pipeline_context import PipelineContext


class OperationCentroidPropagation(PipelineOperation):
    """Pipeline operation that applies actions to centroids and propagates to groups"""

    def __init__(self,
                 target_operation: PipelineOperation,
                 centroid_mapping: Dict[int, str] = None,
                 auto_detect_centroids: bool = True,
                 propagate_partitions: bool = True,
                 propagate_folds: bool = True,
                 operation_name: Optional[str] = None):
        """
        Initialize centroid propagation operation

        Args:
            target_operation: operation to apply to centroids (e.g., OperationSplit)
            centroid_mapping: dict of {group_id: centroid_sample_id}
            auto_detect_centroids: whether to auto-detect centroids from previous clustering
            propagate_partitions: whether to propagate partition assignments
            propagate_folds: whether to propagate fold assignments
            operation_name: custom name for this operation
        """
        self.target_operation = target_operation
        self.centroid_mapping = centroid_mapping or {}
        self.auto_detect_centroids = auto_detect_centroids
        self.propagate_partitions = propagate_partitions
        self.propagate_folds = propagate_folds
        self.operation_name = operation_name

        # Propagation results
        self.propagated_partitions = {}  # {sample_id: partition}
        self.propagated_folds = {}       # {sample_id: fold_id}

    def execute(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Execute centroid-based action propagation"""
        print(f"🔄 Executing {self.get_name()}")

        # Step 1: Detect or use provided centroids
        centroids = self._get_centroids(dataset, context)
        if not centroids:
            raise ValueError("No centroids found for propagation. Run clustering first or provide centroid_mapping.")

        print(f"  🎯 Found {len(centroids)} centroids: {list(centroids.values())}")

        # Step 2: Create centroid-only dataset view
        centroid_sample_ids = list(centroids.values())

        # Save original state
        original_partitions = {}
        original_folds = {}
        all_samples = dataset.select(**context.current_filters)

        for sample_id in all_samples.sample_ids:
            original_partitions[sample_id] = dataset.get_sample_partition(sample_id)
            try:
                original_folds[sample_id] = dataset.get_sample_fold(sample_id)
            except (AttributeError, KeyError):
                original_folds[sample_id] = -1  # No fold assigned

        # Temporarily mark only centroids as the target partition for the operation
        centroid_partition = getattr(self.target_operation, 'source_partition', 'train')
        if hasattr(self.target_operation, 'target_partition'):
            centroid_partition = self.target_operation.target_partition

        # Reset all samples to a different partition, then set centroids to target partition
        for sample_id in all_samples.sample_ids:
            if sample_id in centroid_sample_ids:
                dataset.update_partitions([sample_id], centroid_partition)
            else:
                dataset.update_partitions([sample_id], "temp_non_centroid")

        print(f"  🎯 Applying {self.target_operation.get_name()} to centroids only")

        # Step 3: Apply operation to centroids only
        try:
            self.target_operation.execute(dataset, context)
        except Exception as e:
            # Restore original state on error
            self._restore_state(dataset, original_partitions, original_folds)
            raise ValueError(f"Failed to apply operation to centroids: {e}")

        # Step 4: Collect centroid results
        centroid_results = {}
        for group_id, centroid_id in centroids.items():
            try:
                new_partition = dataset.get_sample_partition(centroid_id)
                new_fold = dataset.get_sample_fold(centroid_id)
            except (AttributeError, KeyError):
                new_partition = original_partitions.get(centroid_id, centroid_partition)
                new_fold = original_folds.get(centroid_id, -1)

            centroid_results[group_id] = {
                'partition': new_partition,
                'fold': new_fold
            }

        print(f"  📋 Centroid results: {centroid_results}")

        # Step 5: Propagate results to all group members
        self.propagated_partitions = {}
        self.propagated_folds = {}

        for sample_id in all_samples.sample_ids:
            try:
                group_id = dataset.get_sample_group(sample_id)
            except (AttributeError, KeyError):
                group_id = None

            if group_id is not None and group_id in centroid_results:
                # Propagate from centroid
                if self.propagate_partitions:
                    new_partition = centroid_results[group_id]['partition']
                    dataset.update_partitions([sample_id], new_partition)
                    self.propagated_partitions[sample_id] = new_partition

                if self.propagate_folds:
                    new_fold = centroid_results[group_id]['fold']
                    if new_fold != -1:
                        dataset.update_folds([sample_id], new_fold)
                        self.propagated_folds[sample_id] = new_fold
            else:
                # Restore original state for samples without group/centroid
                if sample_id in original_partitions:
                    dataset.update_partitions([sample_id], original_partitions[sample_id])
                if sample_id in original_folds and original_folds[sample_id] != -1:
                    dataset.update_folds([sample_id], original_folds[sample_id])

        # Report propagation results
        if self.propagated_partitions:
            partition_counts = {}
            for partition in self.propagated_partitions.values():
                partition_counts[partition] = partition_counts.get(partition, 0) + 1
            print(f"  ✅ Propagated partitions: {partition_counts}")

        if self.propagated_folds:
            fold_counts = {}
            for fold in self.propagated_folds.values():
                fold_counts[fold] = fold_counts.get(fold, 0) + 1
            print(f"  ✅ Propagated folds: {fold_counts}")

    def _get_centroids(self, dataset: SpectraDataset, context: PipelineContext) -> Dict[int, str]:
        """Get centroid mapping from previous operations or provided mapping"""
        if self.centroid_mapping:
            return self.centroid_mapping

        if not self.auto_detect_centroids:
            return {}

        # Try to detect centroids from previous clustering operations
        # Look for clustering operations in context history
        if hasattr(context, 'operation_history'):
            for operation in reversed(context.operation_history):
                if hasattr(operation, 'get_centroids'):
                    centroids = operation.get_centroids()
                    if centroids:
                        return centroids

        # Alternative: find centroids by looking for samples with special markers
        # This would require a convention in the dataset structure
        return {}

    def _restore_state(self, dataset: SpectraDataset,
                      original_partitions: Dict[str, str],
                      original_folds: Dict[str, int]) -> None:
        """Restore original dataset state"""
        for sample_id, partition in original_partitions.items():
            dataset.update_partitions([sample_id], partition)

        for sample_id, fold_id in original_folds.items():
            if fold_id != -1:
                dataset.update_folds([sample_id], fold_id)

    def get_propagated_partitions(self) -> Dict[str, str]:
        """Get mapping of sample_id -> propagated_partition"""
        return self.propagated_partitions.copy()

    def get_propagated_folds(self) -> Dict[str, int]:
        """Get mapping of sample_id -> propagated_fold"""
        return self.propagated_folds.copy()

    def get_name(self) -> str:
        """Get operation name"""
        if self.operation_name:
            return self.operation_name
        return f"CentroidProp({self.target_operation.get_name()})"

    def can_execute(self, dataset: SpectraDataset, context: PipelineContext) -> bool:
        """Check if centroid propagation can be executed"""
        centroids = self._get_centroids(dataset, context)
        return len(centroids) > 0 and self.target_operation.can_execute(dataset, context)
