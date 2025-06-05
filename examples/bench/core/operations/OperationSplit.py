"""
Split Operation - Splits data into different partitions (train/test/validation)

This operation:
1. Splits specified partition into multiple new partitions
2. Updates partition labels in dataset
3. Supports stratified and group-aware splitting
4. Can work with existing folds or create new splits
"""
import numpy as np
from typing import Optional, List, Dict, Union
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GroupShuffleSplit

from PipelineOperation import PipelineOperation
from SpectraDataset import SpectraDataset
from PipelineContext import PipelineContext


class OperationSplit(PipelineOperation):
    """Pipeline operation that splits data into different partitions"""

    def __init__(self,
                 split_ratios: Dict[str, float] = None,
                 source_partition: str = "train",
                 stratified: bool = False,
                 use_groups: bool = False,
                 random_state: int = 42,
                 preserve_folds: bool = False,
                 operation_name: Optional[str] = None):
        """
        Initialize split operation

        Args:
            split_ratios: dictionary of partition_name -> ratio (e.g., {"train": 0.8, "test": 0.2})
            source_partition: partition to split (default: "train")
            stratified: whether to use stratified splitting
            use_groups: whether to use group-aware splitting
            random_state: random state for reproducibility
            preserve_folds: whether to preserve existing fold assignments
            operation_name: custom name for this operation
        """
        self.split_ratios = split_ratios or {"train": 0.8, "test": 0.2}
        self.source_partition = source_partition
        self.stratified = stratified
        self.use_groups = use_groups
        self.random_state = random_state
        self.preserve_folds = preserve_folds
        self.operation_name = operation_name
        self.split_assignments = {}  # {sample_id: new_partition}

        # Validate split ratios
        if abs(sum(self.split_ratios.values()) - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

    def execute(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Execute split: divide source partition into new partitions"""
        print(f"🔄 Executing {self.get_name()}")

        # Get source partition data
        source_view = dataset.select(partition=self.source_partition, **context.current_filters)
        if len(source_view) == 0:
            raise ValueError(f"No data found in partition '{self.source_partition}' for splitting")

        sample_ids = source_view.sample_ids
        n_samples = len(sample_ids)

        print(f"  📊 Splitting {n_samples} samples from '{self.source_partition}' partition")
        print(f"  📋 Split ratios: {self.split_ratios}")
        print(f"  🔧 Stratified: {self.stratified}, Use groups: {self.use_groups}")

        # Prepare data for splitting
        indices = np.arange(n_samples)

        # Get targets and groups if needed
        y = None
        groups = None

        if self.stratified:
            try:
                y = source_view.get_targets("auto")
                print(f"    📈 Using targets for stratified splitting")
            except Exception:
                print(f"    ⚠️ No targets found, falling back to random splitting")
                self.stratified = False

        if self.use_groups:
            try:
                groups = source_view.group_ids
                if groups is not None and len(set(groups)) > 1:
                    print(f"    👥 Using {len(set(groups))} groups for group-aware splitting")
                else:
                    print(f"    ⚠️ No groups found, falling back to random splitting")
                    self.use_groups = False
                    groups = None
            except Exception:
                print(f"    ⚠️ Groups not available, falling back to random splitting")
                self.use_groups = False
                groups = None

        # Perform splitting
        partition_assignments = self._perform_split(indices, y, groups)

        # Apply splits to dataset
        self.split_assignments = {}
        for partition_name, sample_indices in partition_assignments.items():
            for idx in sample_indices:
                sample_id = sample_ids[idx]
                self.split_assignments[sample_id] = partition_name

                # Update dataset partition
                dataset.update_partitions([sample_id], partition_name)

        # Report split results
        print(f"  ✅ Split completed:")
        for partition_name, expected_ratio in self.split_ratios.items():
            actual_count = len([s for s in self.split_assignments.values() if s == partition_name])
            actual_ratio = actual_count / n_samples
            print(f"    {partition_name}: {actual_count} samples ({actual_ratio:.3f}, expected {expected_ratio:.3f})")

    def _perform_split(self, indices: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None) -> Dict[str, List[int]]:
        """Perform the actual splitting logic"""
        n_samples = len(indices)
        partition_names = list(self.split_ratios.keys())

        if len(partition_names) == 2:
            # Binary split
            partition1, partition2 = partition_names
            ratio1 = self.split_ratios[partition1]

            if self.use_groups and groups is not None:
                # Group-aware splitting
                splitter = GroupShuffleSplit(n_splits=1, test_size=(1-ratio1), random_state=self.random_state)
                train_idx, test_idx = next(splitter.split(indices, y, groups))
            elif self.stratified and y is not None:
                # Stratified splitting
                train_idx, test_idx = train_test_split(
                    indices, test_size=(1-ratio1), stratify=y, random_state=self.random_state
                )
            else:
                # Random splitting
                train_idx, test_idx = train_test_split(
                    indices, test_size=(1-ratio1), random_state=self.random_state
                )

            return {partition1: train_idx.tolist(), partition2: test_idx.tolist()}

        else:
            # Multi-way split (sequential binary splits)
            remaining_indices = indices.copy()
            remaining_y = y.copy() if y is not None else None
            remaining_groups = groups.copy() if groups is not None else None
            result = {}

            for i, partition_name in enumerate(partition_names[:-1]):
                ratio = self.split_ratios[partition_name]
                # Adjust ratio for remaining samples
                remaining_ratio = ratio / sum(self.split_ratios[p] for p in partition_names[i:])

                if len(remaining_indices) == 0:
                    result[partition_name] = []
                    continue

                if len(remaining_indices) == 1:
                    result[partition_name] = remaining_indices.tolist()
                    remaining_indices = np.array([])
                    continue

                # Perform split
                if self.use_groups and remaining_groups is not None:
                    splitter = GroupShuffleSplit(n_splits=1, test_size=(1-remaining_ratio), random_state=self.random_state)
                    current_idx, remaining_idx = next(splitter.split(remaining_indices, remaining_y, remaining_groups))
                elif self.stratified and remaining_y is not None:
                    current_idx, remaining_idx = train_test_split(
                        np.arange(len(remaining_indices)), test_size=(1-remaining_ratio),
                        stratify=remaining_y, random_state=self.random_state
                    )
                else:
                    current_idx, remaining_idx = train_test_split(
                        np.arange(len(remaining_indices)), test_size=(1-remaining_ratio),
                        random_state=self.random_state
                    )

                # Store current partition samples
                result[partition_name] = remaining_indices[current_idx].tolist()

                # Update remaining samples
                remaining_indices = remaining_indices[remaining_idx]
                if remaining_y is not None:
                    remaining_y = remaining_y[remaining_idx]
                if remaining_groups is not None:
                    remaining_groups = remaining_groups[remaining_idx]

            # Assign remaining samples to last partition
            result[partition_names[-1]] = remaining_indices.tolist()

            return result

    def get_split_assignments(self) -> Dict[str, str]:
        """Get mapping of sample_id -> new_partition"""
        return self.split_assignments.copy()

    def get_partition_samples(self, partition: str) -> List[str]:
        """Get list of sample IDs in specific partition after split"""
        return [sample_id for sample_id, part in self.split_assignments.items() if part == partition]

    def get_name(self) -> str:
        """Get operation name"""
        if self.operation_name:
            return self.operation_name

        strategy_parts = []
        if self.stratified:
            strategy_parts.append("Stratified")
        if self.use_groups:
            strategy_parts.append("Group")

        strategy_str = "-".join(strategy_parts) if strategy_parts else "Random"
        ratios_str = "-".join([f"{k}:{v:.1f}" for k, v in self.split_ratios.items()])

        return f"Split({strategy_str}[{ratios_str}])"

    def can_execute(self, dataset: SpectraDataset, context: PipelineContext) -> bool:
        """Check if split can be executed"""
        source_view = dataset.select(partition=self.source_partition, **context.current_filters)
        return len(source_view) > 1  # Need at least 2 samples to split
