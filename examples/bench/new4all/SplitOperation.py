"""
SplitOperation - Data splitting operations with various strategies
"""
import numpy as np
import polars as pl
from typing import Dict, List, Optional, Union, Tuple
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold, TimeSeriesSplit
from PipelineOperation import PipelineOperation
from SpectraDataset import SpectraDataset
from SpectraFeatures import SpectraFeatures
from PipelineContext import PipelineContext


class SplitOperation(PipelineOperation):
    """Operation for splitting data into different partitions"""

    def __init__(self,
                 split_strategy: str = "random",
                 split_ratios: Dict[str, float] = None,
                 stratified: bool = False,
                 random_state: int = 42,
                 **split_params):
        """
        Initialize split operation

        Parameters:
        -----------
        split_strategy : str
            Splitting strategy: "random", "stratified", "group", "time_series"
        split_ratios : dict
            Ratios for different partitions, e.g., {"train": 0.7, "val": 0.2, "test": 0.1}
        stratified : bool
            Whether to use stratified splitting
        random_state : int
            Random state for reproducibility
        **split_params : dict
            Additional parameters for splitting strategy
        """
        super().__init__()
        self.split_strategy = split_strategy
        self.split_ratios = split_ratios or {"train": 0.8, "test": 0.2}
        self.stratified = stratified
        self.random_state = random_state
        self.split_params = split_params

        # Validate split ratios
        if abs(sum(self.split_ratios.values()) - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

    def execute(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Execute split operation"""
        if not self.can_execute(dataset, context):
            raise ValueError("Cannot execute split - no data available")

        # Get data for splitting
        n_samples = self.get_n_samples(dataset)        # Get targets for stratification if needed
        y_for_split = None
        if self.stratified and hasattr(dataset, 'target_manager') and dataset.target_manager is not None:
            # Get all sample IDs from the dataset
            all_sample_ids = dataset.indices["sample"].to_list()
            if all_sample_ids:
                y_for_split = dataset.target_manager.get_targets(all_sample_ids)

        # Generate split indices
        split_indices = self.generate_split_indices(n_samples, y_for_split)

        # Apply splits to all data sources
        self.apply_splits_to_dataset(dataset, split_indices)        # Update context with split information
        context.data_splits = {
            'strategy': self.split_strategy,
            'ratios': self.split_ratios,
            'indices': split_indices,
            'n_samples': {partition: len(indices) for partition, indices in split_indices.items()}
        }

        print(f"Data split completed using {self.split_strategy} strategy:")
        for partition, indices in split_indices.items():
            print(f"  {partition}: {len(indices)} samples")

    def can_execute(self, dataset: SpectraDataset, context: PipelineContext) -> bool:
        """Check if split can be executed"""
        return len(dataset) > 1

    def get_name(self) -> str:
        """Get operation name"""
        return f"SplitOperation({self.split_strategy})"

    def get_n_samples(self, dataset: SpectraDataset) -> int:
        """Get number of samples in dataset"""
        return len(dataset)

    def generate_split_indices(self, n_samples: int, y: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Generate split indices based on strategy"""

        if self.split_strategy == "random":
            return self.random_split(n_samples, y)
        elif self.split_strategy == "stratified":
            if y is None:
                raise ValueError("Stratified split requires target labels")
            return self.stratified_split(n_samples, y)
        elif self.split_strategy == "group":
            groups = self.split_params.get('groups')
            if groups is None:
                raise ValueError("Group split requires 'groups' parameter")
            return self.group_split(n_samples, groups)
        elif self.split_strategy == "time_series":
            return self.time_series_split(n_samples)
        else:
            raise ValueError(f"Unknown split strategy: {self.split_strategy}")

    def random_split(self, n_samples: int, y: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Random splitting"""
        indices = np.arange(n_samples)
        split_indices = {}

        remaining_indices = indices.copy()
        remaining_ratio = 1.0

        # Sort partitions by ratio (largest first)
        sorted_partitions = sorted(self.split_ratios.items(), key=lambda x: x[1], reverse=True)

        for i, (partition, ratio) in enumerate(sorted_partitions):
            if i == len(sorted_partitions) - 1:
                # Last partition gets all remaining indices
                split_indices[partition] = remaining_indices
            else:
                # Calculate size for this partition
                current_ratio = ratio / remaining_ratio
                split_size = int(len(remaining_indices) * current_ratio)

                if self.stratified and y is not None:
                    # Use stratified split
                    selected_indices, remaining_indices = train_test_split(
                        remaining_indices,
                        test_size=len(remaining_indices) - split_size,
                        stratify=y[remaining_indices],
                        random_state=self.random_state
                    )
                else:
                    # Use random split
                    selected_indices, remaining_indices = train_test_split(
                        remaining_indices,
                        test_size=len(remaining_indices) - split_size,
                        random_state=self.random_state
                    )

                split_indices[partition] = selected_indices
                remaining_ratio -= ratio

        return split_indices

    def stratified_split(self, n_samples: int, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Stratified splitting"""
        return self.random_split(n_samples, y)  # Uses stratification in random_split

    def group_split(self, n_samples: int, groups: np.ndarray) -> Dict[str, np.ndarray]:
        """Group-based splitting"""
        unique_groups = np.unique(groups)
        np.random.seed(self.random_state)
        np.random.shuffle(unique_groups)

        split_indices = {}
        remaining_groups = unique_groups.copy()
        remaining_ratio = 1.0

        # Sort partitions by ratio (largest first)
        sorted_partitions = sorted(self.split_ratios.items(), key=lambda x: x[1], reverse=True)

        for i, (partition, ratio) in enumerate(sorted_partitions):
            if i == len(sorted_partitions) - 1:
                # Last partition gets all remaining groups
                selected_groups = remaining_groups
            else:
                # Calculate number of groups for this partition
                current_ratio = ratio / remaining_ratio
                n_groups = max(1, int(len(remaining_groups) * current_ratio))

                selected_groups = remaining_groups[:n_groups]
                remaining_groups = remaining_groups[n_groups:]
                remaining_ratio -= ratio

            # Get sample indices for selected groups
            group_mask = np.isin(groups, selected_groups)
            split_indices[partition] = np.where(group_mask)[0]

        return split_indices

    def time_series_split(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Time series splitting (chronological order)"""
        indices = np.arange(n_samples)
        split_indices = {}

        current_start = 0

        # Sort partitions to maintain chronological order
        # Typically: train -> val -> test
        partition_order = ["train", "val", "test"]

        for partition in partition_order:
            if partition in self.split_ratios:
                ratio = self.split_ratios[partition]
                split_size = int(n_samples * ratio)

                split_end = min(current_start + split_size, n_samples)
                split_indices[partition] = indices[current_start:split_end]
                current_start = split_end

        # Handle any remaining partitions
        for partition, ratio in self.split_ratios.items():
            if partition not in split_indices and current_start < n_samples:
                split_size = int(n_samples * ratio)
                split_end = min(current_start + split_size, n_samples)
                split_indices[partition] = indices[current_start:split_end]
                current_start = split_end

        return split_indices

    def apply_splits_to_dataset(self, dataset: SpectraDataset, split_indices: Dict[str, np.ndarray]) -> None:
        """Apply splits to dataset by updating partition labels in indices"""

        if dataset.indices is None or len(dataset.indices) == 0:
            return
              # Update partition column in dataset.indices based on split_indices
        for partition_name, sample_indices in split_indices.items():
            # Update the partition field for these specific sample indices
            mask = dataset.indices['sample'].is_in(sample_indices)
            dataset.indices = dataset.indices.with_columns(
                pl.when(mask).then(pl.lit(partition_name)).otherwise(pl.col('partition')).alias('partition')
            )
          # Split targets if available
        if hasattr(dataset, 'target_manager') and dataset.target_manager is not None:
            self.split_targets(dataset, split_indices)

    def split_targets(self, dataset: SpectraDataset, split_indices: Dict[str, np.ndarray]) -> None:
        """Split targets according to data splits - simplified approach"""
        # For now, we'll skip target splitting since the TargetManager doesn't provide
        # the necessary methods (get_all_targets, add_target, remove_target)
        # In a full implementation, this would need to be coordinated with the
        # TargetManager's design to properly handle split targets
        pass


class SplitStrategy:
    """Strategy for common splitting scenarios"""

    @classmethod
    def train_val_test(cls, train_ratio: float = 0.7, val_ratio: float = 0.2,
                      test_ratio: float = 0.1, stratified: bool = False) -> SplitOperation:
        """Create train/validation/test split"""
        return SplitOperation(
            split_strategy="stratified" if stratified else "random",
            split_ratios={"train": train_ratio, "val": val_ratio, "test": test_ratio},
            stratified=stratified
        )

    @classmethod
    def train_test(cls, train_ratio: float = 0.8, stratified: bool = False) -> SplitOperation:
        """Create simple train/test split"""
        return SplitOperation(
            split_strategy="stratified" if stratified else "random",
            split_ratios={"train": train_ratio, "test": 1.0 - train_ratio},
            stratified=stratified
        )

    @classmethod
    def time_series_split(cls, train_ratio: float = 0.7, val_ratio: float = 0.2,
                         test_ratio: float = 0.1) -> SplitOperation:
        """Create time series split maintaining chronological order"""
        return SplitOperation(
            split_strategy="time_series",
            split_ratios={"train": train_ratio, "val": val_ratio, "test": test_ratio}
        )

    @classmethod
    def group_split(cls, groups: np.ndarray, train_ratio: float = 0.7,
                   val_ratio: float = 0.2, test_ratio: float = 0.1) -> SplitOperation:
        """Create group-based split"""
        return SplitOperation(
            split_strategy="group",
            split_ratios={"train": train_ratio, "val": val_ratio, "test": test_ratio},
            groups=groups
        )
