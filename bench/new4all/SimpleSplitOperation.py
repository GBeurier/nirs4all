"""
Simple split operations (train/test, train/val/test) without cross-validation
"""
import numpy as np
import polars as pl
from typing import Dict, List, Optional, Union
from sklearn.model_selection import train_test_split

try:
    from BaseSplitOperation import BaseSplitOperation
    from SpectraDataset import SpectraDataset
    from PipelineContext import PipelineContext
except ImportError:
    from BaseSplitOperation import BaseSplitOperation
    from SpectraDataset import SpectraDataset
    from PipelineContext import PipelineContext


class SimpleSplitOperation(BaseSplitOperation):
    """Operation for simple data splitting (no cross-validation)."""

    def __init__(self,
                 split_strategy: str = "random",
                 split_ratios: Dict[str, float] = None,
                 stratified: bool = False,
                 random_state: int = 42,
                 **split_params):
        """
        Initialize simple split operation

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
        super().__init__(random_state, **split_params)
        self.split_strategy = split_strategy
        self.split_ratios = split_ratios or {"train": 0.8, "test": 0.2}
        self.stratified = stratified

        # Validate split ratios
        if abs(sum(self.split_ratios.values()) - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

    def execute(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Execute simple split operation."""
        if not self.can_execute(dataset, context):
            raise ValueError("Cannot execute split - no data available")

        # Get data for splitting
        n_samples = self.get_n_samples(dataset)

        # Get targets for stratification if needed
        y_for_split = None
        if self.stratified and hasattr(dataset, 'target_manager') and dataset.target_manager is not None:
            # Get all sample IDs from the dataset
            all_sample_ids = dataset.indices["sample"].to_list()
            if all_sample_ids:
                y_for_split = dataset.target_manager.get_targets(all_sample_ids)

        # Generate split indices
        split_indices = self.generate_split_indices(n_samples, y_for_split)

        # Apply splits to all data sources
        self.apply_splits_to_dataset(dataset, split_indices)

        # Update context with split information
        context.data_splits = {
            'strategy': self.split_strategy,
            'ratios': self.split_ratios,
            'indices': split_indices,
            'n_samples': {partition: len(indices) for partition, indices in split_indices.items()},
            'type': 'simple'
        }

        print(f"Data split completed using {self.split_strategy} strategy:")
        for partition, indices in split_indices.items():
            print(f"  {partition}: {len(indices)} samples")

    def get_name(self) -> str:
        """Get operation name."""
        return f"SimpleSplit({self.split_strategy})"

    def generate_split_indices(self, n_samples: int, y: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Generate split indices based on strategy."""

        if self.split_strategy == "random":
            return self.random_split(n_samples, y)
        elif self.split_strategy == "stratified":
            if y is None:
                raise ValueError("Stratified split requires target labels")
            return self.stratified_split(n_samples, y)
        elif self.split_strategy == "group":
            groups = self.split_params.get("groups")
            if groups is None:
                raise ValueError("Group split requires 'groups' parameter")
            return self.group_split(n_samples, groups)
        elif self.split_strategy == "time_series":
            return self.time_series_split(n_samples)
        else:
            raise ValueError(f"Unknown split strategy: {self.split_strategy}")

    def random_split(self, n_samples: int, y: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Random splitting with optional stratification."""
        indices = np.arange(n_samples)
        split_indices = {}

        remaining_indices = indices.copy()
        remaining_ratio = 1.0

        # Sort partitions by ratio (largest first)
        sorted_partitions = sorted(self.split_ratios.items(), key=lambda x: x[1], reverse=True)

        for i, (partition, ratio) in enumerate(sorted_partitions):
            if i == len(sorted_partitions) - 1:
                # Last partition gets all remaining
                split_indices[partition] = remaining_indices
            else:
                # Calculate size for this partition
                size = int(len(indices) * ratio)

                if y is not None and self.stratified:
                    # Stratified split
                    remaining_y = y[remaining_indices]
                    current_ratio = ratio / remaining_ratio

                    selected_indices, remaining_indices = train_test_split(
                        remaining_indices,
                        test_size=1-current_ratio,
                        stratify=remaining_y,
                        random_state=self.random_state
                    )
                else:
                    # Random split
                    np.random.seed(self.random_state)
                    selected_indices = np.random.choice(
                        remaining_indices, size=size, replace=False
                    )
                    remaining_indices = np.setdiff1d(remaining_indices, selected_indices)

                split_indices[partition] = selected_indices
                remaining_ratio -= ratio

        return split_indices

    def stratified_split(self, n_samples: int, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Stratified splitting."""
        return self.random_split(n_samples, y)  # Uses stratification in random_split

    def group_split(self, n_samples: int, groups: np.ndarray) -> Dict[str, np.ndarray]:
        """Group-based splitting."""
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
                partition_groups = remaining_groups
            else:
                # Calculate number of groups for this partition
                n_groups = int(len(unique_groups) * ratio)
                partition_groups = remaining_groups[:n_groups]
                remaining_groups = remaining_groups[n_groups:]

            # Get sample indices for these groups
            sample_indices = np.where(np.isin(groups, partition_groups))[0]
            split_indices[partition] = sample_indices
            remaining_ratio -= ratio

        return split_indices

    def time_series_split(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Time series splitting (chronological order)."""
        indices = np.arange(n_samples)
        split_indices = {}

        current_start = 0

        # Sort partitions to maintain chronological order
        # Typically: train -> val -> test
        partition_order = ["train", "val", "test"]

        # Handle only partitions that exist in split_ratios
        existing_partitions = [p for p in partition_order if p in self.split_ratios]

        for partition in existing_partitions[:-1]:  # All but last
            ratio = self.split_ratios[partition]
            size = int(n_samples * ratio)
            split_indices[partition] = indices[current_start:current_start + size]
            current_start += size

        # Last partition gets remaining samples
        if existing_partitions:
            last_partition = existing_partitions[-1]
            split_indices[last_partition] = indices[current_start:]

        # Handle any remaining partitions not in standard order
        for partition, ratio in self.split_ratios.items():
            if partition not in split_indices:
                # This shouldn't happen with proper time series splitting
                size = int(n_samples * ratio)
                split_indices[partition] = indices[current_start:current_start + size]
                current_start += size

        return split_indices

    def apply_splits_to_dataset(self, dataset: SpectraDataset, split_indices: Dict[str, np.ndarray]) -> None:
        """Apply splits to dataset by updating partition labels in indices."""

        if dataset.indices is None or len(dataset.indices) == 0:
            raise ValueError("Cannot split empty dataset")

        # Create a copy of indices to work with
        updated_indices = dataset.indices.clone()

        for partition_name, row_indices in split_indices.items():
            # Update partition labels for the specified row indices
            mask = pl.col("row").is_in(row_indices.tolist())
            updated_indices = updated_indices.with_columns([
                pl.when(mask).then(pl.lit(partition_name)).otherwise(pl.col("partition")).alias("partition")
            ])

        # Replace the dataset indices
        dataset.indices = updated_indices


class SplitStrategy:
    """Strategy factory for common splitting scenarios."""

    @classmethod
    def train_val_test(cls, train_ratio: float = 0.7, val_ratio: float = 0.2,
                      test_ratio: float = 0.1, stratified: bool = False) -> SimpleSplitOperation:
        """Create train/validation/test split."""
        return SimpleSplitOperation(
            split_strategy="stratified" if stratified else "random",
            split_ratios={"train": train_ratio, "val": val_ratio, "test": test_ratio},
            stratified=stratified
        )

    @classmethod
    def train_test(cls, train_ratio: float = 0.8, stratified: bool = False) -> SimpleSplitOperation:
        """Create train/test split."""
        test_ratio = 1.0 - train_ratio
        return SimpleSplitOperation(
            split_strategy="stratified" if stratified else "random",
            split_ratios={"train": train_ratio, "test": test_ratio},
            stratified=stratified
        )

    @classmethod
    def time_series_split(cls, train_ratio: float = 0.7, val_ratio: float = 0.2,
                         test_ratio: float = 0.1) -> SimpleSplitOperation:
        """Create time series split."""
        return SimpleSplitOperation(
            split_strategy="time_series",
            split_ratios={"train": train_ratio, "val": val_ratio, "test": test_ratio}
        )

    @classmethod
    def group_split(cls, groups: np.ndarray, train_ratio: float = 0.7,
                   val_ratio: float = 0.2, test_ratio: float = 0.1) -> SimpleSplitOperation:
        """Create group-based split."""
        return SimpleSplitOperation(
            split_strategy="group",
            split_ratios={"train": train_ratio, "val": val_ratio, "test": test_ratio},
            groups=groups
        )
