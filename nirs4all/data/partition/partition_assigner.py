"""
Partition assigner for dataset configuration.

This module provides flexible partition assignment for DataFrames, supporting
multiple assignment methods including static, column-based, percentage-based,
and index-based partitions.

Example:
    >>> assigner = PartitionAssigner()
    >>> # Static partition
    >>> result = assigner.assign(df, partition="train")
    >>> # Column-based partition
    >>> result = assigner.assign(df, {
    ...     "column": "split",
    ...     "train_values": ["train", "training"],
    ...     "test_values": ["test", "validation"]
    ... })
    >>> # Percentage-based partition
    >>> result = assigner.assign(df, {
    ...     "train": "80%",
    ...     "test": "20%",
    ...     "shuffle": True,
    ...     "random_state": 42
    ... })
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd


class PartitionError(Exception):
    """Raised when partition assignment fails."""
    pass


# Type alias for partition specification
PartitionSpec = Union[
    str,                     # Static partition: "train", "test", "predict"
    Dict[str, Any],          # Complex partition specification
    None,                    # Auto-detect (based on file naming, not implemented here)
]

# Partition types
PartitionName = Literal["train", "test", "predict"]


@dataclass
class PartitionResult:
    """Result of a partition assignment operation.

    Attributes:
        train_indices: List of indices assigned to training partition.
        test_indices: List of indices assigned to test partition.
        predict_indices: List of indices assigned to predict partition (no targets).
        train_data: DataFrame subset for training.
        test_data: DataFrame subset for testing.
        predict_data: DataFrame subset for prediction.
        partition_column: Name of column used for partitioning (if column-based).
    """
    train_indices: List[int] = field(default_factory=list)
    test_indices: List[int] = field(default_factory=list)
    predict_indices: List[int] = field(default_factory=list)
    train_data: Optional[pd.DataFrame] = None
    test_data: Optional[pd.DataFrame] = None
    predict_data: Optional[pd.DataFrame] = None
    partition_column: Optional[str] = None

    @property
    def has_train(self) -> bool:
        """Check if training data exists."""
        return len(self.train_indices) > 0

    @property
    def has_test(self) -> bool:
        """Check if test data exists."""
        return len(self.test_indices) > 0

    @property
    def has_predict(self) -> bool:
        """Check if predict data exists."""
        return len(self.predict_indices) > 0

    def get_indices(self, partition: PartitionName) -> List[int]:
        """Get indices for a specific partition."""
        if partition == "train":
            return self.train_indices
        elif partition == "test":
            return self.test_indices
        elif partition == "predict":
            return self.predict_indices
        else:
            raise PartitionError(f"Unknown partition: {partition}")

    def get_data(self, partition: PartitionName) -> Optional[pd.DataFrame]:
        """Get data for a specific partition."""
        if partition == "train":
            return self.train_data
        elif partition == "test":
            return self.test_data
        elif partition == "predict":
            return self.predict_data
        else:
            raise PartitionError(f"Unknown partition: {partition}")


class PartitionAssigner:
    """Flexible partition assigner for DataFrames.

    Supports multiple partition methods:
    - Static: `"train"`, `"test"`, `"predict"` (assign entire DataFrame)
    - Column-based: `{"column": "split", "train_values": [...], "test_values": [...]}`
    - Percentage-based: `{"train": "80%", "test": "20%", "shuffle": True}`
    - Index-based: `{"train": [0,1,2], "test": [3,4,5]}`
    - Index file: `{"train_file": "train_idx.txt", "test_file": "test_idx.txt"}`

    Example:
        >>> assigner = PartitionAssigner()
        >>> result = assigner.assign(df, {"train": "80%", "test": "20%"})
        >>> print(len(result.train_data), len(result.test_data))
    """

    # Recognized partition names
    PARTITION_NAMES = ("train", "test", "predict")

    # Recognized train values in column-based partitioning
    DEFAULT_TRAIN_VALUES = ("train", "training", "cal", "calibration")

    # Recognized test values in column-based partitioning
    DEFAULT_TEST_VALUES = ("test", "testing", "val", "validation", "valid")

    # Recognized predict values
    DEFAULT_PREDICT_VALUES = ("predict", "prediction", "unknown")

    def __init__(
        self,
        default_random_state: Optional[int] = None,
        base_path: Optional[Path] = None,
    ):
        """Initialize the partition assigner.

        Args:
            default_random_state: Default random state for shuffle operations.
            base_path: Base path for resolving relative paths in index files.
        """
        self.default_random_state = default_random_state
        self.base_path = base_path

    def assign(
        self,
        df: pd.DataFrame,
        partition: PartitionSpec,
    ) -> PartitionResult:
        """Assign rows to partitions.

        Args:
            df: The DataFrame to partition.
            partition: Partition specification. Can be:
                - str: Static partition ("train", "test", "predict")
                - dict: Complex partition (column-based, percentage, or index)
                - None: No partitioning (returns empty result)

        Returns:
            PartitionResult with indices and data for each partition.

        Raises:
            PartitionError: If partition specification is invalid.
        """
        if partition is None:
            # No partitioning - return empty result
            return PartitionResult()

        if isinstance(partition, str):
            return self._assign_static(df, partition)

        if isinstance(partition, dict):
            return self._assign_from_dict(df, partition)

        raise PartitionError(
            f"Unsupported partition type: {type(partition).__name__}. "
            f"Expected str, dict, or None."
        )

    def _assign_static(
        self,
        df: pd.DataFrame,
        partition: str,
    ) -> PartitionResult:
        """Assign entire DataFrame to a single partition.

        Args:
            df: The DataFrame to assign.
            partition: Partition name ("train", "test", or "predict").

        Returns:
            PartitionResult with all rows in the specified partition.
        """
        partition_lower = partition.lower()

        if partition_lower not in self.PARTITION_NAMES:
            raise PartitionError(
                f"Invalid partition name: '{partition}'. "
                f"Expected one of: {self.PARTITION_NAMES}"
            )

        indices = list(range(len(df)))
        result = PartitionResult()

        if partition_lower == "train":
            result.train_indices = indices
            result.train_data = df.copy()
        elif partition_lower == "test":
            result.test_indices = indices
            result.test_data = df.copy()
        elif partition_lower == "predict":
            result.predict_indices = indices
            result.predict_data = df.copy()

        return result

    def _assign_from_dict(
        self,
        df: pd.DataFrame,
        partition: Dict[str, Any],
    ) -> PartitionResult:
        """Assign based on dictionary specification.

        Detects the partition method from the dictionary keys:
        - "column": Column-based partitioning
        - "train"/"test" with percentages or lists: Percentage or index-based
        - "train_file"/"test_file": Index file-based
        """
        # Check for column-based partitioning
        if "column" in partition:
            return self._assign_by_column(df, partition)

        # Check for index file-based partitioning
        if "train_file" in partition or "test_file" in partition:
            return self._assign_by_index_file(df, partition)

        # Check for percentage or index-based partitioning
        if "train" in partition or "test" in partition:
            train_spec = partition.get("train")
            test_spec = partition.get("test")
            predict_spec = partition.get("predict")

            # Detect if percentage/range-based (string with % or :)
            def is_range_or_percentage(spec):
                if isinstance(spec, str):
                    return "%" in spec or ":" in spec
                return False

            is_range_based = (
                is_range_or_percentage(train_spec) or
                is_range_or_percentage(test_spec) or
                is_range_or_percentage(predict_spec)
            )

            if is_range_based:
                return self._assign_by_percentage(df, partition)
            else:
                return self._assign_by_indices(df, partition)

        raise PartitionError(
            f"Cannot determine partition method from specification: {partition}. "
            f"Expected 'column', 'train'/'test' with values/percentages, "
            f"or 'train_file'/'test_file'."
        )

    def _assign_by_column(
        self,
        df: pd.DataFrame,
        partition: Dict[str, Any],
    ) -> PartitionResult:
        """Assign based on column values.

        Args:
            df: The DataFrame to partition.
            partition: Dict with keys:
                - column: Column name containing partition labels
                - train_values: Values indicating training data
                - test_values: Values indicating test data
                - predict_values: Values indicating predict data
                - unknown_policy: How to handle unknown values ("error", "ignore", "train")
        """
        column = partition.get("column")
        if column is None:
            raise PartitionError("Column-based partition requires 'column' key.")

        if column not in df.columns:
            raise PartitionError(
                f"Partition column '{column}' not found in DataFrame. "
                f"Available columns: {df.columns.tolist()[:10]}"
            )

        # Get value mappings with defaults
        train_values = partition.get("train_values", list(self.DEFAULT_TRAIN_VALUES))
        test_values = partition.get("test_values", list(self.DEFAULT_TEST_VALUES))
        predict_values = partition.get("predict_values", list(self.DEFAULT_PREDICT_VALUES))
        unknown_policy = partition.get("unknown_policy", "error")

        # Normalize values to lowercase for comparison
        train_values_lower = {str(v).lower() for v in train_values}
        test_values_lower = {str(v).lower() for v in test_values}
        predict_values_lower = {str(v).lower() for v in predict_values}

        # Build indices for each partition
        train_indices = []
        test_indices = []
        predict_indices = []
        unknown_indices = []

        for idx in range(len(df)):
            value = str(df.iloc[idx][column]).lower()

            if value in train_values_lower:
                train_indices.append(idx)
            elif value in test_values_lower:
                test_indices.append(idx)
            elif value in predict_values_lower:
                predict_indices.append(idx)
            else:
                unknown_indices.append(idx)

        # Handle unknown values
        if unknown_indices:
            if unknown_policy == "error":
                unknown_values = df.iloc[unknown_indices][column].unique().tolist()
                raise PartitionError(
                    f"Unknown partition values in column '{column}': {unknown_values}. "
                    f"Expected train: {list(train_values_lower)}, "
                    f"test: {list(test_values_lower)}, "
                    f"predict: {list(predict_values_lower)}. "
                    f"Set unknown_policy='ignore' to skip or 'train' to include in training."
                )
            elif unknown_policy == "train":
                train_indices.extend(unknown_indices)
            # else: ignore (leave unknown indices unassigned)

        return PartitionResult(
            train_indices=train_indices,
            test_indices=test_indices,
            predict_indices=predict_indices,
            train_data=df.iloc[train_indices].copy() if train_indices else None,
            test_data=df.iloc[test_indices].copy() if test_indices else None,
            predict_data=df.iloc[predict_indices].copy() if predict_indices else None,
            partition_column=column,
        )

    def _assign_by_percentage(
        self,
        df: pd.DataFrame,
        partition: Dict[str, Any],
    ) -> PartitionResult:
        """Assign based on percentage splits.

        Args:
            df: The DataFrame to partition.
            partition: Dict with keys:
                - train: Percentage for training (e.g., "80%", "0:80%")
                - test: Percentage for testing (e.g., "20%", "80%:100%")
                - predict: Percentage for prediction
                - shuffle: Whether to shuffle before splitting (default: False)
                - random_state: Random state for shuffling
                - stratify: Column name for stratified splitting
        """
        n_rows = len(df)
        shuffle = partition.get("shuffle", False)
        random_state = partition.get("random_state", self.default_random_state)
        stratify_column = partition.get("stratify")

        # Parse percentages
        train_spec = partition.get("train")
        test_spec = partition.get("test")
        predict_spec = partition.get("predict")

        # Calculate indices
        indices = np.arange(n_rows)

        # Handle stratification
        if stratify_column:
            if stratify_column not in df.columns:
                raise PartitionError(
                    f"Stratify column '{stratify_column}' not found in DataFrame."
                )
            indices = self._stratified_shuffle(df, stratify_column, random_state)
        elif shuffle:
            rng = np.random.RandomState(random_state)
            rng.shuffle(indices)

        # Parse each partition's percentage
        train_range = self._parse_percentage_spec(train_spec, n_rows) if train_spec else None
        test_range = self._parse_percentage_spec(test_spec, n_rows) if test_spec else None
        predict_range = self._parse_percentage_spec(predict_spec, n_rows) if predict_spec else None

        # Assign indices
        train_indices = []
        test_indices = []
        predict_indices = []

        if train_range:
            start, end = train_range
            train_indices = indices[start:end].tolist()

        if test_range:
            start, end = test_range
            test_indices = indices[start:end].tolist()

        if predict_range:
            start, end = predict_range
            predict_indices = indices[start:end].tolist()

        # Validate no overlap
        self._validate_no_overlap(train_indices, test_indices, predict_indices)

        return PartitionResult(
            train_indices=train_indices,
            test_indices=test_indices,
            predict_indices=predict_indices,
            train_data=df.iloc[train_indices].copy() if train_indices else None,
            test_data=df.iloc[test_indices].copy() if test_indices else None,
            predict_data=df.iloc[predict_indices].copy() if predict_indices else None,
        )

    def _parse_percentage_spec(
        self,
        spec: str,
        n_rows: int,
    ) -> tuple:
        """Parse a percentage specification into (start_idx, end_idx).

        Formats:
            "80%" -> (0, int(n_rows * 0.8))
            "0:80%" -> (0, int(n_rows * 0.8))
            "80%:100%" -> (int(n_rows * 0.8), n_rows)
            "20%:80%" -> (int(n_rows * 0.2), int(n_rows * 0.8))
        """
        if not isinstance(spec, str):
            raise PartitionError(f"Percentage spec must be a string, got: {type(spec)}")

        spec = spec.strip()

        # Handle simple percentage like "80%"
        if ":" not in spec:
            if not spec.endswith("%"):
                raise PartitionError(f"Invalid percentage format: '{spec}'")
            pct = float(spec.rstrip("%")) / 100.0
            return (0, int(n_rows * pct))

        # Handle range like "80%:100%" or "0:80%"
        parts = spec.split(":")
        if len(parts) != 2:
            raise PartitionError(f"Invalid percentage range format: '{spec}'")

        start_str, end_str = parts

        # Parse start
        if start_str.strip().endswith("%"):
            start_pct = float(start_str.strip().rstrip("%")) / 100.0
            start_idx = int(n_rows * start_pct)
        elif start_str.strip():
            start_idx = int(start_str.strip())
        else:
            start_idx = 0

        # Parse end
        if end_str.strip().endswith("%"):
            end_pct = float(end_str.strip().rstrip("%")) / 100.0
            end_idx = int(n_rows * end_pct)
        elif end_str.strip():
            end_idx = int(end_str.strip())
        else:
            end_idx = n_rows

        return (start_idx, end_idx)

    def _assign_by_indices(
        self,
        df: pd.DataFrame,
        partition: Dict[str, Any],
    ) -> PartitionResult:
        """Assign based on explicit index lists.

        Args:
            df: The DataFrame to partition.
            partition: Dict with keys:
                - train: List of indices for training
                - test: List of indices for testing
                - predict: List of indices for prediction
        """
        n_rows = len(df)

        train_indices = partition.get("train", [])
        test_indices = partition.get("test", [])
        predict_indices = partition.get("predict", [])

        # Validate and normalize indices
        train_indices = self._validate_indices(train_indices, n_rows, "train")
        test_indices = self._validate_indices(test_indices, n_rows, "test")
        predict_indices = self._validate_indices(predict_indices, n_rows, "predict")

        # Validate no overlap
        self._validate_no_overlap(train_indices, test_indices, predict_indices)

        return PartitionResult(
            train_indices=train_indices,
            test_indices=test_indices,
            predict_indices=predict_indices,
            train_data=df.iloc[train_indices].copy() if train_indices else None,
            test_data=df.iloc[test_indices].copy() if test_indices else None,
            predict_data=df.iloc[predict_indices].copy() if predict_indices else None,
        )

    def _assign_by_index_file(
        self,
        df: pd.DataFrame,
        partition: Dict[str, Any],
    ) -> PartitionResult:
        """Assign based on index files.

        Args:
            df: The DataFrame to partition.
            partition: Dict with keys:
                - train_file: Path to file with training indices
                - test_file: Path to file with test indices
                - predict_file: Path to file with predict indices
        """
        train_file = partition.get("train_file")
        test_file = partition.get("test_file")
        predict_file = partition.get("predict_file")

        train_indices = self._load_indices_from_file(train_file) if train_file else []
        test_indices = self._load_indices_from_file(test_file) if test_file else []
        predict_indices = self._load_indices_from_file(predict_file) if predict_file else []

        n_rows = len(df)

        # Validate indices
        train_indices = self._validate_indices(train_indices, n_rows, "train")
        test_indices = self._validate_indices(test_indices, n_rows, "test")
        predict_indices = self._validate_indices(predict_indices, n_rows, "predict")

        # Validate no overlap
        self._validate_no_overlap(train_indices, test_indices, predict_indices)

        return PartitionResult(
            train_indices=train_indices,
            test_indices=test_indices,
            predict_indices=predict_indices,
            train_data=df.iloc[train_indices].copy() if train_indices else None,
            test_data=df.iloc[test_indices].copy() if test_indices else None,
            predict_data=df.iloc[predict_indices].copy() if predict_indices else None,
        )

    def _load_indices_from_file(self, file_path: str) -> List[int]:
        """Load indices from a file.

        Supports:
        - Text files with one index per line
        - CSV files with indices in first column
        - JSON files with list of indices
        """
        path = Path(file_path)
        if self.base_path and not path.is_absolute():
            path = self.base_path / path

        if not path.exists():
            raise PartitionError(f"Index file not found: {path}")

        suffix = path.suffix.lower()

        try:
            if suffix == ".json":
                import json
                with open(path, "r") as f:
                    indices = json.load(f)
                if not isinstance(indices, list):
                    raise PartitionError(
                        f"JSON file {path} must contain a list of indices."
                    )
                return [int(i) for i in indices]

            elif suffix in (".yaml", ".yml"):
                import yaml
                with open(path, "r") as f:
                    indices = yaml.safe_load(f)
                if not isinstance(indices, list):
                    raise PartitionError(
                        f"YAML file {path} must contain a list of indices."
                    )
                return [int(i) for i in indices]

            elif suffix == ".csv":
                df = pd.read_csv(path, header=None)
                return df.iloc[:, 0].astype(int).tolist()

            else:
                # Assume text file with one index per line
                with open(path, "r") as f:
                    lines = f.readlines()
                return [int(line.strip()) for line in lines if line.strip()]

        except Exception as e:
            raise PartitionError(f"Failed to load indices from {path}: {e}")

    def _validate_indices(
        self,
        indices: List[int],
        n_rows: int,
        partition_name: str,
    ) -> List[int]:
        """Validate and normalize index list."""
        if not isinstance(indices, (list, tuple)):
            indices = [indices]

        validated = []
        for idx in indices:
            idx = int(idx)

            # Handle negative indices
            if idx < 0:
                idx = n_rows + idx

            if idx < 0 or idx >= n_rows:
                raise PartitionError(
                    f"Index {idx} in '{partition_name}' out of range. "
                    f"DataFrame has {n_rows} rows (0-{n_rows - 1})."
                )
            validated.append(idx)

        return validated

    def _validate_no_overlap(
        self,
        train_indices: List[int],
        test_indices: List[int],
        predict_indices: List[int],
    ) -> None:
        """Validate that partition indices don't overlap."""
        train_set = set(train_indices)
        test_set = set(test_indices)
        predict_set = set(predict_indices)

        # Check train-test overlap
        train_test_overlap = train_set & test_set
        if train_test_overlap:
            raise PartitionError(
                f"Train and test partitions overlap at indices: "
                f"{sorted(list(train_test_overlap))[:10]}"
            )

        # Check train-predict overlap
        train_predict_overlap = train_set & predict_set
        if train_predict_overlap:
            raise PartitionError(
                f"Train and predict partitions overlap at indices: "
                f"{sorted(list(train_predict_overlap))[:10]}"
            )

        # Check test-predict overlap
        test_predict_overlap = test_set & predict_set
        if test_predict_overlap:
            raise PartitionError(
                f"Test and predict partitions overlap at indices: "
                f"{sorted(list(test_predict_overlap))[:10]}"
            )

    def _stratified_shuffle(
        self,
        df: pd.DataFrame,
        stratify_column: str,
        random_state: Optional[int],
    ) -> np.ndarray:
        """Create stratified shuffled indices.

        Returns indices shuffled such that when split sequentially,
        each split maintains the original class proportions.

        This works by:
        1. Shuffling indices within each stratum
        2. Interleaving the strata proportionally

        Example: If we have 80 samples of class 0 and 20 of class 1,
        and we want 80% train, we should get:
        - Train: 64 of class 0 + 16 of class 1 (maintains 80/20 ratio)
        - Test: 16 of class 0 + 4 of class 1 (maintains 80/20 ratio)
        """
        rng = np.random.RandomState(random_state)

        # Group by stratify column
        groups = df.groupby(stratify_column)
        n_groups = len(groups)

        # Collect and shuffle indices for each group
        group_indices_list = []
        for name, group in groups:
            group_positions = [
                df.index.get_loc(idx)
                for idx in group.index.tolist()
            ]
            rng.shuffle(group_positions)
            group_indices_list.append(group_positions)

        # Interleave indices from each group proportionally
        # This ensures that any contiguous slice has similar proportions
        result_indices = []
        n_rows = len(df)

        # Use round-robin with proportional allocation
        group_positions = [0] * n_groups  # Track position in each group

        for i in range(n_rows):
            # Find which group to pick from based on proportion
            # We pick from the group that's most "behind" its expected proportion
            best_group = -1
            best_deficit = -float('inf')

            for g_idx, indices in enumerate(group_indices_list):
                if group_positions[g_idx] >= len(indices):
                    continue  # This group is exhausted

                expected = (i + 1) * len(indices) / n_rows
                actual = group_positions[g_idx]
                deficit = expected - actual

                if deficit > best_deficit:
                    best_deficit = deficit
                    best_group = g_idx

            if best_group >= 0:
                result_indices.append(group_indices_list[best_group][group_positions[best_group]])
                group_positions[best_group] += 1

        return np.array(result_indices)

    def concatenate_partitions(
        self,
        results: Sequence[PartitionResult],
    ) -> PartitionResult:
        """Concatenate multiple partition results.

        Useful when combining multiple files with the same partition.
        Indices are adjusted to account for concatenation order.

        Args:
            results: Sequence of PartitionResult objects.

        Returns:
            Combined PartitionResult.
        """
        if not results:
            return PartitionResult()

        combined_train = []
        combined_test = []
        combined_predict = []
        train_dfs = []
        test_dfs = []
        predict_dfs = []

        offset = 0

        for result in results:
            # Adjust indices by offset
            if result.train_indices:
                combined_train.extend([i + offset for i in result.train_indices])
                if result.train_data is not None:
                    train_dfs.append(result.train_data)

            if result.test_indices:
                combined_test.extend([i + offset for i in result.test_indices])
                if result.test_data is not None:
                    test_dfs.append(result.test_data)

            if result.predict_indices:
                combined_predict.extend([i + offset for i in result.predict_indices])
                if result.predict_data is not None:
                    predict_dfs.append(result.predict_data)

            # Update offset (use max index from this result)
            all_indices = (
                result.train_indices +
                result.test_indices +
                result.predict_indices
            )
            if all_indices:
                offset += max(all_indices) + 1

        return PartitionResult(
            train_indices=combined_train,
            test_indices=combined_test,
            predict_indices=combined_predict,
            train_data=pd.concat(train_dfs, ignore_index=True) if train_dfs else None,
            test_data=pd.concat(test_dfs, ignore_index=True) if test_dfs else None,
            predict_data=pd.concat(predict_dfs, ignore_index=True) if predict_dfs else None,
        )
