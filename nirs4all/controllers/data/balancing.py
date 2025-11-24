"""
Balancing utilities for sample augmentation.

This module provides utilities to calculate augmentation counts for balanced datasets
and to apply random transformer selection strategies.
"""
from typing import List, Dict, Optional
import numpy as np


class BalancingCalculator:
    """Calculate augmentation counts for balanced datasets."""

    @staticmethod
    def calculate_balanced_counts(
        base_labels: np.ndarray,
        base_sample_indices: np.ndarray,
        all_labels: np.ndarray,
        all_sample_indices: np.ndarray,
        target_size: Optional[int] = None,
        max_factor: Optional[float] = None,
        ref_percentage: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Dict[int, int]:
        """
        Calculate augmentations per BASE sample considering ALL samples for target.

        Three balancing modes are supported (use exactly one):

        1. target_size mode: Augment each class to a fixed target sample count.
           - target_size: int - desired number of samples per class
           - Example: target_size=100 means each class will have 100 samples
           - No cap: classes can exceed majority class size if target_size > majority

        2. max_factor mode: Augment each class by a multiplier, capped at majority class size.
           - max_factor: float - multiplier applied to each class's current size
           - Target is capped at majority class size (majority class is never augmented)
           - Example: max_factor=3 with majority=100, class=20 → target=min(60, 100)=60
           - Example: max_factor=2 with majority=100, class=100 → target=100 (no augmentation)

        3. ref_percentage mode: Augment each class to a percentage of the majority class.
           - ref_percentage: float - can be any positive value (0.5-2.0, etc)
           - If < 1.0: targets below majority (e.g., 0.8 with majority=100 → target=80)
           - If > 1.0: targets above majority, like a multiplier of majority class
           - Example: ref_percentage=1.5 with majority=100 → target=150

        Args:
            base_labels: Class labels for BASE samples only
            base_sample_indices: BASE sample IDs (these have data to augment)
            all_labels: Class labels for ALL samples (base + augmented)
            all_sample_indices: ALL sample IDs (for calculating target size)
            target_size: Fixed target samples per class (mode 1)
            max_factor: Multiplier for augmentation, capped at majority (mode 2)
            ref_percentage: Target as multiple of reference class (mode 3, can be > 1.0)
            random_state: Random seed for reproducible remainder distribution

        Returns:
            Dict mapping base_sample_id → augmentation_count

        Raises:
            ValueError: If zero or multiple modes are specified, or invalid parameter values
        """
        if len(base_labels) != len(base_sample_indices):
            raise ValueError("base_labels and base_sample_indices must have same length")
        if len(all_labels) != len(all_sample_indices):
            raise ValueError("all_labels and all_sample_indices must have same length")

        if len(base_labels) == 0:
            return {}

        # Validate that exactly one mode is specified
        modes_specified = sum([target_size is not None, max_factor is not None, ref_percentage is not None])
        if modes_specified == 0:
            # Default to max_factor=1.0 if nothing specified
            max_factor = 1.0
        elif modes_specified > 1:
            raise ValueError("Specify exactly one of: target_size, max_factor, or ref_percentage")

        # Count ALL samples per class (to get current distribution)
        all_class_counts = {}
        for label in all_labels:
            label_key = label.item() if hasattr(label, 'item') else label
            all_class_counts[label_key] = all_class_counts.get(label_key, 0) + 1

        # Build mapping: label → list of BASE sample_ids
        label_to_base_samples = {}
        for sample_id, label in zip(base_sample_indices, base_labels):
            label_key = label.item() if hasattr(label, 'item') else label
            label_to_base_samples.setdefault(label_key, []).append(int(sample_id))

        # Calculate target size per class based on mode
        if target_size is not None:
            # Mode 1: Fixed target size per class
            if target_size <= 0:
                raise ValueError(f"target_size must be positive, got {target_size}")
            target_size_per_class = target_size
        elif max_factor is not None:
            # Mode 2: Multiplier - each class augmented by factor (applied to current size)
            if max_factor <= 0:
                raise ValueError(f"max_factor must be positive, got {max_factor}")
            # This mode calculates per-class target differently
            target_size_per_class = None  # Will be calculated per-class
        elif ref_percentage is not None:
            # Mode 3: Percentage of reference (majority) class
            if ref_percentage <= 0:
                raise ValueError(f"ref_percentage must be positive, got {ref_percentage}")
            majority_size = max(all_class_counts.values())
            target_size_per_class = int(majority_size * ref_percentage)
        else:
            # Default: use max_factor=1.0
            target_size_per_class = None

        # Get majority size for max_factor capping
        majority_size = max(all_class_counts.values()) if all_class_counts else 0

        # Setup random number generator for remainder distribution
        rng = np.random.default_rng(random_state)

        # Calculate augmentations per BASE sample
        augmentation_map = {}

        for label, base_samples in label_to_base_samples.items():
            current_total = all_class_counts.get(label, 0)
            base_count = len(base_samples)

            # Calculate target for this class
            if max_factor is not None and target_size_per_class is None:
                # Mode 2: Multiplier - each class augmented by factor (applied to current size)
                # capped at majority class size
                target_from_factor = int(current_total * max_factor)
                class_target = min(target_from_factor, majority_size)
            else:
                # Modes 1, 3, or default
                class_target = target_size_per_class

            if current_total >= class_target:
                # Already balanced - no augmentation needed
                for sample_id in base_samples:
                    augmentation_map[sample_id] = 0
            else:
                # Need augmentation to reach target
                total_needed = class_target - current_total
                aug_per_base = total_needed // base_count
                remainder = total_needed % base_count

                # Initialize all samples with base augmentation count
                for sample_id in base_samples:
                    augmentation_map[sample_id] = aug_per_base

                # Randomly select which samples get the remainder augmentations
                if remainder > 0:
                    remainder_indices = rng.choice(base_count, size=remainder, replace=False)
                    for idx in remainder_indices:
                        augmentation_map[base_samples[idx]] += 1

        return augmentation_map

    @staticmethod
    def calculate_balanced_counts_value_aware(
        base_labels: np.ndarray,
        base_sample_indices: np.ndarray,
        base_values: np.ndarray,
        all_labels: np.ndarray,
        all_sample_indices: np.ndarray,
        target_size: Optional[int] = None,
        max_factor: Optional[float] = None,
        ref_percentage: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Dict[int, int]:
        """
        Calculate augmentations per BASE sample with value-aware distribution.

        This method first distributes augmentations fairly across unique values,
        then distributes within each value group across its samples.

        Useful for binned regression where samples with the same bin value
        should be treated fairly together, not competing individually.

        Args:
            base_labels: Class labels for BASE samples only
            base_sample_indices: BASE sample IDs
            base_values: Actual values (y or bin values) for BASE samples
                         Used to group samples with same value
            all_labels: Class labels for ALL samples (base + augmented)
            all_sample_indices: ALL sample IDs
            target_size: Fixed target samples per class (mode 1)
            max_factor: Multiplier for augmentation (mode 2)
            ref_percentage: Target as multiple of reference class (mode 3)
            random_state: Random seed for reproducibility

        Returns:
            Dict mapping base_sample_id → augmentation_count
        """
        if len(base_labels) != len(base_sample_indices):
            raise ValueError("base_labels and base_sample_indices must have same length")
        if len(base_labels) != len(base_values):
            raise ValueError("base_values must have same length as base_labels")
        if len(all_labels) != len(all_sample_indices):
            raise ValueError("all_labels and all_sample_indices must have same length")

        if len(base_labels) == 0:
            return {}

        # Validate that exactly one mode is specified
        modes_specified = sum([target_size is not None, max_factor is not None, ref_percentage is not None])
        if modes_specified == 0:
            max_factor = 1.0
        elif modes_specified > 1:
            raise ValueError("Specify exactly one of: target_size, max_factor, or ref_percentage")

        # Count ALL samples per class
        all_class_counts = {}
        for label in all_labels:
            label_key = label.item() if hasattr(label, 'item') else label
            all_class_counts[label_key] = all_class_counts.get(label_key, 0) + 1

        # Build mapping: label → list of (sample_id, value) pairs
        label_to_samples_with_values = {}
        for sample_id, label, value in zip(base_sample_indices, base_labels, base_values):
            label_key = label.item() if hasattr(label, 'item') else label
            value_key = value.item() if hasattr(value, 'item') else value
            label_to_samples_with_values.setdefault(label_key, []).append((int(sample_id), value_key))

        # Calculate target size per class
        if target_size is not None:
            if target_size <= 0:
                raise ValueError(f"target_size must be positive, got {target_size}")
            target_size_per_class = target_size
        elif max_factor is not None:
            if max_factor <= 0:
                raise ValueError(f"max_factor must be positive, got {max_factor}")
            target_size_per_class = None
        elif ref_percentage is not None:
            if ref_percentage <= 0:
                raise ValueError(f"ref_percentage must be positive, got {ref_percentage}")
            majority_size = max(all_class_counts.values())
            target_size_per_class = int(majority_size * ref_percentage)
        else:
            target_size_per_class = None

        majority_size = max(all_class_counts.values()) if all_class_counts else 0
        rng = np.random.default_rng(random_state)

        augmentation_map = {}

        for label, samples_with_values in label_to_samples_with_values.items():
            current_total = all_class_counts.get(label, 0)

            # Calculate target for this class
            if max_factor is not None and target_size_per_class is None:
                # Mode 2: Multiplier - each class augmented by factor (applied to current size)
                # capped at majority class size
                target_from_factor = int(current_total * max_factor)
                class_target = min(target_from_factor, majority_size)
            else:
                class_target = target_size_per_class

            if current_total >= class_target:
                # Already balanced
                for sample_id, _ in samples_with_values:
                    augmentation_map[sample_id] = 0
            else:
                # Need augmentation
                total_needed = class_target - current_total

                # Step 1: Group samples by value
                value_to_samples = {}
                for sample_id, value in samples_with_values:
                    value_to_samples.setdefault(value, []).append(sample_id)

                num_unique_values = len(value_to_samples)

                # Step 2: Distribute augmentations across unique values
                aug_per_value = total_needed // num_unique_values
                remainder_values = total_needed % num_unique_values

                # Randomly select which values get the remainder
                value_list = list(value_to_samples.keys())
                if remainder_values > 0:
                    remainder_value_indices = rng.choice(num_unique_values, size=remainder_values, replace=False)
                    remainder_value_set = {value_list[i] for i in remainder_value_indices}
                else:
                    remainder_value_set = set()

                # Step 3: Distribute within each value group
                for value, samples_in_value in value_to_samples.items():
                    # Determine augmentations for this value group
                    value_augmentations = aug_per_value + (1 if value in remainder_value_set else 0)
                    num_samples_in_value = len(samples_in_value)

                    # Distribute within the value group
                    aug_per_sample = value_augmentations // num_samples_in_value
                    remainder_samples = value_augmentations % num_samples_in_value

                    # Initialize all samples with base
                    for sample_id in samples_in_value:
                        augmentation_map[sample_id] = aug_per_sample

                    # Randomly select which samples in this value get remainder
                    if remainder_samples > 0:
                        remainder_sample_indices = rng.choice(num_samples_in_value, size=remainder_samples, replace=False)
                        for idx in remainder_sample_indices:
                            augmentation_map[samples_in_value[idx]] += 1

        return augmentation_map

    @staticmethod
    def apply_random_transformer_selection(transformers: List, augmentation_counts: Dict[int, int], random_state: Optional[int] = None) -> Dict[int, List[int]]:
        """
        Randomly select transformers for each augmentation.

        This method assigns transformer indices to each sample's augmentations,
        supporting reproducible randomization via random_state.

        Args:
            transformers: List of transformer instances (e.g., [SavGol(), Gaussian(), SNV()])
            augmentation_counts: sample_id → number of augmentations to create
            random_state: Random seed for reproducibility. None = non-deterministic

        Returns:
            Dictionary mapping sample_id → list of transformer indices
            For each sample, returns a list of length augmentation_counts[sample_id]
            containing randomly selected transformer indices.

        Examples:
            >>> transformers = [SavGol(), Gaussian(), SNV()]  # 3 transformers
            >>> counts = {10: 2, 11: 3, 12: 0}  # Sample 10 needs 2 augs, 11 needs 3, 12 needs 0
            >>> selection = BalancingCalculator.apply_random_transformer_selection(
            ...     transformers, counts, random_state=42
            ... )
            >>> len(selection[10])  # 2 (two transformer indices)
            >>> len(selection[11])  # 3 (three transformer indices)
            >>> selection[12]  # [] (no augmentations)
            >>> all(0 <= idx < 3 for idx in selection[10])  # True (valid indices)
        """
        if not transformers:
            raise ValueError("transformers list cannot be empty")

        rng = np.random.default_rng(random_state)
        transformer_selection = {}

        for sample_id, count in augmentation_counts.items():
            if count > 0:
                # Randomly select transformer indices for this sample
                selected = rng.integers(0, len(transformers), size=count).tolist()
                transformer_selection[sample_id] = selected
            else:
                # No augmentations needed
                transformer_selection[sample_id] = []

        return transformer_selection
