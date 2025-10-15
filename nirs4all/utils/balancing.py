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
    def calculate_balanced_counts(labels: np.ndarray, sample_indices: np.ndarray, max_factor: float = 1.0) -> Dict[int, int]:
        """
        Calculate how many augmentations needed per sample for balancing.

        This method analyzes class distribution and computes how many synthetic samples
        need to be created for each original sample to achieve a balanced dataset.

        Args:
            labels: Class labels or group IDs (1D array). Can be numeric or string.
            sample_indices: Corresponding sample IDs (same length as labels)
            max_factor: Target fraction of majority class (0.0-1.0). Default 1.0 means
                       minority classes will be balanced to match majority class size.
                       0.8 means minority classes target 80% of majority size.

        Returns:
            Dictionary mapping sample_id → augmentation_count
            For each sample, returns the number of augmented versions to create.

        Examples:
            >>> labels = np.array([0, 0, 0, 0, 1, 1])  # 4 class-0, 2 class-1
            >>> indices = np.array([10, 11, 12, 13, 14, 15])
            >>> counts = BalancingCalculator.calculate_balanced_counts(labels, indices, max_factor=1.0)
            >>> # Class 1 needs augmentation: target=4, current=2, need=2 total
            >>> # Each class-1 sample gets: 2 // 2 = 1 augmentation
            >>> counts[14]  # 1
            >>> counts[15]  # 1
            >>> counts[10]  # 0 (majority class, no augmentation)
        """
        if len(labels) != len(sample_indices):
            raise ValueError(f"labels and sample_indices must have same length, got {len(labels)} and {len(sample_indices)}")

        if not 0.0 <= max_factor <= 1.0:
            raise ValueError(f"max_factor must be between 0.0 and 1.0, got {max_factor}")

        if len(labels) == 0:
            return {}

        # Build mapping: label → list of sample_ids
        label_to_samples = {}
        for sample_id, label in zip(sample_indices, labels):
            # Convert to hashable type (handle numpy types and strings)
            label_key = label.item() if hasattr(label, 'item') else label
            label_to_samples.setdefault(label_key, []).append(int(sample_id))

        # Find majority class size
        class_sizes = {label: len(samples) for label, samples in label_to_samples.items()}
        majority_size = max(class_sizes.values())

        # Calculate target size based on max_factor
        target_size = int(majority_size * max_factor)

        # Calculate augmentations per sample for each class
        augmentation_map = {}

        for label, samples in label_to_samples.items():
            current_size = len(samples)

            if current_size >= target_size:
                # Already balanced or majority class - no augmentation needed
                for sample_id in samples:
                    augmentation_map[sample_id] = 0
            else:
                # Needs augmentation to reach target
                total_needed = target_size - current_size
                base_count = total_needed // current_size  # Base augmentation count per sample
                remainder = total_needed % current_size  # Extra augmentations to distribute

                # Distribute augmentations: first 'remainder' samples get +1 extra
                for i, sample_id in enumerate(samples):
                    augmentation_map[sample_id] = base_count + (1 if i < remainder else 0)

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
