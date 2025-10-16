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
        max_factor: float = 1.0
    ) -> Dict[int, int]:
        """
        Calculate augmentations per BASE sample considering ALL samples for target.

        Args:
            base_labels: Class labels for BASE samples only
            base_sample_indices: BASE sample IDs (these have data to augment)
            all_labels: Class labels for ALL samples (base + augmented)
            all_sample_indices: ALL sample IDs (for calculating target size)
            max_factor: Target fraction of majority class (0.0-1.0).

        Returns:
            Dict mapping base_sample_id → augmentation_count
        """
        if len(base_labels) != len(base_sample_indices):
            raise ValueError(f"base_labels and base_sample_indices must have same length")
        if len(all_labels) != len(all_sample_indices):
            raise ValueError(f"all_labels and all_sample_indices must have same length")

        if not 0.0 <= max_factor <= 1.0:
            raise ValueError(f"max_factor must be between 0.0 and 1.0, got {max_factor}")

        if len(base_labels) == 0:
            return {}

        # Count ALL samples per class (to get target size)
        all_class_counts = {}
        for label in all_labels:
            label_key = label.item() if hasattr(label, 'item') else label
            all_class_counts[label_key] = all_class_counts.get(label_key, 0) + 1

        # Find target size from ALL samples
        majority_size = max(all_class_counts.values())
        target_size = int(majority_size * max_factor)

        # Build mapping: label → list of BASE sample_ids
        label_to_base_samples = {}
        for sample_id, label in zip(base_sample_indices, base_labels):
            label_key = label.item() if hasattr(label, 'item') else label
            label_to_base_samples.setdefault(label_key, []).append(int(sample_id))

        # Calculate augmentations per BASE sample
        augmentation_map = {}

        for label, base_samples in label_to_base_samples.items():
            current_total = all_class_counts.get(label, 0)
            base_count = len(base_samples)

            if current_total >= target_size:
                # Already balanced - no augmentation needed
                for sample_id in base_samples:
                    augmentation_map[sample_id] = 0
            else:
                # Need augmentation to reach target
                total_needed = target_size - current_total
                aug_per_base = total_needed // base_count
                remainder = total_needed % base_count

                for i, sample_id in enumerate(base_samples):
                    augmentation_map[sample_id] = aug_per_base + (1 if i < remainder else 0)

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
