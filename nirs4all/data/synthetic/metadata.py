"""
Metadata generation for synthetic NIRS datasets.

This module provides tools for generating realistic sample metadata including
sample IDs, biological sample groupings, repetitions, and custom columns.

Example:
    >>> from nirs4all.data.synthetic.metadata import MetadataGenerator
    >>>
    >>> generator = MetadataGenerator(random_state=42)
    >>> metadata = generator.generate(
    ...     n_samples=100,
    ...     sample_id_prefix="S",
    ...     n_groups=3,
    ...     n_repetitions=(2, 4)
    ... )
    >>> print(metadata.keys())
    dict_keys(['sample_id', 'bio_sample_id', 'repetition', 'group'])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np


@dataclass
class MetadataGenerationResult:
    """
    Container for generated metadata.

    Attributes:
        sample_ids: Unique sample identifiers.
        bio_sample_ids: Biological sample identifiers (before repetitions).
        repetitions: Repetition number for each sample.
        groups: Group assignments.
        group_indices: Integer group indices (for stratification).
        n_bio_samples: Number of unique biological samples.
        additional_columns: Any extra columns generated.
    """

    sample_ids: np.ndarray
    bio_sample_ids: Optional[np.ndarray] = None
    repetitions: Optional[np.ndarray] = None
    groups: Optional[np.ndarray] = None
    group_indices: Optional[np.ndarray] = None
    n_bio_samples: int = 0
    additional_columns: Optional[Dict[str, np.ndarray]] = None

    def to_dict(self) -> Dict[str, np.ndarray]:
        """
        Convert to dictionary format suitable for DataFrame or SpectroDataset.

        Returns:
            Dictionary with string keys and array values.
        """
        result: Dict[str, np.ndarray] = {"sample_id": self.sample_ids}

        if self.bio_sample_ids is not None:
            result["bio_sample_id"] = self.bio_sample_ids

        if self.repetitions is not None:
            result["repetition"] = self.repetitions

        if self.groups is not None:
            result["group"] = self.groups

        if self.group_indices is not None:
            result["group_idx"] = self.group_indices

        if self.additional_columns:
            result.update(self.additional_columns)

        return result


class MetadataGenerator:
    """
    Generate realistic metadata for synthetic NIRS datasets.

    This class creates sample identifiers, biological sample groupings,
    repetition structures, and group assignments that mimic real
    spectroscopy datasets.

    Attributes:
        rng: NumPy random generator for reproducibility.

    Args:
        random_state: Random seed for reproducibility.

    Example:
        >>> generator = MetadataGenerator(random_state=42)
        >>>
        >>> # Generate with repetitions and groups
        >>> metadata = generator.generate(
        ...     n_samples=100,
        ...     sample_id_prefix="WHEAT",
        ...     n_groups=3,
        ...     group_names=["Field_A", "Field_B", "Field_C"],
        ...     n_repetitions=(2, 4)
        ... )
        >>>
        >>> # Result: Each biological sample has 2-4 spectral measurements
        >>> print(f"Bio samples: {metadata.n_bio_samples}")
        >>> print(f"Total samples: {len(metadata.sample_ids)}")
    """

    def __init__(self, random_state: Optional[int] = None) -> None:
        """
        Initialize the metadata generator.

        Args:
            random_state: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(random_state)
        self._random_state = random_state

    def generate(
        self,
        n_samples: int,
        *,
        sample_id_prefix: str = "S",
        n_groups: Optional[int] = None,
        group_names: Optional[List[str]] = None,
        n_repetitions: Union[int, Tuple[int, int]] = 1,
        bio_sample_prefix: str = "B",
        additional_columns: Optional[Dict[str, Any]] = None,
    ) -> MetadataGenerationResult:
        """
        Generate complete metadata for a synthetic dataset.

        This method handles the complex logic of generating samples with
        repetitions while respecting group structures. When repetitions
        are requested, biological samples are created first, then each
        is replicated 1 or more times to create the final samples.

        Args:
            n_samples: Total number of samples (spectra) to generate.
            sample_id_prefix: Prefix for sample ID strings.
            n_groups: Number of groups (None for no grouping).
            group_names: Optional list of group names. If None and n_groups > 0,
                generates names like "Group_0", "Group_1", etc.
            n_repetitions: Number of repetitions per biological sample.
                If int: fixed number of repetitions.
                If tuple (min, max): random number in range [min, max].
            bio_sample_prefix: Prefix for biological sample IDs.
            additional_columns: Dictionary of additional columns to generate.
                Keys are column names, values can be:
                - Callable(n_samples, rng) -> np.ndarray
                - List of values to randomly sample from
                - Tuple (distribution, params) for numeric data

        Returns:
            MetadataGenerationResult containing all generated metadata.

        Raises:
            ValueError: If n_samples is less than 1 or if n_repetitions
                would make it impossible to generate the requested samples.

        Example:
            >>> generator = MetadataGenerator(random_state=42)
            >>>
            >>> # Simple case: 100 samples, no repetitions
            >>> result = generator.generate(100)
            >>> assert len(result.sample_ids) == 100
            >>>
            >>> # With repetitions: ~50 bio samples, each measured 2 times
            >>> result = generator.generate(100, n_repetitions=2)
            >>> assert result.n_bio_samples == 50
            >>>
            >>> # Variable repetitions
            >>> result = generator.generate(100, n_repetitions=(1, 3))
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        # Parse repetition config
        if isinstance(n_repetitions, int):
            min_reps = max_reps = n_repetitions
        else:
            min_reps, max_reps = n_repetitions

        if min_reps < 1:
            raise ValueError(f"Minimum repetitions must be >= 1, got {min_reps}")

        # Generate samples with repetitions
        if min_reps == max_reps == 1:
            # No repetitions - simple case
            n_bio_samples = n_samples
            bio_sample_ids = None
            repetitions = None
            sample_bio_mapping = np.arange(n_samples)
        else:
            # With repetitions - need to calculate bio samples
            bio_sample_ids, repetitions, sample_bio_mapping, n_bio_samples = (
                self._generate_repetition_structure(
                    n_samples, min_reps, max_reps, bio_sample_prefix
                )
            )

        # Generate sample IDs
        sample_ids = self._generate_sample_ids(n_samples, sample_id_prefix)

        # Generate groups if requested
        groups = None
        group_indices = None
        if n_groups is not None and n_groups > 0:
            groups, group_indices = self._generate_groups(
                n_samples=n_samples,
                n_bio_samples=n_bio_samples,
                sample_bio_mapping=sample_bio_mapping,
                n_groups=n_groups,
                group_names=group_names,
            )

        # Generate additional columns
        extra_columns = None
        if additional_columns:
            extra_columns = self._generate_additional_columns(
                n_samples, additional_columns
            )

        return MetadataGenerationResult(
            sample_ids=sample_ids,
            bio_sample_ids=bio_sample_ids,
            repetitions=repetitions,
            groups=groups,
            group_indices=group_indices,
            n_bio_samples=n_bio_samples,
            additional_columns=extra_columns,
        )

    def _generate_sample_ids(
        self, n_samples: int, prefix: str
    ) -> np.ndarray:
        """Generate unique sample ID strings."""
        # Determine number of digits needed
        n_digits = max(4, len(str(n_samples)))
        return np.array([f"{prefix}{i:0{n_digits}d}" for i in range(n_samples)])

    def _generate_repetition_structure(
        self,
        n_samples: int,
        min_reps: int,
        max_reps: int,
        bio_sample_prefix: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Generate biological sample structure with repetitions.

        Returns:
            Tuple of (bio_sample_ids, repetitions, sample_bio_mapping, n_bio_samples)
        """
        if min_reps == max_reps:
            # Fixed repetitions
            avg_reps = min_reps
        else:
            # Variable repetitions - estimate needed bio samples
            avg_reps = (min_reps + max_reps) / 2

        # Estimate number of biological samples
        n_bio_estimate = int(np.ceil(n_samples / avg_reps))

        # Generate repetition counts for each bio sample
        if min_reps == max_reps:
            rep_counts = np.full(n_bio_estimate, min_reps)
        else:
            rep_counts = self.rng.integers(
                min_reps, max_reps + 1, size=n_bio_estimate
            )

        # Adjust to get exact sample count with a bounded loop
        total_samples = rep_counts.sum()
        max_iterations = n_samples * 2  # Safety limit
        iteration = 0

        while total_samples != n_samples and iteration < max_iterations:
            iteration += 1
            diff = n_samples - total_samples

            if diff > 0:
                # Need more samples
                if diff <= max_reps:
                    # Can add one final bio sample with exact count needed
                    # But ensure it's within valid range
                    if min_reps <= diff <= max_reps:
                        rep_counts = np.append(rep_counts, diff)
                    else:
                        # Add min_reps and continue adjusting
                        rep_counts = np.append(rep_counts, min_reps)
                else:
                    # Add a bio sample with random reps
                    new_reps = self.rng.integers(min_reps, max_reps + 1)
                    rep_counts = np.append(rep_counts, new_reps)
            else:
                # Too many samples - reduce or remove
                if len(rep_counts) == 0:
                    break  # Safety: can't remove from empty
                if rep_counts[-1] > min_reps:
                    # Reduce last bio sample's reps
                    reduction = min(-diff, rep_counts[-1] - min_reps)
                    rep_counts[-1] -= reduction
                else:
                    # Remove last bio sample entirely
                    rep_counts = rep_counts[:-1]

            total_samples = rep_counts.sum()

        # If we couldn't match exactly, force a match by adjusting last element
        if total_samples != n_samples and len(rep_counts) > 0:
            diff = n_samples - total_samples
            if diff > 0:
                rep_counts[-1] += diff
            elif rep_counts[-1] + diff >= 1:
                rep_counts[-1] += diff

        n_bio_samples = len(rep_counts)

        # Generate bio sample IDs
        n_digits = max(4, len(str(n_bio_samples)))
        bio_ids_unique = np.array(
            [f"{bio_sample_prefix}{i:0{n_digits}d}" for i in range(n_bio_samples)]
        )

        # Expand to all samples
        bio_sample_ids = np.repeat(bio_ids_unique, rep_counts)
        sample_bio_mapping = np.repeat(np.arange(n_bio_samples), rep_counts)

        # Generate repetition numbers
        repetitions = np.concatenate([
            np.arange(1, count + 1) for count in rep_counts
        ])

        return bio_sample_ids, repetitions, sample_bio_mapping, n_bio_samples

    def _generate_groups(
        self,
        n_samples: int,
        n_bio_samples: int,
        sample_bio_mapping: np.ndarray,
        n_groups: int,
        group_names: Optional[List[str]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate group assignments.

        Groups are assigned at the biological sample level to ensure all
        repetitions of a sample belong to the same group.
        """
        # Generate group names
        if group_names is None:
            group_names = [f"Group_{i}" for i in range(n_groups)]
        elif len(group_names) != n_groups:
            raise ValueError(
                f"group_names length ({len(group_names)}) must match "
                f"n_groups ({n_groups})"
            )

        # Assign groups to biological samples (balanced)
        bio_group_indices = np.zeros(n_bio_samples, dtype=np.int32)
        samples_per_group = n_bio_samples // n_groups
        remainder = n_bio_samples % n_groups

        idx = 0
        for g in range(n_groups):
            count = samples_per_group + (1 if g < remainder else 0)
            bio_group_indices[idx:idx + count] = g
            idx += count

        # Shuffle group assignments
        self.rng.shuffle(bio_group_indices)

        # Expand to all samples
        group_indices = bio_group_indices[sample_bio_mapping]
        groups = np.array([group_names[i] for i in group_indices])

        return groups, group_indices

    def _generate_additional_columns(
        self,
        n_samples: int,
        columns: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Generate additional metadata columns based on specifications."""
        result = {}

        for col_name, spec in columns.items():
            if callable(spec):
                # User-provided generator function
                result[col_name] = spec(n_samples, self.rng)
            elif isinstance(spec, (list, np.ndarray)):
                # Random sampling from provided values
                result[col_name] = self.rng.choice(spec, size=n_samples)
            elif isinstance(spec, tuple) and len(spec) == 2:
                # Distribution specification
                dist_name, params = spec
                result[col_name] = self._generate_from_distribution(
                    n_samples, dist_name, params
                )
            else:
                raise ValueError(
                    f"Invalid specification for column '{col_name}': {spec}. "
                    f"Expected callable, list, or (distribution, params) tuple."
                )

        return result

    def _generate_from_distribution(
        self,
        n_samples: int,
        dist_name: str,
        params: Dict[str, Any],
    ) -> np.ndarray:
        """Generate values from a named distribution."""
        if dist_name == "uniform":
            low = params.get("low", 0)
            high = params.get("high", 1)
            return self.rng.uniform(low, high, size=n_samples)
        elif dist_name == "normal":
            mean = params.get("mean", 0)
            std = params.get("std", 1)
            return self.rng.normal(mean, std, size=n_samples)
        elif dist_name == "integers":
            low = params.get("low", 0)
            high = params.get("high", 10)
            return self.rng.integers(low, high + 1, size=n_samples)
        elif dist_name == "choice":
            values = params.get("values", [0, 1])
            probs = params.get("probs", None)
            return self.rng.choice(values, size=n_samples, p=probs)
        else:
            raise ValueError(f"Unknown distribution: '{dist_name}'")


def generate_sample_metadata(
    n_samples: int,
    *,
    random_state: Optional[int] = None,
    sample_id_prefix: str = "S",
    n_groups: Optional[int] = None,
    group_names: Optional[List[str]] = None,
    n_repetitions: Union[int, Tuple[int, int]] = 1,
) -> Dict[str, np.ndarray]:
    """
    Convenience function to generate sample metadata.

    This is a simplified interface to MetadataGenerator for common use cases.

    Args:
        n_samples: Total number of samples to generate.
        random_state: Random seed for reproducibility.
        sample_id_prefix: Prefix for sample ID strings.
        n_groups: Number of groups (None for no grouping).
        group_names: Optional list of group names.
        n_repetitions: Repetitions per biological sample.

    Returns:
        Dictionary with metadata arrays.

    Example:
        >>> metadata = generate_sample_metadata(
        ...     n_samples=100,
        ...     random_state=42,
        ...     n_groups=3,
        ...     n_repetitions=(2, 4)
        ... )
        >>> print(metadata.keys())
    """
    generator = MetadataGenerator(random_state=random_state)
    result = generator.generate(
        n_samples=n_samples,
        sample_id_prefix=sample_id_prefix,
        n_groups=n_groups,
        group_names=group_names,
        n_repetitions=n_repetitions,
    )
    return result.to_dict()
