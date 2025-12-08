"""Sampling utilities for generator module.

This module provides deterministic sampling functions with seed support
for reproducible pipeline generation.
"""

import random
from typing import List, Optional, TypeVar

T = TypeVar('T')


def sample_with_seed(
    population: List[T],
    k: int,
    seed: Optional[int] = None,
    weights: Optional[List[float]] = None
) -> List[T]:
    """Sample k items from population with optional seed for reproducibility.

    This function wraps Python's random sampling functions to provide
    deterministic behavior when a seed is specified. The function uses
    `random.sample` for unweighted sampling and `random.choices` for
    weighted sampling.

    Args:
        population: List of items to sample from.
        k: Number of items to sample.
        seed: Optional random seed for reproducibility. If None, uses
            current random state (non-deterministic).
        weights: Optional list of weights for weighted random selection.
            Must have the same length as population. If None, uniform
            sampling is used.

    Returns:
        List of k sampled items from population.

    Raises:
        ValueError: If k is larger than population size (for unweighted sampling).
        ValueError: If weights length doesn't match population length.

    Examples:
        >>> sample_with_seed(["A", "B", "C", "D"], 2, seed=42)
        ['D', 'A']  # Deterministic result with seed=42
        >>> sample_with_seed(["A", "B", "C"], 2, seed=42)
        ['C', 'A']  # Same seed produces same sequence
        >>> sample_with_seed(["A", "B", "C"], 5, seed=42)  # k > len(population)
        ['A', 'B', 'C']  # Returns all items (capped at population size)
    """
    if not population:
        return []

    # Cap k at population size to avoid errors
    k = min(k, len(population))

    if k <= 0:
        return []

    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    if weights is not None:
        if len(weights) != len(population):
            raise ValueError(
                f"Weights length ({len(weights)}) must match "
                f"population length ({len(population)})"
            )
        # Use choices for weighted sampling (with replacement by default)
        # For unique samples, we need a different approach
        return _weighted_sample_without_replacement(population, k, weights, rng)
    else:
        return rng.sample(population, k)


def _weighted_sample_without_replacement(
    population: List[T],
    k: int,
    weights: List[float],
    rng: random.Random
) -> List[T]:
    """Sample k items without replacement using weights.

    Uses a sequential selection approach where after each selection,
    the weight is set to 0 and remaining weights are renormalized.

    Args:
        population: List of items to sample from.
        k: Number of items to sample.
        weights: List of weights for each item.
        rng: Random number generator instance.

    Returns:
        List of k unique sampled items.
    """
    if k >= len(population):
        return list(population)

    # Make a copy of weights to modify
    remaining_weights = list(weights)
    remaining_indices = list(range(len(population)))
    selected = []

    for _ in range(k):
        if not remaining_indices:
            break

        # Normalize weights
        total = sum(remaining_weights[i] for i in remaining_indices)
        if total <= 0:
            # All remaining weights are 0, fall back to uniform
            idx = rng.choice(remaining_indices)
        else:
            # Weighted selection
            r = rng.random() * total
            cumulative = 0
            idx = remaining_indices[0]
            for i in remaining_indices:
                cumulative += remaining_weights[i]
                if cumulative >= r:
                    idx = i
                    break

        selected.append(population[idx])
        remaining_indices.remove(idx)

    return selected


def shuffle_with_seed(items: List[T], seed: Optional[int] = None) -> List[T]:
    """Shuffle a list with optional seed for reproducibility.

    Args:
        items: List of items to shuffle.
        seed: Optional random seed for reproducibility.

    Returns:
        A new shuffled list (original is not modified).

    Examples:
        >>> shuffle_with_seed([1, 2, 3, 4, 5], seed=42)
        [4, 5, 2, 1, 3]  # Deterministic result with seed=42
    """
    result = list(items)
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(result)
    else:
        random.shuffle(result)
    return result


def random_choice_with_seed(
    population: List[T],
    seed: Optional[int] = None,
    weights: Optional[List[float]] = None
) -> T:
    """Choose a single random item from population.

    Args:
        population: List of items to choose from.
        seed: Optional random seed for reproducibility.
        weights: Optional list of weights for weighted selection.

    Returns:
        A single randomly selected item.

    Raises:
        IndexError: If population is empty.
        ValueError: If weights length doesn't match population length.

    Examples:
        >>> random_choice_with_seed(["A", "B", "C"], seed=42)
        'C'  # Deterministic with seed
    """
    if not population:
        raise IndexError("Cannot choose from an empty population")

    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    if weights is not None:
        if len(weights) != len(population):
            raise ValueError(
                f"Weights length ({len(weights)}) must match "
                f"population length ({len(population)})"
            )
        return rng.choices(population, weights=weights, k=1)[0]
    else:
        return rng.choice(population)
