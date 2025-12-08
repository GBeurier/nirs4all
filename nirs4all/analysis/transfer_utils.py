"""
Transfer Selection Utilities.

This module provides utility functions for preprocessing application,
pipeline generation, and dataset handling in transfer learning scenarios.
"""

from copy import deepcopy
from itertools import combinations, permutations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nirs4all.operators.transforms import (
    Detrend,
    FirstDerivative,
    Gaussian,
    Haar,
    IdentityTransformer,
    MultiplicativeScatterCorrection,
    RobustStandardNormalVariate,
    SavitzkyGolay,
    SecondDerivative,
    StandardNormalVariate,
    Wavelet,
)
from nirs4all.operators.transforms.nirs import (
    AreaNormalization,
    ExtendedMultiplicativeScatterCorrection as EMSC,
)


def get_base_preprocessings() -> Dict[str, Any]:
    """
    Get the base set of preprocessing transforms.

    Returns:
        Dictionary mapping names to transformer instances.

    Example:
        >>> preprocessings = get_base_preprocessings()
        >>> snv = preprocessings["snv"]
        >>> X_transformed = snv.fit_transform(X)
    """
    return {
        "snv": StandardNormalVariate(),
        "rsnv": RobustStandardNormalVariate(),
        "msc": MultiplicativeScatterCorrection(scale=False),
        "savgol": SavitzkyGolay(window_length=11, polyorder=3),
        "savgol_15": SavitzkyGolay(window_length=15, polyorder=3),
        "d1": FirstDerivative(),
        "d2": SecondDerivative(),
        "savgol_d1": SavitzkyGolay(window_length=11, polyorder=3, deriv=1),
        "savgol_d2": SavitzkyGolay(window_length=11, polyorder=3, deriv=2),
        "savgol15_d1": SavitzkyGolay(window_length=15, polyorder=3, deriv=1),
        "haar": Haar(),
        "detrend": Detrend(),
        "gaussian": Gaussian(order=1, sigma=2),
        "gaussian2": Gaussian(order=2, sigma=2),
        "emsc": EMSC(),
        "area_norm": AreaNormalization(),
        "wav_sym5": Wavelet("sym5"),
        "wav_coif3": Wavelet("coif3"),
        "identity": IdentityTransformer(),
    }


def apply_pipeline(X: np.ndarray, transforms: List[Any]) -> np.ndarray:
    """
    Apply a sequence of transforms to X.

    Args:
        X: Input data matrix (n_samples, n_features).
        transforms: List of transformer instances.

    Returns:
        Transformed data matrix.

    Example:
        >>> from nirs4all.operators.transforms import StandardNormalVariate, FirstDerivative
        >>> transforms = [StandardNormalVariate(), FirstDerivative()]
        >>> X_transformed = apply_pipeline(X, transforms)
    """
    X_out = X.copy()
    for t in transforms:
        t_copy = deepcopy(t)
        X_out = t_copy.fit_transform(X_out)
    return X_out


def apply_single_preprocessing(
    X: np.ndarray,
    pp_name: str,
    preprocessings: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Apply a single preprocessing by name.

    Args:
        X: Input data matrix (n_samples, n_features).
        pp_name: Name of the preprocessing (e.g., "snv", "d1").
        preprocessings: Optional dictionary of transforms. Uses base if None.

    Returns:
        Transformed data matrix.
    """
    if preprocessings is None:
        preprocessings = get_base_preprocessings()

    if pp_name not in preprocessings:
        raise ValueError(
            f"Unknown preprocessing: {pp_name}. "
            f"Available: {list(preprocessings.keys())}"
        )

    transform = preprocessings[pp_name]
    return apply_pipeline(X, [transform])


def generate_stacked_pipelines(
    preprocessings: Dict[str, Any],
    max_depth: int = 2,
    exclude: Optional[List[str]] = None,
) -> List[Tuple[str, List[str], List[Any]]]:
    """
    Generate stacked pipeline combinations.

    Args:
        preprocessings: Dictionary of available transforms.
        max_depth: Maximum pipeline depth (1 to max_depth).
        exclude: List of preprocessing names to exclude.

    Returns:
        List of (name, component_names, transforms) tuples.

    Example:
        >>> pp = {"snv": snv_transform, "d1": d1_transform}
        >>> pipelines = generate_stacked_pipelines(pp, max_depth=2)
        >>> # Returns: [("snv", ["snv"], [snv]), ("d1", ["d1"], [d1]),
        >>> #           ("snv>d1", ["snv", "d1"], [snv, d1]),
        >>> #           ("d1>snv", ["d1", "snv"], [d1, snv])]
    """
    if exclude is None:
        exclude = []

    names = [n for n in preprocessings.keys() if n not in exclude]
    pipelines = []

    for depth in range(1, max_depth + 1):
        for combo in permutations(names, depth):
            name = ">".join(combo)
            transforms = [preprocessings[n] for n in combo]
            pipelines.append((name, list(combo), transforms))

    return pipelines


def generate_top_k_stacked_pipelines(
    top_k_names: List[str],
    preprocessings: Dict[str, Any],
    max_depth: int = 2,
) -> List[Tuple[str, List[str], List[Any]]]:
    """
    Generate stacked pipeline combinations from top-K selected preprocessings.

    More efficient than generate_stacked_pipelines when starting from
    a reduced set of candidates.

    Args:
        top_k_names: List of preprocessing names from top-K selection.
        preprocessings: Dictionary of available transforms.
        max_depth: Maximum pipeline depth.

    Returns:
        List of (name, component_names, transforms) tuples.
    """
    pipelines = []

    for depth in range(2, max_depth + 1):  # Start at depth 2 (depth 1 already evaluated)
        for combo in permutations(top_k_names, depth):
            name = ">".join(combo)
            try:
                transforms = [preprocessings[n] for n in combo]
                pipelines.append((name, list(combo), transforms))
            except KeyError:
                # Skip if any transform not found
                continue

    return pipelines


def apply_stacked_pipeline(
    X: np.ndarray,
    pipeline_name: str,
    preprocessings: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Apply a stacked pipeline by name (e.g., "snv>d1>msc").

    Args:
        X: Input data matrix (n_samples, n_features).
        pipeline_name: Pipeline name with ">" separator.
        preprocessings: Optional dictionary of transforms.

    Returns:
        Transformed data matrix.
    """
    if preprocessings is None:
        preprocessings = get_base_preprocessings()

    component_names = pipeline_name.split(">")
    transforms = []

    for name in component_names:
        if name not in preprocessings:
            raise ValueError(f"Unknown preprocessing: {name}")
        transforms.append(preprocessings[name])

    return apply_pipeline(X, transforms)


def generate_augmentation_combinations(
    top_k_names: List[str],
    max_order: int = 2,
) -> List[Tuple[str, List[str]]]:
    """
    Generate feature augmentation combinations from top-K pipelines.

    Feature augmentation concatenates outputs from multiple preprocessings.

    Args:
        top_k_names: List of pipeline names from top-K selection.
        max_order: Maximum number of pipelines to combine (2 or 3).

    Returns:
        List of (name, component_names) tuples.

    Example:
        >>> names = ["snv", "d1", "msc"]
        >>> combos = generate_augmentation_combinations(names, max_order=2)
        >>> # Returns 2-way combinations like ("snv+d1", ["snv", "d1"])
    """
    results = []

    for order in range(2, min(max_order + 1, len(top_k_names) + 1)):
        for combo in combinations(top_k_names, order):
            name = "+".join(combo)
            results.append((name, list(combo)))

    return results


def apply_augmentation(
    X: np.ndarray,
    pipeline_names: List[str],
    preprocessings: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Apply multiple pipelines and concatenate their outputs.

    Args:
        X: Input data matrix (n_samples, n_features).
        pipeline_names: List of pipeline names to apply.
        preprocessings: Optional dictionary of transforms.

    Returns:
        Horizontally stacked transformed features.
    """
    if preprocessings is None:
        preprocessings = get_base_preprocessings()

    transformed = []
    for pp_name in pipeline_names:
        X_t = apply_stacked_pipeline(X, pp_name, preprocessings)
        transformed.append(X_t)

    return np.hstack(transformed)


def validate_datasets(
    X_source: np.ndarray,
    X_target: np.ndarray,
    require_same_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and prepare source/target datasets for transfer analysis.

    Args:
        X_source: Source dataset.
        X_target: Target dataset.
        require_same_features: If True, require same number of features.

    Returns:
        Tuple of validated (X_source, X_target) arrays.

    Raises:
        ValueError: If datasets have incompatible shapes.
    """
    X_source = np.asarray(X_source)
    X_target = np.asarray(X_target)

    if X_source.ndim != 2:
        raise ValueError(f"X_source must be 2D, got shape {X_source.shape}")
    if X_target.ndim != 2:
        raise ValueError(f"X_target must be 2D, got shape {X_target.shape}")

    if require_same_features and X_source.shape[1] != X_target.shape[1]:
        raise ValueError(
            f"Feature dimensions must match: source has {X_source.shape[1]}, "
            f"target has {X_target.shape[1]}"
        )

    if X_source.shape[0] < 3:
        raise ValueError(f"X_source needs at least 3 samples, got {X_source.shape[0]}")
    if X_target.shape[0] < 3:
        raise ValueError(f"X_target needs at least 3 samples, got {X_target.shape[0]}")

    return X_source, X_target


def format_pipeline_name(name: str, max_length: int = 30) -> str:
    """
    Format a pipeline name for display.

    Args:
        name: Pipeline name (e.g., "snv>d1>msc").
        max_length: Maximum length before truncation.

    Returns:
        Formatted name with arrows and potential truncation.
    """
    formatted = name.replace(">", "→").replace("+", " + ")
    if len(formatted) > max_length:
        formatted = formatted[:max_length - 3] + "..."
    return formatted
