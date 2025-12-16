"""
Transfer Selection Utilities.

This module provides utility functions for preprocessing application,
pipeline generation, and dataset handling in transfer learning scenarios.

Supports both object-based and string-based preprocessing definitions:
- Object-based (recommended): Pass transformer instances directly
- String-based (legacy): Use string names that resolve to base preprocessings
"""

from copy import deepcopy
from itertools import combinations, permutations
from typing import Any, Dict, List, Optional, Tuple, Union

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


# Type alias for preprocessing items (object or string)
PreprocessingItem = Union[Any, str, None]


def get_transform_name(obj: Any) -> str:
    """
    Get a readable name from a transformer object.

    Args:
        obj: Transformer instance or string.

    Returns:
        Human-readable name for the transform.

    Example:
        >>> get_transform_name(StandardNormalVariate())
        'StandardNormalVariate'
        >>> get_transform_name(SavitzkyGolay(window_length=15))
        'SavitzkyGolay'
    """
    if obj is None:
        return "None"
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return ">".join(get_transform_name(t) for t in obj)
    return type(obj).__name__


def get_transform_signature(obj: Any) -> str:
    """
    Get a unique signature for a transformer (for deduplication).

    Includes class name and parameters if available.

    Args:
        obj: Transformer instance.

    Returns:
        Unique signature string.

    Example:
        >>> get_transform_signature(SavitzkyGolay(window_length=15))
        'SavitzkyGolay(polyorder=3,window_length=15)'
    """
    if obj is None:
        return "None"
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return ">".join(get_transform_signature(t) for t in obj)

    name = type(obj).__name__
    # Try to get sklearn-style params
    if hasattr(obj, "get_params"):
        try:
            params = obj.get_params(deep=False)
            # Sort params for consistent ordering
            param_str = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
            return f"{name}({param_str})"
        except Exception:
            pass
    return name


def normalize_preprocessing(
    item: PreprocessingItem,
    registry: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Normalize a preprocessing item to a transformer object.

    Handles both object instances and string names (for backward compat).

    Args:
        item: Transformer instance, string name, or None.
        registry: Optional name->object mapping for string resolution.

    Returns:
        Transformer instance (or None).

    Raises:
        ValueError: If string name not found in registry.

    Example:
        >>> normalize_preprocessing(StandardNormalVariate())
        StandardNormalVariate()
        >>> normalize_preprocessing("snv")  # looks up in base preprocessings
        StandardNormalVariate()
    """
    if item is None:
        return None

    # Already an object - return as-is
    if not isinstance(item, str):
        return item

    # String name - resolve from registry
    if registry is None:
        registry = get_base_preprocessings()

    if item not in registry:
        raise ValueError(
            f"Unknown preprocessing name: '{item}'. "
            f"Available: {list(registry.keys())}. "
            f"Consider using transformer objects directly instead of strings."
        )

    return registry[item]


def normalize_preprocessing_list(
    items: List[PreprocessingItem],
    registry: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """
    Normalize a list of preprocessing items to transformer objects.

    Args:
        items: List of transformer instances or string names.
        registry: Optional name->object mapping.

    Returns:
        List of transformer instances (None values filtered out).
    """
    if registry is None:
        registry = get_base_preprocessings()

    result = []
    for item in items:
        normalized = normalize_preprocessing(item, registry)
        if normalized is not None:
            result.append(normalized)
    return result


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
        if t is None:
            continue
        t_copy = deepcopy(t)
        X_out = t_copy.fit_transform(X_out)
    return X_out


def apply_preprocessing_objects(
    X: np.ndarray,
    transforms: Union[Any, List[Any]],
) -> np.ndarray:
    """
    Apply preprocessing object(s) to X.

    This is the primary function for object-based preprocessing.

    Args:
        X: Input data matrix (n_samples, n_features).
        transforms: Single transformer or list of transformers.

    Returns:
        Transformed data matrix.

    Example:
        >>> from nirs4all.operators.transforms import StandardNormalVariate, FirstDerivative
        >>> X_t = apply_preprocessing_objects(X, [StandardNormalVariate(), FirstDerivative()])
        >>> X_t = apply_preprocessing_objects(X, StandardNormalVariate())  # single
    """
    if transforms is None:
        return X.copy()

    if not isinstance(transforms, list):
        transforms = [transforms]

    return apply_pipeline(X, transforms)


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
    pipeline: Union[str, List[Any]],
    preprocessings: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Apply a stacked pipeline to X.

    Supports both:
    - Object-based: List of transformer instances
    - String-based (legacy): Pipeline name with ">" separator (e.g., "snv>d1>msc")

    Args:
        X: Input data matrix (n_samples, n_features).
        pipeline: Either a list of transformer objects or a string name.
        preprocessings: Optional dictionary of transforms (for string resolution).

    Returns:
        Transformed data matrix.

    Example:
        >>> # Object-based (recommended)
        >>> apply_stacked_pipeline(X, [StandardNormalVariate(), FirstDerivative()])
        >>> # String-based (legacy)
        >>> apply_stacked_pipeline(X, "snv>d1")
    """
    # Object-based: list of transforms
    if isinstance(pipeline, list):
        return apply_preprocessing_objects(X, pipeline)

    # String-based: parse and resolve names
    if preprocessings is None:
        preprocessings = get_base_preprocessings()

    component_names = pipeline.split(">")
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


def generate_object_stacked_pipelines(
    transforms: List[Any],
    max_depth: int = 2,
) -> List[Tuple[str, List[Any]]]:
    """
    Generate stacked pipeline combinations from transformer objects.

    Object-based alternative to generate_stacked_pipelines.

    Args:
        transforms: List of transformer objects.
        max_depth: Maximum pipeline depth.

    Returns:
        List of (display_name, transforms_list) tuples.

    Example:
        >>> transforms = [StandardNormalVariate(), FirstDerivative()]
        >>> pipelines = generate_object_stacked_pipelines(transforms, max_depth=2)
        >>> # Returns: [("StandardNormalVariate", [SNV()]),
        >>> #           ("FirstDerivative", [D1()]),
        >>> #           ("StandardNormalVariate>FirstDerivative", [SNV(), D1()]),
        >>> #           ...]
    """
    pipelines = []

    for depth in range(1, max_depth + 1):
        for combo in permutations(transforms, depth):
            combo_list = list(combo)
            name = ">".join(get_transform_name(t) for t in combo_list)
            pipelines.append((name, combo_list))

    return pipelines


def generate_object_augmentation_combinations(
    transforms: List[Any],
    max_order: int = 2,
) -> List[Tuple[str, List[Any]]]:
    """
    Generate augmentation combinations from transformer objects.

    Object-based alternative to generate_augmentation_combinations.

    Args:
        transforms: List of transformer objects or stacked lists.
        max_order: Maximum number of transforms to combine.

    Returns:
        List of (display_name, transforms_list) tuples.
    """
    results = []

    for order in range(2, min(max_order + 1, len(transforms) + 1)):
        for combo in combinations(transforms, order):
            combo_list = list(combo)
            name = "+".join(get_transform_name(t) for t in combo_list)
            results.append((name, combo_list))

    return results


def apply_augmentation(
    X: np.ndarray,
    pipelines: List[Union[str, List[Any], Any]],
    preprocessings: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Apply multiple pipelines and concatenate their outputs.

    Supports both object-based and string-based pipeline definitions.

    Args:
        X: Input data matrix (n_samples, n_features).
        pipelines: List of pipelines. Each can be:
            - A transformer object
            - A list of transformer objects (stacked)
            - A string name (legacy, resolved from preprocessings)
        preprocessings: Optional dictionary of transforms (for string resolution).

    Returns:
        Horizontally stacked transformed features.

    Example:
        >>> # Object-based (recommended)
        >>> apply_augmentation(X, [StandardNormalVariate(), [MSC(), FirstDerivative()]])
        >>> # String-based (legacy)
        >>> apply_augmentation(X, ["snv", "msc>d1"])
    """
    if preprocessings is None:
        preprocessings = get_base_preprocessings()

    transformed = []
    for pp in pipelines:
        if isinstance(pp, str):
            # String name - use stacked pipeline parser
            X_t = apply_stacked_pipeline(X, pp, preprocessings)
        elif isinstance(pp, list):
            # List of transforms - apply as stacked
            X_t = apply_preprocessing_objects(X, pp)
        else:
            # Single transform object
            X_t = apply_preprocessing_objects(X, pp)
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


def format_pipeline_name(name: str, max_length: int = 40) -> str:
    """
    Format a pipeline name for display.

    Args:
        name: Pipeline name (e.g., "snv>d1>msc").
        max_length: Maximum length before truncation.

    Returns:
        Formatted name with potential truncation.
    """
    formatted = name.replace("+", " + ")
    if len(formatted) > max_length:
        formatted = formatted[:max_length - 3] + "..."
    return formatted
