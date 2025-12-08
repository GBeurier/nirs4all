"""Pipeline utilities for preprocessing."""

import sys
import os
from copy import deepcopy
from itertools import permutations
from typing import Any, Dict, List, Tuple

import numpy as np

# Add parent directories to path for nirs4all imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
_examples_dir = os.path.dirname(os.path.dirname(_this_dir))
_repo_dir = os.path.dirname(_examples_dir)
if _repo_dir not in sys.path:
    sys.path.insert(0, _repo_dir)

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
    """Get the base set of preprocessing transforms.

    Returns:
        Dictionary mapping names to transformer instances.
    """
    return {
        "snv": StandardNormalVariate(),
        "rsnv": RobustStandardNormalVariate(),
        "msc": MultiplicativeScatterCorrection(scale=False),
        "savgol": SavitzkyGolay(window_length=11, polyorder=3),
        "d1": FirstDerivative(),
        "d2": SecondDerivative(),
        "savgol_d1": SavitzkyGolay(window_length=11, polyorder=3, deriv=1),
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


def apply_pipeline(X: np.ndarray, transforms: List) -> np.ndarray:
    """Apply a sequence of transforms to X.

    Args:
        X: Input data matrix (n_samples, n_features).
        transforms: List of transformer instances.

    Returns:
        Transformed data matrix.
    """
    X_out = X.copy()
    for t in transforms:
        t_copy = deepcopy(t)
        X_out = t_copy.fit_transform(X_out)
    return X_out


def apply_augmentation(X: np.ndarray, transform_list: List[List]) -> np.ndarray:
    """Apply multiple pipelines and concatenate features.

    Args:
        X: Input data matrix (n_samples, n_features).
        transform_list: List of transform sequences to apply.

    Returns:
        Horizontally stacked transformed features.
    """
    transformed = []
    for transforms in transform_list:
        X_t = apply_pipeline(X, transforms)
        transformed.append(X_t)
    return np.hstack(transformed)


def generate_stacked_pipelines(
    preprocessings: Dict[str, Any], max_depth: int = 3
) -> List[Tuple[str, List[str], List]]:
    """Generate all stacked pipeline combinations.

    Args:
        preprocessings: Dictionary of available transforms.
        max_depth: Maximum pipeline depth (1 to max_depth).

    Returns:
        List of (name, component_names, transforms) tuples.
    """
    names = list(preprocessings.keys())
    pipelines = []

    for depth in range(1, max_depth + 1):
        for combo in permutations(names, depth):
            name = ">".join(combo)
            transforms = [preprocessings[n] for n in combo]
            pipelines.append((name, list(combo), transforms))

    return pipelines
