"""
TabPFN Configuration Module
============================
Centralized configuration for TabPFN model variants, inference configs,
and hyperparameter search spaces for use with nirs4all pipelines.

This module provides:
- Model variant definitions (classifier and regressor checkpoints)
- Inference config builders with search space generation
- Utility functions for model path resolution
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path

# TabPFN imports - lazy loaded to avoid import errors when not installed
try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor  # type: ignore
    from tabpfn.model_loading import get_cache_dir  # type: ignore
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    TabPFNClassifier = None
    TabPFNRegressor = None

    def get_cache_dir():
        return Path(".")


# =============================================================================
# Model Variants
# =============================================================================

CLASSIFIER_MODELS = {
    'default': None,  # Uses default model (no model_path needed)
    'default-2': 'tabpfn-v2.5-classifier-v2.5_default-2.ckpt',
    'real': 'tabpfn-v2.5-classifier-v2.5_real.ckpt',
    'large-features-L': 'tabpfn-v2.5-classifier-v2.5_large-features-L.ckpt',
    'large-features-XL': 'tabpfn-v2.5-classifier-v2.5_large-features-XL.ckpt',
    'large-samples': 'tabpfn-v2.5-classifier-v2.5_large-samples.ckpt',
    'real-large-features': 'tabpfn-v2.5-classifier-v2.5_real-large-features.ckpt',
    'real-large-samples-and-features': 'tabpfn-v2.5-classifier-v2.5_real-large-samples-and-features.ckpt',
    'variant': 'tabpfn-v2.5-classifier-v2.5_variant.ckpt',
}

REGRESSOR_MODELS = {
    'default': None,  # Uses default model (no model_path needed)
    'real': 'tabpfn-v2.5-regressor-v2.5_real.ckpt',
    'low-skew': 'tabpfn-v2.5-regressor-v2.5_low-skew.ckpt',
    'quantiles': 'tabpfn-v2.5-regressor-v2.5_quantiles.ckpt',
    'real-variant': 'tabpfn-v2.5-regressor-v2.5_real-variant.ckpt',
    'small-samples': 'tabpfn-v2.5-regressor-v2.5_small-samples.ckpt',
    'variant': 'tabpfn-v2.5-regressor-v2.5_variant.ckpt',
}


# =============================================================================
# Model Path Utilities
# =============================================================================

def get_model_dict(task_type: str) -> dict:
    """Get the model dictionary for a given task type."""
    return CLASSIFIER_MODELS if task_type == 'classification' else REGRESSOR_MODELS


def get_model_class(task_type: str):
    return TabPFNClassifier if task_type == 'classification' else TabPFNRegressor


def get_model_paths(task_type: str, variants: Optional[list[str]] = None) -> list[Optional[str]]:
    """
    Get full paths to model checkpoints for specified variants.

    Args:
        task_type: 'regression' or 'classification'
        variants: List of variant names. If None, uses ['default', 'real']

    Returns:
        List of full paths to checkpoint files (None for default model)
    """
    model_dict = get_model_dict(task_type)

    if variants is None:
        variants = ['default', 'real']

    paths = []
    cache_dir = get_cache_dir()
    for variant in variants:
        if variant not in model_dict:
            raise ValueError(f"Unknown variant: {variant}. Available: {list(model_dict.keys())}")
        ckpt_name = model_dict[variant]
        if ckpt_name is None:
            paths.append(None)  # Default model
        else:
            paths.append(str(cache_dir / ckpt_name))
    return paths


def get_model_path_options(task_type: str, variants: Optional[list[str]] = None) -> list[str]:
    """
    Get model paths for Optuna categorical search (excludes None).

    Args:
        task_type: 'regression' or 'classification'
        variants: List of variant names

    Returns:
        List of valid checkpoint paths (None values filtered out)
    """
    paths = get_model_paths(task_type, variants)
    return [p for p in paths if p is not None]


def create_model(task_type: str, variant: str = 'default', device: str = 'cuda', **kwargs):
    """
    Create a TabPFN model instance.

    Args:
        task_type: 'regression' or 'classification'
        variant: Model variant name
        device: Device to use ('cuda' or 'cpu')
        **kwargs: Additional arguments passed to TabPFN constructor

    Returns:
        TabPFN model instance
    """
    model_class = get_model_class(task_type)
    model_paths = get_model_paths(task_type, [variant])
    model_path = model_paths[0]

    if model_path is not None:
        return model_class(model_path=model_path, device=device, **kwargs)
    else:
        return model_class(device=device, **kwargs)


# =============================================================================
# Inference Config Builders
# =============================================================================

# Preprocessing transform names (subset of full search space for efficiency)
PREPROCESS_NAMES = [
    "quantile_uni_coarse",
    "quantile_norm_coarse",
    "kdi_uni",
    "kdi_alpha_0.3",
    "kdi_alpha_3.0",
    "none",
    "safepower",
    "squashing_scaler_default",
]

PREPROCESS_NAMES_MINIMAL = [
    "quantile_uni_coarse",
    "kdi_uni",
    "none",
    "squashing_scaler_default",
]

GLOBAL_TRANSFORMERS = [None, "svd", "svd_quarter_components"]
GLOBAL_TRANSFORMERS_MINIMAL = [None, "svd"]

CATEGORICAL_NAMES = ["numeric", "ordinal_very_common_categories_shuffled", "none"]
CATEGORICAL_NAMES_MINIMAL = ["numeric"]


def build_preprocess_transform(
    name: str,
    global_transformer: Optional[str] = None,
    categorical_name: str = "numeric",
    append_original: bool = True,
) -> dict:
    """Build a single PREPROCESS_TRANSFORMS entry."""
    return {
        "name": name,
        "global_transformer_name": global_transformer,
        "categorical_name": categorical_name,
        "append_original": append_original,
    }


def build_inference_config(
    fingerprint_feature: bool = True,
    outlier_removal_std: Optional[float] = None,
    min_unique_for_numerical: int = 5,
    preprocess_transforms: Optional[list] = None,
    regression_y_preprocess: Optional[tuple] = None,
) -> dict:
    """
    Build a single inference_config dictionary.

    Args:
        fingerprint_feature: Whether to add fingerprint features
        outlier_removal_std: Outlier removal threshold (None, 7.0, or 12.0)
        min_unique_for_numerical: Min unique values for numerical treatment
        preprocess_transforms: List of preprocess transform dicts
        regression_y_preprocess: Y preprocessing for regression (None or ("safepower",))

    Returns:
        inference_config dict ready for TabPFN
    """
    config = {
        "FINGERPRINT_FEATURE": fingerprint_feature,
        "OUTLIER_REMOVAL_STD": outlier_removal_std,
        "POLYNOMIAL_FEATURES": "no",
        "MIN_UNIQUE_FOR_NUMERICAL_FEATURES": min_unique_for_numerical,
    }

    if preprocess_transforms is not None:
        config["PREPROCESS_TRANSFORMS"] = preprocess_transforms
    else:
        # Default preprocessing
        config["PREPROCESS_TRANSFORMS"] = [
            build_preprocess_transform("quantile_uni_coarse")
        ]

    if regression_y_preprocess is not None:
        config["REGRESSION_Y_PREPROCESS_TRANSFORMS"] = regression_y_preprocess

    return config


def generate_inference_configs(
    task_type: str = 'regression',
    mode: str = 'minimal',
) -> list[dict]:
    """
    Generate a list of inference_config options for Optuna categorical search.

    Args:
        task_type: 'regression' or 'classification'
        mode: 'minimal' (~48 configs), 'standard' (~192 configs), or 'full' (~800+ configs)

    Returns:
        List of inference_config dictionaries
    """
    configs = []

    # Select parameter ranges based on mode
    if mode == 'minimal':
        fingerprint_options = [True, False]
        outlier_std_options = [None, 7.0]
        preprocess_names = PREPROCESS_NAMES_MINIMAL
        global_transformers = GLOBAL_TRANSFORMERS_MINIMAL
        min_unique_options = [5]
        append_original_options = [True]
    elif mode == 'standard':
        fingerprint_options = [True, False]
        outlier_std_options = [None, 7.0, 12.0]
        preprocess_names = PREPROCESS_NAMES_MINIMAL
        global_transformers = GLOBAL_TRANSFORMERS_MINIMAL
        min_unique_options = [1, 5, 10]
        append_original_options = [True, False]
    else:  # full
        fingerprint_options = [True, False]
        outlier_std_options = [None, 7.0, 12.0]
        preprocess_names = PREPROCESS_NAMES
        global_transformers = GLOBAL_TRANSFORMERS
        min_unique_options = [1, 5, 10, 30]
        append_original_options = [True, False]

    # Y preprocessing for regression
    y_preprocess_options = [None, ("safepower",)] if task_type == 'regression' else [None]

    # Build combinations
    for fingerprint in fingerprint_options:
        for outlier_std in outlier_std_options:
            for preprocess_name in preprocess_names:
                for global_tf in global_transformers:
                    for append_orig in append_original_options:
                        for y_preprocess in y_preprocess_options:
                            config = build_inference_config(
                                fingerprint_feature=fingerprint,
                                outlier_removal_std=outlier_std,
                                min_unique_for_numerical=min_unique_options[0],
                                preprocess_transforms=[
                                    build_preprocess_transform(
                                        preprocess_name,
                                        global_tf,
                                        append_original=append_orig,
                                    )
                                ],
                                regression_y_preprocess=y_preprocess,
                            )
                            configs.append(config)

    return configs


# =============================================================================
# Finetune Params Builders
# =============================================================================

def build_model_params(
    task_type: str,
    model_variants: Optional[list[str]] = None,
    inference_config_mode: str = 'minimal',
    include_model_path: bool = True,
    include_inference_config: bool = True,
) -> dict:
    """
    Build model_params dict for nirs4all finetune_params.

    Args:
        task_type: 'regression' or 'classification'
        model_variants: Model variants to include in search
        inference_config_mode: 'minimal', 'standard', or 'full'
        include_model_path: Whether to include model_path in search
        include_inference_config: Whether to include inference_config in search

    Returns:
        model_params dict for finetune_params
    """
    params = {
        # Core TabPFN hyperparameters (from search_space.py)
        "softmax_temperature": [0.75, 0.8, 0.9, 0.95, 1.0, 1.05],
        "average_before_softmax": [True, False],
    }

    # Add model_path options
    if include_model_path and model_variants:
        model_path_options = get_model_path_options(task_type, model_variants)
        if model_path_options:
            params["model_path"] = model_path_options

    # Add inference_config options
    if include_inference_config:
        inference_configs = generate_inference_configs(task_type, inference_config_mode)
        params["inference_config"] = inference_configs

    return params


def build_finetune_params(
    task_type: str,
    n_trials: int = 50,
    verbose: int = 1,
    approach: str = 'grouped',
    eval_mode: str = 'avg',
    sample: str = 'tpe',
    model_variants: Optional[list[str]] = None,
    inference_config_mode: str = 'minimal',
) -> dict:
    """
    Build complete finetune_params dict for nirs4all pipeline.

    Args:
        task_type: 'regression' or 'classification'
        n_trials: Number of Optuna trials
        verbose: Verbosity level
        approach: 'single', 'grouped', or 'individual'
        eval_mode: 'best' or 'avg'
        sample: Sampler type ('tpe', 'grid', 'random')
        model_variants: Model variants to include
        inference_config_mode: 'minimal', 'standard', or 'full'

    Returns:
        Complete finetune_params dict
    """
    return {
        "n_trials": n_trials,
        "verbose": verbose,
        "approach": approach,
        "eval_mode": eval_mode,
        "sample": sample,
        "model_params": build_model_params(
            task_type,
            model_variants=model_variants,
            inference_config_mode=inference_config_mode,
        ),
    }
