"""
Transfer Analysis Module.

This module provides tools for transfer-optimized preprocessing selection
in spectroscopy applications. It helps find preprocessing methods that
minimize distributional distance between source and target datasets
while preserving predictive information.

Main Components:
    TransferPreprocessingSelector: Main class for preprocessing selection.
    TransferResult: Result from evaluating a single preprocessing.
    TransferSelectionResults: Full results with ranking and visualization.
    TransferMetrics: Container for computed transfer metrics.
    TransferMetricsComputer: Fast computation of transfer metrics.

Presets:
    PRESETS: Dictionary of preset configurations.
    get_preset: Get a preset by name.
    list_presets: List available presets with descriptions.

Utilities:
    compute_transfer_score: Compute composite transfer score.
    get_base_preprocessings: Get default preprocessing transforms.
    apply_pipeline: Apply a sequence of transforms.
    apply_stacked_pipeline: Apply stacked preprocessing by name.
    apply_augmentation: Apply and concatenate multiple preprocessings.

Example:
    >>> from nirs4all.analysis import TransferPreprocessingSelector
    >>> selector = TransferPreprocessingSelector(preset='balanced')
    >>> results = selector.fit(X_source, X_target)
    >>> print(results.best.name)
    'snv>d1'
    >>> print(results.to_pipeline_spec())
    'snv>d1'

    >>> # With visualization
    >>> results.plot_ranking()
    >>> results.plot_metrics_comparison()

    >>> # Export to pipeline spec
    >>> spec = results.to_pipeline_spec(top_k=3, use_augmentation=True)
    >>> # {'feature_augmentation': ['snv', 'd1', 'msc']}
"""

# Presets
from nirs4all.analysis.presets import PRESETS, get_preset, list_presets

# Results
from nirs4all.analysis.results import TransferResult, TransferSelectionResults

# Metrics
from nirs4all.analysis.transfer_metrics import (
    TransferMetrics,
    TransferMetricsComputer,
    compute_transfer_score,
)

# Utilities
from nirs4all.analysis.transfer_utils import (
    apply_augmentation,
    apply_pipeline,
    apply_preprocessing_objects,
    apply_single_preprocessing,
    apply_stacked_pipeline,
    format_pipeline_name,
    generate_augmentation_combinations,
    generate_object_augmentation_combinations,
    generate_object_stacked_pipelines,
    generate_stacked_pipelines,
    generate_top_k_stacked_pipelines,
    get_base_preprocessings,
    get_transform_name,
    get_transform_signature,
    normalize_preprocessing,
    normalize_preprocessing_list,
    validate_datasets,
)

# Main selector
from nirs4all.analysis.selector import TransferPreprocessingSelector

__all__ = [
    # Main class
    "TransferPreprocessingSelector",
    # Results
    "TransferResult",
    "TransferSelectionResults",
    # Metrics
    "TransferMetrics",
    "TransferMetricsComputer",
    "compute_transfer_score",
    # Presets
    "PRESETS",
    "get_preset",
    "list_presets",
    # Utilities
    "get_base_preprocessings",
    "get_transform_name",
    "get_transform_signature",
    "normalize_preprocessing",
    "normalize_preprocessing_list",
    "apply_pipeline",
    "apply_preprocessing_objects",
    "apply_single_preprocessing",
    "apply_stacked_pipeline",
    "apply_augmentation",
    "generate_stacked_pipelines",
    "generate_top_k_stacked_pipelines",
    "generate_augmentation_combinations",
    "generate_object_stacked_pipelines",
    "generate_object_augmentation_combinations",
    "validate_datasets",
    "format_pipeline_name",
]
