"""
Systematic Preprocessing Selection Package
==========================================

A comprehensive, systematic evaluation of preprocessing pipelines:

1. Stage 1 - Exhaustive Unsupervised Evaluation:
   - All single preprocessings (depth 1)
   - All stacked pipelines depth 2 (A → B)
   - All stacked pipelines depth 3 (A → B → C)
   - Compute unsupervised metrics for each
   - Output: CSV + visualization

2. Stage 2 - Diversity Analysis:
   - Take top N candidates from Stage 1
   - Compute pairwise distances using 6 metrics:
     * Subspace-based: Grassmann, CKA, RV coefficient
     * Geometry-based: Procrustes, Trustworthiness, Covariance
   - Produce two rankings (subspace diversity vs geometry diversity)
   - Filter out pipelines too similar to better-scored ones

3. Stage 3 - Proxy Model Evaluation:
   - Evaluate all top candidates with Ridge/KNN proxy models
   - Take top_k best performers

4. Stage 4 - Augmentation Evaluation:
   - Generate concatenations from Stage 3 top_k (2nd and 3rd order)
   - Evaluate with proxy models
   - Extend final list with top_k augmentations
   - Result: 2x top_k transformations

5. Final Ranking:
   - Combined ranking of all strategies
"""

from .data_classes import DiversityAnalysis, PipelineResult
from .cache import ResultCache, compute_config_hash, compute_data_hash
from .metrics import (
    compute_pca_metrics,
    compute_snr,
    compute_roughness,
    compute_separation,
    evaluate_unsupervised,
)
from .distances import (
    compute_all_distances,
    compute_cka_distance,
    compute_covariance_distance,
    compute_grassmann_distance,
    compute_procrustes_distance,
    compute_rv_distance,
    compute_trustworthiness_distance,
)
from .pipelines import (
    get_base_preprocessings,
    apply_pipeline,
    apply_augmentation,
    generate_stacked_pipelines,
)
from .proxy import evaluate_with_proxies
from .selector import SystematicSelector, compute_combinatorics, print_combinatorics_overview
from .visualization import (
    plot_results,
    plot_metrics_heatmap,
    plot_distance_heatmap,
    plot_dual_distance_heatmap,
)
from .data_loader import load_data

__all__ = [
    # Data classes
    "DiversityAnalysis",
    "PipelineResult",
    # Caching
    "ResultCache",
    "compute_config_hash",
    "compute_data_hash",
    # Metrics
    "compute_pca_metrics",
    "compute_snr",
    "compute_roughness",
    "compute_separation",
    "evaluate_unsupervised",
    # Distances
    "compute_all_distances",
    "compute_cka_distance",
    "compute_covariance_distance",
    "compute_grassmann_distance",
    "compute_procrustes_distance",
    "compute_rv_distance",
    "compute_trustworthiness_distance",
    # Pipelines
    "get_base_preprocessings",
    "apply_pipeline",
    "apply_augmentation",
    "generate_stacked_pipelines",
    # Evaluation
    "evaluate_with_proxies",
    "SystematicSelector",
    "compute_combinatorics",
    "print_combinatorics_overview",
    # Visualization
    "plot_results",
    "plot_metrics_heatmap",
    "plot_distance_heatmap",
    "plot_dual_distance_heatmap",
    # Data
    "load_data",
]
