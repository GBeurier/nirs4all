"""
Preprocessing Selection Framework
==================================

A systematic framework for evaluating and selecting preprocessing techniques
for NIRS data analysis.

Main Components:
- SystematicSelector: Complete pipeline for preprocessing selection
- Metrics: Unsupervised and supervised evaluation metrics
- Distance functions: Grassmann and CKA for diversity analysis

Usage:
    python systematic_selection.py [--depth 3] [--top 15] [--plots]

    # Or programmatically:
    from selection import SystematicSelector

    selector = SystematicSelector(verbose=1)
    results = selector.run_full_selection(X, y, max_depth=3, top_k=15)
"""

from .systematic_selection import (
    SystematicSelector,
    PipelineResult,
    compute_pca_metrics,
    compute_snr,
    compute_roughness,
    compute_separation,
    evaluate_unsupervised,
    compute_grassmann_distance,
    compute_cka_distance,
    evaluate_with_proxies,
    get_base_preprocessings,
    apply_pipeline,
    apply_augmentation,
    generate_stacked_pipelines,
    plot_results,
    load_data,
)

__all__ = [
    'SystematicSelector',
    'PipelineResult',
    'compute_pca_metrics',
    'compute_snr',
    'compute_roughness',
    'compute_separation',
    'evaluate_unsupervised',
    'compute_grassmann_distance',
    'compute_cka_distance',
    'evaluate_with_proxies',
    'get_base_preprocessings',
    'apply_pipeline',
    'apply_augmentation',
    'generate_stacked_pipelines',
    'plot_results',
    'load_data',
]
