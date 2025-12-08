"""
Preprocessing Selection Framework
==================================

A systematic framework for evaluating and selecting preprocessing techniques
for NIRS data analysis before running full ML/DL pipelines.

Features:
- Stage 1: Exhaustive unsupervised evaluation of all pipeline combinations
- Stage 2: Diversity analysis with 6 distance metrics (Grassmann, CKA, RV, Procrustes, etc.)
- Stage 3: Proxy model validation (Ridge + KNN) on diverse candidates
- Stage 4: Feature augmentation evaluation (2-way and 3-way concatenations)

Usage:
    python run_pp_selection.py [options]

    # Or programmatically:
    from preprocessing_selection.selector import PreprocessingSelector

    selector = PreprocessingSelector(verbose=1)
    results = selector.select(X, y, preprocessings)
"""

from .selector import PreprocessingSelector
from .metrics import (
    compute_snr,
    compute_roughness,
    pca_variance_filter,
    snr_filter,
    roughness_filter,
    distance_separation_filter,
    rv_coefficient,
    cka_score,
    correlation_score,
    pls_score,
    evaluate_unsupervised,
    evaluate_supervised,
)
from .proxy_models import ridge_proxy, knn_proxy, evaluate_proxies
from .combinations import mutual_info_redundancy, grassmann_distance

__all__ = [
    'PreprocessingSelector',
    'compute_snr',
    'compute_roughness',
    'pca_variance_filter',
    'snr_filter',
    'roughness_filter',
    'distance_separation_filter',
    'rv_coefficient',
    'cka_score',
    'correlation_score',
    'pls_score',
    'evaluate_unsupervised',
    'evaluate_supervised',
    'ridge_proxy',
    'knn_proxy',
    'evaluate_proxies',
    'mutual_info_redundancy',
    'grassmann_distance',
]
