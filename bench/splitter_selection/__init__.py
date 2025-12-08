"""
Splitter Selection Framework
============================

A comprehensive framework for evaluating and comparing different data splitting
strategies for NIRS spectral data. The framework helps identify optimal splitting
approaches that maximize model generalization.

Features:
- 16+ splitting strategies (Random, Stratified, Kennard-Stone, SPXY, Duplex, etc.)
- Multi-model ensemble evaluation (Ridge, PLS, XGBoost, SVR, MLP)
- Bootstrap confidence intervals
- Representativeness metrics (spectral coverage, target distribution, leverage)
- Support for both classification and regression tasks

Usage:
    python run_splitter_selection.py --data_dir path/to/data --output_dir results/

    # Or programmatically:
    from splitter_selection.splitter_strategies import get_splitter

    splitter = get_splitter('kennard_stone')
    train_ids, test_ids = splitter.split(X, y, groups)
"""

from .splitter_strategies import (
    SPLITTING_STRATEGIES,
    get_splitter,
    list_strategies,
    SimpleSplitter,
    TargetStratifiedSplitter,
    SpectralPCASplitter,
    SpectralDistanceSplitter,
    HybridSplitter,
    AdversarialSplitter,
    KennardStoneSplitter,
    StratifiedGroupKFoldSplitter,
)

from .unsupervised_splitters import (
    PuchweinSplitter,
    DuplexSplitter,
    ShenkWestSplitter,
    HonigsSplitter,
    HierarchicalClusteringSplitter,
    KMedoidsSplitter,
)

__all__ = [
    'SPLITTING_STRATEGIES',
    'get_splitter',
    'list_strategies',
    'SimpleSplitter',
    'TargetStratifiedSplitter',
    'SpectralPCASplitter',
    'SpectralDistanceSplitter',
    'HybridSplitter',
    'AdversarialSplitter',
    'KennardStoneSplitter',
    'StratifiedGroupKFoldSplitter',
    'PuchweinSplitter',
    'DuplexSplitter',
    'ShenkWestSplitter',
    'HonigsSplitter',
    'HierarchicalClusteringSplitter',
    'KMedoidsSplitter',
]
