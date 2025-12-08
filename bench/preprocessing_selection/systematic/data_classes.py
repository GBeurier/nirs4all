"""Data classes for systematic preprocessing selection."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PipelineResult:
    """Result of a pipeline evaluation."""

    name: str
    depth: int
    pipeline_type: str  # 'single', 'stacked', 'augmented_2', 'augmented_3'
    components: List[str]
    X_transformed: Optional[np.ndarray] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    total_score: float = 0.0
    proxy_scores: Dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0
    # Stage 2 diversity metrics: {pipeline_comparison_key: {metric: value}}
    diversity_scores: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiversityAnalysis:
    """Results of Stage 2 diversity analysis.

    Contains multiple distance matrices for different metric types,
    allowing separate rankings based on subspace-based vs geometry-based
    diversity.
    """

    # Individual distance matrices (pipeline_name x pipeline_name)
    grassmann_matrix: Optional[np.ndarray] = None
    cka_matrix: Optional[np.ndarray] = None
    rv_matrix: Optional[np.ndarray] = None
    procrustes_matrix: Optional[np.ndarray] = None
    trustworthiness_matrix: Optional[np.ndarray] = None
    covariance_matrix: Optional[np.ndarray] = None

    # Combined matrices
    subspace_matrix: Optional[np.ndarray] = None  # Grassmann + CKA + RV
    geometry_matrix: Optional[np.ndarray] = None  # Procrustes + Trust + Cov
    combined_matrix: Optional[np.ndarray] = None  # All combined

    # Pipeline names for matrix indexing
    pipeline_names: List[str] = field(default_factory=list)

    # Diversity rankings (by different criteria)
    subspace_ranking: List[str] = field(default_factory=list)
    geometry_ranking: List[str] = field(default_factory=list)
    combined_ranking: List[str] = field(default_factory=list)
