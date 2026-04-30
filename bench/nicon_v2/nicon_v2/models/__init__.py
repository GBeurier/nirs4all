"""Model factories for nicon_v2 (V0 baselines + improved variants)."""

from .baseline import (
    PLSBaseline,
    RidgeBaseline,
    build_decon_torch,
    build_nicon_torch,
)
from .v1a_minimal_repair import (
    NiconV1a,
    NiconV1aActivationOnly,
    NiconV1aHeadOnly,
    build_nicon_v1a,
    build_nicon_v1a_activation_only,
    build_nicon_v1a_head_only,
)
from .v1b_concat_aug import NiconV1b, build_nicon_v1b
from .v1c_gap_backbone import NiconV1c, build_nicon_v1c
from .v2_aom_cnn import NiconV2A, build_nicon_v2a
from .stacking import StackedRegressor, StackingConfig
from .searched_baseline import SearchedPLS, SearchedRidge

__all__ = [
    "PLSBaseline",
    "RidgeBaseline",
    "build_decon_torch",
    "build_nicon_torch",
    "NiconV1a",
    "NiconV1aHeadOnly",
    "NiconV1aActivationOnly",
    "build_nicon_v1a",
    "build_nicon_v1a_head_only",
    "build_nicon_v1a_activation_only",
    "NiconV1b",
    "build_nicon_v1b",
    "NiconV1c",
    "build_nicon_v1c",
]
