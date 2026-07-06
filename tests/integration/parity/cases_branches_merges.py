"""Parity cases for `branch` and `merge` — the most error-prone DSL surface.

Duplication branches (same samples, parallel pipelines) and separation
branches (disjoint sample groups) compile to fundamentally different DAG
shapes in the future dag-ml bridge, so each shape needs its own case.

Merge modes covered:
- `"predictions"` — stacking; OOF predictions become features for the meta-model.
- `"features"` — feature concatenation across branches.
- `"all"` — both predictions and features.
- `"concat"` — sample reassembly for separation branches.

`by_source` lives in `cases_multi_source.py` because it requires the
multi-source corpus.
"""

from __future__ import annotations

from typing import Any

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, ShuffleSplit

from nirs4all.operators.filters import YOutlierFilter
from nirs4all.operators.models import MetaModel
from nirs4all.operators.models.meta import CoverageStrategy, StackingConfig
from nirs4all.operators.transforms import Detrend, FirstDerivative
from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC
from nirs4all.operators.transforms import StandardNormalVariate as SNV

from ._registry import PipelineCase, register

_BRANCH = frozenset({"branches", "slow"})
_MERGE = frozenset({"merges", "slow"})


# -- Duplication branches ------------------------------------------------------


def _factory_dup_three_way_predictions() -> list[Any]:
    return [
        ShuffleSplit(n_splits=3, random_state=42),
        {
            "branch": {
                "snv_plsr": [SNV(), {"model": PLSRegression(n_components=10)}],
                "msc_rf": [
                    MSC(),
                    {"model": RandomForestRegressor(n_estimators=20, max_depth=6, random_state=42)},
                ],
                "fd_gbr": [
                    FirstDerivative(),
                    {"model": GradientBoostingRegressor(n_estimators=20, max_depth=4, random_state=42)},
                ],
            }
        },
        {"merge": "predictions"},
        {"model": Ridge(alpha=1.0, random_state=42)},
    ]


register(
    PipelineCase(
        name="branch_dup_three_way_merge_predictions",
        description="3 duplication branches (SNV+PLSR / MSC+RF / FirstDeriv+GBR) → merge predictions "
        "→ Ridge meta-learner. Classic stacking pattern.",
        keywords=("branch", "merge", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "stacking_meta_model",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_dup_three_way_predictions,
        expected_min_predictions=3,
        tags=_BRANCH | _MERGE,
    )
)


def _factory_dup_two_way_features() -> list[Any]:
    return [
        ShuffleSplit(n_splits=3, random_state=42),
        {
            "branch": {
                "snv": [SNV()],
                "msc": [MSC()],
            }
        },
        {"merge": "features"},
        {"model": PLSRegression(n_components=15)},
    ]


register(
    PipelineCase(
        name="branch_dup_two_way_merge_features",
        description="2 duplication branches (SNV / MSC) → merge features (concatenated views) → PLSR(15). "
        "Tests feature-level fusion across duplication branches.",
        keywords=("branch", "merge", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_dup_two_way_features,
        expected_min_predictions=3,
        tags=_BRANCH | _MERGE,
    )
)


def _factory_dup_named_with_metamodel() -> list[Any]:
    return [
        KFold(n_splits=3, shuffle=True, random_state=42),
        {
            "branch": {
                "pls_latent": [
                    SNV(),
                    {"concat_transform": [PCA(n_components=10), TruncatedSVD(n_components=10)]},
                    {"name": "PLS_Latent", "model": PLSRegression(n_components=10)},
                ],
                "rf_path": [
                    MSC(),
                    {"name": "RF", "model": RandomForestRegressor(n_estimators=20, max_depth=6, random_state=42)},
                ],
            }
        },
        {
            "name": "Ridge_MetaModel",
            "model": MetaModel(
                model=Ridge(alpha=1.0, random_state=42),
                stacking_config=StackingConfig(
                    coverage_strategy=CoverageStrategy.DROP_INCOMPLETE,
                    min_coverage_ratio=0.95,
                ),
            ),
        },
        {"merge": {"predictions": [
            {"branch": 0, "select": "best", "metric": "rmse"},
            {"branch": 1, "select": "best", "metric": "rmse"},
        ], "output_as": "features"}},
        {"model": Ridge(alpha=1.0, random_state=42)},
    ]


register(
    PipelineCase(
        name="branch_dup_named_with_metamodel",
        description="Named duplication branches (pls_latent / rf_path) with MetaModel + structured merge "
        "(per-branch best-by-rmse selector). Exercises `concat_transform`, `name`, MetaModel stacking.",
        keywords=("branch", "merge", "model", "concat_transform", "name"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "stacking_meta_model",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_dup_named_with_metamodel,
        expected_min_predictions=3,
        tags=_BRANCH | _MERGE | frozenset({"meta_model"}),
    )
)


# -- Separation branches -------------------------------------------------------


def _factory_sep_by_metadata_auto_discover() -> list[Any]:
    return [
        ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
        {
            "branch": {
                "by_metadata": "group",
                "steps": [SNV()],  # auto-discover unique values, same preproc per branch
            }
        },
        {"merge": "concat"},
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="branch_separation_by_metadata_auto",
        description="Separation branch by metadata column 'group' with auto-discovered values + "
        "shared SNV preprocessing → concat merge → PLSR. Tests dynamic-cardinality separation.",
        keywords=("branch", "merge", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
        ),
        dataset_key="with_metadata",
        pipeline_factory=_factory_sep_by_metadata_auto_discover,
        expected_min_predictions=3,
        tags=_BRANCH | _MERGE | frozenset({"separation"}),
    )
)


def _factory_sep_by_tag() -> list[Any]:
    return [
        {"tag": YOutlierFilter(method="zscore", threshold=2.0, tag_name="y_z_outlier")},
        ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
        {
            "branch": {
                "by_tag": "y_z_outlier",
                "steps": {
                    # Per `examples/developer/01_advanced_pipelines/D06_separation_branches.py`,
                    # by_tag keys are Python bools, not strings.
                    True: [MSC()],
                    False: [SNV()],
                },
            }
        },
        {"merge": "concat"},
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="branch_separation_by_tag",
        description="Tag samples by Y-outlier z-score, then separation branch by tag (True vs False) "
        "with different preprocessing per branch → concat merge → PLSR.",
        keywords=("tag", "branch", "merge", "model"),
        capabilities=(
            "filter",
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_sep_by_tag,
        expected_min_predictions=3,
        tags=_BRANCH | _MERGE | frozenset({"separation", "tag"}),
    )
)


def _factory_sep_by_filter() -> list[Any]:
    return [
        ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
        {
            "branch": {
                "by_filter": YOutlierFilter(method="iqr", threshold=2.5),
                "steps": {
                    "passing": [SNV()],
                    "failing": [MSC()],
                },
            }
        },
        {"merge": "concat"},
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="branch_separation_by_filter",
        description="Separation branch using YOutlierFilter (IQR) as the partition function "
        "(passing vs failing) → distinct preprocessing per branch → concat merge → PLSR.",
        keywords=("branch", "merge", "model"),
        capabilities=(
            "filter",
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_sep_by_filter,
        expected_min_predictions=3,
        tags=_BRANCH | _MERGE | frozenset({"separation", "filter"}),
    )
)


# -- Merge "all" mode ----------------------------------------------------------


def _factory_merge_all() -> list[Any]:
    return [
        ShuffleSplit(n_splits=3, random_state=42),
        {
            "branch": {
                "snv_plsr": [SNV(), {"model": PLSRegression(n_components=10)}],
                "msc_pls": [MSC(), {"model": PLSRegression(n_components=8)}],
            }
        },
        {"merge": "all"},
        {"model": Ridge(alpha=1.0, random_state=42)},
    ]


register(
    PipelineCase(
        name="branch_dup_merge_all",
        description="2 duplication branches → merge mode 'all' (features + predictions) → Ridge meta. "
        "Tests the combined-fusion merge path.",
        keywords=("branch", "merge", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "stacking_meta_model",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_merge_all,
        expected_min_predictions=3,
        tags=_BRANCH | _MERGE,
    )
)
