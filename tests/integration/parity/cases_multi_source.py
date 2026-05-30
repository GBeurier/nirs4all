"""Parity cases for multi-source pipelines.

`sample_data/multi` is a 3-source corpus (Xcal_1, Xcal_2, Xcal_3) sharing
targets — perfect for exercising the per-source dispatch, by_source
separation branches, and source-level merging that the dag-ml bridge must
preserve via dag-ml-data's typed multi-source data plan.
"""

from __future__ import annotations

from typing import Any

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, ShuffleSplit

from nirs4all.operators.transforms import FirstDerivative
from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC
from nirs4all.operators.transforms import StandardNormalVariate as SNV

from ._registry import PipelineCase, register

_MULTI = frozenset({"multi_source", "slow"})


def _factory_multi_source_baseline() -> list[Any]:
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="multi_source_baseline_snv_plsr",
        description="3-source NIR corpus + SNV (applied per source by default) + ShuffleSplit(3) + PLSR(10). "
        "Default per-source dispatch parity.",
        keywords=("model",),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "multi_source",
        ),
        dataset_key="multi",
        pipeline_factory=_factory_multi_source_baseline,
        expected_min_predictions=3,
        tags=_MULTI,
    )
)


def _factory_multi_source_per_source_branch_shared() -> list[Any]:
    return [
        ShuffleSplit(n_splits=3, random_state=42),
        {
            "branch": {
                "by_source": True,
                "steps": [SNV()],  # same preproc per source
            }
        },
        {"merge": "concat"},
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="multi_source_by_source_branch_shared_preproc",
        description="by_source branch with shared SNV preprocessing across all sources → concat merge → PLSR(10). "
        "Tests dynamic per-source branching with auto-discovered sources.",
        keywords=("branch", "merge", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "multi_source",
        ),
        dataset_key="multi",
        pipeline_factory=_factory_multi_source_per_source_branch_shared,
        expected_min_predictions=3,
        tags=_MULTI | frozenset({"separation"}),
    )
)


def _factory_multi_source_per_source_distinct() -> list[Any]:
    return [
        ShuffleSplit(n_splits=3, random_state=42),
        {
            "branch": {
                "by_source": True,
                "steps": {
                    # sample_data/multi exposes sources as `source_0`, `source_1`, `source_2`
                    # (verified against `examples/developer/01_advanced_pipelines/D06_separation_branches.py`).
                    "source_0": [SNV()],
                    "source_1": [MSC()],
                    "source_2": [FirstDerivative()],
                },
            }
        },
        {"merge": "concat"},
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="multi_source_by_source_branch_distinct_preproc",
        description="by_source branch with DIFFERENT preprocessing per source (SNV / MSC / FirstDerivative) → "
        "concat merge → PLSR(10).",
        keywords=("branch", "merge", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "multi_source",
        ),
        dataset_key="multi",
        pipeline_factory=_factory_multi_source_per_source_distinct,
        expected_min_predictions=3,
        tags=_MULTI | frozenset({"separation"}),
    )
)


def _factory_multi_source_models_per_source_stacking() -> list[Any]:
    return [
        KFold(n_splits=3, shuffle=True, random_state=42),
        {
            "branch": {
                "by_source": True,
                "steps": [
                    SNV(),
                    {"model": PLSRegression(n_components=10)},
                ],
            }
        },
        {"merge": "predictions"},
        {"model": Ridge(alpha=1.0, random_state=42)},
    ]


register(
    PipelineCase(
        name="multi_source_per_source_models_stacking",
        description="by_source branch trains a PLSR per source, merges via OOF predictions, "
        "Ridge meta-model stacks the per-source predictions.",
        keywords=("branch", "merge", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "multi_source",
            "stacking_meta_model",
        ),
        dataset_key="multi",
        pipeline_factory=_factory_multi_source_models_per_source_stacking,
        expected_min_predictions=3,
        tags=_MULTI | frozenset({"separation"}),
    )
)


def _factory_multi_source_unified_sources_merge() -> list[Any]:
    return [
        SNV(),
        {"merge": {"sources": "concat"}},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": RandomForestRegressor(n_estimators=20, max_depth=8, random_state=42, n_jobs=1)},
    ]


register(
    PipelineCase(
        name="multi_source_sources_concat_then_rf",
        description="SNV per source → unified merge `{'sources': 'concat'}` → RandomForest. "
        "Tests the source-fusion-before-model path.",
        keywords=("merge", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "multi_source",
        ),
        dataset_key="multi",
        pipeline_factory=_factory_multi_source_unified_sources_merge,
        expected_min_predictions=3,
        tags=_MULTI,
    )
)
