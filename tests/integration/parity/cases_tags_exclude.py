"""Parity cases for `tag`, `exclude`, explicit `preprocessing`, `fit_on_all`, `force_layout`.

`tag` marks samples without removing them; `exclude` removes them from
training (and keeps them for prediction). With multiple filters the
`mode: "any"|"all"` selector controls union vs intersection of masks —
that decision-table is exactly what the dag-ml bridge has to encode in
ADR-04.

`fit_on_all`, `force_layout`, and the explicit `preprocessing` keyword
cover the workflow-keyword surface that simple baseline cases don't
exercise because they pass operator instances directly.
"""

from __future__ import annotations

from typing import Any

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit

from nirs4all.operators.filters import XOutlierFilter, YOutlierFilter
from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC
from nirs4all.operators.transforms import StandardNormalVariate as SNV

from ._registry import PipelineCase, register

_TAG_EX = frozenset({"tag_exclude", "slow"})


def _factory_tag_only() -> list[Any]:
    return [
        {"tag": YOutlierFilter(method="iqr", threshold=2.5, tag_name="y_iqr_outlier")},
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="tag_only_no_removal",
        description="Tag samples with `y_iqr_outlier` via YOutlierFilter (no removal), train PLSR on all rows. "
        "Tag persists in metadata for downstream analysis.",
        keywords=("tag", "model"),
        capabilities=(
            "filter",
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_tag_only,
        expected_min_predictions=3,
        tags=_TAG_EX,
    )
)


def _factory_exclude_single() -> list[Any]:
    return [
        {"exclude": YOutlierFilter(method="zscore", threshold=3.0)},
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="exclude_single_y_zscore",
        description="Exclude outliers per Y z-score (single filter, default mode) → SNV → PLSR.",
        keywords=("exclude", "model"),
        capabilities=(
            "filter",
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_exclude_single,
        expected_min_predictions=3,
        tags=_TAG_EX,
    )
)


def _factory_exclude_multi_any() -> list[Any]:
    return [
        {
            "exclude": [
                YOutlierFilter(method="zscore", threshold=5.0),
                XOutlierFilter(method="mahalanobis", threshold=5.0),
            ],
            "mode": "any",
        },
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        # n_components kept low so this passes after the union of both filters
        # has trimmed the sample/feature space.
        {"model": PLSRegression(n_components=2)},
    ]


register(
    PipelineCase(
        name="exclude_multi_any_y_and_x",
        description="Exclude samples flagged by EITHER Y zscore OR X Mahalanobis outlier filter (union, mode='any'). "
        "Exercises the multi-filter union path.",
        keywords=("exclude", "model"),
        capabilities=(
            "filter",
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_exclude_multi_any,
        expected_min_predictions=3,
        tags=_TAG_EX,
    )
)


def _factory_exclude_multi_all() -> list[Any]:
    return [
        {
            "exclude": [
                YOutlierFilter(method="zscore", threshold=2.5),
                XOutlierFilter(method="mahalanobis", threshold=2.5),
            ],
            "mode": "all",
        },
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="exclude_multi_all_y_and_x",
        description="Exclude samples flagged by BOTH Y zscore AND X Mahalanobis (intersection, mode='all'). "
        "Stricter exclusion: only consensus outliers are removed.",
        keywords=("exclude", "model"),
        capabilities=(
            "filter",
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_exclude_multi_all,
        expected_min_predictions=3,
        tags=_TAG_EX,
    )
)


def _factory_tag_then_exclude() -> list[Any]:
    return [
        {"tag": YOutlierFilter(method="iqr", threshold=2.0, tag_name="y_iqr_outlier")},
        {"exclude": YOutlierFilter(method="zscore", threshold=4.0)},
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="tag_then_exclude_combo",
        description="Tag moderate outliers (kept), exclude extreme outliers (removed). "
        "Tests independence of tag and exclude operations.",
        keywords=("tag", "exclude", "model"),
        capabilities=(
            "filter",
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_tag_then_exclude,
        expected_min_predictions=3,
        tags=_TAG_EX,
    )
)


def _factory_preprocessing_explicit() -> list[Any]:
    return [
        {"preprocessing": SNV()},
        {"preprocessing": MSC()},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="preprocessing_explicit_keyword",
        description="Two `{'preprocessing': ...}` explicit keyword steps before model — verifies that the "
        "preprocessing keyword is honored even when transformer instances are present without it.",
        keywords=("preprocessing", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_preprocessing_explicit,
        expected_min_predictions=3,
        tags=frozenset({"keywords", "fast"}),
    )
)


def _factory_fit_on_all() -> list[Any]:
    return [
        {"preprocessing": SNV(), "fit_on_all": True},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="preprocessing_fit_on_all",
        description="`fit_on_all: True` forces preprocessing to fit on train + val + test together "
        "(leakage-aware semantics — must be reproduced verbatim by the bridge).",
        keywords=("preprocessing", "fit_on_all", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_fit_on_all,
        expected_min_predictions=3,
        tags=frozenset({"keywords", "fast", "leakage_aware"}),
    )
)


def _factory_force_layout() -> list[Any]:
    return [
        {"preprocessing": SNV(), "force_layout": "2d"},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="preprocessing_force_layout_2d",
        description="`force_layout: '2d'` pins the input tensor shape going into the preprocessor. "
        "Exercises the layout-coercion path.",
        keywords=("preprocessing", "force_layout", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
        ),
        dataset_key="regression",
        pipeline_factory=_factory_force_layout,
        expected_min_predictions=3,
        tags=frozenset({"keywords", "fast"}),
    )
)
