"""Parity cases for repetitions + sample-level aggregation + rep_to_*.

These are the cases the dag-ml-data schema extensions (`MetadataSchema`,
`GroupSpec`, repetition invariants per ADR-05) must support. Without them
the bridge cannot reproduce nirs4all's per-sample aggregation semantics.

The fixtures `aggregate_mean` (E04) and `aggregate_outliers` (E05) ship
pre-configured repetition columns. Pure-corpus cases that synthesize reps
via `dataset_kwargs` use the `regression_2` corpus with a placeholder
metadata column — those carry `skip_reason` until the fixture's column
name is locked.
"""

from __future__ import annotations

from typing import Any

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, ShuffleSplit

from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC
from nirs4all.operators.transforms import StandardNormalVariate as SNV

from ._registry import PipelineCase, register

_AGG = frozenset({"aggregation", "slow"})
_REP = frozenset({"repetition", "slow"})


def _factory_baseline_for_aggregate() -> list[Any]:
    return [
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="aggregation_rep_mean",
        description="E04 aggregate-mean fixture: repetition column + mean aggregation across reps. "
        "Tests sample-level mean reducer parity (ADR-07 canonical reducer).",
        keywords=("model",),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
        ),
        dataset_key="aggregate_mean",
        pipeline_factory=_factory_baseline_for_aggregate,
        dataset_kwargs={
            "repetition": "sample_id",
            "aggregate": True,
            "aggregate_method": "mean",
        },
        expected_min_predictions=3,
        tags=_AGG,
    )
)


register(
    PipelineCase(
        name="aggregation_rep_median",
        description="E04 fixture with median aggregation method.",
        keywords=("model",),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
        ),
        dataset_key="aggregate_mean",
        pipeline_factory=_factory_baseline_for_aggregate,
        dataset_kwargs={
            "repetition": "sample_id",
            "aggregate": True,
            "aggregate_method": "median",
        },
        expected_min_predictions=3,
        tags=_AGG,
    )
)


register(
    PipelineCase(
        name="aggregation_rep_outlier_exclude",
        description="E05 aggregate-outliers fixture with aggregate_exclude_outliers=True. "
        "Tests robust aggregation with outlier exclusion (ADR-07 robust_mean / exclude_outliers).",
        keywords=("model",),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
        ),
        dataset_key="aggregate_outliers",
        pipeline_factory=_factory_baseline_for_aggregate,
        dataset_kwargs={
            "repetition": "sample_id",
            "aggregate": True,
            "aggregate_method": "mean",
            "aggregate_exclude_outliers": True,
        },
        expected_min_predictions=3,
        tags=_AGG,
    )
)


def _factory_classification_vote_aggregation() -> list[Any]:
    return [
        SNV(),
        KFold(n_splits=3, shuffle=True, random_state=42),
        {
            "model": RandomForestClassifier(
                n_estimators=20,
                max_depth=6,
                random_state=42,
                n_jobs=1,
            )
        },
    ]


register(
    PipelineCase(
        name="aggregation_classification_vote",
        description="Classification with majority-vote aggregation across repetitions. "
        "Tests the vote reducer for classification (ADR-07).",
        keywords=("model",),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "classification_model",
        ),
        dataset_key="aggregate_mean",
        pipeline_factory=_factory_classification_vote_aggregation,
        dataset_kwargs={
            "repetition": "sample_id",
            "aggregate": True,
            "aggregate_method": "vote",
            "task_type": "multiclass_classification",
        },
        task="classification",
        expected_min_predictions=3,
        tags=_AGG | frozenset({"classification"}),
        skip_reason="aggregate_mean fixture is regression-typed; needs a classification rep fixture.",
        skip_kind="fixture",
    )
)


def _factory_rep_to_sources() -> list[Any]:
    return [
        {"rep_to_sources": "sample_id"},
        SNV(),
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="rep_to_sources_basic",
        description="rep_to_sources: each repetition group becomes a separate source. "
        "Then SNV applied per generated source → PLSR.",
        keywords=("rep_to_sources", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "multi_source",
        ),
        dataset_key="aggregate_mean",
        pipeline_factory=_factory_rep_to_sources,
        dataset_kwargs={"repetition": "sample_id"},
        expected_min_predictions=3,
        tags=_REP,
    )
)


def _factory_rep_to_pp() -> list[Any]:
    return [
        {"rep_to_pp": "sample_id"},
        {"_or_": [SNV, MSC]},
        ShuffleSplit(n_splits=3, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]


register(
    PipelineCase(
        name="rep_to_pp_basic",
        description="rep_to_pp: each repetition becomes a preprocessing variant. "
        "Combined with `_or_` over preprocessing choices.",
        keywords=("rep_to_pp", "_or_", "model"),
        capabilities=(
            "preprocessing_transform",
            "cross_validator",
            "sklearn_model",
            "regression_model",
            "generator",
        ),
        dataset_key="aggregate_mean",
        pipeline_factory=_factory_rep_to_pp,
        dataset_kwargs={"repetition": "sample_id"},
        expected_min_predictions=3,
        tags=_REP | frozenset({"generator"}),
    )
)
