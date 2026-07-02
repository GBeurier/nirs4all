"""Regression locks for separation-branch parity cases."""

from __future__ import annotations

import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.random_projection import GaussianRandomProjection

import nirs4all
from nirs4all.operators.filters import YOutlierFilter
from nirs4all.operators.transforms import StandardNormalVariate as SNV
from nirs4all.pipeline.dagml.detect import _detect_separation_preproc_concat

from . import cases_branches_merges  # noqa: F401 - registers branch cases
from ._conformance_helpers import make_dataset
from ._registry import get

pytestmark = [pytest.mark.parity, pytest.mark.slow]


@pytest.mark.parametrize(
    "case_name",
    [
        "branch_separation_by_tag",
        "branch_separation_by_filter",
    ],
)
def test_branch_separation_cases_are_live_parity_cases(case_name: str) -> None:
    """The old legacy-bug skips stay removed once the DSL paths are fixed."""
    case = get(case_name)

    assert case.skip_reason == ""
    assert case.skip_kind == ""


def test_dagml_branch_by_tag_detection_requires_tag_before_branch() -> None:
    """A late tag step must not affect branch membership in the native projection."""
    pipeline = get("branch_separation_by_tag").pipeline
    assert _detect_separation_preproc_concat(pipeline) is not None

    late_tag_pipeline = [*pipeline[1:], pipeline[0]]
    assert _detect_separation_preproc_concat(late_tag_pipeline) is None


def test_dagml_branch_preproc_rejects_unseeded_random_transforms() -> None:
    """The host-side branch projection cannot claim parity for unseeded stochastic transforms."""
    pipeline = [
        ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
        {
            "branch": {
                "by_filter": YOutlierFilter(method="iqr", threshold=2.5),
                "steps": {
                    "passing": [GaussianRandomProjection(n_components=3)],
                    "failing": [SNV()],
                },
            }
        },
        {"merge": "concat"},
        {"model": PLSRegression(n_components=3)},
    ]

    assert _detect_separation_preproc_concat(pipeline) is None


@pytest.mark.parametrize(
    "case_name",
    [
        "branch_separation_by_tag",
        "branch_separation_by_filter",
    ],
)
def test_legacy_engine_runs_separation_branch_dsl(case_name: str) -> None:
    """The legacy engine remains the Python oracle for these branch DSL shapes."""
    case = get(case_name)

    result = nirs4all.run(
        pipeline=case.pipeline,
        dataset=make_dataset(case),
        verbose=0,
        engine="legacy",
    )

    assert not result._is_dagml_engine()  # noqa: SLF001
    assert result.num_predictions >= case.expected_min_predictions


@pytest.mark.parametrize(
    "case_name",
    [
        "branch_separation_by_tag",
        "branch_separation_by_filter",
    ],
)
def test_dagml_engine_runs_separation_branch_dsl_natively(case_name: str) -> None:
    """Native dag-ml owns these branch DSL shapes instead of falling back/skipping."""
    case = get(case_name)

    result = nirs4all.run(
        pipeline=case.pipeline,
        dataset=make_dataset(case),
        verbose=0,
        engine="dag-ml",
    )

    assert result._is_dagml_engine()  # noqa: SLF001
    assert result.num_predictions >= case.expected_min_predictions
