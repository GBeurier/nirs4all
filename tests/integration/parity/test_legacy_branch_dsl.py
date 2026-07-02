"""Legacy-engine regression locks for separation-branch parity cases."""

from __future__ import annotations

import pytest

import nirs4all

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
def test_legacy_engine_runs_separation_branch_dsl(case_name: str) -> None:
    """The explicit legacy compatibility path still owns these branch DSL shapes."""
    case = get(case_name)

    result = nirs4all.run(
        pipeline=case.pipeline,
        dataset=make_dataset(case),
        verbose=0,
        engine="legacy",
    )

    assert not result._is_dagml_engine()  # noqa: SLF001
    assert result.num_predictions >= case.expected_min_predictions
