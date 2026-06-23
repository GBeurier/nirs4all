"""Parity-suite pytest configuration.

Imports every `cases_*` module so the `PipelineCase` registry is fully
populated before pytest's `parametrize` collects tests. Without these
imports the test modules would see an empty registry — module-level
`@pytest.mark.parametrize(...)` resolves arguments at decoration time,
which is before any `pytest_*` hook fires, so the cases must be in the
registry by the time `test_*.py` is imported.

The integration-level `conftest.py` (one directory up) already manages
the per-test workspace and SQLite isolation; nothing about that needs to
be overridden here.
"""

from __future__ import annotations

import pytest

# Side-effect imports: each module registers its cases at module top level.
from . import (  # noqa: F401
    cases_aggregation_reps,
    cases_augmentation,
    cases_baseline,
    cases_branches_merges,
    cases_generators,
    cases_multi_source,
    cases_refit_predict,
    cases_tags_exclude,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--parity-capture`` flag for the gold-baseline test.

    With the flag set, ``test_parity_baseline`` records each case's legacy
    observation under ``baselines/`` instead of enforcing it against a prior
    capture (Layer 1 vs Layer 2 of PARITY_AND_PERF_HARNESS.md).
    """
    parser.addoption(
        "--parity-capture",
        action="store_true",
        default=False,
        help="capture the legacy gold baseline for each parity case instead of enforcing it",
    )
