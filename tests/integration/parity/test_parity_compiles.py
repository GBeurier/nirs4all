"""Fast parity check: every registered case constructs a valid `PipelineConfigs`.

This is the "first-line" parity test. It catches typos in keyword names,
mis-shaped step dicts, and missing operator imports without paying for
a full pipeline run. It runs on every CI commit.

The slow end-to-end smoke lives in `test_parity_smoke.py` and is gated by
the `slow` marker.
"""

from __future__ import annotations

import pytest

from nirs4all.pipeline.config import PipelineConfigs

from ._registry import PipelineCase, all_cases, capability_coverage, keyword_coverage


@pytest.mark.parametrize("case", all_cases(), ids=lambda c: c.name)
def test_pipeline_compiles(case: PipelineCase) -> None:
    """`PipelineConfigs(pipeline, name=case.name)` must construct without raising.

    Cases carrying `skip_reason` are skipped — the reason text generally
    points either at a fixture mismatch or a known nirs4all 0.9.x bug that
    the parity oracle surfaced. The pipeline factory itself is still
    exercised so factory-level typos still surface.
    """
    pipeline = case.pipeline
    assert isinstance(pipeline, list), f"pipeline_factory must return list, got {type(pipeline)}"
    assert pipeline, "pipeline_factory must return non-empty list"

    if case.skip_reason:
        # See TODO(parity-manifest-followup) in test_parity_smoke for why
        # skip_kind="legacy_bug" doesn't xfail in-band today.
        pytest.skip(f"[{case.skip_kind or 'unknown'}] {case.skip_reason}")

    config = PipelineConfigs(pipeline, name=case.name)
    assert config is not None


def test_keyword_coverage_documented() -> None:
    """Every canonical CLAUDE.md keyword has at least one parity case.

    A keyword surfaces in this assertion failure list only if **no** case
    declares it. That is a coverage hole — either a new case must be added
    or the keyword removed from `_registry.CANONICAL_KEYWORDS` (with a
    matching CLAUDE.md update).

    A small allowlist captures the keywords we intentionally don't cover
    yet (each entry must point to an issue or follow-up note).
    """
    intentionally_uncovered = {
        # auto_transfer_preproc is a transfer-learning shortcut; coverage
        # belongs in the transfer-learning suite, not the parity oracle.
        "auto_transfer_preproc",
        # NA-handling kwargs (na_policy / fill_value) require a fixture
        # with missing values; deferred to a follow-up phase.
        "na_policy",
        "fill_value",
        # _depends_on_ is a DEAD keyword — declared in CONSTRAINT_KEYWORDS but
        # never consulted by the constraint engine; placing it on a generator
        # node BREAKS expansion, so it cannot be a runnable parity case. It is
        # documented via engine-level asserts in
        # test_generators_conformance_extra.py rather than a runnable case.
        "_depends_on_",
    }
    coverage = keyword_coverage()
    holes = {kw for kw, cases in coverage.items() if not cases} - intentionally_uncovered
    assert not holes, (
        f"canonical DSL keywords without any parity case: {sorted(holes)} — "
        f"add a case in tests/integration/parity/cases_*.py or extend the "
        f"intentionally_uncovered allowlist with a justification"
    )


def test_capability_coverage_smoke() -> None:
    """Each declared capability label has at least one case.

    Capabilities are free-form coverage tags, not contract surface. The check
    catches typos (`sklearn_modal` instead of `sklearn_model`) without forcing
    coverage of every label.
    """
    coverage = capability_coverage()
    holes = {cap for cap, cases in coverage.items() if not cases}
    # `pytorch_model` / `tensorflow_model` / `jax_model` need separately
    # installed backends; allow them to stay uncovered until those cases
    # are added in a backend-specific suite.
    backend_dependent = {"pytorch_model", "tensorflow_model", "jax_model"}
    holes -= backend_dependent
    # We allow a handful of other declared-but-not-yet-used labels so this
    # check doesn't go red on every new label-only PR.
    assert holes.issubset({"session_api", "explain_path", "retrain_path"}) or not holes, (
        f"capability labels with zero cases: {sorted(holes)}"
    )
