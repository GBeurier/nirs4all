"""STATIC native-vs-fallback boundary + coverage-meter gate (B-010 / DML-003).

The companion to ``test_conformance_dual_engine.py::test_native_fallback_boundary``
(the DYNAMIC per-case guard that runs the real dag-ml leg). This file gates the
STATIC meter in :mod:`coverage_meter`:

* the meter's summary IS ``docs/compatibility.json["coverage_meter"]`` (so the
  published ledger face can never drift from the live registry/allowlist);
* every registered case partitions into exactly one disposition bucket;
* the ``expected_fallback`` / ``unexpected`` boundary matches the live
  :data:`EXPECTED_FALLBACK` allowlist (the LOCK-DROP D1 instrument);
* the partition leaves roll up to the summary identities.

Fast and engine-free (no ``slow`` marker): it reads declared structures only.
"""

from __future__ import annotations

import json

import pytest

from . import coverage_meter as M
from ._registry import CANONICAL_KEYWORDS, all_cases
from .test_conformance_dual_engine import (
    EXPECTED_FALLBACK,
    KNOWN_DIVERGENCES,
    NUM_PREDICTIONS_DIVERGENCE,
)

pytestmark = [pytest.mark.parity]


def _authority_style_summary() -> dict[str, int]:
    """Recompute the ledger summary straight from live structures.

    Independent of the meter's own partition math, so a bug that corrupts BOTH
    the meter and the ledger in the same way is still caught here.
    """
    cases = all_cases()
    non_runnable = sum(1 for c in cases if c.skip_reason)
    legacy_bug = sum(1 for c in cases if c.skip_kind == "legacy_bug")
    skip = sum(1 for c in cases if c.skip_reason and c.skip_kind != "legacy_bug")
    return {
        "registered": len(cases),
        "non_runnable": non_runnable,
        "runnable": len(cases) - non_runnable,
        "fallback": len(EXPECTED_FALLBACK),
        "native": len(cases) - non_runnable - len(EXPECTED_FALLBACK),
        "xfail_strict": len(KNOWN_DIVERGENCES) + legacy_bug,
        "skip": skip,
        "num_predictions_divergence": len(NUM_PREDICTIONS_DIVERGENCE),
        "expected_fallback_target": 0,
    }


def test_meter_summary_matches_compatibility_ledger() -> None:
    """The published coverage_meter key is exactly the live meter summary."""
    summary = M.build_report().summary()
    assert summary == M.load_ledger_coverage_meter(), (
        "docs/compatibility.json[coverage_meter] drifted from the live meter — "
        "regenerate with `python -m tests.integration.parity.coverage_meter --check`"
    )


def test_meter_summary_matches_authority_recomputation() -> None:
    """The meter summary equals an independent registry recomputation + exposes every ledger key."""
    summary = M.build_report().summary()
    assert summary == _authority_style_summary()
    assert tuple(summary) == M.LEDGER_SUMMARY_KEYS


def test_every_case_partitions_into_one_leaf_bucket() -> None:
    report = M.build_report()
    assert len(report.cases) == len(all_cases())
    assert {c.bucket for c in report.cases} <= set(M.LEAF_BUCKETS)
    assert sum(report.leaf_counts().values()) == len(report.cases)
    # Names are unique across the partition (no case classified twice).
    names = [c.name for c in report.cases]
    assert len(names) == len(set(names)) == len(all_cases())


def test_expected_fallback_boundary_matches_allowlist_and_ledger() -> None:
    """The meter's fallback boundary == the live allowlist == the ledger rows; unexpected is empty (static)."""
    report = M.build_report()
    assert set(report.names_in(M.EXPECTED_FALLBACK_BUCKET)) == set(EXPECTED_FALLBACK)
    assert report.names_in(M.UNEXPECTED) == []

    ledger_rows = {row["case"] for row in json.loads(M.COMPATIBILITY_JSON.read_text())["expected_fallback"]}
    assert ledger_rows == set(EXPECTED_FALLBACK)


def test_partition_leaves_roll_up_to_summary() -> None:
    report = M.build_report()
    leaves = report.leaf_counts()
    summary = report.summary()

    native_family = leaves[M.NATIVE] + leaves[M.PYTHON_EXPANDED] + leaves[M.PYTHON_PRE_MATERIALIZED]
    xfail_divergence = sum(1 for c in report.cases if c.bucket == M.XFAIL and c.runnable)
    xfail_legacy_bug = sum(1 for c in report.cases if c.bucket == M.XFAIL and not c.runnable)

    # native REACH (summary["native"]) = native-family + the native-but-divergent xfails.
    assert summary["native"] == native_family + xfail_divergence
    assert summary["fallback"] == leaves[M.EXPECTED_FALLBACK_BUCKET] + leaves[M.UNEXPECTED]
    assert summary["xfail_strict"] == leaves[M.XFAIL] == xfail_divergence + xfail_legacy_bug
    assert summary["skip"] == leaves[M.SKIP]
    assert summary["non_runnable"] == leaves[M.SKIP] + xfail_legacy_bug
    assert summary["runnable"] == native_family + xfail_divergence + summary["fallback"]
    assert summary["runnable"] + summary["non_runnable"] == summary["registered"]


def test_legacy_fallback_is_expected_plus_unexpected() -> None:
    buckets = M.build_report().bucket_counts()
    assert buckets[M.LEGACY_FALLBACK] == buckets[M.EXPECTED_FALLBACK_BUCKET] + buckets[M.UNEXPECTED]


def test_route_keyword_sets_are_canonical_dsl() -> None:
    """The route heuristic only keys off real DSL keywords (renames fail loud, not silent)."""
    route_keywords = M._PRE_MATERIALIZED_KEYWORDS | M._GENERATOR_KEYWORDS
    assert route_keywords <= CANONICAL_KEYWORDS
    assert not (M._PRE_MATERIALIZED_KEYWORDS & M._GENERATOR_KEYWORDS)


def test_unexpected_bucket_flags_off_allowlist_fallback() -> None:
    """A dynamic observation of an off-allowlist fallback surfaces as a regression."""
    report = M.build_report()
    victim = next(c.name for c in report.cases if c.bucket == M.NATIVE)

    observed = M.build_report(observed_fallback={victim})
    classified = {c.name: c.bucket for c in observed.cases}
    assert classified[victim] == M.UNEXPECTED
    assert observed.names_in(M.UNEXPECTED) == [victim]
    # The regression inflates fallback but never moves the gate target off 0.
    assert observed.summary()["fallback"] == report.summary()["fallback"] + 1
    assert observed.summary()["expected_fallback_target"] == 0


def test_observed_fallback_on_allowlist_stays_expected() -> None:
    """An allowlisted case observed falling back is still expected_fallback, not unexpected."""
    allowlisted = sorted(EXPECTED_FALLBACK)[0]
    report = M.build_report(observed_fallback={allowlisted})
    classified = {c.name: c.bucket for c in report.cases}
    assert classified[allowlisted] == M.EXPECTED_FALLBACK_BUCKET
    assert report.names_in(M.UNEXPECTED) == []


def test_known_divergence_outranks_native_route() -> None:
    """A divergent augmentation/rep case is xfail (disposition), not its native route."""
    report = M.build_report()
    classified = {c.name: c for c in report.cases}
    for name in KNOWN_DIVERGENCES:
        assert classified[name].bucket == M.XFAIL
        assert classified[name].subtype == "known_divergence"
        assert classified[name].runnable is True


def test_inventory_is_json_serializable_and_complete() -> None:
    report = M.build_report()
    inventory = json.loads(json.dumps(report.to_inventory()))
    assert inventory["schema"] == "nirs4all.pyref.coverage_meter.v1"
    assert inventory["summary"] == report.summary()
    assert len(inventory["cases"]) == len(all_cases())
    assert set(inventory["buckets"]) == set(M.LEAF_BUCKETS) | {M.LEGACY_FALLBACK}

    markdown = report.to_markdown()
    assert "coverage meter" in markdown
    assert "fallback" in markdown


def test_cli_check_reports_zero_drift_and_emits_artifacts(tmp_path) -> None:
    assert M.main(["--check"]) == 0

    json_path = tmp_path / "meter.json"
    md_path = tmp_path / "meter.md"
    assert M.main(["--json", str(json_path), "--md", str(md_path)]) == 0
    loaded = json.loads(json_path.read_text())
    assert loaded["summary"] == M.build_report().summary()
    assert md_path.read_text().startswith("# PYREF native-vs-fallback coverage meter")
