"""Machine-readable compatibility ledger authority for the PYREF oracle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import _conformance_helpers as helpers

# Side-effect imports keep this module usable outside pytest collection, where
# parity/conftest.py may not have populated the registry yet.
from . import (  # noqa: F401
    cases_aggregation_reps,
    cases_augmentation,
    cases_baseline,
    cases_branches_merges,
    cases_generators,
    cases_generators_conformance,
    cases_multi_source,
    cases_refit_predict,
    cases_tags_exclude,
)
from ._marker_audit import validate_marker_policy
from ._registry import all_cases
from .test_conformance_dual_engine import (
    EXPECTED_FALLBACK,
    KNOWN_DIVERGENCES,
    LEGACY_CV_SCORE_DIVERGENCE,
    NUM_PREDICTIONS_DIVERGENCE,
    SAME_WINNER_CASES,
    UNSEEDED_NONDETERMINISTIC_CASES,
    Y_PRED_TOL_OVERRIDES,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPATIBILITY_JSON = REPO_ROOT / "docs" / "compatibility.json"


def load_compatibility_ledger(path: Path = COMPATIBILITY_JSON) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_compatibility_ledger(ledger: dict[str, Any] | None = None) -> None:
    data = ledger or load_compatibility_ledger()
    _validate_tolerance_bands(data)
    _validate_authority_entries(data)
    _validate_expected_fallback(data)
    _validate_num_prediction_divergences(data)
    _validate_legacy_cv_score_divergences(data)
    _validate_ypred_overrides(data)
    _validate_same_winner_cases(data)
    _validate_coverage_skips(data)
    _validate_coverage_meter(data)
    validate_marker_policy(data)


def _cases_by_name() -> dict[str, Any]:
    return {case.name: case for case in all_cases()}


def _band_map(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {band["band_id"]: band for band in data["tolerance_bands"]}


def _validate_tolerance_bands(data: dict[str, Any]) -> None:
    bands = _band_map(data)
    required = {
        "kernel_snv",
        "kernel_pls",
        "native_export_reproduce",
        "per_case_tight",
        "cross_impl_score",
        "cross_impl_ypred",
        "cross_impl_ypred_firstderiv",
        "classification_label",
        "n/a_semantic",
        "n/a_rng",
    }
    missing = required - set(bands)
    if missing:
        raise AssertionError(f"compatibility tolerance bands missing: {sorted(missing)}")
    assert bands["cross_impl_score"]["abs_tol"] == helpers._DEFAULT_SCORE_TOL  # noqa: SLF001
    assert bands["cross_impl_ypred"]["abs_tol"] == helpers._DEFAULT_YPRED_TOL  # noqa: SLF001
    assert bands["cross_impl_ypred_firstderiv"]["abs_tol"] == 5e-3
    if "run-only nondeterministic" not in bands["n/a_rng"]["enforced_at"]:
        raise AssertionError("n/a_rng band must publish the ledgered run-only nondeterministic contract")
    _validate_per_case_tight_band(bands)


def _validate_per_case_tight_band(bands: dict[str, dict[str, Any]]) -> None:
    """Bind the ``per_case_tight`` band to its sole live instance so it can't drift.

    ``per_case_tight`` is the per-case ``metric_tolerances`` override mechanism.
    Its only live user is ``baseline_vertical_slice`` (rmse AND r2 pinned at the
    same magnitude); the ledger band must equal that value, so a change to the
    case forces a matching ledger update rather than silently diverging (the
    band previously claimed ``1e-6`` while the case enforced ``1e-3``).
    """
    case = _cases_by_name().get("baseline_vertical_slice")
    if case is None or not case.metric_tolerances:
        return
    values = set(case.metric_tolerances.values())
    band = bands["per_case_tight"]["abs_tol"]
    if values != {band}:
        raise AssertionError(
            "per_case_tight band must equal baseline_vertical_slice metric_tolerances: "
            f"band={band} case={sorted(values)}"
        )


def _validate_authority_entries(data: dict[str, Any]) -> None:
    cases = _cases_by_name()
    known = set(KNOWN_DIVERGENCES)
    legacy_bug = {case.name for case in cases.values() if case.skip_kind == "legacy_bug"}
    strict_rows = {
        row["case"]
        for row in data["authority"]
        if row["disposition"] == "xfail_strict"
    }
    expected_strict = known | legacy_bug
    if strict_rows != expected_strict:
        raise AssertionError(
            "compatibility strict authority entries drifted: "
            f"expected={sorted(expected_strict)} actual={sorted(strict_rows)}"
        )

    parity_note_rows = {
        row["case"]
        for row in data["authority"]
        if row["disposition"] == "pass_parity_note"
    }
    expected_parity_notes = set(NUM_PREDICTIONS_DIVERGENCE) | set(LEGACY_CV_SCORE_DIVERGENCE)
    if parity_note_rows != expected_parity_notes:
        raise AssertionError(
            "compatibility parity-note authority entries drifted: "
            f"expected={sorted(expected_parity_notes)} actual={sorted(parity_note_rows)}"
        )

    run_only_rows = {
        row["case"]
        for row in data["authority"]
        if row["disposition"] == "run_only_nondeterministic"
    }
    if run_only_rows != set(UNSEEDED_NONDETERMINISTIC_CASES):
        raise AssertionError(
            "compatibility run-only nondeterministic authority entries drifted: "
            f"expected={sorted(UNSEEDED_NONDETERMINISTIC_CASES)} actual={sorted(run_only_rows)}"
        )

    bands = set(_band_map(data))
    for row in data["authority"]:
        if row["case"] not in cases:
            raise AssertionError(f"compatibility authority references unknown case {row['case']!r}")
        if row["tolerance_band"] not in bands:
            raise AssertionError(
                f"compatibility authority row {row['case']!r} references unknown band {row['tolerance_band']!r}"
            )


def _validate_expected_fallback(data: dict[str, Any]) -> None:
    actual = {row["case"] for row in data["expected_fallback"]}
    expected = set(EXPECTED_FALLBACK)
    if actual != expected:
        raise AssertionError(
            f"EXPECTED_FALLBACK ledger drift: expected={sorted(expected)} actual={sorted(actual)}"
        )
    bad_owner = [row["case"] for row in data["expected_fallback"] if row.get("owner_lane") != "L5"]
    if bad_owner:
        raise AssertionError(f"expected_fallback entries must be owned by L5: {bad_owner}")


def _validate_num_prediction_divergences(data: dict[str, Any]) -> None:
    actual = {row["case"]: row for row in data["num_predictions_divergence"]}
    if set(actual) != set(NUM_PREDICTIONS_DIVERGENCE):
        raise AssertionError(
            "num_predictions divergence case set drifted: "
            f"expected={sorted(NUM_PREDICTIONS_DIVERGENCE)} actual={sorted(actual)}"
        )
    for case_name, expected in NUM_PREDICTIONS_DIVERGENCE.items():
        row = actual[case_name]
        if row["legacy"] != expected["legacy"] or row["dagml"] != expected["dagml"]:
            raise AssertionError(f"num_predictions divergence count drifted for {case_name}")


def _validate_legacy_cv_score_divergences(data: dict[str, Any]) -> None:
    actual = {row["case"]: row for row in data.get("legacy_cv_score_divergence", [])}
    if set(actual) != set(LEGACY_CV_SCORE_DIVERGENCE):
        raise AssertionError(
            "legacy cv score divergence case set drifted: "
            f"expected={sorted(LEGACY_CV_SCORE_DIVERGENCE)} actual={sorted(actual)}"
        )
    for case_name, expected in LEGACY_CV_SCORE_DIVERGENCE.items():
        row = actual[case_name]
        if row.get("metric") != "cv_best_score":
            raise AssertionError(f"legacy cv score divergence metric drifted for {case_name}")
        if row["legacy"] != expected["legacy"] or row["dagml"] != expected["dagml"]:
            raise AssertionError(f"legacy cv score divergence value drifted for {case_name}")


def _validate_ypred_overrides(data: dict[str, Any]) -> None:
    actual = {row["case"]: row["abs_tol"] for row in data["ypred_tol_overrides"]}
    expected = dict(Y_PRED_TOL_OVERRIDES)
    if actual != expected:
        raise AssertionError(
            f"Y_PRED_TOL_OVERRIDES ledger drift: expected={expected} actual={actual}"
        )
    for row in data["ypred_tol_overrides"]:
        if row.get("guard") != "assert_same_winner":
            raise AssertionError(f"ypred override for {row['case']} must keep assert_same_winner guard")


def _validate_same_winner_cases(data: dict[str, Any]) -> None:
    actual = set(data["same_winner_cases"])
    expected = set(SAME_WINNER_CASES)
    if actual != expected:
        raise AssertionError(
            f"SAME_WINNER_CASES ledger drift: expected={sorted(expected)} actual={sorted(actual)}"
        )


def _validate_coverage_skips(data: dict[str, Any]) -> None:
    skip_cases = {
        case.name: case.skip_kind
        for case in all_cases()
        if case.skip_reason and case.skip_kind != "legacy_bug"
    }
    actual = {row["case"]: row["skip_kind"] for row in data["coverage_skips"]}
    if actual != skip_cases:
        raise AssertionError(f"coverage skip ledger drift: expected={skip_cases} actual={actual}")


def _validate_coverage_meter(data: dict[str, Any]) -> None:
    cases = all_cases()
    legacy_bug_count = sum(1 for case in cases if case.skip_kind == "legacy_bug")
    skip_count = sum(1 for case in cases if case.skip_reason and case.skip_kind != "legacy_bug")
    non_runnable = sum(1 for case in cases if case.skip_reason)
    expected = {
        "registered": len(cases),
        "non_runnable": non_runnable,
        "runnable": len(cases) - non_runnable,
        "fallback": len(EXPECTED_FALLBACK),
        "native": len(cases) - non_runnable - len(EXPECTED_FALLBACK),
        "xfail_strict": len(KNOWN_DIVERGENCES) + legacy_bug_count,
        "skip": skip_count,
        "num_predictions_divergence": len(NUM_PREDICTIONS_DIVERGENCE),
        "run_only_nondeterministic": len(UNSEEDED_NONDETERMINISTIC_CASES),
        "expected_fallback_target": 0,
    }
    if data["coverage_meter"] != expected:
        raise AssertionError(
            f"coverage meter drift: expected={expected} actual={data['coverage_meter']}"
        )
