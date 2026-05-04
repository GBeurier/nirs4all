"""Tests for the R9c DIESEL support-level shape mechanism audit (exp12)."""

from __future__ import annotations

import importlib.util
import sys
from csv import DictReader
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest


def _load_module(name: str, filename: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "experiments" / filename
    experiments_dir = str(path.parent)
    if experiments_dir not in sys.path:
        sys.path.insert(0, experiments_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _load_exp12() -> ModuleType:
    return _load_module(
        "exp12_diesel_support_shape_mechanism_audit",
        "exp12_diesel_support_shape_mechanism_audit.py",
    )


def _load_exp09() -> ModuleType:
    return _load_module(
        "exp09_sentinel_morphology_audit",
        "exp09_sentinel_morphology_audit.py",
    )


def _make_dataset(name: str) -> Any:
    from nirsyntheticpfn.evaluation.realism import RealDataset

    return RealDataset(
        source="AOM_regression",
        task="regression",
        database_name=name,
        dataset="ds",
        train_path="train.csv",
        test_path="test.csv",
        ytrain_path="ytrain.csv",
        ytest_path="ytest.csv",
        n_train_declared=10,
        n_test_declared=4,
        p_declared=8,
    )


def _write_empty_cohorts(root: Path) -> None:
    cohort_dir = root / "bench/AOM_v0/benchmarks"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    header = (
        "database_name,dataset,status,reason,n_train,n_test,p,"
        "train_path,test_path,ytrain_path,ytest_path\n"
    )
    (cohort_dir / "cohort_regression.csv").write_text(header, encoding="utf-8")
    (cohort_dir / "cohort_classification.csv").write_text(header, encoding="utf-8")


# ---------------------------------------------------------------------------
# Static module shape.
# ---------------------------------------------------------------------------


def test_exp12_exposes_audited_profiles_and_default_seeds() -> None:
    exp12 = _load_exp12()
    assert exp12.R9C_AUDITED_PROFILES == (
        "r3d_diesel_matrix_v1",
        "r4a_diesel_basis_v1",
        "r4b_diesel_derivative_restore_v1",
        "r4c_diesel_balanced_derivative_v1",
        "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1",
        "r9b_diesel_support_intercept_v1",
        "r9c_diesel_selective_ch_bandwidth_damping_v1",
    )
    assert exp12.R9C_PAIRED_REFERENCE_PROFILES == (
        "r4a_diesel_basis_v1",
        "r4b_diesel_derivative_restore_v1",
        "r4c_diesel_balanced_derivative_v1",
        "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1",
        "r9b_diesel_support_intercept_v1",
        "r3d_diesel_matrix_v1",
    )
    assert exp12.R9C_FOCUS_PROFILE == (
        "r9c_diesel_selective_ch_bandwidth_damping_v1"
    )
    assert exp12.DEFAULT_SEEDS == (20260501, 20260502, 20260503)
    assert exp12.DEFAULT_N_SYNTHETIC_SAMPLES == 64
    assert exp12.DEFAULT_MAX_REAL_SAMPLES == 64
    assert exp12.SUPPORT_LOW_NM == 750.0
    assert exp12.SUPPORT_HIGH_NM == 1550.0
    assert exp12.COMPARISON_SPACE == "uncalibrated_raw"
    assert exp12.R9C_AUDIT_SCOPE == (
        "bench_only_r9c_diesel_support_shape_mechanism_audit"
    )
    assert exp12.DEFAULT_REPORT.name == (
        "r9c_diesel_support_shape_mechanism_audit.md"
    )
    assert exp12.DEFAULT_CSV.name == (
        "r9c_diesel_support_shape_mechanism_audit.csv"
    )


# ---------------------------------------------------------------------------
# Audit/run shape and metadata flags.
# ---------------------------------------------------------------------------


def test_run_audit_returns_blocked_no_real_data_when_cohorts_are_empty(
    tmp_path: Path,
) -> None:
    exp12 = _load_exp12()
    _write_empty_cohorts(tmp_path)

    result = exp12.run_audit(
        root=tmp_path,
        seeds=(20260501,),
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=2,
    )

    assert result["status"] == "blocked_no_real_data"
    assert result["rows"] == []
    assert result["seeds"] == [20260501]
    assert result["audited_profiles"] == list(exp12.R9C_AUDITED_PROFILES)
    assert result["support_low_nm"] == 750.0
    assert result["support_high_nm"] == 1550.0


def test_run_audit_rejects_unknown_profile(tmp_path: Path) -> None:
    exp12 = _load_exp12()
    _write_empty_cohorts(tmp_path)

    with pytest.raises(ValueError):
        exp12.run_audit(
            root=tmp_path,
            seeds=(20260501,),
            n_synthetic_samples=4,
            max_real_samples=4,
            max_sentinel_datasets=2,
            profiles=("not_a_real_profile",),
        )


def test_run_audit_requires_at_least_one_seed(tmp_path: Path) -> None:
    exp12 = _load_exp12()
    _write_empty_cohorts(tmp_path)

    with pytest.raises(ValueError):
        exp12.run_audit(
            root=tmp_path,
            seeds=(),
            n_synthetic_samples=4,
            max_real_samples=4,
            max_sentinel_datasets=2,
        )


def test_blocked_path_records_remediation_profiles_and_audit_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exp12 = _load_exp12()
    exp09 = _load_exp09()
    _write_empty_cohorts(tmp_path)

    candidates = [_make_dataset("DIESEL_bp50_246_b-a")]

    monkeypatch.setattr(
        exp09,
        "discover_local_real_datasets",
        lambda root: (list(candidates), []),
    )
    monkeypatch.setattr(
        exp12,
        "discover_local_real_datasets",
        lambda root: (list(candidates), []),
    )
    monkeypatch.setattr(
        exp12,
        "load_real_spectra",
        lambda dataset, root: (_ for _ in ()).throw(
            RuntimeError("synthetic-test: no real spectra available")
        ),
    )

    profiles = (
        "r3d_diesel_matrix_v1",
        "r9c_diesel_selective_ch_bandwidth_damping_v1",
    )
    seeds = (20260501, 20260502)
    result = exp12.run_audit(
        root=tmp_path,
        seeds=seeds,
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=2,
        sentinel_tokens=["DIESEL"],
        profiles=profiles,
    )

    assert result["seeds"] == list(seeds)
    assert result["audited_profiles"] == list(profiles)
    assert len(result["rows"]) == len(seeds) * len(profiles)
    assert all(row.status == "blocked" for row in result["rows"])
    for row in result["rows"]:
        assert row.audit_oracle is False
        assert row.audit_label_inputs_used is False
        assert row.audit_target_inputs_used is False
        assert row.audit_split_inputs_used is False
        assert row.audit_source_oracle_used is False
        assert row.audit_real_stat_capture is False
        assert row.audit_thresholds_modified is False
        assert row.audit_metrics_modified is False
        assert row.audit_imputed is False
        assert row.audit_replays_real_rows is False
        assert row.audit_scope == exp12.R9C_AUDIT_SCOPE
        assert row.comparison_space == exp12.COMPARISON_SPACE
        assert row.remediation_profile in profiles
        assert row.seed in seeds


# ---------------------------------------------------------------------------
# CSV / Markdown shape.
# ---------------------------------------------------------------------------


def test_write_csv_empty_rows_emits_stable_header(tmp_path: Path) -> None:
    exp12 = _load_exp12()
    csv_path = tmp_path / "empty.csv"

    exp12.write_csv([], csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        records = list(reader)
    assert records == []
    fieldnames = reader.fieldnames
    assert fieldnames is not None
    assert list(fieldnames) == exp12._csv_fieldnames()
    for required in (
        "status",
        "seed",
        "remediation_profile",
        "effective_remediation_profile",
        "dataset",
        "comparison_space",
        "support_low_nm",
        "support_high_nm",
        "support_count",
        "off_support_count",
        "support_weight",
        "off_support_weight",
        "support_mean_delta",
        "off_support_mean_delta",
        "support_weighted_delta",
        "off_support_weighted_delta",
        "global_mean_delta",
        "decomposition_residual",
        "log10_global_std_ratio",
        "mean_curve_corr",
        "morphology_gap_score",
        "dominant_morphology_gap",
        "audit_oracle",
        "audit_real_stat_capture",
        "audit_thresholds_modified",
        "audit_scope",
        "blocked_reason",
    ):
        assert required in fieldnames
    # All paired-delta columns required by the R9c spec must be present
    # (R9c minus each of R4a/R4b/R4c/R8b/R9b/R3d on global_mean,
    # support_mean, morphology_gap_score, mean_curve_corr,
    # support_weighted_delta, off_support_weighted_delta).
    for ref in exp12.R9C_PAIRED_REFERENCE_PROFILES:
        for attr in exp12.PAIRED_DELTA_ATTRS:
            col = f"delta_r9c_minus_{ref}__{attr}"
            assert col in fieldnames, f"missing paired delta column {col!r}"


def _make_compared_row(
    exp12: ModuleType,
    *,
    profile: str,
    seed: int,
    global_mean_delta: float,
    support_weight: float,
    off_support_weight: float,
    morphology_gap_score: float,
    mean_curve_corr: float,
    log10_derivative_std_p50_ratio: float,
    dominant_morphology_gap: str,
    off_support_mean_delta: float = 0.0,
    off_support_weighted_delta: float = 0.0,
    support_mean_delta: float | None = None,
    support_weighted_delta: float | None = None,
) -> Any:
    audit_kwargs = {
        "audit_oracle": False,
        "audit_label_inputs_used": False,
        "audit_target_inputs_used": False,
        "audit_split_inputs_used": False,
        "audit_source_oracle_used": False,
        "audit_learned": False,
        "audit_real_stat_capture": False,
        "audit_thresholds_modified": False,
        "audit_metrics_modified": False,
        "audit_imputed": False,
        "audit_replays_real_rows": False,
        "audit_scope": exp12.R9C_AUDIT_SCOPE,
    }
    if support_mean_delta is None:
        support_mean_delta = global_mean_delta
    if support_weighted_delta is None:
        support_weighted_delta = global_mean_delta
    return exp12.R9cRow(
        status="compared",
        seed=seed,
        remediation_profile=profile,
        effective_remediation_profile=profile,
        source="AOM_regression",
        task="regression",
        dataset="DIESEL_bp50_246_b-a/ds",
        synthetic_preset="fuel",
        effective_matrix_route="diesel_fuel_matrix",
        comparison_space=exp12.COMPARISON_SPACE,
        n_real_samples=8,
        n_synthetic_samples=8,
        n_wavelengths=10,
        wavelength_min=900.0,
        wavelength_max=1550.0,
        support_low_nm=exp12.SUPPORT_LOW_NM,
        support_high_nm=exp12.SUPPORT_HIGH_NM,
        support_count=10,
        off_support_count=0,
        support_weight=support_weight,
        off_support_weight=off_support_weight,
        support_mean_delta=support_mean_delta,
        off_support_mean_delta=off_support_mean_delta,
        support_weighted_delta=support_weighted_delta,
        off_support_weighted_delta=off_support_weighted_delta,
        global_mean_delta=global_mean_delta,
        decomposition_residual=0.0,
        real_global_mean=0.003,
        synthetic_global_mean=0.005,
        real_global_std=0.014,
        synthetic_global_std=0.015,
        log10_global_std_ratio=0.03,
        log10_amplitude_p50_ratio=0.0,
        log10_derivative_std_p50_ratio=log10_derivative_std_p50_ratio,
        mean_curve_corr=mean_curve_corr,
        morphology_gap_score=morphology_gap_score,
        dominant_morphology_gap=dominant_morphology_gap,
        **audit_kwargs,
        blocked_reason="",
    )


def _render_with_rows(
    exp12: ModuleType, rows: list[Any], tmp_path: Path
) -> Any:
    result = {
        "status": "done",
        "rows": rows,
        "real_runnable_count": 1,
        "real_sentinel_candidate_count": 1,
        "real_selected_count": 1,
        "sentinel_tokens": ["DIESEL"],
        "seeds": [20260501],
        "audited_profiles": list(exp12.R9C_AUDITED_PROFILES),
        "support_low_nm": exp12.SUPPORT_LOW_NM,
        "support_high_nm": exp12.SUPPORT_HIGH_NM,
    }
    return exp12.render_markdown(
        result=result,
        report_path=tmp_path / "r9c.md",
        csv_path=tmp_path / "r9c.csv",
        n_synthetic_samples=64,
        max_real_samples=64,
        max_sentinel_datasets=8,
        seeds=[20260501],
        sentinel_tokens=["DIESEL"],
        profiles=list(exp12.R9C_AUDITED_PROFILES),
    )


def test_render_markdown_disclaimer_and_no_gate_claims(tmp_path: Path) -> None:
    exp12 = _load_exp12()
    rows = [
        _make_compared_row(
            exp12,
            profile="r9c_diesel_selective_ch_bandwidth_damping_v1",
            seed=20260501,
            global_mean_delta=0.001,
            support_weight=1.0,
            off_support_weight=0.0,
            morphology_gap_score=1.10,
            mean_curve_corr=0.30,
            log10_derivative_std_p50_ratio=0.05,
            dominant_morphology_gap="mean_shift",
        ),
    ]
    md = _render_with_rows(exp12, rows, tmp_path)

    assert "uncalibrated_raw" in md
    assert "diagnostic-only" in md.lower()
    assert "no calibration" in md.lower()
    assert "no PCA/covariance" in md
    assert "no ML/DL" in md
    assert "B2/B3/B4/B5" in md
    assert "750-1550" in md
    assert "DIESEL" in md
    # Pre-declared mechanistic constants disclaimer (R9c-specific).
    assert "PRE-DECLARED MECHANISTIC CONSTANTS" in md
    assert "general liquid-hydrocarbon NIR overtone prior" in md
    assert "NOT chosen from any R9a or R9b mean-shift residual delta" in md
    # No promotion / no gate claims.
    assert "not a promotion over r3d" in md.lower()
    # Wavelength-dependent shape only - no scalar offset.
    assert "no scalar offset is added by R9c" in md.lower() or (
        "not a scalar offset" in md.lower()
    )
    for flag in (
        "oracle=false",
        "label_inputs_used=false",
        "target_inputs_used=false",
        "split_inputs_used=false",
        "source_oracle_used=false",
        "learned=false",
        "real_stat_capture=false",
        "thresholds_modified=false",
        "metrics_modified=false",
        "imputed=false",
        "replays_real_rows=false",
        "calibration=false",
        "uses_pca=false",
        "captures_noise=false",
        "uses_ml=false",
        "uses_dl=false",
        "adds_offset=false",
        "support_shape_only=true",
    ):
        assert flag in md, f"missing audit flag {flag!r}"
    assert (
        "constants_source=predeclared_general_liquid_hydrocarbon_nir_prior"
    ) in md
    assert "B2 PASS" not in md.upper().replace(":", "")


def test_render_markdown_includes_per_profile_and_paired_delta_tables(
    tmp_path: Path,
) -> None:
    exp12 = _load_exp12()
    rows = [
        _make_compared_row(
            exp12,
            profile="r4c_diesel_balanced_derivative_v1",
            seed=20260501,
            global_mean_delta=0.003,
            support_weight=1.0,
            off_support_weight=0.0,
            morphology_gap_score=1.30,
            mean_curve_corr=0.20,
            log10_derivative_std_p50_ratio=0.07,
            dominant_morphology_gap="mean_shift",
        ),
        _make_compared_row(
            exp12,
            profile="r9c_diesel_selective_ch_bandwidth_damping_v1",
            seed=20260501,
            global_mean_delta=0.001,
            support_weight=1.0,
            off_support_weight=0.0,
            morphology_gap_score=1.10,
            mean_curve_corr=0.30,
            log10_derivative_std_p50_ratio=0.07,
            dominant_morphology_gap="mean_shift",
        ),
    ]
    md = _render_with_rows(exp12, rows, tmp_path)

    assert "## Per-Profile Synthesis (compared rows only)" in md
    assert "median global mean delta" in md
    assert "median support mean delta" in md
    assert "median morphology gap score" in md
    assert "`r4c_diesel_balanced_derivative_v1`" in md
    assert "`r9c_diesel_selective_ch_bandwidth_damping_v1`" in md
    assert "## Paired Deltas: R9c minus reference profile" in md
    assert "r9c vs r4a global" in md
    assert "r9c vs r4b global" in md
    assert "r9c vs r4c global" in md
    assert "r9c vs r8b global" in md
    assert "r9c vs r9b global" in md
    assert "r9c vs r3d global" in md
    assert "r9c vs r4c morphology gap" in md
    assert "r9c vs r9b morphology gap" in md
    assert "r9c vs r3d morphology gap" in md


def test_render_markdown_marks_r9c_rejected_when_medians_worsen(
    tmp_path: Path,
) -> None:
    exp12 = _load_exp12()
    rows = [
        _make_compared_row(
            exp12,
            profile="r3d_diesel_matrix_v1",
            seed=20260501,
            global_mean_delta=0.005,
            support_weight=1.0,
            off_support_weight=0.0,
            morphology_gap_score=1.50,
            mean_curve_corr=0.20,
            log10_derivative_std_p50_ratio=0.05,
            dominant_morphology_gap="mean_shift",
        ),
        _make_compared_row(
            exp12,
            profile="r4c_diesel_balanced_derivative_v1",
            seed=20260501,
            global_mean_delta=0.004,
            support_weight=1.0,
            off_support_weight=0.0,
            morphology_gap_score=1.40,
            mean_curve_corr=0.20,
            log10_derivative_std_p50_ratio=0.05,
            dominant_morphology_gap="mean_shift",
        ),
        _make_compared_row(
            exp12,
            profile="r9b_diesel_support_intercept_v1",
            seed=20260501,
            global_mean_delta=0.006,
            support_weight=1.0,
            off_support_weight=0.0,
            morphology_gap_score=1.60,
            mean_curve_corr=0.20,
            log10_derivative_std_p50_ratio=0.05,
            dominant_morphology_gap="mean_shift",
        ),
        _make_compared_row(
            exp12,
            profile="r9c_diesel_selective_ch_bandwidth_damping_v1",
            seed=20260501,
            global_mean_delta=0.060,
            support_weight=1.0,
            off_support_weight=0.0,
            morphology_gap_score=7.00,
            mean_curve_corr=0.10,
            log10_derivative_std_p50_ratio=0.20,
            dominant_morphology_gap="mean_shift",
        ),
    ]
    md = _render_with_rows(exp12, rows, tmp_path)

    assert "## Diagnostic Outcome" in md
    assert "R9c current empirical outcome: rejected / not promoted" in md
    assert "worsens median global mean delta vs" in md
    assert "worsens median morphology gap score vs" in md
    assert "do not reuse this positive-area support-shape addition" in md


def test_paired_deltas_include_r9c_minus_all_references_and_handle_missing_reference(
    tmp_path: Path,
) -> None:
    exp12 = _load_exp12()
    rows = [
        _make_compared_row(
            exp12,
            profile="r9c_diesel_selective_ch_bandwidth_damping_v1",
            seed=20260501,
            global_mean_delta=0.0015,
            support_weight=1.0,
            off_support_weight=0.0,
            morphology_gap_score=1.05,
            mean_curve_corr=0.30,
            log10_derivative_std_p50_ratio=0.05,
            dominant_morphology_gap="mean_shift",
            support_mean_delta=0.0015,
            support_weighted_delta=0.0015,
        ),
        _make_compared_row(
            exp12,
            profile="r4c_diesel_balanced_derivative_v1",
            seed=20260501,
            global_mean_delta=0.0035,
            support_weight=1.0,
            off_support_weight=0.0,
            morphology_gap_score=1.30,
            mean_curve_corr=0.20,
            log10_derivative_std_p50_ratio=0.05,
            dominant_morphology_gap="mean_shift",
            support_mean_delta=0.0035,
            support_weighted_delta=0.0035,
        ),
        _make_compared_row(
            exp12,
            profile="r3d_diesel_matrix_v1",
            seed=20260501,
            global_mean_delta=0.0050,
            support_weight=1.0,
            off_support_weight=0.0,
            morphology_gap_score=1.50,
            mean_curve_corr=0.10,
            log10_derivative_std_p50_ratio=0.09,
            dominant_morphology_gap="mean_shift",
            support_mean_delta=0.0050,
            support_weighted_delta=0.0050,
        ),
        _make_compared_row(
            exp12,
            profile="r9b_diesel_support_intercept_v1",
            seed=20260501,
            global_mean_delta=0.0042,
            support_weight=1.0,
            off_support_weight=0.0,
            morphology_gap_score=1.40,
            mean_curve_corr=0.18,
            log10_derivative_std_p50_ratio=0.08,
            dominant_morphology_gap="mean_shift",
            support_mean_delta=0.0042,
            support_weighted_delta=0.0042,
        ),
    ]

    paired = exp12._paired_deltas_vs_references(rows)
    assert len(paired) == 1
    entry = paired[0]
    assert entry["seed"] == 20260501
    assert entry["dataset"] == "DIESEL_bp50_246_b-a/ds"
    assert entry["r9c_global_mean_delta"] == pytest.approx(0.0015)
    # r4c reference present: r9c - r4c.
    assert entry[
        "delta_r9c_minus_r4c_diesel_balanced_derivative_v1__global_mean_delta"
    ] == pytest.approx(0.0015 - 0.0035)
    assert entry[
        "delta_r9c_minus_r4c_diesel_balanced_derivative_v1__morphology_gap_score"
    ] == pytest.approx(1.05 - 1.30)
    # r9b reference present.
    assert entry[
        "delta_r9c_minus_r9b_diesel_support_intercept_v1__global_mean_delta"
    ] == pytest.approx(0.0015 - 0.0042)
    # r3d reference present.
    assert entry[
        "delta_r9c_minus_r3d_diesel_matrix_v1__global_mean_delta"
    ] == pytest.approx(0.0015 - 0.0050)
    # r4a, r4b, r8b references missing -> NA.
    for missing_ref in (
        "r4a_diesel_basis_v1",
        "r4b_diesel_derivative_restore_v1",
        "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1",
    ):
        for attr in exp12.PAIRED_DELTA_ATTRS:
            assert (
                entry[f"delta_r9c_minus_{missing_ref}__{attr}"] is None
            ), f"expected NA for missing reference {missing_ref!r} attr {attr!r}"


def test_paired_deltas_skip_blocked_rows_and_missing_r9c(
    tmp_path: Path,
) -> None:
    exp12 = _load_exp12()
    rows = [
        _make_compared_row(
            exp12,
            profile="r4c_diesel_balanced_derivative_v1",
            seed=20260501,
            global_mean_delta=0.003,
            support_weight=1.0,
            off_support_weight=0.0,
            morphology_gap_score=1.10,
            mean_curve_corr=0.20,
            log10_derivative_std_p50_ratio=0.05,
            dominant_morphology_gap="mean_shift",
        ),
        # No r9c compared row - paired table must be empty.
    ]
    assert exp12._paired_deltas_vs_references(rows) == []


def test_render_markdown_emits_off_support_null_note_when_all_zero(
    tmp_path: Path,
) -> None:
    exp12 = _load_exp12()
    rows = [
        _make_compared_row(
            exp12,
            profile="r9c_diesel_selective_ch_bandwidth_damping_v1",
            seed=20260501,
            global_mean_delta=0.001,
            support_weight=1.0,
            off_support_weight=0.0,
            morphology_gap_score=1.10,
            mean_curve_corr=0.30,
            log10_derivative_std_p50_ratio=0.05,
            dominant_morphology_gap="mean_shift",
        ),
    ]
    md = _render_with_rows(exp12, rows, tmp_path)
    assert exp12.OFF_SUPPORT_NULL_NOTE in md


def test_aggregate_by_profile_ignores_blocked_rows(tmp_path: Path) -> None:
    exp12 = _load_exp12()
    compared = _make_compared_row(
        exp12,
        profile="r9c_diesel_selective_ch_bandwidth_damping_v1",
        seed=20260501,
        global_mean_delta=0.001,
        support_weight=1.0,
        off_support_weight=0.0,
        morphology_gap_score=1.10,
        mean_curve_corr=0.30,
        log10_derivative_std_p50_ratio=0.05,
        dominant_morphology_gap="mean_shift",
    )
    blocked = exp12._blocked_row(
        seed=20260501,
        requested_profile="r3d_diesel_matrix_v1",
        effective_profile="r3d_diesel_matrix_v1",
        dataset=_make_dataset("DIESEL_bp50_246_b-a"),
        preset="fuel",
        blocked_reason="synthetic-test",
    )
    summary = exp12._aggregate_by_profile([compared, blocked])
    assert len(summary) == 1
    assert summary[0]["profile"] == (
        "r9c_diesel_selective_ch_bandwidth_damping_v1"
    )
    assert summary[0]["n"] == 1


def test_run_audit_paired_delta_against_synthetic_in_memory_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end paired-delta check on a deterministic monkeypatched cohort.

    The fake synthetic builder produces an R3d-like base for every profile
    except R9c, where a wavelength-dependent shape is added on the support.
    The paired off-support delta of R9c vs R3d must therefore be exactly
    zero, and the support-side R9c vs R3d delta must be non-zero.
    """
    exp12 = _load_exp12()
    exp09 = _load_exp09()
    _write_empty_cohorts(tmp_path)

    diesel_dataset = _make_dataset("DIESEL_bp50_246_b-a")
    wl = np.array(
        [700.0, 800.0, 900.0, 1000.0, 1100.0, 1600.0, 1700.0], dtype=float
    )
    support_mask = (wl >= exp12.SUPPORT_LOW_NM) & (wl <= exp12.SUPPORT_HIGH_NM)
    fake_real = np.zeros((20, wl.size), dtype=float)

    n_rows = 8
    # Shape values per support wavelength (deterministic, wavelength-dependent).
    shape_on_support = np.array(
        [0.001, 0.002, 0.003, 0.004], dtype=float
    )  # one entry per support index
    assert support_mask.sum() == shape_on_support.size

    def fake_stable_dataset_seed(seed: int, dataset: Any, purpose: str) -> int:
        if "real_downsample" in purpose:
            return 11
        if "syn_downsample" in purpose:
            return 12
        return 99

    def fake_build_baseline_synthetic_run(
        *,
        dataset: Any,  # noqa: ARG001
        preset: str,  # noqa: ARG001
        n_samples: int,
        seed: int,  # noqa: ARG001
        remediation_profile: str | None = None,
    ) -> Any:
        assert n_samples == n_rows
        X = np.zeros((n_rows, wl.size), dtype=float)
        # R3d-like base.
        X[:, support_mask] = 0.05
        X[:, ~support_mask] = 0.02
        if remediation_profile == (
            "r9c_diesel_selective_ch_bandwidth_damping_v1"
        ):
            X[:, support_mask] = X[:, support_mask] + shape_on_support[None, :]
        return SimpleNamespace(
            X=X,
            y=np.zeros(n_rows, dtype=float),
            wavelengths=wl,
            metadata={},
        )

    monkeypatch.setattr(
        exp09,
        "discover_local_real_datasets",
        lambda root: ([diesel_dataset], []),
    )
    monkeypatch.setattr(
        exp12,
        "discover_local_real_datasets",
        lambda root: ([diesel_dataset], []),
    )
    monkeypatch.setattr(
        exp12,
        "load_real_spectra",
        lambda dataset, root: (fake_real, wl),
    )
    monkeypatch.setattr(exp09, "_stable_dataset_seed", fake_stable_dataset_seed)
    monkeypatch.setattr(
        exp09,
        "_build_baseline_synthetic_run",
        fake_build_baseline_synthetic_run,
    )

    profiles = (
        "r3d_diesel_matrix_v1",
        "r9c_diesel_selective_ch_bandwidth_damping_v1",
    )
    result = exp12.run_audit(
        root=tmp_path,
        seeds=(20260501,),
        n_synthetic_samples=n_rows,
        max_real_samples=n_rows,
        max_sentinel_datasets=1,
        sentinel_tokens=["DIESEL"],
        profiles=profiles,
    )

    assert result["status"] == "done"
    rows = {row.remediation_profile: row for row in result["rows"]}
    r3d = rows["r3d_diesel_matrix_v1"]
    r9c = rows["r9c_diesel_selective_ch_bandwidth_damping_v1"]
    paired = exp12._paired_deltas_vs_references(result["rows"])
    assert len(paired) == 1
    entry = paired[0]
    # Off-support paired delta of R9c vs R3d is zero by construction.
    assert entry[
        "delta_r9c_minus_r3d_diesel_matrix_v1__off_support_weighted_delta"
    ] == pytest.approx(0.0, abs=1e-12)
    # Support-side paired delta must equal the mean of the wavelength-
    # dependent shape on the support.
    expected_support_delta = float(shape_on_support.mean())
    assert entry[
        "delta_r9c_minus_r3d_diesel_matrix_v1__support_mean_delta"
    ] == pytest.approx(expected_support_delta, abs=1e-12)
    assert (r9c.support_mean_delta or 0.0) - (r3d.support_mean_delta or 0.0) == (
        pytest.approx(expected_support_delta, abs=1e-12)
    )


def test_write_csv_includes_paired_delta_columns_on_r9c_rows(
    tmp_path: Path,
) -> None:
    """CSV must carry paired delta columns (R9c minus reference) populated on
    R9c compared rows and blank on non-R9c rows."""
    exp12 = _load_exp12()
    rows = [
        _make_compared_row(
            exp12,
            profile="r9c_diesel_selective_ch_bandwidth_damping_v1",
            seed=20260501,
            global_mean_delta=0.0015,
            support_weight=1.0,
            off_support_weight=0.0,
            morphology_gap_score=1.05,
            mean_curve_corr=0.30,
            log10_derivative_std_p50_ratio=0.05,
            dominant_morphology_gap="mean_shift",
            support_mean_delta=0.0015,
            support_weighted_delta=0.0015,
        ),
        _make_compared_row(
            exp12,
            profile="r4c_diesel_balanced_derivative_v1",
            seed=20260501,
            global_mean_delta=0.0035,
            support_weight=1.0,
            off_support_weight=0.0,
            morphology_gap_score=1.30,
            mean_curve_corr=0.20,
            log10_derivative_std_p50_ratio=0.05,
            dominant_morphology_gap="mean_shift",
            support_mean_delta=0.0035,
            support_weighted_delta=0.0035,
        ),
    ]
    csv_path = tmp_path / "r9c.csv"
    exp12.write_csv(rows, csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        records = list(reader)
        fieldnames = reader.fieldnames

    assert fieldnames is not None
    assert (
        "delta_r9c_minus_r4c_diesel_balanced_derivative_v1__global_mean_delta"
        in fieldnames
    )
    assert (
        "delta_r9c_minus_r4c_diesel_balanced_derivative_v1__morphology_gap_score"
        in fieldnames
    )

    r9c_record = next(
        rec
        for rec in records
        if rec["remediation_profile"]
        == "r9c_diesel_selective_ch_bandwidth_damping_v1"
    )
    assert float(
        r9c_record[
            "delta_r9c_minus_r4c_diesel_balanced_derivative_v1__global_mean_delta"
        ]
    ) == pytest.approx(0.0015 - 0.0035)
    assert float(
        r9c_record[
            "delta_r9c_minus_r4c_diesel_balanced_derivative_v1__morphology_gap_score"
        ]
    ) == pytest.approx(1.05 - 1.30)

    r4c_record = next(
        rec
        for rec in records
        if rec["remediation_profile"] == "r4c_diesel_balanced_derivative_v1"
    )
    # Non-R9c rows leave the paired delta cells blank.
    assert (
        r4c_record[
            "delta_r9c_minus_r4c_diesel_balanced_derivative_v1__global_mean_delta"
        ]
        == ""
    )
