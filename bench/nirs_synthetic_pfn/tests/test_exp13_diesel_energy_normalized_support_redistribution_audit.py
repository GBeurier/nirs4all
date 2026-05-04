"""Tests for the R9d DIESEL energy-normalized mean-neutral support
redistribution audit (exp13)."""

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


def _load_exp13() -> ModuleType:
    return _load_module(
        "exp13_diesel_energy_normalized_support_redistribution_audit",
        "exp13_diesel_energy_normalized_support_redistribution_audit.py",
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


def test_exp13_exposes_audited_profiles_and_default_seeds() -> None:
    exp13 = _load_exp13()
    assert exp13.R9D_AUDITED_PROFILES == (
        "r3d_diesel_matrix_v1",
        "r4a_diesel_basis_v1",
        "r4b_diesel_derivative_restore_v1",
        "r4c_diesel_balanced_derivative_v1",
        "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1",
        "r9b_diesel_support_intercept_v1",
        "r9c_diesel_selective_ch_bandwidth_damping_v1",
        "r9d_diesel_energy_normalized_support_redistribution_v1",
    )
    assert exp13.R9D_PAIRED_REFERENCE_PROFILES == (
        "r4a_diesel_basis_v1",
        "r4b_diesel_derivative_restore_v1",
        "r4c_diesel_balanced_derivative_v1",
        "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1",
        "r9b_diesel_support_intercept_v1",
        "r9c_diesel_selective_ch_bandwidth_damping_v1",
        "r3d_diesel_matrix_v1",
    )
    assert exp13.R9D_FOCUS_PROFILE == (
        "r9d_diesel_energy_normalized_support_redistribution_v1"
    )
    assert exp13.DEFAULT_SEEDS == (20260501, 20260502, 20260503)
    assert exp13.DEFAULT_N_SYNTHETIC_SAMPLES == 64
    assert exp13.DEFAULT_MAX_REAL_SAMPLES == 64
    assert exp13.SUPPORT_LOW_NM == 750.0
    assert exp13.SUPPORT_HIGH_NM == 1550.0
    assert exp13.COMPARISON_SPACE == "uncalibrated_raw"
    assert exp13.R9D_AUDIT_SCOPE == (
        "bench_only_r9d_diesel_energy_normalized_support_redistribution_audit"
    )
    assert exp13.DEFAULT_REPORT.name == (
        "r9d_diesel_energy_normalized_support_redistribution_audit.md"
    )
    assert exp13.DEFAULT_CSV.name == (
        "r9d_diesel_energy_normalized_support_redistribution_audit.csv"
    )


# ---------------------------------------------------------------------------
# Audit/run shape and metadata flags.
# ---------------------------------------------------------------------------


def test_run_audit_returns_blocked_no_real_data_when_cohorts_are_empty(
    tmp_path: Path,
) -> None:
    exp13 = _load_exp13()
    _write_empty_cohorts(tmp_path)

    result = exp13.run_audit(
        root=tmp_path,
        seeds=(20260501,),
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=2,
    )

    assert result["status"] == "blocked_no_real_data"
    assert result["rows"] == []
    assert result["seeds"] == [20260501]
    assert result["audited_profiles"] == list(exp13.R9D_AUDITED_PROFILES)
    assert result["support_low_nm"] == 750.0
    assert result["support_high_nm"] == 1550.0


def test_run_audit_rejects_unknown_profile(tmp_path: Path) -> None:
    exp13 = _load_exp13()
    _write_empty_cohorts(tmp_path)

    with pytest.raises(ValueError):
        exp13.run_audit(
            root=tmp_path,
            seeds=(20260501,),
            n_synthetic_samples=4,
            max_real_samples=4,
            max_sentinel_datasets=2,
            profiles=("not_a_real_profile",),
        )


def test_run_audit_requires_at_least_one_seed(tmp_path: Path) -> None:
    exp13 = _load_exp13()
    _write_empty_cohorts(tmp_path)

    with pytest.raises(ValueError):
        exp13.run_audit(
            root=tmp_path,
            seeds=(),
            n_synthetic_samples=4,
            max_real_samples=4,
            max_sentinel_datasets=2,
        )


def test_blocked_path_records_remediation_profiles_and_audit_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exp13 = _load_exp13()
    exp09 = _load_exp09()
    _write_empty_cohorts(tmp_path)

    candidates = [_make_dataset("DIESEL_bp50_246_b-a")]

    monkeypatch.setattr(
        exp09,
        "discover_local_real_datasets",
        lambda root: (list(candidates), []),
    )
    monkeypatch.setattr(
        exp13,
        "discover_local_real_datasets",
        lambda root: (list(candidates), []),
    )
    monkeypatch.setattr(
        exp13,
        "load_real_spectra",
        lambda dataset, root: (_ for _ in ()).throw(
            RuntimeError("synthetic-test: no real spectra available")
        ),
    )

    profiles = (
        "r3d_diesel_matrix_v1",
        "r9d_diesel_energy_normalized_support_redistribution_v1",
    )
    seeds = (20260501, 20260502)
    result = exp13.run_audit(
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
        assert row.audit_scope == exp13.R9D_AUDIT_SCOPE
        assert row.comparison_space == exp13.COMPARISON_SPACE
        assert row.remediation_profile in profiles
        assert row.seed in seeds


# ---------------------------------------------------------------------------
# CSV / Markdown shape.
# ---------------------------------------------------------------------------


def test_write_csv_empty_rows_emits_stable_header(tmp_path: Path) -> None:
    exp13 = _load_exp13()
    csv_path = tmp_path / "empty.csv"

    exp13.write_csv([], csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        records = list(reader)
    assert records == []
    fieldnames = reader.fieldnames
    assert fieldnames is not None
    assert list(fieldnames) == exp13._csv_fieldnames()
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
    # All paired-delta columns required by the R9d spec must be present
    # (R9d minus each of R4a/R4b/R4c/R8b/R9b/R9c/R3d on global_mean,
    # support_mean, morphology_gap_score, mean_curve_corr,
    # support_weighted_delta, off_support_weighted_delta).
    for ref in exp13.R9D_PAIRED_REFERENCE_PROFILES:
        for attr in exp13.PAIRED_DELTA_ATTRS:
            col = f"delta_r9d_minus_{ref}__{attr}"
            assert col in fieldnames, f"missing paired delta column {col!r}"


def _make_compared_row(
    exp13: ModuleType,
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
        "audit_scope": exp13.R9D_AUDIT_SCOPE,
    }
    if support_mean_delta is None:
        support_mean_delta = global_mean_delta
    if support_weighted_delta is None:
        support_weighted_delta = global_mean_delta
    return exp13.R9dRow(
        status="compared",
        seed=seed,
        remediation_profile=profile,
        effective_remediation_profile=profile,
        source="AOM_regression",
        task="regression",
        dataset="DIESEL_bp50_246_b-a/ds",
        synthetic_preset="fuel",
        effective_matrix_route="diesel_fuel_matrix",
        comparison_space=exp13.COMPARISON_SPACE,
        n_real_samples=8,
        n_synthetic_samples=8,
        n_wavelengths=10,
        wavelength_min=900.0,
        wavelength_max=1550.0,
        support_low_nm=exp13.SUPPORT_LOW_NM,
        support_high_nm=exp13.SUPPORT_HIGH_NM,
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
    exp13: ModuleType, rows: list[Any], tmp_path: Path
) -> Any:
    result = {
        "status": "done",
        "rows": rows,
        "real_runnable_count": 1,
        "real_sentinel_candidate_count": 1,
        "real_selected_count": 1,
        "sentinel_tokens": ["DIESEL"],
        "seeds": [20260501],
        "audited_profiles": list(exp13.R9D_AUDITED_PROFILES),
        "support_low_nm": exp13.SUPPORT_LOW_NM,
        "support_high_nm": exp13.SUPPORT_HIGH_NM,
    }
    return exp13.render_markdown(
        result=result,
        report_path=tmp_path / "r9d.md",
        csv_path=tmp_path / "r9d.csv",
        n_synthetic_samples=64,
        max_real_samples=64,
        max_sentinel_datasets=8,
        seeds=[20260501],
        sentinel_tokens=["DIESEL"],
        profiles=list(exp13.R9D_AUDITED_PROFILES),
    )


def test_render_markdown_disclaimer_and_no_gate_claims(tmp_path: Path) -> None:
    exp13 = _load_exp13()
    rows = [
        _make_compared_row(
            exp13,
            profile="r9d_diesel_energy_normalized_support_redistribution_v1",
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
    md = _render_with_rows(exp13, rows, tmp_path)

    # mechanistic / diagnostic-only / non-gate disclaimers.
    assert "uncalibrated_raw" in md
    assert "mechanistic" in md.lower()
    assert "diagnostic-only" in md.lower()
    assert "no calibration" in md.lower()
    assert "no PCA/covariance" in md
    assert "no ML/DL" in md
    assert "B2/B3/B4/B5" in md
    assert "750-1550" in md
    assert "DIESEL" in md
    # R9d base / fallback is R3d, not R4c.
    assert "R3d" in md
    assert "base / fallback is R3d" in md
    assert "R4c is included as a reference comparison only" in md
    # No threshold / metric mutation, no integration.
    assert "does not modify any gate threshold" in md.lower()
    assert "does not authorize any nirs4all integration" in md.lower()
    assert "no integration into nirs4all" in md.lower()
    # Pre-declared mechanistic constants disclaimer (R9d-specific).
    assert "PRE-DECLARED MECHANISTIC CONSTANTS" in md
    assert "general liquid-hydrocarbon NIR energy redistribution prior" in md
    assert "NOT chosen from any R9a/R9b/R9c mean-shift residual delta" in md
    # No promotion / no gate claims.
    assert "not a promotion over r3d" in md.lower()
    # Mean-neutral / energy-normalized / no scalar offset.
    assert "no scalar offset is added by R9d" in md.lower() or (
        "not a scalar offset" in md.lower()
    )
    assert "mean-neutral" in md.lower()
    assert "energy-normalized" in md.lower()
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
        "support_redistribution_only=true",
        "support_redistribution_mean_neutral=true",
        "support_redistribution_energy_normalized=true",
    ):
        assert flag in md, f"missing audit flag {flag!r}"
    assert (
        "constants_source=predeclared_general_liquid_hydrocarbon_nir_"
        "energy_redistribution_prior"
    ) in md
    assert "B2 PASS" not in md.upper().replace(":", "")


def test_render_markdown_includes_per_profile_and_paired_delta_tables(
    tmp_path: Path,
) -> None:
    exp13 = _load_exp13()
    rows = [
        _make_compared_row(
            exp13,
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
            exp13,
            profile="r9d_diesel_energy_normalized_support_redistribution_v1",
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
    md = _render_with_rows(exp13, rows, tmp_path)

    assert "## Per-Profile Synthesis (compared rows only)" in md
    assert "median global mean delta" in md
    assert "median support mean delta" in md
    assert "median morphology gap score" in md
    assert "`r4c_diesel_balanced_derivative_v1`" in md
    assert (
        "`r9d_diesel_energy_normalized_support_redistribution_v1`" in md
    )
    assert "## Paired Deltas: R9d minus reference profile" in md
    assert "r9d vs r4a global" in md
    assert "r9d vs r4b global" in md
    assert "r9d vs r4c global" in md
    assert "r9d vs r8b global" in md
    assert "r9d vs r9b global" in md
    assert "r9d vs r9c global" in md
    assert "r9d vs r3d global" in md
    assert "r9d vs r3d morphology gap" in md
    assert "r9d vs r9c morphology gap" in md


def test_render_markdown_marks_r9d_rejected_when_medians_worsen(
    tmp_path: Path,
) -> None:
    exp13 = _load_exp13()
    rows = [
        _make_compared_row(
            exp13,
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
            exp13,
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
            exp13,
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
            exp13,
            profile="r9c_diesel_selective_ch_bandwidth_damping_v1",
            seed=20260501,
            global_mean_delta=0.007,
            support_weight=1.0,
            off_support_weight=0.0,
            morphology_gap_score=1.70,
            mean_curve_corr=0.20,
            log10_derivative_std_p50_ratio=0.05,
            dominant_morphology_gap="mean_shift",
        ),
        _make_compared_row(
            exp13,
            profile="r9d_diesel_energy_normalized_support_redistribution_v1",
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
    md = _render_with_rows(exp13, rows, tmp_path)

    assert "## Diagnostic Outcome" in md
    assert "rejected / not promoted" in md.lower()
    assert "worsens median global mean delta vs" in md
    assert "worsens median morphology gap score vs" in md
    # The rejection message should reference the constraint that an energy-
    # normalized mean-neutral support redistribution alone did not close the
    # morphology gap.
    assert "did not, by itself, close the morphology gap" in md


def test_paired_deltas_include_r9d_minus_all_references_and_handle_missing_reference(
    tmp_path: Path,
) -> None:
    exp13 = _load_exp13()
    rows = [
        _make_compared_row(
            exp13,
            profile="r9d_diesel_energy_normalized_support_redistribution_v1",
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
            exp13,
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
            exp13,
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
            exp13,
            profile="r9c_diesel_selective_ch_bandwidth_damping_v1",
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

    paired = exp13._paired_deltas_vs_references(rows)
    assert len(paired) == 1
    entry = paired[0]
    assert entry["seed"] == 20260501
    assert entry["dataset"] == "DIESEL_bp50_246_b-a/ds"
    assert entry["r9d_global_mean_delta"] == pytest.approx(0.0015)
    # r4c reference present: r9d - r4c.
    assert entry[
        "delta_r9d_minus_r4c_diesel_balanced_derivative_v1__global_mean_delta"
    ] == pytest.approx(0.0015 - 0.0035)
    assert entry[
        "delta_r9d_minus_r4c_diesel_balanced_derivative_v1__morphology_gap_score"
    ] == pytest.approx(1.05 - 1.30)
    # r9c reference present: r9d - r9c.
    assert entry[
        "delta_r9d_minus_r9c_diesel_selective_ch_bandwidth_damping_v1__global_mean_delta"
    ] == pytest.approx(0.0015 - 0.0042)
    # r3d reference present.
    assert entry[
        "delta_r9d_minus_r3d_diesel_matrix_v1__global_mean_delta"
    ] == pytest.approx(0.0015 - 0.0050)
    # r4a, r4b, r8b, r9b references missing -> NA.
    for missing_ref in (
        "r4a_diesel_basis_v1",
        "r4b_diesel_derivative_restore_v1",
        "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1",
        "r9b_diesel_support_intercept_v1",
    ):
        for attr in exp13.PAIRED_DELTA_ATTRS:
            assert (
                entry[f"delta_r9d_minus_{missing_ref}__{attr}"] is None
            ), f"expected NA for missing reference {missing_ref!r} attr {attr!r}"


def test_paired_deltas_skip_when_r9d_missing(tmp_path: Path) -> None:
    exp13 = _load_exp13()
    rows = [
        _make_compared_row(
            exp13,
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
        # No R9d compared row -> paired table must be empty.
    ]
    assert exp13._paired_deltas_vs_references(rows) == []


def test_render_markdown_emits_off_support_null_note_when_all_zero(
    tmp_path: Path,
) -> None:
    exp13 = _load_exp13()
    rows = [
        _make_compared_row(
            exp13,
            profile="r9d_diesel_energy_normalized_support_redistribution_v1",
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
    md = _render_with_rows(exp13, rows, tmp_path)
    assert exp13.OFF_SUPPORT_NULL_NOTE in md


def test_aggregate_by_profile_ignores_blocked_rows(tmp_path: Path) -> None:
    exp13 = _load_exp13()
    compared = _make_compared_row(
        exp13,
        profile="r9d_diesel_energy_normalized_support_redistribution_v1",
        seed=20260501,
        global_mean_delta=0.001,
        support_weight=1.0,
        off_support_weight=0.0,
        morphology_gap_score=1.10,
        mean_curve_corr=0.30,
        log10_derivative_std_p50_ratio=0.05,
        dominant_morphology_gap="mean_shift",
    )
    blocked = exp13._blocked_row(
        seed=20260501,
        requested_profile="r3d_diesel_matrix_v1",
        effective_profile="r3d_diesel_matrix_v1",
        dataset=_make_dataset("DIESEL_bp50_246_b-a"),
        preset="fuel",
        blocked_reason="synthetic-test",
    )
    summary = exp13._aggregate_by_profile([compared, blocked])
    assert len(summary) == 1
    assert summary[0]["profile"] == (
        "r9d_diesel_energy_normalized_support_redistribution_v1"
    )
    assert summary[0]["n"] == 1


def test_run_audit_paired_delta_against_synthetic_in_memory_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end paired-delta check on a deterministic monkeypatched cohort.

    The fake synthetic builder produces an R3d-like base for every profile
    except R9d, where a wavelength-dependent multiplicative redistribution is
    applied on the support and the per-row support mean is preserved by
    construction. The paired off-support delta of R9d vs R3d must therefore
    be exactly zero, and the support-side mean delta of R9d vs R3d must be
    zero (mean-preservation) while at least one wavelength on the support
    must differ from the R3d base.
    """
    exp13 = _load_exp13()
    exp09 = _load_exp09()
    _write_empty_cohorts(tmp_path)

    diesel_dataset = _make_dataset("DIESEL_bp50_246_b-a")
    wl = np.array(
        [700.0, 800.0, 900.0, 1000.0, 1100.0, 1600.0, 1700.0], dtype=float
    )
    support_mask = (wl >= exp13.SUPPORT_LOW_NM) & (wl <= exp13.SUPPORT_HIGH_NM)
    fake_real = np.zeros((20, wl.size), dtype=float)

    n_rows = 8
    # Mean-neutral wavelength-dependent shape on the support.
    raw_support_shape = np.array(
        [-0.02, 0.01, 0.03, -0.02], dtype=float
    )
    assert support_mask.sum() == raw_support_shape.size
    # Force the support-side shape to be exactly mean-zero so the synthetic
    # cohort really preserves the support mean.
    mean_neutral_shape = raw_support_shape - float(raw_support_shape.mean())
    assert float(mean_neutral_shape.mean()) == pytest.approx(0.0, abs=1e-15)

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
            "r9d_diesel_energy_normalized_support_redistribution_v1"
        ):
            # Mean-neutral additive bump on the support: per-row support mean
            # is preserved by construction.
            X[:, support_mask] = (
                X[:, support_mask] + mean_neutral_shape[None, :]
            )
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
        exp13,
        "discover_local_real_datasets",
        lambda root: ([diesel_dataset], []),
    )
    monkeypatch.setattr(
        exp13,
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
        "r9d_diesel_energy_normalized_support_redistribution_v1",
    )
    result = exp13.run_audit(
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
    r9d = rows["r9d_diesel_energy_normalized_support_redistribution_v1"]
    paired = exp13._paired_deltas_vs_references(result["rows"])
    assert len(paired) == 1
    entry = paired[0]
    # Off-support paired delta of R9d vs R3d is zero by construction.
    assert entry[
        "delta_r9d_minus_r3d_diesel_matrix_v1__off_support_weighted_delta"
    ] == pytest.approx(0.0, abs=1e-12)
    # Support-side mean delta is zero (mean-neutral redistribution).
    assert entry[
        "delta_r9d_minus_r3d_diesel_matrix_v1__support_mean_delta"
    ] == pytest.approx(0.0, abs=1e-12)
    assert (r9d.support_mean_delta or 0.0) - (r3d.support_mean_delta or 0.0) == (
        pytest.approx(0.0, abs=1e-12)
    )


def test_write_csv_includes_paired_delta_columns_on_r9d_rows(
    tmp_path: Path,
) -> None:
    """CSV must carry paired delta columns (R9d minus reference) populated on
    R9d compared rows and blank on non-R9d rows."""
    exp13 = _load_exp13()
    rows = [
        _make_compared_row(
            exp13,
            profile="r9d_diesel_energy_normalized_support_redistribution_v1",
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
            exp13,
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
    csv_path = tmp_path / "r9d.csv"
    exp13.write_csv(rows, csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        records = list(reader)
        fieldnames = reader.fieldnames

    assert fieldnames is not None
    assert (
        "delta_r9d_minus_r4c_diesel_balanced_derivative_v1__global_mean_delta"
        in fieldnames
    )
    assert (
        "delta_r9d_minus_r4c_diesel_balanced_derivative_v1__morphology_gap_score"
        in fieldnames
    )

    r9d_record = next(
        rec
        for rec in records
        if rec["remediation_profile"]
        == "r9d_diesel_energy_normalized_support_redistribution_v1"
    )
    assert float(
        r9d_record[
            "delta_r9d_minus_r4c_diesel_balanced_derivative_v1__global_mean_delta"
        ]
    ) == pytest.approx(0.0015 - 0.0035)
    assert float(
        r9d_record[
            "delta_r9d_minus_r4c_diesel_balanced_derivative_v1__morphology_gap_score"
        ]
    ) == pytest.approx(1.05 - 1.30)

    r4c_record = next(
        rec
        for rec in records
        if rec["remediation_profile"] == "r4c_diesel_balanced_derivative_v1"
    )
    # Non-R9d rows leave the paired delta cells blank.
    assert (
        r4c_record[
            "delta_r9d_minus_r4c_diesel_balanced_derivative_v1__global_mean_delta"
        ]
        == ""
    )
