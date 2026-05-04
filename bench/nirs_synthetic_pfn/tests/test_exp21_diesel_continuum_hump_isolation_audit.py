"""Tests for the R9k DIESEL continuum hump isolation audit."""

from __future__ import annotations

import importlib.util
import sys
from csv import DictReader
from pathlib import Path
from types import ModuleType
from typing import Any

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


def _load_exp21() -> ModuleType:
    return _load_module(
        "exp21_diesel_continuum_hump_isolation_audit",
        "exp21_diesel_continuum_hump_isolation_audit.py",
    )


def _make_row(
    exp21: ModuleType,
    *,
    profile: str,
    seed: int = 20260501,
    global_mean_delta: float,
    morphology_gap_score: float,
    derivative: float,
    mean_curve_corr: float,
    guard_clip_fraction: float | None = None,
) -> Any:
    return exp21.R9kRow(
        status="compared",
        seed=seed,
        remediation_profile=profile,
        effective_remediation_profile=profile,
        source="AOM_regression",
        task="regression",
        dataset="DIESEL/DIESEL_bp50_246_b-a",
        synthetic_preset="fuel",
        effective_matrix_route="diesel_fuel_matrix",
        comparison_space=exp21.COMPARISON_SPACE,
        n_real_samples=8,
        n_synthetic_samples=8,
        n_wavelengths=6,
        wavelength_min=750.0,
        wavelength_max=1550.0,
        support_low_nm=exp21.SUPPORT_LOW_NM,
        support_high_nm=exp21.SUPPORT_HIGH_NM,
        support_count=6,
        off_support_count=0,
        support_weight=1.0,
        off_support_weight=0.0,
        support_mean_delta=global_mean_delta,
        off_support_mean_delta=0.0,
        support_weighted_delta=global_mean_delta,
        off_support_weighted_delta=0.0,
        global_mean_delta=global_mean_delta,
        decomposition_residual=0.0,
        real_global_mean=0.003,
        synthetic_global_mean=0.003 + global_mean_delta,
        real_global_std=0.014,
        synthetic_global_std=0.015,
        log10_global_std_ratio=0.03,
        log10_amplitude_p50_ratio=0.0,
        log10_derivative_std_p50_ratio=derivative,
        mean_curve_corr=mean_curve_corr,
        morphology_gap_score=morphology_gap_score,
        dominant_morphology_gap="mean_shift",
        guard_clip_fraction=guard_clip_fraction,
        audit_calibration=False,
        audit_real_stat_capture=False,
        audit_uses_pca=False,
        audit_captures_noise=False,
        audit_uses_ml=False,
        audit_uses_dl=False,
        audit_label_inputs_used=False,
        audit_target_inputs_used=False,
        audit_split_inputs_used=False,
        audit_thresholds_modified=False,
        audit_metrics_modified=False,
        audit_source_oracle_used=False,
        audit_scope=exp21.R9K_AUDIT_SCOPE,
        blocked_reason="",
    )


def _synthetic_rows(exp21: ModuleType) -> list[Any]:
    return [
        _make_row(
            exp21,
            profile="r3d_diesel_matrix_v1",
            global_mean_delta=0.0057,
            morphology_gap_score=1.60,
            derivative=-0.03,
            mean_curve_corr=0.035,
        ),
        _make_row(
            exp21,
            profile="r4a_diesel_basis_v1",
            global_mean_delta=0.0047,
            morphology_gap_score=1.56,
            derivative=-0.24,
            mean_curve_corr=0.053,
        ),
        _make_row(
            exp21,
            profile="r4b_diesel_derivative_restore_v1",
            global_mean_delta=0.0046,
            morphology_gap_score=1.45,
            derivative=-0.09,
            mean_curve_corr=0.040,
        ),
        _make_row(
            exp21,
            profile="r4c_diesel_balanced_derivative_v1",
            global_mean_delta=0.0051,
            morphology_gap_score=1.50,
            derivative=-0.07,
            mean_curve_corr=0.037,
        ),
        _make_row(
            exp21,
            profile="r9e_diesel_pathlength_reference_attenuation_v1",
            global_mean_delta=0.0055,
            morphology_gap_score=1.57,
            derivative=-0.04,
            mean_curve_corr=0.035,
            guard_clip_fraction=0.0,
        ),
        _make_row(
            exp21,
            profile="r9f_diesel_pre_offset_pathlength_reference_attenuation_v1",
            global_mean_delta=0.00565,
            morphology_gap_score=1.595,
            derivative=-0.032,
            mean_curve_corr=0.034,
            guard_clip_fraction=0.0,
        ),
        _make_row(
            exp21,
            profile="r9h_diesel_support_ch_center_drop1720_isolation_v1",
            global_mean_delta=0.0057,
            morphology_gap_score=1.60,
            derivative=-0.03,
            mean_curve_corr=0.035,
        ),
        _make_row(
            exp21,
            profile="r9i_diesel_ch_width_gain_isolation_v1",
            global_mean_delta=0.0052,
            morphology_gap_score=1.52,
            derivative=-0.02,
            mean_curve_corr=0.036,
        ),
        _make_row(
            exp21,
            profile="r9j_diesel_residual_damping_isolation_v1",
            global_mean_delta=0.0050,
            morphology_gap_score=1.54,
            derivative=-0.05,
            mean_curve_corr=0.037,
        ),
        _make_row(
            exp21,
            profile=exp21.R9K_FOCUS_PROFILE,
            global_mean_delta=0.0050,
            morphology_gap_score=1.49,
            derivative=-0.05,
            mean_curve_corr=0.038,
            guard_clip_fraction=None,
        ),
    ]


def test_exp21_constants_and_profile_contract() -> None:
    exp21 = _load_exp21()

    assert exp21.R9K_AUDITED_PROFILES == (
        "r3d_diesel_matrix_v1",
        "r4a_diesel_basis_v1",
        "r4b_diesel_derivative_restore_v1",
        "r4c_diesel_balanced_derivative_v1",
        "r9e_diesel_pathlength_reference_attenuation_v1",
        "r9f_diesel_pre_offset_pathlength_reference_attenuation_v1",
        "r9h_diesel_support_ch_center_drop1720_isolation_v1",
        "r9i_diesel_ch_width_gain_isolation_v1",
        "r9j_diesel_residual_damping_isolation_v1",
        "r9k_diesel_continuum_hump_isolation_v1",
    )
    assert exp21.R9K_FOCUS_PROFILES == (
        "r9k_diesel_continuum_hump_isolation_v1",
    )
    assert exp21.R9K_REFERENCE_PROFILES == (
        "r3d_diesel_matrix_v1",
        "r9i_diesel_ch_width_gain_isolation_v1",
        "r9j_diesel_residual_damping_isolation_v1",
        "r4b_diesel_derivative_restore_v1",
        "r4c_diesel_balanced_derivative_v1",
        "r9e_diesel_pathlength_reference_attenuation_v1",
        "r9f_diesel_pre_offset_pathlength_reference_attenuation_v1",
    )
    assert exp21.R9K_BASE_PROFILE == "r3d_diesel_matrix_v1"
    assert exp21.COMPARISON_SPACE == "uncalibrated_raw"
    assert exp21.DEFAULT_SEEDS == (20260501, 20260502, 20260503)
    assert exp21.DEFAULT_N_SYNTHETIC_SAMPLES == 64
    assert exp21.DEFAULT_MAX_REAL_SAMPLES == 64
    assert exp21.DEFAULT_MAX_SENTINEL_DATASETS == 8
    assert exp21.DEFAULT_SENTINEL_TOKENS == ("DIESEL",)
    assert exp21.DEFAULT_REPORT.name == (
        "r9k_diesel_continuum_hump_isolation_audit.md"
    )
    assert Path(
        "bench/nirs_synthetic_pfn/reports/"
        "r9k_diesel_continuum_hump_isolation_audit.csv"
    ) == exp21.DEFAULT_CSV
    assert exp21.PAIRED_DELTA_ATTRS == (
        "global_mean_delta",
        "support_mean_delta",
        "morphology_gap_score",
        "log10_derivative_std_p50_ratio",
        "mean_curve_corr",
    )
    audit_fields = exp21._audit_fields()
    assert audit_fields["audit_scope"] == exp21.R9K_AUDIT_SCOPE
    forbidden = [
        "audit_calibration",
        "audit_real_stat_capture",
        "audit_uses_pca",
        "audit_captures_noise",
        "audit_uses_ml",
        "audit_uses_dl",
        "audit_label_inputs_used",
        "audit_target_inputs_used",
        "audit_split_inputs_used",
        "audit_thresholds_modified",
        "audit_metrics_modified",
        "audit_source_oracle_used",
    ]
    assert all(audit_fields[name] is False for name in forbidden)


def test_validate_profiles_rejects_unknown_profile() -> None:
    exp21 = _load_exp21()

    with pytest.raises(ValueError, match="at least one"):
        exp21._validate_profiles(())
    with pytest.raises(ValueError, match="unknown R9k profiles"):
        exp21._validate_profiles(("r9z_not_real",))


def test_write_csv_and_render_markdown_isolation_contract(tmp_path: Path) -> None:
    exp21 = _load_exp21()
    rows = _synthetic_rows(exp21)
    csv_path = tmp_path / "r9k.csv"
    report_path = tmp_path / "r9k.md"

    exp21.write_csv(rows, csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        records = list(reader)

    assert len(records) == 10
    assert reader.fieldnames == exp21._csv_fieldnames()
    delta_col = (
        "delta_r9k_diesel_continuum_hump_isolation_v1_minus_"
        "r3d_diesel_matrix_v1__morphology_gap_score"
    )
    r9i_delta_col = (
        "delta_r9k_diesel_continuum_hump_isolation_v1_minus_"
        "r9i_diesel_ch_width_gain_isolation_v1__mean_curve_corr"
    )
    r9j_delta_col = (
        "delta_r9k_diesel_continuum_hump_isolation_v1_minus_"
        "r9j_diesel_residual_damping_isolation_v1__morphology_gap_score"
    )
    dominant_col = (
        "dominant_gap_r9k_diesel_continuum_hump_isolation_v1_minus_"
        "r9f_diesel_pre_offset_pathlength_reference_attenuation_v1"
    )
    assert delta_col in reader.fieldnames
    assert r9i_delta_col in reader.fieldnames
    assert r9j_delta_col in reader.fieldnames
    assert dominant_col in reader.fieldnames
    r9k_record = next(
        row
        for row in records
        if row["remediation_profile"] == exp21.R9K_FOCUS_PROFILE
    )
    assert float(r9k_record[delta_col]) == pytest.approx(-0.11)
    assert r9k_record["audit_scope"] == exp21.R9K_AUDIT_SCOPE
    assert r9k_record["audit_uses_pca"] == "False"
    assert r9k_record["audit_metrics_modified"] == "False"

    result = {
        "status": "done",
        "rows": rows,
        "real_runnable_count": 1,
        "real_sentinel_candidate_count": 1,
        "real_selected_count": 1,
        "sentinel_tokens": ["DIESEL"],
        "seeds": [20260501],
        "audited_profiles": list(exp21.R9K_AUDITED_PROFILES),
        "support_low_nm": exp21.SUPPORT_LOW_NM,
        "support_high_nm": exp21.SUPPORT_HIGH_NM,
    }
    md = exp21.render_markdown(
        result=result,
        report_path=report_path,
        csv_path=csv_path,
        n_synthetic_samples=8,
        max_real_samples=8,
        max_sentinel_datasets=1,
        seeds=[20260501],
        sentinel_tokens=["DIESEL"],
        profiles=exp21.R9K_AUDITED_PROFILES,
    )

    assert "# R9k DIESEL Continuum Hump Isolation Diagnostic Audit" in md
    assert "diagnostic-only" in md.lower()
    assert "not promoted" in md.lower()
    assert "no gate" in md.lower()
    assert "No `nirs4all/` integration" in md
    assert "R3d remains the accepted DIESEL baseline" in md
    assert "R9k changes only `continuum_hump_center_nm = 975.0`" in md
    assert "R9k keeps R3d CH centers, width, and gain" in md
    assert "Damping windows: absent" in md
    assert "975 nm continuum hump: center `975.0`, width `72.0`" in md
    assert "Damping + continuum combination: absent" in md
    assert "Support intercept, support shape, redistribution, attenuation" in md
    assert "does not explain the R4b/R4c morphology gap advantage" in md
    assert f"--report {report_path}" in md
    assert f"--csv {csv_path}" in md
    assert "B2 PASS" not in md.upper().replace(":", "")
