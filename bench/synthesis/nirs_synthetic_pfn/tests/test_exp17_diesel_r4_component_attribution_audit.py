"""Tests for the R9g DIESEL R4 component attribution audit."""

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


def _load_exp17() -> ModuleType:
    return _load_module(
        "exp17_diesel_r4_component_attribution_audit",
        "exp17_diesel_r4_component_attribution_audit.py",
    )


def _make_row(
    exp17: ModuleType,
    *,
    profile: str,
    seed: int = 20260501,
    global_mean_delta: float,
    morphology_gap_score: float,
    derivative: float,
    mean_curve_corr: float,
    guard_clip_fraction: float | None = None,
) -> Any:
    return exp17.R9gRow(
        status="compared",
        seed=seed,
        remediation_profile=profile,
        effective_remediation_profile=profile,
        source="AOM_regression",
        task="regression",
        dataset="DIESEL/DIESEL_bp50_246_b-a",
        synthetic_preset="fuel",
        effective_matrix_route="diesel_fuel_matrix",
        comparison_space=exp17.COMPARISON_SPACE,
        n_real_samples=8,
        n_synthetic_samples=8,
        n_wavelengths=6,
        wavelength_min=750.0,
        wavelength_max=1550.0,
        support_low_nm=exp17.SUPPORT_LOW_NM,
        support_high_nm=exp17.SUPPORT_HIGH_NM,
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
        audit_scope=exp17.R9G_AUDIT_SCOPE,
        blocked_reason="",
    )


def _synthetic_rows(exp17: ModuleType) -> list[Any]:
    return [
        _make_row(
            exp17,
            profile="r3d_diesel_matrix_v1",
            global_mean_delta=0.0057,
            morphology_gap_score=1.60,
            derivative=-0.03,
            mean_curve_corr=0.035,
        ),
        _make_row(
            exp17,
            profile="r4a_diesel_basis_v1",
            global_mean_delta=0.0047,
            morphology_gap_score=1.56,
            derivative=-0.24,
            mean_curve_corr=0.053,
        ),
        _make_row(
            exp17,
            profile="r4b_diesel_derivative_restore_v1",
            global_mean_delta=0.0046,
            morphology_gap_score=1.45,
            derivative=-0.09,
            mean_curve_corr=0.040,
        ),
        _make_row(
            exp17,
            profile="r4c_diesel_balanced_derivative_v1",
            global_mean_delta=0.0051,
            morphology_gap_score=1.50,
            derivative=-0.07,
            mean_curve_corr=0.037,
        ),
        _make_row(
            exp17,
            profile="r9e_diesel_pathlength_reference_attenuation_v1",
            global_mean_delta=0.0055,
            morphology_gap_score=1.57,
            derivative=-0.04,
            mean_curve_corr=0.035,
            guard_clip_fraction=0.0,
        ),
        _make_row(
            exp17,
            profile="r9f_diesel_pre_offset_pathlength_reference_attenuation_v1",
            global_mean_delta=0.00565,
            morphology_gap_score=1.595,
            derivative=-0.032,
            mean_curve_corr=0.034,
            guard_clip_fraction=0.0,
        ),
    ]


def test_exp17_constants_and_profile_contract() -> None:
    exp17 = _load_exp17()

    assert exp17.R9G_AUDITED_PROFILES == (
        "r3d_diesel_matrix_v1",
        "r4a_diesel_basis_v1",
        "r4b_diesel_derivative_restore_v1",
        "r4c_diesel_balanced_derivative_v1",
        "r9e_diesel_pathlength_reference_attenuation_v1",
        "r9f_diesel_pre_offset_pathlength_reference_attenuation_v1",
    )
    assert exp17.R9G_FOCUS_PROFILES == (
        "r4b_diesel_derivative_restore_v1",
        "r4c_diesel_balanced_derivative_v1",
    )
    assert exp17.R9G_BASE_PROFILE == "r3d_diesel_matrix_v1"
    assert exp17.COMPARISON_SPACE == "uncalibrated_raw"
    assert exp17.SUPPORT_LOW_NM == 750.0
    assert exp17.SUPPORT_HIGH_NM == 1550.0
    audit_fields = exp17._audit_fields()
    assert audit_fields["audit_scope"] == exp17.R9G_AUDIT_SCOPE
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
    exp17 = _load_exp17()

    with pytest.raises(ValueError, match="at least one"):
        exp17._validate_profiles(())
    with pytest.raises(ValueError, match="unknown R9g profiles"):
        exp17._validate_profiles(("r4d_not_real",))


def test_write_csv_and_render_markdown_attribution_contract(tmp_path: Path) -> None:
    exp17 = _load_exp17()
    rows = _synthetic_rows(exp17)
    csv_path = tmp_path / "r9g.csv"
    report_path = tmp_path / "r9g.md"

    exp17.write_csv(rows, csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        records = list(reader)

    assert len(records) == 6
    assert reader.fieldnames == exp17._csv_fieldnames()
    delta_col = (
        "delta_r4b_diesel_derivative_restore_v1_minus_"
        "r9e_diesel_pathlength_reference_attenuation_v1__morphology_gap_score"
    )
    dominant_col = (
        "dominant_gap_r4c_diesel_balanced_derivative_v1_minus_"
        "r9f_diesel_pre_offset_pathlength_reference_attenuation_v1"
    )
    assert delta_col in reader.fieldnames
    assert dominant_col in reader.fieldnames
    r4b_record = next(
        row
        for row in records
        if row["remediation_profile"] == "r4b_diesel_derivative_restore_v1"
    )
    assert float(r4b_record[delta_col]) == pytest.approx(-0.12)
    assert r4b_record["audit_scope"] == exp17.R9G_AUDIT_SCOPE
    assert r4b_record["audit_uses_pca"] == "False"
    assert r4b_record["audit_metrics_modified"] == "False"

    result = {
        "status": "done",
        "rows": rows,
        "real_runnable_count": 1,
        "real_sentinel_candidate_count": 1,
        "real_selected_count": 1,
        "sentinel_tokens": ["DIESEL"],
        "seeds": [20260501],
        "audited_profiles": list(exp17.R9G_AUDITED_PROFILES),
        "support_low_nm": exp17.SUPPORT_LOW_NM,
        "support_high_nm": exp17.SUPPORT_HIGH_NM,
    }
    md = exp17.render_markdown(
        result=result,
        report_path=report_path,
        csv_path=csv_path,
        n_synthetic_samples=8,
        max_real_samples=8,
        max_sentinel_datasets=1,
        seeds=[20260501],
        sentinel_tokens=["DIESEL"],
        profiles=exp17.R9G_AUDITED_PROFILES,
    )

    assert "# R9g DIESEL R4 Component Attribution Diagnostic Audit" in md
    assert "diagnostic-only" in md.lower()
    assert "not promoted" in md.lower()
    assert "no gate" in md.lower()
    assert "No `nirs4all/` integration" in md
    assert "No R9e/R9f attenuation amplitude retuning" in md
    assert "Support CH centers/drop 1720" in md
    assert "Width/gain derivative restoration" in md
    assert "Damping windows" in md
    assert "975 nm continuum hump" in md
    assert "non-isolable without new builder variants" in md
    assert "R3d remains the accepted DIESEL baseline" in md
    assert "B2 PASS" not in md.upper().replace(":", "")


def test_cli_writes_report_and_csv_with_patched_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exp17 = _load_exp17()
    rows = _synthetic_rows(exp17)
    result = {
        "status": "done",
        "rows": rows,
        "real_runnable_count": 1,
        "real_sentinel_candidate_count": 1,
        "real_selected_count": 1,
        "sentinel_tokens": ["DIESEL"],
        "seeds": [20260501],
        "audited_profiles": list(exp17.R9G_AUDITED_PROFILES),
        "support_low_nm": exp17.SUPPORT_LOW_NM,
        "support_high_nm": exp17.SUPPORT_HIGH_NM,
    }
    report_path = tmp_path / "report.md"
    csv_path = tmp_path / "report.csv"

    monkeypatch.setattr(exp17, "run_audit", lambda **_: result)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "exp17",
            "--report",
            str(report_path),
            "--csv",
            str(csv_path),
            "--seeds",
            "20260501",
            "--profiles",
            ",".join(exp17.R9G_AUDITED_PROFILES),
        ],
    )

    exp17.main()

    assert report_path.exists()
    assert csv_path.exists()
    assert "component attribution" in report_path.read_text(encoding="utf-8").lower()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        records = list(DictReader(handle))
    assert len(records) == 6
