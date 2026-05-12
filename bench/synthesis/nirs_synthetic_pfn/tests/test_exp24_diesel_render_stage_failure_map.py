"""Tests for the P2-01 DIESEL render-stage failure-map audit."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
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


def _load_exp24() -> ModuleType:
    return _load_module(
        "exp24_diesel_render_stage_failure_map",
        "exp24_diesel_render_stage_failure_map.py",
    )


def _make_row(
    exp24: ModuleType,
    *,
    profile: str,
    global_mean_delta: float,
    support_mean_delta: float,
    morphology_gap_score: float,
    derivative: float,
    mean_curve_corr: float,
    dominant_gap: str = "mean_shift",
    metadata: dict[str, Any] | None = None,
) -> Any:
    metric_fields = {
        "support_low_nm": exp24.SUPPORT_LOW_NM,
        "support_high_nm": exp24.SUPPORT_HIGH_NM,
        "support_count": 6,
        "off_support_count": 0,
        "support_weight": 1.0,
        "off_support_weight": 0.0,
        "support_mean_delta": support_mean_delta,
        "off_support_mean_delta": 0.0,
        "support_weighted_delta": support_mean_delta,
        "off_support_weighted_delta": 0.0,
        "global_mean_delta": global_mean_delta,
        "decomposition_residual": 0.0,
        "real_global_mean": 0.003,
        "synthetic_global_mean": 0.003 + global_mean_delta,
        "real_global_std": 0.014,
        "synthetic_global_std": 0.015,
        "log10_global_std_ratio": 0.03,
        "log10_amplitude_p50_ratio": 0.0,
        "log10_derivative_std_p50_ratio": derivative,
        "mean_curve_corr": mean_curve_corr,
        "morphology_gap_score": morphology_gap_score,
        "dominant_morphology_gap": dominant_gap,
    }
    return exp24.Exp24Row(
        status="compared",
        seed=20260501,
        remediation_profile=profile,
        effective_remediation_profile=profile,
        source="AOM_regression",
        task="regression",
        dataset="DIESEL/DIESEL_bp50_246_b-a",
        synthetic_preset="fuel",
        effective_matrix_route="diesel_fuel_matrix",
        comparison_space=exp24.COMPARISON_SPACE,
        **exp24._row_stage_and_failure(
            profile=profile,
            metadata=metadata,
            status="compared",
            metrics=metric_fields,
        ),
        n_real_samples=8,
        n_synthetic_samples=8,
        n_wavelengths=6,
        wavelength_min=750.0,
        wavelength_max=1550.0,
        **metric_fields,
        guard_clip_fraction=0.0 if profile == exp24.R9E_PROFILE else None,
        **exp24._audit_fields(),
        blocked_reason="",
    )


def _synthetic_rows(exp24: ModuleType) -> list[Any]:
    r9e_metadata = {
        "r2c_mechanistic_remediation": {
            "transform_params": {
                "support_reference_attenuation_application_stage": "after_r3d_output_clip",
                "support_reference_attenuation_support_nm": [750.0, 1550.0],
                "support_reference_attenuation_n_support_bins": 6,
                "support_reference_attenuation_guard_clip_fraction": 0.0,
            }
        }
    }
    return [
        _make_row(
            exp24,
            profile=exp24.R3D_PROFILE,
            global_mean_delta=0.0057,
            support_mean_delta=0.0057,
            morphology_gap_score=1.60,
            derivative=-0.03,
            mean_curve_corr=0.035,
        ),
        _make_row(
            exp24,
            profile=exp24.R9E_PROFILE,
            global_mean_delta=0.0055,
            support_mean_delta=0.0055,
            morphology_gap_score=1.57,
            derivative=-0.04,
            mean_curve_corr=0.035,
            metadata=r9e_metadata,
        ),
        _make_row(
            exp24,
            profile=exp24.R9J_PROFILE,
            global_mean_delta=0.0050,
            support_mean_delta=0.0050,
            morphology_gap_score=1.54,
            derivative=-0.05,
            mean_curve_corr=0.037,
        ),
        _make_row(
            exp24,
            profile=exp24.R9L_PROFILE,
            global_mean_delta=0.0049,
            support_mean_delta=0.0049,
            morphology_gap_score=1.50,
            derivative=-0.052,
            mean_curve_corr=0.038,
        ),
        _make_row(
            exp24,
            profile=exp24.R9M_PROFILE,
            global_mean_delta=0.0050,
            support_mean_delta=0.0050,
            morphology_gap_score=1.49,
            derivative=-0.05,
            mean_curve_corr=0.038,
        ),
        _make_row(
            exp24,
            profile=exp24.R4B_PROFILE,
            global_mean_delta=0.0046,
            support_mean_delta=0.0046,
            morphology_gap_score=1.45,
            derivative=-0.09,
            mean_curve_corr=0.040,
            dominant_gap="derivative_under",
        ),
        _make_row(
            exp24,
            profile=exp24.R4C_PROFILE,
            global_mean_delta=0.0051,
            support_mean_delta=0.0051,
            morphology_gap_score=1.50,
            derivative=-0.07,
            mean_curve_corr=0.037,
        ),
    ]


def test_exp24_constants_and_read_only_contract() -> None:
    exp24 = _load_exp24()

    assert exp24.EXP24_AUDITED_PROFILES == (
        "r3d_diesel_matrix_v1",
        "r9e_diesel_pathlength_reference_attenuation_v1",
        "r9j_diesel_residual_damping_isolation_v1",
        "r9l_diesel_residual_damping_clean_attenuation_v1",
        "r9m_diesel_width_gain_damping_clean_attenuation_v1",
        "r4b_diesel_derivative_restore_v1",
        "r4c_diesel_balanced_derivative_v1",
    )
    assert "r9n" not in ",".join(exp24.EXP24_AUDITED_PROFILES)
    assert exp24.EXP24_REFERENCE_PROFILES == (
        "r3d_diesel_matrix_v1",
        "r4c_diesel_balanced_derivative_v1",
    )
    assert Path("/tmp/exp24_diesel_render_stage_failure_map.md") == exp24.DEFAULT_REPORT
    assert Path("/tmp/exp24_diesel_render_stage_failure_map.csv") == exp24.DEFAULT_CSV
    assert exp24.PROFILE_STAGE_MAP[exp24.R9E_PROFILE]["stage_application"] == "after_r3d_output_clip"
    assert exp24.PROFILE_STAGE_MAP[exp24.R9J_PROFILE]["residual_damping_active"] is True
    assert exp24.PROFILE_STAGE_MAP[exp24.R9M_PROFILE]["ch_width_gain_changed"] is True
    audit_fields = exp24._audit_fields()
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
    assert audit_fields["audit_scope"] == exp24.EXP24_AUDIT_SCOPE


def test_exp24_validation_rejects_unknown_and_r9n() -> None:
    exp24 = _load_exp24()

    with pytest.raises(ValueError, match="at least one"):
        exp24._validate_profiles(())
    with pytest.raises(ValueError, match="unknown exp24 profiles"):
        exp24._validate_profiles(("r9n_not_allowed",))


def test_write_csv_and_render_markdown_failure_map_contract(tmp_path: Path) -> None:
    exp24 = _load_exp24()
    rows = _synthetic_rows(exp24)
    csv_path = tmp_path / "exp24.csv"
    report_path = tmp_path / "exp24.md"

    exp24.write_csv(rows, csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        records = list(reader)

    assert len(records) == 7
    assert reader.fieldnames == exp24._csv_fieldnames()
    assert Counter(row["status"] for row in records) == {"compared": 7}
    assert all(row["blocked_reason"] == "" for row in records)
    assert Counter(row["remediation_profile"] for row in records) == Counter(
        exp24.EXP24_AUDITED_PROFILES
    )
    delta_col = "delta_vs_r3d_diesel_matrix_v1__morphology_gap_score"
    corr_delta_col = "delta_vs_r4c_diesel_balanced_derivative_v1__mean_curve_corr"
    assert delta_col in reader.fieldnames
    assert corr_delta_col in reader.fieldnames
    r9m_record = next(row for row in records if row["remediation_profile"] == exp24.R9M_PROFILE)
    assert float(r9m_record[delta_col]) == pytest.approx(-0.11)
    assert r9m_record["render_stage_family"] == "width_gain_damping_plus_post_clip_attenuation"
    assert r9m_record["audit_uses_pca"] == "False"
    r4b_record = next(row for row in records if row["remediation_profile"] == exp24.R4B_PROFILE)
    assert r4b_record["report_only_failure_axis"] == "derivative_under"

    result = {
        "status": "done",
        "rows": rows,
        "real_runnable_count": 1,
        "real_sentinel_candidate_count": 1,
        "real_selected_count": 1,
        "sentinel_tokens": ["DIESEL"],
        "seeds": [20260501],
        "audited_profiles": list(exp24.EXP24_AUDITED_PROFILES),
        "support_low_nm": exp24.SUPPORT_LOW_NM,
        "support_high_nm": exp24.SUPPORT_HIGH_NM,
    }
    md = exp24.render_markdown(
        result=result,
        report_path=report_path,
        csv_path=csv_path,
        n_synthetic_samples=8,
        max_real_samples=8,
        max_sentinel_datasets=1,
        seeds=[20260501],
        sentinel_tokens=["DIESEL"],
        profiles=exp24.EXP24_AUDITED_PROFILES,
    )

    assert "# P2-01 DIESEL Render-Stage Failure Map" in md
    assert "read-only" in md
    assert "No R9n" in md
    assert "R3d remains the accepted DIESEL baseline" in md
    assert "report_only_failure_axis" in md
    assert "threshold mutation" in md
    assert "delta" in md.lower()
    assert f"--report {report_path}" in md
    assert f"--csv {csv_path}" in md
    assert "B2 PASS" not in md.upper().replace(":", "")
