"""Tests for the R9e DIESEL pathlength/reference attenuation audit."""

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


def _load_exp15() -> ModuleType:
    return _load_module(
        "exp15_diesel_pathlength_reference_attenuation_audit",
        "exp15_diesel_pathlength_reference_attenuation_audit.py",
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _make_row(
    exp15: ModuleType,
    *,
    profile: str,
    seed: int = 20260501,
    global_mean_delta: float = 0.002,
    support_mean_delta: float = 0.002,
    support_weighted_delta: float = 0.002,
    off_support_weighted_delta: float = 0.0,
    morphology_gap_score: float = 1.0,
    derivative: float = 0.04,
    mean_curve_corr: float = 0.5,
    guard_clip_fraction: float | None = None,
) -> Any:
    return exp15.R9eRow(
        status="compared",
        seed=seed,
        remediation_profile=profile,
        effective_remediation_profile=profile,
        source="AOM_regression",
        task="regression",
        dataset="DIESEL/DIESEL_bp50_246_b-a",
        synthetic_preset="fuel",
        effective_matrix_route="diesel_fuel_matrix",
        comparison_space=exp15.COMPARISON_SPACE,
        n_real_samples=8,
        n_synthetic_samples=8,
        n_wavelengths=6,
        wavelength_min=750.0,
        wavelength_max=1550.0,
        support_low_nm=exp15.SUPPORT_LOW_NM,
        support_high_nm=exp15.SUPPORT_HIGH_NM,
        support_count=6,
        off_support_count=0,
        support_weight=1.0,
        off_support_weight=0.0,
        support_mean_delta=support_mean_delta,
        off_support_mean_delta=0.0,
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
        audit_scope=exp15.R9E_AUDIT_SCOPE,
        blocked_reason="",
    )


def test_exp15_constants_and_profile_contract() -> None:
    exp15 = _load_exp15()

    assert exp15.R9E_AUDITED_PROFILES == (
        "r3d_diesel_matrix_v1",
        "r4b_diesel_derivative_restore_v1",
        "r4c_diesel_balanced_derivative_v1",
        "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1",
        "r9d_diesel_energy_normalized_support_redistribution_v1",
        "r9e_diesel_pathlength_reference_attenuation_v1",
    )
    assert exp15.R9E_FOCUS_PROFILE == (
        "r9e_diesel_pathlength_reference_attenuation_v1"
    )
    assert exp15.R9E_BASE_PROFILE == "r3d_diesel_matrix_v1"
    assert exp15.R9E_PAIRED_REFERENCE_PROFILES == (
        "r3d_diesel_matrix_v1",
        "r4c_diesel_balanced_derivative_v1",
        "r9d_diesel_energy_normalized_support_redistribution_v1",
    )
    assert exp15.SUPPORT_LOW_NM == 750.0
    assert exp15.SUPPORT_HIGH_NM == 1550.0
    assert exp15.COMPARISON_SPACE == "uncalibrated_raw"
    assert exp15.R9E_CONSTANTS_SOURCE == (
        "predeclared_generic_blank_reference_pathlength_attenuation_prior"
    )


def test_validate_profiles_rejects_unknown_profile() -> None:
    exp15 = _load_exp15()

    with pytest.raises(ValueError, match="at least one"):
        exp15._validate_profiles(())
    with pytest.raises(ValueError, match="unknown R9e profiles"):
        exp15._validate_profiles(("r9e_not_real",))


def test_write_csv_and_render_markdown_contract_with_synthetic_rows(
    tmp_path: Path,
) -> None:
    exp15 = _load_exp15()
    rows = [
        _make_row(
            exp15,
            profile="r3d_diesel_matrix_v1",
            global_mean_delta=0.004,
            morphology_gap_score=1.40,
            derivative=0.10,
            mean_curve_corr=0.20,
        ),
        _make_row(
            exp15,
            profile="r4c_diesel_balanced_derivative_v1",
            global_mean_delta=0.003,
            morphology_gap_score=1.30,
            derivative=0.08,
            mean_curve_corr=0.25,
        ),
        _make_row(
            exp15,
            profile="r9d_diesel_energy_normalized_support_redistribution_v1",
            global_mean_delta=0.0041,
            morphology_gap_score=1.41,
            derivative=0.11,
            mean_curve_corr=0.19,
        ),
        _make_row(
            exp15,
            profile="r9e_diesel_pathlength_reference_attenuation_v1",
            global_mean_delta=0.002,
            support_mean_delta=0.002,
            support_weighted_delta=0.002,
            morphology_gap_score=1.10,
            derivative=0.07,
            mean_curve_corr=0.35,
            guard_clip_fraction=0.0,
        ),
    ]
    csv_path = tmp_path / "r9e.csv"
    report_path = tmp_path / "r9e.md"

    exp15.write_csv(rows, csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        records = list(reader)

    assert len(records) == 4
    assert reader.fieldnames == exp15._csv_fieldnames()
    assert "guard_clip_fraction" in reader.fieldnames
    assert (
        "delta_r9e_minus_r3d_diesel_matrix_v1__global_mean_delta"
        in reader.fieldnames
    )
    r9e_record = records[-1]
    assert float(r9e_record["guard_clip_fraction"]) == pytest.approx(0.0)
    assert float(
        r9e_record["delta_r9e_minus_r3d_diesel_matrix_v1__global_mean_delta"]
    ) == pytest.approx(-0.002)
    assert float(
        r9e_record[
            "delta_r9e_minus_r9d_diesel_energy_normalized_support_redistribution_v1__mean_curve_corr"
        ]
    ) == pytest.approx(0.16)

    result = {
        "status": "done",
        "rows": rows,
        "real_runnable_count": 1,
        "real_sentinel_candidate_count": 1,
        "real_selected_count": 1,
        "sentinel_tokens": ["DIESEL"],
        "seeds": [20260501],
        "audited_profiles": list(exp15.R9E_AUDITED_PROFILES),
        "support_low_nm": exp15.SUPPORT_LOW_NM,
        "support_high_nm": exp15.SUPPORT_HIGH_NM,
    }
    md = exp15.render_markdown(
        result=result,
        report_path=report_path,
        csv_path=csv_path,
        n_synthetic_samples=8,
        max_real_samples=8,
        max_sentinel_datasets=1,
        seeds=[20260501],
        sentinel_tokens=["DIESEL"],
        profiles=exp15.R9E_AUDITED_PROFILES,
    )

    assert "# R9e DIESEL Pathlength/Reference Attenuation" in md
    assert "diagnostic-only" in md.lower()
    assert "not promoted" in md.lower()
    assert "no gate" in md.lower()
    assert "no `nirs4all/` integration" in md
    assert "R3d remains the accepted baseline" in md
    assert "guard_clip_fraction" in md
    assert "R9e minus references" in md
    assert "predeclared_generic_blank_reference_pathlength_attenuation_prior" in md
    assert "B2 PASS" not in md.upper().replace(":", "")


def test_run_audit_smoke_with_local_real_data_if_available() -> None:
    exp15 = _load_exp15()
    root = _repo_root()
    real_datasets, _ = exp15.discover_local_real_datasets(root)
    sentinel_candidates = exp15._exp09._select_sentinel_datasets(
        real_datasets, ["DIESEL"]
    )
    if not sentinel_candidates:
        pytest.skip("no local DIESEL sentinel real data available")

    result = exp15.run_audit(
        root=root,
        max_sentinel_datasets=1,
        seeds=(20260501,),
        n_synthetic_samples=8,
        max_real_samples=8,
        profiles=(
            "r3d_diesel_matrix_v1",
            "r4c_diesel_balanced_derivative_v1",
            "r9d_diesel_energy_normalized_support_redistribution_v1",
            "r9e_diesel_pathlength_reference_attenuation_v1",
        ),
    )

    assert result["status"] == "done"
    assert result["real_selected_count"] == 1
    assert result["audited_profiles"] == [
        "r3d_diesel_matrix_v1",
        "r4c_diesel_balanced_derivative_v1",
        "r9d_diesel_energy_normalized_support_redistribution_v1",
        "r9e_diesel_pathlength_reference_attenuation_v1",
    ]
    assert len(result["rows"]) == 4
    r9e_rows = [
        row
        for row in result["rows"]
        if row.remediation_profile
        == "r9e_diesel_pathlength_reference_attenuation_v1"
    ]
    assert len(r9e_rows) == 1
    assert r9e_rows[0].guard_clip_fraction == pytest.approx(0.0)
    assert r9e_rows[0].audit_calibration is False
    assert r9e_rows[0].audit_real_stat_capture is False
