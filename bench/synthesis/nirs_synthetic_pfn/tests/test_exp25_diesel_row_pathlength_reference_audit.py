"""Tests for the P2-02 DIESEL row-level pathlength/reference audit."""

from __future__ import annotations

import importlib.util
import sys
from csv import DictReader
from pathlib import Path
from types import ModuleType
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


def _load_exp25() -> ModuleType:
    return _load_module(
        "exp25_diesel_row_pathlength_reference_audit",
        "exp25_diesel_row_pathlength_reference_audit.py",
    )


def _dataset(exp25: ModuleType, name: str) -> Any:
    return exp25._exp09.RealDataset(
        source="AOM_regression",
        task="regression",
        database_name=name.split("_", 1)[0],
        dataset=name,
        train_path="",
        test_path="",
        ytrain_path="",
        ytest_path="",
        n_train_declared=None,
        n_test_declared=None,
        p_declared=None,
    )


def test_exp25_constants_and_contract() -> None:
    exp25 = _load_exp25()

    assert exp25.EXP25_AUDITED_PROFILES == (
        "r3d_diesel_matrix_v1",
        "r9e_diesel_pathlength_reference_attenuation_v1",
        "r9j_diesel_residual_damping_isolation_v1",
        "r9l_diesel_residual_damping_clean_attenuation_v1",
        "r9m_diesel_width_gain_damping_clean_attenuation_v1",
        "r4b_diesel_derivative_restore_v1",
        "r4c_diesel_balanced_derivative_v1",
        "p2a_diesel_row_pathlength_reference_v1",
    )
    assert "r9n" not in ",".join(exp25.EXP25_AUDITED_PROFILES)
    assert exp25.PROFILE_STAGE_MAP[exp25.P2A_PROFILE]["render_stage_family"] == (
        "row_level_pathlength_reference"
    )
    assert exp25.EXP25_REFERENCE_PROFILES == (
        "r3d_diesel_matrix_v1",
        "r9l_diesel_residual_damping_clean_attenuation_v1",
        "r4c_diesel_balanced_derivative_v1",
    )
    assert "report_only_failure_score" not in exp25.EXP25_PAIRED_DELTA_ATTRS
    audit_fields = exp25._audit_fields()
    for name in (
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
    ):
        assert audit_fields[name] is False
    assert audit_fields["audit_scope"] == exp25.EXP25_AUDIT_SCOPE


def test_p2a_builder_applies_full_row_factor_and_records_provenance() -> None:
    exp25 = _load_exp25()
    diesel = _dataset(exp25, "DIESEL_bp50_246_b-a")
    preset = exp25._exp09.select_synthetic_preset_for_dataset(diesel)
    seed = 20260501

    r3d = exp25._exp09._build_baseline_synthetic_run(
        dataset=diesel,
        preset=preset,
        n_samples=8,
        seed=seed,
        remediation_profile=exp25.R3D_PROFILE,
    )
    p2a = exp25._build_synthetic_run(
        dataset=diesel,
        preset=preset,
        n_samples=8,
        seed=seed,
        remediation_profile=exp25.P2A_PROFILE,
    )

    assert p2a.metadata["r2c_mechanistic_remediation"]["profile"] == exp25.P2A_PROFILE
    assert p2a.metadata["r2c_mechanistic_remediation"]["scope"] == (
        "bench_only_p2a_diesel_row_pathlength_reference_remediation"
    )
    params = p2a.metadata["r2c_mechanistic_remediation"]["transform_params"]
    assert params["row_pathlength_reference_factor_range"] == [0.97, 0.985]
    assert "support_reference_attenuation_factor_range" not in params
    assert params["row_pathlength_reference_support_only"] is False
    assert params["row_pathlength_reference_off_support_unchanged"] is False
    assert params["row_pathlength_reference_route_key"] == (
        "_p2a_diesel_row_pathlength_reference_route"
    )
    assert params["diesel_row_pathlength_reference_route_marker"] == "diesel"
    assert params["diesel_row_pathlength_reference_route_real_stat_capture"] is False
    assert params["row_pathlength_reference_calibration"] is False
    assert params["row_pathlength_reference_uses_real_stats"] is False
    assert params["row_pathlength_reference_uses_pca"] is False
    assert params["row_pathlength_reference_uses_ml"] is False
    assert params["row_pathlength_reference_uses_dl"] is False
    assert params["row_pathlength_reference_uses_labels"] is False
    assert params["row_pathlength_reference_uses_targets"] is False
    assert params["row_pathlength_reference_uses_splits"] is False
    assert params["row_pathlength_reference_mutates_thresholds"] is False
    assert params["row_pathlength_reference_mutates_metrics"] is False

    assert not np.array_equal(p2a.X, r3d.X)
    ratio = np.divide(p2a.X, r3d.X, out=np.ones_like(p2a.X), where=r3d.X != 0.0)
    nonzero_ratio = ratio[r3d.X != 0.0]
    assert float(nonzero_ratio.min()) >= 0.97 - 1e-12
    assert float(nonzero_ratio.max()) <= 0.985 + 1e-12
    for row, base_row in zip(ratio, r3d.X, strict=True):
        row_nonzero = row[base_row != 0.0]
        if row_nonzero.size:
            assert np.max(row_nonzero) - np.min(row_nonzero) < 1e-12


def test_p2a_non_diesel_falls_back_to_r3d_byte_identical() -> None:
    exp25 = _load_exp25()
    beer = _dataset(exp25, "BEER_OriginalExtract_60_KS")
    preset = exp25._exp09.select_synthetic_preset_for_dataset(beer)
    seed = 20260501
    effective = exp25._effective_profile_for_dataset(beer, exp25.P2A_PROFILE)

    expected = exp25._exp09._build_baseline_synthetic_run(
        dataset=beer,
        preset=preset,
        n_samples=8,
        seed=seed,
        remediation_profile=effective,
    )
    p2a = exp25._build_synthetic_run(
        dataset=beer,
        preset=preset,
        n_samples=8,
        seed=seed,
        remediation_profile=exp25.P2A_PROFILE,
    )

    assert p2a.metadata["r2c_mechanistic_remediation"] == expected.metadata[
        "r2c_mechanistic_remediation"
    ]
    assert np.array_equal(p2a.X, expected.X)
    assert np.array_equal(p2a.y, expected.y)


def _make_row(
    exp25: ModuleType,
    *,
    profile: str,
    morphology_gap_score: float,
    derivative: float,
    metadata: dict[str, Any] | None = None,
) -> Any:
    metric_fields = {
        "support_low_nm": exp25.SUPPORT_LOW_NM,
        "support_high_nm": exp25.SUPPORT_HIGH_NM,
        "support_count": 6,
        "off_support_count": 0,
        "support_weight": 1.0,
        "off_support_weight": 0.0,
        "support_mean_delta": 0.005,
        "off_support_mean_delta": 0.0,
        "support_weighted_delta": 0.005,
        "off_support_weighted_delta": 0.0,
        "global_mean_delta": 0.005,
        "decomposition_residual": 0.0,
        "real_global_mean": 0.003,
        "synthetic_global_mean": 0.008,
        "real_global_std": 0.014,
        "synthetic_global_std": 0.015,
        "log10_global_std_ratio": 0.03,
        "log10_amplitude_p50_ratio": 0.0,
        "log10_derivative_std_p50_ratio": derivative,
        "mean_curve_corr": 0.035,
        "morphology_gap_score": morphology_gap_score,
        "dominant_morphology_gap": "mean_shift",
    }
    return exp25.Exp25Row(
        status="compared",
        seed=20260501,
        remediation_profile=profile,
        effective_remediation_profile=profile,
        source="AOM_regression",
        task="regression",
        dataset="DIESEL/DIESEL_bp50_246_b-a",
        synthetic_preset="fuel",
        effective_matrix_route="diesel_fuel_matrix",
        comparison_space=exp25.COMPARISON_SPACE,
        **exp25._row_stage_and_failure(
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
        guard_clip_fraction=0.0 if profile == exp25.P2A_PROFILE else None,
        **exp25._audit_fields(),
        blocked_reason="",
    )


def test_write_csv_and_render_markdown_contract(tmp_path: Path) -> None:
    exp25 = _load_exp25()
    metadata = {
        "r2c_mechanistic_remediation": {
            "transform_params": {
                "row_pathlength_reference_factor_range": [0.970, 0.985],
                "row_pathlength_reference_n_wavelengths": 6,
                "row_pathlength_reference_application_stage": (
                    "after_r3d_output_clip_before_audit_alignment"
                ),
                "row_pathlength_reference_applies_to": "full_generated_wavelength_row",
                "row_pathlength_reference_guard_clip_fraction": 0.0,
            }
        }
    }
    rows = [
        _make_row(exp25, profile=exp25.R3D_PROFILE, morphology_gap_score=1.6, derivative=-0.03),
        _make_row(exp25, profile=exp25.R9L_PROFILE, morphology_gap_score=1.5, derivative=-0.05),
        _make_row(
            exp25,
            profile=exp25.P2A_PROFILE,
            morphology_gap_score=1.55,
            derivative=-0.04,
            metadata=metadata,
        ),
    ]
    csv_path = tmp_path / "exp25.csv"
    report_path = tmp_path / "exp25.md"

    exp25.write_csv(rows, csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        records = list(reader)

    assert len(records) == 3
    assert reader.fieldnames is not None
    assert "delta_vs_r3d_diesel_matrix_v1__morphology_gap_score" in reader.fieldnames
    p2a_record = next(row for row in records if row["remediation_profile"] == exp25.P2A_PROFILE)
    assert p2a_record["render_stage_family"] == "row_level_pathlength_reference"
    assert p2a_record["stage_support_nm"] == "full_generated_row"
    assert float(p2a_record["delta_vs_r3d_diesel_matrix_v1__morphology_gap_score"]) == (
        pytest.approx(-0.05)
    )

    md = exp25.render_markdown(
        result={
            "status": "done",
            "rows": rows,
            "real_runnable_count": 1,
            "real_sentinel_candidate_count": 1,
            "real_selected_count": 1,
            "sentinel_tokens": ["DIESEL"],
            "seeds": [20260501],
            "audited_profiles": [exp25.R3D_PROFILE, exp25.R9L_PROFILE, exp25.P2A_PROFILE],
            "support_low_nm": exp25.SUPPORT_LOW_NM,
            "support_high_nm": exp25.SUPPORT_HIGH_NM,
        },
        report_path=report_path,
        csv_path=csv_path,
        n_synthetic_samples=8,
        max_real_samples=8,
        max_sentinel_datasets=1,
        seeds=[20260501],
        sentinel_tokens=["DIESEL"],
        profiles=[exp25.R3D_PROFILE, exp25.R9L_PROFILE, exp25.P2A_PROFILE],
    )

    assert "# P2-02 DIESEL Row-Level Pathlength/Reference Audit" in md
    assert "No R9n" in md
    assert "R3d remains the accepted DIESEL baseline" in md
    assert "support_only: `False`" in md
    assert "not promoted" in md
    assert "B2 PASS" not in md.upper().replace(":", "")
