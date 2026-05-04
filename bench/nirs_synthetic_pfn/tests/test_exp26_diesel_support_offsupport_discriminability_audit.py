"""Tests for the P2-03 DIESEL support/off-support discriminability audit."""

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


def _load_exp26() -> ModuleType:
    return _load_module(
        "exp26_diesel_support_offsupport_discriminability_audit",
        "exp26_diesel_support_offsupport_discriminability_audit.py",
    )


def _dataset(exp26: ModuleType) -> Any:
    return exp26._exp25._exp09.RealDataset(
        source="AOM_regression",
        task="regression",
        database_name="DIESEL",
        dataset="DIESEL_bp50_246_b-a",
        train_path="",
        test_path="",
        ytrain_path="",
        ytest_path="",
        n_train_declared=None,
        n_test_declared=None,
        p_declared=None,
    )


def _metadata() -> tuple[dict[str, Any], dict[str, Any]]:
    r9e_metadata = {
        "r2c_mechanistic_remediation": {
            "transform_params": {
                "support_reference_attenuation_factor_min": 0.98,
                "support_reference_attenuation_factor_max": 0.98,
                "support_reference_attenuation_support_nm": [750.0, 1550.0],
                "support_reference_attenuation_off_support_unchanged": True,
                "support_reference_attenuation_only": True,
            }
        }
    }
    p2a_metadata = {
        "r2c_mechanistic_remediation": {
            "transform_params": {
                "row_pathlength_reference_factor_min": 0.98,
                "row_pathlength_reference_factor_max": 0.98,
                "row_pathlength_reference_applies_to": "full_generated_wavelength_row",
                "row_pathlength_reference_off_support_unchanged": False,
                "row_pathlength_reference_support_only": False,
            }
        }
    }
    return r9e_metadata, p2a_metadata


def test_exp26_constants_and_contract() -> None:
    exp26 = _load_exp26()

    assert exp26.EXP26_CASES == (
        "real_aligned_current_cohort",
        "generated_prior_full_grid_counterfactual",
    )
    assert exp26.R9E_PROFILE == "r9e_diesel_pathlength_reference_attenuation_v1"
    assert exp26.P2A_PROFILE == "p2a_diesel_row_pathlength_reference_v1"
    assert "r9n" not in f"{exp26.R9E_PROFILE},{exp26.P2A_PROFILE},{exp26.EXP26_AUDIT_SCOPE}"
    audit_fields = exp26._audit_fields()
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
    assert audit_fields["audit_scope"] == exp26.EXP26_AUDIT_SCOPE
    assert exp26._anti_leakage_flags_false() is True


def test_current_support_only_grid_is_indistinguishable() -> None:
    exp26 = _load_exp26()
    r9e_metadata, p2a_metadata = _metadata()
    wavelengths = np.asarray([900.0, 1100.0, 1500.0])
    r3d = np.asarray([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    r9e = r3d * 0.98
    p2a = r3d * 0.98

    row = exp26._row_from_arrays(
        case=exp26.REAL_ALIGNED_CASE,
        seed=20260501,
        dataset=_dataset(exp26),
        preset="fuel",
        grid_source="unit",
        grid_note="unit",
        wavelengths=wavelengths,
        r3d_x=r3d,
        r9e_x=r9e,
        p2a_x=p2a,
        r9e_metadata=r9e_metadata,
        p2a_metadata=p2a_metadata,
        n_real_samples=2,
    )

    assert row.support_count == 3
    assert row.off_support_count == 0
    assert row.r9e_vs_p2a_max_abs_delta_inside == pytest.approx(0.0)
    assert row.r9e_vs_p2a_max_abs_delta_outside is None
    assert row.distinguishable_by_off_support is False
    assert row.current_cohort_can_distinguish is False
    assert row.r9e_off_support_unchanged_metadata is True
    assert row.p2a_off_support_unchanged_metadata is False
    assert row.anti_leakage_flags_false is True


def test_generated_grid_with_off_support_is_distinguishable() -> None:
    exp26 = _load_exp26()
    r9e_metadata, p2a_metadata = _metadata()
    wavelengths = np.asarray([700.0, 900.0, 1100.0, 1600.0])
    r3d = np.asarray([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]])
    support = (wavelengths >= exp26.SUPPORT_LOW_NM) & (wavelengths <= exp26.SUPPORT_HIGH_NM)
    r9e = r3d.copy()
    r9e[:, support] *= 0.98
    p2a = r3d * 0.98

    row = exp26._row_from_arrays(
        case=exp26.GENERATED_FULL_GRID_CASE,
        seed=20260501,
        dataset=_dataset(exp26),
        preset="fuel",
        grid_source="unit",
        grid_note="unit",
        wavelengths=wavelengths,
        r3d_x=r3d,
        r9e_x=r9e,
        p2a_x=p2a,
        r9e_metadata=r9e_metadata,
        p2a_metadata=p2a_metadata,
        n_real_samples=0,
    )

    assert row.support_count == 2
    assert row.off_support_count == 2
    assert row.r9e_vs_r3d_max_abs_delta_outside == pytest.approx(0.0)
    assert row.p2a_vs_r3d_max_abs_delta_outside == pytest.approx(0.08)
    assert row.r9e_vs_p2a_max_abs_delta_inside == pytest.approx(0.0)
    assert row.r9e_vs_p2a_max_abs_delta_outside == pytest.approx(0.08)
    assert row.r9e_vs_p2a_outside_minus_inside == pytest.approx(0.08)
    assert row.r9e_vs_p2a_outside_to_inside_ratio == float("inf")
    assert row.p2a_minus_r9e_delta_outside_vs_r3d == pytest.approx(0.08)
    assert row.distinguishable_by_off_support is True
    assert row.current_cohort_can_distinguish is False


def test_write_csv_and_render_markdown_contract(tmp_path: Path) -> None:
    exp26 = _load_exp26()
    r9e_metadata, p2a_metadata = _metadata()
    dataset = _dataset(exp26)
    support_wl = np.asarray([900.0, 1100.0, 1500.0])
    support_r3d = np.asarray([[1.0, 2.0, 3.0]])
    support_row = exp26._row_from_arrays(
        case=exp26.REAL_ALIGNED_CASE,
        seed=20260501,
        dataset=dataset,
        preset="fuel",
        grid_source="unit",
        grid_note="unit",
        wavelengths=support_wl,
        r3d_x=support_r3d,
        r9e_x=support_r3d * 0.98,
        p2a_x=support_r3d * 0.98,
        r9e_metadata=r9e_metadata,
        p2a_metadata=p2a_metadata,
        n_real_samples=1,
    )
    full_wl = np.asarray([700.0, 900.0, 1600.0])
    full_r3d = np.asarray([[1.0, 2.0, 4.0]])
    full_r9e = full_r3d.copy()
    full_r9e[:, [1]] *= 0.98
    full_row = exp26._row_from_arrays(
        case=exp26.GENERATED_FULL_GRID_CASE,
        seed=20260501,
        dataset=dataset,
        preset="fuel",
        grid_source="unit",
        grid_note="unit",
        wavelengths=full_wl,
        r3d_x=full_r3d,
        r9e_x=full_r9e,
        p2a_x=full_r3d * 0.98,
        r9e_metadata=r9e_metadata,
        p2a_metadata=p2a_metadata,
        n_real_samples=0,
    )
    rows = [support_row, full_row]
    csv_path = tmp_path / "exp26.csv"
    report_path = tmp_path / "exp26.md"

    exp26.write_csv(rows, csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        records = list(reader)

    assert len(records) == 2
    assert reader.fieldnames is not None
    assert "r9e_vs_p2a_max_abs_delta_outside" in reader.fieldnames
    real_record = next(row for row in records if row["case"] == exp26.REAL_ALIGNED_CASE)
    assert real_record["off_support_count"] == "0"
    assert real_record["current_cohort_can_distinguish"] == "False"
    full_record = next(row for row in records if row["case"] == exp26.GENERATED_FULL_GRID_CASE)
    assert full_record["distinguishable_by_off_support"] == "True"

    md = exp26.render_markdown(
        result={
            "status": "done",
            "rows": rows,
            "real_runnable_count": 1,
            "real_sentinel_candidate_count": 1,
            "real_selected_count": 1,
            "sentinel_tokens": ["DIESEL"],
            "seeds": [20260501],
            "support_low_nm": exp26.SUPPORT_LOW_NM,
            "support_high_nm": exp26.SUPPORT_HIGH_NM,
        },
        report_path=report_path,
        csv_path=csv_path,
        n_synthetic_samples=1,
        max_real_samples=1,
        max_sentinel_datasets=1,
        seeds=[20260501],
        sentinel_tokens=["DIESEL"],
    )

    assert "# P2-03 DIESEL Support/Off-Support Discriminability Audit" in md
    assert "current_real_aligned_cohort_can_distinguish: `False`" in md
    assert "generated_full_grid_can_distinguish: `True`" in md
    assert "No calibration" in md
    assert "R9n" in md
    assert "B2 PASS" not in md.upper().replace(":", "")
