"""Tests for the P2-05 DIESEL readout/geometry audit."""

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


def _load_exp27() -> ModuleType:
    return _load_module(
        "exp27_diesel_readout_geometry_audit",
        "exp27_diesel_readout_geometry_audit.py",
    )


def _dataset(exp27: ModuleType) -> Any:
    return exp27._exp25._exp09.RealDataset(
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


def _metadata() -> dict[str, Any]:
    return {
        "instrument": {"key": "foss_xds"},
        "mode": "reflectance",
        "builder_config": {
            "features": {
                "instrument": "foss_xds",
                "measurement_mode": "reflectance",
            }
        },
        "r2c_mechanistic_remediation": {
            "transform_params": {
                "readout_space": "blank_referenced_micro_path_ch_overtone_raw_absorbance"
            }
        },
    }


def test_exp27_contract_and_readout_transforms() -> None:
    exp27 = _load_exp27()
    x = np.asarray([[0.0, 1.0, 2.0]])

    assert exp27.SOURCE_PROFILES == (
        "r3d_diesel_matrix_v1",
        "p2a_diesel_row_pathlength_reference_v1",
    )
    assert "r9n" not in f"{exp27.EXP27_AUDIT_SCOPE},{exp27.EXP27_DECISION}"
    assert exp27._anti_leakage_flags_false() is True
    assert np.allclose(exp27.apply_readout_transform(x, exp27.READOUT_IDENTITY), x)
    assert np.allclose(
        exp27.apply_readout_transform(x, exp27.READOUT_TRANSMITTANCE),
        [[1.0, 0.1, 0.01]],
    )
    assert np.allclose(
        exp27.apply_readout_transform(x, exp27.READOUT_BLANK_INTENSITY),
        [[0.0, 0.9, 0.99]],
    )
    with pytest.raises(ValueError, match="unknown readout transform"):
        exp27.apply_readout_transform(x, "retuned_scalar")


def test_row_from_arrays_reports_geometry_block_and_identity_delta() -> None:
    exp27 = _load_exp27()
    dataset = _dataset(exp27)
    real_x = np.asarray([[0.01, 0.02, 0.03], [0.015, 0.025, 0.035]])
    synthetic_x = np.asarray([[0.011, 0.021, 0.031], [0.016, 0.026, 0.036]])
    wavelengths = np.asarray([900.0, 1100.0, 1500.0])
    identity_metrics = exp27._metric_fields(real_x, synthetic_x, wavelengths)

    row = exp27._row_from_arrays(
        seed=20260501,
        source_profile=exp27.R3D_PROFILE,
        readout_transform=exp27.READOUT_IDENTITY,
        dataset=dataset,
        preset="fuel",
        real_x=real_x,
        synthetic_x=synthetic_x,
        wavelengths=wavelengths,
        metadata=_metadata(),
        identity_metrics=identity_metrics,
    )

    assert row.status == "compared"
    assert row.support_count == 3
    assert row.off_support_count == 0
    assert row.geometry_metadata_present is False
    assert row.geometry_audit_status == "blocked_no_source_detector_geometry_metadata"
    assert row.instrument_key == "foss_xds"
    assert row.measurement_mode == "reflectance"
    assert row.metadata_readout_space == "blank_referenced_micro_path_ch_overtone_raw_absorbance"
    assert row.transform_parameters_source == "predeclared_beer_lambert_readout_maps_no_fit"
    assert row.readout_delta_vs_identity__global_mean_delta == pytest.approx(0.0)
    assert row.readout_delta_vs_identity__morphology_gap_score == pytest.approx(0.0)
    assert row.audit_scope == exp27.EXP27_AUDIT_SCOPE
    assert row.audit_calibration is False


def test_write_csv_and_render_markdown_contract(tmp_path: Path) -> None:
    exp27 = _load_exp27()
    dataset = _dataset(exp27)
    real_x = np.asarray([[0.01, 0.02, 0.03]])
    synthetic_x = np.asarray([[0.011, 0.021, 0.031]])
    wavelengths = np.asarray([900.0, 1100.0, 1500.0])
    identity_metrics = exp27._metric_fields(real_x, synthetic_x, wavelengths)
    rows = [
        exp27._row_from_arrays(
            seed=20260501,
            source_profile=exp27.R3D_PROFILE,
            readout_transform=transform,
            dataset=dataset,
            preset="fuel",
            real_x=real_x,
            synthetic_x=synthetic_x,
            wavelengths=wavelengths,
            metadata=_metadata(),
            identity_metrics=identity_metrics,
        )
        for transform in exp27.READOUT_TRANSFORMS
    ]
    csv_path = tmp_path / "exp27.csv"
    report_path = tmp_path / "exp27.md"

    exp27.write_csv(rows, csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        records = list(reader)

    assert len(records) == 3
    assert reader.fieldnames is not None
    assert "geometry_audit_status" in reader.fieldnames
    assert records[0]["readout_transform"] == exp27.READOUT_IDENTITY

    md = exp27.render_markdown(
        result={
            "status": "done",
            "rows": rows,
            "real_runnable_count": 1,
            "real_sentinel_candidate_count": 1,
            "real_selected_count": 1,
            "sentinel_tokens": ["DIESEL"],
            "seeds": [20260501],
        },
        report_path=report_path,
        csv_path=csv_path,
        n_synthetic_samples=1,
        max_real_samples=1,
        max_sentinel_datasets=1,
        seeds=[20260501],
        sentinel_tokens=["DIESEL"],
    )

    assert "# P2-05 DIESEL Readout/Geometry Mechanistic Audit" in md
    assert "source_detector_geometry_metadata_present: `False`" in md
    assert "No geometry scalar is tested" in md
    assert "No calibration" in md
    assert "no R9n" in md
    assert "uncalibrated raw comparison space" in md
    assert "not promotion evidence" in md
    assert "B2 PASS" not in md.upper().replace(":", "")
