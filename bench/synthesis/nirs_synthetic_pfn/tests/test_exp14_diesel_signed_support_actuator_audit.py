"""Tests for the R9e0 DIESEL signed support actuator probe audit."""

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


def _load_exp14() -> ModuleType:
    return _load_module(
        "exp14_diesel_signed_support_actuator_audit",
        "exp14_diesel_signed_support_actuator_audit.py",
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _make_row(
    exp14: ModuleType,
    *,
    profile: str,
    profile_kind: str,
    seed: int = 20260501,
    global_mean_delta: float = 0.002,
    support_mean_delta: float = 0.002,
    support_weighted_delta: float = 0.002,
    off_support_weighted_delta: float = 0.0,
    morphology_gap_score: float = 1.0,
    mean_curve_corr: float = 0.5,
    guard_clip_fraction: float | None = None,
) -> Any:
    is_probe = profile_kind == "probe"
    return exp14.R9e0Row(
        status="compared",
        seed=seed,
        remediation_profile=profile,
        profile_kind=profile_kind,
        profile_registered=not is_probe,
        probe_only=is_probe,
        base_profile=exp14.R9E0_BASE_PROFILE if is_probe else None,
        effective_remediation_profile=profile if not is_probe else None,
        source="AOM_regression",
        task="regression",
        dataset="DIESEL_bp50_246_b-a/ds",
        synthetic_preset="fuel",
        effective_matrix_route="diesel_fuel_matrix",
        comparison_space=exp14.COMPARISON_SPACE,
        n_real_samples=8,
        n_synthetic_samples=8,
        n_wavelengths=6,
        wavelength_min=700.0,
        wavelength_max=1600.0,
        support_low_nm=exp14.SUPPORT_LOW_NM,
        support_high_nm=exp14.SUPPORT_HIGH_NM,
        support_count=4,
        off_support_count=2,
        support_weight=4.0 / 6.0,
        off_support_weight=2.0 / 6.0,
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
        log10_derivative_std_p50_ratio=0.04,
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
        audit_scope=exp14.R9E0_AUDIT_SCOPE,
        blocked_reason="",
    )


def test_probe_definitions_and_constants_are_exact() -> None:
    exp14 = _load_exp14()

    assert exp14.R9E0_AUDITED_PROFILES == (
        "r3d_diesel_matrix_v1",
        "r4a_diesel_basis_v1",
        "r4b_diesel_derivative_restore_v1",
        "r4c_diesel_balanced_derivative_v1",
        "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1",
        "r9b_diesel_support_intercept_v1",
        "r9c_diesel_selective_ch_bandwidth_damping_v1",
        "r9d_diesel_energy_normalized_support_redistribution_v1",
    )
    assert exp14.R9E0_BASE_PROFILE == "r3d_diesel_matrix_v1"
    assert exp14.R9E0_PAIRED_REFERENCE_PROFILES == (
        "r3d_diesel_matrix_v1",
        "r4c_diesel_balanced_derivative_v1",
    )
    assert exp14.R9D_SHAPE_CENTERS_NM == (1150.0, 1210.0, 1390.0, 1460.0)
    assert exp14.R9D_SHAPE_WIDTHS_NM == (40.0, 40.0, 44.0, 48.0)
    assert exp14.SUPPORT_LOW_NM == 750.0
    assert exp14.SUPPORT_HIGH_NM == 1550.0
    assert exp14.COMPARISON_SPACE == "uncalibrated_raw"
    assert exp14.R9E0_CONSTANTS_SOURCE == (
        "predeclared_generic_blank_reference_pathlength_and_"
        "liquid_hydrocarbon_actuator_prior"
    )

    assert [probe.name for probe in exp14.R9E0_PROBES] == [
        "r9e0_negative_blank_intercept_0p0010",
        "r9e0_negative_blank_intercept_0p0020",
        "r9e0_multiplicative_attenuation_0p985",
        "r9e0_multiplicative_attenuation_0p970",
        "r9e0_negative_intercept_0p0010_plus_r9d_shape_0p035",
    ]
    assert [
        (
            probe.intercept,
            probe.multiplicative,
            probe.apply_r9d_shape,
            probe.r9d_shape_strength,
        )
        for probe in exp14.R9E0_PROBES
    ] == [
        (0.0010, 1.0, False, 0.0),
        (0.0020, 1.0, False, 0.0),
        (0.0, 0.985, False, 0.0),
        (0.0, 0.970, False, 0.0),
        (0.0010, 1.0, True, 0.035),
    ]
    assert tuple(probe.name for probe in exp14.R9E0_PROBES) == exp14.R9E0_PROBE_NAMES
    assert set(exp14.R9E0_PROBES_BY_NAME) == set(exp14.R9E0_PROBE_NAMES)


def test_is_probe_only_true_for_predeclared_probes() -> None:
    exp14 = _load_exp14()

    for probe_name in exp14.R9E0_PROBE_NAMES:
        assert exp14.is_probe(probe_name) is True
    for profile in exp14.R9E0_AUDITED_PROFILES:
        assert exp14.is_probe(profile) is False


def test_r9e0_is_not_a_builder_registration() -> None:
    exp14 = _load_exp14()
    from nirsyntheticpfn.adapters import builder_adapter

    assert all("r9e0" not in profile.lower() for profile in exp14.R9E0_AUDITED_PROFILES)
    assert all(
        probe_name not in builder_adapter.ALL_REMEDIATION_PROFILES
        for probe_name in exp14.R9E0_PROBE_NAMES
    )
    assert not any(
        "r9e0" in profile.lower()
        for profile in builder_adapter.ALL_REMEDIATION_PROFILES
    )


def test_apply_probe_keeps_off_support_byte_identical() -> None:
    exp14 = _load_exp14()
    wavelengths = np.asarray([700.0, 800.0, 1200.0, 1600.0])
    base = np.asarray(
        [
            [0.0100, 0.0020, 0.0040, 0.0200],
            [0.0110, 0.0030, 0.0050, 0.0210],
        ],
        dtype=float,
    )
    spec = exp14.R9E0_PROBES_BY_NAME[
        "r9e0_negative_intercept_0p0010_plus_r9d_shape_0p035"
    ]

    probed, _ = exp14.apply_probe(base, wavelengths, spec=spec)
    support = (wavelengths >= 750.0) & (wavelengths <= 1550.0)

    assert np.array_equal(probed[:, ~support], base[:, ~support])
    assert np.any(probed[:, support] != base[:, support])


def test_apply_probe_preserves_off_support_dtype_and_bytes() -> None:
    exp14 = _load_exp14()
    wavelengths = np.asarray([700.0, 800.0, 1200.0, 1600.0])
    base = np.asarray(
        [
            [0.0100, 0.0020, 0.0040, 0.0200],
            [0.0110, 0.0030, 0.0050, 0.0210],
        ],
        dtype=np.float32,
    )
    spec = exp14.R9E0_PROBES_BY_NAME["r9e0_multiplicative_attenuation_0p970"]
    support = (wavelengths >= 750.0) & (wavelengths <= 1550.0)

    probed, _ = exp14.apply_probe(base, wavelengths, spec=spec)

    assert probed.dtype == base.dtype
    assert probed[:, ~support].tobytes() == base[:, ~support].tobytes()
    assert not np.array_equal(probed[:, support], base[:, support])


def test_apply_probe_reports_clip_fraction_for_negative_intercept() -> None:
    exp14 = _load_exp14()
    wavelengths = np.asarray([800.0, 1200.0])
    base = np.asarray(
        [
            [0.0005, 0.0020],
            [0.0001, 0.0010],
        ],
        dtype=float,
    )
    spec = exp14.R9E0_PROBES_BY_NAME["r9e0_negative_blank_intercept_0p0010"]

    probed, clip_fraction = exp14.apply_probe(base, wavelengths, spec=spec)

    assert clip_fraction == pytest.approx(2.0 / 4.0)
    np.testing.assert_allclose(
        probed,
        np.asarray([[0.0, 0.0010], [0.0, 0.0]], dtype=float),
        atol=0.0,
        rtol=0.0,
    )


def test_combo_probe_preserves_post_intercept_support_mean_after_shape() -> None:
    exp14 = _load_exp14()
    wavelengths = np.asarray([700.0, 800.0, 1150.0, 1210.0, 1390.0, 1460.0, 1600.0])
    base = np.asarray(
        [
            [0.040, 0.006, 0.008, 0.010, 0.012, 0.014, 0.050],
            [0.041, 0.020, 0.018, 0.016, 0.014, 0.012, 0.051],
        ],
        dtype=float,
    )
    support = (wavelengths >= 750.0) & (wavelengths <= 1550.0)
    spec = exp14.R9E0_PROBES_BY_NAME[
        "r9e0_negative_intercept_0p0010_plus_r9d_shape_0p035"
    ]
    post_intercept = np.maximum(base[:, support] - 0.0010, 0.0)

    probed, clip_fraction = exp14.apply_probe(base, wavelengths, spec=spec)

    assert clip_fraction == pytest.approx(0.0)
    np.testing.assert_allclose(
        probed[:, support].mean(axis=1),
        post_intercept.mean(axis=1),
        atol=1e-14,
        rtol=0.0,
    )
    assert np.array_equal(probed[:, ~support], base[:, ~support])


def test_validate_targets_rejects_invalid_names() -> None:
    exp14 = _load_exp14()

    with pytest.raises(ValueError, match="at least one"):
        exp14._validate_targets((), ())
    with pytest.raises(ValueError, match="unknown R9e0 builder profiles"):
        exp14._validate_targets(("r9e0_not_a_builder_profile",), ())
    with pytest.raises(ValueError, match="unknown R9e0 probes"):
        exp14._validate_targets((), ("r3d_diesel_matrix_v1",))


def test_write_csv_and_render_markdown_contract_with_synthetic_rows(
    tmp_path: Path,
) -> None:
    exp14 = _load_exp14()
    rows = [
        _make_row(
            exp14,
            profile="r3d_diesel_matrix_v1",
            profile_kind="builder",
            global_mean_delta=0.004,
            support_mean_delta=0.004,
            support_weighted_delta=0.004,
            morphology_gap_score=1.40,
            mean_curve_corr=0.20,
        ),
        _make_row(
            exp14,
            profile="r4c_diesel_balanced_derivative_v1",
            profile_kind="builder",
            global_mean_delta=0.003,
            support_mean_delta=0.003,
            support_weighted_delta=0.003,
            morphology_gap_score=1.30,
            mean_curve_corr=0.25,
        ),
        _make_row(
            exp14,
            profile="r9e0_negative_blank_intercept_0p0010",
            profile_kind="probe",
            global_mean_delta=0.002,
            support_mean_delta=0.002,
            support_weighted_delta=0.002,
            morphology_gap_score=1.10,
            mean_curve_corr=0.35,
            guard_clip_fraction=0.125,
        ),
    ]
    csv_path = tmp_path / "r9e0.csv"
    report_path = tmp_path / "r9e0.md"

    exp14.write_csv(rows, csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        records = list(reader)

    assert len(records) == 3
    assert reader.fieldnames == exp14._csv_fieldnames()
    assert "profile_kind" in reader.fieldnames
    assert "profile_registered" in reader.fieldnames
    assert "probe_only" in reader.fieldnames
    assert (
        "delta_probe_minus_r3d_diesel_matrix_v1__global_mean_delta"
        in reader.fieldnames
    )
    probe_record = records[2]
    assert probe_record["profile_kind"] == "probe"
    assert probe_record["profile_registered"] == "False"
    assert probe_record["probe_only"] == "True"
    assert float(
        probe_record["delta_probe_minus_r3d_diesel_matrix_v1__global_mean_delta"]
    ) == pytest.approx(-0.002)
    assert float(
        probe_record[
            "delta_probe_minus_r4c_diesel_balanced_derivative_v1__morphology_gap_score"
        ]
    ) == pytest.approx(-0.20)

    result = {
        "status": "done",
        "rows": rows,
        "real_runnable_count": 1,
        "real_sentinel_candidate_count": 1,
        "real_selected_count": 1,
        "sentinel_tokens": ["DIESEL"],
        "seeds": [20260501],
        "audited_profiles": ["r3d_diesel_matrix_v1", "r4c_diesel_balanced_derivative_v1"],
        "audited_probes": ["r9e0_negative_blank_intercept_0p0010"],
        "support_low_nm": exp14.SUPPORT_LOW_NM,
        "support_high_nm": exp14.SUPPORT_HIGH_NM,
    }
    md = exp14.render_markdown(
        result=result,
        report_path=report_path,
        csv_path=csv_path,
        n_synthetic_samples=8,
        max_real_samples=8,
        max_sentinel_datasets=1,
        seeds=[20260501],
        sentinel_tokens=["DIESEL"],
        profiles=["r3d_diesel_matrix_v1", "r4c_diesel_balanced_derivative_v1"],
        probes=["r9e0_negative_blank_intercept_0p0010"],
    )

    assert "# R9e0 DIESEL Signed Support Actuator Diagnostic Audit" in md
    assert "probe-only" in md.lower()
    assert "diagnostic-only" in md.lower()
    assert "no probe is registered as a builder profile" in md.lower()
    assert "uncalibrated_raw" in md
    assert "GO/NO-GO" in md
    assert "r9e0_negative_blank_intercept_0p0010" in md
    assert "`-0.0010`" in md
    assert "`-0.0020`" in md
    assert "`X = 0.985 * X`" in md
    assert "`X = 0.970 * X`" in md
    assert "strength 0.035" in md
    assert "delta" in md.lower()
    assert "B2 PASS" not in md.upper().replace(":", "")


def test_run_audit_smoke_with_local_real_data_if_available() -> None:
    exp14 = _load_exp14()
    root = _repo_root()
    real_datasets, _ = exp14.discover_local_real_datasets(root)
    sentinel_candidates = exp14._exp09._select_sentinel_datasets(
        real_datasets, ["DIESEL"]
    )
    if not sentinel_candidates:
        pytest.skip("no local DIESEL sentinel real data available")

    result = exp14.run_audit(
        root=root,
        max_sentinel_datasets=1,
        seeds=(20260501,),
        n_synthetic_samples=8,
        max_real_samples=8,
        profiles=("r3d_diesel_matrix_v1", "r4c_diesel_balanced_derivative_v1"),
        probes=("r9e0_negative_blank_intercept_0p0010",),
    )

    assert result["real_selected_count"] == 1
    assert result["audited_profiles"] == [
        "r3d_diesel_matrix_v1",
        "r4c_diesel_balanced_derivative_v1",
    ]
    assert result["audited_probes"] == ["r9e0_negative_blank_intercept_0p0010"]
    assert len(result["rows"]) == 3
    assert {row.profile_kind for row in result["rows"]} <= {"builder", "probe"}
    assert any(
        row.remediation_profile == "r9e0_negative_blank_intercept_0p0010"
        and row.profile_kind == "probe"
        and row.profile_registered is False
        and row.probe_only is True
        for row in result["rows"]
    )
