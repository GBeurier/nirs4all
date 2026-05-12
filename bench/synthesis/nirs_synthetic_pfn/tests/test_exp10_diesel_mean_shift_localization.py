"""Tests for the R9a DIESEL mean-shift localization audit (exp10)."""

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


def _load_exp10() -> ModuleType:
    return _load_module(
        "exp10_diesel_mean_shift_localization",
        "exp10_diesel_mean_shift_localization.py",
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


def test_exp10_exposes_dieselonly_profiles_and_default_seeds() -> None:
    exp10 = _load_exp10()
    assert exp10.R9A_AUDITED_PROFILES == (
        "r3d_diesel_matrix_v1",
        "r4a_diesel_basis_v1",
        "r4c_diesel_balanced_derivative_v1",
        "r5a_diesel_absorbance_readout_v1",
        "r5b_diesel_transmittance_readout_v1",
        "r5c_diesel_blank_referenced_intensity_v1",
        "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1",
    )
    assert exp10.DEFAULT_SEEDS == (20260430, 20260431, 20260432)
    assert exp10.DEFAULT_N_SYNTHETIC_SAMPLES == 64
    assert exp10.DEFAULT_MAX_REAL_SAMPLES == 64
    assert exp10.SUPPORT_LOW_NM == 750.0
    assert exp10.SUPPORT_HIGH_NM == 1550.0
    assert exp10.COMPARISON_SPACE == "uncalibrated_raw"
    assert exp10.R9A_AUDIT_SCOPE == (
        "bench_only_r9a_diesel_mean_shift_localization_audit"
    )
    assert exp10.DEFAULT_REPORT.name == (
        "r9a_diesel_mean_shift_localization_audit.md"
    )
    assert exp10.DEFAULT_CSV.name == (
        "r9a_diesel_mean_shift_localization_audit.csv"
    )


# ---------------------------------------------------------------------------
# Support decomposition identity and shape behaviour.
# ---------------------------------------------------------------------------


def test_support_decomposition_identity_holds_within_tolerance() -> None:
    exp10 = _load_exp10()
    rng = np.random.default_rng(20260430)
    wl = np.linspace(400.0, 2500.0, 211)  # straddles the 750-1550 nm support
    real = rng.normal(0.5, 0.05, size=(40, wl.size))
    synth = rng.normal(0.55, 0.06, size=(40, wl.size)) + 0.02 * np.sin(
        np.linspace(0.0, 4 * np.pi, wl.size)
    )

    out = exp10.compute_support_decomposition(real, synth, wl)
    expected_global = float((synth.mean(axis=0) - real.mean(axis=0)).mean())
    assert abs(out["global_mean_delta"] - expected_global) <= 1e-12
    reconstructed = (
        out["support_weighted_delta"] + out["off_support_weighted_delta"]
    )
    assert abs(out["global_mean_delta"] - reconstructed) <= 1e-12
    assert abs(out["decomposition_residual"]) <= 1e-12
    assert out["support_count"] + out["off_support_count"] == wl.size
    assert out["support_weight"] + out["off_support_weight"] == pytest.approx(1.0)
    assert out["support_count"] > 0
    assert out["off_support_count"] > 0


def test_support_decomposition_handles_full_off_support_grid() -> None:
    exp10 = _load_exp10()
    wl = np.linspace(2000.0, 2500.0, 64)  # entirely above the support window
    real = np.full((20, wl.size), 0.1, dtype=float)
    synth = np.full((20, wl.size), 0.4, dtype=float)

    out = exp10.compute_support_decomposition(real, synth, wl)
    assert out["support_count"] == 0
    assert out["off_support_count"] == wl.size
    assert out["support_weight"] == pytest.approx(0.0)
    assert out["off_support_weight"] == pytest.approx(1.0)
    assert out["support_mean_delta"] == pytest.approx(0.0)
    assert out["off_support_mean_delta"] == pytest.approx(0.3)
    assert out["global_mean_delta"] == pytest.approx(0.3)
    assert out["off_support_weighted_delta"] == pytest.approx(0.3)
    assert out["support_weighted_delta"] == pytest.approx(0.0)
    assert abs(out["decomposition_residual"]) <= 1e-12


def test_support_decomposition_handles_full_support_grid() -> None:
    exp10 = _load_exp10()
    wl = np.linspace(800.0, 1500.0, 32)  # entirely inside the 750-1550 support
    real = np.zeros((10, wl.size), dtype=float)
    synth = np.full((10, wl.size), 0.05, dtype=float)

    out = exp10.compute_support_decomposition(real, synth, wl)
    assert out["off_support_count"] == 0
    assert out["support_count"] == wl.size
    assert out["off_support_mean_delta"] == pytest.approx(0.0)
    assert out["support_mean_delta"] == pytest.approx(0.05)
    assert out["global_mean_delta"] == pytest.approx(0.05)
    assert out["support_weighted_delta"] == pytest.approx(0.05)
    assert abs(out["decomposition_residual"]) <= 1e-12


# ---------------------------------------------------------------------------
# DIESEL profile relationships (R5a == R4c, R5b/R5c readout maps, R8b support
# mean equality with R4c).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def diesel_runs() -> dict[str, Any]:
    exp09 = _load_exp09()
    diesel = _make_dataset("DIESEL_bp50_246_b-a")
    preset = exp09.select_synthetic_preset_for_dataset(diesel)
    seeds = (20260430,)
    runs: dict[str, Any] = {}
    for profile in (
        "r3d_diesel_matrix_v1",
        "r4c_diesel_balanced_derivative_v1",
        "r5a_diesel_absorbance_readout_v1",
        "r5b_diesel_transmittance_readout_v1",
        "r5c_diesel_blank_referenced_intensity_v1",
        "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1",
    ):
        runs[profile] = exp09._build_baseline_synthetic_run(
            dataset=diesel,
            preset=preset,
            n_samples=8,
            seed=seeds[0],
            remediation_profile=profile,
        )
    return runs


def test_r5a_equals_r4c_on_explicit_diesel_route(diesel_runs: dict[str, Any]) -> None:
    r4c = diesel_runs["r4c_diesel_balanced_derivative_v1"]
    r5a = diesel_runs["r5a_diesel_absorbance_readout_v1"]
    np.testing.assert_array_equal(r5a.X, r4c.X)
    np.testing.assert_array_equal(r5a.y, r4c.y)


def test_r5b_equals_clipped_negative_log10_of_r5a(diesel_runs: dict[str, Any]) -> None:
    r5a = diesel_runs["r5a_diesel_absorbance_readout_v1"]
    r5b = diesel_runs["r5b_diesel_transmittance_readout_v1"]
    expected = np.clip(np.power(10.0, -np.asarray(r5a.X, dtype=float)), 0.0, 1.0)
    np.testing.assert_allclose(r5b.X, expected, rtol=0.0, atol=1e-12)


def test_r5c_equals_one_minus_clipped_negative_log10_of_r5a(
    diesel_runs: dict[str, Any],
) -> None:
    r5a = diesel_runs["r5a_diesel_absorbance_readout_v1"]
    r5c = diesel_runs["r5c_diesel_blank_referenced_intensity_v1"]
    expected = np.clip(
        1.0 - np.power(10.0, -np.asarray(r5a.X, dtype=float)), 0.0, 1.0
    )
    np.testing.assert_allclose(r5c.X, expected, rtol=0.0, atol=1e-12)


def test_r8b_support_mean_equals_r4c_and_off_support_equals_r4c(
    diesel_runs: dict[str, Any],
) -> None:
    """R8b applies a support-mean-preserving multiplicative modulation only on
    the 750-1550 nm support, so the per-row support mean must match R4c
    exactly, and outside the support the readout must be byte-identical to
    R4c (not just equal in mean)."""
    exp10 = _load_exp10()
    r4c = diesel_runs["r4c_diesel_balanced_derivative_v1"]
    r8b = diesel_runs["r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1"]
    wl = np.asarray(r4c.wavelengths, dtype=float).ravel()
    support_mask = (wl >= exp10.SUPPORT_LOW_NM) & (wl <= exp10.SUPPORT_HIGH_NM)
    assert support_mask.any(), "fixture wavelengths must include the support window"

    r4c_X = np.asarray(r4c.X, dtype=float)
    r8b_X = np.asarray(r8b.X, dtype=float)
    # Support row means must agree (the R8b mean-preserving renormalization
    # is exactly enforced by the builder).
    r4c_support_means = r4c_X[:, support_mask].mean(axis=1)
    r8b_support_means = r8b_X[:, support_mask].mean(axis=1)
    np.testing.assert_allclose(
        r8b_support_means, r4c_support_means, rtol=0.0, atol=1e-9
    )
    # Off-support is unchanged by the R8b modulation, so values must match
    # the R4c base byte-identically.
    if (~support_mask).any():
        np.testing.assert_array_equal(
            r8b_X[:, ~support_mask], r4c_X[:, ~support_mask]
        )
    # Sanity: R8b actually modulates the support (so it isn't identical to R4c).
    assert not np.allclose(r8b_X, r4c_X)


# ---------------------------------------------------------------------------
# Audit/run shape and metadata flags.
# ---------------------------------------------------------------------------


def test_run_audit_returns_blocked_no_real_data_when_cohorts_are_empty(
    tmp_path: Path,
) -> None:
    exp10 = _load_exp10()
    _write_empty_cohorts(tmp_path)

    result = exp10.run_audit(
        root=tmp_path,
        seeds=(20260430,),
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=2,
    )

    assert result["status"] == "blocked_no_real_data"
    assert result["rows"] == []
    assert result["seeds"] == [20260430]
    assert result["audited_profiles"] == list(exp10.R9A_AUDITED_PROFILES)
    assert result["support_low_nm"] == 750.0
    assert result["support_high_nm"] == 1550.0


def test_run_audit_rejects_unknown_profile(tmp_path: Path) -> None:
    exp10 = _load_exp10()
    _write_empty_cohorts(tmp_path)

    with pytest.raises(ValueError):
        exp10.run_audit(
            root=tmp_path,
            seeds=(20260430,),
            n_synthetic_samples=4,
            max_real_samples=4,
            max_sentinel_datasets=2,
            profiles=("not_a_real_profile",),
        )


def test_run_audit_requires_at_least_one_seed(tmp_path: Path) -> None:
    exp10 = _load_exp10()
    _write_empty_cohorts(tmp_path)

    with pytest.raises(ValueError):
        exp10.run_audit(
            root=tmp_path,
            seeds=(),
            n_synthetic_samples=4,
            max_real_samples=4,
            max_sentinel_datasets=2,
        )


def test_blocked_path_records_remediation_profiles_and_audit_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exp10 = _load_exp10()
    exp09 = _load_exp09()
    _write_empty_cohorts(tmp_path)

    candidates = [_make_dataset("DIESEL_bp50_246_b-a")]

    monkeypatch.setattr(
        exp09,
        "discover_local_real_datasets",
        lambda root: (list(candidates), []),
    )
    # Force the exp10 module to use the same patched discovery (it imports
    # ``discover_local_real_datasets`` directly from the realism module).
    monkeypatch.setattr(
        exp10,
        "discover_local_real_datasets",
        lambda root: (list(candidates), []),
    )
    monkeypatch.setattr(
        exp10,
        "load_real_spectra",
        lambda dataset, root: (_ for _ in ()).throw(
            RuntimeError("synthetic-test: no real spectra available")
        ),
    )

    profiles = (
        "r3d_diesel_matrix_v1",
        "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1",
    )
    seeds = (20260430, 20260431)
    result = exp10.run_audit(
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
        assert row.audit_scope == exp10.R9A_AUDIT_SCOPE
        assert row.comparison_space == exp10.COMPARISON_SPACE
        assert row.remediation_profile in profiles
        assert row.seed in seeds


def test_max_sentinel_zero_still_keeps_diesel_token_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exp10 = _load_exp10()
    _write_empty_cohorts(tmp_path)

    diesel_dataset = _make_dataset("DIESEL_bp50_246_b-a")
    non_diesel_dataset = _make_dataset("CORN_m5")

    monkeypatch.setattr(
        exp10,
        "discover_local_real_datasets",
        lambda root: ([diesel_dataset, non_diesel_dataset], []),
    )
    monkeypatch.setattr(
        exp10,
        "load_real_spectra",
        lambda dataset, root: (_ for _ in ()).throw(
            RuntimeError("synthetic-test: selected sentinel only")
        ),
    )

    result = exp10.run_audit(
        root=tmp_path,
        seeds=(20260430,),
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=0,
        sentinel_tokens=["DIESEL"],
        profiles=("r3d_diesel_matrix_v1",),
    )

    assert result["real_runnable_count"] == 2
    assert result["real_sentinel_candidate_count"] == 1
    assert result["real_selected_count"] == 1
    assert len(result["rows"]) == 1
    assert result["rows"][0].dataset == "DIESEL_bp50_246_b-a/ds"


# ---------------------------------------------------------------------------
# CSV / Markdown shape.
# ---------------------------------------------------------------------------


def test_write_csv_empty_rows_emits_stable_header(tmp_path: Path) -> None:
    exp10 = _load_exp10()
    csv_path = tmp_path / "empty.csv"

    exp10.write_csv([], csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        records = list(reader)
    assert records == []
    fieldnames = reader.fieldnames
    assert fieldnames == exp10._csv_fieldnames()
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


def test_render_markdown_disclaimer_and_no_gate_claims(tmp_path: Path) -> None:
    exp10 = _load_exp10()
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
        "audit_scope": exp10.R9A_AUDIT_SCOPE,
    }
    row = exp10.R9aRow(
        status="compared",
        seed=20260430,
        remediation_profile="r4c_diesel_balanced_derivative_v1",
        effective_remediation_profile="r4c_diesel_balanced_derivative_v1",
        source="AOM_regression",
        task="regression",
        dataset="DIESEL_bp50_246_b-a/ds",
        synthetic_preset="fuel",
        effective_matrix_route="diesel_fuel_matrix",
        comparison_space=exp10.COMPARISON_SPACE,
        n_real_samples=8,
        n_synthetic_samples=8,
        n_wavelengths=10,
        wavelength_min=900.0,
        wavelength_max=1550.0,
        support_low_nm=exp10.SUPPORT_LOW_NM,
        support_high_nm=exp10.SUPPORT_HIGH_NM,
        support_count=10,
        off_support_count=0,
        support_weight=1.0,
        off_support_weight=0.0,
        support_mean_delta=0.002,
        off_support_mean_delta=0.0,
        support_weighted_delta=0.002,
        off_support_weighted_delta=0.0,
        global_mean_delta=0.002,
        decomposition_residual=0.0,
        real_global_mean=0.003,
        synthetic_global_mean=0.005,
        real_global_std=0.014,
        synthetic_global_std=0.015,
        log10_global_std_ratio=0.03,
        log10_amplitude_p50_ratio=0.0,
        log10_derivative_std_p50_ratio=0.0,
        mean_curve_corr=0.10,
        morphology_gap_score=1.20,
        dominant_morphology_gap="mean_shift",
        **audit_kwargs,
        blocked_reason="",
    )
    result = {
        "status": "done",
        "rows": [row],
        "real_runnable_count": 1,
        "real_sentinel_candidate_count": 1,
        "real_selected_count": 1,
        "sentinel_tokens": ["DIESEL"],
        "seeds": [20260430, 20260431, 20260432],
        "audited_profiles": list(exp10.R9A_AUDITED_PROFILES),
        "support_low_nm": exp10.SUPPORT_LOW_NM,
        "support_high_nm": exp10.SUPPORT_HIGH_NM,
    }

    md = exp10.render_markdown(
        result=result,
        report_path=tmp_path / "r9a.md",
        csv_path=tmp_path / "r9a.csv",
        n_synthetic_samples=64,
        max_real_samples=64,
        max_sentinel_datasets=8,
        seeds=[20260430, 20260431, 20260432],
        sentinel_tokens=["DIESEL"],
        profiles=list(exp10.R9A_AUDITED_PROFILES),
    )

    assert "uncalibrated_raw" in md
    assert "diagnostic-only" in md.lower()
    assert "no calibration" in md.lower()
    assert "no PCA/covariance" in md
    assert "no ML/DL" in md
    assert "B2/B3/B4/B5" in md
    assert "750-1550" in md
    assert "DIESEL" in md
    assert "should appear in `off_support_mean_delta` only" not in md
    assert "support shape/derivative diagnostics" in md
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
    ):
        assert flag in md, f"missing audit flag {flag!r}"
    # No gate-PASS claims.
    assert "B2 PASS" not in md.upper().replace(":", "")
    assert "B3 PASS" not in md.upper().replace(":", "")


def _make_compared_row(
    exp10: ModuleType,
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
        "audit_scope": exp10.R9A_AUDIT_SCOPE,
    }
    return exp10.R9aRow(
        status="compared",
        seed=seed,
        remediation_profile=profile,
        effective_remediation_profile=profile,
        source="AOM_regression",
        task="regression",
        dataset="DIESEL_bp50_246_b-a/ds",
        synthetic_preset="fuel",
        effective_matrix_route="diesel_fuel_matrix",
        comparison_space=exp10.COMPARISON_SPACE,
        n_real_samples=8,
        n_synthetic_samples=8,
        n_wavelengths=10,
        wavelength_min=900.0,
        wavelength_max=1550.0,
        support_low_nm=exp10.SUPPORT_LOW_NM,
        support_high_nm=exp10.SUPPORT_HIGH_NM,
        support_count=10,
        off_support_count=0,
        support_weight=support_weight,
        off_support_weight=off_support_weight,
        support_mean_delta=global_mean_delta,
        off_support_mean_delta=off_support_mean_delta,
        support_weighted_delta=global_mean_delta,
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
    exp10: ModuleType, rows: list[Any], tmp_path: Path
) -> Any:
    result = {
        "status": "done",
        "rows": rows,
        "real_runnable_count": 1,
        "real_sentinel_candidate_count": 1,
        "real_selected_count": 1,
        "sentinel_tokens": ["DIESEL"],
        "seeds": [20260430],
        "audited_profiles": list(exp10.R9A_AUDITED_PROFILES),
        "support_low_nm": exp10.SUPPORT_LOW_NM,
        "support_high_nm": exp10.SUPPORT_HIGH_NM,
    }
    return exp10.render_markdown(
        result=result,
        report_path=tmp_path / "r9a.md",
        csv_path=tmp_path / "r9a.csv",
        n_synthetic_samples=64,
        max_real_samples=64,
        max_sentinel_datasets=8,
        seeds=[20260430],
        sentinel_tokens=["DIESEL"],
        profiles=list(exp10.R9A_AUDITED_PROFILES),
    )


def test_render_markdown_includes_per_profile_synthesis_table(
    tmp_path: Path,
) -> None:
    exp10 = _load_exp10()
    rows = [
        _make_compared_row(
            exp10,
            profile="r4c_diesel_balanced_derivative_v1",
            seed=20260430,
            global_mean_delta=0.001,
            support_weight=1.0,
            off_support_weight=0.0,
            morphology_gap_score=1.10,
            mean_curve_corr=0.20,
            log10_derivative_std_p50_ratio=0.05,
            dominant_morphology_gap="mean_shift",
        ),
        _make_compared_row(
            exp10,
            profile="r4c_diesel_balanced_derivative_v1",
            seed=20260431,
            global_mean_delta=0.003,
            support_weight=1.0,
            off_support_weight=0.0,
            morphology_gap_score=1.30,
            mean_curve_corr=0.22,
            log10_derivative_std_p50_ratio=0.07,
            dominant_morphology_gap="mean_shift",
        ),
        _make_compared_row(
            exp10,
            profile="r3d_diesel_matrix_v1",
            seed=20260430,
            global_mean_delta=0.005,
            support_weight=1.0,
            off_support_weight=0.0,
            morphology_gap_score=1.50,
            mean_curve_corr=0.10,
            log10_derivative_std_p50_ratio=0.09,
            dominant_morphology_gap="amplitude",
        ),
    ]

    md = _render_with_rows(exp10, rows, tmp_path)

    assert "## Per-Profile Synthesis (compared rows only)" in md
    # Aggregated header columns must be present.
    assert "median global mean delta" in md
    assert "median support weight" in md
    assert "median off-support weight" in md
    assert "median morphology gap score" in md
    assert "median mean curve corr" in md
    assert "median log10 deriv std p50 ratio" in md
    assert "dominant gap dist." in md
    # Both profile rows must appear in the synthesis table.
    assert "`r4c_diesel_balanced_derivative_v1`" in md
    assert "`r3d_diesel_matrix_v1`" in md
    # n=2 for r4c, n=1 for r3d.
    assert "| `r4c_diesel_balanced_derivative_v1` | 2 |" in md
    assert "| `r3d_diesel_matrix_v1` | 1 |" in md
    # Dominant gap distribution must be rendered for r4c.
    assert "mean_shift=2" in md


def test_render_markdown_emits_off_support_null_note_when_all_zero(
    tmp_path: Path,
) -> None:
    exp10 = _load_exp10()
    rows = [
        _make_compared_row(
            exp10,
            profile="r4c_diesel_balanced_derivative_v1",
            seed=20260430,
            global_mean_delta=0.001,
            support_weight=1.0,
            off_support_weight=0.0,
            morphology_gap_score=1.10,
            mean_curve_corr=0.20,
            log10_derivative_std_p50_ratio=0.05,
            dominant_morphology_gap="mean_shift",
        ),
    ]

    md = _render_with_rows(exp10, rows, tmp_path)

    assert exp10.OFF_SUPPORT_NULL_NOTE in md
    assert "structurally null" in md


def test_render_markdown_skips_off_support_null_note_when_any_nonzero(
    tmp_path: Path,
) -> None:
    exp10 = _load_exp10()
    rows = [
        _make_compared_row(
            exp10,
            profile="r4c_diesel_balanced_derivative_v1",
            seed=20260430,
            global_mean_delta=0.001,
            support_weight=0.6,
            off_support_weight=0.4,
            morphology_gap_score=1.10,
            mean_curve_corr=0.20,
            log10_derivative_std_p50_ratio=0.05,
            dominant_morphology_gap="mean_shift",
            off_support_mean_delta=0.002,
            off_support_weighted_delta=0.0008,
        ),
    ]

    md = _render_with_rows(exp10, rows, tmp_path)

    assert exp10.OFF_SUPPORT_NULL_NOTE not in md
    assert "structurally null" not in md


def test_aggregate_by_profile_ignores_blocked_rows(tmp_path: Path) -> None:
    exp10 = _load_exp10()
    compared = _make_compared_row(
        exp10,
        profile="r4c_diesel_balanced_derivative_v1",
        seed=20260430,
        global_mean_delta=0.002,
        support_weight=1.0,
        off_support_weight=0.0,
        morphology_gap_score=1.20,
        mean_curve_corr=0.25,
        log10_derivative_std_p50_ratio=0.06,
        dominant_morphology_gap="mean_shift",
    )
    blocked = exp10._blocked_row(
        seed=20260430,
        requested_profile="r3d_diesel_matrix_v1",
        effective_profile="r3d_diesel_matrix_v1",
        dataset=_make_dataset("DIESEL_bp50_246_b-a"),
        preset="fuel",
        blocked_reason="synthetic-test",
    )
    summary = exp10._aggregate_by_profile([compared, blocked])
    assert len(summary) == 1
    assert summary[0]["profile"] == "r4c_diesel_balanced_derivative_v1"
    assert summary[0]["n"] == 1


def test_run_audit_preserves_r8b_r4c_decomposition_under_synthetic_downsample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exp10 = _load_exp10()
    exp09 = _load_exp09()
    _write_empty_cohorts(tmp_path)

    diesel_dataset = _make_dataset("DIESEL_bp50_246_b-a")
    wl = np.array([700.0, 800.0, 900.0, 1600.0, 1700.0], dtype=float)
    support_mask = (wl >= exp10.SUPPORT_LOW_NM) & (wl <= exp10.SUPPORT_HIGH_NM)
    fake_real = np.zeros((20, wl.size), dtype=float)
    row_means = np.linspace(0.02, 0.13, 12)

    def fake_stable_dataset_seed(seed: int, dataset: Any, purpose: str) -> int:
        if purpose == "r9a:real_downsample":
            return 101
        if purpose == "r9a:syn_downsample":
            return 202
        if purpose.endswith(":r4c_diesel_balanced_derivative_v1"):
            return 303
        if purpose.endswith(
            ":r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1"
        ):
            return 404
        return 505

    def fake_build_baseline_synthetic_run(
        *,
        dataset: Any,
        preset: str,
        n_samples: int,
        seed: int,
        remediation_profile: str | None = None,
    ) -> Any:
        assert n_samples == row_means.size
        X = np.zeros((row_means.size, wl.size), dtype=float)
        X[:, support_mask] = row_means[:, None]
        X[:, ~support_mask] = (0.2 + row_means[:, None])
        if remediation_profile == (
            "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1"
        ):
            X[:, support_mask] = np.column_stack(
                [row_means - 0.005, row_means + 0.005]
            )
        return SimpleNamespace(
            X=X,
            y=np.zeros(row_means.size, dtype=float),
            wavelengths=wl,
            metadata={},
        )

    monkeypatch.setattr(
        exp09,
        "discover_local_real_datasets",
        lambda root: ([diesel_dataset], []),
    )
    monkeypatch.setattr(
        exp10,
        "discover_local_real_datasets",
        lambda root: ([diesel_dataset], []),
    )
    monkeypatch.setattr(
        exp10,
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
        "r4c_diesel_balanced_derivative_v1",
        "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1",
    )
    result = exp10.run_audit(
        root=tmp_path,
        seeds=(20260430,),
        n_synthetic_samples=row_means.size,
        max_real_samples=4,
        max_sentinel_datasets=1,
        sentinel_tokens=["DIESEL"],
        profiles=profiles,
    )

    assert result["status"] == "done"
    rows = {row.remediation_profile: row for row in result["rows"]}
    r4c = rows["r4c_diesel_balanced_derivative_v1"]
    r8b = rows["r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1"]
    for attr in (
        "support_mean_delta",
        "off_support_mean_delta",
        "support_weighted_delta",
        "off_support_weighted_delta",
        "global_mean_delta",
        "decomposition_residual",
    ):
        assert getattr(r8b, attr) == pytest.approx(getattr(r4c, attr), abs=1e-12)


# ---------------------------------------------------------------------------
# Smoke test: tiny end-to-end audit on a synthetic in-memory cohort.
# ---------------------------------------------------------------------------


def test_run_audit_smoke_one_seed_one_profile_single_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exp10 = _load_exp10()
    exp09 = _load_exp09()
    _write_empty_cohorts(tmp_path)

    diesel_dataset = _make_dataset("DIESEL_bp50_246_b-a")

    fake_wl = np.linspace(700.0, 1700.0, 80)
    rng = np.random.default_rng(20260430)
    fake_real = rng.normal(0.05, 0.005, size=(16, fake_wl.size))

    monkeypatch.setattr(
        exp09,
        "discover_local_real_datasets",
        lambda root: ([diesel_dataset], []),
    )
    monkeypatch.setattr(
        exp10,
        "discover_local_real_datasets",
        lambda root: ([diesel_dataset], []),
    )
    monkeypatch.setattr(
        exp10,
        "load_real_spectra",
        lambda dataset, root: (fake_real, fake_wl),
    )

    result = exp10.run_audit(
        root=tmp_path,
        seeds=(20260430,),
        n_synthetic_samples=8,
        max_real_samples=8,
        max_sentinel_datasets=2,
        sentinel_tokens=["DIESEL"],
        profiles=("r4c_diesel_balanced_derivative_v1",),
    )

    assert result["status"] == "done"
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row.status == "compared"
    assert row.remediation_profile == "r4c_diesel_balanced_derivative_v1"
    assert row.effective_remediation_profile == "r4c_diesel_balanced_derivative_v1"
    assert row.seed == 20260430
    assert row.comparison_space == exp10.COMPARISON_SPACE
    assert row.audit_scope == exp10.R9A_AUDIT_SCOPE
    assert row.audit_oracle is False
    assert row.audit_real_stat_capture is False
    assert row.audit_thresholds_modified is False
    # The decomposition identity holds on the audit path too.
    assert row.global_mean_delta is not None
    assert row.support_weighted_delta is not None
    assert row.off_support_weighted_delta is not None
    reconstructed = row.support_weighted_delta + row.off_support_weighted_delta
    assert abs((row.global_mean_delta or 0.0) - reconstructed) <= 1e-9
    assert row.decomposition_residual is not None
    assert abs(row.decomposition_residual) <= 1e-9
    # Support window must include at least some wavelengths (701-1700 grid
    # crosses the 750-1550 nm window).
    assert row.support_count > 0
    assert row.off_support_count >= 0
