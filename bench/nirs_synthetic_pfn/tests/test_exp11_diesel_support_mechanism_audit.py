"""Tests for the R9b DIESEL support-level mechanism audit (exp11)."""

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


def _load_exp11() -> ModuleType:
    return _load_module(
        "exp11_diesel_support_mechanism_audit",
        "exp11_diesel_support_mechanism_audit.py",
    )


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


def test_exp11_exposes_audited_profiles_and_default_seeds() -> None:
    exp11 = _load_exp11()
    assert exp11.R9B_AUDITED_PROFILES == (
        "r3d_diesel_matrix_v1",
        "r4a_diesel_basis_v1",
        "r4c_diesel_balanced_derivative_v1",
        "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1",
        "r9b_diesel_support_intercept_v1",
    )
    assert exp11.R9B_PAIRED_REFERENCE_PROFILES == (
        "r3d_diesel_matrix_v1",
        "r4a_diesel_basis_v1",
        "r4c_diesel_balanced_derivative_v1",
    )
    assert exp11.DEFAULT_SEEDS == (20260501, 20260502, 20260503)
    assert exp11.DEFAULT_N_SYNTHETIC_SAMPLES == 64
    assert exp11.DEFAULT_MAX_REAL_SAMPLES == 64
    assert exp11.SUPPORT_LOW_NM == 750.0
    assert exp11.SUPPORT_HIGH_NM == 1550.0
    assert exp11.COMPARISON_SPACE == "uncalibrated_raw"
    assert exp11.R9B_AUDIT_SCOPE == (
        "bench_only_r9b_diesel_support_mechanism_audit"
    )
    assert exp11.DEFAULT_REPORT.name == (
        "r9b_diesel_support_mechanism_audit.md"
    )
    assert exp11.DEFAULT_CSV.name == (
        "r9b_diesel_support_mechanism_audit.csv"
    )


# ---------------------------------------------------------------------------
# Audit/run shape and metadata flags.
# ---------------------------------------------------------------------------


def test_run_audit_returns_blocked_no_real_data_when_cohorts_are_empty(
    tmp_path: Path,
) -> None:
    exp11 = _load_exp11()
    _write_empty_cohorts(tmp_path)

    result = exp11.run_audit(
        root=tmp_path,
        seeds=(20260501,),
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=2,
    )

    assert result["status"] == "blocked_no_real_data"
    assert result["rows"] == []
    assert result["seeds"] == [20260501]
    assert result["audited_profiles"] == list(exp11.R9B_AUDITED_PROFILES)
    assert result["support_low_nm"] == 750.0
    assert result["support_high_nm"] == 1550.0


def test_run_audit_rejects_unknown_profile(tmp_path: Path) -> None:
    exp11 = _load_exp11()
    _write_empty_cohorts(tmp_path)

    with pytest.raises(ValueError):
        exp11.run_audit(
            root=tmp_path,
            seeds=(20260501,),
            n_synthetic_samples=4,
            max_real_samples=4,
            max_sentinel_datasets=2,
            profiles=("not_a_real_profile",),
        )


def test_run_audit_requires_at_least_one_seed(tmp_path: Path) -> None:
    exp11 = _load_exp11()
    _write_empty_cohorts(tmp_path)

    with pytest.raises(ValueError):
        exp11.run_audit(
            root=tmp_path,
            seeds=(),
            n_synthetic_samples=4,
            max_real_samples=4,
            max_sentinel_datasets=2,
        )


def test_blocked_path_records_remediation_profiles_and_audit_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exp11 = _load_exp11()
    exp09 = _load_exp09()
    _write_empty_cohorts(tmp_path)

    candidates = [_make_dataset("DIESEL_bp50_246_b-a")]

    monkeypatch.setattr(
        exp09,
        "discover_local_real_datasets",
        lambda root: (list(candidates), []),
    )
    monkeypatch.setattr(
        exp11,
        "discover_local_real_datasets",
        lambda root: (list(candidates), []),
    )
    monkeypatch.setattr(
        exp11,
        "load_real_spectra",
        lambda dataset, root: (_ for _ in ()).throw(
            RuntimeError("synthetic-test: no real spectra available")
        ),
    )

    profiles = (
        "r3d_diesel_matrix_v1",
        "r9b_diesel_support_intercept_v1",
    )
    seeds = (20260501, 20260502)
    result = exp11.run_audit(
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
        assert row.audit_scope == exp11.R9B_AUDIT_SCOPE
        assert row.comparison_space == exp11.COMPARISON_SPACE
        assert row.remediation_profile in profiles
        assert row.seed in seeds


# ---------------------------------------------------------------------------
# CSV / Markdown shape.
# ---------------------------------------------------------------------------


def test_write_csv_empty_rows_emits_stable_header(tmp_path: Path) -> None:
    exp11 = _load_exp11()
    csv_path = tmp_path / "empty.csv"

    exp11.write_csv([], csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        records = list(reader)
    assert records == []
    fieldnames = reader.fieldnames
    assert fieldnames == exp11._csv_fieldnames()
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


def _make_compared_row(
    exp11: ModuleType,
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
        "audit_scope": exp11.R9B_AUDIT_SCOPE,
    }
    if support_mean_delta is None:
        support_mean_delta = global_mean_delta
    if support_weighted_delta is None:
        support_weighted_delta = global_mean_delta
    return exp11.R9bRow(
        status="compared",
        seed=seed,
        remediation_profile=profile,
        effective_remediation_profile=profile,
        source="AOM_regression",
        task="regression",
        dataset="DIESEL_bp50_246_b-a/ds",
        synthetic_preset="fuel",
        effective_matrix_route="diesel_fuel_matrix",
        comparison_space=exp11.COMPARISON_SPACE,
        n_real_samples=8,
        n_synthetic_samples=8,
        n_wavelengths=10,
        wavelength_min=900.0,
        wavelength_max=1550.0,
        support_low_nm=exp11.SUPPORT_LOW_NM,
        support_high_nm=exp11.SUPPORT_HIGH_NM,
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
    exp11: ModuleType, rows: list[Any], tmp_path: Path
) -> Any:
    result = {
        "status": "done",
        "rows": rows,
        "real_runnable_count": 1,
        "real_sentinel_candidate_count": 1,
        "real_selected_count": 1,
        "sentinel_tokens": ["DIESEL"],
        "seeds": [20260501],
        "audited_profiles": list(exp11.R9B_AUDITED_PROFILES),
        "support_low_nm": exp11.SUPPORT_LOW_NM,
        "support_high_nm": exp11.SUPPORT_HIGH_NM,
    }
    return exp11.render_markdown(
        result=result,
        report_path=tmp_path / "r9b.md",
        csv_path=tmp_path / "r9b.csv",
        n_synthetic_samples=64,
        max_real_samples=64,
        max_sentinel_datasets=8,
        seeds=[20260501],
        sentinel_tokens=["DIESEL"],
        profiles=list(exp11.R9B_AUDITED_PROFILES),
    )


def test_render_markdown_disclaimer_and_no_gate_claims(tmp_path: Path) -> None:
    exp11 = _load_exp11()
    rows = [
        _make_compared_row(
            exp11,
            profile="r9b_diesel_support_intercept_v1",
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
    md = _render_with_rows(exp11, rows, tmp_path)

    assert "uncalibrated_raw" in md
    assert "diagnostic-only" in md.lower()
    assert "no calibration" in md.lower()
    assert "no PCA/covariance" in md
    assert "no ML/DL" in md
    assert "B2/B3/B4/B5" in md
    assert "750-1550" in md
    assert "DIESEL" in md
    # Pre-declared mechanistic constant disclaimer.
    assert "PRE-DECLARED MECHANISTIC CONSTANT" in md
    assert "NOT chosen from any R9a or R9b mean-shift residual delta" in md
    # No promotion / no gate claims.
    assert "not a promotion over R3d" in md.lower() or (
        "not a promotion over r3d" in md.lower()
    )
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
    assert "B2 PASS" not in md.upper().replace(":", "")


def test_render_markdown_includes_per_profile_synthesis_table(
    tmp_path: Path,
) -> None:
    exp11 = _load_exp11()
    rows = [
        _make_compared_row(
            exp11,
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
            exp11,
            profile="r9b_diesel_support_intercept_v1",
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
    md = _render_with_rows(exp11, rows, tmp_path)

    assert "## Per-Profile Synthesis (compared rows only)" in md
    assert "median global mean delta" in md
    assert "median support mean delta" in md
    assert "median morphology gap score" in md
    assert "`r4c_diesel_balanced_derivative_v1`" in md
    assert "`r9b_diesel_support_intercept_v1`" in md
    assert "## Paired Deltas: R9b minus reference profile" in md
    assert "r9b vs r4c global" in md
    assert "r9b vs r4a morphology gap" in md
    assert "r9b vs r3d global" in md


def test_paired_deltas_include_r9b_minus_r4c_and_handle_missing_reference(
    tmp_path: Path,
) -> None:
    exp11 = _load_exp11()
    rows = [
        _make_compared_row(
            exp11,
            profile="r9b_diesel_support_intercept_v1",
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
            exp11,
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
            exp11,
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
    ]

    paired = exp11._paired_deltas_vs_references(rows)
    assert len(paired) == 1
    entry = paired[0]
    assert entry["seed"] == 20260501
    assert entry["dataset"] == "DIESEL_bp50_246_b-a/ds"
    assert entry["r9b_global_mean_delta"] == pytest.approx(0.0015)
    # r4c reference present: -0.0020.
    assert entry[
        "delta_r9b_minus_r4c_diesel_balanced_derivative_v1__global_mean_delta"
    ] == pytest.approx(0.0015 - 0.0035)
    assert entry[
        "delta_r9b_minus_r4c_diesel_balanced_derivative_v1__morphology_gap_score"
    ] == pytest.approx(1.05 - 1.30)
    # r3d reference present.
    assert entry[
        "delta_r9b_minus_r3d_diesel_matrix_v1__global_mean_delta"
    ] == pytest.approx(0.0015 - 0.0050)
    # r4a reference missing -> NA.
    assert entry[
        "delta_r9b_minus_r4a_diesel_basis_v1__global_mean_delta"
    ] is None
    assert entry[
        "delta_r9b_minus_r4a_diesel_basis_v1__morphology_gap_score"
    ] is None


def test_paired_deltas_skip_blocked_rows_and_missing_r9b(
    tmp_path: Path,
) -> None:
    exp11 = _load_exp11()
    rows = [
        _make_compared_row(
            exp11,
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
        # No r9b compared row - paired table must be empty.
    ]
    assert exp11._paired_deltas_vs_references(rows) == []


def test_render_markdown_emits_off_support_null_note_when_all_zero(
    tmp_path: Path,
) -> None:
    exp11 = _load_exp11()
    rows = [
        _make_compared_row(
            exp11,
            profile="r9b_diesel_support_intercept_v1",
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
    md = _render_with_rows(exp11, rows, tmp_path)
    assert exp11.OFF_SUPPORT_NULL_NOTE in md


def test_aggregate_by_profile_ignores_blocked_rows(tmp_path: Path) -> None:
    exp11 = _load_exp11()
    compared = _make_compared_row(
        exp11,
        profile="r9b_diesel_support_intercept_v1",
        seed=20260501,
        global_mean_delta=0.001,
        support_weight=1.0,
        off_support_weight=0.0,
        morphology_gap_score=1.10,
        mean_curve_corr=0.30,
        log10_derivative_std_p50_ratio=0.05,
        dominant_morphology_gap="mean_shift",
    )
    blocked = exp11._blocked_row(
        seed=20260501,
        requested_profile="r3d_diesel_matrix_v1",
        effective_profile="r3d_diesel_matrix_v1",
        dataset=_make_dataset("DIESEL_bp50_246_b-a"),
        preset="fuel",
        blocked_reason="synthetic-test",
    )
    summary = exp11._aggregate_by_profile([compared, blocked])
    assert len(summary) == 1
    assert summary[0]["profile"] == "r9b_diesel_support_intercept_v1"
    assert summary[0]["n"] == 1


def test_run_audit_smoke_one_seed_two_profiles_single_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end audit on a synthetic in-memory cohort: R4c vs R9b."""
    exp11 = _load_exp11()
    exp09 = _load_exp09()
    _write_empty_cohorts(tmp_path)

    diesel_dataset = _make_dataset("DIESEL_bp50_246_b-a")
    fake_wl = np.linspace(700.0, 1700.0, 80)
    rng = np.random.default_rng(20260501)
    fake_real = rng.normal(0.05, 0.005, size=(16, fake_wl.size))

    monkeypatch.setattr(
        exp09,
        "discover_local_real_datasets",
        lambda root: ([diesel_dataset], []),
    )
    monkeypatch.setattr(
        exp11,
        "discover_local_real_datasets",
        lambda root: ([diesel_dataset], []),
    )
    monkeypatch.setattr(
        exp11,
        "load_real_spectra",
        lambda dataset, root: (fake_real, fake_wl),
    )

    profiles = (
        "r4c_diesel_balanced_derivative_v1",
        "r9b_diesel_support_intercept_v1",
    )
    result = exp11.run_audit(
        root=tmp_path,
        seeds=(20260501,),
        n_synthetic_samples=8,
        max_real_samples=8,
        max_sentinel_datasets=2,
        sentinel_tokens=["DIESEL"],
        profiles=profiles,
    )

    assert result["status"] == "done"
    assert len(result["rows"]) == 2
    rows = {row.remediation_profile: row for row in result["rows"]}
    r4c = rows["r4c_diesel_balanced_derivative_v1"]
    r9b = rows["r9b_diesel_support_intercept_v1"]
    assert r4c.status == "compared"
    assert r9b.status == "compared"
    assert r9b.audit_scope == exp11.R9B_AUDIT_SCOPE
    assert r9b.audit_oracle is False
    assert r9b.audit_real_stat_capture is False
    assert r9b.audit_thresholds_modified is False
    # Decomposition identity holds on the audit path.
    for row in (r4c, r9b):
        assert row.global_mean_delta is not None
        assert row.support_weighted_delta is not None
        assert row.off_support_weighted_delta is not None
        reconstructed = row.support_weighted_delta + row.off_support_weighted_delta
        assert abs((row.global_mean_delta or 0.0) - reconstructed) <= 1e-9


def test_run_audit_paired_delta_against_synthetic_in_memory_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Smoke-driver paired delta check: with deterministic monkeypatched X
    where R9b == R4c base + intercept on support, the support_mean_delta
    paired delta must equal exactly the intercept and the off-support delta
    must be zero.
    """
    exp11 = _load_exp11()
    exp09 = _load_exp09()
    _write_empty_cohorts(tmp_path)

    diesel_dataset = _make_dataset("DIESEL_bp50_246_b-a")
    wl = np.array([700.0, 800.0, 900.0, 1000.0, 1100.0, 1600.0, 1700.0], dtype=float)
    support_mask = (wl >= exp11.SUPPORT_LOW_NM) & (wl <= exp11.SUPPORT_HIGH_NM)
    fake_real = np.zeros((20, wl.size), dtype=float)
    intercept = 0.002

    n_rows = 8

    def fake_stable_dataset_seed(seed: int, dataset: Any, purpose: str) -> int:
        # Stable seeds used only for downsample steps; the synthetic builder
        # is replaced wholesale below.
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
        # Simple deterministic R4c-base profile.
        X[:, support_mask] = 0.05
        X[:, ~support_mask] = 0.02
        if remediation_profile == "r9b_diesel_support_intercept_v1":
            X[:, support_mask] = X[:, support_mask] + intercept
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
        exp11,
        "discover_local_real_datasets",
        lambda root: ([diesel_dataset], []),
    )
    monkeypatch.setattr(
        exp11,
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
        "r9b_diesel_support_intercept_v1",
    )
    result = exp11.run_audit(
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
    r4c = rows["r4c_diesel_balanced_derivative_v1"]
    r9b = rows["r9b_diesel_support_intercept_v1"]
    paired = exp11._paired_deltas_vs_references(result["rows"])
    assert len(paired) == 1
    entry = paired[0]
    # Off-support delta of R9b vs R4c is zero by construction.
    assert entry[
        "delta_r9b_minus_r4c_diesel_balanced_derivative_v1__off_support_weighted_delta"
    ] == pytest.approx(0.0, abs=1e-12)
    # Support mean delta of R9b vs R4c equals exactly the intercept on the
    # decomposition support component.
    assert entry[
        "delta_r9b_minus_r4c_diesel_balanced_derivative_v1__support_mean_delta"
    ] == pytest.approx(intercept, abs=1e-12)
    assert (r9b.support_mean_delta or 0.0) - (r4c.support_mean_delta or 0.0) == (
        pytest.approx(intercept, abs=1e-12)
    )


def test_write_csv_includes_paired_delta_columns_on_r9b_rows(
    tmp_path: Path,
) -> None:
    """CSV must carry paired delta columns (R9b minus reference) populated on
    R9b compared rows and blank on non-R9b rows."""
    exp11 = _load_exp11()
    rows = [
        _make_compared_row(
            exp11,
            profile="r9b_diesel_support_intercept_v1",
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
            exp11,
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
    csv_path = tmp_path / "r9b.csv"
    exp11.write_csv(rows, csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        records = list(reader)
        fieldnames = reader.fieldnames

    assert fieldnames is not None
    assert (
        "delta_r9b_minus_r4c_diesel_balanced_derivative_v1__global_mean_delta"
        in fieldnames
    )
    assert (
        "delta_r9b_minus_r4c_diesel_balanced_derivative_v1__morphology_gap_score"
        in fieldnames
    )

    r9b_record = next(
        rec
        for rec in records
        if rec["remediation_profile"] == "r9b_diesel_support_intercept_v1"
    )
    assert float(
        r9b_record[
            "delta_r9b_minus_r4c_diesel_balanced_derivative_v1__global_mean_delta"
        ]
    ) == pytest.approx(0.0015 - 0.0035)
    assert float(
        r9b_record[
            "delta_r9b_minus_r4c_diesel_balanced_derivative_v1__morphology_gap_score"
        ]
    ) == pytest.approx(1.05 - 1.30)

    r4c_record = next(
        rec
        for rec in records
        if rec["remediation_profile"] == "r4c_diesel_balanced_derivative_v1"
    )
    # Non-R9b rows leave the paired delta cells blank.
    assert (
        r4c_record[
            "delta_r9b_minus_r4c_diesel_balanced_derivative_v1__global_mean_delta"
        ]
        == ""
    )
