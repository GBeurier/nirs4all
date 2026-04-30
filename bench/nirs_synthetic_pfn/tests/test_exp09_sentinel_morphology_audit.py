"""Schema and report tests for the R2b sentinel morphology audit."""

from __future__ import annotations

import ast
import importlib.util
import sys
from csv import DictReader
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest


def _load_exp09_module() -> ModuleType:
    name = "exp09_sentinel_morphology_audit"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "experiments/exp09_sentinel_morphology_audit.py"
    experiments_dir = str(path.parent)
    if experiments_dir not in sys.path:
        sys.path.insert(0, experiments_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _exp09_source_path() -> Path:
    return Path(__file__).resolve().parents[1] / "experiments/exp09_sentinel_morphology_audit.py"


def _write_empty_cohorts(root: Path) -> None:
    cohort_dir = root / "bench/AOM_v0/benchmarks"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    header = (
        "database_name,dataset,status,reason,n_train,n_test,p,"
        "train_path,test_path,ytrain_path,ytest_path\n"
    )
    (cohort_dir / "cohort_regression.csv").write_text(header, encoding="utf-8")
    (cohort_dir / "cohort_classification.csv").write_text(header, encoding="utf-8")


def _audit_kwargs() -> dict[str, Any]:
    return {
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
        "audit_scope": "bench_only_r2b_sentinel_morphology_audit",
    }


def _remediation_kwargs(
    *,
    profile: str | None = None,
    effective_matrix_route: str | None = None,
    enabled: bool = False,
    domain_key: str | None = None,
    concentrations_applied: bool = False,
    spectra_applied: bool = False,
    spectra_rule: str | None = None,
    composition_source: str | None = None,
    spectra_source: str | None = None,
    provenance_source: str | None = None,
    route_variant: str | None = None,
    constant_status: str | None = None,
    readout_space: str | None = None,
    calibration_source: str | None = None,
    real_stat_source: str | None = None,
    threshold_source: str | None = None,
) -> dict[str, Any]:
    return {
        "remediation_profile": profile,
        "effective_matrix_route": effective_matrix_route,
        "r2c_remediation_enabled": enabled,
        "r2c_remediation_domain_key": domain_key,
        "r2c_remediation_concentrations_applied": concentrations_applied,
        "r2c_remediation_spectra_applied": spectra_applied,
        "r2c_remediation_spectra_rule": spectra_rule,
        "r2c_remediation_composition_source": composition_source,
        "r2c_remediation_spectra_source": spectra_source,
        "r2c_remediation_provenance_source": provenance_source,
        "r2c_remediation_route_variant": route_variant,
        "r2c_remediation_constant_status": constant_status,
        "r2c_remediation_readout_space": readout_space,
        "r2c_remediation_calibration_source": calibration_source,
        "r2c_remediation_real_stat_source": real_stat_source,
        "r2c_remediation_threshold_source": threshold_source,
    }


def test_run_audit_returns_blocked_no_real_data_when_cohorts_are_empty(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    _write_empty_cohorts(tmp_path)

    result = exp09.run_audit(
        root=tmp_path,
        n_synthetic_samples=12,
        max_real_samples=12,
        max_sentinel_datasets=2,
        seed=1234,
    )

    assert result["status"] == "blocked_no_real_data"
    assert result["rows"] == []
    assert result["real_runnable_count"] == 0
    assert result["real_sentinel_candidate_count"] == 0
    assert result["real_selected_count"] == 0
    assert result["sentinel_tokens"] == list(exp09.DEFAULT_SENTINEL_TOKENS)
    assert result["remediation_profile"] is None


def test_write_csv_empty_rows_still_produces_stable_header(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    csv_path = tmp_path / "empty.csv"

    exp09.write_csv([], csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        records = list(reader)
    assert records == []
    assert reader.fieldnames == exp09._csv_fieldnames()
    for required in (
        "status",
        "dataset",
        "comparison_space",
        "log10_global_std_ratio",
        "log10_amplitude_p50_ratio",
        "log10_derivative_std_p50_ratio",
        "mean_curve_corr",
        "morphology_gap_score",
        "dominant_morphology_gap",
        "audit_oracle",
        "audit_scope",
        "remediation_profile",
        "effective_matrix_route",
        "r2c_remediation_enabled",
        "r2c_remediation_domain_key",
        "r2c_remediation_concentrations_applied",
        "r2c_remediation_spectra_applied",
        "r2c_remediation_spectra_rule",
        "r2c_remediation_composition_source",
        "r2c_remediation_spectra_source",
        "r2c_remediation_provenance_source",
        "r2c_remediation_route_variant",
        "r2c_remediation_constant_status",
        "r2c_remediation_readout_space",
        "r2c_remediation_calibration_source",
        "r2c_remediation_real_stat_source",
        "r2c_remediation_threshold_source",
    ):
        assert required in reader.fieldnames


def test_compute_morphology_metrics_detects_variance_over() -> None:
    exp09 = _load_exp09_module()
    rng = np.random.default_rng(42)
    wl = np.linspace(900.0, 1700.0, 64)
    shared = 0.5 + 0.05 * np.sin(np.linspace(0.0, 2 * np.pi, wl.size))
    # Real: rows almost identical to shared curve (small variance, smooth -> low derivative).
    real = shared[None, :] + rng.normal(0.0, 0.005, size=(40, wl.size))
    # Synthetic: shared curve plus large per-row constant offsets -> high global std,
    # but rows stay smooth so derivative_std stays close to real.
    offsets = np.linspace(-0.5, 0.5, 40).reshape(-1, 1)
    synth = shared[None, :] + offsets + rng.normal(0.0, 0.005, size=(40, wl.size))

    metrics = exp09.compute_morphology_metrics(real, synth, wl)
    assert metrics["log10_global_std_ratio"] is not None
    assert metrics["log10_global_std_ratio"] > 0.5
    assert metrics["dominant_morphology_gap"] == "variance_over"
    assert metrics["morphology_gap_score"] is not None
    assert metrics["morphology_gap_score"] > 0.0


def test_compute_morphology_metrics_detects_amplitude_under() -> None:
    exp09 = _load_exp09_module()
    rng = np.random.default_rng(0)
    wl = np.linspace(900.0, 1700.0, 64)
    base_real = rng.normal(0.0, 0.001, size=(40, wl.size))
    base_synth = rng.normal(0.0, 0.001, size=(40, wl.size))
    # Real has a strong wavelength-dependent slope (large amplitude).
    real = base_real + np.linspace(-1.0, 1.0, wl.size)[None, :]
    # Synthetic shares the same mean curve direction but with very small amplitude.
    synth = base_synth + 0.01 * np.linspace(-1.0, 1.0, wl.size)[None, :]

    metrics = exp09.compute_morphology_metrics(real, synth, wl)
    assert metrics["log10_amplitude_p50_ratio"] is not None
    assert metrics["log10_amplitude_p50_ratio"] < -1.0
    assert metrics["dominant_morphology_gap"] == "amplitude_under"


def test_compute_morphology_metrics_detects_derivative_over() -> None:
    exp09 = _load_exp09_module()
    rng = np.random.default_rng(7)
    wl = np.linspace(900.0, 1700.0, 128)
    # Real spectra: smooth low-frequency curve plus tiny noise.
    smooth = np.sin(np.linspace(0.0, 2 * np.pi, wl.size))
    real = smooth[None, :] + rng.normal(0.0, 0.001, size=(40, wl.size))
    # Synthetic: same smooth curve plus high-frequency noise (large derivative std)
    # but matched amplitude/variance scale to keep derivative dominant.
    hf = rng.normal(0.0, 0.5, size=(40, wl.size))
    synth = smooth[None, :] + hf - hf.mean()

    metrics = exp09.compute_morphology_metrics(real, synth, wl)
    assert metrics["log10_derivative_std_p50_ratio"] is not None
    assert metrics["log10_derivative_std_p50_ratio"] > 1.0
    # The derivative gap must dominate amplitude/variance/mean.
    candidates = {
        "log10_global_std_ratio": abs(metrics["log10_global_std_ratio"] or 0.0),
        "log10_amplitude_p50_ratio": abs(metrics["log10_amplitude_p50_ratio"] or 0.0),
        "log10_derivative_std_p50_ratio": abs(metrics["log10_derivative_std_p50_ratio"]),
    }
    assert max(candidates, key=lambda k: candidates[k]) == "log10_derivative_std_p50_ratio"
    assert metrics["dominant_morphology_gap"] == "derivative_over"


def test_compute_morphology_metrics_detects_mean_curve_inversion() -> None:
    exp09 = _load_exp09_module()
    wl = np.linspace(900.0, 1700.0, 64)
    rng = np.random.default_rng(3)
    base = np.linspace(-1.0, 1.0, wl.size)
    real = base[None, :] + rng.normal(0.0, 0.01, size=(40, wl.size))
    synth = -base[None, :] + rng.normal(0.0, 0.01, size=(40, wl.size))

    metrics = exp09.compute_morphology_metrics(real, synth, wl)
    assert metrics["mean_curve_corr"] is not None
    assert metrics["mean_curve_corr"] < 0.0
    assert metrics["inverted_mean_curve_corr"] is not None
    assert metrics["inverted_mean_curve_corr"] > 0.0
    assert metrics["dominant_morphology_gap"] == "mean_curve_inversion"


def test_mean_curve_inversion_does_not_override_larger_log_ratio_gap() -> None:
    """Inversion is a scored candidate; a much larger amplitude gap must win."""
    exp09 = _load_exp09_module()
    wl = np.linspace(900.0, 1700.0, 64)
    rng = np.random.default_rng(101)
    base = np.linspace(-1.0, 1.0, wl.size)
    # Real: large amplitude wavelength-dependent slope.
    real = base[None, :] * 10.0 + rng.normal(0.0, 0.005, size=(40, wl.size))
    # Synthetic: tiny inverted slope -> inversion magnitude ~1.0 but amplitude
    # gap is ~ log10(0.001 / 10) = -4 in log10, dominating inversion.
    synth = -base[None, :] * 0.001 + rng.normal(0.0, 0.0001, size=(40, wl.size))

    metrics = exp09.compute_morphology_metrics(real, synth, wl)
    assert metrics["mean_curve_corr"] is not None
    assert metrics["mean_curve_corr"] < 0.0  # inversion candidate present
    assert metrics["log10_amplitude_p50_ratio"] is not None
    assert abs(metrics["log10_amplitude_p50_ratio"]) > 1.0
    # Amplitude_under (large log-ratio magnitude) must dominate over inversion.
    assert metrics["dominant_morphology_gap"] == "amplitude_under"


def test_mean_curve_inversion_wins_when_it_is_the_largest_magnitude() -> None:
    exp09 = _load_exp09_module()
    wl = np.linspace(900.0, 1700.0, 64)
    rng = np.random.default_rng(11)
    base = np.linspace(-1.0, 1.0, wl.size)
    real = base[None, :] + rng.normal(0.0, 0.01, size=(40, wl.size))
    synth = -base[None, :] + rng.normal(0.0, 0.01, size=(40, wl.size))

    metrics = exp09.compute_morphology_metrics(real, synth, wl)
    assert metrics["mean_curve_corr"] is not None
    assert metrics["mean_curve_corr"] < -0.5
    # Inversion magnitude (~1.0) must dominate the small variance/amplitude/derivative ratios.
    assert metrics["dominant_morphology_gap"] == "mean_curve_inversion"


def test_compute_morphology_metrics_handles_zero_std_mean_curve() -> None:
    exp09 = _load_exp09_module()
    wl = np.linspace(900.0, 1700.0, 16)
    # Identical constant rows -> mean curve std is zero, corr undefined.
    real = np.zeros((20, wl.size), dtype=float)
    synth = np.zeros((20, wl.size), dtype=float)
    synth[:, :] = 0.1  # constant offset, still flat mean curve

    metrics = exp09.compute_morphology_metrics(real, synth, wl)
    assert metrics["mean_curve_corr"] is None
    assert metrics["inverted_mean_curve_corr"] is None
    # Dominant gap must still resolve without crashing.
    assert isinstance(metrics["dominant_morphology_gap"], str)


def test_morphology_row_to_dict_contains_required_columns() -> None:
    exp09 = _load_exp09_module()
    row = exp09.MorphologyRow(
        status="compared",
        source="AOM_regression",
        task="regression",
        dataset="DB/DS",
        synthetic_preset="grain",
        comparison_space="uncalibrated_raw",
        n_real_samples=16,
        n_synthetic_samples=16,
        n_wavelengths=64,
        wavelength_min=900.0,
        wavelength_max=1700.0,
        real_global_mean=0.5,
        synthetic_global_mean=0.55,
        global_mean_delta=0.05,
        real_global_std=0.1,
        synthetic_global_std=0.2,
        global_std_ratio=2.0,
        log10_global_std_ratio=0.301,
        real_amplitude_p50=0.4,
        synthetic_amplitude_p50=0.4,
        amplitude_p50_ratio=1.0,
        log10_amplitude_p50_ratio=0.0,
        real_derivative_std_p50=0.01,
        synthetic_derivative_std_p50=0.01,
        derivative_std_p50_ratio=1.0,
        log10_derivative_std_p50_ratio=0.0,
        mean_curve_corr=0.8,
        inverted_mean_curve_corr=-0.8,
        morphology_gap_score=0.5,
        dominant_morphology_gap="variance_over",
        **_audit_kwargs(),
        **_remediation_kwargs(),
        blocked_reason="",
    )
    data = row.to_dict()
    expected_subset = {
        "status",
        "source",
        "task",
        "dataset",
        "synthetic_preset",
        "comparison_space",
        "n_real_samples",
        "n_synthetic_samples",
        "n_wavelengths",
        "wavelength_min",
        "wavelength_max",
        "real_global_mean",
        "synthetic_global_mean",
        "global_mean_delta",
        "real_global_std",
        "synthetic_global_std",
        "global_std_ratio",
        "log10_global_std_ratio",
        "real_amplitude_p50",
        "synthetic_amplitude_p50",
        "amplitude_p50_ratio",
        "log10_amplitude_p50_ratio",
        "real_derivative_std_p50",
        "synthetic_derivative_std_p50",
        "derivative_std_p50_ratio",
        "log10_derivative_std_p50_ratio",
        "mean_curve_corr",
        "inverted_mean_curve_corr",
        "morphology_gap_score",
        "dominant_morphology_gap",
        "audit_oracle",
        "audit_label_inputs_used",
        "audit_target_inputs_used",
        "audit_split_inputs_used",
        "audit_source_oracle_used",
        "audit_learned",
        "audit_real_stat_capture",
        "audit_thresholds_modified",
        "audit_metrics_modified",
        "audit_imputed",
        "audit_replays_real_rows",
        "audit_scope",
        "remediation_profile",
        "effective_matrix_route",
        "r2c_remediation_enabled",
        "r2c_remediation_domain_key",
        "r2c_remediation_concentrations_applied",
        "r2c_remediation_spectra_applied",
        "r2c_remediation_spectra_rule",
        "r2c_remediation_composition_source",
        "r2c_remediation_spectra_source",
        "r2c_remediation_provenance_source",
        "r2c_remediation_route_variant",
        "r2c_remediation_constant_status",
        "r2c_remediation_readout_space",
        "r2c_remediation_calibration_source",
        "r2c_remediation_real_stat_source",
        "r2c_remediation_threshold_source",
        "blocked_reason",
    }
    assert expected_subset.issubset(set(data.keys()))


def test_render_markdown_disclaimer_strings_and_no_gate_claims(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    result = {
        "status": "done",
        "rows": [
            exp09.MorphologyRow(
                status="compared",
                source="AOM_regression",
                task="regression",
                dataset="DB/DS",
                synthetic_preset="grain",
                comparison_space="uncalibrated_raw",
                n_real_samples=16,
                n_synthetic_samples=16,
                n_wavelengths=64,
                wavelength_min=900.0,
                wavelength_max=1700.0,
                real_global_mean=0.5,
                synthetic_global_mean=0.55,
                global_mean_delta=0.05,
                real_global_std=0.1,
                synthetic_global_std=0.2,
                global_std_ratio=2.0,
                log10_global_std_ratio=0.301,
                real_amplitude_p50=0.4,
                synthetic_amplitude_p50=0.4,
                amplitude_p50_ratio=1.0,
                log10_amplitude_p50_ratio=0.0,
                real_derivative_std_p50=0.01,
                synthetic_derivative_std_p50=0.01,
                derivative_std_p50_ratio=1.0,
                log10_derivative_std_p50_ratio=0.0,
                mean_curve_corr=0.8,
                inverted_mean_curve_corr=-0.8,
                morphology_gap_score=0.5,
                dominant_morphology_gap="variance_over",
                **_audit_kwargs(),
                **_remediation_kwargs(),
                blocked_reason="",
            ),
        ],
        "real_runnable_count": 5,
        "real_sentinel_candidate_count": 3,
        "real_selected_count": 1,
        "sentinel_tokens": list(exp09.DEFAULT_SENTINEL_TOKENS),
        "remediation_profile": None,
    }
    md = exp09.render_markdown(
        result=result,
        report_path=tmp_path / "r2b.md",
        csv_path=tmp_path / "r2b.csv",
        n_synthetic_samples=16,
        max_real_samples=16,
        max_sentinel_datasets=1,
        seed=1234,
        sentinel_tokens=list(exp09.DEFAULT_SENTINEL_TOKENS),
    )

    assert "uncalibrated_raw" in md
    assert "report-only" in md.lower()
    assert "no calibration" in md.lower()
    assert "no PCA/covariance" in md
    assert "no ML/DL" in md
    assert "B2/B3/B4/B5" in md
    # Must NOT claim a B2 / B3 pass.
    assert "B2 PASS" not in md.upper().replace(":", "")
    assert "B3 PASS" not in md.upper().replace(":", "")
    assert "real_synthetic_scorecards" not in md
    assert "adversarial_auc.md" not in md
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


def test_default_report_paths_do_not_collide_with_existing_gate_reports() -> None:
    exp09 = _load_exp09_module()
    forbidden = {
        "real_synthetic_scorecards.md",
        "real_synthetic_scorecards.csv",
        "adversarial_auc.md",
        "adversarial_auc.csv",
        "transfer_validation.md",
        "transfer_validation.csv",
        "minimal_ablation_attribution.md",
        "minimal_ablation_attribution.csv",
        "encoder_tabpfn_gate.md",
        "encoder_tabpfn_gate.csv",
        "nirs_icl_gate_precheck.md",
        "nirs_icl_gate_precheck.csv",
        "integration_gate_status.md",
        "r2a_mechanistic_sentinel_ablation.md",
        "r2a_mechanistic_sentinel_ablation.csv",
    }
    assert exp09.DEFAULT_REPORT.name not in forbidden
    assert exp09.DEFAULT_CSV.name not in forbidden
    assert exp09.DEFAULT_REPORT.name.startswith("r2b_")
    assert exp09.DEFAULT_CSV.name.startswith("r2b_")


def test_exp09_source_does_not_import_forbidden_symbols() -> None:
    """The audit must not import sklearn, calibration, PCA, adversarial-AUC, or the
    gate-experiment modules ``exp02_real_synthetic_scorecards`` /
    ``exp08_mechanistic_sentinel_ablation`` (which transitively pull
    calibration/PCA/AUC into the audit's namespace at module load).
    """
    source = _exp09_source_path().read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_top_modules = {
        "sklearn",
        "exp02_real_synthetic_scorecards",
        "exp08_mechanistic_sentinel_ablation",
    }
    forbidden_names = {
        "compare_real_synthetic",
        "apply_covariance_calibration",
        "fit_real_marginal_calibration",
        "apply_real_marginal_calibration",
        "compute_pca_overlap",
        "compute_adversarial_auc",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top not in forbidden_top_modules, f"forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            module_full = node.module or ""
            module_top = module_full.split(".")[0]
            assert module_top not in forbidden_top_modules, (
                f"forbidden import-from: {module_full}"
            )
            for alias in node.names:
                assert alias.name not in forbidden_names, (
                    f"forbidden imported name: {alias.name} from {module_full}"
                )


def test_exp09_source_remains_ascii() -> None:
    source = _exp09_source_path().read_text(encoding="utf-8")

    assert source.isascii()


def test_importing_exp09_does_not_load_exp02_or_exp08_modules() -> None:
    """Subprocess isolation check: loading exp09 must not pull exp02 or exp08
    (which would transitively bring calibration/PCA/AUC names into the audit's
    namespace). sklearn itself is loaded upstream by ``nirs4all`` and cannot be
    avoided when generating synthetic spectra; this audit's contract therefore
    targets the gate-experiment modules and the audit-relevant calibration/AUC
    helpers, not the entire sklearn module surface.
    """
    import subprocess

    src_path = (
        Path(__file__).resolve().parents[1] / "src"
    )
    experiments_path = Path(__file__).resolve().parents[1] / "experiments"
    code = (
        "import sys, importlib.util\n"
        f"sys.path.insert(0, {str(experiments_path)!r})\n"
        f"sys.path.insert(0, {str(src_path)!r})\n"
        f"path = {str(experiments_path / 'exp09_sentinel_morphology_audit.py')!r}\n"
        "spec = importlib.util.spec_from_file_location('exp09', path)\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules['exp09'] = module\n"
        "spec.loader.exec_module(module)\n"
        "loaded = set(sys.modules)\n"
        "forbidden = {\n"
        "    'exp02_real_synthetic_scorecards',\n"
        "    'exp08_mechanistic_sentinel_ablation',\n"
        "}\n"
        "intersection = loaded & forbidden\n"
        "assert not intersection, f'forbidden modules loaded: {sorted(intersection)}'\n"
        "# Calibration/PCA/AUC helpers must not be reachable as audit names.\n"
        "for forbidden_name in (\n"
        "    'compare_real_synthetic',\n"
        "    'apply_real_marginal_calibration',\n"
        "    'fit_real_marginal_calibration',\n"
        "    'apply_covariance_calibration',\n"
        "    'compute_pca_overlap',\n"
        "    'compute_adversarial_auc',\n"
        "):\n"
        "    assert not hasattr(module, forbidden_name), (\n"
        "        f'audit namespace exposes forbidden helper: {forbidden_name}'\n"
        "    )\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"isolation check failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert result.stdout.strip().endswith("OK")


def test_exp09_source_does_not_call_forbidden_helpers() -> None:
    """Lightweight identifier scan ensures no forbidden helper is invoked in code."""
    source = _exp09_source_path().read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_calls = {
        "compare_real_synthetic",
        "apply_covariance_calibration",
        "fit_real_marginal_calibration",
        "apply_real_marginal_calibration",
        "compute_pca_overlap",
        "compute_adversarial_auc",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name: str | None = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name is not None:
                assert name not in forbidden_calls, f"forbidden call: {name}"


def test_run_audit_zero_cap_keeps_all_runnable_without_token_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exp09 = _load_exp09_module()
    _write_empty_cohorts(tmp_path)

    from nirsyntheticpfn.evaluation.realism import RealDataset

    candidates = [
        RealDataset(
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
        for name in ("random_db", "other_db", "beer_db")
    ]

    monkeypatch.setattr(
        exp09,
        "discover_local_real_datasets",
        lambda root: (list(candidates), []),
    )
    monkeypatch.setattr(
        exp09,
        "load_real_spectra",
        lambda dataset, root: (_ for _ in ()).throw(
            RuntimeError("synthetic-test: no real spectra available")
        ),
    )

    result = exp09.run_audit(
        root=tmp_path,
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=0,
        seed=1,
        sentinel_tokens=["BEER"],
    )

    assert result["real_runnable_count"] == 3
    assert result["real_sentinel_candidate_count"] == 3
    assert result["real_selected_count"] == 3
    # Every selected row falls through the blocked path because load_real_spectra raises.
    assert all(row.status == "blocked" for row in result["rows"])
    for row in result["rows"]:
        assert row.audit_oracle is False
        assert row.audit_real_stat_capture is False
        assert row.comparison_space == exp09.COMPARISON_SPACE


def test_run_audit_positive_cap_filters_by_tokens_then_truncates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exp09 = _load_exp09_module()
    _write_empty_cohorts(tmp_path)

    from nirsyntheticpfn.evaluation.realism import RealDataset

    candidates = [
        RealDataset(
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
        for name in ("random_db", "beer_db", "corn_db", "diesel_db")
    ]

    monkeypatch.setattr(
        exp09,
        "discover_local_real_datasets",
        lambda root: (list(candidates), []),
    )
    monkeypatch.setattr(
        exp09,
        "load_real_spectra",
        lambda dataset, root: (_ for _ in ()).throw(
            RuntimeError("synthetic-test: no real spectra available")
        ),
    )

    result = exp09.run_audit(
        root=tmp_path,
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=2,
        seed=1,
        sentinel_tokens=["BEER", "CORN", "DIESEL"],
    )

    assert result["real_runnable_count"] == 4
    assert result["real_sentinel_candidate_count"] == 3
    assert result["real_selected_count"] == 2


def test_run_audit_default_remediation_profile_is_none(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    _write_empty_cohorts(tmp_path)

    result = exp09.run_audit(
        root=tmp_path,
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=2,
        seed=1,
    )

    assert result["remediation_profile"] is None


def test_run_audit_rejects_unknown_remediation_profile(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    _write_empty_cohorts(tmp_path)

    with pytest.raises(ValueError):
        exp09.run_audit(
            root=tmp_path,
            n_synthetic_samples=4,
            max_real_samples=4,
            max_sentinel_datasets=2,
            seed=1,
            remediation_profile="not_a_real_profile",
        )


def test_run_audit_propagates_remediation_profile_into_blocked_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exp09 = _load_exp09_module()
    _write_empty_cohorts(tmp_path)

    from nirsyntheticpfn.evaluation.realism import RealDataset

    candidates = [
        RealDataset(
            source="AOM_regression",
            task="regression",
            database_name="diesel_db",
            dataset="ds",
            train_path="train.csv",
            test_path="test.csv",
            ytrain_path="ytrain.csv",
            ytest_path="ytest.csv",
            n_train_declared=10,
            n_test_declared=4,
            p_declared=8,
        )
    ]

    monkeypatch.setattr(
        exp09,
        "discover_local_real_datasets",
        lambda root: (list(candidates), []),
    )
    monkeypatch.setattr(
        exp09,
        "load_real_spectra",
        lambda dataset, root: (_ for _ in ()).throw(
            RuntimeError("synthetic-test: no real spectra available")
        ),
    )

    result = exp09.run_audit(
        root=tmp_path,
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=2,
        seed=1,
        sentinel_tokens=["DIESEL"],
        remediation_profile="r2c_sentinel_matrix_v1",
    )

    assert result["remediation_profile"] == "r2c_sentinel_matrix_v1"
    assert all(row.status == "blocked" for row in result["rows"])
    for row in result["rows"]:
        # Even on the blocked path the remediation profile must be recorded so
        # CSV/MD downstream can audit which run produced these rows.
        assert row.remediation_profile == "r2c_sentinel_matrix_v1"
        assert row.r2c_remediation_enabled is False  # build was never reached
        assert row.r2c_remediation_concentrations_applied is False
        assert row.r2c_remediation_spectra_applied is False
        # Audit hygiene unchanged.
        assert row.audit_oracle is False
        assert row.audit_real_stat_capture is False


def test_r2c_token_overrides_match_diesel_dataset() -> None:
    exp09 = _load_exp09_module()
    from nirsyntheticpfn.evaluation.realism import RealDataset

    diesel_dataset = RealDataset(
        source="AOM_regression",
        task="regression",
        database_name="diesel_db",
        dataset="ds",
        train_path="train.csv",
        test_path="test.csv",
        ytrain_path="ytrain.csv",
        ytest_path="ytest.csv",
        n_train_declared=10,
        n_test_declared=4,
        p_declared=8,
    )

    overrides = exp09._r2c_token_source_overrides(diesel_dataset)
    assert overrides["matrix_type"] == "liquid"
    assert overrides["measurement_mode"] == "transmittance"
    assert "diesel" in overrides["components"]


def test_r2c_token_overrides_match_milk_dataset() -> None:
    exp09 = _load_exp09_module()
    from nirsyntheticpfn.evaluation.realism import RealDataset

    milk_dataset = RealDataset(
        source="AOM_regression",
        task="regression",
        database_name="milk_db",
        dataset="ds",
        train_path="train.csv",
        test_path="test.csv",
        ytrain_path="ytrain.csv",
        ytest_path="ytest.csv",
        n_train_declared=10,
        n_test_declared=4,
        p_declared=8,
    )

    overrides = exp09._r2c_token_source_overrides(milk_dataset)
    assert overrides["matrix_type"] == "emulsion"
    assert overrides["measurement_mode"] == "transflectance"
    assert "water" in overrides["components"]


def test_r2c_token_overrides_empty_for_unrelated_dataset() -> None:
    exp09 = _load_exp09_module()
    from nirsyntheticpfn.evaluation.realism import RealDataset

    other_dataset = RealDataset(
        source="AOM_regression",
        task="regression",
        database_name="random_db",
        dataset="ds",
        train_path="train.csv",
        test_path="test.csv",
        ytrain_path="ytrain.csv",
        ytest_path="ytest.csv",
        n_train_declared=10,
        n_test_declared=4,
        p_declared=8,
    )

    assert exp09._r2c_token_source_overrides(other_dataset) == {}


def test_render_markdown_emits_remediation_profile_in_command_and_table(
    tmp_path: Path,
) -> None:
    exp09 = _load_exp09_module()
    result = {
        "status": "done",
        "rows": [
            exp09.MorphologyRow(
                status="compared",
                source="AOM_regression",
                task="regression",
                dataset="diesel_db/ds",
                synthetic_preset="fuel",
                comparison_space="uncalibrated_raw",
                n_real_samples=16,
                n_synthetic_samples=16,
                n_wavelengths=64,
                wavelength_min=900.0,
                wavelength_max=1700.0,
                real_global_mean=0.5,
                synthetic_global_mean=0.55,
                global_mean_delta=0.05,
                real_global_std=0.1,
                synthetic_global_std=0.2,
                global_std_ratio=2.0,
                log10_global_std_ratio=0.301,
                real_amplitude_p50=0.4,
                synthetic_amplitude_p50=0.4,
                amplitude_p50_ratio=1.0,
                log10_amplitude_p50_ratio=0.0,
                real_derivative_std_p50=0.01,
                synthetic_derivative_std_p50=0.01,
                derivative_std_p50_ratio=1.0,
                log10_derivative_std_p50_ratio=0.0,
                mean_curve_corr=0.8,
                inverted_mean_curve_corr=-0.8,
                morphology_gap_score=0.5,
                dominant_morphology_gap="variance_over",
                **_audit_kwargs(),
                **_remediation_kwargs(
                    profile="r2c_sentinel_matrix_v1",
                    enabled=True,
                    domain_key="petrochem_fuels",
                    concentrations_applied=True,
                    spectra_applied=True,
                    spectra_rule="short_liquid_optical_path_scale",
                ),
                blocked_reason="",
            ),
        ],
        "real_runnable_count": 5,
        "real_sentinel_candidate_count": 3,
        "real_selected_count": 1,
        "sentinel_tokens": list(exp09.DEFAULT_SENTINEL_TOKENS),
        "remediation_profile": "r2c_sentinel_matrix_v1",
    }

    md = exp09.render_markdown(
        result=result,
        report_path=tmp_path / "r2b.md",
        csv_path=tmp_path / "r2b.csv",
        n_synthetic_samples=16,
        max_real_samples=16,
        max_sentinel_datasets=1,
        seed=1234,
        sentinel_tokens=list(exp09.DEFAULT_SENTINEL_TOKENS),
        remediation_profile="r2c_sentinel_matrix_v1",
    )

    assert "--remediation-profile r2c_sentinel_matrix_v1" in md
    assert "Remediation profile: `r2c_sentinel_matrix_v1`" in md
    assert "`short_liquid_optical_path_scale`" in md
    assert "uncalibrated_raw" in md
    # The audit must NOT spuriously turn on calibration / PCA / AUC anywhere.
    assert "no calibration" in md.lower()
    assert "no PCA/covariance" in md
    assert "no ML/DL" in md
    # Diagnostic remediation column present.
    assert "remediation" in md.lower()


def test_remediation_does_not_introduce_new_helper_calls() -> None:
    """Re-assert that wiring --remediation-profile keeps the audit's call surface clean.

    The existing ``test_exp09_source_does_not_call_forbidden_helpers`` covers
    the global contract; this test focuses on the remediation-specific code
    region (functions whose names contain ``r2c`` or ``remediation``).
    """
    source = _exp09_source_path().read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_calls = {
        "compare_real_synthetic",
        "apply_covariance_calibration",
        "fit_real_marginal_calibration",
        "apply_real_marginal_calibration",
        "compute_pca_overlap",
        "compute_adversarial_auc",
    }
    remediation_funcs: list[ast.AST] = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and (
            "r2c" in node.name.lower()
            or "r2d" in node.name.lower()
            or "remediation" in node.name.lower()
        )
    ]
    assert remediation_funcs, "expected at least one remediation-specific function"
    for func in remediation_funcs:
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                target = node.func
                name: str | None = None
                if isinstance(target, ast.Name):
                    name = target.id
                elif isinstance(target, ast.Attribute):
                    name = target.attr
                if name is not None:
                    assert name not in forbidden_calls, (
                        f"forbidden call {name!r} inside remediation function "
                        f"{getattr(func, 'name', '?')}"
                    )


# ---------------------------------------------------------------------------
# R2d profile wiring (opt-in, additive over R2c).
# ---------------------------------------------------------------------------


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


def _make_real_like_fruitpuree_dataset() -> Any:
    from nirsyntheticpfn.evaluation.realism import RealDataset

    row_id = "FruitPuree/Strawberry2C_983_Holland_Acc94.3"
    return RealDataset(
        source=f"AOM_regression {row_id}",
        task=f"regression {row_id}",
        database_name=row_id,
        dataset=row_id,
        train_path="train.csv",
        test_path="test.csv",
        ytrain_path="ytrain.csv",
        ytest_path="ytest.csv",
        n_train_declared=10,
        n_test_declared=4,
        p_declared=8,
    )


def test_exp09_exposes_r2d_profile_constant() -> None:
    exp09 = _load_exp09_module()
    assert "r2d_sentinel_matrix_v1" in exp09.R2D_REMEDIATION_PROFILES
    assert "r2d_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES
    assert "r2c_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES


def test_run_audit_accepts_r2d_profile(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    _write_empty_cohorts(tmp_path)

    result = exp09.run_audit(
        root=tmp_path,
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=2,
        seed=1,
        remediation_profile="r2d_sentinel_matrix_v1",
    )
    assert result["remediation_profile"] == "r2d_sentinel_matrix_v1"


def test_r2d_token_overrides_match_beer_dataset() -> None:
    exp09 = _load_exp09_module()
    overrides = exp09._r2d_token_source_overrides(_make_dataset("beer_db"))
    assert overrides["matrix_type"] == "liquid"
    assert overrides["measurement_mode"] == "transmittance"
    assert "ethanol" in overrides["components"]
    assert "water" in overrides["components"]


def test_r2d_token_overrides_match_corn_dataset() -> None:
    exp09 = _load_exp09_module()
    overrides = exp09._r2d_token_source_overrides(_make_dataset("corn_db"))
    assert overrides["matrix_type"] == "powder"
    assert overrides["measurement_mode"] == "reflectance"
    assert "starch" in overrides["components"]
    assert "protein" in overrides["components"]


def test_r2d_token_overrides_inherit_diesel_and_milk_from_r2c() -> None:
    """R2d must continue to apply DIESEL/MILK overrides byte-for-byte from R2c."""
    exp09 = _load_exp09_module()
    diesel_r2c = exp09._r2c_token_source_overrides(_make_dataset("diesel_db"))
    diesel_r2d = exp09._r2d_token_source_overrides(_make_dataset("diesel_db"))
    assert diesel_r2c == diesel_r2d
    milk_r2c = exp09._r2c_token_source_overrides(_make_dataset("milk_db"))
    milk_r2d = exp09._r2d_token_source_overrides(_make_dataset("milk_db"))
    assert milk_r2c == milk_r2d


def test_r2d_token_overrides_empty_for_unrelated_dataset() -> None:
    exp09 = _load_exp09_module()
    assert exp09._r2d_token_source_overrides(_make_dataset("random_db")) == {}


def test_r2c_token_overrides_do_not_match_beer_or_corn_under_r2c_profile() -> None:
    """Stability guard: R2c must NOT pick up BEER/CORN overrides."""
    exp09 = _load_exp09_module()
    assert exp09._r2c_token_source_overrides(_make_dataset("beer_db")) == {}
    assert exp09._r2c_token_source_overrides(_make_dataset("corn_db")) == {}


def test_remediation_token_dispatcher_returns_r2d_overrides_for_r2d_profile() -> None:
    exp09 = _load_exp09_module()
    beer = _make_dataset("beer_db")
    overrides = exp09._remediation_token_source_overrides(beer, "r2d_sentinel_matrix_v1")
    assert overrides["matrix_type"] == "liquid"
    # Same dataset under R2c profile yields no override (BEER not in R2c rules).
    assert (
        exp09._remediation_token_source_overrides(beer, "r2c_sentinel_matrix_v1") == {}
    )


def test_render_markdown_emits_r2d_profile_in_command(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    result = {
        "status": "blocked_no_real_data",
        "rows": [],
        "real_runnable_count": 0,
        "real_sentinel_candidate_count": 0,
        "real_selected_count": 0,
        "sentinel_tokens": list(exp09.DEFAULT_SENTINEL_TOKENS),
        "remediation_profile": "r2d_sentinel_matrix_v1",
    }
    md = exp09.render_markdown(
        result=result,
        report_path=tmp_path / "r2b.md",
        csv_path=tmp_path / "r2b.csv",
        n_synthetic_samples=16,
        max_real_samples=16,
        max_sentinel_datasets=1,
        seed=1234,
        sentinel_tokens=list(exp09.DEFAULT_SENTINEL_TOKENS),
        remediation_profile="r2d_sentinel_matrix_v1",
    )
    assert "--remediation-profile r2d_sentinel_matrix_v1" in md
    assert "Remediation profile: `r2d_sentinel_matrix_v1`" in md


# ---------------------------------------------------------------------------
# R2f profile wiring (opt-in, additive over R2d for berry/juice sentinels).
# ---------------------------------------------------------------------------


def test_exp09_exposes_r2f_profile_constant() -> None:
    exp09 = _load_exp09_module()
    assert "r2f_sentinel_matrix_v1" in exp09.R2F_REMEDIATION_PROFILES
    assert "r2f_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES
    assert "r2d_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES


def test_run_audit_accepts_r2f_profile(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    _write_empty_cohorts(tmp_path)

    result = exp09.run_audit(
        root=tmp_path,
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=2,
        seed=1,
        remediation_profile="r2f_sentinel_matrix_v1",
    )
    assert result["remediation_profile"] == "r2f_sentinel_matrix_v1"


@pytest.mark.parametrize("name", ["berry_db", "juice_db"])
def test_r2f_token_overrides_match_berry_and_juice(name: str) -> None:
    exp09 = _load_exp09_module()
    overrides = exp09._r2f_token_source_overrides(_make_dataset(name))
    assert overrides["matrix_type"] == "liquid"
    assert overrides["measurement_mode"] == "transmittance"
    assert overrides["components"] == [
        "water",
        "glucose",
        "fructose",
        "sucrose",
        "citric_acid",
        "malic_acid",
        "carotenoid",
    ]
    for forbidden in ("pectin", "polyphenols", "anthocyanin", "tannins"):
        assert forbidden not in overrides["components"]


@pytest.mark.parametrize("name", ["BERRY", "juice_db"])
def test_r2f_token_overrides_match_real_sentinel_casing(name: str) -> None:
    exp09 = _load_exp09_module()
    overrides = exp09._r2f_token_source_overrides(_make_dataset(name))
    assert overrides["matrix_type"] == "liquid"
    assert overrides["measurement_mode"] == "transmittance"
    assert overrides["components"] == [
        "water",
        "glucose",
        "fructose",
        "sucrose",
        "citric_acid",
        "malic_acid",
        "carotenoid",
    ]


@pytest.mark.parametrize("name", ["fruitpuree_db", "puree_db", "FruitPuree"])
def test_r2f_token_overrides_do_not_match_puree_without_specific_rule(name: str) -> None:
    exp09 = _load_exp09_module()
    assert (
        exp09._remediation_token_source_overrides(
            _make_dataset(name),
            "r2f_sentinel_matrix_v1",
        )
        == {}
    )


def test_r2f_token_overrides_do_not_match_real_like_fruitpuree_strawberry() -> None:
    exp09 = _load_exp09_module()
    assert (
        exp09._remediation_token_source_overrides(
            _make_real_like_fruitpuree_dataset(),
            "r2f_sentinel_matrix_v1",
        )
        == {}
    )


def test_r2f_effective_profile_skips_real_like_fruitpuree_strawberry() -> None:
    exp09 = _load_exp09_module()
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_real_like_fruitpuree_dataset(),
            "r2f_sentinel_matrix_v1",
        )
        is None
    )


def test_r2f_build_keeps_real_like_fruitpuree_unremediated() -> None:
    exp09 = _load_exp09_module()
    dataset = _make_real_like_fruitpuree_dataset()
    run = exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=exp09.select_synthetic_preset_for_dataset(dataset),
        n_samples=4,
        seed=123,
        remediation_profile="r2f_sentinel_matrix_v1",
    )

    assert "r2c_mechanistic_remediation" not in run.metadata
    fields = exp09._remediation_fields_from_metadata(
        remediation_profile=exp09._row_remediation_profile(
            dataset,
            "r2f_sentinel_matrix_v1",
        ),
        metadata=run.metadata,
    )
    assert fields["remediation_profile"] is None
    assert fields["r2c_remediation_enabled"] is False
    assert fields["r2c_remediation_concentrations_applied"] is False
    assert fields["r2c_remediation_spectra_applied"] is False
    assert fields["r2c_remediation_spectra_rule"] is None
    assert fields["r2c_remediation_constant_status"] is None
    assert fields["r2c_remediation_readout_space"] is None
    assert fields["r2c_remediation_calibration_source"] is None
    assert fields["r2c_remediation_real_stat_source"] is None
    assert fields["r2c_remediation_threshold_source"] is None


def test_r2f_build_passes_none_to_builder_for_real_like_fruitpuree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exp09 = _load_exp09_module()
    dataset = _make_real_like_fruitpuree_dataset()
    captured: dict[str, Any] = {}

    def fake_builder(
        record: Any,
        *,
        n_samples: int,
        random_seed: int,
        remediation_profile: str | None,
    ) -> Any:
        captured["remediation_profile"] = remediation_profile
        return SimpleNamespace(metadata={})

    monkeypatch.setattr(exp09, "build_synthetic_dataset_run", fake_builder)

    exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=exp09.select_synthetic_preset_for_dataset(dataset),
        n_samples=4,
        seed=123,
        remediation_profile="r2f_sentinel_matrix_v1",
    )

    assert captured["remediation_profile"] is None


def test_run_audit_reports_effective_none_for_r2f_fruitpuree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exp09 = _load_exp09_module()
    _write_empty_cohorts(tmp_path)
    dataset = _make_real_like_fruitpuree_dataset()
    wl = np.linspace(900.0, 1700.0, 8)
    real = np.tile(np.linspace(0.1, 0.8, 8), (6, 1))
    synth = np.tile(np.linspace(0.2, 0.9, 8), (6, 1))

    monkeypatch.setattr(
        exp09,
        "discover_local_real_datasets",
        lambda root: ([dataset], []),
    )
    monkeypatch.setattr(exp09, "load_real_spectra", lambda dataset, root: (real, wl))
    monkeypatch.setattr(
        exp09,
        "_build_baseline_synthetic_run",
        lambda **kwargs: SimpleNamespace(X=synth, wavelengths=wl, metadata={}),
    )

    result = exp09.run_audit(
        root=tmp_path,
        n_synthetic_samples=6,
        max_real_samples=6,
        max_sentinel_datasets=1,
        seed=123,
        sentinel_tokens=["FruitPuree"],
        remediation_profile="r2f_sentinel_matrix_v1",
    )

    assert result["remediation_profile"] == "r2f_sentinel_matrix_v1"
    assert result["status"] == "done"
    row = result["rows"][0]
    assert row.remediation_profile is None
    assert row.r2c_remediation_enabled is False
    assert row.r2c_remediation_concentrations_applied is False
    assert row.r2c_remediation_spectra_applied is False
    assert row.r2c_remediation_spectra_rule is None
    assert row.r2c_remediation_constant_status is None
    assert row.r2c_remediation_readout_space is None
    assert row.r2c_remediation_calibration_source is None
    assert row.r2c_remediation_real_stat_source is None
    assert row.r2c_remediation_threshold_source is None


def test_r2f_token_overrides_inherit_r2d_sentinel_overrides() -> None:
    exp09 = _load_exp09_module()
    for name in ("diesel_db", "milk_db", "beer_db", "corn_db"):
        assert exp09._r2f_token_source_overrides(
            _make_dataset(name)
        ) == exp09._r2d_token_source_overrides(_make_dataset(name))


def test_r2f_puree_exclusion_does_not_disable_r2d_inheritance() -> None:
    exp09 = _load_exp09_module()
    dataset = _make_dataset("beer_fruitpuree_db")
    assert exp09._r2f_token_source_overrides(
        dataset
    ) == exp09._r2d_token_source_overrides(dataset)


def test_r2f_effective_profile_keeps_r2d_inheritance_for_puree_tokens() -> None:
    exp09 = _load_exp09_module()
    dataset = _make_dataset("beer_fruitpuree_db")
    assert (
        exp09._effective_remediation_profile_for_dataset(
            dataset,
            "r2f_sentinel_matrix_v1",
        )
        == "r2f_sentinel_matrix_v1"
    )


def test_r2d_token_overrides_do_not_match_berry_puree_or_juice() -> None:
    exp09 = _load_exp09_module()
    for name in ("berry_db", "BERRY", "fruitpuree_db", "FruitPuree", "juice_db"):
        assert exp09._r2d_token_source_overrides(_make_dataset(name)) == {}


def test_r2d_effective_profile_unchanged_for_berry_puree_and_juice() -> None:
    exp09 = _load_exp09_module()
    for name in ("berry_db", "BERRY", "fruitpuree_db", "FruitPuree", "juice_db"):
        dataset = _make_dataset(name)
        assert (
            exp09._effective_remediation_profile_for_dataset(
                dataset,
                "r2d_sentinel_matrix_v1",
            )
            == "r2d_sentinel_matrix_v1"
        )
        assert exp09._remediation_token_source_overrides(
            dataset,
            "r2d_sentinel_matrix_v1",
        ) == {}


@pytest.mark.parametrize("name", ["berry_db", "BERRY", "juice_db"])
def test_remediation_token_dispatcher_returns_r2f_overrides_for_r2f_profile(
    name: str,
) -> None:
    exp09 = _load_exp09_module()
    dataset = _make_dataset(name)
    overrides = exp09._remediation_token_source_overrides(
        dataset,
        "r2f_sentinel_matrix_v1",
    )
    assert overrides["matrix_type"] == "liquid"
    assert "carotenoid" in overrides["components"]
    assert (
        exp09._remediation_token_source_overrides(dataset, "r2d_sentinel_matrix_v1")
        == {}
    )


@pytest.mark.parametrize("name", ["BERRY", "juice_db"])
def test_r2f_effective_profile_and_build_stay_active_for_berry_and_juice(
    name: str,
) -> None:
    exp09 = _load_exp09_module()
    dataset = _make_dataset(name)
    assert (
        exp09._effective_remediation_profile_for_dataset(
            dataset,
            "r2f_sentinel_matrix_v1",
        )
        == "r2f_sentinel_matrix_v1"
    )

    run = exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=exp09.select_synthetic_preset_for_dataset(dataset),
        n_samples=4,
        seed=123,
        remediation_profile="r2f_sentinel_matrix_v1",
    )
    audit = run.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2f_sentinel_matrix_v1"
    assert audit["domain_key"] == "beverage_juice"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    assert "carotenoid" in run.prior_config["component_keys"]


def test_remediation_token_dispatcher_keeps_puree_unoverridden_for_r2f() -> None:
    exp09 = _load_exp09_module()
    puree = _make_dataset("fruitpuree_db")
    assert (
        exp09._remediation_token_source_overrides(puree, "r2f_sentinel_matrix_v1")
        == {}
    )


def test_render_markdown_emits_r2f_profile_in_command(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    result = {
        "status": "blocked_no_real_data",
        "rows": [],
        "real_runnable_count": 0,
        "real_sentinel_candidate_count": 0,
        "real_selected_count": 0,
        "sentinel_tokens": list(exp09.DEFAULT_SENTINEL_TOKENS),
        "remediation_profile": "r2f_sentinel_matrix_v1",
    }
    md = exp09.render_markdown(
        result=result,
        report_path=tmp_path / "r2b.md",
        csv_path=tmp_path / "r2b.csv",
        n_synthetic_samples=16,
        max_real_samples=16,
        max_sentinel_datasets=1,
        seed=1234,
        sentinel_tokens=list(exp09.DEFAULT_SENTINEL_TOKENS),
        remediation_profile="r2f_sentinel_matrix_v1",
    )
    assert "--remediation-profile r2f_sentinel_matrix_v1" in md
    assert "Remediation profile: `r2f_sentinel_matrix_v1`" in md


# ---------------------------------------------------------------------------
# R2g profile wiring (opt-in, additive over R2f for soil sentinels).
# ---------------------------------------------------------------------------


def test_exp09_exposes_r2g_profile_constant() -> None:
    exp09 = _load_exp09_module()
    assert "r2g_sentinel_matrix_v1" in exp09.R2G_REMEDIATION_PROFILES
    assert "r2g_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES
    assert "r2f_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES


def test_run_audit_accepts_r2g_profile(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    _write_empty_cohorts(tmp_path)

    result = exp09.run_audit(
        root=tmp_path,
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=2,
        seed=1,
        remediation_profile="r2g_sentinel_matrix_v1",
    )
    assert result["remediation_profile"] == "r2g_sentinel_matrix_v1"


@pytest.mark.parametrize("name", ["lucas_topsoil", "PHOSPHORUS_QUARTZ", "soil_db"])
def test_r2g_token_overrides_match_soil_sentinels(name: str) -> None:
    exp09 = _load_exp09_module()
    overrides = exp09._r2g_token_source_overrides(_make_dataset(name))
    assert overrides["matrix_type"] == "powder"
    assert overrides["measurement_mode"] == "reflectance"
    assert overrides["particle_size"] == 75.0
    assert overrides["components"] == [
        "moisture",
        "carbonates",
        "kaolinite",
        "gypsum",
        "cellulose",
        "lignin",
        "protein",
    ]


@pytest.mark.parametrize("name", ["lucas_juice", "PHOSPHORUS_BERRY", "soil_juice_db"])
def test_r2g_soil_token_overrides_take_priority_over_r2f_tokens(name: str) -> None:
    exp09 = _load_exp09_module()
    overrides = exp09._remediation_token_source_overrides(
        _make_dataset(name),
        "r2g_sentinel_matrix_v1",
    )

    assert overrides["matrix_type"] == "powder"
    assert overrides["measurement_mode"] == "reflectance"
    assert "kaolinite" in overrides["components"]
    assert "carotenoid" not in overrides["components"]


@pytest.mark.parametrize("name", ["lucas_topsoil", "PHOSPHORUS_QUARTZ", "soil_db"])
def test_r2g_effective_profile_stays_r2g_for_soil_sentinels(name: str) -> None:
    exp09 = _load_exp09_module()
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset(name),
            "r2g_sentinel_matrix_v1",
        )
        == "r2g_sentinel_matrix_v1"
    )


@pytest.mark.parametrize("name", ["beer_db", "corn_db", "diesel_db", "milk_db", "BERRY"])
def test_r2g_effective_profile_falls_back_to_r2f_for_non_soil_sentinels(
    name: str,
) -> None:
    exp09 = _load_exp09_module()
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset(name),
            "r2g_sentinel_matrix_v1",
        )
        == "r2f_sentinel_matrix_v1"
    )


def test_r2g_soil_token_overrides_use_valid_components() -> None:
    exp09 = _load_exp09_module()
    from nirs4all.synthesis.domains import get_domain_config

    overrides = exp09._r2g_token_source_overrides(_make_dataset("LUCAS_SOIL"))
    valid_components = set(get_domain_config("environmental_soil").typical_components)
    assert set(overrides["components"]).issubset(valid_components)


def test_r2g_token_overrides_inherit_r2f_sentinel_overrides() -> None:
    exp09 = _load_exp09_module()
    for name in ("diesel_db", "milk_db", "beer_db", "corn_db", "BERRY", "juice_db"):
        assert exp09._r2g_token_source_overrides(
            _make_dataset(name)
        ) == exp09._r2f_token_source_overrides(_make_dataset(name))


def test_r2f_and_r2d_token_routing_do_not_pick_up_soil_remediation() -> None:
    exp09 = _load_exp09_module()
    for profile in ("r2d_sentinel_matrix_v1", "r2f_sentinel_matrix_v1"):
        assert (
            exp09._remediation_token_source_overrides(
                _make_dataset("lucas_topsoil"),
                profile,
            )
            == {}
        )


def test_remediation_token_dispatcher_returns_r2g_overrides_for_r2g_profile() -> None:
    exp09 = _load_exp09_module()
    dataset = _make_dataset("PHOSPHORUS_SOIL")
    overrides = exp09._remediation_token_source_overrides(
        dataset,
        "r2g_sentinel_matrix_v1",
    )
    assert overrides["matrix_type"] == "powder"
    assert "kaolinite" in overrides["components"]
    assert (
        exp09._remediation_token_source_overrides(dataset, "r2f_sentinel_matrix_v1")
        == {}
    )


def test_r2g_effective_profile_keeps_real_like_fruitpuree_skipped() -> None:
    exp09 = _load_exp09_module()
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_real_like_fruitpuree_dataset(),
            "r2g_sentinel_matrix_v1",
        )
        is None
    )


def test_r2g_build_uses_effective_profile_for_builder_and_token_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exp09 = _load_exp09_module()
    original_preset_source = exp09._preset_source
    captured_profiles: list[str | None] = []
    captured_overrides: dict[str, dict[str, object] | None] = {}

    def fake_preset_source(
        preset: str,
        *,
        seed: int,
        extra_overrides: dict[str, object] | None = None,
    ) -> dict[str, object]:
        captured_overrides[preset] = extra_overrides
        return cast(
            "dict[str, object]",
            original_preset_source(
                preset,
                seed=seed,
                extra_overrides=extra_overrides,
            ),
        )

    def fake_builder(
        record: Any,
        *,
        n_samples: int,
        random_seed: int,
        remediation_profile: str | None,
    ) -> Any:
        captured_profiles.append(remediation_profile)
        return SimpleNamespace(metadata={})

    monkeypatch.setattr(exp09, "_preset_source", fake_preset_source)
    monkeypatch.setattr(exp09, "build_synthetic_dataset_run", fake_builder)

    for dataset in (
        _make_dataset("LUCAS_SOIL"),
        _make_dataset("beer_db"),
        _make_real_like_fruitpuree_dataset(),
    ):
        exp09._build_baseline_synthetic_run(
            dataset=dataset,
            preset=exp09.select_synthetic_preset_for_dataset(dataset),
            n_samples=4,
            seed=123,
            remediation_profile="r2g_sentinel_matrix_v1",
        )

    assert captured_profiles == [
        "r2g_sentinel_matrix_v1",
        "r2f_sentinel_matrix_v1",
        None,
    ]
    soil_override = captured_overrides["soil"]
    assert soil_override is not None
    soil_components = cast("list[str]", soil_override["components"])
    assert "kaolinite" in soil_components
    wine_override = captured_overrides["wine"]
    assert wine_override is not None
    wine_components = cast("list[str]", wine_override["components"])
    assert "ethanol" in wine_components
    assert "kaolinite" not in wine_components
    assert captured_overrides["juice"] is None


def test_r2g_row_remediation_profile_reports_effective_profiles() -> None:
    exp09 = _load_exp09_module()
    assert (
        exp09._row_remediation_profile(
            _make_dataset("LUCAS_SOIL"),
            "r2g_sentinel_matrix_v1",
        )
        == "r2g_sentinel_matrix_v1"
    )
    assert (
        exp09._row_remediation_profile(
            _make_dataset("beer_db"),
            "r2g_sentinel_matrix_v1",
        )
        == "r2f_sentinel_matrix_v1"
    )
    assert (
        exp09._row_remediation_profile(
            _make_real_like_fruitpuree_dataset(),
            "r2g_sentinel_matrix_v1",
        )
        is None
    )


def test_r2g_build_passes_none_to_builder_for_real_like_fruitpuree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exp09 = _load_exp09_module()
    dataset = _make_real_like_fruitpuree_dataset()
    captured: dict[str, Any] = {}

    def fake_builder(
        record: Any,
        *,
        n_samples: int,
        random_seed: int,
        remediation_profile: str | None,
    ) -> Any:
        captured["remediation_profile"] = remediation_profile
        return SimpleNamespace(metadata={})

    monkeypatch.setattr(exp09, "build_synthetic_dataset_run", fake_builder)

    exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=exp09.select_synthetic_preset_for_dataset(dataset),
        n_samples=4,
        seed=123,
        remediation_profile="r2g_sentinel_matrix_v1",
    )

    assert captured["remediation_profile"] is None


def test_r2f_requested_effective_profile_behavior_is_unchanged() -> None:
    exp09 = _load_exp09_module()
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("LUCAS_SOIL"),
            "r2f_sentinel_matrix_v1",
        )
        == "r2f_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("beer_fruitpuree_db"),
            "r2f_sentinel_matrix_v1",
        )
        == "r2f_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_real_like_fruitpuree_dataset(),
            "r2f_sentinel_matrix_v1",
        )
        is None
    )


def test_r2g_soil_build_records_non_oracle_audit() -> None:
    exp09 = _load_exp09_module()
    run = exp09._build_baseline_synthetic_run(
        dataset=_make_dataset("LUCAS_SOIL"),
        preset="soil",
        n_samples=8,
        seed=123,
        remediation_profile="r2g_sentinel_matrix_v1",
    )

    audit = run.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2g_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2g_sentinel_matrix_remediation"
    assert audit["domain_key"] == "environmental_soil"
    assert audit["real_stat_capture"] is False
    assert audit["thresholds_modified"] is False
    assert audit["metrics_modified"] is False
    forbidden_keys = {
        "real_spectra",
        "real_labels",
        "real_targets",
        "real_splits",
        "auc",
        "metric_overrides",
        "threshold_overrides",
    }
    assert not (set(audit) & forbidden_keys)
    assert not (set(audit["transform_params"]) & forbidden_keys)


def test_render_markdown_emits_r2g_profile_in_command(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    result = {
        "status": "blocked_no_real_data",
        "rows": [],
        "real_runnable_count": 0,
        "real_sentinel_candidate_count": 0,
        "real_selected_count": 0,
        "sentinel_tokens": list(exp09.DEFAULT_SENTINEL_TOKENS),
        "remediation_profile": "r2g_sentinel_matrix_v1",
    }
    md = exp09.render_markdown(
        result=result,
        report_path=tmp_path / "r2b.md",
        csv_path=tmp_path / "r2b.csv",
        n_synthetic_samples=16,
        max_real_samples=16,
        max_sentinel_datasets=1,
        seed=1234,
        sentinel_tokens=list(exp09.DEFAULT_SENTINEL_TOKENS),
        remediation_profile="r2g_sentinel_matrix_v1",
    )
    assert "--remediation-profile r2g_sentinel_matrix_v1" in md
    assert "Remediation profile: `r2g_sentinel_matrix_v1`" in md


# ---------------------------------------------------------------------------
# R2h profile wiring (opt-in BERRY readout, R2g fallback for non-BERRY).
# ---------------------------------------------------------------------------


def test_exp09_exposes_r2h_profile_constant() -> None:
    exp09 = _load_exp09_module()
    assert "r2h_sentinel_matrix_v1" in exp09.R2H_REMEDIATION_PROFILES
    assert "r2h_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES
    assert "r2g_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES


def test_run_audit_accepts_r2h_profile(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    _write_empty_cohorts(tmp_path)

    result = exp09.run_audit(
        root=tmp_path,
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=2,
        seed=1,
        remediation_profile="r2h_sentinel_matrix_v1",
    )
    assert result["remediation_profile"] == "r2h_sentinel_matrix_v1"


def test_run_audit_marks_r2h_rows_with_r2h_audit_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exp09 = _load_exp09_module()
    dataset = _make_dataset("BERRY")
    monkeypatch.setattr(
        exp09,
        "discover_local_real_datasets",
        lambda root: ([dataset], []),
    )
    monkeypatch.setattr(
        exp09,
        "load_real_spectra",
        lambda dataset, root: (_ for _ in ()).throw(
            RuntimeError("synthetic-test: no real spectra available")
        ),
    )

    result = exp09.run_audit(
        root=tmp_path,
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=1,
        seed=1,
        remediation_profile="r2h_sentinel_matrix_v1",
    )

    assert [row.audit_scope for row in result["rows"]] == [
        "bench_only_r2h_sentinel_morphology_audit"
    ]


@pytest.mark.parametrize("name", ["BERRY", "berry_db"])
def test_r2h_token_overrides_match_berry_only(name: str) -> None:
    exp09 = _load_exp09_module()
    overrides = exp09._remediation_token_source_overrides(
        _make_dataset(name),
        "r2h_sentinel_matrix_v1",
    )
    assert overrides["matrix_type"] == "liquid"
    assert overrides["measurement_mode"] == "transmittance"
    assert overrides["particle_size"] == 10.0
    assert overrides["components"] == [
        "water",
        "glucose",
        "fructose",
        "sucrose",
        "citric_acid",
        "malic_acid",
        "carotenoid",
    ]


def test_r2h_effective_profile_routes_berry_to_r2h_and_nonberry_to_r2g() -> None:
    exp09 = _load_exp09_module()
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("BERRY"),
            "r2h_sentinel_matrix_v1",
        )
        == "r2h_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("LUCAS_SOIL"),
            "r2h_sentinel_matrix_v1",
        )
        == "r2g_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("beer_db"),
            "r2h_sentinel_matrix_v1",
        )
        == "r2f_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_real_like_fruitpuree_dataset(),
            "r2h_sentinel_matrix_v1",
        )
        is None
    )


def test_r2h_soil_marker_keeps_r2g_priority_over_berry_token() -> None:
    exp09 = _load_exp09_module()
    dataset = _make_dataset("PHOSPHORUS_BERRY")
    assert (
        exp09._effective_remediation_profile_for_dataset(
            dataset,
            "r2h_sentinel_matrix_v1",
        )
        == "r2g_sentinel_matrix_v1"
    )
    overrides = exp09._remediation_token_source_overrides(
        dataset,
        "r2h_sentinel_matrix_v1",
    )
    assert overrides["matrix_type"] == "powder"
    assert "kaolinite" in overrides["components"]
    assert "carotenoid" not in overrides["components"]


def test_r2h_build_records_non_oracle_berry_audit() -> None:
    exp09 = _load_exp09_module()
    run = exp09._build_baseline_synthetic_run(
        dataset=_make_dataset("BERRY"),
        preset="juice",
        n_samples=8,
        seed=123,
        remediation_profile="r2h_sentinel_matrix_v1",
    )

    audit = run.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2h_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2h_sentinel_matrix_remediation"
    assert audit["domain_key"] == "beverage_juice"
    assert audit["real_stat_capture"] is False
    assert audit["thresholds_modified"] is False
    assert audit["metrics_modified"] is False
    assert audit["transform_params"]["spectra_rule"] == (
        "cloudy_berry_percent_transmittance_readout"
    )
    assert audit["transform_params"]["constant_status"] == "fixed_mechanistic_prior"
    assert audit["transform_params"]["readout_space"] == (
        "apparent_percent_transmittance_intensity"
    )
    assert audit["transform_params"]["calibration_source"] == "none"
    assert audit["transform_params"]["real_stat_source"] == "none"
    assert audit["transform_params"]["threshold_source"] == "none"
    fields = exp09._remediation_fields_from_metadata(
        remediation_profile="r2h_sentinel_matrix_v1",
        metadata=run.metadata,
    )
    assert fields["r2c_remediation_spectra_rule"] == (
        "cloudy_berry_percent_transmittance_readout"
    )
    assert fields["r2c_remediation_constant_status"] == "fixed_mechanistic_prior"
    assert fields["r2c_remediation_readout_space"] == (
        "apparent_percent_transmittance_intensity"
    )
    assert fields["r2c_remediation_provenance_source"] is None
    assert fields["r2c_remediation_route_variant"] is None
    assert fields["r2c_remediation_calibration_source"] == "none"
    assert fields["r2c_remediation_real_stat_source"] == "none"
    assert fields["r2c_remediation_threshold_source"] == "none"
    for field in (
        "r2c_remediation_constant_status",
        "r2c_remediation_readout_space",
        "r2c_remediation_provenance_source",
        "r2c_remediation_route_variant",
        "r2c_remediation_calibration_source",
        "r2c_remediation_real_stat_source",
        "r2c_remediation_threshold_source",
    ):
        assert field in exp09._csv_fieldnames()


def test_render_markdown_emits_r2h_profile_in_command(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    result = {
        "status": "done",
        "rows": [
            exp09.MorphologyRow(
                status="compared",
                source="AOM_regression",
                task="regression",
                dataset="BERRY/brix",
                synthetic_preset="juice",
                comparison_space="uncalibrated_raw",
                n_real_samples=16,
                n_synthetic_samples=16,
                n_wavelengths=64,
                wavelength_min=900.0,
                wavelength_max=1100.0,
                real_global_mean=37.0,
                synthetic_global_mean=42.0,
                global_mean_delta=5.0,
                real_global_std=16.0,
                synthetic_global_std=12.0,
                global_std_ratio=0.75,
                log10_global_std_ratio=-0.125,
                real_amplitude_p50=2.0,
                synthetic_amplitude_p50=11.0,
                amplitude_p50_ratio=5.5,
                log10_amplitude_p50_ratio=0.740,
                real_derivative_std_p50=0.28,
                synthetic_derivative_std_p50=0.17,
                derivative_std_p50_ratio=0.61,
                log10_derivative_std_p50_ratio=-0.215,
                mean_curve_corr=0.68,
                inverted_mean_curve_corr=-0.68,
                morphology_gap_score=1.7,
                dominant_morphology_gap="amplitude_over",
                **_audit_kwargs(),
                **_remediation_kwargs(
                    profile="r2h_sentinel_matrix_v1",
                    enabled=True,
                    domain_key="beverage_juice",
                    concentrations_applied=True,
                    spectra_applied=True,
                    spectra_rule="cloudy_berry_percent_transmittance_readout",
                    constant_status="fixed_mechanistic_prior",
                    readout_space="apparent_percent_transmittance_intensity",
                    calibration_source="none",
                    real_stat_source="none",
                    threshold_source="none",
                ),
                blocked_reason="",
            )
        ],
        "real_runnable_count": 1,
        "real_sentinel_candidate_count": 1,
        "real_selected_count": 1,
        "sentinel_tokens": list(exp09.DEFAULT_SENTINEL_TOKENS),
        "remediation_profile": "r2h_sentinel_matrix_v1",
    }
    md = exp09.render_markdown(
        result=result,
        report_path=tmp_path / "r2b.md",
        csv_path=tmp_path / "r2b.csv",
        n_synthetic_samples=16,
        max_real_samples=16,
        max_sentinel_datasets=1,
        seed=1234,
        sentinel_tokens=list(exp09.DEFAULT_SENTINEL_TOKENS),
        remediation_profile="r2h_sentinel_matrix_v1",
    )
    assert "--remediation-profile r2h_sentinel_matrix_v1" in md
    assert "Remediation profile: `r2h_sentinel_matrix_v1`" in md
    assert "## R2h Constant Provenance" in md
    assert "`constant_status=fixed_mechanistic_prior`" in md
    assert "`readout_space=apparent_percent_transmittance_intensity`" in md
    assert "`cloudy_berry_percent_transmittance_readout`" in md
    assert "`fixed_mechanistic_prior`" in md
    assert "`apparent_percent_transmittance_intensity`" in md
    assert "| constant status | readout space | calibration source | real stat source | threshold source |" in md


def test_render_markdown_uses_r2h_scope_and_title_not_r2b(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    result = {
        "status": "blocked_no_real_data",
        "rows": [],
        "real_runnable_count": 0,
        "real_sentinel_candidate_count": 0,
        "real_selected_count": 0,
        "sentinel_tokens": list(exp09.DEFAULT_SENTINEL_TOKENS),
        "remediation_profile": "r2h_sentinel_matrix_v1",
    }

    md = exp09.render_markdown(
        result=result,
        report_path=tmp_path / "r2h.md",
        csv_path=tmp_path / "r2h.csv",
        n_synthetic_samples=16,
        max_real_samples=16,
        max_sentinel_datasets=1,
        seed=1234,
        sentinel_tokens=list(exp09.DEFAULT_SENTINEL_TOKENS),
        remediation_profile="r2h_sentinel_matrix_v1",
    )

    assert md.startswith("# R2h Sentinel Morphology Audit")
    assert "audit_scope=bench_only_r2h_sentinel_morphology_audit" in md
    assert "# R2b Sentinel Morphology Audit" not in md
    assert "audit_scope=bench_only_r2b_sentinel_morphology_audit" not in md


# ---------------------------------------------------------------------------
# R2i profile wiring (FruitPuree semi-solid, preserving R2h/R2g/R2f routes).
# ---------------------------------------------------------------------------


def test_exp09_exposes_r2i_profile_constant() -> None:
    exp09 = _load_exp09_module()
    assert "r2i_sentinel_matrix_v1" in exp09.R2I_REMEDIATION_PROFILES
    assert "r2i_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES
    assert "r2h_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES


def test_run_audit_accepts_r2i_profile(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    _write_empty_cohorts(tmp_path)

    result = exp09.run_audit(
        root=tmp_path,
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=2,
        seed=1,
        remediation_profile="r2i_sentinel_matrix_v1",
    )
    assert result["remediation_profile"] == "r2i_sentinel_matrix_v1"


def test_r2i_effective_profile_routes_fruitpuree_before_strawberry_berry_match() -> None:
    exp09 = _load_exp09_module()
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_real_like_fruitpuree_dataset(),
            "r2i_sentinel_matrix_v1",
        )
        == "r2i_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("BERRY"),
            "r2i_sentinel_matrix_v1",
        )
        == "r2h_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("LUCAS_SOIL"),
            "r2i_sentinel_matrix_v1",
        )
        == "r2g_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("beer_db"),
            "r2i_sentinel_matrix_v1",
        )
        == "r2f_sentinel_matrix_v1"
    )


def test_r2i_fruitpuree_override_is_paste_not_berry_readout() -> None:
    exp09 = _load_exp09_module()
    overrides = exp09._remediation_token_source_overrides(
        _make_real_like_fruitpuree_dataset(),
        "r2i_sentinel_matrix_v1",
    )

    assert overrides["domain"] == "fruit"
    assert overrides["matrix_type"] == "paste"
    assert overrides["measurement_mode"] == "transflectance"
    assert overrides["particle_size"] == 35.0
    assert "cellulose" in overrides["components"]
    assert "starch" in overrides["components"]


def test_r2i_build_records_fruitpuree_non_oracle_provenance() -> None:
    exp09 = _load_exp09_module()
    run = exp09._build_baseline_synthetic_run(
        dataset=_make_real_like_fruitpuree_dataset(),
        preset=exp09.select_synthetic_preset_for_dataset(
            _make_real_like_fruitpuree_dataset()
        ),
        n_samples=8,
        seed=123,
        remediation_profile="r2i_sentinel_matrix_v1",
    )

    audit = run.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2i_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2i_sentinel_matrix_remediation"
    assert audit["domain_key"] == "agriculture_fruit"
    assert audit["real_stat_capture"] is False
    assert audit["thresholds_modified"] is False
    params = audit["transform_params"]
    assert params["spectra_rule"] == (
        "semi_solid_fruit_puree_short_path_scatter_smoothing"
    )
    assert params["spectra_rule"] != "cloudy_berry_percent_transmittance_readout"
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "semi_solid_puree_raw_absorbance"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert "additive_baseline_range" in params


def test_r2h_fruitpuree_remains_unremediated_after_r2i_addition() -> None:
    exp09 = _load_exp09_module()
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_real_like_fruitpuree_dataset(),
            "r2h_sentinel_matrix_v1",
        )
        is None
    )


def test_render_markdown_emits_r2i_fruitpuree_provenance(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    result = {
        "status": "blocked_no_real_data",
        "rows": [],
        "real_runnable_count": 0,
        "real_sentinel_candidate_count": 0,
        "real_selected_count": 0,
        "sentinel_tokens": list(exp09.DEFAULT_SENTINEL_TOKENS),
        "remediation_profile": "r2i_sentinel_matrix_v1",
    }

    md = exp09.render_markdown(
        result=result,
        report_path=tmp_path / "r2i.md",
        csv_path=tmp_path / "r2i.csv",
        n_synthetic_samples=16,
        max_real_samples=16,
        max_sentinel_datasets=1,
        seed=1234,
        sentinel_tokens=list(exp09.DEFAULT_SENTINEL_TOKENS),
        remediation_profile="r2i_sentinel_matrix_v1",
    )

    assert md.startswith("# R2i Sentinel Morphology Audit")
    assert "## R2i FruitPuree Provenance" in md
    assert "`semi_solid_fruit_puree_short_path_scatter_smoothing`" in md
    assert "`cloudy_berry_percent_transmittance_readout`" in md
    assert "does not reuse" in md


# ---------------------------------------------------------------------------
# R2j profile wiring (DIESEL micro-path, preserving R2i/R2h/R2g/R2f routes).
# ---------------------------------------------------------------------------


def test_exp09_exposes_r2j_profile_constant() -> None:
    exp09 = _load_exp09_module()
    assert "r2j_sentinel_matrix_v1" in exp09.R2J_REMEDIATION_PROFILES
    assert "r2j_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES
    assert "r2i_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES


def test_run_audit_accepts_r2j_profile(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    _write_empty_cohorts(tmp_path)

    result = exp09.run_audit(
        root=tmp_path,
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=2,
        seed=1,
        remediation_profile="r2j_sentinel_matrix_v1",
    )
    assert result["remediation_profile"] == "r2j_sentinel_matrix_v1"


def test_r2j_effective_profile_changes_only_diesel_vs_r2i() -> None:
    exp09 = _load_exp09_module()
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("DIESEL_bp50_246_b-a"),
            "r2j_sentinel_matrix_v1",
        )
        == "r2j_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_real_like_fruitpuree_dataset(),
            "r2j_sentinel_matrix_v1",
        )
        == "r2i_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("BERRY"),
            "r2j_sentinel_matrix_v1",
        )
        == "r2h_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("LUCAS_SOIL"),
            "r2j_sentinel_matrix_v1",
        )
        == "r2g_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("beer_db"),
            "r2j_sentinel_matrix_v1",
        )
        == "r2f_sentinel_matrix_v1"
    )


def test_r2j_diesel_build_records_micro_path_non_oracle_provenance() -> None:
    exp09 = _load_exp09_module()
    diesel = _make_dataset("DIESEL_bp50_246_b-a")
    run = exp09._build_baseline_synthetic_run(
        dataset=diesel,
        preset=exp09.select_synthetic_preset_for_dataset(diesel),
        n_samples=8,
        seed=123,
        remediation_profile="r2j_sentinel_matrix_v1",
    )

    audit = run.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2j_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2j_sentinel_matrix_remediation"
    assert audit["domain_key"] == "petrochem_fuels"
    assert audit["real_stat_capture"] is False
    assert audit["thresholds_modified"] is False
    params = audit["transform_params"]
    assert params["spectra_rule"] == "micro_path_fuel_transmission_absorbance_floor"
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "micro_path_raw_absorbance"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["additive_baseline_range"] == [0.0005, 0.002]


def test_render_markdown_emits_r2j_diesel_provenance(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    result = {
        "status": "blocked_no_real_data",
        "rows": [],
        "real_runnable_count": 0,
        "real_sentinel_candidate_count": 0,
        "real_selected_count": 0,
        "sentinel_tokens": list(exp09.DEFAULT_SENTINEL_TOKENS),
        "remediation_profile": "r2j_sentinel_matrix_v1",
    }

    md = exp09.render_markdown(
        result=result,
        report_path=tmp_path / "r2j.md",
        csv_path=tmp_path / "r2j.csv",
        n_synthetic_samples=16,
        max_real_samples=16,
        max_sentinel_datasets=1,
        seed=1234,
        sentinel_tokens=list(exp09.DEFAULT_SENTINEL_TOKENS),
        remediation_profile="r2j_sentinel_matrix_v1",
    )

    assert md.startswith("# R2j Sentinel Morphology Audit")
    assert "## R2j DIESEL Provenance" in md
    assert "`micro_path_fuel_transmission_absorbance_floor`" in md
    assert "`path_factor_range=[0.03, 0.05]`" in md
    assert "wavelength-uniform multiplicative compression" in md
    assert "`derivative_under`" in md


# ---------------------------------------------------------------------------
# R2k profile wiring (DIESEL CH overtone contrast, preserving R2i routes).
# ---------------------------------------------------------------------------


def test_exp09_exposes_r2k_profile_constant() -> None:
    exp09 = _load_exp09_module()
    assert "r2k_sentinel_matrix_v1" in exp09.R2K_REMEDIATION_PROFILES
    assert "r2k_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES
    assert "r2i_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES


def test_run_audit_accepts_r2k_profile(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    _write_empty_cohorts(tmp_path)

    result = exp09.run_audit(
        root=tmp_path,
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=2,
        seed=1,
        remediation_profile="r2k_sentinel_matrix_v1",
    )
    assert result["remediation_profile"] == "r2k_sentinel_matrix_v1"


def test_r2k_effective_profile_changes_only_diesel_vs_r2i() -> None:
    exp09 = _load_exp09_module()
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("DIESEL_bp50_246_b-a"),
            "r2k_sentinel_matrix_v1",
        )
        == "r2k_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_real_like_fruitpuree_dataset(),
            "r2k_sentinel_matrix_v1",
        )
        == "r2i_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("BERRY"),
            "r2k_sentinel_matrix_v1",
        )
        == "r2h_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("LUCAS_SOIL"),
            "r2k_sentinel_matrix_v1",
        )
        == "r2g_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("beer_db"),
            "r2k_sentinel_matrix_v1",
        )
        == "r2f_sentinel_matrix_v1"
    )


def test_r2k_diesel_build_records_ch_overtone_non_oracle_provenance() -> None:
    exp09 = _load_exp09_module()
    diesel = _make_dataset("DIESEL_bp50_246_b-a")
    run = exp09._build_baseline_synthetic_run(
        dataset=diesel,
        preset=exp09.select_synthetic_preset_for_dataset(diesel),
        n_samples=8,
        seed=123,
        remediation_profile="r2k_sentinel_matrix_v1",
    )

    audit = run.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2k_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2k_sentinel_matrix_remediation"
    assert audit["domain_key"] == "petrochem_fuels"
    assert audit["real_stat_capture"] is False
    assert audit["thresholds_modified"] is False
    assert audit["metrics_modified"] is False
    params = audit["transform_params"]
    assert params["spectra_rule"] == "micro_path_fuel_ch_overtone_contrast_readout"
    assert (
        params["spectra_source"]
        == "beer_lambert_micro_path_with_fixed_ch_overtone_contrast"
    )
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "micro_path_ch_overtone_raw_absorbance"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["contrast_source"] == "fixed_hydrocarbon_ch_overtone_prior"
    assert params["ch_overtone_centers_nm"] == [
        1150.0,
        1210.0,
        1390.0,
        1460.0,
        1720.0,
    ]
    assert params["feature_contrast_range"] == [0.24, 0.34]


def test_render_markdown_emits_r2k_tradeoff_and_provenance(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    result = {
        "status": "blocked_no_real_data",
        "rows": [],
        "real_runnable_count": 0,
        "real_sentinel_candidate_count": 0,
        "real_selected_count": 0,
        "sentinel_tokens": list(exp09.DEFAULT_SENTINEL_TOKENS),
        "remediation_profile": "r2k_sentinel_matrix_v1",
    }

    md = exp09.render_markdown(
        result=result,
        report_path=tmp_path / "r2k.md",
        csv_path=tmp_path / "r2k.csv",
        n_synthetic_samples=16,
        max_real_samples=16,
        max_sentinel_datasets=1,
        seed=1234,
        sentinel_tokens=list(exp09.DEFAULT_SENTINEL_TOKENS),
        remediation_profile="r2k_sentinel_matrix_v1",
    )

    assert md.startswith("# R2k Sentinel Morphology Audit")
    assert "## R2k DIESEL Provenance" in md
    assert "`micro_path_fuel_ch_overtone_contrast_readout`" in md
    assert "`ch_overtone_centers_nm=[1150.0, 1210.0, 1390.0, 1460.0, 1720.0]`" in md
    assert "No real DIESEL spectra" in md
    assert "R2j reduced the DIESEL gap more aggressively" in md
    assert "`derivative_under=9/9`" in md
    assert "higher DIESEL gap to preserve derivative structure better" in md


# ---------------------------------------------------------------------------
# R2l profile wiring (LUCAS raw-soil readout over R2k inheritance).
# ---------------------------------------------------------------------------


def test_exp09_exposes_r2l_profile_constant() -> None:
    exp09 = _load_exp09_module()
    assert "r2l_sentinel_matrix_v1" in exp09.R2L_REMEDIATION_PROFILES
    assert "r2l_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES
    assert "r2k_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES


def test_run_audit_accepts_r2l_profile(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    _write_empty_cohorts(tmp_path)

    result = exp09.run_audit(
        root=tmp_path,
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=2,
        seed=1,
        remediation_profile="r2l_sentinel_matrix_v1",
    )
    assert result["remediation_profile"] == "r2l_sentinel_matrix_v1"


def test_r2l_effective_profile_changes_lucas_and_preserves_phosphorus_r2g() -> None:
    exp09 = _load_exp09_module()
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("LUCAS_SOC_Cropland_8731_NocitaKS"),
            "r2l_sentinel_matrix_v1",
        )
        == "r2l_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("PHOSPHORUS_SOIL"),
            "r2l_sentinel_matrix_v1",
        )
        == "r2g_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("DIESEL_bp50_246_b-a"),
            "r2l_sentinel_matrix_v1",
        )
        == "r2k_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_real_like_fruitpuree_dataset(),
            "r2l_sentinel_matrix_v1",
        )
        == "r2i_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("BERRY"),
            "r2l_sentinel_matrix_v1",
        )
        == "r2h_sentinel_matrix_v1"
    )


def test_r2l_lucas_build_records_mineral_albedo_provenance() -> None:
    exp09 = _load_exp09_module()
    lucas = _make_dataset("LUCAS_SOC_Cropland_8731_NocitaKS")
    run = exp09._build_baseline_synthetic_run(
        dataset=lucas,
        preset=exp09.select_synthetic_preset_for_dataset(lucas),
        n_samples=8,
        seed=123,
        remediation_profile="r2l_sentinel_matrix_v1",
    )

    audit = run.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2l_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2l_sentinel_matrix_remediation"
    assert audit["domain_key"] == "environmental_soil"
    assert audit["real_stat_capture"] is False
    assert audit["thresholds_modified"] is False
    assert audit["metrics_modified"] is False
    params = audit["transform_params"]
    assert params["spectra_rule"] == (
        "lucas_mineral_albedo_absorbance_floor_scatter_readout"
    )
    assert (
        params["spectra_source"]
        == "fixed_mineral_albedo_floor_plus_diffuse_scatter_residual"
    )
    assert params["path_factor_range"] == [0.2, 0.25]
    assert params["additive_baseline_range"] == [0.30103, 0.30103]
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "lucas_raw_soil_apparent_absorbance"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["baseline_source"] == "mineral_albedo_A_equals_minus_log10_0p5"


@pytest.mark.parametrize(
    "dataset",
    (
        _make_dataset("PHOSPHORUS_SOIL"),
        _make_dataset("DIESEL_bp50_246_b-a"),
        _make_dataset("beer_db"),
        _make_dataset("BERRY"),
        _make_real_like_fruitpuree_dataset(),
    ),
)
def test_r2l_non_lucas_draws_are_identical_to_r2k(dataset: Any) -> None:
    exp09 = _load_exp09_module()
    preset = exp09.select_synthetic_preset_for_dataset(dataset)
    r2k_run = exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2k_sentinel_matrix_v1",
    )
    r2l_run = exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2l_sentinel_matrix_v1",
    )

    assert (
        r2l_run.metadata["r2c_mechanistic_remediation"]["profile"]
        == r2k_run.metadata["r2c_mechanistic_remediation"]["profile"]
    )
    np.testing.assert_allclose(r2l_run.X, r2k_run.X)
    np.testing.assert_allclose(r2l_run.y, r2k_run.y)


def test_render_markdown_emits_r2l_lucas_and_phosphorus_provenance(
    tmp_path: Path,
) -> None:
    exp09 = _load_exp09_module()
    rows = [
        exp09.MorphologyRow(
            status="compared",
            source="AOM_regression",
            task="regression",
            dataset="LUCAS/LUCAS_SOC_Cropland_8731_NocitaKS",
            synthetic_preset="soil",
            comparison_space=exp09.COMPARISON_SPACE,
            n_real_samples=8,
            n_synthetic_samples=8,
            n_wavelengths=4,
            wavelength_min=1100.0,
            wavelength_max=2500.0,
            real_global_mean=0.3,
            synthetic_global_mean=0.3,
            global_mean_delta=0.0,
            real_global_std=0.1,
            synthetic_global_std=0.1,
            global_std_ratio=1.0,
            log10_global_std_ratio=0.0,
            real_amplitude_p50=0.1,
            synthetic_amplitude_p50=0.1,
            amplitude_p50_ratio=1.0,
            log10_amplitude_p50_ratio=0.0,
            real_derivative_std_p50=0.01,
            synthetic_derivative_std_p50=0.01,
            derivative_std_p50_ratio=1.0,
            log10_derivative_std_p50_ratio=0.0,
            mean_curve_corr=1.0,
            inverted_mean_curve_corr=-1.0,
            morphology_gap_score=0.0,
            dominant_morphology_gap="mixed",
            **_audit_kwargs(),
            **_remediation_kwargs(
                profile="r2l_sentinel_matrix_v1",
                enabled=True,
                domain_key="environmental_soil",
                concentrations_applied=True,
                spectra_applied=True,
                spectra_rule="lucas_mineral_albedo_absorbance_floor_scatter_readout",
                constant_status="fixed_mechanistic_prior",
                readout_space="lucas_raw_soil_apparent_absorbance",
                calibration_source="none",
                real_stat_source="none",
                threshold_source="none",
            ),
            blocked_reason="",
        ),
        exp09.MorphologyRow(
            status="compared",
            source="AOM_regression",
            task="regression",
            dataset="PHOSPHORUS/LP_spxyG",
            synthetic_preset="soil",
            comparison_space=exp09.COMPARISON_SPACE,
            n_real_samples=8,
            n_synthetic_samples=8,
            n_wavelengths=4,
            wavelength_min=1100.0,
            wavelength_max=2500.0,
            real_global_mean=0.2,
            synthetic_global_mean=0.2,
            global_mean_delta=0.0,
            real_global_std=0.1,
            synthetic_global_std=0.1,
            global_std_ratio=1.0,
            log10_global_std_ratio=0.0,
            real_amplitude_p50=0.1,
            synthetic_amplitude_p50=0.1,
            amplitude_p50_ratio=1.0,
            log10_amplitude_p50_ratio=0.0,
            real_derivative_std_p50=0.01,
            synthetic_derivative_std_p50=0.01,
            derivative_std_p50_ratio=1.0,
            log10_derivative_std_p50_ratio=0.0,
            mean_curve_corr=-0.05,
            inverted_mean_curve_corr=0.05,
            morphology_gap_score=0.0,
            dominant_morphology_gap="mixed",
            **_audit_kwargs(),
            **_remediation_kwargs(
                profile="r2g_sentinel_matrix_v1",
                enabled=True,
                domain_key="environmental_soil",
                concentrations_applied=True,
                spectra_applied=True,
                spectra_rule="diffuse_powder_smoothing_and_scatter_compression",
            ),
            blocked_reason="",
        ),
    ]
    result = {
        "status": "completed",
        "rows": rows,
        "real_runnable_count": 2,
        "real_sentinel_candidate_count": 2,
        "real_selected_count": 2,
        "sentinel_tokens": list(exp09.DEFAULT_SENTINEL_TOKENS),
        "remediation_profile": "r2l_sentinel_matrix_v1",
    }

    md = exp09.render_markdown(
        result=result,
        report_path=tmp_path / "r2l.md",
        csv_path=tmp_path / "r2l.csv",
        n_synthetic_samples=8,
        max_real_samples=8,
        max_sentinel_datasets=2,
        seed=1234,
        sentinel_tokens=list(exp09.DEFAULT_SENTINEL_TOKENS),
        remediation_profile="r2l_sentinel_matrix_v1",
    )

    assert md.startswith("# R2l Sentinel Morphology Audit")
    assert "## R2l LUCAS Soil Provenance" in md
    assert "`lucas_mineral_albedo_absorbance_floor_scatter_readout`" in md
    assert "`additive_baseline_range=[0.30103, 0.30103]`" in md
    assert "PHOSPHORUS preservation check: rows reported on R2g = 1/1" in md


# ---------------------------------------------------------------------------
# R2m profile wiring (MILK raw emulsion readout over R2l inheritance).
# ---------------------------------------------------------------------------


def test_exp09_exposes_r2m_profile_constant() -> None:
    exp09 = _load_exp09_module()
    assert "r2m_sentinel_matrix_v1" in exp09.R2M_REMEDIATION_PROFILES
    assert "r2m_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES
    assert "r2l_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES


def test_run_audit_accepts_r2m_profile(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    _write_empty_cohorts(tmp_path)

    result = exp09.run_audit(
        root=tmp_path,
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=2,
        seed=1,
        remediation_profile="r2m_sentinel_matrix_v1",
    )
    assert result["remediation_profile"] == "r2m_sentinel_matrix_v1"


def test_r2m_effective_profile_changes_milk_and_preserves_r2l_routes() -> None:
    exp09 = _load_exp09_module()
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("MILK"),
            "r2m_sentinel_matrix_v1",
        )
        == "r2m_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("LUCAS_SOC_Cropland_8731_NocitaKS"),
            "r2m_sentinel_matrix_v1",
        )
        == "r2l_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("PHOSPHORUS_SOIL"),
            "r2m_sentinel_matrix_v1",
        )
        == "r2g_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("DIESEL_bp50_246_b-a"),
            "r2m_sentinel_matrix_v1",
        )
        == "r2k_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("BERRY"),
            "r2m_sentinel_matrix_v1",
        )
        == "r2h_sentinel_matrix_v1"
    )


def test_r2m_milk_build_records_inverse_transflectance_provenance() -> None:
    exp09 = _load_exp09_module()
    milk = _make_dataset("MILK_Milk_Fat_1224_KS")
    r2l_run = exp09._build_baseline_synthetic_run(
        dataset=milk,
        preset=exp09.select_synthetic_preset_for_dataset(milk),
        n_samples=8,
        seed=123,
        remediation_profile="r2l_sentinel_matrix_v1",
    )
    r2m_run = exp09._build_baseline_synthetic_run(
        dataset=milk,
        preset=exp09.select_synthetic_preset_for_dataset(milk),
        n_samples=8,
        seed=123,
        remediation_profile="r2m_sentinel_matrix_v1",
    )

    assert r2l_run.metadata["r2c_mechanistic_remediation"]["profile"] == (
        "r2f_sentinel_matrix_v1"
    )
    audit = r2m_run.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2m_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2m_sentinel_matrix_remediation"
    assert audit["domain_key"] == "food_dairy"
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False
    params = audit["transform_params"]
    assert params["spectra_rule"] == (
        "milk_emulsion_scatter_inverse_transflectance_readout"
    )
    assert params["spectra_source"] == (
        "fat_globule_scatter_inverse_beer_lambert_transflectance"
    )
    assert params["milk_readout_variant"] == "shortwave"
    assert params["path_factor_range"] == [0.55, 0.85]
    assert params["detector_dynamic_range"] == [1.8, 2.6]
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "milk_raw_transflectance_intensity"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["scatter_source"] == "fixed_fat_globule_mie_scatter_prior"
    assert params["provenance_source"] == "exp09_dataset_token_milk_route"
    assert params["milk_readout_route_source"] == "exp09_dataset_token"
    assert params["milk_readout_route_marker"] == "milk"
    assert params["milk_readout_route_non_oracle"] is True
    assert params["milk_readout_route_real_stat_capture"] is False
    assert params["milk_readout_route_thresholds_modified"] is False
    assert not np.allclose(r2m_run.X, r2l_run.X)


def test_r2m_labels_row_uses_fullrange_milk_detector_prior() -> None:
    exp09 = _load_exp09_module()
    labels = _make_dataset("MILK_labels_kenstone70_strat")
    run = exp09._build_baseline_synthetic_run(
        dataset=labels,
        preset=exp09.select_synthetic_preset_for_dataset(labels),
        n_samples=8,
        seed=123,
        remediation_profile="r2m_sentinel_matrix_v1",
    )

    params = run.metadata["r2c_mechanistic_remediation"]["transform_params"]
    assert params["milk_readout_variant"] == "fullrange"
    assert params["detector_dynamic_range"] == [1.0, 1.8]
    assert params["output_clip_intensity"] == [0.0, 3.0]


@pytest.mark.parametrize(
    "dataset",
    (
        _make_dataset("PHOSPHORUS_SOIL"),
        _make_dataset("DIESEL_bp50_246_b-a"),
        _make_dataset("beer_db"),
        _make_dataset("BERRY"),
        _make_dataset("LUCAS_SOC_Cropland_8731_NocitaKS"),
        _make_real_like_fruitpuree_dataset(),
    ),
)
def test_r2m_non_milk_draws_are_identical_to_r2l(dataset: Any) -> None:
    exp09 = _load_exp09_module()
    preset = exp09.select_synthetic_preset_for_dataset(dataset)
    r2l_run = exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2l_sentinel_matrix_v1",
    )
    r2m_run = exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2m_sentinel_matrix_v1",
    )

    assert (
        r2m_run.metadata["r2c_mechanistic_remediation"]["profile"]
        == r2l_run.metadata["r2c_mechanistic_remediation"]["profile"]
    )
    np.testing.assert_allclose(r2m_run.X, r2l_run.X)
    np.testing.assert_allclose(r2m_run.y, r2l_run.y)


def test_render_markdown_emits_r2m_non_gate_milk_provenance(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    rows = [
        exp09.MorphologyRow(
            status="compared",
            source="AOM_regression",
            task="regression",
            dataset="MILK/Milk_Fat_1224_KS",
            synthetic_preset="dairy",
            comparison_space=exp09.COMPARISON_SPACE,
            n_real_samples=8,
            n_synthetic_samples=8,
            n_wavelengths=4,
            wavelength_min=1100.0,
            wavelength_max=1700.0,
            real_global_mean=1.0,
            synthetic_global_mean=1.0,
            global_mean_delta=0.0,
            real_global_std=0.1,
            synthetic_global_std=0.1,
            global_std_ratio=1.0,
            log10_global_std_ratio=0.0,
            real_amplitude_p50=0.1,
            synthetic_amplitude_p50=0.1,
            amplitude_p50_ratio=1.0,
            log10_amplitude_p50_ratio=0.0,
            real_derivative_std_p50=0.01,
            synthetic_derivative_std_p50=0.01,
            derivative_std_p50_ratio=1.0,
            log10_derivative_std_p50_ratio=0.0,
            mean_curve_corr=1.0,
            inverted_mean_curve_corr=-1.0,
            morphology_gap_score=0.0,
            dominant_morphology_gap="mixed",
            **_audit_kwargs(),
            **_remediation_kwargs(
                profile="r2m_sentinel_matrix_v1",
                enabled=True,
                domain_key="food_dairy",
                concentrations_applied=True,
                spectra_applied=True,
                spectra_rule="milk_emulsion_scatter_inverse_transflectance_readout",
                provenance_source="exp09_dataset_token_milk_route",
                route_variant="shortwave",
                constant_status="fixed_mechanistic_prior",
                readout_space="milk_raw_transflectance_intensity",
                calibration_source="none",
                real_stat_source="none",
                threshold_source="none",
            ),
            blocked_reason="",
        ),
    ]
    result = {
        "status": "done",
        "rows": rows,
        "real_runnable_count": 1,
        "real_sentinel_candidate_count": 1,
        "real_selected_count": 1,
        "sentinel_tokens": list(exp09.DEFAULT_SENTINEL_TOKENS),
        "remediation_profile": "r2m_sentinel_matrix_v1",
    }

    md = exp09.render_markdown(
        result=result,
        report_path=tmp_path / "r2m.md",
        csv_path=tmp_path / "r2m.csv",
        n_synthetic_samples=8,
        max_real_samples=8,
        max_sentinel_datasets=1,
        seed=1234,
        sentinel_tokens=list(exp09.DEFAULT_SENTINEL_TOKENS),
        remediation_profile="r2m_sentinel_matrix_v1",
    )

    assert md.startswith("# R2m Sentinel Morphology Audit")
    assert "## R2m MILK Dairy Provenance" in md
    assert "`milk_emulsion_scatter_inverse_transflectance_readout`" in md
    assert "changes only MILK/`food_dairy` rows" in md
    assert "Remaining MILK `variance_under` rows" in md
    assert "explicit bench-only MILK dataset-token route marker" in md
    assert "`exp09_dataset_token_milk_route`" in md
    assert "`shortwave`" in md
    assert "No real MILK spectra, marginal statistics, covariance/PCA" in md
    assert "not a B2/B3/B4/B5 gate" in md


# ---------------------------------------------------------------------------
# R2n profile wiring (MANURE21 organic-mineral readout over R2m inheritance).
# ---------------------------------------------------------------------------


def test_exp09_exposes_r2n_profile_constant() -> None:
    exp09 = _load_exp09_module()
    assert "r2n_sentinel_matrix_v1" in exp09.R2N_REMEDIATION_PROFILES
    assert "r2n_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES
    assert "r2m_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES


def test_run_audit_accepts_r2n_profile(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    _write_empty_cohorts(tmp_path)

    result = exp09.run_audit(
        root=tmp_path,
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=2,
        seed=1,
        remediation_profile="r2n_sentinel_matrix_v1",
    )
    assert result["remediation_profile"] == "r2n_sentinel_matrix_v1"


def test_r2n_effective_profile_changes_manure21_and_preserves_r2m_routes() -> None:
    exp09 = _load_exp09_module()
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("MANURE21_All_manure_K2O_SPXY_strat_Manure_type"),
            "r2n_sentinel_matrix_v1",
        )
        == "r2n_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("MILK"),
            "r2n_sentinel_matrix_v1",
        )
        == "r2m_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("LUCAS_SOC_Cropland_8731_NocitaKS"),
            "r2n_sentinel_matrix_v1",
        )
        == "r2l_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("PHOSPHORUS_SOIL"),
            "r2n_sentinel_matrix_v1",
        )
        == "r2g_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("DIESEL_bp50_246_b-a"),
            "r2n_sentinel_matrix_v1",
        )
        == "r2k_sentinel_matrix_v1"
    )


def test_r2n_manure21_build_records_organic_mineral_provenance() -> None:
    exp09 = _load_exp09_module()
    manure = _make_dataset("MANURE21_All_manure_K2O_SPXY_strat_Manure_type")
    r2m_run = exp09._build_baseline_synthetic_run(
        dataset=manure,
        preset=exp09.select_synthetic_preset_for_dataset(manure),
        n_samples=8,
        seed=123,
        remediation_profile="r2m_sentinel_matrix_v1",
    )
    r2n_run = exp09._build_baseline_synthetic_run(
        dataset=manure,
        preset=exp09.select_synthetic_preset_for_dataset(manure),
        n_samples=8,
        seed=123,
        remediation_profile="r2n_sentinel_matrix_v1",
    )

    audit = r2n_run.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2n_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2n_sentinel_matrix_remediation"
    assert audit["domain_key"] == "environmental_soil"
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False
    params = audit["transform_params"]
    assert params["spectra_rule"] == (
        "dried_manure_organic_mineral_albedo_scatter_readout"
    )
    assert params["composition_source"] == (
        "textbook_dried_manure_organic_mineral_composition"
    )
    assert params["spectra_source"] == (
        "fixed_dark_organic_albedo_plus_diffuse_scatter_residual"
    )
    assert params["path_factor_range"] == [0.3, 0.42]
    assert params["additive_baseline_range"] == [0.6, 0.78]
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "dried_ground_manure_raw_apparent_absorbance"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_manure21_route"
    assert params["manure21_readout_route_source"] == "exp09_dataset_token"
    assert params["manure21_readout_route_marker"] == "manure21"
    assert params["manure21_readout_route_non_oracle"] is True
    assert params["manure21_readout_route_real_stat_capture"] is False
    assert params["manure21_readout_route_thresholds_modified"] is False
    assert not np.allclose(r2n_run.X, r2m_run.X)


def test_r2n_manure21_reporting_exposes_effective_matrix_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exp09 = _load_exp09_module()
    _write_empty_cohorts(tmp_path)
    manure = _make_dataset("MANURE21_All_manure_K2O_SPXY_strat_Manure_type")
    wl = np.linspace(1100.0, 1700.0, 8)
    real = np.tile(np.linspace(0.2, 0.8, 8), (8, 1))
    synth = np.tile(np.linspace(0.3, 0.7, 8), (8, 1))
    metadata = {
        "r2c_mechanistic_remediation": {
            "enabled": True,
            "domain_key": "environmental_soil",
            "applied_to_concentrations": True,
            "applied_to_spectra": True,
            "transform_params": {
                "spectra_rule": "dried_manure_organic_mineral_albedo_scatter_readout",
                "composition_source": "textbook_dried_manure_organic_mineral_composition",
                "spectra_source": "fixed_dark_organic_albedo_plus_diffuse_scatter_residual",
                "provenance_source": "exp09_dataset_token_manure21_route",
                "manure21_readout_route_marker": "manure21",
                "constant_status": "fixed_mechanistic_prior",
                "readout_space": "dried_ground_manure_raw_apparent_absorbance",
                "calibration_source": "none",
                "real_stat_source": "none",
                "threshold_source": "none",
            },
        }
    }

    monkeypatch.setattr(
        exp09,
        "discover_local_real_datasets",
        lambda root: ([manure], []),
    )
    monkeypatch.setattr(exp09, "load_real_spectra", lambda dataset, root: (real, wl))
    monkeypatch.setattr(
        exp09,
        "_build_baseline_synthetic_run",
        lambda **kwargs: SimpleNamespace(X=synth, wavelengths=wl, metadata=metadata),
    )

    result = exp09.run_audit(
        root=tmp_path,
        n_synthetic_samples=8,
        max_real_samples=8,
        max_sentinel_datasets=1,
        seed=123,
        sentinel_tokens=["MANURE"],
        remediation_profile="r2n_sentinel_matrix_v1",
    )

    row = result["rows"][0]
    assert row.synthetic_preset == "grain"
    assert row.effective_matrix_route == "manure_organic_mineral_matrix"

    csv_path = tmp_path / "r2n.csv"
    exp09.write_csv(result["rows"], csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        records = list(DictReader(handle))
    assert records[0]["synthetic_preset"] == "grain"
    assert records[0]["effective_matrix_route"] == "manure_organic_mineral_matrix"

    md = exp09.render_markdown(
        result=result,
        report_path=tmp_path / "r2n.md",
        csv_path=csv_path,
        n_synthetic_samples=8,
        max_real_samples=8,
        max_sentinel_datasets=1,
        seed=123,
        sentinel_tokens=["MANURE"],
        remediation_profile="r2n_sentinel_matrix_v1",
    )
    assert "`synthetic_preset` is the selector preset" in md
    assert "`effective_matrix_route` is the builder-reported remediation" in md
    assert "synthetic_preset=grain" in md
    assert "effective_matrix_route=manure_organic_mineral_matrix" in md
    assert (
        "| `MANURE21_All_manure_K2O_SPXY_strat_Manure_type/ds` | `grain` | "
        "`manure_organic_mineral_matrix` |"
    ) in md


@pytest.mark.parametrize(
    "dataset",
    (
        _make_dataset("MILK"),
        _make_dataset("PHOSPHORUS_SOIL"),
        _make_dataset("DIESEL_bp50_246_b-a"),
        _make_dataset("beer_db"),
        _make_dataset("BERRY"),
        _make_dataset("LUCAS_SOC_Cropland_8731_NocitaKS"),
        _make_real_like_fruitpuree_dataset(),
    ),
)
def test_r2n_non_manure_draws_are_identical_to_r2m(dataset: Any) -> None:
    exp09 = _load_exp09_module()
    preset = exp09.select_synthetic_preset_for_dataset(dataset)
    r2m_run = exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2m_sentinel_matrix_v1",
    )
    r2n_run = exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2n_sentinel_matrix_v1",
    )

    assert (
        r2n_run.metadata["r2c_mechanistic_remediation"]["profile"]
        == r2m_run.metadata["r2c_mechanistic_remediation"]["profile"]
    )
    np.testing.assert_allclose(r2n_run.X, r2m_run.X)
    np.testing.assert_allclose(r2n_run.y, r2m_run.y)


def test_render_markdown_emits_r2n_non_gate_manure21_provenance(
    tmp_path: Path,
) -> None:
    exp09 = _load_exp09_module()
    rows = [
        exp09.MorphologyRow(
            status="compared",
            source="AOM_regression",
            task="regression",
            dataset="MANURE21/All_manure_K2O_SPXY_strat_Manure_type",
            synthetic_preset="grain",
            comparison_space=exp09.COMPARISON_SPACE,
            n_real_samples=8,
            n_synthetic_samples=8,
            n_wavelengths=4,
            wavelength_min=1100.0,
            wavelength_max=2500.0,
            real_global_mean=0.8,
            synthetic_global_mean=0.8,
            global_mean_delta=0.0,
            real_global_std=0.1,
            synthetic_global_std=0.1,
            global_std_ratio=1.0,
            log10_global_std_ratio=0.0,
            real_amplitude_p50=0.1,
            synthetic_amplitude_p50=0.1,
            amplitude_p50_ratio=1.0,
            log10_amplitude_p50_ratio=0.0,
            real_derivative_std_p50=0.01,
            synthetic_derivative_std_p50=0.01,
            derivative_std_p50_ratio=1.0,
            log10_derivative_std_p50_ratio=0.0,
            mean_curve_corr=1.0,
            inverted_mean_curve_corr=-1.0,
            morphology_gap_score=0.0,
            dominant_morphology_gap="amplitude_under",
            **_audit_kwargs(),
            **_remediation_kwargs(
                profile="r2n_sentinel_matrix_v1",
                effective_matrix_route="manure_organic_mineral_matrix",
                enabled=True,
                domain_key="environmental_soil",
                concentrations_applied=True,
                spectra_applied=True,
                spectra_rule="dried_manure_organic_mineral_albedo_scatter_readout",
                provenance_source="exp09_dataset_token_manure21_route",
                constant_status="fixed_mechanistic_prior",
                readout_space="dried_ground_manure_raw_apparent_absorbance",
                calibration_source="none",
                real_stat_source="none",
                threshold_source="none",
            ),
            blocked_reason="",
        ),
    ]
    result = {
        "status": "done",
        "rows": rows,
        "real_runnable_count": 1,
        "real_sentinel_candidate_count": 1,
        "real_selected_count": 1,
        "sentinel_tokens": list(exp09.DEFAULT_SENTINEL_TOKENS),
        "remediation_profile": "r2n_sentinel_matrix_v1",
    }

    md = exp09.render_markdown(
        result=result,
        report_path=tmp_path / "r2n.md",
        csv_path=tmp_path / "r2n.csv",
        n_synthetic_samples=8,
        max_real_samples=8,
        max_sentinel_datasets=1,
        seed=1234,
        sentinel_tokens=list(exp09.DEFAULT_SENTINEL_TOKENS),
        remediation_profile="r2n_sentinel_matrix_v1",
    )

    assert md.startswith("# R2n Sentinel Morphology Audit")
    assert "## R2n MANURE21 Manure Provenance" in md
    assert "`dried_manure_organic_mineral_albedo_scatter_readout`" in md
    assert "LUCAS remains R2l, PHOSPHORUS remains R2g, and MILK remains R2m" in md
    assert "explicit bench-only MANURE21 dataset-token marker" in md
    assert "`exp09_dataset_token_manure21_route`" in md
    assert "`manure_organic_mineral_matrix`" in md
    assert "No real MANURE21 spectra, marginal statistics, covariance/PCA" in md
    assert "Remaining MANURE21 `amplitude_under` rows = 1/1" in md
    assert "current failure mode, not a pass" in md
    assert "not a B2/B3/B4/B5 gate" in md


# ---------------------------------------------------------------------------
# R2o profile wiring (BEER fermented-liquid readout over R2n inheritance).
# ---------------------------------------------------------------------------


def test_exp09_exposes_r2o_profile_constant() -> None:
    exp09 = _load_exp09_module()
    assert "r2o_sentinel_matrix_v1" in exp09.R2O_REMEDIATION_PROFILES
    assert "r2o_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES
    assert "r2n_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES


def test_run_audit_accepts_r2o_profile(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    _write_empty_cohorts(tmp_path)

    result = exp09.run_audit(
        root=tmp_path,
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=2,
        seed=1,
        remediation_profile="r2o_sentinel_matrix_v1",
    )
    assert result["remediation_profile"] == "r2o_sentinel_matrix_v1"


def test_r2o_effective_profile_changes_beer_and_preserves_r2n_routes() -> None:
    exp09 = _load_exp09_module()
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("BEER_OriginalExtract_60_KS"),
            "r2o_sentinel_matrix_v1",
        )
        == "r2o_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("MANURE21_All_manure_K2O_SPXY_strat_Manure_type"),
            "r2o_sentinel_matrix_v1",
        )
        == "r2n_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("BERRY"),
            "r2o_sentinel_matrix_v1",
        )
        == "r2h_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_real_like_fruitpuree_dataset(),
            "r2o_sentinel_matrix_v1",
        )
        == "r2i_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("DIESEL_bp50_246_b-a"),
            "r2o_sentinel_matrix_v1",
        )
        == "r2k_sentinel_matrix_v1"
    )


def test_r2o_beer_build_records_fermented_liquid_provenance() -> None:
    exp09 = _load_exp09_module()
    beer = _make_dataset("BEER_OriginalExtract_60_KS")
    preset = exp09.select_synthetic_preset_for_dataset(beer)
    r2n_run = exp09._build_baseline_synthetic_run(
        dataset=beer,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2n_sentinel_matrix_v1",
    )
    r2o_run = exp09._build_baseline_synthetic_run(
        dataset=beer,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2o_sentinel_matrix_v1",
    )

    audit = r2o_run.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2o_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2o_sentinel_matrix_remediation"
    assert audit["domain_key"] == "beverage_wine"
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False
    params = audit["transform_params"]
    assert params["spectra_rule"] == (
        "fermented_beer_turbid_cuvette_absorbance_readout"
    )
    assert params["composition_source"] == "textbook_beer_composition"
    assert params["spectra_source"] == (
        "beer_lambert_long_path_with_fixed_haze_carbonation"
    )
    assert params["path_factor_range"] == [1.75, 2.35]
    assert params["haze_absorbance_baseline_range"] == [1.75, 2.15]
    assert params["haze_slope_absorbance_range"] == [0.06, 0.18]
    assert params["carbonation_residual_absorbance_range"] == [0.0, 0.05]
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "fermented_beer_raw_apparent_absorbance"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_beer_route"
    assert params["beer_readout_route_source"] == "exp09_dataset_token"
    assert params["beer_readout_route_marker"] == "beer"
    assert params["beer_readout_route_non_oracle"] is True
    assert params["beer_readout_route_real_stat_capture"] is False
    assert params["beer_readout_route_thresholds_modified"] is False
    assert not np.allclose(r2o_run.X, r2n_run.X)


@pytest.mark.parametrize(
    "dataset",
    (
        _make_dataset("MANURE21_All_manure_K2O_SPXY_strat_Manure_type"),
        _make_dataset("MILK"),
        _make_dataset("PHOSPHORUS_SOIL"),
        _make_dataset("DIESEL_bp50_246_b-a"),
        _make_dataset("BERRY"),
        _make_dataset("LUCAS_SOC_Cropland_8731_NocitaKS"),
        _make_real_like_fruitpuree_dataset(),
    ),
)
def test_r2o_non_beer_draws_are_identical_to_r2n(dataset: Any) -> None:
    exp09 = _load_exp09_module()
    preset = exp09.select_synthetic_preset_for_dataset(dataset)
    r2n_run = exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2n_sentinel_matrix_v1",
    )
    r2o_run = exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2o_sentinel_matrix_v1",
    )

    assert (
        r2o_run.metadata["r2c_mechanistic_remediation"]["profile"]
        == r2n_run.metadata["r2c_mechanistic_remediation"]["profile"]
    )
    np.testing.assert_allclose(r2o_run.X, r2n_run.X)
    np.testing.assert_allclose(r2o_run.y, r2n_run.y)


def test_r2o_reporting_exposes_beer_effective_matrix_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exp09 = _load_exp09_module()
    _write_empty_cohorts(tmp_path)
    beer = _make_dataset("BEER_OriginalExtract_60_KS")
    wl = np.linspace(1100.0, 1700.0, 8)
    real = np.tile(np.linspace(2.0, 3.0, 8), (8, 1))
    synth = np.tile(np.linspace(2.1, 2.9, 8), (8, 1))
    metadata = {
        "r2c_mechanistic_remediation": {
            "enabled": True,
            "domain_key": "beverage_wine",
            "applied_to_concentrations": True,
            "applied_to_spectra": True,
            "transform_params": {
                "spectra_rule": "fermented_beer_turbid_cuvette_absorbance_readout",
                "composition_source": "textbook_beer_composition",
                "spectra_source": "beer_lambert_long_path_with_fixed_haze_carbonation",
                "provenance_source": "exp09_dataset_token_beer_route",
                "beer_readout_route_marker": "beer",
                "constant_status": "fixed_mechanistic_prior",
                "readout_space": "fermented_beer_raw_apparent_absorbance",
                "calibration_source": "none",
                "real_stat_source": "none",
                "threshold_source": "none",
            },
        }
    }

    monkeypatch.setattr(exp09, "discover_local_real_datasets", lambda root: ([beer], []))
    monkeypatch.setattr(exp09, "load_real_spectra", lambda dataset, root: (real, wl))
    monkeypatch.setattr(
        exp09,
        "_build_baseline_synthetic_run",
        lambda **kwargs: SimpleNamespace(X=synth, wavelengths=wl, metadata=metadata),
    )

    result = exp09.run_audit(
        root=tmp_path,
        n_synthetic_samples=8,
        max_real_samples=8,
        max_sentinel_datasets=1,
        seed=123,
        sentinel_tokens=["BEER"],
        remediation_profile="r2o_sentinel_matrix_v1",
    )

    row = result["rows"][0]
    assert row.synthetic_preset == "wine"
    assert row.effective_matrix_route == "beer_fermented_liquid_matrix"


def test_render_markdown_emits_r2o_non_gate_beer_provenance(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    rows = [
        exp09.MorphologyRow(
            status="compared",
            source="AOM_regression",
            task="regression",
            dataset="BEER/Beer_OriginalExtract_60_KS",
            synthetic_preset="wine",
            comparison_space=exp09.COMPARISON_SPACE,
            n_real_samples=8,
            n_synthetic_samples=8,
            n_wavelengths=4,
            wavelength_min=1100.0,
            wavelength_max=1700.0,
            real_global_mean=2.5,
            synthetic_global_mean=2.5,
            global_mean_delta=0.0,
            real_global_std=0.1,
            synthetic_global_std=0.1,
            global_std_ratio=1.0,
            log10_global_std_ratio=0.0,
            real_amplitude_p50=0.1,
            synthetic_amplitude_p50=0.1,
            amplitude_p50_ratio=1.0,
            log10_amplitude_p50_ratio=0.0,
            real_derivative_std_p50=0.01,
            synthetic_derivative_std_p50=0.01,
            derivative_std_p50_ratio=1.0,
            log10_derivative_std_p50_ratio=0.0,
            mean_curve_corr=1.0,
            inverted_mean_curve_corr=-1.0,
            morphology_gap_score=0.0,
            dominant_morphology_gap="mean_shift",
            **_audit_kwargs(),
            **_remediation_kwargs(
                profile="r2o_sentinel_matrix_v1",
                effective_matrix_route="beer_fermented_liquid_matrix",
                enabled=True,
                domain_key="beverage_wine",
                concentrations_applied=True,
                spectra_applied=True,
                spectra_rule="fermented_beer_turbid_cuvette_absorbance_readout",
                provenance_source="exp09_dataset_token_beer_route",
                constant_status="fixed_mechanistic_prior",
                readout_space="fermented_beer_raw_apparent_absorbance",
                calibration_source="none",
                real_stat_source="none",
                threshold_source="none",
            ),
            blocked_reason="",
        ),
    ]
    result = {
        "status": "done",
        "rows": rows,
        "real_runnable_count": 1,
        "real_sentinel_candidate_count": 1,
        "real_selected_count": 1,
        "sentinel_tokens": list(exp09.DEFAULT_SENTINEL_TOKENS),
        "remediation_profile": "r2o_sentinel_matrix_v1",
    }

    md = exp09.render_markdown(
        result=result,
        report_path=tmp_path / "r2o.md",
        csv_path=tmp_path / "r2o.csv",
        n_synthetic_samples=8,
        max_real_samples=8,
        max_sentinel_datasets=1,
        seed=1234,
        sentinel_tokens=list(exp09.DEFAULT_SENTINEL_TOKENS),
        remediation_profile="r2o_sentinel_matrix_v1",
    )

    assert md.startswith("# R2o Sentinel Morphology Audit")
    assert "## R2o BEER Fermented-Liquid Provenance" in md
    assert "`fermented_beer_turbid_cuvette_absorbance_readout`" in md
    assert "does not reuse the BERRY percent-transmittance/intensity readout" in md
    assert "No real BEER spectra, marginal statistics, covariance/PCA" in md
    assert "Remaining BEER `variance_under` rows = 0/1" in md
    assert "Non-BEER rows accidentally reported on R2o = 0/0" in md
    assert "not a B2/B3/B4/B5 gate" in md


# ---------------------------------------------------------------------------
# R2p profile wiring (PHOSPHORUS mineral-soil readout over R2o inheritance).
# ---------------------------------------------------------------------------


def test_exp09_exposes_r2p_profile_constant() -> None:
    exp09 = _load_exp09_module()
    assert "r2p_sentinel_matrix_v1" in exp09.R2P_REMEDIATION_PROFILES
    assert "r2p_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES
    assert "r2o_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES


def test_run_audit_accepts_r2p_profile(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    _write_empty_cohorts(tmp_path)

    result = exp09.run_audit(
        root=tmp_path,
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=2,
        seed=1,
        remediation_profile="r2p_sentinel_matrix_v1",
    )
    assert result["remediation_profile"] == "r2p_sentinel_matrix_v1"


def test_r2p_effective_profile_changes_phosphorus_and_preserves_r2o_routes() -> None:
    exp09 = _load_exp09_module()
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("PHOSPHORUS_SOIL"),
            "r2p_sentinel_matrix_v1",
        )
        == "r2p_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("BEER_OriginalExtract_60_KS"),
            "r2p_sentinel_matrix_v1",
        )
        == "r2o_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("MANURE21_All_manure_K2O_SPXY_strat_Manure_type"),
            "r2p_sentinel_matrix_v1",
        )
        == "r2n_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("MILK"),
            "r2p_sentinel_matrix_v1",
        )
        == "r2m_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("LUCAS_SOC_Cropland_8731_NocitaKS"),
            "r2p_sentinel_matrix_v1",
        )
        == "r2l_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("DIESEL_bp50_246_b-a"),
            "r2p_sentinel_matrix_v1",
        )
        == "r2k_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("BERRY"),
            "r2p_sentinel_matrix_v1",
        )
        == "r2h_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_real_like_fruitpuree_dataset(),
            "r2p_sentinel_matrix_v1",
        )
        == "r2i_sentinel_matrix_v1"
    )


def test_r2p_token_overrides_attach_explicit_phosphorus_marker_only() -> None:
    exp09 = _load_exp09_module()
    phosphorus_overrides = exp09._remediation_token_source_overrides(
        _make_dataset("PHOSPHORUS_SOIL"),
        "r2p_sentinel_matrix_v1",
    )
    route = phosphorus_overrides["_r2p_phosphorus_readout_route"]
    assert phosphorus_overrides["matrix_type"] == "powder"
    assert phosphorus_overrides["measurement_mode"] == "reflectance"
    assert route == {
        "enabled": True,
        "route_marker": "phosphorus",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }

    lucas_overrides = exp09._remediation_token_source_overrides(
        _make_dataset("LUCAS_SOC_Cropland_8731_NocitaKS"),
        "r2p_sentinel_matrix_v1",
    )
    assert "_r2p_phosphorus_readout_route" not in lucas_overrides
    assert lucas_overrides["_r2l_lucas_soil_route"]["route_marker"] == "lucas"


def test_r2p_phosphorus_build_records_mineral_albedo_provenance() -> None:
    exp09 = _load_exp09_module()
    phosphorus = _make_dataset("PHOSPHORUS_SOIL")
    r2o_run = exp09._build_baseline_synthetic_run(
        dataset=phosphorus,
        preset=exp09.select_synthetic_preset_for_dataset(phosphorus),
        n_samples=8,
        seed=123,
        remediation_profile="r2o_sentinel_matrix_v1",
    )
    r2p_run = exp09._build_baseline_synthetic_run(
        dataset=phosphorus,
        preset=exp09.select_synthetic_preset_for_dataset(phosphorus),
        n_samples=8,
        seed=123,
        remediation_profile="r2p_sentinel_matrix_v1",
    )

    audit = r2p_run.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2p_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2p_sentinel_matrix_remediation"
    assert audit["domain_key"] == "environmental_soil"
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False
    params = audit["transform_params"]
    assert params["spectra_rule"] == (
        "phosphorus_mineral_fertilizer_albedo_residual_readout"
    )
    assert params["composition_source"] == (
        "mechanistic_mineral_organic_topsoil_composition"
    )
    assert params["spectra_source"] == (
        "fixed_phosphate_mineral_albedo_plus_inverted_residual"
    )
    assert params["path_factor_range"] == [0.95, 1.05]
    assert params["additive_baseline_range"] == [0.195, 0.21]
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "phosphorus_raw_mineral_soil_apparent_absorbance"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_phosphorus_route"
    assert params["phosphorus_readout_route_source"] == "exp09_dataset_token"
    assert params["phosphorus_readout_route_marker"] == "phosphorus"
    assert params["phosphorus_readout_route_non_oracle"] is True
    assert params["phosphorus_readout_route_real_stat_capture"] is False
    assert params["phosphorus_readout_route_thresholds_modified"] is False
    assert not np.allclose(r2p_run.X, r2o_run.X)


@pytest.mark.parametrize(
    "dataset",
    (
        _make_dataset("BEER_OriginalExtract_60_KS"),
        _make_dataset("MANURE21_All_manure_K2O_SPXY_strat_Manure_type"),
        _make_dataset("MILK"),
        _make_dataset("LUCAS_SOC_Cropland_8731_NocitaKS"),
        _make_dataset("DIESEL_bp50_246_b-a"),
        _make_dataset("BERRY"),
        _make_real_like_fruitpuree_dataset(),
    ),
)
def test_r2p_non_phosphorus_draws_are_identical_to_r2o(dataset: Any) -> None:
    exp09 = _load_exp09_module()
    preset = exp09.select_synthetic_preset_for_dataset(dataset)
    r2o_run = exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2o_sentinel_matrix_v1",
    )
    r2p_run = exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2p_sentinel_matrix_v1",
    )

    assert (
        r2p_run.metadata["r2c_mechanistic_remediation"]["profile"]
        == r2o_run.metadata["r2c_mechanistic_remediation"]["profile"]
    )
    np.testing.assert_allclose(r2p_run.X, r2o_run.X)
    np.testing.assert_allclose(r2p_run.y, r2o_run.y)


def test_render_markdown_emits_r2p_non_gate_derivative_over_note(
    tmp_path: Path,
) -> None:
    exp09 = _load_exp09_module()
    rows = [
        exp09.MorphologyRow(
            status="compared",
            source="AOM_regression",
            task="regression",
            dataset="PHOSPHORUS/LP_spxyG",
            synthetic_preset="soil",
            comparison_space=exp09.COMPARISON_SPACE,
            n_real_samples=8,
            n_synthetic_samples=8,
            n_wavelengths=4,
            wavelength_min=1100.0,
            wavelength_max=2500.0,
            real_global_mean=0.2,
            synthetic_global_mean=0.2,
            global_mean_delta=0.0,
            real_global_std=0.1,
            synthetic_global_std=0.1,
            global_std_ratio=1.0,
            log10_global_std_ratio=0.0,
            real_amplitude_p50=0.1,
            synthetic_amplitude_p50=0.1,
            amplitude_p50_ratio=1.0,
            log10_amplitude_p50_ratio=0.0,
            real_derivative_std_p50=0.01,
            synthetic_derivative_std_p50=0.02,
            derivative_std_p50_ratio=2.0,
            log10_derivative_std_p50_ratio=0.301,
            mean_curve_corr=1.0,
            inverted_mean_curve_corr=-1.0,
            morphology_gap_score=0.301,
            dominant_morphology_gap="derivative_over",
            **_audit_kwargs(),
            **_remediation_kwargs(
                profile="r2p_sentinel_matrix_v1",
                effective_matrix_route="phosphorus_mineral_soil_matrix",
                enabled=True,
                domain_key="environmental_soil",
                concentrations_applied=True,
                spectra_applied=True,
                spectra_rule="phosphorus_mineral_fertilizer_albedo_residual_readout",
                provenance_source="exp09_dataset_token_phosphorus_route",
                constant_status="fixed_mechanistic_prior",
                readout_space="phosphorus_raw_mineral_soil_apparent_absorbance",
                calibration_source="none",
                real_stat_source="none",
                threshold_source="none",
            ),
            blocked_reason="",
        ),
        exp09.MorphologyRow(
            status="compared",
            source="AOM_regression",
            task="regression",
            dataset="BEER/Beer_OriginalExtract_60_KS",
            synthetic_preset="wine",
            comparison_space=exp09.COMPARISON_SPACE,
            n_real_samples=8,
            n_synthetic_samples=8,
            n_wavelengths=4,
            wavelength_min=1100.0,
            wavelength_max=1700.0,
            real_global_mean=2.5,
            synthetic_global_mean=2.5,
            global_mean_delta=0.0,
            real_global_std=0.1,
            synthetic_global_std=0.1,
            global_std_ratio=1.0,
            log10_global_std_ratio=0.0,
            real_amplitude_p50=0.1,
            synthetic_amplitude_p50=0.1,
            amplitude_p50_ratio=1.0,
            log10_amplitude_p50_ratio=0.0,
            real_derivative_std_p50=0.01,
            synthetic_derivative_std_p50=0.01,
            derivative_std_p50_ratio=1.0,
            log10_derivative_std_p50_ratio=0.0,
            mean_curve_corr=1.0,
            inverted_mean_curve_corr=-1.0,
            morphology_gap_score=0.0,
            dominant_morphology_gap="variance_under",
            **_audit_kwargs(),
            **_remediation_kwargs(
                profile="r2o_sentinel_matrix_v1",
                effective_matrix_route="beer_fermented_liquid_matrix",
                enabled=True,
                domain_key="beverage_wine",
                concentrations_applied=True,
                spectra_applied=True,
                spectra_rule="fermented_beer_turbid_cuvette_absorbance_readout",
                provenance_source="exp09_dataset_token_beer_route",
                constant_status="fixed_mechanistic_prior",
                readout_space="fermented_beer_raw_apparent_absorbance",
                calibration_source="none",
                real_stat_source="none",
                threshold_source="none",
            ),
            blocked_reason="",
        ),
    ]
    result = {
        "status": "done",
        "rows": rows,
        "real_runnable_count": 2,
        "real_sentinel_candidate_count": 2,
        "real_selected_count": 2,
        "sentinel_tokens": list(exp09.DEFAULT_SENTINEL_TOKENS),
        "remediation_profile": "r2p_sentinel_matrix_v1",
    }

    md = exp09.render_markdown(
        result=result,
        report_path=tmp_path / "r2p.md",
        csv_path=tmp_path / "r2p.csv",
        n_synthetic_samples=8,
        max_real_samples=8,
        max_sentinel_datasets=2,
        seed=1234,
        sentinel_tokens=list(exp09.DEFAULT_SENTINEL_TOKENS),
        remediation_profile="r2p_sentinel_matrix_v1",
    )

    assert md.startswith("# R2p Sentinel Morphology Audit")
    assert "## R2p PHOSPHORUS Mineral-Soil Provenance" in md
    assert "`phosphorus_mineral_fertilizer_albedo_residual_readout`" in md
    assert "R2p inherits R2o routing and changes only explicit PHOSPHORUS" in md
    assert "The route is selected only from the explicit bench-only PHOSPHORUS" in md
    assert "No real PHOSPHORUS spectra, marginal statistics, covariance/PCA" in md
    assert "Remaining PHOSPHORUS `derivative_over` rows = 1/1" in md
    assert "Non-PHOSPHORUS rows accidentally reported on R2p = 0/1" in md
    assert "not a B2/B3/B4/B5 gate" in md


# ---------------------------------------------------------------------------
# R2q profile wiring (LUCAS pH Organic humic-soil readout over R2p inheritance).
# ---------------------------------------------------------------------------


def test_exp09_exposes_r2q_profile_constant() -> None:
    exp09 = _load_exp09_module()
    assert "r2q_sentinel_matrix_v1" in exp09.R2Q_REMEDIATION_PROFILES
    assert "r2q_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES
    assert "r2p_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES


def test_r2q_effective_profile_changes_only_lucas_ph_organic() -> None:
    exp09 = _load_exp09_module()
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("LUCAS_pH_Organic_1763_LiuRandomOrganic"),
            "r2q_sentinel_matrix_v1",
        )
        == "r2q_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("LUCAS_SOC_Cropland_8731_NocitaKS"),
            "r2q_sentinel_matrix_v1",
        )
        == "r2l_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("PHOSPHORUS_SOIL"),
            "r2q_sentinel_matrix_v1",
        )
        == "r2p_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("BEER_OriginalExtract_60_KS"),
            "r2q_sentinel_matrix_v1",
        )
        == "r2o_sentinel_matrix_v1"
    )


def test_r2q_token_overrides_attach_explicit_lucas_ph_organic_marker_only() -> None:
    exp09 = _load_exp09_module()
    organic_overrides = exp09._remediation_token_source_overrides(
        _make_dataset("LUCAS_pH_Organic_1763_LiuRandomOrganic"),
        "r2q_sentinel_matrix_v1",
    )
    route = organic_overrides["_r2q_lucas_ph_organic_readout_route"]
    assert organic_overrides["matrix_type"] == "powder"
    assert organic_overrides["measurement_mode"] == "reflectance"
    assert route == {
        "enabled": True,
        "route_marker": "lucas_ph_organic",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }

    lucas_soc_overrides = exp09._remediation_token_source_overrides(
        _make_dataset("LUCAS_SOC_Cropland_8731_NocitaKS"),
        "r2q_sentinel_matrix_v1",
    )
    assert "_r2q_lucas_ph_organic_readout_route" not in lucas_soc_overrides
    assert lucas_soc_overrides["_r2l_lucas_soil_route"]["route_marker"] == "lucas"


def test_r2q_lucas_ph_organic_build_records_humic_provenance() -> None:
    exp09 = _load_exp09_module()
    organic = _make_dataset("LUCAS_pH_Organic_1763_LiuRandomOrganic")
    r2p_run = exp09._build_baseline_synthetic_run(
        dataset=organic,
        preset=exp09.select_synthetic_preset_for_dataset(organic),
        n_samples=8,
        seed=123,
        remediation_profile="r2p_sentinel_matrix_v1",
    )
    r2q_run = exp09._build_baseline_synthetic_run(
        dataset=organic,
        preset=exp09.select_synthetic_preset_for_dataset(organic),
        n_samples=8,
        seed=123,
        remediation_profile="r2q_sentinel_matrix_v1",
    )

    audit = r2q_run.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2q_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2q_sentinel_matrix_remediation"
    assert audit["domain_key"] == "environmental_soil"
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False
    params = audit["transform_params"]
    assert params["spectra_rule"] == "lucas_ph_organic_humic_albedo_oh_readout"
    assert params["composition_source"] == (
        "fixed_lucas_ph_organic_humic_topsoil_composition"
    )
    assert params["spectra_source"] == (
        "fixed_humic_dark_albedo_plus_oh_residual_readout"
    )
    assert params["path_factor_range"] == [0.22, 0.32]
    assert params["additive_baseline_range"] == [0.405, 0.455]
    assert params["humic_slope_absorbance_range"] == [0.015, 0.045]
    assert params["oh_band_absorbance_range"] == [0.005, 0.025]
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "lucas_ph_organic_raw_soil_apparent_absorbance"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_lucas_ph_organic_route"
    assert params["lucas_ph_organic_readout_route_marker"] == "lucas_ph_organic"
    assert params["lucas_ph_organic_readout_route_non_oracle"] is True
    assert params["lucas_ph_organic_readout_route_real_stat_capture"] is False
    assert params["lucas_ph_organic_readout_route_thresholds_modified"] is False
    assert not np.allclose(r2q_run.X, r2p_run.X)


@pytest.mark.parametrize(
    "dataset",
    (
        _make_dataset("LUCAS_SOC_Cropland_8731_NocitaKS"),
        _make_dataset("PHOSPHORUS_SOIL"),
        _make_dataset("BEER_OriginalExtract_60_KS"),
        _make_dataset("MANURE21_All_manure_K2O_SPXY_strat_Manure_type"),
        _make_dataset("MILK"),
        _make_dataset("DIESEL_bp50_246_b-a"),
    ),
)
def test_r2q_non_target_draws_are_identical_to_r2p(dataset: Any) -> None:
    exp09 = _load_exp09_module()
    preset = exp09.select_synthetic_preset_for_dataset(dataset)
    r2p_run = exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2p_sentinel_matrix_v1",
    )
    r2q_run = exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2q_sentinel_matrix_v1",
    )

    assert (
        r2q_run.metadata["r2c_mechanistic_remediation"]["profile"]
        == r2p_run.metadata["r2c_mechanistic_remediation"]["profile"]
    )
    np.testing.assert_allclose(r2q_run.X, r2p_run.X)
    np.testing.assert_allclose(r2q_run.y, r2p_run.y)


# ---------------------------------------------------------------------------
# R2r profile wiring (FruitPuree residual readout over R2q inheritance).
# ---------------------------------------------------------------------------


def test_exp09_exposes_r2r_profile_constant() -> None:
    exp09 = _load_exp09_module()
    assert "r2r_sentinel_matrix_v1" in exp09.R2R_REMEDIATION_PROFILES
    assert "r2r_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES
    assert "r2q_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES


def test_r2r_effective_profile_changes_only_fruitpuree() -> None:
    exp09 = _load_exp09_module()
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_real_like_fruitpuree_dataset(),
            "r2r_sentinel_matrix_v1",
        )
        == "r2r_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("BERRY"),
            "r2r_sentinel_matrix_v1",
        )
        == "r2h_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("juice_db"),
            "r2r_sentinel_matrix_v1",
        )
        == "r2f_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("LUCAS_pH_Organic_1763_LiuRandomOrganic"),
            "r2r_sentinel_matrix_v1",
        )
        == "r2q_sentinel_matrix_v1"
    )


def test_r2r_token_overrides_attach_explicit_fruitpuree_marker_only() -> None:
    exp09 = _load_exp09_module()
    fruitpuree_overrides = exp09._remediation_token_source_overrides(
        _make_real_like_fruitpuree_dataset(),
        "r2r_sentinel_matrix_v1",
    )
    route = fruitpuree_overrides["_r2r_fruitpuree_readout_route"]
    assert fruitpuree_overrides["domain"] == "fruit"
    assert fruitpuree_overrides["matrix_type"] == "paste"
    assert fruitpuree_overrides["measurement_mode"] == "transflectance"
    assert route == {
        "enabled": True,
        "route_marker": "fruitpuree",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }

    berry_overrides = exp09._remediation_token_source_overrides(
        _make_dataset("BERRY"),
        "r2r_sentinel_matrix_v1",
    )
    assert "_r2r_fruitpuree_readout_route" not in berry_overrides
    assert berry_overrides["matrix_type"] == "liquid"


def test_r2r_fruitpuree_build_records_residual_readout_provenance() -> None:
    exp09 = _load_exp09_module()
    fruitpuree = _make_real_like_fruitpuree_dataset()
    preset = exp09.select_synthetic_preset_for_dataset(fruitpuree)
    r2q_run = exp09._build_baseline_synthetic_run(
        dataset=fruitpuree,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2q_sentinel_matrix_v1",
    )
    r2r_run = exp09._build_baseline_synthetic_run(
        dataset=fruitpuree,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2r_sentinel_matrix_v1",
    )

    audit = r2r_run.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2r_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2r_sentinel_matrix_remediation"
    assert audit["domain_key"] == "agriculture_fruit"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False
    params = audit["transform_params"]
    assert params["spectra_rule"] == "strawberry_puree_transflectance_residual_readout"
    assert params["composition_source"] == (
        "textbook_strawberry_puree_cellular_composition"
    )
    assert params["spectra_source"] == (
        "fixed_puree_transflectance_albedo_residual_readout"
    )
    assert params["path_factor_range"] == [0.045, 0.075]
    assert params["additive_baseline_range"] == [0.006, 0.009]
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert (
        params["readout_space"]
        == "strawberry_puree_raw_transflectance_residual_absorbance"
    )
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_fruitpuree_route"
    assert params["fruitpuree_readout_route_marker"] == "fruitpuree"
    assert params["fruitpuree_readout_route_non_oracle"] is True
    assert params["fruitpuree_readout_route_real_stat_capture"] is False
    assert params["fruitpuree_readout_route_thresholds_modified"] is False
    assert params["spectra_rule"] != "cloudy_berry_percent_transmittance_readout"
    assert not np.allclose(r2r_run.X, r2q_run.X)


@pytest.mark.parametrize(
    "dataset",
    (
        _make_dataset("BERRY"),
        _make_dataset("juice_db"),
        _make_dataset("LUCAS_pH_Organic_1763_LiuRandomOrganic"),
        _make_dataset("PHOSPHORUS_SOIL"),
        _make_dataset("BEER_OriginalExtract_60_KS"),
        _make_dataset("DIESEL_bp50_246_b-a"),
    ),
)
def test_r2r_non_fruitpuree_draws_are_identical_to_r2q(dataset: Any) -> None:
    exp09 = _load_exp09_module()
    preset = exp09.select_synthetic_preset_for_dataset(dataset)
    r2q_run = exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2q_sentinel_matrix_v1",
    )
    r2r_run = exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2r_sentinel_matrix_v1",
    )

    assert (
        r2r_run.metadata["r2c_mechanistic_remediation"]["profile"]
        == r2q_run.metadata["r2c_mechanistic_remediation"]["profile"]
    )
    np.testing.assert_allclose(r2r_run.X, r2q_run.X)
    np.testing.assert_allclose(r2r_run.y, r2q_run.y)


def test_exp09_exposes_r2s_profile_constant() -> None:
    exp09 = _load_exp09_module()
    assert "r2s_sentinel_matrix_v1" in exp09.R2S_REMEDIATION_PROFILES
    assert "r2s_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES


def test_r2s_effective_profile_changes_only_diesel_vs_r2r() -> None:
    exp09 = _load_exp09_module()
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("DIESEL_bp50_246_b-a"),
            "r2s_sentinel_matrix_v1",
        )
        == "r2s_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_real_like_fruitpuree_dataset(),
            "r2s_sentinel_matrix_v1",
        )
        == "r2r_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("BEER_OriginalExtract_60_KS"),
            "r2s_sentinel_matrix_v1",
        )
        == "r2o_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("LUCAS_pH_Organic_1763_LiuRandomOrganic"),
            "r2s_sentinel_matrix_v1",
        )
        == "r2q_sentinel_matrix_v1"
    )


def test_r2s_token_overrides_attach_explicit_diesel_marker_only() -> None:
    exp09 = _load_exp09_module()
    diesel_overrides = exp09._remediation_token_source_overrides(
        _make_dataset("DIESEL_bp50_246_b-a"),
        "r2s_sentinel_matrix_v1",
    )
    route = diesel_overrides["_r2s_diesel_readout_route"]
    assert diesel_overrides["matrix_type"] == "liquid"
    assert diesel_overrides["measurement_mode"] == "transmittance"
    assert route == {
        "enabled": True,
        "route_marker": "diesel",
        "source": "exp09_dataset_token",
        "non_oracle": True,
        "no_target_or_label": True,
        "real_stat_capture": False,
        "thresholds_modified": False,
    }

    beer_overrides = exp09._remediation_token_source_overrides(
        _make_dataset("BEER_OriginalExtract_60_KS"),
        "r2s_sentinel_matrix_v1",
    )
    assert "_r2s_diesel_readout_route" not in beer_overrides
    assert "_r2o_beer_readout_route" in beer_overrides


def test_r2s_diesel_build_records_explicit_route_provenance() -> None:
    exp09 = _load_exp09_module()
    diesel = _make_dataset("DIESEL_bp50_246_b-a")
    preset = exp09.select_synthetic_preset_for_dataset(diesel)
    r2r_run = exp09._build_baseline_synthetic_run(
        dataset=diesel,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2r_sentinel_matrix_v1",
    )
    r2s_run = exp09._build_baseline_synthetic_run(
        dataset=diesel,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2s_sentinel_matrix_v1",
    )

    audit = r2s_run.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2s_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2s_sentinel_matrix_remediation"
    assert audit["domain_key"] == "petrochem_fuels"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False
    params = audit["transform_params"]
    assert params["spectra_rule"] == "micro_path_fuel_ch_overtone_contrast_readout"
    assert params["path_factor_range"] == [0.03, 0.045]
    assert params["feature_contrast_range"] == [0.24, 0.34]
    assert params["ch_overtone_gain_range"] == [0.12, 0.2]
    assert (
        params["spectra_source"]
        == "beer_lambert_blank_referenced_micro_path_with_fixed_ch_overtone_contrast"
    )
    assert (
        params["readout_space"]
        == "blank_referenced_micro_path_ch_overtone_raw_absorbance"
    )
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["provenance_source"] == "exp09_dataset_token_diesel_route"
    assert params["diesel_readout_route_marker"] == "diesel"
    assert params["diesel_readout_route_non_oracle"] is True
    assert params["diesel_readout_route_real_stat_capture"] is False
    assert params["diesel_readout_route_thresholds_modified"] is False
    assert not np.allclose(r2s_run.X, r2r_run.X)


@pytest.mark.parametrize(
    "dataset",
    (
        _make_dataset("BERRY"),
        _make_real_like_fruitpuree_dataset(),
        _make_dataset("LUCAS_pH_Organic_1763_LiuRandomOrganic"),
        _make_dataset("LUCAS_SOC_Cropland_8731_NocitaKS"),
        _make_dataset("PHOSPHORUS_SOIL"),
        _make_dataset("BEER_OriginalExtract_60_KS"),
        _make_dataset("MILK_Fat_1224_KS"),
        _make_dataset("MANURE21_All_manure_K2O_SPXY_strat_Manure_type"),
    ),
)
def test_r2s_non_diesel_draws_are_identical_to_r2r(dataset: Any) -> None:
    exp09 = _load_exp09_module()
    preset = exp09.select_synthetic_preset_for_dataset(dataset)
    r2r_run = exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2r_sentinel_matrix_v1",
    )
    r2s_run = exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2s_sentinel_matrix_v1",
    )

    assert (
        r2s_run.metadata["r2c_mechanistic_remediation"]["profile"]
        == r2r_run.metadata["r2c_mechanistic_remediation"]["profile"]
    )
    np.testing.assert_allclose(r2s_run.X, r2r_run.X)
    np.testing.assert_allclose(r2s_run.y, r2r_run.y)


def test_exp09_exposes_r2t_profile_constant() -> None:
    exp09 = _load_exp09_module()
    assert "r2t_sentinel_matrix_v1" in exp09.R2T_REMEDIATION_PROFILES
    assert "r2t_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES
    assert "r2s_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES


def test_r2t_effective_profile_changes_only_manure21_vs_r2s() -> None:
    exp09 = _load_exp09_module()
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("MANURE21_All_manure_K2O_SPXY_strat_Manure_type"),
            "r2t_sentinel_matrix_v1",
        )
        == "r2t_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("DIESEL_bp50_246_b-a"),
            "r2t_sentinel_matrix_v1",
        )
        == "r2s_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_real_like_fruitpuree_dataset(),
            "r2t_sentinel_matrix_v1",
        )
        == "r2r_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("PHOSPHORUS_SOIL"),
            "r2t_sentinel_matrix_v1",
        )
        == "r2p_sentinel_matrix_v1"
    )


def test_r2t_manure21_build_records_heterogeneity_provenance() -> None:
    exp09 = _load_exp09_module()
    manure = _make_dataset("MANURE21_All_manure_K2O_SPXY_strat_Manure_type")
    preset = exp09.select_synthetic_preset_for_dataset(manure)
    r2s_run = exp09._build_baseline_synthetic_run(
        dataset=manure,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2s_sentinel_matrix_v1",
    )
    r2t_run = exp09._build_baseline_synthetic_run(
        dataset=manure,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2t_sentinel_matrix_v1",
    )

    audit = r2t_run.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2t_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2t_sentinel_matrix_remediation"
    assert audit["domain_key"] == "environmental_soil"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False
    params = audit["transform_params"]
    assert params["spectra_rule"] == "dried_manure_heterogeneous_scatter_patch_readout"
    assert params["composition_source"] == "textbook_dried_manure_organic_mineral_composition"
    assert (
        params["spectra_source"]
        == "fixed_dark_organic_albedo_plus_particle_scatter_moisture_mineral_lumps"
    )
    assert params["provenance_source"] == "exp09_dataset_token_manure21_route"
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "dried_ground_manure_raw_apparent_absorbance"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["manure21_readout_route_marker"] == "manure21"
    assert params["manure21_readout_route_non_oracle"] is True
    assert params["manure21_readout_route_real_stat_capture"] is False
    assert params["manure21_readout_route_thresholds_modified"] is False
    fields = exp09._remediation_fields_from_metadata(
        remediation_profile="r2t_sentinel_matrix_v1",
        metadata=r2t_run.metadata,
    )
    assert fields["effective_matrix_route"] == "manure_organic_mineral_matrix"
    assert fields["r2c_remediation_readout_space"] == (
        "dried_ground_manure_raw_apparent_absorbance"
    )
    assert not np.allclose(r2t_run.X, r2s_run.X)


@pytest.mark.parametrize(
    "dataset",
    (
        _make_dataset("DIESEL_bp50_246_b-a"),
        _make_real_like_fruitpuree_dataset(),
        _make_dataset("LUCAS_pH_Organic_1763_LiuRandomOrganic"),
        _make_dataset("LUCAS_SOC_Cropland_8731_NocitaKS"),
        _make_dataset("PHOSPHORUS_SOIL"),
        _make_dataset("BEER_OriginalExtract_60_KS"),
        _make_dataset("MILK_Fat_1224_KS"),
    ),
)
def test_r2t_non_manure_draws_are_identical_to_r2s(dataset: Any) -> None:
    exp09 = _load_exp09_module()
    preset = exp09.select_synthetic_preset_for_dataset(dataset)
    r2s_run = exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2s_sentinel_matrix_v1",
    )
    r2t_run = exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2t_sentinel_matrix_v1",
    )

    assert (
        r2t_run.metadata["r2c_mechanistic_remediation"]["profile"]
        == r2s_run.metadata["r2c_mechanistic_remediation"]["profile"]
    )
    np.testing.assert_allclose(r2t_run.X, r2s_run.X)
    np.testing.assert_allclose(r2t_run.y, r2s_run.y)


def test_render_markdown_emits_r2t_manure_residuals_non_gate_note(
    tmp_path: Path,
) -> None:
    exp09 = _load_exp09_module()

    def _manure_row(dataset: str, dominant_gap: str) -> Any:
        return exp09.MorphologyRow(
            status="compared",
            source="AOM_regression",
            task="regression",
            dataset=dataset,
            synthetic_preset="grain",
            comparison_space=exp09.COMPARISON_SPACE,
            n_real_samples=8,
            n_synthetic_samples=8,
            n_wavelengths=4,
            wavelength_min=1100.0,
            wavelength_max=2500.0,
            real_global_mean=0.88,
            synthetic_global_mean=0.75,
            global_mean_delta=-0.13,
            real_global_std=0.27,
            synthetic_global_std=0.12,
            global_std_ratio=0.44,
            log10_global_std_ratio=-0.36,
            real_amplitude_p50=0.66,
            synthetic_amplitude_p50=0.24,
            amplitude_p50_ratio=0.36,
            log10_amplitude_p50_ratio=-0.44,
            real_derivative_std_p50=0.0046,
            synthetic_derivative_std_p50=0.0036,
            derivative_std_p50_ratio=0.78,
            log10_derivative_std_p50_ratio=-0.11,
            mean_curve_corr=0.8,
            inverted_mean_curve_corr=-0.8,
            morphology_gap_score=1.6,
            dominant_morphology_gap=dominant_gap,
            **_audit_kwargs(),
            **_remediation_kwargs(
                profile="r2t_sentinel_matrix_v1",
                effective_matrix_route="manure_organic_mineral_matrix",
                enabled=True,
                domain_key="environmental_soil",
                concentrations_applied=True,
                spectra_applied=True,
                spectra_rule="dried_manure_heterogeneous_scatter_patch_readout",
                composition_source=(
                    "textbook_dried_manure_organic_mineral_composition"
                ),
                spectra_source=(
                    "fixed_dark_organic_albedo_plus_particle_scatter_moisture_mineral_lumps"
                ),
                provenance_source="exp09_dataset_token_manure21_route",
                constant_status="fixed_mechanistic_prior",
                readout_space="dried_ground_manure_raw_apparent_absorbance",
                calibration_source="none",
                real_stat_source="none",
                threshold_source="none",
            ),
            blocked_reason="",
        )

    rows = [
        _manure_row("MANURE21/All_manure_K2O_SPXY_strat_Manure_type", "amplitude_under"),
        _manure_row("MANURE21/All_manure_CaO_SPXY_strat_Manure_type", "mean_shift"),
    ]
    result = {
        "status": "done",
        "rows": rows,
        "real_runnable_count": 2,
        "real_sentinel_candidate_count": 2,
        "real_selected_count": 2,
        "sentinel_tokens": list(exp09.DEFAULT_SENTINEL_TOKENS),
        "remediation_profile": "r2t_sentinel_matrix_v1",
    }

    md = exp09.render_markdown(
        result=result,
        report_path=tmp_path / "r2t.md",
        csv_path=tmp_path / "r2t.csv",
        n_synthetic_samples=8,
        max_real_samples=8,
        max_sentinel_datasets=2,
        seed=1234,
        sentinel_tokens=list(exp09.DEFAULT_SENTINEL_TOKENS),
        remediation_profile="r2t_sentinel_matrix_v1",
    )

    assert "## R2t MANURE21 Provenance" in md
    assert "MANURE21 diagnostic rows under R2t = 2" in md
    assert "rows still dominated by `mean_shift` = 1/2" in md
    assert "MANURE21 rows still dominated by `amplitude_under` = 1/2" in md
    assert "amplitude transfer remains residual and non-gate" in md
    assert "not a B2/B3/B4/B5 gate" in md


def test_exp09_exposes_r2u_profile_and_keeps_r2t_available() -> None:
    exp09 = _load_exp09_module()
    assert "r2u_sentinel_matrix_v1" in exp09.R2U_REMEDIATION_PROFILES
    assert "r2u_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES
    assert "r2t_sentinel_matrix_v1" in exp09.R2T_REMEDIATION_PROFILES
    assert "r2t_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES


def test_r2u_effective_profile_changes_only_manure21_vs_r2s() -> None:
    exp09 = _load_exp09_module()
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("MANURE21_All_manure_K2O_SPXY_strat_Manure_type"),
            "r2u_sentinel_matrix_v1",
        )
        == "r2u_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("DIESEL_bp50_246_b-a"),
            "r2u_sentinel_matrix_v1",
        )
        == "r2s_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_real_like_fruitpuree_dataset(),
            "r2u_sentinel_matrix_v1",
        )
        == "r2r_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("PHOSPHORUS_SOIL"),
            "r2u_sentinel_matrix_v1",
        )
        == "r2p_sentinel_matrix_v1"
    )


def test_r2u_manure21_build_records_centered_scatter_provenance() -> None:
    exp09 = _load_exp09_module()
    manure = _make_dataset("MANURE21_All_manure_K2O_SPXY_strat_Manure_type")
    preset = exp09.select_synthetic_preset_for_dataset(manure)
    r2s_run = exp09._build_baseline_synthetic_run(
        dataset=manure,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2s_sentinel_matrix_v1",
    )
    r2u_run = exp09._build_baseline_synthetic_run(
        dataset=manure,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2u_sentinel_matrix_v1",
    )

    audit = r2u_run.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2u_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2u_sentinel_matrix_remediation"
    assert audit["domain_key"] == "environmental_soil"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False
    params = audit["transform_params"]
    assert params["spectra_rule"] == "dried_manure_bounded_centered_scatter_readout"
    assert params["composition_source"] == "textbook_dried_manure_organic_mineral_composition"
    assert (
        params["spectra_source"]
        == "fixed_dark_organic_albedo_plus_centered_particle_scatter_bands"
    )
    assert params["additive_baseline_range"] == [0.74, 0.86]
    assert params["provenance_source"] == "exp09_dataset_token_manure21_route"
    assert params["constant_status"] == "fixed_mechanistic_prior"
    assert params["readout_space"] == "dried_ground_manure_raw_apparent_absorbance"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["manure21_readout_route_marker"] == "manure21"
    assert params["manure21_readout_route_non_oracle"] is True
    assert params["manure21_readout_route_real_stat_capture"] is False
    assert params["manure21_readout_route_thresholds_modified"] is False
    fields = exp09._remediation_fields_from_metadata(
        remediation_profile="r2u_sentinel_matrix_v1",
        metadata=r2u_run.metadata,
    )
    assert fields["effective_matrix_route"] == "manure_organic_mineral_matrix"
    assert fields["r2c_remediation_readout_space"] == (
        "dried_ground_manure_raw_apparent_absorbance"
    )
    assert not np.allclose(r2u_run.X, r2s_run.X)


@pytest.mark.parametrize(
    "dataset",
    (
        _make_dataset("DIESEL_bp50_246_b-a"),
        _make_real_like_fruitpuree_dataset(),
        _make_dataset("LUCAS_pH_Organic_1763_LiuRandomOrganic"),
        _make_dataset("LUCAS_SOC_Cropland_8731_NocitaKS"),
        _make_dataset("PHOSPHORUS_SOIL"),
        _make_dataset("BEER_OriginalExtract_60_KS"),
        _make_dataset("MILK_Fat_1224_KS"),
    ),
)
def test_r2u_non_manure_draws_are_identical_to_r2s(dataset: Any) -> None:
    exp09 = _load_exp09_module()
    preset = exp09.select_synthetic_preset_for_dataset(dataset)
    r2s_run = exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2s_sentinel_matrix_v1",
    )
    r2u_run = exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2u_sentinel_matrix_v1",
    )

    assert (
        r2u_run.metadata["r2c_mechanistic_remediation"]["profile"]
        == r2s_run.metadata["r2c_mechanistic_remediation"]["profile"]
    )
    np.testing.assert_allclose(r2u_run.X, r2s_run.X)
    np.testing.assert_allclose(r2u_run.y, r2s_run.y)


def test_exp09_exposes_r2v_profile_and_keeps_r2t_r2u_available() -> None:
    exp09 = _load_exp09_module()
    assert "r2v_sentinel_matrix_v1" in exp09.R2V_REMEDIATION_PROFILES
    assert "r2v_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES
    assert "r2t_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES
    assert "r2u_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES


def test_r2v_effective_profile_changes_only_manure21_vs_r2s() -> None:
    exp09 = _load_exp09_module()
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("MANURE21_All_manure_K2O_SPXY_strat_Manure_type"),
            "r2v_sentinel_matrix_v1",
        )
        == "r2v_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("DIESEL_bp50_246_b-a"),
            "r2v_sentinel_matrix_v1",
        )
        == "r2s_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("PHOSPHORUS_SOIL"),
            "r2v_sentinel_matrix_v1",
        )
        == "r2p_sentinel_matrix_v1"
    )


def test_r2v_manure21_build_records_balanced_centered_scatter_provenance() -> None:
    exp09 = _load_exp09_module()
    manure = _make_dataset("MANURE21_All_manure_K2O_SPXY_strat_Manure_type")
    preset = exp09.select_synthetic_preset_for_dataset(manure)
    r2s_run = exp09._build_baseline_synthetic_run(
        dataset=manure,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2s_sentinel_matrix_v1",
    )
    r2v_run = exp09._build_baseline_synthetic_run(
        dataset=manure,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2v_sentinel_matrix_v1",
    )

    audit = r2v_run.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2v_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2v_sentinel_matrix_remediation"
    assert audit["domain_key"] == "environmental_soil"
    assert audit["applied_to_concentrations"] is True
    assert audit["applied_to_spectra"] is True
    for key in (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "learned",
        "real_stat_capture",
        "thresholds_modified",
        "metrics_modified",
        "imputed",
        "replays_real_rows",
    ):
        assert audit[key] is False
    params = audit["transform_params"]
    assert params["spectra_rule"] == "dried_manure_balanced_centered_scatter_readout"
    assert params["spectra_source"] == (
        "fixed_dark_organic_albedo_plus_balanced_centered_particle_scatter_bands"
    )
    assert params["path_factor_range"] == [0.6, 0.76]
    assert params["additive_baseline_range"] == [0.74, 0.86]
    assert params["scatter_slope_absorbance_range"] == [-0.16, 0.16]
    assert params["moisture_patch_absorbance_range"] == [0.0, 0.105]
    assert params["organic_lump_absorbance_range"] == [0.0, 0.095]
    assert params["mineral_ash_absorbance_range"] == [-0.075, 0.075]
    assert params["heterogeneous_terms_centered"] is True
    assert params["balanced_centered_draws"] is True
    assert params["provenance_source"] == "exp09_dataset_token_manure21_route"
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    assert params["manure21_readout_route_marker"] == "manure21"
    assert params["manure21_readout_route_non_oracle"] is True
    assert params["manure21_readout_route_real_stat_capture"] is False
    assert params["manure21_readout_route_thresholds_modified"] is False
    fields = exp09._remediation_fields_from_metadata(
        remediation_profile="r2v_sentinel_matrix_v1",
        metadata=r2v_run.metadata,
    )
    assert fields["effective_matrix_route"] == "manure_organic_mineral_matrix"
    assert fields["r2c_remediation_readout_space"] == (
        "dried_ground_manure_raw_apparent_absorbance"
    )
    assert not np.allclose(r2v_run.X, r2s_run.X)


@pytest.mark.parametrize(
    "dataset",
    (
        _make_dataset("DIESEL_bp50_246_b-a"),
        _make_real_like_fruitpuree_dataset(),
        _make_dataset("LUCAS_pH_Organic_1763_LiuRandomOrganic"),
        _make_dataset("LUCAS_SOC_Cropland_8731_NocitaKS"),
        _make_dataset("PHOSPHORUS_SOIL"),
        _make_dataset("BEER_OriginalExtract_60_KS"),
        _make_dataset("MILK_Fat_1224_KS"),
    ),
)
def test_r2v_non_manure_draws_are_identical_to_r2s(dataset: Any) -> None:
    exp09 = _load_exp09_module()
    preset = exp09.select_synthetic_preset_for_dataset(dataset)
    r2s_run = exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2s_sentinel_matrix_v1",
    )
    r2v_run = exp09._build_baseline_synthetic_run(
        dataset=dataset,
        preset=preset,
        n_samples=8,
        seed=123,
        remediation_profile="r2v_sentinel_matrix_v1",
    )

    assert (
        r2v_run.metadata["r2c_mechanistic_remediation"]["profile"]
        == r2s_run.metadata["r2c_mechanistic_remediation"]["profile"]
    )
    np.testing.assert_allclose(r2v_run.X, r2s_run.X)
    np.testing.assert_allclose(r2v_run.y, r2s_run.y)


def test_render_markdown_emits_r2v_manure_balanced_scatter_non_gate_note(
    tmp_path: Path,
) -> None:
    exp09 = _load_exp09_module()
    row = exp09.MorphologyRow(
        status="compared",
        source="AOM_regression",
        task="regression",
        dataset="MANURE21/All_manure_K2O_SPXY_strat_Manure_type",
        synthetic_preset="grain",
        comparison_space=exp09.COMPARISON_SPACE,
        n_real_samples=8,
        n_synthetic_samples=8,
        n_wavelengths=4,
        wavelength_min=1100.0,
        wavelength_max=2500.0,
        real_global_mean=0.88,
        synthetic_global_mean=0.80,
        global_mean_delta=-0.08,
        real_global_std=0.27,
        synthetic_global_std=0.17,
        global_std_ratio=0.63,
        log10_global_std_ratio=-0.20,
        real_amplitude_p50=0.66,
        synthetic_amplitude_p50=0.39,
        amplitude_p50_ratio=0.59,
        log10_amplitude_p50_ratio=-0.23,
        real_derivative_std_p50=0.0046,
        synthetic_derivative_std_p50=0.0040,
        derivative_std_p50_ratio=0.87,
        log10_derivative_std_p50_ratio=-0.06,
        mean_curve_corr=0.8,
        inverted_mean_curve_corr=-0.8,
        morphology_gap_score=0.89,
        dominant_morphology_gap="variance_under",
        **_audit_kwargs(),
        **_remediation_kwargs(
            profile="r2v_sentinel_matrix_v1",
            effective_matrix_route="manure_organic_mineral_matrix",
            enabled=True,
            domain_key="environmental_soil",
            concentrations_applied=True,
            spectra_applied=True,
            spectra_rule="dried_manure_balanced_centered_scatter_readout",
            composition_source="textbook_dried_manure_organic_mineral_composition",
            spectra_source=(
                "fixed_dark_organic_albedo_plus_balanced_centered_particle_scatter_bands"
            ),
            provenance_source="exp09_dataset_token_manure21_route",
            constant_status="fixed_mechanistic_prior",
            readout_space="dried_ground_manure_raw_apparent_absorbance",
            calibration_source="none",
            real_stat_source="none",
            threshold_source="none",
        ),
        blocked_reason="",
    )
    result = {
        "status": "done",
        "rows": [row],
        "real_runnable_count": 1,
        "real_sentinel_candidate_count": 1,
        "real_selected_count": 1,
        "sentinel_tokens": list(exp09.DEFAULT_SENTINEL_TOKENS),
        "remediation_profile": "r2v_sentinel_matrix_v1",
    }

    md = exp09.render_markdown(
        result=result,
        report_path=tmp_path / "r2v.md",
        csv_path=tmp_path / "r2v.csv",
        n_synthetic_samples=8,
        max_real_samples=8,
        max_sentinel_datasets=1,
        seed=1234,
        sentinel_tokens=list(exp09.DEFAULT_SENTINEL_TOKENS),
        remediation_profile="r2v_sentinel_matrix_v1",
    )

    assert "## R2v MANURE21 Provenance" in md
    assert "`dried_manure_balanced_centered_scatter_readout`" in md
    assert "R2t/R2u remain available" in md
    assert "do not add a row-uniform lift/downshift" in md
    assert "rows still dominated by `mean_shift` = 0/1" in md
    assert "not a B2/B3/B4/B5 gate" in md


def test_exp09_exposes_r2w_profile_and_changes_only_manure21_vs_r2s() -> None:
    exp09 = _load_exp09_module()
    assert "r2w_sentinel_matrix_v1" in exp09.R2W_REMEDIATION_PROFILES
    assert "r2w_sentinel_matrix_v1" in exp09.ALL_REMEDIATION_PROFILES
    assert "R2W_REMEDIATION_PROFILES" in exp09.__all__
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("MANURE21_All_manure_K2O_SPXY_strat_Manure_type"),
            "r2w_sentinel_matrix_v1",
        )
        == "r2w_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("DIESEL_bp50_246_b-a"),
            "r2w_sentinel_matrix_v1",
        )
        == "r2s_sentinel_matrix_v1"
    )
    assert (
        exp09._effective_remediation_profile_for_dataset(
            _make_dataset("PHOSPHORUS_SOIL"),
            "r2w_sentinel_matrix_v1",
        )
        == "r2p_sentinel_matrix_v1"
    )


def test_r2w_manure21_build_records_albedo_variance_provenance() -> None:
    exp09 = _load_exp09_module()
    manure = _make_dataset("MANURE21_All_manure_K2O_SPXY_strat_Manure_type")
    run = exp09._build_baseline_synthetic_run(
        dataset=manure,
        preset=exp09.select_synthetic_preset_for_dataset(manure),
        n_samples=8,
        seed=123,
        remediation_profile="r2w_sentinel_matrix_v1",
    )

    audit = run.metadata["r2c_mechanistic_remediation"]
    assert audit["profile"] == "r2w_sentinel_matrix_v1"
    assert audit["scope"] == "bench_only_r2w_sentinel_matrix_remediation"
    assert audit["real_stat_capture"] is False
    assert audit["thresholds_modified"] is False
    params = audit["transform_params"]
    assert params["spectra_rule"] == (
        "dried_manure_albedo_variance_centered_scatter_readout"
    )
    assert params["path_factor_range"] == [0.8, 1.0]
    assert params["additive_baseline_range"] == [0.72, 1.0]
    assert params["balanced_centered_draws"] is True
    assert params["calibration_source"] == "none"
    assert params["real_stat_source"] == "none"
    assert params["threshold_source"] == "none"
    fields = exp09._remediation_fields_from_metadata(
        remediation_profile="r2w_sentinel_matrix_v1",
        metadata=run.metadata,
    )
    assert fields["effective_matrix_route"] == "manure_organic_mineral_matrix"


def test_r2w_repeated_seed_summary_csv_matches_report_contract() -> None:
    report_dir = Path(__file__).resolve().parents[1] / "reports"
    csv_path = report_dir / "r2w_repeated_seed_sentinel_morphology_audit.csv"
    md_path = report_dir / "r2w_repeated_seed_sentinel_morphology_audit.md"
    rows = list(DictReader(csv_path.open(newline="", encoding="utf-8")))

    assert len(rows) == 90
    manure_rows = [
        row
        for row in rows
        if row["token_group"] == "MANURE21" and row["status"] == "compared"
    ]
    non_manure_rows = [row for row in rows if row["token_group"] != "MANURE21"]
    blocked_rows = [row for row in rows if row["status"] == "blocked"]
    assert len(manure_rows) == 15

    expected_stats = {
        "r2w": (0.9514, 0.8686, 1.0212),
        "r2v": (1.3167, 1.2091, 1.3881),
        "r2s": (2.2929, 2.1613, 2.4311),
        "r2t": (1.5860, 1.4443, 1.8492),
        "r2u": (1.4425, 1.2948, 1.5592),
        "r2n": (2.2929, 2.1613, 2.4311),
    }
    for profile, expected in expected_stats.items():
        values = np.asarray(
            [float(row[f"{profile}_morphology_gap_score"]) for row in manure_rows],
            dtype=float,
        )
        actual = (
            round(float(values.mean()), 4),
            round(float(values.min()), 4),
            round(float(values.max()), 4),
        )
        assert actual == expected

    assert {
        row["r2w_dominant_morphology_gap"] for row in manure_rows
    } == {"variance_under", "amplitude_under"}
    assert sum(
        row["r2w_dominant_morphology_gap"] == "variance_under"
        for row in manure_rows
    ) == 11
    assert sum(
        row["r2w_dominant_morphology_gap"] == "amplitude_under"
        for row in manure_rows
    ) == 4
    assert all(row["r2w_provenance_ok"] == "True" for row in manure_rows)
    assert all(row["r2w_route_marker_present"] == "True" for row in manure_rows)
    assert all(row["r2w_calibration_source"] == "none" for row in manure_rows)
    assert all(row["r2w_real_stat_source"] == "none" for row in manure_rows)
    assert all(row["r2w_threshold_source"] == "none" for row in manure_rows)

    assert all(row["audit_flags_ok"] == "True" for row in rows)
    assert all(row["non_target_not_r2w_profile"] == "True" for row in rows)
    assert all(
        row["r2w_remediation_profile"] != "r2w_sentinel_matrix_v1"
        for row in non_manure_rows
    )
    assert all(
        row["non_manure_preserved_vs_r2s"] == "True" for row in non_manure_rows
    )
    assert all(row["route_preserved_vs_r2s"] == "True" for row in non_manure_rows)
    assert all(row["metrics_preserved_vs_r2s"] == "True" for row in non_manure_rows)
    assert len(blocked_rows) == 6
    assert all(row["blocked_unchanged_vs_r2s"] == "True" for row in blocked_rows)

    md = md_path.read_text(encoding="utf-8")
    assert "R2w mean/min/max | R2v mean/min/max" in md
    assert "`0.9514/0.8686/1.0212`" in md
    assert "R2w MANURE provenance and explicit route marker present: `True` (15/15)" in md
    assert "Non-target rows accidentally reported on R2w profile: `0/75`" in md


def test_render_markdown_emits_r2u_manure_centered_scatter_non_gate_note(
    tmp_path: Path,
) -> None:
    exp09 = _load_exp09_module()

    row = exp09.MorphologyRow(
        status="compared",
        source="AOM_regression",
        task="regression",
        dataset="MANURE21/All_manure_K2O_SPXY_strat_Manure_type",
        synthetic_preset="grain",
        comparison_space=exp09.COMPARISON_SPACE,
        n_real_samples=8,
        n_synthetic_samples=8,
        n_wavelengths=4,
        wavelength_min=1100.0,
        wavelength_max=2500.0,
        real_global_mean=0.88,
        synthetic_global_mean=0.75,
        global_mean_delta=-0.13,
        real_global_std=0.27,
        synthetic_global_std=0.16,
        global_std_ratio=0.59,
        log10_global_std_ratio=-0.23,
        real_amplitude_p50=0.66,
        synthetic_amplitude_p50=0.36,
        amplitude_p50_ratio=0.55,
        log10_amplitude_p50_ratio=-0.26,
        real_derivative_std_p50=0.0046,
        synthetic_derivative_std_p50=0.0039,
        derivative_std_p50_ratio=0.85,
        log10_derivative_std_p50_ratio=-0.07,
        mean_curve_corr=0.8,
        inverted_mean_curve_corr=-0.8,
        morphology_gap_score=1.0,
        dominant_morphology_gap="amplitude_under",
        **_audit_kwargs(),
        **_remediation_kwargs(
            profile="r2u_sentinel_matrix_v1",
            effective_matrix_route="manure_organic_mineral_matrix",
            enabled=True,
            domain_key="environmental_soil",
            concentrations_applied=True,
            spectra_applied=True,
            spectra_rule="dried_manure_bounded_centered_scatter_readout",
            composition_source="textbook_dried_manure_organic_mineral_composition",
            spectra_source="fixed_dark_organic_albedo_plus_centered_particle_scatter_bands",
            provenance_source="exp09_dataset_token_manure21_route",
            constant_status="fixed_mechanistic_prior",
            readout_space="dried_ground_manure_raw_apparent_absorbance",
            calibration_source="none",
            real_stat_source="none",
            threshold_source="none",
        ),
        blocked_reason="",
    )
    result = {
        "status": "done",
        "rows": [row],
        "real_runnable_count": 1,
        "real_sentinel_candidate_count": 1,
        "real_selected_count": 1,
        "sentinel_tokens": list(exp09.DEFAULT_SENTINEL_TOKENS),
        "remediation_profile": "r2u_sentinel_matrix_v1",
    }

    md = exp09.render_markdown(
        result=result,
        report_path=tmp_path / "r2u.md",
        csv_path=tmp_path / "r2u.csv",
        n_synthetic_samples=8,
        max_real_samples=8,
        max_sentinel_datasets=1,
        seed=1234,
        sentinel_tokens=list(exp09.DEFAULT_SENTINEL_TOKENS),
        remediation_profile="r2u_sentinel_matrix_v1",
    )

    assert "## R2u MANURE21 Provenance" in md
    assert "`dried_manure_bounded_centered_scatter_readout`" in md
    assert "R2t remains available but is not the baseline profile" in md
    assert "without a global continuum lift" in md
    assert "MANURE21 rows still dominated by `variance_under` = 0/1" in md
    assert "variance transfer remains residual and non-gate" in md
    assert "not a B2/B3/B4/B5 gate" in md


def test_render_markdown_emits_r2s_diesel_non_gate_note(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    rows = [
        exp09.MorphologyRow(
            status="compared",
            source="AOM_regression",
            task="regression",
            dataset="DIESEL/DIESEL_bp50_246_b-a",
            synthetic_preset="fuel",
            comparison_space=exp09.COMPARISON_SPACE,
            n_real_samples=8,
            n_synthetic_samples=8,
            n_wavelengths=4,
            wavelength_min=1100.0,
            wavelength_max=1700.0,
            real_global_mean=0.003,
            synthetic_global_mean=0.011,
            global_mean_delta=0.008,
            real_global_std=0.014,
            synthetic_global_std=0.023,
            global_std_ratio=1.6,
            log10_global_std_ratio=0.2,
            real_amplitude_p50=0.04,
            synthetic_amplitude_p50=0.06,
            amplitude_p50_ratio=1.5,
            log10_amplitude_p50_ratio=0.18,
            real_derivative_std_p50=0.002,
            synthetic_derivative_std_p50=0.0022,
            derivative_std_p50_ratio=1.1,
            log10_derivative_std_p50_ratio=0.04,
            mean_curve_corr=0.1,
            inverted_mean_curve_corr=-0.1,
            morphology_gap_score=1.9,
            dominant_morphology_gap="mean_shift",
            **_audit_kwargs(),
            **_remediation_kwargs(
                profile="r2s_sentinel_matrix_v1",
                effective_matrix_route="diesel_fuel_matrix",
                enabled=True,
                domain_key="petrochem_fuels",
                concentrations_applied=True,
                spectra_applied=True,
                spectra_rule="micro_path_fuel_ch_overtone_contrast_readout",
                composition_source="textbook_diesel_composition",
                spectra_source=(
                    "beer_lambert_blank_referenced_micro_path_with_fixed_ch_overtone_contrast"
                ),
                provenance_source="exp09_dataset_token_diesel_route",
                constant_status="fixed_mechanistic_prior",
                readout_space=(
                    "blank_referenced_micro_path_ch_overtone_raw_absorbance"
                ),
                calibration_source="none",
                real_stat_source="none",
                threshold_source="none",
            ),
            blocked_reason="",
        ),
    ]
    result = {
        "status": "done",
        "rows": rows,
        "real_runnable_count": 1,
        "real_sentinel_candidate_count": 1,
        "real_selected_count": 1,
        "sentinel_tokens": list(exp09.DEFAULT_SENTINEL_TOKENS),
        "remediation_profile": "r2s_sentinel_matrix_v1",
    }

    md = exp09.render_markdown(
        result=result,
        report_path=tmp_path / "r2s.md",
        csv_path=tmp_path / "r2s.csv",
        n_synthetic_samples=8,
        max_real_samples=8,
        max_sentinel_datasets=1,
        seed=1234,
        sentinel_tokens=list(exp09.DEFAULT_SENTINEL_TOKENS),
        remediation_profile="r2s_sentinel_matrix_v1",
    )

    assert "## R2s DIESEL Provenance" in md
    assert "explicit bench-only DIESEL dataset-token marker" in md
    assert "DIESEL rows dominated by `derivative_under` = 0/1" in md
    assert "not a B2/B3/B4/B5 gate" in md


def test_render_markdown_emits_r2r_amplitude_under_non_gate_note(
    tmp_path: Path,
) -> None:
    exp09 = _load_exp09_module()
    rows = [
        exp09.MorphologyRow(
            status="compared",
            source="AOM_regression",
            task="regression",
            dataset="FruitPuree/Strawberry2C_983_Holland_Acc94.3",
            synthetic_preset="juice",
            comparison_space=exp09.COMPARISON_SPACE,
            n_real_samples=8,
            n_synthetic_samples=8,
            n_wavelengths=4,
            wavelength_min=900.0,
            wavelength_max=1100.0,
            real_global_mean=0.2,
            synthetic_global_mean=0.2,
            global_mean_delta=0.0,
            real_global_std=0.1,
            synthetic_global_std=0.04,
            global_std_ratio=0.4,
            log10_global_std_ratio=-0.4,
            real_amplitude_p50=0.2,
            synthetic_amplitude_p50=0.06,
            amplitude_p50_ratio=0.3,
            log10_amplitude_p50_ratio=-0.5,
            real_derivative_std_p50=0.01,
            synthetic_derivative_std_p50=0.004,
            derivative_std_p50_ratio=0.4,
            log10_derivative_std_p50_ratio=-0.4,
            mean_curve_corr=0.56,
            inverted_mean_curve_corr=-0.56,
            morphology_gap_score=1.9,
            dominant_morphology_gap="amplitude_under",
            **_audit_kwargs(),
            **_remediation_kwargs(
                profile="r2r_sentinel_matrix_v1",
                effective_matrix_route="fruit_puree_matrix",
                enabled=True,
                domain_key="agriculture_fruit",
                concentrations_applied=True,
                spectra_applied=True,
                spectra_rule="strawberry_puree_transflectance_residual_readout",
                composition_source="textbook_strawberry_puree_cellular_composition",
                spectra_source="fixed_puree_transflectance_albedo_residual_readout",
                provenance_source="exp09_dataset_token_fruitpuree_route",
                constant_status="fixed_mechanistic_prior",
                readout_space=(
                    "strawberry_puree_raw_transflectance_residual_absorbance"
                ),
                calibration_source="none",
                real_stat_source="none",
                threshold_source="none",
            ),
            blocked_reason="",
        ),
    ]
    result = {
        "status": "done",
        "rows": rows,
        "real_runnable_count": 1,
        "real_sentinel_candidate_count": 1,
        "real_selected_count": 1,
        "sentinel_tokens": list(exp09.DEFAULT_SENTINEL_TOKENS),
        "remediation_profile": "r2r_sentinel_matrix_v1",
    }

    md = exp09.render_markdown(
        result=result,
        report_path=tmp_path / "r2r.md",
        csv_path=tmp_path / "r2r.csv",
        n_synthetic_samples=8,
        max_real_samples=8,
        max_sentinel_datasets=1,
        seed=1234,
        sentinel_tokens=list(exp09.DEFAULT_SENTINEL_TOKENS),
        remediation_profile="r2r_sentinel_matrix_v1",
    )

    assert "## R2r FruitPuree Provenance" in md
    assert "FruitPuree rows still dominated by `amplitude_under` = 1/1" in md
    assert "amplitude transfer remains residual and non-gate" in md
    assert "not a B2/B3/B4/B5 gate" in md


def test_render_markdown_emits_r2q_lucas_ph_organic_non_gate_note(
    tmp_path: Path,
) -> None:
    exp09 = _load_exp09_module()
    rows = [
        exp09.MorphologyRow(
            status="compared",
            source="AOM_regression",
            task="regression",
            dataset="LUCAS/LUCAS_pH_Organic_1763_LiuRandomOrganic",
            synthetic_preset="soil",
            comparison_space=exp09.COMPARISON_SPACE,
            n_real_samples=8,
            n_synthetic_samples=8,
            n_wavelengths=4,
            wavelength_min=1100.0,
            wavelength_max=2500.0,
            real_global_mean=0.4,
            synthetic_global_mean=0.4,
            global_mean_delta=0.0,
            real_global_std=0.1,
            synthetic_global_std=0.1,
            global_std_ratio=1.0,
            log10_global_std_ratio=0.0,
            real_amplitude_p50=0.1,
            synthetic_amplitude_p50=0.1,
            amplitude_p50_ratio=1.0,
            log10_amplitude_p50_ratio=0.0,
            real_derivative_std_p50=0.01,
            synthetic_derivative_std_p50=0.01,
            derivative_std_p50_ratio=1.0,
            log10_derivative_std_p50_ratio=0.0,
            mean_curve_corr=-0.05,
            inverted_mean_curve_corr=0.05,
            morphology_gap_score=0.0,
            dominant_morphology_gap="variance_under",
            **_audit_kwargs(),
            **_remediation_kwargs(
                profile="r2q_sentinel_matrix_v1",
                effective_matrix_route="lucas_ph_organic_humic_soil_matrix",
                enabled=True,
                domain_key="environmental_soil",
                concentrations_applied=True,
                spectra_applied=True,
                spectra_rule="lucas_ph_organic_humic_albedo_oh_readout",
                provenance_source="exp09_dataset_token_lucas_ph_organic_route",
                constant_status="fixed_mechanistic_prior",
                readout_space="lucas_ph_organic_raw_soil_apparent_absorbance",
                calibration_source="none",
                real_stat_source="none",
                threshold_source="none",
            ),
            blocked_reason="",
        ),
        exp09.MorphologyRow(
            status="compared",
            source="AOM_regression",
            task="regression",
            dataset="LUCAS/LUCAS_SOC_Cropland_8731_NocitaKS",
            synthetic_preset="soil",
            comparison_space=exp09.COMPARISON_SPACE,
            n_real_samples=8,
            n_synthetic_samples=8,
            n_wavelengths=4,
            wavelength_min=1100.0,
            wavelength_max=2500.0,
            real_global_mean=0.3,
            synthetic_global_mean=0.3,
            global_mean_delta=0.0,
            real_global_std=0.1,
            synthetic_global_std=0.1,
            global_std_ratio=1.0,
            log10_global_std_ratio=0.0,
            real_amplitude_p50=0.1,
            synthetic_amplitude_p50=0.1,
            amplitude_p50_ratio=1.0,
            log10_amplitude_p50_ratio=0.0,
            real_derivative_std_p50=0.01,
            synthetic_derivative_std_p50=0.01,
            derivative_std_p50_ratio=1.0,
            log10_derivative_std_p50_ratio=0.0,
            mean_curve_corr=1.0,
            inverted_mean_curve_corr=-1.0,
            morphology_gap_score=0.0,
            dominant_morphology_gap="variance_under",
            **_audit_kwargs(),
            **_remediation_kwargs(
                profile="r2l_sentinel_matrix_v1",
                effective_matrix_route="lucas_mineral_organic_soil_matrix",
                enabled=True,
                domain_key="environmental_soil",
                concentrations_applied=True,
                spectra_applied=True,
                spectra_rule="lucas_mineral_albedo_absorbance_floor_scatter_readout",
            ),
            blocked_reason="",
        ),
        exp09.MorphologyRow(
            status="compared",
            source="AOM_regression",
            task="regression",
            dataset="PHOSPHORUS/LP_spxyG",
            synthetic_preset="soil",
            comparison_space=exp09.COMPARISON_SPACE,
            n_real_samples=8,
            n_synthetic_samples=8,
            n_wavelengths=4,
            wavelength_min=1100.0,
            wavelength_max=2500.0,
            real_global_mean=0.2,
            synthetic_global_mean=0.2,
            global_mean_delta=0.0,
            real_global_std=0.1,
            synthetic_global_std=0.1,
            global_std_ratio=1.0,
            log10_global_std_ratio=0.0,
            real_amplitude_p50=0.1,
            synthetic_amplitude_p50=0.1,
            amplitude_p50_ratio=1.0,
            log10_amplitude_p50_ratio=0.0,
            real_derivative_std_p50=0.01,
            synthetic_derivative_std_p50=0.01,
            derivative_std_p50_ratio=1.0,
            log10_derivative_std_p50_ratio=0.0,
            mean_curve_corr=1.0,
            inverted_mean_curve_corr=-1.0,
            morphology_gap_score=0.0,
            dominant_morphology_gap="derivative_over",
            **_audit_kwargs(),
            **_remediation_kwargs(
                profile="r2p_sentinel_matrix_v1",
                effective_matrix_route="phosphorus_mineral_soil_matrix",
                enabled=True,
                domain_key="environmental_soil",
                concentrations_applied=True,
                spectra_applied=True,
                spectra_rule="phosphorus_mineral_fertilizer_albedo_residual_readout",
            ),
            blocked_reason="",
        ),
    ]
    result = {
        "status": "done",
        "rows": rows,
        "real_runnable_count": 3,
        "real_sentinel_candidate_count": 3,
        "real_selected_count": 3,
        "sentinel_tokens": list(exp09.DEFAULT_SENTINEL_TOKENS),
        "remediation_profile": "r2q_sentinel_matrix_v1",
    }

    md = exp09.render_markdown(
        result=result,
        report_path=tmp_path / "r2q.md",
        csv_path=tmp_path / "r2q.csv",
        n_synthetic_samples=8,
        max_real_samples=8,
        max_sentinel_datasets=3,
        seed=1234,
        sentinel_tokens=list(exp09.DEFAULT_SENTINEL_TOKENS),
        remediation_profile="r2q_sentinel_matrix_v1",
    )

    assert md.startswith("# R2q Sentinel Morphology Audit")
    assert "## R2q LUCAS pH Organic Provenance" in md
    assert "`lucas_ph_organic_humic_albedo_oh_readout`" in md
    assert "R2q inherits R2p routing and changes only explicit LUCAS pH Organic" in md
    assert "No real LUCAS pH Organic spectra, marginal statistics, covariance/PCA" in md
    assert "LUCAS pH Organic rows dominated by `variance_under` = 1/1" in md
    assert "weak/negative `mean_curve_corr` (<0.2) = 1/1" in md
    assert "Other LUCAS rows preserved on R2l = 1/1" in md
    assert "PHOSPHORUS rows preserved on R2p = 1/1" in md
    assert "Non-target rows accidentally reported on R2q = 0/2" in md
    assert "not a B2/B3/B4/B5 gate" in md
