"""Phase B1 prior predictive checks for executable synthetic dataset runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

import numpy as np

from nirs4all.synthesis.instruments import get_instrument_archetype
from nirs4all.synthesis.measurement_modes import MeasurementMode
from nirs4all.synthesis.prior import NIRSPriorConfig
from nirs4all.synthesis.validation import (
    compute_baseline_curvature,
    compute_derivative_statistics,
    compute_peak_density,
    compute_snr,
)
from nirsyntheticpfn.adapters.builder_adapter import SyntheticDatasetRun

CheckStatus = Literal["passed", "failed", "unsupported", "not_applicable"]
CheckSeverity = Literal["hard", "informational"]
ValidationStatus = Literal["passed", "blocked"]

PHASE_A_GATE_OVERRIDE = "A3_failed_documented"

DEFAULT_PRIOR_CHECK_THRESHOLDS: dict[str, float] = {
    "concentration_sum_tolerance": 1e-6,
    "concentration_min": 0.0,
    "concentration_max": 1.0,
    "regression_min_std": 1e-8,
    "class_min_fraction": 0.10,
    "nonlinear_residual_ratio_min": 0.02,
    "snr_median_min": 3.0,
    "snr_median_max": 1e5,
    "derivative_std_median_min": 1e-7,
    "derivative_std_median_max": 5e-2,
    "baseline_curvature_median_min": 1e-5,
    "baseline_curvature_median_max": 1.0,
    "peak_density_median_min": 5e-2,
    "peak_density_median_max": 12.0,
}


@dataclass(frozen=True)
class PriorCheckResult:
    """Structured result for one prior predictive check."""

    name: str
    status: CheckStatus
    severity: CheckSeverity
    message: str
    metrics: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def blocks_training(self) -> bool:
        return self.severity == "hard" and self.status in {"failed", "unsupported"}

    def to_dict(self) -> dict[str, Any]:
        return cast("dict[str, Any]", _to_builtin(asdict(self)))


@dataclass(frozen=True)
class PriorPredictiveValidation:
    """B1 validation output attached to downstream training decisions."""

    preset: str
    validation_status: ValidationStatus
    downstream_training_status: Literal["allowed", "blocked"]
    phase_a_gate_override: str
    checks: tuple[PriorCheckResult, ...]
    summary: dict[str, Any]

    @property
    def failed_or_blocking_checks(self) -> list[str]:
        return [check.name for check in self.checks if check.blocks_training]

    def to_dict(self) -> dict[str, Any]:
        return cast("dict[str, Any]", _to_builtin(asdict(self)))


def validate_prior_predictive_run(
    run: SyntheticDatasetRun,
    *,
    preset: str = "dataset",
    thresholds: dict[str, float] | None = None,
) -> PriorPredictiveValidation:
    """Run B1 prior predictive checks and return a hard train/block status."""
    resolved_thresholds = {**DEFAULT_PRIOR_CHECK_THRESHOLDS, **(thresholds or {})}
    checks: list[PriorCheckResult] = []

    checks.append(_check_a2_contract(run))
    checks.append(_check_concentrations(run, resolved_thresholds))
    checks.append(_check_concentration_prior_ranges(run))
    checks.append(_check_target_distribution(run, resolved_thresholds))
    checks.append(_check_nonlinear_behavior(run, resolved_thresholds))
    checks.append(_check_wavelengths_and_mode(run))
    checks.extend(_check_spectral_statistics(run, resolved_thresholds))

    status: ValidationStatus = "blocked" if any(check.blocks_training for check in checks) else "passed"
    return PriorPredictiveValidation(
        preset=preset,
        validation_status=status,
        downstream_training_status="allowed" if status == "passed" else "blocked",
        phase_a_gate_override=PHASE_A_GATE_OVERRIDE,
        checks=tuple(checks),
        summary=_summary(run, checks),
    )


def _check_a2_contract(run: SyntheticDatasetRun) -> PriorCheckResult:
    status = str(run.validation_summary.get("status"))
    failures = run.validation_summary.get("failures", [])
    if status == "passed" and not failures:
        return PriorCheckResult(
            name="a2_contract",
            status="passed",
            severity="hard",
            message="A2 dataset contract is passed.",
            details={"validation_summary_status": status},
        )
    return PriorCheckResult(
        name="a2_contract",
        status="failed",
        severity="hard",
        message="A2 dataset contract failed; downstream training is blocked.",
        details={"validation_summary_status": status, "failures": failures},
    )


def _check_concentrations(
    run: SyntheticDatasetRun,
    thresholds: dict[str, float],
) -> PriorCheckResult:
    raw_concentrations = run.latent_metadata.get("concentrations")
    if raw_concentrations is None:
        return PriorCheckResult(
            name="concentration_sums_and_ranges",
            status="unsupported",
            severity="hard",
            message="Latent concentrations are missing; concentration checks cannot run.",
        )

    concentrations = np.asarray(raw_concentrations, dtype=float)
    if concentrations.ndim != 2:
        return PriorCheckResult(
            name="concentration_sums_and_ranges",
            status="failed",
            severity="hard",
            message=f"Concentrations must be 2D, got shape {concentrations.shape}.",
        )

    row_sums = np.sum(concentrations, axis=1)
    min_value = float(np.min(concentrations))
    max_value = float(np.max(concentrations))
    max_sum_error = float(np.max(np.abs(row_sums - 1.0)))
    finite = bool(np.isfinite(concentrations).all())
    passed = (
        finite
        and min_value >= thresholds["concentration_min"] - thresholds["concentration_sum_tolerance"]
        and max_value <= thresholds["concentration_max"] + thresholds["concentration_sum_tolerance"]
        and max_sum_error <= thresholds["concentration_sum_tolerance"]
    )
    return PriorCheckResult(
        name="concentration_sums_and_ranges",
        status="passed" if passed else "failed",
        severity="hard",
        message=(
            "Concentrations are finite, normalized, and within [0, 1]."
            if passed
            else "Concentrations violate finite, normalized, or [0, 1] support checks."
        ),
        metrics={
            "finite": finite,
            "min": min_value,
            "max": max_value,
            "row_sum_min": float(np.min(row_sums)),
            "row_sum_max": float(np.max(row_sums)),
            "max_row_sum_error": max_sum_error,
        },
        thresholds={
            "min": thresholds["concentration_min"],
            "max": thresholds["concentration_max"],
            "row_sum_tolerance": thresholds["concentration_sum_tolerance"],
        },
        details={
            "shape": list(concentrations.shape),
            "row_normalized": run.builder_config.get("concentration_transform", {}).get("row_normalized"),
        },
    )


def _check_concentration_prior_ranges(run: SyntheticDatasetRun) -> PriorCheckResult:
    priors = run.prior_config.get("concentration_prior")
    if not isinstance(priors, dict) or not priors:
        return PriorCheckResult(
            name="concentration_prior_ranges_declared",
            status="unsupported",
            severity="hard",
            message="Concentration prior ranges are missing from the prior config.",
        )

    invalid: list[dict[str, Any]] = []
    for component, prior in priors.items():
        if not isinstance(prior, dict):
            invalid.append({"component": component, "reason": "prior_not_mapping"})
            continue
        try:
            low = float(prior["min_value"])
            high = float(prior["max_value"])
        except Exception:
            invalid.append({"component": component, "reason": "range_not_numeric"})
            continue
        if not np.isfinite([low, high]).all() or low < 0.0 or high > 1.0 or not low < high:
            invalid.append({"component": component, "reason": "invalid_range", "range": [low, high]})

    return PriorCheckResult(
        name="concentration_prior_ranges_declared",
        status="passed" if not invalid else "failed",
        severity="hard",
        message=(
            "Declared component concentration prior ranges are valid probabilities."
            if not invalid
            else "One or more component concentration prior ranges are invalid."
        ),
        metrics={"component_count": len(priors), "invalid_count": len(invalid)},
        details={"invalid": invalid},
    )


def _check_target_distribution(
    run: SyntheticDatasetRun,
    thresholds: dict[str, float],
) -> PriorCheckResult:
    y = np.asarray(run.y)
    target = run.builder_config.get("target", {})
    target_type = target.get("type")
    if y.ndim == 0 or y.shape[0] != run.X.shape[0]:
        return PriorCheckResult(
            name="target_distribution",
            status="failed",
            severity="hard",
            message="Target shape does not match spectra sample count.",
            details={"y_shape": list(y.shape), "X_shape": list(run.X.shape)},
        )
    if not np.isfinite(y).all():
        return PriorCheckResult(
            name="target_distribution",
            status="failed",
            severity="hard",
            message="Target contains non-finite values.",
        )

    if target_type == "regression":
        target_range = target.get("range")
        if not isinstance(target_range, (list, tuple)) or len(target_range) != 2:
            return PriorCheckResult(
                name="target_distribution",
                status="unsupported",
                severity="hard",
                message="Regression target range is missing.",
            )
        low, high = float(target_range[0]), float(target_range[1])
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        y_std = float(np.std(y))
        passed = low <= y_min <= y_max <= high and y_std >= thresholds["regression_min_std"]
        return PriorCheckResult(
            name="target_distribution",
            status="passed" if passed else "failed",
            severity="hard",
            message=(
                "Regression target is finite, variable, and inside the declared range."
                if passed
                else "Regression target violates declared range or variability checks."
            ),
            metrics={"y_min": y_min, "y_max": y_max, "y_std": y_std},
            thresholds={"range": [low, high], "min_std": thresholds["regression_min_std"]},
        )

    if target_type == "classification":
        n_classes = int(target.get("n_classes", 0))
        if not np.all(np.equal(y, np.round(y))):
            return PriorCheckResult(
                name="target_distribution",
                status="failed",
                severity="hard",
                message="Classification target contains non-integer class labels.",
            )
        labels, counts = np.unique(y.astype(int), return_counts=True)
        fractions = counts / counts.sum()
        expected = set(range(n_classes))
        observed = {int(label) for label in labels}
        min_fraction = float(np.min(fractions)) if fractions.size else 0.0
        passed = observed == expected and min_fraction >= thresholds["class_min_fraction"]
        return PriorCheckResult(
            name="target_distribution",
            status="passed" if passed else "failed",
            severity="hard",
            message=(
                "Classification target contains all declared classes with acceptable balance."
                if passed
                else "Classification target is missing classes or is too imbalanced."
            ),
            metrics={
                "observed_classes": sorted(observed),
                "expected_classes": sorted(expected),
                "min_class_fraction": min_fraction,
                "class_counts": {str(int(label)): int(count) for label, count in zip(labels, counts, strict=True)},
            },
            thresholds={"min_class_fraction": thresholds["class_min_fraction"]},
        )

    return PriorCheckResult(
        name="target_distribution",
        status="unsupported",
        severity="hard",
        message=f"Unsupported target type {target_type!r}.",
    )


def _check_nonlinear_behavior(
    run: SyntheticDatasetRun,
    thresholds: dict[str, float],
) -> PriorCheckResult:
    target = run.builder_config.get("target", {})
    if target.get("type") != "regression":
        return PriorCheckResult(
            name="nonlinear_target_behavior",
            status="not_applicable",
            severity="informational",
            message="Nonlinear target behavior is only defined for regression targets.",
        )

    nonlinearity = str(target.get("nonlinearity", "none"))
    if nonlinearity == "none":
        return PriorCheckResult(
            name="nonlinear_target_behavior",
            status="not_applicable",
            severity="informational",
            message="Regression target declares no nonlinearity.",
            details={"declared_nonlinearity": nonlinearity},
        )

    raw_concentrations = run.latent_metadata.get("concentrations")
    if raw_concentrations is None:
        return PriorCheckResult(
            name="nonlinear_target_behavior",
            status="unsupported",
            severity="hard",
            message="Latent concentrations are missing; nonlinear behavior cannot be checked.",
            details={"declared_nonlinearity": nonlinearity},
        )

    concentrations = np.asarray(raw_concentrations, dtype=float)
    component_indices = target.get("component_indices") or list(range(concentrations.shape[1]))
    selected = concentrations[:, [int(idx) for idx in component_indices]]
    y = np.asarray(run.y, dtype=float).reshape(-1)
    design = np.column_stack([np.ones(selected.shape[0]), selected])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ coefficients
    residual_std = float(np.std(y - predicted))
    y_std = float(np.std(y))
    residual_ratio = residual_std / max(y_std, 1e-12)
    passed = residual_ratio >= thresholds["nonlinear_residual_ratio_min"]
    return PriorCheckResult(
        name="nonlinear_target_behavior",
        status="passed" if passed else "failed",
        severity="hard",
        message=(
            "Declared nonlinear regression target leaves measurable non-linear residual."
            if passed
            else "Declared nonlinear regression target is effectively linear under this check."
        ),
        metrics={"linear_residual_ratio": residual_ratio, "target_std": y_std},
        thresholds={"min_residual_ratio": thresholds["nonlinear_residual_ratio_min"]},
        details={"declared_nonlinearity": nonlinearity, "component_indices": list(component_indices)},
    )


def _check_wavelengths_and_mode(run: SyntheticDatasetRun) -> PriorCheckResult:
    wavelengths = np.asarray(run.wavelengths, dtype=float)
    features = run.builder_config.get("features", {})
    instrument_key = str(features.get("instrument", run.metadata.get("instrument", {}).get("key", "")))
    measurement_mode = str(features.get("measurement_mode", run.metadata.get("mode", "")))
    failures: list[str] = []

    if wavelengths.ndim != 1 or wavelengths.size < 2:
        failures.append("wavelength grid must be one-dimensional with at least two values")
    elif not np.isfinite(wavelengths).all() or not np.all(np.diff(wavelengths) > 0):
        failures.append("wavelength grid must be finite and strictly increasing")

    try:
        instrument = get_instrument_archetype(instrument_key)
    except Exception as exc:
        return PriorCheckResult(
            name="wavelengths_and_mode",
            status="unsupported",
            severity="hard",
            message=f"Instrument {instrument_key!r} cannot be resolved: {exc}",
        )

    if wavelengths.size:
        inst_low, inst_high = instrument.wavelength_range
        if float(wavelengths[0]) < inst_low - 1e-9 or float(wavelengths[-1]) > inst_high + 1e-9:
            failures.append("wavelength grid falls outside the instrument range")

    expected_range = features.get("wavelength_range")
    if isinstance(expected_range, (list, tuple)) and len(expected_range) == 2 and wavelengths.size:
        low, high = float(expected_range[0]), float(expected_range[1])
        if float(wavelengths[0]) < low - 1e-9 or float(wavelengths[-1]) > high + 1e-9:
            failures.append("wavelength grid falls outside the builder effective range")
    else:
        failures.append("builder effective wavelength range is missing")

    valid_modes = {mode.value for mode in MeasurementMode}
    if measurement_mode not in valid_modes:
        failures.append(f"unknown measurement mode {measurement_mode!r}")
    elif not _mode_supported_by_category(measurement_mode, instrument.category.value):
        failures.append(
            f"mode {measurement_mode!r} conflicts with instrument category {instrument.category.value!r}"
        )

    return PriorCheckResult(
        name="wavelengths_and_mode",
        status="passed" if not failures else "failed",
        severity="hard",
        message=(
            "Wavelength grid and mode are compatible with the selected instrument."
            if not failures
            else "Wavelength grid or mode conflicts were detected."
        ),
        metrics={
            "wavelength_min": float(wavelengths[0]) if wavelengths.size else None,
            "wavelength_max": float(wavelengths[-1]) if wavelengths.size else None,
            "n_wavelengths": int(wavelengths.size),
        },
        details={
            "instrument": instrument_key,
            "instrument_category": instrument.category.value,
            "measurement_mode": measurement_mode,
            "failures": failures,
        },
    )


def _check_spectral_statistics(
    run: SyntheticDatasetRun,
    thresholds: dict[str, float],
) -> list[PriorCheckResult]:
    X = np.asarray(run.X, dtype=float)
    wavelengths = np.asarray(run.wavelengths, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 5 or wavelengths.size != X.shape[1]:
        return [
            PriorCheckResult(
                name="spectral_statistics",
                status="unsupported",
                severity="hard",
                message="Spectral statistics require a 2D X matrix with at least five wavelengths.",
                details={"X_shape": list(X.shape), "wavelength_shape": list(wavelengths.shape)},
            )
        ]
    if not np.isfinite(X).all():
        return [
            PriorCheckResult(
                name="spectral_statistics",
                status="failed",
                severity="hard",
                message="Spectra contain non-finite values.",
            )
        ]

    snr = compute_snr(X)
    _, derivative_stds = compute_derivative_statistics(X, wavelengths, order=1)
    curvature = compute_baseline_curvature(X)
    peak_density = compute_peak_density(X, wavelengths)

    return [
        _range_check(
            name="spectral_snr",
            values=snr,
            metric_name="median_snr",
            lower=thresholds["snr_median_min"],
            upper=thresholds["snr_median_max"],
            pass_message="Spectral SNR is finite and inside the B1 smoke range.",
            fail_message="Spectral SNR falls outside the B1 smoke range.",
        ),
        _range_check(
            name="derivative_statistics",
            values=derivative_stds,
            metric_name="median_first_derivative_std",
            lower=thresholds["derivative_std_median_min"],
            upper=thresholds["derivative_std_median_max"],
            pass_message="First-derivative variability is inside the B1 smoke range.",
            fail_message="First-derivative variability falls outside the B1 smoke range.",
        ),
        _range_check(
            name="baseline_curvature",
            values=curvature,
            metric_name="median_baseline_curvature",
            lower=thresholds["baseline_curvature_median_min"],
            upper=thresholds["baseline_curvature_median_max"],
            pass_message="Baseline curvature is inside the B1 smoke range.",
            fail_message="Baseline curvature falls outside the B1 smoke range.",
        ),
        _range_check(
            name="peak_density",
            values=peak_density,
            metric_name="median_peak_density_per_100nm",
            lower=thresholds["peak_density_median_min"],
            upper=thresholds["peak_density_median_max"],
            pass_message="Peak density is inside the B1 smoke range.",
            fail_message="Peak density falls outside the B1 smoke range.",
        ),
    ]


def _range_check(
    *,
    name: str,
    values: np.ndarray,
    metric_name: str,
    lower: float,
    upper: float,
    pass_message: str,
    fail_message: str,
) -> PriorCheckResult:
    finite = bool(np.isfinite(values).all())
    median = float(np.median(values)) if values.size else float("nan")
    passed = finite and lower <= median <= upper
    return PriorCheckResult(
        name=name,
        status="passed" if passed else "failed",
        severity="hard",
        message=pass_message if passed else fail_message,
        metrics={
            metric_name: median,
            "finite": finite,
            "q05": float(np.quantile(values, 0.05)) if values.size else None,
            "q95": float(np.quantile(values, 0.95)) if values.size else None,
        },
        thresholds={metric_name: [lower, upper]},
    )


def _mode_supported_by_category(mode: str, category: str) -> bool:
    weights = NIRSPriorConfig().mode_given_category.get(category)
    if weights is None:
        return mode in {"reflectance", "transmittance"}
    return weights.get(mode, 0.0) > 0.0


def _summary(run: SyntheticDatasetRun, checks: list[PriorCheckResult]) -> dict[str, Any]:
    target = run.builder_config.get("target", {})
    wavelengths = np.asarray(run.wavelengths)
    return {
        "domain": run.metadata.get("domain", {}).get("key"),
        "instrument": run.metadata.get("instrument", {}).get("key"),
        "measurement_mode": run.metadata.get("mode"),
        "target_type": target.get("type"),
        "X_shape": list(np.asarray(run.X).shape),
        "y_shape": list(np.asarray(run.y).shape),
        "wavelength_range_nm": [
            float(np.min(wavelengths)),
            float(np.max(wavelengths)),
        ] if wavelengths.size else None,
        "blocking_checks": [check.name for check in checks if check.blocks_training],
        "unsupported_checks": [check.name for check in checks if check.status == "unsupported"],
    }


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value
