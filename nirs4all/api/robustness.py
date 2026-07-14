"""Public audit-only robustness/generalization reports."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from html import escape as html_escape
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from nirs4all.api.result import PredictResult, load_workspace_predict_result
from nirs4all.pipeline.dagml.conformal_contracts import (
    CalibratedPredictionBlock,
    CalibratedRunResult,
    ConformalIntervalBlock,
    ConformalMetricSet,
    evaluate_conformal_prediction,
)
from nirs4all.pipeline.dagml.training_contracts import tcv1_sha256

RobustnessMode = Literal["clean_frozen", "matched_recalibration", "structural_refit"]
ROBUSTNESS_MODES: tuple[RobustnessMode, ...] = (
    "clean_frozen",
    "matched_recalibration",
    "structural_refit",
)
ROBUSTNESS_EXECUTABLE_MODES: tuple[RobustnessMode, ...] = ("clean_frozen",)
RobustnessScenarioKind = Literal[
    "observed",
    "prediction_bias",
    "prediction_noise",
    "spectral_noise",
    "spectral_offset",
    "spectral_scale",
    "spectral_slope",
    "spectral_shift",
]
RobustnessScenarioDistribution = Literal["normal", "uniform"]
ROBUSTNESS_SUMMARY_SCHEMA_ID = "https://nirs4all.org/schemas/robustness-summary/v1"
ROBUSTNESS_SCENARIO_KINDS: tuple[RobustnessScenarioKind, ...] = (
    "observed",
    "prediction_bias",
    "prediction_noise",
    "spectral_noise",
    "spectral_offset",
    "spectral_scale",
    "spectral_slope",
    "spectral_shift",
)
ROBUSTNESS_STOCHASTIC_SCENARIO_KINDS: tuple[RobustnessScenarioKind, ...] = (
    "prediction_noise",
    "spectral_noise",
)
ROBUSTNESS_SCENARIO_DISTRIBUTIONS: tuple[RobustnessScenarioDistribution, ...] = ("normal", "uniform")
_SUPPORTED_SCENARIO_KIND_SET = frozenset(ROBUSTNESS_SCENARIO_KINDS)
_STOCHASTIC_DISTRIBUTION_SCENARIOS = frozenset(ROBUSTNESS_STOCHASTIC_SCENARIO_KINDS)
_SPECTRAL_REPLAY_SCENARIO_KINDS = frozenset(
    (
        "spectral_noise",
        "spectral_offset",
        "spectral_scale",
        "spectral_slope",
        "spectral_shift",
    )
)


_ROBUSTNESS_SUMMARY_SCHEMA: dict[str, Any] = {
    "$id": ROBUSTNESS_SUMMARY_SCHEMA_ID,
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "additionalProperties": False,
    "properties": {
        "conformal_guarantee_status": {
            "additionalProperties": True,
            "properties": {
                "calibrated_coverages": {"type": "array", "items": {"type": "number"}},
                "coverage": {"type": "array", "items": {"type": "number"}},
                "effective_engine": {"type": "string"},
                "guarantee": {"type": "string"},
                "invalidation_reasons": {"type": "array", "items": {"type": "string"}},
                "limitations": {"type": "array", "items": {"type": "string"}},
                "method": {"type": "string"},
                "requested_engine": {"type": "string"},
                "scope": {"type": "string"},
                "status": {"type": "string"},
                "unit": {"type": "string"},
                "version": {"type": "integer", "minimum": 1},
            },
            "type": ["object", "null"],
        },
        "fingerprint": {"type": "string", "minLength": 1},
        "format": {"const": "nirs4all.robustness.summary"},
        "mode": {"enum": ["clean_frozen", "matched_recalibration", "structural_refit"]},
        "report_version": {"type": "integer", "minimum": 1},
        "schema_version": {"const": 1},
        "slice_by": {"type": "array", "items": {"type": "string"}},
        "spectral_replay": {
            "additionalProperties": False,
            "properties": {
                "all_predictions": {"type": "boolean"},
                "predictor_bundle": {"type": "string"},
                "route": {"type": "string"},
                "sample_ids_forwarded": {"type": "boolean"},
                "source": {"enum": ["predictor", "predictor_bundle"]},
            },
            "required": ["route", "sample_ids_forwarded", "source"],
            "type": "object",
        },
        "summary": {
            "type": "array",
            "items": {
                "additionalProperties": True,
                "properties": {
                    "bias": {"type": "number"},
                    "conformal_max_abs_coverage_gap": {"type": ["number", "null"]},
                    "conformal_mean_width_mean": {"type": ["number", "null"]},
                    "conformal_min_observed_coverage": {"type": ["number", "null"]},
                    "delta_bias": {"type": "number"},
                    "delta_mae": {"type": "number"},
                    "delta_max_abs_error": {"type": "number"},
                    "delta_rmse": {"type": "number"},
                    "execution_scope": {"enum": ["baseline", "prediction_replay", "spectral_replay"]},
                    "mae": {"type": "number"},
                    "mae_ratio": {"type": ["number", "null"]},
                    "max_abs_error": {"type": "number"},
                    "n_samples": {"type": "integer", "minimum": 1},
                    "requires_spectral_replay": {"type": "boolean"},
                    "rmse": {"type": "number"},
                    "rmse_ratio": {"type": ["number", "null"]},
                    "scenario": {"type": "object"},
                    "scenario_index": {"type": "integer", "minimum": 0},
                    "scenario_label": {"type": "string"},
                    "severity": {"type": "number"},
                    "worst_slice_key": {"type": ["object", "null"]},
                    "worst_slice_label": {"type": ["string", "null"]},
                    "worst_slice_metric": {"type": "string"},
                    "worst_slice_value": {"type": ["number", "null"]},
                },
                "required": [
                    "bias",
                    "delta_mae",
                    "delta_rmse",
                    "mae",
                    "max_abs_error",
                    "n_samples",
                    "rmse",
                    "scenario",
                    "scenario_index",
                    "scenario_label",
                    "severity",
                ],
                "type": "object",
            },
        },
    },
    "required": ["fingerprint", "format", "mode", "report_version", "schema_version", "slice_by", "summary"],
    "title": "NIRS4All robustness summary artifact",
    "type": "object",
}


def get_robustness_summary_schema() -> dict[str, Any]:
    """Return the JSON Schema for robustness summary artifacts."""

    return cast(dict[str, Any], json.loads(json.dumps(_ROBUSTNESS_SUMMARY_SCHEMA, sort_keys=True)))


def robustness_summary_schema_json(*, indent: int | None = 2) -> str:
    """Serialize the robustness summary JSON Schema deterministically."""

    return (
        json.dumps(
            get_robustness_summary_schema(),
            indent=indent,
            sort_keys=True,
            separators=None if indent is not None else (",", ":"),
        )
        + "\n"
    )


@dataclass(frozen=True)
class RobustnessScenarioSpec:
    """Typed public scenario for ``nirs4all.robustness()`` audit reports."""

    kind: RobustnessScenarioKind | str
    severity: float = 0.0
    distribution: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = _normalize_scenario_kind(self.kind, source="RobustnessScenarioSpec.kind")
        if kind not in _SUPPORTED_SCENARIO_KIND_SET:
            raise NotImplementedError(
                "RobustnessScenarioSpec.kind must be one of 'observed', 'prediction_bias', 'prediction_noise', 'spectral_noise', 'spectral_offset', 'spectral_scale', 'spectral_slope' or 'spectral_shift'"
            )
        if isinstance(self.severity, (bool, np.bool_)) or not isinstance(self.severity, (int, float, np.integer, np.floating)):
            raise ValueError("RobustnessScenarioSpec.severity must be a real numeric scalar")
        severity = float(self.severity)
        if not np.isfinite(severity):
            raise ValueError("RobustnessScenarioSpec.severity must be finite")
        if kind == "observed" and severity != 0.0:
            raise NotImplementedError("observed robustness scenarios require severity=0.0")
        if kind in {"prediction_noise", "spectral_noise"} and severity < 0.0:
            raise ValueError(f"{kind} severity must be non-negative")
        if kind == "spectral_scale" and 1.0 + severity <= 0.0:
            raise ValueError("spectral_scale requires 1.0 + severity to be positive")
        distribution = None
        if self.distribution is not None:
            distribution = _normalize_scenario_distribution(kind, self.distribution, source="RobustnessScenarioSpec.distribution")
        if not isinstance(self.extra, Mapping):
            raise ValueError("RobustnessScenarioSpec.extra must be a mapping")
        extra: dict[str, Any] = {}
        for key, value in self.extra.items():
            if not isinstance(key, str) or not key.strip() or key != key.strip() or "\x00" in key:
                raise ValueError("RobustnessScenarioSpec.extra keys must be canonical non-empty strings")
            if key in {"kind", "severity", "distribution"}:
                raise ValueError(f"RobustnessScenarioSpec.extra must not override {key!r}")
            if key in extra:
                raise ValueError("RobustnessScenarioSpec.extra contains duplicate keys")
            extra[key] = value
        payload: dict[str, Any] = {"kind": kind, "severity": severity, **extra}
        if distribution is not None:
            payload["distribution"] = distribution
        _validate_scenario_payload(payload, source="RobustnessScenarioSpec", validate_kind=True)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(self, "extra", extra)

    def to_dict(self) -> dict[str, Any]:
        """Return the public mapping consumed by ``nirs4all.robustness()``."""

        payload: dict[str, Any] = {"kind": self.kind, "severity": self.severity}
        if self.distribution is not None:
            payload["distribution"] = self.distribution
        for key, value in self.extra.items():
            payload[key] = value
        _validate_scenario_payload(payload, source="RobustnessScenarioSpec", validate_kind=True)
        return payload


@dataclass(frozen=True)
class RobustnessMetricSet:
    """Point-prediction diagnostics for one robustness report cell."""

    n_samples: int
    rmse: float
    mae: float
    bias: float
    max_abs_error: float
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError("RobustnessMetricSet.version must be 1")
        if self.n_samples <= 0:
            raise ValueError("RobustnessMetricSet.n_samples must be positive")
        for name in ("rmse", "mae", "bias", "max_abs_error"):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"RobustnessMetricSet.{name} must be finite")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible metric payload."""

        return {
            "bias": self.bias,
            "fingerprint": self.fingerprint,
            "mae": self.mae,
            "max_abs_error": self.max_abs_error,
            "n_samples": self.n_samples,
            "rmse": self.rmse,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RobustnessMetricSet:
        """Parse and verify a point-metric payload."""

        if not isinstance(payload, Mapping):
            raise TypeError("RobustnessMetricSet payload must be a mapping")
        required = {"bias", "mae", "max_abs_error", "n_samples", "rmse", "version"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"RobustnessMetricSet payload is missing keys {missing}")
        result = cls(
            n_samples=int(payload["n_samples"]),
            rmse=float(payload["rmse"]),
            mae=float(payload["mae"]),
            bias=float(payload["bias"]),
            max_abs_error=float(payload["max_abs_error"]),
            version=int(payload["version"]),
        )
        expected = payload.get("fingerprint")
        if expected is not None and expected != result.fingerprint:
            raise ValueError("RobustnessMetricSet fingerprint mismatch")
        return result

    @property
    def fingerprint(self) -> str:
        """TCV1 fingerprint of the metric payload."""

        return tcv1_sha256(
            {
                "bias": self.bias,
                "mae": self.mae,
                "max_abs_error": self.max_abs_error,
                "n_samples": self.n_samples,
                "rmse": self.rmse,
                "version": self.version,
            }
        )


@dataclass(frozen=True)
class RobustnessSliceResult:
    """Diagnostics for one metadata slice."""

    slice_key: Mapping[str, Any]
    metrics: RobustnessMetricSet
    conformal_metrics: Mapping[float, ConformalMetricSet] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError("RobustnessSliceResult.version must be 1")
        object.__setattr__(
            self,
            "slice_key",
            _strict_json_mapping(self.slice_key, "RobustnessSliceResult.slice_key"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible slice payload."""

        return {
            "conformal_metrics": {str(coverage): metric.to_dict() for coverage, metric in sorted(self.conformal_metrics.items())},
            "fingerprint": self.fingerprint,
            "metrics": self.metrics.to_dict(),
            "slice_key": {key: self.slice_key[key] for key in sorted(self.slice_key)},
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RobustnessSliceResult:
        """Parse and verify one slice payload."""

        if not isinstance(payload, Mapping):
            raise TypeError("RobustnessSliceResult payload must be a mapping")
        required = {"conformal_metrics", "metrics", "slice_key", "version"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"RobustnessSliceResult payload is missing keys {missing}")
        conformal_metrics = _conformal_metrics_from_dict(payload["conformal_metrics"])
        result = cls(
            slice_key=dict(_required_mapping(payload["slice_key"], "RobustnessSliceResult.slice_key")),
            metrics=RobustnessMetricSet.from_dict(_required_mapping(payload["metrics"], "RobustnessSliceResult.metrics")),
            conformal_metrics=conformal_metrics,
            version=int(payload["version"]),
        )
        expected = payload.get("fingerprint")
        if expected is not None and expected != result.fingerprint:
            raise ValueError("RobustnessSliceResult fingerprint mismatch")
        return result

    @property
    def fingerprint(self) -> str:
        """TCV1 fingerprint of the slice payload."""

        return tcv1_sha256(
            {
                "conformal_metrics": {str(coverage): metric.to_dict() for coverage, metric in sorted(self.conformal_metrics.items())},
                "metrics": self.metrics.to_dict(),
                "slice_key": {key: self.slice_key[key] for key in sorted(self.slice_key)},
                "version": self.version,
            }
        )


@dataclass(frozen=True)
class RobustnessScenarioResult:
    """Audit-only result for one scenario/severity cell."""

    scenario: Mapping[str, Any]
    severity: float
    metrics: RobustnessMetricSet
    conformal_metrics: Mapping[float, ConformalMetricSet] = field(default_factory=dict)
    slices: tuple[RobustnessSliceResult, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError("RobustnessScenarioResult.version must be 1")
        if isinstance(self.severity, (bool, np.bool_)) or not isinstance(self.severity, (int, float, np.integer, np.floating)):
            raise ValueError("RobustnessScenarioResult.severity must be a real numeric scalar")
        severity = float(self.severity)
        if not np.isfinite(severity):
            raise ValueError("RobustnessScenarioResult.severity must be finite")
        object.__setattr__(self, "severity", severity)
        object.__setattr__(
            self,
            "scenario",
            _strict_json_mapping(self.scenario, "RobustnessScenarioResult.scenario"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible scenario payload."""

        return {
            "conformal_metrics": {str(coverage): metric.to_dict() for coverage, metric in sorted(self.conformal_metrics.items())},
            "fingerprint": self.fingerprint,
            "metrics": self.metrics.to_dict(),
            "scenario": {key: self.scenario[key] for key in sorted(self.scenario)},
            "severity": self.severity,
            "slices": [slice_result.to_dict() for slice_result in self.slices],
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RobustnessScenarioResult:
        """Parse and verify one scenario payload."""

        if not isinstance(payload, Mapping):
            raise TypeError("RobustnessScenarioResult payload must be a mapping")
        required = {"conformal_metrics", "metrics", "scenario", "severity", "slices", "version"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"RobustnessScenarioResult payload is missing keys {missing}")
        slices = payload["slices"]
        if not isinstance(slices, list):
            raise ValueError("RobustnessScenarioResult.slices must be a list")
        result = cls(
            scenario=dict(_required_mapping(payload["scenario"], "RobustnessScenarioResult.scenario")),
            severity=payload["severity"],
            metrics=RobustnessMetricSet.from_dict(_required_mapping(payload["metrics"], "RobustnessScenarioResult.metrics")),
            conformal_metrics=_conformal_metrics_from_dict(payload["conformal_metrics"]),
            slices=tuple(RobustnessSliceResult.from_dict(_required_mapping(item, "RobustnessScenarioResult.slices[]")) for item in slices),
            version=int(payload["version"]),
        )
        expected = payload.get("fingerprint")
        if expected is not None and expected != result.fingerprint:
            raise ValueError("RobustnessScenarioResult fingerprint mismatch")
        return result

    @property
    def fingerprint(self) -> str:
        """TCV1 fingerprint of the scenario payload."""

        return tcv1_sha256(
            {
                "conformal_metrics": {str(coverage): metric.to_dict() for coverage, metric in sorted(self.conformal_metrics.items())},
                "metrics": self.metrics.to_dict(),
                "scenario": {key: self.scenario[key] for key in sorted(self.scenario)},
                "severity": self.severity,
                "slices": [slice_result.to_dict() for slice_result in self.slices],
                "version": self.version,
            }
        )


@dataclass(frozen=True)
class RobustnessReport:
    """Audit-only robustness/generalization report."""

    mode: RobustnessMode
    scenarios: tuple[RobustnessScenarioResult, ...]
    slice_by: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError("RobustnessReport.version must be 1")
        if self.mode not in ("clean_frozen", "matched_recalibration", "structural_refit"):
            raise ValueError("RobustnessReport.mode is not supported")
        object.__setattr__(
            self,
            "metadata",
            _strict_json_mapping(self.metadata, "RobustnessReport.metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible report payload."""

        return {
            "fingerprint": self.fingerprint,
            "metadata": {key: self.metadata[key] for key in sorted(self.metadata)},
            "mode": self.mode,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "slice_by": list(self.slice_by),
            "version": self.version,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return deterministic JSON with a trailing newline."""

        return (
            json.dumps(
                self.to_dict(),
                indent=indent,
                sort_keys=True,
                separators=None if indent is not None else (",", ":"),
            )
            + "\n"
        )

    def save_json(self, path: str | Path) -> Path:
        """Persist the report as deterministic verified JSON."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    def summary_artifact(self) -> dict[str, Any]:
        """Return the lightweight deterministic summary artifact payload."""

        return _summary_artifact_payload(self)

    def to_summary_json(self, *, indent: int | None = 2) -> str:
        """Return deterministic summary JSON with a trailing newline."""

        return (
            json.dumps(
                self.summary_artifact(),
                indent=indent,
                sort_keys=True,
                separators=None if indent is not None else (",", ":"),
            )
            + "\n"
        )

    def save_summary(self, path: str | Path) -> Path:
        """Persist the lightweight deterministic summary artifact JSON."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_summary_json(), encoding="utf-8")
        return target

    def degradation_rows(self, *, reference: int = 0) -> tuple[dict[str, Any], ...]:
        """Return deterministic per-scenario degradation rows versus a reference."""

        baseline = self._scenario_at(reference)
        rows: list[dict[str, Any]] = []
        for index, scenario in enumerate(self.scenarios):
            rows.append(
                {
                    "delta_bias": scenario.metrics.bias - baseline.metrics.bias,
                    "delta_mae": scenario.metrics.mae - baseline.metrics.mae,
                    "delta_max_abs_error": scenario.metrics.max_abs_error - baseline.metrics.max_abs_error,
                    "delta_rmse": scenario.metrics.rmse - baseline.metrics.rmse,
                    "mae_ratio": _safe_ratio(scenario.metrics.mae, baseline.metrics.mae),
                    "rmse_ratio": _safe_ratio(scenario.metrics.rmse, baseline.metrics.rmse),
                    "scenario": {key: scenario.scenario[key] for key in sorted(scenario.scenario)},
                    "scenario_index": index,
                    "severity": scenario.severity,
                }
            )
        return tuple(rows)

    def worst_slices(self, *, metric: str = "rmse", top_k: int = 1) -> tuple[dict[str, Any], ...]:
        """Return the worst diagnostic slices across all scenarios."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if metric not in {"rmse", "mae", "bias", "abs_bias", "max_abs_error"}:
            raise ValueError("metric must be one of 'rmse', 'mae', 'bias', 'abs_bias', or 'max_abs_error'")
        rows: list[dict[str, Any]] = []
        for scenario_index, scenario in enumerate(self.scenarios):
            for slice_result in scenario.slices:
                value = _slice_metric_value(slice_result.metrics, metric)
                rows.append(
                    {
                        "metric": metric,
                        "scenario": {key: scenario.scenario[key] for key in sorted(scenario.scenario)},
                        "scenario_index": scenario_index,
                        "severity": scenario.severity,
                        "slice_key": {key: slice_result.slice_key[key] for key in sorted(slice_result.slice_key)},
                        "value": value,
                    }
                )
        rows.sort(key=lambda row: (-float(row["value"]), int(row["scenario_index"]), _slice_label(row["slice_key"])))
        return tuple(rows[:top_k])

    def summary_rows(self, *, reference: int = 0, worst_slice_metric: str = "rmse") -> tuple[dict[str, Any], ...]:
        """Return compact per-scenario rows for CI dashboards and Studio."""

        if worst_slice_metric not in {"rmse", "mae", "bias", "abs_bias", "max_abs_error"}:
            raise ValueError("worst_slice_metric must be one of 'rmse', 'mae', 'bias', 'abs_bias', or 'max_abs_error'")
        degradation_by_index = {int(row["scenario_index"]): row for row in self.degradation_rows(reference=reference)}
        rows: list[dict[str, Any]] = []
        for scenario_index, scenario in enumerate(self.scenarios):
            conformal_summary = _scenario_conformal_summary(scenario)
            worst_slice = _scenario_worst_slice(scenario, metric=worst_slice_metric)
            degradation = degradation_by_index[scenario_index]
            rows.append(
                {
                    "bias": scenario.metrics.bias,
                    "conformal_max_abs_coverage_gap": conformal_summary["max_abs_coverage_gap"],
                    "conformal_mean_width_mean": conformal_summary["mean_width_mean"],
                    "conformal_min_observed_coverage": conformal_summary["min_observed_coverage"],
                    "delta_bias": degradation["delta_bias"],
                    "delta_mae": degradation["delta_mae"],
                    "delta_max_abs_error": degradation["delta_max_abs_error"],
                    "delta_rmse": degradation["delta_rmse"],
                    "execution_scope": _scenario_execution_scope(scenario.scenario),
                    "mae": scenario.metrics.mae,
                    "mae_ratio": degradation["mae_ratio"],
                    "max_abs_error": scenario.metrics.max_abs_error,
                    "n_samples": scenario.metrics.n_samples,
                    "requires_spectral_replay": _scenario_requires_spectral_replay(scenario.scenario),
                    "rmse": scenario.metrics.rmse,
                    "rmse_ratio": degradation["rmse_ratio"],
                    "scenario": {key: scenario.scenario[key] for key in sorted(scenario.scenario)},
                    "scenario_index": scenario_index,
                    "scenario_label": _scenario_label(scenario.scenario),
                    "severity": scenario.severity,
                    "worst_slice_key": worst_slice["slice_key"],
                    "worst_slice_label": worst_slice["slice_label"],
                    "worst_slice_metric": worst_slice_metric,
                    "worst_slice_value": worst_slice["value"],
                }
            )
        return tuple(rows)

    def tabular_records(self) -> dict[str, tuple[dict[str, Any], ...]]:
        """Return deterministic flat records used by tabular exports."""

        scenario_rows: list[dict[str, Any]] = []
        conformal_rows: list[dict[str, Any]] = []
        slice_rows: list[dict[str, Any]] = []
        slice_conformal_rows: list[dict[str, Any]] = []
        for scenario_index, scenario in enumerate(self.scenarios):
            scenario_rows.append(
                {
                    "bias": scenario.metrics.bias,
                    "mae": scenario.metrics.mae,
                    "max_abs_error": scenario.metrics.max_abs_error,
                    "n_samples": scenario.metrics.n_samples,
                    "rmse": scenario.metrics.rmse,
                    "scenario_fingerprint": scenario.fingerprint,
                    "scenario_index": scenario_index,
                    "scenario_json": _json_cell(scenario.scenario),
                    "scenario_kind": str(scenario.scenario.get("kind", "scenario")),
                    "scenario_label": _scenario_label(scenario.scenario),
                    "severity": scenario.severity,
                }
            )
            for coverage, metric in sorted(scenario.conformal_metrics.items()):
                conformal_rows.append(
                    {
                        "coverage": float(coverage),
                        "coverage_gap": metric.coverage_gap,
                        "mean_interval_score": metric.mean_interval_score,
                        "mean_width": metric.mean_width,
                        "median_width": metric.median_width,
                        "n_covered": metric.n_covered,
                        "n_missed_above": metric.n_missed_above,
                        "n_missed_below": metric.n_missed_below,
                        "n_samples": metric.n_samples,
                        "observed_coverage": metric.observed_coverage,
                        "scenario_index": scenario_index,
                        "scenario_json": _json_cell(scenario.scenario),
                        "scenario_label": _scenario_label(scenario.scenario),
                        "severity": scenario.severity,
                    }
                )
            for slice_result in scenario.slices:
                slice_rows.append(
                    {
                        "bias": slice_result.metrics.bias,
                        "mae": slice_result.metrics.mae,
                        "max_abs_error": slice_result.metrics.max_abs_error,
                        "n_samples": slice_result.metrics.n_samples,
                        "rmse": slice_result.metrics.rmse,
                        "scenario_index": scenario_index,
                        "scenario_json": _json_cell(scenario.scenario),
                        "scenario_label": _scenario_label(scenario.scenario),
                        "severity": scenario.severity,
                        "slice_key_json": _json_cell(slice_result.slice_key),
                        "slice_label": _slice_label(slice_result.slice_key),
                    }
                )
                for coverage, metric in sorted(slice_result.conformal_metrics.items()):
                    slice_conformal_rows.append(
                        {
                            "coverage": float(coverage),
                            "coverage_gap": metric.coverage_gap,
                            "mean_interval_score": metric.mean_interval_score,
                            "mean_width": metric.mean_width,
                            "median_width": metric.median_width,
                            "n_covered": metric.n_covered,
                            "n_missed_above": metric.n_missed_above,
                            "n_missed_below": metric.n_missed_below,
                            "n_samples": metric.n_samples,
                            "observed_coverage": metric.observed_coverage,
                            "scenario_index": scenario_index,
                            "scenario_json": _json_cell(scenario.scenario),
                            "scenario_label": _scenario_label(scenario.scenario),
                            "severity": scenario.severity,
                            "slice_key_json": _json_cell(slice_result.slice_key),
                            "slice_label": _slice_label(slice_result.slice_key),
                        }
                    )
        worst_rows = []
        for row in self.worst_slices(metric="rmse", top_k=max(1, sum(len(scenario.slices) for scenario in self.scenarios))):
            worst_rows.append(
                {
                    "metric": row["metric"],
                    "scenario_index": row["scenario_index"],
                    "scenario_json": _json_cell(row["scenario"]),
                    "scenario_label": _scenario_label(row["scenario"]),
                    "severity": row["severity"],
                    "slice_key_json": _json_cell(row["slice_key"]),
                    "slice_label": _slice_label(row["slice_key"]),
                    "value": row["value"],
                }
            )
        degradation_rows = []
        for row in self.degradation_rows():
            flat = {key: value for key, value in row.items() if key != "scenario"}
            flat["scenario_json"] = _json_cell(row["scenario"])
            flat["scenario_label"] = _scenario_label(row["scenario"])
            degradation_rows.append(flat)
        summary_rows = []
        for row in self.summary_rows():
            flat = {key: value for key, value in row.items() if key not in {"scenario", "worst_slice_key"}}
            flat["scenario_json"] = _json_cell(row["scenario"])
            flat["worst_slice_key_json"] = _json_cell(row["worst_slice_key"]) if row["worst_slice_key"] is not None else None
            summary_rows.append(flat)
        return {
            "conformal_diagnostics": tuple(conformal_rows),
            "degradation": tuple(degradation_rows),
            "scenario_metrics": tuple(scenario_rows),
            "summary": tuple(summary_rows),
            "slice_conformal_diagnostics": tuple(slice_conformal_rows),
            "slices": tuple(slice_rows),
            "worst_slices": tuple(worst_rows),
        }

    def _scenario_at(self, reference: int) -> RobustnessScenarioResult:
        if not self.scenarios:
            raise ValueError("RobustnessReport requires at least one scenario")
        try:
            return self.scenarios[reference]
        except IndexError as exc:
            raise ValueError("reference scenario index is out of range") from exc

    def to_markdown(self) -> str:
        """Return a deterministic Markdown summary of the report."""

        lines = [
            "# NIRS4All robustness report",
            "",
            "## Summary",
            "",
            f"- Mode: `{self.mode}`",
            f"- Version: `{self.version}`",
            f"- Fingerprint: `{self.fingerprint}`",
            f"- Audit-only: `{_markdown_bool(self.metadata.get('audit_only'))}`",
        ]
        if self.slice_by:
            lines.append(f"- Slice keys: `{', '.join(self.slice_by)}`")
        if self.metadata.get("seed") is not None:
            lines.append(f"- Seed: `{self.metadata['seed']}`")

        guarantee = self.metadata.get("conformal_guarantee_status")
        if isinstance(guarantee, Mapping):
            lines.extend(_markdown_guarantee_section(guarantee))

        summary_rows = self.summary_rows()
        if summary_rows:
            lines.extend(
                [
                    "",
                    "## Scenario summary",
                    "",
                    "| Scenario | Scope | Replay evidence | Severity | n | RMSE | ΔRMSE | RMSE ratio | Min coverage | Max abs coverage gap | Worst slice | Worst slice RMSE |",
                    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
                ]
            )
            for row in summary_rows:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _markdown_cell(str(row["scenario_label"])),
                            _markdown_cell(_format_execution_scope(str(row.get("execution_scope", "")))),
                            "required" if row.get("requires_spectral_replay") else "not required",
                            _format_number(float(row["severity"])),
                            str(row["n_samples"]),
                            _format_number(float(row["rmse"])),
                            _format_number(float(row["delta_rmse"])),
                            _format_optional_number(row["rmse_ratio"]),
                            _format_optional_number(row["conformal_min_observed_coverage"]),
                            _format_optional_number(row["conformal_max_abs_coverage_gap"]),
                            _markdown_cell(str(row["worst_slice_label"] or "n/a")),
                            _format_optional_number(row["worst_slice_value"]),
                        ]
                    )
                    + " |"
                )

        lines.extend(
            [
                "",
                "## Scenario metrics",
                "",
                "| Scenario | Severity | n | RMSE | MAE | Bias | Max abs error |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for scenario in self.scenarios:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown_cell(_scenario_label(scenario.scenario)),
                        _format_number(scenario.severity),
                        str(scenario.metrics.n_samples),
                        _format_number(scenario.metrics.rmse),
                        _format_number(scenario.metrics.mae),
                        _format_number(scenario.metrics.bias),
                        _format_number(scenario.metrics.max_abs_error),
                    ]
                )
                + " |"
            )

        degradation_rows = self.degradation_rows()
        if degradation_rows:
            lines.extend(
                [
                    "",
                    "## Degradation versus reference scenario",
                    "",
                    "| Scenario | Severity | ΔRMSE | RMSE ratio | ΔMAE | MAE ratio | ΔBias | ΔMax abs error |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in degradation_rows:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _markdown_cell(_scenario_label(row["scenario"])),
                            _format_number(float(row["severity"])),
                            _format_number(float(row["delta_rmse"])),
                            _format_optional_number(row["rmse_ratio"]),
                            _format_number(float(row["delta_mae"])),
                            _format_optional_number(row["mae_ratio"]),
                            _format_number(float(row["delta_bias"])),
                            _format_number(float(row["delta_max_abs_error"])),
                        ]
                    )
                    + " |"
                )

        conformal_rows = [(scenario, coverage, metric) for scenario in self.scenarios for coverage, metric in sorted(scenario.conformal_metrics.items())]
        if conformal_rows:
            lines.extend(
                [
                    "",
                    "## Conformal diagnostics",
                    "",
                    "| Scenario | Severity | Coverage | Observed | Gap | Mean width | Median width | Interval score | Covered | Missed below | Missed above |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for scenario, coverage, metric in conformal_rows:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _markdown_cell(_scenario_label(scenario.scenario)),
                            _format_number(scenario.severity),
                            _format_number(coverage),
                            _format_number(metric.observed_coverage),
                            _format_number(metric.coverage_gap),
                            _format_number(metric.mean_width),
                            _format_number(metric.median_width),
                            _format_number(metric.mean_interval_score),
                            str(metric.n_covered),
                            str(metric.n_missed_below),
                            str(metric.n_missed_above),
                        ]
                    )
                    + " |"
                )

        slice_rows = [(scenario, slice_result) for scenario in self.scenarios for slice_result in scenario.slices]
        if slice_rows:
            lines.extend(
                [
                    "",
                    "## Diagnostic slices",
                    "",
                    "| Scenario | Severity | Slice | n | RMSE | MAE | Bias | Max abs error |",
                    "|---|---:|---|---:|---:|---:|---:|---:|",
                ]
            )
            for scenario, slice_result in slice_rows:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _markdown_cell(_scenario_label(scenario.scenario)),
                            _format_number(scenario.severity),
                            _markdown_cell(_slice_label(slice_result.slice_key)),
                            str(slice_result.metrics.n_samples),
                            _format_number(slice_result.metrics.rmse),
                            _format_number(slice_result.metrics.mae),
                            _format_number(slice_result.metrics.bias),
                            _format_number(slice_result.metrics.max_abs_error),
                        ]
                    )
                    + " |"
                )

        worst_slices = self.worst_slices(top_k=3) if any(scenario.slices for scenario in self.scenarios) else ()
        if worst_slices:
            lines.extend(
                [
                    "",
                    "## Worst diagnostic slices",
                    "",
                    "| Scenario | Severity | Slice | Metric | Value |",
                    "|---|---:|---|---|---:|",
                ]
            )
            for row in worst_slices:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _markdown_cell(_scenario_label(row["scenario"])),
                            _format_number(float(row["severity"])),
                            _markdown_cell(_slice_label(row["slice_key"])),
                            _markdown_cell(str(row["metric"])),
                            _format_number(float(row["value"])),
                        ]
                    )
                    + " |"
                )

        slice_conformal_rows = [(scenario, slice_result, coverage, metric) for scenario in self.scenarios for slice_result in scenario.slices for coverage, metric in sorted(slice_result.conformal_metrics.items())]
        if slice_conformal_rows:
            lines.extend(
                [
                    "",
                    "## Slice conformal diagnostics",
                    "",
                    "| Scenario | Severity | Slice | Coverage | Observed | Gap | Mean width | Covered | Missed below | Missed above |",
                    "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for scenario, slice_result, coverage, metric in slice_conformal_rows:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _markdown_cell(_scenario_label(scenario.scenario)),
                            _format_number(scenario.severity),
                            _markdown_cell(_slice_label(slice_result.slice_key)),
                            _format_number(coverage),
                            _format_number(metric.observed_coverage),
                            _format_number(metric.coverage_gap),
                            _format_number(metric.mean_width),
                            str(metric.n_covered),
                            str(metric.n_missed_below),
                            str(metric.n_missed_above),
                        ]
                    )
                    + " |"
                )

        return "\n".join(lines) + "\n"

    def save_markdown(self, path: str | Path) -> Path:
        """Persist the report as a deterministic Markdown summary."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_markdown(), encoding="utf-8")
        return target

    def to_html(self) -> str:
        """Return a deterministic standalone HTML summary of the report."""

        lines = [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>NIRS4All robustness report</title>",
            "<style>",
            "body{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;line-height:1.45;margin:2rem;max-width:1200px}",
            "table{border-collapse:collapse;margin:1rem 0;width:100%}",
            "th,td{border:1px solid #d0d7de;padding:0.35rem 0.5rem;text-align:left}",
            "th{background:#f6f8fa}",
            "td.num{text-align:right;font-variant-numeric:tabular-nums}",
            "code{background:#f6f8fa;padding:0.1rem 0.25rem;border-radius:0.25rem}",
            "</style>",
            "</head>",
            "<body>",
            "<h1>NIRS4All robustness report</h1>",
            "<h2>Summary</h2>",
            "<ul>",
            f"<li>Mode: <code>{_html_text(self.mode)}</code></li>",
            f"<li>Version: <code>{self.version}</code></li>",
            f"<li>Fingerprint: <code>{_html_text(self.fingerprint)}</code></li>",
            f"<li>Audit-only: <code>{_html_text(_markdown_bool(self.metadata.get('audit_only')))}</code></li>",
        ]
        if self.slice_by:
            lines.append(f"<li>Slice keys: <code>{_html_text(', '.join(self.slice_by))}</code></li>")
        if self.metadata.get("seed") is not None:
            lines.append(f"<li>Seed: <code>{_html_text(str(self.metadata['seed']))}</code></li>")
        lines.append("</ul>")

        guarantee = self.metadata.get("conformal_guarantee_status")
        if isinstance(guarantee, Mapping):
            lines.extend(_html_guarantee_section(guarantee))

        summary_rows = self.summary_rows()
        if summary_rows:
            lines.extend(
                [
                    "<h2>Scenario summary</h2>",
                    _html_table(
                        [
                            "Scenario",
                            "Scope",
                            "Replay evidence",
                            "Severity",
                            "n",
                            "RMSE",
                            "ΔRMSE",
                            "RMSE ratio",
                            "Min coverage",
                            "Max abs coverage gap",
                            "Worst slice",
                            "Worst slice RMSE",
                        ],
                        [
                            [
                                str(row["scenario_label"]),
                                _format_execution_scope(str(row.get("execution_scope", ""))),
                                "required" if row.get("requires_spectral_replay") else "not required",
                                _format_number(float(row["severity"])),
                                str(row["n_samples"]),
                                _format_number(float(row["rmse"])),
                                _format_number(float(row["delta_rmse"])),
                                _format_optional_number(row["rmse_ratio"]),
                                _format_optional_number(row["conformal_min_observed_coverage"]),
                                _format_optional_number(row["conformal_max_abs_coverage_gap"]),
                                str(row["worst_slice_label"] or "n/a"),
                                _format_optional_number(row["worst_slice_value"]),
                            ]
                            for row in summary_rows
                        ],
                        numeric_columns={1, 2, 3, 4, 5, 6, 7, 9},
                    ),
                ]
            )

        lines.extend(
            [
                "<h2>Scenario metrics</h2>",
                _html_table(
                    ["Scenario", "Severity", "n", "RMSE", "MAE", "Bias", "Max abs error"],
                    [
                        [
                            _scenario_label(scenario.scenario),
                            _format_number(scenario.severity),
                            str(scenario.metrics.n_samples),
                            _format_number(scenario.metrics.rmse),
                            _format_number(scenario.metrics.mae),
                            _format_number(scenario.metrics.bias),
                            _format_number(scenario.metrics.max_abs_error),
                        ]
                        for scenario in self.scenarios
                    ],
                    numeric_columns={1, 2, 3, 4, 5, 6},
                ),
                "<h2>Degradation versus reference scenario</h2>",
                _html_table(
                    ["Scenario", "Severity", "ΔRMSE", "RMSE ratio", "ΔMAE", "MAE ratio", "ΔBias", "ΔMax abs error"],
                    [
                        [
                            _scenario_label(row["scenario"]),
                            _format_number(float(row["severity"])),
                            _format_number(float(row["delta_rmse"])),
                            _format_optional_number(row["rmse_ratio"]),
                            _format_number(float(row["delta_mae"])),
                            _format_optional_number(row["mae_ratio"]),
                            _format_number(float(row["delta_bias"])),
                            _format_number(float(row["delta_max_abs_error"])),
                        ]
                        for row in self.degradation_rows()
                    ],
                    numeric_columns={1, 2, 3, 4, 5, 6, 7},
                ),
            ]
        )

        conformal_rows = [(scenario, coverage, metric) for scenario in self.scenarios for coverage, metric in sorted(scenario.conformal_metrics.items())]
        if conformal_rows:
            lines.extend(
                [
                    "<h2>Conformal diagnostics</h2>",
                    _html_table(
                        ["Scenario", "Severity", "Coverage", "Observed", "Gap", "Mean width", "Median width", "Interval score", "Covered", "Missed below", "Missed above"],
                        [
                            [
                                _scenario_label(scenario.scenario),
                                _format_number(scenario.severity),
                                _format_number(coverage),
                                _format_number(metric.observed_coverage),
                                _format_number(metric.coverage_gap),
                                _format_number(metric.mean_width),
                                _format_number(metric.median_width),
                                _format_number(metric.mean_interval_score),
                                str(metric.n_covered),
                                str(metric.n_missed_below),
                                str(metric.n_missed_above),
                            ]
                            for scenario, coverage, metric in conformal_rows
                        ],
                        numeric_columns={1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                    ),
                ]
            )

        slice_rows = [(scenario, slice_result) for scenario in self.scenarios for slice_result in scenario.slices]
        if slice_rows:
            lines.extend(
                [
                    "<h2>Diagnostic slices</h2>",
                    _html_table(
                        ["Scenario", "Severity", "Slice", "n", "RMSE", "MAE", "Bias", "Max abs error"],
                        [
                            [
                                _scenario_label(scenario.scenario),
                                _format_number(scenario.severity),
                                _slice_label(slice_result.slice_key),
                                str(slice_result.metrics.n_samples),
                                _format_number(slice_result.metrics.rmse),
                                _format_number(slice_result.metrics.mae),
                                _format_number(slice_result.metrics.bias),
                                _format_number(slice_result.metrics.max_abs_error),
                            ]
                            for scenario, slice_result in slice_rows
                        ],
                        numeric_columns={1, 3, 4, 5, 6, 7},
                    ),
                    "<h2>Worst diagnostic slices</h2>",
                    _html_table(
                        ["Scenario", "Severity", "Slice", "Metric", "Value"],
                        [
                            [
                                _scenario_label(row["scenario"]),
                                _format_number(float(row["severity"])),
                                _slice_label(row["slice_key"]),
                                str(row["metric"]),
                                _format_number(float(row["value"])),
                            ]
                            for row in self.worst_slices(top_k=3)
                        ],
                        numeric_columns={1, 4},
                    ),
                ]
            )

        slice_conformal_rows = [(scenario, slice_result, coverage, metric) for scenario in self.scenarios for slice_result in scenario.slices for coverage, metric in sorted(slice_result.conformal_metrics.items())]
        if slice_conformal_rows:
            lines.extend(
                [
                    "<h2>Slice conformal diagnostics</h2>",
                    _html_table(
                        ["Scenario", "Severity", "Slice", "Coverage", "Observed", "Gap", "Mean width", "Covered", "Missed below", "Missed above"],
                        [
                            [
                                _scenario_label(scenario.scenario),
                                _format_number(scenario.severity),
                                _slice_label(slice_result.slice_key),
                                _format_number(coverage),
                                _format_number(metric.observed_coverage),
                                _format_number(metric.coverage_gap),
                                _format_number(metric.mean_width),
                                str(metric.n_covered),
                                str(metric.n_missed_below),
                                str(metric.n_missed_above),
                            ]
                            for scenario, slice_result, coverage, metric in slice_conformal_rows
                        ],
                        numeric_columns={1, 3, 4, 5, 6, 7, 8, 9},
                    ),
                ]
            )

        lines.extend(["</body>", "</html>"])
        return "\n".join(lines) + "\n"

    def save_html(self, path: str | Path) -> Path:
        """Persist the report as a deterministic standalone HTML summary."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_html(), encoding="utf-8")
        return target

    def save_parquet(self, path: str | Path) -> Path:
        """Persist flat report tables as a portable Parquet directory."""

        try:
            import polars as pl
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ImportError("polars and pyarrow are required for RobustnessReport.save_parquet()") from exc

        target = Path(path)
        if target.exists() and not target.is_dir():
            raise ValueError("save_parquet() target must be a directory path; remove the existing file or choose another path")
        target.mkdir(parents=True, exist_ok=True)
        tables = self.tabular_records()
        written: dict[str, str] = {}
        table_metadata: dict[str, dict[str, Any]] = {}
        for table_name in sorted(tables):
            rows = tables[table_name]
            if not rows:
                continue
            filename = f"{table_name}.parquet"
            row_list = list(rows)
            pl.DataFrame(row_list).write_parquet(target / filename)
            written[table_name] = filename
            table_metadata[table_name] = {
                "file": filename,
                "fingerprint": _table_fingerprint(table_name, row_list),
                "rows": len(row_list),
            }
        (target / "report.json").write_text(self.to_json(), encoding="utf-8")
        manifest = {
            "fingerprint": self.fingerprint,
            "format": "nirs4all.robustness.parquet-directory",
            "mode": self.mode,
            "report_json": "report.json",
            "schema_version": 1,
            "table_metadata": table_metadata,
            "tables": written,
            "version": self.version,
        }
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    def save_artifacts(self, path: str | Path, *, formats: Sequence[str] = ("json", "summary", "markdown", "html", "parquet")) -> dict[str, Path]:
        """Persist a complete deterministic artifact directory with manifest."""

        supported = {
            "html": ("report.html", self.save_html),
            "json": ("report.json", self.save_json),
            "markdown": ("report.md", self.save_markdown),
            "parquet": ("report.parquet", self.save_parquet),
            "summary": ("summary.json", None),
        }
        selected: list[str] = []
        for item in formats:
            name = str(item).strip().lower()
            if not name:
                raise ValueError("save_artifacts() formats must not contain empty names")
            if name not in supported:
                raise ValueError(f"unsupported robustness artifact format: {name!r}")
            if name in selected:
                raise ValueError(f"duplicate robustness artifact format: {name!r}")
            selected.append(name)
        if not selected:
            raise ValueError("save_artifacts() requires at least one artifact format")

        target = Path(path)
        if target.exists() and not target.is_dir():
            raise ValueError("save_artifacts() target must be a directory path; remove the existing file or choose another path")
        target.mkdir(parents=True, exist_ok=True)

        artifacts: dict[str, Path] = {}
        files: dict[str, str] = {}
        for name in selected:
            filename, writer = supported[name]
            artifact_path = target / filename
            if writer is None:
                artifact_path = self.save_summary(artifact_path)
            else:
                artifact_path = writer(artifact_path)
            artifacts[name] = artifact_path
            files[name] = filename

        manifest = {
            "files": files,
            "fingerprint": self.fingerprint,
            "format": "nirs4all.robustness.artifact-directory",
            "formats": selected,
            "mode": self.mode,
            "report_version": self.version,
            "schema_version": 1,
        }
        manifest_path = target / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifacts["manifest"] = manifest_path
        return artifacts

    @classmethod
    def load_artifacts(cls, path: str | Path) -> RobustnessReport:
        """Load and verify a deterministic artifact directory export."""

        source = Path(path)
        if not source.is_dir():
            raise ValueError("load_artifacts() requires a robustness artifact-directory export")
        manifest_path = source / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("RobustnessReport artifact directory requires manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "nirs4all.robustness.artifact-directory":
            raise ValueError("unsupported robustness artifact directory format")
        if int(manifest.get("schema_version", -1)) != 1:
            raise ValueError("unsupported robustness artifact directory schema_version")

        files = manifest.get("files")
        if not isinstance(files, Mapping):
            raise ValueError("RobustnessReport artifact manifest files must be a mapping")
        formats = manifest.get("formats")
        if not isinstance(formats, list) or not all(isinstance(item, str) for item in formats):
            raise ValueError("RobustnessReport artifact manifest formats must be a list of strings")
        if len(set(formats)) != len(formats):
            raise ValueError("RobustnessReport artifact manifest formats must not contain duplicates")
        supported = {"html", "json", "markdown", "parquet", "summary"}
        format_set = set(formats)
        file_set = {str(key) for key in files}
        unknown = sorted((format_set | file_set) - supported)
        if unknown:
            raise ValueError(f"unsupported robustness artifact manifest formats: {unknown}")
        if format_set != file_set:
            raise ValueError("RobustnessReport artifact manifest formats and files keys must match")

        def artifact_path(name: str) -> Path:
            return _resolve_relative_child_path(
                source,
                files[name],
                context="RobustnessReport artifact manifest file paths",
            )

        report: RobustnessReport | None = None
        if "json" in file_set:
            json_path = artifact_path("json")
            if not json_path.is_file():
                raise FileNotFoundError("RobustnessReport artifact JSON file is missing")
            report = cls.load_json(json_path)
        if "parquet" in file_set:
            parquet_path = artifact_path("parquet")
            parquet_report = cls.load_parquet(parquet_path)
            if report is None:
                report = parquet_report
            elif parquet_report.to_dict() != report.to_dict():
                raise ValueError("RobustnessReport artifact Parquet report mismatch")
        if report is None:
            raise FileNotFoundError("RobustnessReport artifact directory requires a JSON or Parquet report artifact")

        if manifest.get("fingerprint") != report.fingerprint:
            raise ValueError("RobustnessReport artifact manifest fingerprint mismatch")
        if str(manifest.get("mode", "")) != report.mode:
            raise ValueError("RobustnessReport artifact manifest mode mismatch")
        if int(manifest.get("report_version", -1)) != report.version:
            raise ValueError("RobustnessReport artifact manifest report_version mismatch")

        if "markdown" in file_set:
            markdown_path = artifact_path("markdown")
            if not markdown_path.is_file():
                raise FileNotFoundError("RobustnessReport artifact Markdown file is missing")
            if markdown_path.read_text(encoding="utf-8") != report.to_markdown():
                raise ValueError("RobustnessReport artifact Markdown file mismatch")
        if "html" in file_set:
            html_path = artifact_path("html")
            if not html_path.is_file():
                raise FileNotFoundError("RobustnessReport artifact HTML file is missing")
            if html_path.read_text(encoding="utf-8") != report.to_html():
                raise ValueError("RobustnessReport artifact HTML file mismatch")
        if "summary" in file_set:
            summary_path = artifact_path("summary")
            if not summary_path.is_file():
                raise FileNotFoundError("RobustnessReport artifact summary file is missing")
            if summary_path.read_text(encoding="utf-8") != report.to_summary_json():
                raise ValueError("RobustnessReport artifact summary file mismatch")
        return report

    @classmethod
    def load_parquet(cls, path: str | Path) -> RobustnessReport:
        """Load a Parquet-directory export and verify the embedded report."""

        source = Path(path)
        if not source.is_dir():
            raise ValueError("load_parquet() requires a Parquet-directory export")
        manifest_path = source / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("Parquet-directory export requires manifest.json and report.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "nirs4all.robustness.parquet-directory":
            raise ValueError("unsupported robustness Parquet export format")
        if int(manifest.get("schema_version", -1)) != 1:
            raise ValueError("unsupported robustness Parquet export schema_version")
        report_path = _resolve_relative_child_path(
            source,
            manifest.get("report_json", ""),
            context="RobustnessReport Parquet manifest report_json path",
        )
        if not report_path.is_file():
            raise FileNotFoundError("Parquet-directory export requires manifest.json and report.json")
        result = cls.load_json(report_path)
        if manifest.get("fingerprint") != result.fingerprint:
            raise ValueError("RobustnessReport Parquet manifest fingerprint mismatch")
        tables = manifest.get("tables")
        if tables is not None:
            if not isinstance(tables, Mapping):
                raise ValueError("RobustnessReport Parquet manifest tables must be a mapping")
            for filename in tables.values():
                _resolve_relative_child_path(
                    source,
                    filename,
                    context="RobustnessReport Parquet manifest table file paths",
                )
        table_metadata = manifest.get("table_metadata")
        if table_metadata is not None:
            if not isinstance(table_metadata, Mapping):
                raise ValueError("RobustnessReport Parquet manifest table_metadata must be a mapping")
            if tables is not None and set(table_metadata) != {str(key) for key in tables}:
                raise ValueError("RobustnessReport Parquet manifest table_metadata and tables keys must match")
            try:
                import polars as pl
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise ImportError("polars and pyarrow are required for RobustnessReport.load_parquet()") from exc
            for table_name, metadata in sorted(table_metadata.items()):
                if not isinstance(metadata, Mapping):
                    raise ValueError("RobustnessReport Parquet table metadata must be mappings")
                filename = metadata.get("file", "")
                table_path = _resolve_relative_child_path(
                    source,
                    filename,
                    context="RobustnessReport Parquet manifest table file paths",
                )
                if tables is not None and str(tables[str(table_name)]) != str(filename):
                    raise ValueError("RobustnessReport Parquet manifest table_metadata and tables files must match")
                if not table_path.is_file():
                    raise FileNotFoundError(f"RobustnessReport Parquet table file is missing: {filename}")
                rows = pl.read_parquet(table_path).to_dicts()
                expected_rows = int(metadata.get("rows", -1))
                if len(rows) != expected_rows:
                    raise ValueError("RobustnessReport Parquet table row count mismatch")
                if metadata.get("fingerprint") != _table_fingerprint(str(table_name), rows):
                    raise ValueError("RobustnessReport Parquet table fingerprint mismatch")
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RobustnessReport:
        """Parse a serialized report and verify its fingerprint."""

        if not isinstance(payload, Mapping):
            raise TypeError("RobustnessReport payload must be a mapping")
        required = {"metadata", "mode", "scenarios", "slice_by", "version"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"RobustnessReport payload is missing keys {missing}")
        scenarios = payload["scenarios"]
        if not isinstance(scenarios, list):
            raise ValueError("RobustnessReport.scenarios must be a list")
        slice_by = payload["slice_by"]
        if not isinstance(slice_by, list) or not all(isinstance(value, str) for value in slice_by):
            raise ValueError("RobustnessReport.slice_by must be a list of strings")
        result = cls(
            mode=str(payload["mode"]),  # type: ignore[arg-type]
            scenarios=tuple(RobustnessScenarioResult.from_dict(_required_mapping(item, "RobustnessReport.scenarios[]")) for item in scenarios),
            slice_by=tuple(slice_by),
            metadata=dict(_required_mapping(payload["metadata"], "RobustnessReport.metadata")),
            version=int(payload["version"]),
        )
        expected = payload.get("fingerprint")
        if expected is not None and expected != result.fingerprint:
            raise ValueError("RobustnessReport fingerprint mismatch")
        return result

    @classmethod
    def from_json(cls, payload: str) -> RobustnessReport:
        """Parse a JSON report and verify its fingerprint."""

        return cls.from_dict(json.loads(payload))

    @classmethod
    def load_json(cls, path: str | Path) -> RobustnessReport:
        """Load a persisted report and verify its fingerprint."""

        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    @property
    def fingerprint(self) -> str:
        """TCV1 fingerprint of the report."""

        return tcv1_sha256(
            {
                "metadata": {key: self.metadata[key] for key in sorted(self.metadata)},
                "mode": self.mode,
                "scenarios": [scenario.to_dict() for scenario in self.scenarios],
                "slice_by": list(self.slice_by),
                "version": self.version,
            }
        )


def robustness(
    result: PredictResult | CalibratedRunResult,
    *,
    y_true: Any,
    X: Any | None = None,
    predictor: Any | None = None,
    predictor_bundle: str | Path | None = None,
    mode: RobustnessMode = "clean_frozen",
    scenarios: Sequence[RobustnessScenarioSpec | Mapping[str, Any]] | None = None,
    slice_by: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    seed: int | None = None,
    workspace_path: str | Path | None = None,
    workspace_name: str = "",
    workspace_robustness_id: str | None = None,
    workspace_metadata: Mapping[str, Any] | None = None,
) -> RobustnessReport:
    """Compute audit-only robustness/generalization diagnostics.

    This public audit gate evaluates already replayed predictions. It does not
    refit or recalibrate. Post-prediction stress scenarios operate on already
    materialized predictions and intervals; spectral stress scenarios require
    explicit ``X`` together with either ``predictor`` or ``predictor_bundle``
    and replay the frozen predictor without refit or recalibration.
    """

    if mode not in ROBUSTNESS_MODES:
        raise ValueError("robustness.mode must be one of 'clean_frozen', 'matched_recalibration' or 'structural_refit'")
    if mode not in ROBUSTNESS_EXECUTABLE_MODES:
        raise NotImplementedError("nirs4all.robustness() currently supports only mode='clean_frozen' audit-only reports")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise ValueError("robustness.seed must be a non-negative integer")
    if predictor is not None and predictor_bundle is not None:
        raise ValueError("robustness() accepts either predictor or predictor_bundle, not both")
    if X is None:
        X = _published_spectral_replay_X(result)
    if predictor is None and predictor_bundle is None:
        predictor_bundle = _published_predictor_bundle(result)

    y_pred, intervals, guarantee_status, sample_ids = _prediction_payload(result)
    truth = _as_1d_float_array(y_true, "y_true")
    if truth.shape != y_pred.shape:
        raise ValueError("y_true must have the same shape as predictions")
    x_array = _as_2d_float_array(X, "X") if X is not None else None
    if x_array is not None and x_array.shape[0] != y_pred.shape[0]:
        raise ValueError("X must have the same number of rows as predictions")

    normalized_scenarios = _normalize_scenarios(scenarios)
    rows = _metadata_rows(metadata if metadata is not None else _result_metadata_for_rows(result), truth.size)
    slice_keys = tuple(str(key) for key in (slice_by or ()))
    _validate_slice_keys(rows, slice_keys)

    scenario_results = tuple(
        _evaluate_scenario(
            scenario,
            truth=truth,
            y_pred=y_pred,
            intervals=intervals,
            metadata_rows=rows,
            slice_by=slice_keys,
            seed=seed,
            X=x_array,
            predictor=predictor,
            predictor_bundle=predictor_bundle,
            sample_ids=sample_ids,
        )
        for scenario in normalized_scenarios
    )
    effective_seed = seed if seed is not None else 0
    report_metadata: dict[str, Any] = {
        "audit_only": True,
        "conformal_guarantee_status": guarantee_status,
        "effective_seed": effective_seed,
        "sample_ids": list(sample_ids) if sample_ids else None,
        "seed": seed,
        "supported_scenario_kinds": list(ROBUSTNESS_SCENARIO_KINDS),
    }
    spectral_replay = _spectral_replay_metadata(
        normalized_scenarios,
        predictor=predictor,
        predictor_bundle=predictor_bundle,
        sample_ids=sample_ids,
    )
    if spectral_replay is not None:
        report_metadata["spectral_replay"] = spectral_replay
    report = RobustnessReport(
        mode=mode,
        scenarios=scenario_results,
        slice_by=slice_keys,
        metadata=report_metadata,
    )
    if workspace_path is not None:
        save_workspace_robustness_report(
            workspace_path,
            report,
            name=workspace_name,
            robustness_id=workspace_robustness_id,
            metadata=workspace_metadata,
        )
    return report


def robustness_from_workspace_prediction(
    workspace_path: str | Path,
    prediction_id: str,
    *,
    y_true: Any,
    X: Any | None = None,
    predictor: Any | None = None,
    predictor_bundle: str | Path | None = None,
    mode: RobustnessMode = "clean_frozen",
    scenarios: Sequence[RobustnessScenarioSpec | Mapping[str, Any]] | None = None,
    slice_by: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    seed: int | None = None,
    save_to_workspace: bool = False,
    workspace_name: str = "",
    workspace_robustness_id: str | None = None,
    workspace_metadata: Mapping[str, Any] | None = None,
) -> RobustnessReport:
    """Compute a robustness report from one persisted workspace prediction.

    The prediction row is loaded with arrays through
    :func:`nirs4all.load_workspace_predict_result`, converted to
    :class:`PredictResult`, and then passed to :func:`nirs4all.robustness`.
    If the stored prediction carries executable row-aligned ``X``/``spectra``
    and ``robustness_evidence.predictor_bundle``/``model_path``,
    ``robustness()`` consumes them as spectral/OOD replay defaults. Set
    ``save_to_workspace=True`` to persist the resulting report back into the
    same workspace with a link to ``prediction_id``.
    """

    prediction = load_workspace_predict_result(workspace_path, prediction_id)
    report = robustness(
        prediction,
        y_true=y_true,
        X=X,
        predictor=predictor,
        predictor_bundle=predictor_bundle,
        mode=mode,
        scenarios=scenarios,
        slice_by=slice_by,
        metadata=metadata,
        seed=seed,
    )
    if save_to_workspace:
        save_workspace_robustness_report(
            workspace_path,
            report,
            name=workspace_name,
            robustness_id=workspace_robustness_id,
            metadata=workspace_metadata,
            prediction_id=prediction_id,
        )
    return report


def save_workspace_robustness_report(
    workspace_path: str | Path,
    report: RobustnessReport,
    *,
    name: str = "",
    robustness_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    pipeline_id: str | None = None,
    chain_id: str | None = None,
    conformal_id: str | None = None,
    prediction_id: str | None = None,
) -> str:
    """Persist a robustness/generalization report in a nirs4all workspace."""

    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    store = WorkspaceStore(Path(workspace_path))
    try:
        return store.save_robustness_result(
            report,
            name=name,
            robustness_id=robustness_id,
            metadata=metadata,
            run_id=run_id,
            pipeline_id=pipeline_id,
            chain_id=chain_id,
            conformal_id=conformal_id,
            prediction_id=prediction_id,
        )
    finally:
        store.close()


def load_workspace_robustness_report(workspace_path: str | Path, robustness_id: str) -> RobustnessReport:
    """Load a verified robustness/generalization report from a nirs4all workspace."""

    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    store = WorkspaceStore(Path(workspace_path))
    try:
        report = store.load_robustness_result(robustness_id)
    finally:
        store.close()
    if not isinstance(report, RobustnessReport):
        raise TypeError("workspace did not return a RobustnessReport")
    return report


def _evaluate_scenario(
    scenario: Mapping[str, Any],
    *,
    truth: np.ndarray,
    y_pred: np.ndarray,
    intervals: Mapping[float, Any],
    metadata_rows: tuple[Mapping[str, Any], ...],
    slice_by: tuple[str, ...],
    seed: int | None,
    X: np.ndarray | None,
    predictor: Any | None,
    predictor_bundle: str | Path | None,
    sample_ids: tuple[str, ...],
) -> RobustnessScenarioResult:
    severity = float(scenario.get("severity", 0.0))
    if not np.isfinite(severity):
        raise ValueError("robustness scenario severity must be finite")
    shifted_pred, shifted_intervals = _apply_audit_scenario(
        scenario,
        y_pred,
        intervals,
        severity,
        seed=seed,
        X=X,
        predictor=predictor,
        predictor_bundle=predictor_bundle,
        sample_ids=sample_ids,
    )
    conformal = _evaluate_conformal_if_available(truth, shifted_pred, shifted_intervals)
    return RobustnessScenarioResult(
        scenario={str(key): scenario[key] for key in sorted(scenario)},
        severity=severity,
        metrics=_point_metrics(truth, shifted_pred),
        conformal_metrics=conformal,
        slices=_slice_results(truth, shifted_pred, shifted_intervals, metadata_rows, slice_by),
    )


def _spectral_replay_metadata(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    predictor: Any | None,
    predictor_bundle: str | Path | None,
    sample_ids: tuple[str, ...],
) -> dict[str, Any] | None:
    if not any(str(scenario.get("kind", "")) in _SPECTRAL_REPLAY_SCENARIO_KINDS for scenario in scenarios):
        return None
    if predictor_bundle is not None:
        return {
            "all_predictions": False,
            "predictor_bundle": str(predictor_bundle),
            "route": "nirs4all.predict",
            "sample_ids_forwarded": bool(sample_ids),
            "source": "predictor_bundle",
        }
    if predictor is not None:
        return {
            "route": "predictor.predict_or_callable",
            "sample_ids_forwarded": False,
            "source": "predictor",
        }
    return None


def _apply_audit_scenario(
    scenario: Mapping[str, Any],
    y_pred: np.ndarray,
    intervals: Mapping[float, Any],
    severity: float,
    *,
    seed: int | None,
    X: np.ndarray | None,
    predictor: Any | None,
    predictor_bundle: str | Path | None,
    sample_ids: tuple[str, ...],
) -> tuple[np.ndarray, Mapping[float, Any]]:
    kind = str(scenario.get("kind", "observed"))
    if kind == "observed":
        if severity != 0.0:
            raise NotImplementedError("observed robustness scenarios require severity=0.0")
        return y_pred, intervals
    if kind == "prediction_bias":
        shifted_pred = y_pred + severity
        shifted_intervals = _shift_intervals(intervals, severity)
        return shifted_pred, shifted_intervals
    if kind == "prediction_noise":
        if severity < 0.0:
            raise ValueError("prediction_noise severity must be non-negative")
        distribution = str(scenario.get("distribution", "normal"))
        if distribution not in ROBUSTNESS_SCENARIO_DISTRIBUTIONS:
            raise NotImplementedError("prediction_noise currently supports only distribution='normal' or 'uniform'")
        noise = _scenario_noise(seed, scenario, distribution, severity, size=y_pred.shape)
        shifted_pred = y_pred + noise
        shifted_intervals = _shift_intervals(intervals, noise)
        return shifted_pred, shifted_intervals
    if kind == "spectral_noise":
        if severity < 0.0:
            raise ValueError("spectral_noise severity must be non-negative")
        distribution = str(scenario.get("distribution", "normal"))
        if distribution not in ROBUSTNESS_SCENARIO_DISTRIBUTIONS:
            raise NotImplementedError("spectral_noise currently supports only distribution='normal' or 'uniform'")
        if X is None or (predictor is None and predictor_bundle is None):
            raise ValueError("spectral_noise requires explicit X and predictor or predictor_bundle arguments")
        noise = _scenario_noise(seed, scenario, distribution, severity, size=X.shape)
        shifted_pred = _predict_with_frozen_predictor(predictor=predictor, predictor_bundle=predictor_bundle, X=X + noise, expected=y_pred.shape[0], sample_ids=sample_ids)
        shifted_intervals = _shift_intervals(intervals, shifted_pred - y_pred)
        return shifted_pred, shifted_intervals
    if kind == "spectral_offset":
        if X is None or (predictor is None and predictor_bundle is None):
            raise ValueError("spectral_offset requires explicit X and predictor or predictor_bundle arguments")
        shifted_pred = _predict_with_frozen_predictor(predictor=predictor, predictor_bundle=predictor_bundle, X=X + severity, expected=y_pred.shape[0], sample_ids=sample_ids)
        shifted_intervals = _shift_intervals(intervals, shifted_pred - y_pred)
        return shifted_pred, shifted_intervals
    if kind == "spectral_scale":
        factor = 1.0 + severity
        if factor <= 0.0:
            raise ValueError("spectral_scale requires 1.0 + severity to be positive")
        if X is None or (predictor is None and predictor_bundle is None):
            raise ValueError("spectral_scale requires explicit X and predictor or predictor_bundle arguments")
        shifted_pred = _predict_with_frozen_predictor(predictor=predictor, predictor_bundle=predictor_bundle, X=X * factor, expected=y_pred.shape[0], sample_ids=sample_ids)
        shifted_intervals = _shift_intervals(intervals, shifted_pred - y_pred)
        return shifted_pred, shifted_intervals
    if kind == "spectral_slope":
        if X is None or (predictor is None and predictor_bundle is None):
            raise ValueError("spectral_slope requires explicit X and predictor or predictor_bundle arguments")
        ramp = np.linspace(-severity / 2.0, severity / 2.0, num=X.shape[1], dtype=float)
        shifted_pred = _predict_with_frozen_predictor(predictor=predictor, predictor_bundle=predictor_bundle, X=X + ramp, expected=y_pred.shape[0], sample_ids=sample_ids)
        shifted_intervals = _shift_intervals(intervals, shifted_pred - y_pred)
        return shifted_pred, shifted_intervals
    if kind == "spectral_shift":
        if X is None or (predictor is None and predictor_bundle is None):
            raise ValueError("spectral_shift requires explicit X and predictor or predictor_bundle arguments")
        shifted_pred = _predict_with_frozen_predictor(predictor=predictor, predictor_bundle=predictor_bundle, X=_shift_feature_axis(X, severity), expected=y_pred.shape[0], sample_ids=sample_ids)
        shifted_intervals = _shift_intervals(intervals, shifted_pred - y_pred)
        return shifted_pred, shifted_intervals
    raise NotImplementedError(_unsupported_scenario_kind_message("unsupported robustness scenario kind; supported audit-only kinds are"))


def _slice_results(
    truth: np.ndarray,
    y_pred: np.ndarray,
    intervals: Mapping[float, Any],
    metadata_rows: tuple[Mapping[str, Any], ...],
    slice_by: tuple[str, ...],
) -> tuple[RobustnessSliceResult, ...]:
    if not slice_by:
        return ()
    grouped: dict[tuple[Any, ...], list[int]] = {}
    for index, row in enumerate(metadata_rows):
        key = tuple(row[column] for column in slice_by)
        grouped.setdefault(key, []).append(index)
    results: list[RobustnessSliceResult] = []
    for key_values, indices in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        idx = np.asarray(indices, dtype=int)
        sliced_intervals = _slice_intervals(intervals, idx)
        results.append(
            RobustnessSliceResult(
                slice_key={column: key_values[position] for position, column in enumerate(slice_by)},
                metrics=_point_metrics(truth[idx], y_pred[idx]),
                conformal_metrics=_evaluate_conformal_if_available(truth[idx], y_pred[idx], sliced_intervals),
            )
        )
    return tuple(results)


def _prediction_payload(result: PredictResult | CalibratedRunResult) -> tuple[np.ndarray, Mapping[float, Any], Mapping[str, Any] | None, tuple[str, ...]]:
    if isinstance(result, CalibratedRunResult):
        return (
            _as_1d_float_array(result.prediction.y_pred, "y_pred"),
            result.prediction.intervals,
            result.conformal_guarantee_status,
            tuple(result.sample_ids),
        )
    if isinstance(result, PredictResult):
        sample_ids: tuple[str, ...] = ()
        if result.sample_indices is not None:
            sample_ids = tuple(str(value) for value in result.sample_indices.tolist())
        return (
            _as_1d_float_array(result.y_pred, "y_pred"),
            result.intervals,
            result.conformal_guarantee_status,
            sample_ids,
        )
    raise TypeError("robustness() requires a PredictResult or CalibratedRunResult")


def _published_spectral_replay_X(result: PredictResult | CalibratedRunResult) -> Any | None:
    """Return published row-aligned spectra carried by a PredictResult, if any."""

    if isinstance(result, CalibratedRunResult):
        result = cast(PredictResult, result.prediction)
    if not isinstance(result, PredictResult) or not isinstance(result.metadata, Mapping):
        return None
    for key in ("X", "spectra"):
        value = result.metadata.get(key)
        if value is not None:
            return value
    return None


def _published_predictor_bundle(result: PredictResult | CalibratedRunResult) -> str | Path | None:
    """Return a published frozen predictor bundle reference, if any."""

    if isinstance(result, CalibratedRunResult):
        result = cast(PredictResult, result.prediction)
    if not isinstance(result, PredictResult):
        return None
    evidence = result.robustness_evidence or {}
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    for source in (evidence, metadata):
        for key in ("predictor_bundle", "model_path"):
            value = source.get(key)
            if value is not None and str(value).strip():
                return Path(value) if isinstance(value, Path) else str(value)
    return None


def _point_metrics(truth: np.ndarray, y_pred: np.ndarray) -> RobustnessMetricSet:
    if truth.size == 0:
        raise ValueError("robustness metrics require at least one sample")
    residual = y_pred - truth
    abs_error = np.abs(residual)
    return RobustnessMetricSet(
        n_samples=int(truth.size),
        rmse=float(np.sqrt(np.mean(residual**2))),
        mae=float(np.mean(abs_error)),
        bias=float(np.mean(residual)),
        max_abs_error=float(np.max(abs_error)),
    )


def _evaluate_conformal_if_available(
    truth: np.ndarray,
    y_pred: np.ndarray,
    intervals: Mapping[float, Any],
) -> Mapping[float, ConformalMetricSet]:
    if not intervals:
        return {}
    block = CalibratedPredictionBlock(
        y_pred=y_pred,
        intervals={float(coverage): interval for coverage, interval in intervals.items()},
    )
    return evaluate_conformal_prediction(y_true=truth, prediction=block)


def _slice_intervals(intervals: Mapping[float, Any], indices: np.ndarray) -> dict[float, ConformalIntervalBlock]:
    sliced: dict[float, ConformalIntervalBlock] = {}
    for coverage, interval in intervals.items():
        qhat = np.asarray(interval.qhat, dtype=float)
        sliced_qhat: float | np.ndarray
        if qhat.ndim == 0:
            sliced_qhat = float(qhat)
        else:
            sliced_qhat = qhat[indices]
        sliced[float(coverage)] = ConformalIntervalBlock(
            coverage=float(coverage),
            qhat=sliced_qhat,
            lower=np.asarray(interval.lower, dtype=float)[indices],
            upper=np.asarray(interval.upper, dtype=float)[indices],
        )
    return sliced


def _shift_intervals(intervals: Mapping[float, Any], offset: float | np.ndarray) -> dict[float, ConformalIntervalBlock]:
    shifted: dict[float, ConformalIntervalBlock] = {}
    for coverage, interval in intervals.items():
        qhat = np.asarray(interval.qhat, dtype=float)
        shifted_qhat: float | np.ndarray = float(qhat) if qhat.ndim == 0 else qhat.copy()
        shifted[float(coverage)] = ConformalIntervalBlock(
            coverage=float(coverage),
            qhat=shifted_qhat,
            lower=np.asarray(interval.lower, dtype=float) + offset,
            upper=np.asarray(interval.upper, dtype=float) + offset,
        )
    return shifted


def _shift_feature_axis(X: np.ndarray, offset: float) -> np.ndarray:
    """Shift spectra along the feature axis by fractional feature units."""

    if X.shape[1] <= 1 or offset == 0.0:
        return X.copy()
    axis = np.arange(X.shape[1], dtype=float)
    source_axis = axis + float(offset)
    shifted = np.vstack(
        [
            np.interp(
                source_axis,
                axis,
                row,
                left=float(row[0]),
                right=float(row[-1]),
            )
            for row in X
        ]
    )
    return np.asarray(shifted, dtype=float)


def _scenario_rng(seed: int | None, scenario: Mapping[str, Any]) -> np.random.Generator:
    base_seed = 0 if seed is None else int(seed)
    scenario_payload = {str(key): scenario[key] for key in sorted(scenario)}
    digest = tcv1_sha256({"seed": base_seed, "scenario": scenario_payload})
    return np.random.default_rng(int(digest[:16], 16) % (2**32))


def _scenario_noise(seed: int | None, scenario: Mapping[str, Any], distribution: str, severity: float, *, size: tuple[int, ...]) -> np.ndarray:
    rng = _scenario_rng(seed, scenario)
    if distribution == "normal":
        return rng.normal(loc=0.0, scale=severity, size=size)
    if distribution == "uniform":
        return rng.uniform(low=-severity, high=severity, size=size)
    raise NotImplementedError("robustness noise currently supports only distribution='normal' or 'uniform'")


def _predict_with_frozen_predictor(
    *,
    predictor: Any | None,
    predictor_bundle: str | Path | None,
    X: np.ndarray,
    expected: int,
    sample_ids: tuple[str, ...],
) -> np.ndarray:
    if predictor_bundle is not None:
        return _predict_with_predictor_bundle(predictor_bundle, X, expected=expected, sample_ids=sample_ids)
    if predictor is None:
        raise ValueError("spectral robustness scenarios require explicit X and predictor or predictor_bundle arguments")
    return _predict_with_predictor(predictor, X, expected=expected)


def _predict_with_predictor_bundle(
    predictor_bundle: str | Path,
    X: np.ndarray,
    *,
    expected: int,
    sample_ids: tuple[str, ...],
) -> np.ndarray:
    import importlib

    predict_module = importlib.import_module("nirs4all.api.predict")

    data: dict[str, Any] = {"X": X}
    if sample_ids:
        data["sample_ids"] = list(sample_ids)
    raw = predict_module.predict(model=predictor_bundle, data=data, all_predictions=False)
    if not isinstance(raw, PredictResult):
        raise TypeError("predictor_bundle replay must return a PredictResult")
    y_pred = _as_1d_float_array(raw.y_pred, "predictor_bundle prediction")
    if y_pred.shape[0] != expected:
        raise ValueError("predictor_bundle predictions must have the same length as the baseline predictions")
    return y_pred


def _predict_with_predictor(predictor: Any, X: np.ndarray, *, expected: int) -> np.ndarray:
    if hasattr(predictor, "predict"):
        raw = predictor.predict(X)
    elif callable(predictor):
        raw = predictor(X)
    else:
        raise TypeError("predictor must expose predict(X) or be callable")
    y_pred = _as_1d_float_array(raw, "predictor prediction")
    if y_pred.shape[0] != expected:
        raise ValueError("predictor predictions must have the same length as the baseline predictions")
    return y_pred


def _normalize_scenarios(scenarios: Sequence[RobustnessScenarioSpec | Mapping[str, Any]] | None) -> tuple[Mapping[str, Any], ...]:
    if scenarios is None:
        return ({"kind": "observed", "severity": 0.0},)
    if isinstance(scenarios, Mapping) or not isinstance(scenarios, Sequence) or isinstance(scenarios, (str, bytes)):
        raise TypeError("robustness.scenarios must be a sequence of mappings")
    normalized = []
    for scenario in scenarios:
        if isinstance(scenario, RobustnessScenarioSpec):
            scenario = scenario.to_dict()
        if not isinstance(scenario, Mapping):
            raise TypeError("each robustness scenario must be a mapping or RobustnessScenarioSpec")
        _validate_mapping_keys(scenario, "robustness.scenarios[]")
        if "kind" not in scenario:
            raise ValueError("each robustness scenario requires a kind")
        payload = {key: scenario[key] for key in sorted(scenario)}
        kind = _normalize_scenario_kind(payload["kind"], source="robustness.scenarios[].kind")
        payload["kind"] = kind
        if "distribution" in payload:
            payload["distribution"] = _normalize_scenario_distribution(kind, payload["distribution"], source="robustness.scenarios[].distribution")
        payload.setdefault("severity", 0.0)
        _validate_scenario_payload(payload, source="robustness.scenarios[]", validate_kind=False)
        normalized.append(payload)
    if not normalized:
        raise ValueError("robustness.scenarios must not be empty")
    return tuple(normalized)


def _normalize_scenario_distribution(kind: str, distribution: Any, *, source: str) -> str:
    if kind not in _STOCHASTIC_DISTRIBUTION_SCENARIOS:
        raise ValueError(f"{source} is supported only for prediction_noise and spectral_noise scenarios")
    if not isinstance(distribution, str) or not distribution.strip():
        raise ValueError(f"{source} must be a non-empty string")
    normalized = distribution.strip()
    if normalized not in ROBUSTNESS_SCENARIO_DISTRIBUTIONS:
        raise NotImplementedError(f"{kind} currently supports only distribution='normal' or 'uniform'")
    return normalized


def _normalize_scenario_kind(kind: Any, *, source: str) -> str:
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError(f"{source} must be a non-empty string")
    return kind.strip()


def _validate_scenario_payload(payload: Mapping[str, Any], *, source: str, validate_kind: bool) -> None:
    _validate_mapping_keys(payload, source)
    kind = _normalize_scenario_kind(payload.get("kind", ""), source=f"{source}.kind")
    if validate_kind and kind not in _SUPPORTED_SCENARIO_KIND_SET:
        raise NotImplementedError(_unsupported_scenario_kind_message(f"{source}.kind must be one of"))
    if "severity" in payload:
        severity = payload["severity"]
        if isinstance(severity, (bool, np.bool_)) or not isinstance(severity, (int, float, np.integer, np.floating)):
            raise ValueError(f"{source}.severity must be a real numeric scalar")
        if not np.isfinite(float(severity)):
            raise ValueError(f"{source}.severity must be finite")
    try:
        tcv1_sha256({key: payload[key] for key in sorted(payload)})
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be TCV1 JSON-native and fingerprintable: {exc}") from exc


def _validate_mapping_keys(payload: Mapping[str, Any], source: str) -> None:
    for key in payload:
        if not isinstance(key, str) or not key.strip() or key != key.strip() or "\x00" in key:
            raise ValueError(f"{source} keys must be canonical non-empty strings")


def _strict_json_mapping(payload: Mapping[str, Any], source: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{source} must be a mapping")
    _validate_mapping_keys(payload, source)
    normalized = {key: payload[key] for key in sorted(payload)}
    try:
        tcv1_sha256(normalized)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be TCV1 JSON-native and fingerprintable: {exc}") from exc
    return normalized


def _unsupported_scenario_kind_message(prefix: str) -> str:
    head = ", ".join(f"'{kind}'" for kind in ROBUSTNESS_SCENARIO_KINDS[:-1])
    return f"{prefix} {head} or '{ROBUSTNESS_SCENARIO_KINDS[-1]}'"


def _metadata_rows(metadata: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None, expected: int) -> tuple[Mapping[str, Any], ...]:
    if metadata is None:
        return tuple({} for _ in range(expected))
    if isinstance(metadata, Mapping):
        columns = {str(key): list(value) for key, value in metadata.items()}
        for key, values in columns.items():
            if len(values) != expected:
                raise ValueError(f"metadata column {key!r} length must match predictions")
        return tuple({key: values[index] for key, values in columns.items()} for index in range(expected))
    rows = list(metadata)
    if len(rows) != expected:
        raise ValueError("metadata rows length must match predictions")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("metadata rows must be mappings")
    return tuple(dict(row) for row in rows)


def _result_metadata_for_rows(result: PredictResult | CalibratedRunResult) -> Mapping[str, Any] | Sequence[Mapping[str, Any]] | None:
    metadata = result.metadata if isinstance(result, (PredictResult, CalibratedRunResult)) else None
    if not isinstance(metadata, Mapping):
        return None
    for key in ("row_metadata", "metadata_rows", "sample_metadata"):
        value = metadata.get(key)
        if value is not None:
            return cast(Mapping[str, Any] | Sequence[Mapping[str, Any]], value)
    return None


def _validate_slice_keys(rows: tuple[Mapping[str, Any], ...], slice_by: tuple[str, ...]) -> None:
    if len(set(slice_by)) != len(slice_by):
        raise ValueError("robustness.slice_by values must be unique")
    for column in slice_by:
        if not column:
            raise ValueError("robustness.slice_by values must be non-empty strings")
        if any(column not in row for row in rows):
            raise ValueError(f"slice column {column!r} is missing from metadata")


def _as_1d_float_array(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return array


def _as_2d_float_array(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return array


def _conformal_metrics_from_dict(payload: Any) -> dict[float, ConformalMetricSet]:
    mapping = _required_mapping(payload, "conformal_metrics")
    result: dict[float, ConformalMetricSet] = {}
    for coverage, metric_payload in mapping.items():
        metric = _conformal_metric_from_dict(_required_mapping(metric_payload, "conformal_metrics[]"))
        cov = float(coverage)
        if cov != float(metric.coverage):
            raise ValueError("conformal metric coverage key does not match payload")
        result[cov] = metric
    return result


def _conformal_metric_from_dict(payload: Mapping[str, Any]) -> ConformalMetricSet:
    required = {
        "coverage",
        "coverage_gap",
        "mean_interval_score",
        "mean_width",
        "median_width",
        "n_covered",
        "n_missed_above",
        "n_missed_below",
        "n_samples",
        "observed_coverage",
        "unit",
        "version",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"ConformalMetricSet payload is missing keys {missing}")
    result = ConformalMetricSet(
        coverage=float(payload["coverage"]),
        observed_coverage=float(payload["observed_coverage"]),
        coverage_gap=float(payload["coverage_gap"]),
        mean_width=float(payload["mean_width"]),
        median_width=float(payload["median_width"]),
        mean_interval_score=float(payload["mean_interval_score"]),
        n_samples=int(payload["n_samples"]),
        n_covered=int(payload["n_covered"]),
        n_missed_below=int(payload["n_missed_below"]),
        n_missed_above=int(payload["n_missed_above"]),
        unit=str(payload["unit"]),
        version=int(payload["version"]),
    )
    expected = payload.get("fingerprint")
    if expected is not None and expected != result.fingerprint:
        raise ValueError("ConformalMetricSet fingerprint mismatch")
    return result


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _markdown_guarantee_section(guarantee: Mapping[str, Any]) -> list[str]:
    lines = [
        "",
        "## Conformal guarantee status",
        "",
    ]
    status = guarantee.get("status")
    if status is not None:
        lines.append(f"- Status: `{status}`")
    requested_engine = guarantee.get("requested_engine")
    if requested_engine is not None:
        lines.append(f"- Requested engine: `{requested_engine}`")
    effective_engine = guarantee.get("effective_engine")
    if effective_engine is not None:
        lines.append(f"- Effective engine: `{effective_engine}`")
    method = guarantee.get("method")
    if method is not None:
        lines.append(f"- Method: `{method}`")
    unit = guarantee.get("unit")
    if unit is not None:
        lines.append(f"- Unit: `{unit}`")
    coverage = guarantee.get("coverage")
    if isinstance(coverage, Sequence) and not isinstance(coverage, (str, bytes)):
        lines.append(f"- Selected coverage: `{', '.join(str(value) for value in coverage)}`")
    calibrated_coverages = guarantee.get("calibrated_coverages")
    if isinstance(calibrated_coverages, Sequence) and not isinstance(calibrated_coverages, (str, bytes)):
        lines.append(f"- Calibrated coverage: `{', '.join(str(value) for value in calibrated_coverages)}`")
    scope = guarantee.get("scope")
    if scope is not None:
        lines.append(f"- Scope: `{scope}`")
    invalidation_reasons = guarantee.get("invalidation_reasons")
    if isinstance(invalidation_reasons, Sequence) and not isinstance(invalidation_reasons, (str, bytes)) and invalidation_reasons:
        lines.append("- Invalidation reasons:")
        for reason in invalidation_reasons:
            lines.append(f"  - {_markdown_cell(str(reason))}")
    limitations = guarantee.get("limitations")
    if isinstance(limitations, Sequence) and not isinstance(limitations, (str, bytes)):
        lines.append("- Limitations:")
        for limitation in limitations:
            lines.append(f"  - {_markdown_cell(str(limitation))}")
    return lines


def _html_guarantee_section(guarantee: Mapping[str, Any]) -> list[str]:
    lines = ["<h2>Conformal guarantee status</h2>", "<ul>"]
    status = guarantee.get("status")
    if status is not None:
        lines.append(f"<li>Status: <code>{_html_text(str(status))}</code></li>")
    requested_engine = guarantee.get("requested_engine")
    if requested_engine is not None:
        lines.append(f"<li>Requested engine: <code>{_html_text(str(requested_engine))}</code></li>")
    effective_engine = guarantee.get("effective_engine")
    if effective_engine is not None:
        lines.append(f"<li>Effective engine: <code>{_html_text(str(effective_engine))}</code></li>")
    method = guarantee.get("method")
    if method is not None:
        lines.append(f"<li>Method: <code>{_html_text(str(method))}</code></li>")
    unit = guarantee.get("unit")
    if unit is not None:
        lines.append(f"<li>Unit: <code>{_html_text(str(unit))}</code></li>")
    coverage = guarantee.get("coverage")
    if isinstance(coverage, Sequence) and not isinstance(coverage, (str, bytes)):
        lines.append(f"<li>Selected coverage: <code>{_html_text(', '.join(str(value) for value in coverage))}</code></li>")
    calibrated_coverages = guarantee.get("calibrated_coverages")
    if isinstance(calibrated_coverages, Sequence) and not isinstance(calibrated_coverages, (str, bytes)):
        lines.append(f"<li>Calibrated coverage: <code>{_html_text(', '.join(str(value) for value in calibrated_coverages))}</code></li>")
    scope = guarantee.get("scope")
    if scope is not None:
        lines.append(f"<li>Scope: <code>{_html_text(str(scope))}</code></li>")
    invalidation_reasons = guarantee.get("invalidation_reasons")
    if isinstance(invalidation_reasons, Sequence) and not isinstance(invalidation_reasons, (str, bytes)) and invalidation_reasons:
        lines.append("<li>Invalidation reasons:<ul>")
        for reason in invalidation_reasons:
            lines.append(f"<li>{_html_text(str(reason))}</li>")
        lines.append("</ul></li>")
    limitations = guarantee.get("limitations")
    if isinstance(limitations, Sequence) and not isinstance(limitations, (str, bytes)):
        lines.append("<li>Limitations:<ul>")
        for limitation in limitations:
            lines.append(f"<li>{_html_text(str(limitation))}</li>")
        lines.append("</ul></li>")
    lines.append("</ul>")
    return lines


def _scenario_label(scenario: Mapping[str, Any]) -> str:
    kind = str(scenario.get("kind", "scenario"))
    extras = [(str(key), scenario[key]) for key in sorted(scenario) if key not in {"kind", "severity"}]
    if not extras:
        return kind
    return f"{kind} ({', '.join(f'{key}={value}' for key, value in extras)})"


def _scenario_requires_spectral_replay(scenario: Mapping[str, Any]) -> bool:
    return str(scenario.get("kind", "")) in _SPECTRAL_REPLAY_SCENARIO_KINDS


def _scenario_execution_scope(scenario: Mapping[str, Any]) -> str:
    kind = str(scenario.get("kind", ""))
    if kind == "observed":
        return "baseline"
    if _scenario_requires_spectral_replay(scenario):
        return "spectral_replay"
    return "prediction_replay"


def _slice_label(slice_key: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}={slice_key[key]}" for key in sorted(slice_key))


def _markdown_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _json_cell(value: Mapping[str, Any]) -> str:
    return json.dumps({key: value[key] for key in sorted(value)}, sort_keys=True, separators=(",", ":"))


def _scenario_conformal_summary(scenario: RobustnessScenarioResult) -> dict[str, float | None]:
    metrics = tuple(metric for _, metric in sorted(scenario.conformal_metrics.items()))
    if not metrics:
        return {
            "max_abs_coverage_gap": None,
            "mean_width_mean": None,
            "min_observed_coverage": None,
        }
    return {
        "max_abs_coverage_gap": max(abs(metric.coverage_gap) for metric in metrics),
        "mean_width_mean": float(np.mean([metric.mean_width for metric in metrics])),
        "min_observed_coverage": min(metric.observed_coverage for metric in metrics),
    }


def _scenario_worst_slice(scenario: RobustnessScenarioResult, *, metric: str) -> dict[str, Any]:
    if not scenario.slices:
        return {"slice_key": None, "slice_label": None, "value": None}
    rows = [
        {
            "slice_key": {key: slice_result.slice_key[key] for key in sorted(slice_result.slice_key)},
            "slice_label": _slice_label(slice_result.slice_key),
            "value": _slice_metric_value(slice_result.metrics, metric),
        }
        for slice_result in scenario.slices
    ]
    rows.sort(key=lambda row: (-float(cast(float, row["value"])), _slice_label(cast(Mapping[str, Any], row["slice_key"]))))
    return rows[0]


def _resolve_relative_child_path(source: Path, filename: Any, *, context: str) -> Path:
    value = str(filename)
    relative = Path(value)
    if not value or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{context} must be relative child paths")
    root = source.resolve()
    candidate = (source / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"{context} must be relative child paths")
    return candidate


def _table_fingerprint(table_name: str, rows: Sequence[Mapping[str, Any]]) -> str:
    return tcv1_sha256(
        {
            "rows": [{str(key): row[key] for key in sorted(row)} for row in rows],
            "table": table_name,
        }
    )


def _summary_artifact_payload(report: RobustnessReport) -> dict[str, Any]:
    payload = {
        "conformal_guarantee_status": _summary_conformal_guarantee_status(report),
        "fingerprint": report.fingerprint,
        "format": "nirs4all.robustness.summary",
        "mode": report.mode,
        "report_version": report.version,
        "schema_version": 1,
        "slice_by": list(report.slice_by),
        "summary": list(report.summary_rows()),
    }
    spectral_replay = _summary_spectral_replay(report)
    if spectral_replay is not None:
        payload["spectral_replay"] = spectral_replay
    return payload


def _summary_conformal_guarantee_status(report: RobustnessReport) -> dict[str, Any] | None:
    guarantee = report.metadata.get("conformal_guarantee_status")
    if not isinstance(guarantee, Mapping):
        return None
    return {str(key): value for key, value in guarantee.items()}


def _summary_spectral_replay(report: RobustnessReport) -> dict[str, Any] | None:
    spectral_replay = report.metadata.get("spectral_replay")
    if not isinstance(spectral_replay, Mapping):
        return None
    return {str(key): spectral_replay[key] for key in sorted(spectral_replay)}


def _html_text(value: str) -> str:
    return html_escape(str(value), quote=True)


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[str]], *, numeric_columns: set[int] | None = None) -> str:
    numeric = numeric_columns or set()
    lines = ["<table>", "<thead>", "<tr>"]
    lines.extend(f"<th>{_html_text(header)}</th>" for header in headers)
    lines.extend(["</tr>", "</thead>", "<tbody>"])
    for row in rows:
        lines.append("<tr>")
        for index, value in enumerate(row):
            class_attr = ' class="num"' if index in numeric else ""
            lines.append(f"<td{class_attr}>{_html_text(value)}</td>")
        lines.append("</tr>")
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines)


def _markdown_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _slice_metric_value(metrics: RobustnessMetricSet, metric: str) -> float:
    if metric == "abs_bias":
        return abs(metrics.bias)
    return float(getattr(metrics, metric))


def _safe_ratio(value: float, baseline: float) -> float | None:
    if baseline == 0.0:
        return None
    return float(value / baseline)


def _format_number(value: float) -> str:
    return f"{float(value):.12g}"


def _format_optional_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return _format_number(value)


def _format_execution_scope(value: str) -> str:
    if value == "baseline":
        return "baseline"
    if value == "prediction_replay":
        return "prediction replay"
    if value == "spectral_replay":
        return "spectral/OOD replay"
    return "unknown"


__all__ = [
    "RobustnessMetricSet",
    "RobustnessReport",
    "RobustnessScenarioSpec",
    "RobustnessScenarioResult",
    "RobustnessSliceResult",
    "get_robustness_summary_schema",
    "load_workspace_robustness_report",
    "robustness",
    "robustness_summary_schema_json",
    "save_workspace_robustness_report",
]
