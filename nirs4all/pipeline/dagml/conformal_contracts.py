"""Internal conformal calibration contracts for the native DAG-ML lane.

This is the Python-side substrate for the planned ``nirs4all.calibrate()``
surface. It intentionally works on already replayed point predictions and
targets; it does not fit, select, refit, or call public prediction APIs.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from .training_contracts import tcv1_sha256

if TYPE_CHECKING:
    from nirs4all.api.robustness import RobustnessReport

ConformalMethod = Literal["split_absolute_residual"]
ConformalMultiTarget = Literal["marginal", "joint_max"]

SUPPORTED_CONFORMAL_KEYS = frozenset({"coverage", "group_by", "method", "multi_target", "unit"})
SUPPORTED_CONFORMAL_METHODS = frozenset({"split_absolute_residual"})
SUPPORTED_CONFORMAL_MULTI_TARGET = frozenset({"marginal", "joint_max"})
SUPPORTED_CONFORMAL_UNITS = frozenset({"physical_sample"})


@dataclass(frozen=True)
class ConformalIntervalBlock:
    """Prediction intervals for one coverage level."""

    coverage: float
    qhat: float | np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        coverage = normalize_coverages(self.coverage)[0]
        lower = _as_1d_interval_json_float_array(self.lower, "lower")
        upper = _as_1d_interval_json_float_array(self.upper, "upper")
        if lower.shape != upper.shape:
            raise ValueError("ConformalIntervalBlock lower and upper arrays must have the same shape")
        if np.any(lower > upper):
            raise ValueError("ConformalIntervalBlock lower bounds must be <= upper bounds")
        qhat = _normalize_interval_qhat(self.qhat, lower.shape)
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "qhat", qhat)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True)
class ConformalCalibrationSpec:
    """Normalized planned ``nirs4all.calibrate(...)`` calibration request."""

    coverage: tuple[float, ...]
    method: ConformalMethod = "split_absolute_residual"
    unit: str = "physical_sample"
    group_by: tuple[str, ...] = ()
    multi_target: ConformalMultiTarget = "marginal"

    def __post_init__(self) -> None:
        coverage = normalize_coverages(self.coverage)
        method = _normalize_supported_contract_string(
            self.method,
            "ConformalCalibrationSpec.method",
            SUPPORTED_CONFORMAL_METHODS,
        )
        unit = _normalize_supported_contract_string(
            self.unit,
            "ConformalCalibrationSpec.unit",
            SUPPORTED_CONFORMAL_UNITS,
        )
        group_by = _normalize_group_by(
            self.group_by,
            "ConformalCalibrationSpec.group_by",
            allow_empty=True,
        )
        multi_target = _normalize_supported_contract_string(
            self.multi_target,
            "ConformalCalibrationSpec.multi_target",
            SUPPORTED_CONFORMAL_MULTI_TARGET,
        )
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "group_by", group_by)
        object.__setattr__(self, "multi_target", multi_target)

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON-like contract form."""

        return {
            "coverage": list(self.coverage),
            "group_by": list(self.group_by),
            "method": self.method,
            "multi_target": self.multi_target,
            "unit": self.unit,
        }

    @property
    def fingerprint(self) -> str:
        """TCV1 fingerprint of the normalized conformal calibration contract."""

        return tcv1_sha256(self.to_dict())


@dataclass(frozen=True)
class CalibratedPredictionBlock:
    """Point predictions and calibrated intervals for one prediction cohort."""

    y_pred: np.ndarray
    intervals: dict[float, ConformalIntervalBlock]
    method: ConformalMethod = "split_absolute_residual"
    unit: str = "physical_sample"
    group_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        y_pred = _as_conformal_observation_array(self.y_pred, "y_pred")
        if self.method not in SUPPORTED_CONFORMAL_METHODS:
            raise ValueError(f"CalibratedPredictionBlock.method must be one of {sorted(SUPPORTED_CONFORMAL_METHODS)}")
        if self.unit not in SUPPORTED_CONFORMAL_UNITS:
            raise ValueError(f"CalibratedPredictionBlock.unit must be one of {sorted(SUPPORTED_CONFORMAL_UNITS)}")
        if not isinstance(self.intervals, Mapping):
            raise ValueError("CalibratedPredictionBlock.intervals must be a mapping")
        if not self.intervals:
            raise ValueError("CalibratedPredictionBlock.intervals must not be empty")
        intervals: dict[float, ConformalIntervalBlock] = {}
        for key, interval in self.intervals.items():
            coverage = normalize_coverages(key)[0]
            if not isinstance(interval, ConformalIntervalBlock):
                raise ValueError("CalibratedPredictionBlock.intervals values must be ConformalIntervalBlock instances")
            if interval.coverage != coverage:
                raise ValueError("CalibratedPredictionBlock interval coverage must match its mapping key")
            if interval.lower.shape != y_pred.shape or interval.upper.shape != y_pred.shape:
                raise ValueError("CalibratedPredictionBlock interval arrays must match y_pred shape")
            if coverage in intervals:
                raise ValueError("CalibratedPredictionBlock.intervals contains duplicate coverage values")
            intervals[coverage] = interval
        group_keys = tuple(_normalize_non_empty_string(key, "CalibratedPredictionBlock.group_keys") for key in self.group_keys)
        if group_keys and len(group_keys) != _n_prediction_rows(y_pred):
            raise ValueError("CalibratedPredictionBlock.group_keys length must match y_pred rows")
        object.__setattr__(self, "y_pred", y_pred)
        object.__setattr__(self, "intervals", intervals)
        object.__setattr__(self, "group_keys", group_keys)

    @property
    def coverages(self) -> tuple[float, ...]:
        """Return materialized coverages in deterministic order."""

        return tuple(sorted(self.intervals))

    def interval(self, coverage: float) -> ConformalIntervalBlock:
        """Return the interval block for an already materialized coverage."""

        cov = normalize_coverages(coverage)[0]
        try:
            return self.intervals[cov]
        except KeyError as exc:
            available = ", ".join(str(value) for value in self.coverages)
            raise KeyError(f"coverage {cov} was not materialized; available coverages: {available}") from exc


@dataclass(frozen=True)
class ConformalMetricSet:
    """Deterministic conformal diagnostics for one materialized coverage."""

    coverage: float
    observed_coverage: float
    coverage_gap: float
    mean_width: float
    median_width: float
    mean_interval_score: float
    n_samples: int
    n_covered: int
    n_missed_below: int
    n_missed_above: int
    unit: str = "physical_sample"
    version: int = 1

    def __post_init__(self) -> None:
        if not _is_integer_scalar(self.version) or self.version != 1:
            raise ValueError("ConformalMetricSet.version must be 1")
        normalize_coverages(self.coverage)
        if self.unit != "physical_sample":
            raise NotImplementedError("conformal V1 metrics currently support only unit='physical_sample'")
        for label, value in (
            ("ConformalMetricSet.n_samples", self.n_samples),
            ("ConformalMetricSet.n_covered", self.n_covered),
            ("ConformalMetricSet.n_missed_below", self.n_missed_below),
            ("ConformalMetricSet.n_missed_above", self.n_missed_above),
        ):
            if not _is_integer_scalar(value):
                raise ValueError(f"{label} must be an integer")
        if self.n_samples <= 0:
            raise ValueError("ConformalMetricSet.n_samples must be positive")
        if self.n_covered < 0 or self.n_missed_below < 0 or self.n_missed_above < 0:
            raise ValueError("ConformalMetricSet counts must be non-negative")
        if self.n_covered + self.n_missed_below + self.n_missed_above != self.n_samples:
            raise ValueError("ConformalMetricSet counts must sum to n_samples")
        observed_coverage = _validate_probability_metric(
            self.observed_coverage,
            "ConformalMetricSet.observed_coverage",
        )
        expected_observed = self.n_covered / self.n_samples
        if not math.isclose(observed_coverage, expected_observed, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("ConformalMetricSet.observed_coverage must match n_covered / n_samples")
        coverage_gap = _validate_finite_metric(self.coverage_gap, "ConformalMetricSet.coverage_gap")
        if not math.isclose(coverage_gap, observed_coverage - float(self.coverage), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("ConformalMetricSet.coverage_gap must equal observed_coverage - coverage")
        _validate_non_negative_metric(self.mean_width, "ConformalMetricSet.mean_width")
        _validate_non_negative_metric(self.median_width, "ConformalMetricSet.median_width")
        _validate_non_negative_metric(self.mean_interval_score, "ConformalMetricSet.mean_interval_score")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-like metric form."""

        return {
            "coverage": self.coverage,
            "coverage_gap": _encode_json_float(self.coverage_gap),
            "fingerprint": self.fingerprint,
            "mean_interval_score": _encode_json_float(self.mean_interval_score),
            "mean_width": _encode_json_float(self.mean_width),
            "median_width": _encode_json_float(self.median_width),
            "n_covered": self.n_covered,
            "n_missed_above": self.n_missed_above,
            "n_missed_below": self.n_missed_below,
            "n_samples": self.n_samples,
            "observed_coverage": _encode_json_float(self.observed_coverage),
            "unit": self.unit,
            "version": self.version,
        }

    @property
    def fingerprint(self) -> str:
        """TCV1 fingerprint of the conformal metric summary."""

        return tcv1_sha256(
            {
                "coverage": self.coverage,
                "coverage_gap": _encode_json_float(self.coverage_gap),
                "mean_interval_score": _encode_json_float(self.mean_interval_score),
                "mean_width": _encode_json_float(self.mean_width),
                "median_width": _encode_json_float(self.median_width),
                "n_covered": self.n_covered,
                "n_missed_above": self.n_missed_above,
                "n_missed_below": self.n_missed_below,
                "n_samples": self.n_samples,
                "observed_coverage": _encode_json_float(self.observed_coverage),
                "unit": self.unit,
                "version": self.version,
            }
        )


@dataclass(frozen=True)
class ConformalCalibrationCohortRow:
    """One physical sample row used for conformal calibration."""

    row_index: int
    sample_id: str
    role: str = "calibration"
    group: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _is_integer_scalar(self.row_index) or self.row_index < 0:
            raise ValueError("ConformalCalibrationCohortRow.row_index must be a non-negative integer")
        sample_id = _validate_required_payload_string_value(
            self.sample_id,
            "ConformalCalibrationCohortRow.sample_id",
        )
        role = _validate_required_payload_string_value(self.role, "ConformalCalibrationCohortRow.role")
        group = _validate_optional_payload_string_value(self.group, "ConformalCalibrationCohortRow.group")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("ConformalCalibrationCohortRow.metadata must be a mapping")
        _validate_json_native(self.metadata, "ConformalCalibrationCohortRow.metadata")
        object.__setattr__(self, "row_index", int(self.row_index))
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-like row form."""

        return {
            "group": self.group,
            "metadata": {key: self.metadata[key] for key in sorted(self.metadata)},
            "role": self.role,
            "row_index": self.row_index,
            "sample_id": self.sample_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConformalCalibrationCohortRow:
        """Parse one calibration cohort row."""

        if not isinstance(payload, Mapping):
            raise TypeError("ConformalCalibrationCohortRow payload must be a mapping")
        required = {"metadata", "role", "row_index", "sample_id"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"ConformalCalibrationCohortRow payload is missing keys {missing}")
        if not isinstance(payload["row_index"], int) or isinstance(payload["row_index"], bool) or payload["row_index"] < 0:
            raise ValueError("ConformalCalibrationCohortRow.row_index must be a non-negative integer")
        if not isinstance(payload["metadata"], Mapping):
            raise ValueError("ConformalCalibrationCohortRow.metadata must be a mapping")
        return cls(
            row_index=payload["row_index"],
            sample_id=payload["sample_id"],
            role=payload["role"],
            group=payload.get("group"),
            metadata=payload["metadata"],
        )


@dataclass(frozen=True)
class ConformalCalibrationCohortManifest:
    """Physical-sample calibration cohort identity manifest."""

    rows: tuple[ConformalCalibrationCohortRow, ...]
    unit: str = "physical_sample"
    version: int = 1

    def __post_init__(self) -> None:
        if not _is_integer_scalar(self.version) or self.version != 1:
            raise ValueError("ConformalCalibrationCohortManifest.version must be 1")
        if self.unit != "physical_sample":
            raise NotImplementedError("conformal V1 cohort manifests currently support only unit='physical_sample'")
        if not self.rows:
            raise ValueError("ConformalCalibrationCohortManifest requires at least one row")
        row_indices = [row.row_index for row in self.rows]
        if row_indices != list(range(len(self.rows))):
            raise ValueError("ConformalCalibrationCohortManifest row_index values must be contiguous from zero")
        sample_ids = [row.sample_id for row in self.rows]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("ConformalCalibrationCohortManifest sample_id values must be unique physical samples")
        roles = {row.role for row in self.rows}
        if roles != {"calibration"}:
            raise ValueError("ConformalCalibrationCohortManifest currently supports only role='calibration'")

    @property
    def n_samples(self) -> int:
        """Number of physical calibration samples."""

        return len(self.rows)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """Calibration physical sample ids in row order."""

        return tuple(row.sample_id for row in self.rows)

    @property
    def groups(self) -> tuple[str | None, ...]:
        """Calibration group labels in row order."""

        return tuple(row.group for row in self.rows)

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON-like manifest form."""

        return {
            "fingerprint": self.fingerprint,
            "n_samples": self.n_samples,
            "rows": [row.to_dict() for row in self.rows],
            "unit": self.unit,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConformalCalibrationCohortManifest:
        """Parse a serialized cohort manifest and verify its fingerprint."""

        if not isinstance(payload, Mapping):
            raise TypeError("ConformalCalibrationCohortManifest payload must be a mapping")
        required = {"rows", "unit", "version"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"ConformalCalibrationCohortManifest payload is missing keys {missing}")
        if not isinstance(payload["rows"], list):
            raise ValueError("ConformalCalibrationCohortManifest.rows must be a list")
        manifest = cls(
            rows=tuple(ConformalCalibrationCohortRow.from_dict(row) for row in payload["rows"]),
            unit=_strict_payload_string(payload["unit"], "ConformalCalibrationCohortManifest.unit"),
            version=_decode_contract_version(payload["version"], "ConformalCalibrationCohortManifest.version"),
        )
        expected_n = payload.get("n_samples")
        if expected_n is not None:
            if not _is_integer_scalar(expected_n):
                raise ValueError("ConformalCalibrationCohortManifest.n_samples must be an integer")
            if int(expected_n) != manifest.n_samples:
                raise ValueError("ConformalCalibrationCohortManifest n_samples mismatch")
        expected = payload.get("fingerprint")
        if expected is not None and expected != manifest.fingerprint:
            raise ValueError("ConformalCalibrationCohortManifest fingerprint mismatch")
        return manifest

    @property
    def fingerprint(self) -> str:
        """TCV1 fingerprint of the calibration cohort manifest."""

        return tcv1_sha256(
            {
                "n_samples": self.n_samples,
                "rows": [row.to_dict() for row in self.rows],
                "unit": self.unit,
                "version": self.version,
            }
        )


@dataclass(frozen=True)
class ConformalCalibrationArtifact:
    """Serializable fitted conformal calibrator artifact."""

    spec: ConformalCalibrationSpec
    calibrator: SplitConformalCalibrator
    calibration_size: int
    target_name: str | None = None
    predictor_fingerprint: str | None = None
    calibration_data_fingerprint: str | None = None
    calibration_cohort: ConformalCalibrationCohortManifest | None = None
    group_calibrators: Mapping[str, SplitConformalCalibrator] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        if not _is_integer_scalar(self.version) or self.version != 1:
            raise ValueError("ConformalCalibrationArtifact.version must be 1")
        _validate_optional_payload_string_value(self.target_name, "ConformalCalibrationArtifact.target_name")
        _validate_optional_payload_string_value(self.predictor_fingerprint, "ConformalCalibrationArtifact.predictor_fingerprint")
        _validate_optional_payload_string_value(self.calibration_data_fingerprint, "ConformalCalibrationArtifact.calibration_data_fingerprint")
        scores = _as_1d_float_array(self.calibrator.scores, "calibrator.scores")
        if np.any(scores < 0.0):
            raise ValueError("calibrator.scores must contain non-negative conformal residuals")
        if self.calibration_size != int(scores.size):
            raise ValueError("calibration_size must match the number of retained conformal scores")
        if self.spec.coverage != self.calibrator.coverages:
            raise ValueError("artifact spec coverage must match calibrator coverage")
        if self.spec.method != self.calibrator.method:
            raise ValueError("artifact spec method must match calibrator method")
        if self.spec.unit != self.calibrator.unit:
            raise ValueError("artifact spec unit must match calibrator unit")
        if tuple(sorted(self.calibrator.qhat_by_coverage)) != self.spec.coverage:
            raise ValueError("calibrator qhat_by_coverage coverages must match artifact spec coverage")
        for coverage in self.spec.coverage:
            expected_qhat = conformal_finite_sample_quantile(scores, coverage)
            if not _conformal_float_equal(self.calibrator.qhat_by_coverage[coverage], expected_qhat):
                raise ValueError("calibrator qhat_by_coverage must match retained conformal scores")
        _validate_group_calibrators(
            self.group_calibrators,
            spec=self.spec,
            calibration_cohort=self.calibration_cohort,
            calibration_size=self.calibration_size,
        )
        if self.calibration_cohort is not None:
            if self.calibration_cohort.n_samples != self.calibration_size:
                raise ValueError("calibration_cohort size must match calibration_size")
            if self.calibration_data_fingerprint is not None and self.calibration_data_fingerprint != self.calibration_cohort.fingerprint:
                raise ValueError("calibration_data_fingerprint must match calibration_cohort fingerprint")

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON-like artifact form."""

        return {
            "calibration_data_fingerprint": self.calibration_data_fingerprint,
            "calibration_cohort": self.calibration_cohort.to_dict() if self.calibration_cohort is not None else None,
            "calibration_size": self.calibration_size,
            "fingerprint": self.fingerprint,
            "group_calibrators": _group_calibrators_to_dict(self.group_calibrators),
            "predictor_fingerprint": self.predictor_fingerprint,
            "qhat_by_coverage": [{"coverage": coverage, "qhat": _encode_json_float(self.calibrator.qhat_by_coverage[coverage])} for coverage in sorted(self.calibrator.qhat_by_coverage)],
            "scores": [_encode_json_float(float(score)) for score in self.calibrator.scores.tolist()],
            "spec": self.spec.to_dict(),
            "target_name": self.target_name,
            "version": self.version,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return deterministic JSON with a trailing newline."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, separators=None if indent is not None else (",", ":")) + "\n"

    def save_json(self, path: str | Path) -> Path:
        """Persist the conformal artifact as deterministic verified JSON."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConformalCalibrationArtifact:
        """Parse a serialized conformal artifact and verify its fingerprint."""

        if not isinstance(payload, Mapping):
            raise TypeError("ConformalCalibrationArtifact payload must be a mapping")
        required = {"calibration_size", "qhat_by_coverage", "scores", "spec", "version"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"ConformalCalibrationArtifact payload is missing keys {missing}")
        if not isinstance(payload["calibration_size"], int) or isinstance(payload["calibration_size"], bool):
            raise ValueError("ConformalCalibrationArtifact.calibration_size must be an integer")
        if not isinstance(payload["qhat_by_coverage"], list):
            raise ValueError("ConformalCalibrationArtifact.qhat_by_coverage must be a list")
        if not isinstance(payload["scores"], list):
            raise ValueError("ConformalCalibrationArtifact.scores must be a list")

        spec = parse_conformal_calibration_spec(
            payload["spec"],
            context="ConformalCalibrationArtifact.spec",
            _allow_normalized_empty_group_by=True,
        )
        qhat_by_coverage: dict[float, float] = {}
        for item in payload["qhat_by_coverage"]:
            if not isinstance(item, Mapping):
                raise ValueError("ConformalCalibrationArtifact.qhat_by_coverage entries must be mappings")
            if "coverage" not in item or "qhat" not in item:
                raise ValueError("ConformalCalibrationArtifact.qhat_by_coverage entries require coverage and qhat")
            coverage = normalize_coverages(item["coverage"])[0]
            if coverage in qhat_by_coverage:
                raise ValueError("ConformalCalibrationArtifact.qhat_by_coverage contains duplicate coverage values")
            qhat_by_coverage[coverage] = _decode_json_float(item["qhat"], "qhat")
        scores = np.asarray([_decode_json_float(score, "score") for score in payload["scores"]], dtype=float)
        if not np.all(np.isfinite(scores)):
            raise ValueError("ConformalCalibrationArtifact.scores must contain only finite values")
        artifact = cls(
            spec=spec,
            calibrator=SplitConformalCalibrator(
                scores=scores,
                coverages=spec.coverage,
                qhat_by_coverage=qhat_by_coverage,
                method=spec.method,
                unit=spec.unit,
            ),
            calibration_size=payload["calibration_size"],
            target_name=_optional_payload_string(payload, "target_name"),
            predictor_fingerprint=_optional_payload_string(payload, "predictor_fingerprint"),
            calibration_data_fingerprint=_optional_payload_string(payload, "calibration_data_fingerprint"),
            calibration_cohort=ConformalCalibrationCohortManifest.from_dict(payload["calibration_cohort"]) if payload.get("calibration_cohort") is not None else None,
            group_calibrators=_group_calibrators_from_payload(payload.get("group_calibrators"), spec=spec),
            version=_decode_contract_version(payload["version"], "ConformalCalibrationArtifact.version"),
        )
        if set(artifact.calibrator.qhat_by_coverage) != set(artifact.spec.coverage):
            raise ValueError("ConformalCalibrationArtifact qhat coverage set must match spec coverage")
        expected = payload.get("fingerprint")
        if expected is not None and expected != artifact.fingerprint:
            raise ValueError("ConformalCalibrationArtifact fingerprint mismatch")
        return artifact

    @classmethod
    def from_json(cls, payload: str) -> ConformalCalibrationArtifact:
        """Parse a JSON conformal artifact and verify its fingerprint."""

        return cls.from_dict(json.loads(payload))

    @classmethod
    def load_json(cls, path: str | Path) -> ConformalCalibrationArtifact:
        """Load a persisted conformal artifact and verify its fingerprint."""

        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    @property
    def fingerprint(self) -> str:
        """TCV1 fingerprint of the artifact summary."""

        return tcv1_sha256(
            {
                "calibration_data_fingerprint": self.calibration_data_fingerprint,
                "calibration_cohort": self.calibration_cohort.to_dict() if self.calibration_cohort is not None else None,
                "calibration_size": self.calibration_size,
                "group_calibrators": _group_calibrators_to_dict(self.group_calibrators),
                "predictor_fingerprint": self.predictor_fingerprint,
                "qhat_by_coverage": [{"coverage": coverage, "qhat": _encode_json_float(self.calibrator.qhat_by_coverage[coverage])} for coverage in sorted(self.calibrator.qhat_by_coverage)],
                "scores": [_encode_json_float(float(score)) for score in self.calibrator.scores.tolist()],
                "spec": self.spec.to_dict(),
                "target_name": self.target_name,
                "version": self.version,
            }
        )


def conformal_guarantee_status(
    artifact: ConformalCalibrationArtifact,
    *,
    effective_engine: str,
    requested_engine: str = "nirs4all.conformal.v1",
    calibration_replay_source: Mapping[str, Any] | None = None,
    selected_coverages: Any | None = None,
    source_calibrated_result_fingerprint: str | None = None,
    invalidation_reasons: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return fail-loud metadata for the statistical guarantee actually carried.

    The payload is intentionally descriptive and JSON-compatible. It is not a
    new proof by itself; it records the method, engine, unit, selected
    coverages, and invalidation state that downstream APIs and Studio can show
    without inferring guarantees from interval arrays or empirical metrics.
    """

    coverages = _normalize_guarantee_coverages(selected_coverages, artifact.spec.coverage)
    effective_engine = _validate_required_payload_string_value(effective_engine, "conformal_guarantee_status.effective_engine")
    requested_engine = _validate_required_payload_string_value(requested_engine, "conformal_guarantee_status.requested_engine")
    source_calibrated_result_fingerprint = _validate_optional_payload_string_value(
        source_calibrated_result_fingerprint,
        "conformal_guarantee_status.source_calibrated_result_fingerprint",
    )
    reasons = _normalize_invalidation_reasons(invalidation_reasons, "conformal_guarantee_status.invalidation_reasons")
    guarantee, scope = _expected_conformal_guarantee_and_scope(artifact)
    status: dict[str, Any] = {
        "artifact_fingerprint": artifact.fingerprint,
        "calibrated_coverages": list(artifact.spec.coverage),
        "calibration_data_fingerprint": artifact.calibration_data_fingerprint,
        "coverage": list(coverages),
        "effective_engine": effective_engine,
        "guarantee": guarantee,
        "group_by": list(artifact.spec.group_by),
        "invalidation_reasons": list(reasons),
        "limitations": list(_expected_conformal_limitations(artifact)),
        "method": artifact.spec.method,
        "multi_target": artifact.spec.multi_target,
        "predictor_fingerprint": artifact.predictor_fingerprint,
        "requested_engine": requested_engine,
        "scope": scope,
        "source_calibrated_result_fingerprint": source_calibrated_result_fingerprint,
        "status": "active" if not reasons else "invalidated",
        "unit": artifact.spec.unit,
        "version": 1,
    }
    if calibration_replay_source is not None:
        _validate_json_native(calibration_replay_source, "conformal_guarantee_status.calibration_replay_source")
        status["calibration_replay_source"] = dict(calibration_replay_source)
    return status


@dataclass(frozen=True)
class SplitConformalCalibrator:
    """Fitted split-absolute-residual calibrator for one target."""

    scores: np.ndarray
    coverages: tuple[float, ...]
    qhat_by_coverage: dict[float, float]
    method: ConformalMethod = "split_absolute_residual"
    unit: str = "physical_sample"

    def __post_init__(self) -> None:
        scores = _as_1d_float_array(self.scores, "SplitConformalCalibrator.scores")
        if np.any(scores < 0.0):
            raise ValueError("SplitConformalCalibrator.scores must contain non-negative conformal residuals")
        coverages = normalize_coverages(self.coverages)
        method = _normalize_supported_contract_string(
            self.method,
            "SplitConformalCalibrator.method",
            SUPPORTED_CONFORMAL_METHODS,
        )
        unit = _normalize_supported_contract_string(
            self.unit,
            "SplitConformalCalibrator.unit",
            SUPPORTED_CONFORMAL_UNITS,
        )
        if not isinstance(self.qhat_by_coverage, Mapping):
            raise ValueError("SplitConformalCalibrator.qhat_by_coverage must be a mapping")
        qhat_by_coverage: dict[float, float] = {}
        for raw_coverage, raw_qhat in self.qhat_by_coverage.items():
            coverage = normalize_coverages(raw_coverage)[0]
            if coverage in qhat_by_coverage:
                raise ValueError("SplitConformalCalibrator.qhat_by_coverage contains duplicate coverage values")
            qhat = _validate_metric_scalar(
                raw_qhat,
                "SplitConformalCalibrator.qhat_by_coverage",
                allow_infinity=True,
            )
            if qhat < 0.0:
                raise ValueError("SplitConformalCalibrator.qhat_by_coverage must contain non-negative qhat values")
            qhat_by_coverage[coverage] = qhat
        if tuple(sorted(qhat_by_coverage)) != coverages:
            raise ValueError("SplitConformalCalibrator.qhat_by_coverage coverages must match coverages")
        for coverage in coverages:
            expected_qhat = conformal_finite_sample_quantile(scores, coverage)
            if not _conformal_float_equal(qhat_by_coverage[coverage], expected_qhat):
                raise ValueError("SplitConformalCalibrator.qhat_by_coverage must match retained conformal scores")
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "coverages", coverages)
        object.__setattr__(self, "qhat_by_coverage", qhat_by_coverage)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "unit", unit)

    def apply(self, y_pred: Any) -> dict[float, ConformalIntervalBlock]:
        """Apply calibrated intervals to point predictions."""

        return apply_split_conformal_calibrator(self, y_pred).intervals

    def apply_block(self, y_pred: Any) -> CalibratedPredictionBlock:
        """Apply calibrated intervals and keep point predictions with them."""

        return apply_split_conformal_calibrator(self, y_pred)


def fit_split_absolute_residual_calibrator(
    y_true: Any,
    y_pred: Any,
    *,
    coverage: float | list[float] | tuple[float, ...],
    unit: str = "physical_sample",
) -> SplitConformalCalibrator:
    """Fit split conformal intervals from calibration targets and predictions."""

    if unit != "physical_sample":
        raise NotImplementedError("conformal V1 currently supports only unit='physical_sample'")
    truth = _as_conformal_observation_array(y_true, "y_true")
    predictions = _as_conformal_observation_array(y_pred, "y_pred")
    if truth.shape != predictions.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    scores = np.abs(truth - predictions)
    coverages = normalize_coverages(coverage)
    qhat_by_coverage = {cov: conformal_finite_sample_quantile(scores, cov) for cov in coverages}
    return SplitConformalCalibrator(
        scores=scores,
        coverages=coverages,
        qhat_by_coverage=qhat_by_coverage,
        unit=unit,
    )


def fit_joint_max_absolute_residual_calibrator(
    y_true: Any,
    y_pred: Any,
    *,
    coverage: float | list[float] | tuple[float, ...],
    unit: str = "physical_sample",
) -> SplitConformalCalibrator:
    """Fit a simultaneous multi-target split conformal calibrator.

    The retained nonconformity score is one scalar per physical sample:
    ``max(abs(y_true - y_pred))`` across target columns.
    """

    if unit != "physical_sample":
        raise NotImplementedError("conformal V1 currently supports only unit='physical_sample'")
    truth = _as_2d_float_array(y_true, "y_true")
    predictions = _as_2d_float_array(y_pred, "y_pred")
    if truth.shape != predictions.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    scores = np.max(np.abs(truth - predictions), axis=1)
    coverages = normalize_coverages(coverage)
    qhat_by_coverage = {cov: conformal_finite_sample_quantile(scores, cov) for cov in coverages}
    return SplitConformalCalibrator(
        scores=scores,
        coverages=coverages,
        qhat_by_coverage=qhat_by_coverage,
        unit=unit,
    )


def fit_conformal_calibration_artifact(
    y_true: Any,
    y_pred: Any,
    *,
    spec: ConformalCalibrationSpec,
    calibration_cohort: ConformalCalibrationCohortManifest | None = None,
    target_name: str | None = None,
    predictor_fingerprint: str | None = None,
    calibration_data_fingerprint: str | None = None,
) -> ConformalCalibrationArtifact:
    """Fit the current V1 conformal artifact from replayed calibration arrays."""

    truth = _as_conformal_target_array(y_true, "y_true", spec=spec)
    predictions = _as_conformal_target_array(y_pred, "y_pred", spec=spec)
    if truth.shape != predictions.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if calibration_cohort is not None:
        if calibration_cohort.n_samples != _n_prediction_rows(truth):
            raise ValueError("calibration_cohort size must match calibration arrays")
        if calibration_data_fingerprint is not None and calibration_data_fingerprint != calibration_cohort.fingerprint:
            raise ValueError("calibration_data_fingerprint must match calibration_cohort fingerprint")
        calibration_data_fingerprint = calibration_cohort.fingerprint
    if spec.multi_target == "joint_max":
        calibrator = fit_joint_max_absolute_residual_calibrator(
            truth,
            predictions,
            coverage=spec.coverage,
            unit=spec.unit,
        )
    else:
        calibrator = fit_split_absolute_residual_calibrator(
            truth,
            predictions,
            coverage=spec.coverage,
            unit=spec.unit,
        )
    group_calibrators: dict[str, SplitConformalCalibrator] = {}
    if spec.group_by:
        if calibration_cohort is None:
            raise ValueError("grouped conformal artifacts require a calibration_cohort with row-aligned group evidence")
        group_calibrators = _fit_group_calibrators(
            y_true=y_true,
            y_pred=y_pred,
            spec=spec,
            calibration_cohort=calibration_cohort,
        )
    return ConformalCalibrationArtifact(
        spec=spec,
        calibrator=calibrator,
        calibration_size=int(calibrator.scores.size),
        target_name=target_name,
        predictor_fingerprint=predictor_fingerprint,
        calibration_data_fingerprint=calibration_data_fingerprint,
        calibration_cohort=calibration_cohort,
        group_calibrators=group_calibrators,
    )


def normalize_conformal_calibration_cohort(
    y_true: Any,
    y_pred: Any,
    *,
    sample_ids: Any,
    groups: Any | None = None,
    metadata: Any | None = None,
    unit: str = "physical_sample",
    role: str = "calibration",
) -> ConformalCalibrationCohortManifest:
    """Normalize row-aligned calibration arrays into a physical-sample manifest."""

    if unit != "physical_sample":
        raise NotImplementedError("conformal V1 cohort manifests currently support only unit='physical_sample'")
    role = _normalize_non_empty_string(role, "role")
    if role != "calibration":
        raise ValueError("conformal calibration cohort rows currently require role='calibration'")
    truth = _as_conformal_observation_array(y_true, "y_true")
    predictions = _as_conformal_observation_array(y_pred, "y_pred")
    if truth.shape != predictions.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    n_rows = _n_prediction_rows(truth)
    ids = _normalize_sample_ids(sample_ids, n_rows)
    group_values = _normalize_optional_groups(groups, n_rows)
    metadata_rows = _normalize_metadata_rows(metadata, n_rows)
    return ConformalCalibrationCohortManifest(
        rows=tuple(
            ConformalCalibrationCohortRow(
                row_index=index,
                sample_id=ids[index],
                role=role,
                group=group_values[index],
                metadata=metadata_rows[index],
            )
            for index in range(n_rows)
        ),
        unit=unit,
    )


def calibrate_replayed_predictions(
    *,
    y_true_calibration: Any,
    y_pred_calibration: Any,
    y_pred: Any,
    spec: ConformalCalibrationSpec,
    calibration_sample_ids: Any,
    prediction_sample_ids: Any,
    calibration_groups: Any | None = None,
    calibration_metadata: Any | None = None,
    prediction_groups: Any | None = None,
    prediction_metadata: Any | None = None,
    result_metadata: Mapping[str, Any] | None = None,
    target_name: str | None = None,
    predictor_fingerprint: str | None = None,
    calibration_replay_source: Mapping[str, Any] | None = None,
) -> CalibratedRunResult:
    """Calibrate already replayed predictions against an explicit calibration cohort."""

    cohort = normalize_conformal_calibration_cohort(
        y_true=y_true_calibration,
        y_pred=y_pred_calibration,
        sample_ids=calibration_sample_ids,
        groups=calibration_groups,
        metadata=calibration_metadata,
        unit=spec.unit,
    )
    artifact = fit_conformal_calibration_artifact(
        y_true=y_true_calibration,
        y_pred=y_pred_calibration,
        spec=spec,
        calibration_cohort=cohort,
        target_name=target_name,
        predictor_fingerprint=predictor_fingerprint,
    )
    predictions = _as_conformal_target_array(y_pred, "y_pred", spec=spec)
    sample_ids = _normalize_sample_ids(prediction_sample_ids, _n_prediction_rows(predictions))
    prediction_group_keys = _normalize_prediction_group_keys(
        spec.group_by,
        groups=prediction_groups,
        metadata=prediction_metadata,
        expected_size=_n_prediction_rows(predictions),
    )
    metadata: dict[str, Any] = {
        **dict(result_metadata or {}),
        "conformal_guarantee_status": conformal_guarantee_status(
            artifact,
            effective_engine="nirs4all.python.replayed_array_calibration",
            calibration_replay_source=calibration_replay_source,
        ),
    }
    if calibration_replay_source is not None:
        metadata["calibration_replay_source"] = dict(calibration_replay_source)
    return CalibratedRunResult(
        artifact=artifact,
        prediction=apply_conformal_artifact_to_predictions(
            artifact,
            predictions,
            group_keys=prediction_group_keys,
        ),
        sample_ids=sample_ids,
        metadata=metadata,
    )


def apply_conformal_artifact_to_predictions(
    artifact: ConformalCalibrationArtifact,
    y_pred: Any,
    *,
    group_keys: Sequence[str] = (),
) -> CalibratedPredictionBlock:
    """Apply a conformal artifact to predictions, including grouped calibrators."""

    predictions = _as_conformal_target_array(y_pred, "y_pred", spec=artifact.spec)
    if not artifact.spec.group_by:
        if group_keys:
            raise ValueError("prediction group_keys require a grouped conformal artifact")
        return _apply_calibrator_for_spec(artifact.calibrator, predictions, spec=artifact.spec)
    keys = tuple(_validate_required_payload_string_value(key, "prediction group key") for key in group_keys)
    if len(keys) != _n_prediction_rows(predictions):
        raise ValueError("prediction group keys length must match y_pred")
    if not artifact.group_calibrators:
        raise ValueError("grouped conformal artifact has no group_calibrators")
    missing = sorted(set(keys) - set(artifact.group_calibrators))
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... (+{len(missing) - 5} more)"
        raise ValueError(f"prediction group keys are absent from the conformal calibration artifact: {preview}{suffix}")
    intervals: dict[float, ConformalIntervalBlock] = {}
    for coverage in artifact.spec.coverage:
        qhat = np.asarray(
            [artifact.group_calibrators[key].qhat_by_coverage[coverage] for key in keys],
            dtype=float,
        )
        intervals[coverage] = ConformalIntervalBlock(
            coverage=coverage,
            qhat=qhat,
            lower=predictions - _broadcast_qhat_to_prediction_shape(qhat, predictions.shape),
            upper=predictions + _broadcast_qhat_to_prediction_shape(qhat, predictions.shape),
        )
    return CalibratedPredictionBlock(
        y_pred=predictions,
        intervals=intervals,
        method=artifact.spec.method,
        unit=artifact.spec.unit,
        group_keys=keys,
    )


def parse_conformal_calibration_spec(
    calibration: Mapping[str, Any],
    *,
    context: str = "nirs4all.calibrate(...)",
    _allow_normalized_empty_group_by: bool = False,
) -> ConformalCalibrationSpec:
    """Validate and normalize a planned conformal calibration mapping."""

    if not isinstance(calibration, Mapping):
        raise TypeError(f"{context} must be a mapping")
    unknown = sorted(set(calibration) - SUPPORTED_CONFORMAL_KEYS)
    if unknown:
        raise ValueError(f"{context} does not support keys {unknown}; supported keys are {sorted(SUPPORTED_CONFORMAL_KEYS)}")
    if "coverage" not in calibration:
        raise ValueError(f"{context}.coverage is required")

    method = _optional_lower_string(calibration, "method", default="split_absolute_residual")
    if method not in SUPPORTED_CONFORMAL_METHODS:
        raise ValueError(f"{context}.method must be one of {sorted(SUPPORTED_CONFORMAL_METHODS)}")

    unit = _optional_lower_string(calibration, "unit", default="physical_sample")
    if unit not in SUPPORTED_CONFORMAL_UNITS:
        raise ValueError(f"{context}.unit must be one of {sorted(SUPPORTED_CONFORMAL_UNITS)}")

    multi_target = _optional_lower_string(calibration, "multi_target", default="marginal")
    if multi_target not in SUPPORTED_CONFORMAL_MULTI_TARGET:
        raise ValueError(f"{context}.multi_target must be one of {sorted(SUPPORTED_CONFORMAL_MULTI_TARGET)}")
    return ConformalCalibrationSpec(
        coverage=normalize_coverages(calibration["coverage"]),
        method=method,  # type: ignore[arg-type]
        unit=unit,
        group_by=_normalize_group_by(
            calibration.get("group_by"),
            f"{context}.group_by",
            allow_empty=_allow_normalized_empty_group_by,
        ),
        multi_target=multi_target,  # type: ignore[arg-type]
    )


def apply_split_conformal_calibrator(
    calibrator: SplitConformalCalibrator,
    y_pred: Any,
) -> CalibratedPredictionBlock:
    """Apply a split conformal calibrator and return a typed prediction block."""

    predictions = _as_conformal_observation_array(y_pred, "y_pred")
    intervals = {
        coverage: ConformalIntervalBlock(
            coverage=coverage,
            qhat=qhat,
            lower=predictions - qhat,
            upper=predictions + qhat,
        )
        for coverage, qhat in calibrator.qhat_by_coverage.items()
    }
    return CalibratedPredictionBlock(
        y_pred=predictions,
        intervals=intervals,
        method=calibrator.method,
        unit=calibrator.unit,
    )


def _apply_calibrator_for_spec(
    calibrator: SplitConformalCalibrator,
    y_pred: np.ndarray,
    *,
    spec: ConformalCalibrationSpec,
) -> CalibratedPredictionBlock:
    if spec.multi_target == "marginal":
        return apply_split_conformal_calibrator(calibrator, y_pred)
    intervals = {
        coverage: ConformalIntervalBlock(
            coverage=coverage,
            qhat=qhat,
            lower=y_pred - qhat,
            upper=y_pred + qhat,
        )
        for coverage, qhat in calibrator.qhat_by_coverage.items()
    }
    return CalibratedPredictionBlock(
        y_pred=y_pred,
        intervals=intervals,
        method=calibrator.method,
        unit=calibrator.unit,
    )


def evaluate_conformal_prediction(
    *,
    y_true: Any,
    prediction: CalibratedPredictionBlock,
) -> dict[float, ConformalMetricSet]:
    """Compute deterministic diagnostics for materialized conformal intervals."""

    truth = _as_conformal_observation_array(y_true, "y_true")
    if truth.shape != prediction.y_pred.shape:
        raise ValueError("y_true must have the same shape as calibrated predictions")
    metrics: dict[float, ConformalMetricSet] = {}
    for coverage in prediction.coverages:
        interval = prediction.interval(coverage)
        lower = _as_interval_array(interval.lower, "lower", truth.shape)
        upper = _as_interval_array(interval.upper, "upper", truth.shape)
        if np.any(lower > upper):
            raise ValueError("conformal interval lower bounds must be <= upper bounds")

        below = truth < lower
        above = truth > upper
        row_below = _row_any(below)
        row_above = _row_any(above) & ~row_below
        covered = ~(row_below | row_above)
        width = upper - lower
        alpha = 1.0 - coverage
        below_penalty = np.where(below, lower - truth, 0.0)
        above_penalty = np.where(above, truth - upper, 0.0)
        interval_score = width + (2.0 / alpha) * below_penalty + (2.0 / alpha) * above_penalty
        observed = float(np.mean(covered))
        metrics[coverage] = ConformalMetricSet(
            coverage=coverage,
            observed_coverage=observed,
            coverage_gap=observed - coverage,
            mean_width=float(np.mean(width)),
            median_width=float(np.median(width)),
            mean_interval_score=float(np.mean(interval_score)),
            n_samples=int(_n_prediction_rows(truth)),
            n_covered=int(np.count_nonzero(covered)),
            n_missed_below=int(np.count_nonzero(row_below)),
            n_missed_above=int(np.count_nonzero(row_above)),
            unit=prediction.unit,
        )
    return metrics


@dataclass(frozen=True)
class CalibratedRunResult:
    """Serializable internal result for one calibrated prediction cohort."""

    artifact: ConformalCalibrationArtifact
    prediction: CalibratedPredictionBlock
    sample_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        if not _is_integer_scalar(self.version) or self.version != 1:
            raise ValueError("CalibratedRunResult.version must be 1")
        _validate_calibrated_result_metadata(self.metadata)
        if self.prediction.method != self.artifact.spec.method:
            raise ValueError("prediction method must match conformal artifact method")
        if self.prediction.unit != self.artifact.spec.unit:
            raise ValueError("prediction unit must match conformal artifact unit")
        if self.prediction.coverages != self.artifact.spec.coverage:
            raise ValueError("prediction coverages must match conformal artifact coverage")
        _validate_calibrated_prediction_matches_artifact(self.artifact, self.prediction)
        n_prediction_rows = _n_prediction_rows(np.asarray(self.prediction.y_pred))
        if n_prediction_rows > 0 and not self.sample_ids:
            raise ValueError("CalibratedRunResult sample_ids are required for physical-sample conformal predictions")
        if self.sample_ids and len(self.sample_ids) != n_prediction_rows:
            raise ValueError("sample_ids length must match the number of calibrated predictions")
        _validate_prediction_sample_ids(self.sample_ids)
        _validate_prediction_sample_ids_are_disjoint_from_calibration(
            self.artifact,
            self.sample_ids,
        )
        _validate_calibrated_result_guarantee_metadata(
            self.metadata,
            artifact=self.artifact,
            prediction_coverages=self.prediction.coverages,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON-like calibrated result form."""

        return {
            "artifact": self.artifact.to_dict(),
            "fingerprint": self.fingerprint,
            "metadata": {str(key): self.metadata[key] for key in sorted(self.metadata)},
            "prediction": _calibrated_prediction_block_to_dict(self.prediction),
            "sample_ids": list(self.sample_ids),
            "version": self.version,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return deterministic JSON with a trailing newline."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, separators=None if indent is not None else (",", ":")) + "\n"

    def save_json(self, path: str | Path) -> Path:
        """Persist the calibrated result as deterministic verified JSON."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    def to_predict_result(self):
        """Convert this calibrated result to the public ``PredictResult`` surface."""

        from nirs4all.api.result import PredictResult

        _validate_calibrated_result_guarantee_metadata(
            self.metadata,
            artifact=self.artifact,
            prediction_coverages=self.prediction.coverages,
        )
        return PredictResult(
            y_pred=self.prediction.y_pred,
            metadata={
                **dict(self.metadata),
                "conformal_artifact": self.artifact.to_dict(),
                "calibrated_result_fingerprint": self.fingerprint,
            },
            sample_indices=np.asarray(self.sample_ids, dtype=object),
            model_name=self.artifact.target_name or "",
            intervals=self.prediction.intervals,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CalibratedRunResult:
        """Parse a serialized calibrated result and verify its fingerprint."""

        if not isinstance(payload, Mapping):
            raise TypeError("CalibratedRunResult payload must be a mapping")
        required = {"artifact", "metadata", "prediction", "sample_ids", "version"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"CalibratedRunResult payload is missing keys {missing}")
        if not isinstance(payload["metadata"], Mapping):
            raise ValueError("CalibratedRunResult.metadata must be a mapping")
        if not isinstance(payload["sample_ids"], list):
            raise ValueError("CalibratedRunResult.sample_ids must be a list")
        if not all(isinstance(sample_id, str) and sample_id for sample_id in payload["sample_ids"]):
            raise ValueError("CalibratedRunResult.sample_ids must contain non-empty strings")
        artifact = ConformalCalibrationArtifact.from_dict(payload["artifact"])
        result = cls(
            artifact=artifact,
            prediction=_calibrated_prediction_block_from_dict(payload["prediction"]),
            sample_ids=tuple(payload["sample_ids"]),
            metadata=dict(payload["metadata"]),
            version=_decode_contract_version(payload["version"], "CalibratedRunResult.version"),
        )
        expected = payload.get("fingerprint")
        if expected is not None and expected != result.fingerprint:
            raise ValueError("CalibratedRunResult fingerprint mismatch")
        return result

    @classmethod
    def from_json(cls, payload: str) -> CalibratedRunResult:
        """Parse a JSON calibrated result and verify its fingerprint."""

        return cls.from_dict(json.loads(payload))

    @classmethod
    def load_json(cls, path: str | Path) -> CalibratedRunResult:
        """Load a persisted calibrated result and verify its fingerprint."""

        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    @property
    def fingerprint(self) -> str:
        """TCV1 fingerprint of the calibrated result summary."""

        return tcv1_sha256(
            {
                "artifact": self.artifact.to_dict(),
                "metadata": {str(key): self.metadata[key] for key in sorted(self.metadata)},
                "prediction": _calibrated_prediction_block_to_dict(self.prediction),
                "sample_ids": list(self.sample_ids),
                "version": self.version,
            }
        )

    @property
    def conformal_guarantee_status(self) -> dict[str, Any] | None:
        """Return persisted guarantee metadata, when present."""

        status = self.metadata.get("conformal_guarantee_status")
        return dict(status) if isinstance(status, Mapping) else None

    @property
    def calibration_replay_source(self) -> dict[str, Any] | None:
        """Return persisted calibration replay provenance, when present."""

        status = self.conformal_guarantee_status
        if status is not None:
            source = status.get("calibration_replay_source")
            if isinstance(source, Mapping):
                return dict(source)
        source = self.metadata.get("calibration_replay_source")
        return dict(source) if isinstance(source, Mapping) else None

    @property
    def tuning_calibration_source(self) -> dict[str, Any] | None:
        """Return persisted native tuning calibration provenance, when present."""

        source = self.metadata.get("tuning_calibration_source")
        return dict(source) if isinstance(source, Mapping) else None

    def metrics(self, y_true: Any) -> dict[float, ConformalMetricSet]:
        """Evaluate materialized conformal intervals against observed targets."""

        return evaluate_conformal_prediction(y_true=y_true, prediction=self.prediction)

    def robustness(
        self,
        *,
        y_true: Any,
        X: Any | None = None,
        predictor: Any | None = None,
        predictor_bundle: str | Path | None = None,
        mode: str = "clean_frozen",
        scenarios: Sequence[Any] | None = None,
        slice_by: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
        seed: int | None = None,
        workspace_path: str | Path | None = None,
        workspace_name: str = "",
        workspace_robustness_id: str | None = None,
        workspace_metadata: Mapping[str, Any] | None = None,
    ) -> RobustnessReport:
        """Compute an audit-only robustness report for this calibrated result."""

        from nirs4all.api.robustness import RobustnessMode, robustness

        return robustness(
            self,
            y_true=y_true,
            X=X,
            predictor=predictor,
            predictor_bundle=predictor_bundle,
            mode=cast(RobustnessMode, mode),
            scenarios=scenarios,
            slice_by=slice_by,
            metadata=metadata,
            seed=seed,
            workspace_path=workspace_path,
            workspace_name=workspace_name,
            workspace_robustness_id=workspace_robustness_id,
            workspace_metadata=workspace_metadata,
        )


def _validate_calibrated_result_guarantee_metadata(
    metadata: Mapping[str, Any],
    *,
    artifact: ConformalCalibrationArtifact,
    prediction_coverages: tuple[float, ...],
) -> None:
    """Fail closed when persisted guarantee metadata no longer matches the artifact."""

    status = metadata.get("conformal_guarantee_status")
    if status is None:
        return
    if not isinstance(status, Mapping):
        raise ValueError("CalibratedRunResult.conformal_guarantee_status metadata must be a mapping")
    required_status_keys = {
        "artifact_fingerprint",
        "calibrated_coverages",
        "calibration_data_fingerprint",
        "coverage",
        "effective_engine",
        "group_by",
        "guarantee",
        "invalidation_reasons",
        "limitations",
        "method",
        "multi_target",
        "predictor_fingerprint",
        "requested_engine",
        "scope",
        "source_calibrated_result_fingerprint",
        "status",
        "unit",
        "version",
    }
    missing_status_keys = sorted(required_status_keys - set(status))
    if missing_status_keys:
        raise ValueError(f"CalibratedRunResult.conformal_guarantee_status is missing keys {missing_status_keys}")
    status_version = status.get("version")
    if status_version is not None and _decode_contract_version(status_version, "CalibratedRunResult.conformal_guarantee_status.version") != 1:
        raise ValueError("CalibratedRunResult.conformal_guarantee_status.version must be 1")
    effective_engine = status.get("effective_engine")
    if effective_engine is not None:
        _validate_required_payload_string_value(effective_engine, "CalibratedRunResult.conformal_guarantee_status.effective_engine")
    requested_engine = status.get("requested_engine")
    if requested_engine is not None:
        _validate_required_payload_string_value(requested_engine, "CalibratedRunResult.conformal_guarantee_status.requested_engine")
    invalidation_reasons = status.get("invalidation_reasons")
    if invalidation_reasons is not None:
        invalidation_reasons = _normalize_invalidation_reasons(invalidation_reasons, "CalibratedRunResult.conformal_guarantee_status.invalidation_reasons")
    status_value = status.get("status")
    if status_value not in {"active", "invalidated"}:
        raise ValueError("CalibratedRunResult.conformal_guarantee_status.status must be 'active' or 'invalidated'")
    expected_status_value = "invalidated" if invalidation_reasons else "active"
    if status_value != expected_status_value:
        raise ValueError("CalibratedRunResult.conformal_guarantee_status.status does not match invalidation_reasons")
    artifact_fingerprint = status.get("artifact_fingerprint")
    if artifact_fingerprint is not None and artifact_fingerprint != artifact.fingerprint:
        raise ValueError("CalibratedRunResult conformal_guarantee_status artifact_fingerprint does not match the conformal artifact")
    _validate_status_optional_string_matches(
        status,
        "predictor_fingerprint",
        artifact.predictor_fingerprint,
        "CalibratedRunResult.conformal_guarantee_status.predictor_fingerprint",
    )
    _validate_status_optional_string_matches(
        status,
        "calibration_data_fingerprint",
        artifact.calibration_data_fingerprint,
        "CalibratedRunResult.conformal_guarantee_status.calibration_data_fingerprint",
    )
    expected_guarantee, expected_scope = _expected_conformal_guarantee_and_scope(artifact)
    guarantee = status.get("guarantee")
    if guarantee is not None and guarantee != expected_guarantee:
        raise ValueError("CalibratedRunResult conformal_guarantee_status guarantee does not match the conformal artifact")
    scope = status.get("scope")
    if scope is not None and scope != expected_scope:
        raise ValueError("CalibratedRunResult conformal_guarantee_status scope does not match the conformal artifact")
    limitations = _normalize_status_limitations(
        status.get("limitations"),
        "CalibratedRunResult.conformal_guarantee_status.limitations",
    )
    if limitations != _expected_conformal_limitations(artifact):
        raise ValueError("CalibratedRunResult conformal_guarantee_status limitations do not match the conformal artifact")
    calibrated_coverages = status.get("calibrated_coverages")
    if calibrated_coverages is not None and tuple(normalize_coverages(calibrated_coverages)) != artifact.spec.coverage:
        raise ValueError("CalibratedRunResult conformal_guarantee_status calibrated_coverages do not match the conformal artifact")
    coverage = status.get("coverage")
    if coverage is not None:
        selected = tuple(normalize_coverages(coverage))
        if any(value not in prediction_coverages for value in selected):
            raise ValueError("CalibratedRunResult conformal_guarantee_status coverage contains non-materialized coverage")
    method = status.get("method")
    if method is not None and method != artifact.spec.method:
        raise ValueError("CalibratedRunResult conformal_guarantee_status method does not match the conformal artifact")
    unit = status.get("unit")
    if unit is not None and unit != artifact.spec.unit:
        raise ValueError("CalibratedRunResult conformal_guarantee_status unit does not match the conformal artifact")
    multi_target = status.get("multi_target")
    if multi_target is not None and multi_target != artifact.spec.multi_target:
        raise ValueError("CalibratedRunResult conformal_guarantee_status multi_target does not match the conformal artifact")
    group_by = status.get("group_by")
    if group_by is not None and tuple(_normalize_group_by(group_by, "CalibratedRunResult.conformal_guarantee_status.group_by", allow_empty=True)) != artifact.spec.group_by:
        raise ValueError("CalibratedRunResult conformal_guarantee_status group_by does not match the conformal artifact")
    metadata_source = _validate_optional_payload_string_value(
        metadata.get("source_calibrated_result_fingerprint"),
        "CalibratedRunResult.source_calibrated_result_fingerprint",
    )
    status_source = _validate_optional_payload_string_value(
        status.get("source_calibrated_result_fingerprint"),
        "CalibratedRunResult.conformal_guarantee_status.source_calibrated_result_fingerprint",
    )
    if metadata_source is not None and status_source is not None and metadata_source != status_source:
        raise ValueError("CalibratedRunResult source_calibrated_result_fingerprint does not match conformal_guarantee_status source_calibrated_result_fingerprint")
    metadata_replay_source = metadata.get("calibration_replay_source")
    status_replay_source = status.get("calibration_replay_source")
    if metadata_replay_source is not None and status_replay_source is not None:
        if not isinstance(metadata_replay_source, Mapping) or not isinstance(status_replay_source, Mapping):
            raise ValueError("CalibratedRunResult calibration_replay_source metadata must be mappings")
        if dict(metadata_replay_source) != dict(status_replay_source):
            raise ValueError("CalibratedRunResult calibration_replay_source does not match conformal_guarantee_status calibration_replay_source")


def _validate_calibrated_result_metadata(metadata: Mapping[str, Any]) -> None:
    """Validate deterministic JSON-compatible result metadata."""

    if not isinstance(metadata, Mapping):
        raise ValueError("CalibratedRunResult.metadata must be a mapping")
    for key, value in metadata.items():
        name = _normalize_non_empty_string(key, "CalibratedRunResult.metadata key")
        if name != key:
            raise ValueError("CalibratedRunResult.metadata keys must not contain surrounding whitespace")
        _validate_json_compatible(value, f"CalibratedRunResult.metadata[{name}]")


def _validate_calibrated_prediction_matches_artifact(
    artifact: ConformalCalibrationArtifact,
    prediction: CalibratedPredictionBlock,
) -> None:
    """Fail closed when materialized intervals no longer derive from the artifact."""

    y_pred = _as_conformal_target_array(prediction.y_pred, "CalibratedRunResult.prediction.y_pred", spec=artifact.spec)
    if artifact.spec.group_by:
        if len(prediction.group_keys) != int(_n_prediction_rows(y_pred)):
            raise ValueError("CalibratedRunResult grouped predictions require one group key per prediction row")
    elif prediction.group_keys:
        raise ValueError("CalibratedRunResult prediction group_keys require a grouped conformal artifact")
    for coverage in prediction.coverages:
        interval = prediction.interval(coverage)
        if interval.coverage != coverage:
            raise ValueError("CalibratedRunResult interval coverage must match its coverage key")
        expected_qhat = _expected_qhat_for_prediction_rows(artifact, coverage, prediction.group_keys)
        if not _conformal_qhat_equal(interval.qhat, expected_qhat):
            raise ValueError("CalibratedRunResult interval qhat must match the conformal artifact")
        lower = _as_interval_array(interval.lower, "lower", y_pred.shape)
        upper = _as_interval_array(interval.upper, "upper", y_pred.shape)
        if np.any(lower > upper):
            raise ValueError("CalibratedRunResult interval lower bounds must be <= upper bounds")
        broadcast_qhat = _broadcast_qhat_to_prediction_shape(expected_qhat, y_pred.shape)
        if not np.array_equal(lower, y_pred - broadcast_qhat) or not np.array_equal(upper, y_pred + broadcast_qhat):
            raise ValueError("CalibratedRunResult interval arrays must be derived from y_pred and the conformal artifact qhat")


def _validate_prediction_sample_ids_are_disjoint_from_calibration(
    artifact: ConformalCalibrationArtifact,
    sample_ids: Sequence[Any],
) -> None:
    """Fail closed when prediction rows reuse physical samples from calibration."""

    if artifact.calibration_cohort is None or not sample_ids:
        return
    calibration_ids = set(artifact.calibration_cohort.sample_ids)
    prediction_ids = set(sample_ids)
    overlap = sorted(calibration_ids & prediction_ids)
    if overlap:
        preview = ", ".join(overlap[:5])
        suffix = "" if len(overlap) <= 5 else f", ... (+{len(overlap) - 5} more)"
        raise ValueError(f"CalibratedRunResult prediction sample_ids must be disjoint from calibration_cohort sample_ids; overlapping physical samples: {preview}{suffix}")


def _validate_prediction_sample_ids(sample_ids: Sequence[Any]) -> None:
    """Validate canonical physical ids for a calibrated prediction cohort."""

    if not sample_ids:
        return
    normalized: list[str] = []
    for sample_id in sample_ids:
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("CalibratedRunResult sample_ids must contain non-empty strings")
        if sample_id != sample_id.strip():
            raise ValueError("CalibratedRunResult sample_ids must not contain surrounding whitespace")
        if "\x00" in sample_id:
            raise ValueError("CalibratedRunResult sample_ids must not contain NUL bytes")
        normalized.append(sample_id)
    if len(set(normalized)) != len(normalized):
        raise ValueError("CalibratedRunResult sample_ids must be unique physical sample identifiers")


def normalize_coverages(coverage: float | list[float] | tuple[float, ...]) -> tuple[float, ...]:
    """Normalize coverage scalar/list to a sorted unique tuple."""

    if isinstance(coverage, (bool, np.bool_)):
        raise ValueError("coverage values must be numeric floats, not booleans")
    if isinstance(coverage, (str, bytes)):
        raise ValueError("coverage values must be numeric floats, not strings")
    if isinstance(coverage, (int, float, np.integer, np.floating)):
        values = [coverage]
    else:
        try:
            values = list(coverage)
        except TypeError as exc:
            raise ValueError("coverage must be a numeric scalar or sequence of numeric scalars") from exc
    normalized: list[float] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("coverage values must be numeric floats, not booleans")
        if isinstance(value, (str, bytes)):
            raise ValueError("coverage values must be numeric floats, not strings")
        cov = float(value)
        if not math.isfinite(cov) or cov <= 0.0 or cov >= 1.0:
            raise ValueError("coverage values must be finite floats in the open interval (0, 1)")
        normalized.append(cov)
    unique = tuple(sorted(set(normalized)))
    if not unique:
        raise ValueError("at least one coverage value is required")
    return unique


def _normalize_guarantee_coverages(selected_coverages: Any | None, available_coverages: tuple[float, ...]) -> tuple[float, ...]:
    if selected_coverages is None:
        return available_coverages
    coverages = normalize_coverages(selected_coverages)
    missing = sorted(set(coverages) - set(available_coverages))
    if missing:
        available = ", ".join(str(value) for value in available_coverages)
        raise KeyError(f"coverage values {missing} are not materialized; available coverages: {available}")
    return coverages


def conformal_finite_sample_quantile(scores: Any, coverage: float) -> float:
    """Return finite-sample split conformal quantile.

    Uses ``k = ceil((n + 1) * coverage)`` over sorted nonconformity scores. If
    ``k > n``, the finite-sample conformal interval is unbounded.
    """

    values = np.sort(_as_1d_float_array(scores, "scores"))
    cov = normalize_coverages(coverage)[0]
    n = values.size
    if n == 0:
        raise ValueError("scores must contain at least one calibration residual")
    k = math.ceil((n + 1) * cov)
    if k > n:
        return float("inf")
    return float(values[k - 1])


def _as_1d_float_array(value: Any, label: str) -> np.ndarray:
    _reject_non_numeric_conformal_values(value, label)
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return array


def _is_integer_scalar(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _decode_contract_version(value: Any, label: str) -> int:
    if not _is_integer_scalar(value):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _validate_metric_scalar(value: Any, label: str, *, allow_infinity: bool) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a numeric scalar, not boolean")
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a numeric scalar, not string")
    if isinstance(value, np.ndarray):
        raise ValueError(f"{label} must be a numeric scalar")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric scalar") from exc
    if math.isnan(number):
        raise ValueError(f"{label} must not be NaN")
    if not allow_infinity and not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _validate_finite_metric(value: Any, label: str) -> float:
    return _validate_metric_scalar(value, label, allow_infinity=False)


def _validate_probability_metric(value: Any, label: str) -> float:
    number = _validate_finite_metric(value, label)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be finite in [0, 1]")
    return number


def _validate_non_negative_metric(value: Any, label: str) -> None:
    number = _validate_metric_scalar(value, label, allow_infinity=True)
    if number < 0.0:
        raise ValueError(f"{label} must be non-negative")


def _reject_non_numeric_conformal_values(value: Any, label: str) -> None:
    """Reject values NumPy/JSON numeric conversion would silently coerce."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must contain numeric values, not booleans")
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must contain numeric values, not strings")
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            raise ValueError(f"{label} must contain numeric values, not booleans")
        if np.issubdtype(value.dtype, np.str_) or np.issubdtype(value.dtype, np.bytes_):
            raise ValueError(f"{label} must contain numeric values, not strings")
        if value.dtype == object:
            for item in value.flat:
                _reject_non_numeric_conformal_values(item, label)
        return
    if isinstance(value, Mapping):
        return
    if isinstance(value, Sequence):
        for item in value:
            _reject_non_numeric_conformal_values(item, label)


def _as_2d_float_array(value: Any, label: str) -> np.ndarray:
    _reject_non_numeric_conformal_values(value, label)
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a two-dimensional array")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"{label} must contain at least one sample and one target")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return array


def _as_conformal_observation_array(value: Any, label: str) -> np.ndarray:
    _reject_non_numeric_conformal_values(value, label)
    array = np.asarray(value, dtype=float)
    if array.ndim not in (1, 2):
        raise ValueError(f"{label} must be a one- or two-dimensional array")
    if array.ndim == 2 and (array.shape[0] <= 0 or array.shape[1] <= 0):
        raise ValueError(f"{label} must contain at least one sample and one target")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return array


def _as_conformal_target_array(value: Any, label: str, *, spec: ConformalCalibrationSpec) -> np.ndarray:
    if spec.multi_target == "joint_max":
        return _as_2d_float_array(value, label)
    return _as_1d_float_array(value, label)


def _n_prediction_rows(array: np.ndarray) -> int:
    if array.ndim == 0:
        raise ValueError("conformal arrays must have at least one dimension")
    return int(array.shape[0])


def _row_any(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 1:
        return cast(np.ndarray, mask)
    return cast(np.ndarray, np.any(mask, axis=1))


def _broadcast_qhat_to_prediction_shape(qhat: Any, prediction_shape: tuple[int, ...]) -> np.ndarray | float:
    _reject_non_numeric_conformal_values(qhat, "conformal interval qhat")
    array = np.asarray(qhat, dtype=float)
    if array.ndim == 0:
        return float(array)
    if array.ndim != 1:
        raise ValueError("conformal interval qhat must be scalar or one-dimensional")
    if len(prediction_shape) == 1:
        if array.shape != prediction_shape:
            raise ValueError("row-aligned qhat length must match y_pred")
        return cast(np.ndarray, array)
    if array.shape != (prediction_shape[0],):
        raise ValueError("row-aligned qhat length must match y_pred rows")
    return cast(np.ndarray, array[:, np.newaxis])


def _normalize_interval_qhat(value: Any, interval_shape: tuple[int, ...]) -> float | np.ndarray:
    _reject_non_numeric_conformal_values(value, "conformal interval qhat")
    array = np.asarray(value, dtype=float)
    if np.any(np.isnan(array)):
        raise ValueError("conformal interval qhat cannot contain NaN")
    if np.any(array < 0.0):
        raise ValueError("conformal interval qhat must be non-negative")
    _broadcast_qhat_to_prediction_shape(array, interval_shape)
    if array.ndim == 0:
        return float(array)
    return array.astype(float, copy=False)


def _as_interval_array(value: Any, label: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    _reject_non_numeric_conformal_values(value, label)
    array = np.asarray(value, dtype=float)
    if array.shape != expected_shape:
        raise ValueError(f"{label} interval array must match y_true shape")
    if np.any(np.isnan(array)):
        raise ValueError(f"{label} interval array cannot contain NaN")
    return array


def _as_1d_interval_json_float_array(value: Any, label: str) -> np.ndarray:
    _reject_non_numeric_conformal_values(value, label)
    array = np.asarray(value, dtype=float)
    if array.ndim not in (1, 2):
        raise ValueError(f"{label} must be a one- or two-dimensional array")
    if np.any(np.isnan(array)):
        raise ValueError(f"{label} interval array cannot contain NaN")
    return array


def _normalize_group_by(value: Any, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        group = value.strip()
        if not group:
            raise ValueError(f"{label} must contain non-empty strings")
        return (group,)
    try:
        values = list(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be null, a string, or a sequence of strings") from exc
    groups: list[str] = []
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{label} must contain non-empty strings")
        groups.append(item.strip())
    unique = tuple(sorted(set(groups)))
    if not unique:
        if allow_empty:
            return ()
        raise ValueError(f"{label} must contain at least one group key when provided")
    if len(unique) != len(groups):
        raise ValueError(f"{label} contains duplicate keys after whitespace normalization")
    return unique


def _normalize_sample_ids(sample_ids: Any, expected_size: int) -> tuple[str, ...]:
    if sample_ids is None:
        raise ValueError("sample_ids are required to build a physical-sample conformal calibration cohort")
    if isinstance(sample_ids, (str, bytes)):
        raise ValueError("sample_ids must be a row-aligned sequence, not a scalar string")
    try:
        values = list(sample_ids)
    except TypeError as exc:
        raise ValueError("sample_ids must be a row-aligned sequence") from exc
    if len(values) != expected_size:
        raise ValueError("sample_ids length must match calibration arrays")
    normalized = tuple(_normalize_canonical_sample_id(value, "sample_ids") for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError("sample_ids must be unique physical sample identifiers")
    return normalized


def _normalize_optional_groups(groups: Any | None, expected_size: int) -> tuple[str | None, ...]:
    if groups is None:
        return tuple(None for _ in range(expected_size))
    if isinstance(groups, (str, bytes)):
        raise ValueError("groups must be a row-aligned sequence, not a scalar string")
    try:
        values = list(groups)
    except TypeError as exc:
        raise ValueError("groups must be a row-aligned sequence") from exc
    if len(values) != expected_size:
        raise ValueError("groups length must match calibration arrays")
    normalized: list[str | None] = []
    for value in values:
        if value is None:
            normalized.append(None)
        else:
            normalized.append(_validate_required_payload_string_value(value, "groups"))
    return tuple(normalized)


def _normalize_metadata_rows(metadata: Any | None, expected_size: int) -> tuple[dict[str, Any], ...]:
    if metadata is None:
        return tuple({} for _ in range(expected_size))
    if isinstance(metadata, Mapping):
        rows: list[dict[str, Any]] = [{} for _ in range(expected_size)]
        for key, column in metadata.items():
            name = _validate_required_payload_string_value(key, "metadata key")
            if isinstance(column, (str, bytes)):
                raise ValueError("metadata columns must be row-aligned sequences, not scalar strings")
            try:
                values = list(column)
            except TypeError as exc:
                raise ValueError("metadata columns must be row-aligned sequences") from exc
            if len(values) != expected_size:
                raise ValueError("metadata column length must match calibration arrays")
            for index, value in enumerate(values):
                _validate_json_compatible(value, f"metadata[{name}]")
                rows[index][name] = value
        return tuple(rows)
    if isinstance(metadata, (str, bytes)):
        raise ValueError("metadata must be a mapping of columns or a sequence of row mappings")
    try:
        row_values = list(metadata)
    except TypeError as exc:
        raise ValueError("metadata must be a mapping of columns or a sequence of row mappings") from exc
    if len(row_values) != expected_size:
        raise ValueError("metadata row count must match calibration arrays")
    rows = []
    for row in row_values:
        if not isinstance(row, Mapping):
            raise ValueError("metadata rows must be mappings")
        normalized_row: dict[str, Any] = {}
        for key, value in row.items():
            name = _validate_required_payload_string_value(key, "metadata key")
            _validate_json_compatible(value, f"metadata[{name}]")
            normalized_row[name] = value
        rows.append(normalized_row)
    return tuple(rows)


def _normalize_prediction_group_keys(
    group_by: tuple[str, ...],
    *,
    groups: Any | None,
    metadata: Any | None,
    expected_size: int,
) -> tuple[str, ...]:
    if not group_by:
        return ()
    group_values = _normalize_optional_groups(groups, expected_size)
    metadata_rows = _normalize_metadata_rows(metadata, expected_size)
    return tuple(
        _encode_group_key(
            _extract_group_key_values(
                group_by,
                group_value=group_values[index],
                metadata=metadata_rows[index],
                context=f"prediction row {index}",
            )
        )
        for index in range(expected_size)
    )


def _fit_group_calibrators(
    *,
    y_true: Any,
    y_pred: Any,
    spec: ConformalCalibrationSpec,
    calibration_cohort: ConformalCalibrationCohortManifest,
) -> dict[str, SplitConformalCalibrator]:
    truth = _as_conformal_target_array(y_true, "y_true", spec=spec)
    predictions = _as_conformal_target_array(y_pred, "y_pred", spec=spec)
    if truth.shape != predictions.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    group_indices: dict[str, list[int]] = {}
    for row in calibration_cohort.rows:
        key = _cohort_row_group_key(row, spec.group_by)
        group_indices.setdefault(key, []).append(row.row_index)
    return {
        key: (
            fit_joint_max_absolute_residual_calibrator(
                truth[indices],
                predictions[indices],
                coverage=spec.coverage,
                unit=spec.unit,
            )
            if spec.multi_target == "joint_max"
            else fit_split_absolute_residual_calibrator(
                truth[indices],
                predictions[indices],
                coverage=spec.coverage,
                unit=spec.unit,
            )
        )
        for key, indices in sorted(group_indices.items())
    }


def _cohort_row_group_key(row: ConformalCalibrationCohortRow, group_by: tuple[str, ...]) -> str:
    return _encode_group_key(
        _extract_group_key_values(
            group_by,
            group_value=row.group,
            metadata=row.metadata,
            context=f"calibration cohort row {row.row_index}",
        )
    )


def _extract_group_key_values(
    group_by: tuple[str, ...],
    *,
    group_value: str | None,
    metadata: Mapping[str, Any],
    context: str,
) -> tuple[Any, ...]:
    values: list[Any] = []
    for key in group_by:
        if key == "group":
            if group_value is None:
                raise ValueError(f"{context} is missing required group value for conformal group_by='group'")
            value = group_value
        else:
            if key not in metadata:
                raise ValueError(f"{context} metadata is missing required conformal group_by key {key!r}")
            value = metadata[key]
            _validate_json_compatible(value, f"{context}.metadata[{key}]")
            if value is None:
                raise ValueError(f"{context} metadata key {key!r} cannot be null for conformal group_by")
        values.append(value)
    return tuple(values)


def _encode_group_key(values: Sequence[Any]) -> str:
    if not values:
        raise ValueError("conformal group key requires at least one value")
    return json.dumps(list(values), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_group_calibrators(
    group_calibrators: Mapping[str, SplitConformalCalibrator],
    *,
    spec: ConformalCalibrationSpec,
    calibration_cohort: ConformalCalibrationCohortManifest | None,
    calibration_size: int,
) -> None:
    if not spec.group_by:
        if group_calibrators:
            raise ValueError("group_calibrators require a conformal spec with group_by")
        return
    if calibration_cohort is None:
        raise ValueError("grouped conformal artifacts require a calibration_cohort")
    if not group_calibrators:
        raise ValueError("grouped conformal artifacts require group_calibrators")
    expected_keys = tuple(sorted(_cohort_row_group_key(row, spec.group_by) for row in calibration_cohort.rows))
    actual_keys = tuple(sorted(group_calibrators))
    if sorted(set(expected_keys)) != list(actual_keys):
        raise ValueError("group_calibrators keys must match calibration_cohort group keys")
    total_scores = 0
    for key, calibrator in group_calibrators.items():
        _normalize_non_empty_string(key, "group_calibrators key")
        scores = _as_1d_float_array(calibrator.scores, f"group_calibrators[{key}].scores")
        if np.any(scores < 0.0):
            raise ValueError("group_calibrators scores must contain non-negative conformal residuals")
        total_scores += int(scores.size)
        if calibrator.coverages != spec.coverage:
            raise ValueError("group_calibrators coverage must match artifact spec coverage")
        if calibrator.method != spec.method:
            raise ValueError("group_calibrators method must match artifact spec method")
        if calibrator.unit != spec.unit:
            raise ValueError("group_calibrators unit must match artifact spec unit")
        if tuple(sorted(calibrator.qhat_by_coverage)) != spec.coverage:
            raise ValueError("group_calibrators qhat_by_coverage coverages must match artifact spec coverage")
        for coverage in spec.coverage:
            expected_qhat = conformal_finite_sample_quantile(scores, coverage)
            if not _conformal_float_equal(calibrator.qhat_by_coverage[coverage], expected_qhat):
                raise ValueError("group_calibrators qhat_by_coverage must match retained conformal scores")
    if total_scores != calibration_size:
        raise ValueError("group_calibrators score counts must sum to calibration_size")


def _group_calibrators_to_dict(group_calibrators: Mapping[str, SplitConformalCalibrator]) -> list[dict[str, Any]]:
    return [
        {
            "group_key": key,
            "n_samples": int(calibrator.scores.size),
            "qhat_by_coverage": [{"coverage": coverage, "qhat": _encode_json_float(calibrator.qhat_by_coverage[coverage])} for coverage in sorted(calibrator.qhat_by_coverage)],
            "scores": [_encode_json_float(float(score)) for score in calibrator.scores.tolist()],
        }
        for key, calibrator in sorted(group_calibrators.items())
    ]


def _group_calibrators_from_payload(
    payload: Any,
    *,
    spec: ConformalCalibrationSpec,
) -> dict[str, SplitConformalCalibrator]:
    if payload is None:
        return {}
    if not isinstance(payload, list):
        raise ValueError("ConformalCalibrationArtifact.group_calibrators must be a list")
    calibrators: dict[str, SplitConformalCalibrator] = {}
    for item in payload:
        if not isinstance(item, Mapping):
            raise ValueError("ConformalCalibrationArtifact.group_calibrators entries must be mappings")
        key = _normalize_non_empty_string(item.get("group_key"), "ConformalCalibrationArtifact.group_calibrators.group_key")
        if key in calibrators:
            raise ValueError("ConformalCalibrationArtifact.group_calibrators contains duplicate group_key values")
        if not isinstance(item.get("scores"), list):
            raise ValueError("ConformalCalibrationArtifact.group_calibrators.scores must be a list")
        if not isinstance(item.get("qhat_by_coverage"), list):
            raise ValueError("ConformalCalibrationArtifact.group_calibrators.qhat_by_coverage must be a list")
        qhat_by_coverage: dict[float, float] = {}
        for qhat_item in item["qhat_by_coverage"]:
            if not isinstance(qhat_item, Mapping) or "coverage" not in qhat_item or "qhat" not in qhat_item:
                raise ValueError("ConformalCalibrationArtifact.group_calibrators qhat entries require coverage and qhat")
            coverage = normalize_coverages(qhat_item["coverage"])[0]
            if coverage in qhat_by_coverage:
                raise ValueError("ConformalCalibrationArtifact.group_calibrators contains duplicate coverage values")
            qhat_by_coverage[coverage] = _decode_json_float(qhat_item["qhat"], "qhat")
        scores = np.asarray([_decode_json_float(score, "score") for score in item["scores"]], dtype=float)
        expected_n = item.get("n_samples")
        if expected_n is not None:
            if not _is_integer_scalar(expected_n):
                raise ValueError("ConformalCalibrationArtifact.group_calibrators n_samples must be an integer")
            if int(expected_n) != int(scores.size):
                raise ValueError("ConformalCalibrationArtifact.group_calibrators n_samples mismatch")
        calibrators[key] = SplitConformalCalibrator(
            scores=scores,
            coverages=spec.coverage,
            qhat_by_coverage=qhat_by_coverage,
            method=spec.method,
            unit=spec.unit,
        )
    return calibrators


def _optional_lower_string(mapping: Mapping[str, Any], key: str, *, default: str) -> str:
    if key not in mapping or mapping[key] is None:
        return default
    return _normalize_non_empty_string(mapping[key], key).lower()


def _normalize_supported_contract_string(value: Any, label: str, supported: frozenset[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be one of {sorted(supported)}")
    normalized = value.strip().lower()
    if normalized not in supported:
        raise ValueError(f"{label} must be one of {sorted(supported)}")
    return normalized


def _normalize_non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _normalize_canonical_sample_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip() or "\x00" in value:
        raise ValueError(f"{label} must contain canonical non-empty strings")
    return value


def _strict_payload_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a string")
    return value


def _validate_json_compatible(value: Any, label: str) -> None:
    _validate_json_native(value, label)
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON-compatible") from exc


def _validate_json_native(value: Any, label: str) -> None:
    """Validate that a value is JSON-native without silent key/type coercion."""

    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must be JSON-compatible")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{label} mapping keys must be non-empty strings")
            if key != key.strip():
                raise ValueError(f"{label} mapping keys must not contain surrounding whitespace")
            _validate_json_native(item, f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_native(item, f"{label}[{index}]")
        return
    raise ValueError(f"{label} must be JSON-compatible")


def _optional_payload_string(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return _validate_optional_payload_string_value(value, f"ConformalCalibrationArtifact.{key}")


def _validate_optional_payload_string_value(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be null or a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} must not contain surrounding whitespace")
    if "\x00" in value:
        raise ValueError(f"{label} must not contain NUL bytes")
    return value


def _validate_required_payload_string_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} must not contain surrounding whitespace")
    if "\x00" in value:
        raise ValueError(f"{label} must not contain NUL bytes")
    return value


def _normalize_invalidation_reasons(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence of non-empty strings")
    reasons: list[str] = []
    for index, reason in enumerate(value):
        reasons.append(_validate_required_payload_string_value(reason, f"{label}[{index}]"))
    return tuple(reasons)


def _normalize_status_limitations(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence of non-empty strings")
    limitations: list[str] = []
    for index, limitation in enumerate(value):
        limitations.append(_validate_required_payload_string_value(limitation, f"{label}[{index}]"))
    return tuple(limitations)


def _validate_status_optional_string_matches(
    status: Mapping[str, Any],
    key: str,
    expected: str | None,
    label: str,
) -> None:
    if expected is None and key not in status:
        return
    value = _validate_optional_payload_string_value(status.get(key), label)
    if value != expected:
        raise ValueError(f"{label} does not match the conformal artifact")


def _expected_conformal_guarantee_and_scope(artifact: ConformalCalibrationArtifact) -> tuple[str, str]:
    if artifact.spec.group_by:
        guarantee = "split_conformal_group_joint_max_simultaneous_coverage" if artifact.spec.multi_target == "joint_max" else "split_conformal_group_marginal_coverage"
        return guarantee, "finite_sample_group_conditional_exchangeability"
    if artifact.spec.multi_target == "joint_max":
        return "split_conformal_joint_max_simultaneous_coverage", "finite_sample_simultaneous_exchangeability"
    return "split_conformal_marginal_coverage", "finite_sample_marginal_exchangeability"


def _expected_conformal_limitations(artifact: ConformalCalibrationArtifact) -> tuple[str, ...]:
    if artifact.spec.group_by:
        return (
            "finite-sample group-conditional coverage requires exchangeable calibration and prediction samples within each materialized group",
            "prediction rows must carry group keys present in the calibration artifact; unseen or missing groups fail closed",
            "coverage is not conditional on robustness slices outside the declared group_by keys",
            "empirical conformal metrics are diagnostics and do not renew the guarantee",
            "the calibrator is stale if predictor, preprocessing, target transform, calibration cohort, or group routing changes",
        )
    if artifact.spec.multi_target == "joint_max":
        return (
            "finite-sample simultaneous coverage requires exchangeable calibration and prediction samples",
            "the joint_max nonconformity score is max(abs residual) across target columns for each physical sample",
            "coverage is for the full target vector, not separate per-target conditional guarantees",
            "empirical conformal metrics are diagnostics and do not renew the guarantee",
            "the calibrator is stale if predictor, preprocessing, target transform, or calibration cohort changes",
        )
    return (
        "finite-sample marginal coverage requires exchangeable calibration and prediction samples",
        "coverage is not conditional on groups, batches, instruments, or robustness slices",
        "empirical conformal metrics are diagnostics and do not renew the guarantee",
        "the calibrator is stale if predictor, preprocessing, target transform, or calibration cohort changes",
    )


def _calibrated_prediction_block_to_dict(block: CalibratedPredictionBlock) -> dict[str, Any]:
    return {
        "group_keys": list(block.group_keys),
        "intervals": [
            {
                "coverage": coverage,
                "lower": _encode_json_float_array(interval.lower),
                "qhat": _encode_json_qhat(interval.qhat),
                "upper": _encode_json_float_array(interval.upper),
            }
            for coverage, interval in sorted(block.intervals.items())
        ],
        "method": block.method,
        "unit": block.unit,
        "y_pred": _encode_json_float_array(block.y_pred),
    }


def _calibrated_prediction_block_from_dict(payload: Mapping[str, Any]) -> CalibratedPredictionBlock:
    if not isinstance(payload, Mapping):
        raise TypeError("CalibratedRunResult.prediction must be a mapping")
    required = {"intervals", "method", "unit", "y_pred"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"CalibratedRunResult.prediction is missing keys {missing}")
    y_pred = _decode_json_float_array(payload["y_pred"], "y_pred")
    intervals: dict[float, ConformalIntervalBlock] = {}
    if not isinstance(payload["intervals"], list):
        raise ValueError("CalibratedRunResult.prediction.intervals must be a list")
    for item in payload["intervals"]:
        if not isinstance(item, Mapping):
            raise ValueError("CalibratedRunResult.prediction.intervals entries must be mappings")
        raw_coverage = item.get("coverage")
        if not isinstance(raw_coverage, (float, int, list, tuple)):
            raise ValueError("CalibratedRunResult.prediction.intervals.coverage must be numeric")
        coverage = normalize_coverages(cast(float | list[float] | tuple[float, ...], raw_coverage))[0]
        if coverage in intervals:
            raise ValueError("CalibratedRunResult.prediction.intervals contains duplicate coverage values")
        lower = _as_1d_interval_json_float_array(_decode_json_float_array(item.get("lower", []), "lower"), "lower")
        upper = _as_1d_interval_json_float_array(_decode_json_float_array(item.get("upper", []), "upper"), "upper")
        if lower.shape != y_pred.shape or upper.shape != y_pred.shape:
            raise ValueError("CalibratedRunResult.prediction interval arrays must match y_pred shape")
        intervals[coverage] = ConformalIntervalBlock(
            coverage=coverage,
            qhat=_decode_json_qhat(item.get("qhat"), "qhat"),
            lower=lower,
            upper=upper,
        )
    group_keys_payload = payload.get("group_keys", [])
    if not isinstance(group_keys_payload, list) or not all(isinstance(key, str) and key for key in group_keys_payload):
        raise ValueError("CalibratedRunResult.prediction.group_keys must be a list of non-empty strings")
    return CalibratedPredictionBlock(
        y_pred=y_pred,
        intervals=intervals,
        method=cast(ConformalMethod, _strict_payload_string(payload["method"], "CalibratedRunResult.prediction.method")),
        unit=_strict_payload_string(payload["unit"], "CalibratedRunResult.prediction.unit"),
        group_keys=tuple(group_keys_payload),
    )


def _encode_json_float(value: float) -> float | str:
    number = float(value)
    if math.isinf(number):
        return "Infinity" if number > 0 else "-Infinity"
    if not math.isfinite(number):
        raise ValueError("conformal artifacts cannot serialize NaN float values")
    return number


def _decode_json_float(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must contain numeric values, not booleans")
    if isinstance(value, np.ndarray):
        raise ValueError(f"{label} must be a JSON numeric scalar, not a NumPy array")
    if isinstance(value, (str, bytes)) and value not in {"Infinity", "-Infinity"}:
        raise ValueError(f"{label} must contain numeric values, not strings")
    if value == "Infinity":
        return float("inf")
    if value == "-Infinity":
        return float("-inf")
    number = float(value)
    if math.isnan(number):
        raise ValueError(f"{label} cannot be NaN")
    return number


def _encode_json_qhat(value: Any) -> float | str | list[float | str]:
    _reject_non_numeric_conformal_values(value, "conformal interval qhat")
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return _encode_json_float(float(array))
    if array.ndim != 1:
        raise ValueError("conformal interval qhat must be scalar or one-dimensional")
    return [_encode_json_float(float(item)) for item in array.tolist()]


def _encode_json_float_array(value: Any) -> list[Any]:
    _reject_non_numeric_conformal_values(value, "conformal arrays")
    array = np.asarray(value, dtype=float)
    if array.ndim not in (1, 2):
        raise ValueError("conformal arrays must be one- or two-dimensional")
    return cast(list[Any], _encode_json_float_array_payload(array.tolist()))


def _encode_json_float_array_payload(value: Any) -> Any:
    if isinstance(value, list):
        return cast(list[Any], [_encode_json_float_array_payload(item) for item in value])
    return _encode_json_float(float(value))


def _decode_json_float_array(value: Any, label: str) -> np.ndarray:
    try:
        decoded = _decode_json_float_array_payload(value, label)
    except TypeError as exc:
        raise ValueError(f"{label} must be a one- or two-dimensional array") from exc
    array = np.asarray(decoded, dtype=float)
    if array.ndim not in (1, 2):
        raise ValueError(f"{label} must be a one- or two-dimensional array")
    if array.ndim == 2 and (array.shape[0] <= 0 or array.shape[1] <= 0):
        raise ValueError(f"{label} must contain at least one sample and one target")
    if np.any(np.isnan(array)):
        raise ValueError(f"{label} cannot contain NaN")
    return array


def _decode_json_float_array_payload(value: Any, label: str) -> Any:
    if isinstance(value, list):
        return [_decode_json_float_array_payload(item, label) for item in value]
    return _decode_json_float(value, label)


def _decode_json_qhat(value: Any, label: str) -> float | np.ndarray:
    if isinstance(value, list):
        decoded = np.asarray([_decode_json_float(item, label) for item in value], dtype=float)
        if decoded.ndim != 1:
            raise ValueError(f"{label} qhat must be scalar or one-dimensional")
        return decoded
    return _decode_json_float(value, label)


def _conformal_float_equal(left: Any, right: Any) -> bool:
    left_number = float(left)
    right_number = float(right)
    if math.isnan(left_number) or math.isnan(right_number):
        return False
    return left_number == right_number


def _conformal_qhat_equal(left: Any, right: Any) -> bool:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape:
        return False
    if np.any(np.isnan(left_array)) or np.any(np.isnan(right_array)):
        return False
    return bool(np.array_equal(left_array, right_array))


def _expected_qhat_for_prediction_rows(
    artifact: ConformalCalibrationArtifact,
    coverage: float,
    group_keys: Sequence[str],
) -> float | np.ndarray:
    if not artifact.spec.group_by:
        return artifact.calibrator.qhat_by_coverage[coverage]
    keys = tuple(group_keys)
    missing = sorted(set(keys) - set(artifact.group_calibrators))
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... (+{len(missing) - 5} more)"
        raise ValueError(f"CalibratedRunResult prediction group keys are absent from the conformal artifact: {preview}{suffix}")
    return np.asarray([artifact.group_calibrators[key].qhat_by_coverage[coverage] for key in keys], dtype=float)


__all__ = [
    "CalibratedPredictionBlock",
    "CalibratedRunResult",
    "ConformalCalibrationArtifact",
    "ConformalCalibrationCohortManifest",
    "ConformalCalibrationCohortRow",
    "ConformalCalibrationSpec",
    "ConformalIntervalBlock",
    "ConformalMetricSet",
    "ConformalMethod",
    "ConformalMultiTarget",
    "SplitConformalCalibrator",
    "SUPPORTED_CONFORMAL_KEYS",
    "SUPPORTED_CONFORMAL_METHODS",
    "SUPPORTED_CONFORMAL_MULTI_TARGET",
    "SUPPORTED_CONFORMAL_UNITS",
    "apply_conformal_artifact_to_predictions",
    "apply_split_conformal_calibrator",
    "calibrate_replayed_predictions",
    "conformal_finite_sample_quantile",
    "conformal_guarantee_status",
    "evaluate_conformal_prediction",
    "fit_conformal_calibration_artifact",
    "fit_split_absolute_residual_calibrator",
    "normalize_coverages",
    "normalize_conformal_calibration_cohort",
    "parse_conformal_calibration_spec",
]
