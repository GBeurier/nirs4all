"""Shared native DAG-ML pipeline objective substrate.

P4 is the seam between a typed tuning contract and optimizer adapters. This
module does not run Optuna or n4m. It evaluates one explicit trial by cloning an
estimator, applying candidate parameter patches, fitting through the estimator's
own native compiler/client path, and extracting one score through an injected
callback. Optimizer adapters in P5 can then consume the same score tape.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone

from nirs4all.core.metrics import eval as evaluate_metric

from .conformal_contracts import evaluate_conformal_prediction, fit_split_absolute_residual_calibrator, normalize_coverages
from .tuning_contracts import DagMLTuningSpec, parse_tuning_spec


@dataclass(frozen=True)
class PipelineObjectiveResult:
    """Result for one evaluated candidate."""

    trial_index: int
    params: Mapping[str, Any]
    score: float
    metric: str
    direction: str
    tuning_fingerprint: str
    diagnostics: Mapping[str, Any]


class PipelineObjective:
    """Evaluate one fixed-topology trial through a supplied estimator seam."""

    def __init__(
        self,
        estimator: Any,
        tuning: DagMLTuningSpec | Mapping[str, Any],
        *,
        score_extractor: Callable[[Any], float],
        clone_estimator: bool = True,
    ) -> None:
        self.estimator = estimator
        self.tuning = tuning if isinstance(tuning, DagMLTuningSpec) else parse_tuning_spec(tuning)
        self.score_extractor = score_extractor
        self.clone_estimator = clone_estimator

    def evaluate(
        self,
        params: Mapping[str, Any],
        X: Any,
        y: Any,
        *,
        trial_index: int = 0,
        sample_ids: Any = None,
        groups: Any = None,
        metadata: Any = None,
    ) -> PipelineObjectiveResult:
        """Fit and score a single candidate without driving an optimizer."""

        if not isinstance(params, Mapping):
            raise TypeError("trial params must be a mapping")
        candidate = self._candidate_estimator()
        applied_patches = self.tuning.parameter_patches(params)
        applied_params = {patch.path: patch.value for patch in applied_patches}
        for patch in applied_patches:
            apply_trial_parameter_patch(candidate, patch.path, patch.value)

        fitted = _fit_with_optional_identity(
            candidate,
            X,
            y,
            sample_ids=sample_ids,
            groups=groups,
            metadata=metadata,
        )
        score = float(self.score_extractor(fitted))
        if not math.isfinite(score):
            raise ValueError("pipeline objective score must be finite")
        score_family = "conformal" if self.tuning.metric.startswith("conformal_") else "objective"
        diagnostics = {
            "engine": self.tuning.engine,
            "metric": self.tuning.metric,
            "direction": self.tuning.direction,
            "score_extractor": "conformal_temporary_calibration" if score_family == "conformal" else "objective",
            "score_family": score_family,
            "search_space_fingerprint": self.tuning.ordered_search_space.fingerprint,
            "final_calibration_scope": "unmodified_by_score_data" if score_family == "conformal" else "not_applicable",
            "tuning_fingerprint": self.tuning.fingerprint,
        }
        return PipelineObjectiveResult(
            trial_index=trial_index,
            params=applied_params,
            score=score,
            metric=self.tuning.metric,
            direction=self.tuning.direction,
            tuning_fingerprint=self.tuning.fingerprint,
            diagnostics=diagnostics,
        )

    def refit_best(
        self,
        tuning_result: Any,
        X: Any,
        y: Any,
        *,
        sample_ids: Any = None,
        groups: Any = None,
        metadata: Any = None,
    ) -> Any:
        """Fit exactly one terminal estimator from a completed tuning result."""

        best_params = getattr(tuning_result, "best_params", None)
        if not isinstance(best_params, Mapping):
            raise TypeError("tuning_result.best_params must be a mapping")
        candidate = self._candidate_estimator()
        for patch in self.tuning.parameter_patches(best_params, context="best params"):
            apply_trial_parameter_patch(candidate, patch.path, patch.value)
        return _fit_with_optional_identity(
            candidate,
            X,
            y,
            sample_ids=sample_ids,
            groups=groups,
            metadata=metadata,
        )

    def _candidate_estimator(self) -> Any:
        if not self.clone_estimator:
            return self.estimator
        try:
            return clone(self.estimator)
        except Exception as exc:  # pragma: no cover - defensive message path
            raise TypeError("PipelineObjective requires a sklearn-cloneable estimator or clone_estimator=False") from exc


def apply_trial_parameter_patch(estimator: Any, path: str, value: Any) -> None:
    """Apply one fail-closed candidate parameter patch to an estimator."""

    if not isinstance(path, str) or not path.strip():
        raise ValueError("trial parameter path must be a non-empty string")
    normalized = path.strip()
    if hasattr(estimator, "get_params") and hasattr(estimator, "set_params"):
        known_params = estimator.get_params(deep=True)
        if normalized in known_params:
            estimator.set_params(**{normalized: value})
            return
        sklearn_alias = normalized.replace(".", "__")
        if sklearn_alias != normalized and sklearn_alias in known_params:
            estimator.set_params(**{sklearn_alias: value})
            return
    _apply_dotted_attribute_patch(estimator, normalized, value)


def _apply_dotted_attribute_patch(estimator: Any, path: str, value: Any) -> None:
    target = estimator
    parts = path.split(".")
    if any(not part for part in parts):
        raise ValueError(f"invalid trial parameter path {path!r}")
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise ValueError(f"trial parameter path {path!r} is not supported by estimator {type(estimator).__name__}")
        target = getattr(target, part)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise ValueError(f"trial parameter path {path!r} is not supported by estimator {type(estimator).__name__}")
    setattr(target, leaf, value)


def make_prediction_score_extractor(
    metric: str,
    X_score: Any,
    y_score: Any,
    *,
    sample_ids: Any = None,
    groups: Any = None,
    metadata: Any = None,
) -> Callable[[Any], float]:
    """Build a ``PipelineObjective`` score extractor from explicit score data.

    The returned callable scores a fitted estimator by calling
    ``estimator.predict(X_score)`` and evaluating the requested metric against
    ``y_score``.  The score cohort is supplied explicitly; this helper does not
    split data, reuse optimizer objective values as test metrics, or infer a
    validation set. Optional ``sample_ids``/``groups``/``metadata`` are validated
    for row alignment and forwarded to ``predict()`` only when the estimator
    signature accepts them.
    """

    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("score metric must be a non-empty string")
    normalized_metric = metric.strip()
    y_score_array = _non_scalar_array(y_score, name="y_score")
    x_samples = _n_samples(X_score)
    if x_samples and x_samples != len(y_score_array):
        raise ValueError(f"X_score contains {x_samples} samples but y_score contains {len(y_score_array)} values")
    _validate_optional_row_aligned(sample_ids, expected=len(y_score_array), name="score sample_ids")
    _validate_optional_row_aligned(groups, expected=len(y_score_array), name="score groups")
    _validate_optional_row_aligned(metadata, expected=len(y_score_array), name="score metadata")

    def _extractor(fitted_estimator: Any) -> float:
        if not hasattr(fitted_estimator, "predict"):
            raise TypeError("prediction score extractor requires fitted_estimator.predict")
        y_pred = _non_scalar_array(
            _predict_with_optional_identity(
                fitted_estimator,
                X_score,
                sample_ids=sample_ids,
                groups=groups,
                metadata=metadata,
            ),
            name="predictions",
        )
        score = evaluate_metric(y_score_array, y_pred, normalized_metric)
        if isinstance(score, dict):
            raise TypeError("prediction score extractor expects a single metric, not a metric list")
        return float(score)

    return _extractor


_CONFORMAL_SCORE_FIELDS = {
    "conformal_abs_coverage_gap": "abs_coverage_gap",
    "conformal_interval_score": "mean_interval_score",
    "conformal_mean_interval_score": "mean_interval_score",
    "conformal_mean_width": "mean_width",
    "conformal_median_width": "median_width",
    "conformal_missed_rate": "missed_rate",
    "conformal_observed_coverage": "observed_coverage",
}


def make_conformal_prediction_score_extractor(
    metric: str,
    X_score: Any,
    y_score: Any,
    calibration_data: Mapping[str, Any],
    *,
    coverage: float = 0.9,
    sample_ids: Any = None,
    groups: Any = None,
    metadata: Any = None,
) -> Callable[[Any], float]:
    """Build a conformal-aware development score extractor.

    This helper is intentionally evaluation-only. For each fitted candidate it:

    1. predicts an explicit development calibration cohort;
    2. fits a temporary split conformal calibrator from that cohort;
    3. predicts the explicit scoring cohort;
    4. evaluates empirical conformal diagnostics on the scoring cohort.

    It does not persist the temporary calibrator and does not touch the final
    calibrator produced by ``run(..., calibration=...)``.
    """

    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("conformal score metric must be a non-empty string")
    normalized_metric = metric.strip().lower()
    try:
        field = _CONFORMAL_SCORE_FIELDS[normalized_metric]
    except KeyError as exc:
        raise ValueError(f"unsupported conformal score metric {metric!r}; supported metrics are {sorted(_CONFORMAL_SCORE_FIELDS)}") from exc
    coverages = normalize_coverages(coverage)
    if len(coverages) != 1:
        raise ValueError("conformal-aware tuning score currently requires exactly one coverage")
    score_coverage = coverages[0]
    if not isinstance(calibration_data, Mapping):
        raise TypeError("conformal calibration score data must be a mapping")
    X_cal = _first_present(calibration_data, "X", "X_calibration", "features")
    y_cal = _first_present(calibration_data, "y", "y_true", "y_calibration", "target", "targets")
    if X_cal is None or y_cal is None:
        raise ValueError("conformal calibration score data requires X/y or X_calibration/y_true")

    y_score_array = _non_scalar_array(y_score, name="y_score")
    x_score_samples = _n_samples(X_score)
    if x_score_samples and x_score_samples != len(y_score_array):
        raise ValueError(f"X_score contains {x_score_samples} samples but y_score contains {len(y_score_array)} values")
    _validate_optional_row_aligned(sample_ids, expected=len(y_score_array), name="score sample_ids")
    _validate_optional_row_aligned(groups, expected=len(y_score_array), name="score groups")
    _validate_optional_row_aligned(metadata, expected=len(y_score_array), name="score metadata")

    y_cal_array = _non_scalar_array(y_cal, name="calibration y_true")
    x_cal_samples = _n_samples(X_cal)
    if x_cal_samples and x_cal_samples != len(y_cal_array):
        raise ValueError(f"calibration X contains {x_cal_samples} samples but calibration y_true contains {len(y_cal_array)} values")
    calibration_sample_ids = _first_present(calibration_data, "sample_ids", "calibration_sample_ids", "physical_sample_ids")
    calibration_groups = _first_present(calibration_data, "groups", "calibration_groups")
    calibration_metadata = _first_present(calibration_data, "metadata", "calibration_metadata")
    _validate_optional_row_aligned(calibration_sample_ids, expected=len(y_cal_array), name="calibration sample_ids")
    _validate_optional_row_aligned(calibration_groups, expected=len(y_cal_array), name="calibration groups")
    _validate_optional_row_aligned(calibration_metadata, expected=len(y_cal_array), name="calibration metadata")

    def _extractor(fitted_estimator: Any) -> float:
        if not hasattr(fitted_estimator, "predict"):
            raise TypeError("conformal score extractor requires fitted_estimator.predict")
        y_cal_pred = _non_scalar_array(
            _predict_with_optional_identity(
                fitted_estimator,
                X_cal,
                sample_ids=calibration_sample_ids,
                groups=calibration_groups,
                metadata=calibration_metadata,
            ),
            name="calibration predictions",
        )
        calibrator = fit_split_absolute_residual_calibrator(
            y_cal_array,
            y_cal_pred,
            coverage=score_coverage,
        )
        y_score_pred = _non_scalar_array(
            _predict_with_optional_identity(
                fitted_estimator,
                X_score,
                sample_ids=sample_ids,
                groups=groups,
                metadata=metadata,
            ),
            name="score predictions",
        )
        metrics = evaluate_conformal_prediction(
            y_true=y_score_array,
            prediction=calibrator.apply_block(y_score_pred),
        )
        metric_set = metrics[score_coverage]
        if field == "abs_coverage_gap":
            return abs(float(metric_set.coverage_gap))
        if field == "missed_rate":
            return 1.0 - float(metric_set.observed_coverage)
        return float(getattr(metric_set, field))

    return _extractor


def _fit_with_optional_identity(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    sample_ids: Any = None,
    groups: Any = None,
    metadata: Any = None,
) -> Any:
    fit_kwargs = {}
    if sample_ids is not None:
        fit_kwargs["sample_ids"] = sample_ids
    if groups is not None:
        fit_kwargs["groups"] = groups
    if metadata is not None:
        fit_kwargs["metadata"] = metadata
    if fit_kwargs:
        return estimator.fit(X, y, **fit_kwargs)
    return estimator.fit(X, y)


def _non_scalar_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError(f"{name} must contain one value per score sample")
    return array


def _validate_optional_row_aligned(value: Any, *, expected: int, name: str) -> None:
    if value is None:
        return
    length = _row_count(value, name=name)
    if length and length != expected:
        raise ValueError(f"{name} contains {length} rows but y_score contains {expected} values")


def _row_count(value: Any, *, name: str) -> int:
    if isinstance(value, Mapping):
        lengths = [_n_samples(column) for column in value.values()]
        non_zero_lengths = [length for length in lengths if length]
        if not non_zero_lengths:
            return 0
        if len(set(non_zero_lengths)) != 1:
            raise ValueError(f"{name} contains inconsistent row lengths")
        return non_zero_lengths[0]
    return _n_samples(value)


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _predict_with_optional_identity(
    estimator: Any,
    X: Any,
    *,
    sample_ids: Any = None,
    groups: Any = None,
    metadata: Any = None,
) -> Any:
    predict = estimator.predict
    accepted = _accepted_kwargs(predict)
    kwargs: dict[str, Any] = {}
    if sample_ids is not None and ("sample_ids" in accepted or "**" in accepted):
        kwargs["sample_ids"] = sample_ids
    if groups is not None and ("groups" in accepted or "**" in accepted):
        kwargs["groups"] = groups
    if metadata is not None and ("metadata" in accepted or "**" in accepted):
        kwargs["metadata"] = metadata
    if kwargs:
        return predict(X, **kwargs)
    return predict(X)


def _accepted_kwargs(callable_obj: Any) -> set[str]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return set()
    accepted: set[str] = set()
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            accepted.add("**")
        elif parameter.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            accepted.add(parameter.name)
    return accepted


def _n_samples(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape:
        try:
            return int(shape[0])
        except (TypeError, ValueError, IndexError):
            pass
    try:
        return len(value)
    except TypeError:
        return 0


__all__ = [
    "PipelineObjective",
    "PipelineObjectiveResult",
    "apply_trial_parameter_patch",
    "make_conformal_prediction_score_extractor",
    "make_prediction_score_extractor",
]
