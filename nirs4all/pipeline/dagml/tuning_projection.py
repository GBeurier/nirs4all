"""Projection helpers for native DAG-ML tuning outputs.

This module projects the internal ``ObjectiveTuningRunResult`` evidence tape
into a public ``RunResult`` carrier.  The default projection carries tuning
evidence only.  Callers may additionally project a refit winner prediction, but
only by supplying the evaluation score and metric explicitly; this seam never
computes or fabricates selection metrics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from nirs4all.api.result import RunResult
from nirs4all.data.predictions import Predictions

from .tuning_adapters import ObjectiveTuningRunResult


def _as_1d_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError(f"{name} must contain one value per projected sample")
    return array


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


def _n_features(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape and len(shape) >= 2:
        try:
            return int(shape[1])
        except (TypeError, ValueError):
            return 0
    return 0


def _estimator_classname(estimator: Any) -> str:
    cls = type(estimator)
    return f"{cls.__module__}.{cls.__qualname__}"


def _winner_predictions(
    result: ObjectiveTuningRunResult,
    *,
    winner_x: Any,
    winner_y_true: Any | None,
    winner_score: float | None,
    winner_metric: str | None,
    winner_sample_ids: Sequence[Any] | None,
    winner_dataset_name: str,
    winner_model_name: str | None,
    winner_task_type: str,
    winner_metadata: Mapping[str, Any] | None,
) -> Predictions:
    if result.refit_estimator is None:
        raise ValueError("winner projection requires a refit_estimator on ObjectiveTuningRunResult")
    if not hasattr(result.refit_estimator, "predict"):
        raise TypeError("winner projection requires refit_estimator.predict")
    if winner_score is None or not winner_metric:
        raise ValueError("winner projection requires explicit winner_score and winner_metric")

    y_pred = _as_1d_array(result.refit_estimator.predict(winner_x), name="winner predictions")
    n_samples = len(y_pred)
    x_samples = _n_samples(winner_x)
    if x_samples and x_samples != n_samples:
        raise ValueError(f"winner_x contains {x_samples} samples but predict() returned {n_samples} predictions")

    y_true = None
    if winner_y_true is not None:
        y_true = _as_1d_array(winner_y_true, name="winner_y_true")
        if len(y_true) != n_samples:
            raise ValueError(f"winner_y_true contains {len(y_true)} samples but predict() returned {n_samples} predictions")

    metadata = dict(winner_metadata or {})
    if winner_sample_ids is not None:
        sample_ids = list(winner_sample_ids)
        if len(sample_ids) != n_samples:
            raise ValueError(f"winner_sample_ids contains {len(sample_ids)} samples but predict() returned {n_samples} predictions")
        metadata["physical_sample_id"] = sample_ids

    predictions = Predictions()
    estimator_name = winner_model_name or type(result.refit_estimator).__name__
    predictions.add_prediction(
        dataset_name=winner_dataset_name,
        model_name=estimator_name,
        model_classname=_estimator_classname(result.refit_estimator),
        fold_id="final",
        sample_indices=np.arange(n_samples),
        metadata=metadata,
        partition="test",
        y_true=y_true,
        y_pred=y_pred,
        test_score=float(winner_score),
        metric=winner_metric,
        task_type=winner_task_type,
        n_samples=n_samples,
        n_features=_n_features(winner_x),
        best_params=dict(result.tuning_result.best_params),
        scores={"test": {winner_metric: float(winner_score)}},
        refit_context="tuning_winner",
    )
    return predictions


def project_objective_tuning_to_run_result(
    result: ObjectiveTuningRunResult,
    *,
    per_dataset: Mapping[str, Any] | None = None,
    winner_x: Any | None = None,
    winner_y_true: Any | None = None,
    winner_score: float | None = None,
    winner_metric: str | None = None,
    winner_sample_ids: Sequence[Any] | None = None,
    winner_dataset_name: str = "tuning_winner",
    winner_model_name: str | None = None,
    winner_task_type: str = "regression",
    winner_metadata: Mapping[str, Any] | None = None,
) -> RunResult:
    """Return a ``RunResult`` carrying a native tuning result.

    The returned object exposes ``tuning_result``, ``tuning_id``,
    ``tuning_best_params`` and ``tuning_best_value``.

    By default, the carrier contains zero prediction rows.  When ``winner_x`` is
    supplied, the function calls the refit estimator's ``predict`` method and
    inserts a single ``fold_id="final"`` prediction row.  In that mode
    ``winner_score`` and ``winner_metric`` are required, because ranking evidence
    must come from an explicit external evaluation rather than this projection
    helper.
    """

    if not isinstance(result, ObjectiveTuningRunResult):
        raise TypeError("project_objective_tuning_to_run_result expects an ObjectiveTuningRunResult")
    predictions = Predictions()
    if winner_x is not None:
        predictions = _winner_predictions(
            result,
            winner_x=winner_x,
            winner_y_true=winner_y_true,
            winner_score=winner_score,
            winner_metric=winner_metric,
            winner_sample_ids=winner_sample_ids,
            winner_dataset_name=winner_dataset_name,
            winner_model_name=winner_model_name,
            winner_task_type=winner_task_type,
            winner_metadata=winner_metadata,
        )
    return RunResult(
        predictions=predictions,
        per_dataset=dict(per_dataset or {}),
        _tuning_result=result.tuning_result,
        _tuning_id=result.tuning_id,
    )


__all__ = ["project_objective_tuning_to_run_result"]
