"""Unit tests for the N0 reduction socle.

Covers the ``ReductionPlan`` parity with ``Predictions.aggregate`` (the adapter
must not change legacy output), prediction levels / unit ids and the typed
``fold_id`` helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

from nirs4all.data.predictions import Predictions
from nirs4all.data.reduction import (
    FitScope,
    LeakedReducerStateError,
    PredictionLevel,
    PredictionScope,
    PredictionUnitId,
    ReducerState,
    ReductionAxis,
    ReductionError,
    ReductionMethod,
    ReductionPlan,
    ReductionRole,
    TaskCompatibility,
    is_real_cv_fold,
    prediction_scope_from_legacy,
)

# ---------------------------------------------------------------------------
# ReductionPlan parity with Predictions.aggregate
# ---------------------------------------------------------------------------


def _assert_aggregate_equal(reduced, legacy):
    assert set(reduced) == set(legacy)
    for key in legacy:
        np.testing.assert_array_equal(np.asarray(reduced[key]), np.asarray(legacy[key]))


def test_reduction_plan_parity_mean_regression():
    rng = np.random.default_rng(0)
    y_pred = rng.normal(size=12).astype(float)
    y_true = rng.normal(size=12).astype(float)
    group_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    plan = ReductionPlan.from_legacy_aggregate("mean")
    reduced = plan.reduce(y_pred, group_ids, y_true=y_true)
    legacy = Predictions.aggregate(y_pred, group_ids, y_true=y_true, method="mean")
    _assert_aggregate_equal(reduced, legacy)


def test_reduction_plan_parity_median():
    y_pred = np.array([1.0, 2.0, 9.0, 4.0, 5.0, 6.0])
    group_ids = np.array([0, 0, 0, 1, 1, 1])
    plan = ReductionPlan.from_legacy_aggregate("median")
    reduced = plan.reduce(y_pred, group_ids)
    legacy = Predictions.aggregate(y_pred, group_ids, method="median")
    _assert_aggregate_equal(reduced, legacy)


def test_reduction_plan_parity_classification_proba():
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    y_proba = np.array(
        [[0.6, 0.4], [0.3, 0.7], [0.2, 0.8], [0.7, 0.3], [0.55, 0.45], [0.1, 0.9]]
    )
    group_ids = np.array([0, 0, 0, 1, 1, 1])
    plan = ReductionPlan.from_legacy_aggregate("vote")
    reduced = plan.reduce(y_pred.astype(float), group_ids, y_proba=y_proba)
    legacy = Predictions.aggregate(y_pred.astype(float), group_ids, y_proba=y_proba, method="vote")
    _assert_aggregate_equal(reduced, legacy)


def test_reduction_plan_parity_exclude_outliers():
    y_pred = np.array([10.0, 10.1, 9.9, 50.0, 1.0, 1.1, 0.9, 1.05])
    group_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    plan = ReductionPlan.from_legacy_aggregate("mean", exclude_outliers=True)
    reduced = plan.reduce(y_pred, group_ids)
    legacy = Predictions.aggregate(y_pred, group_ids, method="mean", exclude_outliers=True)
    _assert_aggregate_equal(reduced, legacy)


def test_reduction_plan_from_by_repetition():
    plan = ReductionPlan.from_by_repetition("median", exclude_outliers=True)
    assert plan.method is ReductionMethod.MEDIAN
    assert plan.exclude_outliers is True
    assert plan.axis is ReductionAxis.UNIT


def test_non_unit_axis_not_executable():
    plan = ReductionPlan.from_test_aggregation("mean")
    assert plan.axis is ReductionAxis.FOLD
    with pytest.raises(NotImplementedError):
        plan.reduce(np.array([1.0]), np.array([0]))


def test_from_test_aggregation_methods():
    assert ReductionPlan.from_test_aggregation("mean").role is ReductionRole.FOLD_ENSEMBLE
    assert ReductionPlan.from_test_aggregation("weighted").axis is ReductionAxis.FOLD
    weighted = ReductionPlan.from_test_aggregation("weighted")
    assert weighted.method is ReductionMethod.WEIGHTED_MEAN
    assert weighted.weight_source == "val_score"
    best = ReductionPlan.from_test_aggregation("best")
    assert best.method is ReductionMethod.CUSTOM
    assert best.custom_name == "best_fold"


def test_reduction_plan_to_dict():
    plan = ReductionPlan.from_legacy_aggregate("mean", exclude_outliers=True)
    d = plan.to_dict()
    assert d["method"] == "mean"
    assert d["axis"] == "unit"
    assert d["output_level"] == "sample"
    assert d["reduction_id"].startswith("reduce-")
    assert d["fingerprint"] == plan.fingerprint()


def test_reduction_plan_from_dict_round_trip_is_stable():
    plan = ReductionPlan(
        role=ReductionRole.PERSIST,
        axis=ReductionAxis.UNIT,
        method=ReductionMethod.WEIGHTED_MEAN,
        weight_source="sample_influence_weight",
        task_compatibility=TaskCompatibility.REGRESSION,
    )
    restored = ReductionPlan.from_dict(plan.to_dict())
    assert restored.to_dict() == plan.to_dict()


def test_weighted_mean_reduction_requires_and_uses_weights():
    plan = ReductionPlan(method=ReductionMethod.WEIGHTED_MEAN, weight_source="sample_influence_weight")
    y_pred = np.array([1.0, 3.0, 10.0, 20.0])
    group_ids = np.array([0, 0, 1, 1])
    weights = np.array([1.0, 3.0, 1.0, 1.0])

    with pytest.raises(ReductionError, match="requires per-row weights"):
        plan.reduce(y_pred, group_ids)

    reduced = plan.reduce(y_pred, group_ids, weights=weights)
    np.testing.assert_allclose(reduced["y_pred"], np.array([2.5, 15.0]))
    np.testing.assert_array_equal(reduced["group_sizes"], np.array([2, 2]))


def test_robust_reduction_is_outlier_excluded_mean():
    y_pred = np.array([10.0, 10.1, 9.9, 50.0, 1.0, 1.1, 0.9, 1.05])
    group_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    plan = ReductionPlan(method=ReductionMethod.ROBUST)

    reduced = plan.reduce(y_pred, group_ids)
    legacy = Predictions.aggregate(y_pred, group_ids, method="mean", exclude_outliers=True)
    _assert_aggregate_equal(reduced, legacy)


def test_as_repetition_aggregation_supports_only_safe_legacy_methods():
    assert ReductionPlan(method=ReductionMethod.ROBUST).as_repetition_aggregation() == ("mean", True)
    with pytest.raises(ReductionError, match="cannot drive by_repetition"):
        ReductionPlan(method=ReductionMethod.WEIGHTED_MEAN).as_repetition_aggregation()


def test_task_compatibility_is_enforced_when_requested():
    plan = ReductionPlan(method=ReductionMethod.MEDIAN)
    assert plan.is_task_compatible("regression") is True
    assert plan.is_task_compatible("classification") is False
    with pytest.raises(ReductionError, match="task_compatibility"):
        plan.validate_task("classification")


def test_fitable_reduction_requires_train_fitted_state_and_rejects_leakage():
    plan = ReductionPlan(method=ReductionMethod.MEAN, fit_scope=FitScope.FOLD_TRAIN)
    y_pred = np.array([1.0, 2.0])
    group_ids = np.array([0, 0])

    with pytest.raises(ReductionError, match="no fitted ReducerState"):
        plan.reduce(y_pred, group_ids)

    leaked = ReducerState(fit_scope=FitScope.FOLD_TRAIN, fit_partition="val", fold_id="0")
    with pytest.raises(LeakedReducerStateError):
        plan.reduce(y_pred, group_ids, state=leaked)

    state = ReducerState(fit_scope=FitScope.FOLD_TRAIN, fit_partition="train", fold_id="0")
    reduced = plan.reduce(y_pred, group_ids, state=state)
    legacy = Predictions.aggregate(y_pred, group_ids, method="mean")
    _assert_aggregate_equal(reduced, legacy)


# ---------------------------------------------------------------------------
# Prediction levels and unit ids
# ---------------------------------------------------------------------------


def test_prediction_unit_id_keys():
    assert PredictionUnitId(PredictionLevel.SAMPLE, physical_sample_id="S1").as_key() == "sample:S1"
    assert (
        PredictionUnitId(PredictionLevel.OBSERVATION, physical_sample_id="S1", source_id="A", rep_id=2).as_key()
        == "source_observation:A:S1:2"
    )
    assert (
        PredictionUnitId(PredictionLevel.COMBO, derived_unit_id="S1::A1xB2").as_key()
        == "derived_combo:S1::A1xB2"
    )


def test_prediction_unit_id_hashable():
    a = PredictionUnitId(PredictionLevel.SAMPLE, physical_sample_id="S1")
    b = PredictionUnitId(PredictionLevel.SAMPLE, physical_sample_id="S1")
    assert a == b
    assert len({a, b}) == 1


# ---------------------------------------------------------------------------
# Typed fold_id helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fold_id", ["0", "1", "12", 0, 3])
def test_is_real_cv_fold_true(fold_id):
    assert is_real_cv_fold(fold_id) is True


@pytest.mark.parametrize("fold_id", ["final", "final_agg", "avg", "w_avg", "ensemble", "all", "0_agg", "", None])
def test_is_real_cv_fold_false(fold_id):
    assert is_real_cv_fold(fold_id) is False


def test_prediction_scope_refit():
    assert prediction_scope_from_legacy("final") is PredictionScope.REFIT
    assert prediction_scope_from_legacy("final_agg") is PredictionScope.REFIT
    assert prediction_scope_from_legacy("0", is_refit=True) is PredictionScope.REFIT


def test_prediction_scope_test_partition():
    assert prediction_scope_from_legacy("0", partition="test") is PredictionScope.TEST


def test_prediction_scope_oof_default():
    assert prediction_scope_from_legacy("0") is PredictionScope.OOF
    assert prediction_scope_from_legacy("avg") is PredictionScope.OOF
    assert prediction_scope_from_legacy("3", partition="val") is PredictionScope.OOF
