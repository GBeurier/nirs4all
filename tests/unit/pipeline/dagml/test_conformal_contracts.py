"""Unit locks for internal conformal calibration contracts."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from nirs4all.pipeline.dagml.conformal_contracts import (
    CalibratedPredictionBlock,
    CalibratedRunResult,
    ConformalCalibrationArtifact,
    ConformalCalibrationCohortManifest,
    ConformalCalibrationCohortRow,
    ConformalCalibrationSpec,
    ConformalIntervalBlock,
    ConformalMetricSet,
    SplitConformalCalibrator,
    apply_split_conformal_calibrator,
    calibrate_replayed_predictions,
    conformal_finite_sample_quantile,
    conformal_guarantee_status,
    evaluate_conformal_prediction,
    fit_conformal_calibration_artifact,
    fit_joint_max_absolute_residual_calibrator,
    fit_split_absolute_residual_calibrator,
    normalize_conformal_calibration_cohort,
    normalize_coverages,
    parse_conformal_calibration_spec,
)


class _StringifiesAs:
    def __init__(self, value: str) -> None:
        self._value = value

    def __str__(self) -> str:
        return self._value


def _valid_metric_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "coverage": 0.8,
        "observed_coverage": 0.5,
        "coverage_gap": -0.3,
        "mean_width": 0.8,
        "median_width": 0.8,
        "mean_interval_score": 3.8,
        "n_samples": 2,
        "n_covered": 1,
        "n_missed_below": 0,
        "n_missed_above": 1,
    }
    payload.update(overrides)
    return payload


def test_conformal_finite_sample_quantile_uses_ceil_n_plus_one_rule() -> None:
    scores = np.asarray([0.4, 0.1, 0.2, 0.3])

    assert conformal_finite_sample_quantile(scores, 0.80) == pytest.approx(0.4)
    assert math.isinf(conformal_finite_sample_quantile(scores, 0.95))


def test_parse_conformal_calibration_spec_normalizes_contract_and_fingerprint() -> None:
    spec = parse_conformal_calibration_spec(
        {
            "method": " Split_Absolute_Residual ",
            "coverage": [0.9, 0.8, 0.9],
            "unit": " Physical_Sample ",
            "group_by": [" instrument ", "batch"],
            "multi_target": " MARGINAL ",
        }
    )

    assert isinstance(spec, ConformalCalibrationSpec)
    assert spec.method == "split_absolute_residual"
    assert spec.coverage == (0.8, 0.9)
    assert spec.unit == "physical_sample"
    assert spec.group_by == ("batch", "instrument")
    assert spec.multi_target == "marginal"
    assert spec.to_dict() == {
        "coverage": [0.8, 0.9],
        "group_by": ["batch", "instrument"],
        "method": "split_absolute_residual",
        "multi_target": "marginal",
        "unit": "physical_sample",
    }
    assert len(spec.fingerprint) == 64
    assert (
        spec.fingerprint
        == parse_conformal_calibration_spec(
            {
                "coverage": [0.8, 0.9],
                "group_by": ["batch", "instrument"],
                "method": "split_absolute_residual",
            }
        ).fingerprint
    )


def test_conformal_calibration_spec_direct_construction_normalizes_contract() -> None:
    spec = ConformalCalibrationSpec(
        coverage=(0.9, 0.8, 0.9),
        method=" Split_Absolute_Residual ",
        unit=" Physical_Sample ",
        group_by=(" instrument ", "batch"),
        multi_target=" MARGINAL ",
    )

    assert spec.coverage == (0.8, 0.9)
    assert spec.method == "split_absolute_residual"
    assert spec.unit == "physical_sample"
    assert spec.group_by == ("batch", "instrument")
    assert spec.multi_target == "marginal"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"coverage": ("0.8",)}, "coverage values must be numeric floats, not strings"),
        ({"coverage": (True,)}, "coverage values must be numeric floats, not booleans"),
        ({"coverage": (0.8,), "method": "weighted"}, "ConformalCalibrationSpec.method must be one of"),
        ({"coverage": (0.8,), "unit": "observation"}, "ConformalCalibrationSpec.unit must be one of"),
        ({"coverage": (0.8,), "group_by": ("group", " group ")}, "duplicate keys"),
        ({"coverage": (0.8,), "multi_target": "per_target"}, "ConformalCalibrationSpec.multi_target must be one of"),
    ],
)
def test_conformal_calibration_spec_direct_construction_rejects_invalid_contract(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        ConformalCalibrationSpec(**kwargs)


@pytest.mark.parametrize(
    ("payload", "error_type", "match"),
    [
        ({}, ValueError, "coverage is required"),
        ({"coverage": 0.8, "unknown": True}, ValueError, "does not support keys"),
        ({"coverage": 0.8, "method": "weighted"}, ValueError, "method must be one of"),
        ({"coverage": 0.8, "unit": "observation"}, ValueError, "unit must be one of"),
        ({"coverage": 0.8, "group_by": []}, ValueError, "at least one group key"),
        ({"coverage": 0.8, "group_by": ["batch", " batch "]}, ValueError, "duplicate keys"),
    ],
)
def test_parse_conformal_calibration_spec_rejects_ambiguous_or_unsupported_payloads(
    payload: dict[str, object],
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        parse_conformal_calibration_spec(payload)


def test_split_absolute_residual_calibrator_applies_multi_coverage_intervals() -> None:
    calibrator = fit_split_absolute_residual_calibrator(
        y_true=np.asarray([1.0, 2.0, 3.0, 4.0]),
        y_pred=np.asarray([0.9, 2.2, 2.6, 4.3]),
        coverage=[0.5, 0.8],
    )

    intervals = calibrator.apply(np.asarray([10.0, 20.0]))

    assert calibrator.scores.tolist() == pytest.approx([0.1, 0.2, 0.4, 0.3])
    assert calibrator.coverages == (0.5, 0.8)
    assert set(intervals) == {0.5, 0.8}
    assert intervals[0.5].qhat == pytest.approx(0.3)
    assert intervals[0.8].qhat == pytest.approx(0.4)
    np.testing.assert_allclose(intervals[0.8].lower, [9.6, 19.6])
    np.testing.assert_allclose(intervals[0.8].upper, [10.4, 20.4])


def test_split_conformal_calibrator_direct_construction_normalizes_contract() -> None:
    calibrator = SplitConformalCalibrator(
        scores=np.asarray([0.4, 0.1, 0.2, 0.3]),
        coverages=(0.8, 0.5, 0.8),
        qhat_by_coverage={0.5: 0.3, 0.8: 0.4},
        method=" Split_Absolute_Residual ",
        unit=" Physical_Sample ",
    )

    assert calibrator.coverages == (0.5, 0.8)
    assert calibrator.method == "split_absolute_residual"
    assert calibrator.unit == "physical_sample"
    np.testing.assert_allclose(calibrator.scores, [0.4, 0.1, 0.2, 0.3])
    assert calibrator.qhat_by_coverage == {0.5: pytest.approx(0.3), 0.8: pytest.approx(0.4)}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"scores": np.asarray([-1.0]), "coverages": (0.8,), "qhat_by_coverage": {0.8: 0.1}},
            "scores must contain non-negative conformal residuals",
        ),
        (
            {"scores": np.asarray([True, True]), "coverages": (0.8,), "qhat_by_coverage": {0.8: True}},
            "scores must contain numeric values, not booleans",
        ),
        (
            {"scores": np.asarray([0.1]), "coverages": (0.8,), "qhat_by_coverage": {}},
            "qhat_by_coverage coverages must match coverages",
        ),
        (
            {"scores": np.asarray([0.1]), "coverages": (0.8,), "qhat_by_coverage": {0.8: -0.1}},
            "qhat_by_coverage must contain non-negative qhat values",
        ),
        (
            {"scores": np.asarray([0.1]), "coverages": (0.8,), "qhat_by_coverage": {0.8: 0.2}},
            "qhat_by_coverage must match retained conformal scores",
        ),
        (
            {"scores": np.asarray([0.1]), "coverages": (0.8,), "qhat_by_coverage": {0.8: 0.1}, "method": "bad"},
            "SplitConformalCalibrator.method must be one of",
        ),
        (
            {"scores": np.asarray([0.1]), "coverages": (0.8,), "qhat_by_coverage": {0.8: 0.1}, "unit": "observation"},
            "SplitConformalCalibrator.unit must be one of",
        ),
    ],
)
def test_split_conformal_calibrator_direct_construction_rejects_invalid_contract(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        SplitConformalCalibrator(**kwargs)


def test_split_absolute_residual_calibrator_can_return_calibrated_prediction_block() -> None:
    calibrator = fit_split_absolute_residual_calibrator(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        coverage=[0.8, 0.5],
    )

    block = apply_split_conformal_calibrator(calibrator, [10.0, 20.0])

    np.testing.assert_allclose(block.y_pred, [10.0, 20.0])
    assert block.coverages == (0.5, 0.8)
    assert block.method == "split_absolute_residual"
    assert block.unit == "physical_sample"
    assert block.interval(0.8).qhat == pytest.approx(0.4)
    np.testing.assert_allclose(block.interval(0.8).lower, [9.6, 19.6])
    np.testing.assert_allclose(block.interval(0.8).upper, [10.4, 20.4])
    assert calibrator.apply_block([10.0, 20.0]).coverages == (0.5, 0.8)


def test_joint_max_calibrator_applies_simultaneous_multi_target_region() -> None:
    calibrator = fit_joint_max_absolute_residual_calibrator(
        y_true=[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
        y_pred=[[0.9, 9.8], [2.3, 20.1], [2.4, 30.2], [4.1, 39.6]],
        coverage=0.8,
    )

    block = calibrator.apply_block([[10.0, 100.0], [20.0, 200.0]])

    np.testing.assert_allclose(calibrator.scores, [0.2, 0.3, 0.6, 0.4])
    assert calibrator.qhat_by_coverage[0.8] == pytest.approx(0.6)
    assert block.y_pred.shape == (2, 2)
    np.testing.assert_allclose(block.interval(0.8).lower, [[9.4, 99.4], [19.4, 199.4]])
    np.testing.assert_allclose(block.interval(0.8).upper, [[10.6, 100.6], [20.6, 200.6]])


def test_evaluate_conformal_prediction_computes_observed_coverage_width_and_interval_score() -> None:
    calibrator = fit_split_absolute_residual_calibrator(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        coverage=[0.5, 0.8],
    )
    block = calibrator.apply_block([10.0, 20.0])

    metrics = evaluate_conformal_prediction(y_true=[10.1, 21.0], prediction=block)

    assert set(metrics) == {0.5, 0.8}
    assert isinstance(metrics[0.8], ConformalMetricSet)
    assert metrics[0.8].n_samples == 2
    assert metrics[0.8].n_covered == 1
    assert metrics[0.8].n_missed_below == 0
    assert metrics[0.8].n_missed_above == 1
    assert metrics[0.8].observed_coverage == pytest.approx(0.5)
    assert metrics[0.8].coverage_gap == pytest.approx(-0.3)
    assert metrics[0.8].mean_width == pytest.approx(0.8)
    assert metrics[0.8].median_width == pytest.approx(0.8)
    assert metrics[0.8].mean_interval_score == pytest.approx(3.8)
    assert metrics[0.8].to_dict()["fingerprint"] == metrics[0.8].fingerprint
    assert metrics[0.5].mean_interval_score == pytest.approx(2.0)


def test_evaluate_conformal_prediction_handles_unbounded_intervals_without_nan_scores() -> None:
    calibrator = fit_split_absolute_residual_calibrator(
        y_true=[1.0, 2.0],
        y_pred=[1.1, 1.9],
        coverage=0.95,
    )

    metrics = evaluate_conformal_prediction(y_true=[10.0], prediction=calibrator.apply_block([10.0]))

    assert metrics[0.95].observed_coverage == pytest.approx(1.0)
    assert math.isinf(metrics[0.95].mean_width)
    assert math.isinf(metrics[0.95].mean_interval_score)
    assert metrics[0.95].to_dict()["mean_width"] == "Infinity"


@pytest.mark.parametrize(
    ("make_block", "match"),
    [
        (
            lambda: ConformalIntervalBlock(
                coverage=0.8,
                qhat=math.nan,
                lower=np.asarray([0.9]),
                upper=np.asarray([1.1]),
            ),
            "qhat cannot contain NaN",
        ),
        (
            lambda: ConformalIntervalBlock(
                coverage=0.8,
                qhat=-0.1,
                lower=np.asarray([0.9]),
                upper=np.asarray([1.1]),
            ),
            "qhat must be non-negative",
        ),
        (
            lambda: ConformalIntervalBlock(
                coverage=0.8,
                qhat=0.1,
                lower=np.asarray([1.2]),
                upper=np.asarray([1.1]),
            ),
            "lower bounds must be <= upper bounds",
        ),
        (
            lambda: ConformalIntervalBlock(
                coverage=0.8,
                qhat=np.asarray([0.1, 0.2]),
                lower=np.asarray([0.9]),
                upper=np.asarray([1.1]),
            ),
            "row-aligned qhat length must match y_pred",
        ),
    ],
)
def test_conformal_interval_block_rejects_invalid_direct_construction(
    make_block: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        make_block()


@pytest.mark.parametrize(
    ("make_block", "match"),
    [
        (
            lambda: CalibratedPredictionBlock(
                y_pred=np.asarray([1.0]),
                intervals={
                    0.8: ConformalIntervalBlock(
                        coverage=0.9,
                        qhat=0.1,
                        lower=np.asarray([0.9]),
                        upper=np.asarray([1.1]),
                    )
                },
            ),
            "interval coverage must match its mapping key",
        ),
        (
            lambda: CalibratedPredictionBlock(
                y_pred=np.asarray([1.0]),
                intervals={
                    0.8: ConformalIntervalBlock(
                        coverage=0.8,
                        qhat=0.1,
                        lower=np.asarray([0.9, 1.9]),
                        upper=np.asarray([1.1, 2.1]),
                    )
                },
            ),
            "interval arrays must match y_pred shape",
        ),
        (
            lambda: CalibratedPredictionBlock(
                y_pred=np.asarray([1.0]),
                intervals={
                    0.8: ConformalIntervalBlock(
                        coverage=0.8,
                        qhat=0.1,
                        lower=np.asarray([0.9]),
                        upper=np.asarray([1.1]),
                    )
                },
                method="bad",
            ),
            "method must be one of",
        ),
        (
            lambda: CalibratedPredictionBlock(
                y_pred=np.asarray([1.0]),
                intervals={
                    0.8: ConformalIntervalBlock(
                        coverage=0.8,
                        qhat=0.1,
                        lower=np.asarray([0.9]),
                        upper=np.asarray([1.1]),
                    )
                },
                group_keys=("g1", "g2"),
            ),
            "group_keys length must match y_pred rows",
        ),
    ],
)
def test_calibrated_prediction_block_rejects_invalid_direct_construction(
    make_block: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        make_block()


def test_conformal_interval_and_prediction_blocks_allow_unbounded_direct_construction() -> None:
    interval = ConformalIntervalBlock(
        coverage=0.95,
        qhat=float("inf"),
        lower=np.asarray([-math.inf]),
        upper=np.asarray([math.inf]),
    )
    block = CalibratedPredictionBlock(
        y_pred=np.asarray([10.0]),
        intervals={0.95: interval},
    )

    assert math.isinf(block.interval(0.95).qhat)
    assert block.interval(0.95).lower.tolist() == [-math.inf]
    assert block.interval(0.95).upper.tolist() == [math.inf]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"observed_coverage": math.nan}, "observed_coverage must not be NaN"),
        ({"observed_coverage": 1.2}, r"observed_coverage must be finite in \[0, 1\]"),
        ({"observed_coverage": "0.5"}, "observed_coverage must be a numeric scalar, not string"),
        ({"coverage_gap": math.nan}, "coverage_gap must not be NaN"),
        ({"coverage_gap": "0.0"}, "coverage_gap must be a numeric scalar, not string"),
        ({"mean_width": -1.0}, "mean_width must be non-negative"),
        ({"median_width": float("-inf")}, "median_width must be non-negative"),
        ({"mean_interval_score": math.nan}, "mean_interval_score must not be NaN"),
        ({"mean_interval_score": True}, "mean_interval_score must be a numeric scalar, not boolean"),
        ({"n_samples": True}, "n_samples must be an integer"),
    ],
)
def test_conformal_metric_set_rejects_invalid_metric_values(
    overrides: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        ConformalMetricSet(**_valid_metric_payload(**overrides))


def test_conformal_metric_set_rejects_metrics_inconsistent_with_counts() -> None:
    with pytest.raises(ValueError, match="observed_coverage must match n_covered / n_samples"):
        ConformalMetricSet(**_valid_metric_payload(observed_coverage=0.75, coverage_gap=-0.05))

    with pytest.raises(ValueError, match="coverage_gap must equal observed_coverage - coverage"):
        ConformalMetricSet(**_valid_metric_payload(coverage_gap=0.0))


def test_conformal_metric_set_allows_positive_infinite_unbounded_interval_metrics() -> None:
    metrics = ConformalMetricSet(
        **_valid_metric_payload(
            mean_width=float("inf"),
            median_width=float("inf"),
            mean_interval_score=float("inf"),
        )
    )

    assert metrics.to_dict()["mean_width"] == "Infinity"
    assert metrics.to_dict()["median_width"] == "Infinity"
    assert metrics.to_dict()["mean_interval_score"] == "Infinity"


def test_calibrated_run_result_metrics_delegates_to_prediction_evaluation() -> None:
    result = calibrate_replayed_predictions(
        y_true_calibration=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0, 20.0],
        spec=parse_conformal_calibration_spec({"coverage": 0.8}),
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
    )

    metrics = result.metrics([10.1, 21.0])

    assert metrics[0.8].n_covered == 1
    assert metrics[0.8].mean_interval_score == pytest.approx(3.8)


def test_calibrated_prediction_block_rejects_non_materialized_coverage() -> None:
    calibrator = fit_split_absolute_residual_calibrator(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        coverage=0.8,
    )
    block = calibrator.apply_block([10.0])

    with pytest.raises(KeyError, match="not materialized"):
        block.interval(0.5)


def test_conformal_calibration_artifact_round_trips_with_verified_fingerprint() -> None:
    spec = parse_conformal_calibration_spec({"coverage": [0.5, 0.8], "method": "split_absolute_residual"})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
        target_name="moisture",
        predictor_fingerprint="predictor-abc",
        calibration_data_fingerprint="calibration-def",
    )

    payload = artifact.to_dict()
    restored = ConformalCalibrationArtifact.from_dict(payload)

    assert payload["fingerprint"] == artifact.fingerprint
    assert restored.to_dict() == payload
    assert restored.target_name == "moisture"
    assert restored.predictor_fingerprint == "predictor-abc"
    assert restored.calibration_data_fingerprint == "calibration-def"
    assert restored.calibration_size == 4
    assert restored.calibrator.qhat_by_coverage[0.8] == pytest.approx(0.4)
    np.testing.assert_allclose(restored.calibrator.apply_block([10.0]).interval(0.8).lower, [9.6])


def test_conformal_calibration_artifact_saves_and_loads_deterministic_json(tmp_path) -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
    )
    target = tmp_path / "conformal-artifact.json"

    saved = artifact.save_json(target)
    first = target.read_text(encoding="utf-8")
    artifact.save_json(target)
    second = target.read_text(encoding="utf-8")

    assert saved == target
    assert first == second
    assert first.endswith("\n")
    assert ConformalCalibrationArtifact.load_json(target).to_dict() == artifact.to_dict()


def test_conformal_calibration_artifact_serializes_unbounded_qhat_without_nonstandard_json() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.95})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0],
        y_pred=[1.1, 1.9],
        spec=spec,
    )

    payload = artifact.to_dict()
    restored = ConformalCalibrationArtifact.from_json(artifact.to_json())

    assert payload["qhat_by_coverage"] == [{"coverage": 0.95, "qhat": "Infinity"}]
    assert '"Infinity"' in artifact.to_json()
    assert math.isinf(restored.calibrator.qhat_by_coverage[0.95])


def test_conformal_calibration_artifact_rejects_fingerprint_mismatch() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})
    payload = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
    ).to_dict()
    payload["fingerprint"] = "0" * 64

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ConformalCalibrationArtifact.from_dict(payload)


def test_conformal_versioned_contracts_reject_boolean_versions_on_construction_and_reload() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
    )
    cohort = normalize_conformal_calibration_cohort(
        y_true=[1.0, 2.0],
        y_pred=[0.9, 2.1],
        sample_ids=["c1", "c2"],
    )
    result = CalibratedRunResult(
        artifact=artifact,
        prediction=artifact.calibrator.apply_block([10.0]),
        sample_ids=("p1",),
    )

    with pytest.raises(ValueError, match="ConformalCalibrationCohortManifest.version must be 1"):
        ConformalCalibrationCohortManifest(rows=cohort.rows, version=True)
    with pytest.raises(ValueError, match="ConformalCalibrationArtifact.version must be 1"):
        ConformalCalibrationArtifact(
            spec=artifact.spec,
            calibrator=artifact.calibrator,
            calibration_size=artifact.calibration_size,
            version=True,
        )
    with pytest.raises(ValueError, match="CalibratedRunResult.version must be 1"):
        CalibratedRunResult(
            artifact=result.artifact,
            prediction=result.prediction,
            sample_ids=result.sample_ids,
            version=True,
        )

    payload = cohort.to_dict()
    payload["version"] = True
    payload.pop("fingerprint")
    with pytest.raises(ValueError, match="ConformalCalibrationCohortManifest.version must be an integer"):
        ConformalCalibrationCohortManifest.from_dict(payload)

    payload = artifact.to_dict()
    payload["version"] = True
    payload.pop("fingerprint")
    with pytest.raises(ValueError, match="ConformalCalibrationArtifact.version must be an integer"):
        ConformalCalibrationArtifact.from_dict(payload)

    payload = result.to_dict()
    payload["version"] = True
    payload.pop("fingerprint")
    with pytest.raises(ValueError, match="CalibratedRunResult.version must be an integer"):
        CalibratedRunResult.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("target_name", "", "must be null or a non-empty string"),
        ("target_name", " moisture ", "must not contain surrounding whitespace"),
        ("target_name", "bad\x00target", "must not contain NUL bytes"),
        ("predictor_fingerprint", " predictor ", "must not contain surrounding whitespace"),
        ("predictor_fingerprint", "bad\x00predictor", "must not contain NUL bytes"),
        ("calibration_data_fingerprint", " ", "must not contain surrounding whitespace"),
        ("predictor_fingerprint", 123, "must be null or a non-empty string"),
    ],
)
def test_conformal_calibration_artifact_rejects_invalid_optional_string_fields(
    field: str,
    value: object,
    match: str,
) -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
    )

    with pytest.raises(ValueError, match=match):
        ConformalCalibrationArtifact(
            spec=artifact.spec,
            calibrator=artifact.calibrator,
            calibration_size=artifact.calibration_size,
            **{field: value},
        )

    payload = artifact.to_dict()
    payload[field] = value
    payload.pop("fingerprint")
    with pytest.raises(ValueError, match=match):
        ConformalCalibrationArtifact.from_dict(payload)


def test_conformal_calibration_artifact_fits_grouped_quantiles_from_cohort() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.5, "group_by": "group"})
    cohort = normalize_conformal_calibration_cohort(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 1.8, 2.5, 3.4],
        sample_ids=["c1", "c2", "c3", "c4"],
        groups=["instrument-a", "instrument-a", "instrument-b", "instrument-b"],
    )

    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 1.8, 2.5, 3.4],
        spec=spec,
        calibration_cohort=cohort,
    )
    payload = artifact.to_dict()
    restored = ConformalCalibrationArtifact.from_dict(payload)

    assert sorted(artifact.group_calibrators) == ['["instrument-a"]', '["instrument-b"]']
    assert artifact.group_calibrators['["instrument-a"]'].qhat_by_coverage[0.5] == pytest.approx(0.2)
    assert artifact.group_calibrators['["instrument-b"]'].qhat_by_coverage[0.5] == pytest.approx(0.6)
    assert restored.to_dict() == payload


def test_conformal_calibration_artifact_rejects_grouped_fit_without_cohort() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8, "group_by": "group"})

    with pytest.raises(ValueError, match="calibration_cohort"):
        fit_conformal_calibration_artifact([1.0, 2.0], [1.1, 1.9], spec=spec)


def test_conformal_calibration_artifact_fits_joint_max_multi_target_scores() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8, "multi_target": "joint_max"})

    artifact = fit_conformal_calibration_artifact(
        y_true=[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
        y_pred=[[0.9, 9.8], [2.3, 20.1], [2.4, 30.2], [4.1, 39.6]],
        spec=spec,
    )
    restored = ConformalCalibrationArtifact.from_dict(artifact.to_dict())

    assert artifact.spec.multi_target == "joint_max"
    np.testing.assert_allclose(artifact.calibrator.scores, [0.2, 0.3, 0.6, 0.4])
    assert artifact.calibrator.qhat_by_coverage[0.8] == pytest.approx(0.6)
    assert restored.to_dict() == artifact.to_dict()


def test_conformal_calibration_artifact_rejects_joint_max_non_2d_arrays() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8, "multi_target": "joint_max"})

    with pytest.raises(ValueError, match="two-dimensional"):
        fit_conformal_calibration_artifact(
            y_true=[1.0, 2.0],
            y_pred=[0.9, 2.1],
            spec=spec,
        )


def test_normalize_conformal_calibration_cohort_preserves_ids_groups_roles_and_metadata() -> None:
    manifest = normalize_conformal_calibration_cohort(
        y_true=[1.0, 2.0, 3.0],
        y_pred=[0.9, 2.1, 2.8],
        sample_ids=["s1", "s2", "s3"],
        groups=["instrument-a", None, "instrument-b"],
        metadata={
            "batch": ["b1", "b1", "b2"],
            "temperature": [20.0, 21.5, 19.0],
        },
    )

    payload = manifest.to_dict()
    restored = ConformalCalibrationCohortManifest.from_dict(payload)

    assert payload["fingerprint"] == manifest.fingerprint
    assert restored.to_dict() == payload
    assert restored.n_samples == 3
    assert restored.sample_ids == ("s1", "s2", "s3")
    assert restored.groups == ("instrument-a", None, "instrument-b")
    assert {row.role for row in restored.rows} == {"calibration"}
    assert restored.rows[1].metadata == {"batch": "b1", "temperature": 21.5}


@pytest.mark.parametrize("sample_ids", [[" s1 ", "s2"], ["bad\x00id", "s2"], [1, "s2"]])
def test_normalize_conformal_calibration_cohort_rejects_coercive_sample_ids(sample_ids: list[object]) -> None:
    with pytest.raises(ValueError, match="sample_ids must contain canonical non-empty strings"):
        normalize_conformal_calibration_cohort(
            y_true=[1.0, 2.0],
            y_pred=[0.9, 2.1],
            sample_ids=sample_ids,
        )


def test_normalize_conformal_calibration_cohort_accepts_row_metadata() -> None:
    manifest = normalize_conformal_calibration_cohort(
        y_true=[1.0, 2.0],
        y_pred=[0.9, 2.1],
        sample_ids=["s1", "s2"],
        metadata=[{"batch": "b1"}, {"batch": "b2"}],
    )

    assert [row.metadata for row in manifest.rows] == [{"batch": "b1"}, {"batch": "b2"}]


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        ({1: ["b1", "b2"]}, "metadata key must be a non-empty string"),
        ({" batch ": ["b1", "b2"]}, "metadata key must not contain surrounding whitespace"),
        ([{"batch": "b1"}, {1: "b2"}], "metadata key must be a non-empty string"),
        ([{"batch": "b1"}, {" batch ": "b2"}], "metadata key must not contain surrounding whitespace"),
    ],
)
def test_normalize_conformal_calibration_cohort_rejects_non_canonical_metadata_keys(
    metadata: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        normalize_conformal_calibration_cohort(
            y_true=[1.0, 2.0],
            y_pred=[0.9, 2.1],
            sample_ids=["s1", "s2"],
            metadata=metadata,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"row_index": False}, "row_index must be a non-negative integer"),
        ({"sample_id": " s1 "}, "sample_id must not contain surrounding whitespace"),
        ({"sample_id": "bad\x00id"}, "sample_id must not contain NUL bytes"),
        ({"role": " calibration "}, "role must not contain surrounding whitespace"),
        ({"role": "bad\x00role"}, "role must not contain NUL bytes"),
        ({"group": " instrument-a "}, "group must not contain surrounding whitespace"),
        ({"group": "bad\x00group"}, "group must not contain NUL bytes"),
        ({"metadata": {1: "value"}}, "metadata mapping keys must be non-empty strings"),
        ({"metadata": {"bad": float("nan")}}, "metadata.bad must be JSON-compatible"),
    ],
)
def test_conformal_calibration_cohort_row_rejects_non_canonical_identity_or_metadata(
    kwargs: dict[str, object],
    match: str,
) -> None:
    payload: dict[str, object] = {
        "row_index": 0,
        "sample_id": "s1",
        "role": "calibration",
        "group": None,
        "metadata": {},
    }
    payload.update(kwargs)

    with pytest.raises(ValueError, match=match):
        ConformalCalibrationCohortRow(**payload)


@pytest.mark.parametrize(
    ("mutate_row", "match"),
    [
        (lambda row: row.update({"sample_id": " s1 "}), "sample_id must not contain surrounding whitespace"),
        (lambda row: row.update({"sample_id": "bad\x00id"}), "sample_id must not contain NUL bytes"),
        (lambda row: row.update({"group": " instrument-a "}), "group must not contain surrounding whitespace"),
        (lambda row: row.update({"group": "bad\x00group"}), "group must not contain NUL bytes"),
        (lambda row: row.update({"metadata": {1: "value"}}), "metadata mapping keys must be non-empty strings"),
    ],
)
def test_conformal_calibration_cohort_manifest_rejects_non_canonical_rows_on_reload(
    mutate_row: Callable[[dict[str, object]], None],
    match: str,
) -> None:
    manifest = normalize_conformal_calibration_cohort(
        y_true=[1.0],
        y_pred=[0.9],
        sample_ids=["s1"],
    )
    payload = manifest.to_dict()
    mutate_row(payload["rows"][0])
    payload.pop("fingerprint")

    with pytest.raises(ValueError, match=match):
        ConformalCalibrationCohortManifest.from_dict(payload)


@pytest.mark.parametrize(
    ("n_samples", "match"),
    [
        (True, "n_samples must be an integer"),
        ("1", "n_samples must be an integer"),
        (2, "n_samples mismatch"),
    ],
)
def test_conformal_calibration_cohort_manifest_rejects_ambiguous_n_samples_on_reload(
    n_samples: object,
    match: str,
) -> None:
    manifest = normalize_conformal_calibration_cohort(
        y_true=[1.0],
        y_pred=[0.9],
        sample_ids=["s1"],
    )
    payload = manifest.to_dict()
    payload["n_samples"] = n_samples
    payload.pop("fingerprint")

    with pytest.raises(ValueError, match=match):
        ConformalCalibrationCohortManifest.from_dict(payload)


def test_conformal_calibration_cohort_manifest_rejects_stringified_unit_on_reload() -> None:
    manifest = normalize_conformal_calibration_cohort(
        y_true=[1.0],
        y_pred=[0.9],
        sample_ids=["s1"],
    )
    payload = manifest.to_dict()
    payload["unit"] = _StringifiesAs("physical_sample")
    payload.pop("fingerprint")

    with pytest.raises(ValueError, match="ConformalCalibrationCohortManifest.unit must be a string"):
        ConformalCalibrationCohortManifest.from_dict(payload)


def test_conformal_calibration_artifact_embeds_calibration_cohort_manifest() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})
    cohort = normalize_conformal_calibration_cohort(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        sample_ids=["s1", "s2", "s3", "s4"],
        groups=["instrument-a", "instrument-a", "instrument-b", "instrument-b"],
    )
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
        calibration_cohort=cohort,
    )

    restored = ConformalCalibrationArtifact.from_dict(artifact.to_dict())

    assert artifact.calibration_data_fingerprint == cohort.fingerprint
    assert restored.calibration_cohort is not None
    assert restored.calibration_cohort.sample_ids == ("s1", "s2", "s3", "s4")
    assert restored.calibration_cohort.groups == ("instrument-a", "instrument-a", "instrument-b", "instrument-b")


@pytest.mark.parametrize(
    ("n_samples", "match"),
    [
        (True, "group_calibrators n_samples must be an integer"),
        ("1", "group_calibrators n_samples must be an integer"),
        (2, "group_calibrators n_samples mismatch"),
    ],
)
def test_conformal_calibration_artifact_rejects_ambiguous_group_calibrator_n_samples_on_reload(
    n_samples: object,
    match: str,
) -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.5, "group_by": "group"})
    cohort = normalize_conformal_calibration_cohort(
        y_true=[1.0, 2.0],
        y_pred=[1.1, 1.8],
        sample_ids=["c1", "c2"],
        groups=["instrument-a", "instrument-b"],
    )
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0],
        y_pred=[1.1, 1.8],
        spec=spec,
        calibration_cohort=cohort,
    )
    payload = artifact.to_dict()
    payload["group_calibrators"][0]["n_samples"] = n_samples
    payload.pop("fingerprint")

    with pytest.raises(ValueError, match=match):
        ConformalCalibrationArtifact.from_dict(payload)


def test_conformal_calibration_artifact_rejects_qhat_not_derived_from_scores() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
    )
    payload = artifact.to_dict()
    payload["qhat_by_coverage"][0]["qhat"] = 0.1

    with pytest.raises(ValueError, match="qhat_by_coverage must match retained conformal scores"):
        ConformalCalibrationArtifact.from_dict(payload)


def test_conformal_calibration_artifact_rejects_numpy_bool_numeric_payloads_on_reload() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
    )
    payload = artifact.to_dict()
    payload["scores"] = [np.bool_(True), np.bool_(True), np.bool_(True), np.bool_(True)]
    payload["qhat_by_coverage"][0]["qhat"] = np.bool_(True)
    payload.pop("fingerprint")

    with pytest.raises(ValueError, match="must contain numeric values, not booleans"):
        ConformalCalibrationArtifact.from_dict(payload)


@pytest.mark.parametrize("value", [np.array(True), np.array(1.0)])
def test_conformal_calibration_artifact_rejects_numpy_array_scalar_payloads_on_reload(value: np.ndarray) -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
    )
    payload = artifact.to_dict()
    payload["scores"] = [value, value, value, value]
    payload["qhat_by_coverage"][0]["qhat"] = value
    payload.pop("fingerprint")

    with pytest.raises(ValueError, match="must be a JSON numeric scalar, not a NumPy array"):
        ConformalCalibrationArtifact.from_dict(payload)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sample_ids": None}, "sample_ids are required"),
        ({"sample_ids": ["s1"]}, "sample_ids length"),
        ({"sample_ids": ["s1", "s1"]}, "unique physical sample"),
        ({"sample_ids": ["s1", "s2"], "groups": ["a"]}, "groups length"),
        ({"sample_ids": ["s1", "s2"], "metadata": {"batch": ["b1"]}}, "metadata column length"),
        ({"sample_ids": ["s1", "s2"], "role": "validation"}, "role='calibration'"),
    ],
)
def test_normalize_conformal_calibration_cohort_rejects_ambiguous_identity_inputs(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        normalize_conformal_calibration_cohort(
            y_true=[1.0, 2.0],
            y_pred=[0.9, 2.1],
            **kwargs,
        )


def test_calibrated_run_result_round_trips_predictions_intervals_and_sample_ids() -> None:
    spec = parse_conformal_calibration_spec({"coverage": [0.5, 0.8]})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
        predictor_fingerprint="predictor-abc",
    )
    result = CalibratedRunResult(
        artifact=artifact,
        prediction=artifact.calibrator.apply_block([10.0, 20.0]),
        sample_ids=("s1", "s2"),
        metadata={
            "source": "unit-test",
            "tuning_calibration_source": {
                "source": "tuning.winner",
                "score_data_role": "hpo_objective_only",
                "score_data_used": False,
            },
        },
    )

    payload = result.to_dict()
    restored = CalibratedRunResult.from_dict(payload)

    assert payload["fingerprint"] == result.fingerprint
    assert restored.to_dict() == payload
    assert restored.sample_ids == ("s1", "s2")
    assert restored.metadata == {
        "source": "unit-test",
        "tuning_calibration_source": {
            "source": "tuning.winner",
            "score_data_role": "hpo_objective_only",
            "score_data_used": False,
        },
    }
    assert restored.tuning_calibration_source == {
        "source": "tuning.winner",
        "score_data_role": "hpo_objective_only",
        "score_data_used": False,
    }
    np.testing.assert_allclose(restored.prediction.y_pred, [10.0, 20.0])
    np.testing.assert_allclose(restored.prediction.interval(0.8).upper, [10.4, 20.4])
    prediction = restored.to_predict_result()
    np.testing.assert_allclose(prediction.y_pred, [10.0, 20.0])
    np.testing.assert_allclose(prediction.interval(0.8).upper, [10.4, 20.4])
    assert prediction.metadata["calibrated_result_fingerprint"] == restored.fingerprint
    assert prediction.tuning_calibration_source == restored.tuning_calibration_source


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("method", _StringifiesAs("split_absolute_residual"), "prediction.method must be a string"),
        ("unit", _StringifiesAs("physical_sample"), "prediction.unit must be a string"),
    ],
)
def test_calibrated_run_result_rejects_stringified_prediction_contract_fields_on_reload(
    field: str,
    value: object,
    match: str,
) -> None:
    spec = parse_conformal_calibration_spec({"coverage": [0.5, 0.8]})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
    )
    result = CalibratedRunResult(
        artifact=artifact,
        prediction=artifact.calibrator.apply_block([10.0, 20.0]),
        sample_ids=("s1", "s2"),
    )
    payload = result.to_dict()
    payload["prediction"][field] = value
    payload.pop("fingerprint")

    with pytest.raises(ValueError, match=match):
        CalibratedRunResult.from_dict(payload)


def test_calibrated_run_result_saves_and_loads_deterministic_json(tmp_path) -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
    )
    result = CalibratedRunResult(
        artifact=artifact,
        prediction=artifact.calibrator.apply_block([10.0]),
        sample_ids=("s1",),
    )
    target = tmp_path / "calibrated-result.json"

    saved = result.save_json(target)
    first = target.read_text(encoding="utf-8")
    result.save_json(target)
    second = target.read_text(encoding="utf-8")

    assert saved == target
    assert first == second
    assert first.endswith("\n")
    assert CalibratedRunResult.load_json(target).to_dict() == result.to_dict()


def test_calibrated_run_result_rejects_inconsistent_prediction_contract() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
    )

    with pytest.raises(ValueError, match="sample_ids length"):
        CalibratedRunResult(
            artifact=artifact,
            prediction=artifact.calibrator.apply_block([10.0, 20.0]),
            sample_ids=("s1",),
        )
    with pytest.raises(ValueError, match="sample_ids are required"):
        CalibratedRunResult(
            artifact=artifact,
            prediction=artifact.calibrator.apply_block([10.0]),
        )
    with pytest.raises(ValueError, match="unique physical sample"):
        CalibratedRunResult(
            artifact=artifact,
            prediction=artifact.calibrator.apply_block([10.0, 20.0]),
            sample_ids=("s1", "s1"),
        )
    with pytest.raises(ValueError, match="non-empty strings"):
        CalibratedRunResult(
            artifact=artifact,
            prediction=artifact.calibrator.apply_block([10.0]),
            sample_ids=("",),
        )
    with pytest.raises(ValueError, match="surrounding whitespace"):
        CalibratedRunResult(
            artifact=artifact,
            prediction=artifact.calibrator.apply_block([10.0]),
            sample_ids=(" s1 ",),
        )
    with pytest.raises(ValueError, match="NUL bytes"):
        CalibratedRunResult(
            artifact=artifact,
            prediction=artifact.calibrator.apply_block([10.0]),
            sample_ids=("bad\x00id",),
        )
    with pytest.raises(ValueError, match="non-empty strings"):
        CalibratedRunResult(
            artifact=artifact,
            prediction=artifact.calibrator.apply_block([10.0]),
            sample_ids=(1,),  # type: ignore[arg-type]
        )

    payload = CalibratedRunResult(
        artifact=artifact,
        prediction=artifact.calibrator.apply_block([10.0]),
        sample_ids=("s1",),
    ).to_dict()
    payload["fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        CalibratedRunResult.from_dict(payload)


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        ({"bad": float("nan")}, "JSON-compatible"),
        ({"bad": object()}, "JSON-compatible"),
        ({" bad ": "value"}, "surrounding whitespace"),
        ({1: "value"}, "metadata key"),
        ({"nested": {1: "value"}}, "mapping keys must be non-empty strings"),
        ({"nested": {" key ": "value"}}, "mapping keys must not contain surrounding whitespace"),
        ({"tuple": ("not", "json-native")}, "JSON-compatible"),
        ([("source", "unit-test")], "metadata must be a mapping"),
    ],
)
def test_calibrated_run_result_rejects_non_strict_json_metadata(
    metadata: object,
    match: str,
) -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
    )

    with pytest.raises(ValueError, match=match):
        CalibratedRunResult(
            artifact=artifact,
            prediction=artifact.calibrator.apply_block([10.0]),
            sample_ids=("s1",),
            metadata=metadata,  # type: ignore[arg-type]
        )


def test_calibrate_replayed_predictions_rejects_non_strict_json_result_metadata() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})

    with pytest.raises(ValueError, match=r"CalibratedRunResult.metadata\[bad\].*JSON-compatible"):
        calibrate_replayed_predictions(
            y_true_calibration=[1.0, 2.0, 3.0, 4.0],
            y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
            y_pred=[10.0],
            spec=spec,
            calibration_sample_ids=["c1", "c2", "c3", "c4"],
            prediction_sample_ids=["p1"],
            result_metadata={"bad": float("nan")},
        )


def test_calibrated_run_result_rejects_intervals_not_derived_from_artifact() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
    )
    result = CalibratedRunResult(
        artifact=artifact,
        prediction=artifact.calibrator.apply_block([10.0, 20.0]),
        sample_ids=("s1", "s2"),
    )

    payload = result.to_dict()
    payload["prediction"]["intervals"][0]["upper"] = [10.0, 200.0]
    with pytest.raises(ValueError, match="interval arrays must be derived"):
        CalibratedRunResult.from_dict(payload)

    payload = result.to_dict()
    payload["prediction"]["intervals"][0]["qhat"] = 0.1
    with pytest.raises(ValueError, match="interval qhat must match"):
        CalibratedRunResult.from_dict(payload)


def test_calibrated_run_result_rejects_stale_guarantee_metadata_on_load() -> None:
    spec = parse_conformal_calibration_spec({"coverage": [0.5, 0.8]})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
    )
    result = CalibratedRunResult(
        artifact=artifact,
        prediction=artifact.calibrator.apply_block([10.0]),
        sample_ids=("s1",),
        metadata={
            "conformal_guarantee_status": conformal_guarantee_status(
                artifact,
                effective_engine="nirs4all.python.replayed_array_calibration",
            )
        },
    )
    payload = result.to_dict()
    payload["metadata"]["conformal_guarantee_status"]["artifact_fingerprint"] = "0" * 64

    with pytest.raises(ValueError, match="artifact_fingerprint"):
        CalibratedRunResult.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("predictor_fingerprint", "predictor-b", "predictor_fingerprint"),
        ("calibration_data_fingerprint", "data-b", "calibration_data_fingerprint"),
        ("guarantee", "fake_guarantee", "guarantee does not match"),
        ("scope", "fake_scope", "scope does not match"),
    ],
)
def test_calibrated_run_result_rejects_stale_guarantee_provenance_on_load(
    field: str,
    value: object,
    match: str,
) -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
        predictor_fingerprint="predictor-a",
        calibration_data_fingerprint="data-a",
    )
    result = CalibratedRunResult(
        artifact=artifact,
        prediction=artifact.calibrator.apply_block([10.0]),
        sample_ids=("s1",),
        metadata={
            "conformal_guarantee_status": conformal_guarantee_status(
                artifact,
                effective_engine="nirs4all.python.replayed_array_calibration",
            )
        },
    )
    payload = result.to_dict()
    payload["metadata"]["conformal_guarantee_status"][field] = value
    payload.pop("fingerprint")

    with pytest.raises(ValueError, match=match):
        CalibratedRunResult.from_dict(payload)


@pytest.mark.parametrize(
    "missing_key",
    [
        "artifact_fingerprint",
        "calibrated_coverages",
        "calibration_data_fingerprint",
        "coverage",
        "effective_engine",
        "guarantee",
        "group_by",
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
    ],
)
def test_calibrated_run_result_rejects_partial_guarantee_status_on_load(missing_key: str) -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
    )
    result = CalibratedRunResult(
        artifact=artifact,
        prediction=artifact.calibrator.apply_block([10.0]),
        sample_ids=("s1",),
        metadata={
            "conformal_guarantee_status": conformal_guarantee_status(
                artifact,
                effective_engine="nirs4all.python.replayed_array_calibration",
            )
        },
    )
    payload = result.to_dict()
    payload["metadata"]["conformal_guarantee_status"].pop(missing_key)
    payload.pop("fingerprint")

    with pytest.raises(ValueError, match="conformal_guarantee_status is missing keys"):
        CalibratedRunResult.from_dict(payload)


@pytest.mark.parametrize(
    ("mutate_status", "match"),
    [
        (
            lambda status: status.update({"status": "fake"}),
            "status must be 'active' or 'invalidated'",
        ),
        (
            lambda status: status.update({"status": "invalidated"}),
            "status does not match invalidation_reasons",
        ),
        (
            lambda status: status.update({"status": "active", "invalidation_reasons": ["predictor changed"]}),
            "status does not match invalidation_reasons",
        ),
    ],
)
def test_calibrated_run_result_rejects_incoherent_guarantee_status_state_on_load(
    mutate_status: Callable[[dict[str, object]], None],
    match: str,
) -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
    )
    result = CalibratedRunResult(
        artifact=artifact,
        prediction=artifact.calibrator.apply_block([10.0]),
        sample_ids=("s1",),
        metadata={
            "conformal_guarantee_status": conformal_guarantee_status(
                artifact,
                effective_engine="nirs4all.python.replayed_array_calibration",
            )
        },
    )
    payload = result.to_dict()
    mutate_status(payload["metadata"]["conformal_guarantee_status"])
    payload.pop("fingerprint")

    with pytest.raises(ValueError, match=match):
        CalibratedRunResult.from_dict(payload)


@pytest.mark.parametrize(
    ("limitations", "match"),
    [
        (["fake limitation"], "limitations do not match"),
        ([], "limitations do not match"),
        ([True], r"limitations\[0\] must be a non-empty string"),
        (
            ["finite-sample marginal coverage requires exchangeable calibration and prediction samples"],
            "limitations do not match",
        ),
        ("fake limitation", "limitations must be a sequence of non-empty strings"),
    ],
)
def test_calibrated_run_result_rejects_stale_or_malformed_guarantee_limitations_on_load(
    limitations: object,
    match: str,
) -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
    )
    result = CalibratedRunResult(
        artifact=artifact,
        prediction=artifact.calibrator.apply_block([10.0]),
        sample_ids=("s1",),
        metadata={
            "conformal_guarantee_status": conformal_guarantee_status(
                artifact,
                effective_engine="nirs4all.python.replayed_array_calibration",
            )
        },
    )
    payload = result.to_dict()
    payload["metadata"]["conformal_guarantee_status"]["limitations"] = limitations
    payload.pop("fingerprint")

    with pytest.raises(ValueError, match=match):
        CalibratedRunResult.from_dict(payload)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"effective_engine": True}, "effective_engine must be a non-empty string"),
        ({"effective_engine": " engine "}, "effective_engine must not contain surrounding whitespace"),
        ({"effective_engine": "bad\x00engine"}, "effective_engine must not contain NUL bytes"),
        (
            {"effective_engine": "engine", "requested_engine": False},
            "requested_engine must be a non-empty string",
        ),
        (
            {"effective_engine": "engine", "requested_engine": "bad\x00engine"},
            "requested_engine must not contain NUL bytes",
        ),
        (
            {"effective_engine": "engine", "source_calibrated_result_fingerprint": " source "},
            "source_calibrated_result_fingerprint must not contain surrounding whitespace",
        ),
        (
            {"effective_engine": "engine", "source_calibrated_result_fingerprint": "bad\x00source"},
            "source_calibrated_result_fingerprint must not contain NUL bytes",
        ),
        (
            {"effective_engine": "engine", "invalidation_reasons": (object(),)},
            r"invalidation_reasons\[0\] must be a non-empty string",
        ),
        (
            {"effective_engine": "engine", "invalidation_reasons": (" reason ",)},
            r"invalidation_reasons\[0\] must not contain surrounding whitespace",
        ),
        (
            {"effective_engine": "engine", "invalidation_reasons": ("bad\x00reason",)},
            r"invalidation_reasons\[0\] must not contain NUL bytes",
        ),
    ],
)
def test_conformal_guarantee_status_rejects_coerced_or_ambiguous_strings(
    kwargs: dict[str, object],
    match: str,
) -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
    )

    with pytest.raises(ValueError, match=match):
        conformal_guarantee_status(artifact, **kwargs)


@pytest.mark.parametrize(
    ("mutate_status", "match"),
    [
        (lambda status: status.update({"version": True}), "conformal_guarantee_status.version must be an integer"),
        (lambda status: status.update({"effective_engine": " engine "}), "effective_engine must not contain surrounding whitespace"),
        (lambda status: status.update({"effective_engine": "bad\x00engine"}), "effective_engine must not contain NUL bytes"),
        (lambda status: status.update({"requested_engine": False}), "requested_engine must be a non-empty string"),
        (
            lambda status: status.update({"source_calibrated_result_fingerprint": " source "}),
            "source_calibrated_result_fingerprint must not contain surrounding whitespace",
        ),
        (
            lambda status: status.update({"source_calibrated_result_fingerprint": "bad\x00source"}),
            "source_calibrated_result_fingerprint must not contain NUL bytes",
        ),
        (
            lambda status: status.update({"invalidation_reasons": [" reason "]}),
            r"invalidation_reasons\[0\] must not contain surrounding whitespace",
        ),
        (
            lambda status: status.update({"invalidation_reasons": ["bad\x00reason"]}),
            r"invalidation_reasons\[0\] must not contain NUL bytes",
        ),
    ],
)
def test_calibrated_run_result_rejects_invalid_guarantee_metadata_strings_on_reload(
    mutate_status: Callable[[dict[str, object]], None],
    match: str,
) -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
    )
    result = CalibratedRunResult(
        artifact=artifact,
        prediction=artifact.calibrator.apply_block([10.0]),
        sample_ids=("s1",),
        metadata={
            "conformal_guarantee_status": conformal_guarantee_status(
                artifact,
                effective_engine="nirs4all.python.replayed_array_calibration",
            )
        },
    )
    payload = result.to_dict()
    mutate_status(payload["metadata"]["conformal_guarantee_status"])
    payload.pop("fingerprint")

    with pytest.raises(ValueError, match=match):
        CalibratedRunResult.from_dict(payload)


def test_calibrated_run_result_rejects_non_materialized_guarantee_coverage() -> None:
    spec = parse_conformal_calibration_spec({"coverage": [0.5, 0.8]})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
    )
    status = conformal_guarantee_status(
        artifact,
        effective_engine="nirs4all.python.replayed_array_calibration",
    )
    status["coverage"] = [0.9]

    with pytest.raises(ValueError, match="non-materialized coverage"):
        CalibratedRunResult(
            artifact=artifact,
            prediction=artifact.calibrator.apply_block([10.0]),
            sample_ids=("s1",),
            metadata={"conformal_guarantee_status": status},
        )


def test_calibrated_run_result_rejects_mismatched_source_fingerprint_metadata() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
    )
    status = conformal_guarantee_status(
        artifact,
        effective_engine="nirs4all.python.replayed_array_apply",
        source_calibrated_result_fingerprint="source-a",
    )

    with pytest.raises(ValueError, match="source_calibrated_result_fingerprint"):
        CalibratedRunResult(
            artifact=artifact,
            prediction=artifact.calibrator.apply_block([10.0]),
            sample_ids=("s1",),
            metadata={
                "conformal_guarantee_status": status,
                "source_calibrated_result_fingerprint": "source-b",
            },
        )


def test_calibrated_run_result_rejects_mismatched_calibration_replay_source_metadata() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
    )
    status = conformal_guarantee_status(
        artifact,
        effective_engine="nirs4all.python.replayed_array_calibration",
        calibration_replay_source={"kind": "replayed_arrays", "route": "provided_arrays", "version": 1},
    )

    with pytest.raises(ValueError, match="calibration_replay_source"):
        CalibratedRunResult(
            artifact=artifact,
            prediction=artifact.calibrator.apply_block([10.0]),
            sample_ids=("s1",),
            metadata={
                "calibration_replay_source": {"kind": "user-overwrite", "route": "provided_arrays", "version": 1},
                "conformal_guarantee_status": status,
            },
        )


def test_calibrate_replayed_predictions_builds_artifact_and_calibrated_result() -> None:
    spec = parse_conformal_calibration_spec({"coverage": [0.5, 0.8]})

    result = calibrate_replayed_predictions(
        y_true_calibration=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0, 20.0],
        spec=spec,
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        calibration_groups=["instrument-a", "instrument-a", "instrument-b", "instrument-b"],
        calibration_metadata={"batch": ["b1", "b1", "b2", "b2"]},
        result_metadata={"phase": "prediction"},
        target_name="moisture",
        predictor_fingerprint="predictor-abc",
    )

    assert result.artifact.target_name == "moisture"
    assert result.artifact.predictor_fingerprint == "predictor-abc"
    assert result.artifact.calibration_cohort is not None
    assert result.artifact.calibration_cohort.sample_ids == ("c1", "c2", "c3", "c4")
    assert result.sample_ids == ("p1", "p2")
    assert result.metadata["phase"] == "prediction"
    assert result.conformal_guarantee_status is not None
    assert result.conformal_guarantee_status["status"] == "active"
    assert result.conformal_guarantee_status["effective_engine"] == "nirs4all.python.replayed_array_calibration"
    assert result.conformal_guarantee_status["predictor_fingerprint"] == "predictor-abc"
    np.testing.assert_allclose(result.prediction.interval(0.8).lower, [9.6, 19.6])
    np.testing.assert_allclose(result.prediction.interval(0.8).upper, [10.4, 20.4])


def test_calibrate_replayed_predictions_applies_grouped_qhat_per_prediction_row() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.5, "group_by": "group"})

    result = calibrate_replayed_predictions(
        y_true_calibration=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 1.8, 2.5, 3.4],
        y_pred=[10.0, 20.0, 30.0],
        spec=spec,
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2", "p3"],
        calibration_groups=["instrument-a", "instrument-a", "instrument-b", "instrument-b"],
        prediction_groups=["instrument-a", "instrument-b", "instrument-a"],
    )

    interval = result.prediction.interval(0.5)
    assert result.artifact.spec.group_by == ("group",)
    assert result.conformal_guarantee_status is not None
    assert result.conformal_guarantee_status["guarantee"] == "split_conformal_group_marginal_coverage"
    assert result.conformal_guarantee_status["group_by"] == ["group"]
    assert result.prediction.group_keys == ('["instrument-a"]', '["instrument-b"]', '["instrument-a"]')
    np.testing.assert_allclose(interval.qhat, [0.2, 0.6, 0.2])
    np.testing.assert_allclose(interval.lower, [9.8, 19.4, 29.8])
    np.testing.assert_allclose(interval.upper, [10.2, 20.6, 30.2])
    assert result.metrics([10.1, 20.7, 29.9])[0.5].n_covered == 2

    restored = CalibratedRunResult.from_dict(result.to_dict())
    np.testing.assert_allclose(restored.prediction.interval(0.5).qhat, [0.2, 0.6, 0.2])


def test_calibrate_replayed_predictions_rejects_grouped_prediction_without_group_keys() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.5, "group_by": "group"})

    with pytest.raises(ValueError, match="missing required group value"):
        calibrate_replayed_predictions(
            y_true_calibration=[1.0, 2.0, 3.0, 4.0],
            y_pred_calibration=[0.9, 1.8, 2.5, 3.4],
            y_pred=[10.0],
            spec=spec,
            calibration_sample_ids=["c1", "c2", "c3", "c4"],
            prediction_sample_ids=["p1"],
            calibration_groups=["instrument-a", "instrument-a", "instrument-b", "instrument-b"],
        )


@pytest.mark.parametrize(
    ("calibration_groups", "prediction_groups", "match"),
    [
        ([" instrument-a ", "instrument-a", "instrument-b", "instrument-b"], ["instrument-a"], "groups must not contain surrounding whitespace"),
        (["bad\x00group", "instrument-a", "instrument-b", "instrument-b"], ["instrument-a"], "groups must not contain NUL bytes"),
        (["instrument-a", "instrument-a", "instrument-b", "instrument-b"], [" instrument-a "], "groups must not contain surrounding whitespace"),
        (["instrument-a", "instrument-a", "instrument-b", "instrument-b"], ["bad\x00group"], "groups must not contain NUL bytes"),
    ],
)
def test_calibrate_replayed_predictions_rejects_coercive_group_labels(
    calibration_groups: list[str],
    prediction_groups: list[str],
    match: str,
) -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.5, "group_by": "group"})

    with pytest.raises(ValueError, match=match):
        calibrate_replayed_predictions(
            y_true_calibration=[1.0, 2.0, 3.0, 4.0],
            y_pred_calibration=[0.9, 1.8, 2.5, 3.4],
            y_pred=[10.0],
            spec=spec,
            calibration_sample_ids=["c1", "c2", "c3", "c4"],
            prediction_sample_ids=["p1"],
            calibration_groups=calibration_groups,
            prediction_groups=prediction_groups,
        )


def test_calibrate_replayed_predictions_rejects_grouped_prediction_unseen_group() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.5, "group_by": "group"})

    with pytest.raises(ValueError, match="absent from the conformal calibration artifact"):
        calibrate_replayed_predictions(
            y_true_calibration=[1.0, 2.0, 3.0, 4.0],
            y_pred_calibration=[0.9, 1.8, 2.5, 3.4],
            y_pred=[10.0],
            spec=spec,
            calibration_sample_ids=["c1", "c2", "c3", "c4"],
            prediction_sample_ids=["p1"],
            calibration_groups=["instrument-a", "instrument-a", "instrument-b", "instrument-b"],
            prediction_groups=["instrument-c"],
        )


def test_grouped_calibrated_result_rejects_edited_qhat_or_group_keys_on_reload() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.5, "group_by": "group"})
    result = calibrate_replayed_predictions(
        y_true_calibration=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 1.8, 2.5, 3.4],
        y_pred=[10.0, 20.0],
        spec=spec,
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        calibration_groups=["instrument-a", "instrument-a", "instrument-b", "instrument-b"],
        prediction_groups=["instrument-a", "instrument-b"],
    )

    payload = result.to_dict()
    payload["prediction"]["intervals"][0]["qhat"] = [0.2, 0.2]
    with pytest.raises(ValueError, match="interval qhat must match"):
        CalibratedRunResult.from_dict(payload)

    payload = result.to_dict()
    payload["prediction"]["group_keys"][1] = '["instrument-a"]'
    with pytest.raises(ValueError, match="interval qhat must match"):
        CalibratedRunResult.from_dict(payload)


def test_calibrate_replayed_predictions_supports_joint_max_multi_target_result() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8, "multi_target": "joint_max"})

    result = calibrate_replayed_predictions(
        y_true_calibration=[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
        y_pred_calibration=[[0.9, 9.8], [2.3, 20.1], [2.4, 30.2], [4.1, 39.6]],
        y_pred=[[10.0, 100.0], [20.0, 200.0]],
        spec=spec,
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
    )

    interval = result.prediction.interval(0.8)
    assert result.artifact.spec.multi_target == "joint_max"
    assert result.conformal_guarantee_status is not None
    assert result.conformal_guarantee_status["guarantee"] == "split_conformal_joint_max_simultaneous_coverage"
    assert result.conformal_guarantee_status["scope"] == "finite_sample_simultaneous_exchangeability"
    assert result.prediction.y_pred.shape == (2, 2)
    np.testing.assert_allclose(interval.lower, [[9.4, 99.4], [19.4, 199.4]])
    np.testing.assert_allclose(interval.upper, [[10.6, 100.6], [20.6, 200.6]])
    assert result.metrics([[10.1, 100.2], [20.7, 200.0]])[0.8].n_covered == 1

    restored = CalibratedRunResult.from_dict(result.to_dict())
    np.testing.assert_allclose(restored.prediction.interval(0.8).lower, interval.lower)


def test_joint_max_calibrated_result_rejects_edited_interval_on_reload() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8, "multi_target": "joint_max"})
    result = calibrate_replayed_predictions(
        y_true_calibration=[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
        y_pred_calibration=[[0.9, 9.8], [2.3, 20.1], [2.4, 30.2], [4.1, 39.6]],
        y_pred=[[10.0, 100.0], [20.0, 200.0]],
        spec=spec,
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
    )

    payload = result.to_dict()
    payload["prediction"]["intervals"][0]["upper"][1][1] = 999.0
    with pytest.raises(ValueError, match="interval arrays must be derived"):
        CalibratedRunResult.from_dict(payload)


def test_calibrate_replayed_predictions_requires_prediction_sample_ids() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})

    with pytest.raises(ValueError, match="sample_ids are required"):
        calibrate_replayed_predictions(
            y_true_calibration=[1.0, 2.0],
            y_pred_calibration=[0.9, 2.1],
            y_pred=[10.0],
            spec=spec,
            calibration_sample_ids=["c1", "c2"],
            prediction_sample_ids=None,
        )


def test_calibrate_replayed_predictions_rejects_prediction_calibration_sample_overlap() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})

    with pytest.raises(ValueError, match="disjoint from calibration_cohort"):
        calibrate_replayed_predictions(
            y_true_calibration=[1.0, 2.0, 3.0, 4.0],
            y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
            y_pred=[10.0, 20.0],
            spec=spec,
            calibration_sample_ids=["c1", "c2", "c3", "c4"],
            prediction_sample_ids=["p1", "c2"],
        )


def test_split_absolute_residual_calibrator_marks_unbounded_small_calibration() -> None:
    calibrator = fit_split_absolute_residual_calibrator(
        y_true=[1.0, 2.0],
        y_pred=[1.1, 1.9],
        coverage=0.95,
    )

    assert math.isinf(calibrator.qhat_by_coverage[0.95])


@pytest.mark.parametrize("coverage", [0.0, 1.0, -0.1, float("nan"), [], True, "0.8", [True], ["0.8"]])
def test_normalize_coverages_rejects_invalid_values(coverage) -> None:
    with pytest.raises(ValueError):
        normalize_coverages(coverage)


def test_split_absolute_residual_calibrator_rejects_non_1d_or_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        fit_split_absolute_residual_calibrator([1.0, 2.0], [1.0], coverage=0.8)
    with pytest.raises(ValueError, match="one-dimensional"):
        fit_split_absolute_residual_calibrator([[1.0]], [[1.0]], coverage=0.8)
    with pytest.raises(NotImplementedError, match="physical_sample"):
        fit_split_absolute_residual_calibrator([1.0], [1.0], coverage=0.8, unit="observation")


def test_conformal_numeric_arrays_reject_boolean_values() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
    )

    with pytest.raises(ValueError, match="y_true.*not booleans"):
        fit_split_absolute_residual_calibrator(
            y_true=[True, False, True, False],
            y_pred=[1.0, 0.0, 1.0, 0.0],
            coverage=0.8,
        )
    with pytest.raises(ValueError, match="y_pred.*not booleans"):
        artifact.calibrator.apply_block([True])
    with pytest.raises(ValueError, match="y_true.*not booleans"):
        evaluate_conformal_prediction(
            y_true=[True],
            prediction=artifact.calibrator.apply_block([10.0]),
        )


def test_conformal_numeric_arrays_reject_string_values() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})
    artifact = fit_conformal_calibration_artifact(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[0.9, 2.2, 2.6, 4.3],
        spec=spec,
    )

    with pytest.raises(ValueError, match="y_true.*not strings"):
        fit_split_absolute_residual_calibrator(
            y_true=["1.0", "2.0", "3.0", "4.0"],
            y_pred=[0.9, 2.2, 2.6, 4.3],
            coverage=0.8,
        )
    with pytest.raises(ValueError, match="y_pred.*not strings"):
        artifact.calibrator.apply_block(["10.0"])
    payload = CalibratedRunResult(
        artifact=artifact,
        prediction=artifact.calibrator.apply_block([10.0]),
        sample_ids=("p1",),
    ).to_dict()
    payload["prediction"]["intervals"][0]["qhat"] = "0.4"
    payload.pop("fingerprint")
    with pytest.raises(ValueError, match="qhat.*not strings"):
        CalibratedRunResult.from_dict(payload)


def test_calibrated_run_result_rejects_boolean_prediction_payload_on_reload() -> None:
    spec = parse_conformal_calibration_spec({"coverage": 0.8})
    result = calibrate_replayed_predictions(
        y_true_calibration=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0],
        spec=spec,
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1"],
    )
    payload = result.to_dict()
    payload["prediction"]["y_pred"] = [True]
    payload.pop("fingerprint")

    with pytest.raises(ValueError, match="y_pred.*not booleans"):
        CalibratedRunResult.from_dict(payload)
