"""Unit tests for audit-only robustness reports."""

from __future__ import annotations

import importlib
import json

import numpy as np
import pytest

import nirs4all
from nirs4all.api.result import PredictResult
from nirs4all.api.robustness import _normalize_scenarios


class FirstColumnPredictor:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float)[:, 0]


class _StringifiesAs:
    def __init__(self, value: str) -> None:
        self._value = value

    def __str__(self) -> str:
        return self._value


def test_public_robustness_reports_point_metrics_and_slices() -> None:
    result = PredictResult(
        y_pred=np.asarray([1.0, 2.0, 4.0, 8.0], dtype=float),
        metadata={"row_metadata": [{"instrument": "a"}, {"instrument": "a"}, {"instrument": "b"}, {"instrument": "b"}]},
        sample_indices=np.asarray(["s1", "s2", "s3", "s4"], dtype=object),
    )

    report = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        slice_by=["instrument"],
        seed=123,
    )

    scenario = report.scenarios[0]
    assert isinstance(report, nirs4all.RobustnessReport)
    assert report.mode == "clean_frozen"
    assert report.metadata["audit_only"] is True
    assert report.metadata["sample_ids"] == ["s1", "s2", "s3", "s4"]
    assert scenario.scenario == {"kind": "observed", "severity": 0.0}
    assert scenario.metrics.n_samples == 4
    assert scenario.metrics.rmse == pytest.approx(np.sqrt(0.75))
    assert scenario.metrics.mae == pytest.approx(0.75)
    assert scenario.metrics.bias == pytest.approx(-0.25)
    assert scenario.metrics.max_abs_error == pytest.approx(1.0)
    assert [slice_result.slice_key for slice_result in scenario.slices] == [{"instrument": "a"}, {"instrument": "b"}]
    assert scenario.slices[0].metrics.n_samples == 2
    assert scenario.slices[0].metrics.bias == pytest.approx(-0.5)
    assert report.to_dict()["fingerprint"] == report.fingerprint


def test_public_robustness_supports_prediction_bias_scenario_without_replay() -> None:
    result = PredictResult(
        y_pred=np.asarray([1.0, 2.0, 4.0, 8.0], dtype=float),
        metadata={"row_metadata": [{"instrument": "a"}, {"instrument": "a"}, {"instrument": "b"}, {"instrument": "b"}]},
    )

    report = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        scenarios=[
            {"kind": "observed", "severity": 0.0},
            {"kind": "prediction_bias", "severity": 0.5},
        ],
        slice_by=["instrument"],
    )

    observed, biased = report.scenarios
    assert report.metadata["supported_scenario_kinds"] == ["observed", "prediction_bias", "prediction_noise", "spectral_noise", "spectral_offset", "spectral_scale", "spectral_slope", "spectral_shift"]
    assert observed.metrics.bias == pytest.approx(-0.25)
    assert "spectral_replay" not in report.metadata
    assert biased.scenario == {"kind": "prediction_bias", "severity": 0.5}
    assert biased.metrics.rmse == pytest.approx(np.sqrt(0.75))
    assert biased.metrics.mae == pytest.approx(0.75)
    assert biased.metrics.bias == pytest.approx(0.25)
    assert biased.metrics.max_abs_error == pytest.approx(1.5)
    assert [slice_result.metrics.bias for slice_result in biased.slices] == pytest.approx([0.0, 0.5])


def test_public_robustness_accepts_typed_scenarios() -> None:
    result = PredictResult(
        y_pred=np.asarray([1.0, 2.0, 4.0, 8.0], dtype=float),
        metadata={"row_metadata": [{"instrument": "a"}, {"instrument": "a"}, {"instrument": "b"}, {"instrument": "b"}]},
    )

    report = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        scenarios=[
            nirs4all.RobustnessScenarioSpec(kind="observed"),
            nirs4all.RobustnessScenarioSpec(kind="prediction_bias", severity=0.5),
        ],
        slice_by=["instrument"],
    )

    observed, biased = report.scenarios
    assert observed.scenario == {"kind": "observed", "severity": 0.0}
    assert biased.scenario == {"kind": "prediction_bias", "severity": 0.5}
    assert biased.metrics.bias == pytest.approx(0.25)


def test_public_robustness_scenario_kinds_constant_matches_report_metadata_and_registry() -> None:
    result = PredictResult(y_pred=np.asarray([1.0, 2.0], dtype=float))

    report = nirs4all.robustness(result, y_true=[1.0, 2.0])
    registry = nirs4all.get_keyword_registry()
    entries = {entry["id"]: entry for entry in registry["entries"]}
    distribution_condition = entries["robustness.scenarios"]["value_schema"]["items"]["allOf"][0]["if"]["properties"]["kind"]["enum"]

    assert nirs4all.ROBUSTNESS_SCENARIO_KINDS == (
        "observed",
        "prediction_bias",
        "prediction_noise",
        "spectral_noise",
        "spectral_offset",
        "spectral_scale",
        "spectral_slope",
        "spectral_shift",
    )
    assert nirs4all.ROBUSTNESS_STOCHASTIC_SCENARIO_KINDS == ("prediction_noise", "spectral_noise")
    assert nirs4all.ROBUSTNESS_SCENARIO_DISTRIBUTIONS == ("normal", "uniform")
    assert report.metadata["supported_scenario_kinds"] == list(nirs4all.ROBUSTNESS_SCENARIO_KINDS)
    assert entries["robustness.scenarios.kind"]["value_schema"]["enum"] == list(nirs4all.ROBUSTNESS_SCENARIO_KINDS)
    assert entries["robustness.scenarios.distribution"]["value_schema"]["enum"] == list(nirs4all.ROBUSTNESS_SCENARIO_DISTRIBUTIONS)
    assert distribution_condition == list(nirs4all.ROBUSTNESS_STOCHASTIC_SCENARIO_KINDS)


def test_public_robustness_mode_constants_match_registry_and_runtime() -> None:
    result = PredictResult(y_pred=np.asarray([1.0, 2.0], dtype=float))
    registry = nirs4all.get_keyword_registry()
    entries = {entry["id"]: entry for entry in registry["entries"]}
    mode_schema = entries["robustness.mode"]["value_schema"]

    assert nirs4all.ROBUSTNESS_MODES == (
        "clean_frozen",
        "matched_recalibration",
        "structural_refit",
    )
    assert nirs4all.ROBUSTNESS_EXECUTABLE_MODES == ("clean_frozen",)
    assert mode_schema["enum"] == list(nirs4all.ROBUSTNESS_MODES)
    assert mode_schema["x-executable-values"] == list(nirs4all.ROBUSTNESS_EXECUTABLE_MODES)

    assert nirs4all.robustness(result, y_true=[1.0, 2.0], mode=nirs4all.ROBUSTNESS_EXECUTABLE_MODES[0]).mode == "clean_frozen"
    with pytest.raises(NotImplementedError, match="clean_frozen"):
        nirs4all.robustness(result, y_true=[1.0, 2.0], mode="matched_recalibration")
    with pytest.raises(ValueError, match="robustness.mode must be one of"):
        nirs4all.robustness(result, y_true=[1.0, 2.0], mode="unknown")


def test_robustness_scenario_spec_serialization_and_validation() -> None:
    assert nirs4all.RobustnessScenarioSpec(
        kind="spectral_offset",
        severity=-0.05,
    ).to_dict() == {
        "kind": "spectral_offset",
        "severity": -0.05,
    }
    assert nirs4all.RobustnessScenarioSpec(
        kind="spectral_scale",
        severity=0.10,
    ).to_dict() == {
        "kind": "spectral_scale",
        "severity": 0.10,
    }
    assert nirs4all.RobustnessScenarioSpec(
        kind="spectral_slope",
        severity=0.20,
    ).to_dict() == {
        "kind": "spectral_slope",
        "severity": 0.20,
    }
    assert nirs4all.RobustnessScenarioSpec(
        kind="spectral_shift",
        severity=1.5,
    ).to_dict() == {
        "kind": "spectral_shift",
        "severity": 1.5,
    }
    assert nirs4all.RobustnessScenarioSpec(
        kind="prediction_noise",
        severity=0.25,
        distribution="normal",
        extra={"label": "seeded"},
    ).to_dict() == {
        "distribution": "normal",
        "kind": "prediction_noise",
        "label": "seeded",
        "severity": 0.25,
    }
    assert nirs4all.RobustnessScenarioSpec(
        kind="prediction_noise",
        severity=0.25,
        distribution="uniform",
    ).to_dict() == {
        "distribution": "uniform",
        "kind": "prediction_noise",
        "severity": 0.25,
    }
    assert nirs4all.RobustnessScenarioSpec(kind=" observed ", severity=0.0).to_dict() == {
        "kind": "observed",
        "severity": 0.0,
    }

    with pytest.raises(NotImplementedError, match="kind must be"):
        nirs4all.RobustnessScenarioSpec(kind="unknown").to_dict()
    with pytest.raises(ValueError, match="kind must be a non-empty string"):
        nirs4all.RobustnessScenarioSpec(kind=_StringifiesAs("observed"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="severity must be a real numeric scalar"):
        nirs4all.RobustnessScenarioSpec(kind="prediction_noise", severity=True)
    with pytest.raises(ValueError, match="severity must be a real numeric scalar"):
        nirs4all.RobustnessScenarioSpec(kind="prediction_noise", severity="1.0")
    with pytest.raises(NotImplementedError, match="observed robustness scenarios require severity=0.0"):
        nirs4all.RobustnessScenarioSpec(kind="observed", severity=0.1).to_dict()
    with pytest.raises(ValueError, match="prediction_noise severity must be non-negative"):
        nirs4all.RobustnessScenarioSpec(kind="prediction_noise", severity=-0.1).to_dict()
    with pytest.raises(ValueError, match="spectral_scale requires 1.0 \\+ severity to be positive"):
        nirs4all.RobustnessScenarioSpec(kind="spectral_scale", severity=-1.0).to_dict()
    with pytest.raises(NotImplementedError, match="distribution='normal' or 'uniform'"):
        nirs4all.RobustnessScenarioSpec(kind="prediction_noise", severity=0.1, distribution="laplace").to_dict()
    with pytest.raises(ValueError, match="distribution must be a non-empty string"):
        nirs4all.RobustnessScenarioSpec(
            kind="prediction_noise",
            severity=0.1,
            distribution=_StringifiesAs("normal"),  # type: ignore[arg-type]
        ).to_dict()
    with pytest.raises(ValueError, match="supported only for prediction_noise and spectral_noise"):
        nirs4all.RobustnessScenarioSpec(kind="spectral_offset", severity=0.1, distribution="normal").to_dict()
    with pytest.raises(ValueError, match="must not override"):
        nirs4all.RobustnessScenarioSpec(kind="prediction_bias", severity=0.1, extra={"kind": "observed"}).to_dict()
    with pytest.raises(ValueError, match="extra keys must be canonical non-empty strings"):
        nirs4all.RobustnessScenarioSpec(kind="observed", extra={1: "coerced"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="extra keys must be canonical non-empty strings"):
        nirs4all.RobustnessScenarioSpec(kind="observed", extra={" label ": "bad"})
    with pytest.raises(ValueError, match="extra keys must be canonical non-empty strings"):
        nirs4all.RobustnessScenarioSpec(kind="observed", extra={"label\x00": "bad"})
    with pytest.raises(ValueError, match="TCV1 JSON-native"):
        nirs4all.RobustnessScenarioSpec(kind="prediction_bias", severity=0.1, extra={"opaque": object()}).to_dict()


def test_raw_robustness_scenario_mappings_reject_coercive_payloads() -> None:
    assert _normalize_scenarios([{"kind": " observed "}]) == ({"kind": "observed", "severity": 0.0},)

    with pytest.raises(ValueError, match="kind must be a non-empty string"):
        _normalize_scenarios([{"kind": _StringifiesAs("observed")}])
    with pytest.raises(ValueError, match="keys must be canonical non-empty strings"):
        _normalize_scenarios([{"kind": "observed", " label ": "bad"}])
    with pytest.raises(ValueError, match="keys must be canonical non-empty strings"):
        _normalize_scenarios([{"kind": "observed", "label\x00": "bad"}])
    with pytest.raises(ValueError, match="keys must be canonical non-empty strings"):
        _normalize_scenarios([{"kind": "observed", 1: "coerced"}])  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="severity must be a real numeric scalar"):
        _normalize_scenarios([{"kind": "prediction_noise", "severity": True}])
    with pytest.raises(ValueError, match="severity must be a real numeric scalar"):
        _normalize_scenarios([{"kind": "prediction_noise", "severity": "1.0"}])
    with pytest.raises(ValueError, match="distribution must be a non-empty string"):
        _normalize_scenarios(
            [
                {
                    "kind": "prediction_noise",
                    "severity": 0.1,
                    "distribution": _StringifiesAs("normal"),
                }
            ]
        )


def test_public_robustness_reports_degradation_and_worst_slices() -> None:
    result = PredictResult(
        y_pred=np.asarray([1.0, 2.0, 4.0, 8.0], dtype=float),
        metadata={"row_metadata": [{"instrument": "a"}, {"instrument": "a"}, {"instrument": "b"}, {"instrument": "b"}]},
    )

    report = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        scenarios=[
            {"kind": "observed", "severity": 0.0},
            {"kind": "prediction_bias", "severity": 0.5},
        ],
        slice_by=["instrument"],
    )

    baseline, biased = report.degradation_rows()
    worst = report.worst_slices(metric="rmse", top_k=2)
    summary = report.summary_rows()

    assert baseline["scenario_index"] == 0
    assert baseline["delta_rmse"] == pytest.approx(0.0)
    assert baseline["rmse_ratio"] == pytest.approx(1.0)
    assert biased["scenario"] == {"kind": "prediction_bias", "severity": 0.5}
    assert biased["delta_bias"] == pytest.approx(0.5)
    assert biased["delta_max_abs_error"] == pytest.approx(0.5)
    assert worst[0]["scenario"] == {"kind": "prediction_bias", "severity": 0.5}
    assert worst[0]["slice_key"] == {"instrument": "b"}
    assert worst[0]["value"] == pytest.approx(np.sqrt(1.25))
    assert summary[0]["scenario_label"] == "observed"
    assert summary[0]["execution_scope"] == "baseline"
    assert summary[0]["requires_spectral_replay"] is False
    assert summary[0]["delta_rmse"] == pytest.approx(0.0)
    assert summary[0]["worst_slice_label"] == "instrument=b"
    assert summary[1]["scenario"] == {"kind": "prediction_bias", "severity": 0.5}
    assert summary[1]["execution_scope"] == "prediction_replay"
    assert summary[1]["requires_spectral_replay"] is False
    assert summary[1]["rmse_ratio"] == pytest.approx(1.0)
    assert summary[1]["worst_slice_value"] == pytest.approx(np.sqrt(1.25))

    with pytest.raises(ValueError, match="reference scenario index"):
        report.degradation_rows(reference=99)
    with pytest.raises(ValueError, match="top_k"):
        report.worst_slices(top_k=0)
    with pytest.raises(ValueError, match="metric must be"):
        report.worst_slices(metric="unknown")
    with pytest.raises(ValueError, match="worst_slice_metric must be"):
        report.summary_rows(worst_slice_metric="unknown")


def test_public_robustness_prediction_noise_is_seeded_and_reproducible() -> None:
    result = PredictResult(
        y_pred=np.asarray([1.0, 2.0, 4.0, 8.0], dtype=float),
        metadata={"row_metadata": [{"instrument": "a"}, {"instrument": "a"}, {"instrument": "b"}, {"instrument": "b"}]},
    )
    scenario = {"kind": "prediction_noise", "severity": 0.25}

    report_a = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        scenarios=[scenario],
        slice_by=["instrument"],
        seed=42,
    )
    report_b = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        scenarios=[scenario],
        slice_by=["instrument"],
        seed=42,
    )
    report_c = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        scenarios=[scenario],
        slice_by=["instrument"],
        seed=43,
    )
    report_default = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        scenarios=[scenario],
    )

    assert report_a.to_dict() == report_b.to_dict()
    assert report_a.fingerprint != report_c.fingerprint
    assert report_a.scenarios[0].metrics.rmse != pytest.approx(report_c.scenarios[0].metrics.rmse)
    assert report_default.metadata["seed"] is None
    assert report_default.metadata["effective_seed"] == 0
    assert (
        report_default.to_dict()
        == nirs4all.robustness(
            result,
            y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
            scenarios=[scenario],
        ).to_dict()
    )


def test_public_robustness_prediction_noise_uniform_distribution_is_seeded() -> None:
    result = PredictResult(
        y_pred=np.asarray([1.0, 2.0, 4.0, 8.0], dtype=float),
        metadata={"row_metadata": [{"instrument": "a"}, {"instrument": "a"}, {"instrument": "b"}, {"instrument": "b"}]},
    )
    scenario = {"kind": "prediction_noise", "severity": 0.25, "distribution": "uniform"}

    report_a = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        scenarios=[scenario],
        seed=42,
    )
    report_b = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        scenarios=[scenario],
        seed=42,
    )
    normal_report = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        scenarios=[{"kind": "prediction_noise", "severity": 0.25, "distribution": "normal"}],
        seed=42,
    )

    assert report_a.to_dict() == report_b.to_dict()
    assert report_a.scenarios[0].scenario == scenario
    assert report_a.scenarios[0].metrics.rmse != pytest.approx(normal_report.scenarios[0].metrics.rmse)


def test_public_robustness_reports_conformal_metrics_without_renewing_guarantee() -> None:
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[1.0, 2.0, 3.0, 4.0],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        as_predict_result=True,
    )

    report = nirs4all.robustness(
        calibrated,
        y_true=[10.0, 19.0],
        metadata={"batch": ["b1", "b2"]},
        slice_by=["batch"],
    )

    scenario = report.scenarios[0]
    assert scenario.conformal_metrics[0.8].observed_coverage == pytest.approx(0.5)
    assert scenario.conformal_metrics[0.8].n_covered == 1
    assert scenario.conformal_metrics[0.8].n_missed_below == 1
    assert report.metadata["conformal_guarantee_status"]["status"] == "active"
    assert report.metadata["conformal_guarantee_status"]["limitations"]
    assert scenario.slices[0].conformal_metrics[0.8].n_samples == 1


def test_robustness_exports_fail_loud_conformal_guarantee_details() -> None:
    result = PredictResult(
        y_pred=np.asarray([10.0, 20.0], dtype=float),
        metadata={
            "conformal_guarantee_status": {
                "calibrated_coverages": [0.8, 0.9],
                "coverage": [0.8],
                "effective_engine": "nirs4all.python.replayed_array_apply",
                "invalidation_reasons": ["predictor fingerprint changed"],
                "limitations": ["finite-sample marginal coverage requires exchangeability"],
                "method": "split_absolute_residual",
                "requested_engine": "nirs4all.conformal.v1",
                "scope": "finite_sample_marginal_exchangeability",
                "status": "invalidated",
                "unit": "physical_sample",
            }
        },
    )

    report = nirs4all.robustness(result, y_true=[10.0, 19.0])
    markdown = report.to_markdown()
    html = report.to_html()

    assert "- Status: `invalidated`" in markdown
    assert "- Requested engine: `nirs4all.conformal.v1`" in markdown
    assert "- Effective engine: `nirs4all.python.replayed_array_apply`" in markdown
    assert "- Method: `split_absolute_residual`" in markdown
    assert "- Unit: `physical_sample`" in markdown
    assert "- Selected coverage: `0.8`" in markdown
    assert "- Calibrated coverage: `0.8, 0.9`" in markdown
    assert "Invalidation reasons:" in markdown
    assert "predictor fingerprint changed" in markdown
    assert "Effective engine: <code>nirs4all.python.replayed_array_apply</code>" in html
    assert "Selected coverage: <code>0.8</code>" in html
    assert "Invalidation reasons:" in html


def test_public_robustness_prediction_bias_shifts_materialized_intervals() -> None:
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[1.0, 2.0, 3.0, 4.0],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        as_predict_result=True,
    )

    report = nirs4all.robustness(
        calibrated,
        y_true=[10.0, 20.0],
        scenarios=[{"kind": "prediction_bias", "severity": 1.0}],
    )

    metric = report.scenarios[0].conformal_metrics[0.8]
    assert metric.observed_coverage == pytest.approx(0.0)
    assert metric.n_covered == 0
    assert metric.n_missed_below == 2
    assert metric.n_missed_above == 0


def test_public_robustness_prediction_noise_shifts_materialized_intervals() -> None:
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[1.0, 2.0, 3.0, 4.0],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        as_predict_result=True,
    )

    report = nirs4all.robustness(
        calibrated,
        y_true=[10.0, 20.0],
        scenarios=[{"kind": "prediction_noise", "severity": 0.75}],
        seed=11,
    )
    repeat = nirs4all.robustness(
        calibrated,
        y_true=[10.0, 20.0],
        scenarios=[{"kind": "prediction_noise", "severity": 0.75}],
        seed=11,
    )

    metric = report.scenarios[0].conformal_metrics[0.8]
    assert report.to_dict() == repeat.to_dict()
    assert metric.n_samples == 2
    assert metric.n_covered + metric.n_missed_below + metric.n_missed_above == 2


def test_public_robustness_spectral_noise_replays_explicit_predictor() -> None:
    X = np.asarray([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0], [8.0, 80.0]], dtype=float)
    result = PredictResult(
        y_pred=np.asarray([1.0, 2.0, 4.0, 8.0], dtype=float),
        metadata={"row_metadata": [{"instrument": "a"}, {"instrument": "a"}, {"instrument": "b"}, {"instrument": "b"}]},
    )
    scenario = {"kind": "spectral_noise", "severity": 0.2}

    report = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        X=X,
        predictor=FirstColumnPredictor(),
        scenarios=[scenario],
        slice_by=["instrument"],
        seed=123,
    )
    repeat = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        X=X,
        predictor=FirstColumnPredictor(),
        scenarios=[scenario],
        slice_by=["instrument"],
        seed=123,
    )
    different_seed = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        X=X,
        predictor=FirstColumnPredictor(),
        scenarios=[scenario],
        seed=124,
    )

    assert report.to_dict() == repeat.to_dict()
    assert report.fingerprint != different_seed.fingerprint
    assert report.metadata["spectral_replay"] == {
        "route": "predictor.predict_or_callable",
        "sample_ids_forwarded": False,
        "source": "predictor",
    }
    assert report.scenarios[0].metrics.n_samples == 4
    assert len(report.scenarios[0].slices) == 2


def test_public_robustness_spectral_noise_uniform_distribution_replays_predictor() -> None:
    X = np.asarray([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0], [8.0, 80.0]], dtype=float)
    result = PredictResult(
        y_pred=np.asarray([1.0, 2.0, 4.0, 8.0], dtype=float),
        metadata={"row_metadata": [{"instrument": "a"}, {"instrument": "a"}, {"instrument": "b"}, {"instrument": "b"}]},
    )
    scenario = {"kind": "spectral_noise", "severity": 0.2, "distribution": "uniform"}

    report = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        X=X,
        predictor=FirstColumnPredictor(),
        scenarios=[scenario],
        seed=123,
    )
    repeat = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        X=X,
        predictor=FirstColumnPredictor(),
        scenarios=[scenario],
        seed=123,
    )
    normal_report = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        X=X,
        predictor=FirstColumnPredictor(),
        scenarios=[{"kind": "spectral_noise", "severity": 0.2, "distribution": "normal"}],
        seed=123,
    )

    assert report.to_dict() == repeat.to_dict()
    assert report.scenarios[0].scenario == scenario
    assert report.scenarios[0].metrics.rmse != pytest.approx(normal_report.scenarios[0].metrics.rmse)


def test_public_robustness_spectral_offset_replays_explicit_predictor_deterministically() -> None:
    X = np.asarray([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0], [8.0, 80.0]], dtype=float)
    result = PredictResult(
        y_pred=np.asarray([1.0, 2.0, 4.0, 8.0], dtype=float),
        metadata={"row_metadata": [{"instrument": "a"}, {"instrument": "a"}, {"instrument": "b"}, {"instrument": "b"}]},
    )

    report = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        X=X,
        predictor=FirstColumnPredictor(),
        scenarios=[{"kind": "spectral_offset", "severity": 0.5}],
        slice_by=["instrument"],
        seed=123,
    )
    repeat_with_different_seed = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        X=X,
        predictor=FirstColumnPredictor(),
        scenarios=[{"kind": "spectral_offset", "severity": 0.5}],
        slice_by=["instrument"],
        seed=999,
    )

    shifted = report.scenarios[0]
    assert shifted.scenario == {"kind": "spectral_offset", "severity": 0.5}
    assert shifted.metrics.bias == pytest.approx(0.25)
    assert shifted.metrics.max_abs_error == pytest.approx(1.5)
    assert [slice_result.metrics.bias for slice_result in shifted.slices] == pytest.approx([0.0, 0.5])
    assert report.to_dict() != repeat_with_different_seed.to_dict()
    assert shifted.metrics.to_dict() == repeat_with_different_seed.scenarios[0].metrics.to_dict()


def test_public_robustness_spectral_offset_replays_predictor_bundle(tmp_path, monkeypatch) -> None:
    import jsonschema

    predict_module = importlib.import_module("nirs4all.api.predict")
    calls = []
    model_path = tmp_path / "model.n4a"
    X = np.asarray([[1.0, 10.0], [2.0, 20.0]], dtype=float)
    result = PredictResult(
        y_pred=np.asarray([1.0, 2.0], dtype=float),
        sample_indices=np.asarray(["p1", "p2"], dtype=object),
    )

    def _fake_predict(**kwargs):
        calls.append(kwargs)
        assert kwargs["model"] == model_path
        assert kwargs["all_predictions"] is False
        np.testing.assert_allclose(kwargs["data"]["X"], X + 0.5)
        assert kwargs["data"]["sample_ids"] == ["p1", "p2"]
        return PredictResult(y_pred=np.asarray([1.5, 2.5], dtype=float))

    monkeypatch.setattr(predict_module, "predict", _fake_predict)

    report = nirs4all.robustness(
        result,
        y_true=np.asarray([1.5, 2.5], dtype=float),
        X=X,
        predictor_bundle=model_path,
        scenarios=[{"kind": "spectral_offset", "severity": 0.5}],
    )

    assert len(calls) == 1
    assert report.metadata["spectral_replay"] == {
        "all_predictions": False,
        "predictor_bundle": str(model_path),
        "route": "nirs4all.predict",
        "sample_ids_forwarded": True,
        "source": "predictor_bundle",
    }
    assert report.summary_artifact()["spectral_replay"] == report.metadata["spectral_replay"]
    assert report.summary_rows()[0]["execution_scope"] == "spectral_replay"
    assert report.summary_rows()[0]["requires_spectral_replay"] is True
    assert report.tabular_records()["summary"][0]["execution_scope"] == "spectral_replay"
    assert report.tabular_records()["summary"][0]["requires_spectral_replay"] is True
    jsonschema.validate(report.summary_artifact(), nirs4all.get_robustness_summary_schema())
    assert report.scenarios[0].metrics.rmse == pytest.approx(0.0)
    assert report.scenarios[0].scenario == {"kind": "spectral_offset", "severity": 0.5}


def test_public_robustness_uses_published_predict_result_spectral_evidence(tmp_path, monkeypatch) -> None:
    predict_module = importlib.import_module("nirs4all.api.predict")
    calls = []
    model_path = tmp_path / "model.n4a"
    X = np.asarray([[1.0, 10.0], [2.0, 20.0]], dtype=float)
    result = PredictResult(
        y_pred=np.asarray([1.0, 2.0], dtype=float),
        metadata={
            "X": X,
            "robustness_evidence": {
                "X": "prediction_arrays.X",
                "predictor_bundle": str(model_path),
            },
        },
        sample_indices=np.asarray(["p1", "p2"], dtype=object),
    )

    def _fake_predict(**kwargs):
        calls.append(kwargs)
        assert kwargs["model"] == str(model_path)
        assert kwargs["all_predictions"] is False
        np.testing.assert_allclose(kwargs["data"]["X"], X + 0.5)
        assert kwargs["data"]["sample_ids"] == ["p1", "p2"]
        return PredictResult(y_pred=np.asarray([1.5, 2.5], dtype=float))

    monkeypatch.setattr(predict_module, "predict", _fake_predict)

    report = nirs4all.robustness(
        result,
        y_true=np.asarray([1.5, 2.5], dtype=float),
        scenarios=[{"kind": "spectral_offset", "severity": 0.5}],
    )

    assert len(calls) == 1
    assert report.metadata["spectral_replay"]["predictor_bundle"] == str(model_path)
    assert report.summary_rows()[0]["execution_scope"] == "spectral_replay"
    assert report.scenarios[0].metrics.rmse == pytest.approx(0.0)


def test_public_predict_result_robustness_uses_store_shaped_spectral_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    predict_module = importlib.import_module("nirs4all.api.predict")
    calls = []
    model_path = tmp_path / "model.n4a"
    X = np.asarray([[1.0, 10.0], [2.0, 20.0]], dtype=float)
    result = PredictResult(
        y_pred=np.asarray([1.0, 2.0], dtype=float),
        metadata={
            "spectra": X,
            "result_metadata": {
                "robustness_evidence": {
                    "spectra": "prediction_arrays.spectra",
                    "model_path": str(model_path),
                }
            },
        },
    )

    def _fake_predict(**kwargs):
        calls.append(kwargs)
        assert kwargs["model"] == str(model_path)
        np.testing.assert_allclose(kwargs["data"]["X"], X * 1.25)
        return PredictResult(y_pred=np.asarray([3.0, 4.0], dtype=float))

    monkeypatch.setattr(predict_module, "predict", _fake_predict)

    report = result.robustness(
        y_true=np.asarray([3.0, 4.0], dtype=float),
        scenarios=[{"kind": "spectral_scale", "severity": 0.25}],
    )

    assert len(calls) == 1
    assert report.metadata["spectral_replay"]["predictor_bundle"] == str(model_path)
    assert report.summary_rows()[0]["requires_spectral_replay"] is True
    assert report.scenarios[0].metrics.rmse == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("scenario", "expected_x"),
    [
        (
            {"kind": "spectral_noise", "severity": 0.0, "distribution": "normal"},
            np.asarray([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]], dtype=float),
        ),
        (
            {"kind": "spectral_scale", "severity": 0.25},
            np.asarray([[1.25, 12.5, 125.0], [2.5, 25.0, 250.0]], dtype=float),
        ),
        (
            {"kind": "spectral_slope", "severity": 1.0},
            np.asarray([[0.5, 10.0, 100.5], [1.5, 20.0, 200.5]], dtype=float),
        ),
        (
            {"kind": "spectral_shift", "severity": 0.5},
            np.asarray([[5.5, 55.0, 100.0], [11.0, 110.0, 200.0]], dtype=float),
        ),
    ],
)
def test_public_robustness_spectral_scenarios_replay_predictor_bundle(
    tmp_path,
    monkeypatch,
    scenario: dict[str, object],
    expected_x: np.ndarray,
) -> None:
    predict_module = importlib.import_module("nirs4all.api.predict")
    calls = []
    model_path = tmp_path / "spectral-model.n4a"
    X = np.asarray([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]], dtype=float)
    result = PredictResult(
        y_pred=np.asarray([1.0, 2.0], dtype=float),
        sample_indices=np.asarray(["p1", "p2"], dtype=object),
    )

    def _fake_predict(**kwargs):
        calls.append(kwargs)
        assert kwargs["model"] == model_path
        assert kwargs["all_predictions"] is False
        np.testing.assert_allclose(kwargs["data"]["X"], expected_x)
        assert kwargs["data"]["sample_ids"] == ["p1", "p2"]
        return PredictResult(y_pred=np.asarray([3.0, 4.0], dtype=float))

    monkeypatch.setattr(predict_module, "predict", _fake_predict)

    report = nirs4all.robustness(
        result,
        y_true=np.asarray([3.0, 4.0], dtype=float),
        X=X,
        predictor_bundle=model_path,
        scenarios=[scenario],
    )

    assert len(calls) == 1
    assert report.scenarios[0].metrics.rmse == pytest.approx(0.0)
    assert report.scenarios[0].scenario == scenario


def test_public_robustness_predictor_bundle_replay_requires_predict_result(tmp_path, monkeypatch) -> None:
    predict_module = importlib.import_module("nirs4all.api.predict")
    model_path = tmp_path / "bad-type-model.n4a"
    result = PredictResult(y_pred=np.asarray([1.0, 2.0], dtype=float))

    def _fake_predict(**_kwargs):
        return {"y_pred": [1.0, 2.0]}

    monkeypatch.setattr(predict_module, "predict", _fake_predict)

    with pytest.raises(TypeError, match="predictor_bundle replay must return a PredictResult"):
        nirs4all.robustness(
            result,
            y_true=np.asarray([1.0, 2.0], dtype=float),
            X=np.asarray([[1.0], [2.0]], dtype=float),
            predictor_bundle=model_path,
            scenarios=[{"kind": "spectral_offset", "severity": 0.0}],
        )


def test_public_robustness_predictor_bundle_replay_requires_matching_prediction_length(tmp_path, monkeypatch) -> None:
    predict_module = importlib.import_module("nirs4all.api.predict")
    model_path = tmp_path / "bad-length-model.n4a"
    result = PredictResult(y_pred=np.asarray([1.0, 2.0], dtype=float))

    def _fake_predict(**_kwargs):
        return PredictResult(y_pred=np.asarray([1.0], dtype=float))

    monkeypatch.setattr(predict_module, "predict", _fake_predict)

    with pytest.raises(ValueError, match="predictor_bundle predictions must have the same length"):
        nirs4all.robustness(
            result,
            y_true=np.asarray([1.0, 2.0], dtype=float),
            X=np.asarray([[1.0], [2.0]], dtype=float),
            predictor_bundle=model_path,
            scenarios=[{"kind": "spectral_offset", "severity": 0.0}],
        )


def test_public_robustness_spectral_scale_replays_explicit_predictor_deterministically() -> None:
    X = np.asarray([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0], [8.0, 80.0]], dtype=float)
    result = PredictResult(
        y_pred=np.asarray([1.0, 2.0, 4.0, 8.0], dtype=float),
        metadata={"row_metadata": [{"instrument": "a"}, {"instrument": "a"}, {"instrument": "b"}, {"instrument": "b"}]},
    )

    report = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        X=X,
        predictor=FirstColumnPredictor(),
        scenarios=[{"kind": "spectral_scale", "severity": 0.25}],
        slice_by=["instrument"],
        seed=123,
    )
    repeat_with_different_seed = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        X=X,
        predictor=FirstColumnPredictor(),
        scenarios=[{"kind": "spectral_scale", "severity": 0.25}],
        slice_by=["instrument"],
        seed=999,
    )

    shifted = report.scenarios[0]
    assert shifted.scenario == {"kind": "spectral_scale", "severity": 0.25}
    assert shifted.metrics.bias == pytest.approx(0.6875)
    assert shifted.metrics.max_abs_error == pytest.approx(3.0)
    assert [slice_result.metrics.bias for slice_result in shifted.slices] == pytest.approx([-0.125, 1.5])
    assert report.to_dict() != repeat_with_different_seed.to_dict()
    assert shifted.metrics.to_dict() == repeat_with_different_seed.scenarios[0].metrics.to_dict()


def test_public_robustness_spectral_slope_replays_explicit_predictor_deterministically() -> None:
    X = np.asarray([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0], [8.0, 80.0]], dtype=float)
    result = PredictResult(
        y_pred=np.asarray([1.0, 2.0, 4.0, 8.0], dtype=float),
        metadata={"row_metadata": [{"instrument": "a"}, {"instrument": "a"}, {"instrument": "b"}, {"instrument": "b"}]},
    )

    report = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        X=X,
        predictor=FirstColumnPredictor(),
        scenarios=[{"kind": "spectral_slope", "severity": 1.0}],
        slice_by=["instrument"],
        seed=123,
    )
    repeat_with_different_seed = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        X=X,
        predictor=FirstColumnPredictor(),
        scenarios=[{"kind": "spectral_slope", "severity": 1.0}],
        slice_by=["instrument"],
        seed=999,
    )

    shifted = report.scenarios[0]
    assert shifted.scenario == {"kind": "spectral_slope", "severity": 1.0}
    assert shifted.metrics.bias == pytest.approx(-0.75)
    assert shifted.metrics.max_abs_error == pytest.approx(1.5)
    assert [slice_result.metrics.bias for slice_result in shifted.slices] == pytest.approx([-1.0, -0.5])
    assert report.to_dict() != repeat_with_different_seed.to_dict()
    assert shifted.metrics.to_dict() == repeat_with_different_seed.scenarios[0].metrics.to_dict()


def test_public_robustness_spectral_shift_replays_explicit_predictor_deterministically() -> None:
    X = np.asarray([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0], [8.0, 80.0]], dtype=float)
    result = PredictResult(
        y_pred=np.asarray([1.0, 2.0, 4.0, 8.0], dtype=float),
        metadata={"row_metadata": [{"instrument": "a"}, {"instrument": "a"}, {"instrument": "b"}, {"instrument": "b"}]},
    )

    report = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        X=X,
        predictor=FirstColumnPredictor(),
        scenarios=[{"kind": "spectral_shift", "severity": 0.5}],
        slice_by=["instrument"],
        seed=123,
    )
    repeat_with_different_seed = nirs4all.robustness(
        result,
        y_true=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
        X=X,
        predictor=FirstColumnPredictor(),
        scenarios=[{"kind": "spectral_shift", "severity": 0.5}],
        slice_by=["instrument"],
        seed=999,
    )

    shifted = report.scenarios[0]
    assert shifted.scenario == {"kind": "spectral_shift", "severity": 0.5}
    assert shifted.metrics.bias == pytest.approx(16.625)
    assert shifted.metrics.max_abs_error == pytest.approx(37.0)
    assert [slice_result.metrics.bias for slice_result in shifted.slices] == pytest.approx([6.25, 27.0])
    assert report.to_dict() != repeat_with_different_seed.to_dict()
    assert shifted.metrics.to_dict() == repeat_with_different_seed.scenarios[0].metrics.to_dict()


def test_public_robustness_spectral_noise_recenters_materialized_intervals() -> None:
    X = np.asarray([[10.0, 1.0], [20.0, 2.0]], dtype=float)
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[1.0, 2.0, 3.0, 4.0],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        as_predict_result=True,
    )

    report = nirs4all.robustness(
        calibrated,
        y_true=[10.0, 20.0],
        X=X,
        predictor=FirstColumnPredictor(),
        scenarios=[{"kind": "spectral_noise", "severity": 0.0}],
        seed=1,
    )

    metric = report.scenarios[0].conformal_metrics[0.8]
    assert metric.observed_coverage == pytest.approx(1.0)
    assert metric.n_covered == 2


def test_public_robustness_spectral_offset_recenters_materialized_intervals() -> None:
    X = np.asarray([[10.0, 1.0], [20.0, 2.0]], dtype=float)
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[1.0, 2.0, 3.0, 4.0],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        as_predict_result=True,
    )

    report = nirs4all.robustness(
        calibrated,
        y_true=[10.25, 20.25],
        X=X,
        predictor=FirstColumnPredictor(),
        scenarios=[{"kind": "spectral_offset", "severity": 0.25}],
    )

    metric = report.scenarios[0].conformal_metrics[0.8]
    assert report.scenarios[0].metrics.rmse == pytest.approx(0.0)
    assert metric.observed_coverage == pytest.approx(1.0)
    assert metric.n_covered == 2


def test_public_robustness_spectral_scale_recenters_materialized_intervals() -> None:
    X = np.asarray([[10.0, 1.0], [20.0, 2.0]], dtype=float)
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[1.0, 2.0, 3.0, 4.0],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        as_predict_result=True,
    )

    report = nirs4all.robustness(
        calibrated,
        y_true=[11.0, 22.0],
        X=X,
        predictor=FirstColumnPredictor(),
        scenarios=[{"kind": "spectral_scale", "severity": 0.1}],
    )

    metric = report.scenarios[0].conformal_metrics[0.8]
    assert report.scenarios[0].metrics.rmse == pytest.approx(0.0)
    assert metric.observed_coverage == pytest.approx(1.0)
    assert metric.n_covered == 2


def test_public_robustness_spectral_slope_recenters_materialized_intervals() -> None:
    X = np.asarray([[10.0, 1.0], [20.0, 2.0]], dtype=float)
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[1.0, 2.0, 3.0, 4.0],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        as_predict_result=True,
    )

    report = nirs4all.robustness(
        calibrated,
        y_true=[9.0, 19.0],
        X=X,
        predictor=FirstColumnPredictor(),
        scenarios=[{"kind": "spectral_slope", "severity": 2.0}],
    )

    metric = report.scenarios[0].conformal_metrics[0.8]
    assert report.scenarios[0].metrics.rmse == pytest.approx(0.0)
    assert metric.observed_coverage == pytest.approx(1.0)
    assert metric.n_covered == 2


def test_public_robustness_spectral_shift_recenters_materialized_intervals() -> None:
    X = np.asarray([[10.0, 20.0], [20.0, 40.0]], dtype=float)
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[1.0, 2.0, 3.0, 4.0],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        as_predict_result=True,
    )

    report = nirs4all.robustness(
        calibrated,
        y_true=[20.0, 40.0],
        X=X,
        predictor=FirstColumnPredictor(),
        scenarios=[{"kind": "spectral_shift", "severity": 1.0}],
    )

    metric = report.scenarios[0].conformal_metrics[0.8]
    assert report.scenarios[0].metrics.rmse == pytest.approx(0.0)
    assert metric.observed_coverage == pytest.approx(1.0)
    assert metric.n_covered == 2


def test_public_robustness_spectral_offset_requires_replay_inputs() -> None:
    result = PredictResult(y_pred=np.asarray([1.0, 2.0], dtype=float))

    with pytest.raises(ValueError, match="either predictor or predictor_bundle"):
        nirs4all.robustness(
            result,
            y_true=np.asarray([1.0, 2.0], dtype=float),
            X=np.asarray([[1.0], [2.0]], dtype=float),
            predictor=FirstColumnPredictor(),
            predictor_bundle="model.n4a",
            scenarios=[{"kind": "spectral_offset", "severity": 0.1}],
        )

    with pytest.raises(ValueError, match="spectral_offset requires explicit X and predictor"):
        nirs4all.robustness(
            result,
            y_true=np.asarray([1.0, 2.0], dtype=float),
            scenarios=[{"kind": "spectral_offset", "severity": 0.1}],
        )

    with pytest.raises(ValueError, match="spectral_scale requires explicit X and predictor"):
        nirs4all.robustness(
            result,
            y_true=np.asarray([1.0, 2.0], dtype=float),
            scenarios=[{"kind": "spectral_scale", "severity": 0.1}],
        )

    with pytest.raises(ValueError, match="spectral_slope requires explicit X and predictor"):
        nirs4all.robustness(
            result,
            y_true=np.asarray([1.0, 2.0], dtype=float),
            scenarios=[{"kind": "spectral_slope", "severity": 0.1}],
        )

    with pytest.raises(ValueError, match="spectral_shift requires explicit X and predictor"):
        nirs4all.robustness(
            result,
            y_true=np.asarray([1.0, 2.0], dtype=float),
            scenarios=[{"kind": "spectral_shift", "severity": 0.1}],
        )

    with pytest.raises(ValueError, match="spectral_scale requires 1.0 \\+ severity to be positive"):
        nirs4all.robustness(
            result,
            y_true=np.asarray([1.0, 2.0], dtype=float),
            X=np.asarray([[1.0], [2.0]], dtype=float),
            predictor=FirstColumnPredictor(),
            scenarios=[{"kind": "spectral_scale", "severity": -1.0}],
        )


def test_public_robustness_report_roundtrips_json_with_fingerprint_verification(tmp_path) -> None:
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[1.0, 2.0, 3.0, 4.0],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        as_predict_result=True,
    )
    report = nirs4all.robustness(
        calibrated,
        y_true=[10.0, 19.0],
        metadata={"batch": ["b1", "b2"]},
        slice_by=["batch"],
    )
    path = report.save_json(tmp_path / "robustness.json")

    restored = nirs4all.RobustnessReport.load_json(path)
    compact = nirs4all.RobustnessReport.from_json(report.to_json(indent=None))

    assert restored.to_dict() == report.to_dict()
    assert compact.to_dict() == report.to_dict()
    assert path.read_text(encoding="utf-8").endswith("\n")

    corrupted = json.loads(path.read_text(encoding="utf-8"))
    corrupted["scenarios"][0]["metrics"]["rmse"] = 123.0
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        nirs4all.RobustnessReport.from_dict(corrupted)


def test_public_robustness_report_persists_in_workspace(tmp_path) -> None:
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[1.0, 2.0, 3.0, 4.0],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        as_predict_result=True,
    )
    report = nirs4all.robustness(
        calibrated,
        y_true=[10.0, 19.0],
        metadata={"batch": ["b1", "b2"]},
        slice_by=["batch"],
    )

    robustness_id = nirs4all.save_workspace_robustness_report(
        tmp_path / "workspace",
        report,
        robustness_id="robustness-main",
        name="Main robustness audit",
        metadata={"source": "unit"},
    )
    restored = nirs4all.load_workspace_robustness_report(tmp_path / "workspace", robustness_id)
    restored_by_fingerprint = nirs4all.load_workspace_robustness_report(tmp_path / "workspace", report.fingerprint)

    assert robustness_id == "robustness-main"
    assert restored.to_dict() == report.to_dict()
    assert restored_by_fingerprint.to_dict() == report.to_dict()

    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    with WorkspaceStore(tmp_path / "workspace") as store:
        rows = store.list_robustness_results().to_dicts()

    assert rows[0]["robustness_id"] == "robustness-main"
    assert rows[0]["name"] == "Main robustness audit"
    assert rows[0]["mode"] == "clean_frozen"
    assert rows[0]["scenario_count"] == 1
    assert rows[0]["slice_by"] == '["batch"]'
    assert json.loads(rows[0]["metadata"]) == {"source": "unit"}

    direct = nirs4all.robustness(
        calibrated,
        y_true=[10.0, 19.0],
        metadata={"batch": ["b1", "b2"]},
        slice_by=["batch"],
        workspace_path=tmp_path / "workspace-direct",
        workspace_robustness_id="robustness-direct",
        workspace_name="Direct robustness audit",
        workspace_metadata={"source": "direct"},
    )
    restored_direct = nirs4all.load_workspace_robustness_report(tmp_path / "workspace-direct", "robustness-direct")

    assert restored_direct.to_dict() == direct.to_dict()
    assert direct.fingerprint == report.fingerprint


def test_public_robustness_workspace_metadata_is_strict_json_native(tmp_path) -> None:
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        as_predict_result=True,
    )
    report = nirs4all.robustness(
        calibrated,
        y_true=[10.0, 19.0],
        metadata={"batch": ["b1", "b2"]},
        slice_by=["batch"],
    )

    robustness_id = nirs4all.save_workspace_robustness_report(
        tmp_path / "workspace",
        report,
        robustness_id="robustness-json",
        metadata={"site": "north", "nested": {"ok": [1, True, None]}},
    )

    assert robustness_id == "robustness-json"

    for metadata, message in (
        ({" bad": 1}, "save_robustness_result.metadata keys must be canonical non-empty strings"),
        ({"bad": object()}, r"save_robustness_result.metadata\[bad\] must be JSON-native"),
        ({"bad": float("nan")}, r"save_robustness_result.metadata\[bad\] must be JSON-native and finite"),
        ({"bad": (1, 2)}, r"save_robustness_result.metadata\[bad\] must be JSON-native"),
    ):
        with pytest.raises(ValueError, match=message):
            nirs4all.save_workspace_robustness_report(tmp_path / "workspace", report, metadata=metadata)
        with pytest.raises(ValueError, match=message):
            nirs4all.robustness(
                calibrated,
                y_true=[10.0, 19.0],
                metadata={"batch": ["b1", "b2"]},
                slice_by=["batch"],
                workspace_path=tmp_path / "workspace-direct-invalid",
                workspace_metadata=metadata,
            )


def test_public_robustness_workspace_ids_are_canonical(tmp_path) -> None:
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        as_predict_result=True,
    )
    report = nirs4all.robustness(
        calibrated,
        y_true=[10.0, 19.0],
        metadata={"batch": ["b1", "b2"]},
        slice_by=["batch"],
    )

    generated = nirs4all.save_workspace_robustness_report(tmp_path / "workspace", report)
    assert isinstance(generated, str)
    assert generated

    for robustness_id in ("", " robustness-main ", "bad\x00id", 123):
        with pytest.raises(ValueError, match="save_robustness_result.robustness_id must be a canonical non-empty string"):
            nirs4all.save_workspace_robustness_report(
                tmp_path / "workspace",
                report,
                robustness_id=robustness_id,  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="save_robustness_result.robustness_id must be a canonical non-empty string"):
            nirs4all.robustness(
                calibrated,
                y_true=[10.0, 19.0],
                metadata={"batch": ["b1", "b2"]},
                slice_by=["batch"],
                workspace_path=tmp_path / "workspace-direct-invalid-id",
                workspace_robustness_id=robustness_id,  # type: ignore[arg-type]
            )


def test_public_robustness_workspace_link_ids_are_canonical(tmp_path) -> None:
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        as_predict_result=True,
    )
    report = nirs4all.robustness(
        calibrated,
        y_true=[10.0, 19.0],
        metadata={"batch": ["b1", "b2"]},
        slice_by=["batch"],
    )

    for kwargs, label in (
        ({"run_id": " run-main "}, "save_robustness_result.run_id"),
        ({"pipeline_id": "pipe\x00main"}, "save_robustness_result.pipeline_id"),
        ({"chain_id": 123}, "save_robustness_result.chain_id"),
        ({"conformal_id": ""}, "save_robustness_result.conformal_id"),
        ({"prediction_id": " pred-main"}, "save_robustness_result.prediction_id"),
    ):
        with pytest.raises(ValueError, match=f"{label} must be a canonical non-empty string"):
            nirs4all.save_workspace_robustness_report(
                tmp_path / "workspace",
                report,
                **kwargs,  # type: ignore[arg-type]
            )


def test_robustness_report_metadata_rejects_coercive_payloads() -> None:
    scenario = nirs4all.RobustnessScenarioResult(
        scenario={"kind": "observed", "severity": 0.0},
        severity=0.0,
        metrics=nirs4all.RobustnessMetricSet(
            n_samples=2,
            rmse=0.0,
            mae=0.0,
            bias=0.0,
            max_abs_error=0.0,
        ),
    )

    with pytest.raises(ValueError, match="RobustnessReport.metadata keys must be canonical non-empty strings"):
        nirs4all.RobustnessReport(
            mode="clean_frozen",
            scenarios=(scenario,),
            metadata={1: "coerced"},  # type: ignore[dict-item]
        )
    with pytest.raises(ValueError, match="RobustnessReport.metadata keys must be canonical non-empty strings"):
        nirs4all.RobustnessReport(
            mode="clean_frozen",
            scenarios=(scenario,),
            metadata={"label\x00": "bad"},
        )
    with pytest.raises(ValueError, match="RobustnessReport.metadata must be TCV1 JSON-native"):
        nirs4all.RobustnessReport(
            mode="clean_frozen",
            scenarios=(scenario,),
            metadata={"opaque": object()},
        )


def test_robustness_result_payload_mappings_reject_coercive_keys() -> None:
    metrics = nirs4all.RobustnessMetricSet(
        n_samples=2,
        rmse=0.0,
        mae=0.0,
        bias=0.0,
        max_abs_error=0.0,
    )

    scenario = nirs4all.RobustnessScenarioResult(
        scenario={"kind": "observed", "severity": 0.0},
        severity=np.float64(0.0),
        metrics=metrics,
        slices=(
            nirs4all.RobustnessSliceResult(
                slice_key={"instrument": "a"},
                metrics=metrics,
            ),
        ),
    )
    report = nirs4all.RobustnessReport(mode="clean_frozen", scenarios=(scenario,), slice_by=("instrument",))

    assert scenario.severity == 0.0
    assert scenario.to_dict()["scenario"] == {"kind": "observed", "severity": 0.0}
    assert scenario.slices[0].to_dict()["slice_key"] == {"instrument": "a"}
    assert report.degradation_rows()[0]["scenario"] == {"kind": "observed", "severity": 0.0}
    assert report.worst_slices()[0]["slice_key"] == {"instrument": "a"}
    assert report.summary_rows()[0]["worst_slice_key"] == {"instrument": "a"}
    assert report.tabular_records()["scenario_metrics"][0]["scenario_json"] == '{"kind":"observed","severity":0.0}'
    assert report.tabular_records()["slices"][0]["slice_key_json"] == '{"instrument":"a"}'

    with pytest.raises(ValueError, match="RobustnessScenarioResult.scenario keys must be canonical non-empty strings"):
        nirs4all.RobustnessScenarioResult(
            scenario={1: "coerced", "severity": 0.0},  # type: ignore[dict-item]
            severity=0.0,
            metrics=metrics,
        )
    with pytest.raises(ValueError, match="RobustnessScenarioResult.scenario keys must be canonical non-empty strings"):
        nirs4all.RobustnessScenarioResult(
            scenario={"kind": "observed", "label\x00": "bad"},
            severity=0.0,
            metrics=metrics,
        )
    with pytest.raises(ValueError, match="RobustnessScenarioResult.scenario must be TCV1 JSON-native"):
        nirs4all.RobustnessScenarioResult(
            scenario={"kind": "observed", "opaque": object()},
            severity=0.0,
            metrics=metrics,
        )
    with pytest.raises(ValueError, match="RobustnessScenarioResult.severity must be a real numeric scalar"):
        nirs4all.RobustnessScenarioResult(
            scenario={"kind": "observed", "severity": True},
            severity=True,
            metrics=metrics,
        )
    with pytest.raises(ValueError, match="RobustnessSliceResult.slice_key keys must be canonical non-empty strings"):
        nirs4all.RobustnessSliceResult(
            slice_key={1: "coerced"},  # type: ignore[dict-item]
            metrics=metrics,
        )
    with pytest.raises(ValueError, match="RobustnessSliceResult.slice_key keys must be canonical non-empty strings"):
        nirs4all.RobustnessSliceResult(
            slice_key={"instrument\x00": "bad"},
            metrics=metrics,
        )
    with pytest.raises(ValueError, match="RobustnessSliceResult.slice_key must be TCV1 JSON-native"):
        nirs4all.RobustnessSliceResult(
            slice_key={"opaque": object()},
            metrics=metrics,
        )

    scenario_payload = scenario.to_dict()
    scenario_payload["scenario"] = {"kind\x00": "observed", "severity": 0.0}
    with pytest.raises(ValueError, match="RobustnessScenarioResult.scenario keys must be canonical non-empty strings"):
        nirs4all.RobustnessScenarioResult.from_dict(scenario_payload)

    bool_severity_payload = scenario.to_dict()
    bool_severity_payload["severity"] = True
    with pytest.raises(ValueError, match="RobustnessScenarioResult.severity must be a real numeric scalar"):
        nirs4all.RobustnessScenarioResult.from_dict(bool_severity_payload)

    slice_payload = scenario.slices[0].to_dict()
    slice_payload["slice_key"] = {"instrument\x00": "bad"}
    with pytest.raises(ValueError, match="RobustnessSliceResult.slice_key keys must be canonical non-empty strings"):
        nirs4all.RobustnessSliceResult.from_dict(slice_payload)


def test_public_robustness_report_exports_deterministic_markdown(tmp_path) -> None:
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[1.0, 2.0, 3.0, 4.0],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        as_predict_result=True,
    )
    report = nirs4all.robustness(
        calibrated,
        y_true=[10.0, 19.0],
        metadata={"batch": ["b1", "b2"]},
        slice_by=["batch"],
        seed=7,
    )

    markdown = report.to_markdown()
    path = report.save_markdown(tmp_path / "robustness.md")

    assert markdown.startswith("# NIRS4All robustness report\n")
    assert path.read_text(encoding="utf-8") == markdown
    assert markdown.endswith("\n")
    assert f"- Fingerprint: `{report.fingerprint}`" in markdown
    assert "- Seed: `7`" in markdown
    assert "## Scenario summary" in markdown
    assert "| Scenario | Scope | Replay evidence | Severity | n | RMSE | ΔRMSE | RMSE ratio | Min coverage | Max abs coverage gap | Worst slice | Worst slice RMSE |" in markdown
    assert "| observed | baseline | not required | 0 | 2 | 0.707106781187 | 0 | 1 | 0.5 | 0.3 | batch=b2 | 1 |" in markdown
    assert "| Scenario | Severity | n | RMSE | MAE | Bias | Max abs error |" in markdown
    assert "## Conformal diagnostics" in markdown
    assert "## Diagnostic slices" in markdown
    assert "## Degradation versus reference scenario" in markdown
    assert "## Worst diagnostic slices" in markdown
    assert "## Slice conformal diagnostics" in markdown
    assert "| observed | 0 | batch=b1 | 1 | 0 | 0 | 0 | 0 |" in markdown
    assert report.to_markdown() == markdown


def test_public_robustness_report_exports_deterministic_html(tmp_path) -> None:
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[1.0, 2.0, 3.0, 4.0],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        as_predict_result=True,
    )
    report = nirs4all.robustness(
        calibrated,
        y_true=[10.0, 19.0],
        metadata={"batch": ["b&1", "b2"]},
        slice_by=["batch"],
        seed=7,
    )

    html = report.to_html()
    path = report.save_html(tmp_path / "robustness.html")

    assert html.startswith("<!doctype html>\n")
    assert path.read_text(encoding="utf-8") == html
    assert html.endswith("\n")
    assert f"<code>{report.fingerprint}</code>" in html
    assert "<h2>Scenario summary</h2>" in html
    assert "<th>Scope</th>" in html
    assert ">baseline</td>" in html
    assert "<th>Replay evidence</th>" in html
    assert ">not required</td>" in html
    assert "<th>Worst slice RMSE</th>" in html
    assert "<h2>Scenario metrics</h2>" in html
    assert "<h2>Conformal diagnostics</h2>" in html
    assert "<h2>Diagnostic slices</h2>" in html
    assert "<h2>Degradation versus reference scenario</h2>" in html
    assert "<h2>Worst diagnostic slices</h2>" in html
    assert "<h2>Slice conformal diagnostics</h2>" in html
    assert "batch=b&amp;1" in html
    assert report.to_html() == html


def test_public_robustness_report_exports_parquet_directory(tmp_path) -> None:
    import polars as pl

    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[1.0, 2.0, 3.0, 4.0],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        as_predict_result=True,
    )
    report = nirs4all.robustness(
        calibrated,
        y_true=[10.0, 19.0],
        metadata={"batch": ["b1", "b2"]},
        slice_by=["batch"],
        seed=7,
    )
    target = report.save_parquet(tmp_path / "robustness-report.parquet")

    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    scenario_df = pl.read_parquet(target / "scenario_metrics.parquet")
    summary_df = pl.read_parquet(target / "summary.parquet")
    degradation_df = pl.read_parquet(target / "degradation.parquet")
    conformal_df = pl.read_parquet(target / "conformal_diagnostics.parquet")
    slices_df = pl.read_parquet(target / "slices.parquet")
    restored = nirs4all.RobustnessReport.load_parquet(target)

    assert manifest["fingerprint"] == report.fingerprint
    assert manifest["format"] == "nirs4all.robustness.parquet-directory"
    assert manifest["report_json"] == "report.json"
    assert manifest["tables"]["scenario_metrics"] == "scenario_metrics.parquet"
    assert manifest["tables"]["summary"] == "summary.parquet"
    assert manifest["table_metadata"]["scenario_metrics"]["file"] == "scenario_metrics.parquet"
    assert manifest["table_metadata"]["scenario_metrics"]["rows"] == 1
    assert manifest["table_metadata"]["scenario_metrics"]["fingerprint"]
    assert (target / "report.json").read_text(encoding="utf-8") == report.to_json()
    assert restored.to_dict() == report.to_dict()
    assert scenario_df.height == 1
    assert scenario_df["scenario_label"].to_list() == ["observed"]
    assert summary_df["scenario_label"].to_list() == ["observed"]
    assert summary_df["execution_scope"].to_list() == ["baseline"]
    assert summary_df["requires_spectral_replay"].to_list() == [False]
    assert summary_df["conformal_min_observed_coverage"].to_list() == pytest.approx([0.5])
    assert degradation_df["delta_rmse"].to_list() == pytest.approx([0.0])
    assert conformal_df["coverage"].to_list() == pytest.approx([0.8])
    assert sorted(slices_df["slice_label"].to_list()) == ["batch=b1", "batch=b2"]

    corrupted_manifest = dict(manifest)
    corrupted_manifest["fingerprint"] = "bad"
    (target / "manifest.json").write_text(json.dumps(corrupted_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        nirs4all.RobustnessReport.load_parquet(target)

    (target / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    for mutate in (
        lambda payload: payload.__setitem__("report_json", "../report.json"),
        lambda payload: payload["tables"].__setitem__("summary", "../summary.parquet"),
        lambda payload: payload["table_metadata"]["summary"].__setitem__("file", "../summary.parquet"),
    ):
        escaping_manifest = json.loads(json.dumps(manifest))
        mutate(escaping_manifest)
        (target / "manifest.json").write_text(json.dumps(escaping_manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="relative child paths"):
            nirs4all.RobustnessReport.load_parquet(target)

    mismatch_manifest = json.loads(json.dumps(manifest))
    mismatch_manifest["table_metadata"]["summary"]["file"] = "scenario_metrics.parquet"
    (target / "manifest.json").write_text(json.dumps(mismatch_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="table_metadata and tables files must match"):
        nirs4all.RobustnessReport.load_parquet(target)

    (target / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    scenario_df.with_columns(pl.lit(999.0).alias("rmse")).write_parquet(target / "scenario_metrics.parquet")
    with pytest.raises(ValueError, match="table fingerprint mismatch"):
        nirs4all.RobustnessReport.load_parquet(target)

    existing_file = tmp_path / "not-a-directory.parquet"
    existing_file.write_text("not parquet", encoding="utf-8")
    with pytest.raises(ValueError, match="target must be a directory"):
        report.save_parquet(existing_file)


def test_public_robustness_report_exports_artifact_directory(tmp_path) -> None:
    import jsonschema

    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[1.0, 2.0, 3.0, 4.0],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        as_predict_result=True,
    )
    report = nirs4all.robustness(
        calibrated,
        y_true=[10.0, 19.0],
        metadata={"batch": ["b1", "b2"]},
        slice_by=["batch"],
        seed=7,
    )

    artifacts = report.save_artifacts(tmp_path / "robustness-artifacts")
    manifest = json.loads(artifacts["manifest"].read_text(encoding="utf-8"))
    summary = json.loads(artifacts["summary"].read_text(encoding="utf-8"))
    schema = nirs4all.get_robustness_summary_schema()

    assert set(artifacts) == {"html", "json", "manifest", "markdown", "parquet", "summary"}
    assert artifacts["json"].read_text(encoding="utf-8") == report.to_json()
    assert artifacts["summary"].read_text(encoding="utf-8") == report.to_summary_json()
    assert artifacts["markdown"].read_text(encoding="utf-8") == report.to_markdown()
    assert artifacts["html"].read_text(encoding="utf-8") == report.to_html()
    assert summary["format"] == "nirs4all.robustness.summary"
    assert summary["fingerprint"] == report.fingerprint
    assert summary["conformal_guarantee_status"]["status"] == "active"
    assert summary["conformal_guarantee_status"]["effective_engine"] == "nirs4all.python.replayed_array_calibration"
    assert summary["conformal_guarantee_status"]["coverage"] == [0.8]
    assert "spectral_replay" not in summary
    assert summary["summary"] == list(report.summary_rows())
    assert report.summary_artifact() == summary
    assert report.save_summary(tmp_path / "summary.json").read_text(encoding="utf-8") == report.to_summary_json()
    row_schema = schema["properties"]["summary"]["items"]["properties"]
    assert row_schema["execution_scope"]["enum"] == ["baseline", "prediction_replay", "spectral_replay"]
    assert row_schema["requires_spectral_replay"]["type"] == "boolean"
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(summary, schema)
    assert schema["properties"]["conformal_guarantee_status"]["properties"]["effective_engine"] == {"type": "string"}
    assert schema["properties"]["spectral_replay"]["required"] == ["route", "sample_ids_forwarded", "source"]
    assert json.loads(nirs4all.robustness_summary_schema_json(indent=None)) == schema
    assert nirs4all.RobustnessReport.load_json(artifacts["json"]).to_dict() == report.to_dict()
    assert nirs4all.RobustnessReport.load_parquet(artifacts["parquet"]).to_dict() == report.to_dict()
    assert nirs4all.RobustnessReport.load_artifacts(tmp_path / "robustness-artifacts").to_dict() == report.to_dict()
    assert manifest == {
        "files": {
            "html": "report.html",
            "json": "report.json",
            "markdown": "report.md",
            "parquet": "report.parquet",
            "summary": "summary.json",
        },
        "fingerprint": report.fingerprint,
        "format": "nirs4all.robustness.artifact-directory",
        "formats": ["json", "summary", "markdown", "html", "parquet"],
        "mode": "clean_frozen",
        "report_version": 1,
        "schema_version": 1,
    }

    subset = report.save_artifacts(tmp_path / "subset", formats=("json", "markdown"))
    subset_manifest = json.loads(subset["manifest"].read_text(encoding="utf-8"))
    assert set(subset) == {"json", "manifest", "markdown"}
    assert subset_manifest["formats"] == ["json", "markdown"]
    assert subset_manifest["files"] == {"json": "report.json", "markdown": "report.md"}
    assert not (tmp_path / "subset" / "report.html").exists()
    assert not (tmp_path / "subset" / "report.parquet").exists()
    assert nirs4all.RobustnessReport.load_artifacts(tmp_path / "subset").to_dict() == report.to_dict()

    parquet_only = report.save_artifacts(tmp_path / "parquet-only", formats=("parquet",))
    assert nirs4all.RobustnessReport.load_artifacts(tmp_path / "parquet-only").to_dict() == report.to_dict()
    assert set(parquet_only) == {"manifest", "parquet"}

    corrupted_markdown = report.save_artifacts(tmp_path / "corrupted-markdown", formats=("json", "markdown"))
    corrupted_markdown["markdown"].write_text("corrupted", encoding="utf-8")
    with pytest.raises(ValueError, match="Markdown file mismatch"):
        nirs4all.RobustnessReport.load_artifacts(tmp_path / "corrupted-markdown")

    corrupted_summary = report.save_artifacts(tmp_path / "corrupted-summary", formats=("json", "summary"))
    corrupted_summary["summary"].write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="summary file mismatch"):
        nirs4all.RobustnessReport.load_artifacts(tmp_path / "corrupted-summary")

    corrupted_manifest_dir = tmp_path / "corrupted-manifest"
    corrupted_manifest = report.save_artifacts(corrupted_manifest_dir)
    corrupted_payload = json.loads(corrupted_manifest["manifest"].read_text(encoding="utf-8"))
    corrupted_payload["fingerprint"] = "bad"
    corrupted_manifest["manifest"].write_text(json.dumps(corrupted_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        nirs4all.RobustnessReport.load_artifacts(corrupted_manifest_dir)

    escaping_dir = tmp_path / "escaping-manifest"
    escaping = report.save_artifacts(escaping_dir, formats=("json",))
    escaping_payload = json.loads(escaping["manifest"].read_text(encoding="utf-8"))
    escaping_payload["files"]["json"] = "../report.json"
    escaping["manifest"].write_text(json.dumps(escaping_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="relative child paths"):
        nirs4all.RobustnessReport.load_artifacts(escaping_dir)

    markdown_only_dir = tmp_path / "markdown-only"
    report.save_artifacts(markdown_only_dir, formats=("markdown",))
    with pytest.raises(FileNotFoundError, match="JSON or Parquet"):
        nirs4all.RobustnessReport.load_artifacts(markdown_only_dir)

    for bad_formats, match in (
        ((), "at least one"),
        (("json", "json"), "duplicate"),
        (("json", "csv"), "unsupported"),
        (("json", ""), "empty"),
    ):
        with pytest.raises(ValueError, match=match):
            report.save_artifacts(tmp_path / "bad-artifacts", formats=bad_formats)

    existing_file = tmp_path / "not-a-directory"
    existing_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="target must be a directory"):
        report.save_artifacts(existing_file)
    with pytest.raises(ValueError, match="artifact-directory export"):
        nirs4all.RobustnessReport.load_artifacts(existing_file)


def test_public_robustness_accepts_calibrated_run_result() -> None:
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.1, 3.1, 4.1],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
    )

    report = nirs4all.robustness(calibrated, y_true=[10.0, 20.0])
    method_report = calibrated.robustness(y_true=[10.0, 20.0])

    assert report.metadata["sample_ids"] == ["p1", "p2"]
    assert report.scenarios[0].conformal_metrics[0.8].observed_coverage == pytest.approx(1.0)
    assert method_report.to_dict() == report.to_dict()


def test_public_predict_result_exposes_robustness_convenience_method() -> None:
    result = nirs4all.PredictResult(
        y_pred=np.asarray([1.0, 2.0], dtype=float),
        metadata={"batch": ["b1", "b2"]},
        sample_indices=np.asarray(["p1", "p2"], dtype=object),
    )

    report = result.robustness(
        y_true=[1.0, 1.5],
        metadata={"batch": ["b1", "b2"]},
        slice_by=["batch"],
    )
    expected = nirs4all.robustness(
        result,
        y_true=[1.0, 1.5],
        metadata={"batch": ["b1", "b2"]},
        slice_by=["batch"],
    )

    assert report.to_dict() == expected.to_dict()
    assert report.summary_rows()[0]["worst_slice_label"] == "batch=b2"


def test_public_robustness_from_workspace_prediction_uses_published_evidence_and_can_save(monkeypatch, tmp_path) -> None:
    robustness_module = importlib.import_module("nirs4all.api.robustness")
    workspace = tmp_path / "workspace"
    prediction = nirs4all.PredictResult(
        y_pred=np.asarray([1.0, 2.0], dtype=float),
        metadata={
            "X": np.asarray([[1.0, 10.0], [2.0, 20.0]], dtype=float),
            "robustness_evidence": {"predictor_bundle": "models/pls.n4a"},
        },
        sample_indices=np.asarray(["s1", "s2"], dtype=object),
    )
    saved_calls = []

    monkeypatch.setattr(robustness_module, "load_workspace_predict_result", lambda path, prediction_id: prediction)
    monkeypatch.setattr(
        robustness_module,
        "_predict_with_predictor_bundle",
        lambda predictor_bundle, X, *, expected, sample_ids: np.asarray(X[:, 0], dtype=float),
    )

    def _fake_save(path, report, **kwargs):
        saved_calls.append((path, report, kwargs))
        return "robustness-001"

    monkeypatch.setattr(robustness_module, "save_workspace_robustness_report", _fake_save)

    report = nirs4all.robustness_from_workspace_prediction(
        workspace,
        "pred-001",
        y_true=[1.0, 2.5],
        scenarios=[{"kind": "spectral_offset", "severity": 0.5}],
        save_to_workspace=True,
        workspace_name="Spectral audit",
        workspace_robustness_id="robustness-001",
        workspace_metadata={"source": "unit"},
    )

    assert report.metadata["spectral_replay"] == {
        "all_predictions": False,
        "predictor_bundle": "models/pls.n4a",
        "route": "nirs4all.predict",
        "sample_ids_forwarded": True,
        "source": "predictor_bundle",
    }
    assert report.summary_rows()[0]["execution_scope"] == "spectral_replay"
    assert saved_calls == [
        (
            workspace,
            report,
            {
                "name": "Spectral audit",
                "robustness_id": "robustness-001",
                "metadata": {"source": "unit"},
                "prediction_id": "pred-001",
            },
        )
    ]


def test_public_robustness_convenience_methods_forward_predictor_bundle(monkeypatch) -> None:
    robustness_module = importlib.import_module("nirs4all.api.robustness")
    result = nirs4all.PredictResult(y_pred=np.asarray([1.0, 2.0], dtype=float))
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0],
        y_pred_calibration=[1.0, 2.0],
        y_pred=[1.0, 2.0],
        calibration_sample_ids=["c1", "c2"],
        prediction_sample_ids=["p1", "p2"],
    )
    calls = []

    def _fake_robustness(source, **kwargs):
        calls.append((source, kwargs))
        return nirs4all.RobustnessReport(
            mode="clean_frozen",
            scenarios=(
                nirs4all.RobustnessScenarioResult(
                    scenario={"kind": "observed", "severity": 0.0},
                    severity=0.0,
                    metrics=nirs4all.RobustnessMetricSet(
                        n_samples=2,
                        rmse=0.0,
                        mae=0.0,
                        bias=0.0,
                        max_abs_error=0.0,
                    ),
                ),
            ),
        )

    monkeypatch.setattr(robustness_module, "robustness", _fake_robustness)

    report = result.robustness(
        y_true=[1.0, 2.0],
        predictor_bundle="model.n4a",
    )
    calibrated_report = calibrated.robustness(
        y_true=[1.0, 2.0],
        predictor_bundle="calibrated-model.n4a",
    )

    assert isinstance(report, nirs4all.RobustnessReport)
    assert isinstance(calibrated_report, nirs4all.RobustnessReport)
    assert calls == [
        (
            result,
            {
                "y_true": [1.0, 2.0],
                "X": None,
                "predictor": None,
                "predictor_bundle": "model.n4a",
                "mode": "clean_frozen",
                "scenarios": None,
                "slice_by": None,
                "metadata": None,
                "seed": None,
                "workspace_path": None,
                "workspace_name": "",
                "workspace_robustness_id": None,
                "workspace_metadata": None,
            },
        ),
        (
            calibrated,
            {
                "y_true": [1.0, 2.0],
                "X": None,
                "predictor": None,
                "predictor_bundle": "calibrated-model.n4a",
                "mode": "clean_frozen",
                "scenarios": None,
                "slice_by": None,
                "metadata": None,
                "seed": None,
                "workspace_path": None,
                "workspace_name": "",
                "workspace_robustness_id": None,
                "workspace_metadata": None,
            },
        ),
    ]


def test_public_robustness_rejects_non_audit_modes_and_perturbations() -> None:
    result = PredictResult(y_pred=np.asarray([1.0, 2.0]))

    with pytest.raises(NotImplementedError, match="clean_frozen"):
        nirs4all.robustness(result, y_true=[1.0, 2.0], mode="matched_recalibration")

    with pytest.raises(NotImplementedError, match="supported audit-only kinds"):
        nirs4all.robustness(
            result,
            y_true=[1.0, 2.0],
            scenarios=[{"kind": "gaussian_noise", "severity": 0.01}],
        )

    with pytest.raises(NotImplementedError, match="observed robustness scenarios require severity=0.0"):
        nirs4all.robustness(
            result,
            y_true=[1.0, 2.0],
            scenarios=[{"kind": "observed", "severity": 0.01}],
        )

    with pytest.raises(ValueError, match="prediction_noise severity must be non-negative"):
        nirs4all.robustness(
            result,
            y_true=[1.0, 2.0],
            scenarios=[{"kind": "prediction_noise", "severity": -0.01}],
        )

    with pytest.raises(NotImplementedError, match="distribution='normal' or 'uniform'"):
        nirs4all.robustness(
            result,
            y_true=[1.0, 2.0],
            scenarios=[{"kind": "prediction_noise", "severity": 0.01, "distribution": "laplace"}],
        )

    with pytest.raises(ValueError, match="supported only for prediction_noise and spectral_noise"):
        nirs4all.robustness(
            result,
            y_true=[1.0, 2.0],
            scenarios=[{"kind": "prediction_bias", "severity": 0.01, "distribution": "normal"}],
        )

    with pytest.raises(ValueError, match="TCV1 JSON-native"):
        nirs4all.robustness(
            result,
            y_true=[1.0, 2.0],
            scenarios=[{"kind": "prediction_bias", "severity": 0.01, "opaque": object()}],
        )

    with pytest.raises(ValueError, match="spectral_noise requires explicit X and predictor"):
        nirs4all.robustness(
            result,
            y_true=[1.0, 2.0],
            scenarios=[{"kind": "spectral_noise", "severity": 0.01}],
        )

    with pytest.raises(ValueError, match="spectral_noise severity must be non-negative"):
        nirs4all.robustness(
            result,
            y_true=[1.0, 2.0],
            X=np.asarray([[1.0], [2.0]], dtype=float),
            predictor=FirstColumnPredictor(),
            scenarios=[{"kind": "spectral_noise", "severity": -0.01}],
        )

    with pytest.raises(NotImplementedError, match="distribution='normal' or 'uniform'"):
        nirs4all.robustness(
            result,
            y_true=[1.0, 2.0],
            X=np.asarray([[1.0], [2.0]], dtype=float),
            predictor=FirstColumnPredictor(),
            scenarios=[{"kind": "spectral_noise", "severity": 0.01, "distribution": "laplace"}],
        )


def test_public_robustness_validates_metadata_and_shapes() -> None:
    result = PredictResult(y_pred=np.asarray([1.0, 2.0]))

    with pytest.raises(ValueError, match="same shape"):
        nirs4all.robustness(result, y_true=[1.0])

    with pytest.raises(ValueError, match="missing from metadata"):
        nirs4all.robustness(result, y_true=[1.0, 2.0], metadata={"batch": ["a", "b"]}, slice_by=["instrument"])

    with pytest.raises(ValueError, match="values must be unique"):
        nirs4all.robustness(result, y_true=[1.0, 2.0], metadata={"batch": ["a", "b"]}, slice_by=["batch", "batch"])

    with pytest.raises(ValueError, match="non-negative integer"):
        nirs4all.robustness(result, y_true=[1.0, 2.0], seed=-1)

    with pytest.raises(ValueError, match="same number of rows"):
        nirs4all.robustness(
            result,
            y_true=[1.0, 2.0],
            X=np.asarray([[1.0]], dtype=float),
        )
