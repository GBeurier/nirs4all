"""Unit tests for the public conformal calibration API boundary."""

from __future__ import annotations

import importlib
import json
import math
import sqlite3
import zipfile

import numpy as np
import pytest

import nirs4all
from nirs4all.api.calibrate import Nirs4AllCalibrationNotImplementedError
from nirs4all.api.result import PredictResult
from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.dagml.conformal_contracts import CalibratedRunResult


class _CalibrationIdentityAwarePredictor:
    def __init__(self) -> None:
        self.seen_sample_ids = None
        self.seen_groups = None
        self.seen_metadata = None

    def predict(self, X, *, sample_ids=None, groups=None, metadata=None):
        self.seen_sample_ids = list(sample_ids) if sample_ids is not None else None
        self.seen_groups = list(groups) if groups is not None else None
        self.seen_metadata = list(metadata) if metadata is not None else None
        identities_match = (
            self.seen_sample_ids == ["c1", "c2", "c3", "c4"]
            and self.seen_groups == ["batch-a", "batch-a", "batch-b", "batch-b"]
            and self.seen_metadata == [{"site": "north"}, {"site": "north"}, {"site": "south"}, {"site": "south"}]
        )
        if identities_match:
            return np.asarray([0.1, 0.8, 2.4, 2.7], dtype=float)
        return np.full(len(X), 99.0)


def _make_calibration_spectro_dataset() -> SpectroDataset:
    dataset = SpectroDataset("native-calibration")
    dataset.add_samples(
        np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=float),
        indexes={"partition": "calibration"},
    )
    dataset.add_targets(np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float))
    dataset.add_metadata(
        np.asarray(
            [
                ["c1", "batch-a", "north"],
                ["c2", "batch-a", "north"],
                ["c3", "batch-b", "south"],
                ["c4", "batch-b", "south"],
            ],
            dtype=object,
        ),
        headers=["sample_id", "batch", "site"],
    )
    return dataset


def _write_calibration_dataset_config(tmp_path) -> str:
    x_path = tmp_path / "X_train.csv"
    y_path = tmp_path / "Y_train.csv"
    metadata_path = tmp_path / "M_train.csv"
    config_path = tmp_path / "calibration_dataset.json"

    x_path.write_text("1000;1100\n1.0;1.1\n2.0;2.1\n3.0;3.1\n4.0;4.1\n", encoding="utf-8")
    y_path.write_text("target\n1.0\n2.0\n3.0\n4.0\n", encoding="utf-8")
    metadata_path.write_text(
        "sample_id;batch;site\nc1;batch-a;north\nc2;batch-a;north\nc3;batch-b;south\nc4;batch-b;south\n",
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(
            {
                "name": "file-calibration",
                "train_x": str(x_path),
                "train_y": str(y_path),
                "train_group": str(metadata_path),
                "global_params": {"delimiter": ";", "has_header": True},
            }
        ),
        encoding="utf-8",
    )
    return str(config_path)


def test_public_conformal_vocabulary_constants_match_registry_and_runtime() -> None:
    registry = nirs4all.get_keyword_registry()
    entries = {entry["id"]: entry for entry in registry["entries"]}

    assert nirs4all.CONFORMAL_CALIBRATION_METHODS == ("split_absolute_residual",)
    assert nirs4all.CONFORMAL_CALIBRATION_UNITS == ("physical_sample",)
    assert nirs4all.CONFORMAL_MULTI_TARGET_POLICIES == ("marginal", "joint_max")
    assert nirs4all.CONFORMAL_EXECUTABLE_MULTI_TARGET_POLICIES == ("marginal", "joint_max")
    assert entries["calibrate.method"]["value_schema"]["enum"] == list(nirs4all.CONFORMAL_CALIBRATION_METHODS)
    assert entries["calibrate.unit"]["value_schema"]["const"] == nirs4all.CONFORMAL_CALIBRATION_UNITS[0]
    assert entries["calibrate.multi_target"]["value_schema"]["enum"] == list(nirs4all.CONFORMAL_MULTI_TARGET_POLICIES)
    assert entries["calibrate.group_by"]["status"] == "partial"
    assert entries["calibrate.prediction_groups"]["status"] == "partial"
    assert entries["calibrate.prediction_metadata"]["status"] == "partial"

    result = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0],
        y_pred_calibration=[1.1, 1.8, 3.2],
        y_pred=[1.0, 2.0],
        calibration_sample_ids=["c1", "c2", "c3"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        method=nirs4all.CONFORMAL_CALIBRATION_METHODS[0],
        unit=nirs4all.CONFORMAL_CALIBRATION_UNITS[0],
        multi_target=nirs4all.CONFORMAL_EXECUTABLE_MULTI_TARGET_POLICIES[0],
    )
    assert result.artifact.spec.method == "split_absolute_residual"
    assert result.artifact.spec.unit == "physical_sample"
    assert result.artifact.spec.multi_target == "marginal"
    joint = nirs4all.calibrate(
        y_true=[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        y_pred_calibration=[[1.1, 9.8], [1.8, 20.3], [3.2, 30.1]],
        y_pred=[[1.0, 10.0], [2.0, 20.0]],
        calibration_sample_ids=["c1", "c2", "c3"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.75,
        multi_target="joint_max",
    )
    assert joint.artifact.spec.multi_target == "joint_max"
    assert joint.conformal_guarantee_status is not None
    assert joint.conformal_guarantee_status["guarantee"] == "split_conformal_joint_max_simultaneous_coverage"
    np.testing.assert_allclose(joint.prediction.interval(0.75).lower, [[0.7, 9.7], [1.7, 19.7]])


def test_public_calibrate_replayed_arrays_returns_calibrated_result() -> None:
    result = nirs4all.calibrate(
        calibration_data={
            "y_true": [1.0, 2.0, 3.0, 4.0],
            "y_pred": [0.9, 2.2, 2.6, 4.3],
            "sample_ids": ["c1", "c2", "c3", "c4"],
            "groups": ["a", "a", "b", "b"],
            "metadata": {"batch": ["b1", "b1", "b2", "b2"]},
        },
        y_pred=[10.0, 20.0],
        prediction_sample_ids=["p1", "p2"],
        coverage=[0.5, 0.8],
        target_name="moisture",
        predictor_fingerprint="predictor-abc",
    )

    assert isinstance(result, CalibratedRunResult)
    assert result.sample_ids == ("p1", "p2")
    assert result.artifact.target_name == "moisture"
    assert result.artifact.predictor_fingerprint == "predictor-abc"
    assert result.artifact.calibration_cohort is not None
    assert result.artifact.calibration_cohort.sample_ids == ("c1", "c2", "c3", "c4")
    assert result.conformal_guarantee_status is not None
    assert result.conformal_guarantee_status["status"] == "active"
    assert result.conformal_guarantee_status["requested_engine"] == "nirs4all.conformal.v1"
    assert result.conformal_guarantee_status["effective_engine"] == "nirs4all.python.replayed_array_calibration"
    assert result.conformal_guarantee_status["coverage"] == [0.5, 0.8]
    assert result.conformal_guarantee_status["artifact_fingerprint"] == result.artifact.fingerprint
    assert result.metadata["calibration_replay_source"] == {
        "dataset_backed": False,
        "kind": "replayed_arrays",
        "requires_model_replay": False,
        "route": "provided_arrays",
        "version": 1,
    }
    assert result.calibration_replay_source == result.metadata["calibration_replay_source"]
    assert result.conformal_guarantee_status["calibration_replay_source"] == result.metadata["calibration_replay_source"]
    np.testing.assert_allclose(result.prediction.interval(0.8).lower, [9.6, 19.6])
    np.testing.assert_allclose(result.prediction.interval(0.8).upper, [10.4, 20.4])


def test_public_calibrate_replayed_arrays_supports_group_by_with_prediction_groups() -> None:
    result = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 1.8, 2.5, 3.4],
        y_pred=[10.0, 20.0, 30.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2", "p3"],
        calibration_groups=["instrument-a", "instrument-a", "instrument-b", "instrument-b"],
        prediction_groups=["instrument-a", "instrument-b", "instrument-a"],
        group_by="group",
        coverage=0.5,
    )

    assert result.artifact.spec.group_by == ("group",)
    assert result.conformal_guarantee_status is not None
    assert result.conformal_guarantee_status["guarantee"] == "split_conformal_group_marginal_coverage"
    assert result.conformal_guarantee_status["scope"] == "finite_sample_group_conditional_exchangeability"
    np.testing.assert_allclose(result.prediction.interval(0.5).qhat, [0.2, 0.6, 0.2])
    np.testing.assert_allclose(result.prediction.interval(0.5).lower, [9.8, 19.4, 29.8])


def test_public_calibrate_group_by_fails_closed_for_unseen_prediction_group() -> None:
    with pytest.raises(ValueError, match="absent from the conformal calibration artifact"):
        nirs4all.calibrate(
            y_true=[1.0, 2.0, 3.0, 4.0],
            y_pred_calibration=[0.9, 1.8, 2.5, 3.4],
            y_pred=[10.0],
            calibration_sample_ids=["c1", "c2", "c3", "c4"],
            prediction_sample_ids=["p1"],
            calibration_groups=["instrument-a", "instrument-a", "instrument-b", "instrument-b"],
            prediction_groups=["instrument-c"],
            group_by="group",
            coverage=0.5,
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
def test_public_calibrate_group_by_rejects_coercive_group_labels(
    calibration_groups: list[str],
    prediction_groups: list[str],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        nirs4all.calibrate(
            y_true=[1.0, 2.0, 3.0, 4.0],
            y_pred_calibration=[0.9, 1.8, 2.5, 3.4],
            y_pred=[10.0],
            calibration_sample_ids=["c1", "c2", "c3", "c4"],
            prediction_sample_ids=["p1"],
            calibration_groups=calibration_groups,
            prediction_groups=prediction_groups,
            group_by="group",
            coverage=0.5,
        )


def test_conformal_quantile_identity_and_roundtrip_fixture(tmp_path) -> None:
    """V1.7 fixture: exact qhat, physical ids, store/workspace reload and coverage selection."""

    store_path = tmp_path / "calibrated-store"
    workspace_path = tmp_path / "workspace"
    result = nirs4all.calibrate(
        calibration_data={
            "y_true": [0.0, 0.0, 0.0, 0.0, 0.0],
            "y_pred": [0.05, -0.10, 0.20, -0.40, 0.80],
            "sample_ids": ["cal-001", "cal-002", "cal-003", "cal-004", "cal-005"],
            "groups": ["instrument-a", "instrument-a", "instrument-b", "instrument-b", "instrument-c"],
            "metadata": {"site": ["north", "north", "south", "south", "west"]},
            "predictor_fingerprint": "fixture-predictor-v1",
        },
        y_pred=[10.0, 20.0],
        prediction_sample_ids=["pred-001", "pred-002"],
        coverage=[0.5, 0.8, 0.9],
        target_name="protein",
        store_path=store_path,
        workspace_path=workspace_path,
        workspace_conformal_id="quantile-fixture",
        workspace_name="Quantile fixture",
    )

    assert result.artifact.calibration_cohort is not None
    assert result.artifact.calibration_cohort.sample_ids == (
        "cal-001",
        "cal-002",
        "cal-003",
        "cal-004",
        "cal-005",
    )
    assert result.sample_ids == ("pred-001", "pred-002")
    assert not set(result.artifact.calibration_cohort.sample_ids) & set(result.sample_ids)
    assert result.artifact.predictor_fingerprint == "fixture-predictor-v1"
    assert result.artifact.calibrator.scores.tolist() == [0.05, 0.1, 0.2, 0.4, 0.8]
    assert result.artifact.calibrator.qhat_by_coverage[0.5] == 0.2
    assert result.artifact.calibrator.qhat_by_coverage[0.8] == 0.8
    assert math.isinf(result.artifact.calibrator.qhat_by_coverage[0.9])
    np.testing.assert_allclose(result.prediction.interval(0.5).lower, [9.8, 19.8])
    np.testing.assert_allclose(result.prediction.interval(0.5).upper, [10.2, 20.2])
    np.testing.assert_allclose(result.prediction.interval(0.8).lower, [9.2, 19.2])
    assert np.isneginf(result.prediction.interval(0.9).lower).all()
    assert np.isposinf(result.prediction.interval(0.9).upper).all()

    from_json = nirs4all.CalibratedRunResult.from_json(result.to_json())
    from_store = nirs4all.load_calibrated_result(store_path)
    from_workspace = nirs4all.load_workspace_calibrated_result(workspace_path, "quantile-fixture")
    for restored in (from_json, from_store, from_workspace):
        assert restored.fingerprint == result.fingerprint
        assert restored.artifact.fingerprint == result.artifact.fingerprint
        assert restored.artifact.calibration_data_fingerprint == result.artifact.calibration_data_fingerprint
        assert restored.artifact.calibration_cohort is not None
        assert restored.artifact.calibration_cohort.fingerprint == result.artifact.calibration_cohort.fingerprint
        assert restored.sample_ids == result.sample_ids

    selected = nirs4all.predict(
        model=from_workspace,
        data={"y_pred": [30.0, 40.0], "sample_ids": ["pred-003", "pred-004"]},
        coverage=0.5,
    )

    assert selected.interval_coverages == (0.5,)
    assert selected.metadata["selected_interval_coverages"] == [0.5]
    assert selected.conformal_guarantee_status is not None
    assert selected.conformal_guarantee_status["coverage"] == [0.5]
    assert selected.conformal_guarantee_status["source_calibrated_result_fingerprint"] == result.fingerprint
    np.testing.assert_allclose(selected.interval(0.5).lower, [29.8, 39.8])
    np.testing.assert_allclose(selected.interval(0.5).upper, [30.2, 40.2])


def test_public_calibrate_accepts_explicit_replayed_array_arguments() -> None:
    result = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1"],
        coverage=0.8,
    )

    assert result.artifact.spec.coverage == (0.8,)
    np.testing.assert_allclose(result.prediction.interval(0.8).upper, [10.4])


def test_public_calibrate_rejects_prediction_calibration_sample_overlap() -> None:
    with pytest.raises(ValueError, match="overlapping physical samples: c2"):
        nirs4all.calibrate(
            y_true=[1.0, 2.0, 3.0, 4.0],
            y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
            y_pred=[10.0, 20.0],
            calibration_sample_ids=["c1", "c2", "c3", "c4"],
            prediction_sample_ids=["p1", "c2"],
            coverage=0.8,
        )


def test_public_calibrate_rejects_duplicate_prediction_sample_ids() -> None:
    with pytest.raises(ValueError, match="unique physical sample"):
        nirs4all.calibrate(
            y_true=[1.0, 2.0, 3.0, 4.0],
            y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
            y_pred=[10.0, 20.0],
            calibration_sample_ids=["c1", "c2", "c3", "c4"],
            prediction_sample_ids=["p1", "p1"],
            coverage=0.8,
        )


def test_public_calibrate_rejects_coercive_prediction_sample_ids() -> None:
    for prediction_sample_ids in ([" p1"], ["p1 "], ["bad\x00id"], [123]):
        with pytest.raises(ValueError, match="canonical non-empty strings"):
            nirs4all.calibrate(
                y_true=[1.0, 2.0, 3.0, 4.0],
                y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
                y_pred=[10.0],
                calibration_sample_ids=["c1", "c2", "c3", "c4"],
                prediction_sample_ids=prediction_sample_ids,
                coverage=0.8,
            )


def test_public_calibrate_accepts_tuple_calibration_data() -> None:
    result = nirs4all.calibrate(
        calibration_data=(
            [1.0, 2.0, 3.0, 4.0],
            [0.9, 2.2, 2.6, 4.3],
            ["c1", "c2", "c3", "c4"],
            ["batch-a", "batch-a", "batch-b", "batch-b"],
            {"site": ["north", "north", "south", "south"]},
        ),
        y_pred=[10.0, 20.0],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
    )

    assert result.artifact.calibration_cohort is not None
    assert result.artifact.calibration_cohort.sample_ids == ("c1", "c2", "c3", "c4")
    assert result.artifact.calibration_cohort.groups == ("batch-a", "batch-a", "batch-b", "batch-b")
    np.testing.assert_allclose(result.prediction.interval(0.8).lower, [9.6, 19.6])


def test_public_calibrate_accepts_typed_calibration_data() -> None:
    result = nirs4all.calibrate(
        calibration_data=nirs4all.ConformalCalibrationData(
            y_true=[1.0, 2.0, 3.0, 4.0],
            y_pred=[0.9, 2.2, 2.6, 4.3],
            sample_ids=["c1", "c2", "c3", "c4"],
            groups=["batch-a", "batch-a", "batch-b", "batch-b"],
            metadata={"site": ["north", "north", "south", "south"]},
            predictor_fingerprint="predictor-typed",
        ),
        y_pred=[10.0, 20.0],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
    )

    assert result.artifact.predictor_fingerprint == "predictor-typed"
    assert result.artifact.calibration_cohort is not None
    assert result.artifact.calibration_cohort.sample_ids == ("c1", "c2", "c3", "c4")
    np.testing.assert_allclose(result.prediction.interval(0.8).upper, [10.4, 20.4])


def test_public_calibrate_accepts_replayed_array_mapping_aliases() -> None:
    result = nirs4all.calibrate(
        calibration_data={
            "y_true": [1.0, 2.0, 3.0, 4.0],
            "y_pred_calibration": [0.9, 2.2, 2.6, 4.3],
            "calibration_sample_ids": ["c1", "c2", "c3", "c4"],
            "calibration_groups": ["batch-a", "batch-a", "batch-b", "batch-b"],
            "calibration_metadata": {"site": ["north", "north", "south", "south"]},
            "predictor_fingerprint": "predictor-aliases",
        },
        y_pred=[10.0, 20.0],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
    )

    assert result.artifact.predictor_fingerprint == "predictor-aliases"
    assert result.artifact.calibration_cohort is not None
    assert result.artifact.calibration_cohort.sample_ids == ("c1", "c2", "c3", "c4")
    assert result.artifact.calibration_cohort.groups == ("batch-a", "batch-a", "batch-b", "batch-b")
    assert tuple(row.metadata["site"] for row in result.artifact.calibration_cohort.rows) == ("north", "north", "south", "south")
    np.testing.assert_allclose(result.prediction.interval(0.8).upper, [10.4, 20.4])


def test_public_calibrate_replayed_array_mapping_rejects_ambiguous_aliases() -> None:
    with pytest.raises(ValueError, match="accepts only one alias for y_pred"):
        nirs4all.calibrate(
            calibration_data={
                "y_true": [1.0, 2.0, 3.0, 4.0],
                "y_pred": [0.9, 2.2, 2.6, 4.3],
                "y_pred_calibration": [0.9, 2.2, 2.6, 4.3],
                "sample_ids": ["c1", "c2", "c3", "c4"],
            },
            y_pred=[10.0],
            prediction_sample_ids=["p1"],
            coverage=0.8,
        )

    with pytest.raises(ValueError, match="accepts only one alias for sample_ids"):
        nirs4all.calibrate(
            calibration_data={
                "y_true": [1.0, 2.0, 3.0, 4.0],
                "calibration_predictions": [0.9, 2.2, 2.6, 4.3],
                "sample_ids": ["c1", "c2", "c3", "c4"],
                "physical_sample_ids": ["c1", "c2", "c3", "c4"],
            },
            y_pred=[10.0],
            prediction_sample_ids=["p1"],
            coverage=0.8,
        )


def test_public_calibrate_rejects_ambiguous_tuple_calibration_data() -> None:
    with pytest.raises(ValueError, match="calibration_data tuple must be"):
        nirs4all.calibrate(
            calibration_data=(
                [1.0, 2.0, 3.0, 4.0],
                [0.9, 2.2, 2.6, 4.3],
                ["c1", "c2", "c3", "c4"],
                None,
                None,
                "extra",
            ),
            y_pred=[10.0],
            prediction_sample_ids=["p1"],
            coverage=0.8,
        )


def test_public_calibrate_accepts_prediction_dict_with_extra_result_fields() -> None:
    prediction_entry = {
        "id": "pred-1",
        "dataset_name": "calibration",
        "model_name": "PLSRegression",
        "partition": "test",
        "fold_id": "final",
        "metric": "rmse",
        "y_true": np.array([1.0, 2.0, 3.0, 4.0]),
        "y_pred": np.array([0.9, 2.2, 2.6, 4.3]),
        "metadata": {
            "physical_sample_id": ["c1", "c2", "c3", "c4"],
            "batch": ["b1", "b1", "b2", "b2"],
        },
        "scores": {"test": {"rmse": 0.25}},
    }

    result = nirs4all.calibrate(
        calibration_data=prediction_entry,
        y_pred=[10.0],
        prediction_sample_ids=["p1"],
        coverage=0.8,
    )

    assert result.artifact.calibration_cohort is not None
    assert result.artifact.calibration_cohort.sample_ids == ("c1", "c2", "c3", "c4")
    np.testing.assert_allclose(result.prediction.interval(0.8).lower, [9.6])


def test_public_calibrate_accepts_explicit_spectro_dataset_calibration_cohort() -> None:
    result = nirs4all.calibrate(
        calibration_data={
            "dataset": _make_calibration_spectro_dataset(),
            "selector": {"partition": "calibration"},
            "sample_id_column": "sample_id",
            "group_column": "batch",
            "metadata_columns": ["site"],
            "y_pred": [0.1, 0.8, 2.4, 2.7],
        },
        y_pred=[10.0, 20.0],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
    )

    assert result.artifact.calibration_cohort is not None
    assert result.artifact.calibration_cohort.sample_ids == ("c1", "c2", "c3", "c4")
    assert result.artifact.calibration_cohort.groups == ("batch-a", "batch-a", "batch-b", "batch-b")
    assert tuple(row.metadata for row in result.artifact.calibration_cohort.rows) == (
        {"site": "north"},
        {"site": "north"},
        {"site": "south"},
        {"site": "south"},
    )
    np.testing.assert_allclose(result.prediction.interval(0.8).lower, [9.6, 19.6])
    np.testing.assert_allclose(result.prediction.interval(0.8).upper, [10.4, 20.4])


def test_public_calibrate_dataset_backed_mapping_accepts_calibration_aliases() -> None:
    result = nirs4all.calibrate(
        calibration_data={
            "dataset": _make_calibration_spectro_dataset(),
            "selector": {"partition": "calibration"},
            "calibration_sample_ids": ["c1", "c2", "c3", "c4"],
            "calibration_groups": ["batch-a", "batch-a", "batch-b", "batch-b"],
            "calibration_metadata": {"site": ["north", "north", "south", "south"]},
            "y_pred_calibration": [0.1, 0.8, 2.4, 2.7],
        },
        y_pred=[10.0],
        prediction_sample_ids=["p1"],
        coverage=0.8,
    )

    assert result.artifact.calibration_cohort is not None
    assert result.artifact.calibration_cohort.sample_ids == ("c1", "c2", "c3", "c4")
    assert result.artifact.calibration_cohort.groups == ("batch-a", "batch-a", "batch-b", "batch-b")
    assert tuple(row.metadata for row in result.artifact.calibration_cohort.rows) == (
        {"site": "north"},
        {"site": "north"},
        {"site": "south"},
        {"site": "south"},
    )
    np.testing.assert_allclose(result.prediction.interval(0.8).upper, [10.4])


def test_public_calibrate_dataset_backed_mapping_rejects_ambiguous_sample_id_aliases() -> None:
    with pytest.raises(ValueError, match="accepts only one alias for sample_ids"):
        nirs4all.calibrate(
            calibration_data={
                "dataset": _make_calibration_spectro_dataset(),
                "selector": {"partition": "calibration"},
                "sample_ids": ["c1", "c2", "c3", "c4"],
                "calibration_sample_ids": ["c1", "c2", "c3", "c4"],
                "y_pred": [0.1, 0.8, 2.4, 2.7],
            },
            y_pred=[10.0],
            prediction_sample_ids=["p1"],
            coverage=0.8,
        )


def test_public_calibrate_dataset_backed_mapping_rejects_ambiguous_group_metadata_aliases() -> None:
    with pytest.raises(ValueError, match="accepts only one alias for groups"):
        nirs4all.calibrate(
            calibration_data={
                "dataset": _make_calibration_spectro_dataset(),
                "selector": {"partition": "calibration"},
                "sample_ids": ["c1", "c2", "c3", "c4"],
                "groups": ["batch-a", "batch-a", "batch-b", "batch-b"],
                "calibration_groups": ["batch-a", "batch-a", "batch-b", "batch-b"],
                "y_pred": [0.1, 0.8, 2.4, 2.7],
            },
            y_pred=[10.0],
            prediction_sample_ids=["p1"],
            coverage=0.8,
        )

    with pytest.raises(ValueError, match="accepts only one alias for metadata"):
        nirs4all.calibrate(
            calibration_data={
                "dataset": _make_calibration_spectro_dataset(),
                "selector": {"partition": "calibration"},
                "sample_ids": ["c1", "c2", "c3", "c4"],
                "metadata": {"site": ["north", "north", "south", "south"]},
                "calibration_metadata": {"site": ["north", "north", "south", "south"]},
                "y_pred": [0.1, 0.8, 2.4, 2.7],
            },
            y_pred=[10.0],
            prediction_sample_ids=["p1"],
            coverage=0.8,
        )


def test_public_calibrate_accepts_typed_spectro_dataset_calibration_cohort() -> None:
    result = nirs4all.calibrate(
        calibration_data=nirs4all.ConformalCalibrationData(
            dataset=_make_calibration_spectro_dataset(),
            selector={"partition": "calibration"},
            sample_id_column="sample_id",
            group_column="batch",
            metadata_columns=["site"],
            y_pred=[0.1, 0.8, 2.4, 2.7],
        ),
        y_pred=[10.0, 20.0],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
    )

    assert result.artifact.calibration_cohort is not None
    assert result.artifact.calibration_cohort.sample_ids == ("c1", "c2", "c3", "c4")
    assert result.artifact.calibration_cohort.groups == ("batch-a", "batch-a", "batch-b", "batch-b")
    assert tuple(row.metadata for row in result.artifact.calibration_cohort.rows) == (
        {"site": "north"},
        {"site": "north"},
        {"site": "south"},
        {"site": "south"},
    )
    np.testing.assert_allclose(result.prediction.interval(0.8).lower, [9.6, 19.6])


def test_public_calibrate_accepts_typed_dataset_backed_predictor() -> None:
    predictor = _CalibrationIdentityAwarePredictor()

    result = nirs4all.calibrate(
        calibration_data=nirs4all.ConformalCalibrationData(
            dataset=_make_calibration_spectro_dataset(),
            selector={"partition": "calibration"},
            sample_id_column="sample_id",
            group_column="batch",
            metadata_columns=["site"],
            predictor=predictor,
            predictor_fingerprint="typed-predictor",
        ),
        y_pred=[10.0],
        prediction_sample_ids=["p1"],
        coverage=0.8,
    )

    assert predictor.seen_sample_ids == ["c1", "c2", "c3", "c4"]
    assert predictor.seen_groups == ["batch-a", "batch-a", "batch-b", "batch-b"]
    assert predictor.seen_metadata == [{"site": "north"}, {"site": "north"}, {"site": "south"}, {"site": "south"}]
    assert result.artifact.predictor_fingerprint == "typed-predictor"
    np.testing.assert_allclose(result.prediction.interval(0.8).lower, [9.6])


def test_conformal_calibration_data_rejects_incomplete_array_payload() -> None:
    with pytest.raises(ValueError, match="requires y_true/y_pred or dataset/selector"):
        nirs4all.ConformalCalibrationData(
            y_true=[1.0, 2.0],
            sample_ids=["c1", "c2"],
        ).to_dict()


def test_conformal_calibration_data_rejects_dataset_payload_without_selector() -> None:
    with pytest.raises(ValueError, match="dataset-backed form requires selector"):
        nirs4all.ConformalCalibrationData(
            dataset=_make_calibration_spectro_dataset(),
            y_pred=[0.1, 0.8, 2.4, 2.7],
            sample_id_column="sample_id",
        ).to_dict()


def test_conformal_calibration_data_rejects_coercive_dataset_selector_payloads() -> None:
    dataset = _make_calibration_spectro_dataset()

    with pytest.raises(ValueError, match="selector keys must be canonical non-empty strings"):
        nirs4all.ConformalCalibrationData(
            dataset=dataset,
            selector={1: "calibration"},  # type: ignore[dict-item]
            y_pred=[0.1, 0.8, 2.4, 2.7],
            sample_id_column="sample_id",
        ).to_dict()

    with pytest.raises(ValueError, match="selector keys must be canonical non-empty strings"):
        nirs4all.ConformalCalibrationData(
            dataset=dataset,
            selector={" partition ": "calibration"},
            y_pred=[0.1, 0.8, 2.4, 2.7],
            sample_id_column="sample_id",
        ).to_dict()

    with pytest.raises(ValueError, match="include_augmented must be a boolean"):
        nirs4all.ConformalCalibrationData(
            dataset=dataset,
            selector={"partition": "calibration"},
            y_pred=[0.1, 0.8, 2.4, 2.7],
            include_augmented="yes",  # type: ignore[arg-type]
            sample_id_column="sample_id",
        ).to_dict()
    with pytest.raises(ValueError, match="sample_id_column must be a canonical non-empty string"):
        nirs4all.ConformalCalibrationData(
            dataset=dataset,
            selector={"partition": "calibration"},
            y_pred=[0.1, 0.8, 2.4, 2.7],
            sample_id_column=1,  # type: ignore[arg-type]
        ).to_dict()
    with pytest.raises(ValueError, match="group_column must be a canonical non-empty string"):
        nirs4all.ConformalCalibrationData(
            dataset=dataset,
            selector={"partition": "calibration"},
            y_pred=[0.1, 0.8, 2.4, 2.7],
            sample_id_column="sample_id",
            group_column=" batch ",
        ).to_dict()
    with pytest.raises(ValueError, match="metadata_columns contains duplicate column names"):
        nirs4all.ConformalCalibrationData(
            dataset=dataset,
            selector={"partition": "calibration"},
            y_pred=[0.1, 0.8, 2.4, 2.7],
            sample_id_column="sample_id",
            metadata_columns=["site", "site"],
        ).to_dict()
    with pytest.raises(ValueError, match=r"ConformalCalibrationData.selector\[bad\] must be JSON-native"):
        nirs4all.ConformalCalibrationData(
            dataset=dataset,
            selector={"bad": object()},
            y_pred=[0.1, 0.8, 2.4, 2.7],
            sample_id_column="sample_id",
        ).to_dict()

    base_payload = {
        "dataset": dataset,
        "y_pred": [0.1, 0.8, 2.4, 2.7],
        "sample_id_column": "sample_id",
    }
    with pytest.raises(ValueError, match="Dataset-backed calibration_data.selector keys must be canonical non-empty strings"):
        nirs4all.calibrate(
            calibration_data={**base_payload, "selector": {1: "calibration"}},  # type: ignore[dict-item]
            y_pred=[10.0],
            prediction_sample_ids=["p1"],
        )
    with pytest.raises(ValueError, match="Dataset-backed calibration_data.selector keys must be canonical non-empty strings"):
        nirs4all.calibrate(
            calibration_data={**base_payload, "selector": {" partition ": "calibration"}},
            y_pred=[10.0],
            prediction_sample_ids=["p1"],
        )
    with pytest.raises(ValueError, match="Dataset-backed calibration_data.include_augmented must be a boolean"):
        nirs4all.calibrate(
            calibration_data={**base_payload, "selector": {"partition": "calibration"}, "include_augmented": "yes"},
            y_pred=[10.0],
            prediction_sample_ids=["p1"],
        )
    with pytest.raises(ValueError, match="Dataset-backed calibration_data.sample_id_column must be a canonical non-empty string"):
        nirs4all.calibrate(
            calibration_data={**base_payload, "sample_id_column": 1, "selector": {"partition": "calibration"}},
            y_pred=[10.0],
            prediction_sample_ids=["p1"],
        )
    with pytest.raises(ValueError, match="Dataset-backed calibration_data.group_column must be a canonical non-empty string"):
        nirs4all.calibrate(
            calibration_data={**base_payload, "selector": {"partition": "calibration"}, "group_column": " batch "},
            y_pred=[10.0],
            prediction_sample_ids=["p1"],
        )
    with pytest.raises(ValueError, match="Dataset-backed calibration_data.metadata_columns contains duplicate column names"):
        nirs4all.calibrate(
            calibration_data={**base_payload, "selector": {"partition": "calibration"}, "metadata_columns": ["site", "site"]},
            y_pred=[10.0],
            prediction_sample_ids=["p1"],
        )
    with pytest.raises(ValueError, match=r"Dataset-backed calibration_data.selector\[bad\] must be JSON-native"):
        nirs4all.calibrate(
            calibration_data={**base_payload, "selector": {"bad": object()}},
            y_pred=[10.0],
            prediction_sample_ids=["p1"],
        )


def test_conformal_calibration_data_rejects_ambiguous_dataset_replay_lanes() -> None:
    with pytest.raises(ValueError, match="at most one of y_pred, predictor, predictor_bundle"):
        nirs4all.ConformalCalibrationData(
            dataset=_make_calibration_spectro_dataset(),
            selector={"partition": "calibration"},
            y_pred=[0.1, 0.8, 2.4, 2.7],
            predictor=_CalibrationIdentityAwarePredictor(),
            sample_id_column="sample_id",
        ).to_dict()


def test_conformal_calibration_data_accepts_dataset_replay_aliases_as_canonical_payload() -> None:
    payload = nirs4all.ConformalCalibrationData(
        dataset=_make_calibration_spectro_dataset(),
        selector={"partition": "calibration"},
        sample_id_column="sample_id",
        model_bundle="model.n4a",
    ).to_dict()

    assert payload["predictor_bundle"] == "model.n4a"
    assert "model_bundle" not in payload

    payload = nirs4all.ConformalCalibrationData(
        dataset=_make_calibration_spectro_dataset(),
        selector={"partition": "calibration"},
        sample_id_column="sample_id",
        workspace_chain_id="chain-42",
        workspace_path="workspace",
    ).to_dict()

    assert payload["predictor_chain_id"] == "chain-42"
    assert "workspace_chain_id" not in payload


def test_conformal_calibration_data_rejects_coercive_workspace_chain_ids() -> None:
    for kwargs in (
        {"predictor_chain_id": ""},
        {"predictor_chain_id": " chain-42 "},
        {"predictor_chain_id": "bad\x00id"},
        {"workspace_chain_id": 123},
    ):
        with pytest.raises(ValueError, match="ConformalCalibrationData.predictor_chain_id must be a canonical non-empty string"):
            nirs4all.ConformalCalibrationData(
                dataset=_make_calibration_spectro_dataset(),
                selector={"partition": "calibration"},
                sample_id_column="sample_id",
                workspace_path="workspace",
                **kwargs,  # type: ignore[arg-type]
            ).to_dict()


def test_conformal_calibration_data_rejects_ambiguous_dataset_replay_aliases() -> None:
    with pytest.raises(ValueError, match="only one alias for predictor_bundle"):
        nirs4all.ConformalCalibrationData(
            dataset=_make_calibration_spectro_dataset(),
            selector={"partition": "calibration"},
            sample_id_column="sample_id",
            predictor_bundle="model-a.n4a",
            model_bundle="model-b.n4a",
        ).to_dict()

    with pytest.raises(ValueError, match="only one alias for predictor_chain_id"):
        nirs4all.ConformalCalibrationData(
            dataset=_make_calibration_spectro_dataset(),
            selector={"partition": "calibration"},
            sample_id_column="sample_id",
            predictor_chain_id="chain-a",
            workspace_chain_id="chain-b",
        ).to_dict()


def test_conformal_calibration_data_rejects_dataset_payload_with_y_true() -> None:
    with pytest.raises(ValueError, match="derives y_true from dataset/selector"):
        nirs4all.ConformalCalibrationData(
            dataset=_make_calibration_spectro_dataset(),
            selector={"partition": "calibration"},
            y_true=[1.0, 2.0, 3.0, 4.0],
            y_pred=[0.1, 0.8, 2.4, 2.7],
            sample_id_column="sample_id",
        ).to_dict()


def test_public_calibrate_accepts_dataset_config_path_calibration_cohort(tmp_path) -> None:
    result = nirs4all.calibrate(
        calibration_data={
            "dataset": _write_calibration_dataset_config(tmp_path),
            "selector": {"partition": "train"},
            "sample_id_column": "sample_id",
            "group_column": "batch",
            "metadata_columns": ["site"],
            "y_pred": [0.1, 0.8, 2.4, 2.7],
        },
        y_pred=[10.0],
        prediction_sample_ids=["p1"],
        coverage=0.8,
    )

    assert result.artifact.calibration_cohort is not None
    assert result.artifact.calibration_cohort.sample_ids == ("c1", "c2", "c3", "c4")
    assert result.artifact.calibration_cohort.groups == ("batch-a", "batch-a", "batch-b", "batch-b")
    assert tuple(row.metadata for row in result.artifact.calibration_cohort.rows) == (
        {"site": "north"},
        {"site": "north"},
        {"site": "south"},
        {"site": "south"},
    )
    np.testing.assert_allclose(result.prediction.interval(0.8).lower, [9.6])


def test_public_calibrate_replays_dataset_backed_predictor_with_identities() -> None:
    predictor = _CalibrationIdentityAwarePredictor()

    result = nirs4all.calibrate(
        calibration_data={
            "dataset": _make_calibration_spectro_dataset(),
            "selector": {"partition": "calibration"},
            "sample_id_column": "sample_id",
            "group_column": "batch",
            "metadata_columns": ["site"],
            "predictor": predictor,
            "predictor_fingerprint": "predictor-from-memory",
        },
        y_pred=[10.0],
        prediction_sample_ids=["p1"],
        coverage=0.8,
    )

    assert predictor.seen_sample_ids == ["c1", "c2", "c3", "c4"]
    assert predictor.seen_groups == ["batch-a", "batch-a", "batch-b", "batch-b"]
    assert predictor.seen_metadata == [{"site": "north"}, {"site": "north"}, {"site": "south"}, {"site": "south"}]
    assert result.artifact.predictor_fingerprint == "predictor-from-memory"
    np.testing.assert_allclose(result.prediction.interval(0.8).lower, [9.6])


def test_public_calibrate_replays_dataset_backed_predictor_bundle_with_identities(tmp_path, monkeypatch) -> None:
    from nirs4all.api.result import PredictResult

    predict_module = importlib.import_module("nirs4all.api.predict")
    calls = []

    def _fake_predict(**kwargs):
        calls.append(kwargs)
        data = kwargs["data"]
        assert kwargs["model"] == tmp_path / "model.n4a"
        assert list(data["sample_ids"]) == ["c1", "c2", "c3", "c4"]
        assert list(data["groups"]) == ["batch-a", "batch-a", "batch-b", "batch-b"]
        assert data["metadata"] == [{"site": "north"}, {"site": "north"}, {"site": "south"}, {"site": "south"}]
        return PredictResult(y_pred=np.asarray([0.1, 0.8, 2.4, 2.7], dtype=float))

    monkeypatch.setattr(predict_module, "predict", _fake_predict)

    result = nirs4all.calibrate(
        calibration_data={
            "dataset": _make_calibration_spectro_dataset(),
            "selector": {"partition": "calibration"},
            "sample_id_column": "sample_id",
            "group_column": "batch",
            "metadata_columns": ["site"],
            "predictor_bundle": tmp_path / "model.n4a",
        },
        y_pred=[10.0],
        prediction_sample_ids=["p1"],
        coverage=0.8,
    )

    assert len(calls) == 1
    assert result.artifact.predictor_fingerprint == str(tmp_path / "model.n4a")
    assert result.metadata["calibration_replay_source"] == {
        "dataset_backed": True,
        "kind": "dataset_predictor_bundle",
        "predictor_bundle": str(tmp_path / "model.n4a"),
        "requires_model_replay": True,
        "route": "nirs4all.predict",
        "version": 1,
    }
    assert result.calibration_replay_source == result.metadata["calibration_replay_source"]
    assert result.conformal_guarantee_status["calibration_replay_source"] == result.metadata["calibration_replay_source"]
    np.testing.assert_allclose(result.prediction.interval(0.8).lower, [9.6])


def test_public_calibrate_replays_typed_dataset_backed_predictor_bundle(tmp_path, monkeypatch) -> None:
    from nirs4all.api.result import PredictResult

    predict_module = importlib.import_module("nirs4all.api.predict")
    calls = []

    def _fake_predict(**kwargs):
        calls.append(kwargs)
        data = kwargs["data"]
        assert kwargs["model"] == tmp_path / "model.n4a"
        assert list(data["sample_ids"]) == ["c1", "c2", "c3", "c4"]
        assert list(data["groups"]) == ["batch-a", "batch-a", "batch-b", "batch-b"]
        assert data["metadata"] == [{"site": "north"}, {"site": "north"}, {"site": "south"}, {"site": "south"}]
        return PredictResult(y_pred=np.asarray([0.1, 0.8, 2.4, 2.7], dtype=float))

    monkeypatch.setattr(predict_module, "predict", _fake_predict)

    result = nirs4all.calibrate(
        calibration_data=nirs4all.ConformalCalibrationData(
            dataset=_make_calibration_spectro_dataset(),
            selector={"partition": "calibration"},
            sample_id_column="sample_id",
            group_column="batch",
            metadata_columns=["site"],
            predictor_bundle=tmp_path / "model.n4a",
        ),
        y_pred=[10.0],
        prediction_sample_ids=["p1"],
        coverage=0.8,
    )

    assert len(calls) == 1
    assert result.artifact.predictor_fingerprint == str(tmp_path / "model.n4a")
    np.testing.assert_allclose(result.prediction.interval(0.8).lower, [9.6])


def test_public_calibrate_replays_dataset_backed_predictor_result_with_identities(monkeypatch) -> None:
    from nirs4all.api.result import PredictResult

    predict_module = importlib.import_module("nirs4all.api.predict")
    predictor_entry = {"id": "pred-77", "model_name": "PLSRegression", "chain_id": "chain-77"}
    calls = []

    def _fake_predict(**kwargs):
        calls.append(kwargs)
        data = kwargs["data"]
        assert kwargs["model"] == predictor_entry
        assert kwargs["workspace_path"] == "workspace-a"
        assert list(data["sample_ids"]) == ["c1", "c2", "c3", "c4"]
        assert list(data["groups"]) == ["batch-a", "batch-a", "batch-b", "batch-b"]
        assert data["metadata"] == [{"site": "north"}, {"site": "north"}, {"site": "south"}, {"site": "south"}]
        return PredictResult(y_pred=np.asarray([0.1, 0.8, 2.4, 2.7], dtype=float))

    monkeypatch.setattr(predict_module, "predict", _fake_predict)

    result = nirs4all.calibrate(
        calibration_data={
            "dataset": _make_calibration_spectro_dataset(),
            "selector": {"partition": "calibration"},
            "sample_id_column": "sample_id",
            "group_column": "batch",
            "metadata_columns": ["site"],
            "predictor_result": predictor_entry,
            "workspace_path": "workspace-a",
        },
        y_pred=[10.0],
        prediction_sample_ids=["p1"],
        coverage=0.8,
    )

    assert len(calls) == 1
    assert result.artifact.predictor_fingerprint == "chain_id:chain-77"
    np.testing.assert_allclose(result.prediction.interval(0.8).lower, [9.6])


def test_public_calibrate_replays_typed_dataset_backed_predictor_result(monkeypatch) -> None:
    from nirs4all.api.result import PredictResult

    predict_module = importlib.import_module("nirs4all.api.predict")
    predictor_entry = {"id": "pred-77", "model_name": "PLSRegression", "chain_id": "chain-77"}
    calls = []

    def _fake_predict(**kwargs):
        calls.append(kwargs)
        data = kwargs["data"]
        assert kwargs["model"] == predictor_entry
        assert kwargs["workspace_path"] == "workspace-a"
        assert list(data["sample_ids"]) == ["c1", "c2", "c3", "c4"]
        assert list(data["groups"]) == ["batch-a", "batch-a", "batch-b", "batch-b"]
        assert data["metadata"] == [{"site": "north"}, {"site": "north"}, {"site": "south"}, {"site": "south"}]
        return PredictResult(y_pred=np.asarray([0.1, 0.8, 2.4, 2.7], dtype=float))

    monkeypatch.setattr(predict_module, "predict", _fake_predict)

    result = nirs4all.calibrate(
        calibration_data=nirs4all.ConformalCalibrationData(
            dataset=_make_calibration_spectro_dataset(),
            selector={"partition": "calibration"},
            sample_id_column="sample_id",
            group_column="batch",
            metadata_columns=["site"],
            predictor_result=predictor_entry,
            workspace_path="workspace-a",
        ),
        y_pred=[10.0],
        prediction_sample_ids=["p1"],
        coverage=0.8,
    )

    assert len(calls) == 1
    assert result.artifact.predictor_fingerprint == "chain_id:chain-77"
    np.testing.assert_allclose(result.prediction.interval(0.8).lower, [9.6])


def test_public_calibrate_replays_dataset_backed_run_result_like_object(monkeypatch) -> None:
    from nirs4all.api.result import PredictResult

    class _RunResultLike:
        best = {"prediction_id": "pred-abc", "model_name": "Ridge"}

    predict_module = importlib.import_module("nirs4all.api.predict")

    def _fake_predict(**kwargs):
        assert kwargs["model"] == _RunResultLike.best
        return PredictResult(y_pred=np.asarray([0.1, 0.8, 2.4, 2.7], dtype=float))

    monkeypatch.setattr(predict_module, "predict", _fake_predict)

    result = nirs4all.calibrate(
        calibration_data={
            "dataset": _make_calibration_spectro_dataset(),
            "selector": {"partition": "calibration"},
            "sample_id_column": "sample_id",
            "predictor_result": _RunResultLike(),
        },
        y_pred=[10.0],
        prediction_sample_ids=["p1"],
        coverage=0.8,
    )

    assert result.artifact.predictor_fingerprint == "prediction_id:pred-abc"
    np.testing.assert_allclose(result.prediction.interval(0.8).lower, [9.6])


def test_public_calibrate_replays_dataset_backed_workspace_chain_with_identities(monkeypatch) -> None:
    from nirs4all.api.result import PredictResult

    predict_module = importlib.import_module("nirs4all.api.predict")
    calls = []

    def _fake_predict(**kwargs):
        calls.append(kwargs)
        data = kwargs["data"]
        assert kwargs["chain_id"] == "chain-42"
        assert kwargs["workspace_path"] == "workspace-a"
        assert "model" not in kwargs
        assert list(data["sample_ids"]) == ["c1", "c2", "c3", "c4"]
        assert list(data["groups"]) == ["batch-a", "batch-a", "batch-b", "batch-b"]
        assert data["metadata"] == [{"site": "north"}, {"site": "north"}, {"site": "south"}, {"site": "south"}]
        return PredictResult(y_pred=np.asarray([0.1, 0.8, 2.4, 2.7], dtype=float))

    monkeypatch.setattr(predict_module, "predict", _fake_predict)

    result = nirs4all.calibrate(
        calibration_data={
            "dataset": _make_calibration_spectro_dataset(),
            "selector": {"partition": "calibration"},
            "sample_id_column": "sample_id",
            "group_column": "batch",
            "metadata_columns": ["site"],
            "predictor_chain_id": "chain-42",
            "workspace_path": "workspace-a",
        },
        y_pred=[10.0],
        prediction_sample_ids=["p1"],
        coverage=0.8,
    )

    assert len(calls) == 1
    assert result.artifact.predictor_fingerprint == "workspace:workspace-a#chain:chain-42"
    np.testing.assert_allclose(result.prediction.interval(0.8).lower, [9.6])


def test_public_calibrate_replays_typed_dataset_backed_workspace_chain(monkeypatch) -> None:
    from nirs4all.api.result import PredictResult

    predict_module = importlib.import_module("nirs4all.api.predict")
    calls = []

    def _fake_predict(**kwargs):
        calls.append(kwargs)
        data = kwargs["data"]
        assert kwargs["chain_id"] == "chain-42"
        assert kwargs["workspace_path"] == "workspace-a"
        assert "model" not in kwargs
        assert list(data["sample_ids"]) == ["c1", "c2", "c3", "c4"]
        assert list(data["groups"]) == ["batch-a", "batch-a", "batch-b", "batch-b"]
        assert data["metadata"] == [{"site": "north"}, {"site": "north"}, {"site": "south"}, {"site": "south"}]
        return PredictResult(y_pred=np.asarray([0.1, 0.8, 2.4, 2.7], dtype=float))

    monkeypatch.setattr(predict_module, "predict", _fake_predict)

    result = nirs4all.calibrate(
        calibration_data=nirs4all.ConformalCalibrationData(
            dataset=_make_calibration_spectro_dataset(),
            selector={"partition": "calibration"},
            sample_id_column="sample_id",
            group_column="batch",
            metadata_columns=["site"],
            predictor_chain_id="chain-42",
            workspace_path="workspace-a",
        ),
        y_pred=[10.0],
        prediction_sample_ids=["p1"],
        coverage=0.8,
    )

    assert len(calls) == 1
    assert result.artifact.predictor_fingerprint == "workspace:workspace-a#chain:chain-42"
    np.testing.assert_allclose(result.prediction.interval(0.8).lower, [9.6])


def test_public_calibrate_dataset_backed_predictor_rejects_ambiguous_replayed_predictions() -> None:
    with pytest.raises(ValueError, match="exactly one of y_pred, predictor, predictor_bundle, predictor_result, or predictor_chain_id"):
        nirs4all.calibrate(
            calibration_data={
                "dataset": _make_calibration_spectro_dataset(),
                "selector": {"partition": "calibration"},
                "sample_id_column": "sample_id",
                "predictor": _CalibrationIdentityAwarePredictor(),
                "y_pred": [0.1, 0.8, 2.4, 2.7],
            },
            y_pred=[10.0],
            prediction_sample_ids=["p1"],
            coverage=0.8,
        )


def test_public_calibrate_dataset_backed_predictor_bundle_rejects_ambiguous_replayed_predictions(tmp_path) -> None:
    with pytest.raises(ValueError, match="exactly one of y_pred, predictor, predictor_bundle, predictor_result, or predictor_chain_id"):
        nirs4all.calibrate(
            calibration_data={
                "dataset": _make_calibration_spectro_dataset(),
                "selector": {"partition": "calibration"},
                "sample_id_column": "sample_id",
                "predictor_bundle": tmp_path / "model.n4a",
                "y_pred": [0.1, 0.8, 2.4, 2.7],
            },
            y_pred=[10.0],
            prediction_sample_ids=["p1"],
            coverage=0.8,
        )


def test_public_calibrate_dataset_backed_predictor_chain_requires_workspace_path() -> None:
    with pytest.raises(ValueError, match="predictor_chain_id requires an explicit workspace_path"):
        nirs4all.calibrate(
            calibration_data={
                "dataset": _make_calibration_spectro_dataset(),
                "selector": {"partition": "calibration"},
                "sample_id_column": "sample_id",
                "predictor_chain_id": "chain-42",
            },
            y_pred=[10.0],
            prediction_sample_ids=["p1"],
            coverage=0.8,
        )


def test_public_calibrate_dataset_backed_predictor_chain_id_is_canonical() -> None:
    for predictor_chain_id in ("", " chain-42 ", "bad\x00id", 123):
        with pytest.raises(ValueError, match="Dataset-backed calibration_data.predictor_chain_id must be a canonical non-empty string"):
            nirs4all.calibrate(
                calibration_data={
                    "dataset": _make_calibration_spectro_dataset(),
                    "selector": {"partition": "calibration"},
                    "sample_id_column": "sample_id",
                    "predictor_chain_id": predictor_chain_id,
                    "workspace_path": "workspace-a",
                },
                y_pred=[10.0],
                prediction_sample_ids=["p1"],
                coverage=0.8,
            )

    with pytest.raises(ValueError, match="Dataset-backed calibration_data.predictor_chain_id must be a canonical non-empty string"):
        nirs4all.calibrate(
            calibration_data={
                "dataset": _make_calibration_spectro_dataset(),
                "selector": {"partition": "calibration"},
                "sample_id_column": "sample_id",
                "workspace_chain_id": " chain-42 ",
                "workspace_path": "workspace-a",
            },
            y_pred=[10.0],
            prediction_sample_ids=["p1"],
            coverage=0.8,
        )


def test_public_calibrate_accepts_spectro_dataset_with_top_level_calibration_predictions() -> None:
    result = nirs4all.calibrate(
        calibration_data={
            "dataset": _make_calibration_spectro_dataset(),
            "selector": {"partition": "calibration"},
            "sample_id_column": "sample_id",
            "group_column": "batch",
            "metadata_columns": ["site"],
        },
        y_pred_calibration=[0.1, 0.8, 2.4, 2.7],
        y_pred=[10.0],
        prediction_sample_ids=["p1"],
        coverage=0.8,
    )

    assert result.artifact.calibration_cohort is not None
    assert result.artifact.calibration_cohort.sample_ids == ("c1", "c2", "c3", "c4")
    np.testing.assert_allclose(result.prediction.interval(0.8).lower, [9.6])


def test_public_calibrate_spectro_dataset_requires_selector() -> None:
    with pytest.raises(ValueError, match="explicit selector"):
        nirs4all.calibrate(
            calibration_data={
                "dataset": _make_calibration_spectro_dataset(),
                "sample_id_column": "sample_id",
                "y_pred": [0.1, 0.8, 2.4, 2.7],
            },
            y_pred=[10.0],
            prediction_sample_ids=["p1"],
            coverage=0.8,
        )


def test_public_calibrate_spectro_dataset_requires_replayed_predictions_or_predictor() -> None:
    with pytest.raises(ValueError, match="y_pred/y_pred_calibration or an explicit predictor replay source"):
        nirs4all.calibrate(
            calibration_data={
                "dataset": _make_calibration_spectro_dataset(),
                "selector": {"partition": "calibration"},
                "sample_id_column": "sample_id",
            },
            y_pred=[10.0],
            prediction_sample_ids=["p1"],
            coverage=0.8,
        )


def test_public_calibrate_spectro_dataset_rejects_mixed_truth_arrays() -> None:
    with pytest.raises(ValueError, match="must not also provide X/y_true"):
        nirs4all.calibrate(
            calibration_data={
                "dataset": _make_calibration_spectro_dataset(),
                "selector": {"partition": "calibration"},
                "sample_id_column": "sample_id",
                "y_true": [1.0, 2.0, 3.0, 4.0],
                "y_pred": [0.1, 0.8, 2.4, 2.7],
            },
            y_pred=[10.0],
            prediction_sample_ids=["p1"],
            coverage=0.8,
        )


def test_public_calibrate_accepts_predict_result_when_truth_is_explicit() -> None:
    calibration_predictions = PredictResult(
        y_pred=np.array([0.9, 2.2, 2.6, 4.3]),
        sample_indices=np.array(["c1", "c2", "c3", "c4"], dtype=object),
        metadata={"source": "unit"},
    )

    result = nirs4all.calibrate(
        calibration_data=calibration_predictions,
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[10.0],
        prediction_sample_ids=["p1"],
        coverage=0.8,
    )

    assert result.artifact.calibration_cohort is not None
    assert result.artifact.calibration_cohort.sample_ids == ("c1", "c2", "c3", "c4")
    np.testing.assert_allclose(result.prediction.interval(0.8).upper, [10.4])


def test_public_calibrate_rejects_prediction_dict_without_physical_sample_ids() -> None:
    with pytest.raises(Nirs4AllCalibrationNotImplementedError, match="already replayed arrays"):
        nirs4all.calibrate(
            calibration_data={
                "id": "pred-1",
                "y_true": [1.0, 2.0, 3.0, 4.0],
                "y_pred": [0.9, 2.2, 2.6, 4.3],
            },
            y_pred=[10.0],
            prediction_sample_ids=["p1"],
            coverage=0.8,
        )


def test_public_calibrate_can_return_predict_result_with_intervals() -> None:
    result = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        target_name="moisture",
        as_predict_result=True,
    )

    assert isinstance(result, PredictResult)
    assert result.model_name == "moisture"
    assert result.interval_coverages == (0.8,)
    assert list(result.sample_indices) == ["p1", "p2"]
    assert "conformal_artifact" in result.metadata
    assert "calibrated_result_fingerprint" in result.metadata
    assert result.conformal_guarantee_status is not None
    assert result.conformal_guarantee_status["unit"] == "physical_sample"
    assert result.conformal_guarantee_status["coverage"] == [0.8]
    assert result.calibration_replay_source is not None
    assert result.calibration_replay_source["kind"] == "explicit_replayed_arrays"
    np.testing.assert_allclose(result.interval(0.8).lower, [9.6, 19.6])


def test_public_calibrate_can_persist_and_reload_calibrated_result(tmp_path) -> None:
    store_path = tmp_path / "calibrated-store"

    result = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        store_path=store_path,
    )
    restored = nirs4all.load_calibrated_result(store_path)

    assert (store_path / "manifest.json").is_file()
    assert restored.to_dict() == result.to_dict()


def test_public_calibrate_can_persist_and_reload_workspace_result(tmp_path) -> None:
    workspace = tmp_path / "workspace"

    result = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        workspace_path=workspace,
        workspace_conformal_id="calib-main",
        workspace_name="main calibration",
        workspace_metadata={"purpose": "unit-test"},
    )
    restored = nirs4all.load_workspace_calibrated_result(workspace, "calib-main")
    restored_by_fingerprint = nirs4all.load_workspace_calibrated_result(workspace, result.fingerprint)
    restored_prediction = nirs4all.load_workspace_calibrated_predict_result(workspace, "calib-main")

    assert (workspace / "store.sqlite").is_file()
    assert restored.to_dict() == result.to_dict()
    assert restored_by_fingerprint.to_dict() == result.to_dict()
    assert isinstance(restored_prediction, PredictResult)
    np.testing.assert_allclose(restored_prediction.y_pred, [10.0, 20.0])
    np.testing.assert_allclose(restored_prediction.interval(0.8).lower, [9.6, 19.6])
    assert restored_prediction.metadata["calibrated_result_fingerprint"] == result.fingerprint


def test_public_calibrate_workspace_metadata_is_strict_json_native(tmp_path) -> None:
    workspace = tmp_path / "workspace"

    nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        workspace_path=workspace,
        workspace_conformal_id="calib-json",
        workspace_metadata={"site": "north", "nested": {"ok": [1, True, None]}},
    )

    for metadata, message in (
        ({" bad": 1}, "save_conformal_result.metadata keys must be canonical non-empty strings"),
        ({"bad": object()}, r"save_conformal_result.metadata\[bad\] must be JSON-native"),
        ({"bad": float("inf")}, r"save_conformal_result.metadata\[bad\] must be JSON-native and finite"),
        ({"bad": (1, 2)}, r"save_conformal_result.metadata\[bad\] must be JSON-native"),
    ):
        with pytest.raises(ValueError, match=message):
            nirs4all.calibrate(
                y_true=[1.0, 2.0, 3.0, 4.0],
                y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
                y_pred=[10.0, 20.0],
                calibration_sample_ids=["c1", "c2", "c3", "c4"],
                prediction_sample_ids=["p1", "p2"],
                coverage=0.8,
                workspace_path=workspace,
                workspace_metadata=metadata,
            )


def test_public_calibrate_workspace_ids_are_canonical(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    result = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
    )

    generated = nirs4all.save_workspace_calibrated_result(workspace, result)
    assert isinstance(generated, str)
    assert generated

    for conformal_id in ("", " calib-main ", "bad\x00id", 123):
        with pytest.raises(ValueError, match="save_conformal_result.conformal_id must be a canonical non-empty string"):
            nirs4all.save_workspace_calibrated_result(
                workspace,
                result,
                conformal_id=conformal_id,  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="save_conformal_result.conformal_id must be a canonical non-empty string"):
            nirs4all.calibrate(
                y_true=[1.0, 2.0, 3.0, 4.0],
                y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
                y_pred=[10.0, 20.0],
                calibration_sample_ids=["c1", "c2", "c3", "c4"],
                prediction_sample_ids=["p1", "p2"],
                coverage=0.8,
                workspace_path=workspace,
                workspace_conformal_id=conformal_id,  # type: ignore[arg-type]
            )


def test_public_calibrate_workspace_link_ids_are_canonical(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    result = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
    )

    for kwargs, label in (
        ({"run_id": " run-main "}, "save_conformal_result.run_id"),
        ({"pipeline_id": "pipe\x00main"}, "save_conformal_result.pipeline_id"),
        ({"chain_id": 123}, "save_conformal_result.chain_id"),
        ({"prediction_id": ""}, "save_conformal_result.prediction_id"),
    ):
        with pytest.raises(ValueError, match=f"{label} must be a canonical non-empty string"):
            nirs4all.save_workspace_calibrated_result(
                workspace,
                result,
                **kwargs,  # type: ignore[arg-type]
            )


def test_public_calibrate_result_metadata_is_strict_json_native() -> None:
    result = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1"],
        coverage=0.8,
        result_metadata={"site": "north", "nested": {"ok": [1, True, None]}},
    )
    assert result.metadata["site"] == "north"
    assert result.metadata["nested"] == {"ok": [1, True, None]}

    for metadata in ({" bad": 1}, {"bad": object()}, {"bad": float("nan")}, {"bad": (1, 2)}):
        with pytest.raises(ValueError, match=r"calibrate.result_metadata"):
            nirs4all.calibrate(
                y_true=[1.0, 2.0, 3.0, 4.0],
                y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
                y_pred=[10.0],
                calibration_sample_ids=["c1", "c2", "c3", "c4"],
                prediction_sample_ids=["p1"],
                coverage=0.8,
                result_metadata=metadata,
            )


def test_public_predict_calibrated_result_metadata_is_strict_json_native() -> None:
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1"],
        coverage=0.8,
    )
    predicted = nirs4all.predict_calibrated(
        calibrated,
        y_pred=[20.0],
        prediction_sample_ids=["p2"],
        result_metadata={"site": "north", "nested": {"ok": [1, True, None]}},
        as_predict_result=False,
    )
    assert isinstance(predicted, CalibratedRunResult)
    assert predicted.metadata["site"] == "north"
    assert predicted.metadata["nested"] == {"ok": [1, True, None]}

    for metadata in ({" bad": 1}, {"bad": object()}, {"bad": float("nan")}, {"bad": (1, 2)}):
        with pytest.raises(ValueError, match=r"predict_calibrated.result_metadata"):
            nirs4all.predict_calibrated(
                calibrated,
                y_pred=[20.0],
                prediction_sample_ids=["p2"],
                result_metadata=metadata,
            )


def test_public_calibrate_grouped_result_persists_and_reloads_from_workspace(tmp_path) -> None:
    workspace = tmp_path / "workspace"

    result = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 1.8, 2.5, 3.4],
        y_pred=[10.0, 20.0, 30.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2", "p3"],
        calibration_groups=["instrument-a", "instrument-a", "instrument-b", "instrument-b"],
        prediction_groups=["instrument-a", "instrument-b", "instrument-a"],
        group_by="group",
        coverage=0.5,
        workspace_path=workspace,
        workspace_conformal_id="grouped-calib",
    )

    restored = nirs4all.load_workspace_calibrated_result(workspace, "grouped-calib")
    restored_prediction = nirs4all.load_workspace_calibrated_predict_result(workspace, "grouped-calib")

    assert restored.to_dict() == result.to_dict()
    assert restored.prediction.group_keys == ('["instrument-a"]', '["instrument-b"]', '["instrument-a"]')
    np.testing.assert_allclose(restored.prediction.interval(0.5).qhat, [0.2, 0.6, 0.2])
    assert isinstance(restored_prediction, PredictResult)
    np.testing.assert_allclose(restored_prediction.interval(0.5).qhat, [0.2, 0.6, 0.2])
    assert restored_prediction.metadata["conformal_guarantee_status"]["guarantee"] == "split_conformal_group_marginal_coverage"


def test_workspace_calibrated_result_reload_rejects_missing_prediction_sample_ids(tmp_path) -> None:
    workspace = tmp_path / "workspace"

    nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        workspace_path=workspace,
        workspace_conformal_id="corrupt-missing-sample-ids",
    )

    db_path = workspace / "store.sqlite"
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT result_json FROM conformal_results WHERE conformal_id = ?",
            ("corrupt-missing-sample-ids",),
        ).fetchone()
        assert row is not None
        payload = json.loads(row[0])
        payload["sample_ids"] = []
        payload.pop("fingerprint", None)
        conn.execute(
            "UPDATE conformal_results SET result_json = ? WHERE conformal_id = ?",
            (json.dumps(payload, sort_keys=True), "corrupt-missing-sample-ids"),
        )

    with pytest.raises(ValueError, match="sample_ids are required"):
        nirs4all.load_workspace_calibrated_result(workspace, "corrupt-missing-sample-ids")
    with pytest.raises(ValueError, match="sample_ids are required"):
        nirs4all.load_workspace_calibrated_predict_result(workspace, "corrupt-missing-sample-ids")


def test_public_save_workspace_calibrated_result_returns_generated_id(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    result = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1"],
        coverage=0.8,
    )

    conformal_id = nirs4all.save_workspace_calibrated_result(workspace, result)

    assert isinstance(conformal_id, str)
    assert nirs4all.load_workspace_calibrated_result(workspace, conformal_id).to_dict() == result.to_dict()


def test_public_calibrate_can_export_and_reload_calibrated_result_bundle(tmp_path) -> None:
    bundle_path = tmp_path / "calibrated-result.n4a"

    result = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        bundle_path=bundle_path,
    )
    restored = nirs4all.load_calibrated_result(bundle_path)

    assert bundle_path.is_file()
    assert restored.to_dict() == result.to_dict()


def test_public_export_calibrated_result_writes_n4a_bundle(tmp_path) -> None:
    result = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1"],
        coverage=0.8,
    )
    bundle_path = tmp_path / "manual-export.n4a"

    saved = nirs4all.export_calibrated_result(result, bundle_path)
    restored = nirs4all.load_calibrated_result(bundle_path)

    assert saved == bundle_path
    assert restored.to_dict() == result.to_dict()


def test_public_predict_calibrated_applies_loaded_calibrator_to_new_predictions() -> None:
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p0"],
        coverage=0.8,
        target_name="moisture",
    )

    result = nirs4all.predict_calibrated(
        calibrated,
        y_pred=[30.0, 40.0],
        prediction_sample_ids=["p1", "p2"],
    )

    assert isinstance(result, PredictResult)
    assert result.model_name == "moisture"
    assert list(result.sample_indices) == ["p1", "p2"]
    assert "source_calibrated_result_fingerprint" in result.metadata
    assert result.conformal_guarantee_status is not None
    assert result.conformal_guarantee_status["effective_engine"] == "nirs4all.python.replayed_array_apply"
    assert result.conformal_guarantee_status["source_calibrated_result_fingerprint"] == calibrated.fingerprint
    assert result.calibration_replay_source == calibrated.calibration_replay_source
    assert result.conformal_guarantee_status["calibration_replay_source"] == calibrated.calibration_replay_source
    np.testing.assert_allclose(result.interval(0.8).lower, [29.6, 39.6])
    np.testing.assert_allclose(result.interval(0.8).upper, [30.4, 40.4])


def test_public_predict_calibrated_preserves_source_replay_provenance_over_user_metadata() -> None:
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p0"],
        coverage=0.8,
    )

    result = nirs4all.predict_calibrated(
        calibrated,
        y_pred=[30.0],
        prediction_sample_ids=["p1"],
        result_metadata={
            "calibration_replay_source": {"kind": "user-overwrite"},
            "conformal_guarantee_status": {"status": "user-overwrite"},
        },
    )

    assert result.calibration_replay_source == calibrated.calibration_replay_source
    assert result.conformal_guarantee_status is not None
    assert result.conformal_guarantee_status["calibration_replay_source"] == calibrated.calibration_replay_source
    assert result.conformal_guarantee_status["effective_engine"] == "nirs4all.python.replayed_array_apply"


def test_public_predict_calibrated_rejects_calibration_sample_reuse() -> None:
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p0"],
        coverage=0.8,
    )

    with pytest.raises(ValueError, match="overlapping physical samples: c2"):
        nirs4all.predict_calibrated(
            calibrated,
            y_pred=[30.0, 40.0],
            prediction_sample_ids=["p1", "c2"],
        )


def test_public_predict_calibrated_loads_bundle_source(tmp_path) -> None:
    bundle_path = tmp_path / "calibrated-result.n4a"
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p0"],
        coverage=0.8,
        bundle_path=str(bundle_path),
    )

    result = nirs4all.predict_calibrated(
        str(bundle_path),
        y_pred=[30.0],
        prediction_sample_ids=["p1"],
        as_predict_result=False,
    )

    assert isinstance(result, CalibratedRunResult)
    assert result.artifact.fingerprint == calibrated.artifact.fingerprint
    assert result.sample_ids == ("p1",)
    np.testing.assert_allclose(result.prediction.interval(0.8).upper, [30.4])


def test_public_predict_calibrated_rejects_bad_prediction_ids() -> None:
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p0"],
        coverage=0.8,
    )

    with pytest.raises(ValueError, match="length"):
        nirs4all.predict_calibrated(calibrated, y_pred=[30.0, 40.0], prediction_sample_ids=["p1"])

    for prediction_sample_ids in ([" p1"], ["p1 "], ["bad\x00id"], [123]):
        with pytest.raises(ValueError, match="prediction_sample_ids must contain canonical non-empty strings"):
            nirs4all.predict_calibrated(
                calibrated,
                y_pred=[30.0],
                prediction_sample_ids=prediction_sample_ids,
            )


def test_public_conformal_metrics_accepts_calibrated_result_path_and_predict_result(tmp_path) -> None:
    bundle_path = tmp_path / "calibrated-result.n4a"
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0, 20.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        coverage=0.8,
        bundle_path=str(bundle_path),
    )
    predict_result = nirs4all.predict_calibrated(
        calibrated,
        y_pred=[10.0, 20.0],
        prediction_sample_ids=["p1", "p2"],
    )

    from_result = nirs4all.conformal_metrics(calibrated, y_true=[10.1, 21.0])
    from_path = nirs4all.conformal_metrics(str(bundle_path), y_true=[10.1, 21.0])
    from_predict = nirs4all.conformal_metrics(predict_result, y_true=[10.1, 21.0])

    assert from_result[0.8].observed_coverage == pytest.approx(0.5)
    assert from_result[0.8].mean_width == pytest.approx(0.8)
    assert from_result[0.8].mean_interval_score == pytest.approx(3.8)
    assert from_path[0.8].to_dict() == from_result[0.8].to_dict()
    assert from_predict[0.8].to_dict() == from_result[0.8].to_dict()


def test_public_conformal_metrics_rejects_predict_result_without_intervals() -> None:
    with pytest.raises(ValueError, match="does not contain conformal intervals"):
        nirs4all.conformal_metrics(PredictResult(y_pred=np.array([1.0])), y_true=[1.0])


def test_public_predict_routes_calibrated_replayed_array_request() -> None:
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p0"],
        coverage=[0.5, 0.8],
        target_name="moisture",
    )

    result = nirs4all.predict(
        model=calibrated,
        data={"y_pred": [30.0, 40.0], "sample_ids": ["p1", "p2"], "metadata": {"batch": "new"}},
        coverage=0.8,
    )

    assert isinstance(result, PredictResult)
    assert result.model_name == "moisture"
    assert result.interval_coverages == (0.8,)
    assert result.metadata["selected_interval_coverages"] == [0.8]
    assert result.metadata["batch"] == "new"
    assert result.conformal_guarantee_status is not None
    assert result.conformal_guarantee_status["coverage"] == [0.8]
    assert result.conformal_guarantee_status["selected_coverages"] == [0.8]
    np.testing.assert_allclose(result.interval(0.8).lower, [29.6, 39.6])


def test_public_predict_rejects_invalid_coverage_selector_values() -> None:
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p0"],
        coverage=[0.5, 0.8],
    )

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        nirs4all.predict(
            model=calibrated,
            data={"y_pred": [30.0], "sample_ids": ["p1"]},
            coverage=1.0,
        )


def test_public_predict_rejects_duplicate_coverage_selector_values() -> None:
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p0"],
        coverage=[0.5, 0.8],
    )

    with pytest.raises(ValueError, match="unique"):
        nirs4all.predict(
            model=calibrated,
            data={"y_pred": [30.0], "sample_ids": ["p1"]},
            coverage=[0.8, 0.8],
        )


def test_public_predict_rejects_unmaterialized_coverage_selector() -> None:
    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p0"],
        coverage=[0.5, 0.8],
    )

    with pytest.raises(ValueError, match=r"coverage 0\.9 was not materialized; available coverages: 0\.5, 0\.8"):
        nirs4all.predict(
            model=calibrated,
            data={"y_pred": [30.0], "sample_ids": ["p1"]},
            coverage=0.9,
        )


def test_public_predict_routes_calibrated_bundle_request(tmp_path) -> None:
    bundle_path = tmp_path / "calibrated-result.n4a"
    nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p0"],
        coverage=0.8,
        bundle_path=str(bundle_path),
    )

    result = nirs4all.predict(
        model=str(bundle_path),
        data={"y_pred": [30.0], "prediction_sample_ids": ["p1"]},
        coverage=0.8,
    )

    assert result.interval_coverages == (0.8,)
    np.testing.assert_allclose(result.interval(0.8).upper, [30.4])


def test_public_attach_calibrated_result_to_model_bundle_roundtrips(tmp_path) -> None:
    model_bundle = tmp_path / "model.n4a"
    with zipfile.ZipFile(model_bundle, "w") as archive:
        archive.writestr("manifest.json", '{"bundle_format_version": "1.0"}')
        archive.writestr("pipeline.json", "{}")

    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p0"],
        coverage=0.8,
    )

    attached = nirs4all.attach_calibrated_result_to_bundle(model_bundle, calibrated)
    restored = nirs4all.load_calibrated_result(attached)

    assert attached == tmp_path / "model.calibrated.n4a"
    assert restored.to_dict() == calibrated.to_dict()
    with zipfile.ZipFile(attached, "r") as archive:
        members = set(archive.namelist())
    assert "manifest.json" in members
    assert "pipeline.json" in members
    assert "conformal/manifest.json" in members
    assert "conformal/artifact.json" in members
    assert "conformal/calibrated_result.json" in members


def test_public_predict_routes_attached_conformal_model_bundle(tmp_path, monkeypatch) -> None:
    model_bundle = tmp_path / "model.n4a"
    with zipfile.ZipFile(model_bundle, "w") as archive:
        archive.writestr("manifest.json", '{"bundle_format_version": "1.0"}')
        archive.writestr("pipeline.json", "{}")

    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p0"],
        coverage=[0.5, 0.8],
        target_name="moisture",
    )
    attached = nirs4all.attach_calibrated_result_to_bundle(str(model_bundle), calibrated)

    def fake_predict_from_model(**_kwargs):
        return PredictResult(
            y_pred=np.array([30.0, 40.0]),
            metadata={"raw": "ok"},
            model_name="bundle_model",
            preprocessing_steps=["SNV"],
        )

    predict_module = importlib.import_module("nirs4all.api.predict")
    monkeypatch.setattr(predict_module, "_predict_from_model", fake_predict_from_model)

    result = nirs4all.predict(
        model=str(attached),
        data={"X": np.ones((2, 3)), "sample_ids": ["p1", "p2"]},
        coverage=0.8,
    )

    assert result.model_name == "moisture"
    assert result.preprocessing_steps == ["SNV"]
    assert result.interval_coverages == (0.8,)
    assert result.metadata["raw"] == "ok"
    assert result.metadata["point_prediction_model_name"] == "bundle_model"
    assert result.conformal_guarantee_status is not None
    assert result.conformal_guarantee_status["coverage"] == [0.8]
    np.testing.assert_allclose(result.interval(0.8).lower, [29.6, 39.6])


def test_public_predict_attached_conformal_model_bundle_rejects_invalid_sidecar(tmp_path) -> None:
    model_bundle = tmp_path / "model.n4a"
    with zipfile.ZipFile(model_bundle, "w") as archive:
        archive.writestr("manifest.json", '{"bundle_format_version": "1.0"}')
        archive.writestr("pipeline.json", "{}")

    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p0"],
        coverage=0.8,
    )
    attached = nirs4all.attach_calibrated_result_to_bundle(str(model_bundle), calibrated)
    with zipfile.ZipFile(attached, "a") as archive:
        archive.writestr("conformal/extra.json", "{}")

    with pytest.raises(ValueError, match="unexpected conformal sidecar members"):
        nirs4all.predict(
            model=str(attached),
            data={"X": np.ones((1, 3)), "sample_ids": ["p1"]},
            coverage=0.8,
        )


def test_public_predict_attached_conformal_model_bundle_rejects_missing_prediction_sample_ids(tmp_path, monkeypatch) -> None:
    model_bundle = tmp_path / "model.n4a"
    with zipfile.ZipFile(model_bundle, "w") as archive:
        archive.writestr("manifest.json", '{"bundle_format_version": "1.0"}')
        archive.writestr("pipeline.json", "{}")

    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p0"],
        coverage=0.8,
    )
    attached = nirs4all.attach_calibrated_result_to_bundle(str(model_bundle), calibrated)
    corrupted = tmp_path / "model.corrupted-missing-sample-ids.n4a"
    with zipfile.ZipFile(attached, "r") as src, zipfile.ZipFile(corrupted, "w") as dst:
        for member in src.namelist():
            if member == "conformal/calibrated_result.json":
                payload = json.loads(src.read(member).decode("utf-8"))
                payload["sample_ids"] = []
                payload.pop("fingerprint", None)
                dst.writestr(member, json.dumps(payload, sort_keys=True))
            else:
                dst.writestr(member, src.read(member))

    def fail_if_model_prediction_runs(**_kwargs):
        raise AssertionError("invalid conformal sidecar must fail before model prediction")

    predict_module = importlib.import_module("nirs4all.api.predict")
    monkeypatch.setattr(predict_module, "_predict_from_model", fail_if_model_prediction_runs)

    with pytest.raises(ValueError, match="sample_ids are required"):
        nirs4all.predict(
            model=str(corrupted),
            data={"X": np.ones((1, 3)), "sample_ids": ["p1"]},
            coverage=0.8,
        )


def test_public_predict_attached_conformal_model_bundle_rejects_incomplete_sidecar(tmp_path) -> None:
    model_bundle = tmp_path / "model.n4a"
    with zipfile.ZipFile(model_bundle, "w") as archive:
        archive.writestr("manifest.json", '{"bundle_format_version": "1.0"}')
        archive.writestr("pipeline.json", "{}")
        archive.writestr("conformal/manifest.json", "{}")

    with pytest.raises(ValueError, match="does not contain a complete conformal sidecar"):
        nirs4all.predict(
            model=str(model_bundle),
            data={"X": np.ones((1, 3)), "sample_ids": ["p1"]},
            coverage=0.8,
        )


def test_public_predict_attached_conformal_model_bundle_rejects_duplicate_sidecar(tmp_path) -> None:
    model_bundle = tmp_path / "model.n4a"
    with zipfile.ZipFile(model_bundle, "w") as archive:
        archive.writestr("manifest.json", '{"bundle_format_version": "1.0"}')
        archive.writestr("pipeline.json", "{}")

    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p0"],
        coverage=0.8,
    )
    attached = nirs4all.attach_calibrated_result_to_bundle(str(model_bundle), calibrated)
    with zipfile.ZipFile(attached, "a") as archive, pytest.warns(UserWarning, match="Duplicate name"):
        archive.writestr("conformal/artifact.json", "{}")

    with pytest.raises(ValueError, match="duplicate members"):
        nirs4all.predict(
            model=str(attached),
            data={"X": np.ones((1, 3)), "sample_ids": ["p1"]},
            coverage=0.8,
        )


def test_public_predict_model_bundle_without_conformal_sidecar_stays_outside_coverage_lane(tmp_path) -> None:
    model_bundle = tmp_path / "model.n4a"
    with zipfile.ZipFile(model_bundle, "w") as archive:
        archive.writestr("manifest.json", '{"bundle_format_version": "1.0"}')
        archive.writestr("pipeline.json", "{}")

    with pytest.raises(NotImplementedError, match="attached conformal sidecar"):
        nirs4all.predict(
            model=str(model_bundle),
            data={"X": np.ones((1, 3)), "sample_ids": ["p1"]},
            coverage=0.8,
        )


def test_public_predict_attached_conformal_model_bundle_rejects_unmaterialized_coverage(tmp_path, monkeypatch) -> None:
    model_bundle = tmp_path / "model.n4a"
    with zipfile.ZipFile(model_bundle, "w") as archive:
        archive.writestr("manifest.json", '{"bundle_format_version": "1.0"}')
        archive.writestr("pipeline.json", "{}")

    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p0"],
        coverage=[0.5, 0.8],
    )
    attached = nirs4all.attach_calibrated_result_to_bundle(str(model_bundle), calibrated)

    def fake_predict_from_model(**_kwargs):
        return PredictResult(y_pred=np.array([30.0]), metadata={}, model_name="bundle_model")

    predict_module = importlib.import_module("nirs4all.api.predict")
    monkeypatch.setattr(predict_module, "_predict_from_model", fake_predict_from_model)

    with pytest.raises(ValueError, match=r"coverage 0\.9 was not materialized; available coverages: 0\.5, 0\.8"):
        nirs4all.predict(
            model=str(attached),
            data={"X": np.ones((1, 3)), "sample_ids": ["p1"]},
            coverage=0.9,
        )


def test_public_predict_attached_conformal_model_bundle_rejects_all_predictions(tmp_path) -> None:
    model_bundle = tmp_path / "model.n4a"
    with zipfile.ZipFile(model_bundle, "w") as archive:
        archive.writestr("manifest.json", '{"bundle_format_version": "1.0"}')
        archive.writestr("pipeline.json", "{}")

    calibrated = nirs4all.calibrate(
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0],
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p0"],
        coverage=0.8,
    )
    attached = nirs4all.attach_calibrated_result_to_bundle(str(model_bundle), calibrated)

    with pytest.raises(NotImplementedError, match="all_predictions=False"):
        nirs4all.predict(
            model=str(attached),
            data={"X": np.ones((2, 3)), "sample_ids": ["p1", "p2"]},
            coverage=0.8,
            all_predictions=True,
        )


def test_public_predict_coverage_without_calibrated_request_fails_closed() -> None:
    with pytest.raises(NotImplementedError, match="calibrated replayed-array"):
        nirs4all.predict(model={"model_name": "plain"}, data=np.array([[1.0, 2.0]]), coverage=0.8)


def test_public_predict_rejects_y_pred_data_with_non_calibrated_model() -> None:
    with pytest.raises(TypeError, match="CalibratedRunResult"):
        nirs4all.predict(model={"model_name": "plain"}, data={"y_pred": [1.0], "sample_ids": ["p1"]})


def test_public_calibrate_missing_replayed_arrays_fails_with_structured_error() -> None:
    with pytest.raises(Nirs4AllCalibrationNotImplementedError, match="already replayed arrays") as excinfo:
        nirs4all.calibrate(coverage=0.8)

    assert excinfo.value.calibration_spec.coverage == (0.8,)
    assert "automatic predictor lookup for dataset/replay calibration forms" in excinfo.value.missing_gates


def test_public_calibrate_rejects_unsupported_dataset_calibration_source() -> None:
    with pytest.raises(Nirs4AllCalibrationNotImplementedError, match="missing gates"):
        nirs4all.calibrate(
            calibration_data={"dataset": object()},
            y_pred=[10.0],
            prediction_sample_ids=["p1"],
            coverage=0.8,
        )
