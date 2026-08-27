"""Transition-release backend selector coverage for public helper APIs."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.api.explain import explain
from nirs4all.api.native_result import NativeMethodsRunResult
from nirs4all.api.native_retrain_lineage import NativeRetrainLineage
from nirs4all.api.predict import predict
from nirs4all.api.retrain import retrain
from nirs4all.api.run import run
from nirs4all.pipeline.dagml.native_archive_replay import (
    NativeArchiveConformalInterval,
    NativeArchivePrediction,
)
from nirs4all.pipeline.engine import require_legacy_engine


def test_require_legacy_engine_accepts_legacy() -> None:
    assert require_legacy_engine("predict", "legacy") == "legacy"


def test_run_native_executes_the_methods_subset_without_constructing_a_legacy_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public native lane never constructs ``PipelineRunner``."""

    def legacy_runner(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("run(engine='native') constructed a legacy PipelineRunner")

    run_module = importlib.import_module("nirs4all.api.run")
    monkeypatch.setattr(run_module, "PipelineRunner", legacy_runner)

    class Estimator:
        training_outcome_ = {
            "outcome_fingerprint": "a" * 64,
            "score_set": {
                "schema_version": 2,
                "selection_metric": "rmse",
                "reports": [
                    {
                        "producer_node": "model:methods",
                        "producer_port": "oof",
                        "partition": "validation",
                        "fold_id": "fold0",
                        "level": "sample",
                        "metrics": {"rmse": 0.5},
                        "row_count": 2,
                        "target_names": ["y"],
                        "target_width": 1,
                        "variant_id": "variant:base",
                    },
                    {
                        "producer_node": "model:methods",
                        "producer_port": "oof",
                        "partition": "validation",
                        "fold_id": "avg",
                        "level": "sample",
                        "metrics": {"rmse": 0.5},
                        "row_count": 2,
                        "target_names": ["y"],
                        "target_width": 1,
                        "variant_id": "variant:base",
                    },
                ],
            },
        }

        def export_native_archive(self, *_args, **_kwargs):  # noqa: ANN001
            raise AssertionError("training must not export during run")

    observed: dict[str, object] = {}

    def native_fit(pipeline, X, y, **kwargs):  # noqa: ANN001
        observed.update(pipeline=pipeline, X=np.asarray(X), y=np.asarray(y), kwargs=kwargs)
        return Estimator()

    monkeypatch.setattr("nirs4all.api.native_training.fit_native_pipeline", native_fit)

    result = run(
        [{"split": "stub"}, {"model": "stub"}],
        {"X": np.asarray([[1.0], [2.0]]), "y": np.asarray([1.0, 2.0]), "sample_ids": ["s1", "s2"]},
        engine="native",
        save_charts=False,
    )

    assert isinstance(result, NativeMethodsRunResult)
    assert nirs4all.NativeMethodsRunResult is NativeMethodsRunResult
    assert result.cv_best_score == pytest.approx(0.5)
    assert observed["kwargs"] == {
        "sample_ids": ["s1", "s2"],
        "groups": None,
        "metadata": None,
        "seed": 12345,
    }


def test_run_native_environment_selection_is_also_fail_closed_for_unsupported_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("N4A_ENGINE", "native")

    with pytest.raises(ValueError, match="missing required keys"):
        run([], {})


def test_run_native_attaches_identity_bound_conformal_calibration_without_legacy_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public native lane transports one attested replay to DAG-ML."""

    outcome = {
        "outcome_fingerprint": "c" * 64,
        "score_set": {
            "schema_version": 2,
            "selection_metric": "rmse",
            "reports": [
                {
                    "producer_node": "model:methods",
                    "producer_port": "oof",
                    "partition": "validation",
                    "fold_id": "avg",
                    "level": "sample",
                    "metrics": {"rmse": 0.25},
                    "row_count": 2,
                    "target_names": ["y"],
                    "target_width": 1,
                    "variant_id": "variant:base",
                }
            ],
        },
    }
    attached: dict[str, object] = {}
    replay_outcome = {"replay": "native-attested"}

    class TrainingResult:
        @property
        def outcome(self):  # noqa: ANN201
            return outcome

        def attach_conformal_calibration(self, replay, **kwargs):  # noqa: ANN001
            assert replay is replay_outcome
            attached.update(kwargs)
            outcome["conformal_calibration"] = {"schema_version": 2, "binding_id": kwargs["binding_id"]}
            return outcome["conformal_calibration"]

        def export_portable_predictor_package(self, package_id, **kwargs):  # noqa: ANN001
            assert package_id == "package:native"
            assert kwargs == {"fitted_artifact_mode": "portable_required", "artifact_load_mode": "native_portable"}
            return {"package_id": package_id, "schema_version": 2, "reexported": True}

    class Estimator:
        dagml_module = "fake_dag_ml"
        predictor_package_ = {"package_id": "package:native", "schema_version": 2}
        prediction_compiler = SimpleNamespace(methods_library_path="/native/libn4m.so")

        def __init__(self) -> None:
            self.training_result_ = TrainingResult()
            self.training_outcome_ = outcome

        def execute_compiled_replay(self, execution):  # noqa: ANN001
            assert execution == "compiled-calibration-replay"
            return replay_outcome

    estimator = Estimator()
    compile_observed: dict[str, object] = {}

    def compile_calibration(package, X, y, **kwargs):  # noqa: ANN001
        assert package is estimator.predictor_package_
        compile_observed.update(X=np.asarray(X), y=np.asarray(y), kwargs=kwargs)
        return SimpleNamespace(
            execution="compiled-calibration-replay",
            binding_id="binding:prediction",
            calibration_relations={"records": [{"sample_id": "cal-1"}, {"sample_id": "cal-2"}]},
            truth={"sample_ids": ["cal-1", "cal-2"], "values": [[1.5], [2.5]]},
        )

    monkeypatch.setattr("nirs4all.api.native_training.fit_native_pipeline", lambda *_args, **_kwargs: estimator)
    monkeypatch.setattr("nirs4all.api.native_training.compile_methods_conformal_calibration_replay", compile_calibration)
    monkeypatch.setattr("nirs4all.api.native_training.validate_native_methods_package", lambda package: package)
    monkeypatch.setattr(
        importlib.import_module("nirs4all.api.run"),
        "PipelineRunner",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("native calibration constructed a legacy runner")),
    )

    result = run(
        [{"split": "stub"}, {"model": "stub"}],
        {"X": np.asarray([[1.0], [2.0]]), "y": np.asarray([1.0, 2.0]), "sample_ids": ["train-1", "train-2"]},
        engine="native",
        save_charts=False,
        calibration={
            "X": np.asarray([[3.0], [4.0]]),
            "y": np.asarray([1.5, 2.5]),
            "sample_ids": ["cal-1", "cal-2"],
            "coverages": [0.8, 0.95],
        },
    )

    assert isinstance(result, NativeMethodsRunResult)
    assert np.array_equal(compile_observed["X"], np.asarray([[3.0], [4.0]]))
    assert np.array_equal(compile_observed["y"], np.asarray([1.5, 2.5]))
    assert compile_observed["kwargs"] == {
        "sample_ids": ["cal-1", "cal-2"],
        "groups": None,
        "metadata": None,
        "methods_library_path": "/native/libn4m.so",
        "dagml_module": "fake_dag_ml",
    }
    assert attached == {
        "binding_id": "binding:prediction",
        "calibration_relations": {"records": [{"sample_id": "cal-1"}, {"sample_id": "cal-2"}]},
        "truth": {"sample_ids": ["cal-1", "cal-2"], "values": [[1.5], [2.5]]},
        "coverages": [0.8, 0.95],
        "multi_target_policy": '"marginal"',
        "small_sample_policy": '"error"',
    }
    assert estimator.predictor_package_ == {"package_id": "package:native", "schema_version": 2, "reexported": True}
    assert result.native_conformal_calibration == {"schema_version": 2, "binding_id": "binding:prediction"}


@pytest.mark.parametrize(
    ("calibration", "message"),
    [
        ({}, "missing required keys"),
        ({"X": [], "y": [], "sample_ids": [], "coverages": [0.9, 0.9]}, "non-empty and unique"),
        ({"X": [], "y": [], "sample_ids": [], "coverages": [1.0]}, "strictly between zero and one"),
        ({"X": [], "y": [], "sample_ids": [], "coverages": [0.9], "legacy": True}, "unsupported keys"),
    ],
)
def test_run_native_refuses_ambiguous_conformal_calibration_before_fit(
    monkeypatch: pytest.MonkeyPatch,
    calibration: dict[str, object],
    message: str,
) -> None:
    monkeypatch.setattr(
        "nirs4all.api.native_training.fit_native_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("native fit was reached")),
    )
    with pytest.raises((TypeError, ValueError), match=message):
        run(
            [{"split": "stub"}, {"model": "stub"}],
            {"X": np.asarray([[1.0], [2.0]]), "y": np.asarray([1.0, 2.0]), "sample_ids": ["train-1", "train-2"]},
            engine="native",
            save_charts=False,
            calibration=calibration,
        )


def test_run_native_routes_strict_methods_hpo_through_the_native_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public HPO lane is scheduler metadata, never a Python objective."""

    class Estimator:
        training_outcome_ = {
            "outcome_fingerprint": "b" * 64,
            "selected_variant_id": "hpo:trial:0",
            "methods_hpo_resume_state": {"schema_version": 1, "checkpoint": {"format": "N4MOPT"}},
            "score_set": {
                "schema_version": 2,
                "selection_metric": "rmse",
                "reports": [
                    {
                        "producer_node": "model:methods",
                        "producer_port": "oof",
                        "partition": "validation",
                        "fold_id": "avg",
                        "level": "sample",
                        "metrics": {"rmse": 0.25},
                        "row_count": 2,
                        "target_names": ["y"],
                        "target_width": 1,
                        "variant_id": "hpo:trial:0",
                    }
                ],
            },
        }

    observed: dict[str, object] = {}

    def native_fit(*_args, **kwargs):  # noqa: ANN002, ANN003
        observed.update(kwargs)
        return Estimator()

    monkeypatch.setattr("nirs4all.api.native_training.fit_native_pipeline", native_fit)
    result = run(
        [{"split": "stub"}, {"model": "stub"}],
        {"X": np.asarray([[1.0], [2.0]]), "y": np.asarray([1.0, 2.0]), "sample_ids": ["s1", "s2"]},
        engine="native",
        save_charts=False,
        random_state=17,
        tuning={"engine": "methods-hpo", "trials": 3},
    )

    assert isinstance(result, NativeMethodsRunResult)
    assert result.native_selected_variant_id == "hpo:trial:0"
    assert result.native_methods_hpo_resume_state == {"schema_version": 1, "checkpoint": {"format": "N4MOPT"}}
    operation = observed["methods_hpo_operation"]
    assert operation == {
        "operation_id": "hpo:methods",
        "study": {
            "controller_id": "controller:methods.hpo",
            "study_id": "study:nirs4all.native.pls",
            "methods_abi": "n4m-abi-2.2",
            "search_space": {
                "parameters": [
                    {
                        "kind": "int",
                        "name": "n_components",
                        "low": 1,
                        "high": 3,
                        "step": 1,
                        "log": False,
                    }
                ]
            },
            "optimizer": {
                "sampler": "random",
                "pruner": "none",
                "direction": "minimize",
                "metric": "rmse",
                "seed": 17,
                "n_startup_trials": 0,
                "max_resource": 0,
                "reduction_factor": 0,
            },
        },
        "trials": 3,
        "parameter_paths": {"n_components": "n_components"},
    }


@pytest.mark.parametrize(
    "tuning, message",
    [
        ({"trials": 2}, "requires engine='methods-hpo'"),
        ({"engine": "methods-hpo", "trials": 0}, "integer in 1..64"),
        ({"engine": "methods-hpo", "trials": 2, "score_data": {}}, "unsupported keys"),
        ({"engine": "methods-hpo", "trials": 2, "sampler": "sobol"}, "sampler is unsupported"),
        ({"engine": "methods-hpo", "trials": 2, "pruner": "asha"}, "pruner is unsupported"),
    ],
)
def test_run_native_refuses_generic_or_partial_tuning_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tuning: dict[str, object],
    message: str,
) -> None:
    monkeypatch.setattr(
        "nirs4all.api.native_training.fit_native_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("native execution was reached")),
    )
    with pytest.raises(ValueError, match=message):
        run(
            [{"split": "stub"}, {"model": "stub"}],
            {"X": np.asarray([[1.0], [2.0]]), "y": np.asarray([1.0, 2.0]), "sample_ids": ["s1", "s2"]},
            engine="native",
            save_charts=False,
            tuning=tuning,
        )


def test_native_tpe_median_hpo_descriptor_is_closed_and_attested() -> None:
    """TPE/Median is native scheduler configuration, never a Python objective."""

    from nirs4all.api.native_training import _native_methods_hpo_operation

    operation = _native_methods_hpo_operation(
        {"engine": "methods-hpo", "trials": 4, "sampler": "tpe", "pruner": "median"},
        seed=51,
    )

    assert operation is not None
    assert operation["trials"] == 4
    assert operation["study"]["optimizer"] == {
        "sampler": "tpe",
        "pruner": "median",
        "direction": "minimize",
        "metric": "rmse",
        "seed": 51,
        "n_startup_trials": 2,
        "max_resource": 0,
        "reduction_factor": 0,
    }


@pytest.mark.parametrize(
    ("pipeline", "dataset", "kwargs", "message"),
    [
        ({"model": "stub"}, {"X": [[1.0]], "y": [1.0], "sample_ids": ["s1"]}, {}, "list pipeline"),
        ([], (np.asarray([[1.0]]), np.asarray([1.0])), {}, "explicit mapping dataset"),
        ([], {"X": [[1.0]], "y": [1.0], "sample_ids": ["s1"]}, {"refit": {"mode": "full"}}, "requires refit=True"),
    ],
)
def test_run_native_rejects_broad_legacy_shapes_before_native_execution(
    monkeypatch: pytest.MonkeyPatch,
    pipeline,
    dataset,
    kwargs,
    message,
) -> None:
    monkeypatch.setattr(
        "nirs4all.api.native_training.run_native_methods",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("native execution was reached")),
    )

    with pytest.raises((TypeError, NotImplementedError), match=message):
        run(pipeline, dataset, engine="native", save_charts=False, **kwargs)


@pytest.mark.parametrize(
    ("operation", "call"),
    [
        (
            "predict",
            lambda: predict(model={"model_name": "dummy"}, data=np.zeros((2, 3)), engine="dag-ml"),
        ),
        (
            "predict",
            lambda: predict(chain_id="chain-1", data=np.zeros((2, 3)), engine="dag-ml"),
        ),
        (
            "explain",
            lambda: explain({"model_name": "dummy"}, np.zeros((2, 3)), engine="dag-ml"),
        ),
        (
            "retrain",
            lambda: retrain({"model_name": "dummy"}, (np.zeros((2, 3)), np.zeros(2)), engine="dag-ml"),
        ),
    ],
)
def test_public_helpers_reject_dagml_until_native_paths_exist(operation: str, call) -> None:
    with pytest.raises(NotImplementedError, match=rf"nirs4all\.{operation}.*dag-ml"):
        call()


def test_retrain_native_refits_the_attested_selected_methods_variant_without_legacy_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native full retrain delegates one V3 operation without a legacy runner."""

    class Estimator:
        pass

    source = object.__new__(NativeMethodsRunResult)
    source._native_estimator = Estimator()  # noqa: SLF001
    observed: dict[str, object] = {}
    expected = object()

    def native_retrain(source_arg, dataset, *, name):  # noqa: ANN001
        observed.update(source=source_arg, dataset=dataset, name=name)
        return expected

    retrain_module = importlib.import_module("nirs4all.api.retrain")
    monkeypatch.setattr(retrain_module, "refit_native_methods", native_retrain)
    monkeypatch.setattr(
        retrain_module,
        "PipelineRunner",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("native retrain constructed PipelineRunner")),
    )

    dataset = {
        "X": np.asarray([[1.0], [2.0], [3.0], [4.0]]),
        "y": np.asarray([1.0, 2.0, 3.0, 4.0]),
        "sample_ids": ["next-0", "next-1", "next-2", "next-3"],
    }
    assert retrain(source, dataset, name="next") is expected
    assert observed["source"] is source
    assert observed["dataset"] is dataset
    assert observed["name"] == "next"




@pytest.mark.parametrize("engine", ["legacy", "dag-ml", "dual"])
def test_retrain_native_source_refuses_explicit_non_native_engine(engine: str) -> None:
    source = object.__new__(NativeMethodsRunResult)
    with pytest.raises(ValueError, match="explicit non-native engine"):
        retrain(source, {"X": [], "y": [], "sample_ids": []}, engine=engine)


@pytest.mark.parametrize(
    ("mode", "message"),
    [("transfer", "only mode='full'"), ("finetune", "only mode='full'")],
)
def test_retrain_native_refuses_unqualified_modes_before_execution(mode: str, message: str) -> None:
    source = object.__new__(NativeMethodsRunResult)
    with pytest.raises(NotImplementedError, match=message):
        retrain(source, {"X": [], "y": [], "sample_ids": []}, engine="native", mode=mode)


def test_predict_native_archive_is_explicit_and_never_constructs_a_legacy_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def native_predict(path, X, *, sample_ids, methods_library_path, groups, metadata):  # noqa: ANN001
        observed.update(path=str(path), X=np.asarray(X), sample_ids=list(sample_ids), methods_library_path=methods_library_path, groups=groups, metadata=metadata)
        return NativeArchivePrediction(
            values=np.asarray([[2.0], [3.0]]),
            sample_ids=("p1", "p2"),
            intervals={},
            conformal_guarantee_status=None,
        )

    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.predict_methods_archive_v2_raw_result",
        native_predict,
    )

    result = predict(
        model="portable.n4a",
        data={"X": np.asarray([[1.0], [2.0]]), "sample_ids": ["p1", "p2"]},
        engine="native",
    )

    assert result.y_pred.tolist() == [[2.0], [3.0]]
    assert result.metadata["engine"] == "native"
    assert observed["sample_ids"] == ["p1", "p2"]


def test_predict_native_refit_result_is_direct_and_never_constructs_a_legacy_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A detached V3 child is a native model, not a bundle-loader surrogate."""

    from nirs4all.api.native_refit_result import NativeMethodsRefitResult

    result = object.__new__(NativeMethodsRefitResult)
    observed: dict[str, object] = {}

    def native_predict(X, *, sample_ids, groups=None, metadata=None):  # noqa: ANN001
        observed.update(X=np.asarray(X), sample_ids=list(sample_ids), groups=groups, metadata=metadata)
        return type(
            "Prediction",
            (),
            {
                "y_pred": np.asarray([[3.0]]),
                "metadata": {"engine": "native", "sample_ids": ["p1"]},
                "intervals": {},
            },
        )()

    monkeypatch.setattr(result, "predict", native_predict)
    predict_module = importlib.import_module("nirs4all.api.predict")
    monkeypatch.setattr(
        predict_module,
        "PipelineRunner",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy runner constructed")),
    )
    prediction = predict(
        model=result,
        data={"X": np.asarray([[2.0]]), "sample_ids": ["p1"]},
        engine="native",
    )
    assert prediction.y_pred.tolist() == [[3.0]]
    assert observed["sample_ids"] == ["p1"]


def test_predict_native_run_result_is_direct_and_never_constructs_a_legacy_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh native run result is immediately usable for identity-bound PREDICT."""

    result = object.__new__(NativeMethodsRunResult)
    observed: dict[str, object] = {}

    class Estimator:
        def predict_with_identity(self, X, *, sample_ids, groups=None, metadata=None):  # noqa: ANN001
            observed.update(X=np.asarray(X), sample_ids=list(sample_ids), groups=groups, metadata=metadata)
            return np.asarray([[4.0]])

    result._native_estimator = Estimator()  # noqa: SLF001
    predict_module = importlib.import_module("nirs4all.api.predict")
    monkeypatch.setattr(
        predict_module,
        "PipelineRunner",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy runner constructed")),
    )

    prediction = predict(
        model=result,
        data={"X": np.asarray([[2.0]]), "sample_ids": ["p1"]},
    )

    assert prediction.y_pred.tolist() == [[4.0]]
    assert prediction.metadata == {"engine": "native", "sample_ids": ["p1"]}
    assert observed["sample_ids"] == ["p1"]


def test_predict_native_archive_accepts_raw_matrix_with_explicit_keyword_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The native public boundary must not require a synthetic mapping wrapper."""

    observed: dict[str, object] = {}

    def native_predict(path, X, *, sample_ids, methods_library_path, groups, metadata):  # noqa: ANN001
        observed.update(path=str(path), X=np.asarray(X), sample_ids=list(sample_ids))
        return NativeArchivePrediction(
            values=np.asarray([[2.0], [3.0]]),
            sample_ids=("p1", "p2"),
            intervals={},
            conformal_guarantee_status=None,
        )

    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.predict_methods_archive_v2_raw_result",
        native_predict,
    )

    result = predict(
        model="portable.n4a",
        data=np.asarray([[1.0], [2.0]]),
        sample_ids=["p1", "p2"],
        engine="native",
    )

    assert result.y_pred.tolist() == [[2.0], [3.0]]
    assert observed["path"] == "portable.n4a"
    assert observed["sample_ids"] == ["p1", "p2"]


def test_predict_native_archive_refuses_two_identity_sources_before_replay() -> None:
    with pytest.raises(ValueError, match="either in data or as an explicit keyword"):
        predict(
            model="portable.n4a",
            data={"X": np.asarray([[1.0]]), "sample_ids": ["p1"]},
            sample_ids=["p1"],
            engine="native",
        )


def test_predict_native_archive_selects_dagml_materialized_conformal_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interval = NativeArchiveConformalInterval(
        coverage=0.9,
        lower=np.asarray([[1.0], [2.0]]),
        upper=np.asarray([[3.0], [4.0]]),
        qhat=1.0,
        calibration_fingerprint="a" * 64,
    )

    def native_predict(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return NativeArchivePrediction(
            values=np.asarray([[2.0], [3.0]]),
            sample_ids=("p1", "p2"),
            intervals={0.9: interval},
            conformal_guarantee_status={
                "status": "active",
                "coverage": [0.9],
                "calibration_fingerprint": "a" * 64,
            },
        )

    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.native_archive_replay.predict_methods_archive_v2_raw_result",
        native_predict,
    )
    result = predict(
        model="portable.n4a",
        data={"X": np.asarray([[1.0], [2.0]]), "sample_ids": ["p1", "p2"]},
        coverage=0.9,
        engine="native",
    )

    assert result.interval_coverages == (0.9,)
    assert result.interval(0.9) is interval
    assert result.conformal_guarantee_status == {
        "status": "active",
        "coverage": [0.9],
        "calibration_fingerprint": "a" * 64,
    }


@pytest.mark.parametrize(
    ("model", "data", "kwargs", "message"),
    [
        ("portable.n4a", np.zeros((1, 2)), {}, "data={'X': matrix, 'sample_ids': explicit_ids}"),
        ("portable.joblib", {"X": [[1.0]], "sample_ids": ["p1"]}, {}, "Archive V2 .n4a"),
        ("portable.n4a", {"X": [[1.0]], "sample_ids": ["p1"]}, {"coverage": 0.9}, "not materialized"),
    ],
)
def test_predict_native_archive_fails_closed_before_execution(monkeypatch: pytest.MonkeyPatch, model, data, kwargs, message) -> None:
    if kwargs.get("coverage") is not None:
        monkeypatch.setattr(
            "nirs4all.pipeline.dagml.native_archive_replay.predict_methods_archive_v2_raw_result",
            lambda *_args, **_kwargs: NativeArchivePrediction(
                values=np.asarray([[1.0]]),
                sample_ids=("p1",),
                intervals={},
                conformal_guarantee_status=None,
            ),
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError), match=message):
        predict(model=model, data=data, engine="native", **kwargs)


@pytest.mark.parametrize(
    "operation",
    [
        "predict",
        "explain",
        "retrain",
    ],
)
def test_public_helpers_refuse_dagml_env_without_legacy_fallback(monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
    monkeypatch.setenv("N4A_ENGINE", "dag-ml")

    with pytest.raises(NotImplementedError, match=rf"nirs4all\.{operation} does not have a dag-ml execution path"):
        require_legacy_engine(operation)
