"""Unit locks for optimizer adapters over the shared PipelineObjective."""

from __future__ import annotations

import base64
import json
import sqlite3
from typing import Any

import pytest
from sklearn.base import BaseEstimator

from nirs4all.pipeline.dagml.pipeline_objective import PipelineObjective
from nirs4all.pipeline.dagml.tuning_adapters import (
    N4MPipelineObjectiveAdapter,
    ObjectiveTuningRunResult,
    _categorical_codecs,
    _CategoricalCodec,
    _make_optuna_pruner,
    _validate_optuna_resume_trial_contract,
    optimize_pipeline_objective,
    run_pipeline_objective_tuning,
)
from nirs4all.pipeline.dagml.tuning_contracts import TrialResult, TuningResult, parse_tuning_spec, tcv1_sha256
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


class _AdapterEstimator(BaseEstimator):
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_ids: Any = None,
        groups: Any = None,
        metadata: Any = None,
    ) -> _AdapterEstimator:
        self.sample_ids_ = sample_ids
        self.groups_ = groups
        self.metadata_ = metadata
        self.training_result_ = {"score": float(self.alpha)}
        return self


def test_objective_tuning_run_result_direct_construction_validates_contract() -> None:
    tuning = parse_tuning_spec({"engine": "optuna", "space": {"alpha": [0.1]}})
    tuning_result = TuningResult(
        tuning=tuning,
        best_params={"alpha": 0.1},
        best_value=0.1,
        trials=(TrialResult(number=0, params={"alpha": 0.1}, value=0.1, state="COMPLETE", diagnostics={}),),
        optimizer="optuna",
    )

    result = ObjectiveTuningRunResult(tuning_result=tuning_result, refit_estimator=None, tuning_id="tuning-1")

    assert result.tuning_result == tuning_result
    with pytest.raises(ValueError, match="tuning_result must be a TuningResult"):
        ObjectiveTuningRunResult(tuning_result=object(), refit_estimator=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tuning_id must be a canonical non-empty string"):
        ObjectiveTuningRunResult(tuning_result=tuning_result, refit_estimator=None, tuning_id=" tuning-1 ")
    with pytest.raises(ValueError, match="tuning_id must not contain NUL"):
        ObjectiveTuningRunResult(tuning_result=tuning_result, refit_estimator=None, tuning_id="bad\x00id")


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: _CategoricalCodec(()), "choices must be a non-empty tuple"),
        (lambda: _CategoricalCodec(("a", "a")), "choices must not contain duplicates"),
        (lambda: _CategoricalCodec(({"kind": "passthrough"},)), "without a decoder must be optimizer-native"),
        (lambda: _CategoricalCodec(("a",), decoder={}), "decoder must be a non-empty mapping"),
        (lambda: _CategoricalCodec(("a",), decoder={"b": 1}), "decoder keys must match choices in order"),
        (lambda: _CategoricalCodec(("a", "b"), decoder={"a": 1, "b": 1}), "decoder values must not contain duplicates"),
        (lambda: _CategoricalCodec(("a",), decoder={"a": object()}), "decoder values must contain TCV1-compatible"),
    ],
)
def test_categorical_codec_direct_construction_rejects_invalid_contracts(factory: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


def test_categorical_codec_direct_construction_decodes_valid_contract() -> None:
    native = _CategoricalCodec(("a", 2, True))
    encoded = _CategoricalCodec(("opt0", "opt1"), decoder={"opt0": {"kind": "passthrough"}, "opt1": ["x"]})

    assert native.decode("a") == "a"
    assert native.encode(2) == 2
    assert encoded.decode("opt0") == {"kind": "passthrough"}
    assert encoded.encode(["x"]) == "opt1"


class _ChoiceEstimator(BaseEstimator):
    def __init__(self, mode: Any = "scaled") -> None:
        self.mode = mode

    def fit(self, X: Any, y: Any) -> _ChoiceEstimator:
        self.training_result_ = {"score": 0.0 if self.mode == {"kind": "passthrough"} else 10.0}
        return self


class _SometimesFailingEstimator(BaseEstimator):
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: Any, y: Any) -> _SometimesFailingEstimator:
        if self.alpha > 0.5:
            raise RuntimeError("candidate fit failed")
        self.training_result_ = {"score": float(self.alpha)}
        return self


class _SometimesPrunedEstimator(BaseEstimator):
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: Any, y: Any) -> _SometimesPrunedEstimator:
        if self.alpha > 0.5:
            from optuna.exceptions import TrialPruned

            raise TrialPruned("candidate pruned")
        self.training_result_ = {"score": float(self.alpha)}
        return self


class _AlwaysFailingEstimator(BaseEstimator):
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: Any, y: Any) -> _AlwaysFailingEstimator:
        raise RuntimeError("all candidates fail")


class _RefitFailingEstimator(BaseEstimator):
    fit_calls: int = 0

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: Any, y: Any) -> _RefitFailingEstimator:
        type(self).fit_calls += 1
        if type(self).fit_calls > 2:
            raise RuntimeError("terminal refit failed")
        self.training_result_ = {"score": float(self.alpha)}
        return self


class _FakeOptunaState:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeOptunaTrial:
    def __init__(
        self,
        number: Any,
        *,
        params: dict[str, Any] | None = None,
        value: float | None = 0.2,
        state: str = "COMPLETE",
    ) -> None:
        self.number = number
        self.params = {} if params is None else params
        self.system_attrs: dict[str, Any] = {}
        self.value = value
        self.state = _FakeOptunaState(state)


class _FakeOptunaStudy:
    def __init__(self, trials: tuple[_FakeOptunaTrial, ...]) -> None:
        self.trials = trials


def test_optuna_adapter_drives_pipeline_objective_and_returns_typed_result() -> None:
    objective = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "seed": 7,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    result = optimize_pipeline_objective(
        objective,
        [[1.0], [2.0]],
        [1.0, 2.0],
        sample_ids=["s1", "s2"],
        groups=["g", "g"],
        metadata={"s1": {"batch": "a"}, "s2": {"batch": "a"}},
    )

    assert result.optimizer == "optuna"
    assert result.best_params == {"alpha": 0.2}
    assert result.best_value == pytest.approx(0.2)
    assert result.n_trials == 2
    assert {trial.state for trial in result.trials} == {"COMPLETE"}
    assert len(result.fingerprint) == 64
    assert result.trials[0].diagnostics["score_family"] == "objective"
    assert result.trials[0].diagnostics["score_extractor"] == "objective"
    assert "search_space_fingerprint" in result.trials[0].diagnostics


def test_optuna_adapter_decodes_json_native_categorical_choices() -> None:
    objective = PipelineObjective(
        _ChoiceEstimator(),
        {
            "engine": "optuna",
            "space": {"mode": [{"kind": "passthrough"}, "scaled"]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "seed": 7,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    result = optimize_pipeline_objective(objective, [[1.0]], [1.0])

    assert result.best_params == {"mode": {"kind": "passthrough"}}
    assert {trial.params["mode"] for trial in result.trials if isinstance(trial.params["mode"], str)} == {"scaled"}
    assert any(trial.params["mode"] == {"kind": "passthrough"} for trial in result.trials)


def test_optuna_adapter_decodes_named_categorical_options_to_values() -> None:
    objective = PipelineObjective(
        _ChoiceEstimator(),
        {
            "engine": "optuna",
            "space": {
                "mode": {
                    "type": "categorical",
                    "options": {
                        "identity": {"kind": "passthrough"},
                        "scaled": "scaled",
                    },
                }
            },
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "seed": 7,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    result = optimize_pipeline_objective(objective, [[1.0]], [1.0])

    assert result.best_params == {"mode": {"kind": "passthrough"}}
    assert {trial.params["mode"] for trial in result.trials if isinstance(trial.params["mode"], str)} == {"scaled"}
    assert any(trial.params["mode"] == {"kind": "passthrough"} for trial in result.trials)


def test_optuna_adapter_records_failed_trials_and_keeps_successful_best() -> None:
    objective = PipelineObjective(
        _SometimesFailingEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "seed": 7,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    result = optimize_pipeline_objective(objective, [[1.0]], [1.0])

    assert result.best_params == {"alpha": 0.2}
    assert result.best_value == pytest.approx(0.2)
    assert {trial.state for trial in result.trials} == {"COMPLETE", "FAIL"}
    failed = next(trial for trial in result.trials if trial.state == "FAIL")
    assert failed.value is None
    assert failed.diagnostics["error_type"] == "RuntimeError"
    assert failed.diagnostics["score_extractor"] == "failed"


def test_optuna_adapter_records_pruned_trials_separately_from_failures() -> None:
    pytest.importorskip("optuna")
    objective = PipelineObjective(
        _SometimesPrunedEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "seed": 7,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    result = optimize_pipeline_objective(objective, [[1.0]], [1.0])

    assert result.best_params == {"alpha": 0.2}
    assert result.best_value == pytest.approx(0.2)
    assert {trial.state for trial in result.trials} == {"COMPLETE", "PRUNED"}
    pruned = next(trial for trial in result.trials if trial.state == "PRUNED")
    assert pruned.value is None
    assert pruned.diagnostics["error_type"] == "TrialPruned"
    assert pruned.diagnostics["score_extractor"] == "pruned"
    assert result.summary_artifact()["trial_states"] == {"COMPLETE": 1, "PRUNED": 1}
    pruned_summary = next(row for row in result.summary_artifact()["trials"] if row["state"] == "PRUNED")
    assert pruned_summary["diagnostics"]["score_extractor"] == "pruned"


@pytest.mark.parametrize(("direction", "expected_best_value"), [("minimize", 1.0e308), ("maximize", -1.0e308)])
def test_run_pipeline_objective_tuning_skips_refit_when_all_trials_fail(
    direction: str,
    expected_best_value: float,
) -> None:
    objective = PipelineObjective(
        _AlwaysFailingEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.8]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": direction,
            "seed": 7,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    result = run_pipeline_objective_tuning(objective, [[1.0]], [1.0], refit=True)

    assert result.refit_estimator is None
    assert result.tuning_result.best_params == {}
    assert result.tuning_result.best_value == expected_best_value
    assert {trial.state for trial in result.tuning_result.trials} == {"FAIL"}
    assert len(result.tuning_result.fingerprint) == 64
    assert result.tuning_result.summary_artifact()["best_value"] == expected_best_value
    assert result.tuning_result.summary_artifact()["trial_states"] == {"FAIL": 2}
    assert "nirs4all.tuning.summary" in result.tuning_result.to_summary_json()


def test_run_pipeline_objective_tuning_persists_workspace_result_when_all_trials_fail(tmp_path) -> None:
    objective = PipelineObjective(
        _AlwaysFailingEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.8]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "seed": 7,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    result = run_pipeline_objective_tuning(
        objective,
        [[1.0]],
        [1.0],
        refit=True,
        workspace_path=tmp_path / "workspace",
        workspace_tuning_id="objective-tune-all-failed",
        workspace_name="all failed objective tuning",
        workspace_metadata={"purpose": "all-failed-unit-test"},
    )

    store = WorkspaceStore(tmp_path / "workspace")
    try:
        restored = store.load_tuning_result("objective-tune-all-failed")
    finally:
        store.close()

    assert result.refit_estimator is None
    assert restored.to_dict() == result.tuning_result.to_dict()
    assert restored.best_params == {}
    assert restored.summary_artifact()["trial_states"] == {"FAIL": 2}


def test_optuna_adapter_restores_logging_after_caught_trial_failures() -> None:
    optuna = pytest.importorskip("optuna")
    previous = optuna.logging.get_verbosity()
    objective = PipelineObjective(
        _AlwaysFailingEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "seed": 7,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    result = run_pipeline_objective_tuning(objective, [[1.0]], [1.0], refit=True)

    assert optuna.logging.get_verbosity() == previous
    assert result.tuning_result.trials[0].state == "FAIL"
    assert result.tuning_result.trials[0].diagnostics["error_type"] == "RuntimeError"


@pytest.mark.parametrize(
    "extra",
    [
        {"resume": True},
        {"resume": True, "storage": "sqlite:///study.db"},
        {"resume": True, "study_name": "pls-study"},
    ],
)
def test_optuna_adapter_rejects_ambiguous_resume_without_storage_and_study_name(extra: dict[str, Any]) -> None:
    objective = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": [0.2]},
            "sampler": "grid",
            "n_trials": 1,
            **extra,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    with pytest.raises(ValueError, match="requires explicit storage and study_name when resume=True"):
        optimize_pipeline_objective(objective, [[1.0]], [1.0])


def test_optuna_adapter_allows_storage_without_resume_for_new_study(tmp_path) -> None:
    objective = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": [0.2]},
            "sampler": "grid",
            "n_trials": 1,
            "storage": f"sqlite:///{tmp_path / 'study.db'}",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    result = optimize_pipeline_objective(objective, [[1.0]], [1.0])

    assert result.best_params == {"alpha": 0.2}
    assert result.trials[0].state == "COMPLETE"


def test_optuna_adapter_enqueues_force_params_as_first_public_trial() -> None:
    objective = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "force_params": {"alpha": 0.2},
            "sampler": "grid",
            "n_trials": 1,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    result = optimize_pipeline_objective(objective, [[1.0]], [1.0])

    assert result.best_params == {"alpha": 0.2}
    assert result.trials[0].params == {"alpha": 0.2}
    assert result.trials[0].value == pytest.approx(0.2)


def test_optuna_resume_reuses_matching_force_params_without_duplicate_enqueue(tmp_path) -> None:
    storage = f"sqlite:///{tmp_path / 'resume-force.db'}"
    initial = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "force_params": {"alpha": 0.2},
            "sampler": "grid",
            "n_trials": 1,
            "storage": storage,
            "study_name": "force-resume",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "force_params": {"alpha": 0.2},
            "sampler": "grid",
            "n_trials": 1,
            "storage": storage,
            "study_name": "force-resume",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    first = optimize_pipeline_objective(initial, [[1.0]], [1.0])
    second = optimize_pipeline_objective(resumed, [[1.0]], [1.0])

    assert first.trials[0].params == {"alpha": 0.2}
    assert second.trials[0].params == {"alpha": 0.2}
    assert len(second.trials) == 1
    assert [trial.params for trial in second.trials].count({"alpha": 0.2}) == 1
    assert second.trials[0].diagnostics["score_extractor"] == "optuna_storage"


def test_optuna_resume_treats_n_trials_as_target_total(tmp_path) -> None:
    storage = f"sqlite:///{tmp_path / 'resume-target-total.db'}"
    initial = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "force_params": {"alpha": 0.2},
            "sampler": "random",
            "n_trials": 1,
            "storage": storage,
            "study_name": "target-total",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed_same_total = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "force_params": {"alpha": 0.2},
            "sampler": "random",
            "n_trials": 1,
            "storage": storage,
            "study_name": "target-total",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed_next_total = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "force_params": {"alpha": 0.2},
            "sampler": "random",
            "n_trials": 2,
            "storage": storage,
            "study_name": "target-total",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    first = optimize_pipeline_objective(initial, [[1.0]], [1.0])
    second = optimize_pipeline_objective(resumed_same_total, [[1.0]], [1.0])
    third = optimize_pipeline_objective(resumed_next_total, [[1.0]], [1.0])

    assert len(first.trials) == 1
    assert len(second.trials) == 1
    assert len(third.trials) == 2
    assert second.trials[0].diagnostics["score_extractor"] == "optuna_storage"
    assert third.trials[0].diagnostics["score_extractor"] == "optuna_storage"


def test_optuna_storage_persists_native_contract_fingerprints(tmp_path) -> None:
    optuna = pytest.importorskip("optuna")
    storage = f"sqlite:///{tmp_path / 'contract-attrs.db'}"
    objective = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "sampler": "random",
            "n_trials": 1,
            "storage": storage,
            "study_name": "contract-attrs",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    result = optimize_pipeline_objective(objective, [[1.0]], [1.0])
    study = optuna.load_study(study_name="contract-attrs", storage=storage)

    assert result.optimizer == "optuna"
    assert study.user_attrs["nirs4all_format"] == "nirs4all.optuna.pipeline_objective"
    assert study.user_attrs["nirs4all_schema_version"] == 1
    assert len(study.user_attrs["nirs4all_optimizer_contract_fingerprint"]) == 64
    assert study.user_attrs["nirs4all_search_space_fingerprint"] == objective.tuning.ordered_search_space.fingerprint


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("delete_value", "missing a numeric finite value"),
        ("positive_infinity", "non-finite value"),
    ],
)
def test_optuna_resume_rejects_complete_storage_trial_without_finite_value(
    tmp_path,
    mutation: str,
    match: str,
) -> None:
    pytest.importorskip("optuna")
    storage_path = tmp_path / f"invalid-complete-value-{mutation}.db"
    storage = f"sqlite:///{storage_path}"
    initial = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "force_params": {"alpha": 0.2},
            "sampler": "random",
            "n_trials": 1,
            "storage": storage,
            "study_name": "invalid-complete-value",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "force_params": {"alpha": 0.2},
            "sampler": "random",
            "n_trials": 1,
            "storage": storage,
            "study_name": "invalid-complete-value",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    optimize_pipeline_objective(initial, [[1.0]], [1.0])
    with sqlite3.connect(storage_path) as connection:
        if mutation == "delete_value":
            connection.execute("delete from trial_values")
        else:
            connection.execute("update trial_values set value = NULL, value_type = 'INF_POS'")

    with pytest.raises(ValueError, match=match):
        optimize_pipeline_objective(resumed, [[1.0]], [1.0])


@pytest.mark.parametrize("state", ["FAIL", "PRUNED", "RUNNING", "WAITING"])
def test_optuna_resume_rejects_non_complete_storage_trial_with_final_value(
    tmp_path,
    state: str,
) -> None:
    pytest.importorskip("optuna")
    storage_path = tmp_path / f"invalid-{state.lower()}-value.db"
    storage = f"sqlite:///{storage_path}"
    initial = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "force_params": {"alpha": 0.2},
            "sampler": "random",
            "n_trials": 1,
            "storage": storage,
            "study_name": "invalid-non-complete-value",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "force_params": {"alpha": 0.2},
            "sampler": "random",
            "n_trials": 1,
            "storage": storage,
            "study_name": "invalid-non-complete-value",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    optimize_pipeline_objective(initial, [[1.0]], [1.0])
    with sqlite3.connect(storage_path) as connection:
        connection.execute("update trials set state = ?", (state,))

    with pytest.raises(ValueError, match=f"storage {state} trial 0 carries a final value"):
        optimize_pipeline_objective(resumed, [[1.0]], [1.0])


@pytest.mark.parametrize("state", ["COMPLETE", "FAIL", "PRUNED"])
def test_optuna_resume_rejects_terminal_storage_trial_without_search_space_params(
    tmp_path,
    state: str,
) -> None:
    pytest.importorskip("optuna")
    storage_path = tmp_path / f"missing-{state.lower()}-params.db"
    storage = f"sqlite:///{storage_path}"
    initial = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "force_params": {"alpha": 0.2},
            "sampler": "random",
            "n_trials": 1,
            "storage": storage,
            "study_name": "missing-terminal-params",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "force_params": {"alpha": 0.2},
            "sampler": "random",
            "n_trials": 1,
            "storage": storage,
            "study_name": "missing-terminal-params",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    optimize_pipeline_objective(initial, [[1.0]], [1.0])
    with sqlite3.connect(storage_path) as connection:
        connection.execute("delete from trial_params")
        if state != "COMPLETE":
            connection.execute("update trials set state = ?", (state,))
            connection.execute("delete from trial_values")

    with pytest.raises(ValueError, match="materialized trial params do not match tuning.space keys"):
        optimize_pipeline_objective(resumed, [[1.0]], [1.0])


def test_optuna_resume_rejects_waiting_storage_trial_with_incompatible_params(
    tmp_path,
) -> None:
    pytest.importorskip("optuna")
    storage_path = tmp_path / "waiting-incompatible-params.db"
    storage = f"sqlite:///{storage_path}"
    initial = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "sampler": "random",
            "n_trials": 1,
            "storage": storage,
            "study_name": "in-flight-params",
            "seed": 7,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "sampler": "random",
            "n_trials": 1,
            "storage": storage,
            "study_name": "in-flight-params",
            "resume": True,
            "seed": 7,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    optimize_pipeline_objective(initial, [[1.0]], [1.0])
    with sqlite3.connect(storage_path) as connection:
        connection.execute("update trials set state = 'WAITING'")
        connection.execute("delete from trial_values")
        connection.execute("update trial_params set param_value = 9.0")

    with pytest.raises(ValueError, match="value 9.0.*outside the current tuning.space range"):
        optimize_pipeline_objective(resumed, [[1.0]], [1.0])


def test_optuna_resume_rejects_waiting_storage_trial_with_incompatible_fixed_params(
    tmp_path,
) -> None:
    pytest.importorskip("optuna")
    storage_path = tmp_path / "waiting-incompatible-fixed-params.db"
    storage = f"sqlite:///{storage_path}"
    initial = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "sampler": "random",
            "n_trials": 1,
            "storage": storage,
            "study_name": "queued-fixed-params",
            "seed": 7,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "sampler": "random",
            "n_trials": 2,
            "storage": storage,
            "study_name": "queued-fixed-params",
            "resume": True,
            "seed": 7,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    optimize_pipeline_objective(initial, [[1.0]], [1.0])
    with sqlite3.connect(storage_path) as connection:
        study_id = connection.execute(
            "select study_id from studies where study_name = ?",
            ("queued-fixed-params",),
        ).fetchone()[0]
        connection.execute(
            "insert into trials(number, study_id, state) values (?, ?, 'WAITING')",
            (1, study_id),
        )
        trial_id = connection.execute("select last_insert_rowid()").fetchone()[0]
        connection.execute(
            "insert into trial_system_attributes(trial_id, key, value_json) values (?, ?, ?)",
            (trial_id, "fixed_params", json.dumps({"alpha": 9.0})),
        )

    with pytest.raises(ValueError, match="value 9.0.*outside the current tuning.space range"):
        optimize_pipeline_objective(resumed, [[1.0]], [1.0])


def test_optuna_resume_rejects_waiting_storage_trial_with_invalid_fixed_params_payload(
    tmp_path,
) -> None:
    pytest.importorskip("optuna")
    storage_path = tmp_path / "waiting-invalid-fixed-params.db"
    storage = f"sqlite:///{storage_path}"
    initial = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "sampler": "random",
            "n_trials": 1,
            "storage": storage,
            "study_name": "invalid-fixed-params",
            "seed": 7,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "sampler": "random",
            "n_trials": 2,
            "storage": storage,
            "study_name": "invalid-fixed-params",
            "resume": True,
            "seed": 7,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    optimize_pipeline_objective(initial, [[1.0]], [1.0])
    with sqlite3.connect(storage_path) as connection:
        study_id = connection.execute(
            "select study_id from studies where study_name = ?",
            ("invalid-fixed-params",),
        ).fetchone()[0]
        connection.execute(
            "insert into trials(number, study_id, state) values (?, ?, 'WAITING')",
            (1, study_id),
        )
        trial_id = connection.execute("select last_insert_rowid()").fetchone()[0]
        connection.execute(
            "insert into trial_system_attributes(trial_id, key, value_json) values (?, ?, ?)",
            (trial_id, "fixed_params", json.dumps(["alpha", 0.2])),
        )

    with pytest.raises(ValueError, match="fixed_params are not a mapping"):
        optimize_pipeline_objective(resumed, [[1.0]], [1.0])


def test_optuna_resume_rejects_waiting_storage_trial_with_divergent_fixed_params(
    tmp_path,
) -> None:
    pytest.importorskip("optuna")
    storage_path = tmp_path / "waiting-divergent-fixed-params.db"
    storage = f"sqlite:///{storage_path}"
    initial = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "force_params": {"alpha": 0.2},
            "sampler": "random",
            "n_trials": 1,
            "storage": storage,
            "study_name": "divergent-fixed-params",
            "seed": 7,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "force_params": {"alpha": 0.2},
            "sampler": "random",
            "n_trials": 1,
            "storage": storage,
            "study_name": "divergent-fixed-params",
            "resume": True,
            "seed": 7,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    optimize_pipeline_objective(initial, [[1.0]], [1.0])
    with sqlite3.connect(storage_path) as connection:
        trial_id = connection.execute("select trial_id from trials where number = 0").fetchone()[0]
        connection.execute("update trials set state = 'WAITING'")
        connection.execute("delete from trial_values")
        connection.execute(
            "update trial_system_attributes set value_json = ? where trial_id = ? and key = ?",
            (json.dumps({"alpha": 0.3}), trial_id, "fixed_params"),
        )

    with pytest.raises(ValueError, match="materialized params differ from fixed_params"):
        optimize_pipeline_objective(resumed, [[1.0]], [1.0])


@pytest.mark.parametrize("keep_params", [False, True])
def test_optuna_resume_rejects_running_storage_trial(
    tmp_path,
    keep_params: bool,
) -> None:
    pytest.importorskip("optuna")
    storage_path = tmp_path / f"running-trial-{'with' if keep_params else 'without'}-params.db"
    storage = f"sqlite:///{storage_path}"
    initial = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "sampler": "random",
            "n_trials": 1,
            "storage": storage,
            "study_name": "running-trial",
            "seed": 7,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "sampler": "random",
            "n_trials": 1,
            "storage": storage,
            "study_name": "running-trial",
            "resume": True,
            "seed": 7,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    optimize_pipeline_objective(initial, [[1.0]], [1.0])
    with sqlite3.connect(storage_path) as connection:
        connection.execute("update trials set state = 'RUNNING'")
        connection.execute("delete from trial_values")
        if not keep_params:
            connection.execute("delete from trial_params")

    with pytest.raises(ValueError, match="cannot resume a study with RUNNING trials"):
        optimize_pipeline_objective(resumed, [[1.0]], [1.0])


def test_optuna_resume_rejects_non_integer_storage_trial_number(tmp_path) -> None:
    pytest.importorskip("optuna")
    storage_path = tmp_path / "non-integer-trial-number.db"
    storage = f"sqlite:///{storage_path}"
    initial = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": [0.2, 0.4]},
            "sampler": "grid",
            "n_trials": 2,
            "storage": storage,
            "study_name": "non-integer-trial-number",
            "seed": 7,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": [0.2, 0.4]},
            "sampler": "grid",
            "n_trials": 2,
            "storage": storage,
            "study_name": "non-integer-trial-number",
            "resume": True,
            "seed": 7,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    optimize_pipeline_objective(initial, [[1.0]], [1.0])
    with sqlite3.connect(storage_path) as connection:
        connection.execute("update trials set number = ? where number = 1", ("trial-one",))

    with pytest.raises(ValueError, match="trial numbers are not canonical integers"):
        optimize_pipeline_objective(resumed, [[1.0]], [1.0])


def test_optuna_resume_contract_rejects_duplicate_trial_numbers() -> None:
    objective = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "sampler": "random",
            "n_trials": 2,
            "storage": "sqlite:////tmp/nirs4all-fake-optuna.db",
            "study_name": "fake-duplicate-numbers",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    study = _FakeOptunaStudy(
        (
            _FakeOptunaTrial(0, params={"alpha": 0.2}, value=0.2),
            _FakeOptunaTrial(0, params={"alpha": 0.4}, value=0.4),
        )
    )

    with pytest.raises(ValueError, match="duplicate trial number 0"):
        _validate_optuna_resume_trial_contract(
            study,
            objective.tuning,
            _categorical_codecs(objective.tuning.space),
        )


def test_optuna_resume_rejects_study_contract_fingerprint_mismatch(tmp_path) -> None:
    storage = f"sqlite:///{tmp_path / 'contract-mismatch.db'}"
    initial = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "sampler": "random",
            "metric": "rmse",
            "n_trials": 1,
            "storage": storage,
            "study_name": "contract-mismatch",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    changed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "sampler": "random",
            "metric": "mae",
            "n_trials": 2,
            "storage": storage,
            "study_name": "contract-mismatch",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    optimize_pipeline_objective(initial, [[1.0]], [1.0])

    with pytest.raises(ValueError, match="study contract mismatch"):
        optimize_pipeline_objective(changed, [[1.0]], [1.0])


def test_optuna_resume_rejects_non_empty_legacy_study_without_native_fingerprints(tmp_path) -> None:
    optuna = pytest.importorskip("optuna")
    storage = f"sqlite:///{tmp_path / 'legacy-without-attrs.db'}"
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=7),
        storage=storage,
        study_name="legacy-without-attrs",
    )
    study.optimize(lambda trial: trial.suggest_float("alpha", 0.0, 1.0), n_trials=1)
    objective = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "sampler": "random",
            "n_trials": 2,
            "storage": storage,
            "study_name": "legacy-without-attrs",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    with pytest.raises(ValueError, match="without nirs4all optimizer contract fingerprints"):
        optimize_pipeline_objective(objective, [[1.0]], [1.0])


def test_optuna_resume_reconstructs_failed_storage_diagnostics(tmp_path) -> None:
    storage = f"sqlite:///{tmp_path / 'resume-failed-diagnostics.db'}"
    initial = PipelineObjective(
        _SometimesFailingEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "force_params": {"alpha": 0.9},
            "sampler": "grid",
            "n_trials": 1,
            "storage": storage,
            "study_name": "failed-diagnostics",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "force_params": {"alpha": 0.9},
            "sampler": "grid",
            "n_trials": 2,
            "storage": storage,
            "study_name": "failed-diagnostics",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    first = optimize_pipeline_objective(initial, [[1.0]], [1.0])
    second = optimize_pipeline_objective(resumed, [[1.0]], [1.0])

    assert first.trials[0].state == "FAIL"
    assert [trial.state for trial in second.trials] == ["FAIL", "COMPLETE"]
    assert second.trials[0].diagnostics["score_extractor"] == "failed"
    assert second.trials[0].diagnostics["error_type"] == "OptunaFailedTrial"
    assert "search_space_fingerprint" in second.trials[0].diagnostics
    assert second.summary_artifact()["trial_states"] == {"COMPLETE": 1, "FAIL": 1}


def test_optuna_resume_reconstructs_pruned_storage_diagnostics(tmp_path) -> None:
    pytest.importorskip("optuna")
    storage = f"sqlite:///{tmp_path / 'resume-pruned-diagnostics.db'}"
    initial = PipelineObjective(
        _SometimesPrunedEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "force_params": {"alpha": 0.9},
            "sampler": "grid",
            "n_trials": 1,
            "storage": storage,
            "study_name": "pruned-diagnostics",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "force_params": {"alpha": 0.9},
            "sampler": "grid",
            "n_trials": 2,
            "storage": storage,
            "study_name": "pruned-diagnostics",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    first = optimize_pipeline_objective(initial, [[1.0]], [1.0])
    second = optimize_pipeline_objective(resumed, [[1.0]], [1.0])

    assert first.trials[0].state == "PRUNED"
    assert [trial.state for trial in second.trials] == ["PRUNED", "COMPLETE"]
    assert second.trials[0].diagnostics["score_extractor"] == "pruned"
    assert second.trials[0].diagnostics["error_type"] == "TrialPruned"
    assert "tuning_fingerprint" in second.trials[0].diagnostics
    assert second.summary_artifact()["trial_states"] == {"COMPLETE": 1, "PRUNED": 1}


def test_optuna_resume_rejects_existing_trials_with_changed_search_space(tmp_path) -> None:
    storage = f"sqlite:///{tmp_path / 'resume-space-mismatch.db'}"
    initial = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": [0.2]},
            "sampler": "grid",
            "n_trials": 1,
            "storage": storage,
            "study_name": "space-mismatch",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    changed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"beta": [0.1]},
            "sampler": "grid",
            "n_trials": 1,
            "storage": storage,
            "study_name": "space-mismatch",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    optimize_pipeline_objective(initial, [[1.0]], [1.0])

    with pytest.raises(ValueError, match="materialized trial params do not match tuning.space keys"):
        optimize_pipeline_objective(changed, [[1.0]], [1.0])


def test_optuna_resume_rejects_existing_trials_with_changed_categorical_choices(tmp_path) -> None:
    storage = f"sqlite:///{tmp_path / 'resume-categorical-mismatch.db'}"
    initial = PipelineObjective(
        _ChoiceEstimator(),
        {
            "engine": "optuna",
            "space": {"mode": ["old", "stable"]},
            "force_params": {"mode": "old"},
            "sampler": "grid",
            "n_trials": 1,
            "storage": storage,
            "study_name": "categorical-mismatch",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    changed = PipelineObjective(
        _ChoiceEstimator(),
        {
            "engine": "optuna",
            "space": {"mode": ["new", "stable"]},
            "sampler": "grid",
            "n_trials": 1,
            "storage": storage,
            "study_name": "categorical-mismatch",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    optimize_pipeline_objective(initial, [[1.0]], [1.0])

    with pytest.raises(ValueError, match="categorical value 'old'.*not present in the current tuning.space choices"):
        optimize_pipeline_objective(changed, [[1.0]], [1.0])


def test_optuna_resume_rejects_existing_trials_with_changed_numeric_range(tmp_path) -> None:
    storage = f"sqlite:///{tmp_path / 'resume-range-mismatch.db'}"
    initial = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 1.0)},
            "force_params": {"alpha": 0.9},
            "sampler": "random",
            "n_trials": 1,
            "storage": storage,
            "study_name": "range-mismatch",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    changed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": (0.0, 0.5)},
            "sampler": "random",
            "n_trials": 1,
            "storage": storage,
            "study_name": "range-mismatch",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    optimize_pipeline_objective(initial, [[1.0]], [1.0])

    with pytest.raises(ValueError, match="value 0.9.*outside the current tuning.space range"):
        optimize_pipeline_objective(changed, [[1.0]], [1.0])


def test_optuna_resume_rejects_existing_trials_with_changed_numeric_step(tmp_path) -> None:
    storage = f"sqlite:///{tmp_path / 'resume-step-mismatch.db'}"
    initial = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": {"type": "float", "low": 0.0, "high": 1.0}},
            "force_params": {"alpha": 0.9},
            "sampler": "random",
            "n_trials": 1,
            "storage": storage,
            "study_name": "step-mismatch",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    changed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": {"type": "float", "low": 0.0, "high": 1.0, "step": 0.2}},
            "sampler": "random",
            "n_trials": 1,
            "storage": storage,
            "study_name": "step-mismatch",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    optimize_pipeline_objective(initial, [[1.0]], [1.0])

    with pytest.raises(ValueError, match="value 0.9.*outside the current tuning.space range"):
        optimize_pipeline_objective(changed, [[1.0]], [1.0])


def test_optuna_resume_rejects_changed_force_params_for_existing_study(tmp_path) -> None:
    storage = f"sqlite:///{tmp_path / 'resume-force-mismatch.db'}"
    initial = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "force_params": {"alpha": 0.2},
            "sampler": "grid",
            "n_trials": 1,
            "storage": storage,
            "study_name": "force-resume-mismatch",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    changed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "force_params": {"alpha": 0.9},
            "sampler": "grid",
            "n_trials": 1,
            "storage": storage,
            "study_name": "force-resume-mismatch",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    optimize_pipeline_objective(initial, [[1.0]], [1.0])

    with pytest.raises(ValueError, match="cannot change tuning.force_params when resume=True"):
        optimize_pipeline_objective(changed, [[1.0]], [1.0])


def test_optuna_adapter_encodes_force_params_for_named_categorical_options() -> None:
    objective = PipelineObjective(
        _ChoiceEstimator(),
        {
            "engine": "optuna",
            "space": {
                "mode": {
                    "type": "categorical",
                    "options": {
                        "identity": {"kind": "passthrough"},
                        "scaled": "scaled",
                    },
                }
            },
            "force_params": {"mode": {"kind": "passthrough"}},
            "sampler": "grid",
            "n_trials": 1,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    result = optimize_pipeline_objective(objective, [[1.0]], [1.0])

    assert result.best_params == {"mode": {"kind": "passthrough"}}
    assert result.trials[0].params == {"mode": {"kind": "passthrough"}}


def test_run_pipeline_objective_tuning_separates_optimization_and_terminal_refit() -> None:
    objective = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "direction": "minimize",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    result = run_pipeline_objective_tuning(
        objective,
        [[1.0], [2.0]],
        [1.0, 2.0],
        sample_ids=["s1", "s2"],
        groups=["g", "g"],
        metadata={"s1": {"batch": "a"}, "s2": {"batch": "a"}},
        refit=True,
    )

    assert result.tuning_result.best_params == {"alpha": 0.2}
    assert result.refit_estimator is not None
    assert result.refit_estimator.alpha == 0.2
    assert result.refit_estimator.sample_ids_ == ["s1", "s2"]


def test_run_pipeline_objective_tuning_can_skip_terminal_refit() -> None:
    objective = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    result = run_pipeline_objective_tuning(objective, [[1.0]], [1.0], refit=False)

    assert result.tuning_result.best_params == {"alpha": 0.2}
    assert result.refit_estimator is None


def test_run_pipeline_objective_tuning_can_persist_workspace_result_before_refit(tmp_path) -> None:
    objective = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    result = run_pipeline_objective_tuning(
        objective,
        [[1.0], [2.0]],
        [1.0, 2.0],
        sample_ids=["s1", "s2"],
        refit=True,
        workspace_path=tmp_path / "workspace",
        workspace_tuning_id="objective-tune-main",
        workspace_name="objective tuning",
        workspace_metadata={"purpose": "unit-test"},
    )

    store = WorkspaceStore(tmp_path / "workspace")
    try:
        restored = store.load_tuning_result("objective-tune-main")
    finally:
        store.close()

    assert result.tuning_id == "objective-tune-main"
    assert result.refit_estimator is not None
    assert restored.to_dict() == result.tuning_result.to_dict()


def test_run_pipeline_objective_tuning_persists_workspace_result_when_terminal_refit_fails(tmp_path) -> None:
    _RefitFailingEstimator.fit_calls = 0
    objective = PipelineObjective(
        _RefitFailingEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    with pytest.raises(RuntimeError, match="terminal refit failed"):
        run_pipeline_objective_tuning(
            objective,
            [[1.0], [2.0]],
            [1.0, 2.0],
            refit=True,
            workspace_path=tmp_path / "workspace",
            workspace_tuning_id="objective-tune-before-refit-failure",
            workspace_name="objective tuning before refit failure",
            workspace_metadata={"purpose": "refit-failure-unit-test"},
        )

    store = WorkspaceStore(tmp_path / "workspace")
    try:
        restored = store.load_tuning_result("objective-tune-before-refit-failure")
    finally:
        store.close()

    assert restored.best_params == {"alpha": 0.2}
    assert restored.best_value == pytest.approx(0.2)
    assert [trial.state for trial in restored.trials] == ["COMPLETE", "COMPLETE"]


class _FakeEnum:
    TPE = "tpe"
    RANDOM = "random"
    GRID = "grid"
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    MEDIAN = "median"
    ASHA = "asha"
    HYPERBAND = "hyperband"
    RACING = "racing"
    FAILED = "failed"
    PRUNED = "pruned"
    CANCELLED = "cancelled"


class _FakeSearchSpace:
    def __init__(self) -> None:
        self.params: dict[str, tuple[str, Any]] = {}

    def add_categorical(self, name: str, choices: list[Any]) -> None:
        if any(isinstance(choice, (dict, list, tuple, set)) for choice in choices):
            raise AssertionError("fake n4m search space expects optimizer-native categorical labels")
        self.params[name] = ("categorical", list(choices))

    def add_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False) -> None:
        self.params[name] = ("int", (low, high, step, log))

    def add_float(self, name: str, low: float, high: float, step: float = 0.0, log: bool = False) -> None:
        self.params[name] = ("float", (low, high, step, log))


class _FakeTrial:
    def __init__(self, trial_id: int, values: dict[str, Any]) -> None:
        self.id = trial_id
        self.values = values

    def get_int(self, name: str) -> int:
        return int(self.values[name])

    def get_float(self, name: str) -> float:
        return float(self.values[name])

    def get_category(self, name: str) -> tuple[int, Any]:
        index, label = self.values[name]
        return index, label


class _FakeTrialRecord:
    def __init__(
        self,
        *,
        trial_id: int,
        params: dict[str, Any],
        status: str,
        score: Any,
    ) -> None:
        self.id = trial_id
        self.params = params
        self.status = status
        self.score = score


class _FakeOptimizer:
    last_kwargs: dict[str, Any] = {}
    last_enqueued: list[dict[str, Any]] = []

    def __init__(self, space: _FakeSearchSpace, **_kwargs: Any) -> None:
        self.space = space
        type(self).last_kwargs = dict(_kwargs)
        type(self).last_enqueued = []
        self.trials = [_FakeTrial(index, self._trial_values(index)) for index in range(2)]
        self.told: list[tuple[int, float]] = []
        self.failed: list[tuple[int, str, str]] = []
        self.records: list[_FakeTrialRecord] = []

    def enqueue(self, params: dict[str, Any]) -> None:
        type(self).last_enqueued.append(dict(params))
        values: dict[str, Any] = {}
        for name, (kind, spec) in self.space.params.items():
            if name not in params:
                continue
            if kind == "categorical":
                choices = spec
                label = params[name]
                values[name] = (choices.index(label), label)
            else:
                values[name] = params[name]
        self.trials.insert(0, _FakeTrial(99, values))

    def ask(self) -> _FakeTrial:
        return self.trials[len(self.told)]

    def tell(self, trial_id: int, score: float) -> None:
        self.told.append((trial_id, score))
        trial = next(trial for trial in self.trials if trial.id == trial_id)
        self.records.append(
            _FakeTrialRecord(
                trial_id=trial_id,
                params=self._trial_params(trial),
                status="COMPLETED",
                score=score,
            )
        )

    def tell_result(self, trial_id: int, status: str, *, error: str = "") -> None:
        self.failed.append((trial_id, status, error))
        self.told.append((trial_id, float("inf")))
        trial = next(trial for trial in self.trials if trial.id == trial_id)
        self.records.append(
            _FakeTrialRecord(
                trial_id=trial_id,
                params=self._trial_params(trial),
                status=str(status).upper(),
                score=None,
            )
        )

    def best(self) -> tuple[_FakeTrial, float]:
        trial_id, score = min(self.told, key=lambda item: item[1])
        return next(trial for trial in self.trials if trial.id == trial_id), score

    def get_trials(self) -> list[_FakeTrialRecord]:
        return list(self.records)

    def save(self) -> bytes:
        return json.dumps(
            {
                "records": [
                    {
                        "id": record.id,
                        "params": record.params,
                        "score": record.score,
                        "status": record.status,
                    }
                    for record in self.records
                ],
                "space_params": self.space.params,
                "told": self.told,
            },
            sort_keys=True,
        ).encode()

    @classmethod
    def load(cls, checkpoint: bytes) -> _FakeOptimizer:
        payload = json.loads(checkpoint.decode())
        space = _FakeSearchSpace()
        for name, (kind, spec) in payload["space_params"].items():
            if kind == "categorical":
                space.add_categorical(name, spec)
            elif kind == "int":
                low, high, step, log = spec
                space.add_int(name, low, high, step, log)
            else:
                low, high, step, log = spec
                space.add_float(name, low, high, step, log)
        restored = cls(space)
        restored.told = [(int(trial_id), float(score)) for trial_id, score in payload["told"]]
        restored.records = [
            _FakeTrialRecord(
                trial_id=record["id"],
                params=dict(record["params"]),
                status=str(record["status"]),
                score=record["score"],
            )
            for record in payload["records"]
        ]
        return restored

    def close(self) -> None:
        self.closed = True

    def _trial_params(self, trial: _FakeTrial) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name, (kind, spec) in self.space.params.items():
            if kind == "categorical":
                _, label = trial.get_category(name)
                params[name] = label
            elif kind == "int":
                params[name] = trial.get_int(name)
            else:
                params[name] = trial.get_float(name)
        return params

    def _trial_values(self, trial_index: int) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name, (kind, spec) in self.space.params.items():
            if kind == "categorical":
                choices = spec
                index = trial_index % len(choices)
                values[name] = (index, choices[index])
            elif kind == "int":
                low, high, _step, _log = spec
                values[name] = high if trial_index == 0 else low
            else:
                low, high, _step, _log = spec
                values[name] = high if trial_index == 0 else low
        return values


class _FakeN4MApi:
    SearchSpace = _FakeSearchSpace
    Optimizer = _FakeOptimizer
    Sampler = _FakeEnum
    Direction = _FakeEnum
    Pruner = _FakeEnum
    TrialStatus = _FakeEnum


class _FakeN4MApiWithoutEnqueue:
    class Optimizer(_FakeOptimizer):
        enqueue = None

    SearchSpace = _FakeSearchSpace
    Sampler = _FakeEnum
    Direction = _FakeEnum
    Pruner = _FakeEnum
    TrialStatus = _FakeEnum


class _FakeOptimizerWithoutFailureTell(_FakeOptimizer):
    tell = None
    tell_result = None


class _FakeN4MApiWithoutFailureTell:
    SearchSpace = _FakeSearchSpace
    Optimizer = _FakeOptimizerWithoutFailureTell
    Sampler = _FakeEnum
    Direction = _FakeEnum
    Pruner = _FakeEnum
    TrialStatus = _FakeEnum


def _rewrite_fake_n4m_checkpoint_record(
    checkpoint_path: Any,
    *,
    params: dict[str, Any] | None = None,
    status: str,
    score: Any = None,
) -> None:
    manifest = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(base64.b64decode(manifest["checkpoint_b64"]).decode())
    if params is not None:
        checkpoint["records"][0]["params"] = params
    checkpoint["records"][0]["status"] = status
    checkpoint["records"][0]["score"] = score
    checkpoint_b64 = base64.b64encode(json.dumps(checkpoint, sort_keys=True).encode()).decode("ascii")
    manifest["checkpoint_b64"] = checkpoint_b64
    manifest["checkpoint_fingerprint"] = tcv1_sha256({"checkpoint_b64": checkpoint_b64})
    checkpoint_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reverse_fake_n4m_checkpoint_records(checkpoint_path: Any) -> None:
    manifest = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(base64.b64decode(manifest["checkpoint_b64"]).decode())
    checkpoint["records"] = list(reversed(checkpoint["records"]))
    checkpoint_b64 = base64.b64encode(json.dumps(checkpoint, sort_keys=True).encode()).decode("ascii")
    manifest["checkpoint_b64"] = checkpoint_b64
    manifest["checkpoint_fingerprint"] = tcv1_sha256({"checkpoint_b64": checkpoint_b64})
    checkpoint_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _duplicate_fake_n4m_checkpoint_record_id(checkpoint_path: Any) -> None:
    manifest = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(base64.b64decode(manifest["checkpoint_b64"]).decode())
    checkpoint["records"][1]["id"] = checkpoint["records"][0]["id"]
    checkpoint_b64 = base64.b64encode(json.dumps(checkpoint, sort_keys=True).encode()).decode("ascii")
    manifest["checkpoint_b64"] = checkpoint_b64
    manifest["checkpoint_fingerprint"] = tcv1_sha256({"checkpoint_b64": checkpoint_b64})
    checkpoint_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rewrite_fake_n4m_checkpoint_record_id(checkpoint_path: Any, trial_id: Any) -> None:
    manifest = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(base64.b64decode(manifest["checkpoint_b64"]).decode())
    checkpoint["records"][0]["id"] = trial_id
    checkpoint_b64 = base64.b64encode(json.dumps(checkpoint, sort_keys=True).encode()).decode("ascii")
    manifest["checkpoint_b64"] = checkpoint_b64
    manifest["checkpoint_fingerprint"] = tcv1_sha256({"checkpoint_b64": checkpoint_b64})
    checkpoint_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_optuna_pruner_mapping_is_explicit() -> None:
    optuna = pytest.importorskip("optuna")

    assert type(_make_optuna_pruner(optuna, parse_tuning_spec({"engine": "optuna", "space": {"alpha": [0.2]}}))).__name__ == "NopPruner"
    assert (
        type(
            _make_optuna_pruner(
                optuna,
                parse_tuning_spec({"engine": "optuna", "space": {"alpha": [0.2]}, "pruner": "median"}),
            )
        ).__name__
        == "MedianPruner"
    )
    assert (
        type(
            _make_optuna_pruner(
                optuna,
                parse_tuning_spec({"engine": "optuna", "space": {"alpha": [0.2]}, "pruner": "successive_halving"}),
            )
        ).__name__
        == "SuccessiveHalvingPruner"
    )
    assert (
        type(
            _make_optuna_pruner(
                optuna,
                parse_tuning_spec({"engine": "optuna", "space": {"alpha": [0.2]}, "pruner": "hyperband"}),
            )
        ).__name__
        == "HyperbandPruner"
    )


def test_n4m_adapter_drives_pipeline_objective_with_ask_tell() -> None:
    objective = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {"engine": "n4m", "space": {"alpha": [0.9, 0.2]}, "sampler": "tpe", "n_trials": 2},
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    result = N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(objective, [[1.0]], [1.0])

    assert result.optimizer == "n4m"
    assert result.best_params == {"alpha": 0.2}
    assert result.best_value == pytest.approx(0.2)
    assert result.n_trials == 2
    assert {trial.state for trial in result.trials} == {"COMPLETE"}
    assert result.trials[0].diagnostics["score_family"] == "objective"
    assert result.trials[0].diagnostics["score_extractor"] == "objective"


def test_optuna_and_n4m_adapters_publish_same_ordered_search_space_fingerprint() -> None:
    tuning_space = {"alpha": [0.9, 0.2]}
    optuna_objective = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {"engine": "optuna", "space": tuning_space, "sampler": "grid", "n_trials": 2, "seed": 7},
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    n4m_objective = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {"engine": "n4m", "space": tuning_space, "sampler": "grid", "n_trials": 2, "seed": 7},
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    optuna_result = optimize_pipeline_objective(optuna_objective, [[1.0]], [1.0])
    n4m_result = N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(n4m_objective, [[1.0]], [1.0])

    expected_fingerprint = optuna_objective.tuning.ordered_search_space.fingerprint
    assert expected_fingerprint == n4m_objective.tuning.ordered_search_space.fingerprint
    assert optuna_result.trials[0].diagnostics["search_space_fingerprint"] == expected_fingerprint
    assert n4m_result.trials[0].diagnostics["search_space_fingerprint"] == expected_fingerprint
    assert optuna_result.tuning.fingerprint != n4m_result.tuning.fingerprint


def test_optuna_and_n4m_force_params_publish_same_public_trial_and_search_space_fingerprint() -> None:
    tuning_space = {"alpha": [0.9, 0.2]}
    force_params = {"alpha": 0.2}
    optuna_objective = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "optuna",
            "space": tuning_space,
            "force_params": force_params,
            "sampler": "grid",
            "n_trials": 1,
            "seed": 7,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    n4m_objective = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": tuning_space,
            "force_params": force_params,
            "sampler": "grid",
            "n_trials": 1,
            "seed": 7,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    optuna_result = optimize_pipeline_objective(optuna_objective, [[1.0]], [1.0])
    n4m_result = N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(n4m_objective, [[1.0]], [1.0])

    expected_fingerprint = optuna_objective.tuning.ordered_search_space.fingerprint
    assert optuna_result.trials[0].params == force_params
    assert n4m_result.trials[0].params == force_params
    assert optuna_result.best_params == force_params
    assert n4m_result.best_params == force_params
    assert optuna_result.trials[0].diagnostics["search_space_fingerprint"] == expected_fingerprint
    assert n4m_result.trials[0].diagnostics["search_space_fingerprint"] == expected_fingerprint
    assert optuna_result.trials[0].diagnostics["tuning_fingerprint"] == optuna_objective.tuning.fingerprint
    assert n4m_result.trials[0].diagnostics["tuning_fingerprint"] == n4m_objective.tuning.fingerprint
    assert optuna_objective.tuning.fingerprint != n4m_objective.tuning.fingerprint


def test_optuna_and_n4m_force_params_publish_same_named_categorical_public_value() -> None:
    tuning_space = {
        "mode": {
            "type": "categorical",
            "options": {
                "identity": {"kind": "passthrough"},
                "scaled": "scaled",
            },
        }
    }
    force_params = {"mode": {"kind": "passthrough"}}
    optuna_objective = PipelineObjective(
        _ChoiceEstimator(),
        {
            "engine": "optuna",
            "space": tuning_space,
            "force_params": force_params,
            "sampler": "grid",
            "n_trials": 1,
            "seed": 7,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    n4m_objective = PipelineObjective(
        _ChoiceEstimator(),
        {
            "engine": "n4m",
            "space": tuning_space,
            "force_params": force_params,
            "sampler": "grid",
            "n_trials": 1,
            "seed": 7,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    optuna_result = optimize_pipeline_objective(optuna_objective, [[1.0]], [1.0])
    n4m_result = N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(n4m_objective, [[1.0]], [1.0])

    expected_fingerprint = optuna_objective.tuning.ordered_search_space.fingerprint
    assert optuna_result.trials[0].params == force_params
    assert n4m_result.trials[0].params == force_params
    assert optuna_result.best_params == force_params
    assert n4m_result.best_params == force_params
    assert optuna_result.trials[0].diagnostics["search_space_fingerprint"] == expected_fingerprint
    assert n4m_result.trials[0].diagnostics["search_space_fingerprint"] == expected_fingerprint


def test_n4m_adapter_enqueues_force_params_as_first_public_trial() -> None:
    objective = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {"engine": "n4m", "space": {"alpha": [0.9, 0.2]}, "force_params": {"alpha": 0.2}, "n_trials": 1},
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    result = N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(objective, [[1.0]], [1.0])

    assert result.best_params == {"alpha": 0.2}
    assert result.trials[0].params == {"alpha": 0.2}
    assert result.trials[0].number == 99
    assert _FakeOptimizer.last_enqueued == [{"alpha": 0.2}]


def test_n4m_adapter_rejects_force_params_without_enqueue_support() -> None:
    objective = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {"engine": "n4m", "space": {"alpha": [0.9, 0.2]}, "force_params": {"alpha": 0.2}, "n_trials": 1},
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    with pytest.raises(ValueError, match=r"requires optimizer\.enqueue"):
        N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApiWithoutEnqueue).optimize(objective, [[1.0]], [1.0])


def test_n4m_adapter_decodes_json_native_categorical_choices() -> None:
    objective = PipelineObjective(
        _ChoiceEstimator(),
        {"engine": "n4m", "space": {"mode": [{"kind": "passthrough"}, "scaled"]}, "sampler": "tpe", "n_trials": 2},
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    result = N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(objective, [[1.0]], [1.0])

    assert result.best_params == {"mode": {"kind": "passthrough"}}
    assert result.trials[0].params == {"mode": {"kind": "passthrough"}}
    assert result.trials[1].params == {"mode": "scaled"}


def test_n4m_adapter_decodes_named_categorical_options_to_values() -> None:
    objective = PipelineObjective(
        _ChoiceEstimator(),
        {
            "engine": "n4m",
            "space": {
                "mode": {
                    "type": "categorical",
                    "options": {
                        "identity": {"kind": "passthrough"},
                        "scaled": "scaled",
                    },
                }
            },
            "sampler": "tpe",
            "n_trials": 2,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    result = N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(objective, [[1.0]], [1.0])

    assert result.best_params == {"mode": {"kind": "passthrough"}}
    assert result.trials[0].params == {"mode": {"kind": "passthrough"}}
    assert result.trials[1].params == {"mode": "scaled"}


def test_n4m_adapter_threads_optimizer_pruner() -> None:
    objective = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "random",
            "pruner": "successive_halving",
            "seed": 11,
            "n_trials": 2,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(objective, [[1.0]], [1.0])

    assert _FakeOptimizer.last_kwargs == {
        "sampler": "random",
        "direction": "minimize",
        "seed": 11,
        "pruner": "asha",
    }


def test_n4m_adapter_accepts_normalized_sampler_and_pruner_values() -> None:
    objective = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": " RANDOM ",
            "pruner": " HyperBand ",
            "seed": 11,
            "n_trials": 2,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(objective, [[1.0]], [1.0])

    assert _FakeOptimizer.last_kwargs == {
        "sampler": "random",
        "direction": "minimize",
        "seed": 11,
        "pruner": "hyperband",
    }


def test_n4m_adapter_records_failed_trials_and_keeps_successful_best() -> None:
    objective = PipelineObjective(
        _SometimesFailingEstimator(alpha=1.0),
        {"engine": "n4m", "space": {"alpha": [0.9, 0.2]}, "sampler": "tpe", "n_trials": 2},
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    result = N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(objective, [[1.0]], [1.0])

    assert result.best_params == {"alpha": 0.2}
    assert result.best_value == pytest.approx(0.2)
    assert {trial.state for trial in result.trials} == {"COMPLETE", "FAIL"}
    failed = next(trial for trial in result.trials if trial.state == "FAIL")
    assert failed.value is None
    assert failed.diagnostics["error_type"] == "RuntimeError"
    assert failed.diagnostics["score_extractor"] == "failed"


def test_n4m_adapter_records_pruned_trials_separately_from_failures() -> None:
    pytest.importorskip("optuna")
    objective = PipelineObjective(
        _SometimesPrunedEstimator(alpha=1.0),
        {"engine": "n4m", "space": {"alpha": [0.9, 0.2]}, "sampler": "tpe", "n_trials": 2},
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    result = N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(objective, [[1.0]], [1.0])

    assert result.best_params == {"alpha": 0.2}
    assert result.best_value == pytest.approx(0.2)
    assert {trial.state for trial in result.trials} == {"COMPLETE", "PRUNED"}
    pruned = next(trial for trial in result.trials if trial.state == "PRUNED")
    assert pruned.value is None
    assert pruned.diagnostics["error_type"] == "TrialPruned"
    assert pruned.diagnostics["score_extractor"] == "pruned"
    assert result.summary_artifact()["trial_states"] == {"COMPLETE": 1, "PRUNED": 1}
    pruned_summary = next(row for row in result.summary_artifact()["trials"] if row["state"] == "PRUNED")
    assert pruned_summary["diagnostics"]["score_extractor"] == "pruned"


def test_n4m_adapter_fails_closed_when_binding_cannot_record_failed_trials() -> None:
    objective = PipelineObjective(
        _SometimesFailingEstimator(alpha=1.0),
        {"engine": "n4m", "space": {"alpha": [0.9, 0.2]}, "sampler": "tpe", "n_trials": 2},
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    with pytest.raises(ValueError, match=r"requires optimizer\.tell_result"):
        N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApiWithoutFailureTell).optimize(objective, [[1.0]], [1.0])


def test_n4m_adapter_fails_closed_when_binding_cannot_record_pruned_trials() -> None:
    pytest.importorskip("optuna")
    objective = PipelineObjective(
        _SometimesPrunedEstimator(alpha=1.0),
        {"engine": "n4m", "space": {"alpha": [0.9, 0.2]}, "sampler": "tpe", "n_trials": 2},
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    with pytest.raises(ValueError, match=r"TrialStatus\.PRUNED"):
        N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApiWithoutFailureTell).optimize(objective, [[1.0]], [1.0])


@pytest.mark.parametrize(("direction", "expected_best_value"), [("minimize", 1.0e308), ("maximize", -1.0e308)])
def test_n4m_adapter_all_failed_trials_keep_tcv1_compatible_summary(
    direction: str,
    expected_best_value: float,
) -> None:
    objective = PipelineObjective(
        _AlwaysFailingEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "direction": direction,
            "space": {"alpha": [0.9, 0.8]},
            "sampler": "tpe",
            "n_trials": 2,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    result = N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(objective, [[1.0]], [1.0])

    assert result.best_params == {}
    assert result.best_value == expected_best_value
    assert {trial.state for trial in result.trials} == {"FAIL"}
    assert len(result.fingerprint) == 64
    assert result.summary_artifact()["best_value"] == expected_best_value
    assert result.summary_artifact()["trial_states"] == {"FAIL": 2}
    assert "nirs4all.tuning.summary" in result.to_summary_json()


def test_n4m_adapter_supports_grid_sampler_when_native_enum_exists() -> None:
    objective = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {"engine": "n4m", "space": {"alpha": [0.9, 0.2]}, "sampler": "grid", "n_trials": 2},
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    result = N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(objective, [[1.0]], [1.0])

    assert result.optimizer == "n4m"
    assert result.best_params == {"alpha": 0.2}
    assert result.best_value == pytest.approx(0.2)
    assert _FakeOptimizer.last_kwargs["sampler"] == "grid"


def test_n4m_adapter_grid_sampler_fails_closed_for_old_bindings() -> None:
    class _SamplerWithoutGrid:
        TPE = "tpe"
        RANDOM = "random"

    class _ApiWithoutGrid(_FakeN4MApi):
        Sampler = _SamplerWithoutGrid

    objective = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {"engine": "n4m", "space": {"alpha": [0.9, 0.2]}, "sampler": "grid", "n_trials": 2},
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    with pytest.raises(ValueError, match="requires optimizer enum 'GRID'"):
        N4MPipelineObjectiveAdapter(optimizer_api=_ApiWithoutGrid).optimize(objective, [[1.0]], [1.0])


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"resume": True}, "requires storage='file:///"),
        ({"storage": "sqlite:///study.db", "study_name": "n4m-study"}, "only supports storage='file:///directory'"),
        ({"storage": "file:///tmp/n4m-study"}, "requires study_name"),
        ({"study_name": "n4m-study"}, "requires storage='file:///"),
        ({"storage": "file:///tmp/n4m-study", "study_name": "../bad"}, "filename-safe"),
    ],
)
def test_n4m_adapter_rejects_ambiguous_optimizer_state_persistence_controls(
    extra: dict[str, Any],
    match: str,
) -> None:
    objective = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "tpe",
            "n_trials": 2,
            **extra,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    with pytest.raises(ValueError, match=match):
        N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(objective, [[1.0]], [1.0])


def test_n4m_adapter_persists_file_checkpoint_and_resumes_remaining_trials(tmp_path) -> None:
    storage = (tmp_path / "n4m-checkpoints").as_uri()
    first = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 1,
            "storage": storage,
            "study_name": "pls-study",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "storage": storage,
            "study_name": "pls-study",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    first_result = N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(first, [[1.0]], [1.0])
    checkpoint_path = tmp_path / "n4m-checkpoints" / "pls-study.n4mopt.json"
    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    second_result = N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(resumed, [[1.0]], [1.0])

    assert first_result.trials[0].params == {"alpha": 0.9}
    assert checkpoint_payload["format"] == "nirs4all.n4m.optimizer_checkpoint"
    assert checkpoint_payload["schema_version"] == 1
    assert checkpoint_payload["study_name"] == "pls-study"
    assert "checkpoint_b64" in checkpoint_payload
    assert second_result.best_params == {"alpha": 0.2}
    assert second_result.best_value == pytest.approx(0.2)
    assert [trial.params for trial in second_result.trials] == [{"alpha": 0.9}, {"alpha": 0.2}]
    assert [trial.state for trial in second_result.trials] == ["COMPLETE", "COMPLETE"]
    assert second_result.trials[0].diagnostics["score_extractor"] == "native_checkpoint"
    assert second_result.summary_artifact()["persistence"]["optimizer_state_resume_supported"] is True
    assert second_result.summary_artifact()["persistence"]["resume"] is True
    assert second_result.summary_artifact()["persistence"]["storage_configured"] is True


def test_n4m_adapter_resume_orders_checkpoint_records_by_trial_id(tmp_path) -> None:
    storage_dir = tmp_path / "n4m-checkpoints"
    storage = storage_dir.as_uri()
    first = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "storage": storage,
            "study_name": "ordered-study",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "storage": storage,
            "study_name": "ordered-study",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    first_result = N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(first, [[1.0]], [1.0])
    _reverse_fake_n4m_checkpoint_records(storage_dir / "ordered-study.n4mopt.json")
    resumed_result = N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(resumed, [[1.0]], [1.0])

    assert [trial.number for trial in first_result.trials] == [0, 1]
    assert [trial.number for trial in resumed_result.trials] == [0, 1]
    assert [trial.params for trial in resumed_result.trials] == [{"alpha": 0.9}, {"alpha": 0.2}]
    assert [trial.diagnostics["score_extractor"] for trial in resumed_result.trials] == ["native_checkpoint", "native_checkpoint"]


def test_n4m_adapter_resume_rejects_duplicate_checkpoint_trial_ids(tmp_path) -> None:
    storage_dir = tmp_path / "n4m-checkpoints"
    storage = storage_dir.as_uri()
    first = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "storage": storage,
            "study_name": "duplicate-id-study",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "storage": storage,
            "study_name": "duplicate-id-study",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(first, [[1.0]], [1.0])
    _duplicate_fake_n4m_checkpoint_record_id(storage_dir / "duplicate-id-study.n4mopt.json")

    with pytest.raises(ValueError, match="duplicate trial ids"):
        N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(resumed, [[1.0]], [1.0])


def test_n4m_adapter_resume_rejects_non_integer_checkpoint_trial_id(tmp_path) -> None:
    storage_dir = tmp_path / "n4m-checkpoints"
    storage = storage_dir.as_uri()
    first = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 1,
            "storage": storage,
            "study_name": "non-integer-id-study",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "storage": storage,
            "study_name": "non-integer-id-study",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(first, [[1.0]], [1.0])
    _rewrite_fake_n4m_checkpoint_record_id(storage_dir / "non-integer-id-study.n4mopt.json", "trial-zero")

    with pytest.raises(ValueError, match="trial id 'trial-zero' is not an integer"):
        N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(resumed, [[1.0]], [1.0])


@pytest.mark.parametrize(
    ("score", "match"),
    [
        (None, "missing a numeric finite score"),
        (True, "missing a numeric finite score"),
        (float("nan"), "non-finite score"),
    ],
)
def test_n4m_adapter_resume_rejects_complete_checkpoint_record_without_finite_score(
    tmp_path,
    score: object,
    match: str,
) -> None:
    storage_dir = tmp_path / "n4m-checkpoints"
    storage = storage_dir.as_uri()
    first = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 1,
            "storage": storage,
            "study_name": "bad-complete-score-study",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "storage": storage,
            "study_name": "bad-complete-score-study",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(first, [[1.0]], [1.0])
    _rewrite_fake_n4m_checkpoint_record(
        storage_dir / "bad-complete-score-study.n4mopt.json",
        status="COMPLETED",
        score=score,
    )

    with pytest.raises(ValueError, match=match):
        N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(resumed, [[1.0]], [1.0])


def test_n4m_adapter_resume_preserves_failed_checkpoint_diagnostics(tmp_path) -> None:
    storage = (tmp_path / "n4m-checkpoints").as_uri()
    first = PipelineObjective(
        _SometimesFailingEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 1,
            "storage": storage,
            "study_name": "failure-study",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _SometimesFailingEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "storage": storage,
            "study_name": "failure-study",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    first_result = N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(first, [[1.0]], [1.0])
    second_result = N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(resumed, [[1.0]], [1.0])

    assert first_result.trials[0].state == "FAIL"
    assert second_result.best_params == {"alpha": 0.2}
    assert [trial.state for trial in second_result.trials] == ["FAIL", "COMPLETE"]
    assert second_result.trials[0].diagnostics["score_extractor"] == "failed"
    assert second_result.summary_artifact()["trial_states"] == {"COMPLETE": 1, "FAIL": 1}


def test_n4m_adapter_resume_preserves_pruned_checkpoint_diagnostics(tmp_path) -> None:
    pytest.importorskip("optuna")
    storage = (tmp_path / "n4m-checkpoints").as_uri()
    first = PipelineObjective(
        _SometimesPrunedEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 1,
            "storage": storage,
            "study_name": "pruned-study",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _SometimesPrunedEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "storage": storage,
            "study_name": "pruned-study",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    first_result = N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(first, [[1.0]], [1.0])
    second_result = N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(resumed, [[1.0]], [1.0])

    assert first_result.trials[0].state == "PRUNED"
    assert second_result.best_params == {"alpha": 0.2}
    assert [trial.state for trial in second_result.trials] == ["PRUNED", "COMPLETE"]
    assert second_result.trials[0].diagnostics["score_extractor"] == "pruned"
    assert second_result.summary_artifact()["trial_states"] == {"COMPLETE": 1, "PRUNED": 1}


def test_n4m_adapter_resume_preserves_cancelled_checkpoint_diagnostics(tmp_path) -> None:
    storage_dir = tmp_path / "n4m-checkpoints"
    storage = storage_dir.as_uri()
    first = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 1,
            "storage": storage,
            "study_name": "cancelled-study",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "storage": storage,
            "study_name": "cancelled-study",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    first_result = N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(first, [[1.0]], [1.0])
    checkpoint_path = storage_dir / "cancelled-study.n4mopt.json"
    _rewrite_fake_n4m_checkpoint_record(checkpoint_path, status="CANCELLED")
    second_result = N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(resumed, [[1.0]], [1.0])

    assert first_result.trials[0].state == "COMPLETE"
    assert second_result.best_params == {"alpha": 0.2}
    assert [trial.state for trial in second_result.trials] == ["CANCELLED", "COMPLETE"]
    assert second_result.trials[0].value is None
    assert second_result.trials[0].diagnostics["score_extractor"] == "cancelled"
    assert second_result.summary_artifact()["trial_states"] == {"CANCELLED": 1, "COMPLETE": 1}


@pytest.mark.parametrize("status", ["FAILED", "PRUNED", "CANCELLED"])
def test_n4m_adapter_resume_rejects_non_complete_checkpoint_record_with_final_score(
    tmp_path,
    status: str,
) -> None:
    storage_dir = tmp_path / "n4m-checkpoints"
    storage = storage_dir.as_uri()
    first = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 1,
            "storage": storage,
            "study_name": "invalid-non-complete-score",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "storage": storage,
            "study_name": "invalid-non-complete-score",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(first, [[1.0]], [1.0])
    checkpoint_path = storage_dir / "invalid-non-complete-score.n4mopt.json"
    _rewrite_fake_n4m_checkpoint_record(checkpoint_path, status=status, score=0.9)

    expected_status = "FAIL" if status == "FAILED" else status
    with pytest.raises(ValueError, match=f"checkpoint {expected_status} trial 0 carries a final score"):
        N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(resumed, [[1.0]], [1.0])


def test_n4m_adapter_resume_rejects_non_terminal_checkpoint_record(tmp_path) -> None:
    storage_dir = tmp_path / "n4m-checkpoints"
    storage = storage_dir.as_uri()
    first = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 1,
            "storage": storage,
            "study_name": "running-study",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "storage": storage,
            "study_name": "running-study",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(first, [[1.0]], [1.0])
    _rewrite_fake_n4m_checkpoint_record(storage_dir / "running-study.n4mopt.json", status="RUNNING")

    with pytest.raises(ValueError, match="non-terminal or unsupported trial status 'RUNNING'"):
        N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(resumed, [[1.0]], [1.0])


def test_n4m_adapter_resume_rejects_checkpoint_record_with_changed_param_keys(tmp_path) -> None:
    storage_dir = tmp_path / "n4m-checkpoints"
    storage = storage_dir.as_uri()
    first = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 1,
            "storage": storage,
            "study_name": "record-key-study",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "storage": storage,
            "study_name": "record-key-study",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(first, [[1.0]], [1.0])
    _rewrite_fake_n4m_checkpoint_record(
        storage_dir / "record-key-study.n4mopt.json",
        params={"beta": 0.9},
        status="COMPLETED",
        score=0.9,
    )

    with pytest.raises(ValueError, match="trial params that do not match tuning.space keys"):
        N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(resumed, [[1.0]], [1.0])


def test_n4m_adapter_resume_rejects_checkpoint_record_with_changed_numeric_range(tmp_path) -> None:
    storage_dir = tmp_path / "n4m-checkpoints"
    storage = storage_dir.as_uri()
    first = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": {"type": "float", "low": 0.0, "high": 1.0}},
            "sampler": "random",
            "n_trials": 1,
            "storage": storage,
            "study_name": "record-range-study",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": {"type": "float", "low": 0.0, "high": 1.0}},
            "sampler": "random",
            "n_trials": 2,
            "storage": storage,
            "study_name": "record-range-study",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(first, [[1.0]], [1.0])
    _rewrite_fake_n4m_checkpoint_record(
        storage_dir / "record-range-study.n4mopt.json",
        params={"alpha": 1.5},
        status="COMPLETED",
        score=1.5,
    )

    with pytest.raises(ValueError, match="trial value 1.5.*outside the current tuning.space range"):
        N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(resumed, [[1.0]], [1.0])


def test_n4m_adapter_resume_rejects_checkpoint_record_with_changed_numeric_step(tmp_path) -> None:
    storage_dir = tmp_path / "n4m-checkpoints"
    storage = storage_dir.as_uri()
    first = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": {"type": "float", "low": 0.0, "high": 1.0, "step": 0.2}},
            "sampler": "random",
            "n_trials": 1,
            "storage": storage,
            "study_name": "record-step-study",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": {"type": "float", "low": 0.0, "high": 1.0, "step": 0.2}},
            "sampler": "random",
            "n_trials": 2,
            "storage": storage,
            "study_name": "record-step-study",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(first, [[1.0]], [1.0])
    _rewrite_fake_n4m_checkpoint_record(
        storage_dir / "record-step-study.n4mopt.json",
        params={"alpha": 0.9},
        status="COMPLETED",
        score=0.9,
    )

    with pytest.raises(ValueError, match="trial value 0.9.*outside the current tuning.space range"):
        N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(resumed, [[1.0]], [1.0])


def test_n4m_adapter_resume_rejects_checkpoint_record_with_changed_categorical_choices(tmp_path) -> None:
    storage_dir = tmp_path / "n4m-checkpoints"
    storage = storage_dir.as_uri()
    space = {"mode": ["old", "stable"]}
    first = PipelineObjective(
        _ChoiceEstimator(),
        {
            "engine": "n4m",
            "space": space,
            "sampler": "grid",
            "n_trials": 1,
            "storage": storage,
            "study_name": "record-choice-study",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _ChoiceEstimator(),
        {
            "engine": "n4m",
            "space": space,
            "sampler": "grid",
            "n_trials": 2,
            "storage": storage,
            "study_name": "record-choice-study",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(first, [[1.0]], [1.0])
    _rewrite_fake_n4m_checkpoint_record(
        storage_dir / "record-choice-study.n4mopt.json",
        params={"mode": "removed"},
        status="COMPLETED",
        score=10.0,
    )

    with pytest.raises(ValueError, match="trial categorical value 'removed'.*not present"):
        N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(resumed, [[1.0]], [1.0])


def test_n4m_adapter_resume_decodes_checkpoint_categorical_public_values(tmp_path) -> None:
    storage = (tmp_path / "n4m-checkpoints").as_uri()
    space = {
        "mode": {
            "type": "categorical",
            "options": {
                "identity": {"kind": "passthrough"},
                "scaled": "scaled",
            },
        }
    }
    first = PipelineObjective(
        _ChoiceEstimator(),
        {
            "engine": "n4m",
            "space": space,
            "sampler": "grid",
            "n_trials": 1,
            "storage": storage,
            "study_name": "choice-study",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    resumed = PipelineObjective(
        _ChoiceEstimator(),
        {
            "engine": "n4m",
            "space": space,
            "sampler": "grid",
            "n_trials": 2,
            "storage": storage,
            "study_name": "choice-study",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    first_result = N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(first, [[1.0]], [1.0])
    second_result = N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(resumed, [[1.0]], [1.0])

    assert first_result.trials[0].params == {"mode": {"kind": "passthrough"}}
    assert second_result.trials[0].diagnostics["score_extractor"] == "native_checkpoint"
    assert second_result.trials[0].params == {"mode": {"kind": "passthrough"}}
    assert second_result.trials[1].params == {"mode": "scaled"}
    assert second_result.best_params == {"mode": {"kind": "passthrough"}}


def test_n4m_adapter_resume_rejects_mismatched_checkpoint_contract(tmp_path) -> None:
    storage = (tmp_path / "n4m-checkpoints").as_uri()
    initial = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 1,
            "storage": storage,
            "study_name": "pls-study",
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    changed = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        {
            "engine": "n4m",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "random",
            "n_trials": 2,
            "storage": storage,
            "study_name": "pls-study",
            "resume": True,
        },
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(initial, [[1.0]], [1.0])

    with pytest.raises(ValueError, match="optimizer_contract_fingerprint"):
        N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(changed, [[1.0]], [1.0])


def test_n4m_adapter_rejects_existing_checkpoint_without_resume(tmp_path) -> None:
    storage = (tmp_path / "n4m-checkpoints").as_uri()
    payload = {
        "engine": "n4m",
        "space": {"alpha": [0.9, 0.2]},
        "sampler": "grid",
        "n_trials": 1,
        "storage": storage,
        "study_name": "pls-study",
    }
    objective = PipelineObjective(
        _AdapterEstimator(alpha=1.0),
        payload,
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(objective, [[1.0]], [1.0])

    with pytest.raises(ValueError, match="existing optimizer checkpoint"):
        N4MPipelineObjectiveAdapter(optimizer_api=_FakeN4MApi).optimize(
            PipelineObjective(
                _AdapterEstimator(alpha=1.0),
                payload,
                score_extractor=lambda fitted: fitted.training_result_["score"],
            ),
            [[1.0]],
            [1.0],
        )
