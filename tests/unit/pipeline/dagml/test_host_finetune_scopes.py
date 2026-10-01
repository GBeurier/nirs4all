"""Proposal paths and durable local scopes retain native checkpoint authority."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nirs4all.pipeline.dagml.framework_estimator import DagMLFrameworkEstimator
from nirs4all.pipeline.dagml.host_finetune import _estimator_proposal, run_scoped_finetune
from nirs4all.pipeline.dagml.torch_estimator import DagMLTorchEstimator


class ConfiguredEstimator(BaseEstimator):
    def __init__(self, inference_config: dict[str, Any] | None = None) -> None:
        self.inference_config = inference_config


def test_nested_proposal_uses_actual_sklearn_parameter_paths() -> None:
    model = Pipeline([("model", Ridge())])
    values = {"model": {"alpha": 0.25, "tol": 0.002}}
    before = copy.deepcopy(values)
    assert _estimator_proposal(model, values) == {"model__alpha": 0.25, "model__tol": 0.002}
    assert values == before
    assert model.get_params(deep=True)["model__alpha"] == 1.0


def test_actual_dictionary_parameter_is_preserved_without_flattening() -> None:
    config = {"thresholds": {"low": 0.1, "high": 0.9}, "output": "mean"}
    proposal = _estimator_proposal(ConfiguredEstimator(), {"inference_config": config})
    assert proposal == {"inference_config": config}
    assert proposal["inference_config"] is config


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow", "jax"])
def test_framework_proposals_preserve_unset_factory_argument_routing(framework: str) -> None:
    adapter = DagMLTorchEstimator() if framework == "pytorch" else DagMLFrameworkEstimator(framework=framework)
    values = {"filters1": 8, "inference_config": {"threshold": 0.75}}
    assert "filters1" not in adapter.get_params(deep=True)
    assert "inference_config" not in adapter.get_params(deep=True)
    proposal = _estimator_proposal(adapter, values)
    assert proposal == values
    assert proposal["inference_config"] is values["inference_config"]
    assert adapter.factory_params is None
    adapter.set_params(**proposal)
    assert adapter.factory_params == values


@pytest.mark.parametrize("proposal", [{"unknown": 1.0}, {"model": {"unknown": 1.0}}])
def test_unknown_proposal_path_is_refused(proposal: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="not supported"):
        _estimator_proposal(Pipeline([("model", Ridge())]), proposal)


@pytest.mark.parametrize("reverse", [False, True])
def test_nested_and_flat_proposals_for_the_same_parameter_conflict(reverse: bool) -> None:
    items = [("model", {"alpha": 0.25}), ("model__alpha", 0.5)]
    with pytest.raises(ValueError, match="conflicting proposals"):
        _estimator_proposal(Pipeline([("model", Ridge())]), dict(reversed(items) if reverse else items))


def _local_inputs(tmp_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    x = np.arange(24, dtype=float).reshape(12, 2)
    y = 0.5 * x[:, 0] + np.asarray([0.0, 0.2, -0.1] * 4)
    config = {"engine": "optuna", "sampler": "random", "seed": 17, "n_trials": 1,
              "approach": "grouped", "eval_mode": "mean", "model_params": {"alpha": [0.1, 1.0]},
              "storage": f"sqlite:///{tmp_path / 'local.sqlite3'}", "study_name": "local"}
    scope = {"node_id": "model:local", "variant_id": "host_hpo:trial:0000000000", "phase": "FIT_CV",
             "fold_id": "outer:0", "training_sample_ids": [f"wire:{row}" for row in range(len(y))],
             "recipe_fingerprint": "a" * 64, "source_names": ["nir"],
             "data_content_fingerprint": "b" * 64, "target_content_fingerprint": "c" * 64,
             "optimizer_fingerprint": "d" * 64}
    return x, y, config, scope


def _local_search(x: np.ndarray, y: np.ndarray, config: dict[str, Any], scope: dict[str, Any], *,
                  with_mean: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    folds = [(list(range(6)), list(range(6, 12))), (list(range(6, 12)), list(range(6)))]
    return run_scoped_finetune(Ridge(), [StandardScaler(with_mean=with_mean)], x, y, config,
                              scope=scope, task_type="regression",
                              inner_cv={"folds": folds, "group_by_sample": None})


def test_local_user_study_identity_survives_resume_and_budget_extension(tmp_path: Path) -> None:
    x, y, config, scope = _local_inputs(tmp_path)
    _, first = _local_search(x, y, config, scope)
    name = first["optimizer"]["study_name"]
    initial = optuna.load_study(study_name=name, storage=config["storage"])
    checkpoint = copy.deepcopy(initial.user_attrs["nirs4all_dagml_host_hpo_checkpoint_v1"])
    resumed_scope = {**scope, "optimizer_fingerprint": "e" * 64}
    _, unchanged = _local_search(x, y, {**config, "resume": True}, resumed_scope)
    _, extended = _local_search(x, y, {**config, "n_trials": 3, "resume": True}, resumed_scope)
    assert unchanged["optimizer"]["study_name"] == extended["optimizer"]["study_name"] == name
    summaries = optuna.get_all_study_summaries(storage=config["storage"])
    assert [summary.study_name for summary in summaries] == [name]
    saved = optuna.load_study(study_name=name, storage=config["storage"])
    assert len(saved.trials) == 3
    assert saved.user_attrs["nirs4all_dagml_host_hpo_checkpoint_v1"]["trials"][:1] == checkpoint["trials"]


@pytest.mark.parametrize("mutation", ["data", "targets", "recipe"])
def test_local_user_study_rejects_changed_science_without_new_study_or_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    x, y, config, scope = _local_inputs(tmp_path)
    _, first = _local_search(x, y, config, scope)
    name = first["optimizer"]["study_name"]
    initial = optuna.load_study(study_name=name, storage=config["storage"])
    checkpoint = copy.deepcopy(initial.user_attrs["nirs4all_dagml_host_hpo_checkpoint_v1"])
    if mutation == "data":
        x = x.copy()
        x[0, 0] += 10
        scope = {**scope, "data_content_fingerprint": "e" * 64}
    elif mutation == "targets":
        y = y.copy()
        y[0] += 10
        scope = {**scope, "target_content_fingerprint": "e" * 64}
    else:
        scope = {**scope, "recipe_fingerprint": "e" * 64}
    monkeypatch.setattr(Ridge, "fit", lambda *a, **k: pytest.fail("incompatible local resume fit a model"))
    monkeypatch.setattr(StandardScaler, "fit", lambda *a, **k: pytest.fail("incompatible local resume fit preprocessing"))
    with pytest.raises(Exception, match="host HPO checkpoint objective/graph/controller/data/fold binding mismatch"):
        _local_search(x, y, {**config, "n_trials": 3, "resume": True}, scope, with_mean=mutation != "recipe")
    assert [summary.study_name for summary in optuna.get_all_study_summaries(storage=config["storage"])] == [name]
    saved = optuna.load_study(study_name=name, storage=config["storage"])
    assert len(saved.trials) == 1
    assert saved.user_attrs["nirs4all_dagml_host_hpo_checkpoint_v1"] == checkpoint
