"""Durable multimodal search with real DAG scheduling and native N4M state."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from nirs4all_io import MultimodalDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

import nirs4all
from nirs4all.operators.models.multimodal import MultimodalRegressor
from nirs4all.pipeline.dagml.multimodal_tuning import MultimodalTuningStopped
from tests.integration.api.test_multimodal_dagml import _cohort, _model


def _tuning(directory: Path, *, resume: bool = False, failed_candidate: bool = False) -> dict[str, Any]:
    return {
        "engine": "n4m", "sampler": "random", "seed": 7 if failed_candidate else 19,
        "metric": "rmse", "direction": "minimize", "n_trials": 4,
        "storage": directory.as_uri(), "study_name": "multimodal", "resume": resume,
        "space": {
            "model__alpha": [0.1, 1.0],
            "source_weights__image": [0.5, 1.0],
            "transformers__image__n_components": [1, 2, 99] if failed_candidate else [1, 2],
        },
    }


def _run(cohort: MultimodalDataset, tuning: dict[str, Any], workspace: Path, *, n_splits: int = 3) -> Any:
    return nirs4all.run(
        [GroupKFold(n_splits), {"model": _model()}], cohort, tuning=tuning,
        engine="dag-ml", workspace_path=workspace, verbose=0, save_charts=False,
        random_state=19, refit=True,
    )


def _checkpoint(directory: Path) -> dict[str, Any]:
    checkpoint: dict[str, Any] = json.loads((directory / "multimodal.n4mopt.json").read_text(encoding="utf-8"))
    return checkpoint


def _stop_after(count: int, events: list[dict[str, Any]]) -> Any:
    def callback(event: dict[str, Any]) -> bool:
        events.append(copy.deepcopy(event))
        return len(event["checkpoint"]["trials"]) < count

    return callback


@pytest.mark.parametrize("perturb_test_labels", [False, True])
def test_stopped_search_resumes_native_history_identically_to_continuous_search(perturb_test_labels: bool, tmp_path: Path) -> None:
    cohort = _cohort(unequal_groups=True)
    directory = tmp_path / "resumed-study"
    events: list[dict[str, Any]] = []
    interrupted = {**_tuning(directory), "progress_callback": _stop_after(2, events)}
    with pytest.raises(MultimodalTuningStopped) as stopped:
        _run(cohort, interrupted, tmp_path / "interrupted-workspace")
    assert stopped.value.evidence["status"] == "cancelled"
    prefix = _checkpoint(directory)["native_checkpoint"]["trials"]
    assert len(prefix) == 2
    assert all(record["state"] == "complete" for record in prefix)
    assert stopped.value.evidence["checkpoint"]["trials"] == prefix
    assert events and len(events[-1]["checkpoint"]["trials"]) == 2

    resumed_cohort = cohort
    if perturb_test_labels:
        targets = np.array(cohort.y, copy=True)
        targets[12:] += [1000, -2000, 3000, -4000]
        resumed_cohort = MultimodalDataset(
            cohort.sources, sample_ids=cohort.sample_ids, y=targets,
            groups=cohort.groups, partitions=cohort.partitions, name=cohort.name,
        )
    resumed = _run(resumed_cohort, _tuning(directory, resume=True), tmp_path / "resumed-workspace")
    continuous = _run(cohort, _tuning(tmp_path / "continuous-study"), tmp_path / "continuous-workspace")
    try:
        assert resumed.tuning_result is not None and continuous.tuning_result is not None
        resumed_trials = resumed.tuning_result.trials
        continuous_trials = continuous.tuning_result.trials
        assert [trial.number for trial in resumed_trials] == [0, 1, 2, 3]
        assert [trial.to_dict() for trial in resumed_trials] == [trial.to_dict() for trial in continuous_trials]
        assert all(trial.state == "COMPLETE" for trial in resumed_trials)
        assert all(trial.diagnostics["test_used"] is False for trial in resumed_trials)
        assert resumed.tuning_best_params == continuous.tuning_best_params
        assert resumed.tuning_best_value == pytest.approx(continuous.tuning_best_value, abs=1e-12)
        assert resumed.tuning_best_value == min(trial.value for trial in resumed_trials)
        saved = _checkpoint(directory)["native_checkpoint"]["trials"]
        assert len(saved) == 4
        assert saved[:2] == prefix
        assert saved == _checkpoint(tmp_path / "continuous-study")["native_checkpoint"]["trials"]
        assert len(resumed._dagml_refit_artifacts) == 1
        if perturb_test_labels:
            assert resumed.best_rmse != pytest.approx(continuous.best_rmse)
        predictor = resumed._dagml_refit_artifacts[0]["estimator"]
        baseline = continuous._dagml_refit_artifacts[0]["estimator"]
        new_data = _cohort(prediction=True)
        np.testing.assert_allclose(
            predictor.predict([new_data.sources[name].values for name in predictor.source_names]),
            baseline.predict([new_data.sources[name].values for name in baseline.source_names]),
            rtol=1e-12, atol=1e-12,
        )
    finally:
        resumed.close()
        continuous.close()


def test_failed_raw_encoder_trial_is_checkpointed_and_not_evaluated_again_on_resume(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cohort = _cohort(unequal_groups=True)
    directory = tmp_path / "failed-study"
    # Native random seed 7 proposes 99, 1, 1, 99 components for this space.
    # 99 exceeds the raw fold's eight rows, so sklearn PCA raises a real error.
    config = _tuning(directory, failed_candidate=True)
    seen: list[int] = []
    original_fit = MultimodalRegressor.fit

    def observe_fit(self: MultimodalRegressor, X: Any, y: Any) -> MultimodalRegressor:
        seen.append(int(self.get_params(deep=True)["transformers__image__n_components"]))
        return original_fit(self, X, y)

    monkeypatch.setattr(MultimodalRegressor, "fit", observe_fit)
    with pytest.raises(Exception, match="n_components=99"):
        _run(cohort, {**config, "progress_callback": _stop_after(1, [])}, tmp_path / "failed-workspace")
    prefix = _checkpoint(directory)["native_checkpoint"]["trials"]
    assert len(prefix) == 1
    failed = prefix[0]
    assert failed["state"] == "failed"
    assert failed.get("evidence", failed)["params"]["transformers.image.n_components"] == 99
    assert seen == [99]
    seen.clear()
    # Limit the resumed total to three trials; the failing fourth proposal is
    # outside this budget, so any repeated 99 is an invalid retry of trial 0.
    resumed = _run(cohort, {**config, "resume": True, "n_trials": 3}, tmp_path / "recovered-workspace")
    try:
        assert 99 not in seen
        assert resumed.tuning_result is not None
        trials = resumed.tuning_result.trials
        assert [trial.number for trial in trials] == [0, 1, 2]
        assert [trial.state for trial in trials] == ["FAIL", "COMPLETE", "COMPLETE"]
        assert trials[0].value is None
        assert all(trial.value is not None and np.isfinite(trial.value) for trial in trials[1:])
        assert resumed.tuning_best_params["transformers.image.n_components"] == 1
        saved = _checkpoint(directory)["native_checkpoint"]["trials"]
        assert len(saved) == 3 and saved[0] == failed
    finally:
        resumed.close()


def test_search_can_stop_before_first_trial_without_fitting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    directory = tmp_path / "empty-study"
    events: list[dict[str, Any]] = []

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("search stopped before its first trial reached estimator.fit")

    monkeypatch.setattr(MultimodalRegressor, "fit", forbidden)
    with pytest.raises(MultimodalTuningStopped) as stopped:
        _run(_cohort(), {**_tuning(directory), "progress_callback": _stop_after(0, events)}, tmp_path / "stopped")
    assert stopped.value.evidence["status"] == "cancelled"
    assert stopped.value.evidence["checkpoint"]["trials"] == []
    assert _checkpoint(directory)["native_checkpoint"]["trials"] == []
    assert events and all(event["checkpoint"]["trials"] == [] for event in events)


def test_completed_two_trial_budget_can_extend_to_four_without_changing_history(tmp_path: Path) -> None:
    cohort = _cohort(unequal_groups=True)
    directory = tmp_path / "extended-study"
    initial = _run(cohort, {**_tuning(directory), "n_trials": 2}, tmp_path / "initial")
    try:
        assert initial.tuning_result is not None
        assert len(initial.tuning_result.trials) == 2
        prefix = _checkpoint(directory)["native_checkpoint"]["trials"]
    finally:
        initial.close()
    resumed = _run(cohort, _tuning(directory, resume=True), tmp_path / "extended")
    continuous = _run(cohort, _tuning(tmp_path / "continuous-study"), tmp_path / "continuous")
    try:
        assert resumed.tuning_result is not None and continuous.tuning_result is not None
        assert [trial.number for trial in resumed.tuning_result.trials] == [0, 1, 2, 3]
        assert [trial.to_dict() for trial in resumed.tuning_result.trials] == [
            trial.to_dict() for trial in continuous.tuning_result.trials
        ]
        assert resumed.tuning_best_params == continuous.tuning_best_params
        assert resumed.tuning_best_value == pytest.approx(continuous.tuning_best_value, abs=1e-12)
        saved = _checkpoint(directory)["native_checkpoint"]["trials"]
        assert saved[:2] == prefix
        assert saved == _checkpoint(tmp_path / "continuous-study")["native_checkpoint"]["trials"]
    finally:
        resumed.close()
        continuous.close()


@pytest.mark.parametrize("mutation", ["features", "training_labels", "schema", "folds"])
def test_resume_refuses_changed_training_contract_before_fit(mutation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cohort = _cohort(unequal_groups=True)
    directory = tmp_path / mutation
    with pytest.raises(MultimodalTuningStopped):
        _run(cohort, {**_tuning(directory), "progress_callback": _stop_after(1, [])}, tmp_path / "initial")
    checkpoint_path = directory / "multimodal.n4mopt.json"
    before = checkpoint_path.read_bytes()
    sources = dict(cohort.sources)
    targets = np.array(cohort.y, copy=True)
    if mutation == "features":
        source = sources["nir"]
        values = np.array(source.values, copy=True)
        values[0, 0] += 10
        sources["nir"] = replace(source, values=values)
    elif mutation == "training_labels":
        targets[0] += 1000
    elif mutation == "schema":
        sources["nir"] = replace(sources["nir"], axis_units={"wavelength": "um"})
    changed = MultimodalDataset(
        sources, sample_ids=cohort.sample_ids, y=targets,
        groups=cohort.groups, partitions=cohort.partitions, name=cohort.name,
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("changed training contract reached estimator.fit")

    monkeypatch.setattr(MultimodalRegressor, "fit", forbidden)
    with pytest.raises(Exception, match="(?i)(checkpoint|fingerprint|mismatch|contract)"):
        _run(changed, _tuning(directory, resume=True), tmp_path / "invalid-resume", n_splits=2 if mutation == "folds" else 3)
    assert checkpoint_path.read_bytes() == before


def _run_stochastic(tuning: dict[str, Any], workspace: Path, *, random_state: int = 23) -> Any:
    # An unset estimator RNG must follow the public run seed on each native task.
    model = _model().set_params(model=RandomForestRegressor(n_estimators=9, max_depth=3, n_jobs=1))
    config = {**tuning, "space": {"model__max_depth": [2, 3, 4], "source_weights__image": [0.5, 1.0]}}
    return nirs4all.run(
        [GroupKFold(3), {"model": model}], _cohort(unequal_groups=True), tuning=config,
        engine="dag-ml", workspace_path=workspace, verbose=0, save_charts=False,
        random_state=random_state, refit=True,
    )


def test_stochastic_search_resume_matches_continuous_despite_progress_rng_use(tmp_path: Path) -> None:
    directory = tmp_path / "stochastic-resumed"

    def consume_rng_and_stop(event: dict[str, Any]) -> bool:
        np.random.normal(size=1000)
        return len(event["checkpoint"]["trials"]) < 2

    with pytest.raises(MultimodalTuningStopped):
        _run_stochastic(
            {**_tuning(directory), "progress_callback": consume_rng_and_stop}, tmp_path / "interrupted",
        )
    prefix = _checkpoint(directory)["native_checkpoint"]["trials"]
    assert len(prefix) == 2
    np.random.normal(size=777)
    resumed = _run_stochastic(_tuning(directory, resume=True), tmp_path / "resumed")
    continuous_directory = tmp_path / "stochastic-continuous"
    continuous = _run_stochastic(_tuning(continuous_directory), tmp_path / "continuous")
    try:
        assert resumed.tuning_result is not None and continuous.tuning_result is not None
        assert [trial.to_dict() for trial in resumed.tuning_result.trials] == [
            trial.to_dict() for trial in continuous.tuning_result.trials
        ]
        assert resumed.tuning_best_params == continuous.tuning_best_params
        assert resumed.tuning_best_value == continuous.tuning_best_value
        resumed_trials = _checkpoint(directory)["native_checkpoint"]["trials"]
        assert resumed_trials[:2] == prefix
        assert resumed_trials == _checkpoint(continuous_directory)["native_checkpoint"]["trials"]
        predictor = resumed._dagml_refit_artifacts[0]["estimator"]
        baseline = continuous._dagml_refit_artifacts[0]["estimator"]
        new_data = _cohort(prediction=True)
        np.testing.assert_array_equal(
            predictor.predict([new_data.sources[name].values for name in predictor.source_names]),
            baseline.predict([new_data.sources[name].values for name in baseline.source_names]),
        )
    finally:
        resumed.close()
        continuous.close()


def test_stochastic_resume_refuses_changed_public_run_seed_before_fit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    directory = tmp_path / "stochastic-seed"
    with pytest.raises(MultimodalTuningStopped):
        _run_stochastic({**_tuning(directory), "progress_callback": _stop_after(1, [])}, tmp_path / "initial")
    checkpoint_path = directory / "multimodal.n4mopt.json"
    before = checkpoint_path.read_bytes()

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("changed public run seed reached stochastic estimator.fit")

    monkeypatch.setattr(RandomForestRegressor, "fit", forbidden)
    with pytest.raises(Exception, match="(?i)(checkpoint|fingerprint|mismatch|contract)"):
        _run_stochastic(_tuning(directory, resume=True), tmp_path / "invalid-resume", random_state=24)
    assert checkpoint_path.read_bytes() == before


@pytest.mark.parametrize("direction", [None, "maximize", "minimize"])
def test_r2_search_infers_maximization_and_preserves_explicit_direction(direction: str | None, tmp_path: Path) -> None:
    config = {
        **_tuning(tmp_path / "r2-study"), "metric": " R2 ",
        "space": {"model__alpha": [0.01, 1000.0]},
    }
    if direction is None:
        config.pop("direction")
    else:
        config["direction"] = direction
    result = _run(_cohort(), config, tmp_path / "workspace")
    try:
        tuning = result.tuning_result
        assert tuning is not None
        expected_direction = direction or "maximize"
        assert tuning.tuning.metric == "r2"
        assert tuning.tuning.direction == expected_direction
        values = [trial.value for trial in tuning.trials if trial.value is not None]
        assert len(values) == 4
        assert min(values) < 0 and max(values) > 0.9
        assert tuning.best_value == (max(values) if expected_direction == "maximize" else min(values))
        assert tuning.best_params["model.alpha"] == (0.01 if expected_direction == "maximize" else 1000.0)
    finally:
        result.close()


def test_rmse_search_keeps_implicit_minimization(tmp_path: Path) -> None:
    config = {**_tuning(tmp_path / "rmse-study"), "space": {"model__alpha": [0.01, 1000.0]}}
    config.pop("direction")
    result = _run(_cohort(), config, tmp_path / "workspace")
    try:
        tuning = result.tuning_result
        assert tuning is not None and tuning.tuning.direction == "minimize"
        assert tuning.best_value == min(trial.value for trial in tuning.trials if trial.value is not None)
        assert tuning.best_params["model.alpha"] == 0.01
    finally:
        result.close()
