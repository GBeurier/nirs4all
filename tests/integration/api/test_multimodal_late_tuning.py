"""Durable tuning of raw-source branches and their native OOF meta-model."""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from nirs4all_io import MultimodalDataset
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import nirs4all
from nirs4all.api.result import RunResult
from nirs4all.operators.models.multimodal import TensorPCA
from nirs4all.pipeline.dagml.multimodal_tuning import MultimodalTuningStopped
from tests.integration.api.test_multimodal_late_fusion import _cohort, _pipeline
from tests.integration.api.test_multimodal_tuning import _checkpoint, _stop_after


def _tuning(directory: Path, *, resume: bool = False, n_trials: int = 4) -> dict[str, Any]:
    return {
        "engine": "n4m", "sampler": "random", "seed": 19, "metric": "rmse",
        "direction": "minimize", "n_trials": n_trials,
        "storage": directory.as_uri(), "study_name": "multimodal", "resume": resume,
        "space": {
            "branches.image.0.n_components": [1, 2],
            "branches.nir.1.alpha": [0.01, 10.0],
            "branches.metadata.0.numeric.with_mean": [False, True],
            "meta.alpha": [0.01, 10.0],
        },
    }


def _run(cohort: MultimodalDataset, tuning: dict[str, Any], workspace: Path, *, pipeline: list[Any] | None = None) -> Any:
    return nirs4all.run(
        _pipeline() if pipeline is None else pipeline, cohort, tuning=tuning,
        engine="dag-ml", refit=True, save_artifacts=True, save_charts=False,
        random_state=19, verbose=0, workspace_path=workspace,
    )


def _forbid_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Invalid configuration or replay reached fit")

    for cls in (Ridge, LogisticRegression, TensorPCA, ColumnTransformer, StandardScaler, OneHotEncoder):
        for method in ("fit", "fit_transform", "partial_fit"):
            if hasattr(cls, method):
                monkeypatch.setattr(cls, method, forbidden)


@pytest.fixture(autouse=True)
def native_scheduler_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *a, **k: pytest.fail("Legacy scheduler executed"))
    monkeypatch.setattr("nirs4all.pipeline.dagml.run_paths._run_model_on_precomputed_matrix", lambda *a, **k: pytest.fail("Python CV loop executed"))


def test_late_search_tunes_real_branches_and_returns_meta_ensemble(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cohort = _cohort()
    fits: list[tuple[int, int, frozenset[int]]] = []
    ridge_alphas: list[float] = []
    trial_fits: list[list[tuple[int, int, frozenset[int]]]] = []
    original_pca, original_ridge = TensorPCA.fit, Ridge.fit

    def pca_fit(self: TensorPCA, X: Any, y: Any = None) -> TensorPCA:
        values = np.asarray(X)
        assert self.n_components is not None
        fits.append((values.ndim, int(self.n_components), frozenset(int(v) for v in values.reshape(len(values), -1)[:, 0])))
        return original_pca(self, X, y)

    def ridge_fit(self: Ridge, X: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
        ridge_alphas.append(float(self.alpha))
        return original_ridge(self, X, y, *args, **kwargs)

    monkeypatch.setattr(TensorPCA, "fit", pca_fit)
    monkeypatch.setattr(Ridge, "fit", ridge_fit)

    def completed(event: dict[str, Any]) -> bool:
        if len(event["checkpoint"]["trials"]) > len(trial_fits):
            already_seen = sum(map(len, trial_fits))
            trial_fits.append(list(fits[already_seen:]))
        return True

    result = _run(cohort, {**_tuning(tmp_path / "study"), "progress_callback": completed}, tmp_path / "workspace")
    try:
        assert isinstance(result, RunResult)
        assert result.tuning_result is not None
        trials = result.tuning_result.trials
        assert [trial.number for trial in trials] == [0, 1, 2, 3]
        assert all(trial.state == "COMPLETE" and trial.diagnostics["test_used"] is False for trial in trials)
        assert all(trial.value is not None for trial in trials)
        assert result.tuning_best_value == min(trial.value for trial in trials if trial.value is not None)
        components = {trial.params["branches.image.0.n_components"] for trial in trials}
        assert components == {1, 2}
        assert {count for rank, count, _ in fits if rank == 4} == components
        assert len(trial_fits) == len(trials)
        for trial, calls in zip(trials, trial_fits, strict=True):
            assert {count for rank, count, _ in calls if rank == 4} == {trial.params["branches.image.0.n_components"]}
            assert any(len(rows) < 16 for _, _, rows in calls), "Each trial must rebuild its inner OOF encoders"
        assert {0.01, 10.0} <= set(ridge_alphas)
        assert any(len(rows) < 16 for _, _, rows in fits)
        assert all(rows <= set(range(24)) for _, _, rows in fits)
        for group in set(cohort.groups[:24]):
            members = {i for i, value in enumerate(cohort.groups[:24]) if value == group}
            assert all(not rows.intersection(members) or members <= rows for _, _, rows in fits)
        meta = [item for item in result._dagml_refit_artifacts if item["controller_id"] == "controller:nirs4all.meta_model"]
        assert len(meta) == 1
        assert meta[0]["estimator"].n_features_in_ == 4
        assert meta[0]["estimator"].alpha == result.tuning_best_params["meta.alpha"]
        assert result.execution_engine == "dag-ml"
        # The public result must export the selected ensemble directly.
        archive = result.export(tmp_path / "selected-ensemble.n4a")
        with zipfile.ZipFile(archive) as bundle:
            manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["multimodal_host"]["tuning"] == result.tuning_result.to_dict()
        new = _cohort(prediction=True)
        expected = nirs4all.predict(archive, new).y_pred
        result.close()
        shutil.rmtree(tmp_path / "workspace")
        _forbid_fit(monkeypatch)
        reordered = MultimodalDataset(dict(reversed(list(new.sources.items()))), sample_ids=new.sample_ids, partitions=new.partitions)
        replay = nirs4all.predict(archive, reordered)
        np.testing.assert_array_equal(replay.y_pred, expected)
        assert replay.y_pred.shape == (len(new),)
        assert replay.metadata["sample_ids"] == list(new.sample_ids)
        assert replay.metadata["training_performed"] is False
        assert replay.metadata["artifact_integrity_verified"] is True
    finally:
        result.close()


def test_late_search_resume_matches_continuous_and_ignores_test_targets(tmp_path: Path) -> None:
    cohort = _cohort()
    directory = tmp_path / "resumed"
    events: list[dict[str, Any]] = []
    with pytest.raises(MultimodalTuningStopped):
        _run(cohort, {**_tuning(directory), "progress_callback": _stop_after(2, events)}, tmp_path / "stopped")
    prefix = _checkpoint(directory)["native_checkpoint"]["trials"]
    assert len(prefix) == 2 and events[-1]["checkpoint"]["trials"] == prefix
    poisoned = _cohort(poison=list(range(24, 30)))
    resumed = _run(poisoned, _tuning(directory, resume=True), tmp_path / "resumed-workspace")
    continuous = _run(cohort, _tuning(tmp_path / "continuous"), tmp_path / "continuous-workspace")
    try:
        assert resumed.tuning_result is not None and continuous.tuning_result is not None
        assert [trial.to_dict() for trial in resumed.tuning_result.trials] == [trial.to_dict() for trial in continuous.tuning_result.trials]
        assert resumed.tuning_best_params == continuous.tuning_best_params
        assert resumed.tuning_best_value == continuous.tuning_best_value
        saved = _checkpoint(directory)["native_checkpoint"]["trials"]
        assert saved[:2] == prefix
        assert saved == _checkpoint(tmp_path / "continuous")["native_checkpoint"]["trials"]
        new = _cohort(prediction=True)
        a = nirs4all.predict(resumed.export(tmp_path / "resumed.n4a"), new)
        b = nirs4all.predict(continuous.export(tmp_path / "continuous.n4a"), new)
        np.testing.assert_array_equal(a.y_pred, b.y_pred)
    finally:
        resumed.close()
        continuous.close()


@pytest.mark.parametrize("path", ["branches.unknown.0.alpha", "branches.image.9.n_components", "branches.image.0.unknown", "meta.unknown", "branches.image.0.n_components.none"])
def test_invalid_late_parameter_path_fails_before_fit_or_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    directory = tmp_path / "study"
    _forbid_fit(monkeypatch)
    with pytest.raises(ValueError):
        _run(_cohort(), {**_tuning(directory), "space": {path: [1, 2]}}, tmp_path / "workspace")
    assert not (directory / "multimodal.n4mopt.json").exists()


def test_public_step_indices_include_none_and_nested_sklearn_aliases(tmp_path: Path) -> None:
    pipeline = _pipeline()
    pipeline[1]["branch"]["steps"]["image"].insert(0, None)
    pipeline[-1] = make_pipeline(StandardScaler(), Ridge())
    config = {**_tuning(tmp_path / "aliases", n_trials=1), "space": {
        "branches__image__1__n_components": [1],
        "branches.metadata.0.numeric__with_mean": [False],
        "meta__ridge__alpha": [0.7],
    }}
    result = _run(_cohort(), config, tmp_path / "workspace", pipeline=pipeline)
    try:
        assert result.tuning_best_params == {
            "branches.image.1.n_components": 1,
            "branches.metadata.0.numeric.with_mean": False,
            "meta.ridge.alpha": 0.7,
        }
    finally:
        result.close()


@pytest.mark.parametrize("mutation", ["schema", "folds", "training_labels"])
def test_late_resume_changed_contract_refused_before_fit_and_checkpoint_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    cohort = _cohort()
    directory = tmp_path / "study"
    with pytest.raises(MultimodalTuningStopped):
        _run(cohort, {**_tuning(directory), "progress_callback": _stop_after(1, [])}, tmp_path / "initial")
    path = directory / "multimodal.n4mopt.json"
    before = path.read_bytes()
    pipeline = _pipeline()
    sources = dict(cohort.sources)
    y = np.array(cohort.y, copy=True)
    if mutation == "schema":
        sources["nir"] = replace(sources["nir"], axis_units={"wavelength": "um"})
    elif mutation == "folds":
        pipeline[0] = GroupKFold(2)
    else:
        y[0] += 1_000
    changed = MultimodalDataset(sources, sample_ids=cohort.sample_ids, y=y, groups=cohort.groups, partitions=cohort.partitions, name=cohort.name)
    _forbid_fit(monkeypatch)
    with pytest.raises(Exception, match="(?i)(checkpoint|fingerprint|mismatch|contract)"):
        _run(changed, _tuning(directory, resume=True), tmp_path / "invalid", pipeline=pipeline)
    assert path.read_bytes() == before


def test_fixed_late_trial_outer_validation_labels_do_not_enter_its_fit(tmp_path: Path) -> None:
    cohort = _cohort()
    validation = next(GroupKFold(3).split(np.zeros((24, 1)), groups=cohort.groups[:24]))[1]
    space = {"branches.image.0.n_components": [1], "meta.alpha": [0.7]}
    before = _run(cohort, {**_tuning(tmp_path / "before", n_trials=1), "space": space}, tmp_path / "before-workspace")
    after = _run(_cohort(poison=validation.tolist()), {**_tuning(tmp_path / "after", n_trials=1), "space": space}, tmp_path / "after-workspace")
    try:
        left = before.predictions.filter_predictions(fold_id="0", partition="val", load_arrays=True)
        right = after.predictions.filter_predictions(fold_id="0", partition="val", load_arrays=True)
        assert len(left) == len(right) == 1
        assert left[0]["sample_indices"] == right[0]["sample_indices"]
        np.testing.assert_array_equal(left[0]["y_pred"], right[0]["y_pred"])
        # Validation labels affect HPO scores; this assertion holds a trial fixed.
    finally:
        before.close()
        after.close()


@pytest.mark.parametrize("task", ["multioutput", "classification"])
def test_late_search_complete_targets_support_task_and_direct_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: str) -> None:
    original = _cohort()
    pipeline = _pipeline()
    config = _tuning(tmp_path / "study", n_trials=1)
    if task == "multioutput":
        y = np.column_stack([original.y, -2 * original.y + 1])
        names, task_type = ["sugar", "protein"], "regression"
    else:
        y = np.where(np.arange(len(original)) % 2, "cultivar-a", "cultivar-b")
        names, task_type = ["cultivar"], "classification"
        for branch in pipeline[1]["branch"]["steps"].values():
            branch[-1] = LogisticRegression(max_iter=300)
        pipeline[-1] = LogisticRegression(max_iter=300)
        config.update(metric="accuracy", direction="maximize", space={"branches.image.0.n_components": [1], "meta.C": [0.5]})
    cohort = MultimodalDataset(original.sources, sample_ids=original.sample_ids, y=y, target_names=names, task_type=task_type, groups=original.groups, partitions=original.partitions)
    result = _run(cohort, config, tmp_path / "workspace", pipeline=pipeline)
    try:
        assert isinstance(result, RunResult) and result.tuning_result is not None
        archive = result.export(tmp_path / "ensemble.n4a")
        new = _cohort(prediction=True)
        prediction = MultimodalDataset(new.sources, sample_ids=new.sample_ids, partitions=new.partitions, target_names=names, task_type=task_type)
        result.close()
        shutil.rmtree(tmp_path / "workspace")
        _forbid_fit(monkeypatch)
        replay = nirs4all.predict(archive, prediction)
        assert replay.metadata["training_performed"] is False
        if task == "multioutput":
            assert replay.y_pred.shape == (len(new), 2)
            assert np.isfinite(replay.y_pred).all()
        else:
            assert set(np.asarray(replay.y_pred).reshape(-1)) <= set(y)
    finally:
        result.close()


def test_failed_late_encoder_trial_is_preserved_and_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "failed-study"
    # Real N4M random seed 6 proposes 99, 1, 1 for this one-dimensional space.
    config = {**_tuning(directory, n_trials=3), "seed": 6, "space": {"branches.image.0.n_components": [1, 99]}}
    seen: list[int] = []
    original_fit = TensorPCA.fit

    def record(self: TensorPCA, X: Any, y: Any = None) -> TensorPCA:
        if np.ndim(X) == 4:
            assert self.n_components is not None
            seen.append(int(self.n_components))
        return original_fit(self, X, y)

    monkeypatch.setattr(TensorPCA, "fit", record)
    with pytest.raises(Exception, match="n_components=99"):
        _run(_cohort(), config, tmp_path / "failed-workspace")
    prefix = _checkpoint(directory)["native_checkpoint"]["trials"]
    assert len(prefix) == 1 and prefix[0]["state"] == "failed"
    assert prefix[0].get("evidence", prefix[0])["params"]["branches.image.0.n_components"] == 99
    assert seen == [99]
    seen.clear()
    recovered = _run(_cohort(), {**config, "resume": True}, tmp_path / "recovered-workspace")
    try:
        assert recovered.tuning_result is not None
        assert [trial.state for trial in recovered.tuning_result.trials] == ["FAIL", "COMPLETE", "COMPLETE"]
        assert [trial.number for trial in recovered.tuning_result.trials] == [0, 1, 2]
        assert recovered.tuning_result.trials[0].value is None
        assert seen and set(seen) == {1}
        assert _checkpoint(directory)["native_checkpoint"]["trials"][:1] == prefix
        assert recovered.tuning_best_params["branches.image.0.n_components"] == 1
    finally:
        recovered.close()


def test_none_branch_step_is_not_a_tunable_operator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _pipeline()
    pipeline[1]["branch"]["steps"]["image"].insert(0, None)
    config = {**_tuning(tmp_path / "study"), "space": {"branches.image.0.n_components": [1]}}
    _forbid_fit(monkeypatch)
    with pytest.raises(ValueError, match="(?i)(path|operator|parameter)"):
        _run(_cohort(), config, tmp_path / "workspace", pipeline=pipeline)
    assert not (tmp_path / "study" / "multimodal.n4mopt.json").exists()


def test_completed_checkpoint_refits_random_estimators_identically_despite_callback_rng(tmp_path: Path) -> None:
    pipeline = _pipeline()
    pipeline[1]["branch"]["steps"]["nir"][-1] = RandomForestRegressor(n_estimators=7, max_depth=2, n_jobs=1)
    pipeline[-1] = RandomForestRegressor(n_estimators=7, max_depth=2, n_jobs=1)
    directory = tmp_path / "completed-study"
    config = {**_tuning(directory, n_trials=2), "space": {
        "branches.image.0.n_components": [1, 2], "branches.nir.1.max_depth": [2, 3], "meta.max_depth": [2, 3],
    }}

    def consume_rng(event: dict[str, Any]) -> bool:
        np.random.seed(703 + len(event["checkpoint"]["trials"]))
        np.random.normal(size=257)
        return True

    initial = _run(_cohort(), {**config, "progress_callback": consume_rng}, tmp_path / "initial", pipeline=pipeline)
    prefix = _checkpoint(directory)["native_checkpoint"]["trials"]
    resumed = _run(_cohort(), {**config, "resume": True}, tmp_path / "resumed", pipeline=pipeline)
    try:
        assert initial.tuning_result is not None and resumed.tuning_result is not None
        assert [trial.to_dict() for trial in initial.tuning_result.trials] == [trial.to_dict() for trial in resumed.tuning_result.trials]
        assert _checkpoint(directory)["native_checkpoint"]["trials"] == prefix
        assert initial.tuning_best_params == resumed.tuning_best_params
        new = _cohort(prediction=True)
        before = nirs4all.predict(initial.export(tmp_path / "initial.n4a"), new)
        after = nirs4all.predict(resumed.export(tmp_path / "resumed.n4a"), new)
        np.testing.assert_array_equal(before.y_pred, after.y_pred)
    finally:
        initial.close()
        resumed.close()


def test_resampled_late_search_resume_keeps_partitioned_refit_oof_and_direct_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cohort = _cohort()
    pipeline = _pipeline()
    pipeline[0] = GroupShuffleSplit(n_splits=3, train_size=0.6, random_state=19)
    directory = tmp_path / "resampled-study"
    config = _tuning(directory, n_trials=2)
    with pytest.raises(MultimodalTuningStopped):
        _run(cohort, {**config, "progress_callback": _stop_after(1, [])}, tmp_path / "stopped", pipeline=pipeline)
    prefix = _checkpoint(directory)["native_checkpoint"]["trials"]
    assert len(prefix) == 1
    resumed = _run(cohort, {**config, "resume": True}, tmp_path / "resumed", pipeline=pipeline)
    continuous = _run(cohort, _tuning(tmp_path / "continuous-study", n_trials=2), tmp_path / "continuous", pipeline=pipeline)
    try:
        assert resumed.tuning_result is not None and continuous.tuning_result is not None
        assert [trial.to_dict() for trial in resumed.tuning_result.trials] == [trial.to_dict() for trial in continuous.tuning_result.trials]
        assert resumed.tuning_best_params == continuous.tuning_best_params
        assert _checkpoint(directory)["native_checkpoint"]["trials"][:1] == prefix
        outer = list(pipeline[0].split(np.zeros((24, 1)), groups=cohort.groups[:24]))
        evidence = next(iter(resumed.per_dataset.values()))["stacking_evaluation"]
        assert evidence == next(iter(continuous.per_dataset.values()))["stacking_evaluation"]
        assert evidence["outer_partition_mode"] == "resampled"
        assert evidence["outer_validation_occurrences"] == sum(len(validation) for _, validation in outer)
        assert evidence["outer_validation_sample_count"] == len({int(row) for _, validation in outer for row in validation})
        assert evidence["training_sample_count"] == 24
        assert evidence["refit_oof"] == "partitioned_inner_v1"
        assert evidence["refit_oof_is_selection_evidence"] is False
        # These are native preparation outputs, not the overlapping outer OOFs.
        by_producer: dict[str, list[str]] = {}
        for frame in resumed._dagml_node_results:
            for block in frame.get("predictions", []):
                if str(block.get("fold_id", "")).startswith("stacking.refit.inner.") and block["partition"] == "validation":
                    by_producer.setdefault(block["producer_node"], []).extend(block["sample_ids"])
        assert len(by_producer) == 4
        assert all(len(ids) == len(set(ids)) == 24 for ids in by_producer.values())
        assert resumed._dagml_score_set is not None
        assert not any(
            str(report.get("fold_id", "")).startswith("stacking.refit.inner.")
            for report in resumed._dagml_score_set["reports"] if report["producer_node"] == "merge:stack"
        )
        _forbid_fit(monkeypatch)
        archive = resumed.export(tmp_path / "resampled.n4a")
        reference = continuous.export(tmp_path / "continuous.n4a")
        with zipfile.ZipFile(archive) as bundle:
            manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["stacking_evaluation"] == evidence
        resumed.close()
        continuous.close()
        shutil.rmtree(tmp_path / "resumed")
        shutil.rmtree(tmp_path / "continuous")
        new = _cohort(prediction=True)
        replay, expected = nirs4all.predict(archive, new), nirs4all.predict(reference, new)
        np.testing.assert_array_equal(replay.y_pred, expected.y_pred)
        assert replay.y_pred.shape == (len(new),)
        assert replay.metadata["training_performed"] is False
        assert replay.metadata["sample_ids"] == list(new.sample_ids)
    finally:
        resumed.close()
        continuous.close()
