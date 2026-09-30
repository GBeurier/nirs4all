"""Global native HPO preserves fixed training controls through selection and replay."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from nirs4all_io import DataProvider, TensorSource
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold

import nirs4all
from nirs4all.operators.models.multimodal import TensorPCA
from nirs4all.pipeline.dagml.multimodal_tuning import MultimodalTuningStopped
from tests.integration.api.test_multimodal_dagml import _cohort, _model
from tests.integration.api.test_multimodal_late_fusion import _cohort as _late_cohort
from tests.integration.api.test_multimodal_late_fusion import _pipeline as _late_pipeline
from tests.integration.api.test_multimodal_targets import _classification_cohort, _classifier
from tests.integration.api.test_multimodal_tuning import _stop_after


def _recipe(late: bool) -> tuple[list[Any], Any, dict[str, Any]]:
    if not late:
        return [GroupKFold(3), {
            "model": _model(), "name": "controlled-multimodal",
            "train_params": {"model__tol": 0.003}, "refit_params": {"model__alpha": 9.0},
        }], _cohort(), {"model__alpha": [0.1, 1.0]}
    pipeline = _late_pipeline()
    for index, branch in enumerate(pipeline[1]["branch"]["steps"].values()):
        branch[-1] = {
            "model": branch[-1], "train_params": {"alpha": float(index + 2), "tol": 0.003},
            "refit_params": {"alpha": float(index + 12)},
        }
    pipeline[-1] = {
        "model": pipeline[-1], "name": "controlled-meta",
        "train_params": {"alpha": 17.0}, "refit_params": {"alpha": 27.0},
    }
    return pipeline, _late_cohort(), {"branches.image.0.n_components": [1, 2]}


def _run(pipeline: list[Any], cohort: Any, space: dict[str, Any], workspace: Path, **tuning: Any) -> Any:
    return nirs4all.run(
        pipeline, cohort, tuning={"engine": "n4m", "sampler": "random", "seed": 19,
                                 "n_trials": 2, "space": space, **tuning},
        engine="dag-ml", workspace_path=workspace, save_charts=False,
        verbose=0, random_state=19, refit=True, save_artifacts=True,
    )


@pytest.fixture(autouse=True)
def native_scheduler_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *a, **k: pytest.fail("legacy scheduler executed"))
    monkeypatch.setattr("nirs4all.pipeline.dagml.run_paths._run_model_on_precomputed_matrix", lambda *a, **k: pytest.fail("Python CV loop executed"))


@pytest.mark.parametrize("fixed_alpha", [None, 3.0])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_direct_global_search_controls_and_refit_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixed_alpha: float | None, n_jobs: int) -> None:
    pipeline, cohort, space = _recipe(False)
    if fixed_alpha is not None:
        pipeline[-1]["train_params"]["model__alpha"] = fixed_alpha
    original_controls = copy.deepcopy(pipeline[-1]["train_params"])
    fits: list[tuple[int, float, float]] = []
    trial_fits: list[list[tuple[int, float, float]]] = []
    original = Ridge.fit

    def record(self: Ridge, X: Any, y: Any, **kwargs: Any) -> Any:
        fits.append((len(X), self.alpha, self.tol))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(Ridge, "fit", record)
    result = _run(pipeline, cohort, space, tmp_path / "workspace", n_jobs=n_jobs,
                  progress_callback=lambda event: trial_fits.append(list(fits)))
    try:
        assert [trial.state for trial in result.tuning_result.trials] == ["COMPLETE"] * 2
        expected_alpha = fixed_alpha if fixed_alpha is not None else result.tuning_best_params["model.alpha"]
        assert fits[-4:] == [(8, expected_alpha, 0.003)] * 3 + [(12, 9.0, 0.003)]
        if n_jobs == 1:
            assert len(trial_fits[-1]) == 6
            assert {rows for rows, _, _ in trial_fits[-1]} == {8}
            assert {tol for _, _, tol in trial_fits[-1]} == {0.003}
            assert all(alpha == fixed_alpha if fixed_alpha is not None else alpha in {0.1, 1.0}
                       for _, alpha, _ in trial_fits[-1])
        assert pipeline[-1]["train_params"] == original_controls
        assert pipeline[-1]["refit_params"] == {"model__alpha": 9.0}
        assert pipeline[-1]["model"].model.alpha == 0.2
        fitted = result._dagml_refit_artifacts[0]["estimator"]
        assert fitted._nirs4all_training_controls["phase"] == "REFIT"
        prediction = _cohort(prediction=True)
        expected = fitted.predict([prediction.sources[name].values for name in fitted.source_names])
        archive = result.export(tmp_path / "controlled.n4a")
    finally:
        result.close()
    shutil.rmtree(tmp_path / "workspace")
    monkeypatch.setattr(Ridge, "fit", lambda *a, **k: pytest.fail("archive replay reached fit"))
    monkeypatch.setattr(TensorPCA, "fit", lambda *a, **k: pytest.fail("archive replay reached encoder fit"))
    np.testing.assert_array_equal(nirs4all.predict(archive, prediction, engine="dag-ml").y_pred.ravel(), expected.ravel())


def test_late_global_search_controls_each_branch_and_meta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, cohort, space = _recipe(True)
    original_steps = copy.deepcopy([branch[-1]["train_params"] for branch in pipeline[1]["branch"]["steps"].values()])
    fits: list[tuple[int, float]] = []
    trial_fits: list[list[tuple[int, float]]] = []
    original = Ridge.fit

    def record(self: Ridge, X: Any, y: Any, **kwargs: Any) -> Any:
        fits.append((len(X), self.alpha))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(Ridge, "fit", record)
    result = _run(pipeline, cohort, space, tmp_path / "workspace",
                  progress_callback=lambda event: trial_fits.append(list(fits)))
    try:
        assert [trial.state for trial in result.tuning_result.trials] == ["COMPLETE"] * 2
        assert {alpha for _, alpha in trial_fits[-1]} == {2, 3, 4, 5, 17}
        assert all(rows < 24 for rows, _ in trial_fits[-1])
        assert sorted(alpha for rows, alpha in fits if rows == 24) == [12, 13, 14, 15, 27]
        assert [branch[-1]["train_params"] for branch in pipeline[1]["branch"]["steps"].values()] == original_steps
        meta = next(item["estimator"] for item in result._dagml_refit_artifacts if item["controller_id"] == "controller:nirs4all.meta_model")
        assert meta.alpha == 27
        assert meta._nirs4all_training_controls["phase"] == "REFIT"
        prediction = _late_cohort(prediction=True)
        archive = result.export(tmp_path / "late-controlled.n4a")
        expected = nirs4all.predict(archive, prediction, engine="dag-ml").y_pred
    finally:
        result.close()
    shutil.rmtree(tmp_path / "workspace")
    monkeypatch.setattr(Ridge, "fit", lambda *a, **k: pytest.fail("archive replay reached fit"))
    monkeypatch.setattr(TensorPCA, "fit", lambda *a, **k: pytest.fail("archive replay reached encoder fit"))
    np.testing.assert_array_equal(nirs4all.predict(archive, prediction, engine="dag-ml").y_pred, expected)


@pytest.mark.parametrize("late", [False, True])
def test_resume_same_controls_matches_continuous_search(tmp_path: Path, late: bool) -> None:
    pipeline, cohort, space = _recipe(late)
    options = {"storage": (tmp_path / "study").as_uri(), "study_name": "controls"}
    with pytest.raises(MultimodalTuningStopped):
        _run(pipeline, cohort, space, tmp_path / "initial", **options, progress_callback=_stop_after(1, []))
    resumed = _run(pipeline, cohort, space, tmp_path / "resumed", **options, resume=True)
    continuous = _run(pipeline, cohort, space, tmp_path / "continuous")
    try:
        assert [trial.to_dict() for trial in resumed.tuning_result.trials] == [trial.to_dict() for trial in continuous.tuning_result.trials]
        assert resumed.tuning_best_params == continuous.tuning_best_params
        assert resumed.tuning_best_value == continuous.tuning_best_value
        prediction = _late_cohort(prediction=True) if late else _cohort(prediction=True)
        left = resumed.export(tmp_path / "resumed.n4a")
        right = continuous.export(tmp_path / "continuous.n4a")
        np.testing.assert_array_equal(nirs4all.predict(left, prediction, engine="dag-ml").y_pred,
                                      nirs4all.predict(right, prediction, engine="dag-ml").y_pred)
    finally:
        resumed.close()
        continuous.close()


def test_classification_global_search_applies_controls_in_native_scopes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = [GroupKFold(3), {"model": _classifier(), "train_params": {"model__tol": 0.002}, "refit_params": {"model__C": 0.7}}]
    fits: list[tuple[int, float, float]] = []
    original = LogisticRegression.fit

    def record(self: LogisticRegression, X: Any, y: Any, **kwargs: Any) -> Any:
        fits.append((len(X), self.C, self.tol))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(LogisticRegression, "fit", record)
    result = _run(pipeline, _classification_cohort(), {"model__C": [0.1, 1.0]}, tmp_path / "workspace")
    try:
        assert result.tuning_result.tuning.metric == "balanced_accuracy"
        assert [trial.state for trial in result.tuning_result.trials] == ["COMPLETE"] * 2
        assert fits[-1] == (12, 0.7, 0.002)
        assert all(rows < 12 and C in {0.1, 1.0} and tol == 0.002 for rows, C, tol in fits[:-1])
        assert pipeline[-1]["model"].model.C == 0.3
    finally:
        result.close()


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_generated_view_global_search_controls_survive_worker_and_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_jobs: int) -> None:
    pipeline, base, space = _recipe(False)

    def generate(**_: Any) -> dict[str, Any]:
        return {"sample_ids": list(base.sample_ids), "sources": {"nir": base.sources["nir"]}}

    def generate_view(*, sample_ids: list[str], seed: int, **_: Any) -> dict[str, Any]:
        source = base.take(sample_ids).sources["nir"]
        return {"sample_ids": sample_ids, "sources": {"nir": TensorSource(
            np.asarray(source.values) + float(seed % 7), sample_ids,
            representation_id=source.representation_id, axis_units=source.axis_units,
            axis_coordinates=source.axis_coordinates,
        )}}

    provider = DataProvider(generate, generate_view=generate_view,
                            provider_id="qualification.global-training-controls", base=base, replace_sources=["nir"])
    result = nirs4all.run(
        pipeline, provider, tuning={"engine": "n4m", "sampler": "random", "seed": 19,
                                   "n_trials": 2, "space": space, "n_jobs": n_jobs},
        engine="dag-ml", results_path=tmp_path / "native", save_charts=False,
        verbose=0, random_state=19, refit=True, save_artifacts=False,
    )
    try:
        assert [trial.state for trial in result.tuning_result.trials] == ["COMPLETE"] * 2
        fitted = result._dagml_refit_artifacts[0]["estimator"]
        assert fitted._nirs4all_training_controls["phase"] == "REFIT"
        assert fitted._nirs4all_training_controls["model_params"] == {"model__alpha": 9.0, "model__tol": 0.003}
        prediction = _cohort(prediction=True)
        expected = fitted.predict([prediction.sources[name].values for name in fitted.source_names])
        archive = result.export(tmp_path / "generated-controlled.n4a")
    finally:
        result.close()
    shutil.rmtree(tmp_path / "native")
    monkeypatch.setattr(DataProvider, "materialize", lambda *a, **k: pytest.fail("archive replay called a provider"))
    monkeypatch.setattr(DataProvider, "materialize_view", lambda *a, **k: pytest.fail("archive replay regenerated a view"))
    monkeypatch.setattr(Ridge, "fit", lambda *a, **k: pytest.fail("archive replay reached fit"))
    np.testing.assert_array_equal(nirs4all.predict(archive, prediction, engine="dag-ml").y_pred.ravel(), expected.ravel())


@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize("control", ["train_params", "refit_params"])
def test_resume_changed_training_controls_refused_before_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, late: bool, control: str) -> None:
    pipeline, cohort, space = _recipe(late)
    directory = tmp_path / "study"
    options = {"storage": directory.as_uri(), "study_name": "controls"}
    with pytest.raises(MultimodalTuningStopped):
        _run(pipeline, cohort, space, tmp_path / "initial", **options, progress_callback=_stop_after(1, []))
    before = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    changed = copy.deepcopy(pipeline)
    changed[-1][control]["alpha" if late else "model__alpha"] = 31.0
    monkeypatch.setattr(Ridge, "fit", lambda *a, **k: pytest.fail("changed checkpoint reached fit"))
    monkeypatch.setattr(TensorPCA, "fit", lambda *a, **k: pytest.fail("changed checkpoint reached encoder fit"))
    with pytest.raises(Exception, match="(?i)(checkpoint|fingerprint|mismatch|contract)"):
        _run(changed, cohort, space, tmp_path / "invalid", **options, resume=True)
    assert {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()} == before


@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize("declaration, message", [
    ({"train_params": {"nonexistent_parameter": 1}}, "unrecognized training parameters"),
    ({"refit_params": {"warm_start": True}}, "warm.start"),
    ({"train_params": []}, "mapping"),
    ({"finetune_params": {"n_trials": 2}}, "tuning.space"),
])
def test_invalid_global_training_controls_refuse_before_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, late: bool, declaration: dict[str, Any], message: str) -> None:
    pipeline, cohort, space = _recipe(late)
    pipeline[-1].update(declaration)
    monkeypatch.setattr(Ridge, "fit", lambda *a, **k: pytest.fail("invalid declaration reached fit"))
    monkeypatch.setattr(TensorPCA, "fit", lambda *a, **k: pytest.fail("invalid declaration reached encoder fit"))
    monkeypatch.setattr("nirs4all.pipeline.dagml.multimodal_tuning.HostSearchOptimizer.__init__",
                        lambda *a, **k: pytest.fail("invalid declaration opened a study"))
    with pytest.raises((ValueError, TypeError, NotImplementedError), match=message):
        _run(pipeline, cohort, space, tmp_path / "invalid", storage=(tmp_path / "study").as_uri())
    assert not (tmp_path / "study").exists()
