"""Real source coordinates survive fresh native inner HPO preprocessing."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

import dag_ml
import numpy as np
import optuna
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.data import SpectroDataset
from nirs4all.operators.transforms import Resampler
from nirs4all.pipeline.dagml import host_finetune


def _spectral_data(*, unit: str = "cm-1", dense: bool = False, shift: float = 0.0) -> tuple[SpectroDataset, list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(918)
    widths = [12, 9] if dense else [12]
    axes = [np.linspace(1000.125, 1200.875, widths[0]) + shift]
    if dense:
        axes.append(np.linspace(800.375, 1600.625, widths[1]) + shift)
    values = [rng.normal(size=(24, width)) for width in widths]
    for index, block in enumerate(values):
        block[:, 0] = np.arange(len(block)) + index * 100
    y = 2.0 * values[0][:, 1] - values[0][:, 3]
    if dense:
        y = y + 0.7 * values[1][:, 2]
    headers = [[str(value) for value in (axis if unit == "cm-1" else 10_000_000 / axis)] for axis in axes]
    dataset = SpectroDataset("native_inner_source_axes")
    for rows, partition in ((slice(0, 18), "train"), (slice(18, 24), "test")):
        samples = [block[rows] for block in values]
        dataset.add_samples(samples if dense else samples[0], {"partition": partition},
                            headers=headers if dense else headers[0], header_unit=[unit] * len(widths) if dense else unit)
        dataset.add_targets(y[rows])
    return dataset, values, axes


def _pipeline(engine: str, *, dense: bool = False, durable: dict[str, Any] | None = None) -> list[Any]:
    local = {"engine": engine, "sampler": "random", "seed": 13, "n_trials": 2,
             "approach": "grouped", "eval_mode": "mean", "model_params": {"alpha": [0.1, 1.0]},
             **(durable or {})}
    model = {"model": Ridge(), "finetune_params": local, "train_params": {"tol": 0.003}, "refit_params": {"alpha": 0.7}}
    stages = [Resampler(target_wavelengths=np.linspace(1010.25, 1190.75, 7)), StandardScaler(), model]
    if dense:
        return [KFold(3), {"branch": {"by_source": True, "steps": stages}},
                {"merge": "predictions"}, {"model": Ridge(alpha=0.2)}]
    return [*stages[:-1], KFold(3), model]


def _run(pipeline: list[Any], dataset: SpectroDataset, workspace: Path) -> Any:
    return nirs4all.run(pipeline, dataset, engine="dag-ml", workspace_path=workspace,
                        save_charts=False, verbose=0, refit=True, save_artifacts=True)


@pytest.fixture(autouse=True)
def native_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *a, **k: pytest.fail("legacy scheduler executed"))
    monkeypatch.setattr("nirs4all.pipeline.dagml.run_paths._run_model_on_precomputed_matrix", lambda *a, **k: pytest.fail("Python CV loop executed"))


def _capture_inner(monkeypatch: pytest.MonkeyPatch, original: SpectroDataset, raw: list[np.ndarray], axes: list[np.ndarray],
                   unit: str, *, dense: bool) -> list[dict[str, Any]]:
    captures: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    search = host_finetune.run_scoped_finetune
    native = dag_ml.run_host_hpo_search_in_process
    fit = Resampler.fit

    def record_search(*args: Any, **kwargs: Any) -> Any:
        scoped = kwargs["scoped_dataset"]
        assert isinstance(scoped, SpectroDataset)
        assert scoped.features_sources() == 1
        assert kwargs["source_index"] == (0 if dense else None)
        scope = copy.deepcopy(kwargs["scope"])
        source = int(scope["source_names"][0].rsplit("_", 1)[1])
        assert source in range(len(raw))
        assert scoped.headers(0) == original.headers(source)
        assert scoped.header_unit(0) == unit
        assert len(scope["recipe_fingerprint"]) == len(scope["data_content_fingerprint"]) == 64
        values = np.asarray(scoped.x({"partition": "train"}, layout="2d"))
        rows = tuple(int(value) for value in values[:, 0])
        assert set(rows) <= set(range(source * 100, source * 100 + 18))
        expected = {frozenset(rows[row] for row in train) for train, _ in kwargs["inner_cv"]["folds"]}
        capture = {"scope": scope, "source": source, "rows": rows, "expected": expected,
                   "fits": [], "instances": [], "native": []}
        captures.append(capture)
        active.append(capture)
        try:
            answer = search(*args, **kwargs)
            capture["evidence"] = answer[1]
            return answer
        finally:
            active.pop()

    def record_native(*args: Any, **kwargs: Any) -> Any:
        if active:
            dsl, envelope = args[:2]
            capture = active[-1]
            assert len(envelope["data_content_fingerprint"]) == 64
            np.testing.assert_allclose(np.asarray(envelope["_host_feature_axes"]["src0"], dtype=float),
                                       axes[capture["source"]], rtol=1e-12, atol=1e-10)
            declared = {step["source_id"] for step in envelope["plan"]["steps"] if step["kind"] == "materialize"}
            assert declared == {"src0"}
            assert dsl["data_bindings"]
            for binding in dsl["data_bindings"]:
                assert binding["source_ids"] == ["src0"]
            capture["native"].append(copy.deepcopy(envelope))
        return native(*args, **kwargs)

    def record_fit(self: Resampler, X: Any, y: Any = None, wavelengths: Any = None) -> Any:
        if active:
            capture = active[-1]
            np.testing.assert_allclose(wavelengths, axes[capture["source"]], rtol=1e-12, atol=1e-10)
            rows = frozenset(int(value) for value in np.asarray(X)[:, 0])
            assert rows in capture["expected"], "inner Resampler received validation/test rows or outer-fitted features"
            assert len(rows) < len(capture["rows"])
            capture["fits"].append(rows)
            capture["instances"].append(self)
        return fit(self, X, y, wavelengths=wavelengths)

    monkeypatch.setattr(host_finetune, "run_scoped_finetune", record_search)
    monkeypatch.setattr(dag_ml, "run_host_hpo_search_in_process", record_native)
    monkeypatch.setattr(Resampler, "fit", record_fit)
    return captures


def _check_and_replay(result: Any, captures: list[dict[str, Any]], dataset: SpectroDataset, tmp_path: Path,
                      monkeypatch: pytest.MonkeyPatch, *, dense: bool) -> None:
    assert captures
    assert {capture["source"] for capture in captures} == ({0, 1} if dense else {0})
    assert {capture["scope"]["phase"] for capture in captures} >= {"FIT_CV", "REFIT"}
    for capture in captures:
        assert set(capture["fits"]) == capture["expected"]
        assert len({id(instance) for instance in capture["instances"]}) == len(capture["instances"])
        assert len(capture["native"]) == 1
        assert capture["evidence"]["evaluation"]["outer_validation_used"] is False
        assert capture["evidence"]["evaluation"]["test_used"] is False
        assert len(capture["evidence"]["trials"]) == 2
        if capture["scope"]["phase"] == "REFIT":
            assert capture["evidence"]["effective_selected_model_params"]["alpha"] == 0.7
    exported = result.runs[-1] if dense else result
    archive = exported.export(tmp_path / "source-axes.n4a")
    expected = nirs4all.predict(archive, dataset, engine="dag-ml").y_pred
    result.close()
    shutil.rmtree(tmp_path / "workspace")
    monkeypatch.setattr(Resampler, "fit", lambda *a, **k: pytest.fail("archive prediction fit a Resampler"))
    monkeypatch.setattr(Ridge, "fit", lambda *a, **k: pytest.fail("archive prediction fit a model"))
    monkeypatch.setattr(host_finetune, "run_scoped_finetune", lambda *a, **k: pytest.fail("archive prediction opened local HPO"))
    np.testing.assert_array_equal(nirs4all.predict(archive, dataset, engine="dag-ml").y_pred, expected)


@pytest.mark.parametrize("engine", ["optuna", "n4m"])
@pytest.mark.parametrize("unit", ["cm-1", "nm"])
def test_scalar_local_hpo_fresh_inner_resampler_keeps_real_spectral_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine: str, unit: str,
) -> None:
    dataset, raw, axes = _spectral_data(unit=unit)
    captures = _capture_inner(monkeypatch, dataset, raw, axes, unit, dense=False)
    result = _run(_pipeline(engine), dataset, tmp_path / "workspace")
    try:
        _check_and_replay(result, captures, dataset, tmp_path, monkeypatch, dense=False)
    finally:
        result.close()


@pytest.mark.parametrize("engine", ["optuna", "n4m"])
def test_dense_by_source_local_hpo_keeps_source_axes_and_native_local_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine: str,
) -> None:
    dataset, raw, axes = _spectral_data(dense=True)
    captures = _capture_inner(monkeypatch, dataset, raw, axes, "cm-1", dense=True)
    result = _run(_pipeline(engine, dense=True), dataset, tmp_path / "workspace")
    try:
        _check_and_replay(result, captures, dataset, tmp_path, monkeypatch, dense=True)
    finally:
        result.close()


def test_scalar_local_resume_changed_coordinates_refuses_before_fit_in_same_studies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = f"sqlite:///{tmp_path / 'spectral.sqlite3'}"
    durable = {"n_trials": 1, "storage": storage, "study_name": "spectral"}
    scopes: list[dict[str, Any]] = []
    active: list[bool] = []
    search = host_finetune.run_scoped_finetune
    fit = Resampler.fit

    def record_search(*args: Any, **kwargs: Any) -> Any:
        scopes.append(copy.deepcopy(kwargs["scope"]))
        active.append(True)
        try:
            return search(*args, **kwargs)
        finally:
            active.pop()

    def forbid_inner_fit(self: Resampler, *args: Any, **kwargs: Any) -> Any:
        # Outer preprocessing precedes the model's local search; only a fresh
        # inner candidate fit would violate native resume preflight here.
        assert not active, "incompatible axis resume fit inner preprocessing"
        return fit(self, *args, **kwargs)

    monkeypatch.setattr(host_finetune, "run_scoped_finetune", record_search)
    dataset, _, _ = _spectral_data()
    first = _run(_pipeline("optuna", durable=durable), dataset, tmp_path / "first")
    first.close()
    names = sorted(summary.study_name for summary in optuna.get_all_study_summaries(storage=storage))
    before = {name: copy.deepcopy(optuna.load_study(study_name=name, storage=storage).user_attrs[
        "nirs4all_dagml_host_hpo_checkpoint_v1"]) for name in names}
    original_scopes = copy.deepcopy(scopes)
    scopes.clear()
    changed, _, _ = _spectral_data(shift=3.0)
    monkeypatch.setattr(Resampler, "fit", forbid_inner_fit)
    monkeypatch.setattr(Ridge, "fit", lambda *a, **k: pytest.fail("incompatible axis resume fit a model"))
    with pytest.raises(Exception, match="host HPO checkpoint objective/graph/controller/data/fold binding mismatch"):
        _run(_pipeline("optuna", durable={**durable, "resume": True, "n_trials": 2}), changed, tmp_path / "rejected")
    assert scopes
    assert scopes[0]["training_sample_ids"] == original_scopes[0]["training_sample_ids"]
    assert scopes[0]["data_content_fingerprint"] != original_scopes[0]["data_content_fingerprint"]
    assert sorted(summary.study_name for summary in optuna.get_all_study_summaries(storage=storage)) == names
    for name in names:
        study = optuna.load_study(study_name=name, storage=storage)
        assert len(study.trials) == 1
        assert study.user_attrs["nirs4all_dagml_host_hpo_checkpoint_v1"] == before[name]
