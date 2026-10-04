"""Public named-input Torch runs: genuine native scopes, independent FIT oracle."""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import textwrap
import zipfile
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path
from typing import Any

import dag_ml
import joblib
import numpy as np
import pytest
import torch
from nirs4all_io import MultimodalDataset, TensorSource
from sklearn.model_selection import GroupKFold

import nirs4all
from nirs4all.operators.models.multimodal import MultimodalRegressor
from nirs4all.pipeline.dagml import full_train, in_process_runner
from nirs4all.pipeline.dagml.general_archive import load_general_archive
from nirs4all.pipeline.dagml.named_torch_estimator import DagMLNamedTorchEstimator
from nirs4all.pipeline.dagml.native_results import _rehydrate_artifacts, _write_model_artifacts, read_native_results
from nirs4all.pipeline.dagml.torch_estimator import DagMLTorchEstimator
from tests.fixtures import named_torch_training as oracle

pytestmark = pytest.mark.torch
_ACTIVE: ContextVar[dict[str, Any] | None] = ContextVar("named_torch_native_fit", default=None)


def _cohort(count: int = 2, *, prediction: bool = False, target_name: str = "y") -> MultimodalDataset:
    rng = np.random.default_rng(791 if prediction else 781)
    rows = 5 if prediction else 16
    ids = [f"{'predict' if prediction else 'sample'}_{i:02d}" for i in range(rows)]
    sources = {}
    targets = np.zeros(rows, dtype=np.float32)
    for index, name in enumerate(("nir", "image", "series", "metadata")[:count]):
        values = rng.normal(size=(rows, index + 2)).astype(np.float32)
        targets += (index + 1) * values[:, 0] + 0.3 * values[:, 1]
        order = rng.permutation(rows)
        sources[name] = TensorSource(values[order], [ids[i] for i in order], representation_id="tabular_numeric",
                                     feature_names=[f"feature_{j}" for j in range(values.shape[1])])
    return MultimodalDataset(sources, sample_ids=ids, y=None if prediction else targets,
        groups=None if prediction else [f"group_{i // 2}" for i in range(rows)],
        partitions=["predict"] * rows if prediction else ["train"] * 12 + ["test"] * 4,
        target_names=[target_name], name="named_prediction" if prediction else "named_training")


def _model(cohort: Any, *, template: bool = False) -> MultimodalRegressor:
    if template:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(21)
            model: Any = oracle.joint_factory(input_shapes={name: (source.values.shape[1],) for name, source in cohort.sources.items()})
    else:
        model = DagMLTorchEstimator(factory_path="tests.fixtures.named_torch_training.joint_factory",
            factory_params={"hidden_units": 3}, device="cpu", force_layout="2d", task_type="regression",
            epochs=2, batch_size=4, patience=2, lr=0.01)
    return MultimodalRegressor(dict.fromkeys(cohort.sources, "passthrough"), model=model,
                              fusion="intermediate", backend="sklearn")


def _run(cohort: Any, root: Path, *, template: bool = False, cv: bool = True, random_state: int = 21, save_artifacts: bool = True) -> Any:
    step = {"model": _model(cohort, template=template)}
    if template:
        step["train_params"] = {"epochs": 2, "batch_size": 4, "patience": 2, "lr": 0.01,
                                "device": "cpu", "force_layout": "2d", "task_type": "regression"}
    pipeline = [GroupKFold(3), step] if cv else [step]
    return nirs4all.run(pipeline, cohort, engine="dag-ml", refit=True, random_state=random_state, cpu_threads=1,
        gpu_devices=[], save_artifacts=save_artifacts, save_charts=False, workspace_path=root, verbose=0)


def _ids(task: dict[str, Any], partition: str) -> list[str]:
    views = [view for view in task["data_views"].values() if view["partition"] == partition]
    assert views and len(views) >= 2
    assert all(view["sample_ids"] == views[0]["sample_ids"] for view in views)
    return list(views[0]["sample_ids"])


class _Observer:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, cohort: Any) -> None:
        self.cohort = cohort
        self.rows = {sample: index for index, sample in enumerate(cohort.sample_ids)}
        self.records: list[dict[str, Any]] = []
        self.modules: list[Any] = []
        self.materializations: list[dict[str, Any]] = []
        from nirs4all.pipeline.dagml.fixed_cohort_views import FixedCohortViewStore

        original_materialize = FixedCohortViewStore.__call__

        def materialize(view_store: Any, call: dict[str, Any]) -> dict[str, Any]:
            receipt = original_materialize(view_store, call)
            self.materializations.append({"request": deepcopy(call["request"]), "receipt": deepcopy(receipt)})
            return receipt

        monkeypatch.setattr(FixedCohortViewStore, "__call__", materialize)
        original_fit = DagMLNamedTorchEstimator.fit
        original_new = DagMLNamedTorchEstimator._new_named_model

        def module(estimator: Any, shapes: Any) -> Any:
            result = original_new(estimator, shapes)
            context = _ACTIVE.get()
            assert context is not None
            context["initial_state"] = {name: value.detach().clone() for name, value in result.state_dict().items()}
            context["rng_after_init"] = torch.get_rng_state().clone()
            context["effective_seed"] = torch.random.default_generator.initial_seed()
            return result

        def fit(estimator: Any, X: Any, y: Any) -> Any:
            context = _ACTIVE.get()
            assert context is not None, "FIT escaped the native task callback"
            task = context["task"]
            ids = _ids(task, "full_train" if task["phase"] == "REFIT" else "fold_train")
            assert all(cohort.partitions[self.rows[sample]] == "train" for sample in ids)
            assert not hasattr(estimator, "model_"), "a fold reused fitted weights"
            expected = self.features(ids)
            assert tuple(X) == tuple(expected)
            for name in expected:
                np.testing.assert_array_equal(X[name], expected[name])
            target = np.asarray(cohort.y)[[self.rows[sample] for sample in ids]].reshape(-1, 1)
            np.testing.assert_array_equal(np.asarray(y).reshape(-1, 1), target)
            if task["phase"] == "FIT_CV":
                heldout = _ids(task, "fold_validation")
                assert set(ids).isdisjoint(heldout)
                assert all(cohort.partitions[self.rows[sample]] == "train" for sample in heldout)
                assert {cohort.groups[self.rows[sample]] for sample in ids}.isdisjoint(
                    {cohort.groups[self.rows[sample]] for sample in heldout})
            context["rng_before_fit"] = torch.get_rng_state().clone()
            result = original_fit(estimator, X, y)
            rng_after_fit = torch.get_rng_state().clone()
            independent = oracle.fit_reference(expected, target, initial_state=context["initial_state"],
                rng_after_init=context["rng_after_init"], params=estimator.get_params(deep=False))
            assert torch.equal(torch.get_rng_state(), rng_after_fit), "independent oracle perturbed the next native task"
            for name, value in estimator.model_.state_dict().items():
                np.testing.assert_allclose(value.detach().numpy(), independent.state_dict()[name].detach().numpy(), rtol=2e-6, atol=2e-6)
            for name in expected:
                layer = estimator.model_.encoders[name]
                assert layer.weight.grad is not None and torch.count_nonzero(layer.weight.grad).item() > 0
                assert not torch.equal(layer.weight.detach(), context["initial_state"][f"encoders.{name}.weight"])
            assert all(estimator.model_ is not previous for previous in self.modules)
            self.modules.append(estimator.model_)
            context.update(reference=independent, fit_ids=ids, estimator=estimator)
            return result

        def observe(original: Any) -> Any:
            def callback(task: Any, resolver: Any, lookup: Any, store: Any, *args: Any, **kwargs: Any) -> Any:
                if task["node_plan"]["kind"] != "model":
                    return original(task, resolver, lookup, store, *args, **kwargs)
                context = {"task": deepcopy(task), "rng_before_callback": torch.get_rng_state().clone()}
                self.assert_materialized(task)
                token = _ACTIVE.set(context)
                try:
                    response = original(task, resolver, lookup, store, *args, **kwargs)
                    assert torch.equal(torch.get_rng_state(), context["rng_before_callback"]), "native named CPU RNG scope leaked"
                    if task["phase"] == "PREDICT":
                        owners = [record for record in self.records if record["task"]["phase"] == "REFIT"
                                  and record["task"]["node_plan"]["node_id"] == task["node_plan"]["node_id"]
                                  and record["task"].get("variant_id") == task.get("variant_id")]
                        assert len(owners) == 1
                        context["reference"] = owners[0]["reference"]
                    assert "reference" in context
                    for block in response.get("predictions", []):
                        expected = oracle.predict_reference(context["reference"], self.features(block["sample_ids"]))
                        np.testing.assert_allclose(block["values"], expected, rtol=2e-6, atol=2e-6)
                        assert block["target_names"] == list(cohort.target_names)
                    context["response"] = deepcopy(response)
                    self.records.append(context)
                    return response
                finally:
                    _ACTIVE.reset(token)
            return callback

        monkeypatch.setattr(in_process_runner, "run_node", observe(in_process_runner.run_node))
        monkeypatch.setattr(full_train, "run_node", observe(full_train.run_node))
        monkeypatch.setattr(DagMLNamedTorchEstimator, "_new_named_model", module)
        monkeypatch.setattr(DagMLNamedTorchEstimator, "fit", fit)

    def features(self, ids: list[str]) -> dict[str, np.ndarray]:
        rows = [self.rows[sample] for sample in ids]
        return {name: np.asarray(source.values)[rows] for name, source in self.cohort.sources.items()}

    def assert_materialized(self, task: dict[str, Any]) -> None:
        """Observe genuine native provider receipts before the numerical owner."""
        views, receipts = task["data_views"], task["data_view_receipts"]
        assert views and set(receipts) == set(views)
        for key, view in views.items():
            receipt = receipts[key]
            assert set(receipt) == {"handle", "view_key", "sample_ids", "schema_fingerprint", "content_fingerprint"}
            assert receipt["handle"] == task["input_handles"][key]
            assert receipt["sample_ids"] == view["sample_ids"]
            matches = [record for record in self.materializations
                       if record["receipt"] == receipt and record["request"]["view"] == view]
            assert len(matches) == 1, "callback did not consume an actually materialized native provider view"
            request = matches[0]["request"]
            port = request["input_name"]
            assert key in (f"data:{port}", f"data:{port}:validation", f"data:{port}:test")
            assert request["node_id"] == task["node_plan"]["node_id"]
            assert request["phase"] == task["phase"]
            assert request["fold_id"] == task.get("fold_id")
            assert receipt["view_key"] == request["view_key"]
            assert view["source_ids"] == request["binding"]["source_ids"]
            assert len(view["source_ids"]) == 1
            for field in ("schema_fingerprint", "content_fingerprint"):
                digest = receipt[field]
                assert len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)

    def observe_prediction(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        """Observe the real PyO3 PREDICT callback, not a replacement provider."""
        import importlib

        extension = importlib.import_module("dag_ml._dag_ml")
        execute = extension.execute_phase_in_process
        tasks: list[dict[str, Any]] = []

        def observed_execute(*args: Any, **kwargs: Any) -> Any:
            if len(args) < 5 or args[4] != "PREDICT":
                return execute(*args, **kwargs)
            arguments = list(args)
            callback = arguments[3]

            def observed_callback(task: dict[str, Any]) -> dict[str, Any]:
                if task["node_plan"]["kind"] == "model":
                    self.assert_materialized(task)
                    tasks.append(deepcopy(task))
                before = torch.get_rng_state().clone()
                result = callback(task)
                assert torch.equal(torch.get_rng_state(), before), "named PREDICT leaked CPU RNG"
                return result

            arguments[3] = observed_callback
            return execute(*arguments, **kwargs)

        monkeypatch.setattr(extension, "execute_phase_in_process", observed_execute)
        return tasks


@pytest.fixture(autouse=True)
def native_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.delenv("N4A_NATIVE_RESULTS", raising=False)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *a, **k: pytest.fail("legacy executed"))
    monkeypatch.setattr(dag_ml, "run_host_hpo_search_in_process", lambda *a, **k: pytest.fail("unexpected HPO"))


@pytest.mark.parametrize("count", [2, 3, 4])
@pytest.mark.parametrize("template", [False, True])
@pytest.mark.parametrize("cv", [False, True])
def test_public_named_run_uses_real_joint_tensors_and_native_train_scopes(
    count: int, template: bool, cv: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    cohort = _cohort(count, target_name="concentration")
    observer = _Observer(monkeypatch, cohort)
    result = _run(cohort, tmp_path / "workspace", template=template, cv=cv)
    try:
        fits = [record for record in observer.records if record["task"]["phase"] in ("FIT_CV", "REFIT")]
        assert len(fits) == (4 if cv else 1)
        assert [record["task"]["phase"] for record in fits].count("FIT_CV") == (3 if cv else 0)
        refits = [record for record in fits if record["task"]["phase"] == "REFIT"]
        assert len(refits) == 1
        artifacts = result._dagml_refit_artifacts
        assert len(artifacts) == 1
        artifact = artifacts[0]
        origin = artifact["named_refit_origin"]
        assert artifact["content_fingerprint"] == artifact["named_refit_fingerprint"]
        assert origin["phase"] == "REFIT" and origin["fold_id"] is None
        assert origin["fit_sample_ids"] == refits[0]["fit_ids"]
        assert origin["effective_seed"] == refits[0]["effective_seed"]
        assert origin["target_names"] == ["concentration"]
        assert origin["multimodal_input_schema"] == artifact["estimator"].multimodal_input_schema
        assert origin["native_seed"] == refits[0]["task"]["seed"]
        assert [port["name"] for port in origin["model_input"]["ports"]] == list(cohort.sources)
        native = read_native_results(result._dagml_results_dir)
        loaded = native["artifacts"][0]
        assert loaded["named_refit_origin"] == origin
        assert loaded["content_fingerprint"] == artifact["content_fingerprint"]
        assert loaded["serialization_fingerprint"] != loaded["content_fingerprint"]
        archive = result.export(tmp_path / "named.n4a")
        frozen = load_general_archive(archive)
        assert frozen["artifact"]["named_refit_origin"] == origin
        prediction = _cohort(count, prediction=True, target_name="concentration")
        expected = oracle.predict_reference(refits[0]["reference"],
                                           {name: source.values for name, source in prediction.sources.items()})
        predict_tasks = observer.observe_prediction(monkeypatch)
        actual = nirs4all.predict(archive, prediction, engine="dag-ml", verbose=0)
        assert len(predict_tasks) == 1
        assert predict_tasks[0]["phase"] == "PREDICT"
        np.testing.assert_allclose(actual.values.reshape(-1, 1), expected, rtol=2e-6, atol=2e-6)
    finally:
        result.close()


@pytest.mark.parametrize("lane", ["cv", "no_splitter", "predict"])
@pytest.mark.parametrize("damage", ["absent", "content"])
def test_named_public_paths_refuse_missing_or_changed_real_provider_receipts_before_numerics(
    lane: str, damage: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from nirs4all.pipeline.dagml.fixed_cohort_views import FixedCohortViewStore

    result = None
    archive = None
    if lane == "predict":
        result = _run(_cohort(), tmp_path / "original", cv=False)
        archive = result.export(tmp_path / "original.n4a")
    original = FixedCohortViewStore.__call__
    calls: list[dict[str, Any]] = []
    numerical_calls: list[bool] = []

    def damage_actual_receipt(store: Any, call: dict[str, Any]) -> Any:
        receipt = original(store, call)
        assert receipt["sample_ids"] == call["request"]["view"]["sample_ids"]
        calls.append(deepcopy(call))
        if damage == "absent":
            return None
        changed = deepcopy(receipt)
        changed["content_fingerprint"] = "0" * 64
        assert changed != receipt
        return changed

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        numerical_calls.append(True)
        pytest.fail("numerical constructor/FIT/PREDICT entered without the original materialized receipt")

    monkeypatch.setattr(FixedCohortViewStore, "__call__", damage_actual_receipt)
    monkeypatch.setattr(DagMLNamedTorchEstimator, "_new_named_model", forbidden)
    monkeypatch.setattr(DagMLNamedTorchEstimator, "fit", forbidden)
    monkeypatch.setattr(DagMLNamedTorchEstimator, "predict", forbidden)
    try:
        with pytest.raises((ValueError, dag_ml.DagMlRuntimeError, dag_ml.DagMlCompatibilityError), match="(?i)(receipt|materializ|view)"):
            if lane == "predict":
                assert archive is not None
                nirs4all.predict(archive, _cohort(prediction=True), engine="dag-ml", verbose=0)
            else:
                _run(_cohort(), tmp_path / "refused", cv=lane == "cv")
        assert calls, "the real fixed-cohort provider was never invoked"
        assert numerical_calls == [], "a converted callback error concealed numerical execution"
        assert calls[0]["request"]["phase"] == {"cv": "FIT_CV", "no_splitter": "REFIT", "predict": "PREDICT"}[lane]
    finally:
        if result is not None:
            result.close()


@pytest.mark.parametrize("cv", [False, True])
def test_public_random_state_controls_real_named_native_tasks_and_learned_weights(
    cv: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    cohort = _cohort()
    runs: list[dict[str, Any]] = []
    for index, seed in enumerate((21, 21, 29)):
        # Deliberately change ambient Torch RNG. Native task seeds, not this
        # caller state, must determine the real initialization and shuffling.
        with monkeypatch.context() as scoped, torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(900 + index)
            observer = _Observer(scoped, cohort)
            result = _run(cohort, tmp_path / f"seed_{index}", cv=cv, random_state=seed)
            try:
                fits = [record for record in observer.records if record["task"]["phase"] in {"FIT_CV", "REFIT"}]
                assert len(fits) == (4 if cv else 1)
                assert all(type(record["task"].get("seed")) is int for record in fits)
                artifact = result._dagml_refit_artifacts[0]
                refit = next(record for record in fits if record["task"]["phase"] == "REFIT")
                assert artifact["named_refit_origin"]["native_seed"] == refit["task"]["seed"]
                assert artifact["named_refit_origin"]["effective_seed"] == refit["effective_seed"]
                runs.append({"tasks": [deepcopy(record["task"]) for record in fits],
                             "origin": deepcopy(artifact["named_refit_origin"]),
                             "states": [{name: value.detach().clone() for name, value in record["estimator"].model_.state_dict().items()} for record in fits]})
            finally:
                result.close()
    def coordinates(run: dict[str, Any]) -> list[Any]:
        return [(task["phase"], task.get("fold_id"), task["node_plan"]["node_id"], task.get("variant_id")) for task in run["tasks"]]
    assert coordinates(runs[0]) == coordinates(runs[1]) == coordinates(runs[2])
    assert [task["seed"] for task in runs[0]["tasks"]] == [task["seed"] for task in runs[1]["tasks"]]
    assert [task["seed"] for task in runs[0]["tasks"]] != [task["seed"] for task in runs[2]["tasks"]]
    assert runs[0]["origin"]["native_seed"] == runs[1]["origin"]["native_seed"] != runs[2]["origin"]["native_seed"]
    assert runs[0]["origin"]["effective_seed"] == runs[1]["origin"]["effective_seed"] != runs[2]["origin"]["effective_seed"]
    for first, repeat in zip(runs[0]["states"], runs[1]["states"], strict=True):
        assert all(torch.equal(first[name], repeat[name]) for name in first)
    assert any(not torch.equal(runs[0]["states"][-1][name], runs[2]["states"][-1][name]) for name in runs[0]["states"][-1])


def _change_full_input_schema(schema: dict[str, Any], change: str) -> None:
    name = next(iter(schema))
    descriptor = schema[name]
    if change == "source_names":
        descriptor["source_id"] = name + "_changed"
        schema[name + "_changed"] = schema.pop(name)
    elif change == "feature_names":
        descriptor["feature_names"][0] += "_changed"
    elif change == "axes":
        descriptor["axes"][1] += "_changed"
        descriptor["native_representation"]["axes"][1]["name"] += "_changed"
    elif change == "units":
        axis = descriptor["axes"][1]
        descriptor["axis_units"][axis] = "changed_unit"
        descriptor["native_representation"]["axes"][1]["unit"] = "changed_unit"
    else:
        raise AssertionError(change)


@pytest.mark.parametrize("change", ["source_names", "feature_names", "axes", "units"])
def test_named_refit_schema_is_original_at_write_export_and_native_read(change: str, tmp_path: Path) -> None:
    result = _run(_cohort(), tmp_path / "workspace", save_artifacts=False)
    try:
        assert result._dagml_results_dir is None, "test requires the public memory-only export path"
        artifact = result._dagml_refit_artifacts[0]
        original_origin = deepcopy(artifact["named_refit_origin"])
        assert artifact["estimator"].multimodal_input_schema == original_origin["multimodal_input_schema"]
        carrier = tmp_path / "carrier"
        carrier.mkdir()
        references = _write_model_artifacts(carrier, [artifact])
        _change_full_input_schema(artifact["estimator"].multimodal_input_schema, change)
        assert artifact["named_refit_origin"] == original_origin
        refused = tmp_path / "refused"
        refused.mkdir()
        with pytest.raises(ValueError, match="named Torch"):
            _write_model_artifacts(refused, [artifact])
        assert not list(refused.rglob("*.joblib"))
        with pytest.raises(ValueError, match="named Torch"):
            result.export(tmp_path / "refused.n4a")
        assert not (tmp_path / "refused.n4a").exists()
        # Re-sign only the carrier bytes, preserving the independent genuine
        # native anchor. The reader must still reject the changed schema.
        path = carrier / references[0]["uri"]
        payload = joblib.load(path)
        _change_full_input_schema(payload["estimator"].multimodal_input_schema, change)
        joblib.dump(payload, path)
        references[0]["serialization_fingerprint"] = hashlib.sha256(path.read_bytes()).hexdigest()
        references[0]["size_bytes"] = path.stat().st_size
        with pytest.raises(ValueError, match="named Torch"):
            _rehydrate_artifacts(carrier, references)
        assert artifact["named_refit_origin"] == original_origin
    finally:
        result.close()


@pytest.mark.parametrize("tamper", ["weights", "params", "ports", "scope", "ids"])
def test_native_named_persistence_refuses_changed_state_without_replacing_origin(tamper: str, tmp_path: Path) -> None:
    result = _run(_cohort(), tmp_path / "workspace")
    try:
        artifact = deepcopy(result._dagml_refit_artifacts[0])
        original_origin = deepcopy(artifact["named_refit_origin"])
        if tamper == "weights":
            with torch.no_grad():
                artifact["estimator"].model_.head.weight.add_(0.01)
        elif tamper == "params":
            artifact["estimator"].lr *= 2
        elif tamper == "ports":
            artifact["estimator"]._nirs4all_named_model_input["ports"][0]["metadata"]["feature_shape"][0] += 1
        elif tamper == "scope":
            artifact["named_refit_origin"]["phase"] = "FIT_CV"
        else:
            artifact["named_refit_origin"]["fit_sample_ids"] = artifact["named_refit_origin"]["fit_sample_ids"][1:]
        directory = tmp_path / "tampered"
        directory.mkdir()
        with pytest.raises(ValueError, match="named Torch"):
            _write_model_artifacts(directory, [artifact])
        assert not list(directory.rglob("*.joblib"))
        assert result._dagml_refit_artifacts[0]["named_refit_origin"] == original_origin
    finally:
        result.close()


def test_native_named_artifact_bytes_are_checked_before_joblib_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run(_cohort(), tmp_path / "workspace")
    try:
        directory = tmp_path / "carrier"
        directory.mkdir()
        refs = _write_model_artifacts(directory, result._dagml_refit_artifacts)
        (directory / refs[0]["uri"]).write_bytes(b"corrupted carrier")
        monkeypatch.setattr(joblib, "load", lambda *a, **k: pytest.fail("unverified carrier unpickled"))
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            _rehydrate_artifacts(directory, refs)
    finally:
        result.close()


def test_native_named_persistence_refuses_an_estimator_from_another_genuine_refit(tmp_path: Path) -> None:
    first = _run(_cohort(), tmp_path / "first")
    second = _run(_cohort(), tmp_path / "second", cv=False, random_state=29)
    try:
        artifact = deepcopy(first._dagml_refit_artifacts[0])
        other = second._dagml_refit_artifacts[0]
        assert artifact["named_refit_origin"]["model_input"] == other["named_refit_origin"]["model_input"]
        assert artifact["named_refit_origin"] != other["named_refit_origin"]
        artifact["estimator"] = deepcopy(other["estimator"])
        directory = tmp_path / "swapped"
        directory.mkdir()
        with pytest.raises(ValueError, match="named Torch"):
            _write_model_artifacts(directory, [artifact])
        assert not list(directory.rglob("*.joblib"))
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("change", ["weights", "params", "origin", "missing_origin", "source_names", "feature_names", "axes", "units"])
def test_archive_rejects_resigned_carrier_with_unchanged_native_origin(change: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run(_cohort(), tmp_path / "workspace")
    try:
        archive = result.export(tmp_path / "original.n4a")
        with zipfile.ZipFile(archive) as source:
            members = {name: source.read(name) for name in source.namelist()}
        manifest = json.loads(members["manifest.json"])
        member = next(name for name in members if name.startswith("artifacts/") and name.endswith(".joblib"))
        model = joblib.load(io.BytesIO(members[member]))
        if change == "weights":
            with torch.no_grad():
                model.estimator.model_.head.weight.add_(0.01)
        elif change == "params":
            model.estimator.lr *= 2
        elif change == "origin":
            manifest["named_torch_refit"]["named_refit_origin"]["fit_sample_ids"].reverse()
        elif change == "missing_origin":
            manifest.pop("named_torch_refit")
        else:
            _change_full_input_schema(model.estimator.multimodal_input_schema, change)
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        members[member] = buffer.getvalue()
        manifest["artifact_integrity"][member] = "sha256:" + hashlib.sha256(members[member]).hexdigest()
        members["manifest.json"] = json.dumps(manifest).encode()
        changed = tmp_path / "changed.n4a"
        with zipfile.ZipFile(changed, "w") as target:
            for name, value in members.items():
                target.writestr(name, value)
        with pytest.raises(ValueError, match="named Torch"):
            load_general_archive(changed)
        monkeypatch.setattr(DagMLNamedTorchEstimator, "predict", lambda *a, **k: pytest.fail("changed state reached numerical PREDICT"))
        with pytest.raises(ValueError, match="named Torch"):
            nirs4all.predict(changed, _cohort(prediction=True), engine="dag-ml", verbose=0)
    finally:
        result.close()


def test_fresh_installed_named_archive_replay_is_prediction_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    python = os.environ.get("NIRS4ALL_NAMED_TORCH_INSTALLED_PYTHON")
    if not python:
        if os.environ.get("NIRS4ALL_REQUIRE_NAMED_TORCH_INSTALLED") == "1":
            pytest.fail("NIRS4ALL_NAMED_TORCH_INSTALLED_PYTHON is mandatory")
        pytest.skip("set a fresh installed interpreter for mandatory named Torch replay qualification")
    training = _cohort(4)
    observer = _Observer(monkeypatch, training)
    result = _run(training, tmp_path / "workspace")
    try:
        archive = result.export(tmp_path / "named.n4a")
        prediction = _cohort(4, prediction=True)
        refits = [record for record in observer.records if record["task"]["phase"] == "REFIT"]
        assert len(refits) == 1
        expected = oracle.predict_reference(refits[0]["reference"],
                                           {name: source.values for name, source in prediction.sources.items()})
        data = {"ids": list(prediction.sample_ids), "sources": {
            name: {"values": source.values.tolist(), "sample_ids": list(source.sample_ids), "feature_names": source.feature_names}
            for name, source in prediction.sources.items()}}
        inputs = tmp_path / "inputs.json"
        inputs.write_text(json.dumps(data))
        clean = tmp_path / "clean"
        clean.mkdir()
        script = textwrap.dedent("""\
            import importlib,json,pathlib,sys,types
            import numpy as np
            import nirs4all,dag_ml
            from nirs4all_io import MultimodalDataset,TensorSource
            prefix=pathlib.Path(sys.prefix).resolve()
            origins=['nirs4all','dag_ml','dag_ml._dag_ml','nirs4all.api.result','nirs4all.pipeline.dagml.named_torch_estimator','nirs4all.pipeline.dagml.named_torch','nirs4all.pipeline.dagml.native_results','nirs4all.pipeline.dagml.general_archive','nirs4all.pipeline.dagml.general_replay','nirs4all.pipeline.dagml.fixed_cohort_views','nirs4all.pipeline.dagml.node_runner','nirs4all.controllers.models.torch_model']
            for name in origins:
                assert pathlib.Path(importlib.import_module(name).__file__).resolve().is_relative_to(prefix),name
            # Mount only the test-owned user factory, never an SDK source root.
            resource=pathlib.Path(sys.argv[3])/'tests'
            for name,folder in [('tests',resource),('tests.fixtures',resource/'fixtures')]:
                package=types.ModuleType(name)
                package.__path__=[str(folder)]
                sys.modules[name]=package
            import tests.fixtures.named_torch_training
            from nirs4all.pipeline.dagml.named_torch_estimator import DagMLNamedTorchEstimator
            from nirs4all.controllers.models.torch_model import PyTorchModelController
            from nirs4all.pipeline.dagml.fixed_cohort_views import FixedCohortViewStore
            from nirs4all.pipeline import PipelineRunner
            def forbidden(*args,**kwargs):raise AssertionError('FIT/HPO entered during archive PREDICT')
            DagMLNamedTorchEstimator.fit=forbidden
            PyTorchModelController._train_model=forbidden
            PipelineRunner.run=forbidden
            dag_ml.run_host_hpo_search_in_process=forbidden
            materializations=[]
            original_materialize=FixedCohortViewStore.__call__
            def materialize(store,call):
                receipt=original_materialize(store,call)
                materializations.append({'request':call['request'],'receipt':receipt})
                return receipt
            FixedCohortViewStore.__call__=materialize
            extension=importlib.import_module('dag_ml._dag_ml')
            original_execute=extension.execute_phase_in_process
            observed_tasks=[]
            def execute(*args,**kwargs):
                assert args[4]=='PREDICT'
                arguments=list(args)
                callback=arguments[3]
                def checked(task):
                    receipts=task['data_view_receipts']
                    assert receipts and set(receipts)==set(task['data_views'])
                    for key,view in task['data_views'].items():
                        receipt=receipts[key]
                        assert receipt['sample_ids']==view['sample_ids']
                        assert receipt['handle']==task['input_handles'][key]
                        matches=[item for item in materializations if item['receipt']==receipt and item['request']['view']==view]
                        assert len(matches)==1,'native PREDICT did not materialize its actual named table'
                    observed_tasks.append(task)
                    return callback(task)
                arguments[3]=checked
                return original_execute(*arguments,**kwargs)
            extension.execute_phase_in_process=execute
            payload=json.loads(pathlib.Path(sys.argv[2]).read_text())
            sources={name:TensorSource(np.asarray(item['values'],dtype=np.float32),item['sample_ids'],representation_id='tabular_numeric',feature_names=item['feature_names']) for name,item in payload['sources'].items()}
            cohort=MultimodalDataset(sources,sample_ids=payload['ids'],partitions=['predict']*len(payload['ids']),target_names=['y'])
            result=nirs4all.predict(sys.argv[1],cohort,engine='dag-ml',verbose=0)
            assert len(observed_tasks)==1
            for name in origins:
                assert pathlib.Path(importlib.import_module(name).__file__).resolve().is_relative_to(prefix),name
            print(json.dumps({'values':result.values.tolist(),'prefix':str(prefix),'materialized_views':len(materializations)}))
            """)
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run([python, "-I", "-B", "-c", script, str(archive), str(inputs), str(Path(__file__).resolve().parents[3])],
            cwd=clean, env=environment, text=True, capture_output=True, timeout=120, check=False)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        assert payload["materialized_views"] == 4
        values = payload["values"]
        np.testing.assert_allclose(np.asarray(values).reshape(-1, 1), expected, rtol=2e-6, atol=2e-6)
    finally:
        result.close()
