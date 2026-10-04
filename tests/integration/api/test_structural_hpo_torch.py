"""Actual native Torch topology HPO, independent scoped fits and cold replay."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import textwrap
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path
from typing import Any

import dag_ml
import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

import nirs4all
from nirs4all.pipeline.dagml import multimodal_tuning
from nirs4all.pipeline.dagml.cancellation import DagRunCancelled
from nirs4all.pipeline.dagml.general_archive import load_general_archive
from nirs4all.pipeline.dagml.structural_tuning import _prepare_structure
from nirs4all.pipeline.dagml.torch_estimator import DagMLTorchEstimator
from nirs4all.pipeline.dagml.tuning_contracts import tcv1_sha256
from tests.fixtures import torch_topology_oracle as oracle
from tests.integration.api.test_structural_hpo_preprocessing_chains import _enqueue_recipes

pytestmark = pytest.mark.torch
_PATH = Path(__file__).resolve().parents[3] / "examples/user/04_models/U24_structural_hpo_torch.py"
_SPEC = importlib.util.spec_from_file_location("structural_torch_example", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)
_ACTIVE: ContextVar[dict[str, Any] | None] = ContextVar("torch_oracle_native_task", default=None)
_SEARCH: ContextVar[bool] = ContextVar("torch_oracle_native_search", default=False)


@pytest.fixture(autouse=True)
def native_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *a, **k: pytest.fail("legacy scheduler executed"))
    monkeypatch.setattr("nirs4all.pipeline.dagml.run_paths._run_model_on_precomputed_matrix", lambda *a, **k: pytest.fail("host CV scheduler executed"))


def _run(cohort: Any, root: Path, *, fusion: str = "search", tuning: Any = None, pipeline: Any = None) -> Any:
    return nirs4all.run(example.make_pipeline(fusion) if pipeline is None else pipeline, cohort,
        tuning=example.make_tuning(root / "study", fusion=fusion) if tuning is None else tuning,
        engine="dag-ml", workspace_path=root / "workspace", random_state=17, refit=True,
        cpu_threads=1, gpu_devices=[], save_artifacts=True, save_charts=False, verbose=0)


def _view_ids(task: dict[str, Any], partition: str) -> list[str]:
    views = [view for view in task.get("data_views", {}).values() if view["partition"] == partition]
    assert views and all(view["sample_ids"] == views[0]["sample_ids"] for view in views)
    return list(views[0]["sample_ids"])


class _Observer:
    """Use actual native tasks only as scope declarations, never as numeric oracle."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, cohort: Any) -> None:
        self.cohort = cohort
        self.rows = {sample: index for index, sample in enumerate(cohort.sample_ids)}
        self.records: list[dict[str, Any]] = []
        self.modules: list[Any] = []
        self.stores: list[Any] = []
        original_run = multimodal_tuning.run_node
        original_torch = DagMLTorchEstimator.fit
        original_ridge = Ridge.fit
        original_search = dag_ml.run_host_hpo_search_in_process

        def search(*args: Any, **kwargs: Any) -> Any:
            token = _SEARCH.set(True)
            try:
                return original_search(*args, **kwargs)
            finally:
                _SEARCH.reset(token)

        def run_node(task: dict[str, Any], resolver: Any, lookup: Any, store: Any, *args: Any, **kwargs: Any) -> Any:
            node = lookup(task["node_plan"]["node_id"])
            if task["node_plan"]["kind"] != "model":
                return original_run(task, resolver, lookup, store, *args, **kwargs)
            context = {"task": deepcopy(task), "node": deepcopy(node), "store": store,
                       "profile": kwargs["graph_metadata"]["python_torch_profile"], "search": _SEARCH.get()}
            self.stores.append(store)  # prevent object-ID reuse between candidate owners
            token = _ACTIVE.set(context)
            try:
                response = original_run(task, resolver, lookup, store, *args, **kwargs)
                context["response"] = deepcopy(response)
                expected = []
                for block in response.get("predictions", []):
                    assert block["target_names"] == list(cohort.target_names)
                    if context.get("raw"):
                        predicted = oracle.predict_torch(context["oracle"], self.features(block["sample_ids"], node["metadata"]["source_selection"]))
                    else:
                        delivery = ":outer" if task["phase"] == "FIT_CV" else ":refit"
                        features = self.meta_features(context, block["sample_ids"], delivery)
                        temporary = _ACTIVE.set(None)
                        try:
                            predicted = oracle.fit_ridge_predict(context["X"], context["y"], features, alpha=context["alpha"])
                        finally:
                            _ACTIVE.reset(temporary)
                    np.testing.assert_allclose(block["values"], predicted, rtol=3e-6, atol=3e-6)
                    expected.append({**deepcopy(block), "values": predicted.tolist()})
                context["expected"] = expected
                self.records.append(context)
                return response
            finally:
                _ACTIVE.reset(token)

        def torch_fit(estimator: Any, X: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
            context = _ACTIVE.get()
            if context is None:
                return original_torch(estimator, X, y, *args, **kwargs)
            task, node = context["task"], context["node"]
            ids = _view_ids(task, "full_train" if task["phase"] == "REFIT" else "fold_train")
            expected = self.features(ids, node["metadata"]["source_selection"])
            self.assert_train(ids)
            if task["phase"] == "FIT_CV":
                heldout = _view_ids(task, "fold_validation")
                self.assert_train(heldout)
                assert set(ids).isdisjoint(heldout)
                train_groups = {cohort.groups[self.rows[sample]] for sample in ids}
                assert train_groups.isdisjoint({cohort.groups[self.rows[sample]] for sample in heldout})
                if task["fold_id"] in {"fold0", "fold1", "fold2"}:
                    pool = [index for index, partition in enumerate(cohort.partitions) if partition == "train"]
                    groups = np.asarray(cohort.groups)[pool]
                    fold = int(task["fold_id"].removeprefix("fold"))
                    train, validation = list(GroupKFold(3).split(np.zeros((len(pool), 1)), groups=groups))[fold]
                    assert set(ids) == {cohort.sample_ids[pool[index]] for index in train}
                    assert set(heldout) == {cohort.sample_ids[pool[index]] for index in validation}
            np.testing.assert_allclose(X, expected, rtol=0, atol=0)
            np.testing.assert_array_equal(np.asarray(y).ravel(), self.targets(ids).ravel())
            assert not hasattr(estimator, "model_"), "candidate reused a fitted module"
            params = deepcopy(estimator.get_params(deep=False))
            seed = int(tcv1_sha256({"seed": context["profile"]["seed"], "variant": task.get("variant_id"),
                "fold": task.get("fold_id"), "node": node["id"], "phase": task["phase"]})[:8], 16)
            result = original_torch(estimator, X, y, *args, **kwargs)
            independent = oracle.fit_torch(params, expected, self.targets(ids), seed=seed)
            for actual, reference in zip(estimator.model_.parameters(), independent.parameters(), strict=True):
                assert actual.device.type == "cpu" and str(actual.dtype) == "torch.float32"
                np.testing.assert_allclose(actual.detach().cpu().numpy(), reference.detach().cpu().numpy(), rtol=3e-6, atol=3e-6)
            assert all(estimator.model_ is not previous for previous in self.modules)
            self.modules.append(estimator.model_)
            context.update(raw=True, oracle=independent, train_ids=ids, params=params, estimator=estimator)
            return result

        def ridge_fit(estimator: Any, X: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
            context = _ACTIVE.get()
            if context is None:
                return original_ridge(estimator, X, y, *args, **kwargs)
            specs = [value for key, value in context["task"]["prediction_inputs"].items()
                     if not key.endswith((":outer", ":refit", ":predict", ":test"))]
            assert specs
            ids = list(specs[0]["sample_ids"])
            self.assert_train(ids)
            expected = self.meta_features(context, ids, "")
            np.testing.assert_allclose(X, expected, rtol=3e-6, atol=3e-6)
            np.testing.assert_array_equal(np.asarray(y).ravel(), self.targets(ids).ravel())
            context.update(raw=False, X=expected, y=self.targets(ids), alpha=estimator.alpha, train_ids=ids)
            return original_ridge(estimator, X, y, *args, **kwargs)

        monkeypatch.setattr(multimodal_tuning, "run_node", run_node)
        monkeypatch.setattr(DagMLTorchEstimator, "fit", torch_fit)
        monkeypatch.setattr(Ridge, "fit", ridge_fit)
        monkeypatch.setattr(dag_ml, "run_host_hpo_search_in_process", search)

    def features(self, ids: list[str], names: list[str]) -> np.ndarray:
        rows = [self.rows[sample] for sample in ids]
        return np.concatenate([self.cohort.sources[name].values[rows] for name in names], axis=1)

    def targets(self, ids: list[str]) -> np.ndarray:
        # The public TargetConverter stores numeric targets in float32 before
        # native callbacks expose them as float64; preserve that input contract
        # while computing the reference fits independently.
        return np.asarray(self.cohort.y, dtype=np.float32)[[self.rows[sample] for sample in ids]].astype(np.float64).reshape(-1, 1)

    def assert_train(self, ids: list[str]) -> None:
        assert ids and len(ids) == len(set(ids))
        assert all(self.cohort.partitions[self.rows[sample]] == "train" for sample in ids)

    def meta_features(self, context: dict[str, Any], ids: list[str], suffix: str) -> np.ndarray:
        task = context["task"]
        ordered = context["node"]["metadata"]["prediction_source_order"]
        columns = []
        for producer in ordered:
            specs = [spec for key, spec in task["prediction_inputs"].items() if spec["producer_node"] == producer
                     and (key.endswith(suffix) if suffix else not key.endswith((":outer", ":refit", ":predict", ":test")))]
            assert len(specs) == 1
            spec = specs[0]
            assert set(spec["sample_ids"]) == set(ids)
            values = {}
            for sample in ids:
                matches = []
                for record in self.records:
                    if (not record.get("raw") or record["store"] is not context["store"]
                            or record["node"]["id"] != producer or record["task"]["variant_id"] != task["variant_id"]):
                        continue
                    for block in record["expected"]:
                        if block["partition"] != spec["partition"] or sample not in block["sample_ids"]:
                            continue
                        if not suffix:
                            assert spec["partition"] == "validation" and spec.get("fold_ids")
                            if block["partition"] != "validation" or block.get("fold_id") not in spec["fold_ids"]:
                                continue
                            assert set(record["train_ids"]).isdisjoint(block["sample_ids"])
                            assert set(record["train_ids"]) | set(block["sample_ids"]) == set(ids)
                            train_groups = {self.cohort.groups[self.rows[s]] for s in record["train_ids"]}
                            assert train_groups.isdisjoint({self.cohort.groups[self.rows[s]] for s in block["sample_ids"]})
                        elif record["task"]["phase"] != task["phase"] or record["task"].get("fold_id") != task.get("fold_id"):
                            continue
                        matches.append(block["values"][block["sample_ids"].index(sample)])
                assert len(matches) == 1, (producer, sample, suffix, len(matches))
                values[sample] = matches[0]
            actual = dict(zip(spec["sample_ids"], spec["values"], strict=True))
            column = np.asarray([values[sample] for sample in ids])
            np.testing.assert_allclose([actual[sample] for sample in ids], column, rtol=3e-6, atol=3e-6)
            columns.append(column)
        return np.concatenate(columns, axis=1)

    def full_refit_prediction(self, captured: Any, new: Any) -> np.ndarray:
        graph = captured.package["effective_plan"]["graph_plan"]["graph"]
        predictions = {}
        refits = [record for record in self.records if record["task"]["phase"] == "REFIT"]
        for node in graph["nodes"]:
            if node["kind"] != "model":
                continue
            records = [record for record in refits if record["node"]["id"] == node["id"]]
            assert len(records) == 1
            record = records[0]
            if record["raw"]:
                matrix = np.concatenate([new.sources[name].values for name in node["metadata"]["source_selection"]], axis=1)
                predictions[node["id"]] = oracle.predict_torch(record["oracle"], matrix)
        terminal = captured.package["output_bindings"][0]["node_id"]
        record = next(record for record in refits if record["node"]["id"] == terminal)
        if record["raw"]:
            return predictions[terminal]
        return oracle.fit_ridge_predict(record["X"], record["y"], np.concatenate(
            [predictions[producer] for producer in record["node"]["metadata"]["prediction_source_order"]], axis=1), alpha=record["alpha"])


def _assert_selection(result: Any, observer: _Observer) -> None:
    evidence = result.structural_tuning_evidence
    catalogue = result.structural_tuning_search_request["request"]["structural_catalogue"]
    recipes = {entry["recipe_id"]: entry for entry in catalogue["entries"]}
    for trial, public in zip(evidence["trials"], result.tuning_result.trials, strict=True):
        recipe = recipes[trial["params"]["__recipe__"]]
        assert set(trial["params"]) == {"__recipe__", *recipe["parameter_bindings"]}
        assert public.state == "COMPLETE" and public.value == trial["score"]
        scores = {}
        for record in observer.records:
            if (not record["search"] or record["node"]["id"] != recipe["target_node"]
                    or record["task"]["variant_id"] != trial["variant_id"]):
                continue
            for block in record["expected"]:
                if block["partition"] == "validation" and block.get("fold_id") in {"fold0", "fold1", "fold2"}:
                    assert block["fold_id"] not in scores
                    scores[block["fold_id"]] = float(np.sqrt(np.mean((np.asarray(block["values"]) - observer.targets(block["sample_ids"])) ** 2)))
        assert set(scores) == {"fold0", "fold1", "fold2"}
        assert trial["objective_fold_scores"] == pytest.approx(scores, abs=5e-6)
        assert trial["score"] == pytest.approx(np.mean(list(scores.values())), abs=5e-6)
    winner = next(trial for trial in evidence["trials"] if trial["trial_index"] == evidence["selected_trial_index"])
    assert winner["score"] == min(trial["score"] for trial in evidence["trials"])
    terminal = result.structural_tuning_training_request["options"]["outputs"][0]["node_id"]
    assert terminal == recipes[winner["params"]["__recipe__"]]["target_node"]
    captured = result._dagml_refit_artifacts[0]["estimator"]
    assert {record["node_id"] for record in captured.package["execution_bundle"]["refit_artifacts"]} == {
        node["id"] for node in result._dagml_graph["nodes"] if node["kind"] == "model"}


def test_all_topologies_have_independent_train_only_numeric_oracles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cohort = example.make_dataset()
    tuning = example.make_tuning(tmp_path / "study")
    prepared = _prepare_structure(example.make_pipeline(), cohort, tuning, {"random_state": 17})
    _enqueue_recipes(monkeypatch, prepared["catalogue"], prepared["catalogue"]["entries"])
    observer = _Observer(monkeypatch, cohort)
    with _run(cohort, tmp_path, tuning=tuning) as result:
        _assert_selection(result, observer)
        selected = result._dagml_refit_artifacts[0]["estimator"]
        new = example.make_dataset(29, prediction=True)
        expected = observer.full_refit_prediction(selected, new)
        archive = result.export(tmp_path / "all-topologies.n4a")
        actual = nirs4all.predict(archive, new, engine="dag-ml")
        np.testing.assert_allclose(actual.y_pred, expected, rtol=5e-6, atol=5e-6)
        assert actual.metadata["training_performed"] is False


@pytest.mark.parametrize("fusion,train_only,target_name", [
    ("early", False, "y"), ("late", False, "y"), ("late", True, "y"),
    ("early", False, "concentration"), ("late", False, "concentration"), ("late", True, "concentration"),
])
def test_forced_complete_closure_and_strict_sidecar_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fusion: str, train_only: bool, target_name: str,
) -> None:
    cohort = example.make_dataset(train_only=train_only, target_name=target_name)
    observer = _Observer(monkeypatch, cohort)
    tuning = example.make_tuning(tmp_path / "study", fusion=fusion)
    tuning["n_trials"] = 1
    with _run(cohort, tmp_path, fusion=fusion, tuning=tuning) as result:
        captured = result._dagml_refit_artifacts[0]["estimator"]
        assert captured.package["output_bindings"][0]["target_names"] == [target_name]
        assert result._dagml_graph["metadata"]["python_torch_profile"]["target_names"] == [target_name]
        _assert_selection(result, observer)
        outcome = result.structural_tuning_training_outcome
        terminal = captured.package["output_bindings"][0]["node_id"]
        if fusion == "late":
            terminal_refit = next(record for record in observer.records if record["node"]["id"] == terminal and record["task"]["phase"] == "REFIT")
            blocks = terminal_refit["response"]["predictions"]
            if train_only:
                assert blocks == []
                bound = next(output for output in outcome["outputs"] if output["binding"]["node_id"] == terminal)
                assert bound["artifact_only"] is True and bound["predictions"] == [] and bound["aggregated_predictions"] == []
            else:
                test_ids = {sample for sample, partition in zip(cohort.sample_ids, cohort.partitions, strict=True) if partition == "test"}
                assert blocks and all(block["partition"] == "test" and block.get("fold_id") is None for block in blocks)
                assert {sample for block in blocks for sample in block["sample_ids"]} == test_ids
                reports = [report for report in outcome["score_set"]["reports"] if report["producer_node"] == terminal and report.get("fold_id") is None]
                assert reports and all(report["partition"] == "test" for report in reports)
        new = example.make_dataset(29, prediction=True, target_name=target_name)
        expected = observer.full_refit_prediction(captured, new)
        archive = result.export(tmp_path / "forced.n4a")
    shutil.rmtree(tmp_path / "workspace")
    shutil.rmtree(tmp_path / "study")
    loaded = load_general_archive(archive)["artifact"]["estimator"]
    captured = loaded.estimator if hasattr(loaded, "estimator") else loaded
    np.testing.assert_allclose(nirs4all.predict(archive, new, engine="dag-ml").y_pred, expected, rtol=5e-6, atol=5e-6)
    if fusion == "late":
        invalid = deepcopy(captured)
        invalid.artifacts.pop(next(iter(invalid.artifacts)))
        with pytest.raises(ValueError, match="sidecars"):
            invalid.validate()
        for parameter in ("lr", "alpha"):
            invalid = deepcopy(captured)
            estimator = next(bundle["estimator"] for bundle in invalid.artifacts.values() if parameter in bundle["estimator"].get_params(deep=False))
            estimator.set_params(**{parameter: estimator.get_params(deep=False)[parameter] * 2})
            with pytest.raises(ValueError, match="signed parameters"):
                invalid.validate()


@pytest.mark.parametrize("tamper", ["swap_estimators", "swap_bundles", "foreign_fit_scope"])
def test_same_shape_and_controls_cannot_replace_signed_fitted_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tamper: str,
) -> None:
    cohort = example.make_dataset(target_name="concentration")
    observer = _Observer(monkeypatch, cohort)
    pipeline = [GroupKFold(3), {"_or_": [example.make_late(("series", "metadata"))]}]
    tuning = example.make_tuning(tmp_path / "study", fusion="late")
    tuning.update(n_trials=1, space={"late.series.lr": [0.01], "late.metadata.lr": [0.01], "late.meta.alpha": [1.0]})
    with _run(cohort, tmp_path, tuning=tuning, pipeline=pipeline) as result:
        archive = result.export(tmp_path / "compatible-branches.n4a")
    loaded = load_general_archive(archive)["artifact"]["estimator"]
    captured = loaded.estimator if hasattr(loaded, "estimator") else loaded
    raw = [record for record in captured.package["execution_bundle"]["refit_artifacts"]
           if record["controller_id"] == "controller:nirs4all.model"]
    assert len(raw) == 2
    first, second = [record["artifact"]["id"] for record in raw]
    original = captured.artifacts[first]["estimator"]
    other = captured.artifacts[second]["estimator"]
    assert original.get_params(deep=False) == other.get_params(deep=False)
    assert original.n_features_in_ == other.n_features_in_ == 2
    invalid = deepcopy(captured)
    if tamper == "swap_estimators":
        invalid.artifacts[first]["estimator"], invalid.artifacts[second]["estimator"] = (
            invalid.artifacts[second]["estimator"], invalid.artifacts[first]["estimator"])
    elif tamper == "swap_bundles":
        invalid.artifacts[first], invalid.artifacts[second] = invalid.artifacts[second], invalid.artifacts[first]
    else:
        foreign = next(record for record in observer.records if record["raw"] and not record["search"]
            and record["node"]["id"] == raw[0]["node_id"] and record["task"]["phase"] == "FIT_CV")
        assert foreign["estimator"].get_params(deep=False) == original.get_params(deep=False)
        assert foreign["estimator"].n_features_in_ == original.n_features_in_
        assert len(foreign["train_ids"]) < 36
        invalid.artifacts[first]["estimator"] = deepcopy(foreign["estimator"])
    with pytest.raises(ValueError):
        invalid.validate()


def test_resume_preserves_trials_and_refuses_changed_excluded_content_before_fit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cohort = example.make_dataset(target_name="concentration")
    tuning = example.make_tuning(tmp_path / "study", fusion="early")
    tuning["n_trials"] = 3
    tuning["progress_callback"] = lambda event: len(event["checkpoint"]["trials"]) < 1
    with pytest.raises(DagRunCancelled):
        _run(cohort, tmp_path, fusion="early", tuning=tuning)
    observer = _Observer(monkeypatch, cohort)
    resumed = example.make_tuning(tmp_path / "study", fusion="early", resume=True)
    resumed["n_trials"] = 3
    with _run(cohort, tmp_path, fusion="early", tuning=resumed) as result:
        assert len(result.structural_tuning_evidence["trials"]) == 3
        search_ids = {record["task"]["variant_id"] for record in observer.records if record["search"]}
        assert result.structural_tuning_evidence["trials"][0]["variant_id"] not in search_ids
    before = {path.name: path.read_bytes() for path in (tmp_path / "study").iterdir() if path.is_file()}
    source = cohort.sources["metadata"]
    changed_values = source.values.copy()
    changed_values[0, 0] += 0.5
    changed_source = example.TensorSource(changed_values, cohort.sample_ids,
        representation_id=source.representation_id, axes=source.axes, feature_names=source.feature_names)
    changed = example.MultimodalDataset({**cohort.sources, "metadata": changed_source},
        sample_ids=cohort.sample_ids, y=cohort.y, target_names=cohort.target_names,
        task_type=cohort.task_type, groups=cohort.groups, partitions=cohort.partitions, name=cohort.name)
    callbacks = len(observer.records)
    monkeypatch.setattr(DagMLTorchEstimator, "fit", lambda *a, **k: pytest.fail("changed resume invoked FIT"))
    from dag_ml import DagMlRuntimeError
    with pytest.raises((ValueError, RuntimeError, DagMlRuntimeError)):
        _run(changed, tmp_path, fusion="early", tuning=resumed)
    assert len(observer.records) == callbacks
    assert before == {path.name: path.read_bytes() for path in (tmp_path / "study").iterdir() if path.is_file()}


def test_fresh_installed_late_archive_has_full_native_predict_without_fit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    interpreter = os.environ.get("NIRS4ALL_TORCH_TOPOLOGY_INSTALLED_PYTHON")
    if not interpreter:
        if os.environ.get("NIRS4ALL_REQUIRE_TORCH_TOPOLOGY_INSTALLED") == "1":
            pytest.fail("mandatory fresh replay needs NIRS4ALL_TORCH_TOPOLOGY_INSTALLED_PYTHON")
        pytest.skip("fresh installed child is supplied by the qualification gate")
    cohort = example.make_dataset(target_name="concentration")
    observer = _Observer(monkeypatch, cohort)
    tuning = example.make_tuning(tmp_path / "training/study", fusion="late")
    tuning["n_trials"] = 1
    new = example.make_dataset(29, prediction=True, target_name="concentration")
    with _run(cohort, tmp_path / "training", fusion="late", tuning=tuning) as result:
        captured = result._dagml_refit_artifacts[0]["estimator"]
        prediction = observer.full_refit_prediction(captured, new)
        archive = Path(result.export(tmp_path / "cold.n4a"))
        graph = deepcopy(result._dagml_graph)
    shutil.rmtree(tmp_path / "training")
    paths = ["pipeline/dagml/structural_torch.py", "pipeline/dagml/structural_tuning.py", "pipeline/dagml/torch_topology_replay.py",
        "pipeline/dagml/node_runner.py", "pipeline/dagml/torch_estimator.py", "pipeline/dagml/general_archive.py",
        "pipeline/dagml/general_replay.py", "controllers/models/torch_model.py", "operators/models/pytorch/mlp.py"]
    root = Path(nirs4all.__file__).resolve().parent
    expected = {"prediction": prediction.tolist(), "graph": graph, "sources": {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths},
        "dag_extension": hashlib.sha256(Path(importlib.import_module("dag_ml._dag_ml").__file__).read_bytes()).hexdigest(),
        "archive": hashlib.sha256(archive.read_bytes()).hexdigest()}
    (tmp_path / "expected.json").write_text(json.dumps(expected, allow_nan=False), encoding="utf-8")
    np.savez(tmp_path / "sources.npz", **{name: source.values for name, source in new.sources.items()})
    script = textwrap.dedent('''\
        import hashlib, importlib, json, pathlib, sys
        import dag_ml, numpy as np, nirs4all
        from nirs4all_io import MultimodalDataset, TensorSource
        from sklearn.linear_model import Ridge
        from n4m.model_selection.optimizer import Optimizer
        from nirs4all.pipeline.dagml.torch_estimator import DagMLTorchEstimator
        from nirs4all.pipeline.dagml.general_archive import load_general_archive
        from nirs4all.pipeline.dagml.host_search_checkpoint import HostSearchOptimizer
        expected=json.loads(pathlib.Path(sys.argv[3]).read_text())
        root=pathlib.Path(nirs4all.__file__).resolve().parent
        assert 'site-packages' in root.parts, root
        for name,digest in expected['sources'].items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
        assert hashlib.sha256(pathlib.Path(importlib.import_module('dag_ml._dag_ml').__file__).read_bytes()).hexdigest()==expected['dag_extension']
        archive=pathlib.Path(sys.argv[1])
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected['archive']
        def forbidden(*args,**kwargs): raise AssertionError('cold replay reached FIT/HPO')
        DagMLTorchEstimator.fit=forbidden
        Ridge.fit=forbidden
        Optimizer.__init__=forbidden
        Optimizer.load=classmethod(forbidden)
        HostSearchOptimizer.__init__=forbidden
        nirs4all.run=forbidden
        dag_ml.execute_training=forbidden
        dag_ml.run_host_hpo_search_in_process=forbidden
        dag_ml.prepare_host_hpo_topology_catalogue=forbidden
        data=np.load(sys.argv[2]); ids=[f'new_{i}' for i in range(9)]
        order=expected['graph']['metadata']['python_torch_profile']['source_order']
        sources={name:TensorSource(data[name],ids,representation_id='tabular_numeric',axes=('sample','feature'),
            feature_names=[f'{name}:{i}' for i in range(data[name].shape[1])]) for name in order}
        names=expected['graph']['metadata']['python_torch_profile']['target_names']
        cohort=MultimodalDataset(sources,sample_ids=ids,y=None,target_names=names,task_type='regression',
            groups=[f'group_{i//3}' for i in range(9)],partitions=['train']*9,name='torch-structural-fixture')
        loaded=load_general_archive(archive)['artifact']['estimator']
        captured=loaded.estimator if hasattr(loaded,'estimator') else loaded
        assert captured.package['effective_plan']['graph_plan']['graph']==expected['graph']
        assert len(captured.artifacts)==3
        public=nirs4all.predict(archive,cohort,engine='dag-ml')
        with nirs4all.load_session(archive) as session: replay=session.predict(cohort)
        for result in (public,replay):
            np.testing.assert_allclose(result.y_pred,expected['prediction'],rtol=5e-6,atol=5e-6)
            assert result.metadata['training_performed'] is False
            assert result.metadata['phase']=='PREDICT'
            assert result.metadata['artifact_integrity_verified'] is True
        np.testing.assert_array_equal(public.y_pred,replay.y_pred)
        print(json.dumps({'fit_hpo_calls':0,'models':len(captured.artifacts),'sources':expected['sources']}))
    ''')
    process = subprocess.run([interpreter, "-I", "-c", script, str(archive), str(tmp_path / "sources.npz"), str(tmp_path / "expected.json")],
        cwd=tmp_path, text=True, capture_output=True, timeout=180, check=False)
    assert process.returncode == 0, process.stderr[-6000:]
    receipt = json.loads(process.stdout.splitlines()[-1])
    assert receipt == {"fit_hpo_calls": 0, "models": 3, "sources": expected["sources"]}
