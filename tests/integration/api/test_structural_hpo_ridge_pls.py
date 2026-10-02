"""Real native recipe search, grouped scientific oracle and no-fit archive replay."""

from __future__ import annotations

import base64
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.pipeline.dagml import node_runner
from nirs4all.pipeline.dagml.cancellation import DagRunCancelled
from nirs4all.pipeline.dagml.identity import mint_identity
from nirs4all.pipeline.dagml.tuning_contracts import tcv1_sha256

_EXAMPLE = Path(__file__).resolve().parents[3] / "examples/user/04_models/U17_structural_hpo_ridge_pls.py"
_SPEC = importlib.util.spec_from_file_location("structural_hpo_public_example", _EXAMPLE)
assert _SPEC is not None and _SPEC.loader is not None
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


@pytest.fixture(autouse=True)
def native_execution_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """A public structural search must never enter a Python scheduler or CV loop."""
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *a, **k: pytest.fail("legacy scheduler executed"))
    monkeypatch.setattr("nirs4all.pipeline.dagml.run_paths._run_model_on_precomputed_matrix", lambda *a, **k: pytest.fail("host CV scheduler executed"))


def _run(dataset: Any, root: Path, *, tuning: dict[str, Any] | None = None, pipeline: list[Any] | None = None) -> Any:
    return nirs4all.run(
        example.make_pipeline() if pipeline is None else pipeline, dataset,
        tuning=example.make_tuning(root / "study") if tuning is None else tuning,
        engine="dag-ml", workspace_path=root / "workspace", random_state=17,
        refit=True, verbose=0, save_charts=False, save_artifacts=True,
    )


def _checkpoint(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "structural-ridge-pls.n4mopt.json").read_text(encoding="utf-8"))


def _stop_after(count: int) -> Any:
    def progress(event: dict[str, Any]) -> bool:
        return len(event["checkpoint"]["trials"]) < count

    return progress


def _view_ids(task: dict[str, Any], partition: str) -> list[str]:
    views = [view for view in task.get("data_views", {}).values() if view.get("partition") == partition]
    assert views, f"missing native {partition} view"
    ordered = list(views[0]["sample_ids"])
    assert len(ordered) == len(set(ordered))
    assert all(list(view["sample_ids"]) == ordered for view in views)
    return ordered


class _NativeObserver:
    """Record real callbacks, sklearn FIT inputs and native optimizer lifecycle."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from n4m.model_selection.optimizer import Optimizer, Trial

        self.entries: list[dict[str, Any]] = []
        self.callbacks: list[dict[str, Any]] = []
        self.candidate_factories: list[tuple[int, int]] = []
        self.searches: list[dict[str, Any]] = []
        self.stores: dict[int, Any] = {}
        self.store_candidates: dict[int, set[tuple[int, str]]] = {}
        self.optimizers: list[Any] = []
        self.model_refs: list[Any] = []
        self.active_checks: list[tuple[int, str, bool]] = []
        self.original_fits = {"Ridge": Ridge.fit, "PLSRegression": PLSRegression.fit, "StandardScaler": StandardScaler.fit}
        self.fail_fold: str | None = None
        active: ContextVar[dict[str, Any] | None] = ContextVar("structural_native_task", default=None)
        search_scope: ContextVar[int | None] = ContextVar("structural_search_invocation", default=None)
        candidate_scope: ContextVar[int | None] = ContextVar("structural_candidate_invocation", default=None)
        native_search = dag_ml.run_host_hpo_search_in_process
        original_optimizer_init = Optimizer.__init__
        original_optimizer_load = Optimizer.load
        original_active = Trial.is_active

        def observe_optimizer(optimizer: Any, *args: Any, **kwargs: Any) -> None:
            original_optimizer_init(optimizer, *args, **kwargs)
            self.optimizers.append(optimizer)

        def observe_load(cls: Any, *args: Any, **kwargs: Any) -> Any:
            optimizer = original_optimizer_load(*args, **kwargs)
            self.optimizers.append(optimizer)
            return optimizer

        def observe_active(trial: Any, name: str) -> bool:
            value = original_active(trial, name)
            self.active_checks.append((trial.id, name, value))
            return value

        def callback(operator: Any, candidate: int | None = None) -> Any:
            def observe(task: dict[str, Any]) -> Any:
                observed = {"task": deepcopy(task), "search": search_scope.get(), "candidate": candidate}
                self.callbacks.append(observed)
                token = candidate_scope.set(candidate)
                try:
                    result = operator(task)
                    observed["result"] = deepcopy(result)
                    return result
                finally:
                    candidate_scope.reset(token)

            return observe

        def observe_search(*args: Any, **kwargs: Any) -> Any:
            search_index = len(self.searches)
            call: dict[str, Any] = {"dsl": deepcopy(args[0]), "envelope": deepcopy(args[1]),
                                   "controller_manifests": deepcopy(args[2]), "request": deepcopy(args[3])}
            self.searches.append(call)
            forwarded = list(args)
            forwarded[4] = callback(args[4])
            factory = kwargs.get("candidate_callback_factory")
            if factory is not None:
                def observe_factory(index: int) -> Any:
                    self.candidate_factories.append((search_index, index))
                    return callback(factory(index), index)

                kwargs["candidate_callback_factory"] = observe_factory
            token = search_scope.set(search_index)
            try:
                result = native_search(*forwarded, **kwargs)
                call["result"] = deepcopy(result)
                return result
            finally:
                search_scope.reset(token)

        def wrap_node(original: Any) -> Any:
            def observe(task: dict[str, Any], resolver: Any, lookup: Any, store: Any, *args: Any, **kwargs: Any) -> Any:
                index = search_scope.get()
                self.stores[id(store)] = store
                if index is not None:
                    candidate = str(candidate_scope.get())
                    self.store_candidates.setdefault(id(store), set()).add((index, candidate))
                observed = {"task": deepcopy(task), "search": index, "candidate": candidate_scope.get(), "store": id(store)}
                token = active.set(observed)
                try:
                    return original(task, resolver, lookup, store, *args, **kwargs)
                finally:
                    active.reset(token)

            return observe

        def wrap_fit(name: str) -> Any:
            original = self.original_fits[name]

            def observe(model: Any, X: Any, y: Any = None, *args: Any, **kwargs: Any) -> Any:
                context = active.get()
                if context is None:
                    return original(model, X, y, *args, **kwargs)
                self.model_refs.append(model)
                entry = {**context, "kind": name, "model_id": id(model), "params": deepcopy(model.get_params(deep=False)),
                         "X": np.array(X, copy=True), "y": None if y is None else np.array(y, copy=True),
                         "fresh": not hasattr(model, "n_features_in_")}
                self.entries.append(entry)
                result = original(model, X, y, *args, **kwargs)
                entry["fitted"] = True
                for attribute in ("coef_", "intercept_", "mean_", "var_", "scale_", "x_weights_"):
                    if hasattr(model, attribute):
                        entry[attribute] = np.array(getattr(model, attribute), copy=True)
                if context["task"].get("fold_id") == self.fail_fold and self.fail_fold is not None and name != "StandardScaler":
                    raise RuntimeError("injected failure after real structural candidate FIT")
                return result

            return observe

        monkeypatch.setattr(dag_ml, "run_host_hpo_search_in_process", observe_search)
        monkeypatch.setattr(node_runner, "run_model_node", wrap_node(node_runner.run_model_node))
        monkeypatch.setattr(node_runner, "_run_fitted_transform_node", wrap_node(node_runner._run_fitted_transform_node))
        monkeypatch.setattr(Optimizer, "__init__", observe_optimizer)
        monkeypatch.setattr(Optimizer, "load", classmethod(observe_load))
        monkeypatch.setattr(Trial, "is_active", observe_active)
        for name, cls in (("Ridge", Ridge), ("PLSRegression", PLSRegression), ("StandardScaler", StandardScaler)):
            monkeypatch.setattr(cls, "fit", wrap_fit(name))

    def assert_closed(self, *, candidates_expected: bool = True) -> None:
        assert self.optimizers, "no real Methods optimizer was created"
        for optimizer in self.optimizers:
            with pytest.raises(RuntimeError, match="closed"):
                optimizer.ask()
        assert bool(self.store_candidates) == candidates_expected, "unexpected search candidate store presence"
        for identifier, candidates in self.store_candidates.items():
            assert len(candidates) == 1, "learned state store crossed native candidate identities"
            assert not self.stores[identifier], "candidate retained learned handles after search terminal"


def _raw(dataset: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    X = np.asarray(dataset.x({}, layout="2d"))
    y = np.asarray(dataset.y({})).reshape(-1)
    groups = np.asarray(dataset.metadata_column("batch")).reshape(-1)
    identity = mint_identity(dataset)
    rows = {identity.to_wire(int(sample)): row for row, sample in enumerate(dataset.index_column("sample"))}
    return X, y, groups, rows


def _assert_fit_oracle(observer: _NativeObserver, dataset: Any) -> tuple[dict[int, dict[str, float]], np.ndarray]:
    """Use a separate sklearn estimator for every actual native training scope."""
    X, y, groups, rows = _raw(dataset)
    expected_folds = list(GroupKFold(3).split(X[:36], y[:36], groups[:36]))
    models = [entry for entry in observer.entries if entry["kind"] != "StandardScaler"]
    assert models and {entry["kind"] for entry in models} == {"Ridge", "PLSRegression"}
    assert all(entry["fresh"] for entry in observer.entries)
    assert len({entry["model_id"] for entry in observer.entries}) == len(observer.entries)
    scores: dict[int, dict[str, float]] = {}
    final_prediction: np.ndarray | None = None
    for entry in models:
        task = entry["task"]
        phase = task["phase"]
        train = np.array([rows[sample] for sample in _view_ids(task, "full_train" if phase == "REFIT" else "fold_train")])
        assert np.all(train < 36), "held-out test rows entered FIT"
        if phase == "FIT_CV":
            validation = np.array([rows[sample] for sample in _view_ids(task, "fold_validation")])
            assert np.all(validation < 36)
            assert set(groups[train]).isdisjoint(groups[validation])
            assert any(set(train) == set(fit) and set(validation) == set(heldout) for fit, heldout in expected_folds)
        else:
            assert phase == "REFIT" and set(train) == set(range(36))
        learned_scalers = [item for item in observer.entries if item["kind"] == "StandardScaler"
                           and item["search"] == entry["search"] and item["store"] == entry["store"]
                           and item["task"]["phase"] == phase and item["task"].get("fold_id") == task.get("fold_id")]
        assert len(learned_scalers) <= 1
        features = X[train]
        prediction_features = X[validation] if phase == "FIT_CV" else X[36:]
        if learned_scalers:
            scaler = StandardScaler()
            observer.original_fits["StandardScaler"](scaler, features)
            actual = learned_scalers[0]
            np.testing.assert_array_equal(actual["X"], features)
            np.testing.assert_allclose(actual["mean_"], scaler.mean_, rtol=0, atol=0)
            np.testing.assert_allclose(actual["var_"], scaler.var_, rtol=0, atol=0)
            features = scaler.transform(features)
            prediction_features = scaler.transform(prediction_features)
        np.testing.assert_allclose(entry["X"], features, rtol=1e-7, atol=1e-7)
        np.testing.assert_array_equal(entry["y"].reshape(-1), y[train])
        fitted = Ridge(**entry["params"]) if entry["kind"] == "Ridge" else PLSRegression(**entry["params"])
        observer.original_fits[entry["kind"]](fitted, features, entry["y"])
        np.testing.assert_allclose(entry["coef_"], fitted.coef_, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(entry["intercept_"], fitted.intercept_, rtol=2e-6, atol=2e-6)
        prediction = np.asarray(fitted.predict(prediction_features)).reshape(-1)
        if entry["search"] is not None:
            assert entry["candidate"] is not None
            assert task["phase"] == "FIT_CV", "HPO candidate was refitted"
            score = float(np.sqrt(np.mean((prediction - y[validation]) ** 2)))
            fold_scores = scores.setdefault(entry["candidate"], {})
            assert task["fold_id"] not in fold_scores
            fold_scores[task["fold_id"]] = score
        elif phase == "REFIT":
            assert final_prediction is None, "an unselected recipe was refitted"
            final_prediction = prediction
    assert final_prediction is not None
    return scores, final_prediction


def _catalogue(result: Any) -> dict[str, Any]:
    request = result.structural_tuning_search_request
    catalogue = request["request"]["structural_catalogue"]
    assert catalogue["schema_version"] == 1 and catalogue["selector_path"] == "__recipe__"
    assert len(catalogue["entries"]) == 4
    assert len({entry["recipe_id"] for entry in catalogue["entries"]}) == 4
    assert len({entry["variant"]["variant_id"] for entry in catalogue["entries"]}) == 4
    assert len({entry["variant_label"] for entry in catalogue["entries"]}) == 4
    assert {frozenset(entry["parameter_bindings"]) for entry in catalogue["entries"]} == {
        frozenset({"model.alpha"}), frozenset({"model.n_components"}),
    }
    compiled = dag_ml.prepare_host_hpo_structural_catalogue(
        request["dsl"], request["envelope"], request["controller_manifests"],
        {"model.alpha": "alpha", "model.n_components": "n_components"}, selector_path="__recipe__",
    )
    assert compiled == catalogue, "saved recipe identities differed from native compilation"
    return catalogue


def _assert_native_selection(result: Any, observer: _NativeObserver | None = None) -> None:
    catalogue = _catalogue(result)
    evidence = result.structural_tuning_evidence
    recipes = {entry["recipe_id"]: entry for entry in catalogue["entries"]}
    trials = result.tuning_result.trials
    assert len(trials) == len(evidence["trials"]) == 8
    assert [trial.state for trial in trials] == ["COMPLETE"] * 8
    for public, native in zip(trials, evidence["trials"], strict=True):
        assert public.number == native["trial_index"]
        recipe = recipes[native["params"]["__recipe__"]]
        assert set(native["params"]) == {"__recipe__", *recipe["parameter_bindings"]}
        assert public.params == {path: value for path, value in native["params"].items() if path != "__recipe__"}
        assert set(public.params) == set(recipe["parameter_bindings"])
        assert public.value == native["score"]
        assert public.diagnostics["engine"] == "dag-ml" and public.diagnostics["test_used"] is False
        assert set(native["objective_fold_scores"]) == {"fold0", "fold1", "fold2"}
        assert native["score"] == pytest.approx(np.mean(list(native["objective_fold_scores"].values())), abs=1e-12)
        if observer is not None:
            fits = [entry for entry in observer.entries if entry["search"] == 0 and entry["candidate"] == native["trial_index"]
                    and entry["kind"] != "StandardScaler"]
            assert len(fits) == 3
            for fit in fits:
                task = fit["task"]
                assert task["variant_id"] == native["variant_id"]
                assert task["node_plan"]["node_id"] == recipe["target_node"]
                overrides = task["node_plan"].get("params", {})
                expected = {binding["param_path"]: native["params"][path] for path, binding in recipe["parameter_bindings"].items()}
                declared = next(node for node in recipe["graph"]["nodes"] if node["id"] == recipe["target_node"])
                assert overrides == {**declared.get("params", {}), **expected}
                assert ("n_components" if "model.alpha" in public.params else "alpha") not in overrides, "inactive axis reached native FIT plan"
                assert fit["kind"] == ("Ridge" if "model.alpha" in public.params else "PLSRegression")
                for key, value in expected.items():
                    assert fit["params"][key] == value
            active = {name: value for index, name, value in observer.active_checks if index == native["trial_index"] and name.startswith("model.")}
            assert active == {"model.alpha": "model.alpha" in public.params, "model.n_components": "model.n_components" in public.params}
    selected = next(trial for trial in evidence["trials"] if trial["trial_index"] == evidence["selected_trial_index"])
    assert selected["score"] == min(trial["score"] for trial in evidence["trials"])
    assert evidence["selected_params"] == selected["params"]
    assert result.tuning_best_params == {path: value for path, value in selected["params"].items() if path != "__recipe__"}
    assert result.tuning_best_value == selected["score"]
    assert len(result._dagml_refit_artifacts) == 1
    selected_recipe = recipes[selected["params"]["__recipe__"]]
    selected_graph = result.structural_tuning_training_request["graph"]
    assert result._dagml_graph == selected_graph
    recipe_nodes = {node["id"]: node for node in selected_recipe["graph"]["nodes"]}
    assert {node["id"] for node in selected_graph["nodes"]} == set(recipe_nodes)
    for node in selected_graph["nodes"]:
        original = recipe_nodes[node["id"]]
        assert {key: value for key, value in node.items() if key != "params"} == {
            key: value for key, value in original.items() if key != "params"
        }
        active = {binding["param_path"]: selected["params"][path]
                  for path, binding in selected_recipe["parameter_bindings"].items() if binding["node_id"] == node["id"]}
        assert node.get("params", {}) == {**original.get("params", {}), **active}
    assert selected_graph.get("edges", []) == selected_recipe["graph"].get("edges", [])
    assert [node["id"] for node in selected_graph["nodes"] if node["kind"] == "model"] == [selected_recipe["target_node"]]
    assert len([node for node in selected_graph["nodes"] if node["kind"] == "transform"]) <= 1
    selection = result.structural_tuning_training_request["campaign"]["metadata"]["host_hpo_structural_selection"]
    assert selection["recipe_id"] == selected_recipe["recipe_id"]
    assert selection["variant_label"] == selected_recipe["variant_label"]
    assert selection["selected_trial_index"] == selected["trial_index"]
    assert selection["selected_params"] == selected["params"]
    assert result._dagml_refit_artifacts[0]["estimator"].structural_tuning_evidence == evidence
    if observer is not None:
        refits = [entry for entry in observer.entries if entry["kind"] != "StandardScaler" and entry["task"]["phase"] == "REFIT"]
        assert len(refits) == 1
        assert refits[0]["task"]["node_plan"]["node_id"] == selected_recipe["target_node"]
        for path, value in result.tuning_best_params.items():
            assert refits[0]["params"][selected_recipe["parameter_bindings"][path]["param_path"]] == value


@pytest.mark.parametrize("seed", [17, 23])
def test_public_structure_search_matches_train_only_grouped_oracle(seed: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observer = _NativeObserver(monkeypatch)
    dataset = example.make_dataset(seed)
    with _run(dataset, tmp_path) as result:
        _assert_native_selection(result, observer)
        assert result.structural_tuning_search_request == {key: value for key, value in observer.searches[0].items() if key != "result"}
        scores, expected = _assert_fit_oracle(observer, dataset)
        assert set(scores) == set(range(8))
        for trial in result.structural_tuning_evidence["trials"]:
            assert trial["objective_fold_scores"] == pytest.approx(scores[trial["trial_index"]], abs=2e-6)
            assert trial["score"] == pytest.approx(np.mean(list(scores[trial["trial_index"]].values())), abs=2e-6)
        archive = result.export(tmp_path / "winner.n4a")
        prediction = nirs4all.predict(archive, dataset.x({"partition": "test"}, layout="2d"), engine="dag-ml")
        np.testing.assert_allclose(prediction.y_pred.ravel(), expected, rtol=2e-6, atol=2e-6)
        assert result.best_rmse == pytest.approx(np.sqrt(np.mean((expected - _raw(dataset)[1][36:]) ** 2)), abs=2e-6)
        assert prediction.metadata["training_performed"] is False
    observer.assert_closed()


def _changed_dataset(dataset: Any, mutation: str) -> Any:
    from nirs4all.data import SpectroDataset

    X, y, groups, _ = _raw(dataset)
    X, y, groups = X.copy(), y.copy(), groups.copy()
    if mutation == "features":
        X[0, 0] += 0.25
    elif mutation == "targets":
        y[0] += 0.5
    elif mutation == "groups":
        groups[0] = groups[4]
    elif mutation == "test_targets":
        y[36:] += np.linspace(1000, 5000, 12)
    else:
        raise AssertionError(mutation)
    changed = SpectroDataset(dataset.name)
    changed.add_samples(X[:36], {"partition": "train"})
    changed.add_samples(X[36:], {"partition": "test"})
    changed.add_targets(y)
    changed.add_metadata(groups[:, None], headers=["batch"])
    return changed


@pytest.mark.parametrize("mutation", ["features", "targets", "groups", "folds", "model_order", "scaler_order", "space", "seed"])
def test_resume_refuses_changed_signed_training_and_structure_before_callback(mutation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observer = _NativeObserver(monkeypatch)
    dataset = example.make_dataset()
    tuning = example.make_tuning(tmp_path / "study")
    with pytest.raises(DagRunCancelled):
        _run(dataset, tmp_path, tuning={**tuning, "progress_callback": _stop_after(2)})
    checkpoint = tmp_path / "study/structural-ridge-pls.n4mopt.json"
    before = checkpoint.read_bytes()
    counts = len(observer.callbacks), len(observer.entries)
    changed = _changed_dataset(dataset, mutation) if mutation in {"features", "targets", "groups"} else dataset
    pipeline = example.make_pipeline()
    tuning = {**tuning, "resume": True}
    if mutation == "folds":
        pipeline[1]["split"] = GroupKFold(2)
    elif mutation == "model_order":
        pipeline[2]["model"]["_or_"].reverse()
    elif mutation == "scaler_order":
        pipeline[0]["_or_"].reverse()
    elif mutation == "space":
        tuning["space"] = {**tuning["space"], "model.alpha": {"type": "float", "low": 0.02, "high": 10.0, "log": True}}
    elif mutation == "seed":
        tuning["seed"] = 23
    with pytest.raises(Exception, match="(?i)(checkpoint|fingerprint|contract|catalogue|space|structur|fold|group|seed)"):
        _run(changed, tmp_path / "invalid", tuning=tuning, pipeline=pipeline)
    assert (len(observer.callbacks), len(observer.entries)) == counts
    assert checkpoint.read_bytes() == before
    observer.assert_closed()


@pytest.mark.parametrize("mutation", ["pair_seal", "native_history", "optimizer_payload", "catalogue_order", "activation_mask"])
def test_resume_refuses_tampered_pair_before_real_callback(mutation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observer = _NativeObserver(monkeypatch)
    dataset = example.make_dataset()
    tuning = example.make_tuning(tmp_path / "study")
    with pytest.raises(DagRunCancelled):
        _run(dataset, tmp_path, tuning={**tuning, "progress_callback": _stop_after(2)})
    path = tmp_path / "study/structural-ridge-pls.n4mopt.json"
    payload = _checkpoint(tmp_path / "study")
    if mutation == "pair_seal":
        payload["pair_fingerprint"] = "0" * 64
    elif mutation == "native_history":
        record = payload["native_checkpoint"]["trials"][0]
        record.get("evidence", record)["params"]["__recipe__"] = "recipe:unknown"
    elif mutation == "optimizer_payload":
        payload["checkpoint_b64"] = "invalid native checkpoint bytes"
        payload["checkpoint_fingerprint"] = tcv1_sha256({"checkpoint_b64": payload["checkpoint_b64"]})
    elif mutation == "catalogue_order":
        payload["structural_binding"]["catalogue"]["entries"].reverse()
    else:
        mask = payload["structural_binding"]["activation_masks"][0]
        mask["active_paths"] = ["__recipe__", "model.alpha", "model.n_components"]
    if mutation != "pair_seal":
        payload["pair_fingerprint"] = tcv1_sha256({key: payload[key] for key in ("checkpoint_fingerprint", "native_checkpoint", "structural_binding")})
    path.write_text(json.dumps(payload), encoding="utf-8")
    tampered = path.read_bytes()
    counts = len(observer.callbacks), len(observer.entries)
    with pytest.raises(Exception, match="(?i)(checkpoint|fingerprint|history|contract|payload|base64|native)"):
        _run(dataset, tmp_path / "invalid", tuning={**tuning, "resume": True})
    assert (len(observer.callbacks), len(observer.entries)) == counts
    assert path.read_bytes() == tampered
    observer.assert_closed()


def _native_history(records: list[Any]) -> list[dict[str, Any]]:
    """Compare semantic native history independently of time and selector encoding."""
    return [
        {
            "id": record.id, "ask_sequence": record.ask_sequence, "terminal_sequence": record.terminal_sequence,
            "params": record.params, "active": {path: detail.active for path, detail in record.param_details.items()},
            "status": record.status, "score": record.score, "rung": record.rung,
            "intermediates": record.intermediates, "error": record.error,
        }
        for record in records
    ]


def _substituted_native_optimizer(payload: dict[str, Any], mutation: str, tuning: dict[str, Any]) -> bytes:
    """Build/load a valid replacement with unchanged historical native trial facts."""
    from n4m.model_selection.optimizer import ConstraintKind, Direction, Optimizer, Sampler, SearchSpace, TrialStatus

    original = base64.b64decode(payload["checkpoint_b64"], validate=True)
    with Optimizer.load(original) as optimizer:
        records = optimizer.get_trials()
    assert len(records) == len(payload["native_checkpoint"]["trials"])
    assert all(record.status == TrialStatus.COMPLETED for record in records)
    catalogue = payload["structural_binding"]["catalogue"]
    selector = catalogue["selector_path"]
    recipe_ids = [entry["recipe_id"] for entry in catalogue["entries"]]
    # Recipe identities come from the native catalogue, whereas record.id is a trial identity.
    observed_recipes = {record.params[selector] for record in records}
    unseen = next(entry for entry in catalogue["entries"] if entry["recipe_id"] not in observed_recipes)
    assert unseen["recipe_id"] not in observed_recipes
    mutated_order = list(reversed(recipe_ids)) if mutation == "selector_order" else recipe_ids
    seed = tuning["seed"] + 1 if mutation == "seed" else tuning["seed"]
    with SearchSpace() as space:
        alpha = tuning["space"]["model.alpha"]
        components = tuning["space"]["model.n_components"]
        space.add_float("model.alpha", alpha["low"], 100.0 if mutation == "numeric_bounds" else alpha["high"], log=alpha["log"])
        space.add_int("model.n_components", components["low"], components["high"])
        space.add_categorical(selector, mutated_order)
        for path in sorted(tuning["space"]):
            for entry in catalogue["entries"]:
                if path not in entry["parameter_bindings"]:
                    continue
                if mutation == "unseen_condition" and entry["recipe_id"] == unseen["recipe_id"]:
                    continue
                space.add_constraint(ConstraintKind.CONDITION_IN, [path, selector], ["", entry["recipe_id"]])
        with Optimizer(space, sampler=Sampler.RANDOM, direction=Direction.MINIMIZE, seed=seed) as replacement:
            for record in records:
                forced = {**record.params, selector: mutated_order.index(record.params[selector])}
                replacement.enqueue(forced)
                trial = replacement.ask()
                assert trial.id == record.id
                for intermediate in record.intermediates:
                    assert replacement.tell_intermediate(trial.id, intermediate.step, intermediate.score) == intermediate.should_prune
                replacement.tell(trial.id, record.score)
            assert _native_history(replacement.get_trials()) == _native_history(records)
            substituted = replacement.save()
    assert substituted != original
    with Optimizer.load(substituted) as reloaded:
        assert _native_history(reloaded.get_trials()) == _native_history(records)
    return substituted


@pytest.mark.parametrize("history_count", [0, 1], ids=["initial-stop-empty-history", "completed-trial-unseen-recipe"])
@pytest.mark.parametrize("mutation", ["numeric_bounds", "selector_order", "seed", "unseen_condition"])
def test_resume_attests_valid_substituted_native_optimizer_before_ask_or_callback(
    mutation: str, history_count: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from n4m.model_selection.optimizer import Optimizer

    observer = _NativeObserver(monkeypatch)
    dataset = example.make_dataset()
    tuning = example.make_tuning(tmp_path / "study")
    with pytest.raises(DagRunCancelled):
        _run(dataset, tmp_path, tuning={**tuning, "progress_callback": _stop_after(history_count)})
    path = tmp_path / "study/structural-ridge-pls.n4mopt.json"
    payload = _checkpoint(tmp_path / "study")
    assert len(payload["native_checkpoint"]["trials"]) == history_count
    history = deepcopy(payload["native_checkpoint"])
    binding = deepcopy(payload["structural_binding"])
    payload["checkpoint_b64"] = base64.b64encode(_substituted_native_optimizer(payload, mutation, tuning)).decode("ascii")
    payload["checkpoint_fingerprint"] = tcv1_sha256({"checkpoint_b64": payload["checkpoint_b64"]})
    pair = {key: payload[key] for key in ("checkpoint_fingerprint", "native_checkpoint", "structural_binding")}
    payload["pair_fingerprint"] = tcv1_sha256(pair)
    assert payload["native_checkpoint"] == history and payload["structural_binding"] == binding
    assert payload["checkpoint_fingerprint"] == tcv1_sha256({"checkpoint_b64": payload["checkpoint_b64"]})
    assert payload["pair_fingerprint"] == tcv1_sha256(pair)
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    substituted_bytes = path.read_bytes()
    counts = len(observer.callbacks), len(observer.entries), len(observer.candidate_factories)

    def forbidden_ask(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("substituted native optimizer reached ask before contract refusal")

    def forbidden_fit(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("substituted native optimizer reached FIT before contract refusal")

    with monkeypatch.context() as guard:
        guard.setattr(Optimizer, "ask", forbidden_ask)
        for cls in (Ridge, PLSRegression, StandardScaler):
            guard.setattr(cls, "fit", forbidden_fit)
        with pytest.raises(ValueError, match="native optimizer checkpoint space or options contract mismatch"):
            _run(dataset, tmp_path / "invalid", tuning={**tuning, "resume": True})
    assert (len(observer.callbacks), len(observer.entries), len(observer.candidate_factories)) == counts
    assert path.read_bytes() == substituted_bytes
    observer.assert_closed(candidates_expected=bool(history_count))


@pytest.mark.parametrize("path,value", [
    ("model.alpha", 0.001), ("model.alpha", 10.01),
    ("model.n_components", 0), ("model.n_components", 4),
    ("model.n_components", 1.5), ("model.n_components", True),
], ids=["alpha-below", "alpha-above", "components-below", "components-above", "components-fraction", "components-boolean"])
def test_public_active_numeric_proposal_refused_before_factory_or_fit(
    path: str, value: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from n4m.model_selection.optimizer import Optimizer

    from nirs4all.pipeline.dagml import tuning_adapters

    observer = _NativeObserver(monkeypatch)
    dataset = example.make_dataset()
    tuning = example.make_tuning(tmp_path / "study")
    with pytest.raises(DagRunCancelled):
        _run(dataset, tmp_path, tuning={**tuning, "progress_callback": _stop_after(0)})
    checkpoint = tmp_path / "study/structural-ridge-pls.n4mopt.json"
    before = checkpoint.read_bytes()
    catalogue = observer.searches[0]["request"]["structural_catalogue"]
    selector = catalogue["selector_path"]
    recipe = next(entry for entry in catalogue["entries"] if path in entry["parameter_bindings"])
    recipe_index = next(index for index, entry in enumerate(catalogue["entries"]) if entry["recipe_id"] == recipe["recipe_id"])
    original_ask = Optimizer.ask
    original_params = tuning_adapters._n4m_trial_params
    decoded: list[dict[str, Any]] = []

    def ask_selected(optimizer: Any) -> Any:
        optimizer.enqueue({selector: recipe_index, path: 0.5 if path == "model.alpha" else 2})
        return original_ask(optimizer)

    def invalid_value(*args: Any, **kwargs: Any) -> dict[str, Any]:
        params = original_params(*args, **kwargs)
        assert kwargs.get("active_only") is True
        assert params[selector] == recipe["recipe_id"] and set(params) == {selector, path}
        params[path] = value
        decoded.append(deepcopy(params))
        return params

    def forbidden_fit(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("out-of-domain active proposal reached FIT")

    counts = len(observer.callbacks), len(observer.entries), len(observer.candidate_factories)
    with monkeypatch.context() as guard:
        guard.setattr(Optimizer, "ask", ask_selected)
        guard.setattr(tuning_adapters, "_n4m_trial_params", invalid_value)
        for cls in (Ridge, PLSRegression, StandardScaler):
            guard.setattr(cls, "fit", forbidden_fit)
        with pytest.raises(dag_ml.DagMlRuntimeError) as error:
            _run(dataset, tmp_path / "invalid", tuning={**tuning, "resume": True})
    assert type(error.value) is dag_ml.DagMlRuntimeError
    assert str(error.value) == (
        "runtime validation failed: python callback raised an exception: "
        f"structural proposal value is outside the declared search domain: {path}"
    )
    assert len(decoded) == 1 and decoded[0][path] == value
    assert (len(observer.callbacks), len(observer.entries), len(observer.candidate_factories)) == counts
    assert checkpoint.read_bytes() == before
    observer.assert_closed(candidates_expected=False)


def test_heldout_targets_do_not_change_any_native_search_score_or_recipe(tmp_path: Path) -> None:
    dataset = example.make_dataset()
    with _run(dataset, tmp_path / "baseline") as baseline, _run(_changed_dataset(dataset, "test_targets"), tmp_path / "perturbed") as perturbed:
        assert [trial.to_dict() for trial in baseline.tuning_result.trials] == [trial.to_dict() for trial in perturbed.tuning_result.trials]
        for left, right in zip(baseline.structural_tuning_evidence["trials"], perturbed.structural_tuning_evidence["trials"], strict=True):
            for field in ("params", "score", "variant_id", "objective_fold_scores"):
                assert left[field] == right[field]
        assert baseline.tuning_best_params == perturbed.tuning_best_params
        assert baseline.tuning_best_value == perturbed.tuning_best_value
        assert baseline.best_rmse != pytest.approx(perturbed.best_rmse)


@pytest.mark.parametrize("forced", [{"model.alpha": 0.1}, {"model.n_components": 2}, {"__recipe__": "recipe:unknown"},
                                     {"model.alpha": 0.1, "model.n_components": 2}])
def test_public_forced_axes_refused_without_recipe_guess_or_cold_fallback(forced: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observer = _NativeObserver(monkeypatch)
    tuning = {**example.make_tuning(tmp_path / "study"), "force_params": forced}
    with pytest.raises(Exception, match="force_params|recipe|selector|unknown"):
        _run(example.make_dataset(), tmp_path, tuning=tuning)
    assert not observer.callbacks and not observer.entries and not observer.optimizers


def test_failed_real_candidate_cleans_handles_and_does_not_change_next_candidate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observer = _NativeObserver(monkeypatch)
    dataset = example.make_dataset()
    tuning = example.make_tuning(tmp_path / "study")
    observer.fail_fold = "fold1"
    with pytest.raises(Exception, match="injected failure after real structural candidate FIT"):
        _run(dataset, tmp_path, tuning={**tuning, "progress_callback": _stop_after(1)})
    checkpoint = _checkpoint(tmp_path / "study")
    assert checkpoint["native_checkpoint"]["trials"][0]["state"] == "failed"
    observer.assert_closed()
    observer.fail_fold = None
    before = len(observer.callbacks)
    with _run(dataset, tmp_path / "recovered", tuning={**tuning, "resume": True}) as recovered:
        assert recovered.tuning_result.trials[0].state == "FAIL"
        assert [trial.state for trial in recovered.tuning_result.trials[1:]] == ["COMPLETE"] * 7
        calls = [call for call in observer.callbacks[before:] if call["search"] == 1]
        assert {call["candidate"] for call in calls} == set(range(1, 8))
        assert all(entry["fresh"] for entry in observer.entries)
    observer.assert_closed()


def test_native_parallel_workers_match_sequential_recipe_scores_and_are_reaped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from nirs4all.pipeline.dagml.host_hpo_candidate import HostHpoCandidate

    children: list[Any] = []
    original = HostHpoCandidate.__init__

    def observe(child: Any, *args: Any, **kwargs: Any) -> None:
        original(child, *args, **kwargs)
        children.append(child)

    monkeypatch.setattr(HostHpoCandidate, "__init__", observe)
    dataset = example.make_dataset()
    with _run(dataset, tmp_path / "sequential") as sequential, _run(
        dataset, tmp_path / "parallel", tuning={**example.make_tuning(tmp_path / "parallel/study"), "n_jobs": 2},
    ) as parallel:
        _assert_native_selection(parallel)
        assert [trial.to_dict() for trial in parallel.tuning_result.trials] == [trial.to_dict() for trial in sequential.tuning_result.trials]
        assert parallel.structural_tuning_evidence["trials"] == sequential.structural_tuning_evidence["trials"]
        assert parallel.tuning_best_params == sequential.tuning_best_params
        X_new = dataset.x({"partition": "test"}, layout="2d")
        left, right = sequential.export(tmp_path / "sequential.n4a"), parallel.export(tmp_path / "parallel.n4a")
        np.testing.assert_array_equal(nirs4all.predict(left, X_new, engine="dag-ml").y_pred,
                                      nirs4all.predict(right, X_new, engine="dag-ml").y_pred)
    assert len(children) == 8 and len({child._process.pid for child in children}) == 8
    assert all(child._closed and child._process.poll() is not None and not Path(child._private_dir.name).exists() for child in children)


def test_stop_resume_keeps_native_recipe_history_and_matches_continuous_search(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observer = _NativeObserver(monkeypatch)
    dataset = example.make_dataset()
    root = tmp_path / "resumed"
    tuning = example.make_tuning(root / "study")
    with pytest.raises(DagRunCancelled):
        _run(dataset, root, tuning={**tuning, "progress_callback": _stop_after(2)})
    initial = _checkpoint(root / "study")["native_checkpoint"]["trials"]
    assert len(initial) == 2 and all(record["state"] == "complete" for record in initial)
    before = len(observer.callbacks)
    with _run(dataset, root, tuning={**tuning, "resume": True}) as resumed, _run(dataset, tmp_path / "continuous") as continuous:
        assert _checkpoint(root / "study")["native_checkpoint"]["trials"][:2] == initial
        assert [trial.to_dict() for trial in resumed.tuning_result.trials] == [trial.to_dict() for trial in continuous.tuning_result.trials]
        assert resumed.structural_tuning_evidence["trials"] == continuous.structural_tuning_evidence["trials"]
        assert resumed.tuning_best_params == continuous.tuning_best_params
        resumed_callbacks = [call for call in observer.callbacks[before:] if call["search"] == 1]
        assert {call["candidate"] for call in resumed_callbacks} == set(range(2, 8)), "resume repeated terminal trials"
        X_new = dataset.x({"partition": "test"}, layout="2d")
        left, right = resumed.export(tmp_path / "resumed.n4a"), continuous.export(tmp_path / "continuous.n4a")
        np.testing.assert_array_equal(nirs4all.predict(left, X_new, engine="dag-ml").y_pred,
                                      nirs4all.predict(right, X_new, engine="dag-ml").y_pred)
    observer.assert_closed()


@pytest.mark.parametrize("mutation", ["unknown_recipe", "inactive_axis", "missing_active_axis"])
def test_invalid_native_proposal_refused_before_candidate_factory_or_fit(mutation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observer = _NativeObserver(monkeypatch)
    tuning = example.make_tuning(tmp_path / "study")
    with pytest.raises(DagRunCancelled):
        _run(example.make_dataset(), tmp_path, tuning={**tuning, "progress_callback": _stop_after(2)})
    saved = observer.searches[0]
    catalogue = saved["request"]["structural_catalogue"]
    recipe = next(entry for entry in catalogue["entries"] if "model.alpha" in entry["parameter_bindings"])
    proposal = {"__recipe__": recipe["recipe_id"], "model.alpha": 0.5}
    if mutation == "unknown_recipe":
        proposal["__recipe__"] = "recipe:unknown"
    elif mutation == "inactive_axis":
        proposal["model.n_components"] = 2
    else:
        proposal.pop("model.alpha")
    checkpoint_path = tmp_path / "study/structural-ridge-pls.n4mopt.json"
    checkpoint_bytes = checkpoint_path.read_bytes()
    counts = len(observer.callbacks), len(observer.entries)
    factory_calls: list[int] = []

    def optimizer(event: dict[str, Any]) -> Any:
        assert event["operation"] == "ask", "invalid proposal reached optimizer tell/fail"
        return deepcopy(proposal)

    def forbidden(task: Any) -> Any:
        pytest.fail("invalid structural proposal reached native operator callback")

    def factory(index: int) -> Any:
        factory_calls.append(index)
        pytest.fail("invalid structural proposal constructed a candidate provider")

    with pytest.raises(Exception, match="(?i)(recipe|structur|parameter|inactive|active|proposal|binding)"):
        dag_ml.run_host_hpo_search_in_process(
            saved["dsl"], saved["envelope"], saved["controller_manifests"], saved["request"],
            forbidden, optimizer, candidate_callback_factory=factory,
        )
    assert not factory_calls
    assert (len(observer.callbacks), len(observer.entries)) == counts
    assert checkpoint_path.read_bytes() == checkpoint_bytes
    observer.assert_closed()


def test_fresh_installed_winner_archive_replays_without_fit_or_search(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    installed_python = os.environ.get("NIRS4ALL_STRUCTURAL_HPO_INSTALLED_PYTHON")
    if not installed_python:
        if os.environ.get("NIRS4ALL_REQUIRE_STRUCTURAL_HPO_INSTALLED") == "1":
            pytest.fail("mandatory installed proof requires NIRS4ALL_STRUCTURAL_HPO_INSTALLED_PYTHON")
        pytest.skip("fresh installed Python is supplied by the qualification gate")
    observer = _NativeObserver(monkeypatch)
    dataset = example.make_dataset()
    training_root = tmp_path / "training"
    X_new = dataset.x({"partition": "test"}, layout="2d")
    with _run(dataset, training_root) as result:
        _assert_native_selection(result, observer)
        _, expected_prediction = _assert_fit_oracle(observer, dataset)
        evidence = deepcopy(result.structural_tuning_evidence)
        selected_graph = deepcopy(result.structural_tuning_training_request["graph"])
        archive = Path(result.export(tmp_path / "winner.n4a"))
    observer.assert_closed()
    shutil.rmtree(training_root)
    np.save(tmp_path / "X.npy", X_new)
    production_paths = [
        "api/run.py", "pipeline/dagml_bridge.py", "pipeline/dagml/structural_tuning.py", "pipeline/dagml/host_search_checkpoint.py",
        "pipeline/dagml/tuning_adapters.py", "pipeline/dagml/node_runner.py",
        "pipeline/dagml/host_hpo_candidate.py", "pipeline/dagml/host_hpo_candidate_worker.py",
        "pipeline/dagml/general_archive.py",
    ]
    package_root = Path(nirs4all.__file__).resolve().parent
    expected_sources = {name: hashlib.sha256((package_root / name).read_bytes()).hexdigest() for name in production_paths}
    extension = importlib.import_module("dag_ml._dag_ml")
    expectation = {
        "prediction": expected_prediction.tolist(), "evidence": evidence, "graph": selected_graph,
        "source_sha256": expected_sources,
        "dag_extension_sha256": hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    (tmp_path / "expected.json").write_text(json.dumps(expectation, allow_nan=False), encoding="utf-8")
    script = textwrap.dedent("""\
        import hashlib, importlib, json, pathlib, sys
        import dag_ml, numpy as np, nirs4all
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from n4m.model_selection.optimizer import Optimizer
        from nirs4all.pipeline.dagml.general_archive import load_general_archive
        from nirs4all.pipeline.dagml.host_search_checkpoint import HostSearchOptimizer
        root = pathlib.Path(nirs4all.__file__).resolve().parent
        assert 'site-packages' in root.parts, root
        expected = json.loads(pathlib.Path(sys.argv[3]).read_text())
        for name, digest in expected['source_sha256'].items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
        extension = importlib.import_module('dag_ml._dag_ml')
        assert hashlib.sha256(pathlib.Path(extension.__file__).read_bytes()).hexdigest() == expected['dag_extension_sha256']
        archive = pathlib.Path(sys.argv[1])
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected['archive_sha256']
        assert callable(getattr(Optimizer, 'configuration_matches', None)), 'installed Methods binding lacks native contract attestation'
        def forbidden(*args, **kwargs):
            raise AssertionError('archive replay reached FIT/HPO')
        for cls in (Ridge, PLSRegression, StandardScaler):
            cls.fit = forbidden
        Optimizer.__init__ = forbidden
        Optimizer.load = classmethod(forbidden)
        HostSearchOptimizer.__init__ = forbidden
        nirs4all.run = forbidden
        nirs4all.run_host_hpo_search = forbidden
        nirs4all.execute_training = forbidden
        dag_ml.run_host_hpo_search_in_process = forbidden
        dag_ml.execute_training = forbidden
        dag_ml.prepare_host_hpo_structural_catalogue = forbidden
        dag_ml.resolve_host_hpo_structural_winner = forbidden
        captured = load_general_archive(archive)['artifact']['estimator']
        while hasattr(captured, 'estimator'):
            captured = captured.estimator
        assert captured.structural_tuning_evidence == expected['evidence']
        fitted_steps = [operator for _, operator in captured.steps] if hasattr(captured, 'steps') else [captured]
        expected_scalers = sum(node['kind'] == 'transform' for node in expected['graph']['nodes'])
        assert sum(type(operator) is StandardScaler for operator in fitted_steps) == expected_scalers
        model = fitted_steps[-1]
        chosen = expected['evidence']['selected_params']
        assert type(model) is (Ridge if 'model.alpha' in chosen else PLSRegression)
        for path, value in chosen.items():
            if path != '__recipe__':
                assert model.get_params(deep=False)[path.removeprefix('model.')] == value
        X = np.load(sys.argv[2])
        public = nirs4all.predict(archive, X, engine='dag-ml')
        with nirs4all.load_session(archive) as session:
            replayed = session.predict(X)
        for result in (public, replayed):
            np.testing.assert_allclose(result.y_pred.ravel(), np.asarray(expected['prediction']), rtol=2e-6, atol=2e-6)
            assert result.metadata['training_performed'] is False
            assert result.metadata['phase'] == 'PREDICT'
            assert result.metadata['artifact_integrity_verified'] is True
        np.testing.assert_array_equal(public.y_pred, replayed.y_pred)
        print(json.dumps({'installed_package': str(root), 'source_sha256': expected['source_sha256'],
                          'evidence': captured.structural_tuning_evidence, 'fit_hpo_calls': 0,
                          'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}))
    """)
    process = subprocess.run(
        [installed_python, "-I", "-c", script, str(archive), str(tmp_path / "X.npy"), str(tmp_path / "expected.json")],
        cwd=tmp_path, capture_output=True, text=True, check=False, timeout=180,
    )
    assert process.returncode == 0, process.stderr[-6000:]
    receipt = json.loads(process.stdout.splitlines()[-1])
    assert receipt["fit_hpo_calls"] == 0 and receipt["evidence"] == evidence
    assert receipt["source_sha256"] == expected_sources
    assert receipt["archive_sha256"] == expectation["archive_sha256"]
    assert not training_root.exists()
