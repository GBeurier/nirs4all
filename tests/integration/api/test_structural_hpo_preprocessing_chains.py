"""Native structural chain search with independent fold oracle and fitted replay."""

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
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.operators.transforms import SNV, SavitzkyGolay
from nirs4all.pipeline.dagml import node_runner
from nirs4all.pipeline.dagml.cancellation import DagRunCancelled
from nirs4all.pipeline.dagml.structural_tuning import _prepare_structure
from tests.integration.api import test_structural_hpo_ridge_pls as baseline

_EXAMPLE = Path(__file__).resolve().parents[3] / "examples/user/04_models/U18_structural_hpo_preprocessing_chains.py"
_SPEC = importlib.util.spec_from_file_location("structural_hpo_chains_example", _EXAMPLE)
assert _SPEC is not None and _SPEC.loader is not None
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)
_MODELS = {"Ridge": Ridge, "PLSRegression": PLSRegression}


@pytest.fixture(autouse=True)
def native_execution_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *a, **k: pytest.fail("legacy scheduler executed"))
    monkeypatch.setattr("nirs4all.pipeline.dagml.run_paths._run_model_on_precomputed_matrix", lambda *a, **k: pytest.fail("host CV scheduler executed"))


def _run(dataset: Any, root: Path, *, tuning: dict[str, Any] | None = None, pipeline: list[Any] | None = None) -> Any:
    return baseline._run(dataset, root, tuning=example.make_tuning(root / "study") if tuning is None else tuning,
                         pipeline=example.make_pipeline() if pipeline is None else pipeline)


def _operator_name(node: dict[str, Any]) -> str:
    operator = node["operator"]
    return (operator["class"] if isinstance(operator, dict) else operator).rsplit(".", 1)[-1]


def _transforms(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    return [node for node in recipe["graph"]["nodes"] if node["kind"] == "transform"]


def _native_catalogue(pipeline: list[Any], dataset: Any, tuning: dict[str, Any]) -> dict[str, Any]:
    prepared = _prepare_structure(pipeline, dataset, tuning, {"random_state": 17})
    catalogue = prepared["catalogue"]
    assert catalogue == dag_ml.prepare_host_hpo_structural_catalogue(
        prepared["dsl"], prepared["envelope"], prepared["manifests"],
        {"model.alpha": "alpha", "model.n_components": "n_components"}, selector_path="__recipe__",
    )
    return catalogue


def _enqueue_recipes(monkeypatch: pytest.MonkeyPatch, catalogue: dict[str, Any], recipes: list[dict[str, Any]]) -> None:
    """Queue genuine native selector values; leave numeric proposals native-owned."""
    from n4m.model_selection.optimizer import Optimizer

    original = Optimizer.ask
    ids = [entry["recipe_id"] for entry in catalogue["entries"]]
    selected = iter(recipes)

    def ask(optimizer: Any) -> Any:
        recipe = next(selected, None)
        if recipe is not None:
            optimizer.enqueue({catalogue["selector_path"]: ids.index(recipe["recipe_id"])})
        return original(optimizer)

    monkeypatch.setattr(Optimizer, "ask", ask)


class _ChainObserver(baseline._NativeObserver):
    """Extend the shared real-native observer for stateless NIRS transform FITs."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        super().__init__(monkeypatch)
        active: ContextVar[dict[str, Any] | None] = ContextVar("structural_chain_transform", default=None)
        original_node = node_runner._run_fitted_transform_node

        def node(task: dict[str, Any], resolver: Any, lookup: Any, store: Any, *args: Any, **kwargs: Any) -> Any:
            observed = next((call for call in reversed(self.callbacks) if call["task"] == task), None)
            context = {"task": deepcopy(task), "store": id(store), "search": None if observed is None else observed["search"],
                       "candidate": None if observed is None else observed["candidate"]}
            token = active.set(context)
            try:
                return original_node(task, resolver, lookup, store, *args, **kwargs)
            finally:
                active.reset(token)

        monkeypatch.setattr(node_runner, "_run_fitted_transform_node", node)
        for name, cls in (("StandardNormalVariate", SNV), ("SavitzkyGolay", SavitzkyGolay)):
            original_fit = cls.fit
            self.original_fits[name] = original_fit

            def fit(operator: Any, X: Any, y: Any = None, *args: Any, _name: str = name, _original: Any = original_fit, **kwargs: Any) -> Any:
                context = active.get()
                if context is None:
                    return _original(operator, X, y, *args, **kwargs)
                self.model_refs.append(operator)
                entry = {**context, "kind": _name, "model_id": id(operator), "params": deepcopy(operator.get_params(deep=False)),
                         "X": np.array(X, copy=True), "y": None if y is None else np.array(y, copy=True),
                         "fresh": not hasattr(operator, "n_features_in_")}
                self.entries.append(entry)
                result = _original(operator, X, y, *args, **kwargs)
                entry["fitted"] = True
                return result

            monkeypatch.setattr(cls, "fit", fit)


def _assert_selection(result: Any, catalogue: dict[str, Any], observer: _ChainObserver | None = None) -> None:
    assert result.structural_tuning_search_request["request"]["structural_catalogue"] == catalogue
    recipes = {entry["recipe_id"]: entry for entry in catalogue["entries"]}
    evidence = result.structural_tuning_evidence
    trials = evidence["trials"]
    assert len(result.tuning_result.trials) == len(trials)
    for public, native in zip(result.tuning_result.trials, trials, strict=True):
        recipe = recipes[native["params"]["__recipe__"]]
        assert public.state == "COMPLETE" and public.number == native["trial_index"]
        assert set(native["params"]) == {"__recipe__", *recipe["parameter_bindings"]}
        assert public.params == {path: value for path, value in native["params"].items() if path != "__recipe__"}
        assert public.value == native["score"]
        assert set(native["objective_fold_scores"]) == {"fold0", "fold1", "fold2"}
        assert native["score"] == pytest.approx(np.mean(list(native["objective_fold_scores"].values())), abs=1e-12)
        if observer is not None:
            actual = [entry for entry in observer.entries if entry["search"] == 0 and entry["candidate"] == native["trial_index"]]
            expected_nodes = {node["id"] for node in recipe["graph"]["nodes"] if node["kind"] in {"transform", "model"}}
            assert len(actual) == 3 * len(expected_nodes), "unexpected duplicate/missing native FIT"
            assert {(entry["task"]["node_plan"]["node_id"], entry["task"]["fold_id"]) for entry in actual} == {
                (node, fold) for node in expected_nodes for fold in ("fold0", "fold1", "fold2")
            }
            for entry in actual:
                assert entry["task"]["variant_id"] == native["variant_id"]
                if entry["kind"] in _MODELS:
                    expected = {binding["param_path"]: native["params"][path] for path, binding in recipe["parameter_bindings"].items()}
                    assert all(entry["params"][path] == value for path, value in expected.items())
                    inactive = "n_components" if entry["kind"] == "Ridge" else "alpha"
                    assert inactive not in entry["task"]["node_plan"].get("params", {})
            active = {name: value for index, name, value in observer.active_checks if index == native["trial_index"] and name.startswith("model.")}
            assert active == {"model.alpha": "model.alpha" in public.params, "model.n_components": "model.n_components" in public.params}
    winner = next(trial for trial in trials if trial["trial_index"] == evidence["selected_trial_index"])
    assert winner["score"] == min(trial["score"] for trial in trials)
    recipe = recipes[winner["params"]["__recipe__"]]
    graph = result.structural_tuning_training_request["graph"]
    assert result._dagml_graph == graph
    assert {node["id"] for node in graph["nodes"]} == {node["id"] for node in recipe["graph"]["nodes"]}
    assert graph.get("edges", []) == recipe["graph"].get("edges", [])
    for node in graph["nodes"]:
        declared = next(original for original in recipe["graph"]["nodes"] if original["id"] == node["id"])
        overrides = {binding["param_path"]: winner["params"][path] for path, binding in recipe["parameter_bindings"].items() if binding["node_id"] == node["id"]}
        assert node.get("params", {}) == {**declared.get("params", {}), **overrides}
        assert {key: value for key, value in node.items() if key != "params"} == {key: value for key, value in declared.items() if key != "params"}
    assert result.tuning_best_params == {path: value for path, value in winner["params"].items() if path != "__recipe__"}
    assert result.tuning_best_value == winner["score"] and len(result._dagml_refit_artifacts) == 1
    selection = result.structural_tuning_training_request["campaign"]["metadata"]["host_hpo_structural_selection"]
    assert selection["recipe_id"] == recipe["recipe_id"] and selection["selected_trial_index"] == winner["trial_index"]
    if observer is not None:
        refits = [entry for entry in observer.entries if entry["task"]["phase"] == "REFIT"]
        expected_ids = {node["id"] for node in recipe["graph"]["nodes"] if node["kind"] in {"transform", "model"}}
        assert len(refits) == len(expected_ids)
        assert {entry["task"]["node_plan"]["node_id"] for entry in refits} == expected_ids


def _oracle_transform(kind: str, params: dict[str, Any], train: np.ndarray, predict: np.ndarray, observer: _ChainObserver) -> tuple[np.ndarray, np.ndarray, Any]:
    if kind == "StandardScaler":
        independent = StandardScaler(**params)
        observer.original_fits[kind](independent, train)
        return independent.transform(train), independent.transform(predict), independent
    if kind == "StandardNormalVariate":
        def snv(values: np.ndarray) -> np.ndarray:
            values = values.copy()
            if params["with_mean"]:
                values = values - values.mean(axis=1, keepdims=True)
            if params["with_std"]:
                scale = values.std(axis=1, ddof=params["ddof"], keepdims=True)
                scale[scale == 0] = 1.0
                values = values / scale
            return values
        return snv(train), snv(predict), None
    assert kind == "SavitzkyGolay"
    arguments = {key: params[key] for key in ("window_length", "polyorder", "deriv", "delta")}
    return savgol_filter(train, **arguments), savgol_filter(predict, **arguments), None


def _assert_fit_oracle(observer: _ChainObserver, dataset: Any, catalogue: dict[str, Any], result: Any) -> np.ndarray:
    X, y, groups, rows = baseline._raw(dataset)
    folds = list(GroupKFold(3).split(X[:36], y[:36], groups[:36]))
    assert all(entry["fresh"] and entry["fitted"] for entry in observer.entries)
    assert len({entry["model_id"] for entry in observer.entries}) == len(observer.entries), "candidate/fold reused an operator"
    scopes = [(entry["search"], entry["candidate"], entry["task"]["phase"], entry["task"].get("fold_id"), entry["task"]["node_plan"]["node_id"])
              for entry in observer.entries]
    assert len(scopes) == len(set(scopes)), "native FIT was performed twice for one node/fold"
    recipes = {entry["recipe_id"]: entry for entry in catalogue["entries"]}
    trials = {trial["trial_index"]: trial for trial in result.structural_tuning_evidence["trials"]}
    selected = result.structural_tuning_evidence["selected_params"]["__recipe__"]
    scores: dict[int, dict[str, float]] = {}
    final: np.ndarray | None = None
    for model in (entry for entry in observer.entries if entry["kind"] in _MODELS):
        task = model["task"]
        refit = task["phase"] == "REFIT"
        train = np.array([rows[sample] for sample in baseline._view_ids(task, "full_train" if refit else "fold_train")])
        predict = np.arange(36, 48) if refit else np.array([rows[sample] for sample in baseline._view_ids(task, "fold_validation")])
        assert np.all(train < 36)
        if refit:
            assert model["search"] is None and set(train) == set(range(36))
        else:
            assert task["phase"] == "FIT_CV" and np.all(predict < 36)
            assert set(groups[train]).isdisjoint(groups[predict])
            assert any(set(train) == set(fit) and set(predict) == set(heldout) for fit, heldout in folds)
        recipe_id = trials[model["candidate"]]["params"]["__recipe__"] if model["search"] is not None else selected
        recipe = recipes[recipe_id]
        features, prediction_features = X[train], X[predict]
        for node in _transforms(recipe):
            fits = [entry for entry in observer.entries if entry["search"] == model["search"] and entry["candidate"] == model["candidate"]
                    and entry["store"] == model["store"] and entry["task"]["phase"] == task["phase"]
                    and entry["task"].get("fold_id") == task.get("fold_id") and entry["task"]["node_plan"]["node_id"] == node["id"]]
            assert len(fits) == 1
            actual = fits[0]
            assert actual["kind"] == _operator_name(node) and actual["params"] == node["params"]
            assert baseline._view_ids(actual["task"], "full_train" if refit else "fold_train") == baseline._view_ids(task, "full_train" if refit else "fold_train")
            np.testing.assert_allclose(actual["X"], features, rtol=2e-6, atol=2e-6)
            features, prediction_features, independent = _oracle_transform(actual["kind"], actual["params"], features, prediction_features, observer)
            if independent is not None:
                for name in ("mean_", "var_", "scale_"):
                    np.testing.assert_allclose(actual[name], getattr(independent, name), rtol=0, atol=0)
        np.testing.assert_allclose(model["X"], features, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(model["y"].reshape(-1), y[train])
        fitted = _MODELS[model["kind"]](**model["params"])
        observer.original_fits[model["kind"]](fitted, features, model["y"])
        np.testing.assert_allclose(model["coef_"], fitted.coef_, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(model["intercept_"], fitted.intercept_, rtol=2e-6, atol=2e-6)
        prediction = np.asarray(fitted.predict(prediction_features)).reshape(-1)
        if model["search"] is not None:
            assert not refit
            scores.setdefault(model["candidate"], {})[task["fold_id"]] = float(np.sqrt(np.mean((prediction - y[predict]) ** 2)))
        elif refit:
            assert final is None
            final = prediction
    assert set(scores) == set(trials)
    for trial_index, trial in trials.items():
        assert trial["objective_fold_scores"] == pytest.approx(scores[trial_index], abs=2e-6)
    assert final is not None
    return final


def test_all_six_native_recipes_have_ordered_identity_and_real_train_only_fits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = example.make_dataset()
    pipeline = example.make_pipeline()
    source_X = baseline._raw(dataset)[0].copy()
    source_params = [[operator.get_params(deep=False) for operator in branch] if isinstance(branch, list)
                     else None if branch is None else branch.get_params(deep=False) for branch in pipeline[0]["_or_"]]
    tuning = {**example.make_tuning(tmp_path / "study"), "n_trials": 6}
    catalogue = _native_catalogue(pipeline, dataset, tuning)
    assert len(catalogue["entries"]) == 6
    for field in ("recipe_id", "variant_label"):
        assert len({entry[field] for entry in catalogue["entries"]}) == 6
    assert len({entry["variant"]["variant_id"] for entry in catalogue["entries"]}) == 6
    signatures = {(tuple(_operator_name(node) for node in _transforms(entry)), _operator_name(next(node for node in entry["graph"]["nodes"] if node["id"] == entry["target_node"])))
                  for entry in catalogue["entries"]}
    assert signatures == {(chain, model) for chain in ((), ("StandardScaler",), ("StandardNormalVariate", "SavitzkyGolay")) for model in _MODELS}
    observer = _ChainObserver(monkeypatch)
    _enqueue_recipes(monkeypatch, catalogue, catalogue["entries"])
    with _run(dataset, tmp_path, tuning=tuning, pipeline=pipeline) as result:
        _assert_selection(result, catalogue, observer)
        assert {trial["params"]["__recipe__"] for trial in result.structural_tuning_evidence["trials"]} == {entry["recipe_id"] for entry in catalogue["entries"]}
        expected = _assert_fit_oracle(observer, dataset, catalogue, result)
        archive = result.export(tmp_path / "winner.n4a")
        actual = nirs4all.predict(archive, source_X[36:], engine="dag-ml")
        np.testing.assert_allclose(actual.y_pred.ravel(), expected, rtol=2e-6, atol=2e-6)
        assert actual.metadata["training_performed"] is False
    np.testing.assert_array_equal(baseline._raw(dataset)[0], source_X)
    after = [[operator.get_params(deep=False) for operator in branch] if isinstance(branch, list)
             else None if branch is None else branch.get_params(deep=False) for branch in pipeline[0]["_or_"]]
    assert after == source_params
    assert all(not hasattr(operator, "n_features_in_") for branch in pipeline[0]["_or_"] if branch is not None
               for operator in (branch if isinstance(branch, list) else [branch]))
    observer.assert_closed()


def test_four_operator_branch_and_repeated_savgol_match_independent_oracle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = example.make_dataset()
    pipeline = example.make_pipeline()
    pipeline[0]["_or_"] = [None, [StandardScaler(with_mean=False), SNV(ddof=1), SavitzkyGolay(window_length=3, polyorder=1), SavitzkyGolay(window_length=5, polyorder=2)]]
    tuning = {**example.make_tuning(tmp_path / "study"), "n_trials": 2}
    catalogue = _native_catalogue(pipeline, dataset, tuning)
    chained = [entry for entry in catalogue["entries"] if len(_transforms(entry)) == 4]
    assert len(chained) == 2
    observer = _ChainObserver(monkeypatch)
    _enqueue_recipes(monkeypatch, catalogue, chained)
    with _run(dataset, tmp_path, tuning=tuning, pipeline=pipeline) as result:
        _assert_selection(result, catalogue, observer)
        expected = _assert_fit_oracle(observer, dataset, catalogue, result)
        actual = nirs4all.predict(result.export(tmp_path / "long-chain.n4a"), baseline._raw(dataset)[0][36:], engine="dag-ml")
        np.testing.assert_allclose(actual.y_pred.ravel(), expected, rtol=2e-6, atol=2e-6)
    observer.assert_closed()


@pytest.mark.parametrize("mutation", ["chain_order", "constructor", "alternative_order", "repeat"])
def test_resume_rejects_changed_order_or_each_chain_constructor_before_callback(mutation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observer = _ChainObserver(monkeypatch)
    dataset = example.make_dataset()
    tuning = {**example.make_tuning(tmp_path / "study"), "n_trials": 6}
    with pytest.raises(DagRunCancelled):
        _run(dataset, tmp_path, tuning={**tuning, "progress_callback": baseline._stop_after(2)})
    path = tmp_path / "study/structural-preprocessing-chains.n4mopt.json"
    before = path.read_bytes()
    counts = len(observer.callbacks), len(observer.entries), len(observer.candidate_factories)
    pipeline = example.make_pipeline()
    chain = pipeline[0]["_or_"][2]
    if mutation == "chain_order":
        chain.reverse()
    elif mutation == "constructor":
        chain[1].set_params(polyorder=1)
    elif mutation == "alternative_order":
        pipeline[0]["_or_"].reverse()
    else:
        chain.append(SavitzkyGolay(window_length=3, polyorder=1))
    with pytest.raises(Exception, match="(?i)(checkpoint|fingerprint|contract|catalogue|structur)"):
        _run(dataset, tmp_path / "refused", tuning={**tuning, "resume": True}, pipeline=pipeline)
    assert (len(observer.callbacks), len(observer.entries), len(observer.candidate_factories)) == counts
    assert path.read_bytes() == before
    observer.assert_closed()


def test_chain_stop_resume_matches_native_continuous_history_without_repeating_trials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observer = _ChainObserver(monkeypatch)
    dataset = example.make_dataset()
    root = tmp_path / "resumed"
    tuning = {**example.make_tuning(root / "study"), "n_trials": 6}
    with pytest.raises(DagRunCancelled):
        _run(dataset, root, tuning={**tuning, "progress_callback": baseline._stop_after(2)})
    initial = json.loads((root / "study/structural-preprocessing-chains.n4mopt.json").read_text())["native_checkpoint"]["trials"]
    before = len(observer.callbacks)
    with _run(dataset, root, tuning={**tuning, "resume": True}) as resumed, _run(
        dataset, tmp_path / "continuous", tuning={**example.make_tuning(tmp_path / "continuous/study"), "n_trials": 6},
    ) as continuous:
        assert [trial.to_dict() for trial in resumed.tuning_result.trials] == [trial.to_dict() for trial in continuous.tuning_result.trials]
        assert resumed.structural_tuning_evidence["trials"] == continuous.structural_tuning_evidence["trials"]
        saved = json.loads((root / "study/structural-preprocessing-chains.n4mopt.json").read_text())["native_checkpoint"]["trials"]
        assert saved[:2] == initial
        resumed_calls = [call for call in observer.callbacks[before:] if call["search"] == 1]
        assert {call["candidate"] for call in resumed_calls} == set(range(2, 6))
        X_new = baseline._raw(dataset)[0][36:]
        np.testing.assert_array_equal(nirs4all.predict(resumed.export(tmp_path / "resumed.n4a"), X_new, engine="dag-ml").y_pred,
                                      nirs4all.predict(continuous.export(tmp_path / "continuous.n4a"), X_new, engine="dag-ml").y_pred)
    observer.assert_closed()


def test_chain_parallel_candidate_workers_match_sequential_and_release_resources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from nirs4all.pipeline.dagml.host_hpo_candidate import HostHpoCandidate

    children: list[Any] = []
    original = HostHpoCandidate.__init__

    def observe(child: Any, *args: Any, **kwargs: Any) -> None:
        original(child, *args, **kwargs)
        children.append(child)

    monkeypatch.setattr(HostHpoCandidate, "__init__", observe)
    dataset = example.make_dataset()
    with _run(dataset, tmp_path / "sequential", tuning={**example.make_tuning(tmp_path / "sequential/study"), "n_trials": 6}) as sequential, _run(
        dataset, tmp_path / "parallel", tuning={**example.make_tuning(tmp_path / "parallel/study"), "n_trials": 6, "n_jobs": 2},
    ) as parallel:
        assert parallel.structural_tuning_evidence["trials"] == sequential.structural_tuning_evidence["trials"]
        _assert_selection(parallel, _native_catalogue(example.make_pipeline(), dataset, example.make_tuning(tmp_path / "catalogue")))
        X_new = baseline._raw(dataset)[0][36:]
        np.testing.assert_array_equal(nirs4all.predict(sequential.export(tmp_path / "sequential.n4a"), X_new, engine="dag-ml").y_pred,
                                      nirs4all.predict(parallel.export(tmp_path / "parallel.n4a"), X_new, engine="dag-ml").y_pred)
    assert len(children) == 6 and len({child._process.pid for child in children}) == 6
    assert all(child._closed and child._process.poll() is not None and not Path(child._private_dir.name).exists() for child in children)


def test_fresh_installed_chain_winner_replays_without_fit_hpo_or_training_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    installed_python = os.environ.get("NIRS4ALL_STRUCTURAL_HPO_INSTALLED_PYTHON")
    if not installed_python:
        if os.environ.get("NIRS4ALL_REQUIRE_STRUCTURAL_HPO_INSTALLED") == "1":
            pytest.fail("mandatory installed proof requires NIRS4ALL_STRUCTURAL_HPO_INSTALLED_PYTHON")
        pytest.skip("fresh installed Python is supplied by the qualification gate")
    dataset = example.make_dataset()
    pipeline = example.make_pipeline()
    tuning = {**example.make_tuning(tmp_path / "study"), "n_trials": 2}
    catalogue = _native_catalogue(pipeline, dataset, tuning)
    chosen = next(entry for entry in catalogue["entries"] if len(_transforms(entry)) == 2 and "model.alpha" in entry["parameter_bindings"])
    observer = _ChainObserver(monkeypatch)
    _enqueue_recipes(monkeypatch, catalogue, [chosen, chosen])
    root = tmp_path / "training"
    X_new = baseline._raw(dataset)[0][36:]
    with _run(dataset, root, tuning=tuning, pipeline=pipeline) as result:
        _assert_selection(result, catalogue, observer)
        expected_prediction = _assert_fit_oracle(observer, dataset, catalogue, result)
        archive = Path(result.export(tmp_path / "chain-winner.n4a"))
        evidence = deepcopy(result.structural_tuning_evidence)
        graph = deepcopy(result.structural_tuning_training_request["graph"])
    observer.assert_closed()
    shutil.rmtree(root)
    shutil.rmtree(tmp_path / "study")
    np.save(tmp_path / "X.npy", X_new)
    source_names = ["pipeline/dagml_bridge.py", "pipeline/dagml/structural_tuning.py", "pipeline/dagml/node_runner.py",
                    "pipeline/dagml/general_archive.py", "operators/transforms/scalers.py", "operators/transforms/nirs.py"]
    package = Path(nirs4all.__file__).resolve().parent
    extension = importlib.import_module("dag_ml._dag_ml")
    expected = {"prediction": expected_prediction.tolist(), "evidence": evidence, "graph": graph,
                "source_sha256": {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in source_names},
                "dag_extension_sha256": hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
    (tmp_path / "expected.json").write_text(json.dumps(expected, allow_nan=False), encoding="utf-8")
    script = textwrap.dedent("""\
        import hashlib, importlib, json, pathlib, sys
        import dag_ml, numpy as np, nirs4all
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from n4m.model_selection.optimizer import Optimizer
        from nirs4all.operators.transforms import SNV, SavitzkyGolay
        from nirs4all.pipeline.dagml.general_archive import load_general_archive
        from nirs4all.pipeline.dagml.host_search_checkpoint import HostSearchOptimizer
        expected = json.loads(pathlib.Path(sys.argv[3]).read_text())
        root = pathlib.Path(nirs4all.__file__).resolve().parent
        assert 'site-packages' in root.parts, root
        for name, digest in expected['source_sha256'].items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
        extension = importlib.import_module('dag_ml._dag_ml')
        assert hashlib.sha256(pathlib.Path(extension.__file__).read_bytes()).hexdigest() == expected['dag_extension_sha256']
        archive = pathlib.Path(sys.argv[1])
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected['archive_sha256']
        def forbidden(*args, **kwargs):
            raise AssertionError('chain archive replay reached FIT/HPO')
        for cls in (Ridge, PLSRegression, StandardScaler, SNV, SavitzkyGolay):
            cls.fit = forbidden
        Optimizer.__init__ = forbidden
        Optimizer.load = classmethod(forbidden)
        HostSearchOptimizer.__init__ = forbidden
        for name in ('run', 'run_host_hpo_search', 'execute_training'):
            setattr(nirs4all, name, forbidden)
        for name in ('run_host_hpo_search_in_process', 'execute_training', 'prepare_host_hpo_structural_catalogue', 'resolve_host_hpo_structural_winner'):
            setattr(dag_ml, name, forbidden)
        captured = load_general_archive(archive)['artifact']['estimator']
        while hasattr(captured, 'estimator'):
            captured = captured.estimator
        assert captured.structural_tuning_evidence == expected['evidence']
        def operators(value):
            if hasattr(value, 'transformer'):
                return operators(value.transformer)
            if hasattr(value, 'steps'):
                return [operator for step in value.steps for operator in operators(step[1] if isinstance(step, tuple) else step)]
            return [value]
        fitted = operators(captured)
        assert [type(operator).__name__ for operator in fitted] == ['StandardNormalVariate', 'SavitzkyGolay', 'Ridge']
        transforms = [node for node in expected['graph']['nodes'] if node['kind'] == 'transform']
        for operator, node in zip(fitted[:-1], transforms, strict=True):
            assert operator.get_params(deep=False) == node['params']
        assert fitted[-1].alpha == expected['evidence']['selected_params']['model.alpha']
        X = np.load(sys.argv[2])
        public = nirs4all.predict(archive, X, engine='dag-ml')
        with nirs4all.load_session(archive) as session:
            replay = session.predict(X)
        for result in (public, replay):
            np.testing.assert_allclose(result.y_pred.ravel(), expected['prediction'], rtol=2e-6, atol=2e-6)
            assert result.metadata['training_performed'] is False
            assert result.metadata['phase'] == 'PREDICT'
            assert result.metadata['artifact_integrity_verified'] is True
        np.testing.assert_array_equal(public.y_pred, replay.y_pred)
        print(json.dumps({'source_sha256': expected['source_sha256'], 'archive_sha256': expected['archive_sha256'], 'fit_hpo_calls': 0,
                          'evidence': captured.structural_tuning_evidence, 'chain': [type(operator).__name__ for operator in fitted]}))
    """)
    process = subprocess.run([installed_python, "-I", "-c", script, str(archive), str(tmp_path / "X.npy"), str(tmp_path / "expected.json")],
                             cwd=tmp_path, capture_output=True, text=True, check=False, timeout=180)
    assert process.returncode == 0, process.stderr[-6000:]
    receipt = json.loads(process.stdout.splitlines()[-1])
    assert receipt["fit_hpo_calls"] == 0 and receipt["evidence"] == evidence
    assert receipt["source_sha256"] == expected["source_sha256"] and receipt["archive_sha256"] == expected["archive_sha256"]
    assert receipt["chain"] == ["StandardNormalVariate", "SavitzkyGolay", "Ridge"]
    assert not root.exists() and not (tmp_path / "study").exists()
