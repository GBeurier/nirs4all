"""Native source-subset HPO, independent fold oracle and fitted archive replay."""

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
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.data import SpectroDataset
from nirs4all.operators.transforms import SNV, SavitzkyGolay
from nirs4all.pipeline.dagml import node_runner
from nirs4all.pipeline.dagml.cancellation import DagRunCancelled
from nirs4all.pipeline.dagml.structural_tuning import _prepare_structure
from tests.integration.api import test_structural_hpo_preprocessing_chains as chains
from tests.integration.api import test_structural_hpo_ridge_pls as baseline

_EXAMPLE = Path(__file__).resolve().parents[3] / "examples/user/04_models/U19_structural_hpo_source_subsets.py"
_SPEC = importlib.util.spec_from_file_location("structural_hpo_sources_example", _EXAMPLE)
assert _SPEC is not None and _SPEC.loader is not None
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


@pytest.fixture(autouse=True)
def native_execution_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *a, **k: pytest.fail("legacy scheduler executed"))
    monkeypatch.setattr("nirs4all.pipeline.dagml.run_paths._run_model_on_precomputed_matrix", lambda *a, **k: pytest.fail("host CV scheduler executed"))


def _run(dataset: Any, root: Path, *, tuning: dict[str, Any] | None = None, pipeline: list[Any] | None = None) -> Any:
    return baseline._run(dataset, root, tuning=example.make_tuning(root / "study") if tuning is None else tuning,
                         pipeline=example.make_pipeline() if pipeline is None else pipeline)


def _native_catalogue(pipeline: list[Any], dataset: Any, tuning: dict[str, Any]) -> dict[str, Any]:
    prepared = _prepare_structure(pipeline, dataset, tuning, {"random_state": 17})
    catalogue = prepared["catalogue"]
    assert catalogue == dag_ml.prepare_host_hpo_structural_catalogue(
        prepared["dsl"], prepared["envelope"], prepared["manifests"],
        {"model.alpha": "alpha", "model.n_components": "n_components"}, selector_path="__recipe__",
    )
    return catalogue


def _selection(recipe: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    nodes = [node for node in recipe["graph"]["nodes"] if "nirs4all_structural_source_selection" in node.get("metadata", {})]
    assert len(nodes) == 1
    node = nodes[0]
    selection = node["metadata"]["nirs4all_structural_source_selection"]
    assert selection["schema"] == "nirs4all.structural-source-selection.v1"
    assert selection["input_source_widths"] == [6, 3, 4]
    assert selection["input_width"] == 13
    assert selection["source_order"] == [f"source_{index}" for index in selection["source_indices"]]
    return node, selection


class _SourceObserver(chains._ChainObserver):
    """Record the actual ColumnTransformer FIT alongside shared native observers."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        super().__init__(monkeypatch)
        active: ContextVar[dict[str, Any] | None] = ContextVar("structural_source_projection", default=None)
        original_node = node_runner._run_fitted_transform_node
        original_fit = ColumnTransformer.fit
        self.original_fits["ColumnTransformer"] = original_fit

        def node(task: dict[str, Any], resolver: Any, lookup: Any, store: Any, *args: Any, **kwargs: Any) -> Any:
            observed = next((call for call in reversed(self.callbacks) if call["task"] == task), None)
            context = {"task": deepcopy(task), "store": id(store), "search": None if observed is None else observed["search"],
                       "candidate": None if observed is None else observed["candidate"]}
            token = active.set(context)
            try:
                return original_node(task, resolver, lookup, store, *args, **kwargs)
            finally:
                active.reset(token)

        def fit(operator: Any, X: Any, y: Any = None, *args: Any, **kwargs: Any) -> Any:
            context = active.get()
            if context is None:
                return original_fit(operator, X, y, *args, **kwargs)
            self.model_refs.append(operator)
            entry = {**context, "kind": "ColumnTransformer", "model_id": id(operator),
                     "params": json.loads(json.dumps(operator.get_params(deep=False), allow_nan=False)),
                     "X": np.array(X, copy=True), "y": None if y is None else np.array(y, copy=True),
                     "fresh": not hasattr(operator, "n_features_in_")}
            self.entries.append(entry)
            result = original_fit(operator, X, y, *args, **kwargs)
            entry["fitted"] = True
            entry["n_features_in_"] = operator.n_features_in_
            return result

        monkeypatch.setattr(node_runner, "_run_fitted_transform_node", node)
        monkeypatch.setattr(ColumnTransformer, "fit", fit)


def _blocks(dataset: Any) -> list[np.ndarray]:
    return [np.asarray(block) for block in dataset.x({}, layout="2d", concat_source=False)]


def _source_oracle_transform(kind: str, params: dict[str, Any], train: np.ndarray, predict: np.ndarray, observer: _SourceObserver) -> tuple[np.ndarray, np.ndarray, Any]:
    """Preserve the projected array layout in independent row-wise SNV math."""
    if kind != "StandardNormalVariate":
        return chains._oracle_transform(kind, params, train, predict, observer)

    def snv(values: np.ndarray) -> np.ndarray:
        values = values.copy(order="K")
        if params["with_mean"]:
            values = values - values.mean(axis=1, keepdims=True)
        if params["with_std"]:
            scale = values.std(axis=1, ddof=params["ddof"], keepdims=True)
            scale[scale == 0] = 1.0
            values = values / scale
        return values

    return snv(train), snv(predict), None


def _assert_fit_oracle(observer: _SourceObserver, dataset: Any, catalogue: dict[str, Any], result: Any) -> np.ndarray:
    """Fit an independent ordered hstack/chain/sklearn oracle on native fit rows."""
    X, y, groups, rows = baseline._raw(dataset)
    blocks = _blocks(dataset)
    folds = list(GroupKFold(3).split(X[:36], y[:36], groups[:36]))
    assert all(entry["fresh"] and entry["fitted"] for entry in observer.entries)
    assert len({entry["model_id"] for entry in observer.entries}) == len(observer.entries)
    scopes = [(entry["search"], entry["candidate"], entry["task"]["phase"], entry["task"].get("fold_id"), entry["task"]["node_plan"]["node_id"])
              for entry in observer.entries]
    assert len(scopes) == len(set(scopes)), "native FIT was repeated for one node/fold"
    recipes = {entry["recipe_id"]: entry for entry in catalogue["entries"]}
    trials = {trial["trial_index"]: trial for trial in result.structural_tuning_evidence["trials"]}
    selected = result.structural_tuning_evidence["selected_params"]["__recipe__"]
    scores: dict[int, dict[str, float]] = {}
    final: np.ndarray | None = None
    for model in (entry for entry in observer.entries if entry["kind"] in chains._MODELS):
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
        selector, selection = _selection(recipe)
        indices = selection["source_indices"]
        # Integer column projection produces a Fortran-contiguous buffer. Keep
        # that layout after the independent block concatenation: float32 row
        # reductions and sklearn centering depend on the accumulation order.
        features = np.asfortranarray(np.hstack([blocks[index][train] for index in indices]))
        prediction_features = np.asfortranarray(np.hstack([blocks[index][predict] for index in indices]))
        assert features.shape[1] == selection["selected_width"]
        columns = [column for index in indices for column in range(sum([6, 3, 4][:index]), sum([6, 3, 4][:index + 1]))]
        assert selection["columns"] == columns
        np.testing.assert_array_equal(features, X[train][:, columns])
        for node in chains._transforms(recipe):
            fits = [entry for entry in observer.entries if entry["search"] == model["search"] and entry["candidate"] == model["candidate"]
                    and entry["store"] == model["store"] and entry["task"]["phase"] == task["phase"]
                    and entry["task"].get("fold_id") == task.get("fold_id") and entry["task"]["node_plan"]["node_id"] == node["id"]]
            assert len(fits) == 1
            actual = fits[0]
            assert actual["kind"] == chains._operator_name(node) and actual["params"] == node["params"]
            assert baseline._view_ids(actual["task"], "full_train" if refit else "fold_train") == baseline._view_ids(task, "full_train" if refit else "fold_train")
            if node["id"] == selector["id"]:
                np.testing.assert_array_equal(actual["X"], X[train])
                assert actual["n_features_in_"] == 13
                assert actual["params"]["transformers"] == [["selected", "passthrough", columns]]
                assert actual["params"]["remainder"] == "drop"
                continue
            np.testing.assert_allclose(actual["X"], features, rtol=2e-6, atol=2e-6)
            assert actual["X"].flags.c_contiguous == features.flags.c_contiguous
            assert actual["X"].flags.f_contiguous == features.flags.f_contiguous
            features, prediction_features, independent = _source_oracle_transform(actual["kind"], actual["params"], features, prediction_features, observer)
            if independent is not None:
                for name in ("mean_", "var_", "scale_"):
                    np.testing.assert_allclose(actual[name], getattr(independent, name), rtol=0, atol=0)
        np.testing.assert_allclose(model["X"], features, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(model["y"].reshape(-1), y[train])
        fitted = chains._MODELS[model["kind"]](**model["params"])
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


class _NamedDenseDataset(SpectroDataset):
    """Exercise the existing source-name hook without typed multimodal inputs."""

    def __init__(self, name: str, source_names: list[str]) -> None:
        super().__init__(name)
        self._source_names = tuple(source_names)

    def source_name(self, index: int) -> str:
        return self._source_names[index]


def _prediction_dataset(blocks: list[np.ndarray]) -> SpectroDataset:
    dataset = SpectroDataset("new-source-subset-observations")
    dataset.add_samples(blocks, {"partition": "test"})
    return dataset


def test_every_source_recipe_has_native_identity_and_train_only_ordered_projection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = example.make_dataset()
    pipeline = example.make_pipeline()
    original_blocks = [block.copy() for block in _blocks(dataset)]
    original_source_choices = deepcopy(pipeline[0])
    original_scaler_params = pipeline[1]["_or_"][1].get_params(deep=False)
    tuning = {**example.make_tuning(tmp_path / "study"), "n_trials": 12}
    catalogue = _native_catalogue(pipeline, dataset, tuning)
    assert len(catalogue["entries"]) == 12
    for field in ("recipe_id", "variant_label"):
        assert len({entry[field] for entry in catalogue["entries"]}) == 12
    assert len({entry["variant"]["variant_id"] for entry in catalogue["entries"]}) == 12
    signatures = set()
    for recipe in catalogue["entries"]:
        selector, selection = _selection(recipe)
        transforms = tuple(chains._operator_name(node) for node in chains._transforms(recipe) if node["id"] != selector["id"])
        model = chains._operator_name(next(node for node in recipe["graph"]["nodes"] if node["id"] == recipe["target_node"]))
        signatures.add((tuple(selection["source_indices"]), transforms, model))
        assert all(node.get("metadata", {}).get("nirs4all_structural_dense_concat") is True for node in recipe["graph"]["nodes"])
    assert signatures == {(sources, transform, model) for sources in ((0,), (0, 2), (2, 0)) for transform in ((), ("StandardScaler",)) for model in chains._MODELS}
    observer = _SourceObserver(monkeypatch)
    chains._enqueue_recipes(monkeypatch, catalogue, catalogue["entries"])
    with _run(dataset, tmp_path, tuning=tuning, pipeline=pipeline) as result:
        chains._assert_selection(result, catalogue, observer)
        assert {trial["params"]["__recipe__"] for trial in result.structural_tuning_evidence["trials"]} == {entry["recipe_id"] for entry in catalogue["entries"]}
        expected = _assert_fit_oracle(observer, dataset, catalogue, result)
        archive = result.export(tmp_path / "winner.n4a")
        for values in (np.hstack([block[36:] for block in original_blocks]), _prediction_dataset([block[36:] for block in original_blocks])):
            actual = nirs4all.predict(archive, values, engine="dag-ml")
            np.testing.assert_allclose(actual.y_pred.ravel(), expected, rtol=2e-6, atol=2e-6)
            assert actual.metadata["training_performed"] is False
    for actual_block, original_block in zip(_blocks(dataset), original_blocks, strict=True):
        np.testing.assert_array_equal(actual_block, original_block)
    assert pipeline[0] == original_source_choices
    assert pipeline[1]["_or_"][1].get_params(deep=False) == original_scaler_params
    assert not hasattr(pipeline[1]["_or_"][1], "n_features_in_")
    observer.assert_closed()


def test_selected_source_order_precedes_the_whole_nirs_preprocessing_chain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = example.make_dataset()
    pipeline = example.make_pipeline()
    pipeline[1]["_or_"] = [None, [StandardScaler(with_mean=False), SNV(ddof=1), SavitzkyGolay(window_length=5, polyorder=2)]]
    tuning = {**example.make_tuning(tmp_path / "study"), "n_trials": 2}
    catalogue = _native_catalogue(pipeline, dataset, tuning)
    selected = [entry for entry in catalogue["entries"] if _selection(entry)[1]["source_indices"] == [2, 0] and len(chains._transforms(entry)) == 4]
    assert len(selected) == 2
    observer = _SourceObserver(monkeypatch)
    chains._enqueue_recipes(monkeypatch, catalogue, selected)
    with _run(dataset, tmp_path, tuning=tuning, pipeline=pipeline) as result:
        chains._assert_selection(result, catalogue, observer)
        expected = _assert_fit_oracle(observer, dataset, catalogue, result)
        prediction = nirs4all.predict(result.export(tmp_path / "source-chain.n4a"), baseline._raw(dataset)[0][36:], engine="dag-ml")
        np.testing.assert_allclose(prediction.y_pred.ravel(), expected, rtol=2e-6, atol=2e-6)
    observer.assert_closed()


def _rebuild_dataset(dataset: Any, *, blocks: list[np.ndarray] | None = None, y: np.ndarray | None = None) -> SpectroDataset:
    rebuilt = SpectroDataset(dataset.name)
    values = _blocks(dataset) if blocks is None else blocks
    rebuilt.add_samples([block[:36] for block in values], {"partition": "train"})
    rebuilt.add_samples([block[36:] for block in values], {"partition": "test"})
    rebuilt.add_targets(np.asarray(dataset.y({})).reshape(-1) if y is None else y)
    rebuilt.add_metadata(np.asarray(dataset.metadata_column("batch"))[:, None], headers=["batch"])
    return rebuilt


@pytest.mark.parametrize("mutation", ["source_order", "choice_order", "selected_subset", "source_widths", "unused_training_source", "unused_test_source", "preprocessing_constructor"])
def test_resume_refuses_changed_source_catalogue_layout_or_signed_data_without_callbacks(
    mutation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    observer = _SourceObserver(monkeypatch)
    dataset = example.make_dataset()
    tuning = {**example.make_tuning(tmp_path / "study"), "n_trials": 4}
    with pytest.raises(DagRunCancelled):
        _run(dataset, tmp_path, tuning={**tuning, "progress_callback": baseline._stop_after(1)})
    checkpoint = tmp_path / "study/structural-source-subsets.n4mopt.json"
    before = checkpoint.read_bytes()
    counts = len(observer.callbacks), len(observer.entries), len(observer.candidate_factories)
    pipeline = example.make_pipeline()
    if mutation == "source_order":
        pipeline[0]["_or_"][1]["merge"]["sources"]["sources"].reverse()
    elif mutation == "choice_order":
        pipeline[0]["_or_"].reverse()
    elif mutation == "selected_subset":
        pipeline[0]["_or_"][0]["merge"]["sources"]["sources"] = [2]
    elif mutation == "preprocessing_constructor":
        pipeline[1]["_or_"][1].set_params(with_mean=False)
    else:
        blocks = [block.copy() for block in _blocks(dataset)]
        if mutation == "source_widths":
            blocks = [np.hstack([blocks[0], blocks[1][:, :1]]), blocks[1][:, 1:], blocks[2]]
        else:
            rows = slice(0, 36) if mutation == "unused_training_source" else slice(36, 48)
            blocks[1][rows] += 100
        dataset = _rebuild_dataset(dataset, blocks=blocks)
    with pytest.raises(Exception, match="(?i)(checkpoint|fingerprint|contract|catalogue|structur)"):
        _run(dataset, tmp_path / "refused", tuning={**tuning, "resume": True}, pipeline=pipeline)
    assert (len(observer.callbacks), len(observer.entries), len(observer.candidate_factories)) == counts
    assert checkpoint.read_bytes() == before
    observer.assert_closed()


def test_unused_source_and_external_targets_do_not_change_any_candidate_score(tmp_path: Path) -> None:
    dataset = example.make_dataset()
    blocks = [block.copy() for block in _blocks(dataset)]
    blocks[1] = blocks[1] * -12 + 300
    targets = np.asarray(dataset.y({})).reshape(-1).copy()
    targets[36:] += 1000
    changed = _rebuild_dataset(dataset, blocks=blocks, y=targets)
    tuning = {**example.make_tuning(tmp_path / "original/study"), "n_trials": 4}
    with _run(dataset, tmp_path / "original", tuning=tuning) as original, _run(
        changed, tmp_path / "changed", tuning={**tuning, "storage": (tmp_path / "changed/study").resolve().as_uri()},
    ) as other:
        assert [trial.to_dict() for trial in original.tuning_result.trials] == [trial.to_dict() for trial in other.tuning_result.trials]
        assert original.tuning_best_params == other.tuning_best_params
        selected_signatures = []
        for result in (original, other):
            selector = result.structural_tuning_evidence["selected_params"]["__recipe__"]
            recipe = next(entry for entry in result.structural_tuning_search_request["request"]["structural_catalogue"]["entries"] if entry["recipe_id"] == selector)
            source_node, selection = _selection(recipe)
            selected_signatures.append((selection["source_indices"], selection["columns"],
                                        [(chains._operator_name(node), node["params"]) for node in recipe["graph"]["nodes"] if node["id"] != source_node["id"]]))
        assert selected_signatures[0] == selected_signatures[1]
        X = baseline._raw(dataset)[0][36:]
        changed_X = baseline._raw(changed)[0][36:]
        np.testing.assert_array_equal(nirs4all.predict(original.export(tmp_path / "original.n4a"), X, engine="dag-ml").y_pred,
                                      nirs4all.predict(other.export(tmp_path / "other.n4a"), changed_X, engine="dag-ml").y_pred)


def test_source_subset_stop_resume_matches_continuous_native_history(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observer = _SourceObserver(monkeypatch)
    dataset = example.make_dataset()
    root = tmp_path / "resumed"
    tuning = {**example.make_tuning(root / "study"), "n_trials": 4}
    with pytest.raises(DagRunCancelled):
        _run(dataset, root, tuning={**tuning, "progress_callback": baseline._stop_after(1)})
    initial = json.loads((root / "study/structural-source-subsets.n4mopt.json").read_text())["native_checkpoint"]["trials"]
    before = len(observer.callbacks)
    with _run(dataset, root, tuning={**tuning, "resume": True}) as resumed, _run(
        dataset, tmp_path / "continuous", tuning={**example.make_tuning(tmp_path / "continuous/study"), "n_trials": 4},
    ) as continuous:
        assert [trial.to_dict() for trial in resumed.tuning_result.trials] == [trial.to_dict() for trial in continuous.tuning_result.trials]
        assert resumed.structural_tuning_evidence["trials"] == continuous.structural_tuning_evidence["trials"]
        saved = json.loads((root / "study/structural-source-subsets.n4mopt.json").read_text())["native_checkpoint"]["trials"]
        assert saved[:1] == initial
        assert {call["candidate"] for call in observer.callbacks[before:] if call["search"] == 1} == set(range(1, 4))
        X_new = baseline._raw(dataset)[0][36:]
        np.testing.assert_array_equal(nirs4all.predict(resumed.export(tmp_path / "resumed.n4a"), X_new, engine="dag-ml").y_pred,
                                      nirs4all.predict(continuous.export(tmp_path / "continuous.n4a"), X_new, engine="dag-ml").y_pred)
    observer.assert_closed()


def test_source_subset_parallel_workers_match_sequential_and_close_resources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from nirs4all.pipeline.dagml.host_hpo_candidate import HostHpoCandidate

    children: list[Any] = []
    original_init = HostHpoCandidate.__init__

    def observe(child: Any, *args: Any, **kwargs: Any) -> None:
        original_init(child, *args, **kwargs)
        children.append(child)

    monkeypatch.setattr(HostHpoCandidate, "__init__", observe)
    dataset = example.make_dataset()
    with _run(dataset, tmp_path / "sequential", tuning={**example.make_tuning(tmp_path / "sequential/study"), "n_trials": 4}) as sequential, _run(
        dataset, tmp_path / "parallel", tuning={**example.make_tuning(tmp_path / "parallel/study"), "n_trials": 4, "n_jobs": 2},
    ) as parallel:
        assert parallel.structural_tuning_evidence["trials"] == sequential.structural_tuning_evidence["trials"]
        chains._assert_selection(parallel, _native_catalogue(example.make_pipeline(), dataset, example.make_tuning(tmp_path / "catalogue")))
        X_new = baseline._raw(dataset)[0][36:]
        np.testing.assert_array_equal(nirs4all.predict(sequential.export(tmp_path / "sequential.n4a"), X_new, engine="dag-ml").y_pred,
                                      nirs4all.predict(parallel.export(tmp_path / "parallel.n4a"), X_new, engine="dag-ml").y_pred)
    assert len(children) == 4 and len({child._process.pid for child in children}) == 4
    assert all(child._closed and child._process.poll() is not None and not Path(child._private_dir.name).exists() for child in children)


def test_source_candidate_failure_closes_projection_handles_and_resumes_fresh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observer = _SourceObserver(monkeypatch)
    dataset = example.make_dataset()
    tuning = {**example.make_tuning(tmp_path / "study"), "n_trials": 4}
    observer.fail_fold = "fold1"
    with pytest.raises(Exception, match="injected failure after real structural candidate FIT"):
        _run(dataset, tmp_path, tuning={**tuning, "progress_callback": baseline._stop_after(1)})
    checkpoint = json.loads((tmp_path / "study/structural-source-subsets.n4mopt.json").read_text())
    assert checkpoint["native_checkpoint"]["trials"][0]["state"] == "failed"
    observer.assert_closed()
    observer.fail_fold = None
    before = len(observer.callbacks)
    with _run(dataset, tmp_path / "recovered", tuning={**tuning, "resume": True}) as recovered:
        assert recovered.tuning_result.trials[0].state == "FAIL"
        assert [trial.state for trial in recovered.tuning_result.trials[1:]] == ["COMPLETE"] * 3
        calls = [call for call in observer.callbacks[before:] if call["search"] == 1]
        assert {call["candidate"] for call in calls} == set(range(1, 4))
        assert all(entry["fresh"] for entry in observer.entries)
    observer.assert_closed()


@pytest.mark.parametrize("mutation", ["missing_source", "unequal_width_swap", "source_widths", "flat_width"])
def test_winner_rejects_changed_full_input_layout_before_prediction(mutation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = example.make_dataset()
    tuning = {**example.make_tuning(tmp_path / "study"), "n_trials": 1}
    catalogue = _native_catalogue(example.make_pipeline(), dataset, tuning)
    recipe = next(entry for entry in catalogue["entries"] if _selection(entry)[1]["source_indices"] == [0, 2]
                  and len(chains._transforms(entry)) == 2 and "model.alpha" in entry["parameter_bindings"])
    chains._enqueue_recipes(monkeypatch, catalogue, [recipe])
    with _run(dataset, tmp_path, tuning=tuning) as result:
        archive = result.export(tmp_path / "layout.n4a")
    blocks = [block[36:].copy() for block in _blocks(dataset)]
    if mutation == "missing_source":
        values: Any = _prediction_dataset([blocks[0], blocks[2]])
    elif mutation == "unequal_width_swap":
        values = _prediction_dataset([blocks[2], blocks[1], blocks[0]])
    elif mutation == "source_widths":
        values = _prediction_dataset([np.hstack([blocks[0], blocks[1][:, :1]]), blocks[1][:, 1:], blocks[2]])
    else:
        values = np.hstack(blocks)[:, :-1]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid full input layout reached projection or estimator prediction")

    monkeypatch.setattr(ColumnTransformer, "transform", forbidden)
    monkeypatch.setattr(chains._MODELS["Ridge"], "predict", forbidden)
    with pytest.raises(ValueError, match="(?i)(source|layout|width|feature)"):
        nirs4all.predict(archive, values, engine="dag-ml")


def test_equal_width_named_source_swap_is_refused_before_archive_prediction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fixture = example.make_dataset()
    blocks = _blocks(fixture)
    blocks[2] = np.hstack([blocks[2], blocks[0][:, :2]])
    dataset = _NamedDenseDataset("named-source-order", ["nir", "auxiliary", "reference"])
    dataset.add_samples([block[:36] for block in blocks], {"partition": "train"})
    dataset.add_samples([block[36:] for block in blocks], {"partition": "test"})
    dataset.add_targets(np.asarray(fixture.y({})).reshape(-1))
    dataset.add_metadata(np.asarray(fixture.metadata_column("batch"))[:, None], headers=["batch"])
    tuning = {**example.make_tuning(tmp_path / "study"), "n_trials": 1}
    catalogue = _native_catalogue(example.make_pipeline(), dataset, tuning)
    recipe = next(entry for entry in catalogue["entries"]
                  if len(chains._transforms(entry)) == 2 and "model.alpha" in entry["parameter_bindings"]
                  and any(node.get("metadata", {}).get("nirs4all_structural_source_selection", {}).get("source_indices") == [0, 2]
                          for node in entry["graph"]["nodes"]))
    selector = next(node["metadata"]["nirs4all_structural_source_selection"] for node in recipe["graph"]["nodes"]
                    if "nirs4all_structural_source_selection" in node.get("metadata", {}))
    assert selector["source_order"] == ["nir", "reference"]
    assert selector["input_source_widths"] == [6, 3, 6]
    chains._enqueue_recipes(monkeypatch, catalogue, [recipe])
    with _run(dataset, tmp_path, tuning=tuning) as result:
        archive = result.export(tmp_path / "named-layout.n4a")
    correct = _NamedDenseDataset("new-named-source-order", ["nir", "auxiliary", "reference"])
    correct.add_samples([block[36:] for block in blocks], {"partition": "test"})
    prediction = nirs4all.predict(archive, correct, engine="dag-ml")
    assert prediction.metadata["training_performed"] is False
    np.testing.assert_array_equal(prediction.y_pred, nirs4all.predict(archive, np.hstack([block[36:] for block in blocks]), engine="dag-ml").y_pred)
    swapped = _NamedDenseDataset("new-swapped-source-order", ["reference", "auxiliary", "nir"])
    swapped.add_samples([blocks[2][36:], blocks[1][36:], blocks[0][36:]], {"partition": "test"})
    assert swapped.num_features == correct.num_features == [6, 3, 6]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("named source-order drift reached archive projection/prediction")

    monkeypatch.setattr(ColumnTransformer, "transform", forbidden)
    monkeypatch.setattr(chains._MODELS["Ridge"], "predict", forbidden)
    with pytest.raises(ValueError, match="source names or order"):
        nirs4all.predict(archive, swapped, engine="dag-ml")


def test_fresh_installed_source_chain_archive_replays_without_fit_or_hpo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    installed_python = os.environ.get("NIRS4ALL_STRUCTURAL_HPO_INSTALLED_PYTHON")
    if not installed_python:
        if os.environ.get("NIRS4ALL_REQUIRE_STRUCTURAL_HPO_INSTALLED") == "1":
            pytest.fail("mandatory installed proof requires NIRS4ALL_STRUCTURAL_HPO_INSTALLED_PYTHON")
        pytest.skip("fresh installed Python is supplied by the qualification gate")
    dataset = example.make_dataset()
    pipeline = example.make_pipeline()
    pipeline[1]["_or_"] = [None, [SNV(), SavitzkyGolay(window_length=5, polyorder=2)]]
    tuning = {**example.make_tuning(tmp_path / "study"), "n_trials": 1}
    catalogue = _native_catalogue(pipeline, dataset, tuning)
    chosen = next(entry for entry in catalogue["entries"] if _selection(entry)[1]["source_indices"] == [2, 0]
                  and len(chains._transforms(entry)) == 3 and "model.alpha" in entry["parameter_bindings"])
    observer = _SourceObserver(monkeypatch)
    chains._enqueue_recipes(monkeypatch, catalogue, [chosen])
    root = tmp_path / "training"
    X_new = baseline._raw(dataset)[0][36:]
    with _run(dataset, root, tuning=tuning, pipeline=pipeline) as result:
        chains._assert_selection(result, catalogue, observer)
        expected_prediction = _assert_fit_oracle(observer, dataset, catalogue, result)
        archive = Path(result.export(tmp_path / "source-chain-winner.n4a"))
        evidence = deepcopy(result.structural_tuning_evidence)
        graph = deepcopy(result.structural_tuning_training_request["graph"])
    observer.assert_closed()
    shutil.rmtree(root)
    shutil.rmtree(tmp_path / "study")
    np.save(tmp_path / "X.npy", X_new)
    source_names = ["pipeline/dagml_bridge.py", "pipeline/dagml/structural_tuning.py", "pipeline/dagml/structural_sources.py",
                    "pipeline/dagml/node_runner.py", "pipeline/dagml/multimodal_contracts.py", "pipeline/dagml/general_archive.py", "api/result.py"]
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
        from sklearn.compose import ColumnTransformer
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from n4m.model_selection.optimizer import Optimizer
        from nirs4all.data import SpectroDataset
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
            raise AssertionError('source archive replay reached FIT/HPO')
        for cls in (ColumnTransformer, Ridge, PLSRegression, StandardScaler, SNV, SavitzkyGolay):
            cls.fit = forbidden
        ColumnTransformer.fit_transform = forbidden
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
        assert [type(operator).__name__ for operator in fitted] == ['ColumnTransformer', 'StandardNormalVariate', 'SavitzkyGolay', 'Ridge']
        selector = next(node for node in expected['graph']['nodes'] if 'nirs4all_structural_source_selection' in node.get('metadata', {}))
        columns = selector['metadata']['nirs4all_structural_source_selection']['columns']
        assert captured.dense_concat_input_layout == selector['metadata']['nirs4all_structural_source_selection']['input_source_layout']
        assert json.loads(json.dumps(fitted[0].get_params(deep=False))) == selector['params']
        assert fitted[0].n_features_in_ == 13
        assert fitted[0].transformers[0][2] == columns
        assert columns == list(range(9, 13)) + list(range(0, 6))
        assert fitted[-1].alpha == expected['evidence']['selected_params']['model.alpha']
        X = np.load(sys.argv[2])
        dataset = SpectroDataset('installed-new-observations')
        dataset.add_samples([X[:, :6], X[:, 6:9], X[:, 9:]], {'partition':'test'})
        public = nirs4all.predict(archive, dataset, engine='dag-ml')
        with nirs4all.load_session(archive) as session:
            replay = session.predict(X)
        for result in (public, replay):
            np.testing.assert_allclose(result.y_pred.ravel(), expected['prediction'], rtol=2e-6, atol=2e-6)
            assert result.metadata['training_performed'] is False
            assert result.metadata['phase'] == 'PREDICT'
            assert result.metadata['artifact_integrity_verified'] is True
        np.testing.assert_array_equal(public.y_pred, replay.y_pred)
        modified = X.copy()
        modified[:, 6:9] += 1000
        np.testing.assert_array_equal(nirs4all.predict(archive, modified, engine='dag-ml').y_pred, replay.y_pred)
        print(json.dumps({'source_sha256': expected['source_sha256'], 'archive_sha256': expected['archive_sha256'], 'fit_hpo_calls': 0,
                          'evidence': captured.structural_tuning_evidence, 'chain': [type(operator).__name__ for operator in fitted], 'columns':columns}))
    """)
    process = subprocess.run([installed_python, "-I", "-c", script, str(archive), str(tmp_path / "X.npy"), str(tmp_path / "expected.json")],
                             cwd=tmp_path, capture_output=True, text=True, check=False, timeout=180)
    assert process.returncode == 0, process.stderr[-6000:]
    receipt = json.loads(process.stdout.splitlines()[-1])
    assert receipt["fit_hpo_calls"] == 0 and receipt["evidence"] == evidence
    assert receipt["source_sha256"] == expected["source_sha256"] and receipt["archive_sha256"] == expected["archive_sha256"]
    assert receipt["chain"] == ["ColumnTransformer", "StandardNormalVariate", "SavitzkyGolay", "Ridge"]
    assert receipt["columns"] == list(range(9, 13)) + list(range(6))
    assert not root.exists() and not (tmp_path / "study").exists()
