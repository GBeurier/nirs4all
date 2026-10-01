"""Real public SGD CV-weight transfer, native ownership and zero-fit archive replay."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import textwrap
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import dag_ml
import numpy as np
import pytest
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.data import SpectroDataset
from nirs4all.pipeline.dagml import in_process_runner, node_runner
from nirs4all.pipeline.dagml.cli_runner import assemble_cv_refit_dsl
from nirs4all.pipeline.dagml.cv_weight_transfer import CVWeightTransferStore
from nirs4all.pipeline.dagml.envelope import build_envelope
from nirs4all.pipeline.dagml.folds import _build_folds
from nirs4all.pipeline.dagml.host_search_checkpoint import HostSearchOptimizer
from nirs4all.pipeline.dagml.identity import mint_identity
from nirs4all.pipeline.dagml.multimodal_tuning import _evaluate_host_task
from nirs4all.pipeline.dagml.resolver import MaterializationResolver
from nirs4all.pipeline.dagml.rt import RtError
from nirs4all.pipeline.dagml.tuning_contracts import parse_tuning_spec
from nirs4all.pipeline.dagml_bridge import controller_manifests
from tests.integration.parity._dagml_cli import dagml_cli_path


def _dataset(*, offset: float = 0.0) -> SpectroDataset:
    rng = np.random.default_rng(41)
    X = rng.normal(scale=0.5, size=(30, 3)).astype(np.float32)
    X[:, 0] = np.linspace(-0.6, 0.6, 30, dtype=np.float32)
    y = (3 * X[:, 0] - 2 * X[:, 1] + 0.7 * X[:, 2] + 0.25 + offset).astype(np.float32)
    dataset = SpectroDataset("cv-weight-transfer")
    dataset.add_samples(X[:24], {"partition": "train"})
    dataset.add_samples(X[24:], {"partition": "test"})
    dataset.add_targets(y)
    return dataset


def _model(**params: Any) -> SGDRegressor:
    return SGDRegressor(**{
        "loss": "squared_error", "penalty": "l2", "alpha": 0.01,
        "learning_rate": "constant", "eta0": 0.01,
        "random_state": 19, "shuffle": False, "max_iter": 2, "tol": None, **params,
    })


def _pipeline(fold: str = "fold1", **model_params: Any) -> list[Any]:
    return [KFold(3, shuffle=True, random_state=31), {
        "model": _model(**model_params),
        "refit_params": {"warm_start": True, "warm_start_fold": fold, "max_iter": 3},
    }]


def _run(pipeline: list[Any], dataset: SpectroDataset, workspace: Path, **kwargs: Any) -> Any:
    return nirs4all.run(
        pipeline, dataset, engine="dag-ml", workspace_path=workspace,
        save_charts=False, save_artifacts=True, verbose=0, random_state=19, **kwargs,
    )


class _FitObserver:
    """Observe real public fits and lifecycle; every successful fit calls sklearn."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.entries: list[dict[str, Any]] = []
        self.stores: list[CVWeightTransferStore] = []
        self.captures: list[tuple[str, bool, int, int]] = []
        self.graphs: list[dict[str, Any]] = []
        self.campaign_outcomes: dict[int, dict[str, Any]] = {}
        self.store_campaigns: dict[int, int | None] = {}
        self.fail_phase: str | None = None
        self.fail_fold: str | None = None
        self.original_fit = SGDRegressor.fit
        active: ContextVar[dict[str, Any] | None] = ContextVar("observed_native_sgd_task", default=None)
        campaign: ContextVar[int | None] = ContextVar("observed_native_sgd_campaign", default=None)
        original_node = node_runner.run_model_node
        original_store = CVWeightTransferStore.__init__
        original_capture = CVWeightTransferStore.capture
        original_campaign = in_process_runner.run_cv_refit_bundle

        def observe_campaign(**kwargs: Any) -> Any:
            campaign_index = len(self.graphs)
            token = campaign.set(campaign_index)
            self.graphs.append(copy.deepcopy(kwargs["graph"]))
            try:
                outcome = original_campaign(**kwargs)
                self.campaign_outcomes[campaign_index] = outcome
                return outcome
            finally:
                campaign.reset(token)

        def observe_node(task: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            token = active.set(copy.deepcopy(task))
            try:
                return original_node(task, *args, **kwargs)
            finally:
                active.reset(token)

        def observe_store(store: CVWeightTransferStore, *args: Any, **kwargs: Any) -> None:
            original_store(store, *args, **kwargs)
            self.stores.append(store)
            self.store_campaigns[id(store)] = campaign.get()

        def observe_capture(store: CVWeightTransferStore, request: Any, identity: Any, fold_id: str, *args: Any, **kwargs: Any) -> bool:
            captured = original_capture(store, request, identity, fold_id, *args, **kwargs)
            self.captures.append((fold_id, captured, store.pending_count, store.pending_bytes))
            return captured

        def observe_fit(model: SGDRegressor, X: Any, y: Any, **kwargs: Any) -> Any:
            task = active.get()
            assert task is not None, "SGD fit escaped the native model callback"
            entry: dict[str, Any] = {
                "task": task, "campaign": campaign.get(), "params": copy.deepcopy(model.get_params(deep=False)),
                "X": np.array(X, copy=True), "y": np.array(y, copy=True),
                "initializers": {key: np.array(value, copy=True) for key, value in kwargs.items() if key in {"coef_init", "intercept_init"} and value is not None},
                "initializer_refs": {key: value for key, value in kwargs.items() if key in {"coef_init", "intercept_init"} and value is not None},
                "fresh": not hasattr(model, "coef_"),
            }
            self.entries.append(entry)
            if task["phase"] == self.fail_phase and (self.fail_fold is None or task.get("fold_id") == self.fail_fold):
                raise RuntimeError("injected native SGD fit failure")
            fitted = self.original_fit(model, X, y, **kwargs)
            entry.update(coef=model.coef_.copy(), intercept=model.intercept_.copy(),
                         coef_ref=model.coef_, intercept_ref=model.intercept_, counter=float(model.t_))
            return fitted

        monkeypatch.setattr(node_runner, "run_model_node", observe_node)
        monkeypatch.setattr(in_process_runner, "run_cv_refit_bundle", observe_campaign)
        monkeypatch.setattr(CVWeightTransferStore, "__init__", observe_store)
        monkeypatch.setattr(CVWeightTransferStore, "capture", observe_capture)
        monkeypatch.setattr(SGDRegressor, "fit", observe_fit)

    def assert_cleared(self) -> None:
        assert self.stores, "the native campaign never opened a transfer store"
        assert all(store.pending_count == 0 and store.pending_bytes == 0 for store in self.stores)
        assert all(count <= 2 for _, _, count, _ in self.captures)
        for store in self.stores:
            with pytest.raises(RuntimeError, match="store is closed"):
                store.__enter__()


@pytest.fixture(autouse=True)
def native_execution_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *a, **k: pytest.fail("legacy scheduler executed"))
    monkeypatch.setattr("nirs4all.pipeline.dagml.run_paths._run_model_on_precomputed_matrix", lambda *a, **k: pytest.fail("Python CV loop executed"))


def _source_for_refit(entries: list[dict[str, Any]], refit: dict[str, Any], fold: str) -> dict[str, Any]:
    task = refit["task"]
    source = [entry for entry in entries if entry["task"]["phase"] == "FIT_CV"
              and entry["campaign"] == refit["campaign"]
              and entry["task"].get("fold_id") == fold
              and entry["task"]["run_id"] == task["run_id"]
              and entry["task"].get("variant_id") == task.get("variant_id")
              and entry["task"]["node_plan"]["node_id"] == task["node_plan"]["node_id"]
              and entry["task"]["node_plan"]["controller_id"] == task["node_plan"]["controller_id"]]
    assert len(source) == 1
    return source[0]


def _assert_transfer(observer: _FitObserver, entries: list[dict[str, Any]], fitted: SGDRegressor, fold: str) -> None:
    refit = next(entry for entry in entries if entry["task"]["phase"] == "REFIT")
    source = _source_for_refit(entries, refit, fold)
    assert refit["fresh"] and all(entry["fresh"] for entry in entries)
    for entry in entries:
        if entry["task"]["phase"] == "FIT_CV":
            assert entry["initializers"] == {} and entry["params"]["warm_start"] is False
    np.testing.assert_array_equal(refit["initializers"]["coef_init"], source["coef"])
    np.testing.assert_array_equal(refit["initializers"]["intercept_init"], source["intercept"])
    assert not np.shares_memory(refit["initializer_refs"]["coef_init"], source["coef_ref"])
    assert not np.shares_memory(refit["initializer_refs"]["intercept_init"], source["intercept_ref"])
    assert refit["params"]["warm_start"] is True
    assert refit["counter"] == 1 + refit["params"]["max_iter"] * len(refit["X"])
    oracle = SGDRegressor(**refit["params"])
    observer.original_fit(oracle, refit["X"], refit["y"],
                          coef_init=source["coef"].copy(), intercept_init=source["intercept"].copy())
    np.testing.assert_array_equal(fitted.coef_, oracle.coef_)
    np.testing.assert_array_equal(fitted.intercept_, oracle.intercept_)
    cold = SGDRegressor(**refit["params"])
    observer.original_fit(cold, refit["X"], refit["y"])
    assert not np.array_equal(fitted.coef_, cold.coef_), "the requested transfer was indistinguishable from a cold fit"
    provenance = fitted._nirs4all_cv_weight_transfer
    assert provenance["schema"] == "nirs4all.cv-weight-transfer.v1"
    assert provenance["source_fold_id"] == fold
    assert provenance["run_id"] == refit["task"]["run_id"]
    assert provenance["node_id"] == refit["task"]["node_plan"]["node_id"]
    assert provenance["controller_id"] == refit["task"]["node_plan"]["controller_id"]
    assert provenance["variant_label"] == (refit["task"].get("variant_id") or "base")
    assert provenance["coef_sha256"] == hashlib.sha256(source["coef"].tobytes(order="C")).hexdigest()
    assert provenance["intercept_sha256"] == hashlib.sha256(source["intercept"].tobytes(order="C")).hexdigest()
    assert provenance["weight_dtype"] == source["coef"].dtype.str
    assert provenance["coef_dtype"] == source["coef"].dtype.str
    assert provenance["intercept_dtype"] == source["intercept"].dtype.str
    assert provenance["coef_shape"] == [refit["X"].shape[1]] and provenance["intercept_shape"] == [1]
    assert provenance["cv_fit_budget"] == {"max_iter": 2, "tol": None}
    assert provenance["refit_fit_budget"] == {"max_iter": 3, "tol": None}
    assert provenance["optimization_counter_policy"] == "reset_by_fit"
    assert provenance["training_rows_retained"] is False and provenance["initializers_detached"] is True
    json.dumps(provenance, allow_nan=False)
    observer.assert_cleared()


@pytest.mark.parametrize("fold", ["fold0", "fold1", "fold2"])
@pytest.mark.parametrize("schedule", ["constant", "invscaling"])
def test_public_native_folds_transfer_real_weights_and_match_independent_sklearn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fold: str, schedule: str,
) -> None:
    observer = _FitObserver(monkeypatch)
    dataset = _dataset()
    identity = mint_identity(dataset)
    with _run(_pipeline(fold, learning_rate=schedule), dataset, tmp_path / "workspace") as result:
        assert len(observer.entries) == 4
        fitted = result._dagml_refit_artifacts[0]["estimator"]
        _assert_transfer(observer, observer.entries, fitted, fold)
        assert [captured_fold for captured_fold, captured, _, _ in observer.captures if captured] == [fold]
        assert max(count for _, _, count, _ in observer.captures) == 1
        train_X = dataset.x({"partition": "train"}, layout="2d")
        for index, (train, val) in enumerate(KFold(3, shuffle=True, random_state=31).split(train_X)):
            entry = next(item for item in observer.entries if item["task"].get("fold_id") == f"fold{index}")
            train_ids, val_ids = node_runner._train_predict_ids(entry["task"])
            assert train_ids == [identity.to_wire(int(row)) for row in train]
            assert val_ids == [identity.to_wire(int(row)) for row in val]
            np.testing.assert_array_equal(entry["X"], train_X[train])
            oracle = SGDRegressor(**entry["params"])
            observer.original_fit(oracle, entry["X"], entry["y"])
            np.testing.assert_array_equal(entry["coef"], oracle.coef_)
        refit = observer.entries[-1]
        full_train_ids, _ = node_runner._train_predict_ids(refit["task"])
        assert full_train_ids == [identity.to_wire(row) for row in range(24)]
        assert len(refit["X"]) == 24
        assert any(view["partition"] == "full_train" for view in refit["task"]["data_views"].values())
        np.testing.assert_array_equal(refit["X"], train_X)


def test_public_generated_variants_and_separate_runs_capture_their_own_native_fold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observer = _FitObserver(monkeypatch)
    fingerprints: list[dict[float, str]] = []
    for run_index, offset in enumerate((0.0, 1.5)):
        start = len(observer.entries)
        dataset = _dataset(offset=offset)
        identity = mint_identity(dataset)
        pipeline = _pipeline()
        pipeline[-1]["alpha"] = {"_or_": [0.01, 0.2]}
        with _run(pipeline, dataset, tmp_path / f"run-{run_index}") as result:
            entries = observer.entries[start:]
            assert len([entry for entry in entries if entry["task"]["phase"] == "FIT_CV"]) == 6
            campaigns = sorted({entry["campaign"] for entry in entries})
            assert len(campaigns) == 2
            run_fingerprints = {}
            campaign_scores = {}
            for campaign in campaigns:
                scoped = [entry for entry in entries if entry["campaign"] == campaign]
                assert len(scoped) == 4
                outcome = observer.campaign_outcomes[campaign]
                assert len(outcome["refit_artifacts"]) == 1
                fitted = outcome["refit_artifacts"][0]["estimator"]
                _assert_transfer(observer, scoped, fitted, "fold1")
                source = _source_for_refit(scoped, scoped[-1], "fold1")
                assert source["params"]["alpha"] == fitted.alpha
                stores = [store for store in observer.stores if observer.store_campaigns[id(store)] == campaign]
                assert len(stores) == 1 and stores[0].pending_count == stores[0].pending_bytes == 0
                foreign = [entry for entry in entries if entry["campaign"] != campaign and entry["task"]["phase"] == "FIT_CV"]
                assert all(not np.shares_memory(scoped[-1]["initializer_refs"]["coef_init"], entry["coef_ref"]) for entry in foreign)
                run_fingerprints[fitted.alpha] = fitted._nirs4all_cv_weight_transfer["coef_sha256"]
                squared_errors = []
                validation_rows = []
                for entry in scoped:
                    if entry["task"]["phase"] != "FIT_CV":
                        continue
                    oracle = SGDRegressor(**entry["params"])
                    observer.original_fit(oracle, entry["X"], entry["y"])
                    np.testing.assert_array_equal(entry["coef"], oracle.coef_)
                    np.testing.assert_array_equal(entry["intercept"], oracle.intercept_)
                    _, validation_ids = node_runner._train_predict_ids(entry["task"])
                    rows = [identity.to_int(sample) for sample in validation_ids]
                    validation_rows.extend(rows)
                    X_val = np.asarray(dataset.x_rows(rows, layout="2d"))
                    y_val = np.asarray(dataset.y({"sample": rows}), dtype=float).ravel()
                    squared_errors.extend((oracle.predict(X_val).astype(float) - y_val) ** 2)
                assert sorted(validation_rows) == list(range(24))
                avg = next(report for report in outcome["scores"]["reports"]
                           if report["partition"] == "validation" and report["fold_id"] == "avg")
                campaign_scores[campaign] = avg["metrics"]["rmse"]
                assert campaign_scores[campaign] == pytest.approx(float(np.sqrt(np.mean(squared_errors))), abs=1e-10)
            winner = min(campaigns, key=campaign_scores.__getitem__)
            assert len(result._dagml_refit_artifacts) == 1
            assert result._dagml_refit_artifacts[0]["estimator"] is observer.campaign_outcomes[winner]["refit_artifacts"][0]["estimator"]
            assert result.cv_best_score == pytest.approx(campaign_scores[winner], abs=1e-10)
            fingerprints.append(run_fingerprints)
    assert fingerprints[0].keys() == fingerprints[1].keys() == {0.01, 0.2}
    assert all(fingerprints[0][alpha] != fingerprints[1][alpha] for alpha in fingerprints[0])
    assert len({id(store) for store in observer.stores}) == 4
    observer.assert_cleared()


@pytest.mark.parametrize("mutation", ["missing_selector", "best", "last", "ridge", "average", "early_stopping", "optimal", "transform", "target_transform", "local_hpo"])
def test_closed_public_profile_refuses_unsupported_requests_before_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    pipeline = _pipeline()
    step = pipeline[-1]
    if mutation == "missing_selector":
        del step["refit_params"]["warm_start_fold"]
    elif mutation in {"best", "last"}:
        step["refit_params"]["warm_start_fold"] = mutation
    elif mutation == "ridge":
        step["model"] = Ridge()
    elif mutation in {"average", "early_stopping"}:
        step["model"].set_params(**{mutation: True})
    elif mutation == "optimal":
        step["model"].set_params(learning_rate="optimal")
    elif mutation == "transform":
        pipeline.insert(0, StandardScaler())
    elif mutation == "target_transform":
        pipeline.insert(0, {"y_processing": StandardScaler()})
    else:
        step["finetune_params"] = {"approach": "single", "sampler": "grid", "n_trials": 2, "model_params": {"alpha": [0.01, 0.2]}}
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("unsupported warm-start profile reached a numerical fit")
    monkeypatch.setattr(SGDRegressor, "fit", forbidden)
    monkeypatch.setattr(Ridge, "fit", forbidden)
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    diagnostics = {"missing_selector": "explicit warm_start_fold", "best": "best", "last": "last", "ridge": "closed SGDRegressor",
                   "average": "average=False", "early_stopping": "early_stopping=False", "optimal": "schedules",
                   "transform": "preprocessing", "target_transform": "preprocessing", "local_hpo": "finetune_params"}
    with pytest.raises(RtError, match="(?i)(warm.start|weight.transfer)") as refusal:
        _run(pipeline, _dataset(), tmp_path / "invalid")
    assert refusal.value.verb == "run" and refusal.value.cause == "unsupported_shape"
    assert diagnostics[mutation] in refusal.value.message


@pytest.mark.parametrize("mutation", ["missing_fold", "changed_recipe"])
def test_incompatible_or_missing_snapshot_never_cold_fits_refit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    observer = _FitObserver(monkeypatch)
    pipeline = _pipeline("fold9" if mutation == "missing_fold" else "fold1")
    if mutation == "changed_recipe":
        pipeline[-1]["refit_params"]["alpha"] = 0.2
    with pytest.raises(Exception, match="(?i)(fold|recipe|captured|weight.transfer)"):
        _run(pipeline, _dataset(), tmp_path / "invalid")
    assert not any(entry["task"]["phase"] == "REFIT" for entry in observer.entries)
    if observer.stores:
        observer.assert_cleared()


@pytest.mark.parametrize("termination", ["cv_error", "refit_error", "cancel"])
def test_real_run_clears_captured_weights_after_error_or_cancellation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, termination: str) -> None:
    observer = _FitObserver(monkeypatch)
    options: dict[str, Any] = {}
    if termination == "cancel":
        options["should_stop"] = lambda: any(store.pending_count for store in observer.stores)
    elif termination == "cv_error":
        observer.fail_phase, observer.fail_fold = "FIT_CV", "fold1"
    else:
        observer.fail_phase = "REFIT"
    with pytest.raises(Exception, match="(?i)(cancel|injected native SGD fit failure)"):
        _run(_pipeline("fold0"), _dataset(), tmp_path / termination, **options)
    assert any(captured for _, captured, _, _ in observer.captures)
    observer.assert_cleared()


def test_public_native_global_hpo_isolates_cv_proposals_and_selected_refit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observer = _FitObserver(monkeypatch)
    dataset = _dataset()
    identity = mint_identity(dataset)
    pool = list(range(24))
    pipeline = _pipeline()
    folds = _build_folds(pipeline[0], dataset, pool, set())
    envelope = build_envelope(dataset, identity, sample_ints=pool)
    dsl = assemble_cv_refit_dsl([pipeline[-1]], identity, envelope, folds, dsl_id="sgd-cv-weight-hpo", n_splits=3)
    graph = json.loads(dag_ml.compile_pipeline_dsl_graph_json(json.dumps(dsl)))
    target = next(node["id"] for node in graph["nodes"] if node["kind"] == "model")
    nodes = {node["id"]: node for node in graph["nodes"]}
    resolver = MaterializationResolver(dataset, identity)
    spec = parse_tuning_spec({"engine": "n4m", "sampler": "sobol", "seed": 19, "n_trials": 3, "space": {"alpha": (0.001, 0.3)}})
    optimizer = HostSearchOptimizer(spec, n_folds=3)
    store: dict[Any, Any] = {}

    def evaluate(task: dict[str, Any]) -> dict[str, Any]:
        return _evaluate_host_task(task, resolver=resolver, nodes=nodes, graph=graph,
                                   model_store=store, view_store=None, operator_seed=19)

    def progress(event: dict[str, Any]) -> bool:
        if event["operation"] != "prepare_terminal":
            optimizer.checkpoint(event)
        return True

    try:
        outcome = nirs4all.run_host_hpo_search(
            dsl, envelope, controller_manifests(),
            {"target_node": target, "trial_budget": 3, "metric": "rmse", "direction": "minimize",
             "fold_score_reduction": "mean", "optimizer_descriptor": spec.to_dict()},
            evaluate, optimizer, progress_callback=progress,
        )
    finally:
        node_runner.clear_cv_weight_transfers(store)
        optimizer.close()
    search_entries = list(observer.entries)
    assert outcome["status"] == "completed" and len(outcome["trials"]) == 3
    selected = next(trial for trial in outcome["trials"] if trial["trial_index"] == outcome["selected_trial_index"])
    assert selected["score"] == min(trial["score"] for trial in outcome["trials"])
    assert outcome["selected_params"] == selected["params"]
    assert len(search_entries) == 9 and all(entry["task"]["phase"] == "FIT_CV" for entry in search_entries)
    assert all(entry["initializers"] == {} for entry in search_entries)
    assert {entry["params"]["alpha"] for entry in search_entries} == {trial["params"]["alpha"] for trial in outcome["trials"]}
    for trial in outcome["trials"]:
        trial_entries = [entry for entry in search_entries if entry["params"]["alpha"] == trial["params"]["alpha"]]
        assert len(trial_entries) == 3
        fold_scores = []
        for entry in trial_entries:
            oracle = SGDRegressor(**entry["params"])
            observer.original_fit(oracle, entry["X"], entry["y"])
            _, validation_ids = node_runner._train_predict_ids(entry["task"])
            rows = [identity.to_int(sample) for sample in validation_ids]
            X_val = np.asarray(dataset.x_rows(rows, layout="2d"))
            y_val = np.asarray(dataset.y({"sample": rows}), dtype=float).ravel()
            fold_scores.append(float(np.sqrt(np.mean((oracle.predict(X_val).astype(float) - y_val) ** 2))))
        assert trial["score"] == pytest.approx(float(np.mean(fold_scores)), abs=1e-10)
    observer.assert_cleared()
    selected_alpha = outcome["selected_params"]["alpha"]
    start = len(observer.entries)
    with _run(_pipeline(alpha=selected_alpha), dataset, tmp_path / "winner") as result:
        entries = observer.entries[start:]
        assert len(entries) == 4
        fitted = result._dagml_refit_artifacts[0]["estimator"]
        _assert_transfer(observer, entries, fitted, "fold1")
        assert all(entry["params"]["alpha"] == selected_alpha for entry in entries)
        source = _source_for_refit(entries, entries[-1], "fold1")
        assert all(not np.shares_memory(entries[-1]["initializer_refs"]["coef_init"], entry["coef_ref"]) for entry in search_entries)
        assert fitted._nirs4all_cv_weight_transfer["coef_sha256"] == hashlib.sha256(source["coef"].tobytes()).hexdigest()


def test_real_inprocess_and_subprocess_refit_agree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli = dagml_cli_path()
    if not cli.is_file():
        if os.environ.get("NIRS4ALL_REQUIRE_CV_WEIGHT_CLI") == "1":
            pytest.fail("mandatory CV-weight subprocess gate requires N4A_DAGML_CLI")
        pytest.skip("the real native DAG-ML CLI is required for subprocess agreement")
    monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    observed = []
    for mode in ("1", "0"):
        monkeypatch.setenv("N4A_DAGML_INPROCESS", mode)
        with _run(_pipeline(), _dataset(), tmp_path / f"mode-{mode}") as result:
            fitted = result._dagml_refit_artifacts[0]["estimator"]
            observed.append((fitted.coef_.copy(), fitted.intercept_.copy(), copy.deepcopy(fitted._nirs4all_cv_weight_transfer), result.cv_best_score))
    np.testing.assert_array_equal(observed[0][0], observed[1][0])
    np.testing.assert_array_equal(observed[0][1], observed[1][1])
    # Separate executions may have distinct native run IDs; recipe, source
    # fold, representation and source weight bytes must still agree exactly.
    for _, _, provenance, _ in observed:
        assert isinstance(provenance["run_id"], str) and provenance["run_id"]
    assert {key: value for key, value in observed[0][2].items() if key != "run_id"} == {
        key: value for key, value in observed[1][2].items() if key != "run_id"
    }
    assert observed[0][3] == observed[1][3]


@pytest.mark.parametrize("termination", ["close", "eof", "error"])
def test_real_worker_shutdown_clears_an_unconsumed_native_fold_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, termination: str,
) -> None:
    observer = _FitObserver(monkeypatch)
    dataset = _dataset()
    with _run(_pipeline("fold0"), dataset, tmp_path / "capture"):
        pass
    tasks = [entry["task"] for entry in observer.entries if entry["task"]["phase"] == "FIT_CV"]
    first = next(task for task in tasks if task["fold_id"] == "fold0")
    frames = [{"type": "init"}, {"type": "task", "task": first}]
    if termination == "close":
        frames.append({"type": "close"})
    elif termination == "error":
        frames.append({"type": "task", "task": next(task for task in tasks if task["fold_id"] == "fold1")})
    with (tmp_path / "dataset.pkl").open("wb") as stream:
        pickle.dump(dataset, stream)
    assert len(observer.graphs) == 1
    (tmp_path / "graph.json").write_text(json.dumps(observer.graphs[0]), encoding="utf-8")
    script = textwrap.dedent("""\
        import json, sys
        from sklearn.linear_model import SGDRegressor
        from nirs4all.pipeline.dagml.cv_weight_transfer import CVWeightTransferStore
        from nirs4all.pipeline.dagml.process_adapter import _build_handler, run_jsonl_loop
        stores, closed, captured = [], [], []
        original_init = CVWeightTransferStore.__init__
        original_close = CVWeightTransferStore.close
        original_capture = CVWeightTransferStore.capture
        original_fit = SGDRegressor.fit
        fit_count = 0
        def init(store, *args, **kwargs):
            original_init(store, *args, **kwargs)
            stores.append(store)
        def close(store):
            before = store.pending_count
            original_close(store)
            closed.append([before, store.pending_count, store.pending_bytes])
        def capture(store, *args, **kwargs):
            result = original_capture(store, *args, **kwargs)
            if result:
                captured.append(store.pending_count)
            return result
        def fit(model, X, y, **kwargs):
            global fit_count
            fit_count += 1
            if sys.argv[1] == 'error' and fit_count == 2:
                raise RuntimeError('injected worker CV error')
            return original_fit(model, X, y, **kwargs)
        CVWeightTransferStore.__init__ = init
        CVWeightTransferStore.close = close
        CVWeightTransferStore.capture = capture
        SGDRegressor.fit = fit
        error = None
        try:
            run_jsonl_loop(sys.stdin, sys.stdout, _build_handler())
        except RuntimeError as exc:
            assert sys.argv[1] == 'error' and str(exc) == 'injected worker CV error'
            error = str(exc)
        assert captured == [1]
        assert stores and all(store.pending_count == 0 and store.pending_bytes == 0 for store in stores)
        assert closed and any(before == 1 for before, count, size in closed)
        assert all(count == 0 and size == 0 for before, count, size in closed)
        assert (error is not None) == (sys.argv[1] == 'error')
        print(json.dumps({'termination': sys.argv[1], 'closed': closed, 'captured': captured, 'error': error}))
    """)
    process = subprocess.run(
        [sys.executable, "-c", script, termination],
        input="".join(json.dumps(frame) + "\n" for frame in frames),
        cwd=Path(nirs4all.__file__).resolve().parent.parent,
        env={**os.environ, "N4A_DAGML_DATASET_PICKLE": str(tmp_path / "dataset.pkl"), "N4A_DAGML_GRAPH_PATH": str(tmp_path / "graph.json")},
        capture_output=True, text=True, check=False, timeout=180,
    )
    assert process.returncode == 0, process.stderr[-5000:]
    receipt = json.loads(process.stdout.splitlines()[-1])
    assert receipt["termination"] == termination and receipt["captured"] == [1]
    assert all(count == 0 and size == 0 for _, count, size in receipt["closed"])


def test_fresh_installed_archive_keeps_provenance_and_never_fits_or_searches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    installed_python = os.environ.get("NIRS4ALL_CV_WEIGHT_INSTALLED_PYTHON")
    if not installed_python:
        if os.environ.get("NIRS4ALL_REQUIRE_CV_WEIGHT_INSTALLED") == "1":
            pytest.fail("mandatory installed proof requires NIRS4ALL_CV_WEIGHT_INSTALLED_PYTHON")
        pytest.skip("fresh installed Python is supplied by the qualification gate")
    observer = _FitObserver(monkeypatch)
    dataset = _dataset()
    X_new = dataset.x({"partition": "test"}, layout="2d")
    workspace = tmp_path / "workspace"
    with _run(_pipeline(), dataset, workspace) as result:
        fitted = result._dagml_refit_artifacts[0]["estimator"]
        _assert_transfer(observer, observer.entries, fitted, "fold1")
        expected = fitted.predict(X_new)
        provenance = copy.deepcopy(fitted._nirs4all_cv_weight_transfer)
        archive = result.export(tmp_path / "captured.n4a")
    shutil.rmtree(workspace)
    np.save(tmp_path / "X.npy", X_new)
    production_paths = ["pipeline/dagml/" + name + ".py" for name in (
        "cv_weight_transfer", "training_controls", "node_runner", "in_process_runner", "process_adapter", "multimodal_tuning",
    )]
    package_root = Path(nirs4all.__file__).resolve().parent
    expected_sources = {name: hashlib.sha256((package_root / name).read_bytes()).hexdigest() for name in production_paths}
    extension = importlib.import_module("dag_ml._dag_ml")
    expectation = {"prediction": expected.tolist(), "provenance": provenance, "source_sha256": expected_sources,
                   "dag_extension_sha256": hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest()}
    (tmp_path / "expected.json").write_text(json.dumps(expectation), encoding="utf-8")
    script = textwrap.dedent("""\
        import hashlib, importlib, json, pathlib, sys
        import dag_ml, numpy as np, nirs4all
        from sklearn.linear_model import SGDRegressor
        from nirs4all.pipeline.dagml.general_archive import load_general_archive
        from nirs4all.pipeline.dagml.host_search_checkpoint import HostSearchOptimizer
        root = pathlib.Path(nirs4all.__file__).resolve().parent
        assert 'site-packages' in root.parts, root
        expected = json.loads(pathlib.Path(sys.argv[3]).read_text())
        for name, digest in expected['source_sha256'].items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
        extension = importlib.import_module('dag_ml._dag_ml')
        assert hashlib.sha256(pathlib.Path(extension.__file__).read_bytes()).hexdigest() == expected['dag_extension_sha256']
        def forbidden(*args, **kwargs):
            raise AssertionError('archive replay reached FIT/HPO')
        SGDRegressor.fit = forbidden
        SGDRegressor.partial_fit = forbidden
        HostSearchOptimizer.__init__ = forbidden
        nirs4all.run = forbidden
        nirs4all.run_host_hpo_search = forbidden
        nirs4all.execute_training = forbidden
        dag_ml.run_host_hpo_search_in_process = forbidden
        dag_ml.execute_training = forbidden
        captured = load_general_archive(sys.argv[1])['artifact']['estimator']
        while hasattr(captured, 'estimator'):
            captured = captured.estimator
        assert captured._nirs4all_cv_weight_transfer == expected['provenance']
        X = np.load(sys.argv[2])
        public = nirs4all.predict(sys.argv[1], X, engine='dag-ml')
        with nirs4all.load_session(sys.argv[1]) as session:
            replayed = session.predict(X)
        for result in (public, replayed):
            np.testing.assert_array_equal(result.y_pred.ravel(), np.asarray(expected['prediction']).ravel())
            assert result.metadata['training_performed'] is False
            assert result.metadata['phase'] == 'PREDICT'
            assert result.metadata['artifact_integrity_verified'] is True
        print(json.dumps({'installed_package': str(root), 'source_sha256': expected['source_sha256'],
                          'provenance': expected['provenance'], 'fit_hpo_calls': 0, 'predictions': public.y_pred.tolist()}))
    """)
    process = subprocess.run(
        [installed_python, "-I", "-c", script, str(archive), str(tmp_path / "X.npy"), str(tmp_path / "expected.json")],
        cwd=tmp_path, capture_output=True, text=True, check=False, timeout=180,
    )
    assert process.returncode == 0, process.stderr[-5000:]
    receipt = json.loads(process.stdout.splitlines()[-1])
    assert receipt["fit_hpo_calls"] == 0 and receipt["provenance"] == provenance
    assert receipt["source_sha256"] == expected_sources
    assert not workspace.exists()
