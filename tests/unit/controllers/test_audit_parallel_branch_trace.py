"""Parallel workers retain isolated runtimes, strict traces and replayable models."""

import copy
import threading

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import nirs4all
from nirs4all.config.cache_config import CacheConfig
from nirs4all.controllers.data.branch import BranchController
from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
from nirs4all.pipeline.execution.executor import PipelineExecutor
from nirs4all.pipeline.steps.step_runner import StepRunner
from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry
from nirs4all.pipeline.trace.recorder import TraceRecorder


def _dataset():
    rng = np.random.default_rng(531)
    dataset = SpectroDataset("parallel_trace")
    dataset.add_samples(rng.normal(size=(24, 8)), {"partition": "train"})
    dataset.add_samples(rng.normal(size=(8, 8)) + 2, {"partition": "test"})
    dataset.add_targets(rng.normal(size=32))
    return dataset


def test_worker_preparation_isolates_runtime_without_mutating_parent(tmp_path, monkeypatch):
    dataset = _dataset()
    recorder = TraceRecorder(pipeline_uid="parent")
    registry = ArtifactRegistry(tmp_path, dataset.name, pipeline_id="parent")
    runner = StepRunner(show_spinner=False)
    runtime = RuntimeContext(store=object(), trace_recorder=recorder, artifact_registry=registry,
                             step_runner=runner, operation_count=9, processing_counter=4,
                             artifact_load_counter={0: 2}, best_refit_chains={})
    context = ExecutionContext()
    context.custom["_runtime_context"] = runtime
    controller = BranchController()
    defs = [{"steps": [StandardScaler()]} for _ in range(3)]
    args = controller._build_parallel_worker_args([[(i, d)] for i, d in enumerate(defs)], defs, dataset, context, [], runtime)
    workers = [arg[-1] for arg in args]
    assert len({id(worker) for worker in workers}) == 3
    assert all(worker is not runtime and worker.step_runner is not runner for worker in workers)
    assert all(worker.store is runtime.store and worker.artifact_registry is registry for worker in workers)
    assert all(arg[2].custom["_runtime_context"] is arg[-1] for arg in args)
    workers[0].next_op()
    workers[0].artifact_load_counter[0] = 10
    assert workers[1].operation_count == runtime.operation_count == 9
    assert workers[1].artifact_load_counter == {} and runtime.artifact_load_counter == {0: 2}
    assert runtime.trace_recorder is recorder and runtime.step_runner is runner
    assert runtime.best_refit_chains == {} and all(worker.best_refit_chains is None for worker in workers)
    original = copy.deepcopy

    def failing_copy(value, *args, **kwargs):
        if value is dataset:
            raise ValueError("dataset copy failed")
        return original(value, *args, **kwargs)

    monkeypatch.setattr(copy, "deepcopy", failing_copy)
    with pytest.raises(ValueError, match="dataset copy failed"):
        controller._build_parallel_worker_args([[(0, defs[0])]], defs, dataset, context, [], runtime)
    assert runtime.store is not None and runtime.artifact_registry is registry
    assert runtime.trace_recorder is recorder and runtime.step_runner is runner


class _ConcurrentScaler(StandardScaler):
    barrier = None
    threads = []

    def fit(self, X, y=None, sample_weight=None):
        self.threads.append(threading.get_ident())
        if self.barrier is not None:
            self.barrier.wait(timeout=10)
        return super().fit(X, y, sample_weight=sample_weight)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("cow", [False, True])
@pytest.mark.parametrize("ancestor", [False, True])
def test_tuple_and_transform_workers_have_single_active_trace_owner(tmp_path, parallel, cow, ancestor):
    dataset = _dataset()
    raw = dataset.x({}).copy()
    train = dataset.x({"partition": "train"}).copy()
    expected = [np.hstack([StandardScaler().fit(train).transform(raw), PCA(2).fit(train).transform(raw)]),
                StandardScaler().fit(train).transform(raw)]
    registry = ArtifactRegistry(tmp_path, dataset.name, pipeline_id="parallel_trace")
    recorder = TraceRecorder(pipeline_uid="parallel_trace")
    if ancestor:
        recorder.enter_branch(7)
    context = ExecutionContext()
    context.selector.partition = None
    context.selector.branch_path = [7] if ancestor else []
    runner = StepRunner(show_spinner=False)
    runtime = RuntimeContext(step_runner=runner, artifact_registry=registry, trace_recorder=recorder,
                             pipeline_uid="parallel_trace", pipeline_name="parallel_trace", step_number=2,
                             cache_config=CacheConfig(use_cow_snapshots=cow))
    _ConcurrentScaler.threads = []
    _ConcurrentScaler.barrier = threading.Barrier(2) if parallel else None
    step = {"branch": [[{"concat_transform": [_ConcurrentScaler(), PCA(2)]}], [_ConcurrentScaler()]],
            "parallel": parallel, "n_jobs": 2 if parallel else 1}
    executor = PipelineExecutor(runner, save_charts=False)
    executor.step_number = 2
    artifacts = []
    try:
        result = executor._execute_single_step(step, dataset, context, runtime, all_artifacts=artifacts)
    finally:
        _ConcurrentScaler.barrier = None
    assert len(set(_ConcurrentScaler.threads)) == (2 if parallel else 1)
    branches = result.custom["branch_contexts"]
    assert len(branches) == 2
    for branch, oracle in zip(branches, expected, strict=True):
        restored = copy.deepcopy(dataset)
        BranchController()._restore_features(restored, branch["features_snapshot"], use_cow=cow)
        np.testing.assert_allclose(restored.x({}), oracle, rtol=1e-5, atol=1e-6)
        assert branch["chain_snapshot"] is not None
    records = registry.get_artifacts_for_step("parallel_trace", 2)
    trace_ids = [aid for step in recorder.trace.steps for aid in step.artifacts.artifact_ids]
    assert len(records) == 3
    assert len(trace_ids) == len(set(trace_ids)) == 3
    assert set(trace_ids) == {record.artifact_id for record in records} == {a["artifact_id"] for a in artifacts}
    for record in records:
        owner, = [step for step in recorder.trace.steps if record.artifact_id in step.artifacts.artifact_ids]
        assert owner.branch_path == ([7] if ancestor else []) + record.branch_path[-1:]
        assert record.branch_path == owner.branch_path
    assert recorder.current_step is None and recorder.current_branch_path() == ([7] if ancestor else [])
    with pytest.raises(RuntimeError, match="no step is active"):
        recorder.record_artifact("unowned")


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("refit", [False, True])
def test_parallel_supervised_models_match_fold_oracles_and_archive(tmp_path, parallel, refit):
    dataset = _dataset()
    X = dataset.x({}).copy()
    y = dataset.y({}).ravel().copy()
    folds = KFold(2, shuffle=True, random_state=17)
    definitions = [(3, StandardScaler(), 1.), (4, MinMaxScaler(), 2.), (5, PCA(2), 3.)]
    branches = [[SelectKBest(f_regression, k=k), clone(transform), Ridge(alpha=alpha)] for k, transform, alpha in definitions]
    with nirs4all.run([folds, {"branch": branches, "parallel": parallel, "n_jobs": 2 if parallel else 1}], dataset,
                      engine="legacy", refit=refit, workspace_path=tmp_path / "workspace", save_charts=False, verbose=0) as result:
        rows = result.predictions.filter_predictions(load_arrays=True)
        cv_scores = []
        for branch_id, (k, transform, alpha) in enumerate(definitions):
            test_predictions = []
            expected_oof = np.empty(24)
            for train, val in folds.split(X[:24]):
                fitted = make_pipeline(SelectKBest(f_regression, k=k), clone(transform), Ridge(alpha=alpha)).fit(X[train], y[train])
                expected_oof[val] = fitted.predict(X[val])
                test_predictions.append(fitted.predict(X[24:]))
            val_rows = [row for row in rows if row["branch_id"] == branch_id and row["partition"] == "val" and str(row["fold_id"]) in ("0", "1")]
            assert len(val_rows) == 2
            for row in val_rows:
                np.testing.assert_allclose(np.asarray(row["y_pred"]).ravel(), expected_oof[list(row["sample_indices"])], rtol=1e-5, atol=1e-6)
            test_rows = [row for row in rows if row["branch_id"] == branch_id and row["partition"] == "test" and row["fold_id"] == "avg"]
            assert len(test_rows) == 1
            np.testing.assert_allclose(np.asarray(test_rows[0]["y_pred"]).ravel(), np.mean(test_predictions, axis=0), rtol=1e-5, atol=1e-6)
            cv_scores.append(np.mean([np.sqrt(np.mean((y[val] - expected_oof[val]) ** 2)) for _, val in folds.split(X[:24])]))
        best_branch = int(np.argmin(cv_scores))
        k, transform, alpha = definitions[best_branch]
        expected = make_pipeline(SelectKBest(f_regression, k=k), clone(transform), Ridge(alpha=alpha)).fit(X[:24], y[:24]).predict(X[24:]) if refit else np.mean([
            make_pipeline(SelectKBest(f_regression, k=k), clone(transform), Ridge(alpha=alpha)).fit(X[train], y[train]).predict(X[24:])
            for train, _ in folds.split(X[:24])
        ], axis=0)
        archive = result.export(tmp_path / "parallel.n4a")
        replay = nirs4all.predict(archive, X[24:], engine="legacy", verbose=0)
        np.testing.assert_allclose(np.asarray(replay.y_pred).ravel(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("refit", [False, True])
def test_physical_branch_substeps_replay_workspace_and_archive(tmp_path, parallel, refit):
    """Replay real fitted substeps and tuples without a fold-preprocessing wrapper."""
    dataset = _dataset()
    X, y = dataset.x({}).copy(), dataset.y({}).ravel().copy()
    prefix = StandardScaler().fit(X[:24]).transform(X).astype(np.float32)
    specifications = [
        [{"concat_transform": [StandardScaler(), PCA(2)]}, PCA(3)],
        [MinMaxScaler(), {"concat_transform": [StandardScaler(), PCA(2)]}],
        [StandardScaler()],
    ]
    folds = KFold(2, shuffle=True, random_state=17)
    expected_tests, expected_final, scores = [], [], []
    for branch_id, steps in enumerate(specifications):
        current = prefix.copy()
        for step in steps:
            if isinstance(step, dict):
                current = np.hstack([clone(op).fit(current[:24]).transform(current) for op in step["concat_transform"]]).astype(np.float32)
            else:
                current = clone(step).fit(current[:24]).transform(current).astype(np.float32)
        test_folds, fold_scores = [], []
        for train, val in folds.split(current[:24]):
            fitted = Ridge(alpha=branch_id + 1).fit(current[train], y[train])
            test_folds.append(fitted.predict(current[24:]))
            fold_scores.append(np.sqrt(np.mean((fitted.predict(current[val]) - y[val]) ** 2)))
        expected_tests.append(np.mean(test_folds, axis=0))
        expected_final.append(Ridge(alpha=branch_id + 1).fit(current[:24], y[:24]).predict(current[24:]))
        scores.append(np.mean(fold_scores))
    # Legacy unsupervised-only stages retain their declared global train fit;
    # this oracle tests physical artifact replay, independently of CV wrappers.
    branches = [steps + [Ridge(alpha=i + 1)] for i, steps in enumerate(specifications)]
    with nirs4all.run([StandardScaler(), folds, {"branch": branches, "parallel": parallel, "n_jobs": 2 if parallel else 1}], dataset,
                      engine="legacy", refit=refit, workspace_path=tmp_path / "workspace", save_charts=False, verbose=0) as result:
        rows = result.predictions.filter_predictions(partition="test", fold_id="avg", load_arrays=True)
        assert len(rows) == 3
        for row in rows:
            np.testing.assert_allclose(np.asarray(row["y_pred"]).ravel(), expected_tests[row["branch_id"]], rtol=1e-5, atol=1e-6)
        winner = int(np.argmin(scores))
        expected = expected_final[winner] if refit else expected_tests[winner]
        before = nirs4all.predict(result.best, X[24:], engine="legacy", workspace_path=tmp_path / "workspace", verbose=0)
        np.testing.assert_allclose(np.asarray(before.y_pred).ravel(), expected, rtol=1e-5, atol=1e-6)
        archive = result.export(tmp_path / "physical_substeps.n4a")
        after = nirs4all.predict(archive, X[24:], engine="legacy", verbose=0)
        np.testing.assert_allclose(np.asarray(after.y_pred).ravel(), expected, rtol=1e-5, atol=1e-6)
