"""Scientific correctness and replay regressions from the October 2026 audit."""

from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.controllers.data.branch import BranchController
from nirs4all.controllers.data.merge import MergeController, detect_disjoint_branches
from nirs4all.controllers.transforms.transformer import TransformerMixinController
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.operators.data.merge import MergeConfig
from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
from nirs4all.pipeline.execution.executor import PipelineExecutor
from nirs4all.pipeline.steps.step_runner import StepRunner
from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry


def _context():
    context = ExecutionContext()
    context.selector.partition = None
    return context


def _dataset(n_train=12, n_test=4, n_features=6):
    rng = np.random.default_rng(19)
    dataset = SpectroDataset("audit")
    dataset.add_samples(rng.normal(size=(n_train, n_features)), {"partition": "train"})
    dataset.add_targets(rng.normal(size=n_train))
    if n_test:
        dataset.add_samples(rng.normal(size=(n_test, n_features)), {"partition": "test"})
        dataset.add_targets(rng.normal(size=n_test))
    return dataset


@pytest.mark.parametrize("first,second", [(PCA(2), PCA(4)), (StandardScaler(), StandardScaler(with_std=False))])
def test_stateful_cache_distinguishes_parameters_and_fit_cohort(tmp_path, first, second):
    """CTM-01: the same input and chain must not erase a parameter sweep."""
    dataset = _dataset()
    registry = ArtifactRegistry(workspace=tmp_path, dataset="audit", pipeline_id="p")
    runtime = RuntimeContext(artifact_registry=registry, step_number=1, pipeline_name="p")
    context = _context()
    controller = TransformerMixinController()
    runner = StepRunner()
    runner.execute(first, deepcopy(dataset), context, runtime)
    runtime.reset_processing_counter()
    assert controller._try_cache_lookup(runtime, context, dataset, type(first).__name__, 0, first) is not None
    assert controller._try_cache_lookup(runtime, context, dataset, type(second).__name__, 0, second) is None
    assert controller._try_cache_lookup(runtime, context, dataset, type(first).__name__, 0, first, fit_on_all=True) is None
    context.custom["sample_partition"] = {"sample_indices": [0, 2, 4]}
    assert controller._try_cache_lookup(runtime, context, dataset, type(first).__name__, 0, first) is None


def test_stacking_same_named_models_stay_in_their_own_branch():
    """CTD-01: same-named source models produce distinct OOF columns."""
    dataset = _dataset(n_train=4, n_test=0)
    predictions = Predictions()
    for branch_id, values in [(0, [1., 2., 3., 4.]), (1, [10., 20., 30., 40.])]:
        predictions.add_prediction(dataset_name="audit", model_name="Ridge", branch_id=branch_id,
                                   partition="val", sample_indices=[0, 1, 2, 3], y_pred=np.array(values),
                                   y_true=np.arange(4.), fold_id=0, step_idx=1)
    context = _context().with_step_number(3)
    controller = MergeController()
    config = MergeConfig(collect_predictions=True)
    for branch_id, expected in [(0, [1., 2., 3., 4.]), (1, [10., 20., 30., 40.])]:
        result = controller._collect_branch_predictions_oof(dataset, context, predictions, ["Ridge"], branch_id, config)
        assert result is not None
        np.testing.assert_allclose(result["Ridge"], expected)


def test_disjoint_prediction_merge_uses_ids_and_full_stored_row_count():
    """CTD-02: excluded gaps and test rows retain their absolute addresses."""
    dataset = _dataset(n_train=4, n_test=2)
    dataset._indexer.mark_excluded([0])
    predictions = Predictions()
    predictions.add_prediction(dataset_name="audit", model_name="Ridge", branch_id=0, partition="val",
                               sample_indices=[1, 3], y_pred=np.array([11., 13.]), fold_id=0, step_idx=1)
    for fold_id, value in [(0, 40.), (1, 44.)]:
        predictions.add_prediction(dataset_name="audit", model_name="Ridge", branch_id=0, partition="test",
                                   sample_indices=[4], y_pred=np.array([value]), fold_id=fold_id, step_idx=1)
    context = _context().with_step_number(3)
    context.custom["sample_partition"] = {"sample_indices": [1, 3], "all_sample_indices": [1, 3, 4]}
    analysis = detect_disjoint_branches([{"branch_id": 0, "context": context}])
    assert analysis.branch_sample_indices == {0: [1, 3, 4]}
    merged = MergeController()._collect_disjoint_oof_predictions(
        dataset, context, predictions, "train", {0: [{"name": "Ridge"}]}, analysis.branch_sample_indices, dataset.num_samples, 1)
    assert merged.shape == (6, 1)
    np.testing.assert_allclose(merged[[1, 3, 4], 0], [11., 13., 42.])
    assert np.isnan(merged[0, 0])


@pytest.mark.parametrize("mode", ["by_metadata", "by_tag", "by_filter"])
def test_separation_models_keep_test_membership(mode):
    """CTD-03: all separation modes expose test membership to model selection."""
    dataset = _dataset(n_train=12, n_test=4)
    dataset.add_metadata(np.array([[v] for v in ["A", "B"] * 8]), headers=["site"])
    dataset.add_tag("audit_group", "bool")
    dataset.set_tag("audit_group", list(range(16)), [True, False] * 8)
    if mode == "by_filter":
        from nirs4all.operators.filters import XOutlierFilter
        spec = {"by_filter": XOutlierFilter(method="isolation_forest", contamination=0.3, random_state=19), "steps": []}
    else:
        spec = {mode: "site" if mode == "by_metadata" else "audit_group", "steps": []}
    runtime = RuntimeContext(step_runner=StepRunner(), step_number=1)
    context = _context()
    result = StepRunner().execute({"branch": spec}, dataset, context, runtime)
    for branch in result.updated_context.custom["branch_contexts"]:
        partition = branch["context"].custom["sample_partition"]
        assert partition["sample_indices"] == partition["all_sample_indices"]
    routed = {sid for branch in result.updated_context.custom["branch_contexts"]
              for sid in branch["context"].custom["sample_partition"]["sample_indices"]}
    assert routed == set(range(16))
    from nirs4all.controllers.models import SklearnModelController
    test_count = 0
    for branch in result.updated_context.custom["branch_contexts"]:
        model_context = branch["context"]
        _, _, X_test, _, _, _ = SklearnModelController().get_xy(dataset, model_context)
        test_count += len(X_test)
    assert test_count == 4


@pytest.mark.parametrize("syntax", ["merge_sources", "merge"])
@pytest.mark.parametrize("strategy", ["concat", "stack"])
def test_source_merges_replace_sources_and_keep_excluded_rows(syntax, strategy):
    """CTD-04/07: merging creates one source without dropping stored rows."""
    dataset = SpectroDataset("multi")
    first = np.arange(24.).reshape(6, 4)
    second = first + 100
    dataset.add_samples([first, second], {"partition": "train"})
    dataset.add_targets(np.arange(6.))
    dataset._indexer.mark_excluded([0])
    step = {syntax: strategy if syntax == "merge_sources" else {"sources": strategy}}
    result = StepRunner().execute(step, dataset, _context(), RuntimeContext(step_number=1))
    assert dataset.n_sources == 1
    assert result.updated_context.selector.processing == [list(dataset.features_processings(0))]
    if strategy == "concat":
        np.testing.assert_array_equal(dataset.x({}, include_excluded=True), np.concatenate([first, second], axis=1))
    else:
        assert dataset.x({}, "3d", include_excluded=True).shape == (6, 2, 4)


@pytest.mark.parametrize("use_cow", [False, True])
def test_branch_snapshot_isolates_all_sample_state(use_cow):
    """CTD-05: restoration includes exclusions, rows, targets, tags and folds."""
    dataset = _dataset(n_train=4, n_test=0)
    dataset.set_folds([([0, 1], [2, 3])])
    original = dataset.x({}).copy()
    controller = BranchController()
    snapshot = controller._snapshot_features(dataset, use_cow)
    dataset._indexer.mark_excluded([0])
    dataset.add_tag("branch_only", "bool")
    dataset.add_samples(np.ones((1, 6)), {"partition": "train"})
    dataset.add_targets([99.])
    dataset.set_folds([([0, 1, 4], [2, 3])])
    controller._restore_features(dataset, snapshot, use_cow)
    assert dataset.num_samples == 4
    assert dataset.y({}).shape[0] == 4
    assert not dataset.has_tag("branch_only")
    assert dataset.folds == [([0, 1], [2, 3])]
    np.testing.assert_array_equal(dataset.x({}), original)
    # Executor must use the same complete restoration after the branch step.
    dataset._indexer.mark_excluded([1])
    PipelineExecutor(Mock())._restore_branch_snapshot(dataset, {"features_snapshot": snapshot, "use_cow": use_cow}, None)
    np.testing.assert_array_equal(dataset.x({}), original)
    controller._release_snapshot(snapshot, use_cow)


def test_feature_merge_includes_actual_pre_branch_features():
    """CTD-09: include_original preserves the input, rather than the last branch."""
    from sklearn.preprocessing import MinMaxScaler
    dataset = _dataset(n_train=8, n_test=0)
    original = dataset.x({}).copy()
    runner = StepRunner()
    executor = PipelineExecutor(runner)
    runtime = RuntimeContext(step_runner=runner)
    _, merged_dataset = executor._execute_steps(
        [{"branch": [[StandardScaler()], [MinMaxScaler()]]},
         {"merge": {"features": "all", "include_original": True}}],
        dataset, _context(), runtime, Predictions(), [],
    )
    merged = merged_dataset.x({})
    assert merged.shape == (8, 18)
    np.testing.assert_allclose(merged[:, :6], original)


@pytest.mark.parametrize("preprocessing", ["concat_transform", "resampler", "transfer", "transfer_augmentation"])
def test_tuple_preprocessing_artifacts_predict_and_export(tmp_path, preprocessing, monkeypatch):
    """CTD-06: tuple artifacts are linked to the chain and replay by saved name."""
    dataset = _dataset(n_train=24, n_test=6, n_features=8)
    if preprocessing == "concat_transform":
        step = {"concat_transform": [PCA(2), StandardScaler()]}
    elif preprocessing.startswith("transfer"):
        from types import SimpleNamespace

        from nirs4all.analysis import TransferPreprocessingSelector
        spec = {"feature_augmentation": ["snv", "msc"]} if preprocessing == "transfer_augmentation" else "msc"
        recommendation = SimpleNamespace(name="msc", transfer_score=1.0, improvement_pct=0.0)
        selection = SimpleNamespace(best=recommendation, ranking=[], top_k=lambda n: [], to_pipeline_spec=lambda **kw: spec)
        monkeypatch.setattr(TransferPreprocessingSelector, "fit", lambda *args, **kw: selection)
        step = {"auto_transfer_preproc": {"verbose": 0}}
    else:
        from nirs4all.operators.transforms import Resampler
        dataset._features.sources[0]._header_mgr.set_headers([str(value) for value in np.linspace(4000, 5000, 8)], unit="cm-1")
        step = Resampler(target_wavelengths=np.linspace(4000, 5000, 4))
    X_test = dataset.x({"partition": "test"})
    prediction_data = SpectroDataset("prediction")
    prediction_data.add_samples(X_test, {"partition": "test"}, headers=dataset.headers(0), header_unit=dataset.header_unit(0))
    prediction_data.add_targets(np.linspace(0.13, 0.83, len(X_test)))
    with nirs4all.run([step, KFold(3), Ridge()], dataset, engine="legacy", workspace_path=tmp_path / "workspace",
                      verbose=0, save_charts=False, refit=False) as result:
        best = result.best
        expected = result.predictions.filter_predictions(model_name=best["model_name"], partition="test", fold_id="avg", load_arrays=True)[0]["y_pred"]
        predicted = nirs4all.predict(best, prediction_data, engine="legacy", workspace_path=tmp_path / "workspace", verbose=0)
        np.testing.assert_allclose(predicted.y_pred.ravel(), expected.ravel(), rtol=1e-5, atol=1e-6)
        archive = result.export(tmp_path / "model.n4a")
        replayed = nirs4all.predict(archive, prediction_data, engine="legacy", verbose=0)
        np.testing.assert_allclose(replayed.y_pred.ravel(), expected.ravel(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("refit", [False, True])
@pytest.mark.parametrize("selector_kind", ["select_k_best", "mcuve"])
def test_supervised_cv_is_fold_local_and_replays(tmp_path, refit, selector_kind, nested):
    """EXE-01: match independently fitted folds and round-trip their selectors."""
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.pipeline import make_pipeline

    from nirs4all.operators.transforms.feature_selection import MCUVE

    rng = np.random.default_rng(19)
    X = rng.normal(size=(48, 80))
    y = rng.normal(size=48)
    selector = SelectKBest(f_regression, k=5) if selector_kind == "select_k_best" else MCUVE(n_components=2, n_iterations=12, random_state=17)
    feature_steps = [StandardScaler(), selector, StandardScaler()]
    steps = ([{"branch": [feature_steps]}] if nested else feature_steps) + [KFold(3), Ridge()]
    dataset = SpectroDataset("supervised_cv")
    dataset.add_samples(X[:36], {"partition": "train"})
    dataset.add_samples(X[36:], {"partition": "test"})
    dataset.add_targets(y)
    expected_test = []
    expected_oof = np.zeros(36)
    from sklearn.base import clone
    for train, val in KFold(3).split(X[:36]):
        model = make_pipeline(StandardScaler(), clone(selector), StandardScaler(), Ridge()).fit(X[train], y[train])
        expected_oof[val] = model.predict(X[val])
        expected_test.append(model.predict(X[36:]))
    with nirs4all.run(steps, dataset, engine="legacy", workspace_path=tmp_path / "workspace", verbose=0,
                      save_charts=False, refit=refit) as result:
        vals = result.predictions.filter_predictions(partition="val", load_arrays=True)
        for record in vals:
            if record["fold_id"] not in ("avg", "w_avg", "final"):
                np.testing.assert_allclose(record["y_pred"].ravel(), expected_oof[record["sample_indices"]], rtol=1e-5, atol=1e-6)
        archive = result.export(tmp_path / "supervised.n4a")
        expected = (make_pipeline(StandardScaler(), clone(selector), StandardScaler(), Ridge()).fit(X[:36], y[:36]).predict(X[36:])
                    if refit else np.mean(expected_test, axis=0))
        predicted = nirs4all.predict(archive, X[36:], engine="legacy", verbose=0)
        np.testing.assert_allclose(predicted.y_pred.ravel(), expected, rtol=1e-5, atol=1e-6)
        # Exercise a trace-bearing bundle as well as index-based export replay.
        from types import SimpleNamespace

        from nirs4all.pipeline.bundle.loader import BundleLoader
        from nirs4all.pipeline.trace.execution_trace import StepExecutionMode
        loader = BundleLoader(archive)
        model_index = loader.metadata.model_step_index
        trace_steps = [SimpleNamespace(step_index=i, operator_type="model" if i == model_index else "transform",
                                       branch_path=[], execution_mode=StepExecutionMode.TRAIN)
                       for i in range(1, model_index + 1)]
        loader.trace = SimpleNamespace(get_steps_up_to_model=lambda: trace_steps)
        np.testing.assert_allclose(loader.predict(X[36:]).ravel(), expected, rtol=1e-5, atol=1e-6)
        if not refit:
            predicted = nirs4all.predict(result.best, X[36:], engine="legacy", workspace_path=tmp_path / "workspace", verbose=0)
            np.testing.assert_allclose(predicted.y_pred.ravel(), expected, rtol=1e-5, atol=1e-6)


def test_supervised_cv_without_test_partition(tmp_path):
    from sklearn.feature_selection import SelectKBest, f_regression

    dataset = _dataset(n_train=24, n_test=0, n_features=12)
    with nirs4all.run([SelectKBest(f_regression, k=3), KFold(3), Ridge()], dataset, engine="legacy",
                      workspace_path=tmp_path, refit=False, verbose=0, save_charts=False) as result:
        rows = result.predictions.filter_predictions(partition="val", fold_id="avg", load_arrays=True)
        assert len(rows) == 1 and len(rows[0]["y_pred"]) == 24


def test_supervised_cv_meta_model_preserves_oof_features(tmp_path):
    from sklearn.feature_selection import SelectKBest, f_regression

    from nirs4all.operators.models.meta import MetaModel

    dataset = _dataset(n_train=36, n_test=6, n_features=12)
    with nirs4all.run([SelectKBest(f_regression, k=3), KFold(3), Ridge(), MetaModel(Ridge())], dataset,
                      engine="legacy", workspace_path=tmp_path, refit=False, verbose=0, save_charts=False) as result:
        base = result.predictions.filter_predictions(model_name="Ridge", partition="val", fold_id="avg", load_arrays=True)[0]
        meta = result.predictions.filter_predictions(partition="val", load_arrays=True)
        meta = [r for r in meta if "MetaModel" in r["model_name"] and r["fold_id"] not in ("avg", "w_avg")]
        by_id = dict(zip(base["sample_indices"], base["y_pred"].ravel(), strict=True))
        features = np.array([by_id[i] for i in range(36)]).reshape(-1, 1)
        y = dataset.y({"partition": "train"}).ravel()
        for record, (train, val) in zip(meta, KFold(3).split(features), strict=True):
            expected = Ridge().fit(features[train], y[train]).predict(features[val])
            np.testing.assert_allclose(record["y_pred"].ravel(), expected, rtol=1e-5, atol=1e-6)
