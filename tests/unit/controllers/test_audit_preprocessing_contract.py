"""Scientific support and refusal contracts for fold-local legacy preprocessing."""

from copy import deepcopy

import numpy as np
import pytest
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import nirs4all
from nirs4all.controllers.transforms.transformer import TransformerMixinController
from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.augmentation import LocalMixupAugmenter, MixupAugmenter
from nirs4all.operators.augmentation.native import NativeRoleAugmenter
from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
from nirs4all.pipeline.execution.preprocessing import FoldPreprocessing
from nirs4all.pipeline.steps.step_runner import StepRunner
from nirs4all.utils import framework


def _dataset(sources=1):
    rng = np.random.default_rng(37)
    X = [rng.normal(size=(30, 8 + 2 * i)) for i in range(sources)]
    y = rng.normal(size=30)
    dataset = SpectroDataset("cv_contract")
    dataset.add_samples(X[0][:24] if sources == 1 else [x[:24] for x in X], {"partition": "train"})
    dataset.add_samples(X[0][24:] if sources == 1 else [x[24:] for x in X], {"partition": "test"})
    dataset.add_targets(y)
    return dataset, X, y


@pytest.mark.parametrize("kind", ["python", "local_python", "native", "local_native", "wrapped_native", "pipeline"])
@pytest.mark.parametrize("balanced", [False, True])
def test_mixup_refuses_before_mutating_labels_or_samples(kind, balanced):
    from n4m import roles

    operators = {
        "python": MixupAugmenter(random_state=3),
        "local_python": LocalMixupAugmenter(random_state=3),
        "native": roles.Mixup(seed=3),
        "local_native": roles.LocalMixup(seed=3),
        "wrapped_native": NativeRoleAugmenter(roles.Mixup(seed=3)),
        "pipeline": make_pipeline(StandardScaler(), MixupAugmenter(random_state=3)),
    }
    dataset, _, _ = _dataset()
    before = deepcopy(dataset)
    config = {"transformers": [FunctionTransformer(), operators[kind]], "selection": "all", "count": 2}
    if balanced:
        config.update(balance="y", max_factor=2)
    with pytest.raises(RuntimeError, match="joint X/y mixing"):
        StepRunner(verbose=0).execute({"sample_augmentation": config}, dataset, ExecutionContext(), RuntimeContext())
    np.testing.assert_array_equal(dataset.x({}), before.x({}))
    np.testing.assert_array_equal(dataset.y({}), before.y({}))
    assert dataset._indexer.df.equals(before._indexer.df)


@pytest.mark.parametrize("operator", [MixupAugmenter(random_state=3), LocalMixupAugmenter(random_state=3)])
def test_bare_mixup_pipeline_refuses_but_direct_x_only_api_remains(operator):
    dataset, X, _ = _dataset()
    with pytest.raises(ValueError, match="unmixed labels"):
        TransformerMixinController.validate_target_preserving_operator(operator)
    assert operator.fit_transform(X[0]).shape == X[0].shape
    np.testing.assert_allclose(dataset.x({}), X[0], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("engine", ["legacy", "dag-ml"])
@pytest.mark.parametrize("kind", ["python", "local_python", "native", "local_native"])
def test_public_sample_augmentation_refuses_unmixed_targets(tmp_path, engine, kind):
    from n4m import roles

    operator = {
        "python": MixupAugmenter(random_state=3), "local_python": LocalMixupAugmenter(random_state=3),
        "native": roles.Mixup(seed=3), "local_native": roles.LocalMixup(seed=3),
    }[kind]
    dataset, _, _ = _dataset()
    before = deepcopy(dataset)
    pipeline = [{"sample_augmentation": {"transformers": [FunctionTransformer(), operator], "count": 2}}, KFold(2), Ridge()]
    with pytest.raises((ValueError, RuntimeError), match="joint X/y mixing"):
        nirs4all.run(pipeline, dataset, engine=engine, workspace_path=tmp_path, refit=False, verbose=0, save_charts=False)
    np.testing.assert_array_equal(dataset.x({}), before.x({}))
    np.testing.assert_array_equal(dataset.y({}), before.y({}))
    assert dataset._indexer.df.equals(before._indexer.df)


@pytest.mark.parametrize("merge_position", ["before_selector", "after_selector"])
@pytest.mark.parametrize("refit", [False, True])
def test_multisource_merge_cv_matches_independent_fold_oracle_and_replays(tmp_path, merge_position, refit):
    dataset, sources, y = _dataset(sources=2)
    selector = SelectKBest(f_regression, k=3)
    merge = {"merge_sources": "concat"}
    stages = [StandardScaler(), merge, selector] if merge_position == "before_selector" else [StandardScaler(), selector, merge]
    expected_oof = np.zeros(24)
    expected_test = []

    def fit_predict(train, evaluate):
        transformed_train, transformed_evaluate = [], []
        for source in sources:
            preprocessor = StandardScaler() if merge_position == "before_selector" else make_pipeline(StandardScaler(), SelectKBest(f_regression, k=3))
            preprocessor.fit(source[train], y[train])
            transformed_train.append(preprocessor.transform(source[train]))
            transformed_evaluate.append(preprocessor.transform(source[evaluate]))
        train_x = np.concatenate(transformed_train, axis=1)
        eval_x = np.concatenate(transformed_evaluate, axis=1)
        if merge_position == "before_selector":
            selection = SelectKBest(f_regression, k=3).fit(train_x, y[train])
            train_x, eval_x = selection.transform(train_x), selection.transform(eval_x)
        return Ridge().fit(train_x, y[train]).predict(eval_x)

    for train, val in KFold(3).split(sources[0][:24]):
        expected_oof[val] = fit_predict(train, val)
        expected_test.append(fit_predict(train, np.arange(24, 30)))
    with nirs4all.run(stages + [KFold(3), Ridge()], dataset, engine="legacy", workspace_path=tmp_path / "ws",
                      save_charts=False, verbose=0, refit=refit) as result:
        for record in result.predictions.filter_predictions(partition="val", load_arrays=True):
            if record["fold_id"] not in ("avg", "w_avg", "final"):
                np.testing.assert_allclose(record["y_pred"].ravel(), expected_oof[record["sample_indices"]], rtol=1e-5, atol=1e-6)
        expected = fit_predict(np.arange(24), np.arange(24, 30)) if refit else np.mean(expected_test, axis=0)
        archive = result.export(tmp_path / "sources.n4a")
        raw = np.concatenate([source[24:] for source in sources], axis=1)
        np.testing.assert_allclose(nirs4all.predict(archive, raw, engine="legacy", verbose=0).y_pred.ravel(), expected, rtol=1e-5, atol=1e-6)
        if not refit:
            replay = nirs4all.predict(result.best, raw, engine="legacy", workspace_path=tmp_path / "ws", verbose=0)
            np.testing.assert_allclose(replay.y_pred.ravel(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("layout", ["2d", "2d_interleaved", "3d", "3d_transpose"])
def test_fold_replay_preserves_multiple_processing_layouts(layout):
    dataset, X, y = _dataset()
    dataset.update_features(source_processings=[""], features=[X[0] * 2 + 1], processings=["second"], source=0)
    context = ExecutionContext().with_processing([["raw", "second"]])
    plan = FoldPreprocessing.capture(dataset, context)
    plan.replay.steps = [(1, StandardScaler()), (2, SelectKBest(f_regression, k=3))]
    plan.active = True
    train = np.arange(12)
    replay = plan.prepare_fold(dataset, context.with_layout(layout), train)
    raw = plan.raw_features(np.arange(24, 30), layout)
    actual = replay.transform(raw)
    expected = []
    for values in (X[0], X[0] * 2 + 1):
        stage = make_pipeline(StandardScaler(), SelectKBest(f_regression, k=3)).fit(values[train], y[train])
        expected.append(stage.transform(values[24:]))
    expected_array = np.stack(expected, axis=1)
    if layout in ("2d_interleaved", "3d_transpose"):
        expected_array = expected_array.transpose(0, 2, 1)
    if layout.startswith("2d"):
        expected_array = expected_array.reshape(6, -1)
    np.testing.assert_allclose(actual, expected_array, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("learned_before", [False, True])
def test_post_snapshot_augmentation_is_explicitly_refused(tmp_path, learned_before):
    dataset, _, _ = _dataset()
    selector = SelectKBest(f_regression, k=3)
    augmentation = {"sample_augmentation": {"transformers": [FunctionTransformer()], "count": 1}}
    prefix = [selector, augmentation] if learned_before else [StandardScaler(), augmentation, selector]
    with pytest.raises(RuntimeError, match="place sample_augmentation before"):
        nirs4all.run(prefix + [KFold(3), Ridge()], dataset, engine="legacy", workspace_path=tmp_path,
                     refit=False, verbose=0, save_charts=False)


def test_supervised_branch_feature_merge_is_explicitly_refused(tmp_path):
    dataset, _, _ = _dataset()
    stages = [{"branch": [[SelectKBest(f_regression, k=3)], [StandardScaler()]]}, {"merge": "features"}, KFold(3), Ridge()]
    with pytest.raises(RuntimeError, match="branch feature merge is unsupported"):
        nirs4all.run(stages, dataset, engine="legacy", workspace_path=tmp_path, refit=False, verbose=0, save_charts=False)


@pytest.mark.parametrize("per_source", [False, True])
def test_supervised_source_branch_routing_is_explicitly_refused(tmp_path, per_source):
    dataset, _, _ = _dataset(sources=2)
    selection = [SelectKBest(f_regression, k=3)]
    stages = {"source_0": selection, "source_1": selection} if per_source else selection
    pipeline = [{"branch": {"by_source": True, "steps": stages}}, {"merge_sources": "concat"}, KFold(3), Ridge()]
    with pytest.raises(RuntimeError, match="by_source branch routing is unsupported"):
        nirs4all.run(pipeline, dataset, engine="legacy", workspace_path=tmp_path, refit=False, verbose=0, save_charts=False)


def test_augmentation_before_preprocessing_keeps_fold_children_local(tmp_path):
    dataset, X, y = _dataset()
    stages = [{"sample_augmentation": {"transformers": [FunctionTransformer()], "count": 1, "selection": "all"}},
              StandardScaler(), SelectKBest(f_regression, k=3), KFold(3), Ridge()]
    expected_oof = np.zeros(24)
    for train, val in KFold(3).split(X[0][:24]):
        duplicated_x, duplicated_y = np.tile(X[0][train], (2, 1)), np.tile(y[train], 2)
        model = make_pipeline(StandardScaler(), SelectKBest(f_regression, k=3), Ridge()).fit(duplicated_x, duplicated_y)
        expected_oof[val] = model.predict(X[0][val])
    with nirs4all.run(stages, dataset, engine="legacy", workspace_path=tmp_path, refit=False, verbose=0, save_charts=False) as result:
        for record in result.predictions.filter_predictions(partition="val", load_arrays=True):
            if record["fold_id"] not in ("avg", "w_avg", "final"):
                np.testing.assert_allclose(record["y_pred"].ravel(), expected_oof[record["sample_indices"]], rtol=1e-5, atol=1e-6)


@framework("tensorflow")
def _sum_network(input_shape, params=None):
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape), tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, kernel_initializer="ones", bias_initializer="zeros"),
    ])


@pytest.mark.tensorflow
def test_tensorflow_supervised_cv_preserves_layout_and_archive(tmp_path):
    pytest.importorskip("tensorflow")
    dataset, X, y = _dataset()
    stages = [StandardScaler(), SelectKBest(f_regression, k=3), KFold(2),
              {"model": _sum_network, "train_params": {"epochs": 1, "learning_rate": 0.0, "verbose": 0}}]
    expected_oof, expected_test = np.zeros(24), []
    for train, val in KFold(2).split(X[0][:24]):
        features = make_pipeline(StandardScaler(), SelectKBest(f_regression, k=3)).fit(X[0][train], y[train])
        expected_oof[val] = features.transform(X[0][val]).sum(axis=1)
        expected_test.append(features.transform(X[0][24:]).sum(axis=1))
    with nirs4all.run(stages, dataset, engine="legacy", workspace_path=tmp_path / "ws", refit=False, verbose=0, save_charts=False) as result:
        for record in result.predictions.filter_predictions(partition="val", load_arrays=True):
            if record["fold_id"] not in ("avg", "w_avg", "final"):
                np.testing.assert_allclose(record["y_pred"].ravel(), expected_oof[record["sample_indices"]], rtol=1e-5, atol=1e-6)
        archive = result.export(tmp_path / "neural.n4a")
        replay = nirs4all.predict(archive, X[0][24:], engine="legacy", verbose=0)
        np.testing.assert_allclose(replay.y_pred.ravel(), np.mean(expected_test, axis=0), rtol=1e-5, atol=1e-6)


def test_resampler_keeps_skipped_source_headers_and_features():
    """CTD-10: the local skip guard preserves an unselected source's axis."""
    from nirs4all.operators.transforms import Resampler

    dataset, _, _ = _dataset(sources=2)
    dataset._features.sources[0].set_headers([str(v) for v in np.linspace(1000, 1200, 8)], unit="cm-1")
    dataset._features.sources[1].set_headers([str(v) for v in np.linspace(1100, 2400, 10)], unit="nm")
    headers = dataset.headers(1)
    original = dataset.x({}, "3d", concat_source=False)[1].copy()
    context = ExecutionContext().with_processing([["raw"], []])
    context.selector.partition = None
    result = StepRunner(verbose=0).execute(Resampler(target_wavelengths=np.linspace(1020, 1180, 4)), dataset, context, RuntimeContext())
    assert dataset.headers(1) == headers
    assert dataset.header_unit(1) == "nm"
    assert result.updated_context.selector.processing[1] == []
    np.testing.assert_array_equal(dataset.x({}, "3d", concat_source=False)[1], original)
