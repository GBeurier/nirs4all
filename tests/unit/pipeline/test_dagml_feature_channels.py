"""Channel-local transforms retain numerical meaning before 2D materialization."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from nirs4all.operators.transforms.concat import FeatureConcat
from nirs4all.pipeline.dagml_bridge import pipeline_to_dsl


@pytest.mark.parametrize("downstream", [RobustScaler(), PCA(n_components=2)])
def test_downstream_transform_fits_independently_per_channel(downstream):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(36, 6)) * np.arange(1, 7)
    held_out = rng.normal(size=(9, 6)) + 100
    pipeline = [{"feature_augmentation": [StandardScaler()], "action": "extend"}, downstream, {"model": Ridge()}]
    dsl = pipeline_to_dsl(pipeline)
    assert len(dsl["pipeline"]) == 2
    node = dsl["pipeline"][0]
    actual = FeatureConcat(**node["params"]).fit(X).transform(held_out)
    raw = clone(downstream).fit(X).transform(held_out)
    augmented = make_pipeline(StandardScaler(), clone(downstream)).fit(X).transform(held_out)
    np.testing.assert_allclose(actual, np.hstack([raw, augmented]), atol=1e-12)
    if isinstance(downstream, PCA):
        assert actual.shape[1] == 4  # A global PCA would silently emit only 2 columns.
    assert not hasattr(downstream, "n_features_in_")


def test_splitter_and_target_processing_remain_distinct_dsl_steps():
    splitter = KFold(n_splits=3)
    dsl = pipeline_to_dsl([
        {"feature_augmentation": [StandardScaler()]}, splitter,
        {"y_processing": MinMaxScaler()}, RobustScaler(), PCA(n_components=2), {"model": Ridge()},
    ])
    assert len(dsl["pipeline"]) == 4
    assert "KFold" in dsl["pipeline"][1]["class"]
    assert "y_processing" in dsl["pipeline"][2]
    assert [len(chain) for chain in dsl["pipeline"][0]["params"]["operations"]] == [2, 3]


def test_transform_cannot_retroactively_change_a_preceding_model():
    with pytest.raises(NotImplementedError, match="sequential model checkpoints"):
        pipeline_to_dsl([
            {"feature_augmentation": [StandardScaler()]}, {"model": Ridge()},
            RobustScaler(), {"model": Ridge(alpha=2)},
        ])


@pytest.mark.parametrize("action", ["extend", "add", "replace"])
def test_repeated_augmentation_keeps_stored_and_active_layers_distinct(action):
    X = np.random.default_rng(17).normal(size=(30, 6)).astype(np.float32)
    held_out = np.random.default_rng(18).normal(size=(8, 6)).astype(np.float32) + 40
    pipeline = [
        {"feature_augmentation": [StandardScaler()], "action": "extend"},
        {"feature_augmentation": [RobustScaler()], "action": action}, MinMaxScaler(), {"model": Ridge()},
    ]
    # Operation-major insertion order, all stored layers reach the 2D model.
    expected_chains = [[], [StandardScaler()], [RobustScaler()]]
    if action != "extend":
        expected_chains.append([StandardScaler(), RobustScaler()])
    active = range(2, len(expected_chains)) if action == "replace" else range(len(expected_chains))
    expected_blocks = []
    for index, chain in enumerate(expected_chains):
        operations = [*chain, *([MinMaxScaler()] if index in active else [])]
        expected_blocks.append(make_pipeline(*operations).fit(X).transform(held_out) if operations else held_out)
    node = pipeline_to_dsl(pipeline)["pipeline"][0]
    actual = FeatureConcat(**node["params"]).fit(X).transform(held_out)
    np.testing.assert_allclose(actual, np.hstack(expected_blocks), rtol=1e-6, atol=1e-6)


def test_repeated_replace_reactivates_previously_inactive_stored_layer():
    X = np.random.default_rng(19).normal(size=(30, 6))
    pipeline = [
        {"feature_augmentation": [StandardScaler()], "action": "replace"},
        {"feature_augmentation": [RobustScaler()], "action": "replace"}, MinMaxScaler(), {"model": Ridge()},
    ]
    node = pipeline_to_dsl(pipeline)["pipeline"][0]
    expected = np.hstack([
        MinMaxScaler().fit_transform(X), StandardScaler().fit_transform(X),
        make_pipeline(StandardScaler(), RobustScaler(), MinMaxScaler()).fit_transform(X),
    ])
    np.testing.assert_allclose(FeatureConcat(**node["params"]).fit_transform(X), expected, atol=1e-12)


@pytest.mark.parametrize("with_cv", [False, True])
def test_public_export_replay_matches_channel_local_oracle(tmp_path, with_cv):
    import nirs4all
    from nirs4all.data import SpectroDataset

    rng = np.random.default_rng(314159)
    X = rng.normal(size=(42, 8)) * np.arange(1, 9)
    y = X @ rng.normal(size=8)
    new_X = rng.normal(size=(7, 8))
    dataset = SpectroDataset("channels")
    dataset.add_samples(X, {"partition": "train"})
    dataset.add_targets(y)
    # Compare identical public storage precision, not float64 inputs against
    # the dataset's default float32 spectral materialization.
    X = dataset.x({"partition": "train"}, layout="2d")
    y = np.asarray(dataset.y({"partition": "train"}), dtype=float).ravel()
    new_X = new_X.astype(X.dtype)
    base = MinMaxScaler().fit(X)
    raw = base.transform(X)
    raw_new = base.transform(new_X)
    chains = [PCA(n_components=2), make_pipeline(StandardScaler(), PCA(n_components=2))]
    fitted = [chain.fit(raw, y) for chain in chains]
    expected_model = Ridge().fit(np.hstack([chain.transform(raw) for chain in fitted]), y)
    expected = expected_model.predict(np.hstack([chain.transform(raw_new) for chain in fitted]))
    pipeline = [MinMaxScaler(), {"feature_augmentation": [StandardScaler()]}, PCA(n_components=2)]
    if with_cv:
        pipeline.append(KFold(n_splits=3))
    result = nirs4all.run(
        [*pipeline, Ridge()],
        dataset, workspace_path=tmp_path / "workspace", verbose=0, save_charts=False,
    )
    try:
        assert result.execution_engine == "dag-ml"
        assert bool(np.isfinite(result.cv_best_score)) is with_cv
        archive = result.export(tmp_path / "channels.n4a")
        prediction = nirs4all.predict(archive, new_X, verbose=0)
        np.testing.assert_allclose(np.asarray(prediction.y_pred).ravel(), expected, atol=1e-10)
    finally:
        result.close()
