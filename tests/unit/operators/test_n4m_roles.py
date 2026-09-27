"""n4m generic roles (``n4m.roles``) inside nirs4all pipelines.

Transformers, selectors, regressors and splitters are scikit-learn objects and
run as they are; sample filters and augmenters reach the exclude / tag /
branch and ``sample_augmentation`` steps through small adapters. Every role
runs on the default and the legacy engine.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.operators.augmentation.native import NativeRoleAugmenter, as_augmenter
from nirs4all.operators.filters.native import NativeRoleFilter, as_sample_filter

roles = pytest.importorskip("n4m.roles")

pytestmark = pytest.mark.methods


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(80, 2))
    loadings = rng.normal(size=(2, 50))
    X = scores @ loadings + 1.0 + 0.05 * rng.normal(size=(80, 50))
    y = scores[:, 0] - 0.3 * scores[:, 1]
    return X, y


def run(pipeline, data, engine):
    return nirs4all.run(
        pipeline=pipeline,
        dataset=data,
        verbose=0,
        save_artifacts=False,
        save_charts=False,
        engine=engine,
    )


@pytest.mark.parametrize("engine", [None, "legacy"])
def test_native_regressor_matches_sklearn(data, engine):
    native = run([KFold(3), {"model": roles.PLSRegression(n_components=3)}], data, engine)
    reference = run([KFold(3), {"model": PLSRegression(n_components=3)}], data, engine)
    np.testing.assert_allclose(native.cv_best_score, reference.cv_best_score, rtol=1e-6)


@pytest.mark.parametrize("engine", [None, "legacy"])
@pytest.mark.parametrize(
    "steps",
    [
        [roles.SNV(), KFold(3)],
        [roles.VarianceFilter(top_k=20), KFold(3)],
        [{"exclude": roles.YOutlierFilter(threshold=1.5)}, KFold(3)],
        [{"tag": roles.HighLeverageFilter()}, KFold(3)],
        [roles.SPXYFold(n_splits=3)],
        [{"sample_augmentation": {"transformers": [roles.GaussianNoise(sigma=0.01)], "count": 2}}, KFold(3)],
        # Optional axis: the dataset has no wavelength headers.
        [{"sample_augmentation": {"transformers": [roles.WavelengthShift()], "count": 1}}, KFold(3)],
    ],
    ids=["transformer", "selector", "exclude", "tag", "splitter", "augmenter", "augmenter-axis"],
)
def test_roles_run_in_pipelines(data, engine, steps):
    result = run([*steps, {"model": roles.CPPLS(n_components=3)}], data, engine)
    assert np.isfinite(result.cv_best_score)


def test_sample_filter_adapter(data):
    X, y = data
    role = roles.YOutlierFilter(threshold=0.5)
    adapted = as_sample_filter(role)
    assert isinstance(adapted, NativeRoleFilter)
    assert adapted.exclusion_reason == "YOutlierFilter"
    mask = adapted.fit(X, y).get_mask(X, y)
    np.testing.assert_array_equal(mask, roles.YOutlierFilter(threshold=0.5).fit(X, y).get_mask(X, y))
    assert not mask.all()
    assert as_sample_filter(object()) is None


def test_augmenter_adapter_draws_a_new_seed_per_call(data):
    X, _ = data
    adapted = as_augmenter(roles.GaussianNoise(sigma=0.1, seed=3))
    assert isinstance(adapted, NativeRoleAugmenter)
    adapted.fit(X)
    first, second = adapted.transform(X), adapted.transform(X)
    np.testing.assert_array_equal(first, roles.GaussianNoise(sigma=0.1, seed=3).augment(X))
    np.testing.assert_array_equal(second, roles.GaussianNoise(sigma=0.1, seed=4).augment(X))
    assert as_augmenter(PLSRegression()) is not None
    assert not isinstance(as_augmenter(PLSRegression()), NativeRoleAugmenter)


def test_role_token_round_trips_through_serialization():
    from nirs4all.pipeline.config.component_serialization import deserialize_component, serialize_component

    assert serialize_component(roles.SNV()) == "n4m:preprocessing.scatter.snv"
    token = serialize_component(roles.CPPLS(n_components=3))
    assert token == {"class": "n4m:models.pls.cppls", "params": {"n_components": 3}}
    assert deserialize_component(token).get_params() == roles.CPPLS(n_components=3).get_params()
    assert isinstance(deserialize_component("n4m:preprocessing.scatter.snv"), roles.SNV)


@pytest.mark.parametrize("engine", [None, "legacy"])
def test_json_recipe_of_role_tokens_runs(data, engine, tmp_path):
    import json

    recipe = {
        "pipeline": [
            "n4m:preprocessing.scatter.snv",
            {"exclude": {"class": "n4m:filters.y_outlier", "params": {"threshold": 2.0}}},
            {"class": "sklearn.model_selection.KFold", "params": {"n_splits": 3}},
            {"model": {"class": "n4m:models.pls.cppls", "params": {"n_components": 3}}},
        ]
    }
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe), encoding="utf-8")
    direct = [
        roles.SNV(),
        {"exclude": roles.YOutlierFilter(threshold=2.0)},
        KFold(3),
        {"model": roles.CPPLS(n_components=3)},
    ]
    np.testing.assert_allclose(run(str(path), data, engine).cv_best_score, run(direct, data, engine).cv_best_score, rtol=1e-12)


@pytest.mark.parametrize("make", [
    lambda: roles.HighLeverageFilter(),  # live instance
    lambda: "n4m:filters.high_leverage",  # default parameters serialize to a bare token
    lambda: {"class": "n4m:filters.y_outlier", "params": {"threshold": 2.1}},  # explicit parameters
], ids=["instance", "token", "dict"])
def test_filters_resolve_from_every_representation(make):
    from nirs4all.operators.filters.native import resolve_sample_filter

    resolved = resolve_sample_filter(make())
    assert isinstance(resolved, NativeRoleFilter)
    assert resolve_sample_filter("sklearn.preprocessing.StandardScaler") is None


@pytest.mark.parametrize("engine", [None, "legacy"])
def test_default_filter_runs_in_by_filter_branch(data, engine):
    # Default parameters make the filter serialize to "n4m:filters.high_leverage".
    pipeline = [
        {"branch": {"by_filter": roles.HighLeverageFilter(), "steps": [roles.SNV()]}},
        {"merge": "concat"},
        KFold(3),
        {"model": PLSRegression(n_components=2)},
    ]
    assert np.isfinite(run(pipeline, data, engine).cv_best_score)


def test_dagml_branch_filter_resolves_default_role_token():
    from nirs4all.pipeline.config.component_serialization import serialize_component
    from nirs4all.pipeline.dagml.run_paths import _branch_filter_from_spec

    token = serialize_component(roles.HighLeverageFilter())
    assert token == "n4m:filters.high_leverage"
    assert isinstance(_branch_filter_from_spec(token), NativeRoleFilter)
