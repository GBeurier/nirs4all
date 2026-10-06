"""Unequal-source witness for the declared DAG early-fusion contract."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

import nirs4all
from nirs4all.data import SpectroDataset
from nirs4all.operators.transforms.scalers import StandardNormalVariate


def test_plain_dag_preprocessing_uses_concatenated_unequal_sources():
    rng = np.random.default_rng(142)
    left = rng.normal(size=(48, 7)) + 10
    right = rng.normal(size=(48, 3)) * 4 - 5
    target = left[:, 0] - right[:, 0] + rng.normal(size=48) * 0.1
    dataset = SpectroDataset("unequal-source-fusion")
    dataset.add_samples([left[:36], right[:36]], {"partition": "train"})
    dataset.add_samples([left[36:], right[36:]], {"partition": "test"})
    dataset.add_targets(target)
    result = nirs4all.run([StandardNormalVariate(), KFold(3), {"model": Ridge(0.2)}], dataset,
                         engine="dag-ml", refit=True, verbose=0, save_artifacts=False, save_charts=False)
    # Compare the same stored float32 inputs/targets as the native callbacks.
    fused = dataset.x({}, layout="2d", concat_source=True)
    stored_target = dataset.y({}).ravel()
    reference = make_pipeline(StandardNormalVariate(), Ridge(0.2)).fit(fused[:36], stored_target[:36])
    expected = root_mean_squared_error(stored_target[36:], reference.predict(fused[36:]))
    np.testing.assert_allclose(result.best_rmse, expected, rtol=1e-6, atol=1e-7)
    separately_normalized = np.hstack([StandardNormalVariate().fit_transform(fused[:, :7]), StandardNormalVariate().fit_transform(fused[:, 7:])])
    per_source = Ridge(0.2).fit(separately_normalized[:36], stored_target[:36])
    different = root_mean_squared_error(stored_target[36:], per_source.predict(separately_normalized[36:]))
    assert abs(expected - different) > 0.1, "witness must distinguish both scientific contracts"


def test_explicit_source_concat_uses_each_transformed_source_once():
    rng = np.random.default_rng(192)
    left = rng.normal(size=(48, 7)) + 10
    right = rng.normal(size=(48, 3)) * 4 - 5
    target = left[:, 0] - right[:, 0] + rng.normal(size=48) * 0.1
    dataset = SpectroDataset("explicit-unequal-source-concat")
    dataset.add_samples([left[:36], right[:36]], {"partition": "train"})
    dataset.add_samples([left[36:], right[36:]], {"partition": "test"})
    dataset.add_targets(target)
    model_params = {"n_estimators": 19, "max_depth": 5, "random_state": 23, "n_jobs": 1}
    result = nirs4all.run(
        [StandardNormalVariate(), {"merge": {"sources": "concat"}}, KFold(3),
         {"model": RandomForestRegressor(**model_params)}],
        dataset, engine="dag-ml", refit=True, verbose=0, save_artifacts=False, save_charts=False,
    )
    stored = dataset.x({}, layout="2d", concat_source=True)
    stored_target = dataset.y({}).ravel()
    transformed = np.hstack([
        StandardNormalVariate().fit_transform(stored[:, :7]),
        StandardNormalVariate().fit_transform(stored[:, 7:]),
    ])
    reference = RandomForestRegressor(**model_params).fit(transformed[:36], stored_target[:36])
    expected = root_mean_squared_error(stored_target[36:], reference.predict(transformed[36:]))
    np.testing.assert_allclose(result.best_rmse, expected, rtol=1e-6, atol=1e-7)
    duplicated = np.hstack([transformed, transformed[:, 7:]])
    historical = RandomForestRegressor(**model_params).fit(duplicated[:36], stored_target[:36])
    wrong = root_mean_squared_error(stored_target[36:], historical.predict(duplicated[36:]))
    assert abs(expected - wrong) > 0.001, "witness must detect the duplicated-source layout"
