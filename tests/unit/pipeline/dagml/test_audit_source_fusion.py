"""Unequal-source witness for the declared DAG early-fusion contract."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

import nirs4all
from nirs4all.data import SpectroDataset
from nirs4all.operators.transforms import FirstDerivative, MultiplicativeScatterCorrection
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


def test_distinct_source_concat_lowering_assembles_three_sources_once(tmp_path):
    """New distinct-source runs use three blocks; old fitted layouts remain replayable."""
    import joblib
    from sklearn.base import clone

    from nirs4all.pipeline.dagml.node_runner import _SourceConcatEstimator
    from nirs4all.pipeline.dagml.run_paths import _source_preprocessing_metadata

    rng = np.random.default_rng(1192)
    blocks = [rng.normal(size=(48, width)).astype(np.float32) + index * 4 for index, width in enumerate((7, 5, 6))]
    target = blocks[0][:, 0] - 3 * blocks[1][:, 1] + 2 * blocks[2][:, 2]
    source_steps = {"source_2": [FirstDerivative()], "source_0": [StandardNormalVariate()], "source_1": [MultiplicativeScatterCorrection()]}
    names = ["source_0", "source_1", "source_2"]
    metadata = _source_preprocessing_metadata(source_steps, {"source_order": names, "source_ids": ["src0", "src1", "src2"]})
    assert [source["source_name"] for source in metadata["sources"]] == names
    assert [source["source_index"] for source in metadata["sources"]] == [0, 1, 2]
    templates = [source_steps[name] for name in names]
    train, test = [block[:36] for block in blocks], [block[36:] for block in blocks]
    fitted = _SourceConcatEstimator(Ridge(5.0), templates,
                                    preserve_legacy_sources_after_merge=metadata["preserve_legacy_sources_after_merge"]).fit(train, target[:36])
    transforms = [clone(steps[0]).fit(block) for steps, block in zip(templates, train, strict=True)]
    processed_train = [transform.transform(block) for transform, block in zip(transforms, train, strict=True)]
    processed_test = [transform.transform(block) for transform, block in zip(transforms, test, strict=True)]
    expected_train, expected_test = np.hstack(processed_train), np.hstack(processed_test)
    assert fitted._model.n_features_in_ == expected_train.shape[1] == sum(block.shape[1] for block in processed_train)
    reference = Ridge(5.0).fit(expected_train, target[:36])
    np.testing.assert_allclose(fitted.predict(test), reference.predict(expected_test), rtol=1e-6, atol=1e-7)

    # Preserve the declared historical layout rather than silently reinterpreting
    # already fitted model coefficients with a different feature matrix.
    historical = _SourceConcatEstimator(Ridge(5.0), templates, preserve_legacy_sources_after_merge=True).fit(train, target[:36])
    duplicated_train = np.hstack([expected_train, *processed_train[1:]])
    duplicated_test = np.hstack([expected_test, *processed_test[1:]])
    assert historical._model.n_features_in_ == duplicated_train.shape[1] > expected_train.shape[1]
    old_reference = Ridge(5.0).fit(duplicated_train, target[:36])
    archive = tmp_path / "historical-source-layout.joblib"
    joblib.dump(historical, archive)
    restored = joblib.load(archive)
    np.testing.assert_allclose(restored.predict(test), old_reference.predict(duplicated_test), rtol=1e-6, atol=1e-7)
    assert np.max(np.abs(fitted.predict(test) - restored.predict(test))) > 0.01, "witness must expose duplicated-source predictions"
