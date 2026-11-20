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


def test_distinct_source_final_captures_upstream_prefix_and_cold_replay(tmp_path):
    """Unequal sources retain train-only prefix/state/order outside their workspace."""
    from sklearn.base import clone
    from sklearn.preprocessing import MinMaxScaler

    from tests.integration.pipeline._cold_replay import run_cold_replay

    rng = np.random.default_rng(2219)
    blocks = [rng.normal(size=(48, width)).astype(np.float32) + 3 * index for index, width in enumerate((7, 5, 6))]
    for block in blocks:
        block[36:] += 20  # A held-out sentinel detects fitting any prefix on TEST.
    targets = blocks[0][:, 0] - blocks[1][:, 1] + 2 * blocks[2][:, 2]
    dataset = SpectroDataset("distinct-source-final-prefix")
    dataset.add_samples([block[:36] for block in blocks], {"partition": "train"})
    dataset.add_samples([block[36:] for block in blocks], {"partition": "test"})
    dataset.add_targets(targets)
    operators = [StandardNormalVariate(), MultiplicativeScatterCorrection(), FirstDerivative()]
    prefix = MinMaxScaler(feature_range=(1, 2))
    result = nirs4all.run(
        [prefix, KFold(2, shuffle=True, random_state=17),
         {"branch": {"by_source": True, "steps": {f"source_{index}": [operator] for index, operator in enumerate(operators)}}},
         {"merge": "concat"}, {"model": Ridge(5.0)}],
        dataset, engine="legacy", workspace_path=tmp_path / "workspace", save_charts=False, verbose=0,
    )
    train_blocks, test_blocks = [], []
    for operator, block in zip(operators, blocks, strict=True):
        fitted_prefix = clone(prefix).fit(block[:36])
        training = np.asarray(fitted_prefix.transform(block[:36]), dtype=np.float32)
        heldout = np.asarray(fitted_prefix.transform(block[36:]), dtype=np.float32)
        transform = clone(operator).fit(training)
        train_blocks.append(np.asarray(transform.transform(training), dtype=np.float32))
        test_blocks.append(np.asarray(transform.transform(heldout), dtype=np.float32))
    oracle = Ridge(5.0).fit(np.hstack(train_blocks), targets[:36]).predict(np.hstack(test_blocks))
    final = next(row for row in result.predictions.iter_entries()
                 if row["partition"] == "test" and str(row["fold_id"]) == "final")
    np.testing.assert_allclose(np.asarray(final["y_pred"]).ravel(), oracle, rtol=1e-5, atol=1e-8)
    archive = result.export(tmp_path / "final.n4a", source=final)
    result.close()
    np.savez(tmp_path / "inputs.npz", X=np.hstack([block[36:] for block in blocks]), expected=oracle)
    script = r'''
import pathlib,sys,numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from nirs4all.operators.transforms import StandardNormalVariate,MultiplicativeScatterCorrection,FirstDerivative
from nirs4all.pipeline.storage import WorkspaceStore
from nirs4all.pipeline.bundle import BundleLoader
blocked=pathlib.Path(sys.argv[1]).resolve()
def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=pathlib.Path(args[0]).resolve()
        assert path!=blocked and blocked not in path.parents,'Replay opened original workspace'
sys.addaudithook(audit)
def refuse(*args,**kwargs):raise AssertionError('Replay attempted fit or workspace construction')
WorkspaceStore.__init__=refuse
for cls in (Ridge,MinMaxScaler,StandardNormalVariate,MultiplicativeScatterCorrection,FirstDerivative):cls.fit=refuse
values=np.load(sys.argv[3])
with BundleLoader(sys.argv[2]) as loader:
    np.testing.assert_allclose(np.asarray(loader.predict(values['X'])).ravel(),values['expected'],rtol=1e-5,atol=1e-8)
'''
    completed = run_cold_replay(script, [str(tmp_path / "workspace"), str(archive), str(tmp_path / "inputs.npz")], tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_historical_source_concat_requires_its_raw_input_closure(tmp_path):
    """A later source merge cannot qualify a flat historical preprocessing list."""
    from unittest.mock import patch

    import pytest

    from nirs4all.pipeline.analysis.topology import require_legacy_raw_input_closure
    from nirs4all.pipeline.bundle import BundleLoader

    steps = [{"branch": {"by_source": True, "steps": {"source_0": [StandardNormalVariate()], "source_1": [MultiplicativeScatterCorrection()]}}},
             {"merge": "concat"}, {"model": Ridge()}]
    with pytest.raises(NotImplementedError, match="source-local raw-input dependency closure was not captured"):
        require_legacy_raw_input_closure(steps, 3, {"branch_id": None})
    # An explicitly selected model within a source precedes the merged input.
    require_legacy_raw_input_closure(steps, 1, {"branch_path": [0], "branch_id": 0})

    rng = np.random.default_rng(2217)
    left = rng.normal(size=(24, 5)).astype(np.float32) + 5
    right = rng.normal(size=(24, 3)).astype(np.float32) - 4
    dataset = SpectroDataset("historical-source-snv")
    dataset.add_samples([left[:18], right[:18]], {"partition": "train"})
    dataset.add_samples([left[18:], right[18:]], {"partition": "test"})
    dataset.add_targets(left[:, 0] - right[:, 0])
    # Reproduce an old producer without graph capture. Its actual training,
    # fitted objects and predictions are untouched; only capture/admission is
    # bypassed while constructing the deliberately incomplete historical file.
    with patch("nirs4all.pipeline.execution.fitted.prepare_sequential_fitted_capture", return_value=False):
        result = nirs4all.run(
            [KFold(2), {"branch": {"by_source": True, "steps": [StandardNormalVariate()]}}, {"merge": "concat"}, {"model": Ridge(.2)}],
            dataset, engine="legacy", workspace_path=tmp_path / "historical", save_charts=False, verbose=0,
        )
    final = next(row for row in result.predictions.iter_entries()
                 if row["partition"] == "test" and str(row["fold_id"]) == "final")
    raw = np.hstack([left[18:], right[18:]])
    with patch("nirs4all.pipeline.analysis.topology.require_legacy_raw_input_closure"):
        archive = result.export(tmp_path / "historical-partial.n4a", source=final)
        with BundleLoader(archive) as loader:
            wrong = np.asarray(loader.predict(raw)).ravel()
    result.close()
    assert np.isfinite(wrong).all()
    assert np.max(np.abs(wrong - np.asarray(final["y_pred"]).ravel())) > 0.01, "historical flat SNV must be observably wrong"
    with patch("nirs4all.pipeline.bundle.loader.load_host_joblib", side_effect=AssertionError("deserialized partial fitted state")) as load:
        with pytest.raises(NotImplementedError, match="source-local raw-input dependency closure was not captured"):
            BundleLoader(archive)
        assert load.call_count == 0
