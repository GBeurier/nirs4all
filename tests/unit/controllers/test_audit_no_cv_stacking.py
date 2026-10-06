"""Explicit in-sample stacking remains separate from OOF coverage protection."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge

import nirs4all
from nirs4all.controllers.models.stacking import TrainingSetReconstructor
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.operators.models.meta import CoverageStrategy, MetaModel, StackingConfig
from nirs4all.pipeline.config.context import ExecutionContext


def _cohort():
    rng = np.random.default_rng(72)
    dataset = SpectroDataset("no_cv_stack")
    dataset.add_samples(rng.normal(size=(8, 4)), {"partition": "train"})
    dataset.add_samples(rng.normal(size=(3, 4)) + 9, {"partition": "test"})
    dataset.add_targets(rng.normal(size=11))
    dataset._indexer.mark_excluded([1])
    return dataset


def _predictions(dataset, *, validation=False, missing_train=False):
    store = Predictions()
    train_ids = [7, 0, 6, 2, 5, 3, 4]
    test_ids = [10, 8, 9]
    for name, offset in [("base_a", 10), ("base_b", 20)]:
        if not missing_train:
            # A deliberately misplaced held-out row must never enter train features.
            ids = train_ids + [8]
            store.add_prediction(dataset_name=dataset.name, model_name=name, partition="train", fold_id=0,
                                 sample_indices=ids, y_pred=np.array([sid + offset for sid in train_ids] + [999.]), step_idx=1)
        store.add_prediction(dataset_name=dataset.name, model_name=name, partition="test", fold_id=0,
                             sample_indices=test_ids, y_pred=np.array([sid + offset + 100 for sid in test_ids]), step_idx=1)
        if validation:
            store.add_prediction(dataset_name=dataset.name, model_name=name, partition="val", fold_id=0,
                                 sample_indices=train_ids, y_pred=np.array([-sid - offset for sid in train_ids]), step_idx=1)
    return store


@pytest.mark.parametrize("strategy", list(CoverageStrategy))
def test_explicit_no_cv_uses_training_predictions_in_stable_id_order(strategy):
    dataset = _cohort()
    config = StackingConfig(allow_no_cv=True, coverage_strategy=strategy, min_coverage_ratio=1.)
    reconstructor = TrainingSetReconstructor(_predictions(dataset), ["base_a", "base_b"], stacking_config=config)
    with pytest.warns(UserWarning, match="in-sample training predictions.*not OOF"):
        result = reconstructor.reconstruct(dataset, ExecutionContext().with_step_number(2))
    train_ids = np.array([0, 2, 3, 4, 5, 6, 7])
    np.testing.assert_array_equal(result.X_train_meta, np.column_stack([train_ids + 10, train_ids + 20]))
    np.testing.assert_array_equal(result.X_test_meta, [[118., 128.], [119., 129.], [120., 130.]])
    np.testing.assert_array_equal(result.y_train.ravel(), dataset.y({"partition": "train"}).ravel())
    assert result.valid_train_mask.all() and result.coverage_ratio == 1.
    assert result.n_folds == 0
    assert result.validation_result.is_valid
    assert [warning.code for warning in result.validation_result.warnings] == ["IN_SAMPLE_STACKING"]


@pytest.mark.parametrize("allow_no_cv", [False, True])
def test_declared_cv_uses_oof_even_when_in_sample_opt_in_is_set(allow_no_cv):
    dataset = _cohort()
    dataset.set_folds([([0, 2, 3], [4, 5, 6, 7])])
    reconstructor = TrainingSetReconstructor(
        _predictions(dataset, validation=True), ["base_a", "base_b"],
        stacking_config=StackingConfig(allow_no_cv=allow_no_cv, coverage_strategy=CoverageStrategy.STRICT),
    )
    result = reconstructor.reconstruct(dataset, ExecutionContext().with_step_number(2))
    ids = np.array([0, 2, 3, 4, 5, 6, 7])
    np.testing.assert_array_equal(result.X_train_meta, np.column_stack([-ids - 10, -ids - 20]))
    assert result.n_folds == 1
    assert not any(warning.code == "IN_SAMPLE_STACKING" for warning in result.validation_result.warnings)


@pytest.mark.parametrize("has_cv,allow_no_cv,missing_train", [(False, False, False), (True, True, False), (False, True, True)])
def test_zero_coverage_still_refuses_without_eligible_prediction_features(has_cv, allow_no_cv, missing_train):
    dataset = _cohort()
    if has_cv:
        dataset.set_folds([([0, 2, 3], [4, 5, 6, 7])])
    reconstructor = TrainingSetReconstructor(
        _predictions(dataset, missing_train=missing_train), ["base_a", "base_b"],
        stacking_config=StackingConfig(allow_no_cv=allow_no_cv, coverage_strategy=CoverageStrategy.DROP_INCOMPLETE, min_coverage_ratio=.3),
    )
    with pytest.raises(ValueError, match="Coverage ratio 0.0%.*minimum required 30.0%"):
        reconstructor.reconstruct(dataset, ExecutionContext().with_step_number(2))


@pytest.mark.parametrize("strategy", [CoverageStrategy.IMPUTE_MEAN, CoverageStrategy.IMPUTE_FOLD_MEAN])
@pytest.mark.parametrize("has_cv", [False, True])
def test_mean_imputation_requires_eligible_training_predictions(strategy, has_cv):
    dataset = _cohort()
    if has_cv:
        dataset.set_folds([([0, 2, 3], [4, 5, 6, 7])])
    reconstructor = TrainingSetReconstructor(
        _predictions(dataset), ["base_a", "base_b"],
        stacking_config=StackingConfig(coverage_strategy=strategy),
    )
    with pytest.raises(ValueError, match="zero eligible training coverage"):
        reconstructor.reconstruct(dataset, ExecutionContext().with_step_number(2))


class _ObservedRidge(Ridge):
    fits = []

    def fit(self, X, y, **kwargs):
        self.fits.append((np.asarray(X).copy(), np.asarray(y).copy()))
        return super().fit(X, y, **kwargs)


@pytest.mark.parametrize("strategy", list(CoverageStrategy))
def test_public_legacy_no_cv_meta_fit_matches_independent_in_sample_oracle(tmp_path, strategy):
    dataset = _cohort()
    x_train, y_train = dataset.x({"partition": "train"}), dataset.y({"partition": "train"}).ravel()
    x_test = dataset.x({"partition": "test"})
    base = Ridge(alpha=2.).fit(x_train, y_train)
    expected_features = base.predict(x_train).reshape(-1, 1)
    expected = Ridge(alpha=3.).fit(expected_features, y_train).predict(base.predict(x_test).reshape(-1, 1))
    _ObservedRidge.fits.clear()
    pipeline = [Ridge(alpha=2.), {"model": MetaModel(_ObservedRidge(alpha=3.), stacking_config=StackingConfig(
        allow_no_cv=True, coverage_strategy=strategy, min_coverage_ratio=1.,
    ))}]
    with pytest.warns(UserWarning, match="in-sample training predictions"):
        result = nirs4all.run(pipeline, dataset, engine="legacy", refit=False, workspace_path=tmp_path,
                              save_artifacts=False, save_charts=False, verbose=0)
    try:
        assert len(_ObservedRidge.fits) == 1
        fitted_x, fitted_y = _ObservedRidge.fits[0]
        np.testing.assert_allclose(fitted_x, expected_features)
        np.testing.assert_allclose(fitted_y.ravel(), y_train)
        rows = result.predictions.filter_predictions(partition="test", step_idx=2, load_arrays=True)
        assert len(rows) == 1
        assert list(rows[0]["sample_indices"]) == [8, 9, 10]
        np.testing.assert_allclose(np.asarray(rows[0]["y_pred"]).ravel(), expected)
    finally:
        result.close()
