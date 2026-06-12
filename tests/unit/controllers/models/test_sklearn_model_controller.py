from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge

from nirs4all.controllers.models.sklearn_model import SklearnModelController
from nirs4all.pipeline.config.context import ExecutionPhase
from nirs4all.pipeline.steps.parser import StepParser


class _DummyDataset:
    task_type = None

    def x(self, context, layout=None, concat_source=True):
        return np.zeros((4, 3))


class _Column:
    def __init__(self, values):
        self._values = values

    def to_numpy(self):
        return np.asarray(self._values, dtype=object)


class _MetadataFrame:
    def __init__(self, data):
        self._data = data
        self.columns = list(data)

    def __getitem__(self, column):
        return _Column(self._data[column])


class _RelationDataset(_DummyDataset):
    def __init__(self):
        self._metadata = _MetadataFrame(
            {
                "physical_sample_id": ["S1", "S1", "S1", "S2"],
                "unit_level": ["combo", "combo", "combo", "combo"],
                "origin_sample_id": ["S1", "S1", "S1", "S2"],
                "representation": ["cartesian_full"] * 4,
            }
        )

    def metadata(self, selector):
        assert selector == {"partition": "train"}
        return self._metadata


class _LegacyPhysicalIdDataset(_DummyDataset):
    def __init__(self):
        self._metadata = _MetadataFrame({"physical_sample_id": ["S1", "S1", "S1", "S2"]})

    def metadata(self, selector):
        assert selector == {"partition": "train"}
        return self._metadata


class _RecordingWeightedEstimator(BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        self.fit_shape_ = X.shape
        self.sample_weight_ = sample_weight
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])


class _RecordingUnweightedEstimator(BaseEstimator):
    def fit(self, X, y):
        self.fit_shape_ = X.shape
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])


class TestSklearnModelControllerGetModelInstance:
    def test_builds_model_from_serialized_model_instance_dict(self):
        controller = SklearnModelController()

        model = controller._get_model_instance(
            dataset=_DummyDataset(),
            model_config={
                "model_instance": {
                    "class": "sklearn.linear_model.Ridge",
                    "params": {"alpha": 0.25},
                }
            },
        )

        assert isinstance(model, Ridge)
        assert model.alpha == 0.25

    def test_builds_model_from_class_reference(self):
        controller = SklearnModelController()

        model = controller._get_model_instance(
            dataset=_DummyDataset(),
            model_config={"model_instance": Ridge},
        )

        assert isinstance(model, Ridge)

    def test_raises_clear_error_for_invalid_model_instance_dict(self):
        controller = SklearnModelController()

        with pytest.raises(ValueError, match="Could not instantiate sklearn model"):
            controller._get_model_instance(
                dataset=_DummyDataset(),
                model_config={
                    "model_instance": {
                        "class": "sklearn.cross_decomposition.Ridge",
                    }
                },
            )


class TestStepParserInvalidSerializedComponents:
    def test_invalid_model_class_path_raises_clear_error(self):
        parser = StepParser()

        with pytest.raises(
            ValueError,
            match="Could not deserialize component 'sklearn.cross_decomposition.Ridge'",
        ):
            parser.parse({"model": {"class": "sklearn.cross_decomposition.Ridge"}})


class TestSklearnFitInfluence:
    def _build_model(self, estimator, dataset=None, x_train=None, y_train=None, train_indices=None):
        controller = SklearnModelController()
        controller.verbose = 0
        X_train = x_train if x_train is not None else np.arange(8, dtype=float).reshape(4, 2)
        y_train = y_train if y_train is not None else np.arange(X_train.shape[0], dtype=float)
        return controller._build_and_train_model(
            dataset=dataset or _RelationDataset(),
            model_config={"model_instance": estimator},
            context=SimpleNamespace(custom={}),
            runtime_context=SimpleNamespace(phase=ExecutionPhase.CV),
            identifiers=SimpleNamespace(name="recording"),
            X_train=X_train,
            y_train=y_train,
            X_val=np.empty((0, 2)),
            y_val=np.empty((0,)),
            X_test=np.empty((0, 2)),
            best_params=None,
            train_indices=train_indices,
        )

    def test_passes_equal_sample_weight_when_estimator_supports_it(self):
        model = self._build_model(_RecordingWeightedEstimator())

        assert model.fit_shape_ == (4, 2)
        np.testing.assert_allclose(model.sample_weight_, [1 / 3, 1 / 3, 1 / 3, 1.0])
        assert model._nirs4all_fit_influence_manifest["effective_mode"] == "equal_sample_influence"
        assert model._nirs4all_fit_influence_manifest["has_sample_weight"] is True

    def test_resamples_when_estimator_lacks_sample_weight_support(self):
        model = self._build_model(_RecordingUnweightedEstimator())

        assert model.fit_shape_ == (6, 2)
        assert model._nirs4all_fit_influence_manifest["effective_mode"] == "resample_equalized"
        assert model._nirs4all_fit_influence_manifest["has_resample_indices"] is True

    def test_fold_train_indices_align_relation_metadata(self):
        model = self._build_model(
            _RecordingWeightedEstimator(),
            x_train=np.arange(6, dtype=float).reshape(3, 2),
            y_train=np.arange(3, dtype=float),
            train_indices=[0, 2, 3],
        )

        assert model.fit_shape_ == (3, 2)
        np.testing.assert_allclose(model.sample_weight_, [0.5, 0.5, 1.0])

    def test_legacy_physical_sample_metadata_does_not_enable_fit_influence(self):
        model = self._build_model(_RecordingWeightedEstimator(), dataset=_LegacyPhysicalIdDataset())

        assert model.fit_shape_ == (4, 2)
        assert model.sample_weight_ is None
        assert not hasattr(model, "_nirs4all_fit_influence_manifest")
