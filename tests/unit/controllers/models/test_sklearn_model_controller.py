import numpy as np
import pytest
from sklearn.linear_model import Ridge

from nirs4all.controllers.models.sklearn_model import SklearnModelController
from nirs4all.pipeline.steps.parser import StepParser


class _DummyDataset:
    def x(self, context, layout=None, concat_source=True):
        return np.zeros((4, 3))


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
