"""PyTorch output semantics and fitted input contracts."""

import numpy as np
import pytest

from nirs4all.controllers.models.torch_model import PyTorchModelController
from nirs4all.core.task_type import TaskType


@pytest.mark.torch
def test_torch_training_keeps_multioutput_regression_and_input_layout() -> None:
    torch = pytest.importorskip("torch")
    controller = PyTorchModelController()
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 3))
    X = torch.arange(16, dtype=torch.float32).reshape(4, 1, 4)
    y = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    fitted = controller._train_model(model, X, y, epochs=1, task_type=TaskType.REGRESSION)
    expected = fitted(X.to(next(fitted.parameters()).device)).detach().cpu().numpy()
    actual = controller._predict_model(fitted, X.reshape(4, 4).numpy())

    assert fitted._nirs4all_task_type == "regression"
    assert fitted._nirs4all_input_shape == (1, 4)
    assert actual.shape == (4, 3)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.torch
def test_torch_multiclass_prediction_uses_explicit_task_type() -> None:
    torch = pytest.importorskip("torch")
    controller = PyTorchModelController()
    model = torch.nn.Linear(2, 3)
    model._nirs4all_task_type = "multiclass_classification"
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]))
        model.bias.zero_()

    actual = controller._predict_model(model, np.asarray([[3.0, 1.0], [0.0, 2.0]], dtype=np.float32))

    np.testing.assert_array_equal(actual, np.asarray([[0.0], [1.0]], dtype=np.float32))


@pytest.mark.torch
def test_torch_validation_score_keeps_targets_separate() -> None:
    torch = pytest.importorskip("torch")
    controller = PyTorchModelController()
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.copy_(torch.tensor([1.0, 110.0]))
    X = torch.zeros((4, 2))
    y = torch.tensor([[0.0, 100.0], [1.0, 110.0], [2.0, 120.0], [3.0, 130.0]])

    actual = controller._evaluate_model(model, X, y, metric="r2")
    assert actual == pytest.approx(-0.2)
