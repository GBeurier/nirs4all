"""Tests for PyTorch models."""

from types import SimpleNamespace

import pytest

from nirs4all.utils.backend import TORCH_AVAILABLE


class TestPyTorchLossResolution:
    """Test loss configuration without importing the optional backend."""

    @staticmethod
    def _fake_nn():
        class MSELoss:
            pass

        class L1Loss:
            pass

        class CrossEntropyLoss:
            pass

        return SimpleNamespace(
            MSELoss=MSELoss,
            L1Loss=L1Loss,
            CrossEntropyLoss=CrossEntropyLoss,
        )

    def test_known_alias_is_resolved(self):
        from nirs4all.controllers.models.torch_model import _resolve_loss_function

        nn = self._fake_nn()

        assert isinstance(_resolve_loss_function("mse", nn), nn.MSELoss)

    def test_torch_class_name_and_default_are_resolved(self):
        from nirs4all.controllers.models.torch_model import _resolve_loss_function

        nn = self._fake_nn()

        assert isinstance(_resolve_loss_function("L1Loss", nn), nn.L1Loss)
        assert isinstance(_resolve_loss_function("MSELoss", nn), nn.MSELoss)

    def test_callable_is_preserved(self):
        from nirs4all.controllers.models.torch_model import _resolve_loss_function

        custom_loss = object()

        assert _resolve_loss_function(custom_loss, self._fake_nn()) is custom_loss

    def test_unknown_loss_name_is_rejected(self):
        from nirs4all.controllers.models.torch_model import _resolve_loss_function

        with pytest.raises(ValueError, match="Unknown PyTorch loss 'not_a_loss'"):
            _resolve_loss_function("not_a_loss", self._fake_nn())


@pytest.mark.xdist_group("gpu")
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPyTorchModels:
    """Test suite for PyTorch models."""

    def test_simple_mlp(self):
        import torch
        import torch.nn as nn

        from nirs4all.utils.backend import framework

        @framework('pytorch')
        class SimpleMLP(nn.Module):
            def __init__(self, input_shape):
                super().__init__()
                self.layer = nn.Linear(input_shape[1], 1)
            def forward(self, x):
                return self.layer(x)

        model = SimpleMLP(input_shape=(10, 5))
        x = torch.randn(1, 5)
        y = model(x)
        assert y.shape == (1, 1)
