"""Tests for PyTorch models."""

import pytest
from nirs4all.utils.backend import TORCH_AVAILABLE

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

