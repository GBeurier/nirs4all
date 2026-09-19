"""NICoN validates real spectral width before convolution layer allocation."""
import pytest

torch = pytest.importorskip("torch")
from nirs4all.operators.models.pytorch.nicon import customizable_nicon, customizable_nicon_classification, nicon, nicon_classification


@pytest.mark.parametrize("factory", [nicon, nicon_classification, customizable_nicon, customizable_nicon_classification])
def test_nicon_rejects_short_spectra_with_required_width(factory):
    with pytest.raises(ValueError, match="at least 175 spectral features.*received 64"):
        factory(input_shape=(1, 64))
    model = factory(input_shape=(1, 175))
    assert model(torch.zeros(4, 1, 175)).shape == (4, 1)


def test_custom_nicon_width_uses_actual_configured_kernels_and_strides():
    params = {"kernel_size1": 3, "kernel_size2": 3, "kernel_size3": 3,
              "strides1": 1, "strides2": 1, "strides3": 1, "filters1": 12}
    model = customizable_nicon(input_shape=(1, 7), params=params)
    assert model(torch.zeros(4, 1, 7)).shape == (4, 1)
    assert next(layer for layer in model if isinstance(layer, torch.nn.Conv1d)).out_channels == 12
    with pytest.raises(ValueError, match="at least 7 spectral features"):
        customizable_nicon(input_shape=(1, 6), params=params)
    with pytest.raises(ValueError, match="positive integers"):
        customizable_nicon(input_shape=(1, 64), params={"strides1": 0})
