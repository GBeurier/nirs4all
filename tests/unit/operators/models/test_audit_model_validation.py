"""Remaining scientific estimator contracts and qualified legacy CNN shapes."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from nirs4all.operators.models.sklearn import LWPLS, MBPLS, DiPLS


@pytest.mark.parametrize("estimator", [DiPLS, LWPLS])
def test_remaining_estimators_reject_unknown_parameters_and_unfitted_predict(estimator):
    model = estimator()
    with pytest.raises(ValueError, match="n_componets"):
        model.set_params(n_componets=3)
    assert model.set_params(n_components=2) is model
    with pytest.raises(NotFittedError):
        model.predict(np.zeros((3, 8)))


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("params", [{"method": "SVD"}, {"max_tol": 1e-3}])
def test_mbpls_explicitly_rejects_unsupported_algorithm_controls(backend, params):
    with pytest.raises(ValueError, match="method|max_tol"):
        MBPLS(backend=backend, **params).fit(np.eye(6), np.arange(6))


@pytest.mark.parametrize("length", [512, 1557, 2151, None, 0])
def test_unet_length_constraints_fail_before_building_layers(length, monkeypatch):
    pytest.importorskip("tensorflow")
    from nirs4all.operators.models.tensorflow import generic
    def unexpected_input(*args, **kwargs):
        raise AssertionError("Layers must not be created for an unsupported length")
    monkeypatch.setattr(generic, "Input", unexpected_input)
    with pytest.raises(ValueError, match="UNET.*25"):
        generic.UNET((length, 1), {})


@pytest.mark.parametrize("length", [700, 1050])
def test_unet_supported_lengths_still_build(length):
    tensorflow = pytest.importorskip("tensorflow")
    from nirs4all.operators.models.tensorflow import generic
    try:
        model = generic.UNET((length, 1), {"layer_n": 8, "depth": 0})
        assert model.input_shape == (None, length, 1)
        assert model.output_shape == (None, 1)
    finally:
        tensorflow.keras.backend.clear_session()


@pytest.mark.parametrize("length", [512, 700, 1050, None])
def test_custom_vg_length_constraints_fail_before_building_layers(length, monkeypatch):
    pytest.importorskip("tensorflow")
    from nirs4all.operators.models.tensorflow import generic
    def unexpected_input(*args, **kwargs):
        raise AssertionError("Layers must not be created for an unsupported length")
    monkeypatch.setattr(generic, "Input", unexpected_input)
    with pytest.raises(ValueError, match="Custom_VG_Residuals.*(length|short)"):
        generic.Custom_VG_Residuals((length, 1), {})


@pytest.mark.parametrize("length,params", [(1557, {}), (700, {"block_kernel_size1": 1, "block_kernel_size2": 1, "block_kernel_size3": 1, "kernel_size2": 1})])
def test_custom_vg_shape_validation_honors_configured_kernels(length, params):
    tensorflow = pytest.importorskip("tensorflow")
    from nirs4all.operators.models.tensorflow import generic
    try:
        model = generic.Custom_VG_Residuals((length, 1), params)
        assert model.input_shape == (None, length, 1)
        assert model.output_shape == (None, 1)
    finally:
        tensorflow.keras.backend.clear_session()


@pytest.mark.parametrize("params", [{"strides1": 0}, {"kernel_size2": 0}])
def test_custom_vg_rejects_invalid_convolution_parameters(params):
    pytest.importorskip("tensorflow")
    from nirs4all.operators.models.tensorflow import generic
    with pytest.raises(ValueError, match="positive"):
        generic.Custom_VG_Residuals((1557, 1), params)
