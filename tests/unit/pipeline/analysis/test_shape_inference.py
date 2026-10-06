"""Tests for nirs4all.pipeline.analysis.shape_inference."""

from __future__ import annotations

from nirs4all.pipeline.analysis.shape_inference import (
    DIMENSION_BOUND_PARAMS,
    infer_output_shape,
)


class TestInferOutputShape:
    def test_identity_preprocessing(self):
        assert infer_output_shape("SNV", {}, 100, 500) == (100, 500)
        assert infer_output_shape("SavitzkyGolay", {"window_length": 11}, 100, 500) == (100, 500)

    def test_pls_components_clamped(self):
        assert infer_output_shape("PLSRegression", {"n_components": 10}, 100, 500) == (100, 10)
        # clamped by features
        assert infer_output_shape("PLSRegression", {"n_components": 600}, 100, 500) == (100, 100)
        # default 10
        assert infer_output_shape("PLSRegression", {}, 100, 500) == (100, 10)

    def test_pca_default_keeps_min_dim(self):
        assert infer_output_shape("PCA", {}, 40, 500) == (40, 40)
        assert infer_output_shape("PCA", {"n_components": 5}, 40, 500) == (40, 5)

    def test_crop(self):
        assert infer_output_shape("CropTransformer", {"start": 50, "end": 250}, 10, 500) == (10, 200)
        # An empty slice has no output features
        assert infer_output_shape("CropTransformer", {"start": 400, "end": 100}, 10, 500) == (10, 0)

    def test_resample(self):
        assert infer_output_shape("Resampler", {"n_features": 128}, 10, 500) == (10, 128)
        assert infer_output_shape("ResampleTransformer", {"target_points": 64}, 10, 500) == (10, 64)

    def test_wavelet_halves_per_level(self):
        assert infer_output_shape("Wavelet", {"level": 2}, 10, 512) == (10, 128)
        assert infer_output_shape("Haar", {}, 10, 512) == (10, 256)

    def test_unknown_operator_returns_none(self):
        assert infer_output_shape("SomeFutureOperator", {}, 10, 500) is None


def test_dimension_bound_params_taxonomy():
    assert DIMENSION_BOUND_PARAMS["n_components"] == "features"
    assert DIMENSION_BOUND_PARAMS["n_splits"] == "samples"


def test_data_dependent_pca_shape_is_unknown():
    assert infer_output_shape("PCA", {"n_components": None}, 100, 10) == (100, 10)
    assert infer_output_shape("PCA", {"n_components": "mle"}, 100, 10) is None
    assert infer_output_shape("PCA", {"n_components": 0.95}, 100, 10) is None


def test_crop_shape_uses_real_python_slice_bounds():
    assert infer_output_shape("CropTransformer", {"start": 0, "end": None}, 100, 10) == (100, 10)
    assert infer_output_shape("CropTransformer", {"start": 3, "end": 100}, 100, 10) == (100, 7)
    assert infer_output_shape("CropTransformer", {"start": -3}, 100, 10) == (100, 3)
    assert infer_output_shape("CropTransformer", {"start": 12}, 100, 10) == (100, 0)


def test_resample_operator_parameter_controls_shape():
    assert infer_output_shape("ResampleTransformer", {"num_samples": 4}, 100, 10) == (100, 4)
