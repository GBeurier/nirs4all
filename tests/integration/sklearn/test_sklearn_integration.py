"""
Integration tests for NIRSPipeline sklearn wrapper.

These tests verify end-to-end functionality:
- Training with nirs4all.run()
- Wrapping with NIRSPipeline.from_result()
- Exporting and loading bundles
- sklearn compatibility (predict, score, cross_validate)
- SHAP integration (optional, depends on shap being installed)
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Skip all tests if running in minimal environment
pytest.importorskip("sklearn")

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

import nirs4all
from nirs4all.operators.transforms import StandardNormalVariate
from nirs4all.sklearn import NIRSPipeline, NIRSPipelineClassifier


class TestNIRSPipelineIntegration:
    """Integration tests for NIRSPipeline with real training."""

    @pytest.fixture
    def sample_data(self):
        """Create sample regression data for testing.

        Returns X, y as tuple. Uses 40 train + 10 test samples.
        """
        np.random.seed(42)
        n_train = 40
        n_test = 10
        n_features = 100

        X = np.random.randn(n_train + n_test, n_features)
        y = X[:, :5].sum(axis=1) + np.random.randn(n_train + n_test) * 0.1

        return X, y

    @pytest.fixture
    def trained_result(self, sample_data):
        """Train a simple pipeline and return the result."""
        X, y = sample_data
        n_train = 40  # Match sample_data fixture

        # Create partition info for train/test split
        partition_info = {
            'train': slice(0, n_train),
            'test': slice(n_train, len(X))
        }

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25),
            {"model": PLSRegression(n_components=3)}
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=(X, y, partition_info),
            name="sklearn_integration_test",
            verbose=0,
            save_artifacts=True,
            plots_visible=False
        )

        return result

    def test_from_result_basic(self, trained_result):
        """Test basic from_result() workflow."""
        pipe = NIRSPipeline.from_result(trained_result)

        assert pipe.is_fitted_
        assert pipe.model_step_index is not None

    def test_predict_from_result(self, trained_result, sample_data):
        """Test prediction from result wrapper."""
        pipe = NIRSPipeline.from_result(trained_result)

        # Get test data
        X, y = sample_data
        X_test = X[:10]
        y_test = y[:10]

        y_pred = pipe.predict(X_test)

        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape[0] == 10
        assert not np.isnan(y_pred).any()

    def test_score_from_result(self, trained_result, sample_data):
        """Test score() method from result wrapper."""
        pipe = NIRSPipeline.from_result(trained_result)

        # Get test data
        X, y = sample_data
        X_test = X[:10]
        y_test = y[:10]

        score = pipe.score(X_test, y_test)

        assert isinstance(score, float)
        assert -1 <= score <= 1  # R² can be negative

    def test_model_access(self, trained_result):
        """Test accessing underlying model."""
        pipe = NIRSPipeline.from_result(trained_result)

        model = pipe.model_

        assert model is not None
        assert hasattr(model, 'predict')

    def test_export_and_load_bundle(self, trained_result, tmp_path):
        """Test export to bundle and load workflow."""
        # Export
        bundle_path = trained_result.export(tmp_path / "test_model.n4a")
        assert bundle_path.exists()

        # Load from bundle
        pipe = NIRSPipeline.from_bundle(bundle_path)

        assert pipe.is_fitted_
        assert pipe.model_step_index is not None

    def test_bundle_predict(self, trained_result, tmp_path, sample_data):
        """Test prediction from loaded bundle."""
        # Export and load
        bundle_path = trained_result.export(tmp_path / "test_model.n4a")
        pipe = NIRSPipeline.from_bundle(bundle_path)

        # Get test data
        X, y = sample_data
        X_test = X[:10]

        y_pred = pipe.predict(X_test)

        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape[0] == 10

    def test_predictions_consistent(self, trained_result, tmp_path, sample_data):
        """Test that predictions from result and bundle are consistent."""
        # Create wrappers
        pipe_result = NIRSPipeline.from_result(trained_result)
        bundle_path = trained_result.export(tmp_path / "test_model.n4a")
        pipe_bundle = NIRSPipeline.from_bundle(bundle_path)

        # Get test data
        X, y = sample_data
        X_test = X[:5]

        y_result = pipe_result.predict(X_test)
        y_bundle = pipe_bundle.predict(X_test)

        # Predictions should be identical
        np.testing.assert_allclose(y_result, y_bundle, rtol=1e-5)

    def test_fold_selection(self, trained_result, sample_data):
        """Test selecting different folds."""
        pipe_fold0 = NIRSPipeline.from_result(trained_result, fold=0)
        pipe_fold1 = NIRSPipeline.from_result(trained_result, fold=1)

        # Get test data
        X, y = sample_data
        X_test = X[:5]

        # Both should work (predictions might differ slightly due to fold)
        y_fold0 = pipe_fold0.predict(X_test)
        y_fold1 = pipe_fold1.predict(X_test)

        assert y_fold0.shape == y_fold1.shape

class TestNIRSPipelineSklearnCompatibility:
    """Test sklearn tool compatibility."""

    @pytest.fixture
    def fitted_pipeline(self, tmp_path):
        """Create a fitted pipeline for testing."""
        # Create sample data with train/test split
        np.random.seed(42)
        n_train = 40
        n_test = 10
        n_features = 100
        X = np.random.randn(n_train + n_test, n_features)
        y = X[:, :5].sum(axis=1) + np.random.randn(n_train + n_test) * 0.1

        # Partition info for proper train/test split
        partition_info = {
            'train': slice(0, n_train),
            'test': slice(n_train, n_train + n_test)
        }

        # Train
        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25),
            {"model": PLSRegression(n_components=3)}
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=(X, y, partition_info),
            name="sklearn_compat_test",
            verbose=0,
            save_artifacts=True,
            plots_visible=False
        )

        # Export bundle
        bundle_path = tmp_path / "model.n4a"
        result.export(bundle_path)

        return NIRSPipeline.from_bundle(bundle_path), X, y

    def test_sklearn_r2_score(self, fitted_pipeline):
        """Test compatibility with sklearn r2_score."""
        pipe, X, y = fitted_pipeline

        y_pred = pipe.predict(X)
        r2_manual = r2_score(y, y_pred)
        r2_method = pipe.score(X, y)

        assert pytest.approx(r2_manual, rel=1e-5) == r2_method

    def test_get_set_params(self, fitted_pipeline):
        """Test sklearn get_params/set_params interface."""
        pipe, X, y = fitted_pipeline

        params = pipe.get_params()
        assert isinstance(params, dict)

        pipe.set_params(fold=1)
        new_params = pipe.get_params()
        assert new_params["fold"] == 1

class TestShapIntegration:
    """Test SHAP integration (optional, skipped if shap not installed)."""

    @pytest.fixture
    def fitted_pipeline_with_data(self, tmp_path):
        """Create fitted pipeline and return with test data."""
        # Create sample data with train/test split
        np.random.seed(42)
        n_train = 40
        n_test = 10
        n_features = 100
        X = np.random.randn(n_train + n_test, n_features)
        y = X[:, :5].sum(axis=1) + np.random.randn(n_train + n_test) * 0.1

        # Partition info for proper train/test split
        partition_info = {
            'train': slice(0, n_train),
            'test': slice(n_train, n_train + n_test)
        }

        # Train
        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25),
            {"model": PLSRegression(n_components=3)}
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=(X, y, partition_info),
            name="shap_test",
            verbose=0,
            save_artifacts=True,
            plots_visible=False
        )

        bundle_path = tmp_path / "model.n4a"
        result.export(bundle_path)
        pipe = NIRSPipeline.from_bundle(bundle_path)

        return pipe, X

    @pytest.mark.skipif(
        not pytest.importorskip("shap", reason="SHAP not installed"),
        reason="SHAP not installed"
    )
    def test_shap_kernel_explainer(self, fitted_pipeline_with_data):
        """Test SHAP KernelExplainer with NIRSPipeline."""
        import shap

        pipe, X = fitted_pipeline_with_data

        # Use small samples for speed
        background = X[:10]
        test_samples = X[10:12]

        # Create explainer using pipe.predict
        explainer = shap.KernelExplainer(
            pipe.predict,
            shap.kmeans(background, 5)
        )

        # Compute SHAP values
        shap_values = explainer.shap_values(test_samples, nsamples=50)

        assert shap_values is not None
        assert shap_values.shape == test_samples.shape

    def test_model_access_for_shap(self, fitted_pipeline_with_data):
        """Test that model_ can be accessed for model-specific SHAP."""
        pipe, _ = fitted_pipeline_with_data

        model = pipe.model_
        assert model is not None

        # For PLS, should have coef_ attribute
        assert hasattr(model, 'coef_')
