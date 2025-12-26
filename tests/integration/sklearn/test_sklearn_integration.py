"""
Integration tests for NIRSPipeline sklearn wrapper.

These tests verify end-to-end functionality:
- Training with nirs4all.run()
- Wrapping with NIRSPipeline.from_result()
- Exporting and loading bundles
- sklearn compatibility (predict, score, cross_validate)
- SHAP integration (optional, depends on shap being installed)
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

# Skip all tests if running in minimal environment
pytest.importorskip("sklearn")

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

import nirs4all
from nirs4all.sklearn import NIRSPipeline, NIRSPipelineClassifier
from nirs4all.operators.transforms import StandardNormalVariate


class TestNIRSPipelineIntegration:
    """Integration tests for NIRSPipeline with real training."""

    @pytest.fixture
    def sample_data_path(self, tmp_path):
        """Create sample regression data for testing."""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()

        # Create sample CSV data
        np.random.seed(42)
        n_samples = 50
        n_features = 100

        X = np.random.randn(n_samples, n_features)
        y = X[:, :5].sum(axis=1) + np.random.randn(n_samples) * 0.1

        # Save as CSV (nirs4all format)
        data = np.column_stack([y, X])
        np.savetxt(data_dir / "data.csv", data, delimiter=",")

        return str(data_dir)

    @pytest.fixture
    def trained_result(self, sample_data_path):
        """Train a simple pipeline and return the result."""
        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25),
            {"model": PLSRegression(n_components=3)}
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=sample_data_path,
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

    def test_predict_from_result(self, trained_result, sample_data_path):
        """Test prediction from result wrapper."""
        pipe = NIRSPipeline.from_result(trained_result)

        # Get test data
        from nirs4all.data import DatasetConfigs
        dataset = DatasetConfigs(sample_data_path)
        for config, name in dataset.configs:
            ds = dataset.get_dataset(config, name)
            X_test = ds.x({})[:10]
            y_test = ds.y[:10]
            break

        y_pred = pipe.predict(X_test)

        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape[0] == 10
        assert not np.isnan(y_pred).any()

    def test_score_from_result(self, trained_result, sample_data_path):
        """Test score() method from result wrapper."""
        pipe = NIRSPipeline.from_result(trained_result)

        # Get test data
        from nirs4all.data import DatasetConfigs
        dataset = DatasetConfigs(sample_data_path)
        for config, name in dataset.configs:
            ds = dataset.get_dataset(config, name)
            X_test = ds.x({})[:10]
            y_test = ds.y[:10]
            break

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

    def test_bundle_predict(self, trained_result, tmp_path, sample_data_path):
        """Test prediction from loaded bundle."""
        # Export and load
        bundle_path = trained_result.export(tmp_path / "test_model.n4a")
        pipe = NIRSPipeline.from_bundle(bundle_path)

        # Get test data
        from nirs4all.data import DatasetConfigs
        dataset = DatasetConfigs(sample_data_path)
        for config, name in dataset.configs:
            ds = dataset.get_dataset(config, name)
            X_test = ds.x({})[:10]
            break

        y_pred = pipe.predict(X_test)

        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape[0] == 10

    def test_predictions_consistent(self, trained_result, tmp_path, sample_data_path):
        """Test that predictions from result and bundle are consistent."""
        # Create wrappers
        pipe_result = NIRSPipeline.from_result(trained_result)
        bundle_path = trained_result.export(tmp_path / "test_model.n4a")
        pipe_bundle = NIRSPipeline.from_bundle(bundle_path)

        # Get test data
        from nirs4all.data import DatasetConfigs
        dataset = DatasetConfigs(sample_data_path)
        for config, name in dataset.configs:
            ds = dataset.get_dataset(config, name)
            X_test = ds.x({})[:5]
            break

        y_result = pipe_result.predict(X_test)
        y_bundle = pipe_bundle.predict(X_test)

        # Predictions should be identical
        np.testing.assert_allclose(y_result, y_bundle, rtol=1e-5)

    def test_fold_selection(self, trained_result, sample_data_path):
        """Test selecting different folds."""
        pipe_fold0 = NIRSPipeline.from_result(trained_result, fold=0)
        pipe_fold1 = NIRSPipeline.from_result(trained_result, fold=1)

        # Get test data
        from nirs4all.data import DatasetConfigs
        dataset = DatasetConfigs(sample_data_path)
        for config, name in dataset.configs:
            ds = dataset.get_dataset(config, name)
            X_test = ds.x({})[:5]
            break

        # Both should work (predictions might differ slightly due to fold)
        y_fold0 = pipe_fold0.predict(X_test)
        y_fold1 = pipe_fold1.predict(X_test)

        assert y_fold0.shape == y_fold1.shape


class TestNIRSPipelineSklearnCompatibility:
    """Test sklearn tool compatibility."""

    @pytest.fixture
    def fitted_pipeline(self, tmp_path):
        """Create a fitted pipeline for testing."""
        # Create sample data
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()

        np.random.seed(42)
        n_samples = 50
        n_features = 100
        X = np.random.randn(n_samples, n_features)
        y = X[:, :5].sum(axis=1) + np.random.randn(n_samples) * 0.1
        data = np.column_stack([y, X])
        np.savetxt(data_dir / "data.csv", data, delimiter=",")

        # Train
        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25),
            {"model": PLSRegression(n_components=3)}
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=str(data_dir),
            name="sklearn_compat_test",
            verbose=0,
            save_artifacts=True,
            plots_visible=False
        )

        # Export bundle
        bundle_path = tmp_path / "model.n4a"
        result.export(bundle_path)

        return NIRSPipeline.from_bundle(bundle_path), data_dir

    def test_sklearn_r2_score(self, fitted_pipeline):
        """Test compatibility with sklearn r2_score."""
        pipe, data_dir = fitted_pipeline

        from nirs4all.data import DatasetConfigs
        dataset = DatasetConfigs(str(data_dir))
        for config, name in dataset.configs:
            ds = dataset.get_dataset(config, name)
            X = ds.x({})
            y = ds.y
            break

        y_pred = pipe.predict(X)
        r2_manual = r2_score(y, y_pred)
        r2_method = pipe.score(X, y)

        assert pytest.approx(r2_manual, rel=1e-5) == r2_method

    def test_get_set_params(self, fitted_pipeline):
        """Test sklearn get_params/set_params interface."""
        pipe, _ = fitted_pipeline

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
        # Create sample data
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()

        np.random.seed(42)
        n_samples = 50
        n_features = 100
        X = np.random.randn(n_samples, n_features)
        y = X[:, :5].sum(axis=1) + np.random.randn(n_samples) * 0.1
        data = np.column_stack([y, X])
        np.savetxt(data_dir / "data.csv", data, delimiter=",")

        # Train
        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25),
            {"model": PLSRegression(n_components=3)}
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=str(data_dir),
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
