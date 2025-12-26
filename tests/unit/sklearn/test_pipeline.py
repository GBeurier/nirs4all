"""
Tests for NIRSPipeline sklearn wrapper.

This module tests the sklearn-compatible wrapper functionality:
- NIRSPipeline.from_bundle() loading
- NIRSPipeline.from_result() creation
- predict() and score() methods
- model_ property for SHAP access
- transform() for preprocessing
- get_params() and set_params() sklearn interface
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import Mock, MagicMock, patch

from nirs4all.sklearn import NIRSPipeline, NIRSPipelineClassifier


class TestNIRSPipelineInit:
    """Test NIRSPipeline initialization and basic properties."""

    def test_init_not_fitted(self):
        """Test that a fresh NIRSPipeline is not fitted."""
        pipe = NIRSPipeline()
        assert not pipe.is_fitted_
        assert pipe._bundle_loader is None

    def test_fit_raises_not_implemented(self):
        """Test that fit() raises NotImplementedError."""
        pipe = NIRSPipeline()
        X = np.random.randn(10, 100)
        y = np.random.randn(10)

        with pytest.raises(NotImplementedError) as excinfo:
            pipe.fit(X, y)

        assert "NIRSPipeline.fit() is not supported" in str(excinfo.value)
        assert "nirs4all.run()" in str(excinfo.value)

    def test_predict_not_fitted_raises(self):
        """Test that predict() on unfitted pipeline raises error."""
        pipe = NIRSPipeline()
        X = np.random.randn(10, 100)

        with pytest.raises(RuntimeError) as excinfo:
            pipe.predict(X)

        assert "not fitted" in str(excinfo.value).lower()

    def test_get_params(self):
        """Test sklearn get_params interface."""
        pipe = NIRSPipeline()
        params = pipe.get_params()
        assert isinstance(params, dict)
        assert "fold" in params

    def test_set_params(self):
        """Test sklearn set_params interface."""
        pipe = NIRSPipeline()
        pipe.set_params(fold=2)
        assert pipe._fold == 2

    def test_repr_not_fitted(self):
        """Test repr for unfitted pipeline."""
        pipe = NIRSPipeline()
        assert "not fitted" in repr(pipe)


class TestNIRSPipelineFromBundle:
    """Test NIRSPipeline.from_bundle() functionality."""

    def test_from_bundle_file_not_found(self):
        """Test that from_bundle raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            NIRSPipeline.from_bundle("/nonexistent/path/model.n4a")

    def test_from_bundle_creates_fitted_pipeline(self, tmp_path, mock_bundle):
        """Test that from_bundle creates a fitted pipeline."""
        bundle_path = tmp_path / "test_model.n4a"
        mock_bundle(bundle_path)

        with patch("nirs4all.pipeline.bundle.BundleLoader") as MockLoader:
            mock_loader = MagicMock()
            mock_loader.metadata = MagicMock(
                preprocessing_chain="MinMaxScaler -> SNV",
                model_step_index=3,
                original_manifest={"name": "test_model"}
            )
            mock_loader.fold_weights = {0: 0.5, 1: 0.5}
            mock_loader.predict.return_value = np.array([1.0, 2.0, 3.0])
            MockLoader.return_value = mock_loader

            # Need to patch where it's imported, not defined
            with patch.object(NIRSPipeline, '_from_bundle_internal') as mock_internal:
                # Create a real instance manually
                instance = NIRSPipeline()
                instance._bundle_loader = mock_loader
                instance._is_fitted = True
                instance._preprocessing_chain = "MinMaxScaler -> SNV"
                instance._model_step_index = 3
                instance._fold_weights = {0: 0.5, 1: 0.5}
                mock_internal.return_value = instance

                pipe = NIRSPipeline.from_bundle(bundle_path)

                assert pipe.is_fitted_
                assert pipe.preprocessing_chain == "MinMaxScaler -> SNV"
                assert pipe.model_step_index == 3
                assert pipe.n_folds == 2


class TestNIRSPipelineFromResult:
    """Test NIRSPipeline.from_result() functionality."""

    def test_from_result_no_predictions(self):
        """Test that from_result raises for empty result."""
        mock_result = Mock()
        mock_result.best = {}

        with pytest.raises(ValueError) as excinfo:
            NIRSPipeline.from_result(mock_result)

        assert "No predictions available" in str(excinfo.value)

    def test_from_result_creates_pipeline(self, tmp_path):
        """Test that from_result creates a pipeline via export."""
        # Create a mock instance manually
        instance = NIRSPipeline()
        instance._is_fitted = True
        instance._preprocessing_chain = "MinMaxScaler"

        with patch.object(NIRSPipeline, '_from_bundle_internal', return_value=instance):
            # Setup mock result
            mock_result = Mock()
            mock_result.best = {
                "model_name": "PLSRegression",
                "rmse": 0.5,
                "r2": 0.9
            }
            mock_result._runner = Mock()

            # Mock export to create a real file
            def mock_export(path, source=None):
                Path(path).touch()
                return Path(path)
            mock_result.export = mock_export

            with patch("tempfile.mkdtemp", return_value=str(tmp_path)):
                pipe = NIRSPipeline.from_result(mock_result)

            assert pipe.is_fitted_


class TestNIRSPipelinePredict:
    """Test NIRSPipeline.predict() functionality."""

    def test_predict_returns_array(self, tmp_path):
        """Test that predict returns numpy array."""
        bundle_path = tmp_path / "model.n4a"
        bundle_path.touch()

        mock_loader = MagicMock()
        mock_loader.metadata = MagicMock(
            preprocessing_chain="",
            model_step_index=1,
            original_manifest={}
        )
        mock_loader.fold_weights = {}
        mock_loader.predict.return_value = np.array([1.0, 2.0, 3.0])

        # Create instance manually
        pipe = NIRSPipeline()
        pipe._bundle_loader = mock_loader
        pipe._is_fitted = True

        X = np.random.randn(3, 100)
        y_pred = pipe.predict(X)

        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == (3,)
        mock_loader.predict.assert_called_once()


class TestNIRSPipelineScore:
    """Test NIRSPipeline.score() functionality."""

    def test_score_computes_r2(self, tmp_path):
        """Test that score computes R² correctly."""
        mock_loader = MagicMock()
        mock_loader.metadata = MagicMock(
            preprocessing_chain="",
            model_step_index=1,
            original_manifest={}
        )
        mock_loader.fold_weights = {}
        # Return same as y_true for perfect score
        mock_loader.predict.return_value = np.array([1.0, 2.0, 3.0])

        # Create instance manually
        pipe = NIRSPipeline()
        pipe._bundle_loader = mock_loader
        pipe._is_fitted = True

        X = np.random.randn(3, 100)
        y = np.array([1.0, 2.0, 3.0])

        score = pipe.score(X, y)

        assert isinstance(score, float)
        assert score == pytest.approx(1.0, rel=1e-5)


class TestNIRSPipelineModelAccess:
    """Test NIRSPipeline.model_ property."""

    def test_model_returns_underlying_model(self, tmp_path):
        """Test that model_ returns the underlying model."""
        mock_model = Mock()
        mock_model.__class__.__name__ = "PLSRegression"

        mock_artifact_provider = MagicMock()
        mock_artifact_provider.get_fold_artifacts.return_value = [(0, mock_model)]

        mock_loader = MagicMock()
        mock_loader.metadata = MagicMock(
            preprocessing_chain="",
            model_step_index=2,
            original_manifest={}
        )
        mock_loader.fold_weights = {0: 1.0}
        mock_loader.artifact_provider = mock_artifact_provider

        # Create instance manually
        pipe = NIRSPipeline()
        pipe._bundle_loader = mock_loader
        pipe._is_fitted = True
        pipe._model_step_index = 2

        model = pipe.model_

        assert model is mock_model

    def test_shap_model_alias(self, tmp_path):
        """Test that shap_model is an alias for model_."""
        mock_model = Mock()
        mock_artifact_provider = MagicMock()
        mock_artifact_provider.get_fold_artifacts.return_value = [(0, mock_model)]

        mock_loader = MagicMock()
        mock_loader.metadata = MagicMock(
            preprocessing_chain="",
            model_step_index=2,
            original_manifest={}
        )
        mock_loader.fold_weights = {0: 1.0}
        mock_loader.artifact_provider = mock_artifact_provider

        # Create instance manually
        pipe = NIRSPipeline()
        pipe._bundle_loader = mock_loader
        pipe._is_fitted = True
        pipe._model_step_index = 2

        assert pipe.shap_model is pipe.model_


class TestNIRSPipelineClassifier:
    """Test NIRSPipelineClassifier functionality."""

    def test_init_not_fitted(self):
        """Test that a fresh classifier is not fitted."""
        clf = NIRSPipelineClassifier()
        assert not clf.is_fitted_
        assert clf._classes is None

    def test_fit_raises_not_implemented(self):
        """Test that fit() raises NotImplementedError."""
        clf = NIRSPipelineClassifier()
        X = np.random.randn(10, 100)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        with pytest.raises(NotImplementedError):
            clf.fit(X, y)

    def test_score_computes_accuracy(self, tmp_path):
        """Test that classifier score computes accuracy."""
        mock_loader = MagicMock()
        mock_loader.metadata = MagicMock(
            preprocessing_chain="",
            model_step_index=1,
            original_manifest={}
        )
        mock_loader.fold_weights = {}
        # Return same as y_true for perfect accuracy
        mock_loader.predict.return_value = np.array([0, 1, 0, 1])

        # Create instance manually
        clf = NIRSPipelineClassifier()
        clf._bundle_loader = mock_loader
        clf._is_fitted = True
        clf._classes = np.array([0, 1])

        X = np.random.randn(4, 100)
        y = np.array([0, 1, 0, 1])

        score = clf.score(X, y)

        assert isinstance(score, float)
        assert score == 1.0

    def test_repr(self):
        """Test repr for classifier."""
        clf = NIRSPipelineClassifier()
        assert "NIRSPipelineClassifier" in repr(clf)
        assert "not fitted" in repr(clf)


# Fixtures

@pytest.fixture
def mock_bundle(tmp_path):
    """Create a mock bundle file."""
    def _create(path: Path):
        path.touch()
        return path
    return _create


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 500)
    y = np.random.randn(100)
    return X, y
