"""
Tests for NIRSPipelineClassifier sklearn wrapper.

This module tests the sklearn-compatible classifier wrapper functionality:
- NIRSPipelineClassifier initialization and basic properties
- from_bundle() loading
- from_result() creation
- predict() and score() methods
- predict_proba() method
- classes_ property
- model_ property for SHAP access
- sklearn interface (get_params, set_params)
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from nirs4all.sklearn import NIRSPipelineClassifier


class TestNIRSPipelineClassifierInit:
    """Test NIRSPipelineClassifier initialization and basic properties."""

    def test_init_not_fitted(self):
        """Test that a fresh classifier is not fitted."""
        clf = NIRSPipelineClassifier()
        assert not clf.is_fitted_
        assert clf._bundle_loader is None
        assert clf._classes is None

    def test_fit_raises_not_implemented(self):
        """Test that fit() raises NotImplementedError."""
        clf = NIRSPipelineClassifier()
        X = np.random.randn(10, 100)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        with pytest.raises(NotImplementedError) as excinfo:
            clf.fit(X, y)

        assert "NIRSPipeline.fit() is not supported" in str(excinfo.value)
        assert "nirs4all.run()" in str(excinfo.value)

    def test_predict_not_fitted_raises(self):
        """Test that predict() on unfitted classifier raises error."""
        clf = NIRSPipelineClassifier()
        X = np.random.randn(10, 100)

        with pytest.raises(RuntimeError) as excinfo:
            clf.predict(X)

        assert "not fitted" in str(excinfo.value).lower()

    def test_get_params(self):
        """Test sklearn get_params interface."""
        clf = NIRSPipelineClassifier()
        params = clf.get_params()
        assert isinstance(params, dict)
        assert "fold" in params

    def test_set_params(self):
        """Test sklearn set_params interface."""
        clf = NIRSPipelineClassifier()
        clf.set_params(fold=2)
        assert clf._fold == 2

    def test_set_params_invalidates_model_cache(self):
        """Test that set_params(fold=...) invalidates the cached model."""
        clf = NIRSPipelineClassifier()
        clf._cached_model = Mock()
        clf.set_params(fold=1)
        assert clf._cached_model is None

    def test_repr_not_fitted(self):
        """Test repr for unfitted classifier."""
        clf = NIRSPipelineClassifier()
        assert "NIRSPipelineClassifier" in repr(clf)
        assert "not fitted" in repr(clf)

    def test_repr_fitted_with_classes(self):
        """Test repr for fitted classifier with classes."""
        clf = NIRSPipelineClassifier()
        clf._is_fitted = True
        clf._model_name = "RandomForestClassifier"
        clf._classes = np.array([0, 1, 2])

        r = repr(clf)
        assert "fitted" in r
        assert "RandomForestClassifier" in r
        assert "n_classes=3" in r

class TestNIRSPipelineClassifierFromBundle:
    """Test NIRSPipelineClassifier.from_bundle() functionality."""

    def test_from_bundle_file_not_found(self):
        """Test that from_bundle raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            NIRSPipelineClassifier.from_bundle("/nonexistent/path/model.n4a")

    def test_from_bundle_creates_fitted_classifier(self, tmp_path):
        """Test that from_bundle creates a fitted classifier."""
        bundle_path = tmp_path / "test_model.n4a"
        bundle_path.touch()

        mock_loader = MagicMock()
        mock_loader.metadata = MagicMock(
            preprocessing_chain="MinMaxScaler -> SNV",
            model_step_index=3,
            original_manifest={"name": "test_clf"}
        )
        mock_loader.fold_weights = {0: 0.5, 1: 0.5}
        mock_loader.predict.return_value = np.array([0, 1, 0])

        with patch.object(NIRSPipelineClassifier, '_from_bundle_internal_classifier') as mock_internal:
            instance = NIRSPipelineClassifier()
            instance._bundle_loader = mock_loader
            instance._is_fitted = True
            instance._preprocessing_chain = "MinMaxScaler -> SNV"
            instance._model_step_index = 3
            instance._fold_weights = {0: 0.5, 1: 0.5}
            mock_internal.return_value = instance

            clf = NIRSPipelineClassifier.from_bundle(bundle_path)

            assert clf.is_fitted_
            assert clf.preprocessing_chain == "MinMaxScaler -> SNV"
            assert clf.model_step_index == 3
            assert clf.n_folds == 2

class TestNIRSPipelineClassifierFromResult:
    """Test NIRSPipelineClassifier.from_result() functionality."""

    def test_from_result_no_predictions_raises(self):
        """Test that from_result raises for empty result."""
        mock_result = Mock()
        mock_result.best = {}
        mock_result.final = None

        with pytest.raises(ValueError) as excinfo:
            NIRSPipelineClassifier.from_result(mock_result)

        assert "No predictions available" in str(excinfo.value)

    def test_from_result_creates_classifier(self, tmp_path):
        """Test that from_result creates a classifier via export."""
        instance = NIRSPipelineClassifier()
        instance._is_fitted = True
        instance._preprocessing_chain = "MinMaxScaler"
        instance._classes = np.array([0, 1])

        with patch.object(NIRSPipelineClassifier, '_from_bundle_internal_classifier', return_value=instance):
            mock_result = Mock()
            mock_result.best = {"model_name": "RandomForestClassifier", "classes": [0, 1]}
            mock_result._runner = Mock()
            mock_result.final = None

            def mock_export(path, source=None):
                Path(path).touch()
                return Path(path)
            mock_result.export = mock_export

            with patch("tempfile.mkdtemp", return_value=str(tmp_path)):
                clf = NIRSPipelineClassifier.from_result(mock_result)

            assert clf.is_fitted_

    def test_from_result_extracts_classes_from_source(self, tmp_path):
        """Test that from_result extracts classes from prediction source."""
        instance = NIRSPipelineClassifier()
        instance._is_fitted = True

        with patch.object(NIRSPipelineClassifier, '_from_bundle_internal_classifier', return_value=instance):
            mock_result = Mock()
            mock_result.best = {"model_name": "SVC", "classes": [0, 1, 2]}
            mock_result._runner = Mock()
            mock_result.final = None

            def mock_export(path, source=None):
                Path(path).touch()
            mock_result.export = mock_export

            with patch("tempfile.mkdtemp", return_value=str(tmp_path)):
                clf = NIRSPipelineClassifier.from_result(mock_result)

            assert clf._classes is not None
            np.testing.assert_array_equal(clf._classes, [0, 1, 2])

class TestNIRSPipelineClassifierPredict:
    """Test NIRSPipelineClassifier.predict() functionality."""

    def test_predict_returns_class_labels(self):
        """Test that predict returns class label array."""
        mock_loader = MagicMock()
        mock_loader.predict.return_value = np.array([0, 1, 0, 1])

        clf = NIRSPipelineClassifier()
        clf._bundle_loader = mock_loader
        clf._is_fitted = True
        clf._classes = np.array([0, 1])

        X = np.random.randn(4, 100)
        y_pred = clf.predict(X)

        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == (4,)
        mock_loader.predict.assert_called_once()

    def test_predict_multiclass_probabilities_to_labels(self):
        """Test that multi-class probability outputs are converted to labels."""
        # Return probabilities (n_samples, n_classes)
        proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
        mock_loader = MagicMock()
        mock_loader.predict.return_value = proba

        clf = NIRSPipelineClassifier()
        clf._bundle_loader = mock_loader
        clf._is_fitted = True
        clf._classes = np.array(["cat", "dog"])

        X = np.random.randn(3, 100)
        y_pred = clf.predict(X)

        # argmax of [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]] -> [0, 1, 0] -> ["cat", "dog", "cat"]
        assert y_pred[0] == "cat"
        assert y_pred[1] == "dog"
        assert y_pred[2] == "cat"

    def test_predict_not_initialized_raises(self):
        """Test that predict raises when no bundle loader is set."""
        clf = NIRSPipelineClassifier()
        clf._is_fitted = True  # manually set fitted but no loader

        X = np.random.randn(3, 100)
        with pytest.raises(RuntimeError) as excinfo:
            clf.predict(X)

        assert "not properly initialized" in str(excinfo.value).lower()

    def test_predict_converts_numpy_input(self):
        """Test that predict accepts list input and converts to numpy."""
        mock_loader = MagicMock()
        mock_loader.predict.return_value = np.array([0, 1])

        clf = NIRSPipelineClassifier()
        clf._bundle_loader = mock_loader
        clf._is_fitted = True

        X = [[1.0] * 10, [2.0] * 10]  # list of lists
        y_pred = clf.predict(X)

        assert isinstance(y_pred, np.ndarray)
        # Verify that np.asarray was called (the actual call uses numpy)
        call_args = mock_loader.predict.call_args[0][0]
        assert isinstance(call_args, np.ndarray)

class TestNIRSPipelineClassifierScore:
    """Test NIRSPipelineClassifier.score() functionality."""

    def test_score_computes_accuracy(self):
        """Test that classifier score computes accuracy."""
        mock_loader = MagicMock()
        mock_loader.predict.return_value = np.array([0, 1, 0, 1])

        clf = NIRSPipelineClassifier()
        clf._bundle_loader = mock_loader
        clf._is_fitted = True
        clf._classes = np.array([0, 1])

        X = np.random.randn(4, 100)
        y = np.array([0, 1, 0, 1])

        score = clf.score(X, y)

        assert isinstance(score, float)
        assert score == pytest.approx(1.0)

    def test_score_partial_accuracy(self):
        """Test that classifier score handles partial matches."""
        mock_loader = MagicMock()
        # Predict 3 out of 4 correctly
        mock_loader.predict.return_value = np.array([0, 1, 1, 1])

        clf = NIRSPipelineClassifier()
        clf._bundle_loader = mock_loader
        clf._is_fitted = True

        X = np.random.randn(4, 100)
        y = np.array([0, 1, 0, 1])  # index 2 is wrong

        score = clf.score(X, y)

        assert score == pytest.approx(0.75)

class TestNIRSPipelineClassifierPredictProba:
    """Test NIRSPipelineClassifier.predict_proba() functionality."""

    def test_predict_proba_with_sklearn_model(self):
        """Test predict_proba when underlying model has predict_proba."""
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])

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
        mock_loader.trace = None

        clf = NIRSPipelineClassifier()
        clf._bundle_loader = mock_loader
        clf._is_fitted = True
        clf._model_step_index = 2
        clf._classes = np.array([0, 1])

        X = np.random.randn(2, 10)

        # Mock transform to return X unchanged
        with patch.object(clf, 'transform', return_value=X):
            proba = clf.predict_proba(X)

        assert proba.shape == (2, 2)
        assert proba[0, 0] == pytest.approx(0.8)
        assert proba[1, 1] == pytest.approx(0.7)

    def test_predict_proba_fallback_when_model_returns_proba_directly(self):
        """Test predict_proba fallback when bundle returns probability matrix."""
        proba_matrix = np.array([[0.7, 0.3], [0.4, 0.6], [0.9, 0.1]])

        mock_loader = MagicMock()
        mock_loader.predict.return_value = proba_matrix
        mock_loader.trace = None

        # Mock model without predict_proba
        mock_model = Mock(spec=[])  # no attributes
        mock_artifact_provider = MagicMock()
        mock_artifact_provider.get_fold_artifacts.return_value = [(0, mock_model)]

        mock_loader.metadata = MagicMock(
            preprocessing_chain="",
            model_step_index=1,
            original_manifest={}
        )
        mock_loader.fold_weights = {0: 1.0}
        mock_loader.artifact_provider = mock_artifact_provider

        clf = NIRSPipelineClassifier()
        clf._bundle_loader = mock_loader
        clf._is_fitted = True
        clf._model_step_index = 1
        clf._classes = np.array([0, 1])

        X = np.random.randn(3, 10)

        with patch.object(clf, 'transform', return_value=X):
            proba = clf.predict_proba(X)

        assert proba.shape == (3, 2)

    def test_predict_proba_not_fitted_raises(self):
        """Test that predict_proba raises when not fitted."""
        clf = NIRSPipelineClassifier()
        X = np.random.randn(3, 10)

        with pytest.raises(RuntimeError):
            clf.predict_proba(X)

class TestNIRSPipelineClassifierClasses:
    """Test NIRSPipelineClassifier.classes_ property."""

    def test_classes_returns_array_when_set(self):
        """Test that classes_ returns the set array."""
        clf = NIRSPipelineClassifier()
        clf._is_fitted = True
        clf._classes = np.array([0, 1, 2])

        np.testing.assert_array_equal(clf.classes_, [0, 1, 2])

    def test_classes_raises_when_not_set(self):
        """Test that classes_ raises when classes cannot be determined."""
        clf = NIRSPipelineClassifier()
        clf._is_fitted = True
        clf._classes = None
        # No bundle loader to extract from

        with pytest.raises(RuntimeError) as excinfo:
            _ = clf.classes_

        assert "Could not determine class labels" in str(excinfo.value)

    def test_classes_extracted_from_model(self):
        """Test that classes_ can be extracted from the underlying model."""
        mock_model = Mock()
        mock_model.classes_ = np.array(["A", "B", "C"])

        mock_artifact_provider = MagicMock()
        mock_artifact_provider.get_fold_artifacts.return_value = [(0, mock_model)]

        mock_loader = MagicMock()
        mock_loader.metadata = MagicMock(
            preprocessing_chain="",
            model_step_index=1,
            original_manifest={}
        )
        mock_loader.fold_weights = {0: 1.0}
        mock_loader.artifact_provider = mock_artifact_provider

        clf = NIRSPipelineClassifier()
        clf._bundle_loader = mock_loader
        clf._is_fitted = True
        clf._model_step_index = 1

        classes = clf.classes_
        np.testing.assert_array_equal(classes, ["A", "B", "C"])

class TestNIRSPipelineClassifierInheritance:
    """Test that NIRSPipelineClassifier inherits regressor functionality."""

    def test_is_fitted_property(self):
        """Test that is_fitted_ property works."""
        clf = NIRSPipelineClassifier()
        assert clf.is_fitted_ is False

        clf._is_fitted = True
        assert clf.is_fitted_ is True

    def test_model_step_index_property(self):
        """Test model_step_index property inherited from NIRSPipeline."""
        clf = NIRSPipelineClassifier()
        clf._model_step_index = 5
        assert clf.model_step_index == 5

    def test_fold_weights_property(self):
        """Test fold_weights property inherited from NIRSPipeline."""
        clf = NIRSPipelineClassifier()
        clf._fold_weights = {0: 0.4, 1: 0.6}
        assert clf.fold_weights == {0: 0.4, 1: 0.6}

    def test_n_folds_property(self):
        """Test n_folds property inherited from NIRSPipeline."""
        clf = NIRSPipelineClassifier()
        clf._fold_weights = {0: 0.5, 1: 0.5}
        assert clf.n_folds == 2

    def test_model_name_property(self):
        """Test model_name property inherited from NIRSPipeline."""
        clf = NIRSPipelineClassifier()
        clf._model_name = "SVC"
        assert clf.model_name == "SVC"

    def test_preprocessing_chain_property(self):
        """Test preprocessing_chain property inherited from NIRSPipeline."""
        clf = NIRSPipelineClassifier()
        clf._preprocessing_chain = "SNV -> MinMaxScaler"
        assert clf.preprocessing_chain == "SNV -> MinMaxScaler"

    def test_shap_model_alias(self):
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

        clf = NIRSPipelineClassifier()
        clf._bundle_loader = mock_loader
        clf._is_fitted = True
        clf._model_step_index = 2

        assert clf.shap_model is clf.model_

# Fixtures

@pytest.fixture
def mock_bundle(tmp_path):
    """Create a mock bundle file."""
    def _create(path: Path):
        path.touch()
        return path
    return _create

@pytest.fixture
def fitted_classifier():
    """Create a fitted NIRSPipelineClassifier with mock bundle loader."""
    mock_loader = MagicMock()
    mock_loader.metadata = MagicMock(
        preprocessing_chain="MinMaxScaler",
        model_step_index=1,
        original_manifest={"name": "test_clf"}
    )
    mock_loader.fold_weights = {0: 1.0}
    mock_loader.predict.return_value = np.array([0, 1, 0])

    clf = NIRSPipelineClassifier()
    clf._bundle_loader = mock_loader
    clf._is_fitted = True
    clf._classes = np.array([0, 1])
    return clf

class TestNIRSPipelineClassifierWithFixture:
    """Additional tests using the fitted_classifier fixture."""

    def test_predict_calls_bundle_loader(self, fitted_classifier):
        """Test that predict delegates to bundle loader."""
        X = np.random.randn(3, 100)
        fitted_classifier.predict(X)
        fitted_classifier._bundle_loader.predict.assert_called_once()

    def test_score_uses_accuracy_metric(self, fitted_classifier):
        """Test that score uses accuracy (not R2)."""
        fitted_classifier._bundle_loader.predict.return_value = np.array([0, 1, 0])
        X = np.random.randn(3, 100)
        y = np.array([0, 1, 0])

        score = fitted_classifier.score(X, y)
        # Accuracy should be 1.0 for perfect match
        assert score == pytest.approx(1.0)

    def test_classes_returns_set_value(self, fitted_classifier):
        """Test classes_ returns the configured value."""
        np.testing.assert_array_equal(fitted_classifier.classes_, [0, 1])
