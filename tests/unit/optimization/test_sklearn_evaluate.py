"""Unit tests for SklearnModelController._evaluate_model (Phase 3.1 - ISSUE-8)."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


class TestEvaluateModelRegressor:
    """Tests for _evaluate_model with regressors — should use MSE on predictions."""

    @pytest.fixture
    def controller(self):
        from nirs4all.controllers.models.sklearn_model import SklearnModelController
        return SklearnModelController()

    def test_returns_mse_for_regressor(self, controller):
        """_evaluate_model should return MSE (not re-fit via cross_val_score)."""
        model = PLSRegression(n_components=2)
        X_train = np.random.RandomState(42).randn(50, 10)
        y_train = np.random.RandomState(42).randn(50, 1)
        X_val = np.random.RandomState(43).randn(20, 10)
        y_val = np.random.RandomState(43).randn(20, 1)

        model.fit(X_train, y_train)
        score = controller._evaluate_model(model, X_val, y_val)

        # MSE is always >= 0
        assert score >= 0.0
        assert score != float('inf')

    def test_score_matches_manual_mse(self, controller):
        """Score should match hand-computed MSE."""
        from sklearn.metrics import mean_squared_error

        model = PLSRegression(n_components=2)
        X_train = np.random.RandomState(42).randn(50, 10)
        y_train = np.random.RandomState(42).randn(50, 1)
        X_val = np.random.RandomState(43).randn(20, 10)
        y_val = np.random.RandomState(43).randn(20, 1)

        model.fit(X_train, y_train)
        score = controller._evaluate_model(model, X_val, y_val)

        # Manual MSE
        y_pred = model.predict(X_val).ravel()
        expected_mse = mean_squared_error(y_val.ravel(), y_pred)

        assert abs(score - expected_mse) < 1e-10

    def test_2d_y_val_handled(self, controller):
        """2D y_val should be handled correctly."""
        model = PLSRegression(n_components=2)
        X_train = np.random.RandomState(42).randn(50, 10)
        y_train = np.random.RandomState(42).randn(50, 1)
        X_val = np.random.RandomState(43).randn(20, 10)
        y_val_2d = np.random.RandomState(43).randn(20, 1)

        model.fit(X_train, y_train)
        score = controller._evaluate_model(model, X_val, y_val_2d)
        assert score >= 0.0

class TestEvaluateModelClassifier:
    """Tests for _evaluate_model with classifiers — should use negative balanced accuracy."""

    @pytest.fixture
    def controller(self):
        from nirs4all.controllers.models.sklearn_model import SklearnModelController
        return SklearnModelController()

    def test_returns_negative_balanced_accuracy_for_classifier(self, controller):
        """_evaluate_model should return -balanced_accuracy for classifiers."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X[:80], y[:80])

        score = controller._evaluate_model(model, X[80:], y[80:])

        # Negative balanced accuracy: should be in [-1, 0]
        assert -1.0 <= score <= 0.0

class TestEvaluateModelErrorHandling:
    """Tests for error handling in _evaluate_model."""

    @pytest.fixture
    def controller(self):
        from nirs4all.controllers.models.sklearn_model import SklearnModelController
        return SklearnModelController()

    def test_returns_inf_on_error(self, controller):
        """Should return inf if prediction fails."""
        # Create a model that will fail to predict (not fitted)
        model = PLSRegression(n_components=2)
        X_val = np.random.randn(10, 5)
        y_val = np.random.randn(10, 1)

        score = controller._evaluate_model(model, X_val, y_val)
        assert score == float('inf')
