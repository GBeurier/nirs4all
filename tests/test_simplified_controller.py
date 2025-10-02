"""
Unit Tests for Simplified Base Model Controller

These tests verify that the new simplified controller works correctly
with the basic functionality as defined in the user's pseudo-code.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from nirs4all.controllers.models.base_model_controller import BaseModelController
from nirs4all.controllers.models.model_controller_helper import ModelControllerHelper
from nirs4all.controllers.models.prediction_store import PredictionStore


# Mock implementations for testing
class TestSklearnController(BaseModelController):
    """Test implementation of BaseModelController for sklearn models."""

    @classmethod
    def matches(cls, step, operator, keyword):
        return True  # For testing, always match

    def _get_model_instance(self, model_config):
        if 'model_instance' in model_config:
            return model_config['model_instance']
        return LinearRegression()

    def _train_model(self, model, X_train, y_train, X_val=None, y_val=None, **kwargs):
        model.fit(X_train, y_train.ravel())
        return model

    def _predict_model(self, model, X):
        predictions = model.predict(X)
        return predictions.reshape(-1, 1) if predictions.ndim == 1 else predictions

    def _prepare_data(self, X, y, context):
        if X is None or y is None:
            return None, None
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return X, y

    def _evaluate_model(self, model, X_val, y_val):
        from sklearn.metrics import mean_squared_error
        y_pred = model.predict(X_val)
        return mean_squared_error(y_val.ravel(), y_pred)


class TestModelControllerHelper:
    """Test ModelControllerHelper functionality."""

    def test_create_model_id(self):
        """Test model ID creation."""
        helper = ModelControllerHelper()

        # Mock runner
        runner = MagicMock()
        runner.next_op.return_value = 42

        model_id = helper.create_model_id("test_model", runner)
        assert model_id == "test_model_42"
        runner.next_op.assert_called_once()

    def test_create_model_uuid(self):
        """Test model UUID creation."""
        helper = ModelControllerHelper()

        # Mock runner
        runner = MagicMock()
        runner.saver.pipeline_name = "test_pipeline"

        # Without fold
        uuid = helper.create_model_uuid("model_42", runner, 1, "test_config")
        assert "model_42" in uuid

        # With fold
        uuid_fold = helper.create_model_uuid("model_42", runner, 1, "test_config", fold_idx=1)
        assert uuid_fold == "model_42_fold1_test_pipeline"

    def test_clone_model(self):
        """Test model cloning."""
        helper = ModelControllerHelper()

        # Test sklearn model cloning
        original_model = LinearRegression()
        cloned_model = helper.clone_model(original_model)

        assert cloned_model is not original_model
        assert type(cloned_model) == type(original_model)

    def test_calculate_scores_regression(self):
        """Test score calculation for regression."""
        utils = ModelUtils()

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

        scores = utils.calculate_scores(y_true, y_pred, task_type="regression")

        assert 'mse' in scores
        assert 'rmse' in scores
        assert 'mae' in scores
        assert 'r2' in scores
        assert all(isinstance(v, float) for v in scores.values())

    def test_extract_model_name_from_config(self):
        """Test model name extraction from config."""
        helper = ModelControllerHelper()

        # Test with model instance
        model = LinearRegression()
        config1 = {'model_instance': model}
        name1 = helper.extract_name_from_config(config1)
        assert name1 == "LinearRegression"

        # Test with custom name
        config2 = {'name': 'CustomModel', 'model_instance': model}
        name2 = helper.extract_name_from_config(config2)
        assert name2 == "CustomModel"

        # Test with direct model (wrapped in config)
        config3 = {'model_instance': model}
        name3 = helper.extract_name_from_config(config3)
        assert name3 == "LinearRegression"


class TestPredictionStore:
    """Test PredictionStore functionality."""

    def test_ensure_2d(self):
        """Test array dimension ensuring."""
        store = PredictionStore()

        # 1D array
        arr_1d = np.array([1, 2, 3])
        result_1d = store._ensure_2d(arr_1d)
        assert result_1d.shape == (3, 1)

        # 2D array
        arr_2d = np.array([[1, 2], [3, 4]])
        result_2d = store._ensure_2d(arr_2d)
        assert result_2d.shape == (2, 2)

        # Scalar
        arr_0d = np.array(5)
        result_0d = store._ensure_2d(arr_0d)
        assert result_0d.shape == (1, 1)

    def test_create_prediction_csv(self):
        """Test CSV creation."""
        store = PredictionStore()

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])

        csv_content = store.create_prediction_csv(y_true, y_pred)

        assert "y_true" in csv_content
        assert "y_pred" in csv_content
        assert "sample_index" in csv_content
        assert "1.0" in csv_content
        assert "1.1" in csv_content


class TestBaseModelController:
    """Test BaseModelController functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.controller = TestSklearnController()

        # Create mock dataset
        self.dataset = MagicMock()
        self.dataset.num_features = 10
        self.dataset.num_samples = 100
        self.dataset._predictions = MagicMock()
        self.dataset.folds = None

        # Create mock data
        self.X_train = np.random.randn(80, 10)
        self.y_train = np.random.randn(80, 1)
        self.X_test = np.random.randn(20, 10)
        self.y_test = np.random.randn(20, 1)

        # Mock dataset methods
        self.dataset.train_data.return_value = (self.X_train, self.y_train)
        self.dataset.test_data.return_value = (self.X_test, self.y_test)

        # Create mock runner
        self.runner = MagicMock()
        self.runner.next_op.return_value = 1
        self.runner.saver.dataset_name = "test_dataset"
        self.runner.saver.pipeline_name = "test_pipeline"
        self.runner.saver.current_path = "test/path"

        # Mock context
        self.context = {'y': 'numeric'}

    def test_extract_model_config(self):
        """Test model config extraction."""
        model = LinearRegression()

        # Test with dict config
        step = {'model': model, 'name': 'TestModel'}
        config = self.controller._extract_model_config(step)
        assert config['model_instance'] == model
        assert config['name'] == 'TestModel'

        # Test with direct model
        config2 = self.controller._extract_model_config(model)
        assert config2['model_instance'] == model

    def test_train_single_fold(self):
        """Test training without folds."""
        model_config = {'model_instance': LinearRegression()}

        # Mock the data retrieval
        self.controller._get_data_from_dataset = MagicMock(
            return_value=[self.X_train, self.y_train, self.X_test, self.y_test]
        )

        predictions = self.controller.train(
            model_config, self.X_train, self.y_train, self.X_test, self.y_test,
            folds=None, predictions={}, context=self.context,
            runner=self.runner, dataset=self.dataset
        )

        assert isinstance(predictions, dict)
        assert len(predictions) > 0

    def test_train_with_folds(self):
        """Test training with cross-validation folds."""
        model_config = {'model_instance': LinearRegression()}

        # Create mock folds
        folds = [
            (np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])),
            (np.array([4, 5, 6, 7]), np.array([0, 1, 2, 3]))
        ]

        predictions = self.controller.train(
            model_config, self.X_train, self.y_train, self.X_test, self.y_test,
            folds=folds, predictions={}, context=self.context,
            runner=self.runner, dataset=self.dataset
        )

        assert isinstance(predictions, dict)
        assert len(predictions) > 0  # Should have predictions from both folds

    def test_launch_training(self):
        """Test the core training launch functionality."""
        model_config = {'model_instance': LinearRegression()}

        fold_predictions = self.controller.launch_training(
            model_config, self.X_train, self.y_train, self.X_test, self.y_test,
            self.X_test, self.y_test, {}, self.context, self.runner, self.dataset
        )

        assert isinstance(fold_predictions, dict)
        assert len(fold_predictions) > 0

        # Check that predictions contain required keys
        for key, pred_data in fold_predictions.items():
            assert 'dataset' in pred_data
            assert 'pipeline' in pred_data
            assert 'model' in pred_data
            assert 'y_true' in pred_data
            assert 'y_pred' in pred_data

    @patch('nirs4all.controllers.models.base_model_controller.OPTUNA_AVAILABLE', True)
    def test_finetune_no_folds(self):
        """Test finetuning without folds."""
        model_config = {'model_instance': LinearRegression()}
        finetune_params = {'n_trials': 2, 'approach': 'individual'}

        # Mock optuna
        with patch('optuna.create_study') as mock_create_study:
            mock_study = MagicMock()
            mock_study.best_params = {'fit_intercept': True}
            mock_create_study.return_value = mock_study

            best_params = self.controller.finetune(
                model_config, self.X_train, self.y_train, self.X_test, self.y_test,
                folds=None, finetune_params=finetune_params, predictions={},
                context=self.context, runner=self.runner, dataset=self.dataset
            )

            assert best_params == {'fit_intercept': True}

    def test_prediction_mode(self):
        """Test prediction mode execution."""
        import pickle

        # Create a trained model binary
        trained_model = LinearRegression()
        trained_model.fit(self.X_train, self.y_train.ravel())
        model_binary = pickle.dumps(trained_model)

        loaded_binaries = [('trained_model.pkl', model_binary)]
        model_config = {'model_instance': LinearRegression()}

        # Mock prediction data
        self.controller._get_prediction_data = MagicMock(
            return_value={'X': self.X_test, 'y': self.y_test}
        )

        context, binaries = self.controller._execute_prediction_mode(
            model_config, self.dataset, self.context, self.runner, loaded_binaries
        )

        assert context == self.context
        assert len(binaries) > 0
        assert binaries[0][0].endswith('.csv')

    def test_execute_train_mode(self):
        """Test full execute method in train mode."""
        step = {'model': LinearRegression()}

        # Mock the data retrieval
        self.controller._get_data_from_dataset = MagicMock(
            return_value=[self.X_train, self.y_train, self.X_test, self.y_test]
        )

        context, binaries = self.controller.execute(
            step, None, self.dataset, self.context, self.runner, mode="train"
        )

        assert context == self.context
        assert isinstance(binaries, list)


if __name__ == "__main__":
    pytest.main([__file__])