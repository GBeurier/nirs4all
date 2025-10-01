"""
Test Optuna Integration - Test hyperparameter optimization with Optuna

This test verifies that the modular architecture correctly integrates
with Optuna for hyperparameter optimization.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from nirs4all.controllers.models.abstract_model_controller import AbstractModelController
from nirs4all.controllers.models.sklearn_model_controller import SklearnModelController
from nirs4all.controllers.models.config import FinetuneConfig


class TestOptunaIntegration:
    """Test Optuna integration in the modular architecture."""

    def test_optuna_manager_creation(self):
        """Test that OptunaManager can be created."""
        from nirs4all.controllers.models.optuna_manager import OptunaManager

        # Should not raise an error
        manager = OptunaManager()
        assert manager is not None

    def test_optuna_finetuning_workflow(self):
        """Test the complete finetuning workflow with mocked Optuna."""
        # Create mock data
        X_train = np.random.rand(100, 5)
        y_train = np.random.rand(100)
        X_val = np.random.rand(20, 5)
        y_val = np.random.rand(20)
        X_test = np.random.rand(20, 5)
        y_test = np.random.rand(20)

        # Create model config
        model_config = {
            'model_class': 'sklearn.linear_model.LinearRegression',
            'model_params': {},
            'train_params': {'verbose': 0},
            'finetune_params': {
                'n_trials': 2,
                'approach': 'tpe',
                'model_params': {
                    'fit_intercept': [True, False]
                }
            }
        }

        # Create finetune config
        finetune_config = FinetuneConfig.from_dict(model_config['finetune_params'])

        # Create controller
        controller = SklearnModelController()

        # Mock Optuna to avoid actual optimization
        with patch('nirs4all.controllers.models.optuna_manager.optuna') as mock_optuna:
            # Mock study
            mock_study = Mock()
            mock_study.best_params = {'fit_intercept': True}
            mock_study.best_value = 0.1
            mock_optuna.create_study.return_value = mock_study

            # Mock optimization
            mock_study.optimize = Mock()

            # Execute finetuning
            context = {}
            runner = Mock()
            runner.next_op.return_value = 1
            runner.saver = Mock()
            runner.saver.current_path = "test_path"
            runner.saver.dataset_name = "test_dataset"
            runner.saver.pipeline_name = "test_pipeline"

            dataset = Mock()
            dataset._predictions = Mock()
            dataset._predictions.add_prediction = Mock()

            result_context, binaries = controller._execute_finetune_modular(
                model_config, X_train, y_train, X_val, y_val, X_test, y_test,
                model_config.get('train_params', {}), finetune_config,
                context, runner, dataset
            )

            # Verify Optuna was called
            mock_optuna.create_study.assert_called_once()
            mock_study.optimize.assert_called_once()

            # Verify result structure
            assert isinstance(result_context, dict)
            assert isinstance(binaries, list)

    def test_optuna_fallback_when_unavailable(self):
        """Test that finetuning falls back to regular training when Optuna is unavailable."""
        # Create mock data
        X_train = np.random.rand(100, 5)
        y_train = np.random.rand(100)
        X_val = np.random.rand(20, 5)
        y_val = np.random.rand(20)
        X_test = np.random.rand(20, 5)
        y_test = np.random.rand(20)

        # Create model config
        model_config = {
            'model_class': 'sklearn.linear_model.LinearRegression',
            'model_params': {},
            'train_params': {'verbose': 0},
            'finetune_params': {
                'n_trials': 2,
                'approach': 'tpe',
                'model_params': {
                    'fit_intercept': [True, False]
                }
            }
        }

        # Create finetune config
        finetune_config = FinetuneConfig.from_dict(model_config['finetune_params'])

        # Create controller
        controller = SklearnModelController()

        # Mock Optuna to raise ImportError
        with patch('nirs4all.controllers.models.optuna_manager.OPTUNA_AVAILABLE', False):
            with patch.object(controller, '_execute_train_modular') as mock_train:
                mock_train.return_value = ({}, [])

                # Execute finetuning
                context = {}
                runner = Mock()
                dataset = Mock()

                result_context, binaries = controller._execute_finetune_modular(
                    model_config, X_train, y_train, X_val, y_val, X_test, y_test,
                    model_config.get('train_params', {}), finetune_config,
                    context, runner, dataset
                )

                # Verify fallback was called
                mock_train.assert_called_once()

    def test_optuna_parameter_sampling(self):
        """Test parameter sampling functionality."""
        from nirs4all.controllers.models.optuna_manager import OptunaManager

        manager = OptunaManager()

        # Mock trial
        mock_trial = Mock()
        mock_trial.suggest_categorical = Mock(side_effect=['value1', 'value2'])
        mock_trial.suggest_int = Mock(return_value=5)
        mock_trial.suggest_float = Mock(return_value=0.5)

        # Test different parameter types
        finetune_config = {
            'model_params': {
                'categorical_param': ['option1', 'option2'],
                'int_param': ('int', 1, 10),
                'float_param': (0.0, 1.0),
                'dict_param': {'type': 'categorical', 'choices': ['a', 'b']},
                'fixed_param': 'fixed_value'
            }
        }

        params = manager.sample_hyperparameters(mock_trial, finetune_config)

        # Verify parameters were sampled
        assert 'categorical_param' in params
        assert 'int_param' in params
        assert 'float_param' in params
        assert 'dict_param' in params
        assert 'fixed_param' in params
        assert params['fixed_param'] == 'fixed_value'

    def test_grid_search_detection(self):
        """Test detection of when grid search is suitable."""
        from nirs4all.controllers.models.optuna_manager import OptunaManager

        manager = OptunaManager()

        # Grid search suitable (all categorical)
        grid_config = {
            'model_params': {
                'param1': ['a', 'b'],
                'param2': ['x', 'y']
            }
        }
        assert manager._is_grid_search_suitable(grid_config)

        # Grid search not suitable (has ranges)
        mixed_config = {
            'model_params': {
                'categorical': ['a', 'b'],
                'range_param': (1, 10)
            }
        }
        assert not manager._is_grid_search_suitable(mixed_config)
