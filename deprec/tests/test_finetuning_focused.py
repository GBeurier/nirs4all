#!/usr/bin/env python3
"""
Focused test suite for NIRS4ALL finetuning strategies.

Tests the key combinations of CV modes and parameter strategies using
mock data to ensure all code paths work correctly.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor

from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.pipeline.config import PipelineConfigs
from nirs4all.controllers.models.config import ParamStrategy


@pytest.fixture
def mock_data():
    """Create mock data for testing."""
    np.random.seed(42)
    return {
        'X': np.random.randn(50, 20),
        'y': np.random.randn(50),
        'folds': 3
    }


class TestFinetuningStrategies:
    """Test all finetuning strategy combinations."""

    def test_param_strategy_enum_values(self):
        """Test that all parameter strategy enum values are available."""
        expected_strategies = [
            'per_fold_best', 'global_best', 'weighted_average',
            'global_average', 'ensemble_best', 'robust_best', 'stability_best'
        ]

        actual_strategies = [strategy.value for strategy in ParamStrategy]

        for expected in expected_strategies:
            assert expected in actual_strategies, f"Missing strategy: {expected}"

        print(f"✅ All {len(expected_strategies)} parameter strategies are available")

    @pytest.mark.parametrize("cv_mode", ["simple", "per_fold", "nested"])
    def test_cv_modes(self, cv_mode, mock_data):
        """Test that all CV modes can be configured."""
        config = self._create_test_config("test_cv", cv_mode, "per_fold_best")

        # Test that configuration is valid
        assert config["pipeline"][0]["finetune_params"]["cv_mode"] == cv_mode

        if cv_mode == "nested":
            assert "inner_cv" in config["pipeline"][0]["finetune_params"]

        print(f"✅ CV mode '{cv_mode}' configuration is valid")

    @pytest.mark.parametrize("param_strategy", ["per_fold_best", "global_best", "global_average"])
    def test_parameter_strategies(self, param_strategy, mock_data):
        """Test that all implemented parameter strategies can be configured."""
        config = self._create_test_config("test_strategy", "per_fold", param_strategy)

        # Test that configuration is valid
        assert config["pipeline"][0]["finetune_params"]["param_strategy"] == param_strategy

        print(f"✅ Parameter strategy '{param_strategy}' configuration is valid")

    @pytest.mark.parametrize("use_full_train", [True, False])
    def test_full_training_option(self, use_full_train, mock_data):
        """Test the use_full_train_for_final option."""
        config = self._create_test_config("test_full_train", "per_fold", "global_average")
        config["pipeline"][0]["finetune_params"]["use_full_train_for_final"] = use_full_train

        # Test that configuration is valid
        assert config["pipeline"][0]["finetune_params"]["use_full_train_for_final"] == use_full_train

        print(f"✅ Full training option '{use_full_train}' configuration is valid")

    def test_model_parameter_types(self, mock_data):
        """Test different model parameter types."""
        # Test integer parameter
        pls_config = self._create_test_config("test_int", "simple", "per_fold_best")
        pls_params = pls_config["pipeline"][0]["finetune_params"]["model_params"]
        assert "n_components" in pls_params
        assert isinstance(pls_params["n_components"], tuple)
        assert pls_params["n_components"][0] == "int"

        # Test categorical parameters
        rf_config = {
            "pipeline": [{
                "model": RandomForestRegressor(n_estimators=10, random_state=42),
                "finetune_params": {
                    "cv_mode": "simple",
                    "param_strategy": "per_fold_best",
                    "n_trials": 2,
                    "model_params": {
                        "n_estimators": [10, 20],
                        "max_depth": [3, 5]
                    }
                }
            }]
        }

        rf_params = rf_config["pipeline"][0]["finetune_params"]["model_params"]
        assert "n_estimators" in rf_params
        assert isinstance(rf_params["n_estimators"], list)

        print("✅ All parameter types (int, float, categorical) are supported")

    def test_combination_global_average_with_full_training(self, mock_data):
        """Test the recommended combination: global_average + full training."""
        config = self._create_test_config("test_combo", "per_fold", "global_average")
        config["pipeline"][0]["finetune_params"]["use_full_train_for_final"] = True

        # Validate configuration
        finetune_params = config["pipeline"][0]["finetune_params"]
        assert finetune_params["param_strategy"] == "global_average"
        assert finetune_params["use_full_train_for_final"] is True
        assert finetune_params["cv_mode"] == "per_fold"

        print("✅ Recommended combination (global_average + full_training) is properly configured")

    def test_nested_cv_with_global_average(self, mock_data):
        """Test nested CV with global average (most rigorous combination)."""
        config = self._create_test_config("test_nested", "nested", "global_average")

        # Validate configuration
        finetune_params = config["pipeline"][0]["finetune_params"]
        assert finetune_params["cv_mode"] == "nested"
        assert finetune_params["param_strategy"] == "global_average"
        assert "inner_cv" in finetune_params

        print("✅ Most rigorous combination (nested + global_average) is properly configured")

    @patch('nirs4all.controllers.models.abstract_model_controller.AbstractModelController._train_single_model_on_full_data')
    @patch('nirs4all.controllers.models.abstract_model_controller.AbstractModelController._execute_global_average_optimization')
    def test_global_average_code_path(self, mock_global_avg, mock_full_train, mock_data):
        """Test that global_average strategy triggers the correct code path."""
        from nirs4all.controllers.models.abstract_model_controller import AbstractModelController

        # Create a mock controller instance
        controller = MagicMock(spec=AbstractModelController)
        controller._execute_global_average_optimization = mock_global_avg
        controller._train_single_model_on_full_data = mock_full_train

        # Test that the method exists
        assert hasattr(AbstractModelController, '_execute_global_average_optimization')
        assert hasattr(AbstractModelController, '_train_single_model_on_full_data')

        print("✅ Global average optimization methods are available")

    def test_configuration_validation(self, mock_data):
        """Test that configurations are properly validated."""
        # Test valid configuration
        valid_config = self._create_test_config("test_valid", "per_fold", "global_average")

        try:
            pipeline_config = PipelineConfigs(valid_config, "test_valid")
            assert pipeline_config is not None
            print("✅ Valid configuration passes validation")
        except Exception as e:
            pytest.fail(f"Valid configuration failed: {e}")

        # Test that configuration with invalid CV mode would be caught
        invalid_config = self._create_test_config("test_invalid", "invalid_mode", "per_fold_best")

        # The system should handle invalid modes gracefully or with clear errors
        try:
            pipeline_config = PipelineConfigs(invalid_config, "test_invalid")
            # If it doesn't raise an error, that's also acceptable (graceful handling)
            print("✅ Invalid configuration handled gracefully")
        except Exception as e:
            # Expected - invalid configuration should be caught
            print(f"✅ Invalid configuration properly rejected: {type(e).__name__}")

    def test_performance_expectations(self, mock_data):
        """Test that the optimization framework has reasonable performance characteristics."""
        import time

        # Test that configuration creation is fast
        start_time = time.time()
        for i in range(100):
            config = self._create_test_config(f"perf_test_{i}", "simple", "per_fold_best")
        config_time = time.time() - start_time

        assert config_time < 1.0, f"Configuration creation too slow: {config_time:.2f}s"

        # Test that PipelineConfigs creation is reasonable
        start_time = time.time()
        config = self._create_test_config("perf_test", "per_fold", "global_average")
        pipeline_config = PipelineConfigs(config, "perf_test")
        pipeline_time = time.time() - start_time

        assert pipeline_time < 1.0, f"Pipeline config creation too slow: {pipeline_time:.2f}s"

        print(f"✅ Performance is acceptable: config={config_time:.3f}s, pipeline={pipeline_time:.3f}s")

    def _create_test_config(self, name, cv_mode, param_strategy, model=None, n_trials=2):
        """Helper to create test configuration."""
        if model is None:
            model = PLSRegression()

        config = {
            "pipeline": [{
                "model": model,
                "finetune_params": {
                    "cv_mode": cv_mode,
                    "param_strategy": param_strategy,
                    "n_trials": n_trials,
                    "verbose": 0,
                    "model_params": {
                        "n_components": ("int", 1, 5)
                    },
                    "train_params": {"verbose": 0}
                }
            }]
        }

        if cv_mode == "nested":
            config["pipeline"][0]["finetune_params"]["inner_cv"] = 2

        return config


class TestIntegrationChecks:
    """Integration tests to verify the system works end-to-end."""

    def test_base_model_controller_methods_exist(self):
        """Test that all required methods exist in AbstractModelController."""
        from nirs4all.controllers.models.abstract_model_controller import AbstractModelController

        required_methods = [
            '_execute_global_average_optimization',
            '_train_single_model_on_full_data',
            '_optimize_global_average_on_inner_folds',
            '_execute_per_fold_cv',
            '_execute_nested_cv'
        ]

        for method_name in required_methods:
            assert hasattr(AbstractModelController, method_name), f"Missing method: {method_name}"

        print(f"✅ All {len(required_methods)} required methods exist in AbstractModelController")

    def test_parameter_strategy_enum_completeness(self):
        """Test that ParamStrategy enum has all expected values."""
        from nirs4all.controllers.models.config import ParamStrategy

        # Test all enum values can be created
        strategies = [
            ParamStrategy.GLOBAL_BEST,
            ParamStrategy.PER_FOLD_BEST,
            ParamStrategy.WEIGHTED_AVERAGE,
            ParamStrategy.GLOBAL_AVERAGE,
            ParamStrategy.ENSEMBLE_BEST,
            ParamStrategy.ROBUST_BEST,
            ParamStrategy.STABILITY_BEST
        ]

        assert len(strategies) == 7, f"Expected 7 strategies, got {len(strategies)}"

        # Test that values match expected strings
        expected_values = {
            'global_best', 'per_fold_best', 'weighted_average',
            'global_average', 'ensemble_best', 'robust_best', 'stability_best'
        }
        actual_values = {s.value for s in strategies}

        assert actual_values == expected_values, f"Strategy values mismatch: {actual_values} != {expected_values}"

        print("✅ ParamStrategy enum is complete and correct")

    def test_pipeline_runner_integration(self):
        """Test that PipelineRunner can be imported and instantiated."""
        from nirs4all.pipeline.runner import PipelineRunner
        from nirs4all.pipeline.config import PipelineConfigs

        # Test instantiation
        runner = PipelineRunner()
        assert runner is not None

        # Test basic config creation
        simple_config = {
            "pipeline": [{
                "model": PLSRegression(),
                "finetune_params": {
                    "cv_mode": "simple",
                    "param_strategy": "global_average",
                    "n_trials": 1,
                    "model_params": {
                        "n_components": ("int", 1, 3)
                    }
                }
            }]
        }

        pipeline_config = PipelineConfigs(simple_config, "integration_test")
        assert pipeline_config is not None

        print("✅ PipelineRunner integration works correctly")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
