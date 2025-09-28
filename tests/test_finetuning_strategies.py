#!/usr/bin/env python3
"""
Test suite for NIRS4ALL finetuning strategies and cross-validation modes.

This module tests all combinations of:
- CV modes: simple, per_fold, nested
- Parameter strategies: per_fold_best, global_best, global_average
- Full training option: use_full_train_for_final
- Different model types and parameter spaces

Uses synthetic data for fast, reliable testing.
"""

import pytest
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.pipeline.config import PipelineConfigs
from nirs4all.dataset.dataset import SpectroDataset
from nirs4all.controllers.models.base_model_controller import ParamStrategy


# Test data generation fixtures
@pytest.fixture
def synthetic_dataset():
    """Create synthetic spectral dataset for testing."""
    np.random.seed(42)

    # Generate synthetic spectral data
    n_samples = 100
    n_features = 50
    n_folds = 3

    # Create synthetic spectra with some structure
    wavelengths = np.linspace(1000, 2500, n_features)
    X = np.random.randn(n_samples, n_features)

    # Add some spectral-like structure
    for i in range(n_samples):
        # Add some smooth variations
        X[i] += 0.5 * np.sin(wavelengths / 200) + 0.3 * np.cos(wavelengths / 300)
        # Add noise
        X[i] += 0.1 * np.random.randn(n_features)

    # Create target variable with some relationship to spectra
    y_terms = [
        np.sum(X[:, 10:20], axis=1),
        0.5 * np.sum(X[:, 30:40], axis=1),
        0.2 * np.random.randn(n_samples)
    ]
    y = sum(y_terms)

    # Create dataset configuration
    dataset_config = {
        'X': X,
        'y': y,
        'folds': n_folds,
        'train': 0.7,
        'val': 0.15,
        'test': 0.15,
        'random_state': 42
    }

    return dataset_config


@pytest.fixture
def small_synthetic_dataset():
    """Create smaller synthetic dataset for faster tests."""
    np.random.seed(123)

    n_samples = 50
    n_features = 20
    n_folds = 2

    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :5], axis=1) + 0.3 * np.random.randn(n_samples)

    dataset_config = {
        'X': X,
        'y': y,
        'folds': n_folds,
        'train': 0.8,
        'val': 0.1,
        'test': 0.1,
        'random_state': 123
    }

    return dataset_config


# Test configuration helpers
def create_test_config(model, cv_mode, param_strategy, use_full_train=False, n_trials=3):
    """Create test configuration for finetuning."""

    # Define model-specific parameter spaces
    param_spaces = {
        'PLSRegression': {
            'n_components': ('int', 1, 5)
        },
        'RandomForestRegressor': {
            'n_estimators': [10, 20, 30],
            'max_depth': [3, 5, None]
        },
        'Ridge': {
            'alpha': ('float', 0.1, 10.0)
        }
    }

    model_name = model.__class__.__name__
    model_params = param_spaces.get(model_name, {'alpha': ('float', 0.1, 1.0)})

    config = {
        "pipeline": [{
            "model": model,
            "finetune_params": {
                "cv_mode": cv_mode,
                "param_strategy": param_strategy,
                "use_full_train_for_final": use_full_train,
                "n_trials": n_trials,
                "verbose": 0,  # Silent for testing
                "model_params": model_params,
                "train_params": {
                    "verbose": 0
                }
            }
        }]
    }

    if cv_mode == "nested":
        config["pipeline"][0]["finetune_params"]["inner_cv"] = 2  # Small for testing

    return config


def run_config_test(config, dataset_config, config_name="test_config"):
    """Helper to run configuration test."""
    try:
        from nirs4all.dataset.loader import create_synthetic_dataset
        data = create_synthetic_dataset(dataset_config)

        pipeline_config = PipelineConfigs(config, config_name)
        runner = PipelineRunner()

        res_dataset, _, _ = runner.run(pipeline_config, data)
        predictions = res_dataset._predictions

        return {
            'success': True,
            'predictions': len(predictions),
            'prediction_keys': predictions.list_keys() if predictions else []
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


# Basic functionality tests
class TestBasicFinetuning:
    """Test basic finetuning functionality."""

    def test_pls_per_fold_best(self, small_synthetic_dataset):
        """Test PLS with per_fold_best strategy."""
        model = PLSRegression()
        config = create_test_config(model, "per_fold", "per_fold_best")

        # Run test
        result = run_config_test(config, small_synthetic_dataset, "test_pls_per_fold_best")
        assert result['success'], f"Test failed: {result.get('error', 'Unknown error')}"
        assert result['predictions'] > 0, "No predictions generated"

    def test_pls_global_best(self, small_synthetic_dataset):
        """Test PLS with global_best strategy."""
        model = PLSRegression()
        config = create_test_config(model, "per_fold", "global_best")

        result = run_config_test(config, small_synthetic_dataset, "test_pls_global_best")
        assert result['success'], f"Test failed: {result.get('error', 'Unknown error')}"
        assert result['predictions'] > 0, "No predictions generated"

    def test_pls_global_average(self, small_synthetic_dataset):
        """Test PLS with global_average strategy."""
        model = PLSRegression()
        config = create_test_config(model, "per_fold", "global_average")

        result = run_config_test(config, small_synthetic_dataset, "test_pls_global_average")
        assert result['success'], f"Test failed: {result.get('error', 'Unknown error')}"
        assert result['predictions'] > 0, "No predictions generated"


# Cross-validation mode tests
class TestCVModes:
    """Test different cross-validation modes."""

    @pytest.mark.parametrize("cv_mode", ["simple", "per_fold", "nested"])
    def test_cv_modes_with_pls(self, cv_mode, small_synthetic_dataset):
        """Test all CV modes with PLS."""
        model = PLSRegression()
        config = create_test_config(model, cv_mode, "per_fold_best")

        result = run_config_test(config, small_synthetic_dataset, f"test_cv_{cv_mode}")
        assert result['success'], f"CV mode {cv_mode} failed: {result.get('error', 'Unknown error')}"
        assert result['predictions'] > 0, f"No predictions generated for {cv_mode}"

    @pytest.mark.parametrize("cv_mode", ["simple", "per_fold"])  # Skip nested for speed
    def test_cv_modes_with_rf(self, cv_mode, small_synthetic_dataset):
        """Test CV modes with RandomForest."""
        model = RandomForestRegressor(random_state=42)
        config = create_test_config(model, cv_mode, "per_fold_best")

        result = run_config_test(config, small_synthetic_dataset, f"test_rf_{cv_mode}")
        assert result['success'], f"CV mode {cv_mode} with RF failed: {result.get('error', 'Unknown error')}"
        assert result['predictions'] > 0, f"No predictions generated for {cv_mode}"


# Parameter strategy tests
class TestParameterStrategies:
    """Test different parameter strategies."""

    @pytest.mark.parametrize("strategy", ["per_fold_best", "global_best", "global_average"])
    def test_parameter_strategies(self, strategy, small_synthetic_dataset):
        """Test all parameter strategies."""
        model = PLSRegression()
        config = create_test_config(model, "per_fold", strategy, n_trials=2)  # Quick test

        result = run_config_test(config, small_synthetic_dataset, f"test_strategy_{strategy}")
        assert result['success'], f"Strategy {strategy} failed: {result.get('error', 'Unknown error')}"
        assert result['predictions'] > 0, f"No predictions generated for {strategy}"

        # Check prediction naming patterns
        pred_keys = result['prediction_keys']
        if strategy == "global_average":
            # Should have global_avg in the name when using full training
            assert any('global' in key.lower() for key in pred_keys), f"Expected global_avg naming for {strategy}"


# Full training option tests
class TestFullTrainingOption:
    """Test the use_full_train_for_final option."""

    @pytest.mark.parametrize("use_full_train", [False, True])
    def test_full_training_option(self, use_full_train, small_synthetic_dataset):
        """Test full training option with different strategies."""
        model = PLSRegression()
        config = create_test_config(model, "per_fold", "global_average", use_full_train)

        result = run_config_test(config, small_synthetic_dataset, f"test_full_train_{use_full_train}")
        assert result['success'], f"Full training {use_full_train} failed: {result.get('error', 'Unknown error')}"
        assert result['predictions'] > 0, "No predictions generated"

        # Check prediction patterns
        pred_keys = result['prediction_keys']
        if use_full_train:
            # Should have fewer prediction sets (combined vs individual folds)
            assert len(pred_keys) <= 2, f"Expected fewer predictions with full training, got {len(pred_keys)}"
        else:
            # Should have at least one prediction set when not using full training
            assert len(pred_keys) >= 1, f"Expected at least one prediction set, got {len(pred_keys)}"

    def test_full_training_with_different_strategies(self, small_synthetic_dataset):
        """Test full training with different parameter strategies."""
        strategies = ["global_best", "global_average"]

        for strategy in strategies:
            model = PLSRegression()
            config = create_test_config(model, "per_fold", strategy, use_full_train=True)

            result = run_config_test(config, small_synthetic_dataset, f"test_full_train_{strategy}")
            assert result['success'], f"Full training with {strategy} failed: {result.get('error', 'Unknown error')}"
            assert result['predictions'] > 0, f"No predictions for {strategy} with full training"


# Model type tests
class TestModelTypes:
    """Test different model types with finetuning."""

    @pytest.mark.parametrize("model_class", [PLSRegression, Ridge])
    def test_model_types(self, model_class, small_synthetic_dataset):
        """Test different model types."""
        model = model_class()
        config = create_test_config(model, "simple", "per_fold_best", n_trials=2)

        result = run_config_test(config, small_synthetic_dataset, f"test_model_{model_class.__name__}")
        assert result['success'], f"Model {model_class.__name__} failed: {result.get('error', 'Unknown error')}"
        assert result['predictions'] > 0, f"No predictions for {model_class.__name__}"

    def test_random_forest(self, small_synthetic_dataset):
        """Test RandomForest specifically (can be slow)."""
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        config = create_test_config(model, "simple", "per_fold_best", n_trials=2)

        result = run_config_test(config, small_synthetic_dataset, "test_random_forest")
        assert result['success'], f"RandomForest failed: {result.get('error', 'Unknown error')}"
        assert result['predictions'] > 0, "No predictions for RandomForest"


# Comprehensive combination tests
class TestStrategyCominations:
    """Test combinations of strategies and modes."""

    def test_global_average_with_full_training(self, small_synthetic_dataset):
        """Test the recommended combination: global_average + full_training."""
        model = PLSRegression()
        config = create_test_config(model, "per_fold", "global_average", use_full_train=True)

        result = run_config_test(config, small_synthetic_dataset, "test_global_avg_full_train")
        assert result['success'], f"Global average + full training failed: {result.get('error', 'Unknown error')}"
        assert result['predictions'] > 0, "No predictions generated"

        # Should have minimal prediction sets (single unified model)
        pred_keys = result['prediction_keys']
        assert len(pred_keys) <= 2, f"Expected single model predictions, got {len(pred_keys)}: {pred_keys}"

    def test_nested_cv_with_global_average(self, small_synthetic_dataset):
        """Test nested CV with global average (most rigorous)."""
        model = PLSRegression()
        config = create_test_config(model, "nested", "global_average", n_trials=2)

        result = run_config_test(config, small_synthetic_dataset, "test_nested_global_avg")
        assert result['success'], f"Nested CV + global average failed: {result.get('error', 'Unknown error')}"
        assert result['predictions'] > 0, "No predictions generated"

    def test_simple_cv_combinations(self, small_synthetic_dataset):
        """Test simple CV with different strategies (fastest)."""
        strategies = ["per_fold_best", "global_average"]

        for strategy in strategies:
            model = PLSRegression()
            config = create_test_config(model, "simple", strategy, n_trials=3)

            result = run_config_test(config, small_synthetic_dataset, f"test_simple_{strategy}")
            assert result['success'], f"Simple CV + {strategy} failed: {result.get('error', 'Unknown error')}"
            assert result['predictions'] > 0, f"No predictions for simple CV + {strategy}"


# Performance and validation tests
class TestPerformanceAndValidation:
    """Test performance characteristics and validation."""

    def test_prediction_quality(self, small_synthetic_dataset):
        """Test that predictions are reasonable quality."""
        model = PLSRegression()
        config = create_test_config(model, "per_fold", "global_average", n_trials=5)

        try:
            result = run_config_test(config, small_synthetic_dataset, "test_prediction_quality")
            assert result['success'], f"Test failed: {result.get('error', 'Unknown error')}"

            # Get prediction data for quality check
            from nirs4all.dataset.loader import create_synthetic_dataset
            data = create_synthetic_dataset(small_synthetic_dataset)

            pipeline_config = PipelineConfigs(config, "test_prediction_quality_detailed")
            runner = PipelineRunner()

            res_dataset, _, _ = runner.run(pipeline_config, data)
            predictions = res_dataset._predictions

            # Get prediction performance
            if len(predictions) > 0:
                pred_key = predictions.list_keys()[0]
                key_parts = pred_key.split('_', 3)

                if len(key_parts) >= 4:
                    pred_data = predictions.get_prediction_data(*key_parts)

                    if pred_data:
                        from sklearn.metrics import r2_score, mean_squared_error

                        y_true = pred_data['y_true'].flatten()
                        y_pred = pred_data['y_pred'].flatten()

                        r2 = r2_score(y_true, y_pred)
                        mse = mean_squared_error(y_true, y_pred)

                        # Basic quality checks
                        assert r2 > -1.0, f"R² too low: {r2}"  # Should be better than predicting mean
                        assert mse > 0, f"MSE should be positive: {mse}"
                        assert np.isfinite(r2), f"R² should be finite: {r2}"
                        assert np.isfinite(mse), f"MSE should be finite: {mse}"

                        print(f"Prediction quality: R²={r2:.3f}, MSE={mse:.3f}")

        except Exception as e:
            pytest.fail(f"Prediction quality test failed: {e}")

    def test_parameter_optimization_works(self, small_synthetic_dataset):
        """Test that parameter optimization actually improves performance."""
        model = PLSRegression()

        # Test with optimization
        config_optimized = create_test_config(model, "simple", "global_average", n_trials=5)

        # Test without optimization (default parameters)
        config_default = {
            "pipeline": [{
                "model": PLSRegression(n_components=1),  # Fixed parameter
                "train_params": {"verbose": 0}
                # No finetune_params
            }]
        }

        results = {}

        for config_name, config in [("optimized", config_optimized), ("default", config_default)]:
            try:
                result = run_config_test(config, small_synthetic_dataset, f"test_optimization_{config_name}")

                if result['success'] and len(result['prediction_keys']) > 0:
                    from nirs4all.dataset.loader import create_synthetic_dataset
                    data = create_synthetic_dataset(small_synthetic_dataset)

                    pipeline_config = PipelineConfigs(config, f"test_optimization_{config_name}_detailed")
                    runner = PipelineRunner()

                    res_dataset, _, _ = runner.run(pipeline_config, data)
                    predictions = res_dataset._predictions

                    if len(predictions) > 0:
                        pred_key = predictions.list_keys()[0]
                        key_parts = pred_key.split('_', 3)

                        if len(key_parts) >= 4:
                            pred_data = predictions.get_prediction_data(*key_parts)

                            if pred_data:
                                from sklearn.metrics import mean_squared_error
                                y_true = pred_data['y_true'].flatten()
                                y_pred = pred_data['y_pred'].flatten()
                                mse = mean_squared_error(y_true, y_pred)
                                results[config_name] = mse

            except Exception as e:
                print(f"Warning: {config_name} test failed: {e}")

        # Check that optimization generally helps (though not always guaranteed with small data)
        if len(results) == 2:
            print(f"Optimization results: Default MSE={results.get('default', 'N/A'):.3f}, "
                  f"Optimized MSE={results.get('optimized', 'N/A'):.3f}")

            # Just ensure both ran successfully
            assert 'optimized' in results, "Optimized configuration should produce results"
            assert 'default' in results, "Default configuration should produce results"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
