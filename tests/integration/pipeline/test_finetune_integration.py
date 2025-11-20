"""
Integration tests for hyperparameter finetuning (Q3 example).

Tests Optuna-based hyperparameter optimization with different strategies.
Based on Q3_finetune.py example.
"""

import pytest
import numpy as np

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from nirs4all.data import DatasetConfigs
from nirs4all.operators.transforms import (
    Detrend, FirstDerivative, Gaussian, StandardNormalVariate
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

from tests.fixtures.data_generators import TestDataManager


class TestFinetuneIntegration:
    """Integration tests for hyperparameter optimization."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with regression dataset."""
        manager = TestDataManager()
        manager.create_regression_dataset("regression")
        yield manager
        manager.cleanup()

    @pytest.mark.optuna
    def test_basic_hyperparameter_finetuning(self, test_data_manager):
        """Test basic hyperparameter optimization with PLS."""
        pytest.importorskip("optuna")

        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {
                "model": PLSRegression(),
                "name": "PLS-Finetuned",
                "finetune_params": {
                    "n_trials": 5,  # Reduced for testing
                    "verbose": 0,
                    "approach": "single",
                    "eval_mode": "best",
                    "sample": "grid",
                    "model_params": {
                        'n_components': ('int', 1, 10),
                    },
                }
            },
        ]

        pipeline_config = PipelineConfigs(pipeline, "finetune_basic_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should have predictions from finetuned model
        assert predictions.num_predictions > 0

        # Verify finetuning produced results
        best_pred = predictions.get_best(ascending=True)
        assert 'model_name' in best_pred
        assert np.isfinite(best_pred['val_score'])

    @pytest.mark.optuna
    def test_finetune_with_feature_augmentation(self, test_data_manager):
        """Test finetuning combined with feature augmentation."""
        pytest.importorskip("optuna")

        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        preprocessing_options = [Detrend, FirstDerivative, Gaussian, StandardNormalVariate]

        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            {"feature_augmentation": {"_or_": preprocessing_options, "size": 1, "count": 2}},
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {
                "model": PLSRegression(),
                "name": "PLS-Finetuned-Augmented",
                "finetune_params": {
                    "n_trials": 3,
                    "verbose": 0,
                    "approach": "single",
                    "model_params": {
                        'n_components': ('int', 1, 8),
                    },
                }
            },
        ]

        pipeline_config = PipelineConfigs(pipeline, "finetune_augmentation_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should have predictions from multiple augmentation + finetuning combinations
        assert predictions.num_predictions >= 2

    @pytest.mark.optuna
    def test_finetune_different_sampling_strategies(self, test_data_manager):
        """Test different Optuna sampling strategies."""
        pytest.importorskip("optuna")

        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        sampling_strategies = ["grid", "random"]

        for sample_strategy in sampling_strategies:
            pipeline = [
                MinMaxScaler(),
                ShuffleSplit(n_splits=1, test_size=0.25, random_state=42),
                {
                    "model": PLSRegression(),
                    "name": f"PLS-{sample_strategy}",
                    "finetune_params": {
                        "n_trials": 3,
                        "verbose": 0,
                        "sample": sample_strategy,
                        "model_params": {
                            'n_components': ('int', 1, 5),
                        },
                    }
                },
            ]

            pipeline_config = PipelineConfigs(pipeline, f"finetune_{sample_strategy}_test")
            dataset_config = DatasetConfigs(dataset_folder)

            runner = PipelineRunner(save_files=False, verbose=0)
            predictions, _ = runner.run(pipeline_config, dataset_config)

            assert predictions.num_predictions > 0
            print(f"Sampling strategy '{sample_strategy}' completed")

    @pytest.mark.optuna
    def test_finetune_comparison_with_baseline(self, test_data_manager):
        """Test finetuned model vs baseline models."""
        pytest.importorskip("optuna")

        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            # Baseline models
            {"model": PLSRegression(n_components=5), "name": "PLS-5-baseline"},
            {"model": PLSRegression(n_components=10), "name": "PLS-10-baseline"},
            # Finetuned model
            {
                "model": PLSRegression(),
                "name": "PLS-Finetuned",
                "finetune_params": {
                    "n_trials": 5,
                    "verbose": 0,
                    "model_params": {
                        'n_components': ('int', 1, 15),
                    },
                }
            },
        ]

        pipeline_config = PipelineConfigs(pipeline, "finetune_comparison_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should have predictions from all models
        assert predictions.num_predictions >= 3

        # Verify different models
        all_preds = predictions.to_dicts()
        model_names = {pred['model_name'] for pred in all_preds}
        assert len(model_names) >= 3

    @pytest.mark.optuna
    @pytest.mark.tensorflow
    def test_finetune_tensorflow_model(self, test_data_manager):
        """Test finetuning with TensorFlow neural network."""
        pytest.importorskip("optuna")
        pytest.importorskip("tensorflow")
        from nirs4all.operators.models.tensorflow.nicon import customizable_nicon

        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=1, test_size=0.25, random_state=42),
            {
                "model": customizable_nicon,
                "name": "NN-Finetuned",
                "finetune_params": {
                    "n_trials": 3,
                    "verbose": 0,
                    "sample": "random",
                    "approach": "single",
                    "model_params": {
                        "filters_1": [8, 16, 32],
                        "filters_2": [8, 16, 32],
                    },
                    "train_params": {
                        "epochs": 2,  # Minimal for testing
                        "verbose": 0
                    }
                },
                "train_params": {
                    "epochs": 5,
                    "verbose": 0
                }
            }
        ]

        pipeline_config = PipelineConfigs(pipeline, "finetune_tf_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions > 0

    @pytest.mark.optuna
    def test_finetune_eval_modes(self, test_data_manager):
        """Test different evaluation modes for finetuning."""
        pytest.importorskip("optuna")

        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        eval_modes = ["best", "mean"]

        for eval_mode in eval_modes:
            pipeline = [
                MinMaxScaler(),
                ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
                {
                    "model": PLSRegression(),
                    "name": f"PLS-eval-{eval_mode}",
                    "finetune_params": {
                        "n_trials": 3,
                        "verbose": 0,
                        "eval_mode": eval_mode,
                        "model_params": {
                            'n_components': ('int', 1, 8),
                        },
                    }
                },
            ]

            pipeline_config = PipelineConfigs(pipeline, f"finetune_eval_{eval_mode}_test")
            dataset_config = DatasetConfigs(dataset_folder)

            runner = PipelineRunner(save_files=False, verbose=0)
            predictions, _ = runner.run(pipeline_config, dataset_config)

            assert predictions.num_predictions > 0
            print(f"Eval mode '{eval_mode}' completed")
