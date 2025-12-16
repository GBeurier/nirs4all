"""Tests for predictions storage and management."""

import numpy as np
import pytest
from nirs4all.data.predictions import Predictions


@pytest.fixture
def base_prediction_params():
    """Base parameters for creating predictions."""
    return {
        "dataset_name": "test_dataset",
        "dataset_path": "/path/to/dataset",
        "config_name": "test_config",
        "config_path": "/path/to/config",
        "pipeline_uid": "pipe_123",
        "step_idx": 0,
        "op_counter": 1,
        "model_name": "PLS_5",
        "model_classname": "PLSRegression",
        "fold_id": 0,
        "n_samples": 3,
        "n_features": 10,
    }


class TestPredictions:
    """Test suite for Predictions class."""

    def test_initialization(self):
        """Test Predictions initialization."""
        predictions = Predictions()
        assert predictions is not None
        # Test that storage is initialized
        assert predictions._storage is not None

    def test_add_single_prediction(self, base_prediction_params):
        """Test adding a single prediction."""
        predictions = Predictions()
        pred_id = predictions.add_prediction(
            partition="test",
            y_true=np.array([1.0, 2.0, 3.0]),
            y_pred=np.array([1.1, 2.1, 2.9]),
            test_score=0.01,
            val_score=0.015,
            **base_prediction_params
        )

        assert pred_id is not None
        # Verify through top() which is the public API
        # Note: Must specify rank_partition since we only added a "test" partition record
        # Use rank_metric="" to use the stored metric (mse) and precomputed test_score
        all_preds = predictions.top(100, rank_partition="test", rank_metric="")
        assert len(all_preds) >= 1

    def test_add_prediction_with_numpy_arrays(self, base_prediction_params):
        """Test adding prediction with numpy arrays."""
        predictions = Predictions()
        pred_id = predictions.add_prediction(
            partition="test",
            y_true=np.array([1.0, 2.0, 3.0]),
            y_pred=np.array([1.1, 2.1, 2.9]),
            test_score=0.01,
            **base_prediction_params
        )

        assert pred_id is not None
        # Check serialization through top()
        preds = predictions.top(1, rank_partition="test", rank_metric="")
        assert len(preds) == 1
        # Arrays should be available in results
        assert "y_true" in preds[0]
        assert "y_pred" in preds[0]

    def test_filter_by_partition(self, base_prediction_params):
        """Test filtering predictions by partition."""
        predictions = Predictions()

        # Add predictions with different partitions
        for i, partition in enumerate(["test", "val", "test"]):
            predictions.add_prediction(
                partition=partition,
                y_true=np.array([1.0, 2.0]),
                y_pred=np.array([1.1, 2.1]),
                test_score=0.01 * (i + 1),
                model_name=f"Model_{i}",
                **{k: v for k, v in base_prediction_params.items() if k != "model_name"}
            )

        test_preds = predictions.filter_predictions(partition="test")
        assert len(test_preds) == 2
        assert all(p["partition"] == "test" for p in test_preds)

    def test_filter_by_model_name(self, base_prediction_params):
        """Test filtering predictions by model name."""
        predictions = Predictions()

        # Add predictions with different model names
        for i in range(3):
            predictions.add_prediction(
                partition="test",
                y_true=np.array([1.0, 2.0]),
                y_pred=np.array([1.1, 2.1]),
                test_score=0.01 * (i + 1),
                model_name=f"Model_{i}",
                **{k: v for k, v in base_prediction_params.items() if k != "model_name"}
            )

        model_preds = predictions.filter_predictions(model_name="Model_1")
        assert len(model_preds) == 1
        assert model_preds[0]["model_name"] == "Model_1"

    def test_top_single_best(self, base_prediction_params):
        """Test getting single best prediction."""
        predictions = Predictions()

        # Add predictions with different scores
        for i in range(5):
            predictions.add_prediction(
                partition="test",
                y_true=np.array([1.0, 2.0]),
                y_pred=np.array([1.1, 2.1]),
                test_score=0.01 * (i + 1),  # 0.01, 0.02, 0.03, 0.04, 0.05
                model_name=f"Model_{i}",
                **{k: v for k, v in base_prediction_params.items() if k != "model_name"}
            )

        # Best by lowest test_score (using stored metric)
        best = predictions.top(1, rank_partition="test", rank_metric="", ascending=True)
        assert len(best) == 1
        assert best[0]["test_score"] == 0.01

    def test_top_k_multiple_best(self, base_prediction_params):
        """Test getting top K predictions."""
        predictions = Predictions()

        # Add predictions with different scores
        for i in range(5):
            predictions.add_prediction(
                partition="test",
                y_true=np.array([1.0, 2.0]),
                y_pred=np.array([1.1, 2.1]),
                test_score=0.01 * (i + 1),
                model_name=f"Model_{i}",
                **{k: v for k, v in base_prediction_params.items() if k != "model_name"}
            )

        # Top 3 by lowest test_score (using stored metric)
        top_3 = predictions.top(n=3, rank_metric="", ascending=True, rank_partition="test")
        assert len(top_3) == 3
        # Should be sorted by test_score ascending
        assert top_3[0]["test_score"] <= top_3[1]["test_score"]
        assert top_3[1]["test_score"] <= top_3[2]["test_score"]

    def test_top_with_partition_filter(self, base_prediction_params):
        """Test getting best prediction from specific partition."""
        predictions = Predictions()

        # Add predictions to different partitions
        predictions.add_prediction(
            partition="test",
            y_true=np.array([1.0, 2.0]),
            y_pred=np.array([1.1, 2.1]),
            test_score=0.02,
            **base_prediction_params
        )
        predictions.add_prediction(
            partition="val",
            y_true=np.array([1.0, 2.0]),
            y_pred=np.array([1.05, 2.05]),
            test_score=0.01,
            **base_prediction_params
        )

        best_test = predictions.top(1, rank_partition="test", rank_metric="", ascending=True)
        assert len(best_test) == 1
        assert best_test[0]["partition"] == "test"
        assert best_test[0]["test_score"] == 0.02

    def test_top_preserves_pipeline_uid(self, base_prediction_params):
        """Test that top() preserves pipeline_uid metadata."""
        predictions = Predictions()
        predictions.add_prediction(
            partition="test",
            y_true=np.array([1.0, 2.0]),
            y_pred=np.array([1.1, 2.1]),
            test_score=0.01,
            **base_prediction_params
        )

        best = predictions.top(1, rank_partition="test", rank_metric="", ascending=True)
        assert "pipeline_uid" in best[0]
        assert best[0]["pipeline_uid"] == "pipe_123"

    def test_top_k_preserves_all_metadata(self, base_prediction_params):
        """Test that top() preserves all metadata including pipeline_uid."""
        predictions = Predictions()

        for i in range(3):
            predictions.add_prediction(
                partition="test",
                y_true=np.array([1.0, 2.0]),
                y_pred=np.array([1.1, 2.1]),
                test_score=0.01 * (i + 1),
                model_name=f"Model_{i}",
                **{k: v for k, v in base_prediction_params.items() if k != "model_name"}
            )

        top_3 = predictions.top(n=3, rank_metric="test_score", ascending=True)
        for pred in top_3:
            assert "pipeline_uid" in pred
            assert "model_name" in pred
            assert "dataset_name" in pred
            assert pred["pipeline_uid"] == "pipe_123"

    def test_empty_predictions(self):
        """Test operations on empty predictions."""
        predictions = Predictions()

        assert len(predictions.top(1, metric="test_score")) == 0
        assert len(predictions.filter_predictions(partition="test")) == 0

    def test_catalog_unique_models(self, base_prediction_params):
        """Test catalog query for unique model names."""
        predictions = Predictions()

        # Add predictions with different model names
        for i in range(5):
            predictions.add_prediction(
                partition="test",
                y_true=np.array([1.0, 2.0]),
                y_pred=np.array([1.1, 2.1]),
                test_score=0.01,
                model_name=f"Model_{i}",
                **{k: v for k, v in base_prediction_params.items() if k != "model_name"}
            )

        unique_models = predictions.get_unique_values("model_name")
        assert len(unique_models) == 5
        assert "Model_0" in unique_models
        assert "Model_4" in unique_models

    def test_catalog_unique_partitions(self, base_prediction_params):
        """Test catalog query for unique partitions."""
        predictions = Predictions()

        for i, partition in enumerate(["test", "val", "test"]):
            predictions.add_prediction(
                partition=partition,
                y_true=np.array([1.0, 2.0]),
                y_pred=np.array([1.1, 2.1]),
                test_score=0.01,
                **base_prediction_params
            )

        unique_partitions = predictions.get_unique_values("partition")
        assert len(unique_partitions) == 2
        assert "test" in unique_partitions
        assert "val" in unique_partitions

    def test_weights_parameter(self, base_prediction_params):
        """Test handling of weights parameter."""
        predictions = Predictions()
        predictions.add_prediction(
            partition="test",
            y_true=np.array([1.0, 2.0, 3.0]),
            y_pred=np.array([1.1, 2.1, 2.9]),
            test_score=0.01,
            weights=np.array([0.5, 1.0, 0.5]),
            **base_prediction_params
        )

        preds = predictions.top(1, rank_partition="test", rank_metric="")
        assert len(preds) == 1
        # Weights should be available
        assert "weights" in preds[0]

    def test_none_weights_handling(self, base_prediction_params):
        """Test that None weights are handled correctly."""
        predictions = Predictions()
        predictions.add_prediction(
            partition="test",
            y_true=np.array([1.0, 2.0]),
            y_pred=np.array([1.1, 2.1]),
            test_score=0.01,
            weights=None,
            **base_prediction_params
        )

        preds = predictions.top(1, rank_partition="test", rank_metric="")
        assert len(preds) == 1
        # Should handle None gracefully
        assert "weights" in preds[0]


# ========================================================================
# Tests merged from test_predictions_header_units.py
# ========================================================================

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for testing
import matplotlib.pyplot as plt
from nirs4all.controllers.charts.spectra import SpectraChartController


class TestVisualizationHeaderUnit:
    """Test visualization axis labels with different header units"""

    def test_plot_2d_with_cm1_unit(self):
        """Test that 2D plot uses 'Wavenumber (cm⁻¹)' for cm-1 data"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)
        headers = [str(1000 + i*100) for i in range(10)]

        fig, ax = plt.subplots()
        controller._plot_2d_spectra(ax, x_data, y_data, "raw", headers, header_unit="cm-1")

        # Check that x-axis label is correct
        assert ax.get_xlabel() == 'Wavenumber (cm⁻¹)'
        plt.close(fig)

    def test_plot_2d_with_nm_unit(self):
        """Test that 2D plot uses 'Wavelength (nm)' for nm data"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)
        headers = [str(700 + i*50) for i in range(10)]

        fig, ax = plt.subplots()
        controller._plot_2d_spectra(ax, x_data, y_data, "raw", headers, header_unit="nm")

        # Check that x-axis label is correct
        assert ax.get_xlabel() == 'Wavelength (nm)'
        plt.close(fig)

    def test_plot_2d_with_none_unit(self):
        """Test that 2D plot uses 'Features' for none unit"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)
        headers = [f'f{i}' for i in range(10)]

        fig, ax = plt.subplots()
        controller._plot_2d_spectra(ax, x_data, y_data, "raw", headers, header_unit="none")

        # Check that x-axis label is correct
        assert ax.get_xlabel() == 'Features'
        plt.close(fig)

    def test_plot_2d_with_text_headers(self):
        """Test that 2D plot uses 'Features' for text headers"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)
        headers = ['feature_a', 'feature_b', 'feature_c', 'feature_d', 'feature_e',
                   'feature_f', 'feature_g', 'feature_h', 'feature_i', 'feature_j']

        fig, ax = plt.subplots()
        controller._plot_2d_spectra(ax, x_data, y_data, "raw", headers, header_unit="text")

        # Check that x-axis label is correct
        assert ax.get_xlabel() == 'Features'
        plt.close(fig)

    def test_plot_2d_without_headers(self):
        """Test that 2D plot uses 'Features' when no headers provided"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)

        fig, ax = plt.subplots()
        controller._plot_2d_spectra(ax, x_data, y_data, "raw", headers=None, header_unit="cm-1")

        # Check that x-axis label is correct
        assert ax.get_xlabel() == 'Features'
        plt.close(fig)

    def test_plot_3d_with_cm1_unit(self):
        """Test that 3D plot uses 'Wavenumber (cm⁻¹)' for cm-1 data"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)
        headers = [str(1000 + i*100) for i in range(10)]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        controller._plot_3d_spectra(ax, x_data, y_data, "raw", headers, header_unit="cm-1")

        # Check that x-axis label is correct
        assert ax.get_xlabel() == 'Wavenumber (cm⁻¹)'
        plt.close(fig)

    def test_plot_3d_with_nm_unit(self):
        """Test that 3D plot uses 'Wavelength (nm)' for nm data"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)
        headers = [str(700 + i*50) for i in range(10)]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        controller._plot_3d_spectra(ax, x_data, y_data, "raw", headers, header_unit="nm")

        # Check that x-axis label is correct
        assert ax.get_xlabel() == 'Wavelength (nm)'
        plt.close(fig)

    def test_plot_3d_with_none_unit(self):
        """Test that 3D plot uses 'Features' for none unit"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)
        headers = [f'f{i}' for i in range(10)]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        controller._plot_3d_spectra(ax, x_data, y_data, "raw", headers, header_unit="none")

        # Check that x-axis label is correct
        assert ax.get_xlabel() == 'Features'
        plt.close(fig)

    def test_plot_2d_axis_labels_consistency(self):
        """Test that 2D plots have consistent y-axis labels"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)
        headers = [str(1000 + i*100) for i in range(10)]

        fig, ax = plt.subplots()
        controller._plot_2d_spectra(ax, x_data, y_data, "raw", headers, header_unit="cm-1")

        # Check y-axis label
        assert ax.get_ylabel() == 'Intensity'
        plt.close(fig)

    def test_plot_3d_axis_labels_consistency(self):
        """Test that 3D plots have consistent y and z-axis labels"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)
        headers = [str(1000 + i*100) for i in range(10)]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        controller._plot_3d_spectra(ax, x_data, y_data, "raw", headers, header_unit="cm-1")

        # Check axis labels
        assert ax.get_ylabel() == 'y (sorted)'
        assert ax.get_zlabel() == 'Intensity'
        plt.close(fig)

    def test_plot_with_index_unit(self):
        """Test that plot uses 'Features' for index unit"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)
        headers = [str(i) for i in range(10)]

        fig, ax = plt.subplots()
        controller._plot_2d_spectra(ax, x_data, y_data, "raw", headers, header_unit="index")

        # Check that x-axis label is correct for index unit
        assert ax.get_xlabel() == 'Feature Index'
        plt.close(fig)

    def test_plot_title_includes_processing_name(self):
        """Test that plot title includes processing name"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)
        headers = [str(1000 + i*100) for i in range(10)]
        processing_name = "SavitzkyGolay"

        fig, ax = plt.subplots()
        controller._plot_2d_spectra(ax, x_data, y_data, processing_name, headers, header_unit="cm-1")

        # Check that title contains processing name and dimensions
        title = ax.get_title()
        assert processing_name in title
        assert "20 samples" in title
        assert "10 features" in title
        plt.close(fig)

    def test_plot_with_mismatched_headers(self):
        """Test that plot falls back to indices when headers don't match data"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)
        headers = [str(1000 + i*100) for i in range(5)]  # Only 5 headers for 10 features

        fig, ax = plt.subplots()
        controller._plot_2d_spectra(ax, x_data, y_data, "raw", headers, header_unit="cm-1")

        # When headers don't match, should use 'Features'
        assert ax.get_xlabel() == 'Features'
        plt.close(fig)
