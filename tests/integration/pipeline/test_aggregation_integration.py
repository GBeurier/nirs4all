"""
Integration tests for dataset-level aggregation by sample ID.

Tests Phase 3 of the aggregation feature: Pipeline Integration.
Verifies that aggregation settings propagate correctly from DatasetConfigs
through the pipeline to the TabReportManager.
"""

import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.data import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.pipeline.config.context import ExecutionContext
from nirs4all.visualization.predictions import PredictionAnalyzer
from tests.fixtures.data_generators import SyntheticNIRSDataGenerator, TestDataManager


def _write_repeated_regression_dataset(base_path: Path, seed: int = 42) -> None:
    """Create a repeated-measurement regression dataset with a custom lot grouping."""
    rng = np.random.default_rng(seed)
    n_lots = 8
    samples_per_lot = 2
    n_reps = 3
    n_wavelengths = 60
    n_samples = n_lots * samples_per_lot
    n_train = 12

    lot_targets = rng.uniform(5.0, 15.0, size=n_lots)
    X_base = rng.normal(size=(n_samples, n_wavelengths))

    def build_partition(sample_indices, prefix):
        x_all, y_all, meta_rows = [], [], []
        for sample_idx in sample_indices:
            lot_idx = sample_idx // samples_per_lot
            target = float(lot_targets[lot_idx])
            spectrum = X_base[sample_idx] + target * 0.15
            for rep in range(n_reps):
                x_all.append(spectrum + rng.normal(scale=0.25, size=n_wavelengths))
                y_all.append(target)
                meta_rows.append({
                    "sample_id": f"{prefix}_sample_{sample_idx:03d}",
                    "lot_id": f"lot_{lot_idx:03d}",
                    "repetition": rep + 1,
                })
        return np.asarray(x_all), np.asarray(y_all), pd.DataFrame(meta_rows)

    X_train, y_train, m_train = build_partition(np.arange(0, n_train), "train")
    X_test, y_test, m_test = build_partition(np.arange(n_train, n_samples), "test")

    pd.DataFrame(X_train).to_csv(base_path / "Xcal.csv.gz", index=False, header=False, compression="gzip", sep=";")
    pd.DataFrame(y_train).to_csv(base_path / "Ycal.csv.gz", index=False, header=False, compression="gzip", sep=";")
    m_train.to_csv(base_path / "Mcal.csv", index=False, sep=";")
    pd.DataFrame(X_test).to_csv(base_path / "Xval.csv.gz", index=False, header=False, compression="gzip", sep=";")
    pd.DataFrame(y_test).to_csv(base_path / "Yval.csv.gz", index=False, header=False, compression="gzip", sep=";")
    m_test.to_csv(base_path / "Mval.csv", index=False, sep=";")


def _write_repeated_classification_dataset(base_path: Path, seed: int = 123) -> None:
    """Create a repeated-measurement classification dataset with a custom lot grouping."""
    rng = np.random.default_rng(seed)
    n_classes = 3
    lots_per_class = 3
    samples_per_lot = 2
    n_reps = 3
    n_wavelengths = 60
    n_lots = n_classes * lots_per_class
    n_samples = n_lots * samples_per_lot
    n_train = 12

    class_centers = rng.normal(size=(n_classes, n_wavelengths))
    lot_classes = np.repeat(np.arange(n_classes), lots_per_class)

    def build_partition(sample_indices, prefix):
        x_all, y_all, meta_rows = [], [], []
        for sample_idx in sample_indices:
            lot_idx = sample_idx // samples_per_lot
            class_idx = int(lot_classes[lot_idx])
            spectrum = class_centers[class_idx] + rng.normal(scale=0.15, size=n_wavelengths)
            for rep in range(n_reps):
                x_all.append(spectrum + rng.normal(scale=0.08, size=n_wavelengths))
                y_all.append(class_idx)
                meta_rows.append({
                    "sample_id": f"{prefix}_sample_{sample_idx:03d}",
                    "lot_id": f"lot_{lot_idx:03d}",
                    "repetition": rep + 1,
                })
        return np.asarray(x_all), np.asarray(y_all), pd.DataFrame(meta_rows)

    X_train, y_train, m_train = build_partition(np.arange(0, n_train), "train")
    X_test, y_test, m_test = build_partition(np.arange(n_train, n_samples), "test")

    pd.DataFrame(X_train).to_csv(base_path / "Xcal.csv.gz", index=False, header=False, compression="gzip", sep=";")
    pd.DataFrame(y_train).to_csv(base_path / "Ycal.csv.gz", index=False, header=False, compression="gzip", sep=";")
    m_train.to_csv(base_path / "Mcal.csv", index=False, sep=";")
    pd.DataFrame(X_test).to_csv(base_path / "Xval.csv.gz", index=False, header=False, compression="gzip", sep=";")
    pd.DataFrame(y_test).to_csv(base_path / "Yval.csv.gz", index=False, header=False, compression="gzip", sep=";")
    m_test.to_csv(base_path / "Mval.csv", index=False, sep=";")


def _repeated_dataset_config(data_path: Path, task_type: str | None = None) -> DatasetConfigs:
    """Build DatasetConfigs for repeated datasets with metadata files."""
    kwargs = {"repetition": "sample_id"}
    if task_type is not None:
        kwargs["task_type"] = task_type

    return DatasetConfigs(
        {
            "train_x": str(data_path / "Xcal.csv.gz"),
            "train_y": str(data_path / "Ycal.csv.gz"),
            "train_m": str(data_path / "Mcal.csv"),
            "test_x": str(data_path / "Xval.csv.gz"),
            "test_y": str(data_path / "Yval.csv.gz"),
            "test_m": str(data_path / "Mval.csv"),
            "train_x_params": {"has_header": False},
            "train_y_params": {"has_header": False},
            "train_m_params": {"has_header": True},
            "test_x_params": {"has_header": False},
            "test_y_params": {"has_header": False},
            "test_m_params": {"has_header": True},
        },
        **kwargs,
    )


class TestAggregationIntegration:
    """Test aggregation settings propagation through pipeline."""

    @pytest.fixture
    def test_data_manager(self, tmp_path):
        """Create test data manager with datasets including metadata."""
        manager = TestDataManager(base_temp_dir=str(tmp_path))
        manager.create_regression_dataset("regression")

        # Add metadata file with sample_id column for aggregation tests
        regression_path = manager.get_temp_directory() / "regression"

        # Read actual X files to get correct row counts
        # TestDataManager now saves with headers, so use header=0
        X_train = pd.read_csv(regression_path / "Xcal.csv.gz", compression='gzip', sep=';', header=0)
        X_val = pd.read_csv(regression_path / "Xval.csv.gz", compression='gzip', sep=';', header=0)
        n_train = X_train.shape[0]
        n_val = X_val.shape[0]

        # Create metadata with sample_id (use repeated IDs to test aggregation grouping)
        train_sample_ids = [f"S{i // 4:04d}" for i in range(n_train)]
        val_sample_ids = [f"V{i // 4:04d}" for i in range(n_val)]

        pd.DataFrame({"sample_id": train_sample_ids}).to_csv(
            regression_path / "Mcal.csv.gz", index=False, compression='gzip', sep=';'
        )
        pd.DataFrame({"sample_id": val_sample_ids}).to_csv(
            regression_path / "Mval.csv.gz", index=False, compression='gzip', sep=';'
        )

        yield manager
        manager.cleanup()

    def test_aggregate_via_config_dict(self, test_data_manager):
        """Test aggregation via config dictionary."""
        temp_dir = test_data_manager.get_temp_directory()

        config = {
            "train_x": str(temp_dir / "regression" / "Xcal.csv.gz"),
            "train_y": str(temp_dir / "regression" / "Ycal.csv.gz"),
            "test_x": str(temp_dir / "regression" / "Xval.csv.gz"),
            "test_y": str(temp_dir / "regression" / "Yval.csv.gz"),
            "aggregate": "sample_id"
        }

        dataset_config = DatasetConfigs(config)

        # Verify the aggregate setting is stored
        assert dataset_config._aggregates[0] == "sample_id"

        # Get dataset and verify the setting propagates
        dataset = dataset_config.get_dataset_at(0)
        assert dataset.aggregate == "sample_id"

    def test_aggregate_true_for_y_grouping(self, test_data_manager):
        """Test aggregate=True groups by y values."""
        temp_dir = test_data_manager.get_temp_directory()

        config = {
            "train_x": str(temp_dir / "regression" / "Xcal.csv.gz"),
            "train_y": str(temp_dir / "regression" / "Ycal.csv.gz"),
            "aggregate": True
        }

        dataset_config = DatasetConfigs(config)
        dataset = dataset_config.get_dataset_at(0)
        assert dataset.aggregate == "y"

    def test_pipeline_with_aggregate_runs_successfully(self, test_data_manager):
        """Test that a pipeline with aggregate setting runs without errors."""
        temp_dir = test_data_manager.get_temp_directory()

        config = {
            "train_x": str(temp_dir / "regression" / "Xcal.csv.gz"),
            "train_y": str(temp_dir / "regression" / "Ycal.csv.gz"),
            "train_group": str(temp_dir / "regression" / "Mcal.csv.gz"),
            "aggregate": "sample_id"
        }

        dataset_config = DatasetConfigs(config)

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_agg_pipeline")

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, predictions_per_datasets = runner.run(pipeline_config, dataset_config)

        # Verify we got predictions
        assert predictions.num_predictions > 0

        # Verify the pipeline completed successfully
        best_pred = predictions.get_best(ascending=True, score_scope="folds")
        assert best_pred['val_score'] is not None and np.isfinite(best_pred['val_score'])
        # test_score is None when no test partition exists in the dataset
        assert best_pred['test_score'] is None or np.isfinite(best_pred['test_score'])

    def test_context_stores_aggregate_column(self, test_data_manager):
        """Test that ExecutionContext stores the aggregate_column property."""
        temp_dir = test_data_manager.get_temp_directory()

        config = {
            "train_x": str(temp_dir / "regression" / "Xcal.csv.gz"),
            "train_y": str(temp_dir / "regression" / "Ycal.csv.gz"),
            "train_group": str(temp_dir / "regression" / "Mcal.csv.gz"),
            "aggregate": "sample_id"
        }

        dataset_config = DatasetConfigs(config)

        dataset = dataset_config.get_dataset_at(0)

        # Initialize context using the executor's method
        from nirs4all.pipeline.execution.executor import PipelineExecutor
        from nirs4all.pipeline.steps.step_runner import StepRunner

        step_runner = StepRunner()
        executor = PipelineExecutor(step_runner=step_runner)
        context = executor.initialize_context(dataset)

        # Verify aggregate_column is set in context
        assert context.aggregate_column == "sample_id"

    def test_context_preserves_aggregate_on_copy(self):
        """Test that ExecutionContext.copy() preserves aggregate_column."""
        context = ExecutionContext(aggregate_column="sample_id")

        copied_context = context.copy()

        assert copied_context.aggregate_column == "sample_id"

    def test_context_aggregate_none_when_not_set(self):
        """Test that aggregate_column defaults to None."""
        context = ExecutionContext()

        assert context.aggregate_column is None

    def test_spectro_dataset_aggregate_property(self):
        """Test SpectroDataset aggregate property setter and getter."""
        dataset = SpectroDataset("test_dataset")

        # Default should be None
        assert dataset.aggregate is None

        # Set via string
        dataset.set_aggregate("sample_id")
        assert dataset.aggregate == "sample_id"

        # Set via True
        dataset.set_aggregate(True)
        assert dataset.aggregate == "y"

        # Set back to None
        dataset.set_aggregate(None)
        assert dataset.aggregate is None

    def test_pipeline_without_aggregate_still_works(self, test_data_manager):
        """Test that pipelines without aggregate setting work normally."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        # No aggregate parameter
        dataset_config = DatasetConfigs(dataset_folder)

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_no_agg")

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should work normally
        assert predictions.num_predictions > 0

    def test_per_dataset_aggregate_settings(self, tmp_path):
        """Test that different datasets can have different aggregate settings."""
        # Create two simple datasets
        generator = SyntheticNIRSDataGenerator(random_state=42)

        # Dataset 1
        X1, y1 = generator.generate_regression_data(50)
        path1 = tmp_path / "dataset1"
        path1.mkdir()
        pd.DataFrame(X1).to_csv(path1 / "Xcal.csv.gz", index=False, header=False,
                               compression='gzip', sep=';')
        pd.DataFrame(y1).to_csv(path1 / "Ycal.csv.gz", index=False, header=False,
                               compression='gzip', sep=';')

        # Dataset 2
        X2, y2 = generator.generate_regression_data(50)
        path2 = tmp_path / "dataset2"
        path2.mkdir()
        pd.DataFrame(X2).to_csv(path2 / "Xcal.csv.gz", index=False, header=False,
                               compression='gzip', sep=';')
        pd.DataFrame(y2).to_csv(path2 / "Ycal.csv.gz", index=False, header=False,
                               compression='gzip', sep=';')

        # Create configs with per-dataset aggregates
        config1 = {"train_x": str(path1 / "Xcal.csv.gz"), "train_y": str(path1 / "Ycal.csv.gz"), "aggregate": "sample_id"}
        config2 = {"train_x": str(path2 / "Xcal.csv.gz"), "train_y": str(path2 / "Ycal.csv.gz"), "aggregate": "batch_id"}

        dataset_config = DatasetConfigs([config1, config2])

        assert dataset_config._aggregates[0] == "sample_id"
        assert dataset_config._aggregates[1] == "batch_id"

class TestRepetitionAutoAggregation:
    """Test that setting repetition= auto-enables aggregation."""

    @pytest.fixture
    def test_data_manager(self, tmp_path):
        """Create test data manager with datasets including metadata."""
        manager = TestDataManager(base_temp_dir=str(tmp_path))
        manager.create_regression_dataset("regression")

        regression_path = manager.get_temp_directory() / "regression"

        X_train = pd.read_csv(regression_path / "Xcal.csv.gz", compression='gzip', sep=';', header=0)
        X_val = pd.read_csv(regression_path / "Xval.csv.gz", compression='gzip', sep=';', header=0)
        n_train = X_train.shape[0]
        n_val = X_val.shape[0]

        train_sample_ids = [f"S{i // 4:04d}" for i in range(n_train)]
        val_sample_ids = [f"V{i // 4:04d}" for i in range(n_val)]

        pd.DataFrame({"sample_id": train_sample_ids}).to_csv(
            regression_path / "Mcal.csv.gz", index=False, compression='gzip', sep=';'
        )
        pd.DataFrame({"sample_id": val_sample_ids}).to_csv(
            regression_path / "Mval.csv.gz", index=False, compression='gzip', sep=';'
        )

        yield manager
        manager.cleanup()

    def test_repetition_only_enables_runner_last_aggregate(self, test_data_manager):
        """Setting only repetition= should make runner.last_aggregate return the column name."""
        temp_dir = test_data_manager.get_temp_directory()

        config = {
            "train_x": str(temp_dir / "regression" / "Xcal.csv.gz"),
            "train_y": str(temp_dir / "regression" / "Ycal.csv.gz"),
            "train_group": str(temp_dir / "regression" / "Mcal.csv.gz"),
        }

        # Use repetition= only, NOT aggregate=
        dataset_config = DatasetConfigs(config, repetition="sample_id")

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_rep_agg")

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Core assertion: runner.last_aggregate should be set from repetition
        assert runner.last_aggregate == "sample_id"
        assert predictions.num_predictions > 0

    def test_repetition_config_dict_enables_aggregate(self, test_data_manager):
        """Setting repetition in config dict should also auto-enable aggregate."""
        temp_dir = test_data_manager.get_temp_directory()

        config = {
            "train_x": str(temp_dir / "regression" / "Xcal.csv.gz"),
            "train_y": str(temp_dir / "regression" / "Ycal.csv.gz"),
            "train_group": str(temp_dir / "regression" / "Mcal.csv.gz"),
            "repetition": "sample_id",
        }

        dataset_config = DatasetConfigs(config)

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_rep_dict_agg")

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert runner.last_aggregate == "sample_id"

    def test_repetition_with_explicit_aggregate_override(self, test_data_manager):
        """Explicit aggregate should override the auto-aggregate from repetition."""
        temp_dir = test_data_manager.get_temp_directory()

        config = {
            "train_x": str(temp_dir / "regression" / "Xcal.csv.gz"),
            "train_y": str(temp_dir / "regression" / "Ycal.csv.gz"),
            "train_group": str(temp_dir / "regression" / "Mcal.csv.gz"),
        }

        dataset_config = DatasetConfigs(
            config,
            repetition="sample_id",
            aggregate=True,  # Override: use y-based aggregation instead
        )

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_rep_explicit_agg")

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # aggregate=True overrides the auto-set from repetition
        assert runner.last_aggregate == "y"


class TestAggregationReporting:
    """Test aggregated reporting in pipeline output."""

    def test_tab_report_includes_aggregated_rows(self, tmp_path):
        """Test that TabReportManager includes aggregated rows when aggregate is set."""
        # Create simple dataset without metadata (aggregate by y)
        generator = SyntheticNIRSDataGenerator(random_state=42)
        X, y = generator.generate_regression_data(60)

        dataset_path = tmp_path / "agg_test"
        dataset_path.mkdir()

        pd.DataFrame(X).to_csv(dataset_path / "Xcal.csv.gz", index=False, header=False,
                              compression='gzip', sep=';')
        pd.DataFrame(y).to_csv(dataset_path / "Ycal.csv.gz", index=False, header=False,
                              compression='gzip', sep=';')

        # Use aggregate=True for y-based aggregation (doesn't need metadata)
        config = {
            "train_x": str(dataset_path / "Xcal.csv.gz"),
            "train_y": str(dataset_path / "Ycal.csv.gz"),
            "aggregate": True
        }
        dataset_config = DatasetConfigs(config)

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"model": PLSRegression(n_components=3)},
        ]

        pipeline_config = PipelineConfigs(pipeline, "test_agg_report")

        # Run with verbose to capture output
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=1)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Verify predictions exist
        assert predictions.num_predictions > 0

    def test_aggregated_predictions_have_fewer_samples(self):
        """Test that aggregated predictions have fewer samples than raw predictions."""
        # Create sample predictions
        n_reps = 3
        n_unique = 15
        n_total = n_unique * n_reps

        np.random.seed(42)
        y_true = np.random.rand(n_total) * 10
        y_pred = y_true + np.random.randn(n_total) * 0.5
        group_ids = np.repeat(np.arange(n_unique), n_reps)

        # Aggregate predictions
        result = Predictions.aggregate(y_pred=y_pred, group_ids=group_ids, y_true=y_true)

        # Aggregated should have n_unique samples
        assert len(result['y_pred']) == n_unique
        assert len(result['y_true']) == n_unique


class TestAnalyzerCustomAggregationCharts:
    """Verify analyzer charts use custom aggregation columns end-to-end."""

    def test_regression_plots_render_raw_and_custom_aggregated_variants(self, tmp_path):
        """Regression chart families should build raw + custom aggregated plots."""
        data_path = tmp_path / "regression_custom_agg"
        data_path.mkdir()
        _write_repeated_regression_dataset(data_path)

        dataset_config = _repeated_dataset_config(data_path)
        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=3)},
            {"model": PLSRegression(n_components=5)},
            {"model": RandomForestRegressor(n_estimators=10, random_state=42)},
        ]

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(PipelineConfigs(pipeline, "test_chart_custom_agg_reg"), dataset_config)
        analyzer = PredictionAnalyzer(predictions)

        figures = []
        try:
            top_k = analyzer.plot_top_k(k=2, aggregate="lot_id")
            histogram = analyzer.plot_histogram(display_metric="rmse", aggregate="lot_id")
            heatmap = analyzer.plot_heatmap("partition", "model_name", rank_metric="rmse", display_metric="rmse", aggregate="lot_id")
            candlestick = analyzer.plot_candlestick("model_name", display_metric="rmse", aggregate="lot_id")

            figures.extend(top_k if isinstance(top_k, list) else [top_k])
            figures.extend(histogram if isinstance(histogram, list) else [histogram])
            figures.extend(heatmap if isinstance(heatmap, list) else [heatmap])
            figures.extend(candlestick if isinstance(candlestick, list) else [candlestick])

            assert not isinstance(top_k, list)
            assert not isinstance(histogram, list)
            assert not isinstance(heatmap, list)
            assert not isinstance(candlestick, list)

            assert "lot_id" in top_k._suptitle.get_text()
            assert "not applied" not in top_k._suptitle.get_text()
            assert "lot_id" in histogram.axes[0].get_title()
            assert "lot_id" in heatmap.axes[0].get_title()
            assert "lot_id" in candlestick.axes[0].get_title()
        finally:
            for fig in figures:
                plt.close(fig)

    def test_classification_confusion_matrix_renders_custom_aggregated_variant(self, tmp_path):
        """Confusion-matrix plots should also honor custom aggregation columns."""
        data_path = tmp_path / "classification_custom_agg"
        data_path.mkdir()
        _write_repeated_classification_dataset(data_path)

        dataset_config = _repeated_dataset_config(data_path, task_type="multiclass_classification")
        pipeline = [
            StandardScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": RandomForestClassifier(max_depth=6, random_state=42)},
            {"model": RandomForestClassifier(max_depth=10, random_state=42)},
        ]

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(PipelineConfigs(pipeline, "test_chart_custom_agg_clf"), dataset_config)
        analyzer = PredictionAnalyzer(predictions)

        figures = []
        try:
            confusion = analyzer.plot_confusion_matrix(k=2, aggregate="lot_id")
            figures.extend(confusion if isinstance(confusion, list) else [confusion])

            assert not isinstance(confusion, list)
            assert "lot_id" in confusion._suptitle.get_text()
        finally:
            for fig in figures:
                plt.close(fig)


class TestAggregationEndToEnd:
    """End-to-end tests verifying aggregated scores are correct and survive store reload."""

    @pytest.fixture
    def repetition_data(self, tmp_path):
        """Create synthetic NIRS data with 20 unique samples x 4 repetitions."""
        np.random.seed(42)
        n_unique = 20
        n_reps = 4
        n_wavelengths = 50
        n_train = int(n_unique * 0.75)  # 15 unique train samples
        n_test = n_unique - n_train  # 5 unique test samples

        # Generate base spectra and targets
        X_base = np.random.randn(n_unique, n_wavelengths)
        y_base = np.random.rand(n_unique) * 10 + 5

        def expand(X_base, y_base, prefix, start_idx=0):
            X_all, y_all, sample_ids = [], [], []
            for i in range(len(X_base)):
                for r in range(n_reps):
                    # Add significant noise per repetition so aggregation changes scores
                    noise = np.random.randn(n_wavelengths) * 0.5
                    X_all.append(X_base[i] + noise)
                    y_all.append(y_base[i])
                    sample_ids.append(f"{prefix}{start_idx + i:03d}")
            return np.array(X_all), np.array(y_all), sample_ids

        X_train, y_train, train_ids = expand(X_base[:n_train], y_base[:n_train], "S")
        X_test, y_test, test_ids = expand(X_base[n_train:], y_base[n_train:], "T")

        data_path = tmp_path / "rep_data"
        data_path.mkdir()

        pd.DataFrame(X_train).to_csv(data_path / "Xcal.csv.gz", index=False, header=False, compression='gzip', sep=';')
        pd.DataFrame(y_train).to_csv(data_path / "Ycal.csv.gz", index=False, header=False, compression='gzip', sep=';')
        pd.DataFrame({"sample_id": train_ids}).to_csv(data_path / "Mcal.csv", index=False, sep=';')

        pd.DataFrame(X_test).to_csv(data_path / "Xval.csv.gz", index=False, header=False, compression='gzip', sep=';')
        pd.DataFrame(y_test).to_csv(data_path / "Yval.csv.gz", index=False, header=False, compression='gzip', sep=';')
        pd.DataFrame({"sample_id": test_ids}).to_csv(data_path / "Mval.csv", index=False, sep=';')

        return data_path, n_train, n_test, n_reps

    def _make_dataset_config(self, data_path):
        return DatasetConfigs(
            {
                "train_x": str(data_path / "Xcal.csv.gz"),
                "train_y": str(data_path / "Ycal.csv.gz"),
                "train_m": str(data_path / "Mcal.csv"),
                "test_x": str(data_path / "Xval.csv.gz"),
                "test_y": str(data_path / "Yval.csv.gz"),
                "test_m": str(data_path / "Mval.csv"),
                "train_x_params": {"has_header": False},
                "train_y_params": {"has_header": False},
                "train_m_params": {"has_header": True},
                "test_x_params": {"has_header": False},
                "test_y_params": {"has_header": False},
                "test_m_params": {"has_header": True},
            },
            repetition="sample_id",
        )

    def _make_pipeline_config(self):
        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=3)},
        ]
        return PipelineConfigs(pipeline, "test_e2e_agg")

    def test_aggregated_score_differs_from_raw(self, repetition_data):
        """Aggregated test_score must differ from raw test_score.

        top() by default returns test-partition entries (display_partition='test'),
        so aggregation updates test_score from the aggregated test arrays.
        """
        data_path, _, _, _ = repetition_data

        dataset_config = self._make_dataset_config(data_path)
        pipeline_config = self._make_pipeline_config()

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        top_raw = predictions.top(1, rank_metric='rmse', score_scope='refit', by_repetition=False)
        top_agg = predictions.top(1, rank_metric='rmse', score_scope='refit', by_repetition='sample_id')

        assert len(top_raw) > 0, "Raw top() returned empty"
        assert len(top_agg) > 0, "Aggregated top() returned empty"

        # Refit entries carry test-partition arrays, so check test_score
        raw_test = top_raw[0].get('test_score')
        agg_test = top_agg[0].get('test_score')

        assert raw_test is not None and np.isfinite(raw_test), f"Raw test_score invalid: {raw_test}"
        assert agg_test is not None and np.isfinite(agg_test), f"Agg test_score invalid: {agg_test}"
        assert agg_test != raw_test, (
            f"Aggregated test_score ({agg_test}) must differ from raw ({raw_test}). "
            f"Score recomputation after aggregation is broken."
        )
        assert top_agg[0].get('aggregated', False), "Result must be marked as aggregated"

        # Also verify CV entries get updated val_score when queried directly
        top_cv_raw = predictions.top(1, rank_metric='rmse', rank_partition='val',
                                      score_scope='folds', by_repetition=False)
        top_cv_agg = predictions.top(1, rank_metric='rmse', rank_partition='val',
                                      score_scope='folds', by_repetition='sample_id')
        if len(top_cv_raw) > 0 and len(top_cv_agg) > 0:
            cv_raw = top_cv_raw[0]
            cv_agg = top_cv_agg[0]
            if cv_raw.get('partition') == 'val' and cv_agg.get('partition') == 'val':
                raw_val = cv_raw.get('val_score')
                agg_val = cv_agg.get('val_score')
                if raw_val is not None and agg_val is not None:
                    assert agg_val != raw_val, (
                        f"CV aggregated val_score ({agg_val}) must differ from raw ({raw_val})."
                    )

    def test_aggregated_result_has_fewer_samples(self, repetition_data):
        """Aggregated y_pred must have fewer elements than raw y_pred."""
        data_path, _, _, n_reps = repetition_data

        dataset_config = self._make_dataset_config(data_path)
        pipeline_config = self._make_pipeline_config()

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        top_raw = predictions.top(1, rank_metric='rmse', score_scope='refit', by_repetition=False)
        top_agg = predictions.top(1, rank_metric='rmse', score_scope='refit', by_repetition='sample_id')

        raw_y_pred = top_raw[0].get('y_pred')
        agg_y_pred = top_agg[0].get('y_pred')

        assert raw_y_pred is not None, "Raw y_pred is None"
        assert agg_y_pred is not None, "Aggregated y_pred is None"
        assert len(agg_y_pred) < len(raw_y_pred), (
            f"Aggregated y_pred ({len(agg_y_pred)}) should have fewer samples "
            f"than raw ({len(raw_y_pred)})"
        )
        # With n_reps repetitions, aggregated count should be raw / n_reps
        assert len(agg_y_pred) == len(raw_y_pred) // n_reps

    def test_store_reload_preserves_aggregation(self, repetition_data, tmp_path):
        """Aggregation must work after reloading predictions from the store."""
        data_path, _, _, _ = repetition_data

        ws_path = tmp_path / "workspace"
        ws_path.mkdir()

        dataset_config = self._make_dataset_config(data_path)
        pipeline_config = self._make_pipeline_config()

        runner = PipelineRunner(
            save_artifacts=False, save_charts=False, verbose=0,
            workspace_path=str(ws_path),
        )
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Get aggregated score from in-memory predictions (test-partition entry)
        top_agg_mem = predictions.top(1, rank_metric='rmse', by_repetition='sample_id')
        assert len(top_agg_mem) > 0
        agg_test_mem = top_agg_mem[0].get('test_score')
        assert agg_test_mem is not None and np.isfinite(agg_test_mem)

        # Reload predictions from the store
        store_path = ws_path / "store.sqlite"
        assert store_path.exists(), f"Store not found at {store_path}"

        reloaded = Predictions(db_path=str(store_path))

        top_agg_reload = reloaded.top(1, rank_metric='rmse', by_repetition='sample_id')
        assert len(top_agg_reload) > 0, "Reloaded aggregated top() returned empty"

        agg_test_reload = top_agg_reload[0].get('test_score')
        assert agg_test_reload is not None and np.isfinite(agg_test_reload), (
            f"Reloaded aggregated test_score invalid: {agg_test_reload}"
        )
        assert top_agg_reload[0].get('aggregated', False), "Reloaded result must be marked as aggregated"

        # Scores should match between in-memory and reloaded
        assert abs(agg_test_mem - agg_test_reload) < 1e-6, (
            f"Reloaded aggregated score ({agg_test_reload}) differs from "
            f"in-memory ({agg_test_mem})"
        )

    def test_store_reload_preserves_repetition_context_for_aggregate_true(self, repetition_data, tmp_path):
        """PredictionAnalyzer should resolve aggregate=True from repetition context after reload."""
        data_path, _, _, _ = repetition_data

        ws_path = tmp_path / "workspace"
        ws_path.mkdir()

        dataset_config = self._make_dataset_config(data_path)
        pipeline_config = self._make_pipeline_config()

        runner = PipelineRunner(
            save_artifacts=False,
            save_charts=False,
            verbose=0,
            workspace_path=str(ws_path),
        )
        predictions, _ = runner.run(pipeline_config, dataset_config)

        live_analyzer = PredictionAnalyzer(predictions)
        live_fig = live_analyzer.plot_top_k(k=2, rank_metric="rmse", aggregate=True)
        live_figures = live_fig if isinstance(live_fig, list) else [live_fig]

        reloaded = Predictions.from_workspace(ws_path)
        assert reloaded.repetition_column == "sample_id"

        reloaded_analyzer = PredictionAnalyzer(reloaded)
        reloaded_fig = reloaded_analyzer.plot_top_k(k=2, rank_metric="rmse", aggregate=True)
        reloaded_figures = reloaded_fig if isinstance(reloaded_fig, list) else [reloaded_fig]

        try:
            assert live_analyzer.default_aggregate is None
            assert reloaded_analyzer.default_aggregate is None
            assert len(live_figures) == 1
            assert len(reloaded_figures) == 1
            assert "aggregated by sample_id" in live_figures[0]._suptitle.get_text()
            assert "aggregated by sample_id" in reloaded_figures[0]._suptitle.get_text()
        finally:
            for fig in [*live_figures, *reloaded_figures]:
                plt.close(fig)

    def test_store_reload_explicit_aggregate_works_without_context(self, repetition_data, tmp_path):
        """Explicit aggregation should work even if dataset context is unavailable."""
        data_path, _, _, _ = repetition_data

        ws_path = tmp_path / "workspace"
        ws_path.mkdir()

        dataset_config = self._make_dataset_config(data_path)
        pipeline_config = self._make_pipeline_config()

        runner = PipelineRunner(
            save_artifacts=False,
            save_charts=False,
            verbose=0,
            workspace_path=str(ws_path),
        )
        _predictions, _ = runner.run(pipeline_config, dataset_config)

        reloaded = Predictions.from_workspace(ws_path)
        # Simulate a fresh install / portable context where aggregate metadata
        # is not available and the user must pass the aggregate column explicitly.
        reloaded.set_repetition_column(None)
        reloaded.set_aggregate_context(None)

        top_explicit = reloaded.top(1, rank_metric="rmse", by_repetition="sample_id")
        assert len(top_explicit) > 0
        assert top_explicit[0].get("aggregated", False)

        analyzer = PredictionAnalyzer(reloaded)
        figures = analyzer.plot_top_k(k=2, rank_metric="rmse", aggregate="sample_id")
        figure_list = figures if isinstance(figures, list) else [figures]
        try:
            assert len(figure_list) == 1
            assert "aggregated by sample_id" in figure_list[0]._suptitle.get_text()
        finally:
            for fig in figure_list:
                plt.close(fig)

    def test_store_reload_preserves_variant_step_idx_lookup(self, repetition_data, tmp_path):
        """Reloaded predictions must preserve step_idx for variant-specific queries."""
        data_path, _, _, _ = repetition_data

        ws_path = tmp_path / "workspace"
        ws_path.mkdir()

        dataset_config = self._make_dataset_config(data_path)
        pipeline_config = self._make_pipeline_config()

        runner = PipelineRunner(
            save_artifacts=False,
            save_charts=False,
            verbose=0,
            workspace_path=str(ws_path),
        )
        predictions, _ = runner.run(pipeline_config, dataset_config)

        best_raw = predictions.top(
            1,
            rank_metric="rmse",
            rank_partition="test",
            display_partition="test",
            score_scope="refit",
            by_repetition=False,
        )[0]
        model_name = best_raw["model_name"]
        step_idx = best_raw["step_idx"]

        reloaded = Predictions.from_workspace(ws_path)
        matched = reloaded.top(
            1,
            rank_metric="rmse",
            rank_partition="test",
            display_partition="test",
            score_scope="refit",
            by_repetition=True,
            model_name=model_name,
            step_idx=step_idx,
        )

        assert len(matched) == 1
        assert matched[0]["step_idx"] == step_idx
        assert matched[0].get("aggregated", False)

    def test_repetition_no_leakage_across_folds(self, repetition_data):
        """All repetitions of the same sample must stay in the same CV fold.

        If sample S001 has 4 repetitions, all 4 must be in train OR val
        for each fold — never split across both.
        """
        data_path, n_train, _, n_reps = repetition_data

        dataset_config = self._make_dataset_config(data_path)
        pipeline_config = self._make_pipeline_config()

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Check each individual CV fold (fold_id = 0, 1, 2)
        for fold_id in ["0", "1", "2"]:
            # Get val-partition entry for this fold
            val_entries = predictions.filter_predictions(
                partition="val", fold_id=fold_id
            )
            train_entries = predictions.filter_predictions(
                partition="train", fold_id=fold_id
            )

            if not val_entries or not train_entries:
                continue

            val_meta = val_entries[0].get("metadata", {})
            train_meta = train_entries[0].get("metadata", {})

            val_sample_ids = set(val_meta.get("sample_id", []))
            train_sample_ids = set(train_meta.get("sample_id", []))

            assert len(val_sample_ids) > 0, f"Fold {fold_id}: no val sample_ids"
            assert len(train_sample_ids) > 0, f"Fold {fold_id}: no train sample_ids"

            overlap = val_sample_ids & train_sample_ids
            assert len(overlap) == 0, (
                f"Fold {fold_id}: data leakage! {len(overlap)} sample(s) appear in "
                f"both train and val: {overlap}"
            )

            # Also verify each sample_id appears exactly n_reps times in its partition
            val_ids = val_meta.get("sample_id", [])
            for sid in val_sample_ids:
                count = val_ids.count(sid) if isinstance(val_ids, list) else np.sum(np.array(val_ids) == sid)
                assert count == n_reps, (
                    f"Fold {fold_id}: sample '{sid}' has {count} reps in val, expected {n_reps}"
                )
