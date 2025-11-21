import os
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.data.config import DatasetConfigs
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from tests.fixtures.data_generators import TestDataManager

class TestSavingMechanism:

    @pytest.fixture
    def test_data(self):
        """Create test data."""
        manager = TestDataManager()
        manager.create_classification_dataset("classification")
        yield manager
        manager.cleanup()

    def test_fold_chart_saving(self, test_data):
        """
        Test that the FoldChartController (refactored to use StepOutput)
        correctly saves the chart file via the Executor.
        """
        dataset_folder = str(test_data.get_temp_directory() / "classification")

        # Pipeline with just the fold chart
        pipeline = [
            "fold_chart"
        ]

        pipeline_config = PipelineConfigs(pipeline, "saving_test")
        dataset_config = DatasetConfigs(dataset_folder)

        # Create runner with save_files=True
        # We need a temp workspace for the runner to avoid cluttering the real workspace
        with tempfile.TemporaryDirectory() as temp_workspace:
            runner = PipelineRunner(
                workspace_path=temp_workspace,
                save_files=True,
                verbose=0,
                plots_visible=False
            )

            # Run the pipeline
            runner.run(pipeline_config, dataset_config)

            # Check if the file was saved
            run_dir = runner.current_run_dir
            assert run_dir is not None
            assert run_dir.exists()

            # Find the pipeline directory (it should be the only subdirectory starting with digits)
            pipeline_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
            assert len(pipeline_dirs) == 1, f"Expected 1 pipeline dir, found {len(pipeline_dirs)}: {[d.name for d in pipeline_dirs]}"
            pipeline_dir = pipeline_dirs[0]

            # The file name should be "fold_visualization_traintest_split_train.png"
            # The step number prefix is NOT added by the current SimulationSaver.

            expected_file = pipeline_dir / "fold_visualization_traintest_split_train.png"

            # List files in pipeline dir for debugging if assertion fails
            # files = list(pipeline_dir.glob("*"))
            # print(f"Files in pipeline dir: {[f.name for f in files]}")

            assert expected_file.exists(), f"Expected file {expected_file.name} not found in {pipeline_dir}"
            assert expected_file.stat().st_size > 0, "File is empty"

    def test_fold_chart_saving_with_folds(self, test_data):
        """
        Test saving with actual folds (CV mode).
        """
        dataset_folder = str(test_data.get_temp_directory() / "classification")

        from sklearn.model_selection import KFold

        pipeline = [
            {
                "split": KFold(n_splits=3, shuffle=True, random_state=42)
            },
            "fold_chart"
        ]

        pipeline_config = PipelineConfigs(pipeline, "saving_test_folds")
        dataset_config = DatasetConfigs(dataset_folder)

        with tempfile.TemporaryDirectory() as temp_workspace:
            runner = PipelineRunner(
                workspace_path=temp_workspace,
                save_files=True,
                verbose=0,
                plots_visible=False
            )

            runner.run(pipeline_config, dataset_config)

            run_dir = runner.current_run_dir
            assert run_dir is not None

            # Find pipeline dir
            pipeline_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
            assert len(pipeline_dirs) == 1
            pipeline_dir = pipeline_dirs[0]

            # Step 1 is split, Step 2 is fold_chart
            # Name: "fold_visualization_3folds_train.png"
            expected_file = pipeline_dir / "fold_visualization_3folds_train.png"

            # files = list(pipeline_dir.glob("*"))
            # print(f"Files in pipeline dir: {[f.name for f in files]}")

            assert expected_file.exists()
    def test_spectra_chart_saving(self, test_data):
        """Test saving of spectra charts."""
        dataset_folder = str(test_data.get_temp_directory() / "classification")

        pipeline = ["chart_2d"]

        pipeline_config = PipelineConfigs(pipeline, "saving_test_spectra")
        dataset_config = DatasetConfigs(dataset_folder)

        with tempfile.TemporaryDirectory() as temp_workspace:
            runner = PipelineRunner(
                workspace_path=temp_workspace,
                save_files=True,
                verbose=0,
                plots_visible=False
            )

            runner.run(pipeline_config, dataset_config)

            run_dir = runner.current_run_dir
            assert run_dir is not None

            pipeline_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
            assert len(pipeline_dirs) == 1
            pipeline_dir = pipeline_dirs[0]

            # Name: "2D_Chart.png" (or with source suffix if multi-source)
            # Since it's single source: "2D_Chart.png"
            expected_file = pipeline_dir / "2D_Chart.png"

            assert expected_file.exists()

    def test_y_chart_saving(self, test_data):
        """Test saving of Y distribution charts."""
        dataset_folder = str(test_data.get_temp_directory() / "classification")

        pipeline = ["chart_y"]

        pipeline_config = PipelineConfigs(pipeline, "saving_test_y")
        dataset_config = DatasetConfigs(dataset_folder)

        with tempfile.TemporaryDirectory() as temp_workspace:
            runner = PipelineRunner(
                workspace_path=temp_workspace,
                save_files=True,
                verbose=0,
                plots_visible=False
            )

            runner.run(pipeline_config, dataset_config)

            run_dir = runner.current_run_dir
            assert run_dir is not None

            pipeline_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
            assert len(pipeline_dirs) == 1
            pipeline_dir = pipeline_dirs[0]

            # Name: "Y_distribution_train_test.png"
            expected_file = pipeline_dir / "Y_distribution_train_test.png"

            assert expected_file.exists()

    def test_splitter_saving(self, test_data):
        """Test saving of splitter CSVs."""
        dataset_folder = str(test_data.get_temp_directory() / "classification")

        from sklearn.model_selection import KFold

        pipeline = [
            {
                "split": KFold(n_splits=3, shuffle=True, random_state=42)
            }
        ]

        pipeline_config = PipelineConfigs(pipeline, "saving_test_splitter")
        dataset_config = DatasetConfigs(dataset_folder)

        with tempfile.TemporaryDirectory() as temp_workspace:
            runner = PipelineRunner(
                workspace_path=temp_workspace,
                save_files=True,
                verbose=0,
                plots_visible=False
            )

            runner.run(pipeline_config, dataset_config)

            run_dir = runner.current_run_dir
            assert run_dir is not None

            pipeline_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
            assert len(pipeline_dirs) == 1
            pipeline_dir = pipeline_dirs[0]

            # Name: "folds_KFold_seed42.csv"
            expected_file = pipeline_dir / "folds_KFold_seed42.csv"

            assert expected_file.exists()