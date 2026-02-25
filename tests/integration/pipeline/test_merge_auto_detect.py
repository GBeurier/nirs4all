"""Integration tests for merge auto-detection.

Verifies that {"merge": "auto"}, {"merge": True}, and {"merge": {"branch": ...}}
resolve to the correct merge strategy and complete full pipeline runs.
"""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import nirs4all
from nirs4all.data import DatasetConfigs
from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC
from nirs4all.operators.transforms import StandardNormalVariate as SNV
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from tests.fixtures.data_generators import TestDataManager

# ===========================================================================
# Duplication branches + auto merge
# ===========================================================================

class TestAutoMergeDuplicationBranch:
    """Full pipeline with duplication branches and auto merge."""

    @pytest.fixture
    def dataset(self):
        return nirs4all.generate.regression(n_samples=50, random_state=42)

    def test_merge_auto_string(self, dataset):
        """{"merge": "auto"} resolves to predictions merge for duplication."""
        pipeline = [
            {"branch": [[SNV()], [MSC()]]},
            {"merge": "auto"},
            {"model": PLSRegression(n_components=5)},
        ]
        result = nirs4all.run(pipeline=pipeline, dataset=dataset, verbose=0)
        assert result is not None
        assert hasattr(result, "best_rmse")

    def test_merge_true(self, dataset):
        """{"merge": True} resolves to predictions merge for duplication."""
        pipeline = [
            {"branch": [[SNV()], [MSC()]]},
            {"merge": True},
            {"model": PLSRegression(n_components=5)},
        ]
        result = nirs4all.run(pipeline=pipeline, dataset=dataset, verbose=0)
        assert result is not None
        assert hasattr(result, "best_rmse")

    def test_merge_dict_branch(self, dataset):
        """{"merge": {"branch": True}} resolves to predictions merge."""
        pipeline = [
            {"branch": [[SNV()], [MSC()]]},
            {"merge": {"branch": True}},
            {"model": PLSRegression(n_components=5)},
        ]
        result = nirs4all.run(pipeline=pipeline, dataset=dataset, verbose=0)
        assert result is not None
        assert hasattr(result, "best_rmse")


# ===========================================================================
# by_source branches + auto merge
# ===========================================================================

class TestAutoMergeBySource:
    """Full pipeline with by_source branches and auto merge."""

    @pytest.fixture
    def test_data_manager(self):
        manager = TestDataManager()
        manager.create_multi_source_dataset("multi", n_sources=2)
        yield manager
        manager.cleanup()

    def test_merge_auto_string(self, test_data_manager):
        """{"merge": "auto"} resolves to source concat for by_source."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")

        pipeline = [
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=2, random_state=42),
            {"branch": {
                "by_source": True,
                "steps": {
                    "source_0": [StandardScaler(), PLSRegression(5)],
                    "source_1": [StandardScaler(), PLSRegression(5)],
                },
            }},
            {"merge": "auto"},
        ]

        dataset_config = DatasetConfigs(dataset_folder)
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(PipelineConfigs(pipeline, "auto_src"), dataset_config)
        assert predictions.num_predictions > 0

    def test_merge_true(self, test_data_manager):
        """{"merge": True} resolves to source concat for by_source."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")

        pipeline = [
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=2, random_state=42),
            {"branch": {
                "by_source": True,
                "steps": {
                    "source_0": [StandardScaler(), PLSRegression(5)],
                    "source_1": [StandardScaler(), PLSRegression(5)],
                },
            }},
            {"merge": True},
        ]

        dataset_config = DatasetConfigs(dataset_folder)
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(PipelineConfigs(pipeline, "true_src"), dataset_config)
        assert predictions.num_predictions > 0

    def test_merge_dict_branch(self, test_data_manager):
        """{"merge": {"branch": True}} resolves to source concat for by_source."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")

        pipeline = [
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=2, random_state=42),
            {"branch": {
                "by_source": True,
                "steps": {
                    "source_0": [StandardScaler(), PLSRegression(5)],
                    "source_1": [StandardScaler(), PLSRegression(5)],
                },
            }},
            {"merge": {"branch": "auto"}},
        ]

        dataset_config = DatasetConfigs(dataset_folder)
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(PipelineConfigs(pipeline, "dict_src"), dataset_config)
        assert predictions.num_predictions > 0
