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
from tests.fixtures.data_generators import TestDataManager

# ===========================================================================
# Duplication branches + auto merge
# ===========================================================================

class TestAutoMergeDuplicationBranch:
    """Full pipeline with duplication branches and auto merge."""

    @pytest.fixture
    def dataset(self):
        return nirs4all.generate.regression(n_samples=50, random_state=42, engine="legacy")

    def test_merge_auto_string(self, dataset):
        """{"merge": "auto"} resolves to feature merge for duplication."""
        pipeline = [
            {"branch": [[SNV()], [MSC()]]},
            {"merge": "auto"},
            {"model": PLSRegression(n_components=5)},
        ]
        result = nirs4all.run(pipeline=pipeline, dataset=dataset, engine="dag-ml", verbose=0)
        assert all(item["engine"] == "dag-ml" for item in result.per_dataset.values())
        assert np.isfinite(result.best_rmse)

    def test_merge_true(self, dataset):
        """{"merge": True} resolves to feature merge for duplication."""
        pipeline = [
            {"branch": [[SNV()], [MSC()]]},
            {"merge": True},
            {"model": PLSRegression(n_components=5)},
        ]
        result = nirs4all.run(pipeline=pipeline, dataset=dataset, engine="dag-ml", verbose=0)
        assert all(item["engine"] == "dag-ml" for item in result.per_dataset.values())
        assert np.isfinite(result.best_rmse)

    def test_merge_dict_branch(self, dataset):
        """{"merge": {"branch": True}} resolves to feature merge."""
        pipeline = [
            {"branch": [[SNV()], [MSC()]]},
            {"merge": {"branch": True}},
            {"model": PLSRegression(n_components=5)},
        ]
        result = nirs4all.run(pipeline=pipeline, dataset=dataset, engine="dag-ml", verbose=0)
        assert all(item["engine"] == "dag-ml" for item in result.per_dataset.values())
        assert np.isfinite(result.best_rmse)


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

        result = nirs4all.run(pipeline=pipeline, dataset=DatasetConfigs(dataset_folder), engine="dag-ml", verbose=0)
        assert result.num_predictions > 0
        assert set(result.predictions.get_unique_values("branch_name")) == {"source_0", "source_1"}
        val_rows = [row for row in result.predictions.filter_predictions(load_arrays=True) if row["partition"] == "val"]
        assert {row["branch_name"] for row in val_rows} == {"source_0", "source_1"}
        assert all(len(row["y_pred"]) > 0 and np.isfinite(row["val_score"]) for row in val_rows)
        avg_rows = [row for row in val_rows if row["fold_id"] == "avg"]
        assert len(avg_rows) == 2
        assert all(len(row["y_pred"]) == len(row["y_true"]) > 0 for row in avg_rows)
        weighted_rows = [row for row in val_rows if row["fold_id"] == "w_avg"]
        assert len(weighted_rows) == 2
        assert all(len(row["y_pred"]) == len(row["y_true"]) > 0 for row in weighted_rows)

    def test_merge_auto_without_cv(self, test_data_manager):
        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")
        pipeline = [
            {"y_processing": MinMaxScaler()},
            {"branch": {
                "by_source": True,
                "steps": {
                    "source_0": [StandardScaler(), PLSRegression(5)],
                    "source_1": [StandardScaler(), PLSRegression(5)],
                },
            }},
            {"merge": "auto"},
        ]
        result = nirs4all.run(pipeline=pipeline, dataset=DatasetConfigs(dataset_folder), engine="dag-ml", verbose=0)
        assert set(result.predictions.get_unique_values("branch_name")) == {"source_0", "source_1"}
        assert np.isfinite(result.best_rmse)
        assert np.isnan(result.cv_best_score)

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

        result = nirs4all.run(pipeline=pipeline, dataset=DatasetConfigs(dataset_folder), engine="dag-ml", verbose=0)
        assert result.num_predictions > 0
        assert set(result.predictions.get_unique_values("branch_name")) == {"source_0", "source_1"}

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

        result = nirs4all.run(pipeline=pipeline, dataset=DatasetConfigs(dataset_folder), engine="dag-ml", verbose=0)
        assert result.num_predictions > 0
        assert set(result.predictions.get_unique_values("branch_name")) == {"source_0", "source_1"}
