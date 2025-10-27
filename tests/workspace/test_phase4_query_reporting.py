"""
Test Phase 4: Query and Reporting

Tests query methods for Predictions catalog.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import polars as pl
from nirs4all.dataset.predictions import Predictions


class TestPhase4QueryReporting:
    """Test Phase 4 query and reporting methods."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions dataframe."""
        pred = Predictions()
        pred._df = pl.DataFrame({
            "prediction_id": ["pred_001", "pred_002", "pred_003", "pred_004"],
            "dataset_name": ["wheat", "wheat", "corn", "corn"],
            "config_name": ["baseline_pls", "advanced_pls", "baseline_pls", "advanced_rf"],
            "test_score": [0.45, 0.52, 0.38, 0.61],
            "train_score": [0.32, 0.39, 0.28, 0.55],
            "val_score": [0.41, 0.48, 0.35, 0.58],
            "model_name": ["PLSRegression", "PLSRegression", "PLSRegression", "RandomForest"],
            "pipeline_hash": ["abc123", "def456", "abc123", "ghi789"]
        })
        return pred

    def test_query_best(self, sample_predictions):
        """Test finding best pipelines by metric."""
        pred = sample_predictions

        # Get top 2 by test_score
        best = pred.query_best(metric="test_score", n=2, ascending=False)

        assert best.height == 2
        # Should be sorted descending (best first)
        scores = best["test_score"].to_list()
        assert scores[0] >= scores[1]
        assert scores[0] == 0.61  # Best score

    def test_query_best_with_dataset_filter(self, sample_predictions):
        """Test finding best pipelines filtered by dataset."""
        pred = sample_predictions

        # Get top 2 for wheat only
        best = pred.query_best(dataset_name="wheat", metric="test_score", n=2)

        assert best.height == 2
        # All should be wheat
        datasets = best["dataset_name"].to_list()
        assert all(d == "wheat" for d in datasets)

        # Best wheat score should be 0.52
        scores = best["test_score"].to_list()
        assert max(scores) == 0.52

    def test_filter_by_criteria_single(self, sample_predictions):
        """Test filtering by single criterion."""
        pred = sample_predictions

        # Filter by dataset
        filtered = pred.filter_by_criteria(dataset_name="corn")

        assert filtered.height == 2
        datasets = filtered["dataset_name"].to_list()
        assert all(d == "corn" for d in datasets)

    def test_filter_by_criteria_metric_threshold(self, sample_predictions):
        """Test filtering by metric threshold."""
        pred = sample_predictions

        # Filter by test_score >= 0.50
        filtered = pred.filter_by_criteria(
            metric_thresholds={"test_score": 0.50}
        )

        assert filtered.height == 2  # Only 0.52 and 0.61
        scores = filtered["test_score"].to_list()
        assert all(s >= 0.50 for s in scores)

    def test_filter_by_criteria_multiple(self, sample_predictions):
        """Test filtering by multiple criteria."""
        pred = sample_predictions

        # Filter by dataset AND metric threshold
        filtered = pred.filter_by_criteria(
            dataset_name="wheat",
            metric_thresholds={"test_score": 0.50}
        )

        assert filtered.height == 1  # Only wheat with 0.52
        assert filtered["dataset_name"][0] == "wheat"
        assert filtered["test_score"][0] == 0.52

    def test_compare_across_datasets(self, sample_predictions):
        """Test comparing same pipeline across datasets."""
        pred = sample_predictions

        # Compare baseline_pls (hash abc123) across datasets
        comparison = pred.compare_across_datasets("abc123", metric="test_score")

        assert comparison.height == 2  # wheat and corn
        datasets = comparison["dataset_name"].to_list()
        assert "wheat" in datasets
        assert "corn" in datasets

        # Check aggregated stats columns exist
        assert "test_score_min" in comparison.columns
        assert "test_score_max" in comparison.columns
        assert "test_score_mean" in comparison.columns
        assert "num_predictions" in comparison.columns

    def test_list_runs(self, sample_predictions):
        """Test listing runs with summary statistics."""
        pred = sample_predictions

        runs = pred.list_runs()

        assert runs.height >= 1
        assert "dataset_name" in runs.columns
        assert "num_pipelines" in runs.columns
        assert "best_score" in runs.columns

        # Should have counts per dataset
        wheat_rows = runs.filter(pl.col("dataset_name") == "wheat")
        if wheat_rows.height > 0:
            assert wheat_rows["num_pipelines"][0] == 2

    def test_list_runs_with_filter(self, sample_predictions):
        """Test listing runs filtered by dataset."""
        pred = sample_predictions

        runs = pred.list_runs(dataset_name="corn")

        # All rows should be for corn
        datasets = runs["dataset_name"].to_list()
        assert all(d == "corn" for d in datasets)

    def test_get_summary_stats(self, sample_predictions):
        """Test getting summary statistics for a metric."""
        pred = sample_predictions

        stats = pred.get_summary_stats(metric="test_score")

        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "count" in stats

        # Verify values
        assert stats["min"] == 0.38
        assert stats["max"] == 0.61
        assert stats["count"] == 4
        assert 0.38 < stats["mean"] < 0.61

    def test_query_on_loaded_parquet(self, temp_workspace):
        """Test querying predictions loaded from Parquet."""
        catalog_dir = temp_workspace / "catalog"

        # Save predictions
        pred_save = Predictions()
        pred_save._df = pl.DataFrame({
            "dataset_name": ["wheat", "corn", "wheat"],
            "test_score": [0.45, 0.38, 0.52],
            "config_name": ["pls1", "pls1", "pls2"]
        })
        pred_save.save_to_parquet(catalog_dir, "pred_001")

        # Load and query
        pred_load = Predictions.load_from_parquet(catalog_dir)

        # Verify all rows loaded
        assert pred_load._df.height >= 3

        best = pred_load.query_best(metric="test_score", n=2)

        assert best.height == 2
        scores = best["test_score"].to_list()
        # Should include the best score (0.52) or at least high scores
        assert max(scores) >= 0.45

    def test_query_best_ascending(self, sample_predictions):
        """Test finding worst pipelines (ascending order)."""
        pred = sample_predictions

        # Get worst 2 by test_score
        worst = pred.query_best(metric="test_score", n=2, ascending=True)

        assert worst.height == 2
        scores = worst["test_score"].to_list()
        assert scores[0] <= scores[1]  # Ascending order
        assert scores[0] == 0.38  # Worst score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
