"""
Tests for the prediction ranking system.

This module tests the core ranking logic in PredictionRanker, including:
- Aggregation applied before ranking
- Group-by filtering (single and multi-column)
- Consistent results across charts
- Deprecated parameter handling
"""

import json
import warnings
import numpy as np
import pytest
from nirs4all.data.predictions import Predictions, _make_group_key


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
        "n_samples": 10,
        "n_features": 100,
        "task_type": "regression",
        "metric": "rmse",
    }


@pytest.fixture
def predictions_with_multiple_models(base_prediction_params):
    """Create predictions with multiple models and folds."""
    predictions = Predictions()

    # Create 5 different models, each with 2 folds and 3 partitions
    models = [
        ("PLS_3", "PLSRegression"),
        ("PLS_5", "PLSRegression"),
        ("PLS_10", "PLSRegression"),
        ("SVR_rbf", "SVR"),
        ("RF_100", "RandomForestRegressor"),
    ]

    for model_name, model_classname in models:
        for fold_id in range(2):
            for partition in ["train", "val", "test"]:
                # Create different scores for each combination
                # Score formula: model_idx * 0.1 + fold_id * 0.01 + partition_offset
                model_idx = models.index((model_name, model_classname))
                partition_offset = {"train": 0.001, "val": 0.002, "test": 0.003}[partition]
                score = 0.1 + model_idx * 0.05 + fold_id * 0.01 + partition_offset

                y_true = np.array([1.0, 2.0, 3.0, 4.0])
                y_pred = y_true + np.random.randn(4) * score

                predictions.add_prediction(
                    partition=partition,
                    y_true=y_true,
                    y_pred=y_pred,
                    train_score=score if partition == "train" else None,
                    val_score=score if partition == "val" else None,
                    test_score=score if partition == "test" else None,
                    model_name=model_name,
                    model_classname=model_classname,
                    fold_id=fold_id,
                    preprocessings="snv",
                    **{k: v for k, v in base_prediction_params.items() if k not in ["model_name", "model_classname", "fold_id"]}
                )

    return predictions


@pytest.fixture
def predictions_with_metadata_for_aggregation(base_prediction_params):
    """Create predictions with metadata for sample aggregation testing."""
    predictions = Predictions()

    # Create prediction with 4 scans per sample (12 samples = 3 unique IDs x 4 scans)
    y_true = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0])
    # Predictions vary per scan but should aggregate well
    y_pred_good = np.array([1.1, 0.9, 1.0, 1.0, 2.1, 1.9, 2.0, 2.0, 3.1, 2.9, 3.0, 3.0])
    y_pred_bad = np.array([1.5, 0.5, 1.2, 0.8, 2.5, 1.5, 2.2, 1.8, 3.5, 2.5, 3.2, 2.8])

    sample_ids = ["S1", "S1", "S1", "S1", "S2", "S2", "S2", "S2", "S3", "S3", "S3", "S3"]
    metadata = {"ID": sample_ids}

    # Calculate actual RMSE for the good and bad models
    from nirs4all.core import metrics as evaluator
    good_rmse = evaluator.eval(y_true, y_pred_good, "rmse")
    bad_rmse = evaluator.eval(y_true, y_pred_bad, "rmse")

    # Add two models with different performance
    for partition in ["train", "val", "test"]:
        # Good model - add explicit scores
        predictions.add_prediction(
            partition=partition,
            y_true=y_true,
            y_pred=y_pred_good,
            model_name="good_model",
            model_classname="PLSRegression",
            fold_id=0,
            metadata=metadata,
            train_score=good_rmse if partition == "train" else None,
            val_score=good_rmse if partition == "val" else None,
            test_score=good_rmse if partition == "test" else None,
            **{k: v for k, v in base_prediction_params.items() if k not in ["model_name", "model_classname", "fold_id"]}
        )

        # Bad model - add explicit scores
        predictions.add_prediction(
            partition=partition,
            y_true=y_true,
            y_pred=y_pred_bad,
            model_name="bad_model",
            model_classname="PLSRegression",
            fold_id=0,
            metadata=metadata,
            train_score=bad_rmse if partition == "train" else None,
            val_score=bad_rmse if partition == "val" else None,
            test_score=bad_rmse if partition == "test" else None,
            **{k: v for k, v in base_prediction_params.items() if k not in ["model_name", "model_classname", "fold_id"]}
        )

    return predictions


class TestRankingWithRepetition:
    """Test ranking with by_repetition parameter (new name for aggregate)."""

    def test_by_repetition_with_column_name(self, predictions_with_metadata_for_aggregation):
        """Verify by_repetition with explicit column name works."""
        predictions = predictions_with_metadata_for_aggregation

        # Should work without deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            results = predictions.top(
                n=2,
                rank_metric="rmse",
                rank_partition="val",
                by_repetition="ID"  # New parameter name
            )

            # No deprecation warning for new parameter
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0

        assert len(results) == 2
        assert results[0]["model_name"] == "good_model"
        assert results[0].get("aggregated", False) is True

    def test_by_repetition_true_with_context(self, predictions_with_metadata_for_aggregation):
        """Verify by_repetition=True uses context from set_repetition_column."""
        predictions = predictions_with_metadata_for_aggregation

        # Set context
        predictions.set_repetition_column("ID")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            results = predictions.top(
                n=2,
                rank_metric="rmse",
                rank_partition="val",
                by_repetition=True  # Should use "ID" from context
            )

            # No warnings - context was set
            all_warnings = [x for x in w]
            assert len(all_warnings) == 0

        assert len(results) == 2
        assert results[0].get("aggregated", False) is True

    def test_by_repetition_true_warns_without_context(self, predictions_with_metadata_for_aggregation):
        """Verify by_repetition=True warns when no context is set."""
        predictions = predictions_with_metadata_for_aggregation
        # Don't set context

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            results = predictions.top(
                n=2,
                rank_metric="rmse",
                rank_partition="val",
                by_repetition=True  # No context set
            )

            # Should warn about missing context
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) >= 1
            assert "by_repetition=True" in str(user_warnings[0].message)

        # Results returned without aggregation
        assert len(results) >= 1

    def test_repetition_column_property(self, predictions_with_metadata_for_aggregation):
        """Verify repetition_column property works."""
        predictions = predictions_with_metadata_for_aggregation

        # Initially None
        assert predictions.repetition_column is None

        # Set column
        predictions.set_repetition_column("ID")
        assert predictions.repetition_column == "ID"


class TestRankingWithAggregation:
    """Test ranking uses aggregated scores when aggregate is provided."""

    def test_top_with_aggregation_uses_aggregated_scores(self, predictions_with_metadata_for_aggregation):
        """Verify ranking uses aggregated scores when aggregate is provided."""
        predictions = predictions_with_metadata_for_aggregation

        # Get top 2 models with aggregation by sample ID
        results = predictions.top(
            n=2,
            rank_metric="rmse",
            rank_partition="val",
            aggregate="ID"
        )

        assert len(results) == 2
        # The "good_model" should rank first (lower RMSE after aggregation)
        assert results[0]["model_name"] == "good_model"
        assert results[1]["model_name"] == "bad_model"
        # Verify aggregation was applied
        assert results[0].get("aggregated", False) is True

    def test_aggregation_reduces_sample_count(self, predictions_with_metadata_for_aggregation):
        """Verify aggregation reduces the number of samples in y_pred."""
        predictions = predictions_with_metadata_for_aggregation

        # Get with aggregation
        results_agg = predictions.top(
            n=1,
            rank_metric="rmse",
            rank_partition="val",
            display_partition="val",  # Use val partition for display too
            aggregate="ID"
        )

        # Get without aggregation
        results_no_agg = predictions.top(
            n=1,
            rank_metric="rmse",
            rank_partition="val",
            display_partition="val"  # Use val partition for display too
        )

        assert len(results_agg) > 0, "Expected results with aggregation"
        assert len(results_no_agg) > 0, "Expected results without aggregation"
        # With aggregation: 12 samples -> 3 unique IDs
        assert len(results_agg[0]["y_pred"]) == 3
        # Without aggregation: 12 samples
        assert len(results_no_agg[0]["y_pred"]) == 12


class TestGroupByFiltering:
    """Test group_by filtering for ranking."""

    def test_group_by_single_column(self, predictions_with_multiple_models):
        """Verify group_by=['model_name'] with n=1 keeps one per model."""
        predictions = predictions_with_multiple_models

        results = predictions.top(
            n=1,  # n=1 means top 1 per group
            rank_metric="rmse",
            rank_partition="val",
            group_by=["model_name"]
        )

        # With n=1 and group_by, should have 1 per group = 5 unique models
        # (n=1 means top 1 per group)
        model_names = [r["model_name"] for r in results]
        assert len(set(model_names)) == len(model_names)  # All unique
        assert len(results) == 5

    def test_group_by_string_same_as_list(self, predictions_with_multiple_models):
        """Verify group_by='model_name' works same as group_by=['model_name']."""
        predictions = predictions_with_multiple_models

        results_str = predictions.top(
            n=1,  # n=1 per group
            rank_metric="rmse",
            rank_partition="val",
            group_by="model_name"
        )

        results_list = predictions.top(
            n=1,  # n=1 per group
            rank_metric="rmse",
            rank_partition="val",
            group_by=["model_name"]
        )

        assert len(results_str) == len(results_list)
        for r1, r2 in zip(results_str, results_list):
            assert r1["model_name"] == r2["model_name"]

    def test_group_by_multiple_columns(self, predictions_with_multiple_models):
        """Verify group_by with multiple columns works."""
        predictions = predictions_with_multiple_models

        # Group by (model_classname) - with n=1 should give 1 per class = 3 results
        # PLSRegression, SVR, RandomForestRegressor
        results = predictions.top(
            n=1,  # n=1 per group
            rank_metric="rmse",
            rank_partition="val",
            group_by=["model_classname"]
        )

        # Should have 3 unique model classes (1 per class)
        classnames = [r["model_classname"] for r in results]
        assert len(set(classnames)) == len(classnames)  # All unique
        assert len(results) == 3

        # Also test multi-column grouping with model_name + fold_id
        # 5 models x 2 folds = 10 combinations, with n=1 = 10 results
        results_multi = predictions.top(
            n=1,  # n=1 per group
            rank_metric="rmse",
            rank_partition="val",
            group_by=["model_name", "fold_id"]
        )
        assert len(results_multi) == 10  # 5 models x 2 folds

    def test_group_by_preserves_global_sort_order(self, predictions_with_multiple_models):
        """Verify group_by takes first (best) per group from global sort."""
        predictions = predictions_with_multiple_models

        # Get all results without grouping
        all_results = predictions.top(
            n=100,
            rank_metric="rmse",
            rank_partition="val",
            ascending=True  # Lower RMSE is better
        )

        # Get grouped results (n=1 means best 1 per group)
        grouped_results = predictions.top(
            n=1,  # n=1 per group to get best per model
            rank_metric="rmse",
            rank_partition="val",
            group_by=["model_name"],
            ascending=True
        )

        # Verify that for each model_name in grouped results,
        # it has the same rank_score as the first occurrence in all_results
        for grouped in grouped_results:
            model_name = grouped["model_name"]
            # Find first occurrence in all_results
            first_in_all = next(r for r in all_results if r["model_name"] == model_name)
            assert grouped["rank_score"] == first_in_all["rank_score"]

    def test_group_by_adds_group_key(self, predictions_with_multiple_models):
        """Verify group_by adds group_key field to each result."""
        predictions = predictions_with_multiple_models

        results = predictions.top(
            n=2,  # top 2 per group
            rank_metric="rmse",
            rank_partition="val",
            group_by=["model_name"]
        )

        # Each result should have group_key
        for result in results:
            assert "group_key" in result
            assert isinstance(result["group_key"], tuple)
            # group_key should contain model_name (lowercase due to case-insensitive)
            assert len(result["group_key"]) == 1

    def test_return_grouped_returns_dict(self, predictions_with_multiple_models):
        """Verify return_grouped=True returns dict of group -> results."""
        predictions = predictions_with_multiple_models

        grouped = predictions.top(
            n=2,  # top 2 per group
            rank_metric="rmse",
            rank_partition="val",
            group_by=["model_classname"],
            return_grouped=True
        )

        # Should be a dict
        assert isinstance(grouped, dict)

        # Should have 3 groups (PLS, SVR, RF)
        assert len(grouped) == 3

        # Each group should have up to 2 results
        for group_key, results in grouped.items():
            assert isinstance(group_key, tuple)
            assert len(results) <= 2
            # All results in this group should have matching group_key
            for r in results:
                assert r.get("group_key") == group_key

    def test_top_n_per_group_semantic(self, predictions_with_multiple_models):
        """Verify n means 'top N per group' when group_by is used."""
        predictions = predictions_with_multiple_models

        # Get top 2 per model_classname
        # 3 classes * 2 per class = 6 total
        results = predictions.top(
            n=2,
            rank_metric="rmse",
            rank_partition="val",
            group_by=["model_classname"]
        )

        # Count per group
        group_counts = {}
        for r in results:
            key = r["group_key"]
            group_counts[key] = group_counts.get(key, 0) + 1

        # Each group should have at most 2
        for key, count in group_counts.items():
            assert count <= 2


class TestDeprecatedParameters:
    """Test deprecated parameter handling."""

    def test_aggregate_emits_deprecation_warning(self, predictions_with_metadata_for_aggregation):
        """Verify aggregate parameter emits deprecation warning."""
        predictions = predictions_with_metadata_for_aggregation

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            predictions.top(
                n=2,
                rank_metric="rmse",
                rank_partition="val",
                aggregate="ID"  # Deprecated
            )

            # Check for deprecation warning
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "aggregate" in str(deprecation_warnings[0].message)
            assert "by_repetition" in str(deprecation_warnings[0].message)

    def test_best_per_model_emits_deprecation_warning(self, predictions_with_multiple_models):
        """Verify best_per_model emits deprecation warning."""
        predictions = predictions_with_multiple_models

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            predictions.top(
                n=5,
                rank_metric="rmse",
                rank_partition="val",
                best_per_model=True
            )

            # Check for deprecation warning
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "best_per_model" in str(deprecation_warnings[0].message)

    def test_best_per_model_same_as_group_by_model_name(self, predictions_with_multiple_models):
        """Verify best_per_model=True gives same result as group_by=['model_name'] with n=1."""
        predictions = predictions_with_multiple_models

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            # best_per_model=True should behave like group_by=['model_name'] with n=1
            results_deprecated = predictions.top(
                n=1,  # n=1 per group
                rank_metric="rmse",
                rank_partition="val",
                best_per_model=True
            )

        results_new = predictions.top(
            n=1,  # n=1 per group
            rank_metric="rmse",
            rank_partition="val",
            group_by=["model_name"]
        )

        assert len(results_deprecated) == len(results_new)
        for r1, r2 in zip(results_deprecated, results_new):
            assert r1["model_name"] == r2["model_name"]
            assert r1["rank_score"] == r2["rank_score"]

    def test_best_per_model_same_as_group_by_model_name(self, predictions_with_multiple_models):
        """Verify best_per_model=True gives same result as group_by=['model_name'] with n=1."""
        predictions = predictions_with_multiple_models

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            # best_per_model=True should behave like group_by=['model_name'] with n=1
            results_deprecated = predictions.top(
                n=1,  # n=1 per group
                rank_metric="rmse",
                rank_partition="val",
                best_per_model=True
            )

        results_new = predictions.top(
            n=1,  # n=1 per group
            rank_metric="rmse",
            rank_partition="val",
            group_by=["model_name"]
        )

        assert len(results_deprecated) == len(results_new)
        for r1, r2 in zip(results_deprecated, results_new):
            assert r1["model_name"] == r2["model_name"]
            assert r1["rank_score"] == r2["rank_score"]


class TestMakeGroupKey:
    """Test the _make_group_key helper function."""

    def test_case_insensitive_model_name(self):
        """Verify model_name is compared case-insensitively."""
        row1 = {"model_name": "PLS_5", "model_classname": "PLSRegression"}
        row2 = {"model_name": "pls_5", "model_classname": "PLSRegression"}

        key1 = _make_group_key(row1, ["model_name"])
        key2 = _make_group_key(row2, ["model_name"])

        assert key1 == key2

    def test_none_value_handling(self):
        """Verify None values are handled correctly in group keys."""
        row1 = {"model_name": "PLS_5", "preprocessings": None}
        row2 = {"model_name": "PLS_5", "preprocessings": None}
        row3 = {"model_name": "PLS_5", "preprocessings": "snv"}

        key1 = _make_group_key(row1, ["model_name", "preprocessings"])
        key2 = _make_group_key(row2, ["model_name", "preprocessings"])
        key3 = _make_group_key(row3, ["model_name", "preprocessings"])

        assert key1 == key2  # Both have None preprocessings
        assert key1 != key3  # Different preprocessings

    def test_numeric_column_handling(self):
        """Verify numeric columns are handled correctly."""
        row1 = {"model_name": "PLS_5", "fold_id": 0}
        row2 = {"model_name": "PLS_5", "fold_id": 0}
        row3 = {"model_name": "PLS_5", "fold_id": 1}

        key1 = _make_group_key(row1, ["model_name", "fold_id"])
        key2 = _make_group_key(row2, ["model_name", "fold_id"])
        key3 = _make_group_key(row3, ["model_name", "fold_id"])

        assert key1 == key2  # Same fold_id
        assert key1 != key3  # Different fold_id

    def test_list_to_tuple_conversion(self):
        """Verify list values are converted to tuples for hashability."""
        row = {"model_name": "PLS_5", "some_list": [1, 2, 3]}

        # Should not raise error (list is converted to tuple)
        key = _make_group_key(row, ["model_name", "some_list"])
        assert isinstance(key, tuple)


class TestConsistentResults:
    """Test consistent results across multiple calls."""

    def test_same_parameters_same_results(self, predictions_with_multiple_models):
        """Verify same parameters give same results across multiple calls."""
        predictions = predictions_with_multiple_models

        results1 = predictions.top(
            n=5,
            rank_metric="rmse",
            rank_partition="val",
            group_by=["model_name"]
        )

        results2 = predictions.top(
            n=5,
            rank_metric="rmse",
            rank_partition="val",
            group_by=["model_name"]
        )

        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1["model_name"] == r2["model_name"]
            assert r1["rank_score"] == r2["rank_score"]


class TestMetricDirection:
    """Test ascending/descending based on metric type."""

    def test_rmse_ascending_by_default(self, predictions_with_multiple_models):
        """Verify RMSE defaults to ascending (lower is better)."""
        predictions = predictions_with_multiple_models

        results = predictions.top(
            n=10,
            rank_metric="rmse",
            rank_partition="val"
        )

        # Scores should be in ascending order
        scores = [r["rank_score"] for r in results if r["rank_score"] is not None]
        assert scores == sorted(scores)

    def test_r2_descending_by_default(self, predictions_with_multiple_models):
        """Verify R2 defaults to descending (higher is better)."""
        predictions = predictions_with_multiple_models

        # Add predictions with R2 metric
        base_params = {
            "dataset_name": "test_dataset",
            "dataset_path": "/path/to/dataset",
            "config_name": "test_config",
            "config_path": "/path/to/config",
            "pipeline_uid": "pipe_r2",
            "step_idx": 0,
            "op_counter": 1,
            "n_samples": 10,
            "n_features": 100,
            "task_type": "regression",
            "metric": "r2",
        }

        predictions_r2 = Predictions()
        for i, score in enumerate([0.8, 0.9, 0.7, 0.95, 0.85]):
            predictions_r2.add_prediction(
                partition="val",
                y_true=np.array([1.0, 2.0, 3.0]),
                y_pred=np.array([1.0, 2.0, 3.0]) * (1 + (1 - score) * 0.1),
                val_score=score,
                model_name=f"model_{i}",
                model_classname="PLSRegression",
                fold_id=0,
                **{k: v for k, v in base_params.items() if k not in ["model_name", "model_classname", "fold_id"]}
            )

        results = predictions_r2.top(
            n=5,
            rank_metric="r2",
            rank_partition="val"
        )

        # Scores should be in descending order (higher R2 is better)
        scores = [r["rank_score"] for r in results if r["rank_score"] is not None]
        assert scores == sorted(scores, reverse=True)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_predictions(self):
        """Verify empty predictions returns empty list."""
        predictions = Predictions()
        results = predictions.top(n=5, rank_metric="rmse", rank_partition="val")
        assert len(results) == 0

    def test_n_larger_than_available(self, predictions_with_multiple_models):
        """Verify n larger than available per group returns all available per group."""
        predictions = predictions_with_multiple_models

        # With group_by, n=1000 means up to 1000 per group
        # Each model has 2 folds, so 2 per model = 10 total
        results = predictions.top(
            n=1000,
            rank_metric="rmse",
            rank_partition="val",
            group_by=["model_name"]
        )

        # Should return all predictions per model (2 folds each), so 5 models * 2 = 10
        assert len(results) == 10

    def test_group_by_nonexistent_column(self, predictions_with_multiple_models):
        """Verify group_by with non-existent column still works (groups by None)."""
        predictions = predictions_with_multiple_models

        # Should not raise error - all rows will have None for this column
        # All go into one group (None), so n=10 returns top 10 from that group
        results = predictions.top(
            n=10,
            rank_metric="rmse",
            rank_partition="val",
            group_by=["nonexistent_column"]
        )

        # All predictions have None for nonexistent_column, forming one group
        # n=10 means top 10 from that group
        assert len(results) == 10

    def test_aggregation_column_not_in_metadata(self, predictions_with_multiple_models):
        """Verify aggregation with missing column warns and falls back."""
        predictions = predictions_with_multiple_models

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            results = predictions.top(
                n=5,
                rank_metric="rmse",
                rank_partition="val",
                aggregate="nonexistent_id"
            )

            # Should have warnings about missing column
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) >= 1

        # Results should still be returned (without aggregation)
        assert len(results) >= 1
