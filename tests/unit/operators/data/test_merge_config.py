"""Tests for merge operator configuration and parsing.

This module tests:
- MergeMode, SelectionStrategy, AggregationStrategy, ShapeMismatchStrategy enums
- BranchPredictionConfig dataclass validation
- MergeConfig dataclass validation
- MergeConfigParser for all syntax variants
"""

import warnings

import pytest

from nirs4all.controllers.data.merge import MergeConfigParser
from nirs4all.operators.data.merge import (
    AggregationStrategy,
    BranchPredictionConfig,
    MergeConfig,
    MergeMode,
    SelectionStrategy,
    ShapeMismatchStrategy,
)


class TestMergeMode:
    """Test suite for MergeMode enum."""

    def test_features_mode(self):
        """Test FEATURES mode value."""
        assert MergeMode.FEATURES.value == "features"

    def test_predictions_mode(self):
        """Test PREDICTIONS mode value."""
        assert MergeMode.PREDICTIONS.value == "predictions"

    def test_all_mode(self):
        """Test ALL mode value."""
        assert MergeMode.ALL.value == "all"

    def test_from_string(self):
        """Test creating MergeMode from string value."""
        assert MergeMode("features") == MergeMode.FEATURES
        assert MergeMode("predictions") == MergeMode.PREDICTIONS
        assert MergeMode("all") == MergeMode.ALL

class TestSelectionStrategy:
    """Test suite for SelectionStrategy enum."""

    def test_all_strategy(self):
        """Test ALL strategy value."""
        assert SelectionStrategy.ALL.value == "all"

    def test_best_strategy(self):
        """Test BEST strategy value."""
        assert SelectionStrategy.BEST.value == "best"

    def test_top_k_strategy(self):
        """Test TOP_K strategy value."""
        assert SelectionStrategy.TOP_K.value == "top_k"

    def test_explicit_strategy(self):
        """Test EXPLICIT strategy value."""
        assert SelectionStrategy.EXPLICIT.value == "explicit"

class TestAggregationStrategy:
    """Test suite for AggregationStrategy enum."""

    def test_separate_strategy(self):
        """Test SEPARATE strategy value."""
        assert AggregationStrategy.SEPARATE.value == "separate"

    def test_mean_strategy(self):
        """Test MEAN strategy value."""
        assert AggregationStrategy.MEAN.value == "mean"

    def test_weighted_mean_strategy(self):
        """Test WEIGHTED_MEAN strategy value."""
        assert AggregationStrategy.WEIGHTED_MEAN.value == "weighted_mean"

    def test_proba_mean_strategy(self):
        """Test PROBA_MEAN strategy value."""
        assert AggregationStrategy.PROBA_MEAN.value == "proba_mean"

class TestShapeMismatchStrategy:
    """Test suite for ShapeMismatchStrategy enum."""

    def test_error_strategy(self):
        """Test ERROR strategy value."""
        assert ShapeMismatchStrategy.ERROR.value == "error"

    def test_allow_strategy(self):
        """Test ALLOW strategy value."""
        assert ShapeMismatchStrategy.ALLOW.value == "allow"

    def test_pad_strategy(self):
        """Test PAD strategy value."""
        assert ShapeMismatchStrategy.PAD.value == "pad"

    def test_truncate_strategy(self):
        """Test TRUNCATE strategy value."""
        assert ShapeMismatchStrategy.TRUNCATE.value == "truncate"

class TestBranchPredictionConfig:
    """Test suite for BranchPredictionConfig dataclass."""

    def test_minimal_init(self):
        """Test initialization with only required branch parameter."""
        config = BranchPredictionConfig(branch=0)

        assert config.branch == 0
        assert config.select == "all"
        assert config.metric is None
        assert config.aggregate == "separate"
        assert config.weight_metric is None
        assert config.proba is False
        assert config.sources == "all"

    def test_branch_by_name(self):
        """Test initialization with branch name."""
        config = BranchPredictionConfig(branch="spectral_path")

        assert config.branch == "spectral_path"

    def test_select_all(self):
        """Test select='all' is valid."""
        config = BranchPredictionConfig(branch=0, select="all")
        assert config.get_selection_strategy() == SelectionStrategy.ALL

    def test_select_best(self):
        """Test select='best' is valid."""
        config = BranchPredictionConfig(branch=0, select="best", metric="rmse")
        assert config.get_selection_strategy() == SelectionStrategy.BEST

    def test_select_top_k(self):
        """Test select={'top_k': N} is valid."""
        config = BranchPredictionConfig(branch=0, select={"top_k": 3})
        assert config.get_selection_strategy() == SelectionStrategy.TOP_K

    def test_select_explicit_list(self):
        """Test select=['Model1', 'Model2'] is valid."""
        config = BranchPredictionConfig(
            branch=0, select=["PLS", "RandomForest"]
        )
        assert config.get_selection_strategy() == SelectionStrategy.EXPLICIT

    def test_invalid_select_string(self):
        """Test invalid select string raises ValueError."""
        with pytest.raises(ValueError, match="string select must be 'all' or 'best'"):
            BranchPredictionConfig(branch=0, select="invalid")

    def test_invalid_select_dict_missing_top_k(self):
        """Test dict select without top_k raises ValueError."""
        with pytest.raises(ValueError, match="dict select must contain 'top_k' key"):
            BranchPredictionConfig(branch=0, select={"invalid": 3})

    def test_invalid_select_top_k_not_int(self):
        """Test top_k with non-integer raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            BranchPredictionConfig(branch=0, select={"top_k": "three"})

    def test_invalid_select_top_k_negative(self):
        """Test top_k with negative value raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            BranchPredictionConfig(branch=0, select={"top_k": -1})

    def test_invalid_select_list_non_strings(self):
        """Test list select with non-strings raises ValueError."""
        with pytest.raises(ValueError, match="list select must contain only string"):
            BranchPredictionConfig(branch=0, select=[1, 2, 3])

    def test_invalid_select_empty_list(self):
        """Test empty list select raises ValueError."""
        with pytest.raises(ValueError, match="list select cannot be empty"):
            BranchPredictionConfig(branch=0, select=[])

    def test_valid_aggregates(self):
        """Test all valid aggregate values."""
        for agg in ["separate", "mean", "weighted_mean", "proba_mean"]:
            config = BranchPredictionConfig(branch=0, aggregate=agg)
            assert config.aggregate == agg

    def test_invalid_aggregate(self):
        """Test invalid aggregate raises ValueError."""
        with pytest.raises(ValueError, match="aggregate must be one of"):
            BranchPredictionConfig(branch=0, aggregate="invalid")

    def test_valid_metrics(self):
        """Test all valid metric values."""
        valid_metrics = ["rmse", "mae", "r2", "mse", "accuracy", "f1", "auc", "log_loss"]
        for metric in valid_metrics:
            config = BranchPredictionConfig(branch=0, metric=metric)
            assert config.metric == metric

    def test_invalid_metric(self):
        """Test invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="metric must be one of"):
            BranchPredictionConfig(branch=0, metric="invalid_metric")

    def test_proba_mean_sets_proba_true(self):
        """Test aggregate='proba_mean' auto-sets proba=True with warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = BranchPredictionConfig(
                branch=0, aggregate="proba_mean", proba=False
            )
            assert config.proba is True
            assert len(w) == 1
            assert "proba=True" in str(w[0].message)

    def test_get_aggregation_strategy(self):
        """Test get_aggregation_strategy returns correct enum."""
        config = BranchPredictionConfig(branch=0, aggregate="weighted_mean")
        assert config.get_aggregation_strategy() == AggregationStrategy.WEIGHTED_MEAN

class TestMergeConfig:
    """Test suite for MergeConfig dataclass."""

    def test_default_init(self):
        """Test initialization with defaults."""
        config = MergeConfig()

        assert config.collect_features is False
        assert config.feature_branches == "all"
        assert config.collect_predictions is False
        assert config.prediction_branches == "all"
        assert config.prediction_configs is None
        assert config.model_filter is None
        assert config.use_proba is False
        assert config.include_original is False
        assert config.on_missing == "error"
        assert config.on_shape_mismatch == "error"
        assert config.unsafe is False
        assert config.output_as == "features"
        assert config.source_names is None

    def test_feature_collection(self):
        """Test feature collection configuration."""
        config = MergeConfig(
            collect_features=True, feature_branches=[0, 2]
        )
        assert config.collect_features is True
        assert config.feature_branches == [0, 2]
        assert config.get_merge_mode() == MergeMode.FEATURES

    def test_prediction_collection(self):
        """Test prediction collection configuration."""
        config = MergeConfig(
            collect_predictions=True, prediction_branches=[1]
        )
        assert config.collect_predictions is True
        assert config.prediction_branches == [1]
        assert config.get_merge_mode() == MergeMode.PREDICTIONS

    def test_all_collection(self):
        """Test collecting both features and predictions."""
        config = MergeConfig(
            collect_features=True, collect_predictions=True
        )
        assert config.get_merge_mode() == MergeMode.ALL

    def test_invalid_on_missing(self):
        """Test invalid on_missing raises ValueError."""
        with pytest.raises(ValueError, match="on_missing must be one of"):
            MergeConfig(on_missing="invalid")

    def test_valid_on_missing(self):
        """Test all valid on_missing values."""
        for value in ["error", "warn", "skip"]:
            config = MergeConfig(on_missing=value)
            assert config.on_missing == value

    def test_invalid_on_shape_mismatch(self):
        """Test invalid on_shape_mismatch raises ValueError."""
        with pytest.raises(ValueError, match="on_shape_mismatch must be one of"):
            MergeConfig(on_shape_mismatch="invalid")

    def test_valid_on_shape_mismatch(self):
        """Test all valid on_shape_mismatch values."""
        for value in ["error", "allow", "pad", "truncate"]:
            config = MergeConfig(on_shape_mismatch=value)
            assert config.on_shape_mismatch == value

    def test_invalid_output_as(self):
        """Test invalid output_as raises ValueError."""
        with pytest.raises(ValueError, match="output_as must be one of"):
            MergeConfig(output_as="invalid")

    def test_valid_output_as(self):
        """Test all valid output_as values."""
        for value in ["features", "sources", "dict"]:
            config = MergeConfig(output_as=value)
            assert config.output_as == value

    def test_unsafe_warning(self):
        """Test unsafe=True with predictions emits warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = MergeConfig(collect_predictions=True, unsafe=True)
            assert config.unsafe is True
            assert len(w) == 1
            assert "DATA LEAKAGE" in str(w[0].message)

    def test_source_names_warning_without_sources(self):
        """Test source_names with wrong output_as emits warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = MergeConfig(
                source_names=["a", "b"], output_as="features"
            )
            assert len(w) == 1
            assert "source_names is only used" in str(w[0].message)

    def test_has_per_branch_config_false(self):
        """Test has_per_branch_config returns False when not set."""
        config = MergeConfig()
        assert config.has_per_branch_config() is False

    def test_has_per_branch_config_true(self):
        """Test has_per_branch_config returns True when set."""
        config = MergeConfig(
            prediction_configs=[BranchPredictionConfig(branch=0)]
        )
        assert config.has_per_branch_config() is True

    def test_has_per_branch_config_empty_list(self):
        """Test has_per_branch_config returns False for empty list."""
        config = MergeConfig(prediction_configs=[])
        assert config.has_per_branch_config() is False

    def test_get_feature_branches_all(self):
        """Test get_feature_branches with 'all'."""
        config = MergeConfig(
            collect_features=True, feature_branches="all"
        )
        assert config.get_feature_branches(3) == [0, 1, 2]

    def test_get_feature_branches_specific(self):
        """Test get_feature_branches with specific indices."""
        config = MergeConfig(
            collect_features=True, feature_branches=[0, 2]
        )
        assert config.get_feature_branches(5) == [0, 2]

    def test_get_prediction_configs_per_branch(self):
        """Test get_prediction_configs with per-branch configs."""
        pred_configs = [
            BranchPredictionConfig(branch=0, select="best"),
            BranchPredictionConfig(branch=1, aggregate="mean"),
        ]
        config = MergeConfig(
            collect_predictions=True, prediction_configs=pred_configs
        )
        result = config.get_prediction_configs(n_branches=2)
        assert result == pred_configs

    def test_get_prediction_configs_legacy_all(self):
        """Test get_prediction_configs with legacy 'all' format."""
        config = MergeConfig(
            collect_predictions=True, prediction_branches="all"
        )
        result = config.get_prediction_configs(n_branches=3)

        assert len(result) == 3
        assert all(isinstance(r, BranchPredictionConfig) for r in result)
        assert [r.branch for r in result] == [0, 1, 2]

    def test_get_prediction_configs_legacy_specific(self):
        """Test get_prediction_configs with legacy specific indices."""
        config = MergeConfig(
            collect_predictions=True, prediction_branches=[0, 2]
        )
        result = config.get_prediction_configs(n_branches=3)

        assert len(result) == 2
        assert [r.branch for r in result] == [0, 2]

    def test_get_prediction_configs_with_model_filter(self):
        """Test get_prediction_configs with model_filter."""
        config = MergeConfig(
            collect_predictions=True,
            prediction_branches="all",
            model_filter=["PLS", "RF"]
        )
        result = config.get_prediction_configs(n_branches=2)

        assert len(result) == 2
        assert all(r.select == ["PLS", "RF"] for r in result)

    def test_get_merge_mode_error_when_nothing_collected(self):
        """Test get_merge_mode raises error when nothing is collected."""
        config = MergeConfig()
        with pytest.raises(ValueError, match="neither collect_features nor"):
            config.get_merge_mode()

    def test_get_shape_mismatch_strategy(self):
        """Test get_shape_mismatch_strategy returns correct enum."""
        config = MergeConfig(on_shape_mismatch="pad")
        assert config.get_shape_mismatch_strategy() == ShapeMismatchStrategy.PAD

class TestMergeConfigParser:
    """Test suite for MergeConfigParser."""

    class TestSimpleStringParsing:
        """Test parsing simple string formats."""

        def test_parse_features(self):
            """Test parsing 'features' string."""
            config = MergeConfigParser.parse("features")

            assert config.collect_features is True
            assert config.collect_predictions is False
            assert config.feature_branches == "all"

        def test_parse_predictions(self):
            """Test parsing 'predictions' string."""
            config = MergeConfigParser.parse("predictions")

            assert config.collect_predictions is True
            assert config.collect_features is False
            assert config.prediction_branches == "all"

        def test_parse_all(self):
            """Test parsing 'all' string."""
            config = MergeConfigParser.parse("all")

            assert config.collect_features is True
            assert config.collect_predictions is True

        def test_parse_invalid_string(self):
            """Test parsing invalid string raises ValueError."""
            with pytest.raises(ValueError, match="Unknown merge mode"):
                MergeConfigParser.parse("invalid")

    class TestDictParsing:
        """Test parsing dictionary formats."""

        def test_parse_features_all(self):
            """Test parsing {'features': 'all'}."""
            config = MergeConfigParser.parse({"features": "all"})

            assert config.collect_features is True
            assert config.feature_branches == "all"

        def test_parse_features_list(self):
            """Test parsing {'features': [0, 2]}."""
            config = MergeConfigParser.parse({"features": [0, 2]})

            assert config.collect_features is True
            assert config.feature_branches == [0, 2]

        def test_parse_features_dict(self):
            """Test parsing {'features': {'branches': [1, 2]}}."""
            config = MergeConfigParser.parse(
                {"features": {"branches": [1, 2]}}
            )

            assert config.collect_features is True
            assert config.feature_branches == [1, 2]

        def test_parse_predictions_all(self):
            """Test parsing {'predictions': 'all'}."""
            config = MergeConfigParser.parse({"predictions": "all"})

            assert config.collect_predictions is True
            assert config.prediction_branches == "all"

        def test_parse_predictions_list_indices(self):
            """Test parsing {'predictions': [0, 1]} (legacy format)."""
            config = MergeConfigParser.parse({"predictions": [0, 1]})

            assert config.collect_predictions is True
            assert config.prediction_branches == [0, 1]

        def test_parse_predictions_dict(self):
            """Test parsing {'predictions': {'branches': [0], 'models': ['PLS']}}."""
            config = MergeConfigParser.parse({
                "predictions": {"branches": [0], "models": ["PLS"], "proba": True}
            })

            assert config.collect_predictions is True
            assert config.prediction_branches == [0]
            assert config.model_filter == ["PLS"]
            assert config.use_proba is True

        def test_parse_predictions_per_branch(self):
            """Test parsing per-branch prediction configs."""
            config = MergeConfigParser.parse({
                "predictions": [
                    {"branch": 0, "select": "best", "metric": "rmse"},
                    {"branch": 1, "aggregate": "mean"}
                ]
            })

            assert config.collect_predictions is True
            assert config.has_per_branch_config()
            assert len(config.prediction_configs) == 2
            assert config.prediction_configs[0].branch == 0
            assert config.prediction_configs[0].select == "best"
            assert config.prediction_configs[1].aggregate == "mean"

        def test_parse_mixed(self):
            """Test parsing mixed features and predictions."""
            config = MergeConfigParser.parse({
                "features": [1],
                "predictions": [0]
            })

            assert config.collect_features is True
            assert config.feature_branches == [1]
            assert config.collect_predictions is True
            assert config.prediction_branches == [0]

        def test_parse_global_options(self):
            """Test parsing global options."""
            config = MergeConfigParser.parse({
                "features": "all",
                "include_original": True,
                "on_missing": "warn",
                "on_shape_mismatch": "allow",
                "output_as": "sources"
            })

            assert config.include_original is True
            assert config.on_missing == "warn"
            assert config.on_shape_mismatch == "allow"
            assert config.output_as == "sources"

        def test_parse_unsafe(self):
            """Test parsing unsafe option."""
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                config = MergeConfigParser.parse({
                    "predictions": "all",
                    "unsafe": True
                })

            assert config.unsafe is True

        def test_parse_empty_dict_raises(self):
            """Test parsing empty dict raises ValueError."""
            with pytest.raises(ValueError, match="at least one of"):
                MergeConfigParser.parse({})

        def test_parse_no_features_or_predictions_raises(self):
            """Test parsing dict without features or predictions raises."""
            with pytest.raises(ValueError, match="at least one of"):
                MergeConfigParser.parse({"include_original": True})

    class TestPerBranchParsing:
        """Test parsing per-branch prediction configurations."""

        def test_parse_branch_config_minimal(self):
            """Test parsing minimal branch config."""
            config = MergeConfigParser.parse({
                "predictions": [{"branch": 0}]
            })

            assert len(config.prediction_configs) == 1
            pc = config.prediction_configs[0]
            assert pc.branch == 0
            assert pc.select == "all"
            assert pc.aggregate == "separate"

        def test_parse_branch_config_full(self):
            """Test parsing fully specified branch config."""
            config = MergeConfigParser.parse({
                "predictions": [{
                    "branch": "spectral_path",
                    "select": {"top_k": 2},
                    "metric": "r2",
                    "aggregate": "weighted_mean",
                    "weight_metric": "r2",
                    "proba": False,
                    "sources": ["NIR"]
                }]
            })

            pc = config.prediction_configs[0]
            assert pc.branch == "spectral_path"
            assert pc.select == {"top_k": 2}
            assert pc.metric == "r2"
            assert pc.aggregate == "weighted_mean"
            assert pc.weight_metric == "r2"
            assert pc.proba is False
            assert pc.sources == ["NIR"]

        def test_parse_branch_config_missing_branch_raises(self):
            """Test parsing branch config without 'branch' key raises."""
            with pytest.raises(ValueError, match="must have 'branch' key"):
                MergeConfigParser.parse({
                    "predictions": [{"select": "best"}]
                })

        def test_parse_multiple_branch_configs(self):
            """Test parsing multiple branch configurations."""
            config = MergeConfigParser.parse({
                "predictions": [
                    {"branch": 0, "select": "best", "metric": "rmse"},
                    {"branch": 1, "select": ["PLS", "RF"], "aggregate": "separate"},
                    {"branch": 2, "aggregate": "mean"}
                ]
            })

            assert len(config.prediction_configs) == 3

    class TestEdgeCases:
        """Test edge cases and error handling."""

        def test_parse_merge_config_passthrough(self):
            """Test parsing MergeConfig instance returns itself."""
            original = MergeConfig(collect_features=True)
            result = MergeConfigParser.parse(original)
            assert result is original

        def test_parse_invalid_type_raises(self):
            """Test parsing invalid type raises ValueError."""
            with pytest.raises(ValueError, match="Invalid merge config type"):
                MergeConfigParser.parse(123)

        def test_parse_features_true(self):
            """Test {'features': True} is equivalent to 'all'."""
            config = MergeConfigParser.parse({"features": True})
            assert config.feature_branches == "all"

        def test_parse_predictions_true(self):
            """Test {'predictions': True} is equivalent to 'all'."""
            config = MergeConfigParser.parse({"predictions": True})
            assert config.prediction_branches == "all"

        def test_parse_non_integer_branch_indices_raises(self):
            """Test non-integer branch indices raise ValueError."""
            with pytest.raises(ValueError, match="Branch indices must be integers"):
                MergeConfigParser.parse({"features": ["a", "b"]})

        def test_parse_empty_predictions_list_raises(self):
            """Test empty predictions list raises ValueError."""
            with pytest.raises(ValueError, match="cannot be empty"):
                MergeConfigParser.parse({"predictions": []})

class TestMergeConfigParserIntegration:
    """Integration tests for complex parsing scenarios."""

    def test_complex_asymmetric_merge(self):
        """Test parsing complex asymmetric merge configuration."""
        config = MergeConfigParser.parse({
            "predictions": [
                {"branch": 0, "select": "best", "metric": "rmse"},
                {"branch": 1, "select": {"top_k": 2}, "aggregate": "mean"}
            ],
            "features": [2],
            "include_original": True,
            "output_as": "features"
        })

        assert config.collect_predictions is True
        assert config.collect_features is True
        assert len(config.prediction_configs) == 2
        assert config.feature_branches == [2]
        assert config.include_original is True
        assert config.get_merge_mode() == MergeMode.ALL

    def test_stacking_like_config(self):
        """Test parsing MetaModel-like stacking configuration."""
        config = MergeConfigParser.parse({
            "predictions": {
                "branches": "all",
                "models": ["PLS", "RF", "XGB"],
                "proba": False
            }
        })

        assert config.collect_predictions is True
        assert config.prediction_branches == "all"
        assert config.model_filter == ["PLS", "RF", "XGB"]
        assert config.use_proba is False

    def test_feature_concat_like_config(self):
        """Test parsing feature concatenation configuration."""
        config = MergeConfigParser.parse({
            "features": "all",
            "on_shape_mismatch": "allow"
        })

        assert config.collect_features is True
        assert config.feature_branches == "all"
        assert config.get_shape_mismatch_strategy() == ShapeMismatchStrategy.ALLOW
