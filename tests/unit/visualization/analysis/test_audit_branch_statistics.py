"""Numerical and sampling regressions for VIZ-17/18/19, without rendering."""

from unittest.mock import Mock

import numpy as np
import pytest
from scipy import stats

from nirs4all.data.predictions import Predictions
from nirs4all.visualization.analysis.branch import BranchAnalyzer
from nirs4all.visualization.predictions import PredictionAnalyzer


def _analyzer(rows):
    predictions = Mock(spec=Predictions)
    predictions.num_predictions = len(rows)
    predictions.filter_predictions.side_effect = lambda **filters: [
        row for row in rows if all(row.get(key) == value for key, value in filters.items())
    ]
    return BranchAnalyzer(predictions)


def _rows(branch, values, route="score"):
    rows = []
    for index, value in enumerate(values):
        for partition in ("train", "val", "test"):
            row = {"branch_name": branch, "branch_id": ord(branch), "partition": partition,
                   "fold_id": f"fold_{index}", "metric": "rmse"}
            if route == "score":
                row["test_score"] = value
            elif route == "scores":
                row["scores"] = {"test": {"rmse": value}}
            else:
                row["y_true"] = np.zeros(3)
                row["y_pred"] = np.full(3, value)
            rows.append(row)
    for fold in ("avg", "w_avg", "fold_avg", "fold_w_avg", "custom_agg", "final", "fold_final"):
        rows.append({"branch_name": branch, "branch_id": ord(branch), "partition": "test",
                     "fold_id": fold, "metric": "rmse", "test_score": 100})
    rows.append({"branch_name": branch, "branch_id": ord(branch), "partition": "test",
                 "fold_id": "refit-uuid", "refit_context": "{}", "metric": "rmse", "test_score": 200})
    return rows


@pytest.mark.parametrize("route", ["score", "scores", "arrays"])
def test_branch_counts_means_and_tests_use_only_requested_partition_cv_folds(route):
    a = [1., 2., 3.]
    b = [2., 3., 5., 6.]
    analyzer = _analyzer(_rows("A", a, route) + _rows("B", b, route))
    summary = analyzer.summary(["rmse"]).to_dict()
    assert summary["A"]["count"] == 3 and summary["B"]["count"] == 4
    assert summary["A"]["rmse_mean"] == pytest.approx(2)
    assert summary["B"]["rmse_mean"] == pytest.approx(4)
    assert analyzer._collect_scores(_rows("A", a, route), "rmse", "test") == a
    ranked = analyzer.rank_branches("rmse")
    assert [row["branch_name"] for row in ranked] == ["A", "B"]
    assert [row["count"] for row in ranked] == [3, 4]
    comparison = analyzer.compare("A", "B")
    expected = stats.ttest_ind(a, b)
    assert comparison["n1"] == 3 and comparison["n2"] == 4
    assert comparison["statistic"] == pytest.approx(expected.statistic)
    assert comparison["p_value"] == pytest.approx(expected.pvalue)


def test_pooled_effect_size_matches_sample_variance_for_unequal_groups():
    analyzer = _analyzer(_rows("A", [1., 2., 3.]) + _rows("B", [2., 3., 5., 6.]))
    # Means 2 and 4; sums of squared deviations 2 and 10, pooled df 5.
    assert analyzer.compare("A", "B")["effect_size"] == pytest.approx(-2 / np.sqrt(12 / 5))


@pytest.mark.parametrize("fold,context", [("final", None), ("fold_final", None), ("uuid", "{}")])
def test_refit_only_population_is_not_duplicated_across_partitions(fold, context):
    rows = [{"branch_name": "A", "partition": partition, "fold_id": fold, "refit_context": context,
             "metric": "rmse", "test_score": 2.} for partition in ("train", "val", "test")]
    analyzer = _analyzer(rows)
    assert analyzer.summary(["rmse"])["A"]["count"] == 1
    assert analyzer._collect_scores(rows, "rmse", "test") == [2.]
    with pytest.raises(ValueError, match="Insufficient data"):
        analyzer.compare("A", "A")


def test_aggregate_partition_dictionary_is_one_unit_even_with_val_anchor():
    row = {"branch_name": "A", "partition": "val", "fold_id": "fold_0",
           "partitions": {"test": {"rmse": .25}, "val": {"rmse": .5}}}
    analyzer = _analyzer([row])
    analyzer.predictions.top.return_value = [row]
    summary = analyzer.summary(["rmse"], aggregate="sample_id")
    assert summary["A"]["count"] == 1
    assert summary["A"]["rmse_mean"] == .25
    assert analyzer._collect_scores([row], "rmse", "test") == [.25]
    assert analyzer._collect_scores([row], "rmse", "train") == []


def test_other_partition_and_ensemble_only_rows_do_not_make_samples():
    rows = [{"branch_name": "A", "partition": "val", "fold_id": "fold_0", "metric": "rmse", "test_score": 1},
            {"branch_name": "A", "partition": "test", "fold_id": "fold_avg", "metric": "rmse", "test_score": 2}]
    analyzer = _analyzer(rows)
    assert len(analyzer.summary(["rmse"])) == 0
    assert analyzer._collect_scores(rows, "rmse", "test") == []


@pytest.mark.parametrize("values,expected", [([1., 2., 3.], 1.), ([2.], np.nan), ([], np.nan)])
def test_prediction_branch_summary_uses_sample_std_for_t_confidence_intervals(values, expected):
    predictions = Mock(spec=Predictions)
    predictions.repetition_column = None
    analyzer = PredictionAnalyzer(predictions)
    rows = [{"branch_name": "A", "partitions": {"test": {"rmse": value}}} for value in values]
    analyzer.get_cached_predictions = Mock(return_value=rows)
    summary = analyzer.branch_summary(metrics=["rmse"], as_dataframe=False)
    if not values:
        assert summary == {}
    elif np.isnan(expected):
        assert np.isnan(summary["A"]["rmse_std"])
    else:
        assert summary["A"]["rmse_std"] == pytest.approx(expected)
        # The supplied standard deviation now produces the classical t CI.
        half_width = stats.t.ppf(.975, 2) * summary["A"]["rmse_std"] / np.sqrt(3)
        assert half_width == pytest.approx(2.4841377117)
