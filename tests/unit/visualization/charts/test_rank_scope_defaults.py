"""Chart ranking follows the requested CV metric without rebinding refit scores."""

import numpy as np
import pytest

from nirs4all.data.predictions import Predictions
from nirs4all.visualization.charts.top_k_comparison import TopKComparisonChart
from nirs4all.visualization.predictions import PredictionAnalyzer


@pytest.fixture
def predictions():
    rows = Predictions()
    for name, validation, test in [("validation winner", .9, .1), ("test winner", .2, .8)]:
        for partition in ("val", "test"):
            rows.add_prediction(
                dataset_name="data", model_name=name, model_classname="Ridge", config_name=name,
                fold_id=0, partition=partition, metric="rmse", task_type="regression",
                val_score=1 - validation, test_score=1 - test,
                scores={"val": {"r2": validation, "rmse": 1 - validation},
                        "test": {"r2": test, "rmse": 1 - test}},
                y_true=np.arange(4.), y_pred=np.arange(4.) + .2, sample_indices=np.arange(4),
            )
    rows.add_prediction(
        dataset_name="data", model_name="refit winner", model_classname="Ridge", config_name="refit winner",
        fold_id="final", partition="test", metric="rmse", task_type="regression", val_score=.01,
        test_score=100., scores={"test": {"r2": -100., "rmse": 100.}},
        y_true=np.arange(4.), y_pred=np.arange(4.) + 100., sample_indices=np.arange(4),
    )
    return rows


def query(predictions, cached, **kwargs):
    analyzer = PredictionAnalyzer(predictions) if cached else None
    return TopKComparisonChart(predictions, analyzer=analyzer)._get_ranked_predictions(n=1, **kwargs)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("partition,winner", [("val", "validation winner"), ("test", "test winner")])
def test_default_chart_query_ranks_cv_evidence_in_requested_metric(predictions, cached, partition, winner):
    result = query(predictions, cached, rank_metric="r2", rank_partition=partition)
    assert result[0]["model_name"] == winner
    assert result[0]["rank_score"] == (.9 if partition == "val" else .8)


@pytest.mark.parametrize("cached", [False, True])
def test_explicit_refit_scope_keeps_selection_metric_and_ignores_test_ranking(predictions, cached):
    result = query(predictions, cached, rank_metric="rmse", rank_partition="test", score_scope="refit")
    assert result[0]["model_name"] == "refit winner"
    assert result[0]["rank_score"] == .01


@pytest.mark.parametrize("cached", [False, True])
def test_explicit_refit_scope_still_rejects_another_metric(predictions, cached):
    query(predictions, cached, rank_metric="r2", score_scope="cv")
    with pytest.raises(ValueError, match="selection evidence uses 'rmse'"):
        query(predictions, cached, rank_metric="r2", score_scope="refit")


def test_cache_does_not_mix_cv_and_refit_selection(predictions):
    analyzer = PredictionAnalyzer(predictions)
    cv = analyzer.get_cached_predictions(n=1, rank_metric="rmse")
    refit = analyzer.get_cached_predictions(n=1, rank_metric="rmse", score_scope="refit")
    assert cv[0]["model_name"] == "validation winner"
    assert refit[0]["model_name"] == "refit winner"
    assert analyzer.get_cached_predictions(n=1, rank_metric="rmse")[0]["rank_score"] == cv[0]["rank_score"]


@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize("partition,winner", [("val", "validation winner"), ("test", "test winner")])
def test_top_k_entry_points_use_cv_ranking_without_drawing(predictions, monkeypatch, public, partition, winner):
    # Exercise the real public/direct render selection path, then capture its
    # structured rows instead of drawing a chart.
    captured = []
    original = TopKComparisonChart._get_ranked_predictions
    sentinel = object()

    def capture(self, **kwargs):
        captured.extend(original(self, **kwargs))
        return []

    monkeypatch.setattr(TopKComparisonChart, "_get_ranked_predictions", capture)
    monkeypatch.setattr(TopKComparisonChart, "_create_empty_figure", lambda *args: sentinel)
    render = PredictionAnalyzer(predictions).plot_top_k if public else TopKComparisonChart(predictions).render
    assert render(k=1, rank_metric="r2", rank_partition=partition) is sentinel
    assert captured[0]["model_name"] == winner
    assert all(row["model_name"] != "refit winner" for row in captured)
