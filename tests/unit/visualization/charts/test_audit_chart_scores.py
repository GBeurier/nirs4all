"""Regression evidence for VIZ-01, VIZ-02 and VIZ-03 chart data contracts."""

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from nirs4all.data.predictions import Predictions
from nirs4all.visualization.charts.candlestick import CandlestickChart
from nirs4all.visualization.charts.confusion_matrix import ConfusionMatrixChart
from nirs4all.visualization.charts.heatmap import HeatmapChart
from nirs4all.visualization.charts.histogram import ScoreHistogramChart
from nirs4all.visualization.charts.top_k_comparison import TopKComparisonChart
from nirs4all.visualization.predictions import PredictionAnalyzer


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def _add(predictions, dataset, model, *, error=0.2, secondary=None, classification=False, test_score=None, fold_id=0, branch_name=None):
    targets = np.array([0., 0., 1., 1.])
    metric = 'accuracy' if classification else 'rmse'
    scores = {'val': {metric: 1 - error if classification else error}, 'test': {metric: error}}
    if test_score is not None:
        scores['test'][metric] = test_score
    if secondary:
        scores['test'].update(secondary)
    for partition in ('val', 'test'):
        predictions.add_prediction(
            dataset_name=dataset, model_name=model, model_classname=model,
            config_name=model, fold_id=fold_id, partition=partition, metric=metric, branch_name=branch_name,
            val_score=scores['val'][metric], test_score=scores['test'][metric], scores=scores,
            task_type='binary_classification' if classification else 'regression',
            y_true=targets, y_pred=targets if classification else targets + error,
            sample_indices=np.arange(4),
        )


@pytest.mark.parametrize('classification', [False, True])
@pytest.mark.parametrize('cached', [False, True])
def test_top_k_charts_keep_validation_winner_when_test_order_disagrees(classification, cached):
    predictions = Predictions()
    # Display scores deliberately oppose validation ranking.
    _add(predictions, 'data', 'CV winner', error=0.1, classification=classification, test_score=0.1 if classification else 0.9)
    _add(predictions, 'data', 'test winner', error=0.2, classification=classification, test_score=0.99 if classification else 0.01)
    analyzer = PredictionAnalyzer(predictions) if cached else None
    chart_type = ConfusionMatrixChart if classification else TopKComparisonChart
    chart = chart_type(predictions, analyzer=analyzer)
    fig = chart.render(k=1, rank_metric='accuracy' if classification else 'rmse', rank_partition='val', score_scope='cv')
    titles = '\n'.join(ax.get_title() for ax in fig.axes)
    assert 'CV winner' in titles
    assert 'test winner' not in titles


@pytest.mark.parametrize('axes', [('dataset_name', 'model_name'), ('partition', 'model_name'), ('model_name', 'partition')])
def test_aggregated_heatmap_fills_every_cell_in_both_partition_orientations(monkeypatch, axes):
    predictions = Predictions()
    for dataset in ('A', 'B'):
        for model, error in [('Ridge', 0.2), ('PLS', 0.4)]:
            _add(predictions, dataset, model, error=error)
    chart = HeatmapChart(predictions)
    captured = {}

    def capture(matrix, normalized, counts, x_labels, y_labels, *args):
        captured.update(matrix=matrix, counts=counts, x=x_labels, y=y_labels)
        return Figure()

    monkeypatch.setattr(chart, '_render_heatmap_aggregated', capture)
    chart.render(*axes, aggregate='y', rank_metric='rmse', score_scope='cv')
    assert captured['matrix'].shape == (2, 2)
    assert np.isfinite(captured['matrix']).all()
    # Two grouped target values per dataset; partition-only axes combine datasets.
    assert (captured['counts'] == (2 if axes[0] == 'dataset_name' else 4)).all()
    assert set(captured['x']) == ({'A', 'B'} if axes[0] == 'dataset_name' else {'val', 'test'} if axes[0] == 'partition' else {'ridge', 'pls'})


@pytest.mark.parametrize('metric,values', [('r2', [-0.2, -3.5]), ('mse', [1.5e-6, 3e3])])
@pytest.mark.parametrize('kind', ['heatmap', 'candlestick', 'histogram'])
def test_secondary_json_metrics_keep_negative_and_exponent_values(monkeypatch, metric, values, kind):
    predictions = Predictions()
    for index, value in enumerate(values):
        _add(predictions, 'data', f'model{index}', secondary={metric: value, 'undefined': float('nan'), 'infinite': float('inf')})
    if kind == 'heatmap':
        chart = HeatmapChart(predictions)
        captured = []
        monkeypatch.setattr(chart, '_render_heatmap', lambda matrix, *args: captured.extend(matrix.ravel()) or Figure())
        chart.render('dataset_name', 'model_name', rank_metric='rmse', display_metric=metric, score_scope='cv')
        actual = captured
    elif kind == 'histogram':
        chart = ScoreHistogramChart(predictions)
        captured = []
        monkeypatch.setattr(chart, '_plot_histogram', lambda ax, scores, *args: captured.extend(scores))
        chart.render(display_metric=metric, clip_outliers=False, score_scope='cv')
        actual = captured
    else:
        chart = CandlestickChart(predictions)
        fig = chart.render(variable='model_name', display_metric=metric, clip_outliers=False, score_scope='cv')
        actual = [line.get_ydata()[0] for line in fig.axes[0].lines[::3]]
    np.testing.assert_allclose(sorted(actual), sorted(values))
