"""Population, aggregation and class-label witnesses for VIZ-13–16/20."""

import numpy as np
import polars as pl
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from nirs4all.controllers.charts.targets import YChartController
from nirs4all.data import SpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.config.context import DataSelector, ExecutionContext
from nirs4all.visualization.charts.candlestick import CandlestickChart
from nirs4all.visualization.charts.confusion_matrix import ConfusionMatrixChart
from nirs4all.visualization.charts.heatmap import HeatmapChart
from nirs4all.visualization.charts.histogram import ScoreHistogramChart
from nirs4all.visualization.predictions import PredictionAnalyzer
from tests.unit.visualization.charts.test_audit_chart_scores import _add


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.mark.parametrize('layout', ['stacked', 'staggered'])
def test_audit_local_minor_target_histograms_use_absolute_fold_ids(monkeypatch, layout):
    """Excluded IDs leave holes; both train and validation values must retain identity."""
    dataset = SpectroDataset('absolute_target_ids')
    dataset.add_samples(np.arange(16.).reshape(8, 2), {'partition': 'train'})
    dataset.add_targets(np.arange(8.))
    dataset._indexer.mark_excluded([1, 3], reason='audit holes')
    # Valid partial train/validation populations, as supported by ShuffleSplit.
    folds = [([0, 2], [4, 5]), ([4, 5], [0, 2])]
    dataset.set_folds(folds)
    controller = YChartController()
    captured = []
    monkeypatch.setattr(controller, '_plot_categorical_fold',
                        lambda ax, train, val, *args, **kwargs: captured.append((train.copy(), val.copy())))
    context = ExecutionContext(selector=DataSelector(processing=[['raw']]))
    _, name = controller._create_fold_grid_histogram(dataset, context, dataset.folds, layout=layout)

    assert name == f'Y_distribution_2folds_{layout}'
    assert len(captured) == 2
    for (train_values, val_values), (train_ids, val_ids) in zip(captured, folds, strict=True):
        # The target is the original sample ID: this oracle does not remap through base IDs.
        np.testing.assert_array_equal(train_values, train_ids)
        np.testing.assert_array_equal(val_values, val_ids)


@pytest.mark.parametrize('aggregate', [None, 'y'])
@pytest.mark.parametrize('method,expected', [('best', 10.), ('worst', 100.), ('mean', 130 / 3), ('median', 20.)])
def test_heatmap_display_reduces_full_cell_population_independently(monkeypatch, aggregate, method, expected):
    predictions = Predictions()
    for fold, score in enumerate((10., 20., 100.)):
        _add(predictions, 'data', 'Ridge', error=0.1 * (fold + 1), test_score=score, fold_id=fold)
    if aggregate:
        # Actual grouped array errors must reflect each desired display score.
        for row in predictions.iter_entries():
            if row['partition'] == 'test':
                row['y_pred'] = np.asarray(row['y_true']) + row['test_score']
    chart = HeatmapChart(predictions)
    captured = []
    monkeypatch.setattr(chart, '_render_heatmap_aggregated' if aggregate else '_render_heatmap', lambda matrix, *args: captured.extend(matrix.ravel()) or Figure())
    chart.render('dataset_name', 'model_name', rank_metric='rmse', rank_agg='best', display_agg=method, aggregate=aggregate, score_scope='cv')
    np.testing.assert_allclose(captured, [expected])


@pytest.mark.parametrize('aggregate', [None, 'y'])
def test_heatmap_filter_matches_original_mixed_case_model(monkeypatch, aggregate):
    predictions = Predictions()
    _add(predictions, 'data', 'PLS8')
    chart = HeatmapChart(predictions)
    captured = []
    monkeypatch.setattr(chart, '_render_heatmap_aggregated' if aggregate else '_render_heatmap', lambda matrix, *args: captured.extend(matrix.ravel()) or Figure())
    chart.render('dataset_name', 'model_name', rank_metric='rmse', model_name='PLS8', aggregate=aggregate, score_scope='cv')
    np.testing.assert_allclose(captured, [0.2])


@pytest.mark.parametrize('scope,expected', [('refit', 1), ('final', 1), ('cv', 2), ('folds', 2), ('all', 5)])
@pytest.mark.parametrize('kind', ['heatmap', 'candlestick', 'histogram'])
def test_fast_charts_respect_requested_fold_population(monkeypatch, scope, expected, kind):
    predictions = Predictions()
    for fold in (0, 1, 'avg', 'w_avg', 'final', 'final_agg'):
        _add(predictions, 'data', 'Ridge', fold_id=fold)
    if kind == 'heatmap':
        chart = HeatmapChart(predictions)
        captured = []
        monkeypatch.setattr(chart, '_render_heatmap', lambda matrix, normalized, counts, *args: captured.extend(counts.ravel()) or Figure())
        chart.render('dataset_name', 'model_name', rank_metric='rmse', score_scope=scope)
        assert captured == [expected]
    elif kind == 'histogram':
        chart = ScoreHistogramChart(predictions)
        captured = []
        monkeypatch.setattr(chart, '_plot_histogram', lambda ax, scores, *args: captured.extend(scores))
        chart.render(display_metric='rmse', score_scope=scope)
        assert len(captured) == expected
    else:
        chart = CandlestickChart(predictions)
        original = predictions.to_dataframe
        # Scores identify the scope population through the plotted mean.
        monkeypatch.setattr(predictions, 'to_dataframe', lambda: original().with_columns(
            pl.when(pl.col('fold_id') == 'final')
            .then(10.).otherwise(1.).alias('test_score')
        ))
        fig = chart.render(variable='model_name', display_metric='rmse', score_scope=scope, clip_outliers=False)
        expected_mean = 10. if scope in ('refit', 'final') else 1. if scope in ('cv', 'folds') else 14 / 5
        assert fig.axes[0].lines[1].get_ydata()[0] == pytest.approx(expected_mean)


def test_refit_histogram_with_only_cv_rows_reports_empty_population():
    predictions = Predictions()
    _add(predictions, 'data', 'Ridge')
    fig = ScoreHistogramChart(predictions).render(score_scope='refit')
    assert any('No predictions found' in text.get_text() for text in fig.axes[0].texts)


def test_heatmap_refit_scope_without_validation_scores_reports_missing_data():
    predictions = Predictions()
    _add(predictions, 'data', 'Ridge', fold_id=0)
    _add(predictions, 'data', 'Ridge', fold_id='final')
    # A refit model has test observations but no held-out validation partition.
    for row in predictions.iter_entries():
        if row['fold_id'] == 'final':
            row['val_score'] = None
            row['scores']['val']['rmse'] = None
    fig = HeatmapChart(predictions).render(
        'model_name', 'preprocessings', rank_metric='rmse',
        display_metric='rmse', display_partition='val', score_scope='refit',
    )
    assert not fig.axes[0].images
    assert any('scope=refit, rank=val/rmse, display=val/rmse' in text.get_text() for text in fig.axes[0].texts)
    cv = HeatmapChart(predictions).render(
        'model_name', 'preprocessings', rank_metric='rmse',
        display_metric='rmse', display_partition='val', score_scope='cv',
    )
    assert cv.axes[0].images


@pytest.mark.parametrize('kind', ['heatmap', 'candlestick', 'histogram'])
def test_scope_filter_preserves_empty_chart_message(kind):
    predictions = Predictions()
    if kind == 'heatmap':
        fig = HeatmapChart(predictions).render('dataset_name', 'model_name', rank_metric='rmse')
    elif kind == 'candlestick':
        fig = CandlestickChart(predictions).render('model_name', display_metric='rmse')
    else:
        fig = ScoreHistogramChart(predictions).render(display_metric='rmse')
    assert any('No predictions found' in text.get_text() for text in fig.axes[0].texts)


@pytest.mark.parametrize('labels', [[0, 2], ['cat', 'dog']])
def test_confusion_matrix_ticks_show_actual_class_labels(labels):
    predictions = Predictions()
    for partition in ('val', 'test'):
        predictions.add_prediction(
            dataset_name='data', model_name='classifier', fold_id=0, partition=partition,
            metric='accuracy', val_score=1., test_score=1., task_type='multiclass_classification',
            scores={'val': {'accuracy': 1.}, 'test': {'accuracy': 1.}},
            y_true=np.array(labels * 2), y_pred=np.array(labels * 2),
        )
    fig = ConfusionMatrixChart(predictions).render(k=1, rank_metric='accuracy', score_scope='cv', show_scores=False)
    assert [tick.get_text() for tick in fig.axes[0].get_xticklabels()] == [str(label) for label in labels]
    assert [tick.get_text() for tick in fig.axes[0].get_yticklabels()] == [str(label) for label in labels]


def test_boxplot_uses_matplotlib_37_compatible_signature(monkeypatch):
    predictions = Predictions()
    for name in ('A', 'B'):
        _add(predictions, 'data', 'Ridge' + name, branch_name=name)
    original = Axes.boxplot
    calls = []

    def compatible_boxplot(ax, data, **kwargs):
        assert 'tick_labels' not in kwargs
        calls.append(kwargs)
        return original(ax, data, **kwargs)

    monkeypatch.setattr(Axes, 'boxplot', compatible_boxplot)
    fig = PredictionAnalyzer(predictions).plot_branch_boxplot(rank_metric='rmse', score_scope='cv')
    assert calls
    assert {tick.get_text() for tick in fig.axes[0].get_xticklabels()} == {'A', 'B'}
