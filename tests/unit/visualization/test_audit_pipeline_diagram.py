"""Quantitative and textual diagram evidence for VIZ-23, VIZ-26 and VIZ-27."""

import numpy as np
import pytest
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from nirs4all.data.predictions import Predictions
from nirs4all.visualization.pipeline_diagram import PipelineDiagram
from nirs4all.visualization.predictions import PredictionAnalyzer


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def _predictions():
    predictions = Predictions()
    for partition in ('val', 'test'):
        predictions.add_prediction(
            dataset_name='data', model_name='Ridge', fold_id=0, partition=partition,
            metric='rmse', val_score=0.2, test_score=0.8,
            scores={'val': {'rmse': 0.2, 'r2': 0.7}, 'test': {'rmse': 0.8, 'r2': -0.3}},
            y_true=np.array([0., 1.]), y_pred=np.array([0.1, 1.1]),
        )
    return predictions


@pytest.mark.parametrize('metric,partition,score', [('rmse', 'val', '0.20'), ('rmse', 'test', '0.80'), ('r2', 'val', '0.70'), ('r2', 'test', '-0.30')])
def test_branch_diagram_displays_requested_metric_and_partition(metric, partition, score):
    analyzer = PredictionAnalyzer(_predictions())
    fig = analyzer.plot_branch_diagram(metric=metric, partition=partition)
    text = '\n'.join(item.get_text() for item in fig.axes[0].texts)
    assert f'{metric} [{partition}] {score}' in text


def test_hiding_metrics_does_not_hide_shapes(monkeypatch):
    predictions = _predictions()
    original = PipelineDiagram._format_shape_display
    calls = []

    def shapes(diagram, node):
        calls.append(node.id)
        return original(diagram, node)

    monkeypatch.setattr(PipelineDiagram, '_format_shape_display', shapes)
    fig = PredictionAnalyzer(predictions).plot_branch_diagram(show_metrics=False, metric='r2')
    assert calls
    assert not any('★' in text.get_text() for text in fig.axes[0].texts)


@pytest.mark.parametrize('key,value', [('by_metadata', 'site'), ('by_tag', 'outlier')])
def test_separation_branch_does_not_invent_keyword_branches(key, value):
    diagram = PipelineDiagram([{'branch': {key: value, 'steps': [Ridge()]}}, {'merge': {'features': 'all'}}])
    diagram.render(initial_shape=(20, 1, 8))
    labels = [node.label for node in diagram.nodes.values()]
    assert f'Separate {key}: {value}' in labels
    assert 'Ridge' in labels
    assert 'Merge (features)' in labels
    assert 'steps' not in labels
    assert key not in labels


def test_separation_branch_preserves_explicit_group_steps():
    diagram = PipelineDiagram([{'branch': {'by_metadata': 'site', 'steps': {'A': [StandardScaler()], 'B': [Ridge()]}}}])
    diagram.render(initial_shape=(20, 1, 8))
    labels = [node.label for node in diagram.nodes.values()]
    assert all(label in labels for label in ('A', 'B', 'StandardScaler', 'Ridge'))


def test_arrowheads_connect_to_final_multiline_box_bounds(monkeypatch):
    diagram = PipelineDiagram([StandardScaler(), Ridge()])
    captured = {}
    original = diagram._compute_layout

    def layout():
        captured.update(original())
        return captured

    monkeypatch.setattr(diagram, '_compute_layout', layout)
    fig = diagram.render(initial_shape=(20, 1, 8))
    arrows = [text for text in fig.axes[0].texts if getattr(text, 'arrow_patch', None) is not None]
    assert len(arrows) == len(diagram.edges)
    for arrow, (source, target) in zip(arrows, diagram.edges, strict=True):
        start, end = captured[source], captured[target]
        assert start['height'] > diagram._node_height
        assert end['height'] > diagram._node_height
        assert arrow.xy[1] == pytest.approx(end['y'] + end['height'] / 2 + 0.1)
        assert arrow.xyann[1] == pytest.approx(start['y'] - start['height'] / 2 - 0.1)
