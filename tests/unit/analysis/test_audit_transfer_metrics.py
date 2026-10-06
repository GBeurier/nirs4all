"""Numerical regression witnesses for VIZ-05–08, VIZ-24/25 and VIZ-28."""

from dataclasses import replace

import numpy as np
import pytest
from matplotlib import pyplot as plt
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.analysis.results import TransferResult, TransferSelectionResults
from nirs4all.analysis.selector import TransferPreprocessingSelector
from nirs4all.analysis.transfer_metrics import TransferMetrics, TransferMetricsComputer, compute_transfer_score
from nirs4all.analysis.transfer_utils import apply_stacked_pipeline
from nirs4all.visualization.analysis.transfer import PreprocPCAEvaluator


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def _metrics(**changes):
    return replace(TransferMetrics(1., 1., 0., 1., 0., 1., 2., 1., 1.), **changes)


def test_centroid_distance_retains_instrument_offset_and_common_translation():
    source = np.random.default_rng(12).normal(size=(60, 4))
    target = source + 5
    computer = TransferMetricsComputer(n_components=4)
    forward = computer.compute(source, target, compute_trust=False)
    translated = computer.compute(source + 100, target + 100, compute_trust=False)
    reverse = computer.compute(target, source, compute_trust=False)
    assert forward.centroid_distance == pytest.approx(10.)
    assert translated.centroid_distance == pytest.approx(forward.centroid_distance)
    assert reverse.centroid_distance == pytest.approx(forward.centroid_distance)


def test_cross_dataset_pca_evaluator_reports_real_centroid_reduction():
    source = np.random.default_rng(7).normal(size=(60, 4))
    raw = {'A': source, 'B': source + 5}
    centered = {name: values - values.mean(axis=0) for name, values in raw.items()}
    evaluator = PreprocPCAEvaluator(r_components=4, knn=5).fit(raw, {'centered': centered})
    row = evaluator.cross_dataset_df_.iloc[0]
    assert row['centroid_dist_raw'] == pytest.approx(10.)
    assert row['centroid_dist_pp'] == pytest.approx(0., abs=1e-10)
    assert row['centroid_improvement'] == pytest.approx(1.)


@pytest.mark.parametrize('spread,expected', [(0., 1.), (0.5, 0.75), (2., 0.), (4., -1.)])
def test_spread_score_uses_baseline(spread, expected):
    weights = {'centroid': 0., 'cka': 0., 'spread': 1., 'evr': 0.}
    assert compute_transfer_score(_metrics(spread_distance=spread), _metrics(), weights) == pytest.approx(expected)


@pytest.mark.parametrize('raw,preprocessed,expected', [(1., 0., 0.), (1., 0.5, 0.5), (0., 0., 1.), (0., 1., 0.)])
def test_zero_evr_does_not_receive_full_preservation_credit(raw, preprocessed, expected):
    weights = {'centroid': 0., 'cka': 0., 'spread': 0., 'evr': 1.}
    assert compute_transfer_score(_metrics(evr_source=preprocessed), _metrics(evr_source=raw), weights) == pytest.approx(expected)


@pytest.mark.parametrize('noise', [0.3, 1., 5.])
def test_trustworthiness_matches_sklearn_for_distinct_distances(noise):
    rng = np.random.default_rng(15)
    reference = rng.normal(size=(60, 8))
    embedded = reference[:, :3] + rng.normal(scale=noise, size=(60, 3))
    expected = trustworthiness(reference, embedded, n_neighbors=5)
    assert TransferMetricsComputer(k_neighbors=5)._trustworthiness(reference, embedded) == pytest.approx(expected)
    assert PreprocPCAEvaluator()._trust(reference, embedded, 5) == pytest.approx(expected)


def test_augmented_stacked_components_and_stage4_preserve_transform_objects():
    source = np.random.default_rng(16).normal(size=(40, 4))
    target = source + 2
    selector = TransferPreprocessingSelector(
        preset=None, preprocessings={'scale': StandardScaler(), 'norm': MinMaxScaler()},
        n_components=4, n_jobs=1, stage4_models=[], verbose=0,
    )
    selector.raw_metrics_ = selector.metrics_computer.compute(source, target, compute_trust=False)
    results = selector._parallel_evaluate_preprocessings(
        source, target, [('scale>norm+norm', ['scale>norm', 'norm'])], pipeline_type='augmented',
    )
    assert len(results) == 1
    transforms = results[0].get_transforms()
    assert len(transforms) == 2 and all(transform is not None for transform in transforms)
    expected = apply_stacked_pipeline(source, 'scale>norm', selector.preprocessings)
    np.testing.assert_allclose(transforms[0].fit_transform(source), expected)
    validated = selector._stage4_supervised_validation(source, source[:, 0], results)
    assert validated[0].signal_score is not None
    assert validated[0].transforms is results[0].transforms
    selection = TransferSelectionResults(ranking=validated, raw_metrics={})
    assert len(selection.to_preprocessing_list(top_k=1)) == 1
    assert all(transform is not None for transform in selection.to_preprocessing_list(top_k=1)[0])


@pytest.mark.parametrize('metric,cmap_name', [('centroid_distance', 'RdYlGn_r'), ('cka_similarity', 'RdYlGn')])
def test_metric_comparison_colors_follow_metric_direction(metric, cmap_name):
    ranking = [TransferResult(str(value), 'single', [], value, {metric: value}, 0.) for value in (0.2, 0.8)]
    selection = TransferSelectionResults(ranking=ranking, raw_metrics={})
    fig = selection.plot_metrics_comparison(metrics=[metric])
    colors = [bar.get_facecolor() for bar in fig.axes[0].patches]
    np.testing.assert_allclose(colors[0][:3], plt.get_cmap(cmap_name)(0.)[:3])
    np.testing.assert_allclose(colors[1][:3], plt.get_cmap(cmap_name)(1.)[:3])


@pytest.mark.parametrize('sizes', [(20, 20), (150, 160)])
def test_spread_is_reproducible_without_mutating_global_random_state(sizes):
    rng = np.random.default_rng(91)
    source, target = [rng.normal(size=(size, 3)) for size in sizes]
    evaluator = PreprocPCAEvaluator()
    np.random.seed(34)
    expected_state = np.random.get_state()
    values = [evaluator._compute_spread_distance(source, target) for _ in range(3)]
    assert values == [values[0]] * 3
    actual_state = np.random.get_state()
    np.testing.assert_array_equal(actual_state[1], expected_state[1])
    assert actual_state[2:] == expected_state[2:]
    if sizes == (20, 20):
        assert evaluator._compute_spread_distance(target, source) == pytest.approx(values[0])
