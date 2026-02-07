"""Memory stress test baselines for Phase 1.6.

Three scenarios that establish peak RSS baselines. No hard assertions on RSS
values (they shift with hardware), but the tests must complete without OOM.

Run with: pytest tests/integration/test_memory_stress.py -v -s
Skip in fast CI: pytest -m "not stress"
"""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.utils.memory import format_bytes, get_process_rss_mb

pytestmark = [pytest.mark.stress, pytest.mark.sklearn]


def _make_dataset(n_samples: int, n_features: int, seed: int = 42) -> SpectroDataset:
    """Create a synthetic regression dataset with train/test split."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.sum(X[:, :5], axis=1) + np.random.randn(n_samples).astype(np.float32) * 0.1

    n_train = int(n_samples * 0.8)
    dataset = SpectroDataset(name="stress_test")
    dataset.add_samples(X[:n_train], indexes={"partition": "train"})
    dataset.add_targets(y[:n_train])
    dataset.add_samples(X[n_train:], indexes={"partition": "test"})
    dataset.add_targets(y[n_train:])
    return dataset


def _print_rss(label: str) -> float:
    rss = get_process_rss_mb()
    print(f"  [{label}] RSS: {rss:.1f} MB")
    return rss


class TestGeneratorVariantExplosion:
    """Scenario 1: Large dataset + many generated preprocessing variants.

    5000 samples x 1000 features with 2 _or_ variants x 5 _range_ values
    = 10 pipeline variants. Measures peak RSS to establish the baseline
    that step caching (Phase 2) should improve.
    """

    def test_generator_variant_explosion(self):
        dataset = _make_dataset(5000, 1000)
        rss_before = _print_rss("before")

        pipeline = [
            {"_or_": [MinMaxScaler, StandardScaler]},
            ShuffleSplit(n_splits=1, test_size=0.3),
            {"model": PLSRegression(), "_range_": [1, 11, 2], "param": "n_components"},
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, _ = runner.run(
            PipelineConfigs(pipeline, "stress_generator"),
            dataset
        )

        rss_after = _print_rss("after")
        peak_delta = rss_after - rss_before

        print(f"  Generator variants: {len(predictions)} predictions")
        print(f"  Dataset steady: {format_bytes(dataset.num_samples * 1000 * 4)}")
        print(f"  RSS delta: {peak_delta:.1f} MB")

        assert len(predictions) > 0


class TestBranchHeavyPipeline:
    """Scenario 2: 6 duplication branches with post-branch model steps.

    Measures peak RSS to establish the baseline that CoW snapshots
    (Phase 3) should improve.
    """

    def test_branch_heavy_pipeline(self):
        dataset = _make_dataset(3000, 700)
        rss_before = _print_rss("before")

        pipeline = [
            ShuffleSplit(n_splits=1, test_size=0.3),
            {"branch": [
                [StandardScaler(), {"model": PLSRegression(n_components=5)}],
                [MinMaxScaler(), {"model": PLSRegression(n_components=5)}],
                [StandardScaler(), {"model": PLSRegression(n_components=3)}],
                [MinMaxScaler(), {"model": PLSRegression(n_components=3)}],
                [StandardScaler(), {"model": PLSRegression(n_components=8)}],
                [MinMaxScaler(), {"model": PLSRegression(n_components=8)}],
            ]},
            {"merge": "predictions"},
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, _ = runner.run(
            PipelineConfigs(pipeline, "stress_branches"),
            dataset
        )

        rss_after = _print_rss("after")
        peak_delta = rss_after - rss_before

        print("  Branches: 6 duplication branches")
        print(f"  Dataset steady: {format_bytes(dataset.num_samples * 700 * 4)}")
        print(f"  RSS delta: {peak_delta:.1f} MB")

        assert len(predictions) > 0


class TestFeatureAugmentationGrowth:
    """Scenario 3: Feature augmentation with many operations in add mode.

    Measures peak RSS to evaluate whether block-based storage (Phase 4) is
    needed. Each augmentation step adds new processings to the 3D feature array.
    """

    def test_feature_augmentation_growth(self):
        dataset = _make_dataset(2000, 500)
        rss_before = _print_rss("before")

        # Build a pipeline that adds features in "add" mode (via feature_augmentation)
        # This grows the processings dimension from 1 to 11
        augmentation_steps = [
            {"feature_augmentation": MinMaxScaler(), "mode": "add"}
            for _ in range(10)
        ]

        pipeline = [
            *augmentation_steps,
            ShuffleSplit(n_splits=1, test_size=0.3),
            {"model": PLSRegression(n_components=5)},
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, _ = runner.run(
            PipelineConfigs(pipeline, "stress_augmentation"),
            dataset
        )

        rss_after = _print_rss("after")
        peak_delta = rss_after - rss_before

        print("  Augmentation steps: 10 (add mode)")
        print(f"  Dataset steady: {format_bytes(dataset.num_samples * 500 * 4)}")
        print(f"  RSS delta: {peak_delta:.1f} MB")

        assert len(predictions) > 0
