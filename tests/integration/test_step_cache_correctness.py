"""Integration test: step cache correctness.

Runs the same generator pipeline with step caching ON and OFF,
then asserts that all prediction scores are identical within
floating-point tolerance.  This verifies that the step cache
does not alter results.

Run with: pytest tests/integration/test_step_cache_correctness.py -v -s
"""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.config.cache_config import CacheConfig
from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.runner import PipelineRunner

pytestmark = [pytest.mark.sklearn]

def _make_dataset(n_samples: int = 200, n_features: int = 50, seed: int = 42) -> SpectroDataset:
    """Create a deterministic synthetic dataset."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = np.sum(X[:, :3], axis=1) + rng.randn(n_samples).astype(np.float32) * 0.1

    n_train = int(n_samples * 0.8)
    ds = SpectroDataset(name="cache_test")
    ds.add_samples(X[:n_train], indexes={"partition": "train"})
    ds.add_targets(y[:n_train])
    ds.add_samples(X[n_train:], indexes={"partition": "test"})
    ds.add_targets(y[n_train:])
    return ds

def _run_pipeline(pipeline, tmp_path, cache_enabled: bool):
    """Run a pipeline and return sorted test scores."""
    ds = _make_dataset()
    runner = PipelineRunner(
        verbose=0,
        save_artifacts=False,
        workspace_path=str(tmp_path / ("cached" if cache_enabled else "uncached")),
    )
    if cache_enabled:
        runner.cache_config = CacheConfig(
            step_cache_enabled=True,
            step_cache_max_mb=512,
            log_cache_stats=False,
        )
    predictions, _ = runner.run(pipeline, ds, refit=False)
    # Extract all test scores sorted for deterministic comparison
    scores = []
    for pred in predictions.to_dicts():
        scores.append(pred.get("test_score", 0.0))
    return sorted(scores)

class TestStepCacheCorrectness:
    """Verify cache ON vs OFF produces identical results."""

    def test_generator_or_produces_same_scores(self, tmp_path):
        """_or_ generator with two scalers: cache ON == cache OFF."""
        pipeline = [
            {"_or_": [MinMaxScaler, StandardScaler]},
            ShuffleSplit(n_splits=2, test_size=0.3, random_state=0),
            {"model": PLSRegression(n_components=5)},
        ]

        scores_off = _run_pipeline(pipeline, tmp_path, cache_enabled=False)
        scores_on = _run_pipeline(pipeline, tmp_path, cache_enabled=True)

        assert len(scores_off) == len(scores_on), (
            f"Different number of predictions: {len(scores_off)} vs {len(scores_on)}"
        )
        np.testing.assert_allclose(
            scores_off, scores_on, atol=1e-10,
            err_msg="Step cache produced different scores than uncached execution",
        )

    def test_generator_range_produces_same_scores(self, tmp_path):
        """_range_ generator with parameter sweep: cache ON == cache OFF."""
        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.3, random_state=0),
            {"model": PLSRegression(), "_range_": [2, 8, 2], "param": "n_components"},
        ]

        scores_off = _run_pipeline(pipeline, tmp_path, cache_enabled=False)
        scores_on = _run_pipeline(pipeline, tmp_path, cache_enabled=True)

        assert len(scores_off) == len(scores_on)
        np.testing.assert_allclose(
            scores_off, scores_on, atol=1e-10,
            err_msg="Step cache produced different scores for _range_ pipeline",
        )

    def test_combined_or_and_range(self, tmp_path):
        """Combined _or_ + _range_: 2 scalers x 3 PLS components = 6 variants."""
        pipeline = [
            {"_or_": [MinMaxScaler, StandardScaler]},
            ShuffleSplit(n_splits=1, test_size=0.3, random_state=0),
            {"model": PLSRegression(), "_range_": [2, 8, 2], "param": "n_components"},
        ]

        scores_off = _run_pipeline(pipeline, tmp_path, cache_enabled=False)
        scores_on = _run_pipeline(pipeline, tmp_path, cache_enabled=True)

        assert len(scores_off) == len(scores_on)
        np.testing.assert_allclose(
            scores_off, scores_on, atol=1e-10,
            err_msg="Step cache produced different scores for combined generator pipeline",
        )
