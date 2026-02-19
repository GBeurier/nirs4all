"""Tests for parallel pipeline execution (n_jobs != 1).

Regression test for the loky pickling bug: objects containing threading.RLock
(WorkspaceStore, Explainer→Runner→Orchestrator→Store) cannot be pickled by
joblib's loky backend.  The orchestrator must strip unpicklable objects from
variant_data and use a module-level function (not bound method) for dispatch.
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _make_dataset():
    """Build a small synthetic regression dataset for testing."""
    from nirs4all.synthesis.builder import SyntheticDatasetBuilder

    return (
        SyntheticDatasetBuilder(n_samples=60, random_state=42)
        .with_features(complexity="simple")
        .with_targets(distribution="uniform", range=(10, 50))
        .with_partitions(train_ratio=0.8)
        .build()
    )

class TestParallelExecution:
    """Tests that parallel execution (n_jobs != 1) works without pickling errors."""

    def test_parallel_execution_no_pickling_error(self, tmp_path):
        """Pipeline with n_jobs=2 and multiple variants must not raise PicklingError.

        This is a regression test for the loky PicklingError caused by
        unpicklable threading.RLock objects in WorkspaceStore and Explainer.
        """
        import nirs4all
        from nirs4all.config.cache_config import CacheConfig

        dataset = _make_dataset()

        # _or_ with 2 options generates 2 variants → triggers parallel path
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"_or_": [None, MinMaxScaler()]},
            {"model": PLSRegression(n_components=3)},
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=dataset,
            verbose=0,
            n_jobs=2,
            workspace_path=str(tmp_path / "workspace"),
            cache=CacheConfig(memory_warning_threshold_mb=1024),
        )

        assert result is not None
        assert result.best_rmse is not None

    def test_parallel_vs_sequential_results_consistent(self, tmp_path):
        """Parallel and sequential execution should produce equivalent results."""
        import nirs4all

        dataset = _make_dataset()

        # 2 variants to trigger parallel path
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"_or_": [None, StandardScaler()]},
            {"model": PLSRegression(n_components=5)},
        ]

        result_seq = nirs4all.run(
            pipeline=pipeline,
            dataset=dataset,
            verbose=0,
            n_jobs=1,
            workspace_path=str(tmp_path / "ws_seq"),
            random_state=42,
        )

        result_par = nirs4all.run(
            pipeline=pipeline,
            dataset=dataset,
            verbose=0,
            n_jobs=2,
            workspace_path=str(tmp_path / "ws_par"),
            random_state=42,
        )

        assert result_seq.best_rmse is not None
        assert result_par.best_rmse is not None
        np.testing.assert_allclose(result_seq.best_rmse, result_par.best_rmse, rtol=1e-5)

    def test_parallel_multiple_variants(self, tmp_path):
        """Parallel execution with multiple pipeline variants produces results for each."""
        import nirs4all

        dataset = _make_dataset()

        # _or_ with 3 options generates 3 variants → triggers parallel path
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"_or_": [None, MinMaxScaler(), StandardScaler()]},
            {"model": PLSRegression(n_components=3)},
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=dataset,
            verbose=0,
            n_jobs=2,
            workspace_path=str(tmp_path / "workspace"),
        )

        assert result is not None
        assert result.best_rmse is not None
