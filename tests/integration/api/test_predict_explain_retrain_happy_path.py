"""
Strict happy-path integration tests for predict(), explain(), and retrain() (F-01, NF-API-01).

These tests exercise the full end-to-end path and NEVER swallow exceptions.
Any failure in the pipeline path (run → export → predict/explain/retrain) will
propagate and fail the test, ensuring we detect regressions immediately.
"""

import gc
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_xy():
    """Small deterministic regression dataset as numpy arrays."""
    rng = np.random.default_rng(42)
    X = rng.normal(0.5, 0.1, size=(80, 50)).astype(np.float64)
    # y is a linear combination of the first 5 features
    y = X[:, :5].sum(axis=1) + rng.normal(0, 0.05, size=80)
    return X, y

@pytest.fixture
def simple_pipeline():
    """Minimal PLS pipeline."""
    return [
        MinMaxScaler(),
        ShuffleSplit(n_splits=2, test_size=0.25, random_state=0),
        {"model": PLSRegression(n_components=3)},
    ]

_cleanup_errors = sys.version_info >= (3, 10)
_tmpdir_kwargs = {"ignore_cleanup_errors": True} if _cleanup_errors else {}

# ---------------------------------------------------------------------------
# predict() happy path
# ---------------------------------------------------------------------------

class TestPredictHappyPath:

    def test_predict_from_exported_bundle(self, regression_xy, simple_pipeline):
        """Export a trained model to .n4a, then predict on new data – no exception swallowing."""
        import nirs4all
        from nirs4all.core.logging import reset_logging

        X, y = regression_xy
        X_new = X[:10]

        with tempfile.TemporaryDirectory(**_tmpdir_kwargs) as tmpdir:
            # Train
            result = nirs4all.run(
                pipeline=simple_pipeline,
                dataset=(X, y),
                verbose=0,
                save_artifacts=True,
                workspace_path=tmpdir,
            )

            assert result.num_predictions > 0

            # Export best model
            bundle_path = Path(tmpdir) / "model.n4a"
            result.export(bundle_path)
            assert bundle_path.exists(), "Exported bundle file not found"

            # Predict – must succeed without exception
            pred = nirs4all.predict(
                model=str(bundle_path),
                data=X_new,
                verbose=0,
            )

            # Assert result type and shape
            assert isinstance(pred, nirs4all.PredictResult)
            assert pred.y_pred is not None
            assert len(pred.y_pred) == len(X_new)
            assert np.all(np.isfinite(pred.y_pred))

            reset_logging()
            gc.collect()

    def test_predict_returns_predict_result_type(self, regression_xy, simple_pipeline):
        """predict() return value is a PredictResult with expected attributes."""
        import nirs4all
        from nirs4all.core.logging import reset_logging

        X, y = regression_xy

        with tempfile.TemporaryDirectory(**_tmpdir_kwargs) as tmpdir:
            result = nirs4all.run(
                pipeline=simple_pipeline,
                dataset=(X, y),
                verbose=0,
                save_artifacts=True,
                workspace_path=tmpdir,
            )

            bundle_path = Path(tmpdir) / "model.n4a"
            result.export(bundle_path)

            pred = nirs4all.predict(
                model=str(bundle_path),
                data=X,
                verbose=0,
            )

            # PredictResult contract
            assert hasattr(pred, "y_pred")
            assert hasattr(pred, "values")
            assert hasattr(pred, "shape")
            assert hasattr(pred, "model_name")
            np.testing.assert_array_equal(pred.values, pred.y_pred)

            reset_logging()
            gc.collect()

    def test_predict_shape_matches_input(self, regression_xy, simple_pipeline):
        """y_pred shape matches the number of input samples."""
        import nirs4all
        from nirs4all.core.logging import reset_logging

        X, y = regression_xy

        with tempfile.TemporaryDirectory(**_tmpdir_kwargs) as tmpdir:
            result = nirs4all.run(
                pipeline=simple_pipeline,
                dataset=(X, y),
                verbose=0,
                save_artifacts=True,
                workspace_path=tmpdir,
            )

            bundle_path = Path(tmpdir) / "model.n4a"
            result.export(bundle_path)

            for n_samples in (1, 5, 20):
                pred = nirs4all.predict(
                    model=str(bundle_path),
                    data=X[:n_samples],
                    verbose=0,
                )
                assert len(pred.y_pred) == n_samples

            reset_logging()
            gc.collect()

# ---------------------------------------------------------------------------
# retrain() happy path
# ---------------------------------------------------------------------------

class TestRetrainHappyPath:

    def test_retrain_full_mode_returns_run_result(self, regression_xy, simple_pipeline):
        """retrain(mode='full') with a .n4a bundle trains a new model on new data."""
        import nirs4all
        from nirs4all.core.logging import reset_logging

        X, y = regression_xy
        X_new = X[40:]
        y_new = y[40:]

        with tempfile.TemporaryDirectory(**_tmpdir_kwargs) as tmpdir:
            # Initial training
            result = nirs4all.run(
                pipeline=simple_pipeline,
                dataset=(X[:40], y[:40]),
                verbose=0,
                save_artifacts=True,
                workspace_path=tmpdir,
            )

            bundle_path = Path(tmpdir) / "model.n4a"
            result.export(bundle_path)
            assert bundle_path.exists()

            retrain_workspace = Path(tmpdir) / "retrain_ws"
            retrain_workspace.mkdir()

            # Retrain – must not raise
            retrain_result = nirs4all.retrain(
                source=str(bundle_path),
                data=(X_new, y_new),
                mode="full",
                verbose=0,
                save_artifacts=False,
                workspace_path=str(retrain_workspace),
            )

            assert isinstance(retrain_result, nirs4all.RunResult)
            assert retrain_result.num_predictions > 0

            reset_logging()
            gc.collect()

    def test_retrain_result_has_valid_scores(self, regression_xy, simple_pipeline):
        """retrain() produces predictions with finite (non-NaN) scores."""
        import nirs4all
        from nirs4all.core.logging import reset_logging

        X, y = regression_xy

        with tempfile.TemporaryDirectory(**_tmpdir_kwargs) as tmpdir:
            result = nirs4all.run(
                pipeline=simple_pipeline,
                dataset=(X[:40], y[:40]),
                verbose=0,
                save_artifacts=True,
                workspace_path=tmpdir,
            )

            bundle_path = Path(tmpdir) / "model.n4a"
            result.export(bundle_path)

            retrain_workspace = Path(tmpdir) / "retrain_ws"
            retrain_workspace.mkdir()

            retrain_result = nirs4all.retrain(
                source=str(bundle_path),
                data=(X[40:], y[40:]),
                mode="full",
                verbose=0,
                save_artifacts=False,
                workspace_path=str(retrain_workspace),
            )

            validation = retrain_result.validate(raise_on_failure=False)
            assert validation["nan_count"] == 0, (
                f"retrain() produced NaN scores: {validation['issues']}"
            )

            reset_logging()
            gc.collect()

# ---------------------------------------------------------------------------
# RunResult.validate() after run()
# ---------------------------------------------------------------------------

class TestRunResultValidateAfterRun:
    """validate() on a real run result must pass without raising."""

    def test_validate_passes_on_clean_run(self, regression_xy, simple_pipeline):
        import nirs4all
        from nirs4all.core.logging import reset_logging

        X, y = regression_xy

        result = nirs4all.run(
            pipeline=simple_pipeline,
            dataset=(X, y),
            verbose=0,
            save_artifacts=False,
        )

        report = result.validate()  # Raises on failure
        assert report["valid"] is True
        assert report["nan_count"] == 0
        assert report["total_count"] == result.num_predictions

        reset_logging()
        gc.collect()
