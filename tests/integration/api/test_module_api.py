"""
Integration tests for the nirs4all module-level API.

Tests the new simplified API functions:
- nirs4all.run()
- nirs4all.predict()
- nirs4all.explain()
- nirs4all.retrain()
- nirs4all.session()
"""

import gc
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler


class TestRunFunction:
    """Tests for nirs4all.run() function."""

    def test_run_with_list_pipeline_and_path_dataset(self, sample_regression_data_path):
        """Test run() with pipeline as list and dataset as path."""
        import nirs4all

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.3),
            {"model": PLSRegression(n_components=5)}
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=sample_regression_data_path,
            verbose=0,
            save_artifacts=False
        )

        # Verify result type
        assert isinstance(result, nirs4all.RunResult)

        # Verify predictions are available
        assert result.num_predictions > 0

        # Verify convenience accessors work
        assert result.best is not None
        assert not np.isnan(result.best_score)

    def test_run_with_numpy_arrays(self):
        """Test run() with numpy array dataset."""
        import nirs4all

        # Create sample data
        np.random.seed(42)
        X = np.random.randn(50, 100)
        y = np.random.randn(50)

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.3),
            {"model": PLSRegression(n_components=3)}
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=(X, y),
            verbose=0,
            save_artifacts=False
        )

        assert result.num_predictions > 0
        assert result.best is not None

    def test_run_with_name_parameter(self, sample_regression_data_path):
        """Test run() with custom pipeline name."""
        import nirs4all

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.3),
            {"model": PLSRegression(n_components=5)}
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=sample_regression_data_path,
            name="CustomPipelineName",
            verbose=0,
            save_artifacts=False
        )

        assert result.num_predictions > 0

    def test_run_returns_run_result_with_all_accessors(self, sample_regression_data_path):
        """Test that RunResult has all expected accessor properties."""
        import nirs4all

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.3),
            {"model": PLSRegression(n_components=5)}
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=sample_regression_data_path,
            verbose=0,
            save_artifacts=False
        )

        # Test all accessors exist and work
        assert hasattr(result, 'predictions')
        assert hasattr(result, 'per_dataset')
        assert hasattr(result, 'best')
        assert hasattr(result, 'best_score')
        assert hasattr(result, 'best_rmse')
        assert hasattr(result, 'best_r2')
        assert hasattr(result, 'num_predictions')

        # Test methods
        top_3 = result.top(n=3)
        assert isinstance(top_3, list)

        datasets = result.get_datasets()
        assert isinstance(datasets, list)

        models = result.get_models()
        assert isinstance(models, list)

    def test_run_with_runner_kwargs(self, sample_regression_data_path):
        """Test run() passes kwargs to PipelineRunner correctly."""
        import nirs4all
        from nirs4all.core.logging import reset_logging

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.3),
            {"model": PLSRegression(n_components=5)}
        ]

        # Use ignore_cleanup_errors=True to handle Windows file locking issues
        # where file handles may not be immediately released after close()
        cleanup_errors = sys.version_info >= (3, 10)
        tmpdir_kwargs = {"ignore_cleanup_errors": True} if cleanup_errors else {}

        with tempfile.TemporaryDirectory(**tmpdir_kwargs) as tmpdir:
            result = nirs4all.run(
                pipeline=pipeline,
                dataset=sample_regression_data_path,
                verbose=0,
                save_artifacts=True,
                workspace_path=tmpdir
            )

            assert result.num_predictions > 0

            # Reset logging to close file handlers before temp directory cleanup
            # This is required on Windows where open file handles prevent deletion
            reset_logging()

            # Force garbage collection to release any lingering file handles
            # This helps on Windows where file handles may be held by cyclic refs
            gc.collect()

            # Small delay to allow OS to release file handles (Windows issue)
            if sys.platform == "win32":
                time.sleep(0.1)

    def test_run_with_random_state(self, sample_regression_data_path):
        """Test run() with random_state for reproducibility."""
        import nirs4all

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.3, random_state=42),
            {"model": PLSRegression(n_components=5)}
        ]

        result1 = nirs4all.run(
            pipeline=pipeline,
            dataset=sample_regression_data_path,
            verbose=0,
            save_artifacts=False,
            random_state=42
        )

        result2 = nirs4all.run(
            pipeline=pipeline,
            dataset=sample_regression_data_path,
            verbose=0,
            save_artifacts=False,
            random_state=42
        )

        # With same random state, results should be similar
        assert abs(result1.best_score - result2.best_score) < 0.01


class TestSessionContextManager:
    """Tests for nirs4all.session() context manager."""

    def test_session_basic_usage(self, sample_regression_data_path):
        """Test basic session usage."""
        import nirs4all

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.3),
            {"model": PLSRegression(n_components=5)}
        ]

        with nirs4all.session(verbose=0, save_artifacts=False) as s:
            assert isinstance(s, nirs4all.Session)

            result = nirs4all.run(
                pipeline=pipeline,
                dataset=sample_regression_data_path,
                session=s
            )

            assert result.num_predictions > 0

    def test_session_multiple_runs(self, sample_regression_data_path):
        """Test multiple runs within a session."""
        import nirs4all

        pipeline1 = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.3),
            {"model": PLSRegression(n_components=3)}
        ]

        pipeline2 = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.3),
            {"model": PLSRegression(n_components=10)}
        ]

        with nirs4all.session(verbose=0, save_artifacts=False) as s:
            result1 = nirs4all.run(pipeline1, sample_regression_data_path, session=s)
            result2 = nirs4all.run(pipeline2, sample_regression_data_path, session=s)

            assert result1.num_predictions > 0
            assert result2.num_predictions > 0

    def test_session_lazy_runner_creation(self):
        """Test that session creates runner lazily."""
        import nirs4all

        with nirs4all.session(verbose=0) as s:
            # Runner not created yet
            assert s._runner is None

            # Access runner property triggers creation
            runner = s.runner
            assert runner is not None
            assert s._runner is not None

    def test_session_cleanup_on_exit(self):
        """Test session cleanup on context exit."""
        import nirs4all

        session_obj = None
        with nirs4all.session(verbose=0) as s:
            session_obj = s
            _ = s.runner  # Trigger runner creation

        # After exit, runner should be cleaned up
        assert session_obj._runner is None


class TestResultClasses:
    """Tests for result classes."""

    def test_run_result_summary(self, sample_regression_data_path):
        """Test RunResult.summary() method."""
        import nirs4all

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.3),
            {"model": PLSRegression(n_components=5)}
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=sample_regression_data_path,
            verbose=0,
            save_artifacts=False
        )

        summary = result.summary()
        assert isinstance(summary, str)
        assert "RunResult" in summary

    def test_run_result_str_repr(self, sample_regression_data_path):
        """Test RunResult string representations."""
        import nirs4all

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.3),
            {"model": PLSRegression(n_components=5)}
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=sample_regression_data_path,
            verbose=0,
            save_artifacts=False
        )

        # __repr__ should work
        repr_str = repr(result)
        assert "RunResult" in repr_str

        # __str__ should work (calls summary)
        str_str = str(result)
        assert len(str_str) > 0

    def test_predict_result_properties(self):
        """Test PredictResult properties."""
        from nirs4all.api.result import PredictResult

        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = PredictResult(
            y_pred=y_pred,
            model_name="TestModel",
            preprocessing_steps=["MinMaxScaler", "SNV"]
        )

        assert np.array_equal(result.values, y_pred)
        assert result.shape == (5,)
        assert not result.is_multioutput
        assert len(result) == 5
        assert result.model_name == "TestModel"

    def test_predict_result_to_list(self):
        """Test PredictResult.to_list()."""
        from nirs4all.api.result import PredictResult

        y_pred = np.array([1.0, 2.0, 3.0])
        result = PredictResult(y_pred=y_pred)

        as_list = result.to_list()
        assert isinstance(as_list, list)
        assert as_list == [1.0, 2.0, 3.0]

    def test_explain_result_properties(self):
        """Test ExplainResult properties."""
        from nirs4all.api.result import ExplainResult

        shap_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        result = ExplainResult(
            shap_values=shap_values,
            feature_names=["f1", "f2", "f3"],
            explainer_type="kernel"
        )

        assert result.shape == (2, 3)
        assert len(result.mean_abs_shap) == 3
        assert result.top_features == ["f3", "f2", "f1"]  # Sorted by importance

    def test_explain_result_feature_importance(self):
        """Test ExplainResult.get_feature_importance()."""
        from nirs4all.api.result import ExplainResult

        shap_values = np.array([[0.1, 0.5, 0.3]])
        result = ExplainResult(
            shap_values=shap_values,
            feature_names=["low", "high", "mid"]
        )

        importance = result.get_feature_importance()
        assert isinstance(importance, dict)
        assert "high" in importance
        # High should have highest importance
        assert importance["high"] >= importance["low"]
        assert importance["high"] >= importance["mid"]


class TestPackageExports:
    """Tests for package-level exports."""

    def test_module_level_imports(self):
        """Test that all API functions are importable from nirs4all."""
        import nirs4all

        # Functions
        assert hasattr(nirs4all, 'run')
        assert hasattr(nirs4all, 'predict')
        assert hasattr(nirs4all, 'explain')
        assert hasattr(nirs4all, 'retrain')
        assert hasattr(nirs4all, 'session')

        # Classes
        assert hasattr(nirs4all, 'Session')
        assert hasattr(nirs4all, 'RunResult')
        assert hasattr(nirs4all, 'PredictResult')
        assert hasattr(nirs4all, 'ExplainResult')

        # Legacy (backward compat)
        assert hasattr(nirs4all, 'PipelineRunner')
        assert hasattr(nirs4all, 'PipelineConfigs')

    def test_api_module_imports(self):
        """Test that all API functions are importable from nirs4all.api."""
        from nirs4all.api import (
            run,
            predict,
            explain,
            retrain,
            session,
            Session,
            RunResult,
            PredictResult,
            ExplainResult
        )

        assert callable(run)
        assert callable(predict)
        assert callable(explain)
        assert callable(retrain)
        assert callable(session)


# Fixtures
@pytest.fixture
def sample_regression_data_path():
    """Path to sample regression data."""
    # Try to find the sample data directory
    candidates = [
        Path(__file__).parent.parent.parent.parent / "examples" / "sample_data" / "regression",
        Path("examples/sample_data/regression"),
        Path("sample_data/regression"),
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    # If no sample data found, create temporary data
    pytest.skip("Sample regression data not found")
