"""
Unit tests for nirs4all.api.result module.

Tests the RunResult, PredictResult, and ExplainResult dataclasses.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock

from nirs4all.api.result import RunResult, PredictResult, ExplainResult


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_predictions():
    """Create a mock Predictions object with sample data."""
    mock = Mock()

    # Sample prediction entries
    sample_entries = [
        {
            'id': 'pred_001',
            'model_name': 'PLSRegression',
            'dataset_name': 'wheat',
            'test_score': 0.85,
            'val_score': 0.82,
            'metric': 'rmse',
            'scores': {
                'test': {'rmse': 0.85, 'r2': 0.92, 'mae': 0.65},
                'val': {'rmse': 0.82, 'r2': 0.91, 'mae': 0.62}
            },
            'fold_id': '0',
            'step_idx': 2,
        },
        {
            'id': 'pred_002',
            'model_name': 'RandomForest',
            'dataset_name': 'wheat',
            'test_score': 0.90,
            'val_score': 0.88,
            'metric': 'rmse',
            'scores': {
                'test': {'rmse': 0.90, 'r2': 0.88, 'mae': 0.70},
            },
            'fold_id': '0',
            'step_idx': 2,
        },
    ]

    mock.top.return_value = sample_entries
    mock.num_predictions = 2
    mock.get_datasets.return_value = ['wheat']
    mock.get_models.return_value = ['PLSRegression', 'RandomForest']
    mock.filter_predictions.return_value = sample_entries

    return mock


@pytest.fixture
def mock_runner():
    """Create a mock PipelineRunner."""
    runner = Mock()
    runner.workspace_path = Path('/tmp/workspace')
    runner.export.return_value = Path('/tmp/exports/model.n4a')
    runner.export_model.return_value = Path('/tmp/exports/model.joblib')
    return runner


@pytest.fixture
def run_result(mock_predictions, mock_runner):
    """Create a RunResult instance with mocks."""
    return RunResult(
        predictions=mock_predictions,
        per_dataset={'wheat': {'status': 'success'}},
        _runner=mock_runner
    )


# =============================================================================
# RunResult Tests
# =============================================================================

class TestRunResult:
    """Tests for RunResult dataclass."""

    def test_init(self, mock_predictions):
        """Test basic initialization."""
        result = RunResult(
            predictions=mock_predictions,
            per_dataset={'test': 'data'}
        )
        assert result.predictions == mock_predictions
        assert result.per_dataset == {'test': 'data'}
        assert result._runner is None

    def test_best_property(self, run_result, mock_predictions):
        """Test best property returns first entry from top(1)."""
        mock_predictions.top.return_value = [{'id': 'best_model'}]
        assert run_result.best == {'id': 'best_model'}
        mock_predictions.top.assert_called_with(n=1)

    def test_best_property_empty(self, mock_predictions):
        """Test best property returns empty dict when no predictions."""
        mock_predictions.top.return_value = []
        result = RunResult(predictions=mock_predictions, per_dataset={})
        assert result.best == {}

    def test_best_score(self, run_result, mock_predictions):
        """Test best_score extracts test_score from best."""
        mock_predictions.top.return_value = [{'test_score': 0.75}]
        assert run_result.best_score == 0.75

    def test_best_score_missing(self, mock_predictions):
        """Test best_score returns NaN when not available."""
        mock_predictions.top.return_value = [{'model_name': 'test'}]
        result = RunResult(predictions=mock_predictions, per_dataset={})
        assert np.isnan(result.best_score)

    def test_best_rmse_from_scores(self, run_result, mock_predictions):
        """Test best_rmse extracts from scores dict."""
        mock_predictions.top.return_value = [{
            'scores': {'test': {'rmse': 0.42}}
        }]
        assert run_result.best_rmse == 0.42

    def test_best_rmse_fallback_to_test_score(self, mock_predictions):
        """Test best_rmse falls back to test_score when metric is rmse."""
        mock_predictions.top.return_value = [{
            'metric': 'rmse',
            'test_score': 0.55
        }]
        result = RunResult(predictions=mock_predictions, per_dataset={})
        assert result.best_rmse == 0.55

    def test_best_r2(self, run_result, mock_predictions):
        """Test best_r2 extracts from scores dict."""
        mock_predictions.top.return_value = [{
            'scores': {'test': {'r2': 0.95}}
        }]
        assert run_result.best_r2 == 0.95

    def test_best_accuracy(self, mock_predictions):
        """Test best_accuracy for classification results."""
        mock_predictions.top.return_value = [{
            'scores': {'test': {'accuracy': 0.88}}
        }]
        result = RunResult(predictions=mock_predictions, per_dataset={})
        assert result.best_accuracy == 0.88

    def test_artifacts_path(self, run_result):
        """Test artifacts_path returns runner's workspace_path."""
        assert run_result.artifacts_path == Path('/tmp/workspace')

    def test_artifacts_path_no_runner(self, mock_predictions):
        """Test artifacts_path returns None when no runner."""
        result = RunResult(predictions=mock_predictions, per_dataset={})
        assert result.artifacts_path is None

    def test_num_predictions(self, run_result, mock_predictions):
        """Test num_predictions delegates to predictions."""
        assert run_result.num_predictions == 2

    def test_top(self, run_result, mock_predictions):
        """Test top() delegates to predictions.top()."""
        run_result.top(n=10, rank_metric='r2')
        mock_predictions.top.assert_called_with(n=10, rank_metric='r2')

    def test_filter(self, run_result, mock_predictions):
        """Test filter() delegates to predictions.filter_predictions()."""
        run_result.filter(model_name='PLS', partition='test')
        mock_predictions.filter_predictions.assert_called_with(
            model_name='PLS',
            partition='test'
        )

    def test_get_datasets(self, run_result, mock_predictions):
        """Test get_datasets delegates to predictions."""
        assert run_result.get_datasets() == ['wheat']

    def test_get_models(self, run_result, mock_predictions):
        """Test get_models delegates to predictions."""
        assert run_result.get_models() == ['PLSRegression', 'RandomForest']

    def test_export(self, run_result, mock_runner, mock_predictions):
        """Test export() delegates to runner.export()."""
        mock_predictions.top.return_value = [{'id': 'best'}]
        path = run_result.export('output/model.n4a')

        mock_runner.export.assert_called_with(
            source={'id': 'best'},
            output_path='output/model.n4a',
            format='n4a'
        )
        assert path == Path('/tmp/exports/model.n4a')

    def test_export_with_source(self, run_result, mock_runner):
        """Test export() with explicit source."""
        source = {'id': 'specific_model'}
        run_result.export('output/model.n4a', source=source)

        mock_runner.export.assert_called_with(
            source=source,
            output_path='output/model.n4a',
            format='n4a'
        )

    def test_export_no_runner(self, mock_predictions):
        """Test export() raises when no runner available."""
        result = RunResult(predictions=mock_predictions, per_dataset={})
        with pytest.raises(RuntimeError, match="runner reference not available"):
            result.export('output/model.n4a')

    def test_export_no_predictions(self, mock_runner, mock_predictions):
        """Test export() raises when no predictions and no source."""
        mock_predictions.top.return_value = []
        result = RunResult(
            predictions=mock_predictions,
            per_dataset={},
            _runner=mock_runner
        )
        with pytest.raises(ValueError, match="No predictions available"):
            result.export('output/model.n4a')

    def test_export_model(self, run_result, mock_runner, mock_predictions):
        """Test export_model() delegates to runner.export_model()."""
        mock_predictions.top.return_value = [{'id': 'best'}]
        path = run_result.export_model('output/model.joblib')

        mock_runner.export_model.assert_called_with(
            source={'id': 'best'},
            output_path='output/model.joblib',
            format=None,
            fold=None
        )
        assert path == Path('/tmp/exports/model.joblib')

    def test_summary(self, run_result, mock_predictions):
        """Test summary() returns formatted string."""
        mock_predictions.top.return_value = [{
            'model_name': 'PLSRegression',
            'test_score': 0.85,
            'scores': {'test': {'rmse': 0.85, 'r2': 0.92}}
        }]
        summary = run_result.summary()

        assert 'RunResult' in summary
        assert 'predictions' in summary
        assert 'PLSRegression' in summary

    def test_repr(self, run_result, mock_predictions):
        """Test __repr__ format."""
        mock_predictions.top.return_value = [{'test_score': 0.85}]
        repr_str = repr(run_result)
        assert 'RunResult' in repr_str
        assert 'predictions=' in repr_str

    def test_str(self, run_result, mock_predictions):
        """Test __str__ is same as summary."""
        mock_predictions.top.return_value = [{'model_name': 'test'}]
        assert str(run_result) == run_result.summary()


# =============================================================================
# PredictResult Tests
# =============================================================================

class TestPredictResult:
    """Tests for PredictResult dataclass."""

    def test_init_with_array(self):
        """Test initialization with numpy array."""
        y_pred = np.array([1.0, 2.0, 3.0])
        result = PredictResult(y_pred=y_pred)
        assert np.array_equal(result.y_pred, y_pred)
        assert result.metadata == {}

    def test_init_with_list(self):
        """Test initialization converts list to numpy array."""
        result = PredictResult(y_pred=[1.0, 2.0, 3.0])
        assert isinstance(result.y_pred, np.ndarray)
        assert result.shape == (3,)

    def test_values_property(self):
        """Test values property is alias for y_pred."""
        y_pred = np.array([1.0, 2.0])
        result = PredictResult(y_pred=y_pred)
        assert np.array_equal(result.values, y_pred)

    def test_shape_1d(self):
        """Test shape for 1D predictions."""
        result = PredictResult(y_pred=np.array([1, 2, 3]))
        assert result.shape == (3,)

    def test_shape_2d(self):
        """Test shape for 2D predictions."""
        result = PredictResult(y_pred=np.array([[1, 2], [3, 4]]))
        assert result.shape == (2, 2)

    def test_is_multioutput_false(self):
        """Test is_multioutput for single output."""
        result = PredictResult(y_pred=np.array([1, 2, 3]))
        assert result.is_multioutput is False

    def test_is_multioutput_true(self):
        """Test is_multioutput for multiple outputs."""
        result = PredictResult(y_pred=np.array([[1, 2], [3, 4]]))
        assert result.is_multioutput is True

    def test_len(self):
        """Test __len__ returns number of samples."""
        result = PredictResult(y_pred=np.array([1, 2, 3, 4, 5]))
        assert len(result) == 5

    def test_len_none(self):
        """Test __len__ returns 0 for None y_pred."""
        result = PredictResult(y_pred=None)
        assert len(result) == 0

    def test_to_numpy(self):
        """Test to_numpy returns numpy array."""
        y_pred = np.array([1.0, 2.0, 3.0])
        result = PredictResult(y_pred=y_pred)
        assert np.array_equal(result.to_numpy(), y_pred)

    def test_to_list(self):
        """Test to_list returns Python list."""
        result = PredictResult(y_pred=np.array([1.0, 2.0, 3.0]))
        assert result.to_list() == [1.0, 2.0, 3.0]

    def test_to_list_2d(self):
        """Test to_list flattens 2D array."""
        result = PredictResult(y_pred=np.array([[1, 2], [3, 4]]))
        assert result.to_list() == [1, 2, 3, 4]

    def test_flatten(self):
        """Test flatten returns 1D array."""
        result = PredictResult(y_pred=np.array([[1, 2], [3, 4]]))
        flattened = result.flatten()
        assert flattened.shape == (4,)
        assert list(flattened) == [1, 2, 3, 4]

    def test_to_dataframe_1d(self):
        """Test to_dataframe for 1D predictions."""
        pytest.importorskip('pandas')
        import pandas as pd

        result = PredictResult(y_pred=np.array([1.0, 2.0, 3.0]))
        df = result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert 'y_pred' in df.columns
        assert len(df) == 3

    def test_to_dataframe_2d(self):
        """Test to_dataframe for 2D predictions."""
        pytest.importorskip('pandas')
        import pandas as pd

        result = PredictResult(y_pred=np.array([[1, 2], [3, 4], [5, 6]]))
        df = result.to_dataframe()

        assert 'y_pred_0' in df.columns
        assert 'y_pred_1' in df.columns
        assert len(df) == 3

    def test_to_dataframe_with_indices(self):
        """Test to_dataframe includes sample indices."""
        pytest.importorskip('pandas')

        result = PredictResult(
            y_pred=np.array([1.0, 2.0]),
            sample_indices=np.array([10, 20])
        )
        df = result.to_dataframe(include_indices=True)

        assert 'sample_index' in df.columns
        assert list(df['sample_index']) == [10, 20]

    def test_metadata(self):
        """Test metadata storage."""
        result = PredictResult(
            y_pred=np.array([1.0]),
            metadata={'timing': 0.5, 'uncertainty': [0.1]}
        )
        assert result.metadata['timing'] == 0.5

    def test_model_name(self):
        """Test model_name attribute."""
        result = PredictResult(
            y_pred=np.array([1.0]),
            model_name='PLSRegression'
        )
        assert result.model_name == 'PLSRegression'

    def test_preprocessing_steps(self):
        """Test preprocessing_steps attribute."""
        result = PredictResult(
            y_pred=np.array([1.0]),
            preprocessing_steps=['MinMaxScaler', 'SNV']
        )
        assert result.preprocessing_steps == ['MinMaxScaler', 'SNV']

    def test_repr(self):
        """Test __repr__ format."""
        result = PredictResult(
            y_pred=np.array([1.0, 2.0]),
            model_name='PLS'
        )
        repr_str = repr(result)
        assert 'PredictResult' in repr_str
        assert 'PLS' in repr_str

    def test_str(self):
        """Test __str__ format."""
        result = PredictResult(
            y_pred=np.array([1.0, 2.0]),
            model_name='PLS',
            preprocessing_steps=['SNV']
        )
        str_output = str(result)
        assert 'PredictResult' in str_output
        assert 'PLS' in str_output


# =============================================================================
# ExplainResult Tests
# =============================================================================

class TestExplainResult:
    """Tests for ExplainResult dataclass."""

    def test_init_with_array(self):
        """Test initialization with numpy array."""
        shap_values = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = ExplainResult(shap_values=shap_values)
        assert np.array_equal(result.shap_values, shap_values)

    def test_init_with_shap_explanation(self):
        """Test initialization extracts metadata from shap.Explanation-like object."""
        mock_explanation = Mock()
        mock_explanation.values = np.array([[0.1, 0.2]])
        mock_explanation.feature_names = ['feat_a', 'feat_b']
        mock_explanation.base_values = np.array([0.5])

        result = ExplainResult(shap_values=mock_explanation)

        assert result.feature_names == ['feat_a', 'feat_b']
        assert np.array_equal(result.base_value, np.array([0.5]))
        assert result.n_samples == 1

    def test_values_property_array(self):
        """Test values property with raw array."""
        shap_values = np.array([[0.1, 0.2]])
        result = ExplainResult(shap_values=shap_values)
        assert np.array_equal(result.values, shap_values)

    def test_values_property_explanation(self):
        """Test values property extracts from Explanation object."""
        mock = Mock()
        mock.values = np.array([[1.0, 2.0]])
        # Set feature_names to None to prevent iteration in __post_init__
        del mock.feature_names

        result = ExplainResult(shap_values=mock)
        assert np.array_equal(result.values, np.array([[1.0, 2.0]]))

    def test_shape(self):
        """Test shape property."""
        result = ExplainResult(shap_values=np.array([[0.1, 0.2, 0.3]]))
        assert result.shape == (1, 3)

    def test_mean_abs_shap(self):
        """Test mean_abs_shap calculation."""
        shap_values = np.array([
            [0.1, -0.2, 0.3],
            [-0.1, 0.2, -0.3]
        ])
        result = ExplainResult(shap_values=shap_values)

        expected = np.array([0.1, 0.2, 0.3])  # Mean of absolute values
        assert np.allclose(result.mean_abs_shap, expected)

    def test_top_features_with_names(self):
        """Test top_features with feature names."""
        shap_values = np.array([[0.1, 0.5, 0.3]])
        result = ExplainResult(
            shap_values=shap_values,
            feature_names=['low', 'high', 'mid']
        )

        top = result.top_features
        assert top[0] == 'high'
        assert top[1] == 'mid'
        assert top[2] == 'low'

    def test_top_features_without_names(self):
        """Test top_features returns indices as strings."""
        shap_values = np.array([[0.1, 0.5, 0.3]])
        result = ExplainResult(shap_values=shap_values)

        top = result.top_features
        assert top[0] == '1'  # Index of highest importance

    def test_get_feature_importance(self):
        """Test get_feature_importance returns dict."""
        shap_values = np.array([[0.1, 0.5, 0.3]])
        result = ExplainResult(
            shap_values=shap_values,
            feature_names=['a', 'b', 'c']
        )

        importance = result.get_feature_importance()
        assert 'b' in importance
        assert importance['b'] == 0.5

    def test_get_feature_importance_top_n(self):
        """Test get_feature_importance with top_n limit."""
        shap_values = np.array([[0.1, 0.5, 0.3]])
        result = ExplainResult(
            shap_values=shap_values,
            feature_names=['a', 'b', 'c']
        )

        importance = result.get_feature_importance(top_n=2)
        assert len(importance) == 2
        assert 'b' in importance
        assert 'c' in importance

    def test_get_feature_importance_normalized(self):
        """Test get_feature_importance with normalization."""
        shap_values = np.array([[0.2, 0.3, 0.5]])
        result = ExplainResult(
            shap_values=shap_values,
            feature_names=['a', 'b', 'c']
        )

        importance = result.get_feature_importance(normalize=True)
        total = sum(importance.values())
        assert np.isclose(total, 1.0)

    def test_get_sample_explanation(self):
        """Test get_sample_explanation for single sample."""
        shap_values = np.array([
            [0.1, 0.2],
            [0.3, 0.4]
        ])
        result = ExplainResult(
            shap_values=shap_values,
            feature_names=['feat_a', 'feat_b']
        )

        sample_exp = result.get_sample_explanation(0)
        assert sample_exp['feat_a'] == 0.1
        assert sample_exp['feat_b'] == 0.2

        sample_exp = result.get_sample_explanation(1)
        assert sample_exp['feat_a'] == 0.3
        assert sample_exp['feat_b'] == 0.4

    def test_get_sample_explanation_out_of_range(self):
        """Test get_sample_explanation raises for invalid index."""
        result = ExplainResult(shap_values=np.array([[0.1, 0.2]]))

        with pytest.raises(IndexError):
            result.get_sample_explanation(5)

    def test_to_dataframe(self):
        """Test to_dataframe returns pandas DataFrame."""
        pytest.importorskip('pandas')
        import pandas as pd

        shap_values = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = ExplainResult(
            shap_values=shap_values,
            feature_names=['a', 'b']
        )

        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['a', 'b']
        assert len(df) == 2

    def test_to_dataframe_without_names(self):
        """Test to_dataframe with auto-generated column names."""
        pytest.importorskip('pandas')

        result = ExplainResult(shap_values=np.array([[0.1, 0.2]]))
        df = result.to_dataframe(include_feature_names=False)

        assert 'feature_0' in df.columns
        assert 'feature_1' in df.columns

    def test_visualizations(self):
        """Test visualizations attribute."""
        result = ExplainResult(
            shap_values=np.array([[0.1]]),
            visualizations={
                'summary': Path('/tmp/summary.png'),
                'bar': Path('/tmp/bar.png')
            }
        )
        assert 'summary' in result.visualizations
        assert result.visualizations['summary'] == Path('/tmp/summary.png')

    def test_explainer_type(self):
        """Test explainer_type attribute."""
        result = ExplainResult(
            shap_values=np.array([[0.1]]),
            explainer_type='tree'
        )
        assert result.explainer_type == 'tree'

    def test_model_name(self):
        """Test model_name attribute."""
        result = ExplainResult(
            shap_values=np.array([[0.1]]),
            model_name='RandomForest'
        )
        assert result.model_name == 'RandomForest'

    def test_repr(self):
        """Test __repr__ format."""
        result = ExplainResult(
            shap_values=np.array([[0.1, 0.2]]),
            explainer_type='kernel'
        )
        repr_str = repr(result)
        assert 'ExplainResult' in repr_str
        assert 'kernel' in repr_str

    def test_str(self):
        """Test __str__ format."""
        result = ExplainResult(
            shap_values=np.array([[0.1, 0.2]]),
            feature_names=['a', 'b'],
            model_name='PLS',
            explainer_type='auto',
            n_samples=10
        )
        str_output = str(result)
        assert 'ExplainResult' in str_output
        assert 'PLS' in str_output
        assert 'samples' in str_output


# =============================================================================
# Integration Tests
# =============================================================================

class TestResultIntegration:
    """Integration tests for result classes."""

    def test_run_result_end_to_end(self, mock_predictions, mock_runner):
        """Test complete RunResult workflow."""
        result = RunResult(
            predictions=mock_predictions,
            per_dataset={'wheat': {}},
            _runner=mock_runner
        )

        # Access best
        assert result.best is not None

        # Get metrics
        _ = result.best_score

        # Query
        _ = result.top(n=5)
        _ = result.get_datasets()
        _ = result.get_models()

        # Summary
        summary = result.summary()
        assert isinstance(summary, str)

    def test_predict_result_end_to_end(self):
        """Test complete PredictResult workflow."""
        y_pred = np.random.randn(100, 2)
        indices = np.arange(100)

        result = PredictResult(
            y_pred=y_pred,
            sample_indices=indices,
            model_name='TestModel',
            preprocessing_steps=['Scaler', 'PCA']
        )

        # Access values
        assert result.values.shape == (100, 2)
        assert result.is_multioutput

        # Convert
        assert len(result.to_list()) == 200  # Flattened
        assert result.flatten().shape == (200,)

        # Str representations
        assert 'TestModel' in str(result)

    def test_explain_result_end_to_end(self):
        """Test complete ExplainResult workflow."""
        shap_values = np.random.randn(50, 10)
        feature_names = [f'wavelength_{i}' for i in range(10)]

        result = ExplainResult(
            shap_values=shap_values,
            feature_names=feature_names,
            base_value=0.5,
            model_name='PLSRegression',
            explainer_type='kernel',
            n_samples=50
        )

        # Access values
        assert result.values.shape == (50, 10)
        assert len(result.mean_abs_shap) == 10

        # Feature importance
        importance = result.get_feature_importance(top_n=5, normalize=True)
        assert len(importance) == 5
        # Note: top_n=5 of 10 features with normalization means sum < 1.0
        assert sum(importance.values()) <= 1.0

        # Full importance should sum to 1.0
        full_importance = result.get_feature_importance(normalize=True)
        assert np.isclose(sum(full_importance.values()), 1.0, atol=0.01)

        # Sample explanation
        sample_exp = result.get_sample_explanation(0)
        assert len(sample_exp) == 10

        # Str representations
        assert 'PLSRegression' in str(result)
