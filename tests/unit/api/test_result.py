"""
Unit tests for nirs4all.api.result module.

Tests the RunResult, PredictResult, and ExplainResult dataclasses.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

import nirs4all
from nirs4all.api.result import ExplainResult, PredictResult, RunResult
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.dagml.tuning_contracts import TrialResult, TuningResult, parse_tuning_spec
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

# =============================================================================
# Fixtures
# =============================================================================


def _relation_replay_manifest() -> dict:
    return {
        "version": "1.0",
        "fingerprint": "rel-fp",
        "materialization_manifest": {
            "representation": "stack_padded_masked",
            "fingerprint": "mat-fp",
            "shape": [1, 1],
            "model_shape": [1, 2],
            "headers": ["MIR:1000"],
            "model_headers": ["MIR:1000", "mask:MIR:1000"],
            "source_ids": ["MIR"],
            "representation_plan": {
                "representation": "stack_padded_masked",
                "unit_level": "sample",
                "stage": "stack",
            },
        },
    }


def _tuning_result() -> TuningResult:
    tuning = parse_tuning_spec(
        {
            "engine": "optuna",
            "space": {"model.n_components": [2, 3]},
            "metric": "rmse",
            "direction": "minimize",
            "n_trials": 2,
            "sampler": "grid",
        }
    )
    return TuningResult(
        tuning=tuning,
        best_params={"model.n_components": 2},
        best_value=0.12,
        trials=(
            TrialResult(number=0, params={"model.n_components": 3}, value=0.2, state="COMPLETE", diagnostics={}),
            TrialResult(number=1, params={"model.n_components": 2}, value=0.12, state="COMPLETE", diagnostics={}),
        ),
        optimizer="optuna",
    )


@pytest.fixture
def mock_predictions():
    """Create a mock Predictions object with sample data."""
    mock = Mock()

    # Sample prediction entries
    sample_entries = [
        {
            "id": "pred_001",
            "model_name": "PLSRegression",
            "dataset_name": "wheat",
            "test_score": 0.85,
            "val_score": 0.82,
            "metric": "rmse",
            "scores": {"test": {"rmse": 0.85, "r2": 0.92, "mae": 0.65}, "val": {"rmse": 0.82, "r2": 0.91, "mae": 0.62}},
            "fold_id": "0",
            "step_idx": 2,
        },
        {
            "id": "pred_002",
            "model_name": "RandomForest",
            "dataset_name": "wheat",
            "test_score": 0.90,
            "val_score": 0.88,
            "metric": "rmse",
            "scores": {
                "test": {"rmse": 0.90, "r2": 0.88, "mae": 0.70},
            },
            "fold_id": "0",
            "step_idx": 2,
        },
    ]

    mock.top.return_value = sample_entries
    mock.num_predictions = 2
    mock.get_datasets.return_value = ["wheat"]
    mock.get_models.return_value = ["PLSRegression", "RandomForest"]
    mock.filter_predictions.return_value = sample_entries
    mock.get_best.return_value = None  # Falls back to self.best via top()

    return mock


@pytest.fixture
def mock_runner():
    """Create a mock PipelineRunner."""
    runner = Mock()
    runner.workspace_path = Path("/tmp/workspace")
    runner.export.return_value = Path("/tmp/exports/model.n4a")
    runner.export_model.return_value = Path("/tmp/exports/model.joblib")
    return runner


@pytest.fixture
def run_result(mock_predictions, mock_runner):
    """Create a RunResult instance with mocks."""
    return RunResult(predictions=mock_predictions, per_dataset={"wheat": {"status": "success"}}, _runner=mock_runner)


# =============================================================================
# RunResult Tests
# =============================================================================


class TestRunResult:
    """Tests for RunResult dataclass."""

    def test_init(self, mock_predictions):
        """Test basic initialization."""
        result = RunResult(predictions=mock_predictions, per_dataset={"test": "data"})
        assert result.predictions == mock_predictions
        assert result.per_dataset == {"test": "data"}
        assert result._runner is None

    def test_best_property(self, run_result, mock_predictions):
        """Test best property returns final entry when available."""
        mock_predictions.top.return_value = [{"id": "best_model"}]
        assert run_result.best == {"id": "best_model"}
        # best calls best_final first (score_scope="refit")
        mock_predictions.top.assert_any_call(n=1, score_scope="refit")

    def test_best_property_empty(self, mock_predictions):
        """Test best property returns empty dict when no predictions."""
        mock_predictions.top.return_value = []
        result = RunResult(predictions=mock_predictions, per_dataset={})
        assert result.best == {}

    def test_best_score(self, run_result, mock_predictions):
        """Test best_score extracts test_score from best."""
        mock_predictions.top.return_value = [{"test_score": 0.75}]
        assert run_result.best_score == 0.75

    def test_best_score_missing(self, mock_predictions):
        """Test best_score returns NaN when not available."""
        mock_predictions.top.return_value = [{"model_name": "test"}]
        result = RunResult(predictions=mock_predictions, per_dataset={})
        assert np.isnan(result.best_score)

    def test_best_rmse_from_scores(self, run_result, mock_predictions):
        """Test best_rmse extracts from scores dict."""
        mock_predictions.top.return_value = [{"scores": {"test": {"rmse": 0.42}}}]
        assert run_result.best_rmse == 0.42

    def test_best_rmse_fallback_to_test_score(self, mock_predictions):
        """Test best_rmse falls back to test_score when metric is rmse."""
        mock_predictions.top.return_value = [{"metric": "rmse", "test_score": 0.55}]
        result = RunResult(predictions=mock_predictions, per_dataset={})
        assert result.best_rmse == 0.55

    def test_best_r2(self, run_result, mock_predictions):
        """Test best_r2 extracts from scores dict."""
        mock_predictions.top.return_value = [{"scores": {"test": {"r2": 0.95}}}]
        assert run_result.best_r2 == 0.95

    def test_best_accuracy(self, mock_predictions):
        """Test best_accuracy for classification results."""
        mock_predictions.top.return_value = [{"scores": {"test": {"accuracy": 0.88}}}]
        result = RunResult(predictions=mock_predictions, per_dataset={})
        assert result.best_accuracy == 0.88

    def test_metric_shortcuts_anchor_on_selected_model(self, mock_predictions):
        """best_rmse / best_r2 / best_accuracy ALL read from the SAME selected entry.

        Locks the invariant: the scalar shortcuts describe the model ``best``/``best_score``
        selects, NOT a per-metric re-ranked row. ``get_best`` is configured to return a DIFFERENT
        (lower-rmse / higher-accuracy) entry; if any shortcut re-ranked via ``get_best`` it would
        return that decoy's value instead of the selected entry's.
        """
        selected = {
            "test_score": 13.5,
            "metric": "rmse",
            "scores": {"test": {"rmse": 13.5, "r2": 0.55, "accuracy": 0.20}},
        }
        decoy = {
            "test_score": 11.0,
            "metric": "rmse",
            "scores": {"test": {"rmse": 11.0, "r2": 0.99, "accuracy": 0.90}},
        }
        # ``best`` (and thus best_score) resolves the selected entry via top(); get_best returns a
        # decoy that the OLD per-metric-rerank code would have surfaced.
        mock_predictions.top.return_value = [selected]
        mock_predictions.get_best.return_value = decoy
        result = RunResult(predictions=mock_predictions, per_dataset={})

        assert result.best_score == 13.5
        assert result.best_rmse == 13.5  # selected entry's rmse, NOT the decoy 11.0
        assert result.best_r2 == 0.55  # selected entry's r2, NOT the decoy 0.99
        assert result.best_accuracy == 0.20  # selected entry's accuracy, NOT the decoy 0.90

    def test_artifacts_path(self, run_result):
        """Test artifacts_path returns runner's workspace_path."""
        assert run_result.artifacts_path == Path("/tmp/workspace")

    def test_artifacts_path_no_runner(self, mock_predictions):
        """Test artifacts_path returns None when no runner."""
        result = RunResult(predictions=mock_predictions, per_dataset={})
        assert result.artifacts_path is None

    def test_num_predictions(self, run_result, mock_predictions):
        """Test num_predictions delegates to predictions."""
        assert run_result.num_predictions == 2

    def test_tuning_result_accessors_do_not_fabricate_predictions(self, mock_predictions):
        """RunResult can carry native tuning evidence without prediction rows."""
        mock_predictions.num_predictions = 0
        result = RunResult(
            predictions=mock_predictions,
            per_dataset={},
            _tuning_result=_tuning_result(),
            _tuning_id="tune-main",
        )

        assert result.num_predictions == 0
        assert result.tuning_id == "tune-main"
        assert result.tuning_result is not None
        assert result.tuning_best_params == {"model.n_components": 2}
        assert result.tuning_best_value == pytest.approx(0.12)
        assert "Tuning:" in result.summary()
        assert result.validate(raise_on_failure=False)["valid"] is True

    def test_top(self, run_result, mock_predictions):
        """Test top() delegates to predictions.top()."""
        run_result.top(n=10, rank_metric="r2")
        mock_predictions.top.assert_called_with(n=10, rank_metric="r2")

    def test_filter(self, run_result, mock_predictions):
        """Test filter() delegates to predictions.filter_predictions()."""
        run_result.filter(model_name="PLS", partition="test")
        mock_predictions.filter_predictions.assert_called_with(model_name="PLS", partition="test")

    def test_get_datasets(self, run_result, mock_predictions):
        """Test get_datasets delegates to predictions."""
        assert run_result.get_datasets() == ["wheat"]

    def test_get_models(self, run_result, mock_predictions):
        """Test get_models delegates to predictions."""
        assert run_result.get_models() == ["PLSRegression", "RandomForest"]

    def test_export(self, run_result, mock_runner, mock_predictions):
        """Test export() delegates to runner.export()."""
        mock_predictions.top.return_value = [{"id": "best"}]
        path = run_result.export("output/model.n4a")

        mock_runner.export.assert_called_with(source={"id": "best"}, output_path="output/model.n4a", format="n4a")
        assert path == Path("/tmp/exports/model.n4a")

    def test_export_with_source(self, run_result, mock_runner):
        """Test export() with explicit source."""
        source = {"id": "specific_model"}
        run_result.export("output/model.n4a", source=source)

        mock_runner.export.assert_called_with(source=source, output_path="output/model.n4a", format="n4a")

    def test_export_no_runner(self, mock_predictions):
        """Test export() raises when no runner or workspace path available."""
        result = RunResult(predictions=mock_predictions, per_dataset={})
        with pytest.raises(RuntimeError, match="no workspace path available"):
            result.export("output/model.n4a")

    def test_export_no_predictions(self, mock_runner, mock_predictions):
        """Test export() raises when no predictions and no source."""
        mock_predictions.top.return_value = []
        result = RunResult(predictions=mock_predictions, per_dataset={}, _runner=mock_runner)
        with pytest.raises(ValueError, match="No predictions available"):
            result.export("output/model.n4a")

    def test_export_model(self, run_result, mock_runner, mock_predictions):
        """Test export_model() delegates to runner.export_model()."""
        mock_predictions.top.return_value = [{"id": "best"}]
        path = run_result.export_model("output/model.joblib")

        mock_runner.export_model.assert_called_with(source={"id": "best"}, output_path="output/model.joblib", format=None, fold=None)
        assert path == Path("/tmp/exports/model.joblib")

    def test_summary(self, run_result, mock_predictions):
        """Test summary() returns formatted string."""
        mock_predictions.top.return_value = [{"model_name": "PLSRegression", "test_score": 0.85, "scores": {"test": {"rmse": 0.85, "r2": 0.92}}}]
        summary = run_result.summary()

        assert "RunResult" in summary
        assert "predictions" in summary
        assert "PLSRegression" in summary

    def test_repr(self, run_result, mock_predictions):
        """Test __repr__ format."""
        mock_predictions.top.return_value = [{"test_score": 0.85}]
        repr_str = repr(run_result)
        assert "RunResult" in repr_str
        assert "predictions=" in repr_str

    def test_str(self, run_result, mock_predictions):
        """Test __str__ is same as summary."""
        mock_predictions.top.return_value = [{"model_name": "test"}]
        assert str(run_result) == run_result.summary()

    def test_relation_lineage_from_prediction_metadata(self, mock_predictions):
        """RunResult exposes relation provenance attached to prediction rows."""
        manifest = _relation_replay_manifest()
        row = {
            "id": "pred_rel",
            "chain_id": "chain_rel",
            "model_name": "PLSRegression",
            "test_score": 0.5,
            "n_features": 2,
            "metadata": {"relation_replay_manifest": manifest},
        }
        mock_predictions.filter_predictions.return_value = [row]
        mock_predictions.top.return_value = [row]

        result = RunResult(predictions=mock_predictions, per_dataset={})

        assert result.relation_replay_manifest == manifest
        assert result.relation_materialization_manifest["fingerprint"] == "mat-fp"
        assert result.explanation_level == "stack"
        assert result.get_feature_lineage("mask:MIR:1000")["source_id"] == "MIR"
        assert result.get_feature_lineage(1)["feature_role"] == "presence_mask"


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
        pytest.importorskip("pandas")
        import pandas as pd

        result = PredictResult(y_pred=np.array([1.0, 2.0, 3.0]))
        df = result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "y_pred" in df.columns
        assert len(df) == 3

    def test_to_dataframe_2d(self):
        """Test to_dataframe for 2D predictions."""
        pytest.importorskip("pandas")
        import pandas as pd

        result = PredictResult(y_pred=np.array([[1, 2], [3, 4], [5, 6]]))
        df = result.to_dataframe()

        assert "y_pred_0" in df.columns
        assert "y_pred_1" in df.columns
        assert len(df) == 3

    def test_to_dataframe_with_indices(self):
        """Test to_dataframe includes sample indices."""
        pytest.importorskip("pandas")

        result = PredictResult(y_pred=np.array([1.0, 2.0]), sample_indices=np.array([10, 20]))
        df = result.to_dataframe(include_indices=True)

        assert "sample_index" in df.columns
        assert list(df["sample_index"]) == [10, 20]

    def test_intervals_are_accessible_by_materialized_coverage(self):
        """Test prediction interval accessors."""
        interval = SimpleNamespace(lower=np.array([0.8, 1.8]), upper=np.array([1.2, 2.2]))
        result = PredictResult(y_pred=np.array([1.0, 2.0]), intervals={0.8: interval})

        assert result.interval_coverages == (0.8,)
        assert result.interval(0.8) is interval
        with pytest.raises(KeyError, match="not materialized"):
            result.interval(0.9)
        assert "Intervals: 0.8" in str(result)

    def test_str_reports_conformal_guarantee_status(self):
        """Test __str__ exposes fail-loud conformal guarantee metadata."""
        result = PredictResult(
            y_pred=np.array([1.0, 2.0]),
            intervals={
                0.8: SimpleNamespace(
                    lower=np.array([0.8, 1.8]),
                    upper=np.array([1.2, 2.2]),
                )
            },
            metadata={
                "calibration_replay_source": {
                    "dataset_backed": False,
                    "kind": "replayed_arrays",
                    "requires_model_replay": False,
                    "route": "provided_arrays",
                    "version": 1,
                },
                "conformal_guarantee_status": {
                    "calibration_replay_source": {
                        "dataset_backed": True,
                        "kind": "dataset_predictor_bundle",
                        "predictor_bundle": "model.n4a",
                        "requires_model_replay": True,
                        "route": "nirs4all.predict",
                        "version": 1,
                    },
                    "coverage": [0.8],
                    "effective_engine": "nirs4all.python.replayed_array_apply",
                    "status": "active",
                },
            },
        )

        rendered = str(result)

        assert "Conformal guarantee: active" in rendered
        assert "engine=nirs4all.python.replayed_array_apply" in rendered
        assert "coverage=0.8" in rendered
        assert result.calibration_replay_source == {
            "dataset_backed": True,
            "kind": "dataset_predictor_bundle",
            "predictor_bundle": "model.n4a",
            "requires_model_replay": True,
            "route": "nirs4all.predict",
            "version": 1,
        }

    def test_calibration_replay_source_falls_back_to_top_level_metadata(self):
        """Test replay provenance remains available without a guarantee block."""
        result = PredictResult(
            y_pred=np.array([1.0]),
            metadata={
                "calibration_replay_source": {
                    "dataset_backed": False,
                    "kind": "replayed_arrays",
                    "requires_model_replay": False,
                    "route": "provided_arrays",
                    "version": 1,
                }
            },
        )

        assert result.calibration_replay_source == {
            "dataset_backed": False,
            "kind": "replayed_arrays",
            "requires_model_replay": False,
            "route": "provided_arrays",
            "version": 1,
        }

    def test_tuning_calibration_source_reads_direct_metadata(self):
        """Test native tuning calibration provenance is exposed without nested dict access."""
        result = PredictResult(
            y_pred=np.array([1.0]),
            metadata={
                "tuning_calibration_source": {
                    "source": "tuning.winner",
                    "score_data_role": "hpo_objective_only",
                    "score_data_used": False,
                }
            },
        )

        assert result.tuning_calibration_source == {
            "source": "tuning.winner",
            "score_data_role": "hpo_objective_only",
            "score_data_used": False,
        }

    def test_robustness_evidence_reads_direct_metadata(self):
        """Test published robustness evidence is exposed without nested dict access."""
        result = PredictResult(
            y_pred=np.array([1.0, 2.0]),
            metadata={
                "robustness_evidence": {
                    "X": "prediction_arrays.X",
                    "predictor_bundle": "model.n4a",
                    "publisher": "nirs4all-studio.run-driver",
                }
            },
        )

        assert result.robustness_evidence == {
            "X": "prediction_arrays.X",
            "predictor_bundle": "model.n4a",
            "publisher": "nirs4all-studio.run-driver",
        }
        assert result.spectral_replay_evidence_status == {
            "status": "needs_spectral_replay_evidence",
            "has_X_or_spectra": True,
            "has_executable_X_or_spectra": False,
            "has_predictor_bundle": True,
            "predictor_bundle": "model.n4a",
            "missing": ["row_aligned_executable_X_or_spectra"],
            "source": "metadata.robustness_evidence",
        }

    def test_robustness_evidence_falls_back_to_result_metadata(self):
        """Test store-shaped result_metadata robustness evidence remains public."""
        result = PredictResult(
            y_pred=np.array([1.0, 2.0]),
            metadata={
                "result_metadata": {
                    "robustness_evidence": {
                        "spectra": "prediction_arrays.spectra",
                        "model_path": "exported/model.n4a",
                    }
                }
            },
        )

        assert result.robustness_evidence == {
            "spectra": "prediction_arrays.spectra",
            "model_path": "exported/model.n4a",
        }
        assert result.spectral_replay_evidence_status["status"] == "needs_spectral_replay_evidence"
        assert result.spectral_replay_evidence_status["predictor_bundle"] == "exported/model.n4a"
        assert result.spectral_replay_evidence_status["source"] == "metadata.result_metadata.robustness_evidence"

    def test_spectral_replay_evidence_status_is_ready_with_executable_matrix(self):
        """Test readiness requires an actual row-aligned matrix, not just a marker."""
        result = PredictResult(
            y_pred=np.array([1.0, 2.0]),
            metadata={
                "X": np.array([[1.0, 10.0], [2.0, 20.0]]),
                "robustness_evidence": {
                    "X": "prediction_arrays.X",
                    "predictor_bundle": "model.n4a",
                },
            },
        )

        assert result.spectral_replay_evidence_status == {
            "status": "ready_for_spectral_replay",
            "has_X_or_spectra": True,
            "has_executable_X_or_spectra": True,
            "has_predictor_bundle": True,
            "predictor_bundle": "model.n4a",
            "missing": [],
            "source": "metadata.robustness_evidence",
        }

    def test_spectral_replay_evidence_status_is_fail_closed_when_incomplete(self):
        """Test spectral replay readiness does not infer missing proofs."""
        result = PredictResult(
            y_pred=np.array([1.0, 2.0]),
            metadata={
                "robustness_evidence": {
                    "X": "prediction_arrays.X",
                }
            },
        )

        assert result.spectral_replay_evidence_status == {
            "status": "needs_spectral_replay_evidence",
            "has_X_or_spectra": True,
            "has_executable_X_or_spectra": False,
            "has_predictor_bundle": False,
            "predictor_bundle": None,
            "missing": ["row_aligned_executable_X_or_spectra", "predictor_bundle"],
            "source": "metadata.robustness_evidence",
        }

    def test_spectral_replay_evidence_status_rejects_wrong_row_count(self):
        """Test executable spectral evidence must be row-aligned to y_pred."""
        result = PredictResult(
            y_pred=np.array([1.0, 2.0]),
            metadata={
                "X": np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
                "robustness_evidence": {
                    "X": "prediction_arrays.X",
                    "predictor_bundle": "model.n4a",
                },
            },
        )

        assert result.spectral_replay_evidence_status == {
            "status": "needs_spectral_replay_evidence",
            "has_X_or_spectra": True,
            "has_executable_X_or_spectra": False,
            "has_predictor_bundle": True,
            "predictor_bundle": "model.n4a",
            "missing": ["row_aligned_executable_X_or_spectra"],
            "source": "metadata.robustness_evidence",
        }

    def test_from_prediction_record_preserves_workspace_replay_evidence(self):
        """Test workspace-shaped prediction records convert to replay-ready results."""
        X = np.array([[1.0, 10.0], [2.0, 20.0]])
        record = {
            "prediction_id": "pred-001",
            "model_name": "PLSRegression",
            "preprocessings": '["SNV", "SavitzkyGolay"]',
            "y_pred": [1.1, 1.9],
            "sample_indices": [10, 20],
            "metadata": {"operator": "pipeline"},
            "X": X,
            "result_metadata": {
                "robustness_evidence": {
                    "X": "prediction_arrays.X",
                    "predictor_bundle": "models/pls.n4a",
                }
            },
        }

        result = PredictResult.from_prediction_record(record)

        np.testing.assert_allclose(result.y_pred, np.array([1.1, 1.9]))
        np.testing.assert_array_equal(result.sample_indices, np.array([10, 20]))
        assert result.model_name == "PLSRegression"
        assert result.preprocessing_steps == ["SNV", "SavitzkyGolay"]
        assert result.metadata["operator"] == "pipeline"
        np.testing.assert_allclose(result.metadata["X"], X)
        assert result.robustness_evidence == {
            "X": "prediction_arrays.X",
            "predictor_bundle": "models/pls.n4a",
        }
        assert result.spectral_replay_evidence_status == {
            "status": "ready_for_spectral_replay",
            "has_X_or_spectra": True,
            "has_executable_X_or_spectra": True,
            "has_predictor_bundle": True,
            "predictor_bundle": "models/pls.n4a",
            "missing": [],
            "source": "metadata.result_metadata.robustness_evidence",
        }

    def test_from_prediction_record_requires_loaded_prediction_arrays(self):
        """Test conversion fails loudly when records were loaded without arrays."""
        with pytest.raises(ValueError, match="load_arrays=True"):
            PredictResult.from_prediction_record(
                {
                    "prediction_id": "pred-no-arrays",
                    "model_name": "PLSRegression",
                }
            )

    def test_docstring_describes_conformal_interval_and_guarantee_accessors(self):
        """Test runtime help documents the conformal prediction surface."""
        doc = PredictResult.__doc__ or ""

        assert "intervals: Materialized conformal intervals keyed by coverage" in doc
        assert "interval_coverages: Materialized interval coverages" in doc
        assert "conformal_guarantee_status: Fail-loud guarantee metadata" in doc
        assert "calibration_replay_source: Provenance for conformal calibration prediction replay" in doc
        assert "tuning_calibration_source: Provenance for native tuning-driven calibration" in doc
        assert "robustness_evidence: Published robustness replay evidence metadata" in doc
        assert "spectral_replay_evidence_status: Fail-closed readiness diagnostic" in doc
        assert "from_prediction_record(record): Convert a workspace/store prediction record" in doc
        assert "interval(coverage): Get intervals for an already materialized coverage" in doc

    def test_to_dataframe_includes_prediction_intervals(self):
        """Test to_dataframe emits interval columns when present."""
        pytest.importorskip("pandas")

        result = PredictResult(
            y_pred=np.array([1.0, 2.0]),
            intervals={
                0.8: SimpleNamespace(
                    lower=np.array([0.8, 1.8]),
                    upper=np.array([1.2, 2.2]),
                )
            },
        )

        df = result.to_dataframe()

        assert list(df["interval_0.8_lower"]) == [0.8, 1.8]
        assert list(df["interval_0.8_upper"]) == [1.2, 2.2]

    def test_to_dataframe_rejects_interval_shape_mismatch(self):
        """Test interval arrays must align with predictions."""
        pytest.importorskip("pandas")

        result = PredictResult(
            y_pred=np.array([1.0, 2.0]),
            intervals={
                0.8: SimpleNamespace(
                    lower=np.array([0.8]),
                    upper=np.array([1.2]),
                )
            },
        )

        with pytest.raises(ValueError, match="must match y_pred shape"):
            result.to_dataframe()

    def test_metadata(self):
        """Test metadata storage."""
        result = PredictResult(y_pred=np.array([1.0]), metadata={"timing": 0.5, "uncertainty": [0.1]})
        assert result.metadata["timing"] == 0.5

    def test_relation_lineage_from_metadata(self):
        """Test relation provenance accessors derived from prediction metadata."""
        manifest = _relation_replay_manifest()
        result = PredictResult(y_pred=np.array([1.0]), metadata={"relation_replay_manifest": manifest})

        assert result.relation_replay_manifest == manifest
        assert result.relation_materialization_manifest["fingerprint"] == "mat-fp"
        assert result.explanation_level == "stack"
        assert result.get_feature_lineage("MIR:1000")["source_feature"] == "1000"
        assert result.get_feature_lineage(1)["feature_role"] == "presence_mask"
        assert "Relation provenance: available" in str(result)

    def test_model_name(self):
        """Test model_name attribute."""
        result = PredictResult(y_pred=np.array([1.0]), model_name="PLSRegression")
        assert result.model_name == "PLSRegression"

    def test_preprocessing_steps(self):
        """Test preprocessing_steps attribute."""
        result = PredictResult(y_pred=np.array([1.0]), preprocessing_steps=["MinMaxScaler", "SNV"])
        assert result.preprocessing_steps == ["MinMaxScaler", "SNV"]

    def test_repr(self):
        """Test __repr__ format."""
        result = PredictResult(y_pred=np.array([1.0, 2.0]), model_name="PLS")
        repr_str = repr(result)
        assert "PredictResult" in repr_str
        assert "PLS" in repr_str

    def test_str(self):
        """Test __str__ format."""
        result = PredictResult(y_pred=np.array([1.0, 2.0]), model_name="PLS", preprocessing_steps=["SNV"])
        str_output = str(result)
        assert "PredictResult" in str_output
        assert "PLS" in str_output


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
        mock_explanation.feature_names = ["feat_a", "feat_b"]
        mock_explanation.base_values = np.array([0.5])

        result = ExplainResult(shap_values=mock_explanation)

        assert result.feature_names == ["feat_a", "feat_b"]
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
        shap_values = np.array([[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]])
        result = ExplainResult(shap_values=shap_values)

        expected = np.array([0.1, 0.2, 0.3])  # Mean of absolute values
        assert np.allclose(result.mean_abs_shap, expected)

    def test_top_features_with_names(self):
        """Test top_features with feature names."""
        shap_values = np.array([[0.1, 0.5, 0.3]])
        result = ExplainResult(shap_values=shap_values, feature_names=["low", "high", "mid"])

        top = result.top_features
        assert top[0] == "high"
        assert top[1] == "mid"
        assert top[2] == "low"

    def test_top_features_without_names(self):
        """Test top_features returns indices as strings."""
        shap_values = np.array([[0.1, 0.5, 0.3]])
        result = ExplainResult(shap_values=shap_values)

        top = result.top_features
        assert top[0] == "1"  # Index of highest importance

    def test_get_feature_importance(self):
        """Test get_feature_importance returns dict."""
        shap_values = np.array([[0.1, 0.5, 0.3]])
        result = ExplainResult(shap_values=shap_values, feature_names=["a", "b", "c"])

        importance = result.get_feature_importance()
        assert "b" in importance
        assert importance["b"] == 0.5

    def test_get_feature_importance_top_n(self):
        """Test get_feature_importance with top_n limit."""
        shap_values = np.array([[0.1, 0.5, 0.3]])
        result = ExplainResult(shap_values=shap_values, feature_names=["a", "b", "c"])

        importance = result.get_feature_importance(top_n=2)
        assert len(importance) == 2
        assert "b" in importance
        assert "c" in importance

    def test_get_feature_importance_normalized(self):
        """Test get_feature_importance with normalization."""
        shap_values = np.array([[0.2, 0.3, 0.5]])
        result = ExplainResult(shap_values=shap_values, feature_names=["a", "b", "c"])

        importance = result.get_feature_importance(normalize=True)
        total = sum(importance.values())
        assert np.isclose(total, 1.0)

    def test_get_sample_explanation(self):
        """Test get_sample_explanation for single sample."""
        shap_values = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = ExplainResult(shap_values=shap_values, feature_names=["feat_a", "feat_b"])

        sample_exp = result.get_sample_explanation(0)
        assert sample_exp["feat_a"] == 0.1
        assert sample_exp["feat_b"] == 0.2

        sample_exp = result.get_sample_explanation(1)
        assert sample_exp["feat_a"] == 0.3
        assert sample_exp["feat_b"] == 0.4

    def test_get_sample_explanation_out_of_range(self):
        """Test get_sample_explanation raises for invalid index."""
        result = ExplainResult(shap_values=np.array([[0.1, 0.2]]))

        with pytest.raises(IndexError):
            result.get_sample_explanation(5)

    def test_to_dataframe(self):
        """Test to_dataframe returns pandas DataFrame."""
        pytest.importorskip("pandas")
        import pandas as pd

        shap_values = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = ExplainResult(shap_values=shap_values, feature_names=["a", "b"])

        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2

    def test_to_dataframe_without_names(self):
        """Test to_dataframe with auto-generated column names."""
        pytest.importorskip("pandas")

        result = ExplainResult(shap_values=np.array([[0.1, 0.2]]))
        df = result.to_dataframe(include_feature_names=False)

        assert "feature_0" in df.columns
        assert "feature_1" in df.columns

    def test_visualizations(self):
        """Test visualizations attribute."""
        result = ExplainResult(shap_values=np.array([[0.1]]), visualizations={"summary": Path("/tmp/summary.png"), "bar": Path("/tmp/bar.png")})
        assert "summary" in result.visualizations
        assert result.visualizations["summary"] == Path("/tmp/summary.png")

    def test_explainer_type(self):
        """Test explainer_type attribute."""
        result = ExplainResult(shap_values=np.array([[0.1]]), explainer_type="tree")
        assert result.explainer_type == "tree"

    def test_model_name(self):
        """Test model_name attribute."""
        result = ExplainResult(shap_values=np.array([[0.1]]), model_name="RandomForest")
        assert result.model_name == "RandomForest"

    def test_feature_lineage_by_name_and_index(self):
        """Test relation lineage lookup for explained features."""
        result = ExplainResult(
            shap_values=np.array([[0.1, 0.2]]),
            feature_names=["mir_mean_1200", "raman_mean_800"],
            explanation_level="source_aggregate",
            feature_lineage={
                "mir_mean_1200": {
                    "source_id": "MIR",
                    "component_observation_ids": ["mir:s1:1", "mir:s1:2"],
                }
            },
            lineage_warning="Explaining aggregated features.",
        )

        assert result.get_feature_lineage("mir_mean_1200")["source_id"] == "MIR"
        assert result.get_feature_lineage(0)["component_observation_ids"] == ["mir:s1:1", "mir:s1:2"]
        assert result.get_feature_lineage("missing") == {}
        assert result.explanation_level == "source_aggregate"
        assert result.lineage_warning == "Explaining aggregated features."

    def test_repr(self):
        """Test __repr__ format."""
        result = ExplainResult(shap_values=np.array([[0.1, 0.2]]), explainer_type="kernel")
        repr_str = repr(result)
        assert "ExplainResult" in repr_str
        assert "kernel" in repr_str

    def test_str(self):
        """Test __str__ format."""
        result = ExplainResult(shap_values=np.array([[0.1, 0.2]]), feature_names=["a", "b"], model_name="PLS", explainer_type="auto", n_samples=10)
        str_output = str(result)
        assert "ExplainResult" in str_output
        assert "PLS" in str_output
        assert "samples" in str_output

    def test_str_includes_relation_explainability_metadata(self):
        """Test string output mentions optional relation explainability metadata."""
        result = ExplainResult(
            shap_values=np.array([[0.1, 0.2]]),
            feature_names=["a", "b"],
            explanation_level="source_aggregate",
            feature_lineage={"a": {"source_id": "MIR"}},
            lineage_warning="Aggregated features.",
            n_samples=1,
        )

        str_output = str(result)
        assert "Explanation level: source_aggregate" in str_output
        assert "Feature lineage: available" in str_output
        assert "Lineage warning: Aggregated features." in str_output


# =============================================================================
# Integration Tests
# =============================================================================


class TestResultIntegration:
    """Integration tests for result classes."""

    def test_run_result_end_to_end(self, mock_predictions, mock_runner):
        """Test complete RunResult workflow."""
        result = RunResult(predictions=mock_predictions, per_dataset={"wheat": {}}, _runner=mock_runner)

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

        result = PredictResult(y_pred=y_pred, sample_indices=indices, model_name="TestModel", preprocessing_steps=["Scaler", "PCA"])

        # Access values
        assert result.values.shape == (100, 2)
        assert result.is_multioutput

        # Convert
        assert len(result.to_list()) == 200  # Flattened
        assert result.flatten().shape == (200,)

        # Str representations
        assert "TestModel" in str(result)

    def test_load_workspace_predict_result_converts_stored_prediction(self, tmp_path):
        """Public helper loads a workspace prediction as native PredictResult."""
        workspace = tmp_path / "workspace"
        store = WorkspaceStore(workspace)
        try:
            run_id = store.begin_run("run", config={"metric": "rmse"}, datasets=[{"name": "wheat"}])
            pipeline_id = store.begin_pipeline(
                run_id=run_id,
                name="pls",
                expanded_config=[{"model": "PLSRegression"}],
                generator_choices=[],
                dataset_name="wheat",
                dataset_hash="sha256:wheat",
            )
            chain_id = store.save_chain(
                pipeline_id=pipeline_id,
                steps=[{"step_idx": 0, "operator_class": "PLSRegression", "params": {}, "artifact_id": None, "stateless": False}],
                model_step_idx=0,
                model_class="sklearn.cross_decomposition.PLSRegression",
                preprocessings='["SNV"]',
                fold_strategy="final",
                fold_artifacts={},
                shared_artifacts={},
            )
            predictions = Predictions(store=store)
            prediction_id = predictions.add_prediction(
                dataset_name="wheat",
                model_name="PLSRegression",
                model_classname="sklearn.cross_decomposition.PLSRegression",
                fold_id="final",
                partition="test",
                sample_indices=np.asarray([0, 1], dtype=np.int64),
                metadata={"tuning_calibration_source": {"source": "tuning.winner"}},
                y_true=np.asarray([1.0, 2.0], dtype=float),
                y_pred=np.asarray([1.1, 1.9], dtype=float),
                test_score=0.1,
                metric="rmse",
                task_type="regression",
                n_samples=2,
                n_features=100,
                preprocessings='["SNV"]',
                scores={"test": {"rmse": 0.1}},
                best_params={"n_components": 4},
            )
            predictions.flush(pipeline_id=pipeline_id, chain_id=chain_id)
        finally:
            store.close()

        result = nirs4all.load_workspace_predict_result(workspace, prediction_id)
        results = nirs4all.load_workspace_predict_results(workspace)

        assert isinstance(result, PredictResult)
        np.testing.assert_allclose(result.y_pred, [1.1, 1.9])
        np.testing.assert_array_equal(result.sample_indices, [0, 1])
        assert result.model_name == "PLSRegression"
        assert result.preprocessing_steps == ["SNV"]
        assert result.tuning_calibration_source == {"source": "tuning.winner"}
        assert len(results) == 1
        assert all(isinstance(item, PredictResult) for item in results)
        np.testing.assert_allclose(results[0].y_pred, [1.1, 1.9])

        with pytest.raises(KeyError, match="workspace prediction not found"):
            nirs4all.load_workspace_predict_result(workspace, "missing-prediction")

    def test_save_workspace_predict_result_publishes_executable_evidence(self, tmp_path):
        """Public helper saves PredictResult rows with spectral/OOD evidence."""
        workspace = tmp_path / "workspace"
        X = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        source = PredictResult(
            y_pred=np.asarray([1.1, 1.9], dtype=float),
            metadata={
                "X": X,
                "robustness_evidence": {"predictor_bundle": "bundle.n4a"},
                "tuning_calibration_source": {"source": "tuning.winner"},
            },
            sample_indices=np.asarray([10, 11], dtype=np.int64),
            model_name="PLSRegression",
            preprocessing_steps=["SNV"],
        )

        prediction_id = nirs4all.save_workspace_predict_result(
            workspace,
            source,
            dataset_name="wheat",
            result_metadata={"publisher": "nirs4all.save_workspace_predict_result"},
            task_type="regression",
        )
        restored = nirs4all.load_workspace_predict_result(workspace, prediction_id)

        assert isinstance(restored, PredictResult)
        np.testing.assert_allclose(restored.y_pred, source.y_pred)
        np.testing.assert_array_equal(restored.sample_indices, [10, 11])
        np.testing.assert_allclose(restored.metadata["X"], X)
        assert restored.model_name == "PLSRegression"
        assert restored.preprocessing_steps == ["SNV"]
        assert restored.tuning_calibration_source == {"source": "tuning.winner"}
        assert restored.robustness_evidence == {"predictor_bundle": "bundle.n4a"}
        assert restored.metadata["result_metadata"]["publisher"] == "nirs4all.save_workspace_predict_result"
        assert restored.spectral_replay_evidence_status["status"] == "ready_for_spectral_replay"
        assert restored.spectral_replay_evidence_status["has_executable_X_or_spectra"] is True

    def test_save_workspace_predict_result_metadata_is_strict_json_native(self, tmp_path):
        """Prediction workspace metadata fails closed before sidecar persistence."""

        workspace = tmp_path / "workspace"
        source = PredictResult(
            y_pred=np.asarray([1.1, 1.9], dtype=float),
            sample_indices=np.asarray([10, 11], dtype=np.int64),
            model_name="PLSRegression",
        )

        prediction_id = nirs4all.save_workspace_predict_result(
            workspace,
            source,
            metadata={"site": "north", "nested": {"ok": [1, True, None]}},
            result_metadata={"publisher": "unit", "robustness_evidence": {"predictor_bundle": "model.n4a"}},
        )
        restored = nirs4all.load_workspace_predict_result(workspace, prediction_id)
        assert restored.metadata["site"] == "north"
        assert restored.metadata["nested"] == {"ok": [1, True, None]}
        assert restored.metadata["result_metadata"]["robustness_evidence"] == {"predictor_bundle": "model.n4a"}

        invalid_payloads = (
            {" bad": 1},
            {"bad": object()},
            {"bad": float("nan")},
            {"bad": (1, 2)},
        )
        for payload in invalid_payloads:
            with pytest.raises(ValueError, match=r"save_workspace_predict_result.metadata"):
                nirs4all.save_workspace_predict_result(workspace, source, metadata=payload)
            with pytest.raises(ValueError, match=r"save_workspace_predict_result.result_metadata"):
                nirs4all.save_workspace_predict_result(workspace, source, result_metadata=payload)

    def test_explain_result_end_to_end(self):
        """Test complete ExplainResult workflow."""
        shap_values = np.random.randn(50, 10)
        feature_names = [f"wavelength_{i}" for i in range(10)]

        result = ExplainResult(shap_values=shap_values, feature_names=feature_names, base_value=0.5, model_name="PLSRegression", explainer_type="kernel", n_samples=50)

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
        assert "PLSRegression" in str(result)
