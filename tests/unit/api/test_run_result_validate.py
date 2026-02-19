"""Unit tests for RunResult.validate() (F-06).

Covers: empty predictions, NaN metrics, nan_threshold, raise_on_failure,
and the structure of the returned report dict.
"""

import math

import numpy as np
import pytest

from nirs4all.api.result import RunResult
from nirs4all.data.predictions import Predictions


def _make_run_result(entries: list[dict]) -> RunResult:
    """Build a minimal RunResult backed by an in-memory Predictions."""
    preds = Predictions()
    for entry in entries:
        preds.add_prediction(**entry)
    return RunResult(predictions=preds, per_dataset={})

def _base_entry(**overrides):
    """Return a minimal valid prediction entry dict."""
    base = {
        "dataset_name": "test",
        "model_name": "PLS",
        "config_name": "cfg0",
        "fold_id": "fold_0",
        "partition": "test",
        "val_score": 0.1,
        "test_score": 0.2,
        "train_score": 0.05,
        "y_true": np.array([1.0, 2.0, 3.0]),
        "y_pred": np.array([1.1, 2.1, 3.1]),
    }
    base.update(overrides)
    return base

# ---------------------------------------------------------------------------
# Return-value structure
# ---------------------------------------------------------------------------

class TestValidateReturnStructure:

    def test_returns_dict_with_required_keys(self):
        result = _make_run_result([_base_entry()])
        report = result.validate(raise_on_failure=False)
        assert isinstance(report, dict)
        for key in ("valid", "issues", "nan_count", "total_count"):
            assert key in report, f"Missing key: {key}"

    def test_issues_is_list(self):
        result = _make_run_result([_base_entry()])
        report = result.validate(raise_on_failure=False)
        assert isinstance(report["issues"], list)

    def test_total_count_matches_num_predictions(self):
        entries = [_base_entry(config_name=f"cfg{i}") for i in range(3)]
        result = _make_run_result(entries)
        report = result.validate(raise_on_failure=False)
        assert report["total_count"] == result.num_predictions

# ---------------------------------------------------------------------------
# Empty predictions (check_empty)
# ---------------------------------------------------------------------------

class TestValidateCheckEmpty:

    def test_empty_predictions_returns_invalid(self):
        result = _make_run_result([])
        report = result.validate(raise_on_failure=False)
        assert report["valid"] is False
        assert report["total_count"] == 0
        assert any("No predictions" in issue for issue in report["issues"])

    def test_empty_predictions_raises_by_default(self):
        result = _make_run_result([])
        with pytest.raises(ValueError, match="No predictions"):
            result.validate()

    def test_empty_check_disabled_does_not_flag_empty(self):
        result = _make_run_result([])
        report = result.validate(check_empty=False, raise_on_failure=False)
        # No predictions-empty issue; nan_count check also skipped since total=0
        assert not any("No predictions" in issue for issue in report["issues"])

    def test_nonempty_predictions_passes_empty_check(self):
        result = _make_run_result([_base_entry()])
        report = result.validate(raise_on_failure=False)
        assert not any("No predictions" in issue for issue in report["issues"])

# ---------------------------------------------------------------------------
# NaN metrics (check_nan_metrics)
# ---------------------------------------------------------------------------

class TestValidateNanMetrics:

    def test_no_nan_passes(self):
        result = _make_run_result([_base_entry(test_score=0.5)])
        report = result.validate(raise_on_failure=False)
        assert report["valid"] is True
        assert report["nan_count"] == 0

    def test_nan_test_score_detected(self):
        result = _make_run_result([_base_entry(test_score=float("nan"))])
        report = result.validate(raise_on_failure=False)
        assert report["nan_count"] >= 1
        assert report["valid"] is False

    def test_nan_val_score_detected(self):
        result = _make_run_result([_base_entry(val_score=float("nan"))])
        report = result.validate(raise_on_failure=False)
        assert report["nan_count"] >= 1

    def test_nan_check_disabled_ignores_nans(self):
        result = _make_run_result([_base_entry(test_score=float("nan"))])
        report = result.validate(check_nan_metrics=False, raise_on_failure=False)
        assert report["nan_count"] == 0

    def test_nan_raises_by_default(self):
        result = _make_run_result([_base_entry(test_score=float("nan"))])
        with pytest.raises(ValueError):
            result.validate()

    def test_nan_threshold_zero_rejects_any_nan(self):
        entries = [
            _base_entry(config_name="ok", test_score=0.3),
            _base_entry(config_name="bad", test_score=float("nan")),
        ]
        result = _make_run_result(entries)
        report = result.validate(nan_threshold=0.0, raise_on_failure=False)
        assert report["valid"] is False

    def test_nan_threshold_allows_some_nan(self):
        """nan_threshold=0.6 allows up to 60% NaN entries."""
        entries = [
            _base_entry(config_name="ok1", test_score=0.3),
            _base_entry(config_name="ok2", test_score=0.4),
            _base_entry(config_name="bad1", test_score=float("nan")),
        ]
        result = _make_run_result(entries)
        # 1 out of 3 = ~33% NaN → within threshold of 0.6
        report = result.validate(nan_threshold=0.6, raise_on_failure=False)
        assert report["nan_count"] >= 1
        # Should still be valid because ratio <= threshold
        assert report["valid"] is True

    def test_multiple_nans_count_correctly(self):
        entries = [_base_entry(config_name=f"bad{i}", test_score=float("nan")) for i in range(5)]
        result = _make_run_result(entries)
        report = result.validate(raise_on_failure=False)
        assert report["nan_count"] == 5

# ---------------------------------------------------------------------------
# raise_on_failure
# ---------------------------------------------------------------------------

class TestValidateRaiseOnFailure:

    def test_raise_on_failure_true_raises_value_error(self):
        result = _make_run_result([])
        with pytest.raises(ValueError):
            result.validate(raise_on_failure=True)

    def test_raise_on_failure_false_does_not_raise(self):
        result = _make_run_result([])
        report = result.validate(raise_on_failure=False)
        assert report["valid"] is False  # Still invalid, just no exception

    def test_valid_result_does_not_raise(self):
        result = _make_run_result([_base_entry()])
        report = result.validate()  # Default raise_on_failure=True
        assert report["valid"] is True
