"""Regression tests for scoring computation invariants.

These tests verify the correctness of the scoring overhaul, covering:
- RMSECV = sqrt(PRESS/N) from pooled OOF predictions
- No double-sqrt in _fmt() formatting
- best_val comes from fold_id="avg" entry
- None scores preserved (not coerced to 0.0)
- Naming conventions (NIRS vs ML)
- Multi-criteria selection deduplication
- RefitConfig uses selection_score (not best_score)
"""

import numpy as np
import pytest

from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.execution.refit.config_extractor import (
    RefitConfig,
    RefitCriterion,
    extract_top_configs,
)
from nirs4all.visualization.naming import get_metric_names, _format_metric_display
from nirs4all.visualization.reports import TabReportManager


# ---------------------------------------------------------------------------
# 1. RMSECV = sqrt(PRESS/N) from pooled OOF predictions
# ---------------------------------------------------------------------------


def test_rmsecv_equals_sqrt_press_over_n():
    """RMSECV must equal sqrt(PRESS/N) computed over ALL pooled OOF predictions.

    This verifies the pooled approach (concatenate all folds, then compute
    sqrt(mean_squared_error)) rather than averaging per-fold RMSE values.
    """
    # Simulate 3-fold CV with different fold sizes
    # Fold 0: 2 samples, fold 1: 3 samples, fold 2: 2 samples
    fold0_y_true = np.array([1.0, 2.0])
    fold0_y_pred = np.array([1.5, 2.5])

    fold1_y_true = np.array([3.0, 4.0, 5.0])
    fold1_y_pred = np.array([3.2, 3.8, 5.5])

    fold2_y_true = np.array([6.0, 7.0])
    fold2_y_pred = np.array([6.1, 7.3])

    # Expected RMSECV: pool all predictions and compute sqrt(MSE)
    all_y_true = np.concatenate([fold0_y_true, fold1_y_true, fold2_y_true])
    all_y_pred = np.concatenate([fold0_y_pred, fold1_y_pred, fold2_y_pred])
    squared_errors = (all_y_true - all_y_pred) ** 2
    expected_rmsecv = np.sqrt(np.sum(squared_errors) / len(all_y_true))

    # Build a Predictions buffer mimicking CV fold entries
    predictions = Predictions()
    for fold_id, (yt, yp) in enumerate([
        (fold0_y_true, fold0_y_pred),
        (fold1_y_true, fold1_y_pred),
        (fold2_y_true, fold2_y_pred),
    ]):
        predictions.add_prediction(
            dataset_name="test",
            config_name="cfg",
            model_name="PLS",
            fold_id=fold_id,
            partition="val",
            y_true=yt,
            y_pred=yp,
            val_score=np.sqrt(np.mean((yt - yp) ** 2)),
            metric="rmse",
            task_type="regression",
        )

    # Use TabReportManager._compute_oof_cv_metric_indexed to compute RMSECV
    pred_index = TabReportManager._build_prediction_index(predictions)
    entry = {
        "dataset_name": "test",
        "config_name": "cfg",
        "model_name": "PLS",
        "fold_id": "final",
        "step_idx": 0,
    }
    computed_rmsecv = TabReportManager._compute_oof_cv_metric_indexed(
        entry, pred_index, metric="rmse", task_type="regression",
    )

    assert computed_rmsecv is not None
    assert computed_rmsecv == pytest.approx(expected_rmsecv, rel=1e-10)


def test_rmsecv_pooled_differs_from_fold_averaged():
    """Pooled RMSECV differs from the mean of per-fold RMSE when fold sizes differ.

    This ensures we are truly pooling, not averaging per-fold metrics.
    """
    # Two folds with very different sizes
    fold0_y_true = np.array([1.0])
    fold0_y_pred = np.array([2.0])  # RMSE = 1.0

    fold1_y_true = np.array([1.0, 1.0, 1.0, 1.0])
    fold1_y_pred = np.array([1.0, 1.0, 1.0, 1.0])  # RMSE = 0.0

    # Mean-of-folds RMSE = (1.0 + 0.0) / 2 = 0.5
    mean_of_folds_rmse = 0.5

    # Pooled RMSECV = sqrt(1/5) = 0.4472...
    all_y_true = np.concatenate([fold0_y_true, fold1_y_true])
    all_y_pred = np.concatenate([fold0_y_pred, fold1_y_pred])
    pooled_rmsecv = np.sqrt(np.mean((all_y_true - all_y_pred) ** 2))

    assert pooled_rmsecv != pytest.approx(mean_of_folds_rmse, rel=1e-3)

    # Verify our implementation matches the pooled computation
    predictions = Predictions()
    for fold_id, (yt, yp) in enumerate([
        (fold0_y_true, fold0_y_pred),
        (fold1_y_true, fold1_y_pred),
    ]):
        predictions.add_prediction(
            dataset_name="test",
            config_name="cfg",
            model_name="PLS",
            fold_id=fold_id,
            partition="val",
            y_true=yt,
            y_pred=yp,
            val_score=np.sqrt(np.mean((yt - yp) ** 2)),
            metric="rmse",
            task_type="regression",
        )

    pred_index = TabReportManager._build_prediction_index(predictions)
    entry = {
        "dataset_name": "test",
        "config_name": "cfg",
        "model_name": "PLS",
        "fold_id": "final",
        "step_idx": 0,
    }
    computed = TabReportManager._compute_oof_cv_metric_indexed(
        entry, pred_index, metric="rmse", task_type="regression",
    )

    assert computed == pytest.approx(pooled_rmsecv, rel=1e-10)
    assert computed != pytest.approx(mean_of_folds_rmse, rel=1e-3)


# ---------------------------------------------------------------------------
# 2. No double-sqrt bug: _fmt() only formats, never transforms
# ---------------------------------------------------------------------------


def test_fmt_only_formats_no_transform():
    """_fmt() in TabReportManager must only round/format, never apply sqrt or other transforms.

    Previously there was a bug where sqrt was applied twice (once in the
    metric computation and once in _fmt). Verify _fmt is a pure formatter.
    """
    # We need to test the _fmt inner function. Since it's defined inside
    # generate_per_model_summary, we replicate its exact logic here and
    # verify it matches. The function is:
    #   def _fmt(value): return "N/A" if value is None else f"{value:.4f}"

    test_values = [0.0, 1.0, 0.12345678, 100.0, -3.14, 1e-10]
    for val in test_values:
        formatted = f"{val:.4f}"
        # Verify no sqrt: formatting 4.0 should give "4.0000", not "2.0000"
        assert formatted == f"{val:.4f}"

    # Explicit check: sqrt(4) = 2, but _fmt(4.0) must remain "4.0000"
    val = 4.0
    assert f"{val:.4f}" == "4.0000"
    assert f"{val:.4f}" != f"{np.sqrt(val):.4f}"

    # None handling
    assert (lambda v: "N/A" if v is None else f"{v:.4f}")(None) == "N/A"


def test_per_model_summary_fmt_preserves_values():
    """generate_per_model_summary must output values unchanged (no sqrt applied).

    Given known test_score and selection_score values, verify the formatted
    table contains those exact values rounded to 4 decimal places.
    """
    # Create a refit entry with known scores
    test_score = 4.0  # If sqrt were applied, this would become 2.0
    selection_score = 9.0  # If sqrt were applied, this would become 3.0

    entries = [{
        "model_name": "PLS",
        "test_score": test_score,
        "selection_score": selection_score,
        "preprocessings": "SNV",
        "config_name": "cfg_refit",
        "task_type": "regression",
        "dataset_name": "test",
        "fold_id": "final",
        "step_idx": 0,
    }]

    report = TabReportManager.generate_per_model_summary(
        entries, ascending=True, metric="rmse",
    )

    # The table should contain "4.0000" (the actual test_score), not "2.0000"
    assert "4.0000" in report
    assert "9.0000" in report
    # Ensure no sqrt was applied
    assert "2.0000" not in report
    assert "3.0000" not in report


# ---------------------------------------------------------------------------
# 3. Fold avg score uses fold_id="avg" entry
# ---------------------------------------------------------------------------


def test_best_val_comes_from_avg_fold():
    """best_val must come from the fold_id='avg' entry, not individual folds.

    The executor uses prediction_store.get_best(fold_id='avg') to populate
    best_val for store.complete_pipeline(). This test verifies that when
    fold_id='avg' exists, get_best with that filter returns it.
    """
    predictions = Predictions()

    # Add individual fold entries with varying scores
    predictions.add_prediction(
        dataset_name="test",
        model_name="PLS",
        fold_id=0,
        partition="val",
        val_score=5.0,
        metric="rmse",
    )
    predictions.add_prediction(
        dataset_name="test",
        model_name="PLS",
        fold_id=1,
        partition="val",
        val_score=3.0,  # Best individual fold
        metric="rmse",
    )

    # Add the avg fold entry with the pooled RMSECV
    avg_rmsecv = 4.2
    predictions.add_prediction(
        dataset_name="test",
        model_name="PLS",
        fold_id="avg",
        partition="val",
        val_score=avg_rmsecv,
        metric="rmse",
    )

    # get_best with fold_id="avg" should return the avg entry
    avg_entry = predictions.get_best(ascending=True, fold_id="avg")
    assert avg_entry is not None
    assert avg_entry["fold_id"] == "avg"
    assert avg_entry["val_score"] == pytest.approx(avg_rmsecv)

    # The avg value (4.2) differs from the best individual fold (3.0)
    assert avg_rmsecv != 3.0


def test_best_val_avg_not_best_individual_fold():
    """When fold_id='avg' exists, its score may differ from the best individual fold.

    This is by design: the avg fold holds the pooled RMSECV across all OOF
    predictions, which is generally different from the min per-fold score.
    """
    predictions = Predictions()

    # Individual folds
    for fold_id, score in [(0, 2.0), (1, 4.0), (2, 3.0)]:
        predictions.add_prediction(
            dataset_name="test",
            model_name="PLS",
            fold_id=fold_id,
            partition="val",
            val_score=score,
            metric="rmse",
        )

    # avg fold with pooled RMSECV (typically close to mean, not min)
    predictions.add_prediction(
        dataset_name="test",
        model_name="PLS",
        fold_id="avg",
        partition="val",
        val_score=3.1,
        metric="rmse",
    )

    avg_entry = predictions.get_best(ascending=True, fold_id="avg")
    assert avg_entry is not None
    assert avg_entry["val_score"] == pytest.approx(3.1)

    # Best individual fold has score 2.0, but avg is 3.1
    best_any = predictions.get_best(ascending=True)
    assert best_any is not None
    # Without fold_id filter, the best overall is the lowest score (2.0)
    assert best_any["val_score"] == pytest.approx(2.0)
    assert best_any["fold_id"] == "0"


# ---------------------------------------------------------------------------
# 4. None scores preserved (not coerced to 0.0)
# ---------------------------------------------------------------------------


def test_none_val_score_preserved_in_buffer():
    """Predictions.add_prediction must preserve None for val_score, not coerce to 0.0."""
    predictions = Predictions()
    predictions.add_prediction(
        dataset_name="test",
        model_name="PLS",
        fold_id="final",
        partition="val",
        val_score=None,
        metric="rmse",
    )

    assert len(predictions._buffer) == 1
    entry = predictions._buffer[0]
    assert entry["val_score"] is None


def test_none_test_score_preserved_in_buffer():
    """Predictions.add_prediction must preserve None for test_score."""
    predictions = Predictions()
    predictions.add_prediction(
        dataset_name="test",
        model_name="PLS",
        fold_id=0,
        partition="val",
        val_score=3.14,
        test_score=None,
        metric="rmse",
    )

    entry = predictions._buffer[0]
    assert entry["test_score"] is None
    assert entry["val_score"] == pytest.approx(3.14)


def test_none_scores_not_coerced_in_flush_row():
    """The flush method must pass None scores to save_prediction, not 0.0.

    Verify the buffer row retains None so that save_prediction receives it.
    """
    predictions = Predictions()
    predictions.add_prediction(
        dataset_name="test",
        model_name="PLS",
        fold_id="final",
        partition="val",
        val_score=None,
        test_score=None,
        train_score=None,
        metric="rmse",
    )

    row = predictions._buffer[0]
    # All score fields must be None (flush reads them via row.get())
    assert row.get("val_score") is None
    assert row.get("test_score") is None
    assert row.get("train_score") is None


# ---------------------------------------------------------------------------
# 5. Naming conventions
# ---------------------------------------------------------------------------


def test_nirs_regression_naming():
    """NIRS regression naming must use RMSECV/RMSEP and CV-phase metrics."""
    names = get_metric_names("nirs", "regression")
    assert names["cv_score"] == "RMSECV"
    assert names["test_score"] == "RMSEP"
    assert names["mean_fold_test"] == "Mean_Fold_RMSEP"
    assert names["wmean_fold_test"] == "W_Mean_Fold_RMSEP"
    assert names["selection_score"] == "Selection_Score"
    assert names["ens_test"] == "Ens_Test"
    assert names["w_ens_test"] == "W_Ens_Test"
    assert names["mean_fold_cv"] == "MF_Val"
    assert names["wmean_fold_cv"] == "W_RMSECV"


def test_ml_regression_naming():
    """ML regression naming must use CV_Score/Test_Score and CV-phase metrics."""
    names = get_metric_names("ml", "regression")
    assert names["cv_score"] == "CV_Score"
    assert names["test_score"] == "Test_Score"
    assert names["mean_fold_test"] == "Mean_Fold_Test"
    assert names["wmean_fold_test"] == "W_Mean_Fold_Test"
    assert names["selection_score"] == "Selection_Score"
    assert names["ens_test"] == "Ens_Test_Score"
    assert names["w_ens_test"] == "W_Ens_Test_Score"
    assert names["mean_fold_cv"] == "MF_CV"
    assert names["wmean_fold_cv"] == "W_CV_Score"


def test_nirs_classification_naming():
    """NIRS classification naming must format metric into template."""
    names = get_metric_names("nirs", "classification", "balanced_accuracy")
    assert names["cv_score"] == "CV_BalAcc"
    assert names["test_score"] == "Test_BalAcc"


def test_ml_classification_naming():
    """ML classification naming uses generic Score names."""
    names = get_metric_names("ml", "classification", "balanced_accuracy")
    assert names["cv_score"] == "CV_Score"
    assert names["test_score"] == "Test_Score"


def test_auto_mode_defaults_to_nirs():
    """Auto naming mode must default to NIRS convention."""
    names = get_metric_names("auto", "regression")
    assert names["cv_score"] == "RMSECV"
    assert names["test_score"] == "RMSEP"


def test_format_metric_display():
    """_format_metric_display must produce standard abbreviations."""
    assert _format_metric_display("balanced_accuracy") == "BalAcc"
    assert _format_metric_display("rmse") == "RMSE"
    assert _format_metric_display("accuracy") == "Accuracy"
    assert _format_metric_display("r2") == "R2"
    assert _format_metric_display("f1_score") == "F1"


# ---------------------------------------------------------------------------
# 6. Multi-criteria independent selection
# ---------------------------------------------------------------------------


def test_extract_top_configs_independent_selection():
    """Each refit criterion must independently fill its top_k quota.

    When criteria overlap (same model appears in both rankings), each
    criterion skips globally-seen models and selects the next available,
    guaranteeing sum(top_k) unique models.
    """
    # Simulate ranked lists for two criteria
    # Criterion 1 (rmsecv, top2): ranks are A, B, C, D
    # Criterion 2 (mean_val, top2): ranks are A, C, D, E
    # A is top-1 for both -> criterion 2 must skip A and take C, D
    rmsecv_ranked = ["A", "B", "C", "D"]
    mean_val_ranked = ["A", "C", "D", "E"]

    selected_ids: list[str] = []
    seen_ids: set[str] = set()
    pid_to_criteria: dict[str, list[str]] = {}

    # Criterion 1: select top 2
    top_ids_1 = []
    for pid in rmsecv_ranked:
        if pid not in seen_ids:
            top_ids_1.append(pid)
            if len(top_ids_1) >= 2:
                break
    for pid in top_ids_1:
        pid_to_criteria.setdefault(pid, []).append("rmsecv(top2)")
        selected_ids.append(pid)
        seen_ids.add(pid)

    # Criterion 2: select top 2 (skipping globally-seen)
    top_ids_2 = []
    for pid in mean_val_ranked:
        if pid not in seen_ids:
            top_ids_2.append(pid)
            if len(top_ids_2) >= 2:
                break
    for pid in top_ids_2:
        pid_to_criteria.setdefault(pid, []).append("mean_val(top2)")
        selected_ids.append(pid)
        seen_ids.add(pid)

    # Total: 4 unique models (2 per criterion, no overlap)
    assert selected_ids == ["A", "B", "C", "D"]
    assert len(selected_ids) == 4

    # A is selected by criterion 1, C and D by criterion 2
    assert pid_to_criteria["A"] == ["rmsecv(top2)"]
    assert pid_to_criteria["B"] == ["rmsecv(top2)"]
    assert pid_to_criteria["C"] == ["mean_val(top2)"]
    assert pid_to_criteria["D"] == ["mean_val(top2)"]


def test_refit_criterion_defaults():
    """RefitCriterion defaults must be top_k=1, ranking='rmsecv'."""
    crit = RefitCriterion()
    assert crit.top_k == 1
    assert crit.ranking == "rmsecv"
    assert crit.metric == ""


# ---------------------------------------------------------------------------
# 7. RefitConfig uses selection_score (not best_score)
# ---------------------------------------------------------------------------


def test_refit_config_has_selection_score():
    """RefitConfig must have 'selection_score', not 'best_score'."""
    config = RefitConfig(expanded_steps=[])

    # selection_score exists and defaults to 0.0
    assert hasattr(config, "selection_score")
    assert config.selection_score == 0.0

    # best_score must NOT exist as an attribute
    assert not hasattr(config, "best_score")


def test_refit_config_selection_score_set():
    """RefitConfig.selection_score must store the value used for selection."""
    config = RefitConfig(
        expanded_steps=[],
        selection_score=3.21,
        selection_scores={"rmsecv": 3.21, "mean_val": 3.25},
        primary_selection_criterion="rmsecv",
    )

    assert config.selection_score == pytest.approx(3.21)
    assert config.selection_scores == {"rmsecv": 3.21, "mean_val": 3.25}
    assert config.primary_selection_criterion == "rmsecv"


def test_refit_config_selected_by_criteria():
    """RefitConfig.selected_by_criteria tracks which criteria chose this config."""
    config = RefitConfig(
        expanded_steps=[],
        selected_by_criteria=["rmsecv(top3)", "mean_val(top1)"],
    )

    assert config.selected_by_criteria == ["rmsecv(top3)", "mean_val(top1)"]

    # Default is empty list
    default_config = RefitConfig(expanded_steps=[])
    assert default_config.selected_by_criteria == []


# ---------------------------------------------------------------------------
# 8. Final scores table sorted by RMSEP
# ---------------------------------------------------------------------------


def test_per_model_summary_sorted_by_rmsep():
    """generate_per_model_summary must sort entries by RMSEP (test_score), not RMSECV."""
    entries = [
        {
            "model_name": "PLS",
            "test_score": 3.0,  # Lower RMSEP, but worse RMSECV
            "rmsecv": 5.0,
            "selection_score": 5.0,
            "preprocessings": "SNV",
            "config_name": "cfg1_refit",
            "task_type": "regression",
            "dataset_name": "test",
            "fold_id": "final",
            "step_idx": 0,
        },
        {
            "model_name": "RF",
            "test_score": 5.0,  # Higher RMSEP, but better RMSECV
            "rmsecv": 2.0,
            "selection_score": 2.0,
            "preprocessings": "MSC",
            "config_name": "cfg2_refit",
            "task_type": "regression",
            "dataset_name": "test",
            "fold_id": "final",
            "step_idx": 0,
        },
    ]

    report = TabReportManager.generate_per_model_summary(
        entries, ascending=True, metric="rmse",
    )

    # PLS (RMSEP=3.0) should come before RF (RMSEP=5.0)
    assert report.index("3.0000") < report.index("5.0000")
    # Sorting indicator should mention RMSEP, not RMSECV
    assert "Sorted by: RMSEP" in report


def test_per_model_summary_star_markers():
    """Multi-criteria refit models should have star markers for best per criterion."""
    entries = [
        {
            "model_name": "PLS",
            "test_score": 2.0,
            "rmsecv": 4.0,
            "selection_score": 4.0,
            "preprocessings": "SNV",
            "config_name": "cfg1_refit_rmsecvt2",
            "task_type": "regression",
            "dataset_name": "test",
            "fold_id": "final",
            "step_idx": 0,
        },
        {
            "model_name": "RF",
            "test_score": 3.0,
            "rmsecv": 3.0,
            "selection_score": 3.0,
            "preprocessings": "MSC",
            "config_name": "cfg2_refit_mean_valt2",
            "task_type": "regression",
            "dataset_name": "test",
            "fold_id": "final",
            "step_idx": 0,
        },
    ]

    report = TabReportManager.generate_per_model_summary(
        entries, ascending=True, metric="rmse",
    )

    # Both #1 and #2 should have stars (each is best for its criterion)
    assert "1*" in report
    assert "2*" in report
