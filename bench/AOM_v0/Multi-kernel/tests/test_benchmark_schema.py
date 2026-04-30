"""Tests for benchmark cohort builders and result schema."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.build_cohorts import build_classification_cohort, build_regression_cohort
from benchmarks.run_aompls_benchmark import RESULT_COLUMNS


def test_master_schema_columns_present():
    """The result CSV must include every master_results.csv column."""
    master_columns = [
        "database_name", "dataset", "task", "model", "result_label", "result_dir",
        "status", "status_details", "preprocessing_pipeline", "RMSECV", "RMSE_MF",
        "RMSEP", "MAE_test", "r2_test", "search_mean_score", "seed", "n_splits",
        "best_config_json", "best_model_params_json", "best_fold_scores_json",
        "trial_values_json", "search_results_path", "best_config_path",
        "final_predictions_path", "fold_predictions_path", "rmse_mf_source",
        "artifact_best_config_format", "artifact_search_results_format",
        "artifact_final_predictions_format",
    ]
    missing = [c for c in master_columns if c not in RESULT_COLUMNS]
    assert not missing, f"missing master columns: {missing}"


def test_aom_extra_columns_present():
    """The AOM diagnostic columns from the protocol must be present."""
    aom_columns = [
        "aom_variant", "backend", "engine", "selection", "criterion",
        "orthogonalization", "operator_bank", "selected_operator_sequence_json",
        "selected_operator_scores_json", "n_components_selected", "max_components",
        "fit_time_s", "predict_time_s", "delta_rmsep_vs_master_pls",
        "delta_rmsep_vs_tabpfn_raw", "delta_rmsep_vs_tabpfn_opt", "run_seed",
        "code_version", "notes",
    ]
    missing = [c for c in aom_columns if c not in RESULT_COLUMNS]
    assert not missing, f"missing AOM columns: {missing}"


def test_classification_extras_present():
    cls_columns = ["balanced_accuracy", "macro_f1", "log_loss", "ece"]
    missing = [c for c in cls_columns if c not in RESULT_COLUMNS]
    assert not missing


def test_regression_cohort_builds(tmp_path):
    """If master_results.csv is reachable, build_regression_cohort runs."""
    master = "bench/tabpfn_paper/master_results.csv"
    if not Path(master).exists():
        pytest.skip("master_results.csv not found")
    out = tmp_path / "cohort.csv"
    df = build_regression_cohort(master_path=master, out_path=str(out))
    assert out.exists()
    assert "status" in df.columns
    assert {"ok", "skipped"}.issuperset(set(df["status"].unique().tolist())) or df.empty


def test_classification_cohort_builds(tmp_path):
    out = tmp_path / "cohort.csv"
    df = build_classification_cohort(data_root="bench/tabpfn_paper/data/classification", out_path=str(out))
    assert out.exists()
    if not df.empty:
        assert "status" in df.columns
