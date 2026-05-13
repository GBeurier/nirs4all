"""Parity tests for the Python binding — same JSON fixtures as the C++ side."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from aompls import AOMPLSCompact

ROOT = Path(__file__).resolve().parents[2]  # bench/AOM_lib
REF_DIR = ROOT / "cpp" / "tests" / "reference"
DATASETS = ["BEER", "CORN", "ALPINE"]
CASES = [
    ("kfold5", False),
    ("kfold5_oneSE", True),
    ("spxy5", False),
]


def _load_case(dataset: str, case_name: str) -> Dict:
    with (REF_DIR / f"{dataset}.json").open() as fh:
        return json.load(fh)[case_name]


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("case_name,one_se", CASES)
def test_python_binding_parity(dataset: str, case_name: str, one_se: bool) -> None:
    ref = _load_case(dataset, case_name)
    X = np.asarray(ref["X"], dtype=np.float64)
    y = np.asarray(ref["y"], dtype=np.float64)
    folds: List[List[int]] = ref["fold_test_indices"]

    est = AOMPLSCompact(
        max_components=ref["max_components"],
        cv_mode="external",
        external_folds=folds,
        one_se_rule=one_se,
        random_state=0,
    )
    est.fit(X, y)

    assert est.selected_operator_name_ == ref["selected_operator_name"], (
        f"selected_operator_name: py='{est.selected_operator_name_}', "
        f"ref='{ref['selected_operator_name']}'"
    )
    assert est.selected_operator_index_ == ref["selected_operator_index"]
    assert est.n_components_ == ref["n_components_selected"]

    ref_coef = np.asarray(ref["coef"], dtype=np.float64)
    coef_diff = float(np.max(np.abs(est.coef_ - ref_coef)))
    assert coef_diff < 1e-8, f"coef max|Δ| = {coef_diff:.3e}"

    intercept_diff = abs(est.intercept_ - float(ref["intercept"]))
    assert intercept_diff < 1e-8, f"intercept |Δ| = {intercept_diff:.3e}"

    ref_pred = np.asarray(ref["predictions_train"], dtype=np.float64)
    pred = est.predict(X)
    pred_diff = float(np.max(np.abs(pred - ref_pred)))
    assert pred_diff < 1e-8, f"predict max|Δ| = {pred_diff:.3e}"

    ref_curves = np.asarray(ref["rmse_curves"], dtype=np.float64)
    mask = np.isfinite(est.rmse_curves_) & np.isfinite(ref_curves)
    if mask.any():
        curve_diff = float(np.max(np.abs(est.rmse_curves_[mask] - ref_curves[mask])))
        assert curve_diff < 1e-9, f"rmse_curves max|Δ| = {curve_diff:.3e}"
