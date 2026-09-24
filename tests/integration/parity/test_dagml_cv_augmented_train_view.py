"""Public oracle for synthetic observations in CV in-sample predictions."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.operators.augmentation import GaussianAdditiveNoise
from nirs4all.operators.filters import YOutlierFilter

from ._dagml_cli import dagml_cli_path
from ._datasets import dataset_path


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["pyo3", "cli"])
def test_cv_train_predictions_include_only_fitted_augmentation_children(mechanism, monkeypatch):
    """Train is the fitted augmented cohort; validation and train_pool remain base-grain."""
    if mechanism == "cli":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "pyo3" else "0")

    pipeline = [
        {"sample_augmentation": {
            "transformers": [GaussianAdditiveNoise(sigma=0.01)],
            "count": 1, "selection": "all", "random_state": 42,
        }},
        {"exclude": YOutlierFilter(method="iqr", threshold=1.0)},
        KFold(n_splits=3, shuffle=True, random_state=42),
        {"model": PLSRegression(n_components=3)},
    ]
    path = dataset_path("regression")
    legacy = nirs4all.run(pipeline, path, engine="legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False, verbose=0)
    assert native.execution_engine == "dag-ml"
    assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-8)
    assert native.best_rmse == pytest.approx(legacy.best_rmse, abs=1e-8)

    legacy_rows = legacy.predictions.filter_predictions()
    native_rows = native.predictions.filter_predictions()
    base_pool_count = sum(len(next(row for row in native_rows if row["partition"] == "val"
                                   and str(row["fold_id"]) == str(fold_id))["y_pred"])
                          for fold_id in range(3))
    for fold_id in range(3):
        native_train = next(row for row in native_rows if row["partition"] == "train"
                            and str(row["fold_id"]) == str(fold_id))
        legacy_train = next(row for row in legacy_rows if row["partition"] == "train"
                            and str(row["fold_id"]) == str(fold_id))
        native_val = next(row for row in native_rows if row["partition"] == "val"
                          and str(row["fold_id"]) == str(fold_id))
        assert len(native_train["y_pred"]) == len(legacy_train["y_pred"])
        assert len(native_train["y_pred"]) == 2 * (base_pool_count - len(native_val["y_pred"]))
        np.testing.assert_allclose(np.asarray(native_train["y_true"]).ravel(),
                                   np.asarray(legacy_train["y_true"]).ravel())
        np.testing.assert_allclose(np.asarray(native_train["y_pred"]).ravel(),
                                   np.asarray(legacy_train["y_pred"]).ravel(), atol=1e-8)
        pool = next((row for row in native_rows if row["partition"] == "train_pool"
                     and str(row["fold_id"]) == str(fold_id)), None)
        if pool is not None:
            assert len(pool["y_pred"]) == base_pool_count
