"""Regression cases from the Studio preset and dataset campaign workflows."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import nirs4all

pytestmark = pytest.mark.parity


def test_string_pls_sweep_and_preprocessing_cartesian_persists_every_variant(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(36, 12))
    y = x[:, 0] - 0.5 * x[:, 1]
    pipeline = [
        {"_or_": [None, StandardScaler()]},
        KFold(n_splits=3, shuffle=True, random_state=42),
        {"model": "sklearn.cross_decomposition.PLSRegression", "param": "n_components", "_range_": [1, 3]},
    ]
    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", verbose=0, save_charts=False, workspace_path=tmp_path)
    try:
        assert result.execution_engine == "dag-ml"
        assert np.isfinite(result.cv_best_score)
        # Each preprocessing/model combination keeps its own native score label.
        config_names = {row["config_name"].removesuffix("_refit") for row in result.predictions.to_dicts()}
        assert len(config_names) == 6
    finally:
        result.close()


def test_studio_file_config_dictionary_list_runs_both_datasets(tmp_path):
    configs = []
    rng = np.random.default_rng(43)
    for index in range(2):
        x = rng.normal(size=(24, 8 + index))
        y = x[:, 0] - x[:, 1]
        x_path = tmp_path / f"x{index}.csv"
        y_path = tmp_path / f"y{index}.csv"
        np.savetxt(x_path, x, delimiter=";", header=";".join(f"w{i}" for i in range(x.shape[1])), comments="")
        np.savetxt(y_path, y, delimiter=";", header="target", comments="")
        configs.append({"name": f"campaign{index}", "train_x": str(x_path), "train_y": str(y_path), "delimiter": ";", "has_header": True})
    result = nirs4all.run([KFold(3), {"model": PLSRegression(2)}], configs, engine="dag-ml", verbose=0,
                         save_charts=False, workspace_path=tmp_path / "workspace")
    try:
        assert set(result.get_datasets()) == {"campaign0", "campaign1"}
        assert len(result.runs) == 2
        assert all(np.isfinite(child.cv_best_score) for child in result.runs)
        assert all(any(metadata.get("run_id") for metadata in child.per_dataset.values()) for child in result.runs)
    finally:
        result.close()


@pytest.mark.parametrize("tuned", [False, True])
def test_lightgbm_template_model_is_an_estimator_with_and_without_tuning(tmp_path, tuned):
    lightgbm = pytest.importorskip("lightgbm")
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(47)
    x = rng.normal(size=(36, 8))
    y = np.tile([0, 1, 2], 12)
    step = {"model": lightgbm.LGBMClassifier(n_estimators=5, random_state=42, verbosity=-1, n_jobs=1)}
    if tuned:
        step["finetune_params"] = {"n_trials": 2, "model_params": {"n_estimators": [5, 8]}, "approach": "single"}
    result = nirs4all.run([StandardScaler(), StratifiedKFold(3), step], (x, y), engine="dag-ml",
                         verbose=0, save_charts=False, workspace_path=tmp_path)
    try:
        assert result.execution_engine == "dag-ml"
        assert np.isfinite(result.cv_best_score)
        assert result.best["chain_id"]
    finally:
        result.close()
