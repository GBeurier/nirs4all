"""Public oracle for step-level missing-value replacement."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import nirs4all


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_model_na_replacement_matches_filled_legacy_run_and_replays(
    mechanism: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    rng = np.random.default_rng(42)
    x = rng.normal(size=(24, 6))
    y = 2 * x[:, 0] - x[:, 1]
    x[[2, 8, 16], 0] = np.nan
    fill_value = -2.0
    pipeline = [KFold(3), {"model": Ridge(alpha=1.0), "na_policy": "replace", "fill_value": fill_value}]

    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", save_artifacts=False, verbose=0)
    filled = nirs4all.run(
        [KFold(3), Ridge(alpha=1.0)],
        (np.where(np.isnan(x), fill_value, x), y),
        engine="legacy", save_artifacts=False, verbose=0,
    )
    assert legacy.cv_best_score == pytest.approx(filled.cv_best_score, abs=1e-10)
    native = nirs4all.run(pipeline, (x, y), engine="dag-ml", save_artifacts=False, verbose=0)
    native_filled = nirs4all.run(
        [KFold(3), Ridge(alpha=1.0)],
        (np.where(np.isnan(x), fill_value, x), y),
        engine="dag-ml", save_artifacts=False, verbose=0,
    )
    assert native.cv_best_score == pytest.approx(native_filled.cv_best_score, abs=1e-10)
    assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-5)
    archive = native.export(tmp_path / f"model_na_{mechanism}.n4a")
    expected = Ridge(alpha=1.0).fit(np.where(np.isnan(x), fill_value, x), y).predict(
        np.where(np.isnan(x[:3]), fill_value, x[:3]),
    )
    np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, x[:3]).y_pred).ravel(), expected, atol=1e-5)
    legacy.close()
    filled.close()
    native.close()
    native_filled.close()


@pytest.mark.parity
def test_standard_scaler_fit_scope_explains_existing_legacy_dag_cv_difference() -> None:
    rng = np.random.default_rng(43)
    x = rng.normal(size=(24, 6))
    y = 2 * x[:, 0] - x[:, 1]
    x[[2, 8, 16], 0] = -2.0
    folds = KFold(3)

    legacy = nirs4all.run([folds, StandardScaler(), Ridge()], (x, y), engine="legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run([folds, StandardScaler(), Ridge()], (x, y), engine="dag-ml", save_artifacts=False, verbose=0)
    global_scaled = StandardScaler().fit_transform(x)
    global_oof = cross_val_predict(Ridge(), global_scaled, y, cv=folds)
    fold_oof = cross_val_predict(make_pipeline(StandardScaler(), Ridge()), x, y, cv=folds)
    global_rmse = np.sqrt(np.mean((y - global_oof) ** 2))
    fold_rmse = np.sqrt(np.mean((y - fold_oof) ** 2))

    assert legacy.cv_best_score == pytest.approx(global_rmse, abs=1e-5)
    assert native.cv_best_score == pytest.approx(fold_rmse, abs=1e-5)
    assert fold_rmse - global_rmse > 0.05
    legacy.close()
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_transform_na_replacement_matches_filled_legacy_run_and_replays(
    mechanism: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    rng = np.random.default_rng(43)
    x = rng.normal(size=(24, 6))
    y = 2 * x[:, 0] - x[:, 1]
    x[[2, 8, 16], 0] = np.nan
    fill_value = -2.0
    pipeline = [
        KFold(3),
        {"preprocessing": StandardScaler(), "na_policy": "replace", "fill_value": fill_value},
        Ridge(alpha=1.0),
    ]
    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", save_artifacts=False, verbose=0)
    filled = nirs4all.run(
        [KFold(3), StandardScaler(), Ridge(alpha=1.0)],
        (np.where(np.isnan(x), fill_value, x), y),
        engine="legacy", save_artifacts=False, verbose=0,
    )
    assert legacy.cv_best_score == pytest.approx(filled.cv_best_score, abs=1e-10)
    native = nirs4all.run(pipeline, (x, y), engine="dag-ml", save_artifacts=False, verbose=0)
    native_filled = nirs4all.run(
        [KFold(3), StandardScaler(), Ridge(alpha=1.0)],
        (np.where(np.isnan(x), fill_value, x), y),
        engine="dag-ml", save_artifacts=False, verbose=0,
    )
    assert native.cv_best_score == pytest.approx(native_filled.cv_best_score, abs=1e-10)
    archive = native.export(tmp_path / f"transform_na_{mechanism}.n4a")
    expected = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
        np.where(np.isnan(x), fill_value, x), y,
    ).predict(np.where(np.isnan(x[:3]), fill_value, x[:3]))
    np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, x[:3]).y_pred).ravel(), expected, atol=1e-5)
    legacy.close()
    filled.close()
    native.close()
    native_filled.close()
