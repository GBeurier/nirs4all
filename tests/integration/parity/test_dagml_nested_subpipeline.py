"""Public parity for legacy's linear nested subpipelines."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import nirs4all

from ._dagml_cli import dagml_cli_path


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("splitter_inside", [False, True])
def test_nested_transform_chain_runs_and_replays_like_flat_chain(tmp_path, monkeypatch, mechanism: str, wrapped: bool, splitter_inside: bool) -> None:
    """A grouping list preserves transform order but adds no fitting scope."""
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")

    rng = np.random.default_rng(123)
    x = rng.normal(size=(18, 8))
    y = 2 * x[:, 0] - x[:, 1] + rng.normal(size=18) * 0.1
    first = {"preprocessing": StandardScaler()} if wrapped else StandardScaler()
    if splitter_inside:
        nested = [[KFold(2), first, MinMaxScaler()], Ridge(alpha=1.0)]
        flat = [KFold(2), first, MinMaxScaler(), Ridge(alpha=1.0)]
    else:
        nested = [[first, MinMaxScaler()], KFold(2), Ridge(alpha=1.0)]
        flat = [first, MinMaxScaler(), KFold(2), Ridge(alpha=1.0)]

    legacy_scores = []
    dag_scores = []
    predictions = []
    for shape, pipeline in (("nested", nested), ("flat", flat)):
        legacy = nirs4all.run(
            pipeline, (x, y), engine="legacy", allow_fallback=False,
            workspace_path=tmp_path / f"legacy-{shape}", save_charts=False, verbose=0,
        )
        legacy_scores.append(legacy.cv_best_score)
        legacy.close()

        native = nirs4all.run(
            pipeline, (x, y), engine="dag-ml", allow_fallback=False,
            workspace_path=tmp_path / f"dag-{shape}", save_charts=False, verbose=0,
        )
        assert native.execution_engine == "dag-ml"
        dag_scores.append(native.cv_best_score)
        archive = native.export(tmp_path / f"{shape}.n4a")
        predictions.append(np.asarray(nirs4all.predict(archive, x[:4]).y_pred))
        native.close()

    assert np.isfinite(legacy_scores).all()
    assert legacy_scores[0] == pytest.approx(legacy_scores[1])
    assert dag_scores[0] == pytest.approx(dag_scores[1])

    np.testing.assert_allclose(predictions[0], predictions[1], rtol=1e-6, atol=1e-6)


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("mixed", [False, True])
def test_nested_model_choices_preserve_selection_and_replay(tmp_path, monkeypatch, mechanism: str, mixed: bool) -> None:
    """Legacy compares grouped models just as it compares consecutive models."""
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")

    rng = np.random.default_rng(124)
    x = rng.normal(size=(20, 8))
    y = 2 * x[:, 0] - x[:, 1] + rng.normal(size=20) * 0.1
    group = [StandardScaler(), Ridge(alpha=1.0)] if mixed else [Ridge(alpha=0.1), Ridge(alpha=1.0)]
    nested = [KFold(2), group]
    flat = [KFold(2), *group]

    legacy_scores = []
    dag_scores = []
    predictions = []
    for shape, pipeline in (("nested", nested), ("flat", flat)):
        legacy = nirs4all.run(
            pipeline, (x, y), engine="legacy", allow_fallback=False,
            workspace_path=tmp_path / f"legacy-{shape}", save_charts=False, verbose=0,
        )
        legacy_scores.append(legacy.cv_best_score)
        legacy.close()

        native = nirs4all.run(
            pipeline, (x, y), engine="dag-ml", allow_fallback=False,
            workspace_path=tmp_path / f"dag-{shape}", save_charts=False, verbose=0,
        )
        assert native.execution_engine == "dag-ml"
        dag_scores.append(native.cv_best_score)
        archive = native.export(tmp_path / f"{shape}.n4a")
        predictions.append(np.asarray(nirs4all.predict(archive, x[:4]).y_pred))
        native.close()

    assert np.isfinite(legacy_scores).all()
    assert legacy_scores[0] == pytest.approx(legacy_scores[1])
    assert dag_scores[0] == pytest.approx(dag_scores[1])

    np.testing.assert_allclose(predictions[0], predictions[1], rtol=1e-6, atol=1e-6)


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_nested_model_training_controls_stay_on_selected_model(tmp_path, monkeypatch, mechanism: str) -> None:
    """CV overrides choose Ridge; refit overrides survive archive replay."""
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")

    rng = np.random.default_rng(125)
    x = rng.normal(size=(18, 8))
    y = 2 * x[:, 0] - x[:, 1] + rng.normal(size=18) * 0.1
    controlled = {"model": Ridge(alpha=1.0), "train_params": {"alpha": 0.2}, "refit_params": {"alpha": 3.0}}
    grouped = [KFold(2), [controlled, Lasso(alpha=100.0)]]
    flat = [KFold(2), controlled, Lasso(alpha=100.0)]

    legacy_scores = []
    dag_scores = []
    for shape, pipeline in (("grouped", grouped), ("flat", flat)):
        legacy = nirs4all.run(
            pipeline, (x, y), engine="legacy", allow_fallback=False,
            workspace_path=tmp_path / f"legacy-{shape}", save_charts=False, verbose=0,
        )
        legacy_scores.append(legacy.cv_best_score)
        legacy.close()

        native = nirs4all.run(
            pipeline, (x, y), engine="dag-ml", allow_fallback=False,
            workspace_path=tmp_path / f"dag-{shape}", save_charts=False, verbose=0,
        )
        assert native.best["model_name"] == "Ridge"
        dag_scores.append(native.cv_best_score)
        archive = native.export(tmp_path / f"{shape}.n4a")
        replay = np.asarray(nirs4all.predict(archive, x[:4]).y_pred).reshape(-1)
        np.testing.assert_allclose(replay, Ridge(alpha=3.0).fit(x, y).predict(x[:4]), rtol=1e-6, atol=1e-6)
        native.close()

    assert legacy_scores[0] == pytest.approx(legacy_scores[1])
    assert dag_scores[0] == pytest.approx(dag_scores[1])

    oof = np.empty_like(y)
    for train, validation in KFold(2).split(x):
        oof[validation] = Ridge(alpha=0.2).fit(x[train], y[train]).predict(x[validation])
    assert dag_scores[0] == pytest.approx(root_mean_squared_error(y, oof), abs=1e-6)
