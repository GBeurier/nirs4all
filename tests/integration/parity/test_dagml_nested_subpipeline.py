"""Public parity for legacy's sequential transform-only subpipelines."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import nirs4all

from ._dagml_cli import dagml_cli_path


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_nested_transform_chain_runs_and_replays_like_flat_chain(tmp_path, monkeypatch, mechanism: str) -> None:
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
    nested = [[StandardScaler(), MinMaxScaler()], KFold(2), Ridge(alpha=1.0)]
    flat = [StandardScaler(), MinMaxScaler(), KFold(2), Ridge(alpha=1.0)]

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
