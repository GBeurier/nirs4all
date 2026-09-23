"""A fold chart must show the native scored train/validation membership."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

pytest.importorskip("dag_ml")


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parity
def test_fold_chart_uses_all_legacy_matching_scored_folds(tmp_path, monkeypatch, mechanism: str) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml CLI binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))

    import nirs4all

    rng = np.random.default_rng(21)
    x = rng.normal(size=(18, 8)).astype(np.float32)
    y = (x[:, 0] + x[:, 1]).astype(np.float32)
    pipeline = [KFold(3), "fold_chart", {"model": Ridge()}]
    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", workspace_path=tmp_path / "legacy", save_charts=False, verbose=0)
    expected = [[train.tolist(), val.tolist()] for train, val in KFold(3).split(x)]
    legacy_folds: dict[str, dict[str, list[int]]] = {}
    for row in legacy.predictions.filter_predictions(load_arrays=True):
        fold = str(row.get("fold_id", ""))
        partition = row.get("partition")
        if fold in {"0", "1", "2"} and partition in {"train", "val"}:
            legacy_folds.setdefault(fold, {})[partition] = list(row["sample_indices"])
    assert [[legacy_folds[str(index)]["train"], legacy_folds[str(index)]["val"]] for index in range(3)] == expected

    native = nirs4all.run(pipeline, (x, y), engine="dag-ml", workspace_path=tmp_path / mechanism, save_charts=True, plots_visible=False, verbose=0)
    assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-5)
    chart_files = list((tmp_path / mechanism / "charts").rglob("*.json"))
    assert len(chart_files) == 1
    assert json.loads(chart_files[0].read_text(encoding="utf-8"))["folds"] == expected
    legacy.close()
    native.close()
