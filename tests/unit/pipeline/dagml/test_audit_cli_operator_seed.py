"""The process CLI must use the same seed as its compiled operator campaign."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import nirs4all
from nirs4all.pipeline.dagml.run_backend import _default_dagml_cli


@pytest.mark.parametrize("seed", [None, 0, 17])
def test_real_operator_cli_seed_matches_signed_campaign(monkeypatch, tmp_path, seed):
    cli = _default_dagml_cli()
    if not cli.is_file():
        pytest.skip("local DAG CLI is required for process qualification")
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0")
    monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    original = subprocess.run
    seen = []

    def execute(command, *args, **kwargs):
        if "run-process-dsl-cv-refit-bundle" in command:
            dsl = json.loads(Path(command[command.index("--dsl") + 1]).read_text())
            actual = int(command[command.index("--root-seed") + 1])
            assert actual == dsl["root_seed"] == (0 if seed is None else seed)
            seen.append(actual)
        return original(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", execute)
    features = np.random.default_rng(19).normal(size=(24, 6))
    with nirs4all.run(
        [{"_or_": [StandardScaler(), MinMaxScaler()]}, KFold(3), {"model": Ridge()}],
        {"X": features, "y": features[:, 0] - features[:, 1]},
        engine="dag-ml", random_state=seed, refit=True, workspace_path=tmp_path,
        save_artifacts=False, save_charts=False, verbose=0,
    ) as result:
        assert np.isfinite(result.cv_best_score)
        rows = result.predictions.filter_predictions(partition="val", fold_id="0")
        assert len(rows) == 2
        assert len({row["config_name"] for row in rows}) == 2
    assert seen == [0 if seed is None else seed]
