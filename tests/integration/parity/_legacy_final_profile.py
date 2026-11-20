"""Keep historical CV observations separate from genuine fitted FINAL evidence."""

from copy import copy
from pathlib import Path
from typing import Any

import numpy as np

from nirs4all.api.result import RunResult

from ..pipeline._cold_replay import run_cold_replay
from . import _oracle


def observe_cv(result: RunResult, task: str) -> dict[str, Any]:
    """Observe the actual CV rows without changing the original result or its IDs."""
    predictions = copy(result.predictions)
    predictions._buffer = [row for row in result.predictions.iter_entries() if str(row["fold_id"]) != "final"]
    return _oracle.observe(RunResult(predictions=predictions, per_dataset={}), task)


def assert_final_replay(result: RunResult, dataset: Any, destination: Path, expected_count: int) -> list[dict[str, Any]]:
    """Replay each exact fitted FINAL chain in a cold process without fit or store."""
    rows = result.predictions.filter_predictions(load_arrays=True)
    cv = [row for row in rows if str(row["fold_id"]) != "final"]
    finals = [row for row in rows if str(row["fold_id"]) == "final"]
    assert len(finals) == expected_count
    assert len(rows) == result.num_predictions == len(cv) + len(finals)
    assert all(row["refit_context"] in {"stacking", "standalone"} for row in finals)
    assert not {row["pipeline_uid"] for row in cv} & {row["pipeline_uid"] for row in finals}
    assert not {row["model_artifact_id"] for row in cv} & {row["model_artifact_id"] for row in finals}
    destination.mkdir()
    raw = dataset.get_dataset_at(0)
    arrays, archives = {}, []
    for index, chain_id in enumerate(dict.fromkeys(row["chain_id"] for row in finals)):
        assert chain_id
        chain_rows = [row for row in finals if row["chain_id"] == chain_id]
        assert len({row["model_artifact_id"] for row in chain_rows}) == 1
        assert chain_rows[0]["model_artifact_id"]
        chosen = next((row for row in chain_rows if row["partition"] == "test"), chain_rows[0])
        selector = {"sample": chosen["sample_indices"], "processing": [["raw"]]}
        arrays[f"x{index}"] = raw.x(selector, layout="2d")
        arrays[f"y{index}"] = np.asarray(chosen["y_pred"]).ravel()
        np.testing.assert_array_equal(raw._indexer.x_indices(selector), chosen["sample_indices"])
        archives.append(str(result.export(destination / f"{index}.n4a", source=chosen)))
    np.savez(destination / "inputs.npz", **arrays)
    script = r'''
import pathlib, sys
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from nirs4all.pipeline.bundle import BundleLoader
blocked = pathlib.Path(sys.argv[1]).resolve()
def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = pathlib.Path(args[0]).resolve()
        if path == blocked or blocked in path.parents:
            raise AssertionError('FINAL replay accessed the original workspace')
sys.addaudithook(audit)
def no_fit(*args, **kwargs):
    raise AssertionError('FINAL replay attempted fit')
for cls in (PLSRegression, PCA, TruncatedSVD, RandomForestRegressor, GradientBoostingRegressor, Ridge):
    cls.fit = no_fit
inputs = np.load(sys.argv[2])
for index, archive in enumerate(sys.argv[3:]):
    with BundleLoader(archive) as loader:
        np.testing.assert_allclose(np.asarray(loader.predict(inputs[f'x{index}'])).ravel(),
                                   inputs[f'y{index}'], rtol=1e-5, atol=1e-8)
'''
    assert result._workspace_path is not None
    completed = run_cold_replay(script, [str(result._workspace_path), str(destination / "inputs.npz"), *archives], destination)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return finals
