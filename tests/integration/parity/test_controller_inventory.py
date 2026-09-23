"""Executable legacy-controller coverage boundaries for DAG-ML."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from nirs4all.controllers.registry import CONTROLLER_REGISTRY


def test_coverage_manifest_tracks_every_registered_legacy_controller() -> None:
    manifest = json.loads((Path(__file__).with_name("controller_coverage.json")).read_text(encoding="utf-8"))
    expected = {cls.__name__ for cls in CONTROLLER_REGISTRY}
    assert set(manifest["controllers"]) == expected
    for name, entry in manifest["controllers"].items():
        assert entry["route"] in {"native_graph", "host_operator", "host_preparation", "host_presentation", "gap", "fallback"}, name
        assert entry["coverage"] in {"public", "partial", "none", "not_pipeline"}, name
        for path in entry.get("evidence", []):
            assert (Path(__file__).parents[3] / path).is_file(), (name, path)


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parity
def test_fold_file_loader_imports_sample_ids_into_native_foldset(tmp_path, monkeypatch, mechanism: str) -> None:
    """A saved legacy fold file yields the same CV sample partition in DAG-ML."""
    import nirs4all
    from nirs4all.data.dataset import SpectroDataset

    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")

    features = np.arange(60, dtype=float).reshape(15, 4) / 20
    features[:, 1] = np.sin(features[:, 1])
    target = 1 + features[:, 0] * 2 + features[:, 1] * 3
    dataset = SpectroDataset("nonpositional_fold_ids")
    headers = [str(column) for column in range(4)]
    dataset.add_samples(features[:3], {"partition": "test"}, headers=headers)
    dataset.add_samples(features[3:], {"partition": "train"}, headers=headers)
    dataset.add_targets(target.reshape(-1, 1))
    train_ids = list(map(int, dataset.index_column("sample", {"partition": "train"})))
    assert train_ids[0] == 3  # file IDs are absolute, not offsets into the train matrix
    fold_file = tmp_path / "folds.json"
    fold_file.write_text(json.dumps([
        {"train": train_ids[:6], "val": train_ids[6:]},
        {"train": train_ids[6:], "val": train_ids[:6]},
    ]), encoding="utf-8")
    pipeline = [{"split": str(fold_file)}, {"model": Ridge(alpha=1.0)}]

    legacy = nirs4all.run(
        pipeline, dataset, engine="legacy", refit=False,
        workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert np.isfinite(legacy.cv_best_score)
    legacy.close()

    native = nirs4all.run(
        pipeline, dataset, engine="dag-ml", refit=False,
        workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert native.execution_engine == "dag-ml"
    assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-6)
    native.close()
