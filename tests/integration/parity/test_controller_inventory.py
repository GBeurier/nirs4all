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


@pytest.mark.parity
@pytest.mark.parametrize("gate", [False, 0.25])
@pytest.mark.parametrize("preprocessing", [False, True])
def test_residual_model_native_graph_fits_oof_targets_and_fuses_predictions(tmp_path, gate, preprocessing) -> None:
    """The learner fits OOF-derived residuals; DAG-ML fuses held-out predictions."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    import nirs4all
    from nirs4all.operators.models.residual import ResidualModel

    from ._datasets import dataset_path

    pipeline = [
        KFold(2, shuffle=True, random_state=1),
        *([StandardScaler()] if preprocessing else []),
        {"model": ResidualModel(base=PLSRegression(n_components=2), learner=Ridge(alpha=1.0), gate=gate)},
    ]
    legacy = nirs4all.run(
        pipeline, dataset_path("regression"), engine="legacy", refit=False,
        workspace_path=tmp_path / f"legacy-residual-{gate}-{preprocessing}", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert np.isfinite(legacy.cv_best_score)
    assert np.isfinite(legacy.best_rmse)
    legacy.close()

    native = nirs4all.run(
        pipeline, dataset_path("regression"), engine="dag-ml", refit=True,
        workspace_path=tmp_path / f"native-residual-{gate}-{preprocessing}", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert native.execution_engine == "dag-ml"
    assert np.isfinite(native.cv_best_score)
    assert np.isfinite(native.best_rmse)
    frames = native._dagml_node_results
    by_node_fold = {
        (frame["node_id"], prediction["partition"], prediction.get("fold_id")): prediction
        for frame in frames for prediction in frame.get("predictions", [])
    }
    fusion_id = "model:residual.learner.residual_fusion"
    fusion_rows = 0
    for (node_id, partition, fold_id), fused in by_node_fold.items():
        if node_id != fusion_id or partition not in {"validation", "test"}:
            continue
        fusion_rows += len(fused["sample_ids"])
        base = by_node_fold[("branch:0.node:0", partition, fold_id)]
        learner = by_node_fold[("model:residual.learner", partition, fold_id)]
        base_rows = dict(zip(base["sample_ids"], base["values"], strict=True))
        learner_rows = dict(zip(learner["sample_ids"], learner["values"], strict=True))
        for sample_id, fused_row in zip(fused["sample_ids"], fused["values"], strict=True):
            expected = np.asarray(base_rows[sample_id]) + float(gate if gate is not False else 1) * np.asarray(learner_rows[sample_id])
            assert fused_row == pytest.approx(expected, abs=1e-8)
    if any(frame.get("node_id") == fusion_id for frame in frames):  # CLI bundles omit merge-node frames.
        assert fusion_rows > 0
    from nirs4all.data import DatasetConfigs

    archive = native.export(tmp_path / f"residual_{gate}_{preprocessing}.n4a")
    fresh = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    replay = nirs4all.predict(archive, fresh.x({"partition": "test"}, layout="2d"))
    replay_rmse = np.sqrt(np.mean((np.asarray(fresh.y({"partition": "test"})).ravel() - np.asarray(replay.y_pred).ravel()) ** 2))
    assert replay_rmse == pytest.approx(native.best_rmse, abs=1e-5)
    native.close()
