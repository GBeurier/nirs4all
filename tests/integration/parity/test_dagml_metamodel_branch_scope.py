"""Public oracles for legacy MetaModel branch scope on the native DAG path."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.data.config import DatasetConfigs
from nirs4all.operators.models.meta import BranchScope, MetaModel, StackingConfig
from nirs4all.pipeline.dagml.detect import _detect_sequential_metamodel

from ._datasets import dataset_path


def _backend(monkeypatch: pytest.MonkeyPatch, mechanism: str) -> None:
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")


def _meta_rows(result: nirs4all.RunResult, *, partition: str, fold_id: str) -> list[dict]:
    return [row for row in result.predictions._buffer
            if row.get("model_name") == "MetaModel_Ridge"
            and row.get("partition") == partition and row.get("fold_id") == fold_id]


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("syntax", ["named", "list"])
def test_all_branches_metamodel_keeps_branch_views_and_replays_archive(tmp_path, monkeypatch, mechanism, syntax):
    _backend(monkeypatch, mechanism)
    bodies = {
        "raw": [{"model": PLSRegression(2), "name": "Raw"}],
        "other": [{"model": Ridge(), "name": "Other"}],
    }
    pipeline = [
        KFold(2, shuffle=True, random_state=42),
        {"branch": bodies if syntax == "named" else list(bodies.values())},
        {"model": MetaModel(Ridge(), stacking_config=StackingConfig(branch_scope=BranchScope.ALL_BRANCHES))},
    ]
    path = dataset_path("regression")
    # Legacy evaluates this shape but its refit cannot reconstruct OOF rows.
    legacy = nirs4all.run(pipeline, path, engine="legacy", refit=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0)
    native = nirs4all.run(pipeline, path, engine="dag-ml", allow_fallback=False,
                          workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0)
    try:
        assert native.execution_engine == "dag-ml"
        branch_names = ("raw", "other") if syntax == "named" else ("branch_0", "branch_1")
        for result in (legacy, native):
            for fold_id in ("0", "1"):
                rows = _meta_rows(result, partition="val", fold_id=fold_id)
                assert {(row["branch_id"], row["branch_name"]) for row in rows} == set(enumerate(branch_names))
                by_branch = {row["branch_id"]: row for row in rows}
                assert set(by_branch[0]["sample_indices"]) == set(by_branch[1]["sample_indices"])
                np.testing.assert_allclose(by_branch[0]["y_pred"], by_branch[1]["y_pred"])

        dataset = DatasetConfigs(path).get_dataset_at(0)
        x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
        final_rows = _meta_rows(native, partition="test", fold_id="final")
        assert {row["branch_id"] for row in final_rows} == {0, 1}
        for row in final_rows:
            archive = native.export(tmp_path / f"branch_{row['branch_id']}.n4a", source=row)
            replay = np.asarray(nirs4all.predict(archive, x_test).y_pred).ravel()
            np.testing.assert_allclose(replay, np.asarray(row["y_pred"]).ravel(), atol=1e-4)
    finally:
        legacy.close()
        native.close()


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_specified_scope_selects_named_sequential_source_and_replays_archive(tmp_path, monkeypatch, mechanism):
    _backend(monkeypatch, mechanism)
    pipeline = [
        KFold(2, shuffle=True, random_state=42),
        PLSRegression(2),
        Ridge(alpha=1000),
        {"model": MetaModel(Ridge(), source_models=["PLSRegression"],
                            stacking_config=StackingConfig(branch_scope=BranchScope.SPECIFIED))},
    ]
    detected = _detect_sequential_metamodel(pipeline)
    assert detected is not None and detected[2] == [{"model": "branch:0.node:0"}]
    path = dataset_path("regression")
    legacy = nirs4all.run(pipeline, path, engine="legacy", refit=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0)
    native = nirs4all.run(pipeline, path, engine="dag-ml", allow_fallback=False,
                          workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0)
    try:
        assert np.isfinite(legacy.cv_best_score)
        assert native.execution_engine == "dag-ml"
        assert np.isfinite(native.cv_best_score)
        assert all(_meta_rows(native, partition="val", fold_id=fold) for fold in ("0", "1"))
        row = _meta_rows(native, partition="test", fold_id="final")
        assert len(row) == 1
        archive = native.export(tmp_path / "specified.n4a")
        dataset = DatasetConfigs(path).get_dataset_at(0)
        x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
        replay = np.asarray(nirs4all.predict(archive, x_test).y_pred).ravel()
        np.testing.assert_allclose(replay, np.asarray(row[0]["y_pred"]).ravel(), atol=1e-4)
    finally:
        legacy.close()
        native.close()
