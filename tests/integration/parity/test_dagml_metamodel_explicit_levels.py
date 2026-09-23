"""Explicit MetaModel levels and raised depth limits on the native path."""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.data.config import DatasetConfigs
from nirs4all.operators.models.meta import MetaModel, StackingConfig, StackingLevel
from nirs4all.pipeline.dagml.detect import _detect_named_multi_level_metamodel
from nirs4all.pipeline.dagml.native_results import read_native_results

from ._datasets import dataset_path


def _pipeline(second_limit: int = 4) -> list:
    return [
        KFold(2, shuffle=True, random_state=42),
        Ridge(),
        {"model": MetaModel(Ridge(), source_models=["Ridge"], name="first",
                            stacking_config=StackingConfig(level=StackingLevel.LEVEL_1, max_level=4))},
        {"model": MetaModel(Ridge(), source_models=["first"], name="second",
                            stacking_config=StackingConfig(level=StackingLevel.LEVEL_2, max_level=second_limit))},
    ]


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_explicit_levels_with_raised_depth_limit_run_and_replay(tmp_path, monkeypatch, mechanism):
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")

    pipeline = _pipeline()
    assert _detect_named_multi_level_metamodel(pipeline) is not None
    assert _detect_named_multi_level_metamodel(_pipeline(second_limit=1)) is None
    path = dataset_path("regression")
    legacy = nirs4all.run(pipeline, path, engine="legacy", refit=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0)
    native = nirs4all.run(pipeline, path, engine="dag-ml", allow_fallback=False, refit=True,
                          workspace_path=tmp_path / "native", save_artifacts=True, save_charts=False, verbose=0)
    try:
        assert np.isfinite(legacy.cv_best_score)
        assert {row["model_name"] for row in legacy.predictions._buffer if row["partition"] == "val"} >= {"first", "second"}
        assert native.execution_engine == "dag-ml"
        assert np.isfinite(native.cv_best_score)
        producers = {report["producer_node"] for report in native._dagml_score_set["reports"]}
        assert {"merge:stack", "merge:stack.level2"} <= producers

        final = [row for row in native.predictions._buffer
                 if row.get("partition") == "test" and row.get("fold_id") == "final"]
        assert len(final) == 1
        archive = native.export(tmp_path / "explicit_levels.n4a")
        dataset = DatasetConfigs(path).get_dataset_at(0)
        x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
        replay = np.asarray(nirs4all.predict(archive, x_test).y_pred).ravel()
        np.testing.assert_allclose(replay, np.asarray(final[0]["y_pred"]).ravel(), rtol=1e-5, atol=3e-4)
    finally:
        legacy.close()
        native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_four_automatic_meta_levels_with_raised_limit_replay(tmp_path, monkeypatch, mechanism):
    """A fourth legacy-valid stacking level retains its full native replay chain."""
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")

    features, targets = make_regression(n_samples=36, n_features=6, noise=0.1, random_state=3)
    pipeline = [KFold(2, shuffle=True, random_state=3), Ridge()]
    previous = "Ridge"
    for level in range(1, 5):
        name = f"meta{level}"
        pipeline.append({"model": MetaModel(Ridge(), source_models=[previous], name=name,
                                      stacking_config=StackingConfig(max_level=4))})
        previous = name

    assert _detect_named_multi_level_metamodel(pipeline) is not None
    legacy = nirs4all.run(pipeline, (features, targets), engine="legacy", allow_fallback=False,
                          refit=False, workspace_path=tmp_path / "legacy", save_artifacts=False,
                          save_charts=False, verbose=0)
    native = nirs4all.run(pipeline, (features, targets), engine="dag-ml", allow_fallback=False,
                          refit=True, workspace_path=tmp_path / "native", save_artifacts=True,
                          save_charts=False, verbose=0)
    try:
        assert np.isfinite(legacy.cv_best_score)
        assert {row["model_name"] for row in legacy.predictions._buffer if row["partition"] == "val"} >= {
            "meta1", "meta2", "meta3", "meta4",
        }
        assert native.execution_engine == "dag-ml"
        assert np.isfinite(native.cv_best_score)
        stages = read_native_results(native._dagml_results_dir)["manifest"]["stacking_replay"]["stages"]
        assert len(stages) == 4
        archive = native.export(tmp_path / "four_levels.n4a")
        replay = np.asarray(nirs4all.predict(archive, features[:5]).y_pred).ravel()
        assert np.all(np.isfinite(replay))
    finally:
        legacy.close()
        native.close()
