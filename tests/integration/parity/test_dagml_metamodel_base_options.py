"""Legacy-valid base-only and relation-profile stacking configurations."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.data.config import DatasetConfigs
from nirs4all.operators.models.meta import MetaModel, StackingConfig
from nirs4all.pipeline.dagml.detect import _detect_sequential_metamodel

from ._datasets import dataset_path


@pytest.mark.parametrize("option", ["allow_meta_sources", "relation_profile"])
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_base_stacking_options_preserve_legacy_and_replay(tmp_path, monkeypatch, option, mechanism):
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")

    config = StackingConfig(**{option: False if option == "allow_meta_sources" else True})
    pipeline = [KFold(2, shuffle=True, random_state=42), Ridge(), {"model": MetaModel(Ridge(), stacking_config=config)}]
    assert _detect_sequential_metamodel(pipeline) is not None
    path = dataset_path("regression")
    legacy = nirs4all.run(pipeline, path, engine="legacy", refit=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0)
    native = nirs4all.run(pipeline, path, engine="dag-ml", allow_fallback=False, refit=True,
                          workspace_path=tmp_path / "native", save_artifacts=True, save_charts=False, verbose=0)
    try:
        assert np.isfinite(legacy.cv_best_score)
        assert np.isfinite(native.cv_best_score)
        assert native.execution_engine == "dag-ml"
        final = [row for row in native.predictions._buffer
                 if row.get("partition") == "test" and row.get("fold_id") == "final"]
        assert len(final) == 1
        archive = native.export(tmp_path / f"{option}.n4a")
        dataset = DatasetConfigs(path).get_dataset_at(0)
        x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
        replay = np.asarray(nirs4all.predict(archive, x_test).y_pred).ravel()
        np.testing.assert_allclose(replay, np.asarray(final[0]["y_pred"]).ravel(), rtol=1e-5, atol=3e-4)
    finally:
        legacy.close()
        native.close()
