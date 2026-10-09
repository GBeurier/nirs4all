"""A selected captured predictor can reopen as a fresh executable recipe."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.api.workspace_chain_snapshot import workspace_chain_snapshot
from nirs4all.pipeline.config.component_serialization import deserialize_component

pytestmark = pytest.mark.parity


def test_captured_chain_snapshot_retains_recipe_and_reruns_on_new_cohort(tmp_path):
    rng = np.random.default_rng(41)
    x = rng.normal(size=(30, 8))
    y = x[:, 0] - 0.5 * x[:, 1]
    result = nirs4all.run([KFold(3), StandardScaler(with_std=False), {"model": PLSRegression(3, scale=False)}],
                         (x, y), engine="dag-ml", verbose=0, save_charts=False, workspace_path=tmp_path / "original")
    try:
        chain_id = result.best["chain_id"]
    finally:
        result.close()
    snapshot = workspace_chain_snapshot(tmp_path / "original", chain_id)
    assert snapshot is not None
    restored = deserialize_component(snapshot)
    assert isinstance(restored[0], KFold)
    assert restored[1].with_std is False
    assert restored[-1]["model"].n_components == 3
    assert not hasattr(restored[-1]["model"], "coef_")
    # Different row count proves old captured sample/fold identities are absent.
    rerun = nirs4all.run(snapshot, (x[:24], y[:24]), engine="dag-ml", verbose=0,
                        save_charts=False, workspace_path=tmp_path / "reopened")
    try:
        assert np.isfinite(rerun.cv_best_score)
        assert rerun.best["chain_id"] != chain_id
    finally:
        rerun.close()
