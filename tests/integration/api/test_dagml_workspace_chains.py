"""A persisted native run is visible to ordinary workspace result readers."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def test_general_run_populates_chain_summaries(tmp_path):
    import nirs4all
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    rng = np.random.default_rng(17)
    X = rng.normal(size=(30, 5))
    result = nirs4all.run(
        [StandardScaler(), KFold(3), Ridge()], (X, X @ np.arange(1.0, 6.0)),
        workspace_path=tmp_path, save_artifacts=True,
    )
    with WorkspaceStore(tmp_path) as store:
        chains = store.query_chain_summaries().to_dicts()
    assert chains
    cv_chains = [chain for chain in chains if chain["cv_fold_count"] == 3]
    assert len(cv_chains) == 1, chains
    assert cv_chains[0]["model_name"] == "Ridge"
    assert cv_chains[0]["metric"] == "rmse"
    assert cv_chains[0]["dataset_name"] == "array_dataset"
    assert np.isfinite(cv_chains[0]["cv_val_score"])
    assert result.execution_engine == "dag-ml"
