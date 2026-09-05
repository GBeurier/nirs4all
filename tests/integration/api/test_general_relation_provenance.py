"""Aggregation lineage survives native persistence, archive and workspace replay."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


@pytest.mark.parametrize("representation", ["per_source_aggregate", "sample_aggregate"])
def test_relation_materialization_is_preserved_without_raw_relabeling(tmp_path, representation):
    import nirs4all
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.dagml.general_archive import load_general_archive
    from nirs4all.pipeline.dagml.general_workspace import load_general_workspace_chain
    from nirs4all.pipeline.dagml.native_results import read_native_results

    X = np.random.default_rng(63).normal(size=(24, 3))
    dataset = SpectroDataset("aggregated")
    dataset.add_samples(X, {"partition": "train"})
    dataset.add_targets(X[:, 0] + 0.13)
    recorded = {"representation": representation, "headers": ["MIR:1", "MIR:2", "MIR:3"], "shape": [24, 3], "source_ids": ["MIR"], "fingerprint": "recorded-materialization"}
    dataset._relation_materialization_manifest = recorded
    result = nirs4all.run([KFold(3), Ridge()], dataset, workspace_path=tmp_path)
    assert read_native_results(result._dagml_results_dir)["manifest"]["relation_materialization_manifest"] == recorded
    archive = result.export(tmp_path / "captured.n4a")
    assert load_general_archive(archive)["manifest"]["relation_materialization_manifest"] == recorded
    loaded = load_general_workspace_chain(tmp_path, result.best["chain_id"])
    assert loaded["manifest"]["relation_materialization_manifest"] == recorded
    result.close()
