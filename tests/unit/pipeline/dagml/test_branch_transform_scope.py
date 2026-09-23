"""Native branch transforms fit only the samples selected by their data view."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from nirs4all.pipeline.dagml import node_runner


class _Resolver:
    def __init__(self) -> None:
        self.requested: list[list[str]] = []

    def target_sample_ids(self, ids: list[str]) -> list[str]:
        return ids

    def resolve_targets(self, ids: list[str]) -> dict:
        return {"values": [float(index) for index, _ in enumerate(ids)]}

    def is_multi_source(self) -> bool:
        return False

    def resolve_features(self, ids: list[str], *, include_augmented: bool) -> dict:
        assert not include_augmented
        self.requested.append(list(ids))
        values = {"a1": 1.0, "a2": 3.0, "b1": 100.0, "b2": 200.0}
        return {"values": np.asarray([[values[sample_id]] for sample_id in ids])}


def test_fitted_transform_honors_metadata_branch_view(monkeypatch) -> None:
    monkeypatch.setattr(node_runner, "route_graph_node", lambda *_args, **_kwargs: StandardScaler())
    task = {
        "run_id": "branch-transform-test",
        "phase": "FIT_CV",
        "fold_id": "fold:0",
        "node_plan": {
            "node_id": "transform:group-a",
            "kind": "transform",
            "controller_id": "controller:transform",
            "controller_version": "1",
            "params_fingerprint": "test",
        },
        "data_views": {
            "data:x": {
                "partition": "fold_train",
                "sample_ids": ["a1", "b1", "a2", "b2"],
                "branch_view": {"selector": {"metadata": {"group": "A"}}},
                "extra": {},
            },
        },
    }
    metadata = {sample_id: {"group": sample_id[0].upper()} for sample_id in ["a1", "a2", "b1", "b2"]}
    resolver = _Resolver()
    store = {}

    result = node_runner.run_node(task, resolver, lambda _node_id: {"metadata": {}}, store, sample_metadata=metadata)

    assert resolver.requested == [["a1", "a2"]]
    fitted = store[result["outputs"]["x_out"]["handle"]]
    assert fitted.steps[0].mean_[0] == 2.0
