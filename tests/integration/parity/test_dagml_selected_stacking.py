"""Public parity oracles for selected and weighted stacking inputs."""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.operators.models.meta import MetaModel
from nirs4all.pipeline.dagml.detect import (
    _detect_proba_mean_stacking_branch,
    _detect_sequential_metamodel,
)


def _data():
    return make_regression(n_samples=48, n_features=6, noise=0.1, random_state=42)


def test_sequential_metamodel_selects_named_source(tmp_path):
    meta_by_source = {}
    for source in ("PLSRegression", "Ridge"):
        pipeline = [
            KFold(3, shuffle=True, random_state=42),
            {"model": PLSRegression(n_components=2)},
            {"model": Ridge(alpha=10000)},
            {"model": MetaModel(Ridge(alpha=1), source_models=[source])},
        ]
        detected = _detect_sequential_metamodel(pipeline)
        assert detected is not None
        assert detected[2] == [{"model": "branch:0.node:0"}]
        legacy = nirs4all.run(pipeline, _data(), engine="legacy", refit=False,
                             save_artifacts=False, save_charts=False, verbose=0)
        assert np.isfinite(legacy.cv_best_score)
        native = nirs4all.run(pipeline, _data(), engine="dag-ml", refit=False,
                             allow_fallback=False, workspace_path=tmp_path / source,
                             save_artifacts=False, save_charts=False, verbose=0)
        try:
            assert native.execution_engine == "dag-ml"
            assert np.isfinite(native.cv_best_score)
            meta_by_source[source] = {
                block["fold_id"]: np.asarray(block["values"], dtype=float)
                for node in native._dagml_node_results
                for block in node.get("predictions", [])
                if block.get("producer_node") == "merge:stack"
                and block.get("partition") == "validation"
            }
        finally:
            native.close()
    assert meta_by_source["PLSRegression"].keys() == meta_by_source["Ridge"].keys()
    assert any(
        not np.allclose(meta_by_source["PLSRegression"][fold], meta_by_source["Ridge"][fold])
        for fold in meta_by_source["PLSRegression"]
    )


def test_branch_numeric_prediction_aggregation(tmp_path):
    meta_by_aggregate = {}
    for aggregate in ("mean", "weighted_mean"):
        pipeline = [
            KFold(3, shuffle=True, random_state=42),
            {"branch": [
                [{"model": PLSRegression(n_components=2)}, {"model": Ridge(alpha=10000)}],
                [{"model": Ridge(alpha=1)}, {"model": Ridge(alpha=1000)}],
            ]},
            {"merge": {"predictions": [
                {"branch": 0, "aggregate": aggregate},
                {"branch": 1, "aggregate": aggregate},
            ]}},
            {"model": Ridge(alpha=0.1)},
        ]
        detected = _detect_proba_mean_stacking_branch(pipeline)
        assert detected is not None
        assert [selector["aggregate"] for selector in detected[2]] == [aggregate, aggregate]
        legacy = nirs4all.run(pipeline, _data(), engine="legacy", refit=False,
                             save_artifacts=False, save_charts=False, verbose=0)
        assert np.isfinite(legacy.cv_best_score)
        native = nirs4all.run(pipeline, _data(), engine="dag-ml", refit=False,
                             allow_fallback=False, workspace_path=tmp_path / aggregate,
                             save_artifacts=False, save_charts=False, verbose=0)
        try:
            assert native.execution_engine == "dag-ml"
            assert np.isfinite(native.cv_best_score)
            meta_by_aggregate[aggregate] = {
                block["fold_id"]: np.asarray(block["values"], dtype=float)
                for node in native._dagml_node_results
                for block in node.get("predictions", [])
                if block.get("producer_node") == "merge:stack"
                and block.get("partition") == "validation"
            }
        finally:
            native.close()
    assert meta_by_aggregate["mean"].keys() == meta_by_aggregate["weighted_mean"].keys()
    assert any(
        not np.allclose(meta_by_aggregate["mean"][fold], meta_by_aggregate["weighted_mean"][fold])
        for fold in meta_by_aggregate["mean"]
    )
