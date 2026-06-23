"""Integration tests for the dag-ml-cli runner (migration phase 2b-ii.3).

Two layers: (1) a CI-safe plan-level test that the assembled executable DSL actually produces
non-empty per-node data_bindings + a materialized fold_set (the fix for the empty-data_views /
fold_id=None defect), needing only the dag_ml wheel; (2) a skippable end-to-end test that runs
the real dag-ml-cli binary driving the nirs4all process adapter and checks the FIT_CV OOF
predictions match a direct sklearn KFold OOF — proving dag-ml operationally executes the
nirs4all core for a model-on-raw-features pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from nirs4all.data.config import DatasetConfigs
from nirs4all.pipeline.dagml.cli_runner import assemble_cv_refit_dsl, run_cv_refit_bundle
from nirs4all.pipeline.dagml.envelope import build_envelope
from nirs4all.pipeline.dagml.identity import mint_identity

from ._datasets import dataset_path

pytestmark = [pytest.mark.parity]

pytest.importorskip("dag_ml", reason="dag-ml not installed (nirs4all[dagml])")

_DAGML_CLI = Path(__file__).resolve().parents[3].parent / "dag-ml" / "target" / "release" / "dag-ml-cli"
_N_SPLITS = 3


def _setup(pipeline: list | None = None):
    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    identity = mint_identity(dataset)
    train = dataset.index_column("sample", {"partition": "train"})
    folds = [([train[j] for j in tr], [train[j] for j in va]) for tr, va in KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42).split(train)]
    envelope = build_envelope(dataset, identity, sample_ints=train)  # scoped to the CV universe
    return dataset, identity, train, folds, envelope, pipeline or [{"model": PLSRegression(n_components=5)}]


def test_assembled_dsl_binds_data_and_materializes_folds() -> None:
    """The augmented DSL compiles to a plan whose model node has a data binding + a fold set."""
    import dag_ml

    _, identity, _, folds, envelope, pipeline = _setup()
    from nirs4all.pipeline.dagml_bridge import controller_manifests

    dsl = assemble_cv_refit_dsl(pipeline, identity, envelope, folds, dsl_id="model_only", n_splits=_N_SPLITS)
    manifests = controller_manifests()
    artifact = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, manifests)
    plan = dag_ml.build_execution_plan("p", artifact.graph, artifact.campaign_template, manifests).to_dict()

    model_id = next(node_id for node_id, p in plan["node_plans"].items() if p["kind"] == "model")
    assert plan["node_plans"][model_id]["data_bindings"], "model node must carry a data binding"
    assert plan["fold_set"] is not None and len(plan["fold_set"]["folds"]) == _N_SPLITS


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_end_to_end_cli_oof_matches_sklearn(tmp_path) -> None:
    """dag-ml-cli drives the nirs4all adapter end-to-end; FIT_CV OOF == direct sklearn KFold."""
    import dag_ml

    dataset, identity, train, folds, envelope, pipeline = _setup()
    dsl = assemble_cv_refit_dsl(pipeline, identity, envelope, folds, dsl_id="model_only", n_splits=_N_SPLITS)
    from nirs4all.pipeline.dagml_bridge import controller_manifests

    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()
    outcome = run_cv_refit_bundle(
        dsl=dsl, envelope=envelope, graph=graph, dataset_path=dataset_path("regression"),
        workdir=tmp_path, dagml_cli=str(_DAGML_CLI), venv_python=sys.executable,
    )
    assert outcome["returncode"] == 0, outcome["stdout"][-2000:]

    # Collect dag-ml's OOF (FIT_CV validation) predictions from the adapter result frames.
    dagml_oof: dict[int, float] = {}
    for frame in outcome["results"]:
        result = frame.get("result") if frame.get("type") == "result" else frame
        for block in (result or {}).get("predictions", []):
            if block["partition"] == "validation":
                for sid, value in zip(block["sample_ids"], block["values"], strict=True):
                    dagml_oof[identity.to_int(sid)] = float(value[0])
    assert len(dagml_oof) == len(train)  # every train sample validated exactly once (KFold partition)

    # Direct sklearn KFold OOF on the real data, same folds.
    sklearn_oof: dict[int, float] = {}
    for train_ints, val_ints in folds:
        model = PLSRegression(n_components=5).fit(np.asarray(dataset.x({"sample": train_ints}, layout="2d")), np.asarray(dataset.y({"sample": train_ints})))
        for sample_int in val_ints:
            sklearn_oof[sample_int] = float(np.asarray(model.predict(np.asarray(dataset.x({"sample": [sample_int]}, layout="2d"))))[0][0])

    diffs = [abs(dagml_oof[k] - sklearn_oof[k]) for k in sklearn_oof]
    assert max(diffs) < 1e-5, f"OOF parity drift {max(diffs)} (PLS row-order FP noise is ~1e-6)"


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_end_to_end_cli_snv_pls_chain(tmp_path) -> None:
    """dag-ml-cli runs a PREPROCESSING+model pipeline (SNV->PLS); OOF == sklearn Pipeline(SNV, PLS).

    Proves cross-node feature chaining (A3) end-to-end: the model node reconstructs + applies the
    upstream SNV before fitting, all orchestrated by dag-ml over the process adapter.
    """
    import dag_ml
    from sklearn.pipeline import make_pipeline

    from nirs4all.operators.transforms.scalers import StandardNormalVariate
    from nirs4all.pipeline.dagml_bridge import controller_manifests

    dataset, identity, train, folds, envelope, pipeline = _setup([StandardNormalVariate(), {"model": PLSRegression(n_components=5)}])
    dsl = assemble_cv_refit_dsl(pipeline, identity, envelope, folds, dsl_id="snv_pls", n_splits=_N_SPLITS)
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()
    outcome = run_cv_refit_bundle(
        dsl=dsl, envelope=envelope, graph=graph, dataset_path=dataset_path("regression"),
        workdir=tmp_path, dagml_cli=str(_DAGML_CLI), venv_python=sys.executable,
    )
    assert outcome["returncode"] == 0, outcome["stdout"][-2000:]

    dagml_oof: dict[int, float] = {}
    for frame in outcome["results"]:
        result = frame.get("result") if frame.get("type") == "result" else frame
        for block in (result or {}).get("predictions", []):
            if block["partition"] == "validation":
                for sid, value in zip(block["sample_ids"], block["values"], strict=True):
                    dagml_oof[identity.to_int(sid)] = float(value[0])
    assert len(dagml_oof) == len(train)

    sklearn_oof: dict[int, float] = {}
    for train_ints, val_ints in folds:
        model = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=5))
        model.fit(np.asarray(dataset.x({"sample": train_ints}, layout="2d"), dtype=float), np.asarray(dataset.y({"sample": train_ints}), dtype=float))
        for sample_int in val_ints:
            sklearn_oof[sample_int] = float(np.asarray(model.predict(np.asarray(dataset.x({"sample": [sample_int]}, layout="2d"), dtype=float)))[0][0])

    diffs = [abs(dagml_oof[k] - sklearn_oof[k]) for k in sklearn_oof]
    assert max(diffs) < 1e-4, f"SNV->PLS chained OOF parity drift {max(diffs)}"
