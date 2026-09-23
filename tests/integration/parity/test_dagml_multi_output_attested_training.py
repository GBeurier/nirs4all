"""Real CV training must capture every independent output in one signed DAG outcome."""

from __future__ import annotations

import dag_ml
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from nirs4all.data.config import DatasetConfigs
from nirs4all.pipeline.dagml.attested_by_source import execute_attested_by_source_cv
from nirs4all.pipeline.dagml.cli_runner import data_bindings_for_nodes, split_invocation_for
from nirs4all.pipeline.dagml.envelope import build_envelope
from nirs4all.pipeline.dagml.folds import _build_folds
from nirs4all.pipeline.dagml.identity import mint_identity
from nirs4all.pipeline.dagml.run_paths import _canonical_source_branch
from nirs4all.pipeline.dagml_bridge import controller_manifests

from ._datasets import dataset_path


@pytest.mark.parity
def test_by_source_cv_executes_signed_multi_output_training() -> None:
    spectro = DatasetConfigs(dataset_path("multi")).get_dataset_at(0)
    identity = mint_identity(spectro)
    pool = spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(KFold(n_splits=3), spectro, pool, set())
    envelope = build_envelope(spectro, identity, sample_ints=pool)
    source_names = [f"source_{index}" for index in range(spectro.features_sources())]
    dsl = {
        "id": "nirs4all-attested-by-source-contract",
        "steps": [{"kind": "branch", "mode": "duplication", "branches": [
            _canonical_source_branch([{"model": Ridge(alpha=1.0)}], index)
            for index in range(len(source_names))
        ]}],
    }
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()
    model_ids = [node["id"] for node in graph["nodes"] if node["kind"] == "model"]
    dsl["data_bindings"] = data_bindings_for_nodes(model_ids, envelope)
    dsl["split_invocation"] = split_invocation_for(identity, folds, n_splits=len(folds))
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()

    captured = execute_attested_by_source_cv(
        dsl=dsl, envelope=envelope, graph=graph, spectro=spectro,
        identity=identity, folds=folds, source_names=source_names,
        selection_metric="rmse",
    )
    outcome = captured["training_result"].outcome.to_dict()
    package = captured["portable_package"].to_dict()
    expected = {f"output:source_{index}" for index in range(len(source_names))}
    assert {item["binding"]["binding_id"] for item in outcome["outputs"]} == expected
    assert {item["binding_id"] for item in package["output_bindings"]} == expected
    assert len(captured["refit_artifacts"]) == len(source_names)
    assert len(package["artifact_bindings"]) == len(source_names)
    assert captured["scores"]["reports"]
    assert outcome["training_request_fingerprint"]
