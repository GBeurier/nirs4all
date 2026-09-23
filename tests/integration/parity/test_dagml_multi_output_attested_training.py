"""Real CV training must capture every independent output in one signed DAG outcome."""

from __future__ import annotations

import dag_ml
import numpy as np
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

from ._dagml_cli import dagml_cli_path
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
    assert {report["partition"] for report in captured["scores"]["reports"]} == {"validation", "final", "test"}
    assert all({block["partition"] for block in item["predictions"]} == {"final"} for item in outcome["outputs"])
    assert outcome["training_request_fingerprint"]


@pytest.mark.parity
def test_public_by_source_cv_attested_capture_preserves_cli_predictions(monkeypatch) -> None:
    import nirs4all

    cli = dagml_cli_path()
    if not cli.exists():
        pytest.skip(f"dag-ml-cli binary not built at {cli}")
    pipeline = [
        KFold(n_splits=3),
        {"branch": {"by_source": True, "steps": {
            f"source_{index}": [{"model": Ridge(alpha=1.0)}] for index in range(3)
        }}},
        {"merge": "auto"},
    ]
    monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    attested = nirs4all.run(pipeline, dataset_path("multi"), engine="dag-ml", save_artifacts=False, verbose=0)
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0")
    cli_result = nirs4all.run(pipeline, dataset_path("multi"), engine="dag-ml", save_artifacts=False, verbose=0)
    assert len(attested._dagml_training_outcome["outputs"]) == 3
    assert len(attested._dagml_portable_predictor_package["output_bindings"]) == 3
    assert attested.num_predictions == cli_result.num_predictions == 18
    def key(row: dict) -> tuple:
        return row["branch_id"], row["partition"], row["fold_id"]
    attested_rows = sorted(attested.predictions.filter_predictions(load_arrays=True), key=key)
    cli_rows = sorted(cli_result.predictions.filter_predictions(load_arrays=True), key=key)
    assert [key(row) for row in attested_rows] == [key(row) for row in cli_rows]
    for actual, expected in zip(attested_rows, cli_rows, strict=True):
        np.testing.assert_allclose(actual["y_pred"], expected["y_pred"], atol=1e-6)
