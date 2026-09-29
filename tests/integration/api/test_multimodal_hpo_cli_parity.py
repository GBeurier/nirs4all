"""The CLI Methods optimizer and public Python API must score one fixed DAG alike."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import dag_ml
import numpy as np
import pytest
from nirs4all_io import MultimodalDataset
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

import nirs4all
from nirs4all.data.multimodal import MultimodalSpectroDataset
from nirs4all.pipeline.dagml.cli_runner import assemble_cv_refit_dsl, run_cv_refit_bundle, write_launcher_shim
from nirs4all.pipeline.dagml.envelope import build_envelope
from nirs4all.pipeline.dagml.folds import _build_folds, _split_group_grain
from nirs4all.pipeline.dagml.identity import mint_identity
from nirs4all.pipeline.dagml.in_process_runner import _load_subprocess_refit_artifacts
from nirs4all.pipeline.dagml_bridge import controller_manifests
from tests.integration.api.test_multimodal_dagml import _cohort, _model
from tests.integration.parity._dagml_cli import dagml_cli_path

_DAGML_ROOT = Path(__file__).resolve().parents[4] / "dag-ml"
_OPTIMIZER_ADAPTER = _DAGML_ROOT / "examples/adapters/hpo_n4m_optimizer.sh"


def _command(args: list[str], *, env: dict[str, str] | None = None) -> None:
    process = subprocess.run(args, capture_output=True, text=True, env=env, timeout=180, check=False)
    assert process.returncode == 0, process.stderr[-3000:]


@pytest.mark.skipif(
    not dagml_cli_path().exists() or not _OPTIMIZER_ADAPTER.exists(),
    reason="the sibling DAG-ML CLI and optional N4M adapter are required",
)
def test_cli_n4m_and_public_python_score_same_four_source_dag(tmp_path: Path) -> None:
    cohort: MultimodalDataset = _cohort()
    train_ids = [sample for sample, partition in zip(cohort.sample_ids, cohort.partitions, strict=True)
                 if partition == "train"]
    dataset = MultimodalSpectroDataset(cohort.take(train_ids))
    identity = mint_identity(dataset)
    pool = list(range(dataset.num_samples))
    splitter = GroupKFold(2)
    folds = _build_folds(splitter, dataset, pool, set())
    groups = _split_group_grain(splitter, dataset, pool)
    envelope = build_envelope(dataset, identity, sample_ints=pool, group_by_sample=groups)
    dsl = assemble_cv_refit_dsl(
        [{"model": _model()}], identity, envelope, folds,
        dsl_id="multimodal-hpo", n_splits=len(folds),
    )
    graph = json.loads(dag_ml.compile_pipeline_dsl_graph_json(json.dumps(dsl)))
    target = next(node["id"] for node in graph["nodes"] if node["kind"] == "model")
    for name, content in (
        ("dsl.json", dsl), ("envelope.json", envelope),
        ("graph.json", graph), ("controllers.json", controller_manifests()),
    ):
        (tmp_path / name).write_text(json.dumps(content), encoding="utf-8")
    with (tmp_path / "dataset.pkl").open("wb") as stream:
        pickle.dump(dataset, stream)
    shim = write_launcher_shim(tmp_path / "n4a_adapter", sys.executable)
    cli = str(dagml_cli_path())
    _command([
        cli, "build-pipeline-dsl-plan", "--dsl", str(tmp_path / "dsl.json"),
        "--controllers", str(tmp_path / "controllers.json"),
        "--plan-id", "plan:multimodal-hpo", "--output", str(tmp_path / "plan.json"),
    ])
    controls: dict[str, Any] = {
        "engine": "n4m", "sampler": "sobol", "seed": 19, "metric": "rmse",
        "n_trials": 3, "space": {"model__alpha": (0.01, 1.0)},
    }
    request = {
        "target_node": target, "trial_budget": 2, "metric": "rmse", "direction": "minimize",
        "fold_score_reduction": "mean",
        "optimizer_descriptor": {"n4m": {
            "state_path": str(tmp_path / "optimizer.n4mopt.json"),
            "sampler": "sobol", "pruner": "none", "seed": 19,
            "space": [{"name": "model.alpha", "kind": "float", "low": 0.01, "high": 1.0}],
        }},
    }
    env = {
        **os.environ,
        "N4A_DAGML_DATASET_PICKLE": str(tmp_path / "dataset.pkl"),
        "N4A_DAGML_GRAPH_PATH": str(tmp_path / "graph.json"),
        "N4A_DAGML_HPO_MODE": "1",
        "N4A_RANDOM_STATE": "19",
        "DAGML_N4M_PYTHON": sys.executable,
    }
    for budget in (2, 3):
        request["trial_budget"] = budget
        (tmp_path / "request.json").write_text(json.dumps(request), encoding="utf-8")
        _command([
            cli, "run-host-hpo", "--plan", str(tmp_path / "plan.json"),
            "--envelope", str(tmp_path / "envelope.json"),
            "--request", str(tmp_path / "request.json"),
            "--operator-adapter", str(shim),
            "--optimizer-adapter", str(_OPTIMIZER_ADAPTER),
            "--checkpoint", str(tmp_path / "native.json"),
            "--output", str(tmp_path / "outcome.json"),
        ], env=env)
    cli_outcome = json.loads((tmp_path / "outcome.json").read_text(encoding="utf-8"))
    assert cli_outcome["status"] == "completed"
    assert len(cli_outcome["trials"]) == 3

    result = nirs4all.run(
        [splitter, {"model": _model()}], cohort, tuning=controls,
        engine="dag-ml", workspace_path=tmp_path / "public",
        verbose=0, save_charts=False, random_state=19, refit=True,
    )
    try:
        public_trials = result.tuning_result.trials
        assert [trial["params"] for trial in cli_outcome["trials"]] == [
            trial.params for trial in public_trials
        ]
        assert [trial["score"] for trial in cli_outcome["trials"]] == pytest.approx(
            [trial.value for trial in public_trials], abs=1e-12
        )
        assert cli_outcome["selected_params"] == result.tuning_best_params
        assert len(result._dagml_refit_artifacts) == 1
        selected = clone(_model()).set_params(model__alpha=result.tuning_best_params["model.alpha"])
        selected_dsl = assemble_cv_refit_dsl(
            [{"model": selected}], identity, envelope, folds,
            dsl_id="multimodal-hpo", n_splits=len(folds),
        )
        selected_graph = json.loads(dag_ml.compile_pipeline_dsl_graph_json(json.dumps(selected_dsl)))
        refit_dir = tmp_path / "cli-refit"
        refit = run_cv_refit_bundle(
            dsl=selected_dsl, envelope=envelope, graph=selected_graph,
            dataset_path=str(tmp_path / "unused"), dataset_pickle=str(tmp_path / "dataset.pkl"),
            workdir=refit_dir, dagml_cli=cli, venv_python=sys.executable,
            random_state=19,
        )
        assert refit["returncode"] == 0, refit["stdout"][-3000:]
        artifacts = _load_subprocess_refit_artifacts(refit["results"], refit_dir / "refit_artifacts")
        assert len(artifacts) == 1
        cli_model = artifacts[0]["estimator"]
        public_model = result._dagml_refit_artifacts[0]["estimator"]
        prediction = _cohort(prediction=True)
        np.testing.assert_allclose(
            cli_model.predict([prediction.sources[name].values for name in cli_model.source_names]),
            public_model.predict([prediction.sources[name].values for name in public_model.source_names]),
            rtol=0, atol=0,
        )
    finally:
        result.close()
