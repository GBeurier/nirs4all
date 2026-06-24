"""Integration tests for the dag-ml-cli runner (migration phase 2b-ii.3).

Two layers: (1) a CI-safe plan-level test that the assembled executable DSL actually produces
non-empty per-node data_bindings + a materialized fold_set (the fix for the empty-data_views /
fold_id=None defect), needing only the dag_ml wheel; (2) a skippable end-to-end test that runs
the real dag-ml-cli binary driving the nirs4all process adapter and checks the FIT_CV OOF
predictions match a direct sklearn KFold OOF — proving dag-ml operationally executes the
nirs4all core for a model-on-raw-features pipeline.
"""

from __future__ import annotations

import json
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


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_end_to_end_cli_full_vertical_slice(tmp_path) -> None:
    """The full vertical-slice shape SNV + y_processing(MinMaxScaler) + PLS runs e2e (KFold).

    Exercises X-chaining AND a y_transform node together. MinMaxScaler-y is affine -> a no-op for
    PLS, so the OOF equals Pipeline(SNV, PLS); the point is that the full node shape executes.
    """
    import dag_ml
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler

    from nirs4all.operators.transforms.scalers import StandardNormalVariate
    from nirs4all.pipeline.dagml_bridge import controller_manifests

    pipe = [StandardNormalVariate(), {"y_processing": MinMaxScaler()}, {"model": PLSRegression(n_components=5)}]
    dataset, identity, train, folds, envelope, pipeline = _setup(pipe)
    dsl = assemble_cv_refit_dsl(pipeline, identity, envelope, folds, dsl_id="vslice", n_splits=_N_SPLITS)
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
    assert max(diffs) < 1e-4, f"full-vertical-slice OOF parity drift {max(diffs)}"


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_end_to_end_cli_persists_native_scores(tmp_path) -> None:
    """dag-ml scores NATIVELY: the adapter emits regression_targets, the CLI computes per-fold/final
    RMSE in Rust and persists bundle.scores — and they match sklearn (no Python-side scoring)."""
    import dag_ml
    from sklearn.metrics import mean_squared_error
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

    bundle = json.loads((tmp_path / "bundle.json").read_text())
    scores = bundle.get("scores")
    assert scores is not None, "dag-ml must persist native scores in the bundle"
    validation = {report["fold_id"]: report["metrics"]["rmse"] for report in scores["reports"] if report["partition"] == "validation"}
    assert all(f"fold{i}" in validation for i in range(_N_SPLITS))  # one native score per CV fold

    # dag-ml's native RMSE matches sklearn fold-for-fold (computed in Rust, not Python).
    for index, (train_ints, val_ints) in enumerate(folds):
        model = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=5))
        model.fit(np.asarray(dataset.x({"sample": train_ints}, layout="2d"), dtype=float), np.asarray(dataset.y({"sample": train_ints}), dtype=float))
        pred = np.asarray(model.predict(np.asarray(dataset.x({"sample": val_ints}, layout="2d"), dtype=float))).ravel()
        true = np.asarray(dataset.y({"sample": val_ints}), dtype=float).ravel()
        sklearn_rmse = float(np.sqrt(mean_squared_error(true, pred)))
        assert abs(validation[f"fold{index}"] - sklearn_rmse) < 1e-3, f"fold{index} native RMSE drift"

    # dag-ml also computes the cross-fold OOF average natively (the cv_best_score row).
    avg = [r for r in scores["reports"] if r["partition"] == "validation" and r["fold_id"] == "avg"]
    assert len(avg) == 1, "dag-ml must emit a native cross-fold OOF average (fold_id=avg)"
    oof_pred: dict[int, float] = {}
    oof_true: dict[int, float] = {}
    for train_ints, val_ints in folds:
        model = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=5))
        model.fit(np.asarray(dataset.x({"sample": train_ints}, layout="2d"), dtype=float), np.asarray(dataset.y({"sample": train_ints}), dtype=float))
        pred = np.asarray(model.predict(np.asarray(dataset.x({"sample": val_ints}, layout="2d"), dtype=float))).ravel()
        true = np.asarray(dataset.y({"sample": val_ints}), dtype=float).ravel()
        for position, sample_int in enumerate(val_ints):
            oof_pred[sample_int] = float(pred[position])
            oof_true[sample_int] = float(true[position])
    keys = sorted(oof_pred)
    oof_rmse = float(np.sqrt(mean_squared_error([oof_true[k] for k in keys], [oof_pred[k] for k in keys])))
    assert abs(avg[0]["metrics"]["rmse"] - oof_rmse) < 1e-3, "native OOF-average RMSE != sklearn OOF concat"
    assert avg[0]["row_count"] == len(keys)

    # dag-ml also produces + scores the FINAL model's TEST predictions natively (best_rmse) — the
    # refit model (fit on full train) predicts the held-out test partition in the same run.
    final_test = [r for r in scores["reports"] if r["partition"] == "test" and r["fold_id"] == "final"]
    assert len(final_test) == 1, "dag-ml must emit a native final-test score (best_rmse)"
    test_ints = dataset.index_column("sample", {"partition": "test"})
    final_model = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=5))
    final_model.fit(np.asarray(dataset.x({"sample": train}, layout="2d"), dtype=float), np.asarray(dataset.y({"sample": train}), dtype=float))
    test_pred = np.asarray(final_model.predict(np.asarray(dataset.x({"sample": test_ints}, layout="2d"), dtype=float))).ravel()
    test_true = np.asarray(dataset.y({"sample": test_ints}), dtype=float).ravel()
    sklearn_final_test = float(np.sqrt(mean_squared_error(test_true, test_pred)))
    assert abs(final_test[0]["metrics"]["rmse"] - sklearn_final_test) < 1e-3, "native final-test RMSE != sklearn"
    assert final_test[0]["row_count"] == len(test_ints)


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml() -> None:
    """The public `nirs4all.run(engine="dag-ml")` runs on the dag-ml engine and returns native scores."""
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline

    import nirs4all
    from nirs4all.operators.transforms.scalers import StandardNormalVariate

    pipeline = [StandardNormalVariate(), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}]
    result = nirs4all.run(pipeline, dataset_path("regression"), engine="dag-ml")

    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    train = dataset.index_column("sample", {"partition": "train"})
    test_ints = dataset.index_column("sample", {"partition": "test"})
    folds = [([train[i] for i in tr], [train[i] for i in va]) for tr, va in KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42).split(train)]

    # best_rmse == sklearn final-test (refit on full train, predict test)
    model = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=5))
    model.fit(np.asarray(dataset.x({"sample": train}, layout="2d"), dtype=float), np.asarray(dataset.y({"sample": train}), dtype=float))
    test_pred = np.asarray(model.predict(np.asarray(dataset.x({"sample": test_ints}, layout="2d"), dtype=float))).ravel()
    sklearn_final_test = float(np.sqrt(mean_squared_error(np.asarray(dataset.y({"sample": test_ints}), dtype=float).ravel(), test_pred)))
    assert abs(result.best_rmse - sklearn_final_test) < 1e-3

    # cv_best_score == sklearn OOF-concat
    oof_pred: dict[int, float] = {}
    oof_true: dict[int, float] = {}
    for train_ints, val_ints in folds:
        fold_model = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=5))
        fold_model.fit(np.asarray(dataset.x({"sample": train_ints}, layout="2d"), dtype=float), np.asarray(dataset.y({"sample": train_ints}), dtype=float))
        pred = np.asarray(fold_model.predict(np.asarray(dataset.x({"sample": val_ints}, layout="2d"), dtype=float))).ravel()
        true = np.asarray(dataset.y({"sample": val_ints}), dtype=float).ravel()
        for position, sample_int in enumerate(val_ints):
            oof_pred[sample_int], oof_true[sample_int] = float(pred[position]), float(true[position])
    keys = sorted(oof_pred)
    sklearn_oof = float(np.sqrt(mean_squared_error([oof_true[k] for k in keys], [oof_pred[k] for k in keys])))
    assert abs(result.cv_best_score - sklearn_oof) < 1e-3


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_shufflesplit() -> None:
    """engine="dag-ml" runs a NON-OOF (ShuffleSplit) CV: dag-ml relaxes the OOF check (Resampled
    fold set) and natively scores it. best_rmse + the resampled OOF-average == sklearn (per-sample
    aligned to dodge the storage-vs-request order trap)."""
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import ShuffleSplit
    from sklearn.pipeline import make_pipeline

    import nirs4all
    from nirs4all.operators.transforms.scalers import StandardNormalVariate

    splitter = ShuffleSplit(n_splits=4, test_size=0.25, random_state=42)
    pipeline = [StandardNormalVariate(), splitter, {"model": PLSRegression(n_components=5)}]
    result = nirs4all.run(pipeline, dataset_path("regression"), engine="dag-ml")

    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    train = dataset.index_column("sample", {"partition": "train"})
    test_ints = dataset.index_column("sample", {"partition": "test"})
    folds = [([train[i] for i in tr], [train[i] for i in va]) for tr, va in splitter.split(train)]

    def predict_one(model: object, sample_int: int) -> float:
        return float(np.asarray(model.predict(np.asarray(dataset.x({"sample": [sample_int]}, layout="2d"), dtype=float)))[0][0])

    def true_one(sample_int: int) -> float:
        return float(np.asarray(dataset.y({"sample": [sample_int]}), dtype=float).ravel()[0])

    # best_rmse == sklearn final-test (refit on full train)
    final = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=5))
    final.fit(np.asarray(dataset.x({"sample": train}, layout="2d"), dtype=float), np.asarray(dataset.y({"sample": train}), dtype=float))
    sklearn_final_test = float(np.sqrt(mean_squared_error([true_one(i) for i in test_ints], [predict_one(final, i) for i in test_ints])))
    assert abs(result.best_rmse - sklearn_final_test) < 1e-3

    # cv_best_score == sklearn resampled OOF average (a sample validated in K folds is averaged)
    acc: dict[int, float] = {}
    cnt: dict[int, int] = {}
    tru: dict[int, float] = {}
    for train_ints, val_ints in folds:
        model = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=5))
        model.fit(np.asarray(dataset.x({"sample": train_ints}, layout="2d"), dtype=float), np.asarray(dataset.y({"sample": train_ints}), dtype=float))
        for sample_int in val_ints:
            acc[sample_int] = acc.get(sample_int, 0.0) + predict_one(model, sample_int)
            cnt[sample_int] = cnt.get(sample_int, 0) + 1
            tru[sample_int] = true_one(sample_int)
    keys = sorted(acc)
    sklearn_oof = float(np.sqrt(mean_squared_error([tru[k] for k in keys], [acc[k] / cnt[k] for k in keys])))
    assert abs(result.cv_best_score - sklearn_oof) < 1e-3
