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

from ._datasets import PARSER_FIXTURES, dataset_path

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


def test_dagml_engine_coverage_boundary() -> None:
    """The engine='dag-ml' coverage gate (ADR-17 cutover criterion).

    SUPPORTED today (each has its own e2e parity test above): regression + classification, KFold +
    ShuffleSplit, single + multi-model, any sklearn estimator, preprocessing chains, y_processing,
    generators (_or_/_range_/_grid_), hyperparameter sweeps. UNSUPPORTED features must fail LOUDLY
    (a clear NotImplementedError from the bridge) — never silently produce a wrong result — so the
    default can only be flipped to dag-ml once these are covered. This test pins that boundary; drop
    a keyword from `unsupported` as each gets implemented.
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import MinMaxScaler

    from nirs4all.operators.transforms.scalers import StandardNormalVariate
    from nirs4all.pipeline.dagml_bridge import pipeline_to_dsl

    unsupported = {
        "exclude": {"exclude": {"partition": "val"}},
        "branch": {"branch": [[StandardNormalVariate()], [MinMaxScaler()]]},
        "sample_augmentation": {"sample_augmentation": StandardNormalVariate()},
        "feature_augmentation": {"feature_augmentation": StandardNormalVariate()},
        "merge": {"merge": "predictions"},
    }
    for keyword, step in unsupported.items():
        with pytest.raises(NotImplementedError, match=keyword):
            pipeline_to_dsl([step, {"model": PLSRegression(n_components=5)}], "boundary")


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
    # refit model (fit on full train) predicts the held-out test partition in the same run. The block
    # carries `fold_id=None` (the off-fold convention dag-ml keys on for REFIT-test / merge reassembly).
    final_test = [r for r in scores["reports"] if r["partition"] == "test" and r["fold_id"] is None]
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


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_generator_or() -> None:
    """engine="dag-ml" expands a `_or_` generator, runs each variant, and selects the best by CV.

    Two preprocessings (SNV vs MinMaxScaler) → two variants; dag-ml runs both natively and the
    backend returns the lower-CV one, with best_rmse = that variant's final-test (both == sklearn)."""
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler

    import nirs4all
    from nirs4all.operators.transforms.scalers import StandardNormalVariate

    preprocessings: list[type] = [StandardNormalVariate, MinMaxScaler]
    pipeline = [{"_or_": [StandardNormalVariate(), MinMaxScaler()]}, KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}]
    result = nirs4all.run(pipeline, dataset_path("regression"), engine="dag-ml")

    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    train = dataset.index_column("sample", {"partition": "train"})
    test_ints = dataset.index_column("sample", {"partition": "test"})
    folds = [([train[i] for i in tr], [train[i] for i in va]) for tr, va in KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42).split(train)]

    def predict_one(model: object, sample_int: int) -> float:
        return float(np.asarray(model.predict(np.asarray(dataset.x({"sample": [sample_int]}, layout="2d"), dtype=float)))[0][0])

    def variant_scores(prep_cls: type) -> tuple[float, float]:
        acc: dict[int, float] = {}
        cnt: dict[int, int] = {}
        tru: dict[int, float] = {}
        for train_ints, val_ints in folds:
            model = make_pipeline(prep_cls(), PLSRegression(n_components=5))
            model.fit(np.asarray(dataset.x({"sample": train_ints}, layout="2d"), dtype=float), np.asarray(dataset.y({"sample": train_ints}), dtype=float))
            for sample_int in val_ints:
                acc[sample_int] = acc.get(sample_int, 0.0) + predict_one(model, sample_int)
                cnt[sample_int] = cnt.get(sample_int, 0) + 1
                tru[sample_int] = float(np.asarray(dataset.y({"sample": [sample_int]}), dtype=float).ravel()[0])
        keys = sorted(acc)
        cv = float(np.sqrt(mean_squared_error([tru[k] for k in keys], [acc[k] / cnt[k] for k in keys])))
        final = make_pipeline(prep_cls(), PLSRegression(n_components=5))
        final.fit(np.asarray(dataset.x({"sample": train}, layout="2d"), dtype=float), np.asarray(dataset.y({"sample": train}), dtype=float))
        test_rmse = float(np.sqrt(mean_squared_error([float(np.asarray(dataset.y({"sample": [i]}), dtype=float).ravel()[0]) for i in test_ints], [predict_one(final, i) for i in test_ints])))
        return cv, test_rmse

    scored = {cls: variant_scores(cls) for cls in preprocessings}
    best_cls = min(scored, key=lambda cls: scored[cls][0])  # lowest CV wins
    best_cv, best_test = scored[best_cls]
    assert abs(result.cv_best_score - best_cv) < 1e-3  # backend selected the best-CV variant
    assert abs(result.best_rmse - best_test) < 1e-3  # and reports that variant's final-test


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_param_sweep() -> None:
    """engine="dag-ml" applies sibling hyperparameters and sweeps them: `n_components: {_range_}`.

    Three n_components (3/9/15) → three variants; the sibling param is applied to the model (not the
    default), and the backend selects the best-CV one with cv_best_score == sklearn for that value."""
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline

    import nirs4all
    from nirs4all.operators.transforms.scalers import StandardNormalVariate

    components = [3, 9, 15]
    pipeline = [StandardNormalVariate(), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(), "n_components": {"_range_": [3, 16, 6]}}]
    result = nirs4all.run(pipeline, dataset_path("regression"), engine="dag-ml")

    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    train = dataset.index_column("sample", {"partition": "train"})
    folds = [([train[i] for i in tr], [train[i] for i in va]) for tr, va in KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42).split(train)]

    def oof_cv(n_components: int) -> float:
        acc: dict[int, float] = {}
        cnt: dict[int, int] = {}
        tru: dict[int, float] = {}
        for train_ints, val_ints in folds:
            model = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=n_components))
            model.fit(np.asarray(dataset.x({"sample": train_ints}, layout="2d"), dtype=float), np.asarray(dataset.y({"sample": train_ints}), dtype=float))
            for sample_int in val_ints:
                acc[sample_int] = acc.get(sample_int, 0.0) + float(np.asarray(model.predict(np.asarray(dataset.x({"sample": [sample_int]}, layout="2d"), dtype=float)))[0][0])
                cnt[sample_int] = cnt.get(sample_int, 0) + 1
                tru[sample_int] = float(np.asarray(dataset.y({"sample": [sample_int]}), dtype=float).ravel()[0])
        keys = sorted(acc)
        return float(np.sqrt(mean_squared_error([tru[k] for k in keys], [acc[k] / cnt[k] for k in keys])))

    best_cv = min(oof_cv(nc) for nc in components)  # the sweep must pick the best n_components
    assert abs(result.cv_best_score - best_cv) < 1e-3


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_native_param_sweep(tmp_path) -> None:
    """A PLS `n_components` `_range_` sweep runs as ONE NATIVE dag-ml generation + SELECT run.

    The bridge lowers the sweep to native dag-ml `generators` (no Python `expand_spec`), so the
    compiler expands the variants and dag-ml runs generation + per-variant CV scoring + SELECT +
    refit in a single CLI invocation. We assert both: (1) it is one native run (param_model path,
    a single bundle with a selected_variant_id and exactly one cross-fold OOF average — not the
    per-variant `variant*/` dirs the Python-expand path would create); and (2) the selected
    n_components and `result.best_rmse` MATCH what the Python-expand path selects — computed here
    directly with sklearn KFold (best per-n_components OOF CV), within 1e-3.
    """
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline

    from nirs4all.operators.transforms.scalers import StandardNormalVariate
    from nirs4all.pipeline.dagml.run_backend import _generation_kind, run_via_dagml

    components = [3, 9, 15]  # _range_[3, 16, 6] is end-inclusive: 3, 9, 15 (matches dag-ml range)
    pipeline = [StandardNormalVariate(), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(), "n_components": {"_range_": [3, 16, 6]}}]

    # (1) It is the NATIVE param-level path, run once into the `native/` workdir.
    assert _generation_kind(pipeline) == "param_model"
    result = run_via_dagml(pipeline, dataset_path("regression"), workdir=tmp_path)
    assert (tmp_path / "native" / "bundle.json").exists()
    assert not list(tmp_path.glob("variant*")), "native generation must NOT run the per-variant Python-expand path"
    bundle = json.loads((tmp_path / "native" / "bundle.json").read_text())
    assert bundle.get("selected_variant_id"), "dag-ml must record the natively-selected variant"
    avg_reports = [r for r in bundle["scores"]["reports"] if r["partition"] == "validation" and r.get("fold_id") == "avg"]
    assert len(avg_reports) == 1, "the bundle scores must be the selected variant's (one OOF average)"

    # (2) PARITY — compute the per-n_components OOF CV directly with sklearn KFold and pick the best,
    # exactly as the Python-expand path would. The native run must select that same n_components and
    # report its final-test RMSE.
    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    train = dataset.index_column("sample", {"partition": "train"})
    test_ints = dataset.index_column("sample", {"partition": "test"})
    folds = [([train[i] for i in tr], [train[i] for i in va]) for tr, va in KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42).split(train)]

    def oof_cv(n_components: int) -> float:
        acc: dict[int, float] = {}
        cnt: dict[int, int] = {}
        tru: dict[int, float] = {}
        for train_ints, val_ints in folds:
            model = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=n_components))
            model.fit(np.asarray(dataset.x({"sample": train_ints}, layout="2d"), dtype=float), np.asarray(dataset.y({"sample": train_ints}), dtype=float))
            for sample_int in val_ints:
                acc[sample_int] = acc.get(sample_int, 0.0) + float(np.asarray(model.predict(np.asarray(dataset.x({"sample": [sample_int]}, layout="2d"), dtype=float)))[0][0])
                cnt[sample_int] = cnt.get(sample_int, 0) + 1
                tru[sample_int] = float(np.asarray(dataset.y({"sample": [sample_int]}), dtype=float).ravel()[0])
        keys = sorted(acc)
        return float(np.sqrt(mean_squared_error([tru[k] for k in keys], [acc[k] / cnt[k] for k in keys])))

    scored = {nc: oof_cv(nc) for nc in components}
    best_nc = min(scored, key=lambda nc: scored[nc])  # the n_components Python-expand would select
    assert abs(result.cv_best_score - scored[best_nc]) < 1e-3  # dag-ml selected the same variant by CV

    # The selected variant's final-test RMSE (refit on full train, predict held-out test).
    final = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=best_nc))
    final.fit(np.asarray(dataset.x({"sample": train}, layout="2d"), dtype=float), np.asarray(dataset.y({"sample": train}), dtype=float))
    best_test = float(np.sqrt(mean_squared_error(np.asarray(dataset.y({"sample": test_ints}), dtype=float).ravel(), np.asarray(final.predict(np.asarray(dataset.x({"sample": test_ints}, layout="2d"), dtype=float))).ravel())))
    assert abs(result.best_rmse - best_test) < 1e-3  # and reports that variant's final-test RMSE


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_native_log_range_sweep(tmp_path) -> None:
    """A Ridge `alpha` `_log_range_` sweep runs as ONE NATIVE dag-ml generation + SELECT run.

    Mirrors `test_public_run_engine_dagml_native_param_sweep` for the `_log_range_` list form, whose
    native dag-ml log_range generator now round-trips through `build_execution_plan` (the float-label
    fingerprint drift is fixed by `canonical_generator_number`, dag-ml `2a77a7f`). The bridge lowers
    the sweep to a native dag-ml `generators` entry (no Python `expand_spec`), so the compiler expands
    the variants and dag-ml runs generation + per-variant CV scoring + SELECT + refit in a single CLI
    invocation. We assert both: (1) it is one native run (param_model path, a single bundle with a
    selected_variant_id and exactly one cross-fold OOF average — not the per-variant `variant*/` dirs
    the Python-expand path would create); and (2) the selected `alpha` and `result.best_rmse` MATCH
    what the Python-expand path selects — computed here directly with sklearn KFold (best per-alpha
    OOF CV), within 1e-3.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline

    from nirs4all.operators.transforms.scalers import StandardNormalVariate
    from nirs4all.pipeline.dagml.run_backend import _generation_kind, run_via_dagml

    # _log_range_[1e-3, 1e1, 5] is base-10 geometric end-inclusive: 1e-3, 1e-2, 1e-1, 1e0, 1e1.
    alphas = [1e-3, 1e-2, 1e-1, 1e0, 1e1]
    pipeline = [StandardNormalVariate(), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": Ridge(), "alpha": {"_log_range_": [1e-3, 1e1, 5]}}]

    # (1) It is the NATIVE param-level path, run once into the `native/` workdir.
    assert _generation_kind(pipeline) == "param_model"
    result = run_via_dagml(pipeline, dataset_path("regression"), workdir=tmp_path)
    assert (tmp_path / "native" / "bundle.json").exists()
    assert not list(tmp_path.glob("variant*")), "native generation must NOT run the per-variant Python-expand path"
    bundle = json.loads((tmp_path / "native" / "bundle.json").read_text())
    assert bundle.get("selected_variant_id"), "dag-ml must record the natively-selected variant"
    avg_reports = [r for r in bundle["scores"]["reports"] if r["partition"] == "validation" and r.get("fold_id") == "avg"]
    assert len(avg_reports) == 1, "the bundle scores must be the selected variant's (one OOF average)"

    # (2) PARITY — compute the per-alpha OOF CV directly with sklearn KFold and pick the best, exactly
    # as the Python-expand path would. The native run must select that same alpha and report its
    # final-test RMSE.
    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    train = dataset.index_column("sample", {"partition": "train"})
    test_ints = dataset.index_column("sample", {"partition": "test"})
    folds = [([train[i] for i in tr], [train[i] for i in va]) for tr, va in KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42).split(train)]

    def oof_cv(alpha: float) -> float:
        acc: dict[int, float] = {}
        cnt: dict[int, int] = {}
        tru: dict[int, float] = {}
        for train_ints, val_ints in folds:
            model = make_pipeline(StandardNormalVariate(), Ridge(alpha=alpha))
            model.fit(np.asarray(dataset.x({"sample": train_ints}, layout="2d"), dtype=float), np.asarray(dataset.y({"sample": train_ints}), dtype=float))
            for sample_int in val_ints:
                acc[sample_int] = acc.get(sample_int, 0.0) + float(np.asarray(model.predict(np.asarray(dataset.x({"sample": [sample_int]}, layout="2d"), dtype=float))).ravel()[0])
                cnt[sample_int] = cnt.get(sample_int, 0) + 1
                tru[sample_int] = float(np.asarray(dataset.y({"sample": [sample_int]}), dtype=float).ravel()[0])
        keys = sorted(acc)
        return float(np.sqrt(mean_squared_error([tru[k] for k in keys], [acc[k] / cnt[k] for k in keys])))

    scored = {alpha: oof_cv(alpha) for alpha in alphas}
    best_alpha = min(scored, key=lambda alpha: scored[alpha])  # the alpha Python-expand would select
    assert abs(result.cv_best_score - scored[best_alpha]) < 1e-3  # dag-ml selected the same variant by CV

    # The selected variant's final-test RMSE (refit on full train, predict held-out test).
    final = make_pipeline(StandardNormalVariate(), Ridge(alpha=best_alpha))
    final.fit(np.asarray(dataset.x({"sample": train}, layout="2d"), dtype=float), np.asarray(dataset.y({"sample": train}), dtype=float))
    best_test = float(np.sqrt(mean_squared_error(np.asarray(dataset.y({"sample": test_ints}), dtype=float).ravel(), np.asarray(final.predict(np.asarray(dataset.x({"sample": test_ints}, layout="2d"), dtype=float))).ravel())))
    assert abs(result.best_rmse - best_test) < 1e-3  # and reports that variant's final-test RMSE


def test_generation_kind_routes_conservatively() -> None:
    """The native router is CONSERVATIVE: only a clean `_range_`/`_log_range_` model sweep goes native;
    every other generator shape (mixed, finetune_params, _grid_, dict/modifier sweeps, non-model sweep,
    multi-model) falls back to the Python `expand_spec` path, so no generator is silently dropped.
    No CLI needed."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import MinMaxScaler

    from nirs4all.operators.transforms.scalers import StandardNormalVariate
    from nirs4all.pipeline.dagml.run_backend import _generation_kind

    splitter = KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42)

    # The native cases: a pure `_range_` or `_log_range_` list-form sweep on a model step.
    assert _generation_kind([StandardNormalVariate(), splitter, {"model": PLSRegression(), "n_components": {"_range_": [3, 9, 3]}}]) == "param_model"
    assert _generation_kind([StandardNormalVariate(), splitter, {"model": Ridge(), "alpha": {"_log_range_": [0.001, 10.0, 4]}}]) == "param_model"  # _log_range_ now native (dag-ml 2a77a7f)
    # No generators at all → none.
    assert _generation_kind([StandardNormalVariate(), splitter, {"model": PLSRegression(n_components=5)}]) == "none"

    # Everything below must route to the Python path ("operator") — never native — or a generator
    # would be silently dropped / mis-expanded.
    mixed = [StandardNormalVariate(), {"y_processing": MinMaxScaler(), "feature_range": {"_range_": [0, 1, 1]}}, splitter, {"model": PLSRegression(), "n_components": {"_range_": [3, 9, 3]}}]
    assert _generation_kind(mixed) == "operator"  # a sweep on a non-model step alongside the model sweep
    assert _generation_kind([splitter, {"model": PLSRegression(), "n_components": {"_range_": [3, 9, 3]}, "finetune_params": {"n_trials": 5}}]) == "operator"  # finetune_params
    assert _generation_kind([splitter, {"model": PLSRegression(), "n_components": {"_range_": [3, 9, 3]}, "train_params": {"epochs": 1}}]) == "operator"  # train_params
    assert _generation_kind([splitter, {"model": PLSRegression(), "n_components": {"_grid_": {"n_components": [5, 10]}}}]) == "operator"  # _grid_ (not proven equivalent)
    assert _generation_kind([splitter, {"model": PLSRegression(), "n_components": {"_range_": [3, 16, 1], "count": 3}}]) == "operator"  # modifier-bearing range
    assert _generation_kind([splitter, {"model": PLSRegression(), "n_components": {"_range_": {"from": 3, "to": 9}}}]) == "operator"  # dict-form range
    assert _generation_kind([splitter, {"model": Ridge(), "alpha": {"_log_range_": [0.001, 10.0, 4], "count": 3}}]) == "operator"  # modifier-bearing _log_range_ (only the bare list form is native)
    assert _generation_kind([splitter, {"model": Ridge(), "alpha": {"_log_range_": {"from": 0.001, "to": 10.0, "num": 4}}}]) == "operator"  # dict-form _log_range_
    assert _generation_kind([splitter, {"model": {"_or_": [PLSRegression(), PLSRegression(n_components=3)]}}]) == "operator"  # multi-model
    assert _generation_kind([{"_or_": [StandardNormalVariate(), MinMaxScaler()]}, splitter, {"model": PLSRegression()}]) == "operator"  # operator-level _or_ step


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_classification() -> None:
    """engine="dag-ml" runs a CLASSIFIER: detects the task type, scores accuracy natively, and
    best_accuracy (final-test) == sklearn. Uses LogisticRegression (order-invariant) so the
    resolver-aligned baseline is comparable (RandomForest bootstraps by row order)."""
    from sklearn.linear_model import LogisticRegression

    import nirs4all
    from nirs4all.pipeline.dagml.identity import mint_identity
    from nirs4all.pipeline.dagml.resolver import MaterializationResolver

    pipeline = [KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": LogisticRegression(max_iter=500)}]
    result = nirs4all.run(pipeline, dataset_path("classification"), engine="dag-ml")
    assert result.best_accuracy == result.best_accuracy  # not NaN

    dataset = DatasetConfigs(dataset_path("classification")).get_dataset_at(0)
    identity = mint_identity(dataset)
    resolver = MaterializationResolver(dataset, identity)
    train = dataset.index_column("sample", {"partition": "train"})
    test_ints = dataset.index_column("sample", {"partition": "test"})

    def features(sample_ints: list[int]) -> np.ndarray:
        return np.asarray(resolver.resolve_features([identity.to_wire(i) for i in sample_ints])["values"])

    def targets(sample_ints: list[int]) -> np.ndarray:
        return np.asarray(resolver.resolve_targets([identity.to_wire(i) for i in sample_ints])["values"]).ravel()

    model = LogisticRegression(max_iter=500).fit(features(train), targets(train))
    sklearn_accuracy = float(np.mean(model.predict(features(test_ints)) == targets(test_ints)))
    assert abs(result.best_accuracy - sklearn_accuracy) < 1e-3


def _excluded_train_ints(dataset, train: list[int], threshold: float) -> set[int]:
    """Fit a YOutlierFilter on the full base train pool and return the excluded sample ints.

    Mirrors ExcludeController / run_backend._excluded_sample_ints exactly: fit on the whole train
    pool (``include_augmented=False``), ``get_mask`` (True=keep), exclude where mask is False.
    """
    from nirs4all.operators.filters.y_outlier import YOutlierFilter

    x_train = np.asarray(dataset.x({"partition": "train"}, layout="2d", concat_source=True, include_augmented=False))
    y_train = np.asarray(dataset.y({"partition": "train"}, include_augmented=False)).flatten()
    filt = YOutlierFilter(method="iqr", threshold=threshold)
    filt.fit(x_train, y_train)
    return {int(s) for s, keep in zip(train, filt.get_mask(x_train, y_train), strict=True) if not keep}


def _xy_for_sample_order(dataset, sample_ints: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Real X/y for ``sample_ints`` in request order, avoiding storage-order coupling."""
    sample_ints = [int(s) for s in sample_ints]
    x = np.asarray(dataset.x_rows(sample_ints, layout="2d"), dtype=float)
    y_block = np.asarray(dataset.y({"sample": sample_ints}, include_augmented=False), dtype=float)
    stored = dataset.index_column("sample", {"sample": sample_ints})
    row_of = {int(sample_int): row for row, sample_int in enumerate(stored)}
    y = y_block[[row_of[int(sample_int)] for sample_int in sample_ints]]
    return x, y.ravel()


def _flagged_by_filter(dataset, sample_ints: list[int], filter_obj) -> set[int]:  # noqa: ANN001
    """Sample ints where ``SampleFilter.get_mask`` is false (the tag/exclude polarity)."""
    x, y = _xy_for_sample_order(dataset, sample_ints)
    filter_obj.fit(x, y)
    mask = filter_obj.get_mask(x, y)
    return {int(sample_int) for sample_int, keep in zip(sample_ints, mask, strict=True) if not keep}


def _direct_exclude_oof_and_test(dataset, filter_obj, n_components: int = 5) -> tuple[float, float, set[int]]:  # noqa: ANN001
    """Direct sklearn baseline for default exclude: remove flagged samples from the CV universe."""
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline

    from nirs4all.operators.transforms.scalers import StandardNormalVariate

    train = [int(s) for s in dataset.index_column("sample", {"partition": "train"})]
    test = [int(s) for s in dataset.index_column("sample", {"partition": "test"})]
    excluded = _flagged_by_filter(dataset, train, filter_obj)
    kept = [sample_int for sample_int in train if sample_int not in excluded]
    folds = [([kept[i] for i in tr], [kept[i] for i in va]) for tr, va in KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42).split(kept)]

    oof_pred: dict[int, float] = {}
    oof_true: dict[int, float] = {}
    for train_ints, val_ints in folds:
        model = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=n_components))
        x_train, y_train = _xy_for_sample_order(dataset, train_ints)
        model.fit(x_train, y_train)
        pred = np.asarray(model.predict(_xy_for_sample_order(dataset, val_ints)[0])).ravel()
        true = _xy_for_sample_order(dataset, val_ints)[1]
        for sample_int, value, target in zip(val_ints, pred, true, strict=True):
            oof_pred[int(sample_int)] = float(value)
            oof_true[int(sample_int)] = float(target)
    keys = sorted(oof_pred)
    cv_oof = float(np.sqrt(mean_squared_error([oof_true[k] for k in keys], [oof_pred[k] for k in keys])))

    final = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=n_components))
    final.fit(*_xy_for_sample_order(dataset, kept))
    x_test, y_test = _xy_for_sample_order(dataset, test)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, np.asarray(final.predict(x_test)).ravel())))
    return cv_oof, test_rmse, excluded


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_exclude_default_legacy_parity() -> None:
    """exclude (DEFAULT mode, keep_in_oof=False) on engine="dag-ml" == legacy: excluded removed from CV.

    The default exclude mode removes excluded samples from the CV universe ENTIRELY (verified legacy
    semantic: ExcludeController runs before the splitter, which splits over ``include_excluded=False``).
    Asserts engine="dag-ml"'s ``cv_best_score`` matches the legacy/default engine's on the same pipeline,
    and ``best_rmse`` matches the CLEAN refit-on-kept test score.

    NOTE: legacy's ``best_rmse`` instead carries a get_best quirk (it returns the lowest-val per-fold
    model's test_score, not the refit's), so it is NOT asserted equal here; the dag-ml engine
    intentionally reports the clean refit-on-kept final-test (a known RunResult divergence that is not
    exclude-specific — every dag-ml parity test compares to the clean refit, see the tests above).
    """
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline

    import nirs4all
    from nirs4all.operators.filters.y_outlier import YOutlierFilter
    from nirs4all.operators.transforms.scalers import StandardNormalVariate

    n_comp, threshold = 5, 1.0
    pipeline = lambda: [  # noqa: E731 - fresh filter instance per engine (fit mutates state)
        {"exclude": YOutlierFilter(method="iqr", threshold=threshold)},
        StandardNormalVariate(),
        KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42),
        {"model": PLSRegression(n_components=n_comp)},
    ]
    legacy = nirs4all.run(pipeline(), dataset_path("regression"))
    dagml = nirs4all.run(pipeline(), dataset_path("regression"), engine="dag-ml")

    # cv_best_score: dag-ml default == legacy (both KFold over the kept universe; excluded absent).
    assert abs(dagml.cv_best_score - legacy.cv_best_score) < 1e-3, (dagml.cv_best_score, legacy.cv_best_score)

    # best_rmse: dag-ml == clean refit-on-kept test (the legacy get_best best_rmse quirk is NOT matched).
    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    train = dataset.index_column("sample", {"partition": "train"})
    test_ints = dataset.index_column("sample", {"partition": "test"})
    kept = [s for s in train if s not in _excluded_train_ints(dataset, train, threshold)]
    model = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=n_comp))
    model.fit(np.asarray(dataset.x({"sample": kept}, layout="2d"), dtype=float), np.asarray(dataset.y({"sample": kept}), dtype=float))
    test_pred = np.asarray(model.predict(np.asarray(dataset.x({"sample": test_ints}, layout="2d"), dtype=float))).ravel()
    refit_test_rmse = float(np.sqrt(mean_squared_error(np.asarray(dataset.y({"sample": test_ints}), dtype=float).ravel(), test_pred)))
    assert abs(dagml.best_rmse - refit_test_rmse) < 1e-3, (dagml.best_rmse, refit_test_rmse)


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_tag_round_trip(tmp_path) -> None:
    """`tag` runs non-destructively: relation tags are emitted and the CV pool stays full."""
    from nirs4all.operators.filters.y_outlier import YOutlierFilter
    from nirs4all.operators.transforms.scalers import StandardNormalVariate
    from nirs4all.pipeline.dagml.run_backend import run_via_dagml

    n_comp = 5
    tag_name = "dagml_y_iqr_outlier"
    pipeline = [
        {"tag": YOutlierFilter(method="iqr", threshold=1.0, tag_name=tag_name)},
        StandardNormalVariate(),
        KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42),
        {"model": PLSRegression(n_components=n_comp)},
    ]
    result = run_via_dagml(pipeline, dataset_path("regression"), workdir=tmp_path)

    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    identity = mint_identity(dataset)
    train = [int(s) for s in dataset.index_column("sample", {"partition": "train"})]
    expected_tagged = _flagged_by_filter(dataset, train, YOutlierFilter(method="iqr", threshold=1.0, tag_name=tag_name))
    assert expected_tagged, "tag filter must flag samples for this round-trip test"

    envelope = json.loads((tmp_path / "variant0" / "envelope.json").read_text())
    records = envelope["coordinator_relations"]["records"]
    assert len(records) == len(train), "tag must not remove samples from the CV relation universe"
    by_int = {identity.to_int(record["observation_id"]): record for record in records}
    tagged = {sample_int for sample_int, record in by_int.items() if tag_name in record.get("tags", [])}
    assert tagged == expected_tagged
    assert all(record.get("tags") for record in records if "tags" in record)

    avg = [report for report in json.loads((tmp_path / "variant0" / "bundle.json").read_text())["scores"]["reports"] if report["partition"] == "validation" and report["fold_id"] == "avg"]
    assert len(avg) == 1 and avg[0]["row_count"] == len(train), "tagged samples must stay in the OOF"
    cv_oof, test_rmse = _host_split_oof_and_test(dataset, lambda: KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), n_components=n_comp)
    assert abs(result.cv_best_score - cv_oof) < 1e-3
    assert abs(result.best_rmse - test_rmse) < 1e-3


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_exclude_x_outlier_direct_sklearn_parity() -> None:
    """Default exclude with XOutlierFilter matches direct sklearn OOF/final-test on the kept pool."""
    import nirs4all
    from nirs4all.operators.filters import XOutlierFilter
    from nirs4all.operators.transforms.scalers import StandardNormalVariate

    n_comp = 5
    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    cv_oof, test_rmse, excluded = _direct_exclude_oof_and_test(dataset, XOutlierFilter(), n_components=n_comp)
    assert excluded, "XOutlierFilter must exclude at least one sample for this parity lock"

    pipeline = [
        {"exclude": XOutlierFilter()},
        StandardNormalVariate(),
        KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42),
        {"model": PLSRegression(n_components=n_comp)},
    ]
    result = nirs4all.run(pipeline, dataset_path("regression"), engine="dag-ml")
    assert abs(result.cv_best_score - cv_oof) < 1e-3, (result.cv_best_score, cv_oof)
    assert abs(result.best_rmse - test_rmse) < 1e-3, (result.best_rmse, test_rmse)


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_exclude_keep_in_oof(tmp_path) -> None:
    """exclude (OPT-IN mode, keep_in_oof=True) keeps excluded in the OOF: leakage-pure CV.

    Opt-in mode keeps excluded samples in each fold's VALIDATION (predicted in OOF by a model that
    never trained on them) while dropping them from each fold's TRAIN. Asserts (a) the excluded
    samples DO appear in the validation/OOF predictions and the OOF covers the FULL train universe;
    (b) ``cv_best_score`` differs from the default mode (which removes them from CV); (c) it equals
    the leakage-pure baseline (excluded dropped from fold-train, kept in fold-val).
    """
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline

    import nirs4all
    from nirs4all.operators.filters.y_outlier import YOutlierFilter
    from nirs4all.operators.transforms.scalers import StandardNormalVariate
    from nirs4all.pipeline.dagml.cli_runner import assemble_cv_refit_dsl, run_cv_refit_bundle
    from nirs4all.pipeline.dagml.envelope import build_envelope
    from nirs4all.pipeline.dagml.identity import mint_identity
    from nirs4all.pipeline.dagml_bridge import controller_manifests

    n_comp, threshold = 5, 1.0

    def pipeline(keep: bool) -> list:
        step: dict = {"exclude": YOutlierFilter(method="iqr", threshold=threshold)}
        if keep:
            step["keep_in_oof"] = True
        return [step, StandardNormalVariate(), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=n_comp)}]

    default = nirs4all.run(pipeline(False), dataset_path("regression"), engine="dag-ml")
    optin = nirs4all.run(pipeline(True), dataset_path("regression"), engine="dag-ml")

    # (b) opt-in CV differs from default (excluded kept in OOF vs removed from the CV universe).
    assert abs(optin.cv_best_score - default.cv_best_score) > 1e-3, (optin.cv_best_score, default.cv_best_score)

    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    identity = mint_identity(dataset)
    train = dataset.index_column("sample", {"partition": "train"})
    excluded = _excluded_train_ints(dataset, train, threshold)
    assert excluded, "the filter must exclude at least one sample for this test to be meaningful"

    # (a) the excluded samples ARE predicted in the OOF and the OOF covers the full train universe.
    raw_folds = [([train[i] for i in tr], [train[i] for i in va]) for tr, va in KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42).split(train)]
    pure_folds = [([s for s in tr if s not in excluded], va) for tr, va in raw_folds]
    import dag_ml

    envelope = build_envelope(dataset, identity, sample_ints=list(train), excluded_sample_ints=excluded)
    steps = [StandardNormalVariate(), {"model": PLSRegression(n_components=n_comp)}]
    dsl = assemble_cv_refit_dsl(steps, identity, envelope, pure_folds, dsl_id="exclude_optin", n_splits=_N_SPLITS)
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()
    outcome = run_cv_refit_bundle(dsl=dsl, envelope=envelope, graph=graph, dataset_path=dataset_path("regression"), workdir=tmp_path, dagml_cli=str(_DAGML_CLI), venv_python=sys.executable)
    assert outcome["returncode"] == 0, outcome["stdout"][-2000:]
    oof_ids = {identity.to_int(sid) for frame in outcome["results"] for block in (frame.get("result") or frame).get("predictions", []) if block["partition"] == "validation" for sid in block["sample_ids"]}
    assert excluded <= oof_ids, "excluded samples must be predicted in the OOF (keep_in_oof=True)"
    assert oof_ids == {int(s) for s in train}, "opt-in OOF must cover the full train universe"

    # (c) cv_best_score == leakage-pure baseline (excluded dropped from fold-train, kept in fold-val).
    oof_pred: dict[int, float] = {}
    oof_true: dict[int, float] = {}
    for train_ints, val_ints in pure_folds:
        model = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=n_comp))
        model.fit(np.asarray(dataset.x({"sample": train_ints}, layout="2d"), dtype=float), np.asarray(dataset.y({"sample": train_ints}), dtype=float))
        pred = np.asarray(model.predict(np.asarray(dataset.x({"sample": val_ints}, layout="2d"), dtype=float))).ravel()
        true = np.asarray(dataset.y({"sample": val_ints}), dtype=float).ravel()
        for position, sample_int in enumerate(val_ints):
            oof_pred[sample_int], oof_true[sample_int] = float(pred[position]), float(true[position])
    keys = sorted(oof_pred)
    leakage_pure_oof = float(np.sqrt(mean_squared_error([oof_true[k] for k in keys], [oof_pred[k] for k in keys])))
    assert abs(optin.cv_best_score - leakage_pure_oof) < 1e-3, (optin.cv_best_score, leakage_pure_oof)


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_exclude_sequential() -> None:
    """Multiple `exclude` steps are applied SEQUENTIALLY (legacy parity).

    Two `{"exclude": ...}` steps with different filters: each fits on the CURRENT kept train (the pool
    after earlier exclusions), exactly like legacy ExcludeController (which reads include_excluded=False).
    Asserts engine="dag-ml" cv_best_score matches the legacy engine running the same two-exclude
    pipeline — proving both excludes are consumed (a surviving second exclude would hit the bridge's
    raw-exclude NotImplementedError) and the progressive cleaning matches.
    """
    import nirs4all
    from nirs4all.operators.filters.y_outlier import YOutlierFilter
    from nirs4all.operators.transforms.scalers import StandardNormalVariate

    def pipeline() -> list:
        return [
            {"exclude": YOutlierFilter(method="iqr", threshold=1.5)},
            {"exclude": YOutlierFilter(method="zscore", threshold=2.0)},
            StandardNormalVariate(),
            KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

    legacy = nirs4all.run(pipeline(), dataset_path("regression"))
    dagml = nirs4all.run(pipeline(), dataset_path("regression"), engine="dag-ml")
    assert abs(dagml.cv_best_score - legacy.cv_best_score) < 1e-3, (dagml.cv_best_score, legacy.cv_best_score)


def test_resolve_exclude_all_excluded_guard_keeps_one() -> None:
    """A combined keep-mask that would exclude EVERY train row keeps one sample (legacy guard).

    Mirrors ExcludeController's all-excluded guard (exclude.py:213-222): exclusion must never empty the
    pool. With a filter that excludes everything, _resolve_exclude keeps exactly one sample in the CV
    pool (default mode) so the fold-train pool is non-empty. No CLI needed (pure host-side logic).
    """
    from nirs4all.operators.filters.y_outlier import YOutlierFilter
    from nirs4all.pipeline.dagml.run_backend import _excluded_from_pool, _resolve_exclude

    class _ExcludeAll(YOutlierFilter):
        def get_mask(self, X, y=None):  # noqa: ANN001, ANN202 - test stub
            return np.zeros(len(X), dtype=bool)  # exclude every sample

    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    train = [int(s) for s in dataset.index_column("sample", {"partition": "train"})]

    # Direct: the combined mask excludes all but one.
    excluded = _excluded_from_pool({"exclude": _ExcludeAll(method="iqr")}, dataset, train)
    assert len(train) - len(excluded) == 1, "guard must keep exactly one sample"

    # Through _resolve_exclude (default mode): the CV pool is non-empty (exactly one sample).
    _remaining, pool, envelope_excluded = _resolve_exclude([{"exclude": _ExcludeAll(method="iqr")}], dataset)
    assert len(pool) == 1, pool
    assert envelope_excluded == set()  # default mode marks nothing excluded in the envelope


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_separation_branch_by_metadata() -> None:
    """A by_metadata separation branch + concat merge on engine="dag-ml" scores natively.

    The native path fans the branch into one model node per distinct `group` value (discovered from
    the envelope relation metadata), runs per-partition FIT_CV, and the native concat-merge handler
    reassembles a full-universe OOF. Asserts `cv_best_score` matches a DIRECT sklearn-per-partition
    OOF baseline (one model per group, each CV'd on its partition's folds, OOF reassembled to the full
    universe, scored) within 1e-3.

    PARITY BASELINE = direct sklearn, NOT legacy. Legacy nirs4all by_metadata branch + concat-merge is
    BROKEN here: the disjoint concat reassembly raises `MERGE-E003` ("Branch 0 source 0 has N samples,
    expected M") for both the model-in-branch and SNV-only shapes (the branch's whole-dataset feature
    snapshot is scattered into a partition-sized slot). So the dag-ml native path is a CORRECTION, and
    the parity target is the direct computation — like exclude's legacy get_best best_rmse quirk.

    `best_rmse` is now also native: each per-partition refit model predicts its partition's held-out
    TEST samples (`fold_id=None`), and dag-ml's off-fold concat handler reassembles those into one
    full-universe `(test, fold_id=None)` block. Asserts `best_rmse` == a DIRECT sklearn-per-partition
    TEST baseline (refit one PLS per group on its partition's full train, predict its partition's test
    samples, concat, score) within 1e-3.
    """
    import nirs4all

    n_comp, n_splits = 2, 3
    pipeline = [
        KFold(n_splits=n_splits, shuffle=True, random_state=42),
        {"branch": {"by_metadata": "group", "steps": [{"model": PLSRegression(n_components=n_comp)}]}},
        {"merge": "concat"},
    ]
    result = nirs4all.run(pipeline, dataset_path("with_metadata"), engine="dag-ml")

    dataset = DatasetConfigs(dataset_path("with_metadata")).get_dataset_at(0)
    train = dataset.index_column("sample", {"partition": "train"})
    test = dataset.index_column("sample", {"partition": "test"})
    group_of = {int(s): str(v) for s, v in zip(train, dataset.metadata_column("group", {"partition": "train"}), strict=True)}
    group_of_test = {int(s): str(v) for s, v in zip(test, dataset.metadata_column("group", {"partition": "test"}), strict=True)}
    groups = sorted(set(group_of.values()))
    folds = [([train[i] for i in tr], [train[i] for i in va]) for tr, va in KFold(n_splits=n_splits, shuffle=True, random_state=42).split(train)]

    # DIRECT sklearn-per-partition OOF: one PLS per group, each fit on its partition's fold-train and
    # validated on its partition's fold-validation; OOF reassembled to the full universe and scored.
    oof_pred: dict[int, float] = {}
    oof_true: dict[int, float] = {}
    for train_ints, val_ints in folds:
        for value in groups:
            part_train = [s for s in train_ints if group_of[int(s)] == value]
            part_val = [s for s in val_ints if group_of[int(s)] == value]
            if not part_train or not part_val:  # a small group can be absent from a fold's validation
                continue
            model = PLSRegression(n_components=n_comp)
            model.fit(np.asarray(dataset.x({"sample": part_train}, layout="2d"), dtype=float), np.asarray(dataset.y({"sample": part_train}), dtype=float))
            pred = np.asarray(model.predict(np.asarray(dataset.x({"sample": part_val}, layout="2d"), dtype=float))).ravel()
            true = np.asarray(dataset.y({"sample": part_val}), dtype=float).ravel()
            for position, sample_int in enumerate(part_val):
                oof_pred[int(sample_int)] = float(pred[position])
                oof_true[int(sample_int)] = float(true[position])
    from sklearn.metrics import mean_squared_error

    keys = sorted(oof_pred)
    baseline = float(np.sqrt(mean_squared_error([oof_true[k] for k in keys], [oof_pred[k] for k in keys])))
    assert abs(result.cv_best_score - baseline) < 1e-3, (result.cv_best_score, baseline)

    # DIRECT sklearn-per-partition TEST: one PLS per group refit on its partition's FULL train, predict
    # its partition's test samples; concat to the full test universe, score. This is what dag-ml's
    # off-fold concat handler reassembles from each partition model's `(test, fold_id=None)` REFIT block.
    test_pred: dict[int, float] = {}
    test_true: dict[int, float] = {}
    for value in groups:
        part_train = [s for s in train if group_of[int(s)] == value]
        part_test = [s for s in test if int(s) in group_of_test and group_of_test[int(s)] == value]
        if not part_train or not part_test:
            continue
        model = PLSRegression(n_components=n_comp)
        model.fit(np.asarray(dataset.x({"sample": part_train}, layout="2d"), dtype=float), np.asarray(dataset.y({"sample": part_train}), dtype=float))
        pred = np.asarray(model.predict(np.asarray(dataset.x({"sample": part_test}, layout="2d"), dtype=float))).ravel()
        true = np.asarray(dataset.y({"sample": part_test}), dtype=float).ravel()
        for position, sample_int in enumerate(part_test):
            test_pred[int(sample_int)] = float(pred[position])
            test_true[int(sample_int)] = float(true[position])
    test_keys = sorted(test_pred)
    test_baseline = float(np.sqrt(mean_squared_error([test_true[k] for k in test_keys], [test_pred[k] for k in test_keys])))
    assert abs(result.best_rmse - test_baseline) < 1e-3, (result.best_rmse, test_baseline)


def test_separation_branch_detection() -> None:
    """The branch detector consumes only the handled separation+concat shape; others fall through.

    `_detect_separation_branch` returns the (branch_step, body) tuple for a by_metadata/by_tag
    separation branch with a model in its `steps` and a following `{"merge": "concat"}`; everything
    else (a duplication branch, a different merge mode, a model after the merge) returns None so the
    bridge's raw-branch NotImplementedError still guards it. No CLI needed (pure host-side logic)."""
    from nirs4all.operators.transforms.scalers import StandardNormalVariate
    from nirs4all.pipeline.dagml.run_backend import _detect_separation_branch

    splitter = KFold(n_splits=3, shuffle=True, random_state=42)

    # Handled: by_metadata separation branch (model in branch) + concat merge.
    handled = [splitter, {"branch": {"by_metadata": "group", "steps": [{"model": PLSRegression(n_components=2)}]}}, {"merge": "concat"}]
    detected = _detect_separation_branch(handled)
    assert detected is not None
    branch_step, body = detected
    assert "by_metadata" in branch_step["branch"]
    assert any(isinstance(step, dict) and "model" in step for step in body)

    branch = {"branch": {"by_metadata": "group", "steps": [{"model": PLSRegression(n_components=2)}]}}
    # Not handled (each must fall through to the bridge's loud raw-branch error):
    # a duplication branch (a list, not a by_metadata/by_tag dict).
    assert _detect_separation_branch([splitter, {"branch": [[PLSRegression(n_components=2)]]}, {"merge": "predictions"}]) is None
    # a separation branch but a non-concat merge.
    assert _detect_separation_branch([splitter, {"branch": {"by_metadata": "group", "steps": [{"model": PLSRegression()}]}}, {"merge": "predictions"}]) is None
    # a model placed AFTER the concat merge (a different shape).
    assert _detect_separation_branch([splitter, {"branch": {"by_metadata": "group", "steps": [StandardNormalVariate()]}}, {"merge": "concat"}, {"model": PLSRegression()}]) is None
    # no merge at all.
    assert _detect_separation_branch([splitter, {"branch": {"by_metadata": "group", "steps": [{"model": PLSRegression()}]}}]) is None
    # a top-level transform beside the branch (only the branch body is lowered → would be dropped).
    assert _detect_separation_branch([StandardNormalVariate(), splitter, branch, {"merge": "concat"}]) is None
    # a top-level y_processing / tag step beside the branch.
    assert _detect_separation_branch([splitter, {"y_processing": StandardNormalVariate()}, branch, {"merge": "concat"}]) is None
    # an exclude step beside the branch (the exclusion would be silently lost — out of scope).
    assert _detect_separation_branch([{"exclude": StandardNormalVariate()}, splitter, branch, {"merge": "concat"}]) is None
    # unhandled branch options: explicit `values` grouping / `min_samples` cardinality drop.
    assert _detect_separation_branch([splitter, {"branch": {"by_metadata": "group", "values": {"a": ["group_0"]}, "steps": [{"model": PLSRegression()}]}}, {"merge": "concat"}]) is None
    assert _detect_separation_branch([splitter, {"branch": {"by_metadata": "group", "min_samples": 5, "steps": [{"model": PLSRegression()}]}}, {"merge": "concat"}]) is None


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_separation_branch_unsupported_shapes_fail_loud() -> None:
    """Out-of-scope branch shapes raise NotImplementedError end-to-end — never silently mishandled.

    The detector admits ONLY the exact handled shape; anything `_run_separation_branch` does not honor
    (a top-level preprocessing step that would be dropped, an `exclude` whose exclusion would be lost,
    a `values`/`min_samples` branch whose grouping is not applied) must fall through to the bridge's
    raw-branch NotImplementedError (the coverage-boundary fail-loud guarantee). These are known
    limitations for follow-up slices (top-level preproc+branch, exclude+branch, values/min_samples)."""
    import nirs4all
    from nirs4all.operators.filters.y_outlier import YOutlierFilter
    from nirs4all.operators.transforms.scalers import StandardNormalVariate

    def branch() -> dict:
        return {"branch": {"by_metadata": "group", "steps": [{"model": PLSRegression(n_components=2)}]}}

    def split() -> KFold:
        return KFold(n_splits=3, shuffle=True, random_state=42)

    rejected = {
        "top_level_transform": [StandardNormalVariate(), split(), branch(), {"merge": "concat"}],
        "exclude_plus_branch": [{"exclude": YOutlierFilter(method="iqr", threshold=1.0)}, split(), branch(), {"merge": "concat"}],
        "values_branch": [split(), {"branch": {"by_metadata": "group", "values": {"a": ["group_0"]}, "steps": [{"model": PLSRegression(n_components=2)}]}}, {"merge": "concat"}],
        "min_samples_branch": [split(), {"branch": {"by_metadata": "group", "min_samples": 5, "steps": [{"model": PLSRegression(n_components=2)}]}}, {"merge": "concat"}],
    }
    for label, pipeline in rejected.items():
        with pytest.raises(NotImplementedError):
            nirs4all.run(pipeline, dataset_path("with_metadata"), engine="dag-ml")
        assert label  # name surfaced in the failure if the raise is missing


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_run_via_dagml_sample_augmentation(tmp_path) -> None:
    """`sample_augmentation` runs e2e on dag-ml: synthetic train rows TRAIN, never reach OOF/test.

    The native path runs nirs4all's real augmentation machinery (synthetic train rows), builds
    base-grain folds (a clean OOF partition over base val) + a CV-universe envelope, and the host
    expands each fold's base-train ids to base + their augmented children at fit time. We assert:

    * `cv_best_score` == a DIRECT sklearn-on-augmented-train OOF baseline (per fold, fit SNV->PLS on
      base-train + their augmented children, validate on base-val; OOF reassembled, scored) — within
      1e-3, reconstructed from THIS run's pickled augmented dataset (augmentation is stochastic, so the
      baseline must use the same synthetic rows the run used);
    * it DIFFERS from the no-augmentation OOF baseline (same folds, base-train only) — proving the
      augmented samples actually train. Legacy is a CONFIRMED silent NO-OP (#14): the legacy model
      controller fetches train with include_augmented defaulting False, so augmented rows never reach
      fit and cv_best_score is identical to no-augmentation regardless of magnitude — so the dag-ml
      engine is a CORRECTION and the parity baseline is the direct sklearn-on-aug, NOT legacy;
    * NO augmented child appears in the validation/OOF predictions (the origin-boundary leakage guard).
    """
    import pickle

    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline

    from nirs4all.operators.augmentation import GaussianAdditiveNoise
    from nirs4all.operators.transforms.scalers import StandardNormalVariate
    from nirs4all.pipeline.dagml.run_backend import run_via_dagml

    n_comp = 5
    pipeline = [
        StandardNormalVariate(),
        {"sample_augmentation": {"transformers": [GaussianAdditiveNoise(sigma=0.01)], "count": 1, "selection": "all", "random_state": 42}},
        KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42),
        {"model": PLSRegression(n_components=n_comp)},
    ]
    result = run_via_dagml(pipeline, dataset_path("regression"), workdir=tmp_path)

    # Reconstruct the EXACT augmented dataset the run used (pickled in the run dir) — the synthetic rows
    # are stochastic, so the baseline MUST use the same rows that trained the native run.
    aug_ds = pickle.loads((tmp_path / "augment" / "augmented_dataset.pkl").read_bytes())  # noqa: S301 - test-written
    base_train = [int(s) for s in DatasetConfigs(dataset_path("regression")).get_dataset_at(0).index_column("sample", {"partition": "train"})]
    all_samples = [int(s) for s in aug_ds.index_column("sample", {})]
    all_origins = [int(o) for o in aug_ds.index_column("origin", {})]
    children: dict[int, list[int]] = {}
    augmented_ints: list[int] = []
    for sample_int, origin_int in zip(all_samples, all_origins, strict=True):
        if sample_int != origin_int:
            children.setdefault(origin_int, []).append(sample_int)
            augmented_ints.append(sample_int)
    assert augmented_ints, "augmentation must create synthetic rows for this test to be meaningful"
    folds = [([base_train[i] for i in tr], [base_train[i] for i in va]) for tr, va in KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42).split(base_train)]

    def y_of(sample_int: int) -> float:
        origin_int = all_origins[all_samples.index(sample_int)]
        return float(np.asarray(aug_ds.y({"sample": [origin_int]})).ravel()[0])

    # DIRECT sklearn-on-augmented-train OOF: per fold fit on base-train + their children, val on base-val.
    oof_pred: dict[int, float] = {}
    oof_true: dict[int, float] = {}
    for train_ints, val_ints in folds:
        fit_ints = list(train_ints) + [child for origin in train_ints for child in children.get(origin, [])]
        model = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=n_comp))
        model.fit(np.asarray(aug_ds.x_rows(fit_ints, layout="2d"), dtype=float), np.asarray([y_of(s) for s in fit_ints], dtype=float))
        pred = np.asarray(model.predict(np.asarray(aug_ds.x_rows(list(val_ints), layout="2d"), dtype=float))).ravel()
        for position, sample_int in enumerate(val_ints):
            oof_pred[sample_int], oof_true[sample_int] = float(pred[position]), y_of(sample_int)
    keys = sorted(oof_pred)
    direct_aug = float(np.sqrt(mean_squared_error([oof_true[k] for k in keys], [oof_pred[k] for k in keys])))
    assert abs(result.cv_best_score - direct_aug) < 1e-3, (result.cv_best_score, direct_aug)

    # NO-augmentation OOF baseline (same folds, base-train only) — the augmented run must DIFFER from it
    # (legacy's silent no-op would make them equal; #14).
    no_pred: dict[int, float] = {}
    for train_ints, val_ints in folds:
        model = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=n_comp))
        model.fit(np.asarray(aug_ds.x_rows(list(train_ints), layout="2d"), dtype=float), np.asarray([y_of(s) for s in train_ints], dtype=float))
        pred = np.asarray(model.predict(np.asarray(aug_ds.x_rows(list(val_ints), layout="2d"), dtype=float))).ravel()
        for position, sample_int in enumerate(val_ints):
            no_pred[sample_int] = float(pred[position])
    no_aug = float(np.sqrt(mean_squared_error([oof_true[k] for k in keys], [no_pred[k] for k in keys])))
    assert abs(result.cv_best_score - no_aug) > 1e-3, "augmented samples must actually train (vs legacy no-op #14)"

    # The OOF covers base val only — NO augmented child is ever validated (the leakage guard).
    bundle = json.loads((tmp_path / "augment" / "bundle.json").read_text())
    assert bundle.get("scores") is not None
    avg = [r for r in bundle["scores"]["reports"] if r["partition"] == "validation" and r.get("fold_id") == "avg"]
    assert len(avg) == 1 and avg[0]["row_count"] == len(base_train), "OOF must cover exactly the base train universe"


def test_stateful_augmentation_routes_fold_local() -> None:
    """STATEFUL/SUPERVISED/BALANCED augmentation routes to the FOLD-LOCAL path, not a global fit.

    A stateless per-sample augmenter is leakage-free globally (`_augmentation_is_leakage_free` True →
    global path #8). A stateful augmenter (mixup with stored neighbors, a global-mean scatter reference)
    or the balanced/supervised mode is NOT leakage-free globally (False → fold-local path #32: fit inside
    each fold's train only). This pins the routing signal; the e2e parity test below exercises the path.
    No CLI needed — these are pure predicate checks.
    """
    from nirs4all.operators.augmentation import GaussianAdditiveNoise
    from nirs4all.operators.augmentation.spectral import LocalMixupAugmenter, ScatterSimulationMSC
    from nirs4all.pipeline.dagml.run_backend import _augmentation_is_leakage_free, _operator_is_stateless

    # Operator-level signal: stateless augmenters pass, stateful ones (learn data state in fit) are flagged.
    assert _operator_is_stateless(GaussianAdditiveNoise(sigma=0.01))
    assert not _operator_is_stateless(LocalMixupAugmenter())  # stores X_fit_ neighbors
    assert not _operator_is_stateless(ScatterSimulationMSC(reference_mode="global_mean"))  # stores global_mean_

    # Step-level routing: stateless count-mode → global; balanced/supervised + any stateful → fold-local.
    assert _augmentation_is_leakage_free({"sample_augmentation": {"transformers": [GaussianAdditiveNoise(sigma=0.01)], "count": 1, "selection": "all"}})
    assert not _augmentation_is_leakage_free({"sample_augmentation": {"transformers": [GaussianAdditiveNoise(sigma=0.01)], "balance": "y", "max_factor": 2}})
    assert not _augmentation_is_leakage_free({"sample_augmentation": {"transformers": [LocalMixupAugmenter()], "count": 1}})
    assert not _augmentation_is_leakage_free({"sample_augmentation": {"transformers": [GaussianAdditiveNoise(sigma=0.01), LocalMixupAugmenter()], "count": 1}})


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_run_via_dagml_fold_local_stateful_augmentation(tmp_path) -> None:
    """FOLD-LOCAL stateful augmentation (#32) runs e2e on dag-ml == direct per-fold-augmented sklearn OOF.

    A stateful augmenter (LocalMixup, whose synthetic child interpolates toward a neighbor drawn from the
    fit X) is fit INSIDE each fold's train only — so each fold has its OWN children and a fold's children
    never train into another fold (the global #8 path would fit on the whole train, leaking future
    fold-val neighbors). We assert:

    * `cv_best_score` == a DIRECT per-fold-augmented sklearn OOF (per fold, fit SNV->PLS on base-train +
      THAT FOLD's children, validate on base-val) — reconstructed from THIS run's pickled augmented
      dataset + its `fold_children` map (augmentation is stochastic, so the baseline must reuse the exact
      synthetic rows and per-fold attribution the run used);
    * it DIFFERS from the no-augmentation OOF (children actually train);
    * NO augmented child appears in the validation/OOF (the origin-boundary leakage guard).
    """
    import pickle

    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline

    from nirs4all.operators.augmentation.spectral import LocalMixupAugmenter
    from nirs4all.operators.transforms.scalers import StandardNormalVariate
    from nirs4all.pipeline.dagml.run_backend import run_via_dagml

    n_comp = 5
    pipeline = [
        StandardNormalVariate(),
        {"sample_augmentation": {"transformers": [LocalMixupAugmenter(k_neighbors=5, random_state=7)], "count": 1, "selection": "all", "random_state": 7}},
        KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42),
        {"model": PLSRegression(n_components=n_comp)},
    ]
    result = run_via_dagml(pipeline, dataset_path("regression"), workdir=tmp_path)

    # Reconstruct the EXACT augmented dataset + per-fold children the fold-local run used.
    payload = pickle.loads((tmp_path / "augment" / "augmented_dataset.pkl").read_bytes())  # noqa: S301 - test-written
    assert isinstance(payload, dict), "fold-local augmentation must pickle {dataset, fold_children}"
    aug_ds, fold_children = payload["dataset"], payload["fold_children"]
    assert set(fold_children) == {f"fold{i}" for i in range(_N_SPLITS)} | {"refit"}

    base_train = [int(s) for s in DatasetConfigs(dataset_path("regression")).get_dataset_at(0).index_column("sample", {"partition": "train"})]
    all_samples = [int(s) for s in aug_ds.index_column("sample", {})]
    all_origins = [int(o) for o in aug_ds.index_column("origin", {})]
    assert any(s != o for s, o in zip(all_samples, all_origins, strict=True)), "augmentation must create synthetic rows"
    folds = [([base_train[i] for i in tr], [base_train[i] for i in va]) for tr, va in KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42).split(base_train)]

    def y_of(sample_int: int) -> float:
        origin_int = all_origins[all_samples.index(sample_int)]
        return float(np.asarray(aug_ds.y({"sample": [origin_int]})).ravel()[0])

    # DIRECT per-fold-augmented OOF: per fold, fit on base-train + THAT FOLD's children, val on base-val.
    oof_pred: dict[int, float] = {}
    oof_true: dict[int, float] = {}
    for fold_index, (train_ints, val_ints) in enumerate(folds):
        fold_kids = fold_children[f"fold{fold_index}"]
        fit_ints = list(train_ints) + [child for origin in train_ints for child in fold_kids.get(origin, [])]
        model = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=n_comp))
        model.fit(np.asarray(aug_ds.x_rows(fit_ints, layout="2d"), dtype=float), np.asarray([y_of(s) for s in fit_ints], dtype=float))
        pred = np.asarray(model.predict(np.asarray(aug_ds.x_rows(list(val_ints), layout="2d"), dtype=float))).ravel()
        for position, sample_int in enumerate(val_ints):
            oof_pred[sample_int], oof_true[sample_int] = float(pred[position]), y_of(sample_int)
    keys = sorted(oof_pred)
    direct_aug = float(np.sqrt(mean_squared_error([oof_true[k] for k in keys], [oof_pred[k] for k in keys])))
    assert abs(result.cv_best_score - direct_aug) < 1e-3, (result.cv_best_score, direct_aug)

    # NO-augmentation OOF baseline — the fold-local run must DIFFER (the children actually train).
    no_pred: dict[int, float] = {}
    for train_ints, val_ints in folds:
        model = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=n_comp))
        model.fit(np.asarray(aug_ds.x_rows(list(train_ints), layout="2d"), dtype=float), np.asarray([y_of(s) for s in train_ints], dtype=float))
        pred = np.asarray(model.predict(np.asarray(aug_ds.x_rows(list(val_ints), layout="2d"), dtype=float))).ravel()
        for position, sample_int in enumerate(val_ints):
            no_pred[sample_int] = float(pred[position])
    no_aug = float(np.sqrt(mean_squared_error([oof_true[k] for k in keys], [no_pred[k] for k in keys])))
    assert abs(result.cv_best_score - no_aug) > 1e-3, "augmented samples must actually train"

    # The OOF covers base val only — NO augmented child is ever validated (the leakage guard).
    bundle = json.loads((tmp_path / "augment" / "bundle.json").read_text())
    avg = [r for r in bundle["scores"]["reports"] if r["partition"] == "validation" and r.get("fold_id") == "avg"]
    assert len(avg) == 1 and avg[0]["row_count"] == len(base_train), "OOF must cover exactly the base train universe"


def test_run_cv_refit_bundle_drops_stale_pickle_env(tmp_path, monkeypatch) -> None:
    """A non-augmentation run must NOT inherit a stale N4A_DAGML_DATASET_PICKLE / sample-meta from the env.

    The adapter prioritizes those vars over the dataset path, so a value left in the parent environment
    (an earlier augmentation/branch run, or the caller's shell) would make a plain run load the wrong
    dataset. `run_cv_refit_bundle` must set each var ONLY when it passes the corresponding argument and
    explicitly drop it otherwise. Asserted by capturing the child env handed to subprocess.run — no CLI
    binary or dag-ml needed (the subprocess call is stubbed before it would launch).
    """
    import subprocess

    from nirs4all.pipeline.dagml import cli_runner

    monkeypatch.setenv("N4A_DAGML_DATASET_PICKLE", "/stale/parent/dataset.pkl")
    monkeypatch.setenv("N4A_DAGML_SAMPLE_META_PATH", "/stale/parent/meta.json")

    captured: dict[str, dict[str, str]] = {}

    def _fake_run(args, **kwargs):  # noqa: ANN001, ANN003 - test stub mirroring subprocess.run
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(cli_runner.subprocess, "run", _fake_run)

    # No dataset_pickle and no sample_metadata → both stale vars must be DROPPED from the child env.
    cli_runner.run_cv_refit_bundle(
        dsl={"id": "x", "pipeline": []}, envelope={}, graph={"nodes": [], "edges": []},
        dataset_path="/real/dataset", workdir=tmp_path, dagml_cli="/bin/true", venv_python="/usr/bin/python3",
    )
    env = captured["env"]
    assert "N4A_DAGML_DATASET_PICKLE" not in env, "a non-augmentation run must not carry a stale pickle var"
    assert "N4A_DAGML_SAMPLE_META_PATH" not in env, "a non-branch run must not carry a stale sample-meta var"
    assert env["N4A_DAGML_DATASET_PATH"] == "/real/dataset"

    # WITH a pickle → exactly that value is set (no stale leakage, the fresh value wins).
    cli_runner.run_cv_refit_bundle(
        dsl={"id": "x", "pipeline": []}, envelope={}, graph={"nodes": [], "edges": []},
        dataset_path="/real/dataset", workdir=tmp_path, dagml_cli="/bin/true", venv_python="/usr/bin/python3",
        dataset_pickle=str(tmp_path / "augmented.pkl"),
    )
    assert captured["env"]["N4A_DAGML_DATASET_PICKLE"] == str(tmp_path / "augmented.pkl")


# ---------------------------------------------------------------------------
# Group/domain calibration splitters (backlog #25): the host `_build_folds` must feed the splitter
# the REAL X (and y for supervised) it partitions on, not the sample-int pool. KFold/ShuffleSplit are
# index-only (covered above); KennardStone/SPXY/KMeans/SPXYFold/KBinsStratified are distance/supervised
# NIRS splitters. These run wheel-free (no CLI) — they exercise `_build_folds` against the legacy feed.
# ---------------------------------------------------------------------------


def test_build_folds_feeds_real_xy_to_distance_splitters() -> None:
    """`_build_folds` partitions distance/supervised splitters on REAL X/y == the legacy controller.

    Before the fix `_build_folds` passed the sample-int `pool` AS X, so KennardStone/KMeans raised on
    `cdist`/KMeans (1-D), and SPXY/SPXYFold/KBinsStratified raised "y required". The fix fetches the real
    spectra (`x_rows`, request order) and targets (re-keyed to pool order) and feeds them exactly like
    the legacy `CrossValidatorController`. This pins fold-for-fold identity with the legacy split,
    wheel-free (no dag-ml-cli)."""
    import warnings

    from nirs4all.operators.splitters import (
        KBinsStratifiedSplitter,
        KennardStoneSplitter,
        KMeansSplitter,
        SPXYFold,
        SPXYSplitter,
    )
    from nirs4all.pipeline.dagml.run_backend import _build_folds

    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    pool = [int(s) for s in dataset.index_column("sample", {"partition": "train"})]
    real_x = np.asarray(dataset.x_rows(pool, layout="2d"))
    y_block = np.asarray(dataset.y({"sample": pool}))
    stored = dataset.index_column("sample", {"sample": pool})
    row_of = {int(s): r for r, s in enumerate(stored)}
    real_y = y_block[[row_of[int(s)] for s in pool]].ravel()

    def legacy(splitter, needs_y: bool):
        kwargs = {"y": real_y} if needs_y else {}
        return [(sorted(pool[i] for i in tr), sorted(pool[i] for i in va)) for tr, va in splitter.split(real_x, **kwargs)]

    def via_build_folds(splitter):
        return [(sorted(tr), sorted(va)) for tr, va in _build_folds(splitter, dataset, pool, set())]

    # (name, factory, needs_y) — a fresh instance per call since `.split` returns a one-shot generator.
    cases = [
        ("KennardStone", lambda: KennardStoneSplitter(test_size=0.25), False),
        ("SPXY", lambda: SPXYSplitter(test_size=0.25), True),
        ("KMeans", lambda: KMeansSplitter(test_size=0.25, random_state=42), False),
        ("SPXYFold", lambda: SPXYFold(n_splits=_N_SPLITS), True),
        ("KBinsStratified", lambda: KBinsStratifiedSplitter(test_size=0.25, random_state=42, n_bins=5), True),
    ]
    for name, factory, needs_y in cases:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = legacy(factory(), needs_y)
            actual = via_build_folds(factory())
        assert actual == expected, f"{name}: _build_folds folds diverge from the legacy real-X/y split"


def test_build_folds_rejects_group_required_splitter_loud() -> None:
    """A group-REQUIRED splitter fails loud (the dag-ml path has no group source yet — backlog #21).

    `BinnedStratifiedGroupKFold` cannot run without a group; the engine path carries no
    group_by/repetition group source, so it must raise a clear NotImplementedError naming #21 rather
    than silently splitting without the group constraint."""
    from nirs4all.operators.splitters import BinnedStratifiedGroupKFold
    from nirs4all.pipeline.dagml.run_backend import _build_folds

    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    pool = [int(s) for s in dataset.index_column("sample", {"partition": "train"})]
    with pytest.raises(NotImplementedError, match="#21"):
        _build_folds(BinnedStratifiedGroupKFold(n_splits=3, n_bins=3), dataset, pool, set())


def _host_split_oof_and_test(dataset, make_splitter, n_components: int = 5) -> tuple[float, float]:
    """Direct host-split (the same `_build_folds`) + per-fold sklearn OOF + final-test, sample-aligned.

    Mirrors `run_via_dagml`'s pipeline (SNV → PLS) per fold; every X/y read is by sample int via
    `x_rows`/re-keyed `y` so the baseline never falls into the storage-vs-request order trap. Returns
    `(cv_oof_rmse, final_test_rmse)` — what dag-ml's native `cv_best_score`/`best_rmse` must reproduce.
    """
    import warnings

    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline

    from nirs4all.operators.transforms.scalers import StandardNormalVariate
    from nirs4all.pipeline.dagml.run_backend import _build_folds

    train = [int(s) for s in dataset.index_column("sample", {"partition": "train"})]
    test = [int(s) for s in dataset.index_column("sample", {"partition": "test"})]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        folds = _build_folds(make_splitter(), dataset, train, set())

    def xof(ids):
        return np.asarray(dataset.x_rows(ids, layout="2d"), dtype=float)

    def yof(ids):
        block = np.asarray(dataset.y({"sample": ids}), dtype=float)
        stored = dataset.index_column("sample", {"sample": ids})
        row = {int(s): r for r, s in enumerate(stored)}
        return block[[row[int(s)] for s in ids]].ravel()

    acc: dict[int, float] = {}
    cnt: dict[int, int] = {}
    tru: dict[int, float] = {}
    for train_ids, val_ids in folds:
        if not val_ids:
            continue
        model = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=n_components))
        model.fit(xof(train_ids), yof(train_ids))
        preds = np.asarray(model.predict(xof(val_ids))).ravel()
        true = yof(val_ids)
        for sample_int, pred, target in zip(val_ids, preds, true):
            acc[sample_int] = acc.get(sample_int, 0.0) + float(pred)
            cnt[sample_int] = cnt.get(sample_int, 0) + 1
            tru[sample_int] = float(target)
    keys = sorted(acc)
    cv_oof = float(np.sqrt(mean_squared_error([tru[k] for k in keys], [acc[k] / cnt[k] for k in keys])))

    final = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=n_components))
    final.fit(xof(train), yof(train))
    test_rmse = float(np.sqrt(mean_squared_error(yof(test), np.asarray(final.predict(xof(test))).ravel())))
    return cv_oof, test_rmse


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_kennard_stone() -> None:
    """engine="dag-ml" runs a KennardStone (single train/val, X-distance) split == direct host OOF.

    KennardStone is a DISTANCE splitter — it selects the train subset by max-min spectral distance, so
    the host must feed it the real spectra. The engine's native single-fold OOF (`cv_best_score`) and
    final-test (`best_rmse`) match the direct host-split + sklearn baseline."""
    import nirs4all
    from nirs4all.operators.splitters import KennardStoneSplitter
    from nirs4all.operators.transforms.scalers import StandardNormalVariate

    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    cv_oof, test_rmse = _host_split_oof_and_test(dataset, lambda: KennardStoneSplitter(test_size=0.25))

    pipeline = [StandardNormalVariate(), KennardStoneSplitter(test_size=0.25), {"model": PLSRegression(n_components=5)}]
    result = nirs4all.run(pipeline, dataset_path("regression"), engine="dag-ml")
    assert abs(result.cv_best_score - cv_oof) < 1e-3
    assert abs(result.best_rmse - test_rmse) < 1e-3


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_spxy() -> None:
    """engine="dag-ml" runs an SPXY (single train/val, joint X+Y distance) split == direct host OOF.

    SPXY is a SUPERVISED distance splitter — it partitions on joint feature+target distance, so the host
    must feed it real X AND y (the index-only feed raised "y required" before the fix). The engine's
    native OOF (`cv_best_score`) and final-test (`best_rmse`) match the direct host-split baseline."""
    import nirs4all
    from nirs4all.operators.splitters import SPXYSplitter
    from nirs4all.operators.transforms.scalers import StandardNormalVariate

    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    cv_oof, test_rmse = _host_split_oof_and_test(dataset, lambda: SPXYSplitter(test_size=0.25))

    pipeline = [StandardNormalVariate(), SPXYSplitter(test_size=0.25), {"model": PLSRegression(n_components=5)}]
    result = nirs4all.run(pipeline, dataset_path("regression"), engine="dag-ml")
    assert abs(result.cv_best_score - cv_oof) < 1e-3
    assert abs(result.best_rmse - test_rmse) < 1e-3


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_spxy_fold() -> None:
    """engine="dag-ml" runs an SPXYFold K-fold (joint X+Y distance, OOF partition) == direct host OOF.

    SPXYFold assigns each sample to one of K folds by joint X+Y distance — a clean OOF partition (so
    dag-ml's cross-fold `avg` report scores it). The native cross-fold OOF average (`cv_best_score`) and
    final-test (`best_rmse`) match the direct host-split + sklearn OOF baseline."""
    import nirs4all
    from nirs4all.operators.splitters import SPXYFold
    from nirs4all.operators.transforms.scalers import StandardNormalVariate

    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    cv_oof, test_rmse = _host_split_oof_and_test(dataset, lambda: SPXYFold(n_splits=_N_SPLITS))

    pipeline = [StandardNormalVariate(), SPXYFold(n_splits=_N_SPLITS), {"model": PLSRegression(n_components=5)}]
    result = nirs4all.run(pipeline, dataset_path("regression"), engine="dag-ml")
    assert abs(result.cv_best_score - cv_oof) < 1e-3
    assert abs(result.best_rmse - test_rmse) < 1e-3


# ---------------------------------------------------------------------------
# Supervised + column-changing X-transforms (backlog #24): OSC/EPO are SUPERVISED
# orthogonalizations (fit on X AND a second signal); CARS/MCUVE are feature SELECTORS that CHANGE the
# column count. All four are TransformerMixin X-transforms, so the model node's upstream X-chain
# (`make_pipeline(transform, model)`) already runs them leakage-safe: sklearn's `Pipeline.fit` passes
# the fold-train `y` to each transform's `fit_transform` (so OSC sees the supervised signal, fit on
# fold-train only), and the column-count change propagates cleanly through fold-train fit / fold-val +
# test apply. VERIFIED-NATIVE: no host X-chain change was needed — these tests pin that.
#
# CARS/MCUVE `fit` is RNG Monte-Carlo and therefore ROW-ORDER sensitive: the selected feature set
# depends on the training row order. dag-ml's fold-train / full-train ordering is its own identity-keyed
# pool order (fold-encounter order), NOT sorted-train, so the parity baseline must replicate it — exactly
# like the distance-splitter baselines above use `x_rows` in request order. `_dagml_fold_order` derives
# that order from the same folds the KFold splitter feeds, with no dag-ml internals.
# ---------------------------------------------------------------------------


def _dagml_fold_order(dataset, n_splits: int = _N_SPLITS) -> tuple[list[tuple[list[int], list[int]]], list[int]]:
    """The folds + full-train pool in dag-ml's OWN row order, for an order-sensitive transform baseline.

    Returns ``(folds, full_train_pool)`` where each fold is ``(train_ints, validation_ints)`` in the
    KFold split order dag-ml's ``build_fold_set`` preserves, and ``full_train_pool`` is the fold-encounter
    order ``build_fold_set`` uses for the REFIT ``full_train`` view (first appearance of each sample across
    each fold's train+validation, in fold order). Mirrors ``envelope.build_fold_set`` exactly so an
    order-sensitive selector (CARS/MCUVE) reproduces the engine's selection without touching dag-ml.
    """
    train = [int(s) for s in dataset.index_column("sample", {"partition": "train"})]
    folds = [([train[i] for i in tr], [train[i] for i in va]) for tr, va in KFold(n_splits=n_splits, shuffle=True, random_state=42).split(train)]
    pool: list[int] = []
    seen: set[int] = set()
    for train_ints, validation_ints in folds:
        for sample_int in (*train_ints, *validation_ints):
            if sample_int not in seen:
                seen.add(sample_int)
                pool.append(sample_int)
    return folds, pool


def _transform_oof_and_test(dataset, make_transform, n_components: int = 5) -> tuple[float, float]:
    """Direct sklearn OOF + final-test for ``make_pipeline(transform, PLS)`` in dag-ml's row order.

    Per fold: fit ``make_pipeline(transform, PLS)`` on the fold-train rows, predict the fold-validation
    rows; reassemble the OOF and score it (``cv_best_score``). Refit on the full-train pool (dag-ml's
    fold-encounter order) and score the held-out test (``best_rmse``). Every X/y read is by sample int
    via ``x_rows`` / re-keyed ``y`` so the baseline shares dag-ml's exact row order — required for the
    order-sensitive selectors. Returns ``(cv_oof_rmse, final_test_rmse)``.
    """
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline

    folds, pool = _dagml_fold_order(dataset)
    test = [int(s) for s in dataset.index_column("sample", {"partition": "test"})]

    def xof(ids: list[int]) -> np.ndarray:
        return np.asarray(dataset.x_rows([int(i) for i in ids], layout="2d"), dtype=float)

    def yof(ids: list[int]) -> np.ndarray:
        ids = [int(i) for i in ids]
        block = np.asarray(dataset.y({"sample": ids}), dtype=float)
        stored = dataset.index_column("sample", {"sample": ids})
        row = {int(s): r for r, s in enumerate(stored)}
        return block[[row[int(s)] for s in ids]].ravel()

    oof_pred: dict[int, float] = {}
    oof_true: dict[int, float] = {}
    for train_ids, val_ids in folds:
        model = make_pipeline(make_transform(), PLSRegression(n_components=n_components))
        model.fit(xof(train_ids), yof(train_ids))
        preds = np.asarray(model.predict(xof(val_ids))).ravel()
        true = yof(val_ids)
        for sample_int, pred, target in zip(val_ids, preds, true, strict=True):
            oof_pred[int(sample_int)] = float(pred)
            oof_true[int(sample_int)] = float(target)
    keys = sorted(oof_pred)
    cv_oof = float(np.sqrt(mean_squared_error([oof_true[k] for k in keys], [oof_pred[k] for k in keys])))

    final = make_pipeline(make_transform(), PLSRegression(n_components=n_components))
    final.fit(xof(pool), yof(pool))
    test_rmse = float(np.sqrt(mean_squared_error(yof(test), np.asarray(final.predict(xof(test))).ravel())))
    return cv_oof, test_rmse


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_osc() -> None:
    """engine="dag-ml" runs OSC (SUPERVISED orthogonalization) == direct sklearn Pipeline(OSC, PLS).

    OSC's `fit(X, y)` needs the target to find Y-orthogonal variation to remove; the X-chain runs it as
    `make_pipeline(OSC, PLS)`, so sklearn passes the fold-train `y` to `OSC.fit_transform` (leakage-safe —
    train only) and re-applies the stored deflation at fold-val/test. The native OOF (`cv_best_score`) and
    final-test (`best_rmse`) match the direct baseline, proving the supervised signal reaches OSC's fit."""
    import nirs4all
    from nirs4all.operators.transforms.orthogonalization import OSC

    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    cv_oof, test_rmse = _transform_oof_and_test(dataset, lambda: OSC(n_components=2))

    pipeline = [OSC(n_components=2), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}]
    result = nirs4all.run(pipeline, dataset_path("regression"), engine="dag-ml")
    assert abs(result.cv_best_score - cv_oof) < 1e-3, (result.cv_best_score, cv_oof)
    assert abs(result.best_rmse - test_rmse) < 1e-3, (result.best_rmse, test_rmse)


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_epo() -> None:
    """engine="dag-ml" runs EPO (orthogonalization, fit on a second signal) == sklearn Pipeline(EPO, PLS).

    EPO orthogonalizes X against a second `fit(X, d)` argument (the X-chain feeds it the fold-train `y`
    via sklearn's Pipeline), so it exercises the same supervised-fit path as OSC. The native OOF
    (`cv_best_score`) and final-test (`best_rmse`) match the direct `make_pipeline(EPO, PLS)` baseline."""
    import nirs4all
    from nirs4all.operators.transforms.orthogonalization import EPO

    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    cv_oof, test_rmse = _transform_oof_and_test(dataset, lambda: EPO())

    pipeline = [EPO(), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}]
    result = nirs4all.run(pipeline, dataset_path("regression"), engine="dag-ml")
    assert abs(result.cv_best_score - cv_oof) < 1e-3, (result.cv_best_score, cv_oof)
    assert abs(result.best_rmse - test_rmse) < 1e-3, (result.best_rmse, test_rmse)


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_cars() -> None:
    """engine="dag-ml" runs CARS (feature SELECTION — CHANGES the column count) == sklearn Pipeline(CARS, PLS).

    CARS's `fit(X, y)` selects a wavelength subset; the model then sees FEWER columns. The X-chain runs
    `make_pipeline(CARS, PLS)`, so the column-count reduction propagates leakage-safe through fold-train
    fit / fold-val + test apply (CARS fits the mask on fold-train only). The baseline uses dag-ml's own
    row order (`_dagml_fold_order`) because CARS's RNG Monte-Carlo selection is row-order sensitive — the
    native OOF (`cv_best_score`) and final-test (`best_rmse`) then match it exactly."""
    import nirs4all
    from nirs4all.operators.transforms.feature_selection import CARS

    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    cv_oof, test_rmse = _transform_oof_and_test(dataset, lambda: CARS(n_components=5, n_sampling_runs=20, random_state=42))

    pipeline = [CARS(n_components=5, n_sampling_runs=20, random_state=42), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}]
    result = nirs4all.run(pipeline, dataset_path("regression"), engine="dag-ml")
    assert abs(result.cv_best_score - cv_oof) < 1e-3, (result.cv_best_score, cv_oof)
    assert abs(result.best_rmse - test_rmse) < 1e-3, (result.best_rmse, test_rmse)


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_mcuve() -> None:
    """engine="dag-ml" runs MC-UVE (feature SELECTION — CHANGES the column count) == sklearn Pipeline(MCUVE, PLS).

    MC-UVE's `fit(X, y)` eliminates uninformative wavelengths against a noise baseline, so the model sees
    FEWER columns. Same column-changing X-chain path as CARS, leakage-safe (mask fit on fold-train only),
    same row-order-sensitive RNG selection — so the baseline uses dag-ml's own row order. The native OOF
    (`cv_best_score`) and final-test (`best_rmse`) match the direct `make_pipeline(MCUVE, PLS)` baseline."""
    import nirs4all
    from nirs4all.operators.transforms.feature_selection import MCUVE

    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    cv_oof, test_rmse = _transform_oof_and_test(dataset, lambda: MCUVE(n_components=5, n_iterations=40, random_state=42))

    pipeline = [MCUVE(n_components=5, n_iterations=40, random_state=42), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}]
    result = nirs4all.run(pipeline, dataset_path("regression"), engine="dag-ml")
    assert abs(result.cv_best_score - cv_oof) < 1e-3, (result.cv_best_score, cv_oof)
    assert abs(result.best_rmse - test_rmse) < 1e-3, (result.best_rmse, test_rmse)


# ---------------------------------------------------------------------------
# concat_transform (backlog #27): the top-level single-source REPLACE-mode form hstacks several
# sub-transformers' outputs into one wider 2D feature matrix — exactly `FeatureUnion` semantics. The
# bridge lowers it to ONE `FeatureConcat` X-transform node, so the model's upstream X-chain runs it
# leakage-safe (fit fold-train, apply fold-val/test), like the column-changing X-transforms above.
# Parity is asserted with ROW-INDEPENDENT sub-transformers (SNV / SavitzkyGolay derivative): each row
# is transformed independently, so the result is deterministic and order-insensitive — unlike a
# PCA/TruncatedSVD concat, whose randomized/near-degenerate SVD makes EXACT parity meaningless (the
# components flip/reorder on a hair of row-order or float noise, a property of those reducers, not the
# concat mechanism). The processing-AXIS shapes (multi-processing, the feature_augmentation `add`
# mode, a named layer, a pass-through channel) need the multi-block data-plane and stay fail-loud
# (backlog #29/#31).
# ---------------------------------------------------------------------------


def _concat_oof_and_test(dataset, make_concat_step, n_components: int = 15) -> tuple[float, float]:
    """Direct sklearn OOF + final-test for `make_pipeline(FeatureConcat-equivalent, PLS)`, dag-ml order.

    Builds the host `FeatureConcat` operator the bridge lowers `make_concat_step()` to (so the baseline
    runs the SAME hstack the engine runs) and scores it per dag-ml's fold/refit row order. Returns
    `(cv_oof_rmse, final_test_rmse)`.
    """
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline

    from nirs4all.pipeline.dagml.operator_routing import route_operator
    from nirs4all.pipeline.dagml_bridge import _lower_concat_transform

    lowered = _lower_concat_transform(make_concat_step())

    def make_concat():
        return route_operator("transform", lowered["class"], lowered["params"])

    folds, pool = _dagml_fold_order(dataset)
    test = [int(s) for s in dataset.index_column("sample", {"partition": "test"})]

    def xof(ids):
        return np.asarray(dataset.x_rows([int(i) for i in ids], layout="2d"), dtype=float)

    def yof(ids):
        ids = [int(i) for i in ids]
        block = np.asarray(dataset.y({"sample": ids}), dtype=float)
        stored = dataset.index_column("sample", {"sample": ids})
        row = {int(s): r for r, s in enumerate(stored)}
        return block[[row[int(s)] for s in ids]].ravel()

    oof_pred: dict[int, float] = {}
    oof_true: dict[int, float] = {}
    for train_ids, val_ids in folds:
        model = make_pipeline(make_concat(), PLSRegression(n_components=n_components))
        model.fit(xof(train_ids), yof(train_ids))
        preds = np.asarray(model.predict(xof(val_ids))).ravel()
        true = yof(val_ids)
        for sample_int, pred, target in zip(val_ids, preds, true, strict=True):
            oof_pred[int(sample_int)] = float(pred)
            oof_true[int(sample_int)] = float(target)
    keys = sorted(oof_pred)
    cv_oof = float(np.sqrt(mean_squared_error([oof_true[k] for k in keys], [oof_pred[k] for k in keys])))

    final = make_pipeline(make_concat(), PLSRegression(n_components=n_components))
    final.fit(xof(pool), yof(pool))
    test_rmse = float(np.sqrt(mean_squared_error(yof(test), np.asarray(final.predict(xof(test))).ravel())))
    return cv_oof, test_rmse


def test_concat_transform_bridge_lowers_to_feature_concat() -> None:
    """The bridge lowers a supported `concat_transform` step to ONE `FeatureConcat` transform node."""
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    from nirs4all.pipeline.dagml_bridge import _step_to_dsl

    dsl = _step_to_dsl({"concat_transform": [StandardScaler(), MinMaxScaler()]})
    assert dsl["class"] == "nirs4all.operators.transforms.concat.FeatureConcat"
    ops = dsl["params"]["operations"]
    assert [op["class"] for op in ops] == ["sklearn.preprocessing._data.StandardScaler", "sklearn.preprocessing._data.MinMaxScaler"]


def test_concat_transform_3d_shapes_fail_loud() -> None:
    """The processing-AXIS shapes raise NotImplementedError naming the data-plane (#29/#31), never silent.

    A `name`/`source_processing` selector targets a named 3D layer; a nested concat and a pass-through
    (None) channel both build a multi-block representation; the `feature_augmentation`-nested `add`
    mode grows the processing axis. All need the multi-source/fusion data-plane — they must fail loud.
    """
    from sklearn.decomposition import PCA

    from nirs4all.pipeline.dagml_bridge import _step_to_dsl

    for step in (
        {"concat_transform": {"operations": [PCA(n_components=5)], "name": "fused"}},
        {"concat_transform": {"operations": [PCA(n_components=5)], "source_processing": "snv"}},
        {"concat_transform": [PCA(n_components=5), {"concat_transform": [PCA(n_components=3)]}]},
        {"concat_transform": [None, PCA(n_components=5)]},
    ):
        with pytest.raises(NotImplementedError, match="#29/#31"):
            _step_to_dsl(step)
    # feature_augmentation (the 3D processing axis itself) stays fail-loud at the generic boundary.
    with pytest.raises(NotImplementedError, match="feature_augmentation"):
        _step_to_dsl({"feature_augmentation": PCA(n_components=5)})


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_concat_transform() -> None:
    """engine="dag-ml" runs concat_transform (REPLACE mode, hstack of 2 transformers) == sklearn FeatureUnion.

    SNV ⧺ SavitzkyGolay(deriv=1) concatenated column-wise (each row-independent → deterministic), fed to
    PLS. The bridge lowers the step to one `FeatureConcat` node; the model's X-chain runs `make_pipeline(
    FeatureConcat, PLS)`, fit fold-train, applied fold-val/test. Both native scores (`cv_best_score` OOF
    and `best_rmse` final-test) match the direct sklearn baseline EXACTLY (row-independent, so no SVD/PCA
    order instability)."""
    import nirs4all
    from nirs4all.operators.transforms.nirs import SavitzkyGolay
    from nirs4all.operators.transforms.scalers import StandardNormalVariate

    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)

    def make_step():
        return {"concat_transform": [StandardNormalVariate(), SavitzkyGolay(window_length=11, polyorder=2, deriv=1)]}

    cv_oof, test_rmse = _concat_oof_and_test(dataset, make_step)

    pipeline = [make_step(), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=15)}]
    result = nirs4all.run(pipeline, dataset_path("regression"), engine="dag-ml")
    assert abs(result.cv_best_score - cv_oof) < 1e-6, (result.cv_best_score, cv_oof)
    assert abs(result.best_rmse - test_rmse) < 1e-6, (result.best_rmse, test_rmse)


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_concat_transform_with_chain() -> None:
    """engine="dag-ml" runs concat_transform with a CHAIN entry (`[SG, SNV]`) == sklearn sequential hstack.

    A `concat_transform` operation may be a chain (sequential sub-transformers); the lowered `FeatureConcat`
    applies the chain (`SNV(SG(X))`) before concatenating it beside the single `SNV(X)` channel. Both native
    scores match the direct baseline exactly (row-independent transforms)."""
    import nirs4all
    from nirs4all.operators.transforms.nirs import SavitzkyGolay
    from nirs4all.operators.transforms.scalers import StandardNormalVariate

    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)

    def make_step():
        return {"concat_transform": [StandardNormalVariate(), [SavitzkyGolay(window_length=11, polyorder=2, deriv=1), StandardNormalVariate()]]}

    cv_oof, test_rmse = _concat_oof_and_test(dataset, make_step)

    pipeline = [make_step(), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=15)}]
    result = nirs4all.run(pipeline, dataset_path("regression"), engine="dag-ml")
    assert abs(result.cv_best_score - cv_oof) < 1e-6, (result.cv_best_score, cv_oof)
    assert abs(result.best_rmse - test_rmse) < 1e-6, (result.best_rmse, test_rmse)


# ---------------------------------------------------------------------------
# Repetitions (sample-grain grouping — replicate spectra share one sample). Backlog #21.
# ---------------------------------------------------------------------------

_REPETITION_DS = PARSER_FIXTURES["aggregate_mean"]  # Mcal.csv carries `sample_id` (3 reps/sample)
_REP_COL = "sample_id"


def _group_aware_oof_and_test(dataset, splitter, n_components: int = 5) -> tuple[float, float]:
    """Direct GROUP-aware sklearn OOF + final-test at the REPETITION grain, sample-id aligned.

    Mirrors `_run_repetition`: build the SAME group-aware folds (`_build_group_folds`), then per fold
    fit PLS on the fold-train repetition rows and validate on the fold-val rows; every X/y read is by
    sample int via `x_rows` / re-keyed `y` (never the storage-vs-request order trap). Each rep row is
    scored individually — exactly what nirs4all's `cv_best_score`/`best_rmse` report (the sample-level
    `_agg` aggregation is a separate twin, NOT those two scores). Returns `(cv_oof_rmse, test_rmse)`.
    """
    from sklearn.metrics import mean_squared_error

    from nirs4all.pipeline.dagml.run_backend import _build_group_folds

    train = [int(s) for s in dataset.index_column("sample", {"partition": "train"})]
    test = [int(s) for s in dataset.index_column("sample", {"partition": "test"})]
    folds = _build_group_folds(splitter, dataset, train)

    def xof(ids):
        return np.asarray(dataset.x_rows(ids, layout="2d"), dtype=float)

    def yof(ids):
        block = np.asarray(dataset.y({"sample": ids}, include_augmented=False), dtype=float)
        stored = dataset.index_column("sample", {"sample": ids})
        row = {int(s): r for r, s in enumerate(stored)}
        return block[[row[int(s)] for s in ids]].ravel()

    oof_pred: dict[int, float] = {}
    oof_true: dict[int, float] = {}
    for train_ids, val_ids in folds:
        model = PLSRegression(n_components=n_components)
        model.fit(xof(train_ids), yof(train_ids))
        preds = np.asarray(model.predict(xof(val_ids))).ravel()
        true = yof(val_ids)
        for sample_int, pred, target in zip(val_ids, preds, true, strict=True):
            oof_pred[sample_int], oof_true[sample_int] = float(pred), float(target)
    keys = sorted(oof_pred)
    cv_oof = float(np.sqrt(mean_squared_error([oof_true[k] for k in keys], [oof_pred[k] for k in keys])))

    final = PLSRegression(n_components=n_components)
    final.fit(xof(train), yof(train))
    test_rmse = float(np.sqrt(mean_squared_error(yof(test), np.asarray(final.predict(xof(test))).ravel())))
    return cv_oof, test_rmse


def test_build_group_folds_keeps_repetitions_together() -> None:
    """Group-aware folds never split a sample's replicates across train/val (the leakage invariant).

    Verified for BOTH an index-only splitter (KFold, wrapped by GroupedSplitterWrapper) and a native
    group-REQUIRED splitter (BinnedStratifiedGroupKFold) — the latter is the splitter the non-rep path
    rejects loud (#25/#21); a repetition column now supplies its group so it runs group-leakage-safe.
    """
    from nirs4all.operators.splitters import BinnedStratifiedGroupKFold
    from nirs4all.pipeline.dagml.run_backend import _build_group_folds, _repetition_groups_for_pool

    dataset = DatasetConfigs(str(_REPETITION_DS), repetition=_REP_COL).get_dataset_at(0)
    pool = [int(s) for s in dataset.index_column("sample", {"partition": "train"})]
    assert len(pool) > len({str(g) for g in _repetition_groups_for_pool(dataset, pool)}), "fixture must have >1 rep per sample"

    for splitter in (KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), BinnedStratifiedGroupKFold(n_splits=_N_SPLITS, n_bins=3)):
        folds = _build_group_folds(splitter, dataset, pool)
        groups = _repetition_groups_for_pool(dataset, pool)
        pos = {sample_int: index for index, sample_int in enumerate(pool)}
        # Every rep row validated exactly once (a clean OOF partition over the rep universe).
        validated = [sample_int for _train, val in folds for sample_int in val]
        assert sorted(validated) == sorted(pool), f"{splitter.__class__.__name__}: OOF must cover every rep row once"
        # No sample's group spans train/val in any fold (group-leakage-safe).
        for train_ids, val_ids in folds:
            train_groups = {groups[pos[s]] for s in train_ids}
            val_groups = {groups[pos[s]] for s in val_ids}
            assert not (train_groups & val_groups), f"{splitter.__class__.__name__}: a sample's reps leaked across train/val"


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_repetitions() -> None:
    """engine="dag-ml" runs a REPETITION dataset group-aware == direct group-aware sklearn OOF (#21).

    A repetition dataset (`sample_id` groups 3 replicate spectra per physical sample) runs through the
    native repetition path: folds are GROUP-aware (all reps of a sample on one fold side), the envelope
    emits each row's `group_id`, and dag-ml-data refuses any fold that splits a group. Each rep row is
    scored individually — the repetition grain that `cv_best_score`/`best_rmse` report (sample-level
    `_agg` aggregation is a separate twin, not these scores), so NO aggregation reducer is needed.

    Asserts the native `cv_best_score` and `best_rmse` match a direct group-aware sklearn OOF + final-test
    within 1e-3. (Parity is vs the direct group-aware computation, not legacy: legacy's in-memory RunResult
    has no `fold_id="avg"` OOF entry for this path, so its `cv_best_score` degenerates to the final-test
    value — dag-ml is the correction, the same pattern as augmentation #14 and branch+merge.)
    """
    import nirs4all

    dataset = DatasetConfigs(str(_REPETITION_DS), repetition=_REP_COL).get_dataset_at(0)
    cv_oof, test_rmse = _group_aware_oof_and_test(dataset, KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42))

    pipeline = [{"model": PLSRegression(n_components=5)}, KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42)]
    result = nirs4all.run(pipeline, DatasetConfigs(str(_REPETITION_DS), repetition=_REP_COL), engine="dag-ml")
    assert abs(result.cv_best_score - cv_oof) < 1e-3, (result.cv_best_score, cv_oof)
    assert abs(result.best_rmse - test_rmse) < 1e-3, (result.best_rmse, test_rmse)


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_repetition_unsupported_composition_fails_loud() -> None:
    """A repetition dataset combined with a branch or augmentation FAILS LOUD (the bypass is closed).

    The repetition guard in `run_via_dagml` runs BEFORE the separation-branch and augmentation dispatch
    (both of which build folds WITHOUT the group constraint, so a rep dataset reaching them could split a
    sample's replicates across train/val = silent group leakage). This pins that closure: each composition
    must raise `NotImplementedError` naming `repetition`/`#21` rather than silently take the group-free path.

    The branch/augmentation steps are real shapes (`_detect_separation_branch` / `_is_augmentation_step`
    recognise them) so the guard is exercised on the actual dispatch — the guard raises before any CLI
    subprocess, so no real run happens despite the binary being present.
    """
    from nirs4all.operators.augmentation import GaussianAdditiveNoise
    from nirs4all.pipeline.dagml.run_backend import _detect_separation_branch, _is_augmentation_step, run_via_dagml

    configs = DatasetConfigs(str(_REPETITION_DS), repetition=_REP_COL)

    branch_pipeline = [
        KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42),
        {"branch": {"by_metadata": "group", "steps": [{"model": PLSRegression(n_components=5)}]}},
        {"merge": "concat"},
    ]
    assert _detect_separation_branch(branch_pipeline) is not None, "branch step must reach the dispatch for this to be a real lock"
    with pytest.raises(NotImplementedError, match=r"repetition.*#21"):
        run_via_dagml(branch_pipeline, configs, dagml_cli=str(_DAGML_CLI))

    aug_pipeline = [
        {"sample_augmentation": {"transformers": [GaussianAdditiveNoise(sigma=0.01)], "count": 1, "selection": "all", "random_state": 42}},
        {"model": PLSRegression(n_components=5)},
        KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42),
    ]
    assert any(_is_augmentation_step(step) for step in aug_pipeline), "augmentation step must reach the dispatch for this to be a real lock"
    with pytest.raises(NotImplementedError, match=r"repetition.*#21"):
        run_via_dagml(aug_pipeline, configs, dagml_cli=str(_DAGML_CLI))


def test_duplication_branch_detection() -> None:
    """The duplication detector consumes ONLY the list-branch + avg/mean fusion-merge shape.

    `_detect_duplication_branch` returns `(branches, aggregate)` for a `{"branch": [[A], [B], …]}`
    list-of-lists (each sub-pipeline with a model) followed by an avg/mean fusion merge; everything else
    (a separation branch, a stacking/concat merge, a modelless or single branch, a top-level step beside
    the branch) returns None so the fail-loud paths still guard it. `_fusion_merge_aggregate` /
    `_is_stacking_merge_step` distinguish the fusion tokens from a stacking merge. No CLI needed."""
    from sklearn.linear_model import Ridge

    from nirs4all.operators.transforms.scalers import StandardNormalVariate
    from nirs4all.pipeline.dagml.run_backend import (
        _detect_duplication_branch,
        _detect_separation_branch,
        _fusion_merge_aggregate,
        _is_stacking_merge_step,
    )

    splitter = KFold(n_splits=3, shuffle=True, random_state=42)

    # Handled: two models on the full data + a mean fusion merge (and its spellings).
    handled = [splitter, {"branch": [[{"model": PLSRegression(n_components=5)}], [{"model": Ridge(alpha=1.0)}]]}, {"merge": "mean"}]
    detected = _detect_duplication_branch(handled)
    assert detected is not None
    branches, aggregate = detected
    assert len(branches) == 2 and aggregate == "mean"
    assert _detect_duplication_branch([splitter, {"branch": [[{"model": PLSRegression()}], [{"model": Ridge()}]]}, {"merge": "average"}]) is not None
    explicit = _detect_duplication_branch([splitter, {"branch": [[{"model": PLSRegression()}], [{"model": Ridge()}]]}, {"merge": {"predictions": "all", "aggregate": "mean"}}])
    assert explicit is not None and explicit[1] == "mean"

    # The fusion token mapping (mean->mean value-average, proba_mean->probability-average) vs stacking.
    assert _fusion_merge_aggregate({"merge": "mean"}) == "mean"
    assert _fusion_merge_aggregate({"merge": {"predictions": "all", "aggregate": "proba_mean"}}) == "proba_mean"
    assert _fusion_merge_aggregate({"merge": "predictions"}) is None  # stacking, not fusion
    assert _fusion_merge_aggregate({"merge": {"predictions": "all", "aggregate": "weighted_mean"}}) is None
    assert _is_stacking_merge_step({"merge": "predictions"}) is True
    assert _is_stacking_merge_step({"merge": {"predictions": [{"branch": 0, "select": "best"}]}}) is True
    assert _is_stacking_merge_step({"merge": "mean"}) is False  # a fusion merge is not stacking

    # Not handled (each falls through to a loud path):
    branch = {"branch": [[{"model": PLSRegression()}], [{"model": Ridge()}]]}
    # a STACKING merge (predictions → meta-model) — #10, never fusion.
    assert _detect_duplication_branch([splitter, branch, {"merge": "predictions"}]) is None
    # a concat merge (separation reassembly), not a fusion average.
    assert _detect_duplication_branch([splitter, branch, {"merge": "concat"}]) is None
    # a separation (dict-form) branch is NOT a duplication branch.
    assert _detect_duplication_branch([splitter, {"branch": {"by_metadata": "group", "steps": [{"model": PLSRegression()}]}}, {"merge": "mean"}]) is None
    # a single branch (need ≥2 to fuse).
    assert _detect_duplication_branch([splitter, {"branch": [[{"model": PLSRegression()}]]}, {"merge": "mean"}]) is None
    # a modelless branch (fusion averages model predictions).
    assert _detect_duplication_branch([splitter, {"branch": [[StandardNormalVariate()], [{"model": Ridge()}]]}, {"merge": "mean"}]) is None
    # a top-level transform beside the branch (would be silently dropped).
    assert _detect_duplication_branch([StandardNormalVariate(), splitter, branch, {"merge": "mean"}]) is None
    # no merge / no branch at all.
    assert _detect_duplication_branch([splitter, branch]) is None
    assert _detect_duplication_branch([splitter, {"merge": "mean"}]) is None
    # the separation detector must NOT claim a list-form (duplication) branch.
    assert _detect_separation_branch(handled) is None


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_duplication_branch_fusion() -> None:
    """A duplication branch (2 DIFFERENT models on the FULL data) + avg/mean fusion merge scores natively.

    The native path runs each branch model on the full fold data (no fan-out, no branch_view — every model
    sees the full view), and dag-ml's fusion merge handler AVERAGES the branches' held-out Validation OOF
    per sample (leakage-safe) into one full-universe OOF. Asserts `cv_best_score` == a DIRECT avg-ensemble
    sklearn baseline (per fold: fit each branch on fold-train, average their fold-val per-sample
    predictions; OOF reassembled, scored) within 1e-3, AND that the fused score DIFFERS from either branch
    alone (proves the merge averages, not passes through).

    `best_rmse` is now also native: each branch's REFIT predicts the held-out TEST set (`fold_id=None`),
    and dag-ml's off-fold fusion handler averages those base test predictions per sample into one scored
    `(test, fold_id=None)` block under the merge node. Asserts `best_rmse` == a DIRECT avg-ensemble test
    baseline (refit each branch on FULL train, average their test predictions, score) within 1e-3.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error

    import nirs4all

    n_splits, n_comp, alpha = 3, 5, 1.0
    pipeline = [
        KFold(n_splits=n_splits, shuffle=True, random_state=42),
        {"branch": [[{"model": PLSRegression(n_components=n_comp)}], [{"model": Ridge(alpha=alpha)}]]},
        {"merge": "mean"},
    ]
    result = nirs4all.run(pipeline, dataset_path("regression"), engine="dag-ml")

    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    train = dataset.index_column("sample", {"partition": "train"})
    test = dataset.index_column("sample", {"partition": "test"})
    folds = [([train[i] for i in tr], [train[i] for i in va]) for tr, va in KFold(n_splits=n_splits, shuffle=True, random_state=42).split(train)]

    def branch_oof(model_factory) -> dict[int, float]:  # noqa: ANN001
        pred: dict[int, float] = {}
        for train_ints, val_ints in folds:
            model = model_factory()
            model.fit(np.asarray(dataset.x({"sample": train_ints}, layout="2d"), dtype=float), np.asarray(dataset.y({"sample": train_ints}), dtype=float))
            p = np.asarray(model.predict(np.asarray(dataset.x({"sample": val_ints}, layout="2d"), dtype=float))).ravel()
            for position, sample_int in enumerate(val_ints):
                pred[int(sample_int)] = float(p[position])
        return pred

    true: dict[int, float] = {}
    for _train_ints, val_ints in folds:
        yv = np.asarray(dataset.y({"sample": val_ints}), dtype=float).ravel()
        for position, sample_int in enumerate(val_ints):
            true[int(sample_int)] = float(yv[position])

    pls_oof = branch_oof(lambda: PLSRegression(n_components=n_comp))
    ridge_oof = branch_oof(lambda: Ridge(alpha=alpha))
    keys = sorted(true)
    rmse_pls = float(np.sqrt(mean_squared_error([true[k] for k in keys], [pls_oof[k] for k in keys])))
    rmse_ridge = float(np.sqrt(mean_squared_error([true[k] for k in keys], [ridge_oof[k] for k in keys])))
    rmse_avg = float(np.sqrt(mean_squared_error([true[k] for k in keys], [(pls_oof[k] + ridge_oof[k]) / 2 for k in keys])))

    # The fused OOF == the direct avg-ensemble OOF (the merge averages the branches' held-out predictions).
    assert abs(result.cv_best_score - rmse_avg) < 1e-3, (result.cv_best_score, rmse_avg)
    # And it DIFFERS from either branch alone — proving the merge averages, not passes one branch through.
    assert abs(result.cv_best_score - rmse_pls) > 1e-3, (result.cv_best_score, rmse_pls)
    assert abs(result.cv_best_score - rmse_ridge) > 1e-3, (result.cv_best_score, rmse_ridge)
    # The two branches genuinely differ, so averaging is a real reduction (not a degenerate no-op).
    assert abs(rmse_pls - rmse_ridge) > 1e-3, (rmse_pls, rmse_ridge)

    # best_rmse == a DIRECT avg-ensemble TEST baseline: refit each branch on the FULL train, average
    # their held-out test predictions per sample, score. dag-ml's off-fold fusion handler reassembles
    # exactly that from the branches' `(test, fold_id=None)` REFIT blocks.
    def test_pred(model_factory) -> np.ndarray:  # noqa: ANN001
        model = model_factory()
        model.fit(np.asarray(dataset.x({"sample": train}, layout="2d"), dtype=float), np.asarray(dataset.y({"sample": train}), dtype=float))
        return np.asarray(model.predict(np.asarray(dataset.x({"sample": test}, layout="2d"), dtype=float))).ravel()

    y_test = np.asarray(dataset.y({"sample": test}), dtype=float).ravel()
    avg_test = (test_pred(lambda: PLSRegression(n_components=n_comp)) + test_pred(lambda: Ridge(alpha=alpha))) / 2
    rmse_avg_test = float(np.sqrt(mean_squared_error(y_test, avg_test)))
    assert abs(result.best_rmse - rmse_avg_test) < 1e-3, (result.best_rmse, rmse_avg_test)


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_duplication_branch_unsupported_merge_fails_loud() -> None:
    """A duplication branch with a STACKING merge fails LOUD naming #10; proba-mean fusion fails loud too.

    A `{"branch": [[A], [B]]}` followed by `{"merge": "predictions"}` is STACKING (meta-model over branch
    OOF) — the next slice (#10), not this one — and must raise `NotImplementedError` naming #10 rather than
    silently averaging. A `proba_mean` fusion (classification probability average) has no proba blocks from
    the value-only process adapter, so it also fails loud rather than averaging class labels."""
    from sklearn.linear_model import Ridge

    import nirs4all

    stacking = [
        KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42),
        {"branch": [[{"model": PLSRegression(n_components=5)}], [{"model": Ridge(alpha=1.0)}]]},
        {"merge": "predictions"},
    ]
    with pytest.raises(NotImplementedError, match="#10"):
        nirs4all.run(stacking, dataset_path("regression"), engine="dag-ml")

    proba = [
        KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42),
        {"branch": [[{"model": PLSRegression(n_components=5)}], [{"model": Ridge(alpha=1.0)}]]},
        {"merge": {"predictions": "all", "aggregate": "proba_mean"}},
    ]
    with pytest.raises(NotImplementedError, match="proba"):
        nirs4all.run(proba, dataset_path("regression"), engine="dag-ml")


def test_stacking_branch_detection() -> None:
    """The stacking detector consumes ONLY the duplication-branch + predictions-merge + meta-model shape.

    `_detect_stacking_branch` returns `(branches, meta_learner)` for `{"branch": [[A], [B], …]}` (each
    sub-pipeline with a model) + `{"merge": "predictions"}` + a downstream `{"model": M}` whose M is a
    handled meta-learner (a default `MetaModel` wrapper → its wrapped sklearn model, or a plain sklearn
    estimator). Everything else (no/mis-ordered model, a fusion/concat/per-branch merge, a MetaModel
    carrying unhandled options, a top-level step beside the branch) returns None so the loud #10 path
    still guards it. The duplication detector must NOT claim the stacking shape. No CLI needed."""
    from sklearn.linear_model import Ridge

    from nirs4all.operators.models.meta import CoverageStrategy, MetaModel, StackingConfig, StackingLevel, TestAggregation
    from nirs4all.operators.transforms.scalers import StandardNormalVariate
    from nirs4all.pipeline.dagml.run_backend import _detect_duplication_branch, _detect_stacking_branch

    splitter = KFold(n_splits=3, shuffle=True, random_state=42)
    branch = {"branch": [[{"model": PLSRegression(n_components=5)}], [{"model": Ridge(alpha=1.0)}]]}

    # Handled: a plain downstream model is the meta-learner.
    plain = _detect_stacking_branch([splitter, branch, {"merge": "predictions"}, {"model": Ridge(alpha=0.7)}])
    assert plain is not None
    branches, meta_learner = plain
    assert len(branches) == 2 and isinstance(meta_learner, Ridge) and meta_learner.alpha == 0.7

    # Handled: a MetaModel wrapper → its wrapped sklearn model (with its params).
    wrapped = _detect_stacking_branch([splitter, branch, {"merge": "predictions"}, {"model": MetaModel(model=Ridge(alpha=0.3))}])
    assert wrapped is not None and isinstance(wrapped[1], Ridge) and wrapped[1].alpha == 0.3
    # AUTO and explicit LEVEL_1 are both handled.
    lvl1 = _detect_stacking_branch([splitter, branch, {"merge": "predictions"}, {"model": MetaModel(model=Ridge(), stacking_config=StackingConfig(level=StackingLevel.LEVEL_1))}])
    assert lvl1 is not None

    # Not handled (each falls through to the loud #10 path):
    assert _detect_stacking_branch([splitter, branch, {"merge": "predictions"}]) is None  # no downstream model
    assert _detect_stacking_branch([splitter, branch, {"model": Ridge()}, {"merge": "predictions"}]) is None  # model before merge
    assert _detect_stacking_branch([splitter, branch, {"merge": "mean"}, {"model": Ridge()}]) is None  # fusion, not stacking
    assert _detect_stacking_branch([splitter, branch, {"merge": "concat"}, {"model": Ridge()}]) is None  # concat reassembly
    assert _detect_stacking_branch([splitter, branch, {"merge": {"predictions": [{"branch": 0, "select": "best"}]}}, {"model": Ridge()}]) is None  # per-branch config
    # MetaModel carrying unhandled options.
    assert _detect_stacking_branch([splitter, branch, {"merge": "predictions"}, {"model": MetaModel(model=Ridge(), use_proba=True)}]) is None
    assert _detect_stacking_branch([splitter, branch, {"merge": "predictions"}, {"model": MetaModel(model=Ridge(), source_models=["PLS"])}]) is None
    assert _detect_stacking_branch([splitter, branch, {"merge": "predictions"}, {"model": MetaModel(model=Ridge(), stacking_config=StackingConfig(level=StackingLevel.LEVEL_2))}]) is None
    # A non-default StackingConfig field (other than level) is silently IGNORED by the lowering → fail loud.
    # test_aggregation is the load-bearing one (this slice cannot score test meta-features at all).
    assert _detect_stacking_branch([splitter, branch, {"merge": "predictions"}, {"model": MetaModel(model=Ridge(), stacking_config=StackingConfig(test_aggregation=TestAggregation.WEIGHTED_MEAN))}]) is None
    assert _detect_stacking_branch([splitter, branch, {"merge": "predictions"}, {"model": MetaModel(model=Ridge(), stacking_config=StackingConfig(coverage_strategy=CoverageStrategy.DROP_INCOMPLETE))}]) is None
    assert _detect_stacking_branch([splitter, branch, {"merge": "predictions"}, {"model": MetaModel(model=Ridge(), stacking_config=StackingConfig(min_coverage_ratio=0.5))}]) is None
    # A sibling param or a generator on the meta-model step is silently dropped by the bare-estimator
    # lowering (no _apply_model_params / native generation runs for it) → fail loud, not a silent mis-run.
    assert _detect_stacking_branch([splitter, branch, {"merge": "predictions"}, {"model": Ridge(), "alpha": 0.2}]) is None
    assert _detect_stacking_branch([splitter, branch, {"merge": "predictions"}, {"model": Ridge(), "alpha": {"_range_": [0.1, 1.0, 3]}}]) is None
    # A top-level transform beside the branch (would be silently dropped).
    assert _detect_stacking_branch([StandardNormalVariate(), splitter, branch, {"merge": "predictions"}, {"model": Ridge()}]) is None
    # A modelless base sub-pipeline (the base level needs a model to produce OOF).
    assert _detect_stacking_branch([splitter, {"branch": [[StandardNormalVariate()], [{"model": Ridge()}]]}, {"merge": "predictions"}, {"model": Ridge()}]) is None
    # The duplication (fusion) detector must NOT claim the stacking (predictions-merge) shape.
    assert _detect_duplication_branch([splitter, branch, {"merge": "predictions"}, {"model": Ridge()}]) is None


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_stacking_branch() -> None:
    """A duplication branch (2 base models) + {"merge": "predictions"} + a meta-model scores natively (#10).

    The native path runs each base branch model on the full fold data (held-out Validation OOF); the
    meta-node consumes the base branches' Validation OOF (Option A — leakage-safe, the requires_oof edge
    refuses any train block) and fits the meta-learner on the per-fold OOF meta-feature matrix. Asserts
    `cv_best_score` == a DIRECT sklearn stacking OOF (per fold: base models fit on fold-train, predict
    fold-val → meta-features; meta-model fit on that fold's assembled OOF, predict it; concat across folds;
    scored) within 1e-3, AND that the stacked score DIFFERS from either base branch alone (the meta-model
    genuinely combines).

    `best_rmse` is now also native: in REFIT dag-ml delivers each base producer's held-out TEST
    prediction to the meta-node as a SEPARATE off-fold input keyed `…oof:refit` (partition Test,
    `fold_id=None`), alongside the full Validation OOF the meta fits on. The refit meta-model predicts
    the test set from those base TEST meta-features and emits a scored `(test, fold_id=None)` block.
    Asserts `best_rmse` == a DIRECT sklearn stacking TEST baseline (refit each base on FULL train,
    predict test → meta-features; refit meta on the FULL OOF; predict test, score) within 1e-3.
    LEAKAGE: the meta-model is fit on Validation OOF ONLY; the test meta-features come from the base
    models' TEST predictions, never their OOF/train.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error

    import nirs4all

    n_splits, n_comp, base_alpha, meta_alpha = 3, 5, 1.0, 1.0
    pipeline = [
        KFold(n_splits=n_splits, shuffle=True, random_state=42),
        {"branch": [[{"model": PLSRegression(n_components=n_comp)}], [{"model": Ridge(alpha=base_alpha)}]]},
        {"merge": "predictions"},
        {"model": Ridge(alpha=meta_alpha)},
    ]
    result = nirs4all.run(pipeline, dataset_path("regression"), engine="dag-ml")

    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    train = dataset.index_column("sample", {"partition": "train"})
    test = dataset.index_column("sample", {"partition": "test"})
    folds = [([train[i] for i in tr], [train[i] for i in va]) for tr, va in KFold(n_splits=n_splits, shuffle=True, random_state=42).split(train)]

    def x(ints):  # noqa: ANN001, ANN202
        return np.asarray(dataset.x({"sample": ints}, layout="2d"), dtype=float)

    def y(ints):  # noqa: ANN001, ANN202
        return np.asarray(dataset.y({"sample": ints}), dtype=float).ravel()

    def base_oof(factory) -> dict[int, float]:  # noqa: ANN001
        oof: dict[int, float] = {}
        for train_ints, val_ints in folds:
            model = factory()
            model.fit(x(train_ints), y(train_ints))
            p = np.asarray(model.predict(x(val_ints))).ravel()
            for position, sample_int in enumerate(val_ints):
                oof[int(sample_int)] = float(p[position])
        return oof

    pls_oof = base_oof(lambda: PLSRegression(n_components=n_comp))
    ridge_oof = base_oof(lambda: Ridge(alpha=base_alpha))
    truth: dict[int, float] = {int(s): float(v) for _tr, va in folds for s, v in zip(va, y(va))}

    # DIRECT stacking OOF: per fold, fit the meta-model on that fold's base-OOF meta-features and predict
    # them (the per-fold delivery dag-ml's meta-node receives — only the fold's Validation OOF), concat.
    meta_oof: dict[int, float] = {}
    for _train_ints, val_ints in folds:
        ints = [int(s) for s in val_ints]
        x_meta = np.array([[pls_oof[s], ridge_oof[s]] for s in ints])
        y_meta = np.array([truth[s] for s in ints])
        mm = Ridge(alpha=meta_alpha).fit(x_meta, y_meta)
        p = mm.predict(x_meta)
        for position, sample_int in enumerate(ints):
            meta_oof[sample_int] = float(p[position])

    keys = sorted(truth)
    rmse_stack = float(np.sqrt(mean_squared_error([truth[k] for k in keys], [meta_oof[k] for k in keys])))
    rmse_pls = float(np.sqrt(mean_squared_error([truth[k] for k in keys], [pls_oof[k] for k in keys])))
    rmse_ridge = float(np.sqrt(mean_squared_error([truth[k] for k in keys], [ridge_oof[k] for k in keys])))

    # cv_best_score == the direct sklearn stacking OOF.
    assert abs(result.cv_best_score - rmse_stack) < 1e-3, (result.cv_best_score, rmse_stack)
    # And it DIFFERS from either base branch alone (the meta-model genuinely combines, not a passthrough).
    assert abs(result.cv_best_score - rmse_pls) > 1e-3, (result.cv_best_score, rmse_pls)
    assert abs(result.cv_best_score - rmse_ridge) > 1e-3, (result.cv_best_score, rmse_ridge)

    # best_rmse == a DIRECT sklearn stacking TEST baseline. The meta-model is fit on the FULL Validation
    # OOF (sorted-key column order: PLS then Ridge, matching the meta-node's deterministic producer
    # order); the test meta-features come from each base model REFIT on the FULL train predicting test.
    oof_keys = sorted(truth)
    meta = Ridge(alpha=meta_alpha).fit(np.array([[pls_oof[s], ridge_oof[s]] for s in oof_keys]), np.array([truth[s] for s in oof_keys]))
    pls_test = np.asarray(PLSRegression(n_components=n_comp).fit(x(train), y(train)).predict(x(test))).ravel()
    ridge_test = np.asarray(Ridge(alpha=base_alpha).fit(x(train), y(train)).predict(x(test))).ravel()
    stack_test = np.asarray(meta.predict(np.column_stack([pls_test, ridge_test]))).ravel()
    rmse_stack_test = float(np.sqrt(mean_squared_error(y(test), stack_test)))
    assert abs(result.best_rmse - rmse_stack_test) < 1e-3, (result.best_rmse, rmse_stack_test)


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_stacking_unsupported_config_fails_loud() -> None:
    """Stacking with an IGNORED MetaModel option or a sibling-param/generator meta step fails LOUD (#10).

    Both are silently-dropped-config gaps the dag-ml stacking lowering would otherwise ignore: a non-default
    `StackingConfig` field (e.g. `test_aggregation`, which this slice cannot honor at all — best_rmse is NaN)
    and a sibling param / generator on the bare meta-model step (the bare-estimator lowering never runs
    `_apply_model_params` / native generation for it). Each must raise `NotImplementedError` naming #10
    rather than run with the option silently dropped — the project's never-silently-drop-config discipline."""
    from sklearn.linear_model import Ridge

    import nirs4all
    from nirs4all.operators.models.meta import MetaModel, StackingConfig, TestAggregation

    branch = {"branch": [[{"model": PLSRegression(n_components=5)}], [{"model": Ridge(alpha=1.0)}]]}

    ignored_config = [
        KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42),
        branch,
        {"merge": "predictions"},
        {"model": MetaModel(model=Ridge(), stacking_config=StackingConfig(test_aggregation=TestAggregation.WEIGHTED_MEAN))},
    ]
    with pytest.raises(NotImplementedError, match="#10"):
        nirs4all.run(ignored_config, dataset_path("regression"), engine="dag-ml")

    sibling_param = [
        KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42),
        branch,
        {"merge": "predictions"},
        {"model": Ridge(), "alpha": 0.2},
    ]
    with pytest.raises(NotImplementedError, match="#10"):
        nirs4all.run(sibling_param, dataset_path("regression"), engine="dag-ml")

    swept_meta = [
        KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42),
        branch,
        {"merge": "predictions"},
        {"model": Ridge(), "alpha": {"_range_": [0.1, 1.0, 3]}},
    ]
    with pytest.raises(NotImplementedError, match="#10"):
        nirs4all.run(swept_meta, dataset_path("regression"), engine="dag-ml")


# ---------------------------------------------------------------------------
# MULTI-TARGET y emission (nirs4all-migration #30, slice S0): the host un-ravels the y side of the
# 4-file seam to emit a (sample, target) matrix (target_numeric_matrix) instead of collapsing y to one
# column. dag-ml is already representation-agnostic + multi-target-aware (PredictionBlock /
# RegressionMetricReport are width-aware and emit per-target rmse:yK + a macro-mean), so this is
# HOST-ONLY — no engine change. nirs4all has no multi-target sample_data fixture, so the test
# SYNTHESIZES a multi-target SpectroDataset (3 targets: y0, y0*2+1, cos(y0)) in memory and hands it to
# the adapter via the pickle path (N4A_DAGML_DATASET_PICKLE) exactly like sample_augmentation does.
# ---------------------------------------------------------------------------


def _multi_target_dataset(n_targets: int = 3):
    """A 3-target SpectroDataset built from the regression corpus (X reused, y = [y0, 2*y0+1, cos(y0)]).

    The second target is an affine function of the first, so its per-target RMSE must be ~2x the first's
    — a built-in check that dag-ml scores each target column independently (not a blended number)."""
    from nirs4all.data.dataset import SpectroDataset

    base = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    train = [int(s) for s in base.index_column("sample", {"partition": "train"})]
    test = [int(s) for s in base.index_column("sample", {"partition": "test"})]

    def _xy(ids: list[int]):
        x = np.asarray(base.x_rows(ids, layout="2d"), dtype=float)
        y_block = np.asarray(base.y({"sample": ids}), dtype=float).reshape(len(ids), -1)
        stored = base.index_column("sample", {"sample": ids})
        row_of = {int(s): r for r, s in enumerate(stored)}
        y0 = y_block[[row_of[int(s)] for s in ids], 0]
        cols = [y0, y0 * 2.0 + 1.0, np.cos(y0)][:n_targets]
        return x, np.column_stack(cols)

    x_train, y_train = _xy(train)
    x_test, y_test = _xy(test)
    dataset = SpectroDataset("multi_target_regression")
    headers = [str(i) for i in range(x_train.shape[1])]
    dataset.add_samples(x_train, {"partition": "train"}, headers=headers, header_unit="nm")
    dataset.add_samples(x_test, {"partition": "test"}, headers=headers, header_unit="nm")
    dataset.add_targets(np.vstack([y_train, y_test]))
    return dataset


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_multi_target(tmp_path) -> None:
    """A PLSRegression(n_targets=3) pipeline on engine="dag-ml" runs natively with per-target OOF +
    per-target scoring (S0): dag-ml emits rmse:y0/rmse:y1/rmse:y2 + a macro-mean rmse, each matching a
    hand-computed per-column RMSE within 1e-3, and the macro-mean == the mean of the per-target RMSEs.

    PARITY BASELINE = direct sklearn KFold per-target OOF on the same folds (the engine path is the
    correction: the legacy .ravel() collapsed y to one column). Drives SELECT off the default macro-mean
    metric — per-target SELECT (`select on rmse:y2`) is GAP C / slice S1 and out of scope here.
    """
    import pickle

    import dag_ml
    from sklearn.metrics import mean_squared_error

    from nirs4all.pipeline.dagml.cli_runner import assemble_cv_refit_dsl, run_cv_refit_bundle
    from nirs4all.pipeline.dagml.envelope import build_envelope, num_targets
    from nirs4all.pipeline.dagml.identity import mint_identity
    from nirs4all.pipeline.dagml_bridge import controller_manifests

    n_targets = 3
    dataset = _multi_target_dataset(n_targets)
    assert num_targets(dataset) == n_targets, "fixture must be genuinely multi-target"

    identity = mint_identity(dataset)
    train = dataset.index_column("sample", {"partition": "train"})
    folds = [([train[j] for j in tr], [train[j] for j in va]) for tr, va in KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42).split(train)]
    envelope = build_envelope(dataset, identity, sample_ints=train)
    pipeline = [{"model": PLSRegression(n_components=5)}]
    dsl = assemble_cv_refit_dsl(pipeline, identity, envelope, folds, dsl_id="multi_target", n_splits=_N_SPLITS)
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()

    # Hand the exact in-memory multi-target dataset to the adapter via the pickle path (the regression
    # corpus on disk is single-target; the synthetic y columns live only in memory).
    (tmp_path / "multi_target.pkl").write_bytes(pickle.dumps(dataset))
    outcome = run_cv_refit_bundle(
        dsl=dsl, envelope=envelope, graph=graph, dataset_path="UNUSED", workdir=tmp_path,
        dagml_cli=str(_DAGML_CLI), venv_python=sys.executable, dataset_pickle=str(tmp_path / "multi_target.pkl"),
    )
    assert outcome["returncode"] == 0, outcome["stdout"][-2000:]

    # dag-ml's native cross-fold OOF average report carries per-target + macro-mean keys.
    reports = json.loads((tmp_path / "bundle.json").read_text())["scores"]["reports"]
    avg = [r for r in reports if r["partition"] == "validation" and r["fold_id"] == "avg"]
    assert len(avg) == 1, "dag-ml must emit one native cross-fold OOF average for the multi-target run"
    metrics = avg[0]["metrics"]
    assert avg[0]["target_width"] == n_targets, "the scored block must keep the (sample, target) width"
    assert avg[0]["target_names"] == [f"y{j}" for j in range(n_targets)]

    # Hand-computed per-target OOF RMSE (direct sklearn KFold, same folds, same row-keying as the engine).
    oof_pred: dict[int, np.ndarray] = {}
    oof_true: dict[int, np.ndarray] = {}
    for train_ints, val_ints in folds:
        model = PLSRegression(n_components=5)
        model.fit(np.asarray(dataset.x({"sample": train_ints}, layout="2d"), dtype=float), np.asarray(dataset.y({"sample": train_ints}), dtype=float))
        pred = np.asarray(model.predict(np.asarray(dataset.x({"sample": val_ints}, layout="2d"), dtype=float)), dtype=float).reshape(len(val_ints), -1)
        true_block = np.asarray(dataset.y({"sample": val_ints}), dtype=float).reshape(len(val_ints), -1)
        stored = dataset.index_column("sample", {"sample": val_ints})
        row_of = {int(s): r for r, s in enumerate(stored)}
        true = true_block[[row_of[int(s)] for s in val_ints]]
        for position, sample_int in enumerate(val_ints):
            oof_pred[int(sample_int)] = pred[position]
            oof_true[int(sample_int)] = true[position]
    keys = sorted(oof_pred)
    pred_matrix = np.array([oof_pred[k] for k in keys])
    true_matrix = np.array([oof_true[k] for k in keys])
    per_target = [float(np.sqrt(mean_squared_error(true_matrix[:, j], pred_matrix[:, j]))) for j in range(n_targets)]
    macro = float(np.mean(per_target))

    # Per-target parity: dag-ml's rmse:yK == hand-computed per-column RMSE; macro-mean == mean of them.
    for j in range(n_targets):
        assert abs(metrics[f"rmse:y{j}"] - per_target[j]) < 1e-3, (f"rmse:y{j}", metrics[f"rmse:y{j}"], per_target[j])
    assert abs(metrics["rmse"] - macro) < 1e-3, (metrics["rmse"], macro)
    # dag-ml's own macro-mean is exactly the mean of its per-target keys (no blended-column collapse).
    assert abs(metrics["rmse"] - float(np.mean([metrics[f"rmse:y{j}"] for j in range(n_targets)]))) < 1e-9
    # The affine target (y1 = 2*y0 + 1) must score ~2x the first target — proof the columns are scored
    # independently, not collapsed into one number.
    assert abs(metrics["rmse:y1"] - 2.0 * metrics["rmse:y0"]) < 1e-2, (metrics["rmse:y1"], metrics["rmse:y0"])


def test_multi_target_emission_single_target_unchanged() -> None:
    """The S0 multi-target seam keeps the SINGLE-target path BYTE-IDENTICAL: the schema target
    representation is the legacy ``tabular_numeric`` literal and ``resolve_targets`` still returns a flat
    list (so every existing single-target parity test above is unaffected). No CLI needed."""
    from nirs4all.pipeline.dagml.envelope import _target_representation, num_targets
    from nirs4all.pipeline.dagml.identity import mint_identity
    from nirs4all.pipeline.dagml.resolver import MaterializationResolver

    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    assert num_targets(dataset) == 1

    # The single-target representation is the exact legacy literal (so schema_fingerprint is unchanged).
    legacy = {
        "id": "tabular_numeric", "type_id": "table", "rank": 2,
        "axes": [
            {"name": "sample", "kind": "sample", "unit": None, "size": 130, "variable": False},
            {"name": "target", "kind": "target", "unit": None, "size": 1, "variable": False},
        ],
        "container": "dataframe", "dtype": "float64", "sparse": False, "ragged": False,
    }
    assert _target_representation(dataset, 130) == legacy

    # resolve_targets returns a FLAT list for single-target (the legacy .ravel() shape).
    identity = mint_identity(dataset)
    resolver = MaterializationResolver(dataset, identity)
    wire = [identity.to_wire(int(s)) for s in dataset.index_column("sample", {"partition": "train"})[:5]]
    values = resolver.resolve_targets(wire)["values"]
    assert values and all(not isinstance(v, list) for v in values), "single-target must stay a flat list"


def test_multi_target_emission_widens_target_axis() -> None:
    """A multi-target dataset takes the NEW path: target_numeric_matrix with a target axis of width
    n_targets, and ``resolve_targets`` returns list-of-rows (the (n, n_targets) block). No CLI needed."""
    from nirs4all.pipeline.dagml.envelope import _target_representation, num_targets
    from nirs4all.pipeline.dagml.identity import mint_identity
    from nirs4all.pipeline.dagml.resolver import MaterializationResolver

    dataset = _multi_target_dataset(3)
    assert num_targets(dataset) == 3

    rep = _target_representation(dataset, 130)
    assert rep["id"] == "target_numeric_matrix", "regression multi-target → numeric matrix representation"
    assert rep["type_id"] == "target" and rep["container"] == "array" and rep["dtype"] == "float64"
    target_axis = next(axis for axis in rep["axes"] if axis["kind"] == "target")
    assert target_axis["size"] == 3, "the target axis must widen to n_targets"

    # resolve_targets returns the (n, n_targets) block as list-of-rows.
    identity = mint_identity(dataset)
    resolver = MaterializationResolver(dataset, identity)
    train = dataset.index_column("sample", {"partition": "train"})[:5]
    wire = [identity.to_wire(int(s)) for s in train]
    values = resolver.resolve_targets(wire)["values"]
    assert len(values) == 5 and all(isinstance(row, list) and len(row) == 3 for row in values), "multi-target → list of 3-wide rows"


def test_multi_source_emission_single_source_unchanged() -> None:
    """The S3 multi-source seam keeps the SINGLE-source path BYTE-IDENTICAL: the schema emits one
    ``src0`` ``signal_1d`` source, the plan stays ``materialize → adapt → join → tabular_numeric``, the
    relations carry ``source_id='src0'``, and the model binding stays ``tabular_numeric`` /
    ``source_ids=['src0']`` (so every single-source parity test above is unaffected). No CLI needed."""
    from nirs4all.pipeline.dagml.cli_runner import data_bindings_for, model_node_id
    from nirs4all.pipeline.dagml.envelope import build_envelope, sample_relations, source_ids
    from nirs4all.pipeline.dagml.identity import mint_identity

    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    assert dataset.features_sources() == 1
    assert source_ids(dataset) == ["src0"]

    identity = mint_identity(dataset)
    train = dataset.index_column("sample", {"partition": "train"})
    envelope = build_envelope(dataset, identity, sample_ints=train)

    # The plan is the legacy single-source shape (output_representation tabular_numeric).
    assert envelope["plan"]["output_representation"] == "tabular_numeric"
    assert [step["kind"] for step in envelope["plan"]["steps"]] == ["materialize", "adapt", "join"]
    assert [step["source_id"] for step in envelope["plan"]["steps"]] == ["src0", "src0", None]
    # Single-source relations keep source_id='src0' (BYTE-IDENTICAL).
    assert all(row["source_id"] == "src0" for row in sample_relations(identity, sample_ints=train)["rows"])

    # The model binding stays tabular_numeric / ['src0'].
    binding = data_bindings_for(model_node_id([{"model": PLSRegression(n_components=5)}]), envelope)[0]
    assert binding["output_representation"] == "tabular_numeric"
    assert binding["source_ids"] == ["src0"]
    assert binding["feature_set_id"] == "x"


def test_multi_source_emission_emits_feature_block_set() -> None:
    """A multi-source dataset takes the NEW early-fusion path: the schema declares one ``signal_1d``
    source per ``FeatureSource`` (per-source feature count), the plan joins them to ``feature_block_set``,
    the relations are sample-grain (``source_id=None``), and the model binding carries
    ``output_representation='feature_block_set'`` + every ``source_id``. No CLI needed."""
    from nirs4all.pipeline.dagml.cli_runner import data_bindings_for, model_node_id
    from nirs4all.pipeline.dagml.envelope import build_envelope, sample_relations, source_ids
    from nirs4all.pipeline.dagml.identity import mint_identity

    dataset = DatasetConfigs(dataset_path("multi")).get_dataset_at(0)
    n_sources = dataset.features_sources()
    assert n_sources == 3, "the `multi` corpus has 3 NIR sources"
    assert source_ids(dataset) == ["src0", "src1", "src2"]

    identity = mint_identity(dataset)
    train = dataset.index_column("sample", {"partition": "train"})
    envelope = build_envelope(dataset, identity, sample_ints=train)

    # The schema declares one signal_1d source per FeatureSource, each with its own feature count.
    per_source_features = dataset._feature_accessor.num_features  # noqa: SLF001 - per-source feature widths
    schema_sources = envelope["plan"]["steps"]  # plan materialize steps mirror the schema sources
    materialized = [step["source_id"] for step in schema_sources if step["kind"] == "materialize"]
    assert materialized == ["src0", "src1", "src2"]

    # The plan joins the per-source blocks into a feature_block_set for early fusion.
    assert envelope["plan"]["output_representation"] == "feature_block_set"
    join_step = next(step for step in envelope["plan"]["steps"] if step["kind"] == "join")
    assert join_step["output_representation"] == "feature_block_set"
    assert join_step["input_representation"] == "signal_1d"
    assert join_step["metadata"]["inputs"] == ["src:src0", "src:src1", "src:src2"]

    # Multi-source relations are sample-grain: source_id=None (a sample's blocks span every source).
    assert all(row["source_id"] is None for row in sample_relations(identity, source_id=None, sample_ints=train)["rows"])

    # The model binding carries feature_block_set + every source id.
    binding = data_bindings_for(model_node_id([{"model": PLSRegression(n_components=5)}]), envelope)[0]
    assert binding["output_representation"] == "feature_block_set"
    assert binding["source_ids"] == ["src0", "src1", "src2"]
    assert binding["feature_set_id"] == "x"
    # The fused feature width is the sum of the per-source widths (early-fusion concat by sample_id).
    assert isinstance(per_source_features, list) and len(per_source_features) == n_sources


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_engine_dagml_multi_source_early_fusion() -> None:
    """engine="dag-ml" runs a MULTI-SOURCE pipeline as native early fusion: the engine fuses the N
    per-source blocks by sample_id (host-side, identity-keyed) and the model sees the fused matrix.

    SNV → PLS on the 3-source `multi` corpus. ``cv_best_score`` + ``best_rmse`` == direct sklearn on
    the host-concatenated (early-fused) multi-source matrix, within 1e-3 — proving the engine fusion
    by identity gives the SAME matrix as the host's ``concat_source=True`` hstack (the rewiring is
    behavior-preserving). LEAKAGE: folds/OOF partition over SAMPLES; a sample's per-source blocks all
    land on the same fold side (the fusion is feature-axis, identity-keyed — no cross-sample mixing)."""
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline

    import nirs4all
    from nirs4all.operators.transforms.scalers import StandardNormalVariate

    pipeline = [StandardNormalVariate(), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}]
    result = nirs4all.run(pipeline, dataset_path("multi"), engine="dag-ml")

    dataset = DatasetConfigs(dataset_path("multi")).get_dataset_at(0)
    assert dataset.features_sources() == 3, "this exercises the >1-source early-fusion path"
    train = dataset.index_column("sample", {"partition": "train"})
    test_ints = dataset.index_column("sample", {"partition": "test"})
    folds = [([train[i] for i in tr], [train[i] for i in va]) for tr, va in KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42).split(train)]

    # concat_source=True IS the host-fused (sample-aligned, feature-axis-concatenated) multi-source matrix.
    def fused_x(sample_ints: list[int]) -> np.ndarray:
        return np.asarray(dataset.x({"sample": sample_ints}, layout="2d", concat_source=True), dtype=float)

    def y(sample_ints: list[int]) -> np.ndarray:
        return np.asarray(dataset.y({"sample": sample_ints}), dtype=float)

    # best_rmse == sklearn final-test on the fused matrix (refit on full train, predict test).
    final = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=5))
    final.fit(fused_x(train), y(train))
    sklearn_final_test = float(np.sqrt(mean_squared_error(y(test_ints).ravel(), np.asarray(final.predict(fused_x(test_ints))).ravel())))
    assert abs(result.best_rmse - sklearn_final_test) < 1e-3, (result.best_rmse, sklearn_final_test)

    # cv_best_score == sklearn OOF-concat on the fused matrix.
    oof_pred: dict[int, float] = {}
    oof_true: dict[int, float] = {}
    for train_ints, val_ints in folds:
        fold_model = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=5))
        fold_model.fit(fused_x(train_ints), y(train_ints))
        pred = np.asarray(fold_model.predict(fused_x(val_ints))).ravel()
        true = y(val_ints).ravel()
        for position, sample_int in enumerate(val_ints):
            oof_pred[sample_int], oof_true[sample_int] = float(pred[position]), float(true[position])
    keys = sorted(oof_pred)
    sklearn_oof = float(np.sqrt(mean_squared_error([oof_true[k] for k in keys], [oof_pred[k] for k in keys])))
    assert abs(result.cv_best_score - sklearn_oof) < 1e-3, (result.cv_best_score, sklearn_oof)
