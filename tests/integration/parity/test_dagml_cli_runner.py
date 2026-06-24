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


def test_generation_kind_routes_conservatively() -> None:
    """The native router is CONSERVATIVE: only a clean `_range_` model sweep goes native; every other
    generator shape (mixed, finetune_params, _grid_, _log_range_, non-model sweep, multi-model) falls
    back to the Python `expand_spec` path, so no generator is ever silently dropped. No CLI needed."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import MinMaxScaler

    from nirs4all.operators.transforms.scalers import StandardNormalVariate
    from nirs4all.pipeline.dagml.run_backend import _generation_kind

    splitter = KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42)

    # The ONLY native case: a pure `_range_` sweep on a model step.
    assert _generation_kind([StandardNormalVariate(), splitter, {"model": PLSRegression(), "n_components": {"_range_": [3, 9, 3]}}]) == "param_model"
    # No generators at all → none.
    assert _generation_kind([StandardNormalVariate(), splitter, {"model": PLSRegression(n_components=5)}]) == "none"

    # Everything below must route to the Python path ("operator") — never native — or a generator
    # would be silently dropped / mis-expanded.
    mixed = [StandardNormalVariate(), {"y_processing": MinMaxScaler(), "feature_range": {"_range_": [0, 1, 1]}}, splitter, {"model": PLSRegression(), "n_components": {"_range_": [3, 9, 3]}}]
    assert _generation_kind(mixed) == "operator"  # a sweep on a non-model step alongside the model sweep
    assert _generation_kind([splitter, {"model": PLSRegression(), "n_components": {"_range_": [3, 9, 3]}, "finetune_params": {"n_trials": 5}}]) == "operator"  # finetune_params
    assert _generation_kind([splitter, {"model": PLSRegression(), "n_components": {"_range_": [3, 9, 3]}, "train_params": {"epochs": 1}}]) == "operator"  # train_params
    assert _generation_kind([splitter, {"model": PLSRegression(), "n_components": {"_grid_": {"n_components": [5, 10]}}}]) == "operator"  # _grid_ (not proven equivalent)
    assert _generation_kind([splitter, {"model": Ridge(), "alpha": {"_log_range_": [0.001, 10.0, 4]}}]) == "operator"  # _log_range_ (fingerprint not round-tripping)
    assert _generation_kind([splitter, {"model": PLSRegression(), "n_components": {"_range_": [3, 16, 1], "count": 3}}]) == "operator"  # modifier-bearing range
    assert _generation_kind([splitter, {"model": PLSRegression(), "n_components": {"_range_": {"from": 3, "to": 9}}}]) == "operator"  # dict-form range
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
