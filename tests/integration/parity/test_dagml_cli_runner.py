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

    `best_rmse` is NOT asserted: the native concat-merge handler scores the FIT_CV cross-fold OOF (the
    separation branch's primary result) but does not reassemble the per-partition REFIT test
    predictions, so `best_rmse` is NaN — which also matches legacy (top-level best_rmse is NaN for a
    branch+merge pipeline). cv_best_score is the score a separation branch is meant to produce.
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
    group_of = {int(s): str(v) for s, v in zip(train, dataset.metadata_column("group", {"partition": "train"}), strict=True)}
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


def test_run_via_dagml_stateful_augmentation_fails_loud() -> None:
    """STATEFUL/SUPERVISED/BALANCED augmentation is REJECTED (NotImplementedError), never silently run.

    The first slice augments ONCE globally before folds exist — leakage-free ONLY for stateless
    per-sample augmenters. A stateful augmenter (mixup with stored neighbors, a global-mean scatter
    reference) or the balanced/supervised mode fits on the whole train (future fold-val included), so
    it must fall through to the bridge's raw `sample_augmentation` NotImplementedError (fail-loud).
    Fold-local augmentation to support these leakage-safely is a follow-up slice. No CLI needed — the
    rejection happens before any dag-ml call. Stateless GaussianNoise stays native (the parity test).
    """
    import nirs4all
    from nirs4all.operators.augmentation import GaussianAdditiveNoise
    from nirs4all.operators.augmentation.spectral import LocalMixupAugmenter, ScatterSimulationMSC
    from nirs4all.operators.transforms.scalers import StandardNormalVariate
    from nirs4all.pipeline.dagml.run_backend import _augmentation_is_leakage_free, _operator_is_stateless

    # Operator-level signal: stateless augmenters pass, stateful ones (learn data state in fit) are flagged.
    assert _operator_is_stateless(GaussianAdditiveNoise(sigma=0.01))
    assert not _operator_is_stateless(LocalMixupAugmenter())  # stores X_fit_ neighbors
    assert not _operator_is_stateless(ScatterSimulationMSC(reference_mode="global_mean"))  # stores global_mean_

    # Step-level: balanced/supervised mode and any stateful transformer are NOT leakage-free.
    assert _augmentation_is_leakage_free({"sample_augmentation": {"transformers": [GaussianAdditiveNoise(sigma=0.01)], "count": 1, "selection": "all"}})
    assert not _augmentation_is_leakage_free({"sample_augmentation": {"transformers": [GaussianAdditiveNoise(sigma=0.01)], "balance": "y", "max_factor": 2}})
    assert not _augmentation_is_leakage_free({"sample_augmentation": {"transformers": [LocalMixupAugmenter()], "count": 1}})
    assert not _augmentation_is_leakage_free({"sample_augmentation": {"transformers": [GaussianAdditiveNoise(sigma=0.01), LocalMixupAugmenter()], "count": 1}})

    # End-to-end: a stateful augmentation pipeline raises NotImplementedError naming sample_augmentation.
    stateful_pipelines = {
        "mixup": [StandardNormalVariate(), {"sample_augmentation": {"transformers": [LocalMixupAugmenter()], "count": 1, "selection": "all", "random_state": 42}}, KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}],
        "balanced": [StandardNormalVariate(), {"sample_augmentation": {"transformers": [GaussianAdditiveNoise(sigma=0.01)], "balance": "y", "max_factor": 2, "random_state": 42}}, KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}],
    }
    for label, pipeline in stateful_pipelines.items():
        with pytest.raises(NotImplementedError, match="sample_augmentation"):
            nirs4all.run(pipeline, dataset_path("regression"), engine="dag-ml")
        assert label  # name surfaced in the failure if the raise is missing


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
