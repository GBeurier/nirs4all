"""Resumable benchmark runner for AOM_v0.

Reads a cohort CSV, runs every requested AOM/POP variant on every dataset and
seed, and appends one row per (dataset, seed, variant) immediately to the
results CSV. Already-completed rows are skipped on resume.

Variants implemented (all numpy unless suffixed `-torch`):

- `PLS-standard-numpy`
- `AOM-global-nipals-materialized-numpy`
- `AOM-global-nipals-adjoint-numpy`
- `POP-nipals-materialized-numpy`
- `POP-nipals-adjoint-numpy`
- `AOM-global-simpls-materialized-numpy`
- `AOM-global-simpls-covariance-numpy`
- `POP-simpls-materialized-numpy`
- `POP-simpls-covariance-numpy`
- `Superblock-simpls-numpy`
- `AOM-soft-simpls-covariance-numpy` (experimental)
- Torch equivalents `*-torch` for `nipals_adjoint`, `simpls_covariance`,
  `superblock_simpls`.

Classification variants:

- `PLS-DA-standard`
- `AOM-PLS-DA-global-nipals-adjoint`
- `POP-PLS-DA-nipals-adjoint`
- `AOM-PLS-DA-global-simpls-covariance`
- `POP-PLS-DA-simpls-covariance`
- `AOM-PLS-DA-global-simpls-covariance-torch`
- `POP-PLS-DA-simpls-covariance-torch`

Each row in the output CSV includes the master schema columns plus the
AOM-specific diagnostic columns documented in `BENCHMARK_PROTOCOL.md`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PYPATH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PYPATH_ROOT)

from aompls.classification import AOMPLSDAClassifier, POPPLSDAClassifier  # noqa: E402
from aompls.estimators import AOMPLSRegressor, POPPLSRegressor  # noqa: E402
from aompls.metrics import (  # noqa: E402
    balanced_accuracy,
    expected_calibration_error,
    log_loss,
    macro_f1,
    mae,
    r2,
    rmse,
)


REGRESSION_VARIANTS = [
    {"label": "PLS-standard-numpy", "kind": "regression", "selection": "none", "engine": "pls_standard", "operator_bank": "identity", "backend": "numpy"},
    # AOM_v0 with the production-equivalent default bank (~77 operators)
    {"label": "AOM-global-nipals-adjoint-numpy", "kind": "regression", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "default", "backend": "numpy"},
    {"label": "AOM-global-simpls-covariance-numpy", "kind": "regression", "selection": "global", "engine": "simpls_covariance", "operator_bank": "default", "backend": "numpy"},
    # AOM_v0 with the compact bank (POP-style 9 ops) — sanity comparison
    {"label": "AOM-compact-simpls-covariance-numpy", "kind": "regression", "selection": "global", "engine": "simpls_covariance", "operator_bank": "compact", "backend": "numpy"},
    # POP variants on the compact bank (matches production)
    {"label": "POP-nipals-adjoint-numpy", "kind": "regression", "selection": "per_component", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy"},
    {"label": "POP-simpls-covariance-numpy", "kind": "regression", "selection": "per_component", "engine": "simpls_covariance", "operator_bank": "compact", "backend": "numpy"},
    # Materialised references (slow) — for parity validation only
    {"label": "AOM-global-nipals-materialized-numpy", "kind": "regression", "selection": "global", "engine": "nipals_materialized", "operator_bank": "default", "backend": "numpy"},
    {"label": "AOM-global-simpls-materialized-numpy", "kind": "regression", "selection": "global", "engine": "simpls_materialized", "operator_bank": "default", "backend": "numpy"},
    {"label": "POP-nipals-materialized-numpy", "kind": "regression", "selection": "per_component", "engine": "nipals_materialized", "operator_bank": "compact", "backend": "numpy"},
    {"label": "POP-simpls-materialized-numpy", "kind": "regression", "selection": "per_component", "engine": "simpls_materialized", "operator_bank": "compact", "backend": "numpy"},
    # Superblock and soft (experimental)
    {"label": "Superblock-simpls-numpy", "kind": "regression", "selection": "superblock", "engine": "simpls_covariance", "operator_bank": "compact", "backend": "numpy"},
    {"label": "AOM-soft-simpls-covariance-numpy", "kind": "regression", "selection": "soft", "engine": "simpls_covariance", "operator_bank": "compact", "backend": "numpy", "experimental": True},
    # Production nirs4all baselines (call-through; not AOM_v0 code)
    {"label": "nirs4all-AOM-PLS-default", "kind": "regression", "selection": "external", "engine": "nirs4all_aom", "operator_bank": "production_default", "backend": "numpy"},
    {"label": "nirs4all-POP-PLS-default", "kind": "regression", "selection": "external", "engine": "nirs4all_pop", "operator_bank": "production_compact", "backend": "numpy"},
]

CLASSIFICATION_VARIANTS = [
    {"label": "PLS-DA-standard", "kind": "classification", "selection": "none", "engine": "pls_standard", "operator_bank": "identity", "backend": "numpy"},
    {"label": "AOM-PLS-DA-global-nipals-adjoint", "kind": "classification", "selection": "global", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy"},
    {"label": "POP-PLS-DA-nipals-adjoint", "kind": "classification", "selection": "per_component", "engine": "nipals_adjoint", "operator_bank": "compact", "backend": "numpy"},
    {"label": "AOM-PLS-DA-global-simpls-covariance", "kind": "classification", "selection": "global", "engine": "simpls_covariance", "operator_bank": "compact", "backend": "numpy"},
    {"label": "POP-PLS-DA-simpls-covariance", "kind": "classification", "selection": "per_component", "engine": "simpls_covariance", "operator_bank": "compact", "backend": "numpy"},
]


RESULT_COLUMNS = [
    # master_results schema (subset relevant to AOM/POP)
    "database_name", "dataset", "task", "model", "result_label", "result_dir", "status",
    "status_details", "preprocessing_pipeline", "RMSECV", "RMSE_MF", "RMSEP", "MAE_test",
    "r2_test", "search_mean_score", "seed", "n_splits", "best_config_json",
    "best_model_params_json", "best_fold_scores_json", "trial_values_json",
    "search_results_path", "best_config_path", "final_predictions_path",
    "fold_predictions_path", "rmse_mf_source", "artifact_best_config_format",
    "artifact_search_results_format", "artifact_final_predictions_format",
    # AOM extras
    "aom_variant", "backend", "engine", "selection", "criterion", "orthogonalization",
    "operator_bank", "selected_operator_sequence_json", "selected_operator_scores_json",
    "n_components_selected", "max_components", "fit_time_s", "predict_time_s",
    "delta_rmsep_vs_master_pls", "delta_rmsep_vs_tabpfn_raw", "delta_rmsep_vs_tabpfn_opt",
    "run_seed", "code_version", "notes",
    # classification extras
    "balanced_accuracy", "macro_f1", "log_loss", "ece",
]


def _extract_diagnostics(est, variant: Dict, max_components: int) -> Dict[str, object]:
    """Return a uniform diagnostics dict for both AOM_v0 and external estimators."""
    if hasattr(est, "get_diagnostics"):
        return est.get_diagnostics()
    # Production nirs4all estimators expose different attributes.
    n_components_ = getattr(est, "n_components_", None) or getattr(est, "n_components", max_components)
    selected_idx: List = []
    selected_names: List = []
    if hasattr(est, "selected_operator_idx_"):
        selected_idx = [int(est.selected_operator_idx_)]
    elif hasattr(est, "selected_operator_idxs_"):
        selected_idx = [int(i) for i in est.selected_operator_idxs_]
    if hasattr(est, "selected_operator_names_"):
        selected_names = list(est.selected_operator_names_)
    elif hasattr(est, "Gamma") and est.Gamma is not None:
        Gamma = np.asarray(est.Gamma)
        if Gamma.ndim == 2 and Gamma.size > 0:
            top = np.argmax(Gamma, axis=1)
            selected_idx = [int(t) for t in top]
    return {
        "engine": variant.get("engine", "external"),
        "selection": variant.get("selection", "external"),
        "criterion": "internal_holdout",
        "orthogonalization": "production",
        "operator_bank": variant.get("operator_bank", "production"),
        "selected_operator_indices": selected_idx,
        "selected_operator_names": selected_names,
        "operator_scores": {},
        "n_components_selected": int(n_components_) if n_components_ else max_components,
        "max_components": max_components,
    }


def _coerce_numeric(df: pd.DataFrame) -> np.ndarray:
    """Coerce a possibly mixed-decimal DataFrame to a float ndarray.

    Some CSV files use US decimals (`0.123`) for most values and European
    decimals (`1,23E-04`) for scientific notation. We replace commas with
    dots in object columns before casting.
    """
    try:
        return df.to_numpy(dtype=float)
    except ValueError:
        cleaned = df.copy()
        for col in cleaned.columns:
            if cleaned[col].dtype == object:
                cleaned[col] = (
                    cleaned[col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .astype(float)
                )
        return cleaned.to_numpy(dtype=float)


def _load_csv_array(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep=";")
    return _coerce_numeric(df)


def _load_csv_target(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:
        col = df.columns[0]
        try:
            return df[col].astype(float).to_numpy()
        except (ValueError, TypeError):
            return df[col].to_numpy()
    return df.iloc[:, 0].to_numpy()


def _existing_keys(results_path: Path) -> set:
    if not results_path.exists():
        return set()
    df = pd.read_csv(results_path, dtype=str)
    if df.empty:
        return set()
    keys = set()
    for _, row in df.iterrows():
        keys.add((row.get("database_name", ""), row.get("dataset", ""), row.get("model", ""), row.get("seed", "")))
    return keys


def _append_row(results_path: Path, row: Dict[str, object]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_path.exists()
    with results_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


class _PreprocAOMWrapper:
    """Wrap a non-linear / supervised preprocessing pipeline around an AOM estimator.

    The wrapper fits the pre-processor on the training set, transforms `X`
    once, then fits the inner estimator on the pre-processed `X`. At
    predict time, the same pre-processor is replayed on the test `X`
    before delegating to the inner estimator. Used to expose
    SNV+AOM-PLS, MSC+AOM-PLS, OSC+AOM-PLS, and combinations as benchmark
    variants without modifying the AOM core.
    """

    def __init__(self, preproc, estimator) -> None:
        self.preproc = preproc
        self.estimator = estimator

    def fit(self, X, y):
        try:
            self.preproc.fit(X, y)
        except TypeError:
            self.preproc.fit(X)
        Xt = self.preproc.transform(X)
        self.estimator.fit(Xt, y)
        return self

    def predict(self, X):
        Xt = self.preproc.transform(X)
        return self.estimator.predict(Xt)

    def get_diagnostics(self):
        if hasattr(self.estimator, "get_diagnostics"):
            d = self.estimator.get_diagnostics()
            d["preproc"] = type(self.preproc).__name__
            return d
        return {"preproc": type(self.preproc).__name__}

    @property
    def n_components_(self):
        return getattr(self.estimator, "n_components_", None)

    @property
    def selected_operators_(self):
        return getattr(self.estimator, "selected_operators_", [])

    @property
    def selected_operator_indices_(self):
        return getattr(self.estimator, "selected_operator_indices_", [])

    @property
    def operator_scores_(self):
        return getattr(self.estimator, "operator_scores_", {})

    @property
    def diagnostics_(self):
        return getattr(self.estimator, "diagnostics_", None)


def _build_preproc(name: str):
    """Resolve a preproc shorthand to an instance.

    Supports two shorthand formats:

    Legacy single-step shorthands (kept for backward compatibility):
        ``snv``, ``msc``, ``osc``, ``snv_osc``, ``msc_osc``

    Composable triplet ``<norm>+<baseline>+<osc>`` (paper-style grid):
        norm    in ``{none, snv, msc, emsc1, emsc2}``
        baseline in ``{none, asls}``
        osc     in ``{none, osc1, osc2, osc3}``

    For example, ``snv+asls+osc2`` returns a `PreprocessingPipeline`
    that applies SNV, then ASLSBaseline, then OSC(n_components=2).
    Empty steps (`none`) are skipped. The output preserves the
    sklearn-compatible `fit(X, y) / transform(X)` interface used by
    `_PreprocAOMWrapper`.
    """
    from aompls.preprocessing import (
        StandardNormalVariate,
        MultiplicativeScatterCorrection,
        OrthogonalSignalCorrection,
        PreprocessingPipeline,
        ExtendedMSC,
        ASLSBaseline,
        LocalSNV,
    )
    if name == "snv":
        return StandardNormalVariate()
    if name == "msc":
        return MultiplicativeScatterCorrection()
    if name == "osc":
        return OrthogonalSignalCorrection(n_components=2)
    if name == "snv_osc":
        return PreprocessingPipeline([StandardNormalVariate(), OrthogonalSignalCorrection(n_components=2)])
    if name == "msc_osc":
        return PreprocessingPipeline([MultiplicativeScatterCorrection(), OrthogonalSignalCorrection(n_components=2)])
    if name == "lsnv31":
        return LocalSNV(window=31)
    if name == "lsnv51":
        return LocalSNV(window=51)
    if name == "lsnv101":
        return LocalSNV(window=101)
    if "+" in name:
        norm, baseline, osc = name.split("+")
        steps = []
        if norm == "snv":
            steps.append(StandardNormalVariate())
        elif norm == "msc":
            steps.append(MultiplicativeScatterCorrection())
        elif norm == "emsc1":
            steps.append(ExtendedMSC(degree=1))
        elif norm == "emsc2":
            steps.append(ExtendedMSC(degree=2))
        elif norm != "none":
            raise ValueError(f"unknown norm: {norm!r}")
        if baseline == "asls":
            steps.append(ASLSBaseline())
        elif baseline != "none":
            raise ValueError(f"unknown baseline: {baseline!r}")
        if osc == "osc1":
            steps.append(OrthogonalSignalCorrection(n_components=1))
        elif osc == "osc2":
            steps.append(OrthogonalSignalCorrection(n_components=2))
        elif osc == "osc3":
            steps.append(OrthogonalSignalCorrection(n_components=3))
        elif osc != "none":
            raise ValueError(f"unknown osc: {osc!r}")
        if len(steps) == 1:
            return steps[0]
        return PreprocessingPipeline(steps)
    raise ValueError(f"unknown preproc: {name!r}")


def _build_estimator(variant: Dict, criterion: str, max_components: int, cv: int, seed: int, X=None, y=None):
    kind = variant["kind"]
    selection = variant["selection"]
    engine = variant["engine"]
    bank = variant["operator_bank"]
    backend = variant["backend"]
    preproc_name = variant.get("preproc")
    # If a preproc is requested, wrap whatever inner estimator we'd build.
    if preproc_name:
        inner_variant = {k: v for k, v in variant.items() if k != "preproc"}
        inner = _build_estimator(inner_variant, criterion, max_components, cv, seed, X, y)
        return _PreprocAOMWrapper(_build_preproc(preproc_name), inner)
    # Override criterion / cv / repeats / one_se / max_components from the
    # variant dict so a single config row can express e.g. "cv-5 with
    # 3-repeats and one-SE" or "POP at K=5".
    variant_criterion = variant.get("criterion_override", criterion)
    variant_cv = int(variant.get("cv_override", cv))
    variant_repeats = int(variant.get("repeats_override", 1))
    variant_one_se = bool(variant.get("one_se_override", False))
    variant_kmax = int(variant.get("max_components_override", max_components))
    # Use variant_kmax in place of the global max_components from here on.
    max_components = variant_kmax
    # External: call-through to the production nirs4all estimators.
    if selection == "external":
        if engine == "nirs4all_aom":
            from nirs4all.operators.models.sklearn.aom_pls import AOMPLSRegressor as _NirsAOM
            return _NirsAOM(n_components=max_components, gate="hard")
        if engine == "nirs4all_pop":
            from nirs4all.operators.models.sklearn.pop_pls import POPPLSRegressor as _NirsPOP
            return _NirsPOP(n_components=max_components, auto_select=True)
        raise ValueError(f"unknown external engine: {engine}")
    # Operator explorer — build active bank from training data, then run AOM-global on top.
    if selection == "explorer_global":
        from aompls.operator_explorer import build_active_bank_from_training
        active = build_active_bank_from_training(
            X, y, max_degree=2, beam_width=24, final_top_m=20, cosine_threshold=0.98
        )
        return AOMPLSRegressor(
            n_components="auto",
            max_components=max_components,
            engine=engine,
            selection="global",
            criterion=variant_criterion,
            operator_bank=active,
            cv=variant_cv, repeats=variant_repeats, one_se_rule=variant_one_se,
            random_state=seed,
            backend=backend,
        )
    if selection == "active_superblock":
        cls = AOMPLSRegressor if kind == "regression" else AOMPLSDAClassifier
        return cls(
            n_components="auto",
            max_components=max_components,
            engine=engine,
            selection="active_superblock",
            criterion=variant_criterion,
            operator_bank=bank,
            cv=variant_cv, repeats=variant_repeats, one_se_rule=variant_one_se,
            random_state=seed,
            backend=backend,
        )
    if kind == "regression":
        cls = POPPLSRegressor if selection == "per_component" else AOMPLSRegressor
    else:
        cls = POPPLSDAClassifier if selection == "per_component" else AOMPLSDAClassifier
    if selection == "soft":
        return cls(
            n_components="auto",
            max_components=max_components,
            engine=engine,
            selection="soft",
            criterion=variant_criterion,
            operator_bank=bank,
            cv=variant_cv, repeats=variant_repeats, one_se_rule=variant_one_se,
            random_state=seed,
            backend=backend,
        )
    if selection == "superblock":
        return cls(
            n_components="auto",
            max_components=max_components,
            engine=engine,
            selection="superblock",
            criterion=variant_criterion,
            operator_bank=bank,
            cv=variant_cv, repeats=variant_repeats, one_se_rule=variant_one_se,
            random_state=seed,
            backend=backend,
        )
    if selection == "none":
        return cls(
            n_components="auto",
            max_components=max_components,
            engine=engine,
            selection="none",
            criterion=variant_criterion,
            operator_bank=bank,
            cv=variant_cv, repeats=variant_repeats, one_se_rule=variant_one_se,
            random_state=seed,
            backend=backend,
        )
    return cls(
        n_components="auto",
        max_components=max_components,
        engine=engine,
        selection=selection,
        criterion=variant_criterion,
        operator_bank=bank,
        cv=variant_cv, repeats=variant_repeats, one_se_rule=variant_one_se,
        random_state=seed,
        backend=backend,
    )


def _build_hpo_estimator(variant: Dict, params: Dict, max_components_default: int, cv_default: int, seed: int):
    """Construct an AOM-PLS estimator from an Optuna trial's params dict.

    Reuses the variant dict for the bank/selection/engine and overrides only
    the HPO-tuned fields (norm preproc, asls hyperparameters, cv, max_components,
    one_se_rule). Returns an estimator that wraps a preproc pipeline if needed.
    """
    norm = params.get("norm", "none")
    cv_ = int(params.get("cv", cv_default))
    max_comp = int(params.get("max_components", max_components_default))
    one_se = bool(params.get("one_se_rule", False))
    inner = AOMPLSRegressor(
        n_components="auto",
        max_components=max_comp,
        engine=variant["engine"],
        selection="global",
        criterion="cv",
        operator_bank=variant["operator_bank"],
        cv=cv_,
        random_state=seed,
        backend=variant["backend"],
        repeats=1,
        one_se_rule=one_se,
    )
    if norm == "none":
        return inner
    from aompls.preprocessing import (
        StandardNormalVariate, MultiplicativeScatterCorrection,
    )
    if norm == "snv":
        pp = StandardNormalVariate()
    elif norm == "msc":
        pp = MultiplicativeScatterCorrection()
    elif norm == "asls":
        from nirs4all.operators.transforms.nirs import ASLSBaseline as _ASLS
        pp = _ASLS(
            lam=float(params.get("asls_lambda", 1e6)),
            p=float(params.get("asls_p", 0.01)),
            max_iter=50, tol=1e-3,
        )
    else:
        return inner
    return _PreprocAOMWrapper(pp, inner)


def _run_hpo_search(variant: Dict, Xtr, ytr, max_components: int, cv: int, seed: int,
                     n_trials: int = 25):
    """Run an Optuna HPO search and return (best_params, total_search_time_s)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rng = np.random.RandomState(seed)
    n = Xtr.shape[0]
    perm = rng.permutation(n)
    n_val = max(3, n // 5)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    X_in, y_in = Xtr[train_idx], ytr[train_idx]
    X_va, y_va = Xtr[val_idx], ytr[val_idx]

    def objective(trial):
        norm = trial.suggest_categorical("norm", ["none", "snv", "msc", "asls"])
        params = {"norm": norm}
        if norm == "asls":
            params["asls_lambda"] = trial.suggest_float("asls_lambda", 1e3, 1e9, log=True)
            params["asls_p"] = trial.suggest_float("asls_p", 1e-3, 0.5, log=True)
        params["cv"] = trial.suggest_categorical("cv", [3, 5])
        params["max_components"] = trial.suggest_categorical("max_components", [10, 15, 20])
        params["one_se_rule"] = trial.suggest_categorical("one_se_rule", [False, True])
        try:
            est = _build_hpo_estimator(variant, params, max_components, cv, seed)
            est.fit(X_in, y_in)
            pred = est.predict(X_va)
            if isinstance(pred, np.ndarray) and pred.ndim == 2 and pred.shape[1] == 1:
                pred = pred.ravel()
            return float(np.sqrt(np.mean((y_va.ravel() - pred.ravel()) ** 2)))
        except Exception:
            return float("inf")

    sampler = optuna.samplers.TPESampler(seed=int(seed))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False, gc_after_trial=False)
    search_time = time.perf_counter() - t0
    return study.best_params, search_time


def _run_regression_variant(
    variant: Dict,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    seed: int,
    cohort_row: pd.Series,
    criterion: str,
    max_components: int,
    cv: int,
) -> Dict[str, object]:
    if variant.get("hpo"):
        best_params, search_time = _run_hpo_search(
            variant, Xtr, ytr, max_components, cv, seed,
            n_trials=int(variant.get("hpo_trials", 25)),
        )
        est = _build_hpo_estimator(variant, best_params, max_components, cv, seed)
        t0 = time.perf_counter()
        est.fit(Xtr, ytr)
        final_fit_time = time.perf_counter() - t0
        fit_time = float(search_time + final_fit_time)
        t1 = time.perf_counter()
        yhat = est.predict(Xte)
        if isinstance(yhat, np.ndarray) and yhat.ndim == 2 and yhat.shape[1] == 1:
            yhat = yhat.ravel()
        predict_time = time.perf_counter() - t1
        rmsep = rmse(yte, yhat)
        diag = _extract_diagnostics(est, variant, max_components)
        diag["hpo_best_params"] = best_params
        diag["hpo_search_time_s"] = float(search_time)
        diag["hpo_final_fit_time_s"] = float(final_fit_time)
    else:
        est = _build_estimator(variant, criterion=criterion, max_components=max_components, cv=cv, seed=seed, X=Xtr, y=ytr)
        t0 = time.perf_counter()
        est.fit(Xtr, ytr)
        fit_time = time.perf_counter() - t0
        t1 = time.perf_counter()
        yhat = est.predict(Xte)
        if isinstance(yhat, np.ndarray) and yhat.ndim == 2 and yhat.shape[1] == 1:
            yhat = yhat.ravel()
        predict_time = time.perf_counter() - t1
        rmsep = rmse(yte, yhat)
        diag = _extract_diagnostics(est, variant, max_components)
    ref_pls = cohort_row.get("ref_rmse_pls")
    ref_tabraw = cohort_row.get("ref_rmse_tabpfn_raw")
    ref_tabopt = cohort_row.get("ref_rmse_tabpfn_opt")
    return {
        "database_name": cohort_row["database_name"],
        "dataset": cohort_row["dataset"],
        "task": "regression",
        "model": variant["label"],
        "result_label": variant["label"],
        "result_dir": "",
        "status": "ok",
        "status_details": "",
        "preprocessing_pipeline": diag.get("operator_bank", "compact"),
        "RMSECV": "",
        "RMSE_MF": "",
        "RMSEP": rmsep,
        "MAE_test": mae(yte, yhat),
        "r2_test": r2(yte, yhat),
        "search_mean_score": "",
        "seed": seed,
        "n_splits": cv,
        "best_config_json": "",
        "best_model_params_json": "",
        "best_fold_scores_json": "",
        "trial_values_json": "",
        "search_results_path": "",
        "best_config_path": "",
        "final_predictions_path": "",
        "fold_predictions_path": "",
        "rmse_mf_source": "",
        "artifact_best_config_format": "",
        "artifact_search_results_format": "",
        "artifact_final_predictions_format": "",
        "aom_variant": variant["label"],
        "backend": variant["backend"],
        "engine": variant["engine"],
        "selection": variant["selection"],
        "criterion": criterion,
        "orthogonalization": diag.get("orthogonalization", ""),
        "operator_bank": variant["operator_bank"],
        "selected_operator_sequence_json": json.dumps(diag.get("selected_operator_indices", [])),
        "selected_operator_scores_json": json.dumps(diag.get("operator_scores", {})),
        "n_components_selected": diag.get("n_components_selected", 0),
        "max_components": diag.get("max_components", max_components),
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
        "delta_rmsep_vs_master_pls": (rmsep - float(ref_pls)) if pd.notna(ref_pls) else "",
        "delta_rmsep_vs_tabpfn_raw": (rmsep - float(ref_tabraw)) if pd.notna(ref_tabraw) else "",
        "delta_rmsep_vs_tabpfn_opt": (rmsep - float(ref_tabopt)) if pd.notna(ref_tabopt) else "",
        "run_seed": seed,
        "code_version": "AOM_v0/0.1.0",
        "notes": ("hpo:" + json.dumps(diag.get("hpo_best_params", {}))) if variant.get("hpo") else ("experimental" if variant.get("experimental") else ""),
        "balanced_accuracy": "",
        "macro_f1": "",
        "log_loss": "",
        "ece": "",
    }


def _run_classification_variant(
    variant: Dict,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    seed: int,
    cohort_row: pd.Series,
    criterion: str,
    max_components: int,
    cv: int,
) -> Dict[str, object]:
    est = _build_estimator(variant, criterion=criterion, max_components=max_components, cv=cv, seed=seed)
    t0 = time.perf_counter()
    est.fit(Xtr, ytr)
    fit_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    pred = est.predict(Xte)
    proba = est.predict_proba(Xte)
    predict_time = time.perf_counter() - t1
    diag = est.get_diagnostics()
    return {
        "database_name": cohort_row["database_name"],
        "dataset": cohort_row["dataset"],
        "task": "classification",
        "model": variant["label"],
        "result_label": variant["label"],
        "result_dir": "",
        "status": "ok",
        "status_details": "",
        "preprocessing_pipeline": diag.get("operator_bank", "compact"),
        "RMSECV": "",
        "RMSE_MF": "",
        "RMSEP": "",
        "MAE_test": "",
        "r2_test": "",
        "search_mean_score": "",
        "seed": seed,
        "n_splits": cv,
        "best_config_json": "",
        "best_model_params_json": "",
        "best_fold_scores_json": "",
        "trial_values_json": "",
        "search_results_path": "",
        "best_config_path": "",
        "final_predictions_path": "",
        "fold_predictions_path": "",
        "rmse_mf_source": "",
        "artifact_best_config_format": "",
        "artifact_search_results_format": "",
        "artifact_final_predictions_format": "",
        "aom_variant": variant["label"],
        "backend": variant["backend"],
        "engine": variant["engine"],
        "selection": variant["selection"],
        "criterion": criterion,
        "orthogonalization": diag.get("orthogonalization", ""),
        "operator_bank": variant["operator_bank"],
        "selected_operator_sequence_json": json.dumps(diag.get("selected_operator_indices", [])),
        "selected_operator_scores_json": json.dumps(diag.get("operator_scores", {})),
        "n_components_selected": diag.get("n_components_selected", 0),
        "max_components": diag.get("max_components", max_components),
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
        "delta_rmsep_vs_master_pls": "",
        "delta_rmsep_vs_tabpfn_raw": "",
        "delta_rmsep_vs_tabpfn_opt": "",
        "run_seed": seed,
        "code_version": "AOM_v0/0.1.0",
        "notes": "",
        "balanced_accuracy": balanced_accuracy(yte, pred),
        "macro_f1": macro_f1(yte, pred),
        "log_loss": log_loss(yte, proba, classes=est.classes_),
        "ece": expected_calibration_error(yte, proba, n_bins=10),
    }


def _resolve_target(y: np.ndarray) -> Tuple[np.ndarray, str]:
    """Decide whether the loaded y is regression or classification."""
    if y.dtype.kind in ("i",):
        return y.astype(int), "int"
    if y.dtype.kind in ("f",):
        return y.astype(float), "float"
    # String labels: classification
    return y, "str"


def run_dataset(
    cohort_row: pd.Series,
    variants: List[Dict],
    results_path: Path,
    seeds: Sequence[int],
    criterion: str,
    max_components: int,
    cv: int,
    classification: bool,
    existing_keys: set,
) -> int:
    if cohort_row["status"] != "ok":
        return 0
    Xtr = _load_csv_array(cohort_row["train_path"])
    Xte = _load_csv_array(cohort_row["test_path"])
    ytr = _load_csv_target(cohort_row["ytrain_path"])
    yte = _load_csv_target(cohort_row["ytest_path"])
    if not classification:
        ytr = ytr.astype(float)
        yte = yte.astype(float)
    else:
        # Encode classification labels once so they live in {0, ..., C-1}.
        # Without encoding, ECE/log-loss helpers can be off by a class
        # alignment (Codex code review, HIGH #6).
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        all_labels = np.concatenate([np.asarray(ytr).ravel(), np.asarray(yte).ravel()])
        le.fit(all_labels)
        ytr = le.transform(np.asarray(ytr).ravel()).astype(int)
        yte = le.transform(np.asarray(yte).ravel()).astype(int)
    n_runs = 0
    for variant in variants:
        for seed in seeds:
            key = (str(cohort_row["database_name"]), str(cohort_row["dataset"]), str(variant["label"]), str(seed))
            if key in existing_keys:
                continue
            try:
                if classification:
                    row = _run_classification_variant(
                        variant, Xtr, ytr, Xte, yte, seed, cohort_row, criterion, max_components, cv
                    )
                else:
                    row = _run_regression_variant(
                        variant, Xtr, ytr, Xte, yte, seed, cohort_row, criterion, max_components, cv
                    )
                _append_row(results_path, row)
                existing_keys.add(key)
                n_runs += 1
            except Exception as exc:  # pragma: no cover
                row = {col: "" for col in RESULT_COLUMNS}
                row.update(
                    {
                        "database_name": cohort_row["database_name"],
                        "dataset": cohort_row["dataset"],
                        "task": "regression" if not classification else "classification",
                        "model": variant["label"],
                        "result_label": variant["label"],
                        "status": "skipped",
                        "status_details": str(exc)[:200],
                        "aom_variant": variant["label"],
                        "engine": variant["engine"],
                        "selection": variant["selection"],
                        "operator_bank": variant["operator_bank"],
                        "backend": variant["backend"],
                        "seed": seed,
                        "run_seed": seed,
                        "code_version": "AOM_v0/0.1.0",
                        "notes": "exception during run",
                    }
                )
                _append_row(results_path, row)
                existing_keys.add(key)
    return n_runs


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the AOM_v0 benchmark")
    parser.add_argument("--task", choices=("regression", "classification"), default="regression")
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--master", default="bench/tabpfn_paper/master_results.csv")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--seeds", default="0", help="Comma-separated list of seeds")
    parser.add_argument("--n-jobs", type=int, default=1, help="Currently sequential; reserved for parallelism")
    parser.add_argument("--criterion", default="cv")
    parser.add_argument("--max-components", type=int, default=15)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on the number of datasets")
    parser.add_argument("--variants", default="", help="Comma-separated variant labels to run; empty = all")
    args = parser.parse_args(argv)
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    results_path = workspace / "results.csv"
    cohort_df = pd.read_csv(args.cohort)
    if args.limit > 0:
        cohort_df = cohort_df.head(args.limit)
    if args.task == "regression":
        variants = REGRESSION_VARIANTS
    else:
        variants = CLASSIFICATION_VARIANTS
    if args.variants:
        wanted = set([v.strip() for v in args.variants.split(",") if v.strip()])
        variants = [v for v in variants if v["label"] in wanted]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    existing = _existing_keys(results_path)
    total = 0
    for _, cohort_row in cohort_df.iterrows():
        n = run_dataset(
            cohort_row,
            variants,
            results_path,
            seeds,
            args.criterion,
            args.max_components,
            args.cv,
            args.task == "classification",
            existing,
        )
        total += n
        sys.stdout.write(f"[{cohort_row['database_name']}/{cohort_row['dataset']}] +{n} rows\n")
        sys.stdout.flush()
    print(f"completed {total} new rows -> {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
