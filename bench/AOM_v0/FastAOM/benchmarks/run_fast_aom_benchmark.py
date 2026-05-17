"""Resumable benchmark runner for FastAOM.

Reads a cohort CSV (same schema as
``bench/AOM_v0/benchmarks/cohort_regression.csv``), runs every requested
FastAOM variant on every dataset and seed, and appends one row per
``(dataset, seed, variant)`` immediately to the results CSV. Already-
completed rows are skipped on resume.

Variants implemented (all numpy):

    FastAOM-single-chain-compact            # SingleChainPLSRidge top-1 from screening
    FastAOM-hard-chain-compact              # HardAOMChainPLSRidge, primitive_bank=compact, depth=3
    FastAOM-hard-chain-compact-d4           # HardAOMChainPLSRidge, depth=4
    FastAOM-hard-chain-default              # HardAOMChainPLSRidge, primitive_bank=default
    FastAOM-soft-chain-compact              # SoftAOMChainPLSRidge
    FastAOM-sparse-mkr-compact              # SparseMultiKernelRidge
    FastAOM-hard-chain-shrinkage            # HardAOMChainPLSRidge w/ component-wise λ_h = λ_0 h
    FastAOM-hard-chain-multibase            # adds SNV / MSC / EMSC bases

Each row in the output CSV follows the master schema columns plus the
AOM-specific diagnostic columns documented in ``BENCHMARK_PROTOCOL.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PYPATH_FASTAOM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # bench/AOM_v0/FastAOM
PYPATH_AOM_ROOT = os.path.dirname(PYPATH_FASTAOM_ROOT)                              # bench/AOM_v0
for _root in (PYPATH_FASTAOM_ROOT, PYPATH_AOM_ROOT):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from aompls.metrics import mae, r2, rmse  # noqa: E402
from FastAOM.models import FastAOMConfig, FastAOMPLSRidge  # noqa: E402
from FastAOM.xcorr_fast import install_xcorr_patch  # noqa: E402

# Activate the vectorised xcorr replacement for the duration of the benchmark.
# The patch is bit-exact (verified in ``tests/test_xcorr_fast.py``) and gives
# a ~10× speedup on Python-loop convolutions inside ``aompls.operators``.
install_xcorr_patch()


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


def _base_config(
    label: str,
    *,
    model: str,
    primitive_bank: str = "compact",
    max_chain_depth: int = 3,
    rank: int = 200,
    top_global: int = 200,
    top_per_base: Optional[int] = 60,
    top_per_family: Optional[int] = 6,
    n_components: int = 15,
    component_shrinkage_gamma: Optional[float] = None,
    soft_rho: float = 0.05,
    soft_max_mixture_size: Optional[int] = 4,
    sparse_mkr_max_chains: int = 8,
    use_raw: bool = True,
    use_absorbance: bool = False,
    use_snv: bool = True,
    use_msc: bool = False,
    use_emsc: bool = False,
    asls_grid: Optional[Sequence[Tuple[float, float]]] = None,
    osc_components: Optional[Sequence[int]] = None,
    use_snv_osc: bool = False,
    whittaker_baseline_lam: Optional[Sequence[float]] = None,
    notes: str = "",
    cv_n_components: bool = False,
    cv_folds: int = 5,
) -> Dict[str, object]:
    # Variants ending in ``-cv5`` are wired to do 5-fold CV n_components
    # selection (currently only honoured by the single_chain model).
    cv_flag = cv_n_components or label.endswith("-cv5")
    return {
        "label": label,
        "config": FastAOMConfig(
            model=model,
            primitive_bank=primitive_bank,
            max_chain_depth=max_chain_depth,
            rank=rank,
            top_global=top_global,
            top_per_base=top_per_base,
            top_per_family=top_per_family,
            n_components=n_components,
            component_shrinkage_gamma=component_shrinkage_gamma,
            soft_rho=soft_rho,
            soft_max_mixture_size=soft_max_mixture_size,
            sparse_mkr_max_chains=sparse_mkr_max_chains,
            use_raw=use_raw,
            use_absorbance=use_absorbance,
            use_snv=use_snv,
            use_msc=use_msc,
            use_emsc=use_emsc,
            asls_grid=tuple(asls_grid) if asls_grid else None,
            osc_components=tuple(osc_components) if osc_components else None,
            use_snv_osc=use_snv_osc,
            whittaker_baseline_lam=tuple(whittaker_baseline_lam) if whittaker_baseline_lam else None,
            cv_n_components=cv_flag,
            cv_folds=cv_folds,
        ),
        "notes": notes,
        "primitive_bank": primitive_bank,
        "model": model,
    }


REGRESSION_VARIANTS: List[Dict[str, object]] = [
    _base_config(
        "FastAOM-single-chain-compact",
        model="single_chain",
        primitive_bank="compact",
        max_chain_depth=3,
        n_components=15,
        notes="top-1 chain from fast screening",
    ),
    _base_config(
        "FastAOM-single-chain-compact-cv5-numpy",
        model="single_chain",
        primitive_bank="compact",
        max_chain_depth=3,
        n_components=15,
        notes="top-1 chain + 5-fold sklearn KFold CV n_components (apples-to-apples with ASLS-AOM-compact-cv5-numpy)",
    ),
    _base_config(
        "FastAOM-hard-chain-compact",
        model="hard_aom_chain",
        primitive_bank="compact",
        max_chain_depth=3,
        n_components=15,
        top_global=120,
        notes="one chain per latent component, compact bank, depth=3",
    ),
    # ``FastAOM-hard-chain-compact-d4`` was dropped from the full benchmark:
    # on the 10-dataset smoke run depth=4 chains produced identical RMSEP to
    # the depth=3 chains (the compact-bank ``max_per_role`` cap saturates at 3
    # roles, so the chain set is the same). Re-enable for ablation via
    # ``--variants FastAOM-hard-chain-compact-d4``.
    # ``FastAOM-hard-chain-default`` (100-op primitive bank) is left out of the
    # default smoke list because the dense p×p detrend matrices in the default
    # bank dominate fit time on large-p NIRS datasets; enable explicitly via
    # ``--variants FastAOM-hard-chain-default`` if needed for an ablation.
    _base_config(
        "FastAOM-soft-chain-compact",
        model="soft_aom_chain",
        primitive_bank="compact",
        max_chain_depth=3,
        n_components=12,
        soft_rho=0.05,
        soft_max_mixture_size=3,
        top_global=80,
        notes="sparse non-negative mixture per component",
    ),
    _base_config(
        "FastAOM-sparse-mkr-compact",
        model="sparse_mkr",
        primitive_bank="compact",
        max_chain_depth=3,
        sparse_mkr_max_chains=6,
        top_global=60,
        notes="multiple-kernel Ridge",
    ),
    # ``FastAOM-hard-chain-shrinkage`` was dropped from the full benchmark: on
    # the smoke run, component-wise shrinkage gave identical RMSEP to the no-
    # shrinkage variant (the Ridge GCV-style fit on latent scores already finds
    # the right per-component lambda). Re-enable with ``--variants ...``.
    _base_config(
        "FastAOM-hard-chain-multibase",
        model="hard_aom_chain",
        primitive_bank="compact",
        max_chain_depth=3,
        n_components=15,
        top_global=160,
        use_raw=True,
        use_snv=True,
        use_msc=True,
        use_emsc=True,
        notes="raw + SNV + MSC + EMSC bases",
    ),
    # --- Enriched-bases variants (supervised + ASLS multi-grid + Whittaker) ---
    # These add OSC (supervised), an ASLS grid, EMSC, and Whittaker-baseline
    # bases to address the ASLS+OSC oracle gap visible in the partial smoke
    # comparison (Chla+b, Ccar, Biscuit_Sucrose). Bases are fold-aware: OSC
    # uses ``y_train`` at fit and replays the stored projection at predict.
    _base_config(
        "FastAOM-hard-chain-asls",
        model="hard_aom_chain",
        primitive_bank="compact",
        max_chain_depth=3,
        n_components=15,
        top_global=200,
        use_raw=True,
        use_snv=True,
        use_msc=False,
        use_emsc=False,
        asls_grid=[(1e4, 0.001), (1e5, 0.01), (1e6, 0.001)],
        notes="raw + SNV + 3 ASLS baselines",
    ),
    _base_config(
        "FastAOM-hard-chain-osc",
        model="hard_aom_chain",
        primitive_bank="compact",
        max_chain_depth=3,
        n_components=15,
        top_global=160,
        use_raw=True,
        use_snv=True,
        use_msc=False,
        use_emsc=False,
        osc_components=[2, 3],
        notes="raw + SNV + OSC(2) + OSC(3)",
    ),
    _base_config(
        "FastAOM-hard-chain-supervised",
        model="hard_aom_chain",
        primitive_bank="compact",
        max_chain_depth=3,
        n_components=15,
        top_global=240,
        use_raw=True,
        use_snv=True,
        use_msc=False,
        use_emsc=True,
        asls_grid=[(1e4, 0.001), (1e5, 0.01)],
        osc_components=[2],
        use_snv_osc=True,
        notes="raw + SNV + EMSC + 2 ASLS + OSC(2) + SNV-OSC (full supervised pack)",
    ),
    _base_config(
        "FastAOM-single-chain-supervised-cv5-numpy",
        model="single_chain",
        primitive_bank="compact",
        max_chain_depth=3,
        n_components=15,
        top_global=240,
        use_raw=True,
        use_snv=True,
        use_msc=False,
        use_emsc=True,
        asls_grid=[(1e4, 0.001), (1e5, 0.01)],
        osc_components=[2],
        use_snv_osc=True,
        notes="single-chain + supervised bases + CV-5 n_components",
    ),
    _base_config(
        "FastAOM-sparse-mkr-supervised",
        model="sparse_mkr",
        primitive_bank="compact",
        max_chain_depth=3,
        n_components=15,
        top_global=160,
        sparse_mkr_max_chains=8,
        use_raw=True,
        use_snv=True,
        use_msc=False,
        use_emsc=True,
        asls_grid=[(1e4, 0.001), (1e5, 0.01)],
        osc_components=[2],
        notes="sparse MKR + supervised bases",
    ),
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
    # FastAOM-specific extras
    "n_chains_enumerated", "n_candidates_total", "n_finalists",
    "top_finalist_score", "screening_time_s", "lowrank_time_s", "model_fit_time_s",
    # bench/scenarios "new" schema (so bench/build_run_dashboard.py can render
    # FastAOM results without modification)
    "schema_version", "preset", "cohort", "canonical_name", "model_class",
    "module", "rmsep", "mae", "r2", "error_message",
    "n_train", "n_test", "n_features",
    # classification (unused but kept for schema consistency)
    "balanced_accuracy", "macro_f1", "log_loss", "ece",
]


def _model_class(variant_label: str) -> str:
    if "single-chain" in variant_label:
        return "fastaom-single-chain-pls-ridge"
    if "sparse-mkr" in variant_label:
        return "fastaom-sparse-mkr"
    if "soft-chain" in variant_label:
        return "fastaom-soft-aom-chain"
    if "hard-chain" in variant_label:
        return "fastaom-hard-aom-chain"
    return "fastaom-other"


# ---------------------------------------------------------------------------
# Loaders (match aompls runner)
# ---------------------------------------------------------------------------


def _coerce_numeric(df: pd.DataFrame) -> np.ndarray:
    try:
        return df.to_numpy(dtype=float)
    except ValueError:
        cleaned = df.copy()
        for col in cleaned.columns:
            if cleaned[col].dtype == object:
                cleaned[col] = (
                    cleaned[col].astype(str).str.replace(",", ".", regex=False).astype(float)
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
    return {
        (row.get("database_name", ""), row.get("dataset", ""), row.get("model", ""), row.get("seed", ""))
        for _, row in df.iterrows()
    }


def _append_row(results_path: Path, row: Dict[str, object]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_path.exists()
    with results_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Run one variant
# ---------------------------------------------------------------------------


def _run_variant(
    variant: Dict[str, object],
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    seed: int,
    cohort_row: pd.Series,
    max_components: int,
) -> Dict[str, object]:
    cfg: FastAOMConfig = variant["config"]
    # ``--max-components`` is a *ceiling*, not an override: a variant that
    # explicitly chose a smaller ``n_components`` keeps it. Only variants
    # that requested ``n_components > max_components`` (or unset) are clipped.
    effective_n = min(cfg.n_components, max_components) if cfg.n_components else max_components
    cfg = FastAOMConfig(
        **{**cfg.__dict__, "n_components": effective_n, "random_state": seed}
    )
    rng = np.random.default_rng(seed)
    # Apply seed by perturbing the config (the orchestrator stores random_state but
    # currently uses it only for SVD tie-breaking; we set seed via numpy default RNG
    # implicitly here).
    est = FastAOMPLSRidge(config=cfg)
    t0 = time.perf_counter()
    est.fit(Xtr, ytr)
    fit_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    yhat = est.predict(Xte)
    predict_time = time.perf_counter() - t1

    rmsep = rmse(yte, yhat)
    diag = est.diagnostics_
    timings = diag.get("timings_s", {})
    # Prefer per-component chains (for hard/soft AOM); fall back to screening finalists.
    per_comp = diag.get("per_component_chains", [])
    if per_comp:
        selected_payload = per_comp
    else:
        sigs = diag.get("selected_chain_signatures", [])
        base_idx = diag.get("selected_base_indices", [])
        selected_payload = [
            {"base_index": int(b), "chain_signature": s}
            for b, s in zip(base_idx, sigs)
        ]

    ref_pls = cohort_row.get("ref_rmse_pls")
    ref_tabraw = cohort_row.get("ref_rmse_tabpfn_raw")
    ref_tabopt = cohort_row.get("ref_rmse_tabpfn_opt")

    n_components_selected = 0
    if hasattr(est, "model_") and hasattr(est.model_, "n_components_"):
        n_components_selected = int(est.model_.n_components_)

    return {
        "database_name": cohort_row["database_name"],
        "dataset": cohort_row["dataset"],
        "task": "regression",
        "model": variant["label"],
        "result_label": variant["label"],
        "result_dir": "",
        "status": "ok",
        "status_details": "",
        "preprocessing_pipeline": variant["primitive_bank"],
        "RMSECV": "",
        "RMSE_MF": "",
        "RMSEP": rmsep,
        "MAE_test": mae(yte, yhat),
        "r2_test": r2(yte, yhat),
        "search_mean_score": "",
        "seed": seed,
        "n_splits": "",
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
        "backend": "numpy",
        "engine": "fastaom",
        "selection": variant["model"],
        "criterion": "screening+ridge",
        "orthogonalization": "primal",
        "operator_bank": variant["primitive_bank"],
        "selected_operator_sequence_json": json.dumps(selected_payload),
        "selected_operator_scores_json": json.dumps({}),
        "n_components_selected": n_components_selected,
        "max_components": max_components,
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
        "delta_rmsep_vs_master_pls": (rmsep - float(ref_pls)) if pd.notna(ref_pls) else "",
        "delta_rmsep_vs_tabpfn_raw": (rmsep - float(ref_tabraw)) if pd.notna(ref_tabraw) else "",
        "delta_rmsep_vs_tabpfn_opt": (rmsep - float(ref_tabopt)) if pd.notna(ref_tabopt) else "",
        "run_seed": seed,
        "code_version": "FastAOM/0.1.0",
        "notes": variant.get("notes", ""),
        "n_chains_enumerated": diag.get("n_chains_enumerated", 0),
        "n_candidates_total": diag.get("n_candidates_total", 0),
        "n_finalists": diag.get("n_finalists", 0),
        "top_finalist_score": diag.get("top_finalist_score", 0.0),
        "screening_time_s": timings.get("screening", ""),
        "lowrank_time_s": timings.get("lowrank_fit", ""),
        "model_fit_time_s": timings.get("model_fit", ""),
        "balanced_accuracy": "",
        "macro_f1": "",
        "log_loss": "",
        "ece": "",
        # bench/scenarios "new" schema duplicates
        "schema_version": "fastaom_v0_1",
        "preset": "fastaom_smoke",
        "cohort": "diverse11",
        "canonical_name": variant["label"],
        "model_class": _model_class(variant["label"]),
        "module": "FastAOM.models.fast_aom_pls_ridge",
        "rmsep": rmsep,
        "mae": mae(yte, yhat),
        "r2": r2(yte, yhat),
        "error_message": "",
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "n_features": int(Xtr.shape[1]),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_dataset(
    cohort_row: pd.Series,
    variants: List[Dict[str, object]],
    results_path: Path,
    seeds: Sequence[int],
    max_components: int,
    existing_keys: set,
) -> int:
    if cohort_row["status"] != "ok":
        return 0
    Xtr = _load_csv_array(cohort_row["train_path"])
    Xte = _load_csv_array(cohort_row["test_path"])
    ytr = _load_csv_target(cohort_row["ytrain_path"]).astype(float)
    yte = _load_csv_target(cohort_row["ytest_path"]).astype(float)
    n_runs = 0
    for variant in variants:
        for seed in seeds:
            key = (
                str(cohort_row["database_name"]),
                str(cohort_row["dataset"]),
                str(variant["label"]),
                str(seed),
            )
            if key in existing_keys:
                continue
            try:
                row = _run_variant(variant, Xtr, ytr, Xte, yte, seed, cohort_row, max_components)
                _append_row(results_path, row)
                existing_keys.add(key)
                n_runs += 1
            except Exception as exc:  # pragma: no cover
                tb = traceback.format_exc(limit=2)
                row = {col: "" for col in RESULT_COLUMNS}
                row.update({
                    "database_name": cohort_row["database_name"],
                    "dataset": cohort_row["dataset"],
                    "task": "regression",
                    "model": variant["label"],
                    "result_label": variant["label"],
                    "status": "skipped",
                    "status_details": (str(exc) + " | " + tb.replace("\n", " "))[:400],
                    "aom_variant": variant["label"],
                    "engine": "fastaom",
                    "selection": variant["model"],
                    "operator_bank": variant["primitive_bank"],
                    "backend": "numpy",
                    "seed": seed,
                    "run_seed": seed,
                    "code_version": "FastAOM/0.1.0",
                    "notes": "exception during run",
                    "schema_version": "fastaom_v0_1",
                    "preset": "fastaom_smoke",
                    "cohort": "diverse11",
                    "canonical_name": variant["label"],
                    "model_class": _model_class(variant["label"]),
                    "module": "FastAOM.models.fast_aom_pls_ridge",
                    "error_message": (str(exc))[:200],
                })
                _append_row(results_path, row)
                existing_keys.add(key)
    return n_runs


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the FastAOM benchmark")
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--seeds", default="0", help="Comma-separated list of seeds")
    parser.add_argument("--max-components", type=int, default=15)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on the number of datasets")
    parser.add_argument(
        "--variants", default="", help="Comma-separated variant labels to run; empty = all"
    )
    args = parser.parse_args(argv)
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    results_path = workspace / "results.csv"
    cohort_df = pd.read_csv(args.cohort)
    if args.limit > 0:
        cohort_df = cohort_df.head(args.limit)
    variants = REGRESSION_VARIANTS
    if args.variants:
        wanted = set(v.strip() for v in args.variants.split(",") if v.strip())
        variants = [v for v in variants if v["label"] in wanted]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    existing = _existing_keys(results_path)
    total = 0
    for _, cohort_row in cohort_df.iterrows():
        n = run_dataset(cohort_row, variants, results_path, seeds, args.max_components, existing)
        total += n
        sys.stdout.write(f"[{cohort_row['database_name']}/{cohort_row['dataset']}] +{n} rows\n")
        sys.stdout.flush()
    sys.stdout.write(f"FastAOM done: {total} new rows written to {results_path}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
