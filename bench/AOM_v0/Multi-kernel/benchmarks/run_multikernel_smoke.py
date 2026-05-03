"""Unified smoke benchmark for mkR / MKM / BLUP.

Runs each of the three new models (and a few baselines) on a small subset of
the all57 cohort and writes a single results CSV that lets us compare against
the reference RMSEs (PLS, Ridge, TabPFN-raw/opt, CNN, CatBoost, AOM-PLS).

Usage (from repository root):

```bash
.venv/bin/python bench/AOM_v0/Multi-kernel/benchmarks/run_multikernel_smoke.py \
  --cohort smoke3 \
  --workspace bench/AOM_v0/Multi-kernel/benchmark_runs/smoke3
```

``--cohort`` choices:
- ``smoke3``: ALPINE/ALPINE_P_291_KS, AMYLOSE/Rice_Amylose_313_YbasedSplit,
  BEER/Beer_OriginalExtract_60_KS.
- ``extended12``: 12 datasets sampled across families (small + medium + large n).
- a path to a CSV with explicit dataset selection.

Variants:
- ``mkR-uniform``, ``mkR-kta``, ``mkR-softmax_cv`` (compact bank).
- ``MKM-reml`` (compact bank).
- ``BLUP-reml`` (compact bank, predictions equal MKM by construction).
- ``Ridge-raw`` baseline (sklearn Ridge with alpha=1.0 trace-relative).
- ``PLS-paper-ref`` and ``Ridge-paper-ref`` are read directly from the cohort
  CSV (no fit needed); they appear as ``ref_*`` columns in the output.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve()
BENCHMARKS_DIR = ROOT.parent
MULTI_KERNEL = BENCHMARKS_DIR.parent          # bench/AOM_v0/Multi-kernel
AOM_V0 = MULTI_KERNEL.parent                  # bench/AOM_v0
REPO_ROOT = AOM_V0.parent.parent              # nirs4all
RIDGE_ROOT = AOM_V0 / "Ridge"
MKR_ROOT = MULTI_KERNEL / "MKR"
MKM_ROOT = MULTI_KERNEL / "MkM"
BLUP_ROOT = MULTI_KERNEL / "Blup"

# Order matters: each insert(0, ...) prepends. We want MKR_ROOT first in
# sys.path so ``import aomridge`` picks up the Multi-kernel/MKR copy
# (which has the latest mkR additions) rather than Ridge/aomridge/.
for p in (RIDGE_ROOT, BLUP_ROOT, MKM_ROOT, MULTI_KERNEL, MKR_ROOT):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

# ---------------------------------------------------------------------- code
from aomridge.mkr_estimator import AOMMultiKernelRidge        # noqa: E402
from mkm.estimator import AOMMultiKernelMixedModel             # noqa: E402
from blup.estimator import AOMMultiKernelBLUP                  # noqa: E402
from sklearn.linear_model import Ridge                          # noqa: E402

CODE_VERSION = "Multi-kernel/0.1.0"

RESULT_COLUMNS = [
    "dataset_group", "dataset", "n_train", "n_test", "p",
    "variant", "status", "error",
    "operator_bank", "weight_strategy", "method", "branch_preproc",
    "alpha", "kernel_alignment_max", "fit_time_s", "predict_time_s",
    "converged", "boundary_components",
    "rmsep", "mae", "r2",
    "ref_rmse_pls", "ref_rmse_ridge",
    "ref_rmse_tabpfn_raw", "ref_rmse_tabpfn_opt",
    "ref_rmse_cnn", "ref_rmse_catboost",
    "rel_rmsep_vs_pls", "rel_rmsep_vs_ridge", "rel_rmsep_vs_tabpfn_opt",
    "version", "random_state",
]

SMOKE3 = [
    ("ALPINE", "ALPINE_P_291_KS"),
    ("AMYLOSE", "Rice_Amylose_313_YbasedSplit"),
    ("BEER", "Beer_OriginalExtract_60_KS"),
]

EXTENDED12 = [
    ("ALPINE", "ALPINE_P_291_KS"),
    ("AMYLOSE", "Rice_Amylose_313_YbasedSplit"),
    ("BEER", "Beer_OriginalExtract_60_KS"),
    ("BEEFMARBLING", "Beef_Marbling_RandomSplit"),
    ("BISCUIT", "Biscuit_Fat_40_RandomSplit"),
    ("MILK", "Milk_Lactose_1224_KS"),
    ("PLUMS", "Brix_spxy70"),
    ("QUARTZ", "Quartz_spxy70"),
    ("BERRY", "ph_groupSampleID_stratDateVar_balRows"),
    ("CASSAVA", "Brix_KS"),
    ("CORN", "Moisture_KS"),
    ("WHEAT", "Wheat_Protein_50_KS"),
]


def _load_csv_array(path: Path) -> np.ndarray:
    df = pd.read_csv(path, sep=";")
    arr = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isnan(arr).any():
        col_mean = np.nanmean(arr, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        idx = np.where(np.isnan(arr))
        arr[idx] = np.take(col_mean, idx[1])
    return arr


def _load_csv_target(path: Path) -> np.ndarray:
    df = pd.read_csv(path, sep=";")
    return df.iloc[:, 0].astype(float).to_numpy()


def _resolve_cohort(name: str, all57_csv: Path) -> list[dict]:
    df = pd.read_csv(all57_csv)
    if name == "smoke3":
        keys = SMOKE3
    elif name == "extended12":
        keys = EXTENDED12
    elif Path(name).exists():
        sel_df = pd.read_csv(name)
        return sel_df.to_dict(orient="records")
    else:
        raise ValueError(f"unknown cohort {name!r}")
    out: list[dict] = []
    for db, ds in keys:
        sub = df[(df.database_name == db) & (df.dataset == ds)]
        if sub.empty:
            print(f"WARN: dataset {db}/{ds} not in cohort csv", file=sys.stderr)
            continue
        out.append(sub.iloc[0].to_dict())
    return out


def _pred_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    rmsep = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    if np.var(y_true) > 0:
        r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
    else:
        r2 = float("nan")
    return {"rmsep": rmsep, "mae": mae, "r2": r2}


def _safe(x):
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else x


def _parse_variant(name: str) -> tuple[str, str | None, str, dict]:
    """Return (family, strategy_or_method, branch, extras).

    Variant naming convention:
    ``<family>-<strategy>[-<branch>][-<extras>]``.

    Recognised "extras" tokens:
    - ``default``: use ``operator_bank='default'`` (100 ops) instead of
      compact (9 ops).
    - ``activeNN``: pre-screen the bank to top NN operators by KTA
      (``top_k_active=NN``).
    - ``kta`` / ``norm`` / ``blend``: choose the screening score method.

    Examples:
    - ``mkR-softmax_cv``                       → compact, no branch.
    - ``mkR-softmax_cv-asls``                  → compact, ASLS branch.
    - ``mkR-softmax_cv-default-active15``      → default bank, top-15.
    - ``mkR-softmax_cv-snv-default-active15``  → default+SNV+top-15.
    - ``MKM-reml-default-active10-kta``        → default+top-10+KTA score.
    - ``Ridge-raw``                            → sklearn Ridge baseline.
    """
    parts = name.split("-")
    branch = "none"
    extras = {"operator_bank": "compact",
              "top_k_active": None,
              "screen_score_method": "norm",
              "budget": "default",
              "weight_top_k_post": None,
              "weight_top_k_post_retune_alpha": True}
    branch_options = {"none", "snv", "msc", "asls", "osc", "emsc1"}
    # Multi-branch tokens: combinations of base branches glued by '+' or
    # by short codes (aslmsc=asls+msc, aslmscsnv=asls+msc+snv, etc.).
    multi_branch_codes = {
        "aslmsc": ["asls", "msc"],
        "aslsnv": ["asls", "snv"],
        "mscsnv": ["msc", "snv"],
        "aslmscsnv": ["asls", "msc", "snv"],
    }
    if len(parts) == 1:
        return parts[0], None, branch, extras
    family = parts[0]
    strategy = parts[1] if len(parts) >= 2 else None
    rest = parts[2:]
    for token in rest:
        if token in multi_branch_codes:
            branch = multi_branch_codes[token]
        elif token in branch_options:
            branch = token
        elif token == "default":
            extras["operator_bank"] = "default"
        elif token == "compact":
            extras["operator_bank"] = "compact"
        elif token == "extended":
            extras["operator_bank"] = "extended"
        elif token == "deep":
            extras["operator_bank"] = "deep"
        elif token.startswith("active") and token[6:].isdigit():
            extras["top_k_active"] = int(token[6:])
        elif token in ("kta", "norm", "blend"):
            extras["screen_score_method"] = token
        elif token in ("tuned", "fast"):
            extras["budget"] = token
        elif token.startswith("sparse") and token[6:].isdigit():
            extras["weight_top_k_post"] = int(token[6:])
        elif token == "noretune":
            extras["weight_top_k_post_retune_alpha"] = False
        else:
            raise ValueError(f"unknown variant token {token!r} in {name!r}")
    return family, strategy, branch, extras


def _build_stack_5(operator_bank: str, random_state: int):
    """Build a 5-base-learner StackingRegressor combining the 5 strongest
    multi-kernel variants identified on diverse10:

    - Ridge-raw (winner on n >> p, e.g. ECOSIS)
    - mkR-softmax_cv (winner on smooth high-dim spectra, e.g. ALPINE)
    - mkR-softmax_cv-snv (winner on scatter-affected spectra, e.g. MANURE)
    - MKM-reml (likelihood-based, robust on small n)
    - MKM-reml-asls (winner on asymmetric-baseline spectra, e.g. AMYLOSE)

    A Ridge meta-model combines the base predictions linearly. cv=3 for
    OOF fitting (each base learner fits 4 times: 3 folds + 1 final refit).
    """
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import Ridge as SkRidge
    return StackingRegressor(
        estimators=[
            (
                "ridge_raw",
                SkRidge(alpha=1.0, random_state=random_state),
            ),
            (
                "mkR",
                AOMMultiKernelRidge(
                    operator_bank=operator_bank,
                    weight_strategy="softmax_cv",
                    weight_n_restarts=2, weight_max_iter=15,
                    alpha_grid_size=15, alpha_cv_n_splits=3,
                    branch_preproc="none",
                    random_state=random_state,
                ),
            ),
            (
                "mkR_snv",
                AOMMultiKernelRidge(
                    operator_bank=operator_bank,
                    weight_strategy="softmax_cv",
                    weight_n_restarts=2, weight_max_iter=15,
                    alpha_grid_size=15, alpha_cv_n_splits=3,
                    branch_preproc="snv",
                    random_state=random_state,
                ),
            ),
            (
                "MKM",
                AOMMultiKernelMixedModel(
                    operator_bank=operator_bank,
                    method="reml",
                    n_random_restarts=2, max_iter=50,
                    branch_preproc="none",
                    random_state=random_state,
                ),
            ),
            (
                "MKM_asls",
                AOMMultiKernelMixedModel(
                    operator_bank=operator_bank,
                    method="reml",
                    n_random_restarts=2, max_iter=50,
                    branch_preproc="asls",
                    random_state=random_state,
                ),
            ),
        ],
        final_estimator=SkRidge(alpha=1.0, random_state=random_state),
        cv=3, passthrough=False, n_jobs=1, verbose=0,
    )


def _run_variant(
    variant_name: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    *,
    operator_bank: str = "compact",
    random_state: int = 0,
) -> dict:
    """Fit one variant and return a dict ready to merge into the result row."""
    family, strategy, branch, extras = _parse_variant(variant_name)
    bank = extras.get("operator_bank") or operator_bank
    top_k_active = extras.get("top_k_active")
    score_method = extras.get("screen_score_method", "norm")
    budget = extras.get("budget", "default")
    # Budget tiers — larger = more thorough optimisation but slower fits.
    if budget == "fast":
        mkr_n_restarts, mkr_max_iter, alpha_grid_size, cv_n_splits = 1, 10, 12, 3
        mkm_n_restarts, mkm_max_iter = 2, 30
    elif budget == "tuned":
        mkr_n_restarts, mkr_max_iter, alpha_grid_size, cv_n_splits = 5, 40, 40, 5
        mkm_n_restarts, mkm_max_iter = 6, 150
    else:  # default
        mkr_n_restarts, mkr_max_iter, alpha_grid_size, cv_n_splits = 2, 20, 20, 3
        mkm_n_restarts, mkm_max_iter = 3, 80
    info: dict = {
        "variant": variant_name,
        "operator_bank": bank,
        "weight_strategy": None,
        "method": None,
        "branch_preproc": branch,
        "alpha": None,
        "kernel_alignment_max": None,
        "fit_time_s": None, "predict_time_s": None,
        "converged": None, "boundary_components": None,
        "rmsep": None, "mae": None, "r2": None,
        "status": "ok", "error": None,
    }
    try:
        t0 = time.time()
        if family == "Ridge":
            model = Ridge(alpha=1.0, random_state=random_state)
            model.fit(X_train, y_train)
            info["weight_strategy"] = "n/a"
            info["method"] = "sklearn-Ridge"
            info["alpha"] = 1.0
        elif family == "mkR":
            model = AOMMultiKernelRidge(
                operator_bank=bank,
                weight_strategy=strategy,
                weight_n_restarts=mkr_n_restarts,
                weight_max_iter=mkr_max_iter,
                weight_top_k_post=extras.get("weight_top_k_post"),
                weight_top_k_post_retune_alpha=extras.get("weight_top_k_post_retune_alpha", True),
                alpha_grid_size=alpha_grid_size,
                alpha_cv_n_splits=cv_n_splits,
                branch_preproc=branch,
                kernel_top_k_active=top_k_active,
                kernel_screen_score_method=score_method,
                random_state=random_state,
            )
            model.fit(X_train, y_train)
            info["weight_strategy"] = strategy
            info["method"] = "mkR"
            info["alpha"] = float(model.alpha_)
            info["kernel_alignment_max"] = float(model.kernel_alignment_max_)
        elif family == "MKM":
            model = AOMMultiKernelMixedModel(
                operator_bank=bank,
                method=strategy,
                n_random_restarts=mkm_n_restarts,
                max_iter=mkm_max_iter,
                branch_preproc=branch,
                kernel_top_k_active=top_k_active,
                kernel_screen_score_method=score_method,
                sigma2_top_k_post=extras.get("weight_top_k_post"),
                random_state=random_state,
            )
            model.fit(X_train, y_train)
            info["weight_strategy"] = "REML/ML"
            info["method"] = strategy
            info["kernel_alignment_max"] = float(model.kernel_alignment_max_)
            info["converged"] = bool(model.converged_)
            info["boundary_components"] = ",".join(map(str, model.boundary_components_))
        elif family == "BLUP":
            model = AOMMultiKernelBLUP(
                operator_bank=bank,
                method=strategy,
                n_random_restarts=3,
                max_iter=80,
                branch_preproc=branch,
                kernel_top_k_active=top_k_active,
                kernel_screen_score_method=score_method,
                random_state=random_state,
            )
            model.fit(X_train, y_train)
            info["weight_strategy"] = "REML/ML"
            info["method"] = strategy
            info["kernel_alignment_max"] = float(model.kernel_alignment_max_)
            info["converged"] = bool(model.converged_)
            info["boundary_components"] = ",".join(map(str, model.boundary_components_))
        elif family == "Stack":
            # Stack-5: 5 base learners + Ridge meta-model.
            model = _build_stack_5(operator_bank=operator_bank,
                                    random_state=random_state)
            model.fit(X_train, y_train)
            info["weight_strategy"] = "stacking-OOF-Ridge-meta"
            info["method"] = "Stack-5"
            # Capture the meta-model coefficients (one per base learner).
            try:
                info["alpha"] = float(model.final_estimator_.alpha)
                meta_coef = model.final_estimator_.coef_
                info["boundary_components"] = ",".join(
                    f"{c:.3f}" for c in meta_coef
                )
            except Exception:
                pass
        else:
            raise ValueError(f"unknown variant family {family!r}")
        info["fit_time_s"] = time.time() - t0
        t1 = time.time()
        y_pred = model.predict(X_test)
        info["predict_time_s"] = time.time() - t1
        info.update(_pred_metrics(y_test, y_pred))
    except Exception as exc:                                    # noqa: BLE001
        info["status"] = "error"
        info["error"] = f"{type(exc).__name__}: {exc!s}"
    return info


def _ref_relatives(rmsep: float | None, ref_pls: float | None,
                   ref_ridge: float | None, ref_tabpfn_opt: float | None) -> dict:
    rel = {}
    if rmsep is not None and ref_pls and ref_pls > 0:
        rel["rel_rmsep_vs_pls"] = rmsep / ref_pls
    else:
        rel["rel_rmsep_vs_pls"] = None
    if rmsep is not None and ref_ridge and ref_ridge > 0:
        rel["rel_rmsep_vs_ridge"] = rmsep / ref_ridge
    else:
        rel["rel_rmsep_vs_ridge"] = None
    if rmsep is not None and ref_tabpfn_opt and ref_tabpfn_opt > 0:
        rel["rel_rmsep_vs_tabpfn_opt"] = rmsep / ref_tabpfn_opt
    else:
        rel["rel_rmsep_vs_tabpfn_opt"] = None
    return rel


def _append_row(out_csv: Path, row: dict) -> None:
    new = not out_csv.exists()
    with out_csv.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in RESULT_COLUMNS})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="smoke3",
                        help="'smoke3', 'extended12', or path to a CSV.")
    parser.add_argument("--workspace", default="bench/AOM_v0/Multi-kernel/benchmark_runs/smoke3",
                        type=Path)
    parser.add_argument(
        "--variants", nargs="+", default=[
            "Ridge-raw",
            "mkR-uniform", "mkR-kta", "mkR-softmax_cv",
            "MKM-reml",
            "BLUP-reml",
        ],
    )
    parser.add_argument("--all57-csv", type=Path,
                        default=Path("bench/AOM_v0/Ridge/benchmark_runs/all57_cohort.csv"))
    parser.add_argument("--operator-bank", default="compact")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N datasets (debug).")
    args = parser.parse_args(argv)

    args.workspace.mkdir(parents=True, exist_ok=True)
    out_csv = args.workspace / "results.csv"

    cohort = _resolve_cohort(args.cohort, args.all57_csv)
    if args.limit:
        cohort = cohort[: args.limit]

    print(f"[smoke] cohort: {len(cohort)} datasets, variants: {args.variants}")
    print(f"[smoke] writing to: {out_csv}")
    print()

    for ds_idx, ds_row in enumerate(cohort):
        db = ds_row["database_name"]
        ds = ds_row["dataset"]
        print(f"[{ds_idx + 1}/{len(cohort)}] {db}/{ds}", flush=True)
        try:
            X_train = _load_csv_array(REPO_ROOT / ds_row["train_path"])
            X_test = _load_csv_array(REPO_ROOT / ds_row["test_path"])
            y_train = _load_csv_target(REPO_ROOT / ds_row["ytrain_path"])
            y_test = _load_csv_target(REPO_ROOT / ds_row["ytest_path"])
        except FileNotFoundError as exc:
            print(f"  -> skip (missing path): {exc}")
            continue
        n_train, p = X_train.shape
        n_test = X_test.shape[0]
        ref_pls = float(ds_row.get("ref_rmse_pls") or float("nan"))
        ref_ridge = float(ds_row.get("ref_rmse_ridge") or float("nan"))
        ref_tabpfn_raw = float(ds_row.get("ref_rmse_tabpfn_raw") or float("nan"))
        ref_tabpfn_opt = float(ds_row.get("ref_rmse_tabpfn_opt") or float("nan"))
        ref_cnn = float(ds_row.get("ref_rmse_cnn") or float("nan"))
        ref_cb = float(ds_row.get("ref_rmse_catboost") or float("nan"))
        for variant in args.variants:
            info = _run_variant(
                variant, X_train, y_train, X_test, y_test,
                operator_bank=args.operator_bank,
                random_state=args.random_state,
            )
            row = {
                "dataset_group": db,
                "dataset": ds,
                "n_train": n_train,
                "n_test": n_test,
                "p": p,
                **info,
                "ref_rmse_pls": ref_pls,
                "ref_rmse_ridge": ref_ridge,
                "ref_rmse_tabpfn_raw": ref_tabpfn_raw,
                "ref_rmse_tabpfn_opt": ref_tabpfn_opt,
                "ref_rmse_cnn": ref_cnn,
                "ref_rmse_catboost": ref_cb,
                **_ref_relatives(info["rmsep"], ref_pls, ref_ridge, ref_tabpfn_opt),
                "version": CODE_VERSION,
                "random_state": args.random_state,
            }
            _append_row(out_csv, row)
            status = info["status"]
            rmsep = info.get("rmsep")
            rel_pls = row.get("rel_rmsep_vs_pls")
            print(
                f"  {variant:<25s} status={status:<5s}  RMSEP="
                f"{(f'{rmsep:.4f}' if rmsep is not None else 'NaN'):>10s}  "
                f"vs_PLS={f'{rel_pls:.3f}' if rel_pls else 'NaN'}  "
                f"fit_t={info.get('fit_time_s', 0):.1f}s",
                flush=True,
            )
        print()

    print(f"[smoke] done. results: {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
