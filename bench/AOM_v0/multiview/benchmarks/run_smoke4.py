"""Smoke-4 baseline runner for AOM-multiview Phase 1.

Three variants on the 4-dataset smoke cohort:

- `PLS-standard-numpy` — sklearn PLSRegression reference (auto-`n_components`)
- `AOM-PLS-compact-numpy` — AOMPLSRegressor with the `compact` bank (holdout)
- `MBPLS-blocks3-vanilla` — nirs4all `MBPLS` on 3 equal-width wavelength blocks

Outputs `bench/AOM_v0/multiview/results/smoke4_baseline.csv` with one row per
(dataset, variant, seed). The existing 57-dataset cohort lives at
`bench/AOM_v0/benchmarks/cohort_regression.csv`; we filter it to the 4 datasets
the user nominated for Phase 1 iteration.

Run from the repo root:

    .venv/bin/python bench/AOM_v0/multiview/benchmarks/run_smoke4.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

_HERE = Path(os.path.abspath(__file__)).resolve()
_MULTIVIEW_ROOT = _HERE.parent.parent           # bench/AOM_v0/multiview
_AOM_ROOT = _MULTIVIEW_ROOT.parent              # bench/AOM_v0
for _p in (_MULTIVIEW_ROOT, _AOM_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from aompls.estimators import AOMPLSRegressor  # noqa: E402

from sklearn.cross_decomposition import PLSRegression  # noqa: E402
from sklearn.metrics import mean_squared_error, r2_score  # noqa: E402

from nirs4all.operators.models.sklearn.mbpls import MBPLS  # noqa: E402

from multiview.views import BlockMaskOperator, ViewBuilder  # noqa: E402


SMOKE4_DATASETS = [
    "Beer_OriginalExtract_60_YbaseSplit",
    "All_manure_MgO_SPXY_strat_Manure_type",
    "Chla+b_spxyG_block2deg",
    "grapevine_chloride_556_KS",
]


# ---------------------------------------------------------------------------
# Data loading (mirrors run_aompls_benchmark conventions)
# ---------------------------------------------------------------------------


def _coerce_numeric(df: pd.DataFrame) -> np.ndarray:
    """Coerce a possibly mixed-decimal DataFrame to a float ndarray.

    Some cohort CSVs use US decimals for most values and European decimals
    (`1,23E-04`) for scientific notation. We replace commas with dots in
    object columns before casting. Mirrors `run_aompls_benchmark._coerce_numeric`.
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
        return _coerce_numeric(df).ravel()
    # Some target CSVs include a header row; ravel anyway.
    return _coerce_numeric(df).ravel()


# ---------------------------------------------------------------------------
# Variant runners
# ---------------------------------------------------------------------------


def _split_blocks(X: np.ndarray, K: int) -> List[np.ndarray]:
    """Split X into K equal-width contiguous blocks via the public ViewBuilder."""
    bank = ViewBuilder.blocks_only(K=K, strategy="equal_width").build(p=X.shape[1])
    masks = [op for op in bank if isinstance(op, BlockMaskOperator)]
    return [X[:, m.start : m.end] for m in masks]


def _select_n_components_holdout(
    X: np.ndarray,
    y: np.ndarray,
    max_components: int,
    seed: int,
    holdout_fraction: float = 0.2,
) -> int:
    """Pick `n_components` by single-shot holdout RMSE on training data only."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_val = max(3, int(n * holdout_fraction))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    X_tr, X_va = X[tr_idx], X[val_idx]
    y_tr, y_va = y[tr_idx], y[val_idx]
    best_k = 1
    best_rmse = float("inf")
    upper = min(max_components, max(1, X_tr.shape[0] - 1), X_tr.shape[1])
    for k in range(1, upper + 1):
        try:
            est = PLSRegression(n_components=k)
            est.fit(X_tr, y_tr)
            pred = est.predict(X_va).ravel()
        except Exception:
            continue
        rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_k = k
    return best_k


def _select_n_components_holdout_blocks(
    X_blocks: List[np.ndarray],
    y: np.ndarray,
    max_components: int,
    seed: int,
    holdout_fraction: float = 0.2,
) -> int:
    rng = np.random.default_rng(seed)
    n = X_blocks[0].shape[0]
    perm = rng.permutation(n)
    n_val = max(3, int(n * holdout_fraction))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    X_tr_blocks = [Xb[tr_idx] for Xb in X_blocks]
    X_va_blocks = [Xb[val_idx] for Xb in X_blocks]
    y_tr, y_va = y[tr_idx], y[val_idx]
    best_k = 1
    best_rmse = float("inf")
    n_features_total = sum(Xb.shape[1] for Xb in X_tr_blocks)
    upper = min(max_components, max(1, X_tr_blocks[0].shape[0] - 1), n_features_total)
    for k in range(1, upper + 1):
        try:
            est = MBPLS(n_components=k)
            est.fit(X_tr_blocks, y_tr)
            pred = est.predict(X_va_blocks).ravel()
        except Exception:
            continue
        rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_k = k
    return best_k


def _run_pls_standard(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
    seed: int, max_components: int,
) -> dict:
    t0 = time.perf_counter()
    n_components = _select_n_components_holdout(Xtr, ytr, max_components, seed)
    est = PLSRegression(n_components=n_components)
    est.fit(Xtr, ytr)
    fit_time = time.perf_counter() - t0
    pred = est.predict(Xte).ravel()
    return {
        "rmsep": float(np.sqrt(mean_squared_error(yte, pred))),
        "r2": float(r2_score(yte, pred)),
        "n_components": int(n_components),
        "fit_time_s": float(fit_time),
    }


def _run_aom_pls(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
    seed: int, max_components: int,
) -> dict:
    t0 = time.perf_counter()
    est = AOMPLSRegressor(
        n_components="auto",
        max_components=max_components,
        engine="simpls_covariance",
        selection="global",
        criterion="holdout",
        operator_bank="compact",
        random_state=seed,
    )
    est.fit(Xtr, ytr)
    fit_time = time.perf_counter() - t0
    pred = est.predict(Xte).ravel()
    selected = est.get_selected_operators()
    return {
        "rmsep": float(np.sqrt(mean_squared_error(yte, pred))),
        "r2": float(r2_score(yte, pred)),
        "n_components": int(est.n_components_),
        "fit_time_s": float(fit_time),
        "selected_operators": ",".join(sorted(set(selected))),
    }


def _run_mbpls_blocks3(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
    seed: int, max_components: int, K: int = 3,
) -> dict:
    Xtr_blocks = _split_blocks(Xtr, K)
    Xte_blocks = _split_blocks(Xte, K)
    t0 = time.perf_counter()
    n_components = _select_n_components_holdout_blocks(
        Xtr_blocks, ytr, max_components, seed
    )
    est = MBPLS(n_components=n_components)
    est.fit(Xtr_blocks, ytr)
    fit_time = time.perf_counter() - t0
    pred = est.predict(Xte_blocks).ravel()
    block_sizes = ",".join(str(Xb.shape[1]) for Xb in Xtr_blocks)
    return {
        "rmsep": float(np.sqrt(mean_squared_error(yte, pred))),
        "r2": float(r2_score(yte, pred)),
        "n_components": int(n_components),
        "fit_time_s": float(fit_time),
        "block_sizes": block_sizes,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _existing_keys(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    return {
        (str(r["dataset"]), str(r["variant"]), int(r["seed"]))
        for _, r in df.iterrows()
    }


def _append(path: Path, row: dict, columns: List[str]) -> None:
    df = pd.DataFrame([{c: row.get(c, "") for c in columns}])
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


COLUMNS = [
    "database_name", "dataset", "variant", "seed",
    "n_train", "n_test", "n_features",
    "rmsep", "r2", "n_components", "fit_time_s",
    "selected_operators", "block_sizes",
    "status", "status_details",
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", default="bench/AOM_v0/multiview/results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-components", type=int, default=15)
    parser.add_argument(
        "--cohort",
        default="bench/AOM_v0/benchmarks/cohort_regression.csv",
    )
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    results_path = workspace / "smoke4_baseline.csv"
    existing = _existing_keys(results_path)

    cohort = pd.read_csv(args.cohort)
    smoke = cohort[
        cohort["dataset"].isin(SMOKE4_DATASETS) & (cohort["status"] == "ok")
    ].copy()
    print(
        f"[smoke4] {len(smoke)} datasets x 3 variants on seed {args.seed} "
        f"-> {results_path}"
    )

    runners = [
        ("PLS-standard-numpy", _run_pls_standard),
        ("AOM-PLS-compact-numpy", _run_aom_pls),
        ("MBPLS-blocks3-vanilla", _run_mbpls_blocks3),
    ]

    n_appended = 0
    for _, cohort_row in smoke.iterrows():
        Xtr = _load_csv_array(cohort_row["train_path"])
        Xte = _load_csv_array(cohort_row["test_path"])
        ytr = _load_csv_target(cohort_row["ytrain_path"]).astype(float)
        yte = _load_csv_target(cohort_row["ytest_path"]).astype(float)
        for label, runner in runners:
            key = (str(cohort_row["dataset"]), label, int(args.seed))
            if key in existing:
                print(f"[smoke4] skip already-done {key}")
                continue
            base = {
                "database_name": cohort_row.get("database_name", ""),
                "dataset": cohort_row["dataset"],
                "variant": label,
                "seed": int(args.seed),
                "n_train": int(Xtr.shape[0]),
                "n_test": int(Xte.shape[0]),
                "n_features": int(Xtr.shape[1]),
                "status": "ok",
            }
            try:
                metrics = runner(Xtr, ytr, Xte, yte, args.seed, args.max_components)
                row = {**base, **metrics}
            except Exception as exc:
                row = {
                    **base,
                    "status": "error",
                    "status_details": str(exc)[:200],
                }
            _append(results_path, row, COLUMNS)
            existing.add(key)
            n_appended += 1
            print(
                f"[smoke4] {cohort_row['dataset']:<48s} {label:<28s} "
                f"rmsep={row.get('rmsep', 'NA')} r2={row.get('r2', 'NA')} "
                f"k={row.get('n_components', 'NA')} t={row.get('fit_time_s', 'NA')}"
            )

    print(f"[smoke4] appended {n_appended} rows -> {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
