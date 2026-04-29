"""Build regression and classification cohorts for the AOM_v0 benchmark.

Regression cohort:

- Read `bench/tabpfn_paper/master_results.csv`.
- Extract the unique (database_name, dataset) pairs used as the regression
  oracle.
- For each pair, locate the on-disk dataset under
  `bench/tabpfn_paper/data/regression/<database_name>/<dataset>/`.
- Validate the presence of `Xtrain.csv`, `Xtest.csv`, `Ytrain.csv`, `Ytest.csv`.
- Write `cohort_regression.csv` with columns:

    `database_name, dataset, status, reason, n_train, n_test, p, ref_rmse_pls,
     ref_rmse_tabpfn_raw, ref_rmse_tabpfn_opt, ref_rmse_cnn, ref_rmse_catboost,
     ref_rmse_ridge, train_path, test_path, ytrain_path, ytest_path`

Classification cohort:

- Scan `bench/tabpfn_paper/data/classification/`.
- For each leaf folder containing `Xtrain.csv` and `Ytrain.csv`, record an
  entry. Splits without test files are allowed; the runner builds a
  deterministic split when needed.
- Datasets that fail to parse get a row with `status="skipped"` and a reason.

Skipped or unavailable rows are kept (not dropped). The runner can filter on
`status == "ok"` if desired.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


_DEFAULT_MASTER = "bench/tabpfn_paper/master_results.csv"
_DEFAULT_REG_ROOT = "bench/tabpfn_paper/data/regression"
_DEFAULT_CLF_ROOT = "bench/tabpfn_paper/data/classification"
_DEFAULT_OUT_REG = "bench/AOM_v0/benchmarks/cohort_regression.csv"
_DEFAULT_OUT_CLF = "bench/AOM_v0/benchmarks/cohort_classification.csv"


def _find_split_dir(root: Path, database: str, dataset: str) -> Optional[Path]:
    """Locate the on-disk dataset folder by case-insensitive matching."""
    if not root.is_dir():
        return None
    candidates = [database, database.upper(), database.lower()]
    for cand in candidates:
        path = root / cand / dataset
        if path.is_dir():
            return path
    # Fallback: walk database directories
    for db_dir in root.iterdir():
        if not db_dir.is_dir():
            continue
        if db_dir.name.lower() == database.lower():
            for ds_dir in db_dir.iterdir():
                if ds_dir.is_dir() and ds_dir.name == dataset:
                    return ds_dir
    return None


def _scan_split(split_dir: Path) -> Tuple[bool, str, Dict[str, int]]:
    """Validate a regression/classification split directory."""
    needed = {
        "Xtrain.csv": split_dir / "Xtrain.csv",
        "Ytrain.csv": split_dir / "Ytrain.csv",
        "Xtest.csv": split_dir / "Xtest.csv",
        "Ytest.csv": split_dir / "Ytest.csv",
    }
    missing = [name for name, p in needed.items() if not p.exists()]
    if missing:
        return False, f"missing files: {','.join(missing)}", {}
    try:
        # Read minimal headers to get shapes without loading full data.
        Xtr = pd.read_csv(needed["Xtrain.csv"], sep=";", nrows=1)
        Xte = pd.read_csv(needed["Xtest.csv"], sep=";", nrows=1)
        Ytr = pd.read_csv(needed["Ytrain.csv"], sep=";")
        Yte = pd.read_csv(needed["Ytest.csv"], sep=";")
    except Exception as exc:  # pragma: no cover - I/O dependent
        return False, f"read error: {exc}", {}
    n_train = int(len(Ytr))
    n_test = int(len(Yte))
    p = int(Xtr.shape[1])
    if Xte.shape[1] != p:
        return False, f"feature mismatch train/test ({p} vs {Xte.shape[1]})", {}
    return True, "", {"n_train": n_train, "n_test": n_test, "p": p}


def _aggregate_master_metrics(master_path: Path) -> pd.DataFrame:
    """Pivot master_results.csv to a per-(database,dataset) RMSEP table."""
    df = pd.read_csv(master_path)
    pivot = df.pivot_table(
        index=["database_name", "dataset"],
        columns="model",
        values="RMSEP",
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None
    return pivot


def build_regression_cohort(
    master_path: str = _DEFAULT_MASTER,
    data_root: str = _DEFAULT_REG_ROOT,
    out_path: str = _DEFAULT_OUT_REG,
) -> pd.DataFrame:
    master_path = Path(master_path)
    data_root = Path(data_root)
    if not master_path.exists():
        raise FileNotFoundError(master_path)
    master_pivot = _aggregate_master_metrics(master_path)
    rows: List[Dict] = []
    for _, row in master_pivot.iterrows():
        database = str(row["database_name"])
        dataset = str(row["dataset"])
        split = _find_split_dir(data_root, database, dataset)
        record: Dict[str, object] = {
            "database_name": database,
            "dataset": dataset,
            "ref_rmse_pls": row.get("PLS"),
            "ref_rmse_tabpfn_raw": row.get("TabPFN-Raw"),
            "ref_rmse_tabpfn_opt": row.get("TabPFN-opt"),
            "ref_rmse_cnn": row.get("CNN"),
            "ref_rmse_catboost": row.get("Catboost"),
            "ref_rmse_ridge": row.get("Ridge"),
        }
        if split is None:
            record.update(
                {
                    "status": "skipped",
                    "reason": "dataset folder not found",
                    "n_train": None,
                    "n_test": None,
                    "p": None,
                    "train_path": None,
                    "test_path": None,
                    "ytrain_path": None,
                    "ytest_path": None,
                }
            )
        else:
            ok, reason, shapes = _scan_split(split)
            record.update(
                {
                    "status": "ok" if ok else "skipped",
                    "reason": reason,
                    "n_train": shapes.get("n_train"),
                    "n_test": shapes.get("n_test"),
                    "p": shapes.get("p"),
                    "train_path": str(split / "Xtrain.csv") if ok else None,
                    "test_path": str(split / "Xtest.csv") if ok else None,
                    "ytrain_path": str(split / "Ytrain.csv") if ok else None,
                    "ytest_path": str(split / "Ytest.csv") if ok else None,
                }
            )
        rows.append(record)
    df_out = pd.DataFrame(rows)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out, index=False)
    return df_out


def build_classification_cohort(
    data_root: str = _DEFAULT_CLF_ROOT,
    out_path: str = _DEFAULT_OUT_CLF,
) -> pd.DataFrame:
    data_root = Path(data_root)
    rows: List[Dict] = []
    if not data_root.is_dir():
        df = pd.DataFrame([{"database_name": "", "dataset": "", "status": "skipped", "reason": "classification root missing"}])
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        return df
    for db_dir in sorted(data_root.iterdir()):
        if not db_dir.is_dir():
            continue
        for split_dir in sorted(db_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            ok, reason, shapes = _scan_split(split_dir)
            row: Dict[str, object] = {
                "database_name": db_dir.name,
                "dataset": split_dir.name,
                "status": "ok" if ok else "skipped",
                "reason": reason,
                "n_train": shapes.get("n_train"),
                "n_test": shapes.get("n_test"),
                "p": shapes.get("p"),
                "train_path": str(split_dir / "Xtrain.csv") if ok else None,
                "test_path": str(split_dir / "Xtest.csv") if ok else None,
                "ytrain_path": str(split_dir / "Ytrain.csv") if ok else None,
                "ytest_path": str(split_dir / "Ytest.csv") if ok else None,
            }
            rows.append(row)
    df_out = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    return df_out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build regression and classification cohorts for AOM_v0")
    parser.add_argument("--task", choices=("regression", "classification", "both"), default="both")
    parser.add_argument("--master", default=_DEFAULT_MASTER)
    parser.add_argument("--regression-root", default=_DEFAULT_REG_ROOT)
    parser.add_argument("--classification-root", default=_DEFAULT_CLF_ROOT)
    parser.add_argument("--regression-out", default=_DEFAULT_OUT_REG)
    parser.add_argument("--classification-out", default=_DEFAULT_OUT_CLF)
    args = parser.parse_args(argv)
    if args.task in ("regression", "both"):
        df_reg = build_regression_cohort(args.master, args.regression_root, args.regression_out)
        ok = int((df_reg["status"] == "ok").sum())
        skipped = int((df_reg["status"] == "skipped").sum())
        print(f"[regression] cohort {len(df_reg)} rows, {ok} ok, {skipped} skipped → {args.regression_out}")
    if args.task in ("classification", "both"):
        df_clf = build_classification_cohort(args.classification_root, args.classification_out)
        ok = int((df_clf["status"] == "ok").sum())
        skipped = int((df_clf["status"] == "skipped").sum())
        print(f"[classification] cohort {len(df_clf)} rows, {ok} ok, {skipped} skipped → {args.classification_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
