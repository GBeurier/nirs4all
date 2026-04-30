"""Full 54-dataset multi-kernel benchmark runner with joblib parallelism.

Layout:

- ``--cohort`` defaults to the 54 OK datasets in
  ``bench/AOM_v0/Ridge/benchmark_runs/all57_cohort.csv``.
- ``--variants`` chooses which mkR / MKM / BLUP variants to run.
- ``--n-jobs`` enables process-level parallelism over the dataset axis.
- The result CSV is written incrementally with row-level appends; if it
  already exists, completed `(dataset, variant)` pairs are skipped on
  resume.

Usage:

```bash
.venv/bin/python bench/AOM_v0/Multi-kernel/benchmarks/run_multikernel_full.py \
  --cohort bench/AOM_v0/Ridge/benchmark_runs/all57_cohort.csv \
  --workspace bench/AOM_v0/Multi-kernel/benchmark_runs/all54 \
  --variants Ridge-raw mkR-softmax_cv mkR-softmax_cv-snv mkR-softmax_cv-msc \
             mkR-softmax_cv-asls MKM-reml MKM-reml-asls BLUP-reml-asls \
  --n-jobs 4
```
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve()
BENCHMARKS_DIR = ROOT.parent
MULTI_KERNEL = BENCHMARKS_DIR.parent
AOM_V0 = MULTI_KERNEL.parent
REPO_ROOT = AOM_V0.parent.parent
RIDGE_ROOT = AOM_V0 / "Ridge"
MKR_ROOT = MULTI_KERNEL / "MKR"
MKM_ROOT = MULTI_KERNEL / "MkM"
BLUP_ROOT = MULTI_KERNEL / "Blup"
# Order matters; MKR_ROOT must come first (contains branch_preproc).
for p in (RIDGE_ROOT, BLUP_ROOT, MKM_ROOT, MULTI_KERNEL, MKR_ROOT):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

# Reuse the smoke runner's helpers.
from run_multikernel_smoke import (                                # noqa: E402
    RESULT_COLUMNS, _load_csv_array, _load_csv_target, _pred_metrics,
    _ref_relatives, _run_variant, _append_row,
)

CODE_VERSION = "Multi-kernel/0.1.0-full"


def _locked_append(out_csv: Path, row: dict) -> None:
    """Append one row with an fcntl exclusive lock so concurrent writers
    don't interleave bytes mid-line."""
    import fcntl
    new = not out_csv.exists()
    with open(out_csv, "a", newline="") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
            if new:
                writer.writeheader()
            writer.writerow({k: row.get(k) for k in RESULT_COLUMNS})
            fh.flush()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------


def _process_dataset(
    ds_row: dict,
    variants: list[str],
    repo_root: Path,
    operator_bank: str,
    random_state: int,
    out_csv: Path,
    completed: set[tuple[str, str]],
) -> list[dict]:
    """Run all variants for a single dataset; return list of result rows.

    The ``completed`` set is read by the parent before fork; this function
    skips rows already present.
    """
    db = ds_row["database_name"]
    ds = ds_row["dataset"]
    rows = []
    try:
        X_train = _load_csv_array(repo_root / ds_row["train_path"])
        X_test = _load_csv_array(repo_root / ds_row["test_path"])
        y_train = _load_csv_target(repo_root / ds_row["ytrain_path"])
        y_test = _load_csv_target(repo_root / ds_row["ytest_path"])
    except FileNotFoundError as exc:
        for variant in variants:
            if (ds, variant) in completed:
                continue
            rows.append({
                "dataset_group": db, "dataset": ds,
                "n_train": 0, "n_test": 0, "p": 0,
                "variant": variant, "status": "missing", "error": str(exc),
                "operator_bank": operator_bank,
                "version": CODE_VERSION, "random_state": random_state,
            })
        return rows

    n_train, p = X_train.shape
    n_test = X_test.shape[0]
    ref_pls = float(ds_row.get("ref_rmse_pls") or float("nan"))
    ref_ridge = float(ds_row.get("ref_rmse_ridge") or float("nan"))
    ref_tabpfn_raw = float(ds_row.get("ref_rmse_tabpfn_raw") or float("nan"))
    ref_tabpfn_opt = float(ds_row.get("ref_rmse_tabpfn_opt") or float("nan"))
    ref_cnn = float(ds_row.get("ref_rmse_cnn") or float("nan"))
    ref_cb = float(ds_row.get("ref_rmse_catboost") or float("nan"))

    for variant in variants:
        if (ds, variant) in completed:
            continue
        info = _run_variant(
            variant, X_train, y_train, X_test, y_test,
            operator_bank=operator_bank, random_state=random_state,
        )
        row = {
            "dataset_group": db,
            "dataset": ds,
            "n_train": n_train, "n_test": n_test, "p": p,
            **info,
            "ref_rmse_pls": ref_pls, "ref_rmse_ridge": ref_ridge,
            "ref_rmse_tabpfn_raw": ref_tabpfn_raw,
            "ref_rmse_tabpfn_opt": ref_tabpfn_opt,
            "ref_rmse_cnn": ref_cnn, "ref_rmse_catboost": ref_cb,
            **_ref_relatives(info["rmsep"], ref_pls, ref_ridge, ref_tabpfn_opt),
            "version": CODE_VERSION,
            "random_state": random_state,
        }
        rows.append(row)
    return rows


def _read_completed(out_csv: Path) -> set[tuple[str, str]]:
    if not out_csv.exists():
        return set()
    df = pd.read_csv(out_csv)
    if df.empty:
        return set()
    completed = set()
    for _, r in df.iterrows():
        if r.get("status") == "ok":
            completed.add((str(r["dataset"]), str(r["variant"])))
    return completed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort", type=Path,
        default=Path("bench/AOM_v0/Ridge/benchmark_runs/all57_cohort.csv"),
        help="Path to the cohort CSV (defaults to all57_cohort.csv).",
    )
    parser.add_argument(
        "--workspace", type=Path,
        default=Path("bench/AOM_v0/Multi-kernel/benchmark_runs/all54"),
    )
    parser.add_argument(
        "--variants", nargs="+", default=[
            # Codex-recommended Phase 7 set: 6 variants covering
            # mkR vs MKM, no-branch baseline + 2 most-useful branches.
            "Ridge-raw",                  # external baseline
            "mkR-softmax_cv",             # smoke median winner
            "mkR-softmax_cv-snv",         # branch-best on BEER
            "mkR-softmax_cv-msc",         # second-best on BEER
            "MKM-reml",                   # likelihood reference
            "MKM-reml-asls",              # smoke best on AMYLOSE
            "MKM-reml-msc",               # second branch test for MKM
        ],
    )
    parser.add_argument("--operator-bank", default="compact")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Process-parallelism over datasets (default 1).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after the first N datasets (debug).")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only run datasets with this substring in dataset_group.")
    parser.add_argument("--restrict-status", type=str, default="ok",
                        help="Only datasets with cohort-csv status==value (default 'ok').")
    args = parser.parse_args(argv)

    args.workspace.mkdir(parents=True, exist_ok=True)
    out_csv = args.workspace / "results.csv"

    cohort_df = pd.read_csv(args.cohort)
    if args.restrict_status:
        cohort_df = cohort_df[cohort_df["status"] == args.restrict_status]
    if args.filter:
        cohort_df = cohort_df[
            cohort_df["database_name"].str.contains(args.filter, case=False, na=False)
        ]
    if args.limit:
        cohort_df = cohort_df.head(args.limit)

    cohort = cohort_df.to_dict(orient="records")
    completed = _read_completed(out_csv)
    print(f"[full] cohort: {len(cohort)} datasets, variants: {args.variants}")
    print(f"[full] previously completed (dataset, variant): {len(completed)}")
    print(f"[full] writing to: {out_csv}")
    print(f"[full] n_jobs: {args.n_jobs}")
    print()

    t_start = time.time()

    if args.n_jobs == 1:
        for ds_idx, ds_row in enumerate(cohort):
            print(f"[{ds_idx + 1}/{len(cohort)}] {ds_row['database_name']}/{ds_row['dataset']}")
            rows = _process_dataset(
                ds_row, args.variants, REPO_ROOT,
                args.operator_bank, args.random_state, out_csv, completed,
            )
            for row in rows:
                _append_row(out_csv, row)
                print(f"  {row.get('variant', '?'):<28s}  status={row.get('status', '?'):<7s}  rmsep={row.get('rmsep')}")
            print()
    else:
        # Parallel run with **per-worker incremental writes** via file lock.
        # joblib parallel returns aggregates only at the end; we instead
        # spawn one job per dataset and write rows to the CSV via fcntl
        # exclusive lock so each worker's output appears in real time.
        from joblib import Parallel, delayed

        def _process_and_write(ds_row):
            rows = _process_dataset(
                ds_row, args.variants, REPO_ROOT, args.operator_bank,
                args.random_state, out_csv, completed,
            )
            for row in rows:
                _locked_append(out_csv, row)
            return len(rows)

        list(Parallel(n_jobs=args.n_jobs, prefer="processes", verbose=10)(
            delayed(_process_and_write)(ds_row) for ds_row in cohort
        ))

    t_end = time.time()
    print(f"[full] total wall time: {t_end - t_start:.1f}s")
    print(f"[full] results: {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
