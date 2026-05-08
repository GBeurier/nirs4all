"""FCK static-bank smoke runner for the ``fast12_transfer_core`` cohort.

Runs five FCK-static pipelines + a PLS-baseline reference on each of the
twelve datasets in ``bench/Subset_analysis/rethought_subsets.json`` and
emits a CSV with RMSEP, runtime, and the reference comparisons.

This is a B-side, plan-only smoke runner — it is **not** the canonical
benchmark harness. Once Agent C delivers ``bench/harness/run_benchmark.py``
the same five pipelines will be re-run there for scenario consumption.

Usage
-----
    cd bench/fck_pls
    python run_smoke_fast12.py [--out OUT] [--datasets foo bar ...] [--n-jobs N]

The output CSV is:
    bench/fck_pls/runs/smoke_fast12/results.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, Pipeline

from nirs4all.operators.transforms import (
    ASLSBaseline,
    FCKStaticTransformer,
    StandardNormalVariate,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBSETS_JSON = REPO_ROOT / "bench" / "Subset_analysis" / "rethought_subsets.json"
COHORT_CSV = REPO_ROOT / "bench" / "AOM_v0" / "benchmarks" / "cohort_regression.csv"
DEFAULT_OUT = REPO_ROOT / "bench" / "fck_pls" / "runs" / "smoke_fast12" / "results.csv"

# Reuse the proven dataset loader from nicon_v2 to avoid path/sep drift.
sys.path.insert(0, str(REPO_ROOT / "bench" / "nicon_v2"))
from nicon_v2.datasets import (  # noqa: E402  (import after sys.path edit)
    DatasetSpec,
    load_cohort_manifest,
    load_dataset,
)


def _load_cohort_names(cohort: str) -> list[str]:
    if not SUBSETS_JSON.is_file():
        raise FileNotFoundError(f"Cohort definition not found: {SUBSETS_JSON}")
    payload = json.loads(SUBSETS_JSON.read_text())
    if cohort == "fast12":
        return list(payload["subsets"]["fast12_transfer_core"]["datasets"])
    if cohort == "audit20":
        return list(payload["subsets"]["audit20_transfer_core"]["datasets"])
    raise ValueError(f"Unknown cohort {cohort!r}; expected 'fast12' or 'audit20'.")


def _load_fast12_names() -> list[str]:
    return _load_cohort_names("fast12")


def _select_specs(names: list[str]) -> list[DatasetSpec]:
    all_specs = load_cohort_manifest(cohort="all", csv_path=COHORT_CSV)
    by_name = {s.dataset: s for s in all_specs}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise KeyError(
            f"{len(missing)} datasets in fast12_transfer_core are missing from "
            f"the cohort manifest: {missing}"
        )
    return [by_name[n] for n in names]


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


def _aompls(n_components: int = 10):
    """Lazy AOM-PLS import so the script still runs if AOM_v0 is broken."""
    from nirs4all.operators.models import AOMPLSRegressor  # type: ignore

    return AOMPLSRegressor(n_components=n_components)


def _build_pipelines() -> dict[str, Pipeline]:
    fck = FCKStaticTransformer  # default 16 filters

    pipes: dict[str, Pipeline] = {}
    pipes["PLS-baseline"] = Pipeline(
        [("pls", PLSRegression(n_components=10))]
    )
    pipes["FCK-PLS"] = Pipeline(
        [("fck", fck()), ("pls", PLSRegression(n_components=10))]
    )
    pipes["FCK-Ridge"] = Pipeline(
        [("fck", fck()), ("ridge", Ridge(alpha=1.0))]
    )
    pipes["FCK-AOMPLS"] = Pipeline(
        [("fck", fck()), ("aompls", _aompls(10))]
    )
    # ASLS-FCK-PLS uses asymmetric-LS baseline correction. concat[SNV, FCK] is a
    # plain sklearn FeatureUnion, since nirs4all's native ``concat_transform``
    # keyword only resolves inside a nirs4all pipeline (not a sklearn Pipeline).
    pipes["ASLS-FCK-PLS"] = Pipeline(
        [("asls", ASLSBaseline()), ("fck", fck()), ("pls", PLSRegression(n_components=10))]
    )
    pipes["Concat-SNV-FCK-AOMPLS"] = Pipeline(
        [
            (
                "concat",
                FeatureUnion(
                    [
                        ("snv", StandardNormalVariate()),
                        ("fck", fck()),
                    ]
                ),
            ),
            ("aompls", _aompls(10)),
        ]
    )
    return pipes


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()) ** 2)))


def _run_one(spec: DatasetSpec, pipe_name: str, pipe: Pipeline) -> dict:
    row: dict = {"dataset": spec.dataset, "database_name": spec.database_name, "pipeline": pipe_name,
                 "n_train": spec.n_train, "n_test": spec.n_test, "n_features": spec.n_features,
                 "ref_rmse_pls": spec.ref_rmse_pls,
                 "ref_rmse_paper_ridge": spec.ref_rmse_paper_ridge,
                 "ref_rmse_tabpfn_raw": spec.ref_rmse_tabpfn_raw,
                 "ref_rmse_tabpfn_opt": spec.ref_rmse_tabpfn_opt,
                 "ref_rmse_cnn": spec.ref_rmse_cnn,
                 "ref_rmse_catboost": spec.ref_rmse_catboost,
                 "ref_rmse_aom_ridge_curated_best": spec.ref_rmse_aom_ridge_curated_best}
    try:
        X_train, y_train, X_test, y_test = load_dataset(spec)
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_t = time.perf_counter() - t0
        t1 = time.perf_counter()
        y_pred = pipe.predict(X_test)
        predict_t = time.perf_counter() - t1
        rmsep = _rmse(y_test, y_pred)
        row.update(
            status="OK",
            error_message="",
            rmsep=rmsep,
            fit_time_s=fit_t,
            predict_time_s=predict_t,
        )
        for ref in (
            "ref_rmse_pls",
            "ref_rmse_paper_ridge",
            "ref_rmse_tabpfn_raw",
            "ref_rmse_tabpfn_opt",
            "ref_rmse_cnn",
            "ref_rmse_catboost",
            "ref_rmse_aom_ridge_curated_best",
        ):
            ref_v = row[ref]
            row[f"relative_rmsep_vs_{ref[len('ref_rmse_'):]}"] = (
                float(rmsep / ref_v - 1.0) if ref_v else None
            )
    except Exception as e:  # noqa: BLE001 — capture all failures for the smoke
        row.update(
            status="ERROR",
            error_message=f"{type(e).__name__}: {e}",
            rmsep=None,
            fit_time_s=None,
            predict_time_s=None,
        )
        traceback.print_exc()
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="FCK static benchmark runner (fast12 / audit20)")
    parser.add_argument("--cohort", default="fast12", choices=("fast12", "audit20"),
                        help="Which cohort to run. Default: fast12")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output CSV path. Default depends on cohort.")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Restrict to specific dataset names (default: full cohort)")
    parser.add_argument("--pipelines", nargs="*", default=None,
                        help="Subset of pipeline names to run (default: all)")
    args = parser.parse_args()

    if args.out is None:
        args.out = (REPO_ROOT / "bench" / "fck_pls" / "runs"
                    / f"smoke_{args.cohort}" / "results.csv")

    names = _load_cohort_names(args.cohort)
    if args.datasets:
        names = [n for n in names if n in set(args.datasets)]
    specs = _select_specs(names)
    pipelines = _build_pipelines()
    if args.pipelines:
        pipelines = {k: v for k, v in pipelines.items() if k in set(args.pipelines)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Resume: load any pre-existing rows and skip (dataset, pipeline) pairs we
    # already have. This makes the runner safe to relaunch after a timeout.
    rows: list[dict] = []
    done: set[tuple[str, str]] = set()
    if args.out.is_file():
        prior = pd.read_csv(args.out)
        rows = prior.to_dict(orient="records")
        done = {(r["dataset"], r["pipeline"]) for r in rows}
        print(f"[FCK-smoke] resuming from {args.out}: {len(done)} (dataset, pipeline) pairs already done",
              flush=True)
    for spec in specs:
        for pipe_name, pipe in pipelines.items():
            if (spec.dataset, pipe_name) in done:
                continue
            print(f"[FCK-smoke] {spec.dataset:50s}  {pipe_name}", flush=True)
            row = _run_one(spec, pipe_name, pipe)
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    print(f"\n[FCK-smoke] wrote {len(df)} rows to {args.out}")

    # Quick console summary
    if "rmsep" in df.columns:
        ok = df[df["status"] == "OK"]
        print("\n[FCK-smoke] median rmsep / pipeline (OK rows only):")
        for pipe_name, sub in ok.groupby("pipeline"):
            print(f"  {pipe_name:25s}  median rmsep = {sub['rmsep'].median():.4f}  "
                  f"(n={len(sub)})")
    return 0


# Silence unused-asdict-import warning when the script is imported.
_ = asdict


if __name__ == "__main__":
    sys.exit(main())
