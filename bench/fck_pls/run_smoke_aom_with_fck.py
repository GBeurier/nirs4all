"""AOM-PLS with FCK kernels integrated into the operator bank — smoke runner.

Tests the bullet from `bench/model_exploration_review.md` §6 not yet
covered by the static FCK pipelines: **AOM compact + FCK filters → PLS**.
The 8 FCK kernels (α ∈ {0.5, 1.0, 1.5, 2.0} × scale ∈ {1, 2}, kernel_size
= 31, σ = 3.0, zero-padded boundaries) are added to AOM-PLS's compact
bank (9 ops → 17 ops). AOM-PLS's per-component selector picks
dynamically across the unified vocabulary.

Pipelines compared on the chosen cohort (default: fast12_transfer_core):

| Pipeline | Detail |
|---|---|
| PLS-baseline | reference (n fixed at 10 components) |
| AOMPLS-compact | reference: compact bank only (9 ops) |
| AOMPLS-compact-with-fck | NEW: compact + 8 FCK ops (17 ops) |
| AOMPLS-fck-only | NEW: 8 FCK ops only (auto-prepended identity = 9 ops) |
| FCK-AOMPLS-static | B-side reference: static FCK preprocessing → AOMPLS-compact |

Usage::

    cd bench/fck_pls
    python run_smoke_aom_with_fck.py [--cohort {fast12,audit20}] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBSETS_JSON = REPO_ROOT / "bench" / "Subset_analysis" / "rethought_subsets.json"
COHORT_CSV = REPO_ROOT / "bench" / "AOM_v0" / "benchmarks" / "cohort_regression.csv"

for p in (
    REPO_ROOT / "bench" / "nicon_v2",
    REPO_ROOT / "bench" / "AOM_v0",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from aompls.estimators import AOMPLSRegressor as AOMV0Regressor  # noqa: E402
from nicon_v2.datasets import (  # noqa: E402
    DatasetSpec,
    load_cohort_manifest,
    load_dataset,
)

from nirs4all.operators.transforms import FCKStaticTransformer  # noqa: E402

# Lazy import for the FCK residual head (D-B-016).
_HERE_FCK = Path(__file__).resolve().parent
if str(_HERE_FCK) not in sys.path:
    sys.path.insert(0, str(_HERE_FCK))
from fck_residual import FCKResidualRegressor  # noqa: E402

# AOM-Ridge wrapper (D-B-017). Available because bench/AOM_v0/Ridge is on
# sys.path already (REPO_ROOT/bench/AOM_v0 above also resolves the Ridge
# subpackage via aomridge namespace).
_RIDGE_PATH = REPO_ROOT / "bench" / "AOM_v0" / "Ridge"
if str(_RIDGE_PATH) not in sys.path:
    sys.path.insert(0, str(_RIDGE_PATH))
from aomridge.aom_ridge_pls import AOMRidgePLS, AOMRidgePLSCV  # noqa: E402


def _load_cohort_names(cohort: str) -> list[str]:
    payload = json.loads(SUBSETS_JSON.read_text())
    if cohort == "fast12":
        return list(payload["subsets"]["fast12_transfer_core"]["datasets"])
    if cohort == "audit20":
        return list(payload["subsets"]["audit20_transfer_core"]["datasets"])
    if cohort == "full57":
        # Full-57 = the 61 OK datasets in cohort_regression.csv minus the 4
        # AOM-Ridge coverage holes (per rethought_subsets.json).
        manifest = pd.read_csv(COHORT_CSV)
        ok = manifest[manifest["status"] == "ok"]
        holes = set(payload.get("aom_ridge_coverage_holes_not_in_primary_subsets", []))
        return sorted(d for d in ok["dataset"].tolist() if d not in holes)
    raise ValueError(f"Unknown cohort {cohort!r}")


def _select_specs(names: list[str]) -> list[DatasetSpec]:
    all_specs = load_cohort_manifest(cohort="all", csv_path=COHORT_CSV)
    by_name = {s.dataset: s for s in all_specs}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise KeyError(f"missing datasets: {missing}")
    return [by_name[n] for n in names]


def _build_pipelines() -> dict[str, object]:
    pipes: dict[str, object] = {}
    pipes["PLS-baseline"] = Pipeline([("pls", PLSRegression(n_components=10))])
    pipes["AOMPLS-compact"] = AOMV0Regressor(
        n_components="auto", max_components=15,
        engine="simpls_covariance", selection="global",
        criterion="cv", cv=5, operator_bank="compact",
    )
    pipes["AOMPLS-compact-with-fck"] = AOMV0Regressor(
        n_components="auto", max_components=15,
        engine="simpls_covariance", selection="global",
        criterion="cv", cv=5, operator_bank="compact_with_fck",
    )
    pipes["AOMPLS-fck-only"] = AOMV0Regressor(
        n_components="auto", max_components=15,
        engine="simpls_covariance", selection="global",
        criterion="cv", cv=5, operator_bank="fck_compact",
    )
    pipes["FCK-AOMPLS-static"] = Pipeline([
        ("fck", FCKStaticTransformer()),
        ("aompls", AOMV0Regressor(
            n_components="auto", max_components=15,
            engine="simpls_covariance", selection="global",
            criterion="cv", cv=5, operator_bank="compact",
        )),
    ])
    # D-B-016 — FCKResidualRegressor with AOMPLS-compact teacher and Ridge head.
    from sklearn.linear_model import Ridge
    pipes["FCKResidual-AOMPLS"] = FCKResidualRegressor(
        teacher=AOMV0Regressor(
            n_components="auto", max_components=15,
            engine="simpls_covariance", selection="global",
            criterion="cv", cv=5, operator_bank="compact",
        ),
        fck=FCKStaticTransformer(),
        residual_head=Ridge(alpha=1.0),
        shrinkage_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
        oof_n_folds=5,
        val_fraction=0.2,
        random_state=0,
        catastrophic_threshold=0.5,
    )
    # D-B-017 — AOM-Ridge with FCK in the bank (mirror of D-B-014 on the
    # AOM-Ridge package). AOMRidgePLS already accepts the `compact_with_fck`
    # bank registered in `aompls.banks.bank_by_name`, so no new bank code
    # is needed; we just instantiate the standard estimator with the
    # FCK-augmented bank.
    pipes["AOMRidgePLS-compact"] = AOMRidgePLS(
        operator_bank="compact",
        n_components=10,
        ridge_alpha=1.0,
        cv=5,
    )
    pipes["AOMRidgePLS-compact-with-fck"] = AOMRidgePLS(
        operator_bank="compact_with_fck",
        n_components=10,
        ridge_alpha=1.0,
        cv=5,
    )
    # D-B-017b — CV-tuned AOMRidgePLSCV. The default-hyperparam AOMRidgePLS
    # comparison vs `aom_ridge_curated_best` was unfair (the curated best
    # uses tuned alpha). AOMRidgePLSCV does its own grid search over
    # n_components × ridge_alpha so the comparison is on equal footing.
    pipes["AOMRidgePLSCV-compact"] = AOMRidgePLSCV(
        operator_bank="compact",
        cv=5,
    )
    pipes["AOMRidgePLSCV-compact-with-fck"] = AOMRidgePLSCV(
        operator_bank="compact_with_fck",
        cv=5,
    )
    return pipes


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()) ** 2)))


def _selected_operator_summary(estimator: object) -> str:
    if isinstance(estimator, Pipeline):
        last = estimator.steps[-1][1]
    else:
        last = estimator
    fn = getattr(last, "get_selected_operators", None)
    if not callable(fn):
        return ""
    try:
        ops = fn()
    except Exception:
        return ""
    return ",".join(str(op) for op in ops) if ops else ""


def _run_one(spec: DatasetSpec, pipe_name: str, pipe: object) -> dict:
    row: dict = {
        "dataset": spec.dataset,
        "database_name": spec.database_name,
        "pipeline": pipe_name,
        "n_train": spec.n_train,
        "n_test": spec.n_test,
        "n_features": spec.n_features,
        "ref_rmse_pls": spec.ref_rmse_pls,
        "ref_rmse_paper_ridge": spec.ref_rmse_paper_ridge,
        "ref_rmse_tabpfn_raw": spec.ref_rmse_tabpfn_raw,
        "ref_rmse_tabpfn_opt": spec.ref_rmse_tabpfn_opt,
        "ref_rmse_cnn": spec.ref_rmse_cnn,
        "ref_rmse_catboost": spec.ref_rmse_catboost,
        "ref_rmse_aom_ridge_curated_best": spec.ref_rmse_aom_ridge_curated_best,
    }
    try:
        X_train, y_train, X_test, y_test = load_dataset(spec)
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_t = time.perf_counter() - t0
        t1 = time.perf_counter()
        y_pred = pipe.predict(X_test).ravel()
        predict_t = time.perf_counter() - t1
        rmsep = _rmse(y_test, y_pred)
        row.update(
            status="OK",
            error_message="",
            rmsep=rmsep,
            fit_time_s=fit_t,
            predict_time_s=predict_t,
            selected_operators=_selected_operator_summary(pipe),
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
    except Exception as e:  # noqa: BLE001
        row.update(
            status="ERROR",
            error_message=f"{type(e).__name__}: {e}",
            rmsep=None,
            fit_time_s=None,
            predict_time_s=None,
            selected_operators="",
        )
        traceback.print_exc()
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="AOM-PLS+FCK bank smoke runner")
    parser.add_argument("--cohort", default="fast12", choices=("fast12", "audit20", "full57"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--pipelines", nargs="*", default=None)
    args = parser.parse_args()
    if args.out is None:
        args.out = (REPO_ROOT / "bench" / "fck_pls" / "runs"
                    / f"aom_with_fck_{args.cohort}" / "results.csv")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    names = _load_cohort_names(args.cohort)
    if args.datasets:
        names = [n for n in names if n in set(args.datasets)]
    specs = _select_specs(names)
    pipelines = _build_pipelines()
    if args.pipelines:
        pipelines = {k: v for k, v in pipelines.items() if k in set(args.pipelines)}

    rows: list[dict] = []
    done: set[tuple[str, str]] = set()
    if args.out.is_file():
        prior = pd.read_csv(args.out)
        rows = prior.to_dict(orient="records")
        done = {(r["dataset"], r["pipeline"]) for r in rows}
        print(f"[aom-with-fck] resuming from {args.out}: {len(done)} pairs already done")

    for spec in specs:
        for pipe_name, pipe in pipelines.items():
            if (spec.dataset, pipe_name) in done:
                continue
            print(f"[aom-with-fck] {spec.dataset:50s}  {pipe_name}", flush=True)
            row = _run_one(spec, pipe_name, pipe)
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    print(f"\n[aom-with-fck] wrote {len(df)} rows to {args.out}")
    ok = df[df["status"] == "OK"] if "status" in df.columns else df
    print("\n[aom-with-fck] median rmsep / pipeline (OK rows only):")
    for pipe_name, sub in ok.groupby("pipeline"):
        print(f"  {pipe_name:25s}  median rmsep = {sub['rmsep'].median():.4f}  (n={len(sub)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
