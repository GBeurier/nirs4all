"""Phase 7.5: meta-learner per-dataset variant selection.

Extracts features from each cohort dataset, builds a leave-one-out
classifier that predicts the best multi-view variant from features,
and evaluates the resulting per-dataset RMSEP against TabPFN-opt.

Compares:
- Best single variant (current top: moe-view-soft-pls K=3, 14/58)
- Oracle multi-view (per-dataset best, ceiling 20/58)
- Meta-selector logreg
- Meta-selector rf

Outputs `bench/AOM_v0/multiview/results/meta_selector_full57.csv`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(os.path.abspath(__file__)).resolve()
_MULTIVIEW_ROOT = _HERE.parent.parent
_AOM_ROOT = _MULTIVIEW_ROOT.parent
if str(_AOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_AOM_ROOT))
if str(_MULTIVIEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_MULTIVIEW_ROOT))

from benchmarks.run_smoke4 import _load_csv_array, _load_csv_target  # noqa: E402
from multiview.meta_selector import (  # noqa: E402
    extract_features,
    leave_one_out_select,
    selector_rmsep,
)


VARIANTS = [
    "moe-preproc-soft-pls-compact",
    "moe-view-soft-pls",
    "moe-view-soft-K5",
    "lazy-V2-AOM-combined-compact-holdout",
    "lazy-V1-POP-blocks3-holdout",
    "AOM-PLS-compact-numpy",
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", default="bench/AOM_v0/multiview/results")
    parser.add_argument(
        "--cohort", default="bench/AOM_v0/benchmarks/cohort_regression.csv",
    )
    parser.add_argument(
        "--full57", default="bench/AOM_v0/multiview/results/full57.csv",
    )
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    out_path = workspace / "meta_selector_full57.csv"

    full = pd.read_csv(args.full57)
    full = full[full["status"] == "ok"].copy()
    full["rmsep"] = pd.to_numeric(full["rmsep"], errors="coerce")
    rmsep = full.pivot_table(index="dataset", columns="variant", values="rmsep", aggfunc="mean")
    print(f"[meta] rmsep matrix shape: {rmsep.shape}")

    cohort = pd.read_csv(args.cohort).set_index("dataset")
    valid = cohort[cohort["status"] == "ok"]

    # Extract features for every dataset that has at least 1 variant in rmsep matrix
    print("[meta] extracting features...")
    rows = []
    for ds in rmsep.index:
        if ds not in valid.index:
            continue
        try:
            row = valid.loc[ds]
            Xtr = _load_csv_array(row["train_path"])
            ytr = _load_csv_target(row["ytrain_path"]).astype(float)
            feats = extract_features(Xtr, ytr)
            feats["dataset"] = ds
            rows.append(feats)
            print(f"  ok: {ds:<48s} (n={Xtr.shape[0]}, p={Xtr.shape[1]})")
        except Exception as exc:
            print(f"  skip {ds}: {exc}")
    feature_matrix = pd.DataFrame(rows).set_index("dataset")

    # Leave-one-out predictions
    print(f"\n[meta] LOO meta-selection (logreg)...")
    pred_logreg = leave_one_out_select(
        feature_matrix=feature_matrix,
        rmsep_matrix=rmsep,
        variants=VARIANTS,
        classifier="logreg",
    )
    print(f"\n[meta] LOO meta-selection (rf)...")
    pred_rf = leave_one_out_select(
        feature_matrix=feature_matrix,
        rmsep_matrix=rmsep,
        variants=VARIANTS,
        classifier="rf",
    )

    # Build comparison table
    common = pred_logreg.index
    cmp = pd.DataFrame(index=common)
    cmp["meta_logreg_variant"] = pred_logreg
    cmp["meta_rf_variant"] = pred_rf
    cmp["meta_logreg_rmsep"] = selector_rmsep(pred_logreg, rmsep)
    cmp["meta_rf_rmsep"] = selector_rmsep(pred_rf, rmsep)
    cmp["oracle_rmsep"] = rmsep.loc[common, VARIANTS].min(axis=1)
    cmp["oracle_variant"] = rmsep.loc[common, VARIANTS].idxmin(axis=1)
    for v in VARIANTS:
        cmp[v] = rmsep.loc[common, v]
    cmp["tabpfn_opt"] = cohort.loc[common, "ref_rmse_tabpfn_opt"].astype(float)
    cmp["pls_standard"] = cohort.loc[common, "ref_rmse_pls"].astype(float)
    cmp["ridge_ref"] = cohort.loc[common, "ref_rmse_ridge"].astype(float)
    cmp.to_csv(out_path)
    print(f"\n[meta] saved {len(cmp)} rows -> {out_path}")

    # Win count summary
    print("\n=== Wins vs TabPFN-opt ===")
    valid_tab = cmp.dropna(subset=["tabpfn_opt"])
    rows = []
    for v in VARIANTS + ["oracle_rmsep", "meta_logreg_rmsep", "meta_rf_rmsep"]:
        if v not in valid_tab.columns:
            continue
        wins = (valid_tab[v] < valid_tab["tabpfn_opt"]).sum()
        med_rel = float((valid_tab[v] / valid_tab["pls_standard"]).median())
        rows.append({
            "variant": v,
            "wins_vs_tabpfn": int(wins),
            "median_rel_rmsep_vs_pls": round(med_rel, 4),
        })
    summary = pd.DataFrame(rows).sort_values("wins_vs_tabpfn", ascending=False)
    print(summary.to_string(index=False))

    # Per-prediction variant distribution
    print("\n=== Meta-logreg variant distribution ===")
    print(pred_logreg.value_counts().to_string())
    print("\n=== Meta-rf variant distribution ===")
    print(pred_rf.value_counts().to_string())

    print("\n=== Oracle variant distribution ===")
    print(cmp["oracle_variant"].value_counts().to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
