"""Phase 8 meta-selector v2: include TabPFN + AOM-Ridge as candidates.

Same LOO meta-classifier as run_meta_selector.py but with the heterogeneous
expert pool. Compare:
- best multi-view single (oracle of 5 multi-view variants)
- meta-selector over 6 multi-view (existing 15/58)
- meta-selector over multi-view ∪ {TabPFN, AOM-Ridge}
- oracle of multi-view ∪ {TabPFN, AOM-Ridge}

Outputs `bench/AOM_v0/multiview/results/meta_selector_full57_v2.csv`.
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

from multiview.meta_selector import leave_one_out_select, selector_rmsep  # noqa: E402


VARIANTS_MV = [
    "moe-preproc-soft-pls-compact",
    "moe-view-soft-pls",
    "moe-view-soft-K5",
    "lazy-V2-AOM-combined-compact-holdout",
    "lazy-V1-POP-blocks3-holdout",
    "AOM-PLS-compact-numpy",
]

HETERO_VARIANTS = ["tabpfn-standalone", "aom-ridge-standalone"]
VARIANTS_HETERO = VARIANTS_MV + HETERO_VARIANTS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", default="bench/AOM_v0/multiview/results")
    parser.add_argument(
        "--cohort", default="bench/AOM_v0/benchmarks/cohort_regression.csv",
    )
    parser.add_argument(
        "--full57", default="bench/AOM_v0/multiview/results/full57.csv",
    )
    parser.add_argument(
        "--features", default="bench/AOM_v0/multiview/results/_features_cache.csv",
    )
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    out_path = workspace / "meta_selector_full57_v2.csv"

    full = pd.read_csv(args.full57)
    full = full[full["status"] == "ok"].copy()
    full["rmsep"] = pd.to_numeric(full["rmsep"], errors="coerce")
    rmsep = full.pivot_table(index="dataset", columns="variant", values="rmsep", aggfunc="mean")
    cohort = pd.read_csv(args.cohort).set_index("dataset")

    feature_matrix = pd.read_csv(args.features, index_col="dataset")

    # Drop datasets where we don't have all hetero results
    avail_hetero = HETERO_VARIANTS
    avail_mv = [v for v in VARIANTS_MV if v in rmsep.columns]
    avail_full = [v for v in VARIANTS_HETERO if v in rmsep.columns]
    print(f"[meta-v2] available variants:")
    print(f"  multi-view: {avail_mv}")
    print(f"  hetero:     {[v for v in avail_hetero if v in rmsep.columns]}")
    print(f"  union:      {avail_full}")

    keep = rmsep[avail_full].dropna(thresh=len(avail_full) - 0)  # all-non-NaN rows
    keep_partial = rmsep[avail_full].dropna(thresh=4)  # at least 4 of N variants
    common = keep.index
    print(f"[meta-v2] {len(common)} datasets with all variants available")
    print(f"[meta-v2] {len(keep_partial.index)} datasets with at least 4")

    # Use partial-availability set; meta-selector handles NaN by skipping
    rmsep_keep = rmsep.loc[keep_partial.index, avail_full]
    feature_keep = feature_matrix.reindex(rmsep_keep.index)

    print("\n[meta-v2] LOO meta-selector (logreg) on multi-view only...")
    pred_mv_only = leave_one_out_select(
        feature_keep, rmsep_keep, [v for v in avail_mv if v in rmsep_keep.columns],
        classifier="logreg",
    )
    print("[meta-v2] LOO meta-selector (logreg) on hetero union...")
    pred_hetero = leave_one_out_select(
        feature_keep, rmsep_keep, avail_full, classifier="logreg",
    )
    print("[meta-v2] LOO meta-selector (rf) on hetero union...")
    pred_hetero_rf = leave_one_out_select(
        feature_keep, rmsep_keep, avail_full, classifier="rf",
    )

    # Build summary
    cmp = pd.DataFrame(index=rmsep_keep.index)
    cmp["meta_mv_only_variant"] = pred_mv_only
    cmp["meta_mv_only_rmsep"] = selector_rmsep(pred_mv_only, rmsep_keep)
    cmp["meta_hetero_logreg_variant"] = pred_hetero
    cmp["meta_hetero_logreg_rmsep"] = selector_rmsep(pred_hetero, rmsep_keep)
    cmp["meta_hetero_rf_variant"] = pred_hetero_rf
    cmp["meta_hetero_rf_rmsep"] = selector_rmsep(pred_hetero_rf, rmsep_keep)
    cmp["oracle_mv_rmsep"] = rmsep_keep[[v for v in avail_mv if v in rmsep_keep.columns]].min(axis=1)
    cmp["oracle_hetero_rmsep"] = rmsep_keep[avail_full].min(axis=1)
    for v in avail_full:
        cmp[v] = rmsep_keep[v]
    cmp["tabpfn_opt_ref"] = cohort.loc[rmsep_keep.index, "ref_rmse_tabpfn_opt"].astype(float)
    cmp["pls_ref"] = cohort.loc[rmsep_keep.index, "ref_rmse_pls"].astype(float)
    cmp.to_csv(out_path)
    print(f"\n[meta-v2] saved {len(cmp)} rows -> {out_path}")

    # Win count
    print("\n=== Wins vs TabPFN-opt ===")
    valid = cmp.dropna(subset=["tabpfn_opt_ref"])
    rows = []
    for col_label, col in [
        ("oracle_mv", "oracle_mv_rmsep"),
        ("oracle_hetero", "oracle_hetero_rmsep"),
        ("meta_mv_only", "meta_mv_only_rmsep"),
        ("meta_hetero_logreg", "meta_hetero_logreg_rmsep"),
        ("meta_hetero_rf", "meta_hetero_rf_rmsep"),
    ] + [(v, v) for v in avail_full]:
        valid_v = valid.dropna(subset=[col])
        wins = (valid_v[col] < valid_v["tabpfn_opt_ref"]).sum()
        med_rel = float((valid_v[col] / valid_v["pls_ref"]).median())
        rows.append({
            "variant": col_label,
            "wins_vs_tabpfn": int(wins),
            "n": int(len(valid_v)),
            "median_rel_rmsep_vs_pls": round(med_rel, 4),
        })
    summary = pd.DataFrame(rows).sort_values("wins_vs_tabpfn", ascending=False)
    print(summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
