"""
U07 - AOM Panoply: AOM-PLS, AOM-Ridge, AutoSelector, Blender
=============================================================

Use the AOM family with the same user-defined split that drives the
nirs4all pipeline.

This tutorial covers:

* AOM-PLS with operator-bank selection
* AOM-Ridge global selection
* AOMRidgeAutoSelector across candidate variants
* AOMRidgeBlender, the strongest empirical AOM-Ridge recipe
* FastAOM for faster screened chain calibration

Duration: ~2-5 minutes
Difficulty: *****
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

import nirs4all
from nirs4all.operators.models import (
    AOMPLSRegressor,
    AOMRidgeAutoSelector,
    AOMRidgeBlender,
    AOMRidgeRegressor,
    FastAOMConfig,
    FastAOMPLSRidge,
)
from nirs4all.operators.transforms import StandardNormalVariate

EXAMPLES_ROOT = Path(__file__).resolve().parents[2]


def _load_example_helpers():
    if str(EXAMPLES_ROOT) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_ROOT))
    from example_utils import get_example_output_path, validate_result

    return get_example_output_path, validate_result


get_example_output_path, validate_result = _load_example_helpers()


parser = argparse.ArgumentParser(description="U07 AOM model panoply")
parser.add_argument("--plots", action="store_true", help="Generate plots")
parser.add_argument("--show", action="store_true", help="Display plots interactively")
args = parser.parse_args()


print("\n" + "=" * 60)
print("U07 - AOM Panoply")
print("=" * 60)
print("""
This example uses one user-defined KFold split for the whole pipeline.
For AOM models, nirs4all forwards these same folds into the estimator's
internal CV when train_params.use_pipeline_folds_for_aom is set.
""")


QUICK_AOM_RIDGE_CANDIDATES = [
    {
        "label": "AOMRidge-superblock-identity",
        "selection": "superblock",
        "operator_bank": "identity",
        "block_scaling": "none",
        "extra": {
            "alpha": 1.0,
        },
    },
    {
        "label": "AOMRidge-superblock-compact",
        "selection": "superblock",
        "operator_bank": "compact",
        "block_scaling": "none",
        "extra": {
            "alpha": 1.0,
        },
    },
]

AOM_SPLIT_REQUIRED = {"use_pipeline_folds_for_aom": "required"}

# Replace this step with {"split": GroupKFold(n_splits=4), "group_by": "sample_id"}
# when grouped samples must stay together. The AOM models below will receive that
# grouped split too.
user_split = KFold(n_splits=4, shuffle=True, random_state=42)

pipeline = [
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},
    StandardNormalVariate(),
    user_split,
    {"model": PLSRegression(n_components=5, scale=False), "name": "PLS-5"},
    {
        "model": AOMPLSRegressor(
            n_components="auto",
            max_components=5,
            operator_bank="compact",
            criterion="cv",
            cv=4,
            random_state=42,
        ),
        "name": "AOMPLS-compact",
        "train_params": AOM_SPLIT_REQUIRED,
    },
    {
        "model": AOMRidgeRegressor(
            selection="global",
            operator_bank="compact",
            block_scaling="none",
            alpha=1.0,
            random_state=42,
        ),
        "name": "AOMRidge-global",
        "train_params": AOM_SPLIT_REQUIRED,
    },
    {
        "model": AOMRidgeAutoSelector(
            candidates=QUICK_AOM_RIDGE_CANDIDATES,
            outer_cv=4,
            inner_cv=4,
            outer_cv_kind="kfold",
            random_state=42,
            n_jobs=1,
        ),
        "name": "AOMRidge-auto",
        "train_params": AOM_SPLIT_REQUIRED,
    },
    {
        "model": AOMRidgeBlender(
            candidates=QUICK_AOM_RIDGE_CANDIDATES,
            outer_cv=4,
            inner_cv=4,
            outer_cv_kind="kfold",
            regularizer=0.01,
            random_state=42,
            n_jobs=1,
        ),
        "name": "AOMRidge-blender",
        "train_params": AOM_SPLIT_REQUIRED,
    },
    {
        "model": FastAOMPLSRidge(
            config=FastAOMConfig(
                model="sparse_mkr",
                primitive_bank="compact",
                max_chain_depth=2,
                top_global=12,
                sparse_mkr_max_chains=4,
                random_state=42,
            )
        ),
        "name": "FastAOM-sparse",
    },
]

result = nirs4all.run(
    pipeline=pipeline,
    dataset=str(EXAMPLES_ROOT / "sample_data" / "regression"),
    name="U07_AOM_Panoply",
    verbose=1,
    random_state=42,
)
validate_result(result, "U07_AOM_Panoply", min_predictions=5)

print("\nTop AOM panoply results:")
top_rows = result.top(n=10, display_metrics=["rmse", "r2"])
for rank, pred in enumerate(top_rows, 1):
    model_name = str(pred.get("model_name", "Unknown"))
    rmse = pred.get("rmse", float("nan"))
    r2 = pred.get("r2", float("nan"))
    print(f"  {rank:2d}. {model_name:<24s} RMSE={rmse:.4f}  R2={r2:.4f}")

print("""
Split note:
  AOMPLSRegressor receives cv_splitter from the pipeline.
  AOMRidgeRegressor receives cv from the pipeline.
  AOMRidgeAutoSelector and AOMRidgeBlender receive both outer_cv and inner_cv.
  Set train_params.use_pipeline_folds_for_aom='required' to fail fast if a
  future pipeline cannot provide compatible folds.
""")

if args.plots:
    labels = [str(pred.get("model_name", "Unknown")) for pred in top_rows]
    rmses = [float(pred.get("rmse", 0.0)) for pred in top_rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(labels)), rmses, color="#2f6f73")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("U07 AOM panoply comparison")
    fig.tight_layout()
    if args.show:
        plt.show()
    else:
        out = get_example_output_path("U07_aom_panoply", "aom_panoply_rmse.png")
        fig.savefig(out, dpi=130)
        print(f"Plot saved to {out}")

print("\nDone.")
