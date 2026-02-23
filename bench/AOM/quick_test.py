"""Quick single-dataset test for iterating on AOM prototypes.
Run: python bench/AOM/quick_test.py
Tests on Beer (60 samples, fast) and optionally Rice (313 samples).
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.preprocessing import StandardScaler
import nirs4all
from nirs4all.data.config import DatasetConfigs
from nirs4all.operators.models import AOMPLSRegressor
from nirs4all.operators.splitters.splitters import SPXYFold

# Quick datasets (small)
QUICK_DATASETS = [
    'BEER/Beer_OriginalExtract_60_KS',
    'AMYLOSE/Rice_Amylose_313_YbasedSplit',
]

def run_model(name, model, datasets=None):
    """Test a single model on quick datasets. Returns list of result dicts."""
    if datasets is None:
        datasets = QUICK_DATASETS

    if isinstance(model, list):
        pipeline = [
            SPXYFold(n_splits=3, random_state=42),
            {"y_processing": StandardScaler()}
        ] + model
    else:
        pipeline = [
            SPXYFold(n_splits=3, random_state=42),
            {"y_processing": StandardScaler()},
            {"model": model, "name": name}
        ]

    results = []
    for ds_name in datasets:
        ds_path = f"bench/tabpfn_paper/data/regression/{ds_name}"
        dataset_config = DatasetConfigs([ds_path], task_type="regression")

        start_time = time.time()
        try:
            res = nirs4all.run(
                pipeline=pipeline,
                dataset=dataset_config,
                name=name.replace(" ", "_"),
                verbose=0,
                random_state=42,
                n_jobs=1,
                workspace_path="bench/AOM/workspace"
            )
            duration = time.time() - start_time
            rmse = res.best_rmse
            print(f"  {ds_name.split('/')[0]:12s}: RMSE={rmse:.4f}  ({duration:.1f}s)")
            results.append({"Model": name, "Dataset": ds_name.split('/')[0], "RMSE": rmse, "Time": duration})
        except Exception as e:
            duration = time.time() - start_time
            print(f"  {ds_name.split('/')[0]:12s}: FAILED - {e}  ({duration:.1f}s)")
            results.append({"Model": name, "Dataset": ds_name.split('/')[0], "RMSE": None, "Time": duration})
    return results


if __name__ == "__main__":
    all_results = []

    # Baseline
    print("\n=== Baseline AOM-PLS ===")
    all_results += run_model("Baseline", AOMPLSRegressor(center=True, scale=False))

    # Import models to test (uncomment as needed)
    what_to_test = sys.argv[1] if len(sys.argv) > 1 else "all"

    if what_to_test in ("bandit", "all"):
        from models import BanditAOMPLSRegressor
        print("\n=== Bandit AOM-PLS ===")
        all_results += run_model("Bandit", BanditAOMPLSRegressor(n_components=15))

    if what_to_test in ("darts", "all"):
        from darts_pls import DartsPLSRegressor
        print("\n=== DARTS PLS ===")
        all_results += run_model("DARTS", DartsPLSRegressor(n_components=15, epochs=100))

    if what_to_test in ("zeroshot", "all"):
        from zero_shot_router import ZeroShotRouterPLSRegressor
        print("\n=== Zero-Shot Router ===")
        all_results += run_model("ZeroShot", ZeroShotRouterPLSRegressor(n_components=15))

    if what_to_test in ("moe", "all"):
        from moe_pls import MoEPLSRegressor
        print("\n=== MoE PLS ===")
        all_results += run_model("MoE", MoEPLSRegressor(n_components=15))

    # Summary table
    print("\n" + "="*60)
    df = pd.DataFrame(all_results)
    if not df.empty:
        pivot = df.pivot(index="Model", columns="Dataset", values="RMSE")
        print(pivot.to_string(float_format="%.4f"))
