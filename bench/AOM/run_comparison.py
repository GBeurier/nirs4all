import os
import sys
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add parent dir to path to import models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import BanditAOMPLSRegressor
from zero_shot_router import ZeroShotRouterPLSRegressor
from moe_pls import MoEPLSRegressor
from darts_pls import DartsPLSRegressor

import nirs4all
from nirs4all.data.config import DatasetConfigs
from nirs4all.operators.models import AOMPLSRegressor
from nirs4all.operators.splitters.splitters import SPXYFold

# Datasets to test
DATASETS = [
    'AMYLOSE/Rice_Amylose_313_YbasedSplit',
    'BEER/Beer_OriginalExtract_60_KS',
    'PLUMS/Firmness_spxy70',
    'MILK/Milk_Lactose_1224_KS',
    'PHOSPHORUS/LP_spxyG'
]
DATASETS_PATHS = [f"bench/tabpfn_paper/data/regression/{ds}" for ds in DATASETS]
dataset_config = DatasetConfigs(DATASETS_PATHS, task_type="regression")

# Models to compare
models = {
    "Baseline AOM-PLS": AOMPLSRegressor(center=True, scale=False),
    "Bandit AOM-PLS": BanditAOMPLSRegressor(n_components=15),
    "DARTS PLS": DartsPLSRegressor(n_components=15, epochs=100),
    "Zero-Shot Router": ZeroShotRouterPLSRegressor(n_components=15),
    "MoE PLS": MoEPLSRegressor(n_components=15),
}

results = []

for name, model in models.items():
    print(f"\n{'='*50}\nRunning {name}\n{'='*50}")

    if isinstance(model, list):
        pipeline = [
            SPXYFold(n_splits=3, random_state=42),
            {"y_processing": StandardScaler()}
        ] + model
        pipeline[-1]["name"] = name
    elif isinstance(model, dict):
        model_step = model.copy()
        model_step["name"] = name
        pipeline = [
            SPXYFold(n_splits=3, random_state=42),
            {"y_processing": StandardScaler()},
            model_step
        ]
    else:
        model_step = {"model": model, "name": name}
        pipeline = [
            SPXYFold(n_splits=3, random_state=42),
            {"y_processing": StandardScaler()},
            model_step
        ]

    for ds_name in DATASETS:
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

            best_rmse = res.best_rmse
            results.append({
                "Model": name,
                "Dataset": ds_name.split('/')[-1],
                "RMSE": best_rmse,
                "Time (s)": duration
            })
            print(f"  {ds_name}: RMSE = {best_rmse:.4f}")

        except Exception as e:
            print(f"  {ds_name} Failed: {e}")
            results.append({
                "Model": name,
                "Dataset": ds_name.split('/')[-1],
                "RMSE": None,
                "Time (s)": None
            })

# Save report
df = pd.DataFrame(results)
report_path = "bench/AOM/report.md"

with open(report_path, "w") as f:
    f.write("# AOM-PLS Next Generation Benchmark Report\n\n")
    f.write("Comparison of baseline AOM-PLS with 4 improved prototypes.\n\n")
    f.write("## Models\n\n")
    f.write("- **Baseline AOM-PLS**: Standard AOM-PLS with default operator bank\n")
    f.write("- **Bandit AOM-PLS**: Two-phase bandit (quick R² screen → full NIPALS for top-K), pseudo-linear SNV\n")
    f.write("- **DARTS PLS**: Gumbel-Softmax architecture search over 18 transforms, temperature annealing, adaptive hard-snap/blend\n")
    f.write("- **Zero-Shot Router**: Spectral feature extraction (10+ features), soft scoring of 9 pipelines, conservative validation\n")
    f.write("- **MoE PLS**: 25 expert chains (1st-4th order), two-phase OOF (sklearn screen → AOM-PLS), diversity filtering, Ridge meta-model with safety fallback\n\n")
    f.write("## Results\n\n")
    f.write(df.to_markdown(index=False))
    f.write("\n")

print(f"\nBenchmark complete. Report saved to {report_path}")
