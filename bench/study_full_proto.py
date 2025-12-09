"""
Full Analysis Study - Comprehensive Pipeline for Multiple Datasets
===================================================================
Combines TransferPreprocessingSelector, PLS/OPLS tuning, LWPLS, boosting models,
and TabPFN into a unified analysis workflow.
"""

import argparse
import math
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

os.environ['DISABLE_EMOJIS'] = '0'

import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CatBoostRegressor = None
    CATBOOST_AVAILABLE = False

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.visualization.predictions import PredictionAnalyzer
from nirs4all.analysis import TransferPreprocessingSelector
from nirs4all.operators.models.sklearn import OPLS
from nirs4all.operators.models.sklearn.lwpls import LWPLS
from nirs4all.operators.splitters import SPXYGFold
from nirs4all.operators.transforms import (
    Wavelet, WaveletFeatures, WaveletPCA, WaveletSVD,
    StandardNormalVariate, FirstDerivative, SavitzkyGolay,
)
from nirs4all.operators.models.pytorch.nicon import nicon, customizable_nicon, thin_nicon, nicon_VG

from tabpfn_config import (
    get_model_class,
    get_model_path_options,
    generate_inference_configs,
    TABPFN_AVAILABLE,
)

try:
    from huggingface_hub import login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
except ImportError:
    pass


parser = argparse.ArgumentParser(description="Full Analysis Study")
parser.add_argument("--show", action="store_true", help="Show plots interactively")
parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0-2)")
parser.add_argument("--device", type=str, default="cuda", help="Device: 'cuda' or 'cpu'")
args = parser.parse_args()


FOLDER_LIST = ['_datasets/redox/1700_Brix_StratGroupedKfold', '_datasets/redox/1700_Brix_YearSplit', '_datasets/redox/1700_CondElecCorr_StratGroupedKfold', '_datasets/redox/1700_CondElecCorr_YearSplit', '_datasets/redox/1700_pepH_StratGroupedKfold', '_datasets/redox/1700_pepH_YearSplit', '_datasets/redox/1700_pH_StratGroupedKfold', '_datasets/redox/1700_pH_YearSplit', '_datasets/redox/1700_Temp_Leaf_StratGroupedKfold', '_datasets/redox/1700_Temp_Leaf_YearSplit', '_datasets/redox/Pencil_Brix_StratGroupedKfold', '_datasets/redox/Pencil_Brix_YearSplit', '_datasets/redox/Pencil_CondElecCorr_StratGroupedKfold', '_datasets/redox/Pencil_CondElecCorr_YearSplit', '_datasets/redox/Pencil_pepH_StratGroupedKfold', '_datasets/redox/Pencil_pepH_YearSplit', '_datasets/redox/Pencil_pH_StratGroupedKfold', '_datasets/redox/Pencil_pH_YearSplit', '_datasets/redox/Pencil_Temp_Leaf_StratGroupedKfold', '_datasets/redox/Pencil_Temp_Leaf_YearSplit']

AGGREGATION_KEY_LIST = ["ID_1700_clean" for _ in FOLDER_LIST]

PP_SPEC = {
    "_cartesian_": [
        {"_or_": [None, "msc", "snv", "emsc", "rsnv"]},
        # {"_or_": [None, "savgol", "savgol_15", "gaussian", "gaussian2", "msc", "snv", "emsc", "rsnv"]},
        {"_or_": [None, "d1", "d2", "savgol_d1", "savgol15_d1", "savgol_d2"]},
        # {"_or_": [None, "haar", "detrend", "area_norm", "wav_sym5", "wav_coif3", "msc", "snv", "emsc"]},
    ],
}

SELECTOR_TOP_K = 10 #20
MAX_PP_PIPELINE_1 = 40
PLS_TRIALS = 20 #25
OPLS_TRIALS = 30 #35
RIDGE_TRIALS = 20 #20
TABPFN_TRIALS = 10 #10
TABPFN_MODEL_VARIANTS = ['default', 'real', 'low-skew', 'small-samples']

TABPFN_TRANSFORMERS = [
    PCA(n_components=50),
    PCA(n_components=100),
    TruncatedSVD(n_components=50),
    SparseRandomProjection(n_components=100),
    GaussianRandomProjection(n_components=100),
    [Wavelet('haar'), PCA(n_components=50)],
    [Wavelet('db4'), PCA(n_components=50)],
    [StandardNormalVariate(), PCA(n_components=100)],
    [SavitzkyGolay(), PCA(n_components=100)],
    [FirstDerivative(), PCA(n_components=100)],
    WaveletFeatures(wavelet='db4', max_level=5, n_coeffs_per_level=10),
    WaveletPCA(wavelet='coif3', max_level=4, n_components_per_level=5),
    WaveletSVD(wavelet='db4', max_level=4, n_components_per_level=5),
]


def _format_pipeline_for_display(pipeline_list):
    """Format a list of preprocessing pipelines for display.

    Args:
        pipeline_list: List of pipelines, where each pipeline is a list of transformers

    Returns:
        List of formatted strings for display
    """
    result = []
    for pipeline in pipeline_list:
        if isinstance(pipeline, list):
            names = [type(t).__name__ for t in pipeline]
            result.append(" > ".join(names) if names else "(empty)")
        else:
            result.append(type(pipeline).__name__)
    return result


def run_pipeline_1(dataset_config, filtered_pp_list, aggregation_key):
    """Pipeline 1: PLS and OPLS with transfer-selected preprocessings."""
    pipeline = [
        # {"split": GroupKFold(n_splits=3), "group": aggregation_key},
        {"split": SPXYGFold(n_splits=3), "group": aggregation_key},
        {"y_processing": MinMaxScaler(feature_range=(0.05, 0.9))},
        {"feature_augmentation": {"_or_": filtered_pp_list, "pick": (1, 2), "count": MAX_PP_PIPELINE_1}},
        MinMaxScaler,
        {
            "model": PLSRegression(),
            "name": "PLS-Finetuned",
            "finetune_params": {
                "n_trials": PLS_TRIALS,
                "verbose": 0,
                "approach": "grouped",
                "eval_mode": "avg",
                "sample": "tpe",
                "model_params": {"n_components": ("int", 1, 40)},
            },
        },
        {
            "model": OPLS(),
            "name": "OPLS-Finetuned",
            "finetune_params": {
                "n_trials": OPLS_TRIALS,
                "verbose": 0,
                "approach": "grouped",
                "eval_mode": "avg",
                "sample": "tpe",
                "model_params": {
                    "n_components": ("int", 1, 10),
                    "pls_components": ("int", 1, 40),
                },
            },
        },
    ]

    pipeline_config = PipelineConfigs(pipeline, "pipeline_1_pls_opls")
    runner = PipelineRunner(save_files=True, verbose=0, plots_visible=False)
    predictions, _ = runner.run(pipeline_config, dataset_config)

    # Get many predictions to ensure we find enough unique preprocessings
    # (top predictions may share the same preprocessing across folds)
    top_preds = predictions.top(n=50, rank_metric="rmse", rank_partition="test", aggregate=aggregation_key)
    if not top_preds:
        print("  Warning: No predictions found")
        return predictions, filtered_pp_list[:3], 10, []

    top3_pp_display = [p.get("preprocessings", None) for p in top_preds[:3]]
    best_pred = top_preds[0]
    best_params = best_pred.get("best_params", {})
    best_n_components = best_params.get("n_components", 10)

    # Extract top preprocessing pipelines from the best predictions
    # This parses display strings and deserializes the transformers for reuse
    top_pp_list = runner.manifest_manager.extract_top_preprocessings(
        predictions=list(top_preds),
        top_k=3,
        exclude_scalers=True,
        verbose=True
    )

    # Display the extracted pipelines
    extracted_display = _format_pipeline_for_display(top_pp_list)
    print(f"  *** Extracted {len(top_pp_list)} preprocessing pipeline(s): {extracted_display}")

    # Fallback to original list if extraction fails
    if not top_pp_list:
        print("  Warning: Could not extract preprocessings from manifests, using original list")
        top_pp_list = filtered_pp_list[:min(3, len(filtered_pp_list))]

    return predictions, top_pp_list, best_n_components, top3_pp_display


def run_pipeline_2(dataset_config, top3_pp, best_n_components, aggregation_key):
    """Pipeline 2: LWPLS, Ridge, CatBoost, Nicon, RandomForest."""

    catboost_configs = [
        CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1, verbose=0, allow_writing_files=False),
        CatBoostRegressor(iterations=400, depth=8, learning_rate=0.05, verbose=0, allow_writing_files=False),
        CatBoostRegressor(iterations=300, depth=10, learning_rate=0.08, verbose=0, allow_writing_files=False),
    ] if CATBOOST_AVAILABLE else []

    nicon_configs = [
        {"model": nicon, "name": "nicon"},
        {"model": nicon_VG, "name": "nicon"},
    ]

    # Build feature_augmentation step based on number of pipelines
    # - If multiple pipelines: use _or_ syntax
    # - If single pipeline: use it directly (no _or_ needed)
    # - If empty: skip feature_augmentation
    if len(top3_pp) > 1:
        feature_aug_step = {"feature_augmentation": {"_or_": top3_pp}}
        print(f"  [Pipeline 2] Using _or_ with {len(top3_pp)} pipelines")
    elif len(top3_pp) == 1:
        # Single pipeline - use directly without _or_
        feature_aug_step = {"feature_augmentation": top3_pp[0]}
        print(f"  [Pipeline 2] Using single pipeline directly: {[type(t).__name__ for t in top3_pp[0]]}")
    else:
        # No preprocessings - skip feature_augmentation entirely
        feature_aug_step = None
        print(f"  [Pipeline 2] No preprocessings, skipping feature_augmentation")

    pipeline = [
        # {"split": GroupKFold(n_splits=3), "group": aggregation_key},
        {"split": SPXYGFold(n_splits=3), "group": aggregation_key},
        {"y_processing": MinMaxScaler(feature_range=(0.05, 0.9))},
    ]

    # Add feature_augmentation only if we have one
    if feature_aug_step:
        pipeline.append(feature_aug_step)

    pipeline.extend([
        MinMaxScaler,
        {"model": LWPLS(n_components=best_n_components), "name": "LWPLS"},

        {
            "model": Ridge(),
            "name": "Ridge-Finetuned",
            "finetune_params": {
                "n_trials": RIDGE_TRIALS,
                "verbose": 0,
                "approach": "grouped",
                "model_params": {"alpha": ("log_float", 0.001, 100)},
            },
        },

        # {"model": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42), "name": "RandomForest"},

        *[{"model": cb, "name": f"CatBoost-{i+1}"} for i, cb in enumerate(catboost_configs)],

        *nicon_configs,
    ])

    pipeline_config = PipelineConfigs(pipeline, "pipeline_2_ensemble")
    runner = PipelineRunner(save_files=True, verbose=0, plots_visible=False)
    predictions, _ = runner.run(pipeline_config, dataset_config)
    return predictions


def run_pipeline_3(dataset_config, aggregation_key, top3_pp):
    """Pipeline 3: TabPFN finetuning."""
    if not TABPFN_AVAILABLE:
        print("  TabPFN not available, skipping pipeline 3")
        return None

    TabPFN_class = get_model_class('regression')

    for pp in top3_pp:
        TABPFN_TRANSFORMERS.append([pp, PCA(n_components=100)])

    pipeline = [
        # {"split": GroupKFold(n_splits=3), "group": aggregation_key},
        {"split": SPXYGFold(n_splits=3), "group": aggregation_key},
        {"y_processing": MinMaxScaler()},
        {"concat_transform": {"_or_": TABPFN_TRANSFORMERS, "pick": (1, 3), "count": 20}},
        StandardScaler(),
        {
            "model": TabPFN_class(device=args.device),
            "name": "TabPFN-Finetuned",
            "finetune_params": {
                "n_trials": TABPFN_TRIALS,
                "verbose": args.verbose,
                "approach": "single",
                "eval_mode": "best",
                "sample": "tpe",
                "model_params": {
                    "model_path": get_model_path_options('regression', TABPFN_MODEL_VARIANTS),
                    "inference_config": generate_inference_configs('regression', mode="minimal"),
                },
            },
        },
    ]

    pipeline_config = PipelineConfigs(pipeline, "pipeline_3_tabpfn")
    runner = PipelineRunner(save_files=True, verbose=args.verbose, plots_visible=False)
    predictions, _ = runner.run(pipeline_config, dataset_config)
    return predictions


def display_results(all_results, folder, aggregation_key):
    """Display aggregated best results."""
    print()
    print("=" * 70)
    print(f"AGGREGATED RESULTS: {folder}")
    print("=" * 70)

    combined_top = []
    for name, preds in all_results.items():
        if preds is None:
            continue
        top = preds.top(n=5, rank_metric="rmse", rank_partition="test", aggregate=aggregation_key)
        for p in top:
            p["_source_pipeline"] = name
        combined_top.extend(top)

    combined_top.sort(key=lambda x: x.get("test_score", float('inf')))

    print(f"\nTop 10 models across all pipelines (by RMSE):")
    print("-" * 70)
    for idx, pred in enumerate(combined_top[:10], 1):
        mse = pred.get("test_score", None)
        rmse = math.sqrt(mse) if mse is not None else None
        rmse_str = f"{rmse:.4f}" if rmse else "N/A"
        model_name = pred.get("model_name", "N/A")
        pp = pred.get("preprocessings", "N/A")
        source = pred.get("_source_pipeline", "?")
        print(f"{idx:2d}. RMSE={rmse_str:>8} | {model_name:<25} | {source} | {pp}")

    print()


def main():
    print("=" * 70)
    print("FULL ANALYSIS STUDY")
    print("=" * 70)
    print()

    for folder, aggregation_key in zip(FOLDER_LIST, AGGREGATION_KEY_LIST):
        print(f"\n{'='*70}")
        print(f"PROCESSING: {folder}")
        print(f"Aggregation key: {aggregation_key}")
        print("=" * 70)

        dataset_config = DatasetConfigs(str(folder))

        # Phase 1: Transfer Preprocessing Selection
        print("\n[Phase 1] TransferPreprocessingSelector...")
        t0 = time.time()
        selector = TransferPreprocessingSelector(
            preset="balanced",
            preprocessing_spec=PP_SPEC,
            verbose=args.verbose,
        )
        results = selector.fit(dataset_config)
        filtered_pp_list = results.to_preprocessing_list(top_k=SELECTOR_TOP_K)
        print(f"  Selected {len(filtered_pp_list)} preprocessings in {time.time()-t0:.1f}s")

        # Phase 2: Pipeline 1 - PLS/OPLS
        print("\n[Phase 2] Pipeline 1: PLS/OPLS finetuning...")
        t0 = time.time()
        preds_1, top3_pp, best_n_components, top3_pp_display = run_pipeline_1(dataset_config, filtered_pp_list, aggregation_key)
        print(f"  Completed in {time.time()-t0:.1f}s")
        print(f"  Best n_components: {best_n_components}")
        print(f"  Top {len(top3_pp_display)} preprocessings (display): {top3_pp_display}")
        print(f"  >>> Actual {len(top3_pp)} pipelines passed to Pipeline 2: {_format_pipeline_for_display(top3_pp)}")

        # Phase 3: Pipeline 2 - LWPLS, Ridge, CatBoost, Nicon, RF
        print("\n[Phase 3] Pipeline 2: Ensemble models...")
        t0 = time.time()
        preds_2 = run_pipeline_2(dataset_config, top3_pp, best_n_components, aggregation_key)
        print(f"  Completed in {time.time()-t0:.1f}s")

        # Phase 4: Pipeline 3 - TabPFN
        print("\n[Phase 4] Pipeline 3: TabPFN...")
        t0 = time.time()
        preds_3 = run_pipeline_3(dataset_config, aggregation_key, top3_pp)
        if preds_3:
            print(f"  Completed in {time.time()-t0:.1f}s")

        # Display results
        all_results = {
            "PLS/OPLS": preds_1,
            "Ensemble": preds_2,
            "TabPFN": preds_3,
        }
        display_results(all_results, folder, aggregation_key)

    print("\n" + "=" * 70)
    print("STUDY COMPLETE")
    print("=" * 70)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
