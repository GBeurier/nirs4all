"""
Full Analysis Study - Comprehensive Pipeline for Multiple Datasets
===================================================================
Combines TransferPreprocessingSelector, PLS/OPLS tuning, LWPLS, boosting models,
and TabPFN into a unified analysis workflow.
"""

import argparse
import math
import os
import sys
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

from catboost import CatBoostRegressor

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.visualization.predictions import PredictionAnalyzer
from nirs4all.analysis import TransferPreprocessingSelector
from nirs4all.operators.models.sklearn import OPLS
from nirs4all.operators.models.sklearn.lwpls import LWPLS
from nirs4all.operators.splitters import SPXYGFold
from nirs4all.operators.transforms import (
    Wavelet, WaveletFeatures, WaveletPCA, WaveletSVD,
    StandardNormalVariate, FirstDerivative, SecondDerivative, SavitzkyGolay,
    MultiplicativeScatterCorrection, Detrend, Gaussian, Haar,
    RobustStandardNormalVariate,
)
from nirs4all.operators.transforms.nirs import (
    AreaNormalization,
    ExtendedMultiplicativeScatterCorrection as EMSC,
)
from nirs4all.operators.models.pytorch.nicon import nicon, customizable_nicon, thin_nicon, nicon_VG

from study_tabpfn_config import get_model_class, get_model_path_options, generate_inference_configs

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
parser.add_argument("--workspace", type=str, default="wk", help="Workspace directory path")

# Dataset configuration
parser.add_argument("--datasets", nargs='+', help="Dataset folder paths")
parser.add_argument("--aggregation-keys", nargs='+', help="Aggregation keys (one per dataset)")

# Test mode
parser.add_argument("--test-mode", action="store_true", help="Run in test mode (fast execution)")

# Transfer preprocessing configuration
parser.add_argument("--transfer-preset", type=str, choices=["fast", "balanced", "comprehensive"],
                   help="Transfer preprocessing preset")
parser.add_argument("--transfer-selected", type=int, help="Number of preprocessings to select")

# PLS/OPLS configuration
parser.add_argument("--pls-pp-count", type=int, help="PLS preprocessing count")
parser.add_argument("--pls-pp-top", type=int, help="Top PLS preprocessings to select")
parser.add_argument("--pls-trials", type=int, help="Number of PLS trials")
parser.add_argument("--opls-trials", type=int, help="Number of OPLS trials")

# Other model configurations
parser.add_argument("--test-lwpls", action="store_true", help="Test LWPLS model")
parser.add_argument("--ridge-trials", type=int, help="Number of Ridge trials")

# TabPFN configuration
parser.add_argument("--tabpfn-trials", type=int, help="Number of TabPFN trials")
parser.add_argument("--tabpfn-variants", nargs='+', help="TabPFN model variants")
parser.add_argument("--tabpfn-pp-max-count", type=int, help="TabPFN max preprocessing count")
parser.add_argument("--tabpfn-pp-max-size", type=int, help="TabPFN max preprocessing size")

args = parser.parse_args()


def get_pp_fingerprint(pp):
    """Create a fingerprint string for a preprocessing pipeline to enable comparison."""
    if pp is None:
        return "None"
    if not isinstance(pp, list):
        pp = [pp]
    parts = []
    for item in pp:
        if isinstance(item, list):
            parts.append(f"[{get_pp_fingerprint(item)}]")
        else:
            class_name = type(item).__name__
            params = getattr(item, 'get_params', lambda: {})()
            parts.append(f"{class_name}({params})")
    return "|".join(parts)


def get_best_pp_cp(count_pp, count_cp, runner, predictions, pp_index=0, aggregation_key=None):
    k = 6 * count_pp
    top_preds = predictions.top(n=k, rank_metric="rmse", rank_partition="test")

    best_cp = []
    best_pp = []
    best_pp_fingerprints = set()

    for pred in top_preds:
        if len(best_cp) >= count_cp and len(best_pp) >= count_pp:
            break

        if len(best_cp) < count_cp:
            best_params = pred.get("best_params", {})
            n_components = best_params.get("n_components", 10)
            if n_components not in best_cp:
                best_cp.append(n_components)

        if len(best_pp) < count_pp:
            pp_choice = runner.manifest_manager.extract_generator_choice(pred, choice_index=pp_index, instantiate=True)
            fingerprint = get_pp_fingerprint(pp_choice)
            print(pp_choice)
            if fingerprint not in best_pp_fingerprints:
                best_pp_fingerprints.add(fingerprint)
                best_pp.append(pp_choice)

    return best_pp, best_cp

def expand_tabpfn_pp(top3_pp):
    uniques_pp_fingerprints = set()

    for pp in top3_pp:
        if len(pp) > 1:
            fingerprint = ""
            for sub_pipeline in pp:
                fingerprint += get_pp_fingerprint(sub_pipeline) + ";"
                sub_pipeline.append(PCA(n_components=100))

            if fingerprint not in uniques_pp_fingerprints:
                new_pipeline = {
                    "concat_transform": pp
                }
                print("ADDING CONCAT:", new_pipeline)
                TABPFN_PP.append(new_pipeline)
        else:
            print("SINGLE")
            print(pp)
            pipe = pp[0]
            fingerprint = get_pp_fingerprint(pipe)
            if fingerprint not in uniques_pp_fingerprints:
                pipe.append(PCA(n_components=100))
                uniques_pp_fingerprints.add(fingerprint)
                print("ADDED", pipe)
                TABPFN_PP.append(pipe)

###########################################################

# Default configuration (can be overridden by CLI arguments)
REDOX_FOLDER = '_datasets/redox/'
SUB_FOLDER_LIST = [
    '1700_Brix_StratGroupedKfold',
    '1700_Brix_YearSplit',
    '1700_CondElecCorr_StratGroupedKfold',
    '1700_CondElecCorr_YearSplit',
    '1700_pepH_StratGroupedKfold',
    '1700_pepH_YearSplit',
    '1700_pH_StratGroupedKfold',
    '1700_pH_YearSplit',
    '1700_Temp_Leaf_StratGroupedKfold',
    '1700_Temp_Leaf_YearSplit',
    'Pencil_Brix_StratGroupedKfold',
    'Pencil_Brix_YearSplit',
    'Pencil_CondElecCorr_StratGroupedKfold',
    'Pencil_CondElecCorr_YearSplit',
    'Pencil_pepH_StratGroupedKfold',
    'Pencil_pepH_YearSplit',
    'Pencil_pH_StratGroupedKfold',
    'Pencil_pH_YearSplit',
    'Pencil_Temp_Leaf_StratGroupedKfold',
    'Pencil_Temp_Leaf_YearSplit']

DEFAULT_FOLDER_LIST = [os.path.join(REDOX_FOLDER, sub_folder) for sub_folder in SUB_FOLDER_LIST]
DEFAULT_AGGREGATION_KEY_LIST = ["ID_1700_clean" for _ in DEFAULT_FOLDER_LIST]

# Default test mode settings
DEFAULT_TEST_MODE = False

# Apply CLI arguments or use defaults
TEST_MODE = args.test_mode if hasattr(args, 'test_mode') else DEFAULT_TEST_MODE
FOLDER_LIST = args.datasets if args.datasets else DEFAULT_FOLDER_LIST
AGGREGATION_KEY_LIST = args.aggregation_keys if args.aggregation_keys else DEFAULT_AGGREGATION_KEY_LIST

# Validate dataset and aggregation key lists match
if len(FOLDER_LIST) != len(AGGREGATION_KEY_LIST):
    print(f"Error: Number of datasets ({len(FOLDER_LIST)}) must match number of aggregation keys ({len(AGGREGATION_KEY_LIST)})")
    sys.exit(1)

if TEST_MODE:
    TRANSFER_PP_PRESET = "fast"
    TRANSFER_PP_SELECTED = 2
    PLS_PP_COUNT = 2
    PLS_PP_TOP_SELECTED_COUNT = 1
    PLS_TRIALS = 1
    OPLS_TRIALS = 1
    TEST_LW_PLS = False
    RIDGE_TRIALS = 1
    TABPFN_TRIALS = 1
    TABPFN_MODEL_VARIANTS = ['default']#, 'real']#, 'low-skew', 'small-samples']
    TABPFN_PP_MAX_COUNT = 1
    TABPFN_PP_MAX_SIZE = 1
    WORKSPACE_PATH = args.workspace if hasattr(args, 'workspace') else "wk"

    # Object-based preprocessing spec (recommended)
    GLOBAL_PP = {
        "_cartesian_": [
            {"_or_": [None, SavitzkyGolay(), SavitzkyGolay(window_length=15), Gaussian(order=1, sigma=2), Gaussian(order=2, sigma=2),
                      MultiplicativeScatterCorrection(), StandardNormalVariate(), EMSC(), RobustStandardNormalVariate()]},
            {"_or_": [None, FirstDerivative(), SecondDerivative(), SavitzkyGolay(deriv=1), SavitzkyGolay(window_length=15, deriv=1), SavitzkyGolay(deriv=2)]},
        ],
    }
else:
    TRANSFER_PP_PRESET = "balanced"  # "fast", "balanced", "comprehensive"
    TRANSFER_PP_SELECTED = 10
    PLS_PP_COUNT = 40
    PLS_PP_TOP_SELECTED_COUNT = 10
    PLS_TRIALS = 20
    OPLS_TRIALS = 30
    TEST_LW_PLS = True
    RIDGE_TRIALS = 20
    TABPFN_TRIALS = 10
    TABPFN_MODEL_VARIANTS = ['default', 'real', 'low-skew', 'small-samples']
    TABPFN_PP_MAX_COUNT = 20
    TABPFN_PP_MAX_SIZE = 3
    WORKSPACE_PATH = args.workspace if hasattr(args, 'workspace') else "wk"

    # Object-based preprocessing spec (recommended)
    GLOBAL_PP = {
        "_cartesian_": [
            {"_or_": [None, MultiplicativeScatterCorrection(), StandardNormalVariate(), EMSC(), RobustStandardNormalVariate()]},
            {"_or_": [None, SavitzkyGolay(), SavitzkyGolay(window_length=15), Gaussian(order=1, sigma=2), Gaussian(order=2, sigma=2),
                      MultiplicativeScatterCorrection(), StandardNormalVariate(), EMSC(), RobustStandardNormalVariate()]},
            {"_or_": [None, FirstDerivative(), SecondDerivative(), SavitzkyGolay(deriv=1), SavitzkyGolay(window_length=15, deriv=1), SavitzkyGolay(deriv=2)]},
            {"_or_": [None, Haar(), Detrend(), AreaNormalization(), Wavelet("sym5"), Wavelet("coif3"),
                      MultiplicativeScatterCorrection(), StandardNormalVariate(), EMSC()]},
        ],
    }

    # Override with CLI arguments if provided
    if args.transfer_preset:
        TRANSFER_PP_PRESET = args.transfer_preset
    if args.transfer_selected:
        TRANSFER_PP_SELECTED = args.transfer_selected
    if args.pls_pp_count:
        PLS_PP_COUNT = args.pls_pp_count
    if args.pls_pp_top:
        PLS_PP_TOP_SELECTED_COUNT = args.pls_pp_top
    if args.pls_trials:
        PLS_TRIALS = args.pls_trials
    if args.opls_trials:
        OPLS_TRIALS = args.opls_trials
    if args.test_lwpls:
        TEST_LW_PLS = True
    if args.ridge_trials:
        RIDGE_TRIALS = args.ridge_trials
    if args.tabpfn_trials:
        TABPFN_TRIALS = args.tabpfn_trials
    if args.tabpfn_variants:
        TABPFN_MODEL_VARIANTS = args.tabpfn_variants
    if args.tabpfn_pp_max_count:
        TABPFN_PP_MAX_COUNT = args.tabpfn_pp_max_count
    if args.tabpfn_pp_max_size:
        TABPFN_PP_MAX_SIZE = args.tabpfn_pp_max_size

TABPFN_PP = [
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


def run_pipeline_1(dataset_config, filtered_pp_list, aggregation_key):
    """Pipeline 1: PLS and OPLS with transfer-selected preprocessings."""
    pipeline = [
        # {"split": GroupKFold(n_splits=3), "group": aggregation_key},
        {"split": SPXYGFold(n_splits=3), "group": aggregation_key},
        {"y_processing": MinMaxScaler(feature_range=(0.05, 0.9))},
        {"feature_augmentation": {"_or_": filtered_pp_list, "pick": [1, 2], "count": PLS_PP_COUNT}},
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
    runner = PipelineRunner(workspace_path=WORKSPACE_PATH, save_files=True, verbose=0, plots_visible=False)
    predictions, _ = runner.run(pipeline_config, dataset_config)

    top_pp_list, best_n_components = get_best_pp_cp(
        count_pp=PLS_PP_TOP_SELECTED_COUNT,
        count_cp=1,
        runner=runner,
        predictions=predictions,
        pp_index=0,
        aggregation_key=aggregation_key,
    )

    return predictions, top_pp_list, best_n_components[0]


def run_pipeline_2(dataset_config, top3_pp, best_n_components, aggregation_key):
    """Pipeline 2: LWPLS, Ridge, CatBoost, Nicon, RandomForest."""

    catboost_configs = [
        # CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1, verbose=0, allow_writing_files=False, task_type="GPU", devices="0"),
        # CatBoostRegressor(iterations=400, depth=8, learning_rate=0.05, verbose=0, allow_writing_files=False, task_type="GPU", devices="0"),
        # CatBoostRegressor(iterations=300, depth=10, learning_rate=0.08, verbose=0, allow_writing_files=False, task_type="GPU", devices="0"),
    ]

    nicon_configs = [
        {"model": nicon, "name": "nicon"},
        {"model": nicon_VG, "name": "nicon"},
        {"model": thin_nicon, "name": "nicon"},
    ]

    if len(top3_pp) > 1:
        feature_aug_step = {"feature_augmentation": {"_or_": top3_pp}}
        print(f"  [Pipeline 2] Using _or_ with {len(top3_pp)} pipelines")
    elif len(top3_pp) == 1:
        feature_aug_step = {"feature_augmentation": top3_pp[0]}
        print(f"  [Pipeline 2] Using single pipeline directly: {[type(t).__name__ for t in top3_pp[0]]}")

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

    if TEST_LW_PLS:
        pipeline.append({"model": LWPLS(n_components=best_n_components), "name": "LWPLS"})

    pipeline_config = PipelineConfigs(pipeline, "pipeline_2_ensemble")
    runner = PipelineRunner(workspace_path=WORKSPACE_PATH, save_files=True, verbose=0, plots_visible=False)
    predictions, _ = runner.run(pipeline_config, dataset_config)
    return predictions


def run_pipeline_3(dataset_config, aggregation_key, top3_pp):
    """Pipeline 3: TabPFN finetuning."""

    TabPFN_class = get_model_class('regression')
    expand_tabpfn_pp(top3_pp)
    print(TABPFN_PP)

    # Build model_params, only including model_path if there are valid paths
    model_params = {
        "inference_config": generate_inference_configs('regression', mode="minimal"),
    }
    model_path_options = get_model_path_options('regression', TABPFN_MODEL_VARIANTS)
    if model_path_options:
        model_params["model_path"] = model_path_options

    pipeline = [
        # {"split": GroupKFold(n_splits=3), "group": aggregation_key},
        {"split": SPXYGFold(n_splits=3), "group": aggregation_key},
        {"y_processing": MinMaxScaler()},
        {"concat_transform": {"_or_": TABPFN_PP, "pick": [1, TABPFN_PP_MAX_SIZE], "count": TABPFN_PP_MAX_COUNT}},
        StandardScaler(),
        {
            "model": TabPFN_class(device="cuda"),
            "name": "TabPFN-Finetuned",
            "finetune_params": {
                "n_trials": TABPFN_TRIALS,
                "verbose": args.verbose,
                "approach": "single",
                "eval_mode": "best",
                "sample": "tpe",
                "model_params": model_params,
            },
        },
    ]

    pipeline_config = PipelineConfigs(pipeline, "pipeline_3_tabpfn")
    runner = PipelineRunner(workspace_path=WORKSPACE_PATH, save_files=True, verbose=args.verbose, plots_visible=False)
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

        # Check if advanced config is available (set by study_runner)
        transfer_pp_config = getattr(sys.modules[__name__], 'TRANSFER_PP_CONFIG', None)

        if transfer_pp_config is not None:
            # Advanced mode: use full config dict for TransferPreprocessingSelector
            selector_kwargs = transfer_pp_config.copy()
            # Ensure preprocessing_spec is set if not in config but GLOBAL_PP is available
            if 'preprocessing_spec' not in selector_kwargs and GLOBAL_PP is not None:
                selector_kwargs['preprocessing_spec'] = GLOBAL_PP
            if 'verbose' not in selector_kwargs:
                selector_kwargs['verbose'] = args.verbose
            selector = TransferPreprocessingSelector(**selector_kwargs)
        else:
            # Simple mode: use preset and GLOBAL_PP
            selector = TransferPreprocessingSelector(
                preset=TRANSFER_PP_PRESET,
                preprocessing_spec=GLOBAL_PP,
                verbose=args.verbose,
            )

        results = selector.fit(dataset_config)
        filtered_pp_list = results.to_preprocessing_list(top_k=TRANSFER_PP_SELECTED)
        print(f"  Selected {len(filtered_pp_list)} preprocessings in {time.time()-t0:.1f}s")

        # Phase 2: Pipeline 1 - PLS/OPLS
        print("\n[Phas:e 2] Pipeline 1: PLS/OPLS finetuning...")
        t0 = time.time()
        preds_1, top_pp_list, best_n_components = run_pipeline_1(dataset_config, filtered_pp_list, aggregation_key)
        print(f"  Completed in {time.time()-t0:.1f}s")
        print(f"  Top {len(top_pp_list)} preprocessings for Pipeline 2:")
        print(f"    " + "\n    ".join([str(pp) for pp in top_pp_list]))
        print(f"  Best n_components for PLS: {best_n_components}")


        # Phase 3: Pipeline 2 - LWPLS, Ridge, CatBoost, Nicon, RF
        print("\n[Phase 3] Pipeline 2: Ensemble models...")
        t0 = time.time()
        preds_2 = run_pipeline_2(dataset_config, top_pp_list, best_n_components, aggregation_key)
        print(f"  Completed in {time.time()-t0:.1f}s")

        # Phase 4: Pipeline 3 - TabPFN
        print("\n[Phase 4] Pipeline 3: TabPFN...")
        t0 = time.time()
        preds_3 = run_pipeline_3(dataset_config, aggregation_key, top_pp_list)
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
