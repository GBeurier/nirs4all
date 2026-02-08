"""
Study Proto 2: TabPFN Finetuning with NIRS4All Pipeline
========================================================
Comparable to study_proto_1 (PLS) but using TabPFN models.

This study uses the same dataset (redox/1700_Brix_StratGroupedKfold) and
similar pipeline structure as study_proto_1 for direct comparison.

Key features:
- Uses nirs4all's finetune_params with Optuna for TabPFN inference config tuning
- Tests TabPFN model variants via model_path parameter
- Tests inference_config options (fingerprint, outlier removal, min_unique, etc.)
- Uses concat_transform for feature extraction (PCA, SVD, Wavelet->PCA)

Usage:
    python study_proto_2.py --show
    python study_proto_2.py --device cpu  # For CPU testing
"""

import argparse
import math
import os
import time
from pathlib import Path

# Load environment variables from .env file (in project root)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

os.environ['DISABLE_EMOJIS'] = '0'

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.visualization.predictions import PredictionAnalyzer
from nirs4all.operators.transforms import (
    Wavelet, WaveletFeatures, WaveletPCA, WaveletSVD,
    StandardNormalVariate, FirstDerivative, SavitzkyGolay,
)
from nirs4all.operators.splitters import SPXYGFold, BinnedStratifiedGroupKFold
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

# TabPFN config module (local)
from tabpfn_config import (
    get_model_class,
    create_model,
    get_model_path_options,
    generate_inference_configs,
    TABPFN_AVAILABLE,
    REGRESSOR_MODELS,
    CLASSIFIER_MODELS,
)

# Hugging Face login for TabPFN
try:
    from huggingface_hub import login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("Warning: HF_TOKEN not set. TabPFN may not work properly.")
except ImportError:
    print("Warning: huggingface_hub not installed.")




# CLASSIFIER_MODELS = {
#     'default': 'tabpfn-v2.5-classifier-v2.5_default.ckpt',
#     'default-2': 'tabpfn-v2.5-classifier-v2.5_default-2.ckpt',
#     'large-features-L': 'tabpfn-v2.5-classifier-v2.5_large-features-L.ckpt',
#     'large-features-XL': 'tabpfn-v2.5-classifier-v2.5_large-features-XL.ckpt',
#     'large-samples': 'tabpfn-v2.5-classifier-v2.5_large-samples.ckpt',
#     'real': 'tabpfn-v2.5-classifier-v2.5_real.ckpt',
#     'real-large-features': 'tabpfn-v2.5-classifier-v2.5_real-large-features.ckpt',
#     'real-large-samples-and-features': 'tabpfn-v2.5-classifier-v2.5_real-large-samples-and-features.ckpt',
#     'variant': 'tabpfn-v2.5-classifier-v2.5_variant.ckpt',
# }

# # REGRESSOR models:
# REGRESSOR_MODELS = {
#     'default': 'tabpfn-v2.5-regressor-v2.5_default.ckpt',
#     'low-skew': 'tabpfn-v2.5-regressor-v2.5_low-skew.ckpt',
#     'quantiles': 'tabpfn-v2.5-regressor-v2.5_quantiles.ckpt',
#     'real': 'tabpfn-v2.5-regressor-v2.5_real.ckpt',
#     'real-variant': 'tabpfn-v2.5-regressor-v2.5_real-variant.ckpt',
#     'small-samples': 'tabpfn-v2.5-regressor-v2.5_small-samples.ckpt',
#     'variant': 'tabpfn-v2.5-regressor-v2.5_variant.ckpt',
# }

# =============================================================================
# Parse Arguments
# =============================================================================
parser = argparse.ArgumentParser(description="Study Proto 2 - TabPFN Comparison")
parser.add_argument("--show", action="store_true", help="Show plots interactively")
parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0-2)")
parser.add_argument("--device", type=str, default="cuda", help="Device: 'cuda' or 'cpu'")
args = parser.parse_args()


# =============================================================================
# Configuration
# =============================================================================

# Dataset path - use _datasets if available, else fallback to examples sample_data
DATA_PATH = "_datasets/redox/1700_Brix_StratGroupedKfold/"

# Study parameters
TASK_TYPE = 'regression'
DEVICE = args.device
N_TRIALS = 10  # Number of Optuna trials for finetuning

# Model variants to test
MODEL_VARIANTS = ['default', 'real', 'low-skew'] if TASK_TYPE == 'regression' else ['default', 'real']


# =============================================================================
# Feature Extraction Pipeline Components
# =============================================================================

tabpfn_transformers = [
        # Basic dimensionality reduction
        PCA(n_components=50),
        PCA(n_components=100),
        PCA(n_components=120),
        TruncatedSVD(n_components=50),
        SparseRandomProjection(n_components=100),
        GaussianRandomProjection(n_components=100),

        # Sequential Wavelet -> PCA (global PCA on all wavelet coefficients)
        [Wavelet('haar'), PCA(n_components=50)],
        [Wavelet('db4'), PCA(n_components=50)],
        [Wavelet('coif3'), PCA(n_components=50)],
        [Wavelet('haar'), PCA(n_components=25)],

        # Preprocessing -> PCA
        [StandardNormalVariate(), PCA(n_components=100)],
        [SavitzkyGolay(), PCA(n_components=100)],
        [FirstDerivative(), PCA(n_components=100)],

        # Wavelet statistical features (extracts stats + top coeffs per level)
        WaveletFeatures(wavelet='db4', max_level=5, n_coeffs_per_level=10),
        WaveletFeatures(wavelet='haar', max_level=5, n_coeffs_per_level=10),
        WaveletFeatures(wavelet='coif3', max_level=4, n_coeffs_per_level=8),

        # Multi-scale Wavelet-PCA (PCA per decomposition level)
        WaveletPCA(wavelet='coif3', max_level=4, n_components_per_level=5),
        WaveletPCA(wavelet='haar', max_level=5, n_components_per_level=4),
        WaveletPCA(wavelet='db4', max_level=5, n_components_per_level=3),

        # Multi-scale Wavelet-SVD (SVD per decomposition level)
        WaveletSVD(wavelet='db4', max_level=4, n_components_per_level=5),
        WaveletSVD(wavelet='haar', max_level=5, n_components_per_level=4),
    ]


# =============================================================================
# Main Study
# =============================================================================

def main():
    print("=" * 70)
    print("STUDY PROTO 2 - TabPFN Comparison")
    print("=" * 70)
    print(f"Task type: {TASK_TYPE}")
    print(f"Device: {DEVICE}")
    print(f"Model variants: {MODEL_VARIANTS}")
    print()

    dataset_config = DatasetConfigs(str(DATA_PATH))

    # Get the TabPFN class for this task type
    TabPFN_class = get_model_class(TASK_TYPE)

    # Build pipeline
    pipeline = [
        # Target scaling (same as study_proto_1)
        {"y_processing": MinMaxScaler()},

        # {
        #     "split": SPXYGFold,
        #     "split_params": {
        #         "n_splits": 1,
        #         "test_size": 0.2,
        #         "aggregation": "mean"
        #     },
        #     "group": "ID_1700_clean"
        # },
        {
            "split": GroupKFold(n_splits=3, shuffle=True, random_state=42), "group": "ID_1700_clean"
        },

        # Feature extraction with multiple options (preprocessing handled here, not in TabPFN)
        {
            "concat_transform": {
                "_or_": tabpfn_transformers,
                "pick": (1, 2),
                "count": 10,
            }
        },

        # Standard scaling after feature extraction
        StandardScaler(),

        # TabPFN with finetuning (model_path + inference options)
        {
            "model": TabPFN_class(device=DEVICE),
            "name": "TabPFN-Finetuned",
            "finetune_params": {
                "n_trials": N_TRIALS,
                "verbose": args.verbose,
                "approach": "single",
                "eval_mode": "best",
                "sample": "tpe",
                "model_params": {
                    "model_path": get_model_path_options(TASK_TYPE, MODEL_VARIANTS),
                    "inference_config": generate_inference_configs(TASK_TYPE, mode="minimal"),
                },
            },
        },

        # Baseline: Default TabPFN without finetuning for comparison
        {
            "model": create_model(TASK_TYPE, 'default', device=DEVICE),
            "name": "TabPFN-Default"
        },
    ]

    n_feature_options = len(create_feature_extraction_options())
    print(f"Pipeline configured with {n_feature_options} feature extraction options")
    print(f"Models: {len(MODEL_VARIANTS)} TabPFN variants")
    print()

    # =========================================================================
    # Run Pipeline
    # =========================================================================
    pipeline_config = PipelineConfigs(pipeline, "study_proto_2")

    runner = PipelineRunner(
        save_artifacts=True,
        verbose=args.verbose,
        plots_visible=False,
    )

    t0 = time.time()
    predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)
    training_time = time.time() - t0

    print()
    print(f"Training completed in {training_time:.1f}s ({training_time / 60:.1f} min)")
    print()

    # =========================================================================
    # Results Analysis
    # =========================================================================
    print("=" * 70)
    print("RESULTS ANALYSIS")
    print("=" * 70)
    print()

    ranking_metric = "rmse" if TASK_TYPE == "regression" else "balanced_accuracy"
    n_top = 15

    print(f"Top {n_top} results by {ranking_metric.upper()}:")
    print("-" * 70)
    top_models = predictions.top(n=n_top, rank_metric=ranking_metric, by_repetition="ID_1700_clean")

    for idx, prediction in enumerate(top_models, 10):
        score = prediction.get("test_score", prediction.get("val_score", None))
        if TASK_TYPE == "regression" and score is not None:
            display_score = math.sqrt(score)  # Convert MSE to RMSE
            score_str = f"RMSE={display_score:.4f}"
        else:
            score_str = f"score={score:.4f}" if score is not None else "score=N/A"

        model_name = prediction.get("model_name", "N/A")
        preprocessing = prediction.get("preprocessings", "N/A")

        print(f"{idx}. {score_str:>14} | {model_name:<20} | {preprocessing}")

    print()

    # Best result summary
    print("-" * 70)
    print("Best Result:")
    print("-" * 70)
    best = top_models[0]
    best_score = best.get("test_score", None)
    if TASK_TYPE == "regression" and best_score is not None:
        best_rmse = math.sqrt(best_score)
        print(f"  Test RMSE: {best_rmse:.4f}")
    else:
        print(f"  Test Score: {best_score:.4f}" if best_score else "  Test Score: N/A")
    print(f"  Model: {best.get('model_name', 'N/A')}")
    print(f"  Preprocessing: {best.get('preprocessings', 'N/A')}")
    print()

    # # Per-model summary
    # print("-" * 70)
    # print("Performance by Model:")
    # print("-" * 70)
    # model_scores = {}
    # for pred in predictions.filter_predictions(partition='test'):
    #     model_name = pred.get('model_name', 'Unknown')
    #     score = pred.get('test_score')
    #     if score is not None:
    #         if model_name not in model_scores:
    #             model_scores[model_name] = []
    #         model_scores[model_name].append(score)

    # for model_name, scores in sorted(model_scores.items()):
    #     avg_score = sum(scores) / len(scores)
    #     min_score = min(scores)
    #     if TASK_TYPE == "regression":
    #         avg_rmse = math.sqrt(avg_score)
    #         min_rmse = math.sqrt(min_score)
    #         print(f"  {model_name:<25} avg RMSE={avg_rmse:.4f}, best RMSE={min_rmse:.4f} (n={len(scores)})")
    #     else:
    #         print(f"  {model_name:<25} avg={avg_score:.4f}, best={min_score:.4f} (n={len(scores)})")
    # print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_feature_options = len(create_feature_extraction_options())
    print(f"  Feature extraction options: {n_feature_options}")
    print(f"  Model variants explored: {MODEL_VARIANTS}")
    print(f"  Finetuning trials: {N_TRIALS}")
    print(f"  Training time: {training_time:.1f}s ({training_time / 60:.1f} min)")
    print()

    # =========================================================================
    # Visualizations
    # =========================================================================
    output_dir = DATA_PATH.parent

    # print("Generating visualizations...")
    # analyzer = PredictionAnalyzer(predictions)

    # # Top K plot
    # try:
    #     fig1 = analyzer.plot_top_k(k=10, rank_metric=ranking_metric, rank_partition="test")
    #     if isinstance(fig1, list):
    #         for i, f in enumerate(fig1):
    #             f.savefig(output_dir / f"study_proto_2_top_k_{i}.png", dpi=150, bbox_inches="tight")
    #     else:
    #         fig1.savefig(output_dir / "study_proto_2_top_k.png", dpi=150, bbox_inches="tight")
    #     print("  Saved: study_proto_2_top_k.png")
    # except Exception as e:
    #     print(f"  Warning: Could not create top-k plot: {e}")

    # # Heatmap: model vs preprocessing
    # try:
    #     fig2 = analyzer.plot_heatmap(
    #         x_var="model_name",
    #         y_var="preprocessings",
    #         rank_metric=ranking_metric,
    #         rank_partition="test",
    #     )
    #     fig2.savefig(output_dir / "study_proto_2_heatmap.png", dpi=150, bbox_inches="tight")
    #     print("  Saved: study_proto_2_heatmap.png")
    # except Exception as e:
    #     print(f"  Warning: Could not create heatmap: {e}")

    # print()
    # print("=" * 70)
    # print("STUDY COMPLETE")
    # print("=" * 70)

    # if args.show:
    #     plt.show()


if __name__ == "__main__":
    main()
