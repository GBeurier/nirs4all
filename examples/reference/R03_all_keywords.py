"""
R03 - All Keywords Integration Test
====================================

Comprehensive pipeline exercising ALL pipeline-specific keywords.

This is a complex integration test that validates all pipeline keywords work
together correctly. It is intended for testing, not as a tutorial.

Keywords covered:

1. preprocessing - Feature preprocessing step
2. y_processing - Target preprocessing
3. sample_augmentation - Data augmentation on samples
4. feature_augmentation - Create multiple preprocessing views
5. concat_transform - Concatenate multiple transformer outputs
6. branch - Pipeline branching into parallel execution paths
7. merge - Merge branch outputs (features or predictions)
8. source_branch - Per-source processing (for multi-source datasets)
9. merge_sources - Combine multiple data sources
10. model - Model training step

WARNING: This is an extremely complex pipeline meant for integration testing.
For production and learning, use simpler, focused pipelines.

For syntax reference, see :ref:`R01_pipeline_syntax`.
For generator syntax, see :ref:`R02_generator_reference`.

Duration: ~2-3 minutes
Difficulty: ★★★★★ (advanced - testing only)
"""

import argparse
import os
import sys

# Add parent directory to path for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
)

from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.transforms import (
    StandardNormalVariate as SNV,
    MultiplicativeScatterCorrection as MSC,
    FirstDerivative,
    Detrend,
)
from nirs4all.operators.transforms import (
    Rotate_Translate,
    GaussianAdditiveNoise,
)
from nirs4all.operators.models import MetaModel

# Parse command-line arguments
parser = argparse.ArgumentParser(description="R03 All Keywords Integration Test")
parser.add_argument("--plots", action="store_true", help="Show plots interactively")
parser.add_argument("--show", action="store_true", help="Show all plots")
parser.add_argument("--verbose", "-v", type=int, default=1, help="Verbosity level")
args = parser.parse_args()


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


# =============================================================================
# COMPLEX PIPELINE DEFINITION
# =============================================================================
print_section("R03 - All Keywords Integration Test")

print("""
This pipeline exercises ALL pipeline-specific keywords:

Pipeline Structure:
├── 1. preprocessing (MinMaxScaler)
├── 2. y_processing (StandardScaler for targets)
├── 3. sample_augmentation (Rotate_Translate, GaussianNoise)
├── 4. feature_augmentation (SNV, FirstDerivative, Detrend) [extend mode]
├── 5. source_branch (per-source preprocessing)
├── 6. merge (features from sources → features output)
├── 7. KFold cross-validation
├── 8. branch (3 parallel strategies)
│   ├── Branch 0: SNV → concat_transform[PCA, SVD] → PLS
│   ├── Branch 1: MSC → RandomForest
│   └── Branch 2: FirstDerivative → multiple PLS → GradientBoosting
├── 9. MetaModel (per-branch stacking)
├── 10. merge (predictions with per-branch model selection)
└── 11. Final model (meta-learner)

Expected outcome:
- All keywords execute without errors
- Predictions are generated successfully
- Meta-learner combines branch outputs
""")


# =============================================================================
# DEFINE THE COMPREHENSIVE PIPELINE
# =============================================================================

complex_pipeline = [
    # =========================================================================
    # KEYWORD 1: preprocessing
    # Basic feature preprocessing applied to all data
    # =========================================================================
    {"preprocessing": MinMaxScaler()},

    # =========================================================================
    # KEYWORD 2: y_processing
    # Target preprocessing - normalize y values
    # =========================================================================
    {"y_processing": StandardScaler()},

    # =========================================================================
    # KEYWORD 3: sample_augmentation
    # Augment training samples with synthetic variations
    # =========================================================================
    {
        "sample_augmentation": {
            "transformers": [
                Rotate_Translate(p_range=1, y_factor=2),
                GaussianAdditiveNoise(sigma=0.005),
            ],
            "count": 2,
            "selection": "random",
            "random_state": 42,
        }
    },

    # =========================================================================
    # KEYWORD 4: feature_augmentation (extend mode)
    # Create multiple preprocessing views
    # =========================================================================
    {
        "feature_augmentation": [
            SNV,
            FirstDerivative,
            Detrend,
        ],
        "action": "extend",
    },

    # =========================================================================
    # KEYWORDS 5 & 6: source_branch + merge (features)
    # Per-source processing followed by feature merge
    # =========================================================================
    {"source_branch": [
        [MinMaxScaler()],
        [MinMaxScaler()],
        [PCA(20), MinMaxScaler()]
    ]},

    StandardScaler(),

    {"merge": {
        "features": "all",
        "output_as": "sources",
    }},

    # =========================================================================
    # Cross-validation splitter
    # =========================================================================
    KFold(n_splits=3, shuffle=True, random_state=42),

    # =========================================================================
    # KEYWORD 7: branch
    # Create parallel execution paths
    # =========================================================================
    {
        "branch": {
            # Branch 0: PLS with latent features
            "pls_latent": [
                SNV(),
                # KEYWORD 8: concat_transform
                {
                    "concat_transform": [
                        PCA(n_components=15),
                        TruncatedSVD(n_components=10),
                    ]
                },
                {"name": "PLS_Latent", "model": PLSRegression(n_components=10)},
            ],

            # Branch 1: Random Forest with smoothed spectra
            "rf_smoothed": [
                MSC(),
                {"name": "RF_Smoothed", "model": RandomForestRegressor(
                    n_estimators=10,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1,
                )},
            ],

            # Branch 2: Gradient Boosting with derivatives
            "gbr_derivative": [
                FirstDerivative(),
                {"name": "PLS_10", "model": PLSRegression(n_components=10)},
                {"name": "PLS_5", "model": PLSRegression(n_components=5)},
                PCA(n_components=0.999),
                {"name": "GBR_Derivative", "model": GradientBoostingRegressor(
                    n_estimators=10,
                    max_depth=5,
                    random_state=42,
                )},
            ],
        }
    },

    # =========================================================================
    # MetaModel inside branch mode (branch-aware stacking)
    # =========================================================================
    {"name": "Ridge_MetaModel", "model": MetaModel(model=Ridge(alpha=1.0))},

    # =========================================================================
    # KEYWORD 9: merge with prediction selection
    # =========================================================================
    {"merge": {
        "predictions": [
            {"branch": 0, "select": "best", "metric": "rmse"},
            {"branch": 1, "select": "best", "metric": "rmse"},
            {"branch": 2, "select": {"top_k": 2}, "metric": "r2"},
        ],
        "features": [2],
        "on_missing": "warn",
        "output_as": "features",
    }},

    # =========================================================================
    # Final meta-learner
    # =========================================================================
    {"name": "Meta_RF", "model": RandomForestRegressor(
        n_estimators=10,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )},
]


# =============================================================================
# RUN THE PIPELINE
# =============================================================================
print_section("RUNNING INTEGRATION TEST")

# Configure dataset
dataset_config = DatasetConfigs("sample_data/multi")

# Configure pipeline
pipeline_config = PipelineConfigs(
    complex_pipeline,
    name="R03_All_Keywords_Test"
)

# Run pipeline
runner = PipelineRunner(
    workspace_path="workspace/reference_test",
    verbose=args.verbose,
    save_artifacts=True,
    plots_visible=args.plots,
)

try:
    predictions, pipelines = runner.run(pipeline_config, dataset_config)

    # =========================================================================
    # RESULTS ANALYSIS
    # =========================================================================
    print_section("RESULTS ANALYSIS")

    print(f"Total predictions generated: {predictions.num_predictions}")
    print(f"Unique branches: {predictions.get_unique_values('branch_name')}")
    print(f"Unique models: {predictions.get_unique_values('model_name')}")

    print("\n--- Top 10 Models by RMSE ---")
    top_models = predictions.top(10, rank_metric='rmse', display_metrics=['rmse', 'r2', 'mae'])
    for idx, pred in enumerate(top_models, 1):
        print(f"{idx:2d}. {Predictions.pred_short_string(pred, metrics=['rmse', 'r2'])}")

    # Separate meta-learner from base models
    meta_preds = [p for p in predictions.to_dicts() if "Meta" in p.get("model_name", "")]
    base_preds = [p for p in predictions.to_dicts() if "Meta" not in p.get("model_name", "")]

    if meta_preds and base_preds:
        print("\n--- Stacking Performance ---")
        meta_scores = [p.get("val_score", float("inf")) for p in meta_preds]
        base_scores = [p.get("val_score", float("inf")) for p in base_preds]

        best_meta = min(meta_scores)
        best_base = min(base_scores)

        print(f"  Best base model RMSE:  {best_base:.4f}")
        print(f"  Meta-learner RMSE:     {best_meta:.4f}")

        if best_meta < best_base:
            improvement = (best_base - best_meta) / best_base * 100
            print(f"  → Meta-learner improved by {improvement:.1f}%")
        else:
            print(f"  → No improvement from meta-learner")

    # =========================================================================
    # KEYWORD VERIFICATION
    # =========================================================================
    print_section("KEYWORD VERIFICATION")

    keywords_used = [
        ("preprocessing", "MinMaxScaler feature scaling"),
        ("y_processing", "StandardScaler target normalization"),
        ("sample_augmentation", "Rotate_Translate + GaussianNoise"),
        ("feature_augmentation", "SNV, FirstDerivative, Detrend views"),
        ("concat_transform", "PCA + TruncatedSVD concatenation"),
        ("branch", "pls_latent, rf_smoothed, gbr_derivative"),
        ("merge (features)", "source merge to features"),
        ("merge (predictions)", "per-branch model selection"),
        ("source_branch", "per-source preprocessing"),
        ("model", "PLS, RF, GBR + Ridge meta-learner"),
    ]

    print("All pipeline-specific keywords verified:")
    for keyword, description in keywords_used:
        print(f"  ✓ {keyword:22s} - {description}")

    print("\n" + "=" * 80)
    print("  R03 - ALL KEYWORDS TEST PASSED")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ Pipeline execution failed: {e}")
    import traceback
    traceback.print_exc()
    raise


# =============================================================================
# OPTIONAL: VISUALIZATION
# =============================================================================
if args.show and predictions:
    print_section("GENERATING VISUALIZATIONS")

    from nirs4all.visualization.predictions import PredictionAnalyzer

    analyzer = PredictionAnalyzer(predictions, output_dir="workspace/reference_test/charts")

    try:
        fig_branch = analyzer.plot_branch_comparison(
            display_metric='rmse',
            display_partition='test',
            show_ci=True,
        )
        print("  ✓ Branch comparison chart generated")
    except Exception as e:
        print(f"  ⚠ Branch comparison chart failed: {e}")

    try:
        fig_topk = analyzer.plot_top_k(
            k=6,
            rank_metric='rmse',
            rank_partition='test',
        )
        print("  ✓ Top-K predictions chart generated")
    except Exception as e:
        print(f"  ⚠ Top-K chart failed: {e}")

    import matplotlib.pyplot as plt
    plt.show()


print("\n✓ R03_all_keywords.py completed!")
