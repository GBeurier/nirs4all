"""
R03 - All Keywords Integration Test
====================================

Comprehensive pipeline exercising ALL pipeline-specific keywords.

This is a complex integration test that validates all pipeline keywords work
together correctly. It is intended for testing, not as a tutorial.

Keywords covered (v2):

1. preprocessing - Feature preprocessing step
2. y_processing - Target preprocessing
3. tag - Tag samples without removing (v2)
4. exclude - Exclude samples from training (v2)
5. feature_augmentation - Create multiple preprocessing views
6. concat_transform - Concatenate multiple transformer outputs
7. branch - Pipeline branching (duplication branches)
8. merge - Merge branch outputs (features or predictions)
9. model - Model training step

Note: The following keywords are tested separately:
- sample_augmentation - Tested in other examples (interaction with tag columns)
- by_source branch, merge sources - See D04_merge_sources.py, D06_separation_branches.py

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
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
)

from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions

# Note: sample_augmentation imports (Rotate_Translate, GaussianAdditiveNoise)
# removed due to known interaction issues with tag columns
from nirs4all.operators.filters import YOutlierFilter
from nirs4all.operators.models import MetaModel
from nirs4all.operators.models.meta import StackingConfig
from nirs4all.operators.transforms import (
    Detrend,
    FirstDerivative,
)
from nirs4all.operators.transforms import (
    MultiplicativeScatterCorrection as MSC,
)
from nirs4all.operators.transforms import (
    StandardNormalVariate as SNV,
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

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
This pipeline exercises ALL pipeline-specific keywords (v2):

Pipeline Structure:
├── 1. preprocessing (MinMaxScaler)
├── 2. y_processing (StandardScaler for targets)
├── 3. tag (YOutlierFilter zscore and iqr - v2)
├── 4. exclude (YOutlierFilter zscore - extreme outliers)
├── 5. feature_augmentation (SNV, FirstDerivative) [replace mode]
├── 6. KFold cross-validation
├── 7. branch (3 parallel strategies - duplication branch)
│   ├── Branch 0: SNV → concat_transform[PCA, SVD] → PLS
│   ├── Branch 1: MSC → RandomForest
│   └── Branch 2: FirstDerivative → multiple PLS → GradientBoosting
├── 8. MetaModel (per-branch stacking)
├── 9. merge (predictions with per-branch model selection)
└── 10. Final model (meta-learner)

Note: by_source branch and merge sources are tested in D04 and D06.

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
    # KEYWORD 3: tag (v2)
    # Tag outliers without removing them - for analysis
    # =========================================================================
    {"tag": YOutlierFilter(method="zscore", threshold=3.0, tag_name="zscore_outlier")},
    {"tag": YOutlierFilter(method="iqr", threshold=3.0, tag_name="iqr_outlier")},

    # =========================================================================
    # KEYWORD 4: exclude (v2)
    # Exclude extreme outliers from training (keeps them for prediction)
    # =========================================================================
    {"exclude": YOutlierFilter(method="zscore", threshold=4.0)},

    # =========================================================================
    # KEYWORD 5: feature_augmentation (replace mode)
    # Create multiple preprocessing views
    # Note: Using "replace" instead of "extend" to avoid source multiplication
    # Note: by_source branch tested separately in D04_merge_sources.py and D06_separation_branches.py
    # =========================================================================
    {
        "feature_augmentation": [
            SNV,
            FirstDerivative,
        ],
        "action": "replace",
    },

    # =========================================================================
    # Cross-validation splitter
    # =========================================================================
    KFold(n_splits=3, shuffle=True, random_state=42),

    # =========================================================================
    # KEYWORD 9: branch (duplication branch)
    # Create parallel execution paths - same samples, different preprocessing
    # =========================================================================
    {
        "branch": {
            # Branch 0: PLS with latent features
            "pls_latent": [
                SNV(),
                # KEYWORD 10: concat_transform
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
    {
        "name": "Ridge_MetaModel",
        "model": MetaModel(
            model=Ridge(alpha=1.0),
            stacking_config=StackingConfig(
                coverage_strategy="drop_incomplete",
                min_coverage_ratio=0.95,
            ),
        ),
    },

    # =========================================================================
    # KEYWORD 11: merge with prediction selection
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
            print("  → No improvement from meta-learner")

    # =========================================================================
    # KEYWORD VERIFICATION
    # =========================================================================
    print_section("KEYWORD VERIFICATION")

    keywords_used = [
        ("preprocessing", "MinMaxScaler feature scaling"),
        ("y_processing", "StandardScaler target normalization"),
        ("tag (v2)", "YOutlierFilter zscore + iqr tagging"),
        ("exclude (v2)", "YOutlierFilter zscore extreme exclusion"),
        ("feature_augmentation", "SNV, FirstDerivative views"),
        ("concat_transform", "PCA + TruncatedSVD concatenation"),
        ("branch (duplication)", "pls_latent, rf_smoothed, gbr_derivative"),
        ("merge (predictions)", "per-branch model selection"),
        ("model", "PLS, RF, GBR + Ridge meta-learner"),
        # Note: by_source tested in D04, D06
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
