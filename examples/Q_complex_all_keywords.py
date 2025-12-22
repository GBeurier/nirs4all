"""
Q_complex_all_keywords.py - Comprehensive Pipeline Using ALL Keywords

This is a complex test pipeline that exercises ALL pipeline-specific keywords:
1. preprocessing - Feature preprocessing step
2. feature_augmentation - Create multiple preprocessing views
3. y_processing - Target preprocessing
4. sample_augmentation - Data augmentation on samples
5. concat_transform - Concatenate multiple transformer outputs
6. branch - Pipeline branching into parallel execution paths
7. merge - Merge branch outputs (features or predictions)
8. source_branch - Per-source processing (for multi-source datasets)
9. merge_sources - Combine multiple data sources
10. model - Model training step

This pipeline demonstrates:
- Multi-stage preprocessing with chained operations
- Feature augmentation with different action modes
- Sample augmentation for training data enrichment
- Branching for comparing multiple strategies
- Prediction stacking with meta-learners
- Source-level operations (demonstrated as pass-through for single-source)
- Y-processing for target normalization

WARNING: This is an extremely complex pipeline meant for testing.
For production, use simpler, focused pipelines.
"""

import argparse
import os
import sys

# Add parent directory to path for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    QuantileTransformer,
)

from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.transforms import (
    StandardNormalVariate as SNV,
    MultiplicativeScatterCorrection as MSC,
    FirstDerivative,
    SecondDerivative,
    SavitzkyGolay,
    Detrend,
    Gaussian,
)
from nirs4all.operators.transforms import (
    Rotate_Translate,
    GaussianAdditiveNoise,
    MultiplicativeNoise,
)
from nirs4all.visualization.predictions import PredictionAnalyzer
from nirs4all.visualization.pipeline_diagram import plot_pipeline_diagram, PipelineDiagram

from nirs4all.operators.models import MetaModel

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Complex All Keywords Example")
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
print_section("COMPLEX PIPELINE: All Keywords Test")

print("""
This pipeline exercises ALL pipeline-specific keywords:

Pipeline Structure:
├── 1. preprocessing (MinMaxScaler)
├── 2. y_processing (StandardScaler for targets)
├── 3. sample_augmentation (Rotate_Translate, GaussianNoise)
├── 4. feature_augmentation (SNV, FirstDerivative, Detrend) [extend mode]
├── 5. source_branch ("auto" for single-source compatibility)
├── 6. merge_sources ("concat" to recombine)
├── 7. KFold cross-validation
├── 8. branch (3 parallel strategies)
│   ├── Branch 0: SNV → concat_transform[PCA, SVD] → PLS
│   ├── Branch 1: MSC → SavitzkyGolay → RandomForest
│   └── Branch 2: FirstDerivative → PLS models → PCA → GradientBoosting
├── 9. merge (predictions with PER-BRANCH MODEL SELECTION)
│   ├── Branch 0: best by RMSE
│   ├── Branch 1: best by RMSE
│   └── Branch 2: top 2 by R²
└── 10. model (Ridge meta-learner on stacked predictions)

NOTE on MetaModel vs merge+model:
  - MetaModel is a convenience wrapper equivalent to: merge + model
  - Both approaches exit branch mode and train a meta-learner
  - Use MetaModel for concise syntax, or merge+model for explicit control
  - Do NOT place MetaModel INSIDE branch mode (before merge)

Expected outcome:
- Multiple preprocessing views created via feature_augmentation
- Sample augmentation enriches training data
- Three branches process data with different strategies
- Branch models produce OOF predictions
- Per-branch selection: best from Branch 0 & 1, top 2 from Branch 2
- Predictions are merged and fed to meta-learner
- Final Ridge model learns to combine branch outputs (4 input features)
""")


# =============================================================================
# DEFINE THE MEGA-PIPELINE
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
    # This creates additional training samples with spectral perturbations
    # =========================================================================
    {
        "sample_augmentation": {
            "transformers": [
                Rotate_Translate(p_range=1, y_factor=2),
                GaussianAdditiveNoise(sigma=0.005),
            ],
            "count": 2,  # Create 2 augmented samples per original
            "selection": "random",  # Randomly select which transformer to use
            "random_state": 42,
        }
    },

    # =========================================================================
    # KEYWORD 4: feature_augmentation (extend mode)
    # Create multiple preprocessing views that will be evaluated
    # Using "extend" mode to add independent preprocessing options
    # =========================================================================
    {
        "feature_augmentation": [
            SNV,  # Standard Normal Variate
            FirstDerivative,  # First derivative
            Detrend,  # Detrending
        ],
        "action": "extend",  # Add as independent preprocessing options
    },

    # =========================================================================
    # KEYWORD 8 & 9: source_branch + merge_sources
    # For multi-source datasets, this would process each source differently
    # For single-source, "auto" mode passes through transparently
    # =========================================================================
    {"source_branch": "auto"},  # Per-source processing (pass-through for single-source)
    MinMaxScaler(),  # Re-apply scaling after source branching
    {"merge_sources": "concat"},  # Concatenate sources back together

    # =========================================================================
    # Cross-validation splitter
    # =========================================================================
    KFold(n_splits=3, shuffle=True, random_state=42),

    # =========================================================================
    # KEYWORD 6: branch
    # Create parallel execution paths for different modeling strategies
    # Each branch will produce its own OOF predictions
    # =========================================================================
    {
        "branch": {
            # Branch 0: "pls_latent" - PLS with latent features
            "pls_latent": [
                SNV(),
                # KEYWORD 5: concat_transform
                # Concatenate PCA and SVD features for richer representation
                {
                    "concat_transform": [
                        PCA(n_components=15),
                        TruncatedSVD(n_components=10),
                    ]
                },
                # Model within branch
                {"name": "PLS_Latent", "model": PLSRegression(n_components=10)},
            ],

            # Branch 1: "rf_smoothed" - Random Forest with smoothed spectra
            "rf_smoothed": [
                MSC(),
                SavitzkyGolay(window_length=11, polyorder=2),
                {"name": "RF_Smoothed", "model": RandomForestRegressor(
                    n_estimators=10,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1,
                )},
            ],

            # Branch 2: "gbr_derivative" - Gradient Boosting with derivatives
            "gbr_derivative": [
                FirstDerivative(),
                {"name": "PLS_10", "model": PLSRegression(n_components=10)},
                {"name": "PLS_5", "model": PLSRegression(n_components=5)},
                PCA(n_components=20),  # Reduce dimensionality
                {"name": "GBR_Derivative", "model": GradientBoostingRegressor(
                    n_estimators=10,
                    max_depth=5,
                    random_state=42,
                )},
                {"name": "PLS_Latent2", "model": PLSRegression(n_components=10)},
            ],
        }
    },

    # =========================================================================
    # KEYWORD 7: merge with MODEL SELECTION STRATEGY PER BRANCH
    # Collect OOF predictions from all branches for stacking
    # Using per-branch selection strategies to control which models contribute
    #
    # NOTE: merge ALWAYS exits branch mode. MetaModel is equivalent to
    # merge + model, so it should come AFTER branching ends, not inside it.
    # =========================================================================
    {"merge": {
        "predictions": [
            # Branch 0 (pls_latent): Use single model → best by RMSE
            {"branch": 0, "select": "best", "metric": "rmse"},
            # Branch 1 (rf_smoothed): Use single model → best by RMSE
            {"branch": 1, "select": "best", "metric": "rmse"},
            # Branch 2 (gbr_derivative): Multiple models → top 2 by R²
            {"branch": 2, "select": {"top_k": 2}, "metric": "r2"},
        ],
        "on_missing": "warn",  # Warn if a branch has no predictions
    }},

    # =========================================================================
    # KEYWORD 10: model (Meta-learner)
    # Meta-learner that combines predictions from all branches
    # Ridge regression to learn optimal combination weights
    #
    # This is equivalent to using MetaModel:
    #   {"model": MetaModel(Ridge(alpha=1.0))}
    # But since we already have merge above, we use a regular model.
    # =========================================================================
    {"name": "Meta_Ridge", "model": Ridge(alpha=1.0)},
]


# =============================================================================
# PIPELINE DIAGRAM (STATIC - Before Execution)
# =============================================================================
print_section("PIPELINE DIAGRAM (STATIC)")

print("Generating static pipeline structure diagram...")
print("Shape notation: (samples, features) for 2D layout, [S×P×F] for 3D features\n")

try:
    # Generate static pipeline diagram with estimated shapes
    # This shows the intended pipeline structure before execution
    fig_pipeline = plot_pipeline_diagram(
        pipeline_steps=complex_pipeline,
        show_shapes=True,
        initial_shape=(189, 1, 2151),  # Typical NIRS dataset shape
        title="Complex Pipeline Structure (Static - Pre-Execution)",
        figsize=(16, 12),
    )

    # Save diagram
    import os
    os.makedirs("workspace/complex_test/charts", exist_ok=True)
    fig_pipeline.savefig(
        "workspace/complex_test/charts/pipeline_diagram_static.png",
        dpi=150,
        bbox_inches='tight',
        facecolor='white'
    )
    print("  ✓ Static diagram saved to workspace/complex_test/charts/pipeline_diagram_static.png")

    if args.plots:
        import matplotlib.pyplot as plt
        plt.show(block=False)
except Exception as e:
    print(f"  ⚠ Static diagram generation failed: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# RUN THE COMPLEX PIPELINE
# =============================================================================
print_section("RUNNING COMPLEX PIPELINE")

# Configure dataset
dataset_config = DatasetConfigs("sample_data/multi")

# Configure pipeline
pipeline_config = PipelineConfigs(
    complex_pipeline,
    name="Complex_All_Keywords_Test"
)

# Run pipeline
runner = PipelineRunner(
    workspace_path="workspace/complex_test",
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
        print("\n--- Stacking Performance Comparison ---")
        meta_scores = [p.get("val_score", float("inf")) for p in meta_preds]
        base_scores = [p.get("val_score", float("inf")) for p in base_preds]

        best_meta = min(meta_scores)
        best_base = min(base_scores)
        avg_base = np.mean(base_scores)

        print(f"  Best base model RMSE:    {best_base:.4f}")
        print(f"  Average base model RMSE: {avg_base:.4f}")
        print(f"  Meta-learner RMSE:       {best_meta:.4f}")

        if best_meta < best_base:
            improvement = (best_base - best_meta) / best_base * 100
            print(f"\n  ✓ Meta-learner improved by {improvement:.1f}% over best base model!")
        else:
            print(f"\n  ⚠ Meta-learner did not improve over best base model")

    # =========================================================================
    # PIPELINE DIAGRAM (DYNAMIC - After Execution)
    # =========================================================================
    print_section("PIPELINE DIAGRAM (DYNAMIC)")

    print("Generating dynamic pipeline diagram from execution trace...")
    print("Shape notation: (samples, features) for 2D layout, [S×P×F] for 3D features\n")

    try:
        # Get the execution trace from the runner
        execution_trace = runner.last_execution_trace

        if execution_trace:
            print(f"  Trace contains {len(execution_trace.steps)} executed steps")

            # Generate diagram from trace with actual runtime shapes
            diagram = PipelineDiagram.from_trace(
                execution_trace=execution_trace,
                config={'fontsize': 7, 'figsize': (18, 14)}
            )
            fig_dynamic = diagram.render(
                show_shapes=True,
                title="Complex Pipeline Execution (Dynamic - Actual Shapes)",
            )

            # Save dynamic diagram
            fig_dynamic.savefig(
                "workspace/complex_test/charts/pipeline_diagram_dynamic.png",
                dpi=150,
                bbox_inches='tight',
                facecolor='white'
            )
            print("  ✓ Dynamic diagram saved to workspace/complex_test/charts/pipeline_diagram_dynamic.png")

            if args.plots:
                import matplotlib.pyplot as plt
                plt.show(block=False)
        else:
            print("  ⚠ No execution trace available (trace recording may be disabled)")

    except Exception as e:
        print(f"  ⚠ Dynamic diagram generation failed: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # KEYWORD VERIFICATION
    # =========================================================================
    print_section("KEYWORD VERIFICATION")

    keywords_used = [
        ("preprocessing", "MinMaxScaler feature scaling"),
        ("y_processing", "StandardScaler target normalization"),
        ("sample_augmentation", "Rotate_Translate + GaussianNoise augmenters"),
        ("feature_augmentation", "SNV, FirstDerivative, Detrend views"),
        ("concat_transform", "PCA + TruncatedSVD concatenation"),
        ("branch", "pls_latent, rf_smoothed, gbr_derivative branches"),
        ("merge", "predictions stacking with per-branch selection (best, top_k)"),
        ("source_branch", "auto mode (pass-through for single-source)"),
        ("merge_sources", "concat mode"),
        ("model", "PLS, RF, GBR base models + Ridge meta-learner"),
    ]

    print("All pipeline-specific keywords used in this example:")
    for keyword, description in keywords_used:
        print(f"  ✓ {keyword:22s} - {description}")

    print("\n" + "="*80)
    print("  COMPLEX PIPELINE TEST COMPLETED SUCCESSFULLY!")
    print("="*80)

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

    analyzer = PredictionAnalyzer(predictions, output_dir="workspace/complex_test/charts")

    # Pipeline diagram (already generated above, show it again)
    try:
        fig_pipeline = plot_pipeline_diagram(
            pipeline_steps=complex_pipeline,
            predictions=predictions,
            show_shapes=True,
            initial_shape=(189, 1, 2151),
            title="Complex Pipeline Structure (All Keywords)",
            figsize=(16, 12),
        )
        print("  ✓ Pipeline diagram generated")
    except Exception as e:
        print(f"  ⚠ Pipeline diagram failed: {e}")

    # Branch comparison
    try:
        fig_branch = analyzer.plot_branch_comparison(
            display_metric='rmse',
            display_partition='test',
            show_ci=True,
        )
        print("  ✓ Branch comparison chart generated")
    except Exception as e:
        print(f"  ⚠ Branch comparison chart failed: {e}")

    # Candlestick by model
    try:
        fig_candle = analyzer.plot_candlestick(
            variable="model_name",
            display_metric="rmse",
            display_partition="test",
        )
        print("  ✓ Candlestick chart generated")
    except Exception as e:
        print(f"  ⚠ Candlestick chart failed: {e}")

    # Top-K predictions
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

print("\n✓ Q_complex_all_keywords.py completed!")
