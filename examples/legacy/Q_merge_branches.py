"""
Q_merge_branches.py - Branch Merging and Stacking Examples

This example demonstrates the merge controller for combining branch outputs
and exiting branch mode. Merge is the CORE PRIMITIVE for all branch combination
operations, enabling:

1. Feature merging - Collect features from multiple branches
2. Prediction merging - Stack OOF predictions from branch models (safe by default)
3. Mixed merging - Combine features from some branches, predictions from others
4. Asymmetric branches - Handle branches with different models/features

Key concepts:
- Merge ALWAYS exits branch mode (returns to single-path execution)
- Prediction merging uses OOF reconstruction by default (prevents data leakage)
- `unsafe=True` disables OOF for advanced users (with prominent warnings)
- Merge + model is equivalent to MetaModel (but more flexible)

Examples included:
1. Basic feature merge - Combine features from all branches
2. Prediction merge - Stack predictions for stacking ensemble
3. Mixed merge - Features from some branches, predictions from others
4. Per-branch selection - Choose which models from each branch
5. Per-branch aggregation - Mean, weighted_mean, or separate predictions
6. Asymmetric branches - Handle branches with different capabilities
7. Merge vs MetaModel - Show equivalence between approaches
8. Output targets - Control merge output format (features, sources, dict)

Phase 8 Features:
- Prediction mode support for merge steps
- Full train/predict cycle with bundled models

See also:
- Q30_branching.py - Basic branching examples
- Q_meta_stacking.py - MetaModel stacking examples
- Q_merge_sources.py - Multi-source dataset merging
"""

import argparse
import os
import sys

# Add parent directory to path for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.transforms import (
    StandardNormalVariate as SNV,
    MultiplicativeScatterCorrection as MSC,
    FirstDerivative,
    SecondDerivative,
    SavitzkyGolay,
)
from nirs4all.visualization.predictions import PredictionAnalyzer

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Q_merge_branches Example")
parser.add_argument("--plots", action="store_true", help="Show plots interactively")
parser.add_argument("--show", action="store_true", help="Show all plots")
parser.add_argument("--example", type=int, default=0, help="Run specific example (1-8, 0=all)")
args = parser.parse_args()


# =============================================================================
# Example 1: Basic Feature Merge
# =============================================================================
def example_1_basic_feature_merge():
    """
    Basic feature merge: Collect and concatenate features from all branches.

    Use case: Compare multiple preprocessing strategies, then combine their
    outputs into a single feature set for a downstream model.

    Pipeline structure:
        branch → [SNV | MSC | D1] → merge(features) → PLS

    Result: PLS sees concatenated features from all three preprocessing paths.
    """
    print("=" * 70)
    print("Example 1: Basic Feature Merge")
    print("=" * 70)

    dataset = DatasetConfigs("sample_data/regression")

    pipeline = [
        MinMaxScaler(),
        KFold(n_splits=3, shuffle=True, random_state=42),

        # Create 3 branches with different preprocessing
        {"branch": [
            [SNV()],           # Branch 0: SNV
            [MSC()],           # Branch 1: MSC
            [FirstDerivative()],  # Branch 2: First derivative
        ]},

        # Merge: Collect features from ALL branches and exit branch mode
        # Result: [SNV_features | MSC_features | D1_features]
        {"merge": "features"},

        # Single model on merged features (runs 1x, not 3x)
        PLSRegression(n_components=10),
    ]

    runner = PipelineRunner(workspace_path="workspace/merge_branches", verbose=1)
    predictions, _ = runner.run(pipeline, dataset)

    # Display results
    print("\nResults:")
    for pred in predictions.top(5, "rmse"):
        print(f"  {Predictions.pred_short_string(pred, metrics=['rmse', 'r2'])}")

    print("\n✓ Feature merge complete - 3 preprocessing paths → 1 combined feature set")
    return predictions


# =============================================================================
# Example 2: Prediction Merge (Stacking Foundation)
# =============================================================================
def example_2_prediction_merge():
    """
    Prediction merge: Stack OOF predictions from branch models.

    Use case: Each branch has a trained model. Collect their OOF predictions
    and use as features for a meta-learner.

    IMPORTANT: OOF reconstruction is MANDATORY by default to prevent data leakage.
    Each sample's prediction comes from a model that never saw it during training.

    Pipeline structure:
        branch → [SNV+PLS | MSC+RF | D1+XGB] → merge(predictions) → Ridge

    Result: Ridge trains on stacked OOF predictions from 3 branch models.
    """
    print("\n" + "=" * 70)
    print("Example 2: Prediction Merge (Stacking)")
    print("=" * 70)

    dataset = DatasetConfigs("sample_data/regression")

    pipeline = [
        MinMaxScaler(),
        KFold(n_splits=5, shuffle=True, random_state=42),

        # Each branch has preprocessing + model
        {"branch": [
            [SNV(), PLSRegression(n_components=10)],
            [MSC(), RandomForestRegressor(n_estimators=50, random_state=42)],
            [FirstDerivative(), GradientBoostingRegressor(n_estimators=50, random_state=42)],
        ]},

        # Merge: Collect OOF predictions from all branches
        # Uses OOF reconstruction automatically (safe)
        {"merge": "predictions"},

        # Meta-learner trains on stacked predictions
        Ridge(alpha=1.0),
    ]

    runner = PipelineRunner(workspace_path="workspace/merge_branches", verbose=1)
    predictions, _ = runner.run(pipeline, dataset)

    print("\nResults:")
    for pred in predictions.top(5, "rmse"):
        name = pred.get("model_name", "unknown")
        score = pred.get("val_score", 0)
        print(f"  {name:40s}: {score:.4f}")

    # Compare meta-learner with individual branch models
    print("\n📊 Stacking Performance:")
    meta_preds = [p for p in predictions.to_dicts() if "Ridge" in p.get("model_name", "")]
    base_preds = [p for p in predictions.to_dicts() if "Ridge" not in p.get("model_name", "")]

    if meta_preds and base_preds:
        meta_score = min(p.get("val_score", float("inf")) for p in meta_preds)
        best_base_score = min(p.get("val_score", float("inf")) for p in base_preds)
        print(f"  Best base model: {best_base_score:.4f}")
        print(f"  Meta-learner:    {meta_score:.4f}")

    print("\n✓ Prediction merge complete - OOF predictions stacked safely")
    return predictions


# =============================================================================
# Example 3: Mixed Merge (Features + Predictions)
# =============================================================================
def example_3_mixed_merge():
    """
    Mixed merge: Combine predictions from some branches with features from others.

    Use case: Branch 0 has a strong model (PLS), Branch 1 has good features (PCA).
    Combine PLS predictions with PCA features for the meta-learner.

    Pipeline structure:
        branch → [SNV+PLS | PCA(30)] → merge(predictions=[0], features=[1]) → Ridge

    Result: Ridge trains on [PLS_oof_prediction | PCA_features] = 31 features
    """
    print("\n" + "=" * 70)
    print("Example 3: Mixed Merge (Features + Predictions)")
    print("=" * 70)

    dataset = DatasetConfigs("sample_data/regression")

    pipeline = [
        MinMaxScaler(),
        KFold(n_splits=5, shuffle=True, random_state=42),

        # Branch 0: SNV + PLS (produces predictions)
        # Branch 1: PCA (produces features only - no model)
        {"branch": [
            [SNV(), PLSRegression(n_components=10)],
            [PCA(n_components=30)],
        ]},

        # Mixed merge: predictions from branch 0, features from branch 1
        {"merge": {
            "predictions": [0],  # OOF predictions from branch 0 (PLS)
            "features": [1],     # Features from branch 1 (PCA)
        }},

        # Meta-learner sees: [PLS_pred (1 dim) | PCA_features (30 dim)]
        Ridge(alpha=1.0),
    ]

    runner = PipelineRunner(workspace_path="workspace/merge_branches", verbose=1)
    predictions, _ = runner.run(pipeline, dataset)

    print("\nResults:")
    for pred in predictions.top(5, "rmse"):
        print(f"  {Predictions.pred_short_string(pred, metrics=['rmse', 'r2'])}")

    print("\n✓ Mixed merge complete - PLS predictions + PCA features combined")
    return predictions


# =============================================================================
# Example 4: Per-Branch Model Selection
# =============================================================================
def example_4_per_branch_selection():
    """
    Per-branch model selection: Choose which models to include from each branch.

    Use case: Branch 0 has multiple PLS variants, Branch 1 has ensemble models.
    Select the best PLS from branch 0, top 2 models from branch 1.

    Selection strategies:
    - "all": Include all models
    - "best": Include only the best model (by validation metric)
    - {"top_k": N}: Include top N models
    - ["model1", "model2"]: Explicit model names

    Pipeline structure:
        branch → [PLS(5,10,15) | RF, XGB] → merge(per-branch selection) → Ridge
    """
    print("\n" + "=" * 70)
    print("Example 4: Per-Branch Model Selection")
    print("=" * 70)

    dataset = DatasetConfigs("sample_data/regression")

    pipeline = [
        MinMaxScaler(),
        KFold(n_splits=5, shuffle=True, random_state=42),

        # Branch 0: Multiple PLS variants
        # Branch 1: Ensemble models
        {"branch": [
            [
                SNV(),
                {"name": "PLS_5", "model": PLSRegression(n_components=5)},
                {"name": "PLS_10", "model": PLSRegression(n_components=10)},
                {"name": "PLS_15", "model": PLSRegression(n_components=15)},
            ],
            [
                MSC(),
                {"name": "RF", "model": RandomForestRegressor(n_estimators=50, random_state=42)},
                {"name": "GBR", "model": GradientBoostingRegressor(n_estimators=50, random_state=42)},
            ],
        ]},

        # Per-branch selection:
        # - Branch 0: Best PLS (1 prediction)
        # - Branch 1: All models (2 predictions)
        {"merge": {
            "predictions": [
                {"branch": 0, "select": "best", "metric": "rmse"},
                {"branch": 1, "select": "all"},
            ]
        }},

        Ridge(alpha=1.0),
    ]

    runner = PipelineRunner(workspace_path="workspace/merge_branches", verbose=1)
    predictions, _ = runner.run(pipeline, dataset)

    print("\nResults:")
    for pred in predictions.top(5, "rmse"):
        print(f"  {Predictions.pred_short_string(pred, metrics=['rmse', 'r2'])}")

    print("\n✓ Per-branch selection complete - best PLS + all ensemble models")
    return predictions


# =============================================================================
# Example 5: Per-Branch Aggregation
# =============================================================================
def example_5_per_branch_aggregation():
    """
    Per-branch aggregation: Control how predictions are combined within branches.

    Use case: Average predictions within each branch before stacking.

    Aggregation strategies:
    - "separate": Keep each model's predictions as separate features (default)
    - "mean": Simple average of predictions
    - "weighted_mean": Weight by validation score (better models = higher weight)
    - "proba_mean": Average class probabilities (classification)

    Pipeline structure:
        branch → [PLS variants | RF variants] → merge(aggregate=mean) → Ridge
    """
    print("\n" + "=" * 70)
    print("Example 5: Per-Branch Aggregation")
    print("=" * 70)

    dataset = DatasetConfigs("sample_data/regression")

    pipeline = [
        MinMaxScaler(),
        KFold(n_splits=5, shuffle=True, random_state=42),

        # Multiple models per branch
        {"branch": [
            [
                SNV(),
                PLSRegression(n_components=5),
                PLSRegression(n_components=10),
                PLSRegression(n_components=15),
            ],
            [
                MSC(),
                RandomForestRegressor(n_estimators=30, random_state=42),
                RandomForestRegressor(n_estimators=50, random_state=42),
            ],
        ]},

        # Per-branch aggregation:
        # - Branch 0: Weighted mean of PLS predictions (1 feature)
        # - Branch 1: Mean of RF predictions (1 feature)
        # Total: 2 features for meta-learner
        {"merge": {
            "predictions": [
                {"branch": 0, "select": "all", "aggregate": "weighted_mean", "metric": "rmse"},
                {"branch": 1, "select": "all", "aggregate": "mean"},
            ]
        }},

        Ridge(alpha=1.0),
    ]

    runner = PipelineRunner(workspace_path="workspace/merge_branches", verbose=1)
    predictions, _ = runner.run(pipeline, dataset)

    print("\nResults:")
    for pred in predictions.top(5, "rmse"):
        print(f"  {Predictions.pred_short_string(pred, metrics=['rmse', 'r2'])}")

    print("\n✓ Per-branch aggregation complete - weighted PLS + mean RF")
    return predictions


# =============================================================================
# Example 6: Asymmetric Branches
# =============================================================================
def example_6_asymmetric_branches():
    """
    Asymmetric branches: Handle branches with different capabilities.

    Use case: Some branches have models, others only produce features.
    The merge controller detects this and suggests mixed merge.

    Asymmetric scenarios:
    - Models in some branches, not others
    - Different feature dimensions per branch
    - Different model counts per branch
    """
    print("\n" + "=" * 70)
    print("Example 6: Asymmetric Branches")
    print("=" * 70)

    dataset = DatasetConfigs("sample_data/regression")

    pipeline = [
        MinMaxScaler(),
        KFold(n_splits=5, shuffle=True, random_state=42),

        # Asymmetric branches:
        # - Branch 0: Model (produces predictions)
        # - Branch 1: Features only (no model)
        # - Branch 2: Model (produces predictions)
        {"branch": [
            [SNV(), PLSRegression(n_components=10)],
            [PCA(n_components=20)],  # No model!
            [FirstDerivative(), RandomForestRegressor(n_estimators=50, random_state=42)],
        ]},

        # Mixed merge handles asymmetry:
        # - Predictions from branches 0 and 2
        # - Features from branch 1
        {"merge": {
            "predictions": [0, 2],
            "features": [1],
        }},

        Ridge(alpha=1.0),
    ]

    runner = PipelineRunner(workspace_path="workspace/merge_branches", verbose=1)
    predictions, _ = runner.run(pipeline, dataset)

    print("\nResults:")
    for pred in predictions.top(5, "rmse"):
        print(f"  {Predictions.pred_short_string(pred, metrics=['rmse', 'r2'])}")

    print("\n✓ Asymmetric branches handled - predictions + features combined")
    return predictions


# =============================================================================
# Example 7: Merge vs MetaModel Equivalence
# =============================================================================
def example_7_merge_vs_metamodel():
    """
    Merge vs MetaModel: Show that merge + model is equivalent to MetaModel.

    MetaModel is a convenience wrapper that internally uses merge.
    Both approaches produce the same results.

    Equivalence:
        {"merge": "predictions"}, Ridge()  ≡  MetaModel(Ridge())
    """
    print("\n" + "=" * 70)
    print("Example 7: Merge vs MetaModel Equivalence")
    print("=" * 70)

    from nirs4all.operators.models import MetaModel

    dataset = DatasetConfigs("sample_data/regression")

    # Approach 1: Explicit merge + model
    pipeline_merge = [
        MinMaxScaler(),
        KFold(n_splits=5, shuffle=True, random_state=42),
        {"branch": [
            [SNV(), PLSRegression(n_components=10)],
            [MSC(), RandomForestRegressor(n_estimators=50, random_state=42)],
        ]},
        {"merge": "predictions"},
        {"name": "Ridge_Merge", "model": Ridge(alpha=1.0)},
    ]

    # Approach 2: MetaModel (convenience wrapper)
    pipeline_meta = [
        MinMaxScaler(),
        KFold(n_splits=5, shuffle=True, random_state=42),
        {"branch": [
            [SNV(), PLSRegression(n_components=10)],
            [MSC(), RandomForestRegressor(n_estimators=50, random_state=42)],
        ]},
        {"name": "Ridge_MetaModel", "model": MetaModel(model=Ridge(alpha=1.0))},
    ]

    runner = PipelineRunner(workspace_path="workspace/merge_branches", verbose=0)

    print("Running Approach 1: Merge + Model...")
    predictions1, _ = runner.run(pipeline_merge, dataset)

    print("Running Approach 2: MetaModel...")
    predictions2, _ = runner.run(pipeline_meta, dataset)

    # Compare results
    print("\n📊 Comparison:")

    merge_preds = [p for p in predictions1.to_dicts() if "Ridge" in p.get("model_name", "")]
    meta_preds = [p for p in predictions2.to_dicts() if "Ridge" in p.get("model_name", "")]

    if merge_preds and meta_preds:
        merge_score = min(p.get("val_score", float("inf")) for p in merge_preds)
        meta_score = min(p.get("val_score", float("inf")) for p in meta_preds)
        print(f"  Merge + Model: {merge_score:.4f}")
        print(f"  MetaModel:     {meta_score:.4f}")

        diff = abs(merge_score - meta_score)
        if diff < 0.001:
            print("\n✓ Results are equivalent (< 0.001 difference)")
        else:
            print(f"\n⚠ Small difference: {diff:.6f} (may be due to random state)")

    return predictions1


# =============================================================================
# Example 8: Output Targets
# =============================================================================
def example_8_output_targets():
    """
    Output targets: Control where merged output goes.

    Options:
    - "features" (default): Concatenate into feature matrix
    - "sources": Convert branches to sources for downstream processing
    - "dict": Keep as structured dict for multi-input models

    Use case: After merging, route output to different downstream handlers.
    """
    print("\n" + "=" * 70)
    print("Example 8: Output Targets")
    print("=" * 70)

    dataset = DatasetConfigs("sample_data/regression")

    # Output as features (default)
    pipeline = [
        MinMaxScaler(),
        KFold(n_splits=5, shuffle=True, random_state=42),

        {"branch": [
            [SNV()],
            [MSC()],
        ]},

        # Explicit output_as: "features" (this is the default)
        {"merge": {
            "features": "all",
            "output_as": "features",
        }},

        PLSRegression(n_components=10),
    ]

    runner = PipelineRunner(workspace_path="workspace/merge_branches", verbose=1)
    predictions, _ = runner.run(pipeline, dataset)

    print("\nResults (output_as='features'):")
    for pred in predictions.top(3, "rmse"):
        print(f"  {Predictions.pred_short_string(pred, metrics=['rmse', 'r2'])}")

    print("\n✓ Output target examples complete")
    print("Note: 'sources' and 'dict' targets enable advanced multi-input architectures")
    return predictions


# =============================================================================
# Run Examples
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Q_merge_branches.py - Branch Merging Examples")
    print("=" * 70 + "\n")

    examples = {
        1: ("Basic Feature Merge", example_1_basic_feature_merge),
        2: ("Prediction Merge (Stacking)", example_2_prediction_merge),
        3: ("Mixed Merge (Features + Predictions)", example_3_mixed_merge),
        4: ("Per-Branch Model Selection", example_4_per_branch_selection),
        5: ("Per-Branch Aggregation", example_5_per_branch_aggregation),
        6: ("Asymmetric Branches", example_6_asymmetric_branches),
        7: ("Merge vs MetaModel Equivalence", example_7_merge_vs_metamodel),
        8: ("Output Targets", example_8_output_targets),
    }

    if args.example == 0:
        # Run all examples
        for i, (name, func) in examples.items():
            try:
                func()
            except Exception as e:
                print(f"\n❌ Example {i} ({name}) failed: {e}")
    else:
        # Run specific example
        if args.example in examples:
            name, func = examples[args.example]
            func()
        else:
            print(f"Invalid example number: {args.example}")
            print(f"Available examples: {list(examples.keys())}")

    print("\n" + "=" * 70)
    print("Q_merge_branches.py Complete!")
    print("=" * 70)

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
