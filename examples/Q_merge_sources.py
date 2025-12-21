"""
Q_merge_sources.py - Multi-Source Dataset Merging and Source Branching Examples

This example demonstrates source-level operations for multi-source datasets:
1. Source merging (merge_sources) - Combine features from different data sources
2. Source branching (source_branch) - Per-source pipeline execution
3. Source-specific processing - Different preprocessing per source

Key concepts:
- Sources represent DATA PROVENANCE (different sensors, modalities, instruments)
- Branches represent EXECUTION PATHS (alternative processing strategies)
- Sources and branches are orthogonal dimensions
- source_branch creates per-source pipelines
- merge_sources combines sources after processing

Multi-source use cases:
- NIR + Raman spectroscopy fusion
- Lab + portable instrument transfer
- Spectral + metadata combination
- Multi-modal sensor fusion

Examples included:
1. Basic source merge - Concatenate all source features
2. Source branch - Per-source preprocessing pipelines
3. Source branch + auto-merge - Process and combine automatically
4. Selective source merge - Choose specific sources
5. Source merge strategies - concat, stack, dict outputs
6. Combined source + pipeline branching - Full complexity example
7. Source-aware model training - Different models per source

Phase 9-10 Features:
- merge_sources keyword for multi-source datasets
- source_branch keyword for per-source pipelines
- Auto-merge option for convenience
- Prediction mode support

See also:
- Q6_multisource.py - Basic multi-source examples
- Q_merge_branches.py - Pipeline branching and merging
- Q30_branching.py - Basic branching examples
"""

import argparse
import os
import sys

# Add parent directory to path for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.transforms import (
    StandardNormalVariate as SNV,
    MultiplicativeScatterCorrection as MSC,
    FirstDerivative,
    SavitzkyGolay,
)
from nirs4all.visualization.predictions import PredictionAnalyzer

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Q_merge_sources Example")
parser.add_argument("--plots", action="store_true", help="Show plots interactively")
parser.add_argument("--show", action="store_true", help="Show all plots")
parser.add_argument("--example", type=int, default=0, help="Run specific example (1-7, 0=all)")
args = parser.parse_args()


def create_multi_source_dataset_config():
    """Create a simulated multi-source dataset configuration.

    In production, this would point to actual multi-source data files.
    For this example, we simulate by using the same data as different "sources".

    Returns:
        DatasetConfigs with multiple sources.
    """
    # For this example, we'll use the regression sample data
    # In production, you would have different source files:
    # - "nir_sensor.csv" (NIR spectral data)
    # - "markers.csv" (chemical markers)
    # - "raman.csv" (Raman spectral data)

    # Single-source fallback for demonstration
    # The examples will adapt to single-source if multi-source not available
    return DatasetConfigs("sample_data/regression")


# =============================================================================
# Example 1: Basic Source Merge
# =============================================================================
def example_1_basic_source_merge():
    """
    Basic source merge: Concatenate features from all data sources.

    Use case: You have NIR spectral data and chemical markers.
    Combine them into a single feature matrix for modeling.

    Note: This example uses single-source data for demonstration.
    With multi-source data, merge_sources would concatenate all sources.
    """
    print("=" * 70)
    print("Example 1: Basic Source Merge")
    print("=" * 70)

    dataset = create_multi_source_dataset_config()

    pipeline = [
        MinMaxScaler(),
        KFold(n_splits=3, shuffle=True, random_state=42),

        # Apply preprocessing to all sources
        SNV(),

        # Merge all sources into single feature matrix
        # For single-source, this is a no-op
        # For multi-source: [source1_features | source2_features | ...]
        {"merge_sources": "concat"},

        PLSRegression(n_components=10),
    ]

    runner = PipelineRunner(workspace_path="workspace/merge_sources", verbose=1)
    predictions, _ = runner.run(pipeline, dataset)

    print("\nResults:")
    for pred in predictions.top(5, "rmse"):
        print(f"  {Predictions.pred_short_string(pred, metrics=['rmse', 'r2'])}")

    print("\n✓ Source merge complete")
    print("  For multi-source datasets, this concatenates all source features")
    return predictions


# =============================================================================
# Example 2: Source Branch (Per-Source Pipelines)
# =============================================================================
def example_2_source_branch():
    """
    Source branching: Different preprocessing pipelines per source.

    Use case: NIR data needs SNV + derivatives, markers need variance filtering.
    Each source gets its own tailored preprocessing chain.

    Syntax:
        {"source_branch": {
            "NIR": [SNV(), FirstDerivative()],
            "markers": [VarianceThreshold(), MinMaxScaler()],
        }}

    Note: This example simulates with single source.
    """
    print("\n" + "=" * 70)
    print("Example 2: Source Branching (Per-Source Pipelines)")
    print("=" * 70)

    dataset = create_multi_source_dataset_config()

    # For single-source, we use "auto" mode which just passes through
    # For multi-source, each source would get its specific pipeline
    pipeline = [
        KFold(n_splits=3, shuffle=True, random_state=42),

        # Source-specific preprocessing
        # "auto" mode: Each source processes independently
        {"source_branch": "auto"},

        PLSRegression(n_components=10),
    ]

    runner = PipelineRunner(workspace_path="workspace/merge_sources", verbose=1)
    predictions, _ = runner.run(pipeline, dataset)

    print("\nResults:")
    for pred in predictions.top(5, "rmse"):
        print(f"  {Predictions.pred_short_string(pred, metrics=['rmse', 'r2'])}")

    print("\n✓ Source branch complete")
    print('  With multi-source data, use: {"source_branch": {"NIR": [steps], "markers": [steps]}}')
    return predictions


# =============================================================================
# Example 3: Source Branch with Auto-Merge
# =============================================================================
def example_3_source_branch_auto_merge():
    """
    Source branch with automatic merging after processing.

    Use case: Process each source differently, then automatically combine.
    The _merge_after_ option (default: True) handles this.

    Equivalent to:
        source_branch → merge_sources("concat")
    """
    print("\n" + "=" * 70)
    print("Example 3: Source Branch with Auto-Merge")
    print("=" * 70)

    dataset = create_multi_source_dataset_config()

    pipeline = [
        KFold(n_splits=3, shuffle=True, random_state=42),

        # Source branch with explicit auto-merge
        {"source_branch": {
            "_merge_after_": True,       # Auto-merge after processing (default)
            "_merge_strategy_": "concat",  # Concatenation strategy
            # For multi-source, add source-specific pipelines:
            # "NIR": [SNV(), FirstDerivative()],
            # "markers": [VarianceThreshold()],
        }},

        PLSRegression(n_components=10),
    ]

    runner = PipelineRunner(workspace_path="workspace/merge_sources", verbose=1)
    predictions, _ = runner.run(pipeline, dataset)

    print("\nResults:")
    for pred in predictions.top(5, "rmse"):
        print(f"  {Predictions.pred_short_string(pred, metrics=['rmse', 'r2'])}")

    print("\n✓ Source branch + auto-merge complete")
    return predictions


# =============================================================================
# Example 4: Source Merge Strategies
# =============================================================================
def example_4_source_merge_strategies():
    """
    Source merge strategies: Different ways to combine source features.

    Strategies:
    - "concat" (default): Horizontal concatenation [src1 | src2 | ...]
    - "stack": 3D stacking (requires compatible shapes)
    - "dict": Keep as dictionary for multi-input models

    Use case: Choose the right format for your downstream model.
    """
    print("\n" + "=" * 70)
    print("Example 4: Source Merge Strategies")
    print("=" * 70)

    dataset = create_multi_source_dataset_config()

    # Strategy 1: Concat (default)
    print("\n--- Strategy: concat ---")
    pipeline_concat = [
        MinMaxScaler(),
        KFold(n_splits=3, shuffle=True, random_state=42),
        {"merge_sources": "concat"},
        PLSRegression(n_components=10),
    ]

    runner = PipelineRunner(workspace_path="workspace/merge_sources", verbose=0)
    predictions, _ = runner.run(pipeline_concat, dataset)

    print("Concat results:")
    for pred in predictions.top(2, "rmse"):
        print(f"  {Predictions.pred_short_string(pred, metrics=['rmse'])}")

    # Strategy 2: Dict (for advanced multi-input models)
    print("\n--- Strategy: dict ---")
    print("Dict strategy keeps sources separate as a dictionary.")
    print("Useful for multi-head neural networks with source-specific inputs.")
    print('Syntax: {"merge_sources": {"strategy": "dict"}}')

    # Strategy 3: Stack (requires compatible shapes)
    print("\n--- Strategy: stack ---")
    print("Stack strategy creates 3D array (samples, sources, features).")
    print("Requires all sources to have same feature dimension.")
    print("Useful for CNNs and attention-based models.")
    print('Syntax: {"merge_sources": {"strategy": "stack"}}')

    print("\n✓ Source merge strategies demonstrated")
    return predictions


# =============================================================================
# Example 5: Combined Source + Pipeline Branching
# =============================================================================
def example_5_combined_branching():
    """
    Combined source and pipeline branching: Full complexity example.

    Use case: Multi-source data with multiple preprocessing strategies per source,
    followed by stacking ensemble.

    Pipeline structure:
        source_branch → merge_sources → pipeline_branch → merge(predictions) → Ridge

    This demonstrates the orthogonality of sources and branches:
    - Sources: Different data origins (NIR, markers)
    - Branches: Different processing strategies (SNV vs MSC)
    """
    print("\n" + "=" * 70)
    print("Example 5: Combined Source + Pipeline Branching")
    print("=" * 70)

    dataset = create_multi_source_dataset_config()

    pipeline = [
        KFold(n_splits=3, shuffle=True, random_state=42),

        # Step 1: Source-level processing (each source independently)
        {"source_branch": "auto"},

        # Step 2: Scale features
        MinMaxScaler(),

        # Step 3: Pipeline branching (compare preprocessing strategies)
        {"branch": [
            [SNV(), PLSRegression(n_components=10)],
            [FirstDerivative(), PLSRegression(n_components=10)],
        ]},

        # Step 4: Merge predictions from branches
        {"merge": "predictions"},

        # Step 5: Meta-learner
        Ridge(alpha=1.0),
    ]

    runner = PipelineRunner(workspace_path="workspace/merge_sources", verbose=1)
    predictions, _ = runner.run(pipeline, dataset)

    print("\nResults:")
    for pred in predictions.top(5, "rmse"):
        print(f"  {Predictions.pred_short_string(pred, metrics=['rmse', 'r2'])}")

    print("\n✓ Combined source + pipeline branching complete")
    print("  This pattern enables multi-modal + multi-strategy experiments")
    return predictions


# =============================================================================
# Example 6: Source-Aware Stacking
# =============================================================================
def example_6_source_aware_stacking():
    """
    Source-aware stacking: Different models trained per source, then combined.

    Use case: NIR data works best with PLS, markers work best with RF.
    Train optimal models per source, then stack their predictions.

    This pattern:
    1. Process each source with its optimal preprocessing
    2. Train source-specific models
    3. Combine predictions via meta-learning
    """
    print("\n" + "=" * 70)
    print("Example 6: Source-Aware Stacking")
    print("=" * 70)

    dataset = create_multi_source_dataset_config()

    # For single-source demonstration, we use pipeline branching
    # to simulate source-specific models
    pipeline = [
        MinMaxScaler(),
        KFold(n_splits=3, shuffle=True, random_state=42),

        # Simulate source-specific processing with branches
        {"branch": [
            # "NIR specialist": SNV + PLS
            [SNV(), {"name": "NIR_PLS", "model": PLSRegression(n_components=10)}],
            # "Marker specialist": Feature selection + RF
            [PCA(n_components=20), {"name": "Marker_RF", "model": RandomForestRegressor(n_estimators=50, random_state=42)}],
        ]},

        # Stack source-specialist predictions
        {"merge": "predictions"},

        # Meta-learner combines specialists
        {"name": "Meta_Ridge", "model": Ridge(alpha=1.0)},
    ]

    runner = PipelineRunner(workspace_path="workspace/merge_sources", verbose=1)
    predictions, _ = runner.run(pipeline, dataset)

    print("\nResults:")
    for pred in predictions.top(5, "rmse"):
        name = pred.get("model_name", "unknown")
        score = pred.get("val_score", 0)
        print(f"  {name:40s}: {score:.4f}")

    # Compare meta-learner with specialists
    print("\n📊 Specialist vs Meta-learner:")
    meta_preds = [p for p in predictions.to_dicts() if "Meta" in p.get("model_name", "")]
    specialist_preds = [p for p in predictions.to_dicts() if "Meta" not in p.get("model_name", "") and "Ridge" not in p.get("model_name", "")]

    if meta_preds and specialist_preds:
        meta_score = min(p.get("val_score", float("inf")) for p in meta_preds)
        best_specialist = min(p.get("val_score", float("inf")) for p in specialist_preds)
        print(f"  Best specialist: {best_specialist:.4f}")
        print(f"  Meta-learner:    {meta_score:.4f}")

    print("\n✓ Source-aware stacking complete")
    return predictions


# =============================================================================
# Example 7: Shape Mismatch Handling
# =============================================================================
def example_7_shape_mismatch():
    """
    Shape mismatch handling: Deal with sources of different dimensions.

    Use case: NIR has 500 wavelengths, markers have 50 features.
    The merge controller handles this with on_incompatible options.

    Options:
    - "error": Raise error if shapes differ (for stack strategy)
    - "flatten": Force 2D concatenation (always works)
    - "pad": Zero-pad shorter sources
    - "truncate": Truncate longer sources
    """
    print("\n" + "=" * 70)
    print("Example 7: Shape Mismatch Handling")
    print("=" * 70)

    dataset = create_multi_source_dataset_config()

    pipeline = [
        MinMaxScaler(),
        KFold(n_splits=3, shuffle=True, random_state=42),

        # For multi-source with different dimensions:
        # {"merge_sources": {
        #     "strategy": "concat",          # 2D concat works with any dimensions
        #     "on_incompatible": "flatten",  # Force 2D if needed
        # }},

        # Single-source demonstration
        {"merge_sources": "concat"},

        PLSRegression(n_components=10),
    ]

    runner = PipelineRunner(workspace_path="workspace/merge_sources", verbose=1)
    predictions, _ = runner.run(pipeline, dataset)

    print("\nResults:")
    for pred in predictions.top(3, "rmse"):
        print(f"  {Predictions.pred_short_string(pred, metrics=['rmse', 'r2'])}")

    print("\n✓ Shape mismatch handling demonstrated")
    print("  Use 'concat' strategy for sources with different feature counts")
    print("  Use 'stack' + 'pad'/'truncate' when 3D tensor needed")
    return predictions


# =============================================================================
# Run Examples
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Q_merge_sources.py - Multi-Source Merging Examples")
    print("=" * 70 + "\n")

    print("NOTE: These examples use single-source data for demonstration.")
    print("With multi-source datasets, source_branch and merge_sources")
    print("enable powerful source-specific processing and fusion.\n")

    examples = {
        1: ("Basic Source Merge", example_1_basic_source_merge),
        2: ("Source Branch (Per-Source Pipelines)", example_2_source_branch),
        3: ("Source Branch with Auto-Merge", example_3_source_branch_auto_merge),
        4: ("Source Merge Strategies", example_4_source_merge_strategies),
        5: ("Combined Source + Pipeline Branching", example_5_combined_branching),
        6: ("Source-Aware Stacking", example_6_source_aware_stacking),
        7: ("Shape Mismatch Handling", example_7_shape_mismatch),
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
    print("Q_merge_sources.py Complete!")
    print("=" * 70)

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
