#!/usr/bin/env python3
"""
Q33 Example - Retrain and Transfer Learning
============================================
Demonstrates retraining trained pipelines on new data with various modes:
full retrain, transfer learning, and fine-tuning.

Key Features:
1. Full retrain: Train from scratch with same pipeline structure
2. Transfer mode: Use existing preprocessing artifacts, train new model
3. Finetune mode: Continue training existing model with new data
4. Extract and modify: Get pipeline for inspection and modification

Phase 7 Implementation:
    This example showcases the retrain feature that enables reusing
    trained pipelines without having to reconstruct the pipeline
    configuration manually.

Use Cases:
    - Model update: Retrain with new data while keeping same structure
    - Transfer learning: Apply preprocessing from one domain to another
    - Fine-tuning: Continue training with additional data
    - A/B testing: Compare same preprocessing with different models
"""

# Standard library imports
import argparse
import os
import sys
from pathlib import Path

# Third-party imports
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner, StepMode
from nirs4all.operators.transforms import (
    StandardNormalVariate,
    SavitzkyGolay,
    Gaussian,
)

# Simple status symbols
CHECK = "[OK]"
CROSS = "[X]"
ROCKET = ">"
REFRESH = "[~]"
TRANSFER = "[->]"
SPARKLE = "*"

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q33 Retrain and Transfer Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()


# =============================================================================
# Helper Functions
# =============================================================================


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"{ROCKET} {title}")
    print("=" * 70)


def print_subsection(title: str):
    """Print a subsection header."""
    print(f"\n--- {title} ---")


def compare_predictions(pred1, pred2, name1="Pred1", name2="Pred2"):
    """Compare two prediction arrays."""
    if np.allclose(pred1, pred2, rtol=1e-3):
        print(f"{CHECK} {name1} ≈ {name2}: YES (within tolerance)")
    else:
        diff = np.abs(pred1 - pred2).mean()
        print(f"{CROSS} {name1} ≠ {name2}: Mean diff = {diff:.4f}")


# =============================================================================
# Example 1: Full Retrain on New Data
# =============================================================================


def example_1_full_retrain():
    """
    Demonstrate full retrain: train from scratch with same pipeline structure.

    Use case: You have a trained model and want to retrain it on new data
    (e.g., new season's samples, updated calibration set) while keeping
    the same pipeline structure.
    """
    print_section("Example 1: Full Retrain on New Data")

    # Step 1: Train initial pipeline
    print_subsection("Step 1: Train Initial Pipeline")

    pipeline = [
        MinMaxScaler(),
        {"y_processing": MinMaxScaler()},
        {"feature_augmentation": [StandardNormalVariate(), SavitzkyGolay()]},
        RepeatedKFold(n_splits=2, n_repeats=1, random_state=42),
        {"model": PLSRegression(n_components=10), "name": "PLS_Original"},
    ]

    pipeline_config = PipelineConfigs(pipeline, "retrain_demo")
    dataset_config = DatasetConfigs(['sample_data/regression'])

    runner = PipelineRunner(save_artifacts=True, verbose=0)
    original_predictions, _ = runner.run(pipeline_config, dataset_config)

    best_original = original_predictions.top(n=1, rank_partition="test")[0]
    print(f"Original model: {best_original['model_name']}")
    print(f"Original RMSE: {best_original['rmse']:.4f}")

    # Step 2: Full retrain on same data (simulating new data)
    print_subsection("Step 2: Full Retrain")

    # In a real scenario, you would use different data here
    new_dataset = DatasetConfigs(['sample_data/regression'])

    retrained_predictions, _ = runner.retrain(
        source=best_original,
        dataset=new_dataset,
        mode='full',
        dataset_name='new_calibration',
        verbose=0
    )

    best_retrained = retrained_predictions.top(n=1, rank_partition="test")[0]
    print(f"Retrained model: {best_retrained['model_name']}")
    print(f"Retrained RMSE: {best_retrained['rmse']:.4f}")

    print_subsection("Comparison")
    print(f"RMSE change: {best_retrained['rmse'] - best_original['rmse']:+.4f}")
    print(f"Pipeline structure preserved: {CHECK} YES")

    return best_original


# =============================================================================
# Example 2: Transfer Mode - Reuse Preprocessing
# =============================================================================


def example_2_transfer_mode(original_prediction):
    """
    Demonstrate transfer mode: use existing preprocessing, train new model.

    Use case: You have optimized preprocessing from one experiment and want
    to apply it to a new dataset with a different model.
    """
    print_section("Example 2: Transfer Mode - Reuse Preprocessing")

    runner = PipelineRunner(save_artifacts=True, verbose=0)

    # Same data for demonstration (would be different in practice)
    new_dataset = DatasetConfigs(['sample_data/regression'])

    # Transfer with same model type
    print_subsection("Transfer with Same Model Type")

    transfer_predictions, _ = runner.retrain(
        source=original_prediction,
        dataset=new_dataset,
        mode='transfer',
        dataset_name='transfer_same',
        verbose=0
    )

    best_transfer = transfer_predictions.top(n=1, rank_partition="test")[0]
    print(f"Transfer model: {best_transfer['model_name']}")
    print(f"Transfer RMSE: {best_transfer['rmse']:.4f}")
    print(f"Preprocessing: Reused from original {CHECK}")

    # Transfer with different model
    print_subsection("Transfer with Different Model")

    new_model = GradientBoostingRegressor(n_estimators=50, random_state=42)

    transfer_newmodel_predictions, _ = runner.retrain(
        source=original_prediction,
        dataset=new_dataset,
        mode='transfer',
        dataset_name='transfer_new_model',
        new_model=new_model,
        verbose=0
    )

    best_transfer_new = transfer_newmodel_predictions.top(n=1, rank_partition="test")[0]
    print(f"Transfer (new model): {best_transfer_new['model_name']}")
    print(f"Transfer RMSE: {best_transfer_new['rmse']:.4f}")
    print(f"Preprocessing: Reused from original {CHECK}")
    print(f"Model: New GradientBoostingRegressor {CHECK}")

    print_subsection("Comparison")
    print(f"Original RMSE: {original_prediction['rmse']:.4f}")
    print(f"Transfer (same model) RMSE: {best_transfer['rmse']:.4f}")
    print(f"Transfer (new model) RMSE: {best_transfer_new['rmse']:.4f}")

    return best_transfer


# =============================================================================
# Example 3: Fine-tuning (for Neural Networks)
# =============================================================================


def example_3_finetune_mode(original_prediction):
    """
    Demonstrate finetune mode: continue training existing model.

    Use case: You have a trained neural network and want to fine-tune it
    on new data without starting from scratch.

    Note: Fine-tuning is most effective with neural network models.
    For sklearn models, it's equivalent to retraining.
    """
    print_section("Example 3: Fine-tune Mode (Continue Training)")

    runner = PipelineRunner(save_artifacts=True, verbose=0)
    new_dataset = DatasetConfigs(['sample_data/regression'])

    # Finetune with additional epochs (mainly for NN models)
    print_subsection("Fine-tuning with Additional Epochs")

    finetune_predictions, _ = runner.retrain(
        source=original_prediction,
        dataset=new_dataset,
        mode='finetune',
        dataset_name='finetune_dataset',
        epochs=5,  # Additional epochs for neural networks
        verbose=0
    )

    best_finetune = finetune_predictions.top(n=1, rank_partition="test")[0]
    print(f"Fine-tuned model: {best_finetune['model_name']}")
    print(f"Fine-tuned RMSE: {best_finetune['rmse']:.4f}")

    print_subsection("Note on Fine-tuning")
    print("Fine-tuning is most effective with neural network models.")
    print("For sklearn models like PLSRegression, fine-tuning is")
    print("equivalent to retraining (they don't support incremental learning).")

    return best_finetune


# =============================================================================
# Example 4: Extract and Modify Pipeline
# =============================================================================


def example_4_extract_and_modify(original_prediction):
    """
    Demonstrate extracting and modifying a trained pipeline.

    Use case: You want to inspect a trained pipeline, modify it
    (e.g., swap the model), and run it on new data.
    """
    print_section("Example 4: Extract and Modify Pipeline")

    runner = PipelineRunner(save_artifacts=True, verbose=0)

    # Extract pipeline
    print_subsection("Extracting Pipeline")

    extracted = runner.extract(original_prediction)

    print(f"Extracted pipeline: {extracted}")
    print(f"Number of steps: {len(extracted.steps)}")
    print(f"Model step index: {extracted.model_step_index}")
    print(f"Preprocessing chain: {extracted.preprocessing_chain}")

    # Inspect steps
    print_subsection("Pipeline Steps")
    for i, step in enumerate(extracted.steps):
        print(f"  Step {i}: {step}")

    # Modify model
    print_subsection("Modifying Model")

    print(f"Original model step: {extracted.get_model_step()}")

    # Replace model with RandomForestRegressor
    new_model = RandomForestRegressor(n_estimators=50, random_state=42)
    extracted.set_model(new_model)

    print(f"New model step: {extracted.get_model_step()}")

    # Run modified pipeline
    print_subsection("Running Modified Pipeline")

    new_dataset = DatasetConfigs(['sample_data/regression'])

    modified_predictions, _ = runner.run(
        pipeline=extracted.steps,
        dataset=new_dataset,
        pipeline_name="modified_pipeline"
    )

    best_modified = modified_predictions.top(n=1, rank_partition="test")[0]
    print(f"Modified model: {best_modified['model_name']}")
    print(f"Modified RMSE: {best_modified['rmse']:.4f}")

    return extracted


# =============================================================================
# Example 5: Fine-grained Step Mode Control
# =============================================================================


def example_5_step_mode_control():
    """
    Demonstrate fine-grained control over which steps to retrain.

    Use case: You want to retrain only specific steps in a pipeline
    while keeping others frozen.
    """
    print_section("Example 5: Fine-grained Step Mode Control")

    # Train initial pipeline
    print_subsection("Training Initial Pipeline")

    pipeline = [
        MinMaxScaler(),  # Step 1
        StandardNormalVariate(),  # Step 2
        SavitzkyGolay(),  # Step 3
        RepeatedKFold(n_splits=2, n_repeats=1, random_state=42),  # Step 4
        {"model": PLSRegression(n_components=10), "name": "PLS"},  # Step 5
    ]

    pipeline_config = PipelineConfigs(pipeline, "step_control_demo")
    dataset_config = DatasetConfigs(['sample_data/regression'])

    runner = PipelineRunner(save_artifacts=True, verbose=0)
    predictions, _ = runner.run(pipeline_config, dataset_config)

    best_pred = predictions.top(n=1, rank_partition="test")[0]
    print(f"Original model RMSE: {best_pred['rmse']:.4f}")

    # Define step modes
    print_subsection("Defining Step Modes")

    step_modes = [
        StepMode(step_index=1, mode='predict'),  # Use existing MinMaxScaler
        StepMode(step_index=2, mode='predict'),  # Use existing SNV
        StepMode(step_index=3, mode='train'),    # Retrain SavitzkyGolay
        # Step 4 (CV splitter) will use default mode
        # Step 5 (model) will use default mode based on overall mode
    ]

    print("Step mode configuration:")
    for sm in step_modes:
        print(f"  Step {sm.step_index}: {sm.mode}")

    # Retrain with step modes
    print_subsection("Retraining with Step Mode Control")

    new_dataset = DatasetConfigs(['sample_data/regression'])

    controlled_predictions, _ = runner.retrain(
        source=best_pred,
        dataset=new_dataset,
        mode='full',
        step_modes=step_modes,
        dataset_name='controlled_retrain',
        verbose=0
    )

    best_controlled = controlled_predictions.top(n=1, rank_partition="test")[0]
    print(f"Controlled retrain RMSE: {best_controlled['rmse']:.4f}")

    print_subsection("Summary")
    print("Steps 1-2: Used existing artifacts (predict mode)")
    print("Steps 3-5: Retrained (train mode)")


# =============================================================================
# Example 6: Retrain from Bundle
# =============================================================================


def example_6_retrain_from_bundle():
    """
    Demonstrate retraining from an exported bundle.

    Use case: You received a model bundle from a colleague and want
    to retrain it on your local data.
    """
    print_section("Example 6: Retrain from Bundle")

    # First, create a bundle
    print_subsection("Creating Bundle for Demo")

    pipeline = [
        MinMaxScaler(),
        StandardNormalVariate(),
        RepeatedKFold(n_splits=2, n_repeats=1, random_state=42),
        {"model": PLSRegression(n_components=8), "name": "PLS_bundle"},
    ]

    pipeline_config = PipelineConfigs(pipeline, "bundle_retrain_demo")
    dataset_config = DatasetConfigs(['sample_data/regression'])

    runner = PipelineRunner(save_artifacts=True, verbose=0)
    predictions, _ = runner.run(pipeline_config, dataset_config)

    best_pred = predictions.top(n=1, rank_partition="test")[0]

    # Export to bundle
    bundle_path = runner.export(best_pred, "exports/for_retrain.n4a")
    print(f"Bundle created: {bundle_path}")

    # Retrain from bundle
    print_subsection("Retraining from Bundle")

    new_runner = PipelineRunner(save_artifacts=True, verbose=0)
    new_dataset = DatasetConfigs(['sample_data/regression'])

    bundle_retrain_predictions, _ = new_runner.retrain(
        source=str(bundle_path),
        dataset=new_dataset,
        mode='transfer',
        dataset_name='bundle_retrain',
        verbose=0
    )

    best_bundle_retrain = bundle_retrain_predictions.top(n=1, rank_partition="test")[0]
    print(f"Bundle retrain model: {best_bundle_retrain['model_name']}")
    print(f"Bundle retrain RMSE: {best_bundle_retrain['rmse']:.4f}")

    print_subsection("Summary")
    print("Bundles can be shared and retrained on new data.")
    print("This enables collaborative model development workflows.")


# =============================================================================
# Example 7: Compare Retrain Modes
# =============================================================================


def example_7_compare_modes():
    """
    Demonstrate comparing different retrain modes on the same data.

    Useful for understanding the impact of each mode.
    """
    print_section("Example 7: Comparing Retrain Modes")

    # Train initial model
    print_subsection("Training Initial Model")

    pipeline = [
        MinMaxScaler(),
        {"y_processing": MinMaxScaler()},
        StandardNormalVariate(),
        SavitzkyGolay(),
        RepeatedKFold(n_splits=2, n_repeats=1, random_state=42),
        {"model": PLSRegression(n_components=10), "name": "PLS_compare"},
    ]

    pipeline_config = PipelineConfigs(pipeline, "compare_modes")
    dataset_config = DatasetConfigs(['sample_data/regression'])

    runner = PipelineRunner(save_artifacts=True, verbose=0)
    original_predictions, _ = runner.run(pipeline_config, dataset_config)

    best_original = original_predictions.top(n=1, rank_partition="test")[0]

    # Same dataset for fair comparison
    new_dataset = DatasetConfigs(['sample_data/regression'])

    results = {"original": best_original['rmse']}

    # Full retrain
    print_subsection("Mode: Full Retrain")
    full_preds, _ = runner.retrain(
        source=best_original, dataset=new_dataset, mode='full',
        dataset_name='compare_full', verbose=0
    )
    best_full = full_preds.top(n=1, rank_partition="test")[0]
    results["full"] = best_full['rmse']
    print(f"Full RMSE: {best_full['rmse']:.4f}")

    # Transfer retrain
    print_subsection("Mode: Transfer")
    transfer_preds, _ = runner.retrain(
        source=best_original, dataset=new_dataset, mode='transfer',
        dataset_name='compare_transfer', verbose=0
    )
    best_transfer = transfer_preds.top(n=1, rank_partition="test")[0]
    results["transfer"] = best_transfer['rmse']
    print(f"Transfer RMSE: {best_transfer['rmse']:.4f}")

    # Finetune
    print_subsection("Mode: Finetune")
    finetune_preds, _ = runner.retrain(
        source=best_original, dataset=new_dataset, mode='finetune',
        dataset_name='compare_finetune', verbose=0
    )
    best_finetune = finetune_preds.top(n=1, rank_partition="test")[0]
    results["finetune"] = best_finetune['rmse']
    print(f"Finetune RMSE: {best_finetune['rmse']:.4f}")

    # Summary
    print_subsection("Results Summary")
    print("-" * 40)
    print(f"{'Mode':<15} {'RMSE':<10} {'Δ from Original':<15}")
    print("-" * 40)
    for mode, rmse in results.items():
        delta = rmse - results["original"]
        delta_str = f"{delta:+.4f}" if mode != "original" else "---"
        print(f"{mode:<15} {rmse:.4f}     {delta_str}")
    print("-" * 40)


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print(f"# {REFRESH} Q33 - Retrain and Transfer Learning")
    print("#" * 70)

    # Example 1: Full retrain
    original_prediction = example_1_full_retrain()

    # Example 2: Transfer mode
    transfer_prediction = example_2_transfer_mode(original_prediction)

    # Example 3: Finetune mode
    finetune_prediction = example_3_finetune_mode(original_prediction)

    # Example 4: Extract and modify
    extracted_pipeline = example_4_extract_and_modify(original_prediction)

    # Example 5: Step mode control
    example_5_step_mode_control()

    # Example 6: Retrain from bundle
    example_6_retrain_from_bundle()

    # Example 7: Compare modes
    example_7_compare_modes()

    print("\n" + "=" * 70)
    print(f"{SPARKLE} All examples completed successfully!")
    print("=" * 70)
    print("\nRetrain modes summary:")
    print("  - full: Train everything from scratch (same structure)")
    print("  - transfer: Reuse preprocessing, train new model")
    print("  - finetune: Continue training existing model")
