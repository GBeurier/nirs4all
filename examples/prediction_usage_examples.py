"""
NIRS4All Prediction Mode - Usage Examples and Documentation

This document demonstrates how to use the new prediction functionality in NIRS4All.

## Overview

The prediction feature allows you to:
1. Train a pipeline with binary saving enabled
2. Load the trained pipeline and make predictions on new data
3. Skip unnecessary operations (charts, data splitting) during prediction

## Basic Usage Examples
"""

# Example 1: Training with Binary Saving
def example_training_with_binaries():
    """Train a pipeline with binary saving for later prediction."""
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.pipeline.config import PipelineConfigs
    from nirs4all.dataset.dataset import SpectroDataset

    # Create runner with binary saving enabled (default)
    runner = PipelineRunner(
        results_path="./results/my_experiment",
        save_binaries=True,  # Enable binary saving for prediction support
        verbose=1
    )

    # Define your pipeline steps
    steps = [
        {"class": "sklearn.preprocessing.StandardScaler"},
        {
            "model": {
                "class": "sklearn.ensemble.RandomForestRegressor",
                "params": {"n_estimators": 100}
            },
            "train_params": {"verbose": 1}
        }
    ]

    # Create configuration and run training
    config = PipelineConfigs(steps)
    config.name = "my_trained_pipeline"

    dataset = SpectroDataset.load("./data/training_data")
    result_dataset, history, pipeline = runner.run(config, dataset)

    print("Training completed! Pipeline saved with binaries for prediction.")
    return result_dataset


# Example 2: Making Predictions
def example_prediction():
    """Make predictions using a trained pipeline."""
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset

    # Load new dataset for prediction
    new_dataset = SpectroDataset.load("./data/new_prediction_data")

    # Run prediction using saved pipeline
    result_dataset, final_context = PipelineRunner.predict(
        path="./results/my_experiment/training_data/my_trained_pipeline",
        dataset=new_dataset,
        verbose=1
    )

    print("Prediction completed!")
    print(f"Final context: {final_context}")
    return result_dataset


# Example 3: Complex Pipeline with Charts (skipped in prediction)
def example_complex_pipeline():
    """Complex pipeline demonstrating what gets skipped in prediction mode."""
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.pipeline.config import PipelineConfigs

    steps = [
        # Data preprocessing - EXECUTED in prediction mode
        {"class": "sklearn.preprocessing.StandardScaler"},
        {"class": "sklearn.decomposition.PCA", "params": {"n_components": 10}},

        # Chart operations - SKIPPED in prediction mode
        "chart_2d",
        "chart_3d",

        # Model training/prediction - EXECUTED (prediction mode) in prediction
        {
            "model": {
                "class": "sklearn.ensemble.RandomForestRegressor",
                "params": {"n_estimators": 200, "random_state": 42}
            },
            "train_params": {"verbose": 1}
        },

        # More charts - SKIPPED in prediction mode
        "y_chart"
    ]

    # Training phase
    training_runner = PipelineRunner(
        results_path="./results/complex_pipeline",
        save_binaries=True,
        verbose=2
    )

    return steps


# Example 4: Error Handling
def example_error_handling():
    """Demonstrate error handling for prediction mode."""
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset

    dataset = SpectroDataset.load("./data/test")

    try:
        # Try to use a pipeline without binary metadata
        result = PipelineRunner.predict("./old_pipeline_path", dataset)
    except FileNotFoundError:
        print("Pipeline directory not found")
    except ValueError as e:
        print(f"Configuration error: {e}")
    except RuntimeError as e:
        print(f"Prediction failed: {e}")


# Command Line Examples (for documentation)
def command_line_examples():
    """
    Command line usage examples:

    # Training with binary saving:
    nirs4all train --preset regression_model --data-path ./training_data

    # Making predictions:
    nirs4all predict \
        --pipeline-path ./results/training_data/my_pipeline \
        --data-path ./new_data \
        --output-path ./predictions \
        --verbose
    """
    pass


if __name__ == "__main__":
    print("NIRS4All Prediction Mode - Usage Examples")
    print("=" * 50)
    print("This file contains usage examples for the new prediction functionality.")
    print("See the function definitions above for detailed examples.")
    print("\nKey features:")
    print("- Train with save_binaries=True for prediction support")
    print("- Use PipelineRunner.predict() for inference")
    print("- Chart operations are automatically skipped in prediction mode")
    print("- Transformers use fitted parameters, models make predictions")