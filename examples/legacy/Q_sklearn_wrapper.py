"""
Q_sklearn_wrapper Example - sklearn-Compatible Pipeline Wrapper
================================================================
Demonstrates how to use NIRSPipeline for sklearn compatibility and SHAP integration.

This example shows:
1. Training a pipeline with nirs4all.run()
2. Wrapping the result with NIRSPipeline.from_result()
3. Using the wrapper with sklearn tools (score, predict)
4. Accessing the underlying model for SHAP analysis
5. Loading from an exported bundle with NIRSPipeline.from_bundle()

Phase 4 Implementation - sklearn Integration
"""

# Standard library imports
import argparse
import numpy as np

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

# NIRS4All imports
import nirs4all
from nirs4all.sklearn import NIRSPipeline, NIRSPipelineClassifier
from nirs4all.operators.transforms import StandardNormalVariate, SavitzkyGolay

# Parse command-line arguments
parser = argparse.ArgumentParser(description='sklearn Wrapper Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()


def example_1_basic_wrapper():
    """Example 1: Basic NIRSPipeline wrapper from RunResult.

    Demonstrates the primary use case: train with nirs4all.run(),
    wrap with NIRSPipeline.from_result() for sklearn compatibility.
    """
    print("\n" + "="*60)
    print("Example 1: Basic NIRSPipeline Wrapper from RunResult")
    print("="*60)

    # Define a simple pipeline
    pipeline = [
        MinMaxScaler(),
        StandardNormalVariate(),
        ShuffleSplit(n_splits=3, test_size=0.25),
        {"model": PLSRegression(n_components=10)}
    ]

    # Train with nirs4all
    result = nirs4all.run(
        pipeline=pipeline,
        dataset="sample_data/regression",
        name="sklearn_wrapper_example",
        verbose=1,
        save_artifacts=True,
        plots_visible=False
    )

    print(f"\nTraining complete!")
    print(f"Best RMSE: {result.best_rmse:.4f}")
    print(f"Best R²: {result.best_r2:.4f}")

    # Wrap for sklearn compatibility
    pipe = NIRSPipeline.from_result(result)

    print(f"\nNIRSPipeline created:")
    print(f"  Model name: {pipe.model_name}")
    print(f"  Preprocessing chain: {pipe.preprocessing_chain}")
    print(f"  CV Folds: {pipe.n_folds}")
    print(f"  Is fitted: {pipe.is_fitted_}")

    # Get some test data for demonstration
    # In practice, you'd use new data here
    from nirs4all.data import DatasetConfigs
    dataset = DatasetConfigs("sample_data/regression")
    for config, name in dataset.configs:
        ds = dataset.get_dataset(config, name)
        X_test = ds.x({})[:20]  # First 20 samples
        y_test = ds.y[:20]
        break

    # Use sklearn-style prediction
    y_pred = pipe.predict(X_test)
    print(f"\nPrediction shape: {y_pred.shape}")

    # Use sklearn-style scoring
    r2 = pipe.score(X_test, y_test)
    print(f"R² score: {r2:.4f}")

    # Access underlying model for advanced use
    print(f"\nUnderlying model type: {type(pipe.model_).__name__}")

    return pipe, result


def example_2_export_and_load():
    """Example 2: Export to bundle and load with NIRSPipeline.from_bundle().

    Shows the deployment workflow: export model, load in production.
    """
    print("\n" + "="*60)
    print("Example 2: Export Bundle and Load")
    print("="*60)

    # Train a model
    pipeline = [
        MinMaxScaler(),
        SavitzkyGolay(window_length=11, polyorder=2),
        ShuffleSplit(n_splits=2, test_size=0.25),
        {"model": PLSRegression(n_components=8)}
    ]

    result = nirs4all.run(
        pipeline=pipeline,
        dataset="sample_data/regression",
        name="bundle_export_example",
        verbose=1,
        plots_visible=False
    )

    # Export to bundle
    bundle_path = result.export("exports/sklearn_example.n4a")
    print(f"\nExported bundle: {bundle_path}")

    # Load from bundle (simulates production deployment)
    pipe_loaded = NIRSPipeline.from_bundle(bundle_path)

    print(f"\nLoaded NIRSPipeline from bundle:")
    print(f"  Model name: {pipe_loaded.model_name}")
    print(f"  Is fitted: {pipe_loaded.is_fitted_}")

    # Get test data
    from nirs4all.data import DatasetConfigs
    dataset = DatasetConfigs("sample_data/regression")
    for config, name in dataset.configs:
        ds = dataset.get_dataset(config, name)
        X_test = ds.x({})[:10]
        y_test = ds.y[:10]
        break

    # Predict with loaded bundle
    y_pred = pipe_loaded.predict(X_test)
    r2 = pipe_loaded.score(X_test, y_test)
    print(f"Predictions from bundle: {y_pred.shape}")
    print(f"R² score: {r2:.4f}")

    return pipe_loaded


def example_3_shap_integration():
    """Example 3: SHAP integration with NIRSPipeline.

    Shows how to use SHAP explainers with the sklearn wrapper.
    Requires: pip install shap
    """
    print("\n" + "="*60)
    print("Example 3: SHAP Integration")
    print("="*60)

    try:
        import shap
    except ImportError:
        print("SHAP not installed. Install with: pip install shap")
        print("Skipping SHAP example.")
        return None

    # Train a model
    pipeline = [
        MinMaxScaler(),
        StandardNormalVariate(),
        ShuffleSplit(n_splits=2, test_size=0.25),
        {"model": PLSRegression(n_components=5)}
    ]

    result = nirs4all.run(
        pipeline=pipeline,
        dataset="sample_data/regression",
        name="shap_example",
        verbose=1,
        plots_visible=False
    )

    # Wrap for SHAP
    pipe = NIRSPipeline.from_result(result)

    # Get data for SHAP
    from nirs4all.data import DatasetConfigs
    dataset = DatasetConfigs("sample_data/regression")
    for config, name in dataset.configs:
        ds = dataset.get_dataset(config, name)
        X = ds.x({})
        break

    # Use SHAP with the wrapper's predict function
    print("\nCreating SHAP explainer...")

    # For NIRS data with many features, use sampling
    background = X[:50]  # Background samples
    test_samples = X[50:55]  # Samples to explain

    # Use Kernel SHAP (works with any model)
    explainer = shap.KernelExplainer(pipe.predict, shap.kmeans(background, 10))

    print("Computing SHAP values...")
    shap_values = explainer.shap_values(test_samples, nsamples=100)

    print(f"SHAP values shape: {shap_values.shape}")
    print(f"Mean |SHAP| per feature (first 10): {np.mean(np.abs(shap_values), axis=0)[:10]}")

    # Access underlying model for model-specific SHAP
    model = pipe.model_
    print(f"\nUnderlying model ({type(model).__name__}) can be used with model-specific explainers")

    return shap_values


def example_4_transform():
    """Example 4: Using transform() to get preprocessed features.

    Shows how to access intermediate preprocessing results.
    """
    print("\n" + "="*60)
    print("Example 4: Transform (Preprocessing Only)")
    print("="*60)

    # Train a model
    pipeline = [
        MinMaxScaler(),
        StandardNormalVariate(),
        SavitzkyGolay(window_length=11, polyorder=2),
        ShuffleSplit(n_splits=2, test_size=0.25),
        {"model": PLSRegression(n_components=10)}
    ]

    result = nirs4all.run(
        pipeline=pipeline,
        dataset="sample_data/regression",
        name="transform_example",
        verbose=1,
        plots_visible=False
    )

    pipe = NIRSPipeline.from_result(result)

    # Get test data
    from nirs4all.data import DatasetConfigs
    dataset = DatasetConfigs("sample_data/regression")
    for config, name in dataset.configs:
        ds = dataset.get_dataset(config, name)
        X = ds.x({})[:10]
        break

    print(f"Original X shape: {X.shape}")

    # Transform applies preprocessing without model prediction
    X_transformed = pipe.transform(X)
    print(f"Transformed X shape: {X_transformed.shape}")

    # This is useful for:
    # - Debugging preprocessing
    # - Getting base model predictions in stacking
    # - Feature analysis after preprocessing

    return X_transformed


def example_5_metadata():
    """Example 5: Accessing pipeline metadata.

    Shows how to inspect the wrapped pipeline's configuration.
    """
    print("\n" + "="*60)
    print("Example 5: Pipeline Metadata")
    print("="*60)

    # Load from existing bundle (or create one)
    import os
    bundle_path = "exports/sklearn_example.n4a"

    if not os.path.exists(bundle_path):
        print(f"Bundle not found at {bundle_path}, creating one...")
        example_2_export_and_load()

    pipe = NIRSPipeline.from_bundle(bundle_path)

    print(f"\nPipeline Metadata:")
    print(f"  Model name: {pipe.model_name}")
    print(f"  Preprocessing chain: {pipe.preprocessing_chain}")
    print(f"  Model step index: {pipe.model_step_index}")
    print(f"  Number of CV folds: {pipe.n_folds}")
    print(f"  Fold weights: {pipe.fold_weights}")

    # Get transformers
    transformers = pipe.get_transformers()
    print(f"\n  Transformers ({len(transformers)}):")
    for name, transformer in transformers:
        print(f"    - {name}: {type(transformer).__name__}")

    # String representation
    print(f"\nString representation:")
    print(f"  repr: {repr(pipe)}")
    print(f"  str:\n{pipe}")

    return pipe


def main():
    """Run all examples."""
    print("\n" + "#"*60)
    print("# NIRSPipeline - sklearn-Compatible Wrapper Examples")
    print("#"*60)

    # Run examples
    pipe, result = example_1_basic_wrapper()
    pipe_loaded = example_2_export_and_load()

    # SHAP example (optional, requires shap package)
    shap_values = example_3_shap_integration()

    X_transformed = example_4_transform()
    pipe_meta = example_5_metadata()

    print("\n" + "#"*60)
    print("# All Examples Complete!")
    print("#"*60)

    print("\nKey Takeaways:")
    print("  1. Train with nirs4all.run(), wrap with NIRSPipeline.from_result()")
    print("  2. Export bundles for deployment, load with NIRSPipeline.from_bundle()")
    print("  3. Use pipe.predict() and pipe.score() for sklearn compatibility")
    print("  4. Access pipe.model_ for SHAP or advanced analysis")
    print("  5. Use pipe.transform() for preprocessing-only results")


if __name__ == "__main__":
    main()
