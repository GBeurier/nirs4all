"""
U04 - sklearn Integration: NIRSPipeline Wrapper
================================================

Use trained nirs4all models with sklearn tools and workflows.

This tutorial covers:

* Create NIRSPipeline from training results
* sklearn-compatible predict() and score() methods
* Load from exported bundles for deployment
* Access underlying model for advanced analysis

Prerequisites
-------------
Complete :ref:`U03_workspace_management` first.

Next Steps
----------
See :ref:`07_explainability/U01_shap_basics` for model explainability.

Duration: ~5 minutes
Difficulty: ★★☆☆☆
"""

# Standard library imports
import argparse
from pathlib import Path

# Third-party imports
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

# NIRS4All imports
import nirs4all
from nirs4all.sklearn import NIRSPipeline
from nirs4all.operators.transforms import StandardNormalVariate, SavitzkyGolay
from nirs4all.data import DatasetConfigs

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U04 sklearn Integration Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Section 1: Why sklearn Integration?
# =============================================================================
print("\n" + "=" * 60)
print("U04 - sklearn Integration: NIRSPipeline Wrapper")
print("=" * 60)

print("""
NIRSPipeline provides sklearn compatibility:

  📊 SKLEARN-STYLE API
     - predict(X) - Make predictions
     - score(X, y) - Calculate R² score
     - transform(X) - Apply preprocessing only

  🔧 INTEGRATION BENEFITS
     - Use with sklearn cross-validation
     - Use with SHAP explainers
     - Use with hyperparameter search
     - Standard ML workflow compatibility

  📦 TWO LOADING METHODS
     - from_result(): From training output
     - from_bundle(): From exported .n4a file
""")


# =============================================================================
# Section 2: Training and Wrapping
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Training and Wrapping")
print("-" * 60)

print("""
Train with nirs4all.run(), wrap with NIRSPipeline.from_result().
""")

# Define pipeline
pipeline = [
    MinMaxScaler(),
    StandardNormalVariate(),
    ShuffleSplit(n_splits=3, test_size=0.25),
    {"model": PLSRegression(n_components=10)}
]

# Train
result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    name="sklearn_wrapper",
    verbose=1,
    save_artifacts=True,
    plots_visible=False
)

print(f"\nTraining complete!")
print(f"  Best RMSE: {result.best_rmse:.4f}")
print(f"  Best R²: {result.best_r2:.4f}")

# Wrap for sklearn compatibility
pipe = NIRSPipeline.from_result(result)

print(f"\nNIRSPipeline created:")
print(f"  Model name: {pipe.model_name}")
print(f"  Preprocessing: {pipe.preprocessing_chain}")
print(f"  CV Folds: {pipe.n_folds}")
print(f"  Is fitted: {pipe.is_fitted_}")


# =============================================================================
# Section 3: sklearn-Style Prediction
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: sklearn-Style Prediction")
print("-" * 60)

print("""
Use standard predict() and score() methods.
""")

# Get test data
dataset = DatasetConfigs("sample_data/regression")
for config, name in dataset.configs:
    ds = dataset.get_dataset(config, name)
    X_test = ds.x({})[:20]  # First 20 samples
    y_test = ds.y({})[:20]  # y() is a method, needs selector
    break

# Predict like sklearn
y_pred = pipe.predict(X_test)
print(f"Predictions shape: {y_pred.shape}")
print(f"First 5 predictions: {y_pred[:5].flatten()}")

# Score like sklearn
r2 = pipe.score(X_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nR² score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")


# =============================================================================
# Section 4: Transform (Preprocessing Only)
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Transform (Preprocessing Only)")
print("-" * 60)

print("""
Use transform() to apply preprocessing without prediction.
""")

# Get raw data
X_raw = ds.x({})[:10]
print(f"Original X shape: {X_raw.shape}")

# Transform applies preprocessing only
X_transformed = pipe.transform(X_raw)
print(f"Transformed X shape: {X_transformed.shape}")

# Compare statistics
print(f"\nOriginal stats:")
print(f"  Mean: {X_raw.mean():.4f}, Std: {X_raw.std():.4f}")
print(f"Transformed stats:")
print(f"  Mean: {X_transformed.mean():.4f}, Std: {X_transformed.std():.4f}")


# =============================================================================
# Section 5: Export and Load from Bundle
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Export and Load from Bundle")
print("-" * 60)

print("""
Export to bundle, load in production with from_bundle().
""")

# Export to bundle
exports_dir = Path("exports")
exports_dir.mkdir(exist_ok=True)

bundle_path = result.export("exports/sklearn_wrapper.n4a")
print(f"Exported bundle: {bundle_path}")

# Load from bundle (simulates production deployment)
pipe_loaded = NIRSPipeline.from_bundle(bundle_path)

print(f"\nLoaded NIRSPipeline:")
print(f"  Model name: {pipe_loaded.model_name}")
print(f"  Is fitted: {pipe_loaded.is_fitted_}")

# Verify predictions match
y_pred_loaded = pipe_loaded.predict(X_test)
predictions_match = np.allclose(y_pred, y_pred_loaded)
print(f"\nPredictions match original: {'YES ✓' if predictions_match else 'NO ✗'}")


# =============================================================================
# Section 6: Accessing Underlying Model
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Accessing Underlying Model")
print("-" * 60)

print("""
Access the underlying model for advanced analysis.
""")

# Get underlying model
model = pipe.model_
print(f"Underlying model type: {type(model).__name__}")

# For PLS, we can access coefficients
if hasattr(model, 'coef_'):
    coefs = model.coef_
    print(f"Coefficient shape: {coefs.shape}")
    print(f"Top 5 coefficients: {np.abs(coefs.flatten())[:5]}")

# Get transformers
transformers = pipe.get_transformers()
print(f"\nTransformers ({len(transformers)}):")
for name, transformer in transformers:
    print(f"  - {name}: {type(transformer).__name__}")


# =============================================================================
# Section 7: Pipeline Metadata
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: Pipeline Metadata")
print("-" * 60)

print("""
Inspect the wrapped pipeline's configuration.
""")

print("Pipeline Metadata:")
print(f"  Model name: {pipe.model_name}")
print(f"  Preprocessing chain: {pipe.preprocessing_chain}")
print(f"  Model step index: {pipe.model_step_index}")
print(f"  Number of CV folds: {pipe.n_folds}")
print(f"  Fold weights: {pipe.fold_weights}")

# String representation
print(f"\nString representation:")
print(f"  {repr(pipe)}")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
sklearn Integration Workflow:

  1. TRAIN WITH NIRS4ALL:
     result = nirs4all.run(
         pipeline=[...],
         dataset="sample_data/regression"
     )

  2. WRAP FOR SKLEARN:
     pipe = NIRSPipeline.from_result(result)

  3. SKLEARN-STYLE USAGE:
     y_pred = pipe.predict(X_test)  # Predictions
     r2 = pipe.score(X_test, y_test)  # R² score
     X_prep = pipe.transform(X_raw)  # Preprocessing only

  4. EXPORT AND LOAD:
     # Export
     bundle_path = result.export("model.n4a")

     # Load in production
     pipe = NIRSPipeline.from_bundle(bundle_path)
     y_pred = pipe.predict(X_new)

  5. ACCESS UNDERLYING MODEL:
     model = pipe.model_  # Original sklearn model
     transformers = pipe.get_transformers()  # Preprocessing chain

Key NIRSPipeline Methods:
  - predict(X) → numpy array
  - score(X, y) → float (R²)
  - transform(X) → numpy array
  - from_result(result) → NIRSPipeline
  - from_bundle(path) → NIRSPipeline

Benefits:
  - Use with sklearn cross_val_score
  - Use with SHAP explainers
  - Use with GridSearchCV
  - Standard ML workflow compatibility

Next: See 07_explainability/U01_shap_basics.py - Model explainability with SHAP
""")
