"""
U01 - Multi-Model: Comparing Multiple Models
=============================================

Run and compare multiple models in a single pipeline.

This tutorial covers:

* Defining multiple models in one pipeline
* Using the _or_ generator syntax
* Comparing model performance
* Model selection strategies

Prerequisites
-------------
Complete the preprocessing examples first.

Next Steps
----------
See :ref:`U02_hyperparameter_tuning` for automated tuning.

Duration: ~4 minutes
Difficulty: ★★☆☆☆
"""

# Standard library imports
import argparse

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import StandardNormalVariate, FirstDerivative

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U01 Multi-Model Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Section 1: Why Compare Multiple Models?
# =============================================================================
print("\n" + "=" * 60)
print("U01 - Multi-Model Comparison")
print("=" * 60)

print("""
Different models have different strengths for NIRS data:

  📊 LINEAR MODELS (good baseline)
     PLSRegression - Handles collinearity, interpretable
     Ridge, Lasso  - Regularized regression
     ElasticNet    - Combines L1 + L2 regularization

  🌲 TREE-BASED (handles non-linearity)
     RandomForest       - Ensemble of decision trees
     GradientBoosting   - Sequential boosting
     ExtraTreesRegressor - Extremely randomized trees

  🔷 OTHER
     SVR           - Support Vector Regression
     KNeighbors    - Instance-based learning

nirs4all makes it easy to compare all of these in one run!
""")


# =============================================================================
# Section 2: Basic Multi-Model Pipeline
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Basic Multi-Model Pipeline")
print("-" * 60)

print("""
List multiple models in the pipeline. Each will be trained and evaluated.
""")

pipeline_basic = [
    # Preprocessing
    StandardNormalVariate(),
    FirstDerivative(),
    StandardScaler(),

    # Cross-validation
    ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),

    # Multiple models - each is evaluated
    {"model": PLSRegression(n_components=10)},
    {"model": Ridge(alpha=1.0)},
    {"model": RandomForestRegressor(n_estimators=100, random_state=42)},
]

result = nirs4all.run(
    pipeline=pipeline_basic,
    dataset="sample_data/regression",
    name="BasicMulti",
    verbose=1
)

print(f"\nNumber of model configurations: {result.num_predictions}")
print(f"Best RMSE: {result.best_score:.4f}")

# Show all results
print("\nAll results (ranked by RMSE):")
for i, pred in enumerate(result.top(10, display_metrics=['rmse', 'r2']), 1):
    model_name = pred.get('model_name', 'Unknown')
    print(f"   {i}. {model_name}: RMSE={pred.get('rmse', 0):.4f}")


# =============================================================================
# Section 3: Using _or_ Generator Syntax
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Using _or_ Generator Syntax")
print("-" * 60)

print("""
The _or_ syntax generates pipeline variants for each model.
More compact than listing each model separately.
""")

pipeline_or = [
    # Preprocessing
    StandardNormalVariate(),
    StandardScaler(),

    # Cross-validation
    ShuffleSplit(n_splits=3, random_state=42),

    # _or_ generates variants for each model - each must be wrapped in {"model": ...}
    {"_or_": [
        {"model": PLSRegression(n_components=10)},
        {"model": Ridge(alpha=1.0)},
        {"model": Lasso(alpha=0.1)},
        {"model": ElasticNet(alpha=0.1, l1_ratio=0.5)},
    ]},
]

result_or = nirs4all.run(
    pipeline=pipeline_or,
    dataset="sample_data/regression",
    name="OrSyntax",
    verbose=1
)

print(f"\nVariants generated: {result_or.num_predictions}")
print(f"Best RMSE: {result_or.best_score:.4f}")

for pred in result_or.top(5, display_metrics=['rmse', 'r2']):
    print(f"   {pred.get('model_name', 'Unknown')}: RMSE={pred.get('rmse', 0):.4f}")


# =============================================================================
# Section 4: Comprehensive Model Comparison
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Comprehensive Model Comparison")
print("-" * 60)

print("""
Compare a wide range of models to find the best for your data.
""")

pipeline_comprehensive = [
    # Preprocessing
    StandardNormalVariate(),
    FirstDerivative(),

    # Cross-validation
    ShuffleSplit(n_splits=3, random_state=42),

    # Linear models
    {"model": PLSRegression(n_components=5)},
    {"model": PLSRegression(n_components=10)},
    {"model": PLSRegression(n_components=15)},
    {"model": Ridge(alpha=0.1)},
    {"model": Ridge(alpha=1.0)},
    {"model": Lasso(alpha=0.01)},
    {"model": ElasticNet(alpha=0.1, l1_ratio=0.5)},

    # Tree-based models
    {"model": RandomForestRegressor(n_estimators=50, random_state=42)},
    {"model": RandomForestRegressor(n_estimators=100, random_state=42)},
    {"model": GradientBoostingRegressor(n_estimators=50, random_state=42)},
    {"model": ExtraTreesRegressor(n_estimators=50, random_state=42)},

    # Other models
    {"model": SVR(kernel='rbf', C=1.0)},
    {"model": KNeighborsRegressor(n_neighbors=5)},
    {"model": KNeighborsRegressor(n_neighbors=10)},
]

result_comp = nirs4all.run(
    pipeline=pipeline_comprehensive,
    dataset="sample_data/regression",
    name="Comprehensive",
    verbose=1
)

print(f"\nTotal configurations tested: {result_comp.num_predictions}")
print(f"Best RMSE: {result_comp.best_score:.4f}")

# Show top 10
print("\nTop 10 configurations:")
for i, pred in enumerate(result_comp.top(10, display_metrics=['rmse', 'r2']), 1):
    model_name = pred.get('model_name', 'Unknown')
    print(f"   {i}. {model_name}: RMSE={pred.get('rmse', 0):.4f}")


# =============================================================================
# Section 5: Combined Model + Preprocessing Search
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Combined Model + Preprocessing Search")
print("-" * 60)

print("""
Combine feature_augmentation with multiple models to find
the best preprocessing + model combination.
""")

from nirs4all.operators.transforms import (
    MultiplicativeScatterCorrection,
    Detrend,
    SavitzkyGolay,
)

pipeline_combined = [
    # Explore preprocessing options
    {"feature_augmentation": [
        StandardNormalVariate,
        MultiplicativeScatterCorrection,
        Detrend,
    ], "action": "extend"},

    # Add derivative option
    {"feature_augmentation": [FirstDerivative], "action": "add"},

    # Cross-validation
    ShuffleSplit(n_splits=2, random_state=42),

    # Multiple models
    {"model": PLSRegression(n_components=10)},
    {"model": Ridge(alpha=1.0)},
    {"model": RandomForestRegressor(n_estimators=50, random_state=42)},
]

result_combined = nirs4all.run(
    pipeline=pipeline_combined,
    dataset="sample_data/regression",
    name="Combined",
    verbose=1
)

print(f"\nTotal configurations: {result_combined.num_predictions}")
print(f"Best RMSE: {result_combined.best_score:.4f}")

# Show top results with preprocessing info
print("\nTop 10 configurations (preprocessing + model):")
for i, pred in enumerate(result_combined.top(10, display_metrics=['rmse', 'r2']), 1):
    preproc = pred.get('preprocessings', 'N/A')
    model = pred.get('model_name', 'Unknown')
    print(f"   {i}. {preproc} + {model}: RMSE={pred.get('rmse', 0):.4f}")


# =============================================================================
# Section 6: Classification Multi-Model
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Classification Multi-Model")
print("-" * 60)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

pipeline_classif = [
    StandardNormalVariate(),
    StandardScaler(),

    StratifiedKFold(n_splits=3, shuffle=True, random_state=42),

    {"model": LogisticRegression(max_iter=1000)},
    {"model": RandomForestClassifier(n_estimators=100, random_state=42)},
    {"model": GradientBoostingClassifier(n_estimators=50, random_state=42)},
    {"model": SVC(kernel='rbf', probability=True)},
    {"model": KNeighborsClassifier(n_neighbors=5)},
]

result_classif = nirs4all.run(
    pipeline=pipeline_classif,
    dataset="sample_data/classification",
    name="ClassifMulti",
    verbose=1
)

print(f"\nClassification results:")
for i, pred in enumerate(result_classif.top(5, display_metrics=['rmse', 'r2']), 1):
    model_name = pred.get('model_name', 'Unknown')
    # For classification, 'rmse' is actually error rate
    accuracy = (1 - pred.get('rmse', 0)) * 100
    print(f"   {i}. {model_name}: Accuracy={accuracy:.1f}%")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Multi-Model Comparison Approaches:

  1. List models separately:
     {"model": PLSRegression(n_components=10)},
     {"model": Ridge(alpha=1.0)},
     {"model": RandomForestRegressor()},

  2. Use _or_ generator:
     {"_or_": [{"model": PLS(10)}, {"model": Ridge(1.0)}, {"model": RF()}]}

  3. Combine with preprocessing search:
     {"feature_augmentation": [SNV, MSC], "action": "extend"},
     {"model": PLSRegression()},
     {"model": RandomForest()},

Common Model Families for NIRS:

  LINEAR:
    PLSRegression     - Best for collinear spectral data
    Ridge, Lasso      - Regularized linear models
    ElasticNet        - Combined L1+L2 regularization

  TREE-BASED:
    RandomForest      - Robust ensemble
    GradientBoosting  - Often best non-linear
    ExtraTrees        - Fast, less overfitting

  OTHER:
    SVR, KNeighbors   - Can capture local patterns

Result Analysis:
    result.top(n)       # Get top n configurations
    result.best_score    # Best RMSE achieved
    result.num_predictions  # Total configurations tested

Next: U02_hyperparameter_tuning.py - Automated hyperparameter search
""")
