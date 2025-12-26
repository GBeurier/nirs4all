"""
U15 - Stacking Ensembles: Combining Multiple Models
====================================================

Create ensemble models using sklearn stacking and voting.

This tutorial covers:

* StackingRegressor and StackingClassifier
* VotingRegressor and VotingClassifier
* Base estimators and meta-learners
* Comparing ensembles with individual models

Prerequisites
-------------
Complete :ref:`U13_multi_model` first.

Next Steps
----------
See :ref:`U16_pls_variants` for PLS method variations.

Duration: ~5 minutes
Difficulty: ★★★☆☆
"""

# Standard library imports
import argparse
import matplotlib.pyplot as plt

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    StackingRegressor,
    StackingClassifier,
    VotingRegressor,
    VotingClassifier,
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import (
    StandardNormalVariate,
    FirstDerivative,
    Detrend,
)
from nirs4all.visualization.predictions import PredictionAnalyzer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U15 Stacking Ensembles Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Section 1: Introduction to Stacking
# =============================================================================
print("\n" + "=" * 60)
print("U15 - Stacking Ensembles")
print("=" * 60)

print("""
Stacking combines multiple models (base learners) using a meta-learner.

  📊 HOW STACKING WORKS
     1. Train multiple diverse base models on the data
     2. Generate predictions from each base model (using CV)
     3. Train a meta-learner on the base model predictions
     4. Final prediction combines all base model outputs

  📈 ENSEMBLE TYPES
     StackingRegressor  - Combines predictions via meta-learner
     VotingRegressor    - Simple averaging of predictions
     StackingClassifier - Stacking for classification
     VotingClassifier   - Voting (hard or soft) for classification

  📉 BENEFITS
     ✓ Often better than any single model
     ✓ Reduces overfitting through diversity
     ✓ Combines strengths of different model types
""")


# =============================================================================
# Section 2: Stacking Regressor
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Stacking Regressor")
print("-" * 60)

print("""
Combine PLS, Random Forest, and Gradient Boosting with a Ridge meta-learner.
""")

# Define base estimators
regression_base_estimators = [
    ('pls', PLSRegression(n_components=10)),
    ('rf', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)),
    ('gbr', GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)),
]

# Create Stacking Regressor
stacking_regressor = StackingRegressor(
    estimators=regression_base_estimators,
    final_estimator=Ridge(alpha=1.0),  # Meta-learner
    cv=3,                               # CV for meta-features
    passthrough=False,                  # Don't include original features
    n_jobs=-1
)

# Pipeline with stacking
pipeline_stacking = [
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},
    StandardNormalVariate(),
    FirstDerivative(),

    ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),

    # Individual models for comparison
    {"name": "PLS-10", "model": PLSRegression(n_components=10)},
    {"name": "RF-50", "model": RandomForestRegressor(n_estimators=50, random_state=42)},
    {"name": "GBR-50", "model": GradientBoostingRegressor(n_estimators=50, random_state=42)},

    # Stacking ensemble
    {"name": "Stacking-Ridge", "model": stacking_regressor},
]

result_stacking = nirs4all.run(
    pipeline=pipeline_stacking,
    dataset="sample_data/regression",
    name="StackingReg",
    verbose=1
)

print(f"\nResults comparison:")
for pred in result_stacking.top(5, display_metrics=['rmse', 'r2']):
    model = pred.get('model_name', 'Unknown')
    print(f"   {model}: RMSE={pred.get('rmse', 0):.4f}")


# =============================================================================
# Section 3: Voting Regressor
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Voting Regressor")
print("-" * 60)

print("""
VotingRegressor averages predictions from base models.
Simpler than stacking, no meta-learner training.
""")

# Create Voting Regressor
voting_regressor = VotingRegressor(
    estimators=regression_base_estimators,
    n_jobs=-1
)

pipeline_voting = [
    MinMaxScaler(),
    StandardNormalVariate(),

    ShuffleSplit(n_splits=2, random_state=42),

    {"name": "PLS-10", "model": PLSRegression(n_components=10)},
    {"name": "RF-50", "model": RandomForestRegressor(n_estimators=50, random_state=42)},
    {"name": "Voting-Avg", "model": voting_regressor},
]

result_voting = nirs4all.run(
    pipeline=pipeline_voting,
    dataset="sample_data/regression",
    name="VotingReg",
    verbose=1
)

print(f"\nVoting results:")
for pred in result_voting.top(5, display_metrics=['rmse', 'r2']):
    print(f"   {pred.get('model_name', 'Unknown')}: RMSE={pred.get('rmse', 0):.4f}")


# =============================================================================
# Section 4: Stacking Classifier
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Stacking Classifier")
print("-" * 60)

print("""
Stacking for classification with Logistic Regression meta-learner.
""")

# Classification base estimators
classification_base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)),
    ('gbc', GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)),
    ('lda', LinearDiscriminantAnalysis()),
]

# Stacking Classifier
stacking_classifier = StackingClassifier(
    estimators=classification_base_estimators,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=3,
    passthrough=False,
    n_jobs=-1
)

pipeline_stacking_clf = [
    StandardScaler(),
    StandardNormalVariate(),

    ShuffleSplit(n_splits=2, random_state=42),

    {"name": "RF", "model": RandomForestClassifier(n_estimators=50, random_state=42)},
    {"name": "LDA", "model": LinearDiscriminantAnalysis()},
    {"name": "Stacking-LR", "model": stacking_classifier},
]

result_stacking_clf = nirs4all.run(
    pipeline=pipeline_stacking_clf,
    dataset="sample_data/classification",
    name="StackingClf",
    verbose=1
)

print(f"\nClassification results:")
for pred in result_stacking_clf.top(5, display_metrics=['rmse', 'r2']):
    model = pred.get('model_name', 'Unknown')
    accuracy = (1 - pred.get('rmse', 0)) * 100
    print(f"   {model}: Accuracy={accuracy:.1f}%")


# =============================================================================
# Section 5: Voting Classifier
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Voting Classifier")
print("-" * 60)

print("""
VotingClassifier supports:
  - hard: Majority vote on predicted classes
  - soft: Average predicted probabilities
""")

# Hard voting (majority vote)
voting_hard = VotingClassifier(
    estimators=classification_base_estimators,
    voting='hard',
    n_jobs=-1
)

# Soft voting (probability averaging)
voting_soft = VotingClassifier(
    estimators=classification_base_estimators,
    voting='soft',
    n_jobs=-1
)

pipeline_voting_clf = [
    StandardScaler(),
    StandardNormalVariate(),

    ShuffleSplit(n_splits=2, random_state=42),

    {"name": "RF", "model": RandomForestClassifier(n_estimators=50, random_state=42)},
    {"name": "Voting-Hard", "model": voting_hard},
    {"name": "Voting-Soft", "model": voting_soft},
]

result_voting_clf = nirs4all.run(
    pipeline=pipeline_voting_clf,
    dataset="sample_data/classification",
    name="VotingClf",
    verbose=1
)

print(f"\nVoting classifier results:")
for pred in result_voting_clf.top(5, display_metrics=['rmse', 'r2']):
    model = pred.get('model_name', 'Unknown')
    accuracy = (1 - pred.get('rmse', 0)) * 100
    print(f"   {model}: Accuracy={accuracy:.1f}%")


# =============================================================================
# Section 6: Custom Ensemble Configuration
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Custom Ensemble Configuration")
print("-" * 60)

print("""
Customize ensemble parameters:
  - passthrough: Include original features in meta-learner
  - weights: Weight base estimators differently
  - cv: Cross-validation folds for meta-features
""")

# Weighted voting
weighted_voting = VotingRegressor(
    estimators=regression_base_estimators,
    weights=[0.5, 0.3, 0.2],  # Weight PLS higher
    n_jobs=-1
)

# Stacking with passthrough
stacking_passthrough = StackingRegressor(
    estimators=regression_base_estimators,
    final_estimator=Ridge(alpha=0.5),
    cv=5,                 # More CV folds
    passthrough=True,     # Include original features
    n_jobs=-1
)

pipeline_custom = [
    MinMaxScaler(),
    StandardNormalVariate(),

    ShuffleSplit(n_splits=2, random_state=42),

    {"name": "Voting-Weighted", "model": weighted_voting},
    {"name": "Stacking-Passthrough", "model": stacking_passthrough},
]

result_custom = nirs4all.run(
    pipeline=pipeline_custom,
    dataset="sample_data/regression",
    name="CustomEnsemble",
    verbose=1
)

print(f"\nCustom ensemble results:")
for pred in result_custom.top(5, display_metrics=['rmse', 'r2']):
    print(f"   {pred.get('model_name', 'Unknown')}: RMSE={pred.get('rmse', 0):.4f}")


# =============================================================================
# Section 7: Visualization
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: Visualization")
print("-" * 60)

if args.plots:
    # Visualize regression results
    analyzer = PredictionAnalyzer(result_stacking.predictions)

    fig1 = analyzer.plot_top_k(k=4, rank_metric='rmse')
    fig2 = analyzer.plot_candlestick(variable="model_name", partition="test")

    print("Charts generated (use --show to display)")

    if args.show:
        plt.show()
else:
    print("Use --plots to generate visualization charts")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Ensemble Models:

  STACKING (Learned Combination):
    stacking_reg = StackingRegressor(
        estimators=[
            ('pls', PLSRegression(n_components=10)),
            ('rf', RandomForestRegressor()),
            ('gbr', GradientBoostingRegressor()),
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=3,
        passthrough=False,
    )

  VOTING (Simple Averaging):
    voting_reg = VotingRegressor(
        estimators=[...],
        weights=[0.5, 0.3, 0.2],  # Optional weights
    )

    voting_clf = VotingClassifier(
        estimators=[...],
        voting='soft',  # 'hard' or 'soft'
    )

Key Parameters:
  estimators    - List of (name, model) tuples
  final_estimator - Meta-learner (for stacking)
  cv            - CV folds for meta-feature generation
  passthrough   - Include original features in meta-learner
  weights       - Weight base estimators (for voting)
  voting        - 'hard' or 'soft' (for classification)

When to Use:
  ✓ Diverse models available (linear + tree-based)
  ✓ Need better generalization
  ✓ Computational resources available
  ✗ Interpretability is critical
  ✗ Very small datasets

Next: U16_pls_variants.py - PLS method variations
""")
