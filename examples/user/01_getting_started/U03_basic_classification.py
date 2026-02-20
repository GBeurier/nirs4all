"""
U03 - Basic Classification: NIRS Classification with Multiple Models
=====================================================================

Classification pipeline with Random Forest, XGBoost, and confusion matrix visualization.

This tutorial covers:

* Setting up a classification pipeline
* Using multiple classifiers (Random Forest, XGBoost)
* Confusion matrix visualization
* Classification metrics (accuracy, balanced recall)

Prerequisites
-------------
Complete :ref:`U02_basic_regression` first.

Next Steps
----------
See :ref:`U04_visualization` for advanced visualization techniques.

Duration: ~2 minutes
Difficulty: ★★☆☆☆
"""

# Standard library imports
import argparse

import matplotlib.pyplot as plt

# Third-party imports
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import FirstDerivative, Haar, MultiplicativeScatterCorrection, StandardNormalVariate
from nirs4all.visualization.predictions import PredictionAnalyzer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U03 Basic Classification Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# =============================================================================
# Section 1: Build Classification Pipeline
# =============================================================================
print("\n" + "=" * 60)
print("U03 - Basic Classification Pipeline")
print("=" * 60)

# Build the pipeline
pipeline = [
    # Visualize target distribution
    "y_chart",

    # Feature augmentation with preprocessing options
    {
        "feature_augmentation": [
            FirstDerivative,
            StandardNormalVariate,
            Haar,
            MultiplicativeScatterCorrection
        ]
    },

    # Feature scaling
    StandardScaler(),

    # Cross-validation
    ShuffleSplit(n_splits=3, test_size=0.25),

    # Visualization of fold distribution
    "fold_chart",

    # Random Forest classifier
    {
        "model": RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=42,
            verbose=0
        ),
        "name": "RandomForest"
    },
]

# Add XGBoost if available
if HAS_XGBOOST:
    pipeline.append({
        "model": XGBClassifier(n_estimators=20, max_depth=5, verbosity=0, random_state=42),
        "name": "XGBoost"
    })
    print("   ✓ XGBoost is available")
else:
    print("   ⚠ XGBoost not installed - using Random Forest only")

print("\n📋 Classification Pipeline:")
print("   • Feature augmentation with 4 preprocessing options")
print("   • StandardScaler for feature normalization")
print("   • 3-fold ShuffleSplit cross-validation")
print(f"   • {2 if HAS_XGBOOST else 1} classifier(s)")

# =============================================================================
# Section 2: Train the Pipeline
# =============================================================================
print("\n" + "-" * 60)
print("Training Pipeline")
print("-" * 60)

result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/binary",
    name="BasicClassification",
    verbose=1,
    save_artifacts=True,
    plots_visible=args.plots
)

print("\n📊 Training complete!")
print(f"   Generated {result.num_predictions} predictions")

# =============================================================================
# Section 3: Display Results
# =============================================================================
print("\n" + "-" * 60)
print("Top 5 Models by Accuracy")
print("-" * 60)

# Get predictions object for analysis
predictions = result.predictions

# Display top models with classification metrics
# Use display_metrics to include accuracy and balanced_recall in results
top_models = predictions.top(5, rank_metric='accuracy', display_metrics=['accuracy', 'balanced_recall'])
assert isinstance(top_models, list)
for i, pred in enumerate(top_models, 1):
    model_name = pred.get('model_name', 'unknown')
    preproc = pred.get('preprocessings', 'N/A')
    accuracy = pred.get('accuracy', 0)
    balanced = pred.get('balanced_recall', 0)
    print(f"{i}. {model_name}")
    print(f"   Accuracy: {accuracy:.4f} | Balanced Recall: {balanced:.4f}")
    print(f"   Preprocessing: {preproc}")

# =============================================================================
# Section 4: Visualize Results
# =============================================================================
print("\n" + "-" * 60)
print("Creating Visualizations")
print("-" * 60)

analyzer = PredictionAnalyzer(predictions)

# Confusion matrix for top 4 models
fig1 = analyzer.plot_confusion_matrix(
    k=4,
    rank_metric='accuracy',
    rank_partition='val',
    display_partition='test'
)
print("   ✓ Created confusion matrices for top 4 models")

# Candlestick chart for model comparison
fig2 = analyzer.plot_candlestick(
    variable="model_name",
    display_metric='accuracy',
)
print("   ✓ Created candlestick chart (accuracy)")

# Heatmap: models vs preprocessing
fig3 = analyzer.plot_heatmap(
    x_var="model_name",
    y_var="preprocessings",
    display_metric='accuracy',
)
print("   ✓ Created heatmap: models vs preprocessing")

# Histogram of balanced recall
fig4 = analyzer.plot_histogram(
    display_metric='balanced_recall',
)
print("   ✓ Created histogram (balanced recall)")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. Classification pipeline setup with sklearn classifiers
2. Feature augmentation for preprocessing exploration
3. Classification metrics: accuracy, balanced_recall
4. Confusion matrix visualization

Key classification metrics:
  accuracy         - Overall correct predictions
  balanced_recall  - Average recall per class (handles imbalanced data)
  balanced_accuracy - Average accuracy per class

Key visualization:
  analyzer.plot_confusion_matrix(k=4)  - Confusion matrices for top K

Dataset formats supported:
  - Folder path: auto-detects X/Y files
  - Dict with explicit paths and params
  - Tuple: (X, y) numpy arrays

Next: U04_visualization.py - Advanced visualization techniques
""")

if args.show:
    plt.show()
