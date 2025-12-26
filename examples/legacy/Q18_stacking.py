"""
Q18 Example - Stacking and Ensemble Models
==========================================
Demonstrates NIRS analysis using sklearn stacking and voting ensemble models.
Shows how to combine multiple base learners with a meta-learner for improved predictions.

This example covers:
- StackingRegressor with multiple base estimators and a Ridge meta-learner
- StackingClassifier with RF, XGBoost, and LDA base estimators
- VotingRegressor and VotingClassifier for simple averaging ensembles
- Model serialization/deserialization of ensemble models
- Cross-validation with stacking models
- Prediction with saved stacking models

Note: Stacking models are more computationally expensive but often provide
better generalization by combining diverse base learners.
"""

# Standard library imports
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    StackingRegressor, StackingClassifier,
    VotingRegressor, VotingClassifier,
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.linear_model import Ridge, RidgeClassifier, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR, SVC

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.operators.transforms import (
    Detrend, FirstDerivative, StandardNormalVariate, SavitzkyGolay
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.visualization.predictions import PredictionAnalyzer
# Simple status symbols
CHECK = "[OK]"
CROSS = "[X]"


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q18 Stacking Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
parser.add_argument('--regression', action='store_true', help='Run regression example')
parser.add_argument('--classification', action='store_true', help='Run classification example')
args = parser.parse_args()

# If neither specified, run both
if not args.regression and not args.classification:
    args.regression = True
    args.classification = True


# =============================================================================
# REGRESSION EXAMPLE - Stacking and Voting Regressors
# =============================================================================
if args.regression:
    print("\n" + "=" * 60)
    print("REGRESSION EXAMPLE - Stacking and Voting Ensembles")
    print("=" * 60 + "\n")

    # Define base estimators for regression stacking
    regression_base_estimators = [
        ('pls', PLSRegression(n_components=10)),
        ('rf', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)),
        ('gbr', GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)),
    ]

    # Create Stacking Regressor with Ridge as meta-learner
    stacking_regressor = StackingRegressor(
        estimators=regression_base_estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=3,  # 3-fold CV for generating meta-features
        passthrough=False,  # Don't include original features in meta-learner
        n_jobs=-1
    )

    # Create Voting Regressor (simple averaging ensemble)
    voting_regressor = VotingRegressor(
        estimators=regression_base_estimators,
        n_jobs=-1
    )

    # Define the regression pipeline
    regression_pipeline = [
        MinMaxScaler(),  # Feature scaling
        {"y_processing": MinMaxScaler()},  # Target scaling
        {"feature_augmentation": [Detrend, FirstDerivative, StandardNormalVariate]},
        ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),

        # Individual models for comparison
        {"name": "PLS-10", "model": PLSRegression(n_components=10)},
        {"name": "RF", "model": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)},

        # Stacking ensemble
        {"name": "Stacking-Ridge", "model": stacking_regressor},

        # Voting ensemble
        {"name": "Voting-Avg", "model": voting_regressor},
    ]

    # Configure and run regression pipeline
    regression_pipeline_config = PipelineConfigs(regression_pipeline, "Q18_regression")
    regression_dataset_config = DatasetConfigs('sample_data/regression')

    runner = PipelineRunner(save_artifacts=True, verbose=1, plots_visible=args.plots)
    reg_predictions, reg_per_dataset = runner.run(regression_pipeline_config, regression_dataset_config)

    # Display regression results
    print("\n--- Regression Results ---")
    top_reg_models = reg_predictions.top(5, 'rmse')
    for idx, prediction in enumerate(top_reg_models):
        print(f"{idx+1}. {Predictions.pred_short_string(prediction, metrics=['rmse', 'r2'])}")

    # Visualization
    reg_analyzer = PredictionAnalyzer(reg_predictions)
    fig_reg_topk = reg_analyzer.plot_top_k(k=4, rank_metric='rmse', rank_partition='test')
    fig_reg_candle = reg_analyzer.plot_candlestick(variable="model_name", partition="test")

    # =========================================================================
    # PREDICTION TEST - Verify stacking model can be saved and reloaded
    # =========================================================================
    print("\n" + "-" * 60)
    print("PREDICTION TEST - Stacking Model Save/Reload Roundtrip")
    print("-" * 60)

    # Find the best stacking model prediction
    stacking_predictions = [p for p in reg_predictions.to_dicts() if 'Stacking' in p['model_name']]
    if stacking_predictions:
        # Get best stacking prediction by val score
        best_stacking = min(stacking_predictions, key=lambda p: p.get('val_mse', float('inf')))
        model_id = best_stacking['id']
        model_name = best_stacking['model_name']
        fold_id = best_stacking['fold_id']

        print(f"Source model: {model_name} (id: {model_id}, fold: {fold_id})")

        # Method 1: Predict using prediction entry on the SAME test data
        # Using the same test file that was used during training
        print("\n--- Method 1: Predict with prediction entry ---")
        predictor = PipelineRunner()
        # Use available data file for prediction verification
        prediction_dataset = DatasetConfigs({'X_test': 'sample_data/regression/Xval.csv.gz'})

        method1_predictions, _ = predictor.predict(best_stacking, prediction_dataset, verbose=0)
        print(f"Method 1 predictions (first 5): {method1_predictions[:5].flatten()}")
        print(f"Method 1 predictions shape: {method1_predictions.shape}")

        # Method 2: Predict using model ID
        print("\n--- Method 2: Predict with model ID ---")
        predictor2 = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        method2_predictions, _ = predictor2.predict(model_id, prediction_dataset, verbose=0)
        method2_array = method2_predictions[:5].flatten()
        print(f"Method 2 predictions (first 5): {method2_array}")

        # Compare method 1 and method 2 predictions - they should be identical
        is_identical = np.allclose(method1_predictions, method2_predictions, rtol=1e-5)
        print(f"\nMethod 1 and Method 2 identical: {CHECK + 'YES' if is_identical else CROSS + 'NO'}")
        assert is_identical, "Method 1 and Method 2 predictions do not match!"

        print("\n" + CHECK + " Stacking model prediction roundtrip PASSED!")
    else:
        print("No stacking model predictions found to test.")


# =============================================================================
# CLASSIFICATION EXAMPLE - Stacking and Voting Classifiers
# =============================================================================
if args.classification:
    print("\n" + "=" * 60)
    print("CLASSIFICATION EXAMPLE - Stacking and Voting Ensembles")
    print("=" * 60 + "\n")

    # Define base estimators for classification stacking
    classification_base_estimators = [
        ('rf', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)),
        ('gbc', GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)),
        ('lda', LinearDiscriminantAnalysis()),
    ]

    # Create Stacking Classifier with Logistic Regression as meta-learner
    stacking_classifier = StackingClassifier(
        estimators=classification_base_estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=3,
        passthrough=False,
        n_jobs=-1
    )

    # Create Voting Classifier (hard voting)
    voting_classifier_hard = VotingClassifier(
        estimators=classification_base_estimators,
        voting='hard',
        n_jobs=-1
    )

    # Create Voting Classifier (soft voting - uses probabilities)
    voting_classifier_soft = VotingClassifier(
        estimators=classification_base_estimators,
        voting='soft',
        n_jobs=-1
    )

    # Define the classification pipeline
    classification_pipeline = [
        StandardScaler(),  # Feature scaling
        {"feature_augmentation": [FirstDerivative, StandardNormalVariate, SavitzkyGolay]},
        ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),

        # Individual models for comparison
        {"name": "RF", "model": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)},
        {"name": "LDA", "model": LinearDiscriminantAnalysis()},

        # Stacking ensemble
        {"name": "Stacking-LR", "model": stacking_classifier},

        # Voting ensembles
        {"name": "Voting-Hard", "model": voting_classifier_hard},
        {"name": "Voting-Soft", "model": voting_classifier_soft},
    ]

    # Configure and run classification pipeline
    classification_pipeline_config = PipelineConfigs(classification_pipeline, "Q18_classification")
    classification_dataset_config = DatasetConfigs('sample_data/binary')

    runner = PipelineRunner(save_artifacts=True, verbose=1, plots_visible=args.plots)
    clf_predictions, clf_per_dataset = runner.run(classification_pipeline_config, classification_dataset_config)

    # Display classification results
    print("\n--- Classification Results ---")
    top_clf_models = clf_predictions.top(5, 'accuracy')
    for idx, prediction in enumerate(top_clf_models):
        print(f"{idx+1}. {Predictions.pred_short_string(prediction, metrics=['accuracy', 'balanced_recall'])}")

    # Visualization
    clf_analyzer = PredictionAnalyzer(clf_predictions)
    fig_clf_cm = clf_analyzer.plot_confusion_matrix(k=5, rank_metric='accuracy', rank_partition='test')
    fig_clf_candle = clf_analyzer.plot_candlestick(variable="model_name", display_metric='accuracy')

    # =========================================================================
    # PREDICTION TEST - Verify stacking classifier can be saved and reloaded
    # =========================================================================
    print("\n" + "-" * 60)
    print("PREDICTION TEST - Stacking Classifier Save/Reload Roundtrip")
    print("-" * 60)

    # Find the best stacking model prediction
    stacking_clf_predictions = [p for p in clf_predictions.to_dicts() if 'Stacking' in p['model_name']]
    if stacking_clf_predictions:
        # Get best stacking prediction by val score
        best_stacking_clf = max(stacking_clf_predictions, key=lambda p: p.get('val_balanced_accuracy', 0))
        model_id = best_stacking_clf['id']
        model_name = best_stacking_clf['model_name']

        print(f"Source model: {model_name} (id: {model_id})")

        # Predict using model ID - compare two prediction methods
        print("\n--- Predict with model ID ---")
        predictor = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        prediction_dataset = DatasetConfigs({'X_test': 'sample_data/binary/Xval.csv'})

        method1_predictions, _ = predictor.predict(model_id, prediction_dataset, verbose=0)
        print(f"Method 1 predictions (first 10): {method1_predictions[:10].flatten()}")

        # Predict again with prediction entry
        predictor2 = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        method2_predictions, _ = predictor2.predict(best_stacking_clf, prediction_dataset, verbose=0)
        print(f"Method 2 predictions (first 10): {method2_predictions[:10].flatten()}")

        # Compare both methods
        is_identical = np.array_equal(method1_predictions.astype(int), method2_predictions.astype(int))
        print(f"\nMethod 1 and Method 2 identical: {CHECK + 'YES' if is_identical else CROSS + 'NO'}")
        assert is_identical, "Classification predictions do not match!"

        print("\n" + CHECK + " Stacking classifier prediction roundtrip PASSED!")
    else:
        print("No stacking classifier predictions found to test.")


# =============================================================================
# SHOW ALL PLOTS
# =============================================================================
if args.show:
    plt.show()

print("\n" + "=" * 60)
print("Q18 Stacking Example Complete!")
print("=" * 60)
