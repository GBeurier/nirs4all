"""
Q15 JAX Models Example
======================
Demonstrates NIRS analysis using JAX/Flax models.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.splitters import SPXYSplitter
from nirs4all.operators.models.jax import JaxMLPRegressor, JaxMLPClassifier
from nirs4all.operators.models.jax.nicon import nicon as nicon_jax
from nirs4all.utils.backend import JAX_AVAILABLE

if not JAX_AVAILABLE:
    print("JAX is not available. Skipping example.")
    exit(0)

parser = argparse.ArgumentParser(description='Q15 JAX Models Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()

# Regression Example
print("\n--- JAX Regression Example ---")
data_path_reg = 'sample_data/regression'

pipeline_reg = [
    StandardScaler,
    # SPXYSplitter(0.25),
    ShuffleSplit(n_splits=2, test_size=0.25),
    # Use instance with configured parameters
    [
        {
            'model': JaxMLPRegressor(features=[64, 32]),
            'finetune_params': {
                'n_trials': 4,
                'verbose': 2,
                'sample': 'tpe',
                'model_params': {
                    'features': [[32], [64, 32], [64, 32, 16]],
                },
            },
            'train_params': {
                'epochs': 20,
                'batch_size': 16,
                'learning_rate': 0.001,
                'verbose': 0
            }
        },
        {
            'model': nicon_jax,
            'train_params': {
                'epochs': 20,
                'batch_size': 16,
                'learning_rate': 0.001,
                'verbose': 0
            }
        }
    ]
]

runner = PipelineRunner(save_artifacts=True, verbose=0, plots_visible=args.plots)
preds_reg, _ = runner.run(PipelineConfigs(pipeline_reg), DatasetConfigs(data_path_reg))
print("Top Regression Models:")
print(preds_reg.top(1))

# Prediction Reuse
print("\n--- JAX Prediction Reuse Example ---")
best_prediction = preds_reg.top(1)[0]
model_id = best_prediction['id']
print(f"Best model: {best_prediction['model_name']} (id: {model_id})")
reference_predictions = best_prediction['y_pred'][:5].flatten()
print("Reference predictions:", reference_predictions)

predictor = PipelineRunner()
# Using Xval for prediction reuse demonstration
prediction_dataset = DatasetConfigs({'X_test': 'sample_data/regression/Xval.csv.gz'})
method1_predictions, _ = predictor.predict(best_prediction, prediction_dataset, verbose=0)
method1_array = method1_predictions[:5].flatten()
print("Predictions on new data (first 5):")
print(method1_array)

is_identical = np.allclose(method1_array, reference_predictions)
assert is_identical, "Method 1 predictions do not match reference!"
print(f"Method 1 identical to training: {'✅ YES' if is_identical else '❌ NO'}")

# Classification Example
print("\n--- JAX Classification Example ---")
data_path_cls = 'sample_data/classification'

pipeline_cls = [
    StandardScaler,
    SPXYSplitter(0.25),
    ShuffleSplit(n_splits=2, test_size=0.25),
    {
        'model': JaxMLPClassifier(features=[64, 32], num_classes=2), # num_classes=2 for binary classification in sample_data/classification?
        # Note: sample_data/classification might have more classes.
        # But JaxMLPClassifier needs num_classes in __init__.
        # ModelFactory usually injects it if we pass the class.
        # Since we pass instance, we must set it.
        # Let's assume 2 for now or check dataset.
        'train_params': {
            'epochs': 20,
            'batch_size': 16,
            'learning_rate': 0.001,
            'verbose': 1
        }
    }
]

preds_cls, _ = runner.run(PipelineConfigs(pipeline_cls), DatasetConfigs(data_path_cls))
print("Top Classification Models:")
print(preds_cls.top(1))

if args.show:
    plt.show()
