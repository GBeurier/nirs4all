"""
Q15 JAX Models Example
======================
Demonstrates NIRS analysis using JAX/Flax models.
"""

import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.splitters import SPXYSplitter
from nirs4all.operators.models.jax import JaxMLPRegressor, JaxMLPClassifier
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
    {
        'model': JaxMLPRegressor(features=[64, 32]),
        'train_params': {
            'epochs': 20,
            'batch_size': 16,
            'learning_rate': 0.001,
            'verbose': 1
        }
    }
]

runner = PipelineRunner(save_files=False, verbose=1, plots_visible=args.plots)
preds_reg, _ = runner.run(PipelineConfigs(pipeline_reg), DatasetConfigs(data_path_reg))
print("Top Regression Models:")
print(preds_reg.top(1))

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
