"""
Q16 PyTorch Models Example
==========================
Demonstrates NIRS analysis using PyTorch models.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
import torch.nn as nn

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.splitters import SPXYSplitter
from nirs4all.utils.backend import TORCH_AVAILABLE, framework
from nirs4all.operators.models.pytorch.nicon import nicon


if not TORCH_AVAILABLE:
    print("PyTorch is not available. Skipping example.")
    exit(0)

parser = argparse.ArgumentParser(description='Q16 PyTorch Models Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()

# Define a simple PyTorch model
@framework('pytorch')
class SimplePyTorchMLP(nn.Module):
    def __init__(self, input_shape=None, params=None, **kwargs):
        super().__init__()
        params = params or {}
        params.update(kwargs)
        self.input_shape = input_shape
        hidden_layers = params.get('hidden_layers', [64, 32])

        if input_shape is not None:
            # input_shape is (seq_len, channels) or (features,)
            if len(input_shape) == 1:
                input_dim = input_shape[0]
            else:
                input_dim = input_shape[0] * input_shape[1]
            self.flatten = nn.Flatten()

            layers = []
            in_dim = input_dim
            for h_dim in hidden_layers:
                layers.append(nn.Linear(in_dim, h_dim))
                layers.append(nn.ReLU())
                in_dim = h_dim
            layers.append(nn.Linear(in_dim, 1))
            self.layers = nn.Sequential(*layers)
        else:
            self.layers = None

    def forward(self, x):
        if self.layers is None:
             raise RuntimeError("Model not initialized with input_shape")
        x = self.flatten(x)
        return self.layers(x)

@framework('pytorch')
class SimplePyTorchClassifier(nn.Module):
    def __init__(self, input_shape=None, params=None, num_classes=2, **kwargs):
        super().__init__()
        params = params or {}
        params.update(kwargs)
        self.input_shape = input_shape
        hidden_layers = params.get('hidden_layers', [64, 32])
        self.num_classes = params.get('num_classes', num_classes)

        if input_shape is not None:
            # input_shape is (seq_len, channels) or (features,)
            if len(input_shape) == 1:
                input_dim = input_shape[0]
            else:
                input_dim = input_shape[0] * input_shape[1]
            self.flatten = nn.Flatten()

            layers = []
            in_dim = input_dim
            for h_dim in hidden_layers:
                layers.append(nn.Linear(in_dim, h_dim))
                layers.append(nn.ReLU())
                in_dim = h_dim
            layers.append(nn.Linear(in_dim, self.num_classes))
            self.layers = nn.Sequential(*layers)
        else:
            self.layers = None

    def forward(self, x):
        if self.layers is None:
             raise RuntimeError("Model not initialized with input_shape")
        x = self.flatten(x)
        return self.layers(x)

# Regression Example
print("\n--- PyTorch Regression Example ---")
data_path_reg = 'sample_data/regression'

pipeline_reg = [
    StandardScaler,
    # SPXYSplitter(0.25),
    ShuffleSplit(n_splits=2, test_size=0.25),
    [
        {
            'model': SimplePyTorchMLP(), # Use instance, ModelFactory will inject input_shape
            'finetune_params': {
                'n_trials': 5,
                'verbose': 2,
                'sample': 'hyperband',
                'model_params': {
                    'hidden_layers': [[32], [64, 32], [64, 32, 16]],
                },
                'train_params': {
                    'epochs': 10
                }
            },
            'train_params': {
                'verbose': 0,
                'epochs': 20,
                'learning_rate': 0.001
            }
        },
        {
            'model': nicon,
            'train_params': {
                'epochs': 20,
                'batch_size': 16,
                'learning_rate': 0.001,
                'verbose': 0
            }
        }
    ]
]

runner = PipelineRunner(save_artifacts=True, verbose=1, plots_visible=args.plots)
preds_reg, _ = runner.run(PipelineConfigs(pipeline_reg), DatasetConfigs(data_path_reg))
print("Top Regression Models:")
print(preds_reg.top(1))

# Prediction Reuse
print("\n--- PyTorch Prediction Reuse Example ---")
best_prediction = preds_reg.top(1)[0]
model_id = best_prediction['id']
print(f"Best model: {best_prediction['model_name']} (id: {model_id})")
reference_predictions = best_prediction['y_pred'][:5].flatten()
print("Reference predictions:", reference_predictions)

predictor = PipelineRunner()
# Using Xval for prediction reuse demonstration
prediction_dataset = DatasetConfigs({'X_test': 'sample_data/regression/Xval.csv.gz'})
method1_predictions, _ = predictor.predict(best_prediction, prediction_dataset, verbose=1)
method1_array = method1_predictions[:5].flatten()
print("Predictions on new data (first 5):")
print(method1_array)

is_identical = np.allclose(method1_array, reference_predictions)
assert is_identical, "Method 1 predictions do not match reference!"
print(f"Method 1 identical to training: {'✅ YES' if is_identical else '❌ NO'}")

# Classification Example
print("\n--- PyTorch Classification Example ---")
data_path_cls = 'sample_data/classification'

pipeline_cls = [
    StandardScaler,
    SPXYSplitter(0.25),
    ShuffleSplit(n_splits=2, test_size=0.25),
    {
        'model': SimplePyTorchClassifier(num_classes=2),
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
