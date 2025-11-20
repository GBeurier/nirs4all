"""
Q16 PyTorch Models Example
==========================
Demonstrates NIRS analysis using PyTorch models.
"""

import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
import torch.nn as nn

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.splitters import SPXYSplitter
from nirs4all.utils.backend import TORCH_AVAILABLE, framework

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
    def __init__(self, input_shape=None, params=None):
        super().__init__()
        params = params or {}
        self.input_shape = input_shape

        if input_shape is not None:
            # input_shape is (seq_len, channels) or (features,)
            if len(input_shape) == 1:
                input_dim = input_shape[0]
            else:
                input_dim = input_shape[0] * input_shape[1]
            self.flatten = nn.Flatten()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
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
    SPXYSplitter(0.25),
    ShuffleSplit(n_splits=2, test_size=0.25),
    {
        'model': SimplePyTorchMLP(), # Use instance, ModelFactory will inject input_shape
        'train_params': {
            'verbose': 1
        }
    }
]

runner = PipelineRunner(save_files=False, verbose=1, plots_visible=args.plots)
preds_reg, _ = runner.run(PipelineConfigs(pipeline_reg), DatasetConfigs(data_path_reg))
print("Top Regression Models:")
print(preds_reg.top(1))

if args.show:
    plt.show()
